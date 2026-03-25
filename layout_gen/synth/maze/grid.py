"""Routing grid — 3-D occupancy map for the maze router."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from layout_gen.pdk import PDKRules
from layout_gen.synth.placer import (
    PlacedDevice,
    global_gate_x,
    global_sd_x,
    global_diff_y,
    global_poly_top,
    global_poly_bottom,
)
from layout_gen.cells.standard import _gate_x, _sd_x, _diff_y


# ── Obstacle ID conventions ─────────────────────────────────────────────────

OBSTACLE = -1       # immovable / unroutable
FREE     =  0       # available for any net


@dataclass
class RoutingGrid:
    """3-D occupancy grid for maze routing.

    Coordinates
    -----------
    World: (x_um, y_um)  —  microns, same frame as placed devices.
    Grid:  (gx, gy)      —  integer indices.

    grid_to_world / world_to_grid convert between them.
    ``occupied[layer_idx, gy, gx]`` stores the net-id that owns a cell,
    ``FREE`` if available, or ``OBSTACLE`` if permanently blocked.
    """

    x_origin: float
    y_origin: float
    pitch: float
    nx: int
    ny: int
    layers: list[str]           # ordered, e.g. ["li1", "met1", "met2"]
    occupied: np.ndarray        # int32, shape (n_layers, ny, nx)

    # per-layer half-bloat in grid units (spacing + half wire width)
    bloat: dict[str, int] = field(default_factory=dict)

    # net name → integer id
    _net_ids: dict[str, int] = field(default_factory=dict)
    _next_id: int = 1

    # ── Coordinate helpers ───────────────────────────────────────────────

    def layer_index(self, name: str) -> int:
        return self.layers.index(name)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = round((x - self.x_origin) / self.pitch)
        gy = round((y - self.y_origin) / self.pitch)
        return int(gx), int(gy)

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        x = self.x_origin + gx * self.pitch
        y = self.y_origin + gy * self.pitch
        return x, y

    def in_bounds(self, li: int, gx: int, gy: int) -> bool:
        return 0 <= li < len(self.layers) and 0 <= gx < self.nx and 0 <= gy < self.ny

    # ── Net id management ────────────────────────────────────────────────

    def net_id(self, net: str) -> int:
        """Get or allocate a positive integer id for *net*."""
        if net not in self._net_ids:
            self._net_ids[net] = self._next_id
            self._next_id += 1
        return self._net_ids[net]

    # ── Marking ──────────────────────────────────────────────────────────

    def mark_rect(
        self,
        layer: str,
        x0: float, y0: float,
        x1: float, y1: float,
        net: str | None = None,
        *,
        extra_bloat: int = 0,
    ) -> None:
        """Mark a rectangle as occupied (with spacing bloat).

        Parameters
        ----------
        net : If given, cells are marked with the net's id so the same net
              can share these cells.  If *None*, cells are marked OBSTACLE.
        extra_bloat : Additional bloat beyond the layer's default.
        """
        li = self.layer_index(layer) if layer in self.layers else -1
        if li < 0:
            return

        val = self.net_id(net) if net else OBSTACLE
        b = self.bloat.get(layer, 0) + extra_bloat

        gx0, gy0 = self.world_to_grid(x0, y0)
        gx1, gy1 = self.world_to_grid(x1, y1)
        # Ensure proper ordering
        gx0, gx1 = min(gx0, gx1), max(gx0, gx1)
        gy0, gy1 = min(gy0, gy1), max(gy0, gy1)

        # Apply bloat
        gx0 -= b; gy0 -= b
        gx1 += b; gy1 += b

        # Clamp to bounds
        gx0 = max(gx0, 0); gy0 = max(gy0, 0)
        gx1 = min(gx1, self.nx - 1); gy1 = min(gy1, self.ny - 1)

        self.occupied[li, gy0:gy1 + 1, gx0:gx1 + 1] = val

    def is_free(self, li: int, gx: int, gy: int, net_id: int = 0) -> bool:
        """True if cell can be used by *net_id* (or any net if 0)."""
        if not self.in_bounds(li, gx, gy):
            return False
        v = self.occupied[li, gy, gx]
        if v == FREE:
            return True
        if net_id > 0 and v == net_id:
            return True   # same net can re-use its own cells
        return False


# ── Factory ──────────────────────────────────────────────────────────────────

def build_grid(
    rules: PDKRules,
    placed: dict[str, PlacedDevice],
    *,
    margin: float = 0.5,
) -> RoutingGrid:
    """Create a RoutingGrid and populate it with transistor obstacles.

    Parameters
    ----------
    margin : Extra space (µm) around the device bounding box.
    """
    pitch = rules.mfg_grid
    if pitch <= 0:
        pitch = 0.005

    # ── Determine grid extent from placed devices ────────────────────────
    all_x: list[float] = []
    all_y: list[float] = []
    for dev in placed.values():
        dy0, dy1 = global_diff_y(dev, rules)
        all_y.extend([dy0, dy1])
        for j in range(dev.geom.n_fingers + 1):
            sx0, sx1 = global_sd_x(dev, j, rules)
            all_x.extend([sx0, sx1])
        for j in range(dev.geom.n_fingers):
            gx0, gx1 = global_gate_x(dev, j)
            all_x.extend([gx0, gx1])

    if not all_x or not all_y:
        raise ValueError("No placed devices — cannot build routing grid")

    x_min = min(all_x) - margin
    x_max = max(all_x) + margin
    y_min = min(all_y) - margin
    y_max = max(all_y) + margin

    nx = int(math.ceil((x_max - x_min) / pitch)) + 1
    ny = int(math.ceil((y_max - y_min) / pitch)) + 1

    # ── Determine routable layers ────────────────────────────────────────
    layers: list[str] = []
    if rules.li1_is_met1:
        layers.append("li1")           # li1 == met1, one layer
    else:
        layers.append("li1")
        layers.append("met1")
    # Always include met2 if it exists
    try:
        rules.layer("met2")
        layers.append("met2")
    except KeyError:
        pass

    occupied = np.zeros((len(layers), ny, nx), dtype=np.int32)

    # ── Per-layer bloat = ceil((spacing + half_width) / pitch) ───────────
    bloat: dict[str, int] = {}
    for lyr in layers:
        sec = getattr(rules, lyr, None)
        if not isinstance(sec, dict):
            # li1_is_met1: use met1 rules for li1
            if lyr == "li1" and rules.li1_is_met1:
                sec = rules.met1
            else:
                sec = {}
        sp = sec.get("spacing_min_um", 0.0)
        hw = sec.get("width_min_um", 0.0) / 2
        bloat[lyr] = int(math.ceil((sp + hw) / pitch))

    grid = RoutingGrid(
        x_origin=x_min,
        y_origin=y_min,
        pitch=pitch,
        nx=nx,
        ny=ny,
        layers=layers,
        occupied=occupied,
        bloat=bloat,
    )

    # ── Populate transistor obstacles ────────────────────────────────────
    _populate_transistors(grid, placed, rules)

    return grid


def _populate_transistors(
    grid: RoutingGrid,
    placed: dict[str, PlacedDevice],
    rules: PDKRules,
) -> None:
    """Mark transistor S/D li1 strips and gate poly as obstacles."""
    li1_layer = "li1"

    for name, dev in placed.items():
        # ── S/D li1 strips ───────────────────────────────────────────────
        dy0, dy1 = global_diff_y(dev, rules)
        for j in range(dev.geom.n_fingers + 1):
            sx0, sx1 = global_sd_x(dev, j, rules)
            # S/D li1 strip spans the full diffusion height
            # (may extend slightly beyond due to enclosure, but diff height
            # is a safe approximation for obstacle marking)
            grid.mark_rect(li1_layer, sx0, dy0, sx1, dy1, net=None)

        # ── Gate poly ────────────────────────────────────────────────────
        # Poly runs vertically across the full device height (and endcap)
        pt = global_poly_top(dev)
        pb = global_poly_bottom(dev)
        for j in range(dev.geom.n_fingers):
            gx0, gx1 = global_gate_x(dev, j)
            # Mark poly as obstacle on li1 (contacts cannot be placed here,
            # and li1 routes must avoid poly-covered regions near transistors)
            grid.mark_rect(li1_layer, gx0, pb, gx1, pt, net=None)
