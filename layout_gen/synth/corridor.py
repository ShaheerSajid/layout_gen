"""Routable corridor map — per-layer free-space planes.

Every layer starts fully routable.  Each drawn shape carves out its
footprint **plus** the layer's spacing rule, shrinking the corridor.
Shapes on adjacent layers also carve through inter-layer rules (via
enclosure, contact enclosure, etc.).

Handlers query the corridor to find valid routing regions:

    corridor.is_free("li1", x0, y0, x1, y1)
    corridor.max_extent("li1", x, y0, y1, "right")
"""
from __future__ import annotations

import math
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


class CorridorMap:
    """Per-layer 2D free-space map for routing-aware placement.

    Each routable layer gets a boolean grid (True = free).  Drawing a
    shape calls :meth:`carve`, which marks the shape + spacing bloat
    as blocked on that layer **and** any affected adjacent layers.
    """

    def __init__(
        self,
        rules: PDKRules,
        placed: dict[str, PlacedDevice],
        margin: float = 1.0,
    ):
        self.rules = rules
        self.res = rules.mfg_grid if rules.mfg_grid > 0 else 0.005

        # ── Bounding box from placed devices ──────────────────────────
        xs, ys, x1s, y1s = [], [], [], []
        for d in placed.values():
            xs.append(d.x)
            ys.append(d.y)
            x1s.append(d.x + d.geom.total_x_um)
            y1s.append(d.y + d.geom.total_y_um)

        self.x0 = min(xs) - margin
        self.y0 = min(ys) - margin
        self.x1 = max(x1s) + margin
        self.y1 = max(y1s) + margin
        self.nx = int(math.ceil((self.x1 - self.x0) / self.res)) + 1
        self.ny = int(math.ceil((self.y1 - self.y0) / self.res)) + 1

        # ── Build per-GDS-tuple planes ────────────────────────────────
        # Multiple logical names may alias the same GDS tuple (e.g.
        # li1 and met1 on GF180).  They share one physical plane.
        self._name_to_gds: dict[str, tuple[int, int]] = {}
        self._gds_spacing: dict[tuple[int, int], float] = {}
        self._gds_planes: dict[tuple[int, int], np.ndarray] = {}

        for lyr_name in self._routable_names(rules):
            try:
                gds = rules.layer(lyr_name)
            except KeyError:
                continue
            sec = getattr(rules, lyr_name, None)
            if not isinstance(sec, dict):
                continue
            sp = sec.get("spacing_min_um", 0.0)
            if sp <= 0:
                continue
            self._name_to_gds[lyr_name] = gds
            # Keep the largest spacing when names alias the same GDS layer
            if gds not in self._gds_spacing or sp > self._gds_spacing[gds]:
                self._gds_spacing[gds] = sp
            if gds not in self._gds_planes:
                self._gds_planes[gds] = np.ones(
                    (self.ny, self.nx), dtype=bool,
                )

        # ── Cross-layer effects ───────────────────────────────────────
        # drawn_gds_tuple → [(affected_gds_tuple, bloat_um), ...]
        self._cross: dict[
            tuple[int, int], list[tuple[tuple[int, int], float]]
        ] = self._build_cross_effects(rules)

        # ── GDS tuple reverse lookup (for _rect wrapper) ─────────────
        self._gds_to_names: dict[tuple[int, int], list[str]] = {}
        for n, g in self._name_to_gds.items():
            self._gds_to_names.setdefault(g, []).append(n)

        # ── Pre-populate transistor shapes ────────────────────────────
        self._populate_transistors(placed, rules)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def carve(
        self,
        layer: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        """Mark *shape + spacing bloat* as blocked (logical layer name)."""
        gds = self._name_to_gds.get(layer)
        if gds is not None:
            self._carve_gds(gds, x0, y0, x1, y1)
        # Cross-layer: if this layer name maps to a contact/via GDS
        # that isn't itself routable, we still handle it here.
        try:
            drawn_gds = self.rules.layer(layer)
        except KeyError:
            return
        for affected_gds, bloat in self._cross.get(drawn_gds, []):
            if affected_gds in self._gds_planes:
                self._mark_blocked(
                    self._gds_planes[affected_gds],
                    x0 - bloat, y0 - bloat,
                    x1 + bloat, y1 + bloat,
                )

    def carve_gds(
        self,
        gds: tuple[int, int],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        """Carve using a GDS (layer, datatype) tuple directly."""
        self._carve_gds(gds, x0, y0, x1, y1)
        for affected_gds, bloat in self._cross.get(gds, []):
            if affected_gds in self._gds_planes:
                self._mark_blocked(
                    self._gds_planes[affected_gds],
                    x0 - bloat, y0 - bloat,
                    x1 + bloat, y1 + bloat,
                )

    def is_free(
        self,
        layer: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> bool:
        """True if the rectangle is entirely within the free corridor."""
        gds = self._name_to_gds.get(layer)
        if gds is None or gds not in self._gds_planes:
            return True
        plane = self._gds_planes[gds]
        gx0, gy0 = self._to_grid(x0, y0)
        gx1, gy1 = self._to_grid(x1, y1)
        gx0 = max(0, gx0)
        gy0 = max(0, gy0)
        gx1 = min(self.nx - 1, gx1)
        gy1 = min(self.ny - 1, gy1)
        if gx0 > gx1 or gy0 > gy1:
            return True
        return bool(plane[gy0:gy1 + 1, gx0:gx1 + 1].all())

    def max_extent(
        self,
        layer: str,
        x: float,
        y0: float,
        y1: float,
        direction: str,
    ) -> float:
        """Furthest X reachable from *x* within the y-band before blocked.

        Parameters
        ----------
        direction : ``"left"`` or ``"right"``.

        Returns the world X of the last free grid cell in that direction.
        """
        gds = self._name_to_gds.get(layer)
        if gds is None or gds not in self._gds_planes:
            return self.x1 if direction == "right" else self.x0
        plane = self._gds_planes[gds]

        gx, _ = self._to_grid(x, 0)
        _, gy0 = self._to_grid(0, y0)
        _, gy1 = self._to_grid(0, y1)
        gy0 = max(0, gy0)
        gy1 = min(self.ny - 1, gy1)

        if direction == "right":
            for cgx in range(max(gx, 0), self.nx):
                if not plane[gy0:gy1 + 1, cgx].all():
                    wx, _ = self._to_world(cgx, 0)
                    return wx
            return self.x1
        else:
            for cgx in range(min(gx, self.nx - 1), -1, -1):
                if not plane[gy0:gy1 + 1, cgx].all():
                    wx, _ = self._to_world(cgx, 0)
                    return wx
            return self.x0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(round((x - self.x0) / self.res))
        gy = int(round((y - self.y0) / self.res))
        return gx, gy

    def _to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return self.x0 + gx * self.res, self.y0 + gy * self.res

    def _carve_gds(
        self,
        gds: tuple[int, int],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        """Same-layer carve: shape + spacing bloat."""
        if gds not in self._gds_planes:
            return
        bloat = self._gds_spacing.get(gds, 0.0)
        self._mark_blocked(
            self._gds_planes[gds],
            x0 - bloat, y0 - bloat,
            x1 + bloat, y1 + bloat,
        )

    def _mark_blocked(
        self,
        plane: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        gx0, gy0 = self._to_grid(x0, y0)
        gx1, gy1 = self._to_grid(x1, y1)
        gx0, gx1 = min(gx0, gx1), max(gx0, gx1)
        gy0, gy1 = min(gy0, gy1), max(gy0, gy1)
        gx0 = max(0, gx0)
        gy0 = max(0, gy0)
        gx1 = min(self.nx - 1, gx1)
        gy1 = min(self.ny - 1, gy1)
        if gx0 <= gx1 and gy0 <= gy1:
            plane[gy0:gy1 + 1, gx0:gx1 + 1] = False

    # ── Layer enumeration ─────────────────────────────────────────────

    @staticmethod
    def _routable_names(rules: PDKRules) -> list[str]:
        candidates = ["poly", "li1", "met1", "met2", "met3", "met4", "met5"]
        out = []
        for c in candidates:
            try:
                rules.layer(c)
                out.append(c)
            except KeyError:
                pass
        return out

    # ── Cross-layer effect builder ────────────────────────────────────

    def _build_cross_effects(
        self, rules: PDKRules,
    ) -> dict[tuple[int, int], list[tuple[tuple[int, int], float]]]:
        fx: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}

        # licon1 / contact → carves li1 and poly
        try:
            licon_gds = rules.layer("licon1")
            if "li1" in self._name_to_gds:
                enc = rules.contacts.get(
                    "enclosure_in_li1_2adj_um",
                    rules.contacts.get("enclosure_in_li1_um", 0.0),
                )
                fx.setdefault(licon_gds, []).append(
                    (self._name_to_gds["li1"], enc),
                )
            if "poly" in self._name_to_gds:
                enc = rules.contacts.get("poly_enclosure_um", 0.0)
                fx.setdefault(licon_gds, []).append(
                    (self._name_to_gds["poly"], enc),
                )
        except KeyError:
            pass

        # Via layers from the metal stack
        stack = getattr(rules, "_metal_stack_raw", None)
        if stack is None:
            # Try the parsed via_stack_between as fallback
            return fx

        for entry in stack:
            via_name = entry.get("via")
            via_rules_section = entry.get("via_rules", via_name)
            upper_metal = entry.get("metal")
            if not via_name or not upper_metal:
                continue
            try:
                via_gds = rules.layer(via_name)
            except KeyError:
                continue
            via_sec = getattr(rules, via_rules_section, {})
            if not isinstance(via_sec, dict):
                via_sec = {}

            # Find the lower metal (previous entry)
            idx = stack.index(entry)
            lower_metal = stack[idx - 1]["metal"] if idx > 0 else None

            for metal in [lower_metal, upper_metal]:
                if metal and metal in self._name_to_gds:
                    enc_key_2adj = f"enclosure_in_{metal}_2adj_um"
                    enc_key = f"enclosure_in_{metal}_um"
                    enc = via_sec.get(enc_key_2adj, via_sec.get(enc_key, 0.0))
                    fx.setdefault(via_gds, []).append(
                        (self._name_to_gds[metal], enc),
                    )

        return fx

    # ── Pre-populate transistor shapes ────────────────────────────────

    def _populate_transistors(
        self,
        placed: dict[str, PlacedDevice],
        rules: PDKRules,
    ) -> None:
        """Carve transistor S/D strips and gate poly into the corridor."""
        for dev in placed.values():
            dy0, dy1 = global_diff_y(dev, rules)

            # S/D li1 strips
            for j in range(dev.geom.n_fingers + 1):
                sx0, sx1 = global_sd_x(dev, j, rules)
                self.carve("li1", sx0, dy0, sx1, dy1)

            # Gate poly
            pt = global_poly_top(dev)
            pb = global_poly_bottom(dev)
            for j in range(dev.geom.n_fingers):
                gx0, gx1 = global_gate_x(dev, j)
                self.carve("poly", gx0, pb, gx1, pt)
