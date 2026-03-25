"""Maze router for technology-agnostic standard-cell routing.

Usage::

    maze = MazeRouter(rules, placed)
    route = maze.route_two_pin(comp, net, src_layer, sx, sy, tgt_layer, tx, ty)
"""
from __future__ import annotations

import warnings
from typing import Any

from layout_gen.pdk import PDKRules
from layout_gen.synth.placer import PlacedDevice
from layout_gen.synth.maze.grid import RoutingGrid, build_grid
from layout_gen.synth.maze.solver import a_star, route_problem
from layout_gen.synth.maze.render import render_route
from layout_gen.synth.maze.types import (
    GridPoint, NetRoute, RouteSegment, ViaLocation, RoutingProblem,
)


class MazeRouter:
    """Grid-based maze router that maintains global obstacle state.

    Create once per synthesis run, then call :meth:`route_two_pin` or
    :meth:`route_net` for each connection.  Routed wires are automatically
    registered as obstacles for subsequent routes.
    """

    def __init__(self, rules: PDKRules, placed: dict[str, PlacedDevice]):
        self.rules = rules
        self.placed = placed
        self.grid = build_grid(rules, placed)

        # Build preferred-direction map (layer_index -> "horizontal"/"vertical"/"")
        self._pref_dir: dict[int, str] = {}
        for i, lyr in enumerate(self.grid.layers):
            self._pref_dir[i] = rules.direction(lyr)

        # Reverse lookup: GDS (layer, datatype) tuple → logical layer name
        # Used by _rect() to auto-register drawn shapes on the obstacle map.
        self._layer_name_cache: dict[tuple[int, int], str] = {}
        for lyr_name in self.grid.layers:
            try:
                gds_tuple = rules.layer(lyr_name)
                self._layer_name_cache[gds_tuple] = lyr_name
            except KeyError:
                pass

    # ── Public API ───────────────────────────────────────────────────────

    def mark_obstacle(
        self,
        layer: str,
        x0: float, y0: float,
        x1: float, y1: float,
        net: str | None = None,
    ) -> None:
        """Register an externally drawn shape as an obstacle (or net owner)."""
        self.grid.mark_rect(layer, x0, y0, x1, y1, net)

    def route_two_pin(
        self,
        comp: Any,
        net: str,
        src_layer: str,
        sx: float, sy: float,
        tgt_layer: str,
        tx: float, ty: float,
        *,
        allowed_layers: list[str] | None = None,
        render: bool = True,
    ) -> NetRoute | None:
        """Route a two-pin connection and draw it.

        Parameters
        ----------
        comp : gdsfactory Component to draw on.
        net : Net name.
        src_layer, tgt_layer : Layer names for source/target pins.
        sx, sy, tx, ty : Pin centre coordinates (µm).
        allowed_layers : Restrict routing to these layers.
        render : If True, draw polygons immediately.

        Returns
        -------
        NetRoute on success, None if no path found.
        """
        grid = self.grid

        # Convert to grid coordinates
        src_li = grid.layer_index(src_layer)
        tgt_li = grid.layer_index(tgt_layer)
        sgx, sgy = grid.world_to_grid(sx, sy)
        tgx, tgy = grid.world_to_grid(tx, ty)

        src = GridPoint(src_li, sgx, sgy)
        tgt = GridPoint(tgt_li, tgx, tgy)

        al = None
        if allowed_layers:
            al = [grid.layer_index(l) for l in allowed_layers if l in grid.layers]

        nid = grid.net_id(net)

        # Ensure source and target cells are accessible for this net
        self._clear_pin(src, nid)
        self._clear_pin(tgt, nid)

        path = a_star(
            grid, [src], [tgt], nid,
            allowed_layers=al,
            preferred_dir=self._pref_dir,
        )
        if path is None:
            warnings.warn(
                f"Maze router: no path found for net {net!r} "
                f"from ({sx:.3f},{sy:.3f}) to ({tx:.3f},{ty:.3f})",
                stacklevel=2,
            )
            return None

        # Mark routed cells on grid
        for pt in path:
            if grid.in_bounds(pt.layer, pt.gx, pt.gy):
                grid.occupied[pt.layer, pt.gy, pt.gx] = nid

        # Build route geometry
        route = NetRoute(net=net)
        from layout_gen.synth.maze.solver import _extract_geometry
        _extract_geometry(grid, path, route)

        if render:
            render_route(comp, route, self.rules)
            # Also mark the rendered shapes (with wire width) as obstacles
            self._mark_route_obstacles(route, net)

        return route

    def route_net(
        self,
        comp: Any,
        problem: RoutingProblem,
        *,
        render: bool = True,
    ) -> NetRoute | None:
        """Route a multi-target net and draw it."""
        nid = self.grid.net_id(problem.net)
        for s in problem.sources:
            self._clear_pin(s, nid)
        for t in problem.targets:
            self._clear_pin(t, nid)

        result = route_problem(self.grid, problem, self._pref_dir)
        if result is None:
            warnings.warn(
                f"Maze router: no path for net {problem.net!r}",
                stacklevel=2,
            )
            return None

        if render:
            render_route(comp, result, self.rules)
            self._mark_route_obstacles(result, problem.net)

        return result

    # ── Internals ────────────────────────────────────────────────────────

    def _clear_pin(self, pt: GridPoint, net_id: int) -> None:
        """Ensure a pin location is accessible for routing.

        If the cell is blocked by an obstacle, re-assign it to the net.
        """
        g = self.grid
        if not g.in_bounds(pt.layer, pt.gx, pt.gy):
            return
        v = g.occupied[pt.layer, pt.gy, pt.gx]
        if v != net_id and v != 0:
            # Force-assign so the pin is reachable
            g.occupied[pt.layer, pt.gy, pt.gx] = net_id

    def _mark_route_obstacles(self, route: NetRoute, net: str) -> None:
        """Mark rendered wire shapes as obstacles on the grid."""
        for seg in route.segments:
            lyr_rules = getattr(self.rules, seg.layer, None)
            if isinstance(lyr_rules, dict):
                w = lyr_rules.get("width_min_um", 0.17)
            else:
                w = 0.17
            hw = w / 2

            if abs(seg.x1 - seg.x0) > abs(seg.y1 - seg.y0):
                cy = (seg.y0 + seg.y1) / 2
                self.grid.mark_rect(
                    seg.layer,
                    min(seg.x0, seg.x1), cy - hw,
                    max(seg.x0, seg.x1), cy + hw,
                    net=net,
                )
            else:
                cx = (seg.x0 + seg.x1) / 2
                self.grid.mark_rect(
                    seg.layer,
                    cx - hw, min(seg.y0, seg.y1),
                    cx + hw, max(seg.y0, seg.y1),
                    net=net,
                )
