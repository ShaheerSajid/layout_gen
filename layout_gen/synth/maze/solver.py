"""A* maze router on a 3-D grid."""
from __future__ import annotations

import heapq
from typing import Sequence

from layout_gen.synth.maze.grid import RoutingGrid, FREE
from layout_gen.synth.maze.types import (
    GridPoint, NetRoute, RouteSegment, ViaLocation, RoutingProblem,
)


# ── Cost tuning ──────────────────────────────────────────────────────────────

_STEP_COST      = 1      # one grid step in preferred direction
_WRONG_DIR_COST = 3      # step against preferred direction
_BEND_COST      = 2      # change direction on same layer
_VIA_COST       = 20     # layer change


# ── 2-D neighbours (dx, dy) ─────────────────────────────────────────────────

_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _heuristic(a: GridPoint, b: GridPoint) -> int:
    """Manhattan distance on the grid (ignoring layers)."""
    return abs(a.gx - b.gx) + abs(a.gy - b.gy)


def _multi_target_heuristic(p: GridPoint, targets: Sequence[GridPoint]) -> int:
    """Min Manhattan distance to any target."""
    return min(_heuristic(p, t) for t in targets)


def a_star(
    grid: RoutingGrid,
    sources: list[GridPoint],
    targets: list[GridPoint],
    net_id: int,
    allowed_layers: list[int] | None = None,
    preferred_dir: dict[int, str] | None = None,
) -> list[GridPoint] | None:
    """Find shortest path from any source cell to any target cell.

    Returns the path as a list of GridPoints (source→target), or None if
    no path exists.
    """
    if not sources or not targets:
        return None

    if preferred_dir is None:
        preferred_dir = {}

    target_set = set(targets)

    # Priority queue: (f_cost, g_cost, counter, point, prev_dir)
    # prev_dir: (dx, dy) of the last move, or None for seed cells.
    _counter = 0
    open_heap: list[tuple[int, int, int, GridPoint, tuple[int, int] | None]] = []
    came_from: dict[GridPoint, GridPoint | None] = {}
    g_score: dict[GridPoint, int] = {}

    for s in sources:
        if not grid.in_bounds(s.layer, s.gx, s.gy):
            continue
        g_score[s] = 0
        came_from[s] = None
        h = _multi_target_heuristic(s, targets)
        heapq.heappush(open_heap, (h, 0, _counter, s, None))
        _counter += 1

    while open_heap:
        f, g, _, current, prev_dir = heapq.heappop(open_heap)

        if current in target_set:
            # Reconstruct path
            path = []
            node: GridPoint | None = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        # Already found a better path?
        if g > g_score.get(current, float("inf")):
            continue

        li = current.layer
        pref = preferred_dir.get(li, "")

        # ── Same-layer neighbours ────────────────────────────────────────
        for dx, dy in _DIRS:
            ngx = current.gx + dx
            ngy = current.gy + dy
            if not grid.is_free(li, ngx, ngy, net_id):
                continue
            if allowed_layers is not None and li not in allowed_layers:
                continue

            # Cost
            step = _STEP_COST
            if pref == "horizontal" and dy != 0:
                step = _WRONG_DIR_COST
            elif pref == "vertical" and dx != 0:
                step = _WRONG_DIR_COST

            if prev_dir is not None and (dx, dy) != prev_dir:
                step += _BEND_COST

            ng = g + step
            nb = GridPoint(li, ngx, ngy)
            if ng < g_score.get(nb, float("inf")):
                g_score[nb] = ng
                came_from[nb] = current
                h = _multi_target_heuristic(nb, targets)
                heapq.heappush(open_heap, (ng + h, ng, _counter, nb, (dx, dy)))
                _counter += 1

        # ── Layer-change neighbours (vias) ───────────────────────────────
        for li2 in range(len(grid.layers)):
            if li2 == li:
                continue
            if allowed_layers is not None and li2 not in allowed_layers:
                continue
            if abs(li2 - li) != 1:
                continue  # only adjacent layers
            if not grid.is_free(li2, current.gx, current.gy, net_id):
                continue

            ng = g + _VIA_COST
            nb = GridPoint(li2, current.gx, current.gy)
            if ng < g_score.get(nb, float("inf")):
                g_score[nb] = ng
                came_from[nb] = current
                h = _multi_target_heuristic(nb, targets)
                heapq.heappush(open_heap, (ng + h, ng, _counter, nb, None))
                _counter += 1

    return None  # no path found


def route_problem(
    grid: RoutingGrid,
    problem: RoutingProblem,
    preferred_dir: dict[int, str] | None = None,
) -> NetRoute | None:
    """Route a multi-target net (Steiner-tree approximation).

    Routes to each target sequentially, adding each routed path to the
    source set for the next target.
    """
    nid = grid.net_id(problem.net)
    sources = list(problem.sources)
    route = NetRoute(net=problem.net)

    for target in problem.targets:
        path = a_star(
            grid, sources, [target], nid,
            allowed_layers=problem.allowed_layers,
            preferred_dir=preferred_dir,
        )
        if path is None:
            return None

        # Mark routed path on grid so subsequent targets can re-use
        for pt in path:
            if grid.in_bounds(pt.layer, pt.gx, pt.gy):
                grid.occupied[pt.layer, pt.gy, pt.gx] = nid

        # Add path points as new sources
        sources.extend(path)

        # Convert path to segments and vias
        _extract_geometry(grid, path, route)

    return route


def _extract_geometry(
    grid: RoutingGrid,
    path: list[GridPoint],
    route: NetRoute,
) -> None:
    """Convert a grid path into RouteSegments and ViaLocations."""
    if len(path) < 2:
        return

    i = 0
    while i < len(path):
        # Find the longest run on the same layer
        li = path[i].layer
        layer_name = grid.layers[li]
        j = i + 1
        while j < len(path) and path[j].layer == li:
            j += 1

        # Segment from path[i] to path[j-1] on this layer
        if j - 1 > i:
            x0, y0 = grid.grid_to_world(path[i].gx, path[i].gy)
            x1, y1 = grid.grid_to_world(path[j - 1].gx, path[j - 1].gy)
            route.segments.append(RouteSegment(
                layer=layer_name,
                x0=x0, y0=y0, x1=x1, y1=y1,
                width=0.0,  # filled in by render
            ))

        # Via at layer transition
        if j < len(path) and path[j].layer != li:
            cx, cy = grid.grid_to_world(path[j - 1].gx, path[j - 1].gy)
            route.vias.append(ViaLocation(
                cx=cx, cy=cy,
                from_layer=layer_name,
                to_layer=grid.layers[path[j].layer],
            ))

        i = j
