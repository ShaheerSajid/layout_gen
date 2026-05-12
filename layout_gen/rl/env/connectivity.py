"""
layout_gen.rl.env.connectivity — heuristic per-net connectivity score.

The full notion of net connectivity needs an LVS extractor (e.g. magic
or netgen). For Phase 4 part 4 we use a cheaper, structural heuristic
that only depends on geometry:

  For each net N in the topology:
    1. Look up every device terminal that's electrically tied to N
       (from the topology graph: ``device.terminals[term_name] == N``).
    2. Look up the global position of each such terminal (recorded by
       :func:`place_device_full` and stashed in
       ``LayoutEnv._terminals``).
    3. A terminal is "touched" if any wire rect (``shape_type == 'wire'``,
       net-tag matches N) overlaps the terminal's (x, y) within a
       small tolerance.
    4. ``score_for_net = touched_terminals / total_terminals``.
  ``total_score = sum_{nets} score_for_net``.

Δscore between consecutive env steps is a dense, smooth reward signal
that pushes ROUTE actions toward terminal-aligned segments. It does
not (yet) reward inter-segment connectivity along long paths — for
v1 we trust that segments touching multiple terminals of the same net
will tend to merge naturally as PPO explores.

A trained ROUTE policy gets ``connectivity_score == n_nets`` when every
net has all its terminals touched.
"""
from __future__ import annotations

from typing import Iterable

from layout_gen.synth.geo.state import LayoutState

from layout_gen.rl.topology.parser import TopologyGraph


# How close (in µm) a wire rect's bbox must be to a terminal's (x, y)
# for the terminal to count as "touched". Slightly looser than zero so
# the policy doesn't have to land bins exactly on the port centroid.
DEFAULT_TOUCH_TOL_UM = 0.05


def _touches(rect_x0: float, rect_y0: float,
             rect_x1: float, rect_y1: float,
             px: float, py: float, *, tol: float) -> bool:
    return (rect_x0 - tol <= px <= rect_x1 + tol and
            rect_y0 - tol <= py <= rect_y1 + tol)


def compute_connectivity_score(
    state:     LayoutState,
    topology:  TopologyGraph,
    terminals: dict[tuple[int, str], tuple[float, float, str]],
    *,
    tol_um:    float = DEFAULT_TOUCH_TOL_UM,
) -> float:
    """Sum-of-fractions per-net connectivity score in [0, n_nets].

    Parameters
    ----------
    state :
        Current :class:`LayoutState`. Wire rects (``shape_type='wire'``)
        with a non-empty ``net`` are the only candidates considered for
        "touching".
    topology :
        :class:`TopologyGraph` whose devices' terminal→net mapping
        defines which terminals belong to each net.
    terminals :
        ``{(device_idx, term_name): (x_um, y_um, layer)}`` populated by
        :func:`place_device_full` after each PLACE action.
    tol_um :
        Touch tolerance in µm.

    Returns
    -------
    float :
        ``Σ_nets (touched_terminals / total_terminals)``. Zero before any
        terminals are placed; equal to ``n_nets`` when every net has all
        its terminals touched by appropriately-tagged wire rects.
    """
    # Pre-bucket wire rects by net for O(N+M) lookup.
    wires_by_net: dict[str, list] = {}
    for r in state:
        if r.shape_type == "wire" and r.net:
            wires_by_net.setdefault(r.net, []).append(r)

    score = 0.0
    for net in topology.nets:
        # Collect terminals belonging to this net via the topology graph.
        terminal_points: list[tuple[float, float]] = []
        for (d_idx, term_name) in net.connections:
            pos = terminals.get((d_idx, term_name))
            if pos is None:
                continue   # device not yet placed
            px, py, _layer = pos
            terminal_points.append((px, py))
        if not terminal_points:
            continue

        net_wires = wires_by_net.get(net.name, [])
        if not net_wires:
            continue   # no segments tagged for this net yet → 0 contribution

        touched = 0
        for px, py in terminal_points:
            for r in net_wires:
                if _touches(r.x0, r.y0, r.x1, r.y1, px, py, tol=tol_um):
                    touched += 1
                    break
        score += touched / len(terminal_points)
    return float(score)


__all__ = [
    "DEFAULT_TOUCH_TOL_UM",
    "compute_connectivity_score",
]
