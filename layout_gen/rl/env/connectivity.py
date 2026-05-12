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


def _bbox_overlap(a, b, *, tol: float) -> bool:
    return (a.x0 <= b.x1 + tol and a.x1 + tol >= b.x0 and
            a.y0 <= b.y1 + tol and a.y1 + tol >= b.y0)


# ── Transitive electrical connectivity ──────────────────────────────────────

def compute_electrical_score(
    state:     LayoutState,
    topology:  TopologyGraph,
    terminals: dict[tuple[int, str], tuple[float, float, str]],
    *,
    tol_um:    float = DEFAULT_TOUCH_TOL_UM,
) -> float:
    """Per-net "all terminals electrically connected" score.

    For each net N:
      1. Build a union-find over (this net's wire rects) ∪
         (terminal positions of this net).
      2. Wire-to-wire union when both wires are on the **same logical
         layer** AND their bboxes overlap (with tolerance).
      3. Wire-to-terminal union when the wire's layer matches the
         terminal's layer AND the wire's bbox contains the terminal
         point.
      4. Net contributes 1.0 iff every terminal of the net ends up in
         a single connected component; 0.0 otherwise.

    Returns
    -------
    float :
        Sum over nets, in ``[0, n_nets]``. Strictly stricter than
        :func:`compute_connectivity_score` — a net of 5 disjoint
        terminal-touching wires scores 0 here but ``≥ partial`` there.

    Why this matters
    ----------------
    LVS-style connectivity needs *transitive* unions. The per-terminal-
    touched score rewards a policy for each terminal it grazes; this
    score only rewards completed nets. Use the two together: the
    per-terminal score gives a dense gradient toward the right
    direction, the electrical score rewards finishing the job.

    Cross-layer connectivity (via stacks) is **not** modelled here —
    a real via primitive would need to enter the action space first.
    For now the policy is incentivised to use a single layer per net.
    """
    # Bucket wires per net + per layer for fast same-layer overlap checks.
    wires_by_net: dict[str, list] = {}
    for r in state:
        if r.shape_type == "wire" and r.net:
            wires_by_net.setdefault(r.net, []).append(r)

    score = 0.0
    for net in topology.nets:
        # Resolve this net's terminals.
        net_terms: list[tuple[float, float, str]] = []
        for (d_idx, term_name) in net.connections:
            pos = terminals.get((d_idx, term_name))
            if pos is None:
                continue
            net_terms.append(pos)
        if len(net_terms) <= 1:
            # Singleton-net (or unplaced) nets are trivially "connected".
            # They do still count toward n_nets so they don't penalise
            # the overall score for the policy.
            score += 1.0 if net_terms else 0.0
            continue

        net_wires = wires_by_net.get(net.name, [])

        # Union-find over [terminals..., wires...].
        n_t = len(net_terms)
        n_w = len(net_wires)
        n_total = n_t + n_w
        parent = list(range(n_total))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Same-layer wire-to-wire unions.
        for i in range(n_w):
            for j in range(i + 1, n_w):
                if (net_wires[i].layer == net_wires[j].layer and
                        _bbox_overlap(net_wires[i], net_wires[j], tol=tol_um)):
                    union(n_t + i, n_t + j)

        # Wire-to-terminal unions on matching layer + bbox-contains-point.
        for ti, (tx, ty, tlayer) in enumerate(net_terms):
            for wi, w in enumerate(net_wires):
                if w.layer != tlayer:
                    continue
                if _touches(w.x0, w.y0, w.x1, w.y1, tx, ty, tol=tol_um):
                    union(ti, n_t + wi)

        # Net counts when all terminals share a root.
        roots = {find(i) for i in range(n_t)}
        if len(roots) == 1:
            score += 1.0
    return float(score)


__all__ = [
    "DEFAULT_TOUCH_TOL_UM",
    "compute_connectivity_score",
    "compute_electrical_score",
]
