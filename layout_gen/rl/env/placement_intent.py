"""
layout_gen.rl.env.placement_intent — score policy placements against the
YAML's placement_logic directives.

The connectivity reward pulls device terminals together but is
**axis-blind** — devices that are gate-aligned vertically (same X,
different Y, the canonical inverter / NAND / NOR layout) score
identically to devices side-by-side at same Y. The YAML, however,
does specify the intended axis via ``placement_logic`` directives
like ``align_gate`` and ``abut_x``. This module turns those directives
into a dense scalar that the env can reward Δ on.

Supported directives (Phase 4 part 5)
-------------------------------------
* ``align_gate`` — target's gate X must equal anchor's gate X
  (Y is free; this is the inverter / NAND / NOR pattern).
* ``abut_x``     — target should sit flush against the anchor along
  the X axis (shared-diffusion abutment used in the 6T bitcell).
* ``origin``     — when a directive carries an explicit ``origin``
  tuple, the named device's gate should land near that point.

Each directive's contribution is a clipped linear in [0, 1]:

    score_i = max(0, 1 − distance_i / threshold)

* For ``align_gate``: distance = ``|gate_x_target − gate_x_anchor|``.
* For ``abut_x``: distance = ``|right_edge_anchor − left_edge_target|``.
* For ``origin``: distance = ``‖gate_target − origin‖``.

Total alignment score = Σ_i score_i. Bounded by the number of scored
directives. Devices not yet placed contribute zero (their absent
terminals make the directive un-scoreable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from layout_gen.synth.loader import PlacementDirective

from layout_gen.rl.topology.parser import TopologyGraph


# Default match tolerance: 0 misalignment → 1.0, threshold µm misalignment → 0.0.
# 0.5 µm is wide enough to give a useful gradient over the cell-width range
# we typically discretise into 8–16 position bins.
DEFAULT_THRESHOLD_UM = 0.5


@dataclass
class DirectiveScore:
    """Per-directive breakdown — useful for debugging and for the
    info dict the env hands the trainer."""
    name:     str
    relation: str
    distance: float
    score:    float


def _gate_position(
    terminals: dict[tuple[int, str], tuple[float, float, str]],
    device_idx: int,
) -> tuple[float, float] | None:
    pos = terminals.get((device_idx, "G"))
    return (pos[0], pos[1]) if pos is not None else None


def _drain_position(
    terminals: dict[tuple[int, str], tuple[float, float, str]],
    device_idx: int,
) -> tuple[float, float] | None:
    pos = terminals.get((device_idx, "D"))
    return (pos[0], pos[1]) if pos is not None else None


def _source_position(
    terminals: dict[tuple[int, str], tuple[float, float, str]],
    device_idx: int,
) -> tuple[float, float] | None:
    pos = terminals.get((device_idx, "S"))
    return (pos[0], pos[1]) if pos is not None else None


def score_alignment(
    topology:   TopologyGraph,
    directives: Sequence[PlacementDirective],
    terminals:  dict[tuple[int, str], tuple[float, float, str]],
    *,
    threshold_um: float = DEFAULT_THRESHOLD_UM,
    breakdown:    list[DirectiveScore] | None = None,
) -> float:
    """Sum-of-clipped-linears alignment score.

    Parameters
    ----------
    topology :
        The cell's :class:`TopologyGraph` (used to resolve device names
        in the directives back to indices).
    directives :
        ``CellTemplate.placement_directives``.
    terminals :
        ``{(device_idx, term_name): (x, y, layer)}`` populated by
        :func:`place_device_full` after each PLACE action.
    threshold_um :
        Distance at which a directive's score drops to zero.
    breakdown :
        Optional list to be appended with one :class:`DirectiveScore`
        per scored directive — for diagnostics in
        :class:`LayoutEnv` info dicts.

    Returns
    -------
    float :
        Σ_i max(0, 1 − distance_i / threshold_um) over scored
        directives. Zero before any devices are placed.
    """
    name_to_idx = topology.device_index()
    total = 0.0
    for d in directives:
        idx = name_to_idx.get(d.name)
        if idx is None:
            continue

        # ── Origin directive: named device should land at this point ──
        if d.origin is not None and not d.relation:
            gp = _gate_position(terminals, idx)
            if gp is None:
                continue
            ox, oy = float(d.origin[0]), float(d.origin[1])
            dx, dy = gp[0] - ox, gp[1] - oy
            dist = (dx * dx + dy * dy) ** 0.5
            s = max(0.0, 1.0 - dist / threshold_um)
            if breakdown is not None:
                breakdown.append(DirectiveScore(d.name, "origin", dist, s))
            total += s
            continue

        anchor_idx = name_to_idx.get(d.relative_to) if d.relative_to else None
        if anchor_idx is None:
            continue

        if d.relation == "align_gate":
            gp_t = _gate_position(terminals, idx)
            gp_a = _gate_position(terminals, anchor_idx)
            if gp_t is None or gp_a is None:
                continue
            dist = abs(gp_t[0] - gp_a[0])
            s = max(0.0, 1.0 - dist / threshold_um)
            if breakdown is not None:
                breakdown.append(DirectiveScore(d.name, "align_gate", dist, s))
            total += s

        elif d.relation == "abut_x":
            # We approximate edges using the source/drain X positions,
            # which sit at the outer edges of the device's S/D rails.
            # anchor's "right edge" = max(S.x, D.x) for the anchor
            # target's "left edge"  = min(S.x, D.x) for the target
            sa = _source_position(terminals, anchor_idx)
            da = _drain_position(terminals, anchor_idx)
            st = _source_position(terminals, idx)
            dt = _drain_position(terminals, idx)
            if not all((sa, da, st, dt)):
                continue
            anchor_right = max(sa[0], da[0])
            target_left  = min(st[0], dt[0])
            dist = abs(anchor_right - target_left)
            s = max(0.0, 1.0 - dist / threshold_um)
            if breakdown is not None:
                breakdown.append(DirectiveScore(d.name, "abut_x", dist, s))
            total += s
    return float(total)


# ── Row-type alignment (NMOS=bottom row, PMOS=top row) ─────────────────────

def compute_row_score(
    topology:        TopologyGraph,
    placed_origins:  dict[int, tuple[float, float]],
    cell_height_um:  float,
    *,
    margin_frac:     float = 0.25,
) -> float:
    """Per-device row-alignment score.

    For each placed device, awards full credit when the device's
    origin-y sits anywhere in the *correct* row for its type, and a
    linearly-decaying partial credit when it sits in the wrong row.
    In a sky130-style digital standard cell:

      * **NMOS** (``device.in_nwell == False``) → bottom half
        (``y < cell_height / 2``).
      * **PMOS** (``device.in_nwell == True``)  → top half
        (``y ≥ cell_height / 2``).

    Per-device contribution::

        if on_correct_side(y): s_i = 1.0
        else:                  s_i = max(0, 1 − margin_frac⁻¹ ·
                                       (distance_into_wrong_side / cell_height))

    The wrong-side decay reaches 0 at ``margin_frac × cell_height``
    past the row boundary (default 25% of cell height). Total =
    Σ_i s_i, bounded by the number of placed devices.

    Why this shape
    --------------
    The canonical synth layout puts NMOS at ``y_origin = 0`` (bottom
    rail) and PMOS near ``y_origin = cell_h − device_h``; both are
    *anywhere* in the correct half, so they should both score 1.0.
    A "distance from centre of row" formula would unfairly penalise
    these correct layouts. The wrong-side decay still gives the
    policy a gradient when it lands a device in the wrong half.

    PD-correct for the digital-stdcell flow; set ``row_delta=0`` in
    :class:`RewardConfig` for analog layouts where row assignment is
    free.
    """
    if cell_height_um <= 0:
        return 0.0
    midline = cell_height_um / 2.0
    decay = cell_height_um * margin_frac
    if decay <= 0:
        return 0.0
    total = 0.0
    for d_idx, (_x, y) in placed_origins.items():
        if d_idx >= topology.n_devices:
            continue
        device = topology.devices[d_idx]
        wants_top = bool(device.in_nwell)
        if wants_top:
            on_correct = float(y) >= midline
            distance_wrong = max(0.0, midline - float(y))
        else:
            on_correct = float(y) < midline
            distance_wrong = max(0.0, float(y) - midline)
        if on_correct:
            total += 1.0
        else:
            total += max(0.0, 1.0 - distance_wrong / decay)
    return float(total)


__all__ = [
    "DEFAULT_THRESHOLD_UM",
    "DirectiveScore",
    "score_alignment",
    "compute_row_score",
]
