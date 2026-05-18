"""
layout_gen.rl.env.reward — phase-aware composite reward.

The reward function has to handle three structurally different episode
phases:

  * **PLACE** — the policy adds devices to a (growing) layout. Each
    placement *adds geometry*, which on a real DRC tool tends to add
    violations (well/tap/spacing rules don't fully resolve until the
    cell is more complete). Penalising ``Δviolations`` here teaches
    the policy to never place anything — degenerate. PLACE rewards
    success-of-action heavily and damps DRC delta.
  * **ROUTE** — same problem: each segment may introduce small
    spacing violations. Reward each successful segment; damp DRC.
  * **REPAIR** — every action either makes DRC strictly better or
    strictly worse; here ``Δviolations`` is the right signal.

Each phase has its own multiplier on the DRC-related terms, plus a
phase-specific success bonus. Common terms (step penalty, terminal
bonus, invalid / no-change penalties) stay shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from layout_gen.drc.base import DRCViolation


_PHASES = ("place", "route", "repair")


@dataclass
class RewardConfig:
    """All weights for :func:`compute_reward`.

    Phase-specific weights live in dicts keyed by phase name
    (``"place"`` / ``"route"`` / ``"repair"``); missing entries fall
    back to a sensible default at lookup time.
    """

    # ── Per-phase DRC delta weight ────────────────────────────────────
    # REPAIR is the pure-repair task — full weight on Δviolations.
    # PLACE / ROUTE attenuate so adding geometry doesn't get punished
    # for re-resolving long-range rules.
    drc_delta_per_phase: dict[str, float] = field(default_factory=lambda: {
        "place":  0.05,
        "route":  0.20,
        "repair": 1.00,
    })

    # Per-phase value-delta weight — same idea, just for the sum of
    # measured-µm deficits across violations.
    value_delta_per_phase: dict[str, float] = field(default_factory=lambda: {
        "place":  0.0,
        "route":  0.05,
        "repair": 0.05,
    })

    # ── Phase-specific success bonuses ────────────────────────────────
    # Awarded when the action was valid AND changed the state. Damp
    # them as the phase progresses (mostly to discourage no-op spam,
    # but also because PPO tends to over-emphasise reward sources).
    place_success: float = 1.0
    route_success: float = 0.5

    # ── Connectivity ──────────────────────────────────────────────────
    # Multiplier on Δ(per-net-fraction of terminals touched). With
    # connectivity_score in [0, n_nets], a Δ of +1.0 means "an entire
    # net's worth of terminals just got connected by this action".
    # Most useful in ROUTE phase but applied across all phases (a
    # PLACE action that lands a device at the right spot for an
    # already-routed net should count too).
    connectivity_delta: float = 2.0

    # ── Electrical (transitive) connectivity ──────────────────────────
    # Multiplier on Δ(net-completion count from union-find over wires
    # + terminals). Strictly stricter than connectivity_delta — only
    # rewards nets where ALL terminals end up in a single connected
    # component. Use both: connectivity gives the dense gradient,
    # electrical rewards finishing the job.
    electrical_delta: float = 3.0

    # ── Placement-intent alignment ────────────────────────────────────
    # Multiplier on Δ(sum-of-clipped-linears against the YAML's
    # placement_logic directives). Pulls PLACE actions toward gate-
    # aligned / abutted layouts that match the cell template's
    # intended structure rather than discovering an axis from scratch.
    alignment_delta: float = 1.5

    # ── Row-type alignment (NMOS=bottom, PMOS=top) ────────────────────
    # Multiplier on Δ(per-device row-row-match score). Per-device
    # contribution is 1.0 when the device sits at the y of the row its
    # type expects (NMOS → bottom half, PMOS → top half), tapering
    # linearly to 0 at threshold_frac × cell_height misalignment.
    # Closes the documented nand2/nor2 stacking failure mode where the
    # policy puts the wrong-type device at the right-type row without
    # waiting for the DRC missing-nwell signal to penalise it.
    # Set to 0 for analog layouts where row-assignment is free.
    row_delta: float = 1.0

    # ── Half-perimeter wirelength ─────────────────────────────────────
    # Multiplier on Δ(-Σ HPWL_net). HPWL is negated so "shorter is
    # better" (more positive). Δ is mostly negative during PLACE (each
    # placement can only grow a net's bbox), so a positive weight here
    # *penalises* placements that spread connected terminals apart.
    # This is the standard dense placement-quality signal used by
    # MaskPlace / AlphaChip / R-GCN-PPO.
    hpwl_delta: float = 0.5

    # ── Short-circuit penalty (cheap geometric heuristic) ─────────────
    # Multiplier on -Δ(short_count). A new wire that bridges two nets
    # introduces shorts; punish proportional to count. Only fires when
    # delta is non-zero so an already-shorted episode doesn't keep
    # bleeding reward.
    short_delta: float = 2.0

    # ── LVS reward (truth signal via magic + netgen) ──────────────────
    # Per-step delta in number of LVS mismatches reported by
    # MagicNetgenLVSRunner. ``+Δ(prev_mismatches - curr_mismatches)``
    # so reducing mismatches gives positive reward. Only active when
    # the env is constructed with an ``lvs_runner`` (gated on magic +
    # netgen + a reference SPICE netlist being available).
    lvs_delta: float = 1.0
    # Bonus when crossing the LVS-clean threshold (any → 0 mismatches).
    lvs_clean_bonus: float = 5.0

    # ── Common terms ──────────────────────────────────────────────────
    step:       float = 0.05
    terminal:   float = 5.0     # only fires in REPAIR phase
    # Per-invalid-action penalty. Bumped from 0.5 to 2.0 after the
    # autoregressive PLACE coupling exposed a failure mode where the
    # policy spammed invalid PLACE attempts (no-stacking guard
    # rejections) because each attempt only cost 0.5 — cheap vs. the
    # 1.0 bonus for a successful placement that happens to add a
    # violation. 2.0 keeps invalids strictly worse than any
    # commitment.
    invalid:    float = 2.0
    no_change:  float = 0.2

    # ── Lookup helpers ────────────────────────────────────────────────
    def drc_delta_weight(self, phase: str) -> float:
        return float(self.drc_delta_per_phase.get(phase, 1.0))

    def value_delta_weight(self, phase: str) -> float:
        return float(self.value_delta_per_phase.get(phase, 0.0))


@dataclass
class RewardBreakdown:
    """Per-component reward returned alongside the scalar."""
    drc_delta:          float = 0.0
    value_delta:        float = 0.0
    step:               float = 0.0
    terminal:           float = 0.0
    invalid:            float = 0.0
    no_change:          float = 0.0
    place_success:      float = 0.0
    route_success:      float = 0.0
    connectivity_delta: float = 0.0
    alignment_delta:    float = 0.0
    electrical_delta:   float = 0.0
    hpwl_delta:         float = 0.0
    row_delta:          float = 0.0
    short_delta:        float = 0.0
    lvs_delta:          float = 0.0
    lvs_clean_bonus:    float = 0.0

    @property
    def total(self) -> float:
        return (self.drc_delta + self.value_delta + self.step
                + self.terminal + self.invalid + self.no_change
                + self.place_success + self.route_success
                + self.connectivity_delta + self.alignment_delta
                + self.electrical_delta + self.hpwl_delta
                + self.row_delta
                + self.short_delta + self.lvs_delta
                + self.lvs_clean_bonus)

    def to_dict(self) -> dict[str, float]:
        return {
            "drc_delta":          self.drc_delta,
            "value_delta":        self.value_delta,
            "step":               self.step,
            "terminal":           self.terminal,
            "invalid":            self.invalid,
            "no_change":          self.no_change,
            "place_success":      self.place_success,
            "route_success":      self.route_success,
            "connectivity_delta": self.connectivity_delta,
            "alignment_delta":    self.alignment_delta,
            "electrical_delta":   self.electrical_delta,
            "hpwl_delta":         self.hpwl_delta,
            "row_delta":          self.row_delta,
            "short_delta":        self.short_delta,
            "lvs_delta":          self.lvs_delta,
            "lvs_clean_bonus":    self.lvs_clean_bonus,
            "total":              self.total,
        }


def _sum_values(viols: Sequence[DRCViolation]) -> float:
    s = 0.0
    for v in viols:
        if v.value is not None:
            s += max(float(v.value), 0.0)
    return s


def compute_reward(
    *,
    violations_before:    Sequence[DRCViolation],
    violations_after:     Sequence[DRCViolation],
    state_changed:        bool,
    action_valid:         bool,
    phase:                str = "repair",
    config:               RewardConfig | None = None,
    connectivity_before:  float = 0.0,
    connectivity_after:   float = 0.0,
    alignment_before:     float = 0.0,
    alignment_after:      float = 0.0,
    electrical_before:    float = 0.0,
    electrical_after:     float = 0.0,
    hpwl_before:          float = 0.0,
    hpwl_after:           float = 0.0,
    row_before:           float = 0.0,
    row_after:            float = 0.0,
    short_before:         int   = 0,
    short_after:          int   = 0,
    lvs_mismatches_before: int | None = None,
    lvs_mismatches_after:  int | None = None,
) -> RewardBreakdown:
    """Compute reward from before/after violation lists, action flags,
    and the active episode phase.

    Parameters
    ----------
    phase :
        ``"place"``, ``"route"``, or ``"repair"``. Drives the per-phase
        DRC weight and which success bonus (if any) fires.
    connectivity_before, connectivity_after :
        Per-net connectivity score (see :mod:`layout_gen.rl.env.connectivity`)
        sampled before and after the action. The Δ between them
        contributes ``connectivity_delta * (after - before)`` to the
        reward — strongly rewards ROUTE actions that newly touch a
        terminal of the net they claim.
    """
    cfg = config or RewardConfig()
    rb  = RewardBreakdown()

    n_before = len(violations_before)
    n_after  = len(violations_after)
    rb.drc_delta = cfg.drc_delta_weight(phase) * (n_before - n_after)

    sv_before = _sum_values(violations_before)
    sv_after  = _sum_values(violations_after)
    rb.value_delta = cfg.value_delta_weight(phase) * (sv_before - sv_after)

    rb.step = -cfg.step

    # Terminal only fires in REPAIR (the env's own logic also gates this,
    # but we double-check here so a misconfigured caller can't get a free
    # +5 reward by reaching DRC-clean during PLACE on an empty layout).
    if phase == "repair" and n_after == 0 and n_before > 0:
        rb.terminal = cfg.terminal

    # Phase-specific success bonus: encourage the policy to actually
    # add geometry during generative phases. Awarded only when the
    # action was valid AND the geometry-hash actually changed.
    if action_valid and state_changed:
        if phase == "place":
            rb.place_success = cfg.place_success
        elif phase == "route":
            rb.route_success = cfg.route_success

    rb.connectivity_delta = (
        cfg.connectivity_delta * (connectivity_after - connectivity_before)
    )
    rb.alignment_delta = (
        cfg.alignment_delta * (alignment_after - alignment_before)
    )
    rb.electrical_delta = (
        cfg.electrical_delta * (electrical_after - electrical_before)
    )
    rb.hpwl_delta = (
        cfg.hpwl_delta * (hpwl_after - hpwl_before)
    )
    rb.row_delta = (
        cfg.row_delta * (row_after - row_before)
    )

    # Short-circuit penalty (geometric heuristic, always-on).
    rb.short_delta = -cfg.short_delta * (short_after - short_before)

    # LVS reward (truth signal). lvs_mismatches_* are None when the env
    # was constructed without an LVS runner — in that case skip both
    # the delta and the bonus.
    if lvs_mismatches_before is not None and lvs_mismatches_after is not None:
        rb.lvs_delta = (
            cfg.lvs_delta * (lvs_mismatches_before - lvs_mismatches_after)
        )
        if lvs_mismatches_after == 0 and lvs_mismatches_before > 0:
            rb.lvs_clean_bonus = cfg.lvs_clean_bonus

    if not action_valid:
        rb.invalid = -cfg.invalid
    elif not state_changed:
        rb.no_change = -cfg.no_change

    return rb


__all__ = ["RewardConfig", "RewardBreakdown", "compute_reward"]
