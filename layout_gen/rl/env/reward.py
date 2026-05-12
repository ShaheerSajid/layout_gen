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

    # ── Common terms ──────────────────────────────────────────────────
    step:       float = 0.05
    terminal:   float = 5.0     # only fires in REPAIR phase
    invalid:    float = 0.5
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

    @property
    def total(self) -> float:
        return (self.drc_delta + self.value_delta + self.step
                + self.terminal + self.invalid + self.no_change
                + self.place_success + self.route_success
                + self.connectivity_delta + self.alignment_delta
                + self.electrical_delta)

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

    if not action_valid:
        rb.invalid = -cfg.invalid
    elif not state_changed:
        rb.no_change = -cfg.no_change

    return rb


__all__ = ["RewardConfig", "RewardBreakdown", "compute_reward"]
