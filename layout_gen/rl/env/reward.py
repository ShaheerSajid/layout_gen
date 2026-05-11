"""
layout_gen.rl.env.reward — composite reward for the layout RL env.

Reward terms (Phase 1; LVS / area / topology added in later phases)
------------------------------------------------------------------
* ``r_drc_delta``   : ``+w * (n_before - n_after)``.  Positive for repairs,
                      negative for newly-introduced violations.
* ``r_value_delta`` : ``+w * (sum_value_before - sum_value_after)``.
                      Rewards shrinking the magnitude of remaining
                      violations even when their count is unchanged
                      (dense signal that helps gradient flow).
* ``r_step``        : constant ``-w`` per step — discourages stalling
                      and encourages compact action sequences.
* ``r_terminal``    : ``+w`` when the layout reaches DRC-clean. Anchors
                      the success state.
* ``r_invalid``     : ``-w`` when the action was a structural no-op
                      (target rid not present, or apply raised).
* ``r_no_change``   : ``-w`` when the action ran but the geometry-hash
                      didn't change (e.g. shrinking an already-tiny rect).

All weights are configurable via :class:`RewardConfig`. Defaults are
calibrated so:
  * one repaired violation comfortably outweighs the per-step penalty;
  * an introduced violation hurts more than the per-step penalty saves;
  * value-delta is small relative to count-delta so the policy
    prioritises clearing violations rather than just shrinking them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from layout_gen.drc.base import DRCViolation


@dataclass
class RewardConfig:
    drc_delta:  float = 1.0
    value_delta: float = 0.05
    step:       float = 0.05
    terminal:   float = 5.0
    invalid:    float = 0.5
    no_change:  float = 0.2


@dataclass
class RewardBreakdown:
    """Per-component reward returned alongside the scalar."""
    drc_delta:   float = 0.0
    value_delta: float = 0.0
    step:        float = 0.0
    terminal:    float = 0.0
    invalid:     float = 0.0
    no_change:   float = 0.0

    @property
    def total(self) -> float:
        return (self.drc_delta + self.value_delta + self.step
                + self.terminal + self.invalid + self.no_change)

    def to_dict(self) -> dict[str, float]:
        return {
            "drc_delta":   self.drc_delta,
            "value_delta": self.value_delta,
            "step":        self.step,
            "terminal":    self.terminal,
            "invalid":     self.invalid,
            "no_change":   self.no_change,
            "total":       self.total,
        }


def _sum_values(viols: Sequence[DRCViolation]) -> float:
    s = 0.0
    for v in viols:
        if v.value is not None:
            s += max(float(v.value), 0.0)
    return s


def compute_reward(
    *,
    violations_before: Sequence[DRCViolation],
    violations_after:  Sequence[DRCViolation],
    state_changed:     bool,
    action_valid:      bool,
    config:            RewardConfig | None = None,
) -> RewardBreakdown:
    """Compute reward from before/after violation lists and action flags.

    Parameters
    ----------
    violations_before, violations_after :
        DRC violations before and after applying the action.
    state_changed :
        True iff the geometry-hash of the LayoutState changed during the
        step. False indicates a no-op edit (e.g. shrink past zero clamped
        to a minimum that produced the same geometry, or apply raised
        and the env rolled back).
    action_valid :
        False when the action was structurally invalid (target rid not
        present, decoder returned None). True for all well-formed
        attempts even if they degrade the layout.
    """
    cfg = config or RewardConfig()
    rb  = RewardBreakdown()

    n_before = len(violations_before)
    n_after  = len(violations_after)
    rb.drc_delta = cfg.drc_delta * (n_before - n_after)

    sv_before = _sum_values(violations_before)
    sv_after  = _sum_values(violations_after)
    rb.value_delta = cfg.value_delta * (sv_before - sv_after)

    rb.step = -cfg.step

    if n_after == 0 and n_before > 0:
        rb.terminal = cfg.terminal

    if not action_valid:
        rb.invalid = -cfg.invalid
    elif not state_changed:
        rb.no_change = -cfg.no_change

    return rb


__all__ = ["RewardConfig", "RewardBreakdown", "compute_reward"]
