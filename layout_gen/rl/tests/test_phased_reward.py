"""
layout_gen.rl.tests.test_phased_reward — phase-aware reward tests.

Verifies the per-phase weight lookup, the place_success / route_success
bonuses, and the gating that prevents the terminal bonus from firing
during PLACE / ROUTE phases.
"""
from __future__ import annotations

import pytest

from layout_gen.drc.base import DRCViolation

from layout_gen.rl.env.reward import RewardConfig, compute_reward


def _viol(n: int, value: float = 0.05) -> list[DRCViolation]:
    return [
        DRCViolation(rule="x", description="", layer="met1",
                     x=0, y=0, value=value)
        for _ in range(n)
    ]


# ── Per-phase DRC weight ─────────────────────────────────────────────────────

def test_drc_delta_weighted_per_phase():
    cfg = RewardConfig(
        drc_delta_per_phase={"place": 0.0, "route": 0.5, "repair": 1.0},
        place_success=0.0, route_success=0.0,
    )
    # 5 → 3 violations: -2 net change → +2 delta (good).
    before = _viol(5); after = _viol(3)
    place_rb = compute_reward(violations_before=before, violations_after=after,
                               state_changed=True, action_valid=True,
                               phase="place", config=cfg)
    route_rb = compute_reward(violations_before=before, violations_after=after,
                               state_changed=True, action_valid=True,
                               phase="route", config=cfg)
    repair_rb = compute_reward(violations_before=before, violations_after=after,
                                state_changed=True, action_valid=True,
                                phase="repair", config=cfg)
    assert place_rb.drc_delta == pytest.approx(0.0)
    assert route_rb.drc_delta == pytest.approx(1.0)   # 0.5 * 2
    assert repair_rb.drc_delta == pytest.approx(2.0)  # 1.0 * 2


# ── Phase-specific success bonus ─────────────────────────────────────────────

def test_place_success_bonus_only_in_place():
    cfg = RewardConfig(place_success=3.0, route_success=2.0)
    rb_place = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True,
        phase="place", config=cfg,
    )
    rb_route = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True,
        phase="route", config=cfg,
    )
    rb_repair = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True,
        phase="repair", config=cfg,
    )
    assert rb_place.place_success == 3.0
    assert rb_place.route_success == 0.0
    assert rb_route.route_success == 2.0
    assert rb_route.place_success == 0.0
    assert rb_repair.place_success == 0.0
    assert rb_repair.route_success == 0.0


def test_success_bonus_requires_state_change():
    cfg = RewardConfig(place_success=3.0)
    rb = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=False, action_valid=True,
        phase="place", config=cfg,
    )
    assert rb.place_success == 0.0


def test_success_bonus_requires_valid_action():
    cfg = RewardConfig(place_success=3.0)
    rb = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=False,
        phase="place", config=cfg,
    )
    assert rb.place_success == 0.0


# ── Terminal gating ──────────────────────────────────────────────────────────

def test_terminal_only_in_repair_phase():
    cfg = RewardConfig(terminal=10.0)
    # 3 → 0 violations: would normally fire terminal, but only in REPAIR.
    before = _viol(3); after = []
    for phase in ("place", "route"):
        rb = compute_reward(
            violations_before=before, violations_after=after,
            state_changed=True, action_valid=True,
            phase=phase, config=cfg,
        )
        assert rb.terminal == 0.0, f"phase={phase} should suppress terminal"
    rb_repair = compute_reward(
        violations_before=before, violations_after=after,
        state_changed=True, action_valid=True,
        phase="repair", config=cfg,
    )
    assert rb_repair.terminal == 10.0


# ── Total reward sanity ──────────────────────────────────────────────────────

def test_total_includes_all_components():
    cfg = RewardConfig(
        drc_delta_per_phase={"place": 0.5},
        place_success=2.0, step=0.1,
    )
    before = _viol(2); after = _viol(0)
    rb = compute_reward(
        violations_before=before, violations_after=after,
        state_changed=True, action_valid=True,
        phase="place", config=cfg,
    )
    expected = (
        rb.drc_delta + rb.value_delta + rb.step
        + rb.terminal + rb.invalid + rb.no_change
        + rb.place_success + rb.route_success
    )
    assert rb.total == pytest.approx(expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
