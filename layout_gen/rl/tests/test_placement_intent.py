"""
layout_gen.rl.tests.test_placement_intent — alignment-score tests.

Verifies:
  * align_gate score is 1.0 when gates share an X coord, 0 when far,
    a clipped linear in between.
  * abut_x score rewards diff-edge proximity.
  * origin score keys on the named device's gate position.
  * The env's alignment reward fires Δ when a PLACE action moves a
    device closer to the YAML-specified gate-aligned position.
"""
from __future__ import annotations

import numpy as np
import pytest

from layout_gen.pdk import load_pdk
from layout_gen.synth.loader import PlacementDirective, load_template

from layout_gen.rl.env.action_space import REPAIR_KINDS
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.env.place_action import TransistorCache
from layout_gen.rl.env.placement_intent import (
    DEFAULT_THRESHOLD_UM, DirectiveScore, score_alignment,
)
from layout_gen.rl.topology import graph_from_template
from layout_gen.rl.topology.parser import (
    DeviceNode, NetEdge, TopologyGraph,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _two_device_graph():
    devs = [
        DeviceNode("N", "nmos", "planar_mosfet", 0.5, 0.15, 0, False),
        DeviceNode("P", "pmos", "planar_mosfet", 0.5, 0.15, 0, True),
    ]
    return TopologyGraph(cell_name="t", devices=devs, nets=[])


def _terms_with_gates(gn_x: float, gp_x: float, *,
                       gn_y: float = 1.0, gp_y: float = 2.0):
    return {
        (0, "G"): (gn_x, gn_y, "poly"),
        (1, "G"): (gp_x, gp_y, "poly"),
    }


# ── Pure scoring tests ───────────────────────────────────────────────────────

def test_align_gate_full_score_when_x_match():
    g = _two_device_graph()
    directive = PlacementDirective(
        name="P", relative_to="N", relation="align_gate",
        alignment="gate", orientation="R0",
    )
    terms = _terms_with_gates(gn_x=1.5, gp_x=1.5)
    s = score_alignment(g, [directive], terms)
    assert s == pytest.approx(1.0)


def test_align_gate_zero_when_far():
    g = _two_device_graph()
    directive = PlacementDirective(
        name="P", relative_to="N", relation="align_gate", alignment="gate",
    )
    # Distance >> threshold (0.5) → clipped to zero.
    terms = _terms_with_gates(gn_x=0.0, gp_x=2.0)
    s = score_alignment(g, [directive], terms)
    assert s == 0.0


def test_align_gate_clipped_linear():
    g = _two_device_graph()
    directive = PlacementDirective(
        name="P", relative_to="N", relation="align_gate", alignment="gate",
    )
    # Half-threshold misalignment → score 0.5
    terms = _terms_with_gates(gn_x=0.0, gp_x=DEFAULT_THRESHOLD_UM / 2)
    s = score_alignment(g, [directive], terms)
    assert s == pytest.approx(0.5)


def test_align_gate_y_unconstrained():
    """Y differences must NOT affect the align_gate score (it only
    constrains X)."""
    g = _two_device_graph()
    directive = PlacementDirective(
        name="P", relative_to="N", relation="align_gate", alignment="gate",
    )
    s = score_alignment(g, [directive],
                         _terms_with_gates(1.5, 1.5, gn_y=0.0, gp_y=10.0))
    assert s == pytest.approx(1.0)


def test_origin_directive_scores_distance_to_target_point():
    g = _two_device_graph()
    directive = PlacementDirective(
        name="N", relative_to="", relation="", origin=(0.0, 0.0),
    )
    terms = _terms_with_gates(gn_x=0.0, gp_x=99.0,
                               gn_y=0.0, gp_y=0.0)
    s = score_alignment(g, [directive], terms)
    # N's gate at (0,0) → exactly on origin → score 1.0
    assert s == pytest.approx(1.0)


def test_breakdown_contains_one_entry_per_scored_directive():
    g = _two_device_graph()
    directives = [
        PlacementDirective(name="N", relative_to="",  relation="",
                           origin=(0.0, 0.0)),
        PlacementDirective(name="P", relative_to="N", relation="align_gate",
                           alignment="gate"),
    ]
    breakdown: list[DirectiveScore] = []
    score_alignment(g, directives,
                     _terms_with_gates(0.0, 0.0, gn_y=0.0, gp_y=2.0),
                     breakdown=breakdown)
    assert len(breakdown) == 2
    assert {b.relation for b in breakdown} == {"origin", "align_gate"}


# ── End-to-end env behaviour ─────────────────────────────────────────────────

class _FakeDRC:
    def run(self, state): return []
    def count(self, state): return 0
    def stats(self): return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}
    def clear(self): pass


@pytest.fixture(scope="module")
def rules():
    return load_pdk()


@pytest.fixture(scope="module")
def cache(rules):
    return TransistorCache(rules)


def test_env_alignment_reward_fires_on_placement(rules, cache):
    """The inverter YAML's directive 'P align_gate N' should fire a
    positive alignment_delta when a PLACE action puts P's gate close
    to N's."""
    template = load_template("inverter")
    g = graph_from_template(
        template,
        cell_params={"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}},
    )
    env = LayoutEnv(
        drc=_FakeDRC(),
        poly_cap=64, viol_cap=8, target_cap=64, mag_bins=8,
        max_steps=10,
        enable_place=True,
        topology_graph=g, transistor_cache=cache,
        device_cap=8, x_bins=8, y_bins=8,
        cell_width_um=4.0, cell_height_um=4.0,
        max_place_steps=4,
        placement_directives=template.placement_directives,
    )
    env.reset()

    helper = env._action_helper

    def _place(d_idx: int, x_bin: int, y_bin: int) -> dict:
        action = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
        action[0] = len(REPAIR_KINDS)   # place_device
        action[6] = d_idx
        action[7] = x_bin
        action[8] = y_bin
        action[9] = 0
        _, _, _, _, info = env.step(action)
        return info

    # 1) Place N (device 0) at bin (4, 2). alignment_delta on this
    #    step should be ≥ 0 (origin directive may now satisfy).
    info_n = _place(0, 4, 2)
    assert info_n["reward"]["alignment_delta"] >= 0.0

    # 2) Place P (device 1) at the SAME x_bin → gates aligned in X
    #    (modulo orientation differences). alignment_delta should rise.
    info_p = _place(1, 4, 6)
    assert info_p["reward"]["alignment_delta"] > 0.0, info_p["reward"]


# ── Row-type alignment ─────────────────────────────────────────────────────

def test_row_score_nmos_at_bottom_pmos_at_top():
    """Cell height 2 µm; NMOS expected at y=0.5, PMOS at y=1.5."""
    from layout_gen.rl.env.placement_intent import compute_row_score
    g = _two_device_graph()
    origins = {0: (0.0, 0.5), 1: (0.0, 1.5)}   # canonical layout
    s = compute_row_score(g, origins, cell_height_um=2.0)
    assert s == pytest.approx(2.0)             # 1.0 per device


def test_row_score_zero_when_swapped():
    """NMOS at top + PMOS at bottom → both miss by cell_h/2 = 1.0,
    threshold = 0.25 * cell_h = 0.5 → both score 0."""
    from layout_gen.rl.env.placement_intent import compute_row_score
    g = _two_device_graph()
    origins = {0: (0.0, 1.5), 1: (0.0, 0.5)}   # swapped rows
    s = compute_row_score(g, origins, cell_height_um=2.0)
    assert s == pytest.approx(0.0)


def test_row_score_partial_when_off_centre():
    """NMOS at y=0.6 (0.1 µm off-centre); threshold=0.5 → score 0.8."""
    from layout_gen.rl.env.placement_intent import compute_row_score
    g = _two_device_graph()
    origins = {0: (0.0, 0.6)}                  # NMOS only, slightly high
    s = compute_row_score(g, origins, cell_height_um=2.0)
    assert s == pytest.approx(0.8, abs=1e-6)


def test_row_score_zero_when_no_origins():
    from layout_gen.rl.env.placement_intent import compute_row_score
    g = _two_device_graph()
    assert compute_row_score(g, {}, cell_height_um=2.0) == 0.0


def test_row_score_zero_height_returns_zero():
    """Degenerate cell_height shouldn't crash."""
    from layout_gen.rl.env.placement_intent import compute_row_score
    g = _two_device_graph()
    s = compute_row_score(g, {0: (0.0, 0.0)}, cell_height_um=0.0)
    assert s == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
