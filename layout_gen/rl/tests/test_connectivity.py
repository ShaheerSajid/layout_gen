"""
layout_gen.rl.tests.test_connectivity — connectivity scoring tests.

Verifies the per-net heuristic, the reward delta, and end-to-end env
behaviour:
  * `compute_connectivity_score` is 0 on a layout with no wires.
  * It rises by 1/n_terms when a wire newly touches a terminal.
  * `compute_reward` adds `connectivity_delta` proportional to the
    score change.
  * `LayoutEnv._terminals` is populated by PLACE actions.
  * An env step that adds a wire near a placed terminal earns a
    positive `connectivity_delta` reward component.
"""
from __future__ import annotations

import numpy as np
import pytest

from layout_gen.pdk import load_pdk
from layout_gen.synth.geo.state import LayoutState
from layout_gen.synth.loader import load_template

from layout_gen.rl.env.action_space import REPAIR_KINDS, PLACE_KINDS, ROUTE_KINDS
from layout_gen.rl.env.connectivity import compute_connectivity_score
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.env.place_action import TransistorCache, place_device_full
from layout_gen.rl.env.reward import RewardConfig, compute_reward
from layout_gen.rl.env.route_action import add_route_segment
from layout_gen.rl.topology import graph_from_template
from layout_gen.rl.topology.parser import (
    DeviceNode, NetEdge, TopologyGraph,
)


# ── Module fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rules():
    return load_pdk()


@pytest.fixture(scope="module")
def cache(rules):
    return TransistorCache(rules)


# ── Pure scoring tests ───────────────────────────────────────────────────────

def _toy_graph():
    devices = [
        DeviceNode("N", "nmos", "planar_mosfet", w_um=0.5, l_um=0.15,
                   fingers=0, in_nwell=False),
        DeviceNode("P", "pmos", "planar_mosfet", w_um=0.5, l_um=0.15,
                   fingers=0, in_nwell=True),
    ]
    nets = [
        NetEdge(name="OUT", net_type="signal", rail="none", layer_hint="",
                connections=[(0, "D"), (1, "D")]),
    ]
    return TopologyGraph(cell_name="toy", devices=devices, nets=nets)


def test_score_is_zero_with_no_wires():
    graph = _toy_graph()
    state = LayoutState()
    terms = {(0, "D"): (1.0, 0.5, "li1"), (1, "D"): (3.0, 0.5, "li1")}
    s = compute_connectivity_score(state, graph, terms)
    assert s == 0.0


def test_score_one_terminal_touched():
    graph = _toy_graph()
    state = LayoutState()
    add_route_segment(state, layer="met1",
                       x_um=0.95, y_um=0.45, w_um=0.20, h_um=0.20,
                       net_name="OUT")
    terms = {(0, "D"): (1.0, 0.5, "li1"), (1, "D"): (3.0, 0.5, "li1")}
    s = compute_connectivity_score(state, graph, terms)
    # 1 of 2 terminals touched → 0.5
    assert s == pytest.approx(0.5)


def test_score_full_when_both_touched():
    graph = _toy_graph()
    state = LayoutState()
    add_route_segment(state, layer="met1",
                       x_um=0.95, y_um=0.45, w_um=0.20, h_um=0.20,
                       net_name="OUT")
    add_route_segment(state, layer="met1",
                       x_um=2.95, y_um=0.45, w_um=0.20, h_um=0.20,
                       net_name="OUT")
    terms = {(0, "D"): (1.0, 0.5, "li1"), (1, "D"): (3.0, 0.5, "li1")}
    s = compute_connectivity_score(state, graph, terms)
    assert s == pytest.approx(1.0)


def test_score_ignores_wires_with_wrong_net_tag():
    graph = _toy_graph()
    state = LayoutState()
    # Right position, wrong net tag — must not count.
    add_route_segment(state, layer="met1",
                       x_um=0.95, y_um=0.45, w_um=0.20, h_um=0.20,
                       net_name="VDD")
    terms = {(0, "D"): (1.0, 0.5, "li1"), (1, "D"): (3.0, 0.5, "li1")}
    assert compute_connectivity_score(state, graph, terms) == 0.0


# ── Reward delta ─────────────────────────────────────────────────────────────

def test_reward_includes_connectivity_delta():
    cfg = RewardConfig(
        connectivity_delta=2.0,
        place_success=0.0, route_success=0.0,
        drc_delta_per_phase={"route": 0.0},
    )
    rb = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True,
        phase="route", config=cfg,
        connectivity_before=0.0, connectivity_after=1.0,
    )
    assert rb.connectivity_delta == pytest.approx(2.0)
    # total = drc_delta + value_delta + step + ... + connectivity_delta
    expected = (rb.drc_delta + rb.value_delta + rb.step
                + rb.terminal + rb.invalid + rb.no_change
                + rb.place_success + rb.route_success
                + rb.connectivity_delta)
    assert rb.total == pytest.approx(expected)


def test_reward_connectivity_negative_when_score_drops():
    cfg = RewardConfig(connectivity_delta=2.0)
    rb = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True,
        phase="route", config=cfg,
        connectivity_before=0.5, connectivity_after=0.0,
    )
    assert rb.connectivity_delta == pytest.approx(-1.0)


# ── End-to-end env behaviour ─────────────────────────────────────────────────

class _NoOpDRC:
    def run(self, state): return []
    def count(self, state): return 0
    def stats(self): return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}
    def clear(self): pass


def _inverter_env(rules, cache, *, max_place_steps=4, max_route_steps=4,
                   max_steps=12) -> LayoutEnv:
    g = graph_from_template(
        load_template("inverter"),
        cell_params={"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}},
    )
    return LayoutEnv(
        drc=_NoOpDRC(),
        poly_cap=64, viol_cap=8, target_cap=64, mag_bins=8,
        max_steps=max_steps,
        enable_place=True,
        topology_graph=g, transistor_cache=cache,
        device_cap=8, x_bins=8, y_bins=8,
        cell_width_um=4.0, cell_height_um=4.0,
        max_place_steps=max_place_steps,
        enable_route=True,
        net_cap=8, route_x_bins=8, route_y_bins=8,
        route_w_bins=4, route_h_bins=4,
        max_route_steps=max_route_steps,
    )


def test_env_records_terminal_positions_after_place(rules, cache):
    env = _inverter_env(rules, cache)
    env.reset()
    # Place device 0 (N).
    action = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
    action[0] = len(REPAIR_KINDS)        # place_device kind
    action[6] = 0
    action[7] = action[8] = 4
    action[9] = 0
    env.step(action)
    # _terminals must now have G/S/D entries for device 0.
    keys = {k for k in env._terminals if k[0] == 0}
    assert {"G", "S", "D"}.issubset({k[1] for k in keys})


def test_env_score_rises_when_wire_lands_on_terminal(rules, cache):
    """Drop a wire segment manually right on top of a placed terminal —
    the env's connectivity score should rise by exactly 1 / n_terminals
    of that net (here: 1/1 since net OUT touches one terminal in this
    scenario, the only one we place)."""
    env = _inverter_env(rules, cache)
    env.reset()
    # PLACE device 0 (NMOS), so its terminals exist.
    action = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
    action[0] = len(REPAIR_KINDS)
    action[6] = 0
    action[7] = action[8] = 4
    action[9] = 0
    env.step(action)

    g = env._topology_graph
    # Find the net the NMOS's drain is on (inverter: D -> OUT).
    net_name_for_d = None
    for net in g.nets:
        if (0, "D") in net.connections:
            net_name_for_d = net.name
            break
    assert net_name_for_d is not None

    px, py, _ = env._terminals[(0, "D")]

    score_before = env._connectivity_score()
    # Place a wire centered exactly on the terminal.
    add_route_segment(
        env._state,
        layer="met1",
        x_um=px - 0.10, y_um=py - 0.10, w_um=0.20, h_um=0.20,
        net_name=net_name_for_d,
    )
    score_after = env._connectivity_score()
    assert score_after > score_before, (
        f"score did not rise after wire on terminal: "
        f"before={score_before:.3f} after={score_after:.3f}"
    )


# ── Electrical (transitive) connectivity ────────────────────────────────────

def _two_term_net_at(x1, x2, *, layer="li1") -> tuple:
    g = _toy_graph()
    terms = {
        (0, "D"): (x1, 0.5, layer),
        (1, "D"): (x2, 0.5, layer),
    }
    return g, terms


def test_electrical_zero_when_terminals_only_individually_touched():
    """Two disjoint wires each touching one terminal → connectivity=1.0
    but electrical=0.0 (no path between them)."""
    from layout_gen.rl.env.connectivity import compute_electrical_score
    g, terms = _two_term_net_at(1.0, 3.0)
    state = LayoutState()
    add_route_segment(state, layer="li1",
                       x_um=0.95, y_um=0.45, w_um=0.10, h_um=0.10,
                       net_name="OUT")
    add_route_segment(state, layer="li1",
                       x_um=2.95, y_um=0.45, w_um=0.10, h_um=0.10,
                       net_name="OUT")
    conn = compute_connectivity_score(state, g, terms)
    elec = compute_electrical_score(state, g, terms)
    assert conn == pytest.approx(1.0)   # both terminals touched
    assert elec == 0.0                  # but not connected to each other


def test_electrical_full_when_one_segment_spans_both_terminals():
    """A single wire long enough to overlap both terminals connects
    them. electrical=1.0."""
    from layout_gen.rl.env.connectivity import compute_electrical_score
    g, terms = _two_term_net_at(1.0, 3.0)
    state = LayoutState()
    add_route_segment(state, layer="li1",
                       x_um=0.5, y_um=0.45, w_um=3.0, h_um=0.10,
                       net_name="OUT")
    elec = compute_electrical_score(state, g, terms)
    assert elec == pytest.approx(1.0)


def test_electrical_chain_of_overlapping_segments():
    """Three overlapping wires forming a connected chain between two
    terminals → electrical=1.0."""
    from layout_gen.rl.env.connectivity import compute_electrical_score
    g, terms = _two_term_net_at(0.0, 3.0)
    state = LayoutState()
    # Three overlapping segments of li1 forming a continuous strip.
    add_route_segment(state, layer="li1",
                       x_um=-0.10, y_um=0.45, w_um=1.20, h_um=0.10,
                       net_name="OUT")
    add_route_segment(state, layer="li1",
                       x_um=1.00,  y_um=0.45, w_um=1.20, h_um=0.10,
                       net_name="OUT")
    add_route_segment(state, layer="li1",
                       x_um=2.00,  y_um=0.45, w_um=1.20, h_um=0.10,
                       net_name="OUT")
    elec = compute_electrical_score(state, g, terms)
    assert elec == pytest.approx(1.0)


def test_electrical_rejects_cross_layer_unions():
    """A met1 wire and an li1 wire sharing a bbox don't union (no via
    primitive). Two terminals on li1 connected only via a non-li1 wire
    should NOT score."""
    from layout_gen.rl.env.connectivity import compute_electrical_score
    g, terms = _two_term_net_at(1.0, 3.0)
    state = LayoutState()
    add_route_segment(state, layer="met1",   # wrong layer for li1 terminals
                       x_um=0.5, y_um=0.45, w_um=3.0, h_um=0.10,
                       net_name="OUT")
    elec = compute_electrical_score(state, g, terms)
    assert elec == 0.0


def test_electrical_singleton_net_counts_as_connected():
    """A net with only one terminal is trivially connected (no other
    terminals to fail to reach). Scores 1.0 once that one terminal
    exists, even with zero wires."""
    from layout_gen.rl.env.connectivity import compute_electrical_score
    devs = [
        DeviceNode("N", "nmos", "planar_mosfet", 0.5, 0.15, 0, False),
    ]
    nets = [
        NetEdge(name="VSS", net_type="power", rail="bottom", layer_hint="",
                connections=[(0, "S")]),
    ]
    g = TopologyGraph(cell_name="x", devices=devs, nets=nets)
    state = LayoutState()
    terms = {(0, "S"): (0.0, 0.0, "li1")}
    assert compute_electrical_score(state, g, terms) == pytest.approx(1.0)


# ── HPWL (half-perimeter wirelength) ────────────────────────────────────────

def test_hpwl_zero_when_no_terminals_placed():
    from layout_gen.rl.env.connectivity import compute_hpwl_score
    g = _toy_graph()
    state = LayoutState()
    assert compute_hpwl_score(state, g, {}) == 0.0


def test_hpwl_zero_with_single_terminal_on_net():
    """A net with only one placed terminal has no bbox → 0 HPWL."""
    from layout_gen.rl.env.connectivity import compute_hpwl_score
    g = _toy_graph()
    state = LayoutState()
    terms = {(0, "D"): (1.0, 0.5, "li1")}  # net OUT has 1/2 placed
    assert compute_hpwl_score(state, g, terms) == 0.0


def test_hpwl_negated_sum_of_bbox_half_perimeters():
    """Two terminals on net OUT at x=1,y=0.5 and x=3,y=2.0 →
    half-perimeter = 2 + 1.5 = 3.5; score = -3.5."""
    from layout_gen.rl.env.connectivity import compute_hpwl_score
    g = _toy_graph()
    state = LayoutState()
    terms = {
        (0, "D"): (1.0, 0.5, "li1"),
        (1, "D"): (3.0, 2.0, "li1"),
    }
    assert compute_hpwl_score(state, g, terms) == pytest.approx(-3.5)


def test_hpwl_score_rises_as_terminals_cluster():
    """Moving the second terminal closer to the first → score rises
    (less negative)."""
    from layout_gen.rl.env.connectivity import compute_hpwl_score
    g = _toy_graph()
    far  = {(0, "D"): (0.0, 0.0, "li1"), (1, "D"): (5.0, 5.0, "li1")}
    near = {(0, "D"): (0.0, 0.0, "li1"), (1, "D"): (1.0, 1.0, "li1")}
    s_far  = compute_hpwl_score(LayoutState(), g, far)
    s_near = compute_hpwl_score(LayoutState(), g, near)
    assert s_near > s_far  # near is "better" (closer to 0)
    assert s_far  == pytest.approx(-10.0)
    assert s_near == pytest.approx(-2.0)


def test_reward_includes_hpwl_delta():
    cfg = RewardConfig(
        hpwl_delta=0.5,
        place_success=0.0, route_success=0.0,
        drc_delta_per_phase={"place": 0.0},
    )
    rb = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True,
        phase="place", config=cfg,
        hpwl_before=-5.0, hpwl_after=-3.0,
    )
    # Δhpwl = (-3.0) - (-5.0) = +2.0; weighted = 1.0
    assert rb.hpwl_delta == pytest.approx(1.0)


def test_reward_hpwl_delta_negative_when_score_drops():
    """Placing a device that grows a net's bbox → score becomes more
    negative → hpwl_delta < 0."""
    cfg = RewardConfig(hpwl_delta=1.0)
    rb = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True,
        phase="place", config=cfg,
        hpwl_before=-2.0, hpwl_after=-5.0,
    )
    assert rb.hpwl_delta == pytest.approx(-3.0)


def test_env_exposes_hpwl_in_info(rules, cache):
    """Env.info should include the per-step HPWL score."""
    env = _inverter_env(rules, cache)
    obs, info = env.reset()
    assert "hpwl" in info
    # Empty layout: no terminals placed → 0.
    assert info["hpwl"] == 0.0


def test_env_hpwl_falls_after_placing_far_apart_devices(rules, cache):
    """Place two devices on the same net far apart → hpwl drops (more
    negative)."""
    env = _inverter_env(rules, cache)
    env.reset()
    # PLACE device 0 (NMOS) at one corner.
    action = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
    action[0] = len(REPAIR_KINDS)
    action[6] = 0
    action[7] = 1; action[8] = 1
    action[9] = 0
    _, _, _, _, info0 = env.step(action)
    hpwl0 = info0["hpwl"]

    # PLACE device 1 (PMOS) at the opposite corner.
    action = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
    action[0] = len(REPAIR_KINDS)
    action[6] = 1
    action[7] = 6; action[8] = 6
    action[9] = 0
    _, _, _, _, info1 = env.step(action)
    hpwl1 = info1["hpwl"]

    # After placing the second device, at least one shared net (OUT)
    # has both terminals → bbox materialises → hpwl strictly drops.
    assert hpwl1 < hpwl0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
