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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
