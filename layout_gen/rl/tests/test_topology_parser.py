"""
layout_gen.rl.tests.test_topology_parser — TopologyGraph & encoder layout.

Verifies:
  * graph_from_template handles real cell YAMLs (inverter, bit_cell_6t).
  * Device + net feature vectors land in the documented index slots.
  * Adjacency matches the YAML's terminal → net mapping.
  * Sizing fallbacks pick up cell_params and "_defaults" overrides.
"""
from __future__ import annotations

import pytest

from layout_gen.synth.loader import load_template

from layout_gen.rl.topology.parser import (
    DEVICE_FEAT_DIM, NET_FEAT_DIM,
    DEVICE_TYPES, NET_TYPES, RAIL_POSITIONS,
    device_feature_indices, net_feature_indices,
    encode_device, encode_net,
    graph_from_template,
)


# ── Real-template parsing ────────────────────────────────────────────────────

def test_graph_from_inverter_yaml():
    tmpl = load_template("inverter")
    g = graph_from_template(tmpl)

    assert g.cell_name == "cmos_inverter"
    assert g.n_devices == 2
    assert g.n_nets == 4   # VDD, GND, IN, OUT

    names = {d.name for d in g.devices}
    assert names == {"N", "P"}

    types = {d.name: d.device_type for d in g.devices}
    assert types == {"N": "nmos", "P": "pmos"}


def test_graph_from_bit_cell_6t_yaml():
    tmpl = load_template("bit_cell_6t")
    g = graph_from_template(tmpl)

    assert g.n_devices == 6                     # 4 N + 2 P
    n_pmos = sum(1 for d in g.devices if d.device_type == "pmos")
    n_nmos = sum(1 for d in g.devices if d.device_type == "nmos")
    assert n_pmos == 2 and n_nmos == 4

    # Net count: VDD, GND, Q, Q_, WL, BL, BL_
    assert g.n_nets == 7

    # Adjacency: every Q net connects PD_L.D + PU_L.D + PG_L.S
    di = g.device_index()
    by_name = {n.name: n for n in g.nets}
    q_conns = {(d_idx, term) for (d_idx, term) in by_name["Q"].connections}
    assert (di["PD_L"], "D") in q_conns
    assert (di["PU_L"], "D") in q_conns
    assert (di["PG_L"], "S") in q_conns


def test_layer_hint_from_routing_section():
    """The bit_cell_6t YAML's routing section sets BL/BL_ on met2 and WL on met1."""
    tmpl = load_template("bit_cell_6t")
    g = graph_from_template(tmpl)
    by_name = {n.name: n for n in g.nets}
    assert by_name["BL"].layer_hint == "met2"
    assert by_name["BL_"].layer_hint == "met2"
    assert by_name["WL"].layer_hint == "met1"
    # Q has strategy: local — no layer hint
    assert by_name["Q"].layer_hint == ""


# ── Sizing fallbacks ─────────────────────────────────────────────────────────

def test_sizing_falls_back_to_cell_defaults():
    tmpl = load_template("inverter")
    params = {"_defaults": {"w_N": 0.5, "w_P": 0.7, "l": 0.18}}
    g = graph_from_template(tmpl, cell_params=params)
    by_name = {d.name: d for d in g.devices}
    # N is nmos → w_N
    assert by_name["N"].w_um == pytest.approx(0.5)
    # P is pmos → w_P
    assert by_name["P"].w_um == pytest.approx(0.7)
    # Both → l
    assert by_name["N"].l_um == pytest.approx(0.18)
    assert by_name["P"].l_um == pytest.approx(0.18)


# ── Feature vector layout ────────────────────────────────────────────────────

def test_device_feature_layout_matches_indices():
    idx = device_feature_indices()
    assert idx["type_nmos"] == DEVICE_TYPES.index("nmos")
    assert idx["type_pmos"] == DEVICE_TYPES.index("pmos")
    assert idx["w_um"] == DEVICE_FEAT_DIM - 4
    assert idx["fingers"] == DEVICE_FEAT_DIM - 2
    assert idx["in_nwell"] == DEVICE_FEAT_DIM - 1


def test_net_feature_layout_matches_indices():
    idx = net_feature_indices()
    assert idx["type_power"] == NET_TYPES.index("power")
    assert idx["rail_top"] == len(NET_TYPES) + RAIL_POSITIONS.index("top")
    # layer_met1 should land in the layer-role slice
    layer_base = len(NET_TYPES) + len(RAIL_POSITIONS)
    assert idx["layer_met1"] >= layer_base


def test_encode_device_one_hot_for_nmos():
    from layout_gen.rl.topology.parser import DeviceNode
    d = DeviceNode(name="N1", device_type="nmos", template="planar_mosfet",
                   w_um=0.5, l_um=0.15, fingers=2, in_nwell=False)
    feats = encode_device(d)
    assert len(feats) == DEVICE_FEAT_DIM
    assert feats[DEVICE_TYPES.index("nmos")] == 1.0
    assert feats[DEVICE_TYPES.index("pmos")] == 0.0


def test_encode_net_one_hot_for_power_top():
    from layout_gen.rl.topology.parser import NetEdge
    n = NetEdge(name="VDD", net_type="power", rail="top",
                layer_hint="met1", connections=[])
    feats = encode_net(n)
    assert len(feats) == NET_FEAT_DIM
    assert feats[NET_TYPES.index("power")] == 1.0
    assert feats[len(NET_TYPES) + RAIL_POSITIONS.index("top")] == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
