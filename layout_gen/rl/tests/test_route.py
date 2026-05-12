"""
layout_gen.rl.tests.test_route — ROUTE phase end-to-end smoke tests.

Verifies:
  * `add_route_segment` writes one rect with the right layer / coords
    and tags it with the net name (so LVS-style introspection can
    later trace connectivity).
  * `ActionSpace(enable_route=True)` adds the 6 ROUTE dims after the
    PLACE block (kind grows by N_ROUTE_KINDS).
  * `action_mask_for(phase="route")` enables only ROUTE kinds and
    constrains the net dim to topology-known nets.
  * `LayoutEnv` with both PLACE and ROUTE goes through the full
    PLACE → ROUTE → REPAIR machine and ROUTE actions correctly
    materialise net-tagged metal rects.
  * `MaskableLayoutPolicy(enable_route=True)` produces logits whose
    flat width matches the extended MultiDiscrete.
  * MaskablePPO trains briefly on a three-phase env without crashing.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from gymnasium import spaces

from layout_gen.pdk import load_pdk
from layout_gen.synth.geo.state import LayoutState
from layout_gen.synth.loader import load_template

from layout_gen.rl.env.action_space import (
    ActionSpace, REPAIR_KINDS, PLACE_KINDS, ROUTE_KINDS,
    action_mask_for,
)
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.env.observation import build_observation
from layout_gen.rl.env.place_action import N_ORIENTATIONS, TransistorCache
from layout_gen.rl.env.route_action import (
    N_ROUTE_LAYERS, ROUTE_LAYERS, add_route_segment, layer_from_index,
    layer_index, size_bins,
)
from layout_gen.rl.policy.network import LayoutPolicyConfig
from layout_gen.rl.policy.sb3 import MaskableLayoutPolicy
from layout_gen.rl.topology import graph_from_template
from layout_gen.rl.training.ppo_train import PPOConfig, PPOTrainer


# ── Module fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rules():
    return load_pdk()


@pytest.fixture(scope="module")
def cache(rules):
    return TransistorCache(rules)


# ── add_route_segment ────────────────────────────────────────────────────────

def test_add_route_segment_writes_tagged_rect():
    state = LayoutState()
    r = add_route_segment(
        state, layer="met1",
        x_um=1.0, y_um=2.0, w_um=0.3, h_um=0.05,
        net_name="VDD",
    )
    assert r.layer == "met1"
    assert r.x0 == pytest.approx(1.0)
    assert r.y0 == pytest.approx(2.0)
    assert r.x1 == pytest.approx(1.3)
    assert r.y1 == pytest.approx(2.05)
    assert r.net == "VDD"
    assert r.shape_type == "wire"


def test_layer_index_roundtrip():
    for i, name in enumerate(ROUTE_LAYERS):
        assert layer_index(name) == i
        assert layer_from_index(i) == name


def test_size_bins_log_spaced():
    bins = size_bins(8)
    assert bins.shape == (8,)
    assert bins[0] < bins[-1]
    # Ratio between adjacent bins should be roughly constant (log-spaced).
    ratios = bins[1:] / bins[:-1]
    assert np.allclose(ratios, ratios[0], rtol=0.01)


# ── ActionSpace with enable_route ────────────────────────────────────────────

def test_action_space_shape_with_route_only():
    helper = ActionSpace(target_cap=16, mag_bins=4,
                         enable_route=True, net_cap=8,
                         route_x_bins=4, route_y_bins=4,
                         route_w_bins=4, route_h_bins=4)
    expected = (
        len(REPAIR_KINDS) + len(ROUTE_KINDS),  # PLACE not included
        16, 4, 2, 2, 4,                          # repair dims
        8, N_ROUTE_LAYERS, 4, 4, 4, 4,           # route dims
    )
    assert tuple(helper.gym_space.nvec) == expected


def test_action_space_shape_with_place_and_route():
    helper = ActionSpace(target_cap=16, mag_bins=4,
                         enable_place=True, device_cap=4,
                         x_bins=4, y_bins=4,
                         enable_route=True, net_cap=8,
                         route_x_bins=4, route_y_bins=4,
                         route_w_bins=4, route_h_bins=4)
    expected = (
        len(REPAIR_KINDS) + len(PLACE_KINDS) + len(ROUTE_KINDS),
        16, 4, 2, 2, 4,
        4, 4, 4, N_ORIENTATIONS,                 # place block
        8, N_ROUTE_LAYERS, 4, 4, 4, 4,           # route block
    )
    assert tuple(helper.gym_space.nvec) == expected


def test_action_space_decode_route_kind():
    """The kind index for route_segment is 7 when both PLACE+ROUTE enabled
    (REPAIR=6 + PLACE=1)."""
    helper = ActionSpace(
        target_cap=16, mag_bins=4,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
        cell_width_um=4.0, cell_height_um=2.0,
        enable_route=True, net_cap=8,
        route_x_bins=4, route_y_bins=4, route_w_bins=4, route_h_bins=4,
    )
    # Layout: [kind, target, edge, sx, sy, mag, dev, xb, yb, or,
    #          net, lyr, rxb, ryb, wb, hb]
    raw = [7,  # kind = route_segment (idx 6 = place_device, 7 = route_segment)
           0, 0, 0, 0, 0,
           0, 0, 0, 0,                 # place block (ignored for route kind)
           3,                          # net idx 3
           1,                          # layer = met1
           2, 1,                       # rxb=2 (mid x), ryb=1
           0, 0]                       # smallest size bins
    act = helper.decode(raw, idx_to_rid={})
    assert act.is_route
    assert act.net_idx == 3
    assert act.route_layer == "met1"
    assert act.route_x_um == pytest.approx(2.5)   # bin 2 of 4 over 4.0
    assert act.route_y_um == pytest.approx(0.75)  # bin 1 of 4 over 2.0
    assert act.route_w_um > 0
    assert act.route_h_um > 0


# ── action_mask_for (ROUTE phase) ────────────────────────────────────────────

def test_action_mask_route_phase_enables_only_route_kinds():
    state = LayoutState()
    mask = action_mask_for(
        state, {},
        target_cap=8, mag_bins=4,
        enable_place=True,  device_cap=4, n_devices=2,
        x_bins=4, y_bins=4,
        enable_route=True,  net_cap=8, n_nets=4,
        phase="route",
    )
    n_kinds = len(REPAIR_KINDS) + len(PLACE_KINDS) + len(ROUTE_KINDS)
    kind_mask = mask[:n_kinds]
    repair_block = kind_mask[: len(REPAIR_KINDS)]
    place_block  = kind_mask[len(REPAIR_KINDS):
                              len(REPAIR_KINDS) + len(PLACE_KINDS)]
    route_block  = kind_mask[len(REPAIR_KINDS) + len(PLACE_KINDS):]
    assert not repair_block.any()
    assert not place_block.any()
    assert route_block.all()


def test_action_mask_route_phase_constrains_net_dim():
    state = LayoutState()
    mask = action_mask_for(
        state, {},
        target_cap=8, mag_bins=4,
        enable_place=True, device_cap=4, n_devices=2,
        x_bins=4, y_bins=4,
        enable_route=True, net_cap=8, n_nets=3,
        phase="route",
    )
    # Layout offsets:
    #   kinds (8) + target (8) + edge (4) + sx (2) + sy (2) + mag (4)
    #   + place block (4 + 4 + 4 + 4 = 16) = 44
    #   then net block starts.
    base = 8 + 8 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 4
    net_mask = mask[base:base + 8]
    # First 3 nets enabled, rest off.
    np.testing.assert_array_equal(net_mask[:3], [True, True, True])
    np.testing.assert_array_equal(net_mask[3:], [False] * 5)


# ── LayoutEnv: PLACE → ROUTE → REPAIR ────────────────────────────────────────

class _FakeDRC:
    def run(self, state):
        return []

    def count(self, state) -> int:
        return 0

    def stats(self) -> dict:
        return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}

    def clear(self) -> None:
        pass


def _inverter_env(rules, cache, *, max_place_steps=4, max_route_steps=4,
                   max_steps=12) -> LayoutEnv:
    from layout_gen.rl.topology.parser import graph_from_template as _gft
    g = _gft(load_template("inverter"),
             cell_params={"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}})
    return LayoutEnv(
        drc=_FakeDRC(),
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


def test_env_three_phase_transition(rules, cache):
    env = _inverter_env(rules, cache, max_place_steps=2,
                         max_route_steps=3, max_steps=12)
    obs, info = env.reset()
    assert env.phase == "place"

    nvec = env.action_space.nvec.shape[0]
    seen_phases = ["place"]
    for _ in range(env._action_helper.gym_space.nvec.shape[0]):
        action = env.action_space.sample()
        # Make the action valid for the current phase by setting kind.
        kind_offsets = {
            "place":  len(REPAIR_KINDS),
            "route":  len(REPAIR_KINDS) + len(PLACE_KINDS),
            "repair": 0,
        }
        action[0] = kind_offsets[env.phase]
        obs, _, terminated, truncated, info = env.step(action)
        if info["phase"] != seen_phases[-1]:
            seen_phases.append(info["phase"])
        if terminated or truncated:
            break
    # We should have seen at least PLACE → ROUTE (and ideally → REPAIR).
    assert seen_phases[0] == "place"
    assert "route"  in seen_phases, f"never reached route: {seen_phases}"


def test_env_route_action_tags_net(rules, cache):
    env = _inverter_env(rules, cache, max_place_steps=2,
                         max_route_steps=4, max_steps=10)
    env.reset()
    # Place both devices to advance into ROUTE phase.
    for d_idx in (0, 1):
        action = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
        action[0] = len(REPAIR_KINDS)   # place_device
        action[6] = d_idx
        action[7] = action[8] = 4
        action[9] = 0
        env.step(action)
    assert env.phase == "route"

    # Issue a route action targeting net idx 0 ("VDD" in the inverter).
    action = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
    action[0] = len(REPAIR_KINDS) + len(PLACE_KINDS)   # route_segment
    # Net dim is at offset 6 + 4 = 10 (after PLACE block).
    action[10] = 0  # net 0
    action[11] = 1  # layer met1
    action[12] = 4  # rxb
    action[13] = 4  # ryb
    action[14] = 1  # wb
    action[15] = 1  # hb
    obs, _, _, _, info = env.step(action)
    assert info["action"]["valid"] is True
    # The new rect should carry the net name from the topology.
    routing_rects = [r for r in env.state if r.shape_type == "wire"]
    assert len(routing_rects) >= 1
    assert routing_rects[-1].net != ""


# ── MaskableLayoutPolicy with ROUTE ──────────────────────────────────────────

def test_maskable_policy_route_logits_width():
    cfg = LayoutPolicyConfig(
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
        enable_route=True, net_cap=8,
        route_x_bins=4, route_y_bins=4,
        route_w_bins=4, route_h_bins=4,
    )
    nvec = [
        len(REPAIR_KINDS) + len(PLACE_KINDS) + len(ROUTE_KINDS),
        cfg.target_cap, 4, 2, 2, cfg.mag_bins,
        cfg.device_cap, cfg.x_bins, cfg.y_bins, N_ORIENTATIONS,
        cfg.net_cap, N_ROUTE_LAYERS,
        cfg.route_x_bins, cfg.route_y_bins,
        cfg.route_w_bins, cfg.route_h_bins,
    ]
    obs_space = spaces.Dict({
        "poly_feats": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.poly_cap,
                    __import__("layout_gen.repair.features",
                               fromlist=["POLY_FEAT_DIM"]).POLY_FEAT_DIM),
            dtype=np.float32),
        "poly_mask": spaces.Box(0.0, 1.0, (cfg.poly_cap,), np.float32),
        "viol_feats": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.viol_cap,
                   __import__("layout_gen.rl.env.observation",
                              fromlist=["V_FEAT_DIM"]).V_FEAT_DIM),
            dtype=np.float32),
        "viol_mask": spaces.Box(0.0, 1.0, (cfg.viol_cap,), np.float32),
        "global_feats": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(__import__("layout_gen.rl.env.observation",
                              fromlist=["N_GLOBAL"]).N_GLOBAL,),
            dtype=np.float32),
    })
    act_space = spaces.MultiDiscrete(nvec)
    policy = MaskableLayoutPolicy(
        observation_space=obs_space, action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        layout_config=cfg,
    )
    assert policy.action_dist.action_dims == nvec

    # Smoke a forward to confirm the flat-logits width matches sum(nvec).
    s = LayoutState()
    s.add(layer="met1", x0=0.0, y0=0.0, x1=0.10, y1=0.10)
    obs_struct = build_observation(s, [], poly_cap=cfg.poly_cap,
                                    viol_cap=cfg.viol_cap)
    obs_dict = obs_struct.to_dict()
    batched = {k: torch.from_numpy(np.stack([v])) for k, v in obs_dict.items()}
    mask = action_mask_for(
        s, obs_struct.rid_to_idx,
        target_cap=cfg.target_cap, mag_bins=cfg.mag_bins,
        enable_place=True, phase="route",
        device_cap=cfg.device_cap, n_devices=2,
        x_bins=cfg.x_bins, y_bins=cfg.y_bins,
        enable_route=True, net_cap=cfg.net_cap, n_nets=4,
        route_x_bins=cfg.route_x_bins, route_y_bins=cfg.route_y_bins,
        route_w_bins=cfg.route_w_bins, route_h_bins=cfg.route_h_bins,
    )
    batched_mask = torch.from_numpy(np.stack([mask]))
    actions, values, log_probs = policy.forward(batched, action_masks=batched_mask)
    assert actions.shape == (1, len(nvec))


def test_ppo_with_route_phase_does_not_crash(rules, cache):
    cfg = LayoutPolicyConfig(
        poly_cap=64, viol_cap=8, target_cap=64, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
        enable_place=True, device_cap=8, x_bins=8, y_bins=8,
        enable_route=True, net_cap=8,
        route_x_bins=8, route_y_bins=8,
        route_w_bins=4, route_h_bins=4,
    )

    def _factory():
        return _inverter_env(rules, cache,
                              max_place_steps=2, max_route_steps=4,
                              max_steps=8)

    trainer = PPOTrainer(
        env_factory=_factory,
        config=PPOConfig(
            n_envs=1, n_steps=64, batch_size=32, n_epochs=1,
            learning_rate=3e-4, seed=0, verbose=0,
        ),
        layout_config=cfg,
    )
    trainer.learn(total_timesteps=64)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
