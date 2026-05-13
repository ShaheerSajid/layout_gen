"""
layout_gen.rl.tests.test_place — PLACE phase end-to-end smoke tests.

Verifies:
  * `place_device` materialises a transistor at (x, y) with correct
    orientation (R0 / MX / MY / R180) and caches the rect template
    across calls of identical (type, w, l, fingers).
  * Extended `ActionSpace(enable_place=True)` exposes the 10-dim
    MultiDiscrete and decodes a PLACE action correctly.
  * Phase-aware `action_mask_for` masks REPAIR slots in PLACE phase
    and vice versa, and disables already-placed devices.
  * `LayoutEnv(enable_place=True)` runs an episode that places every
    inverter device in turn, transitions to REPAIR, and remains
    well-typed throughout.
  * `MaskableLayoutPolicy(enable_place=True)` produces the right
    flat-logits width and PPO can train on a PLACE-enabled env.
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
    ActionSpace, EnvAction, REPAIR_KINDS, PLACE_KINDS,
    action_mask_for, magnitude_bins,
)
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.env.place_action import (
    N_ORIENTATIONS, ORIENTATIONS, TransistorCache, orient_rect, place_device,
)
from layout_gen.rl.policy.network import LayoutPolicy, LayoutPolicyConfig
from layout_gen.rl.policy.sb3 import MaskableLayoutPolicy
from layout_gen.rl.topology import (
    TopologyEncoder, TopologyEncoderConfig, graph_from_template,
)
from layout_gen.rl.training.ppo_train import PPOConfig, PPOTrainer
from layout_gen.rl.topology.parser import DeviceNode


# ── Module-level fixtures (PDK is expensive to load) ─────────────────────────

@pytest.fixture(scope="module")
def rules():
    return load_pdk()


@pytest.fixture(scope="module")
def cache(rules):
    return TransistorCache(rules)


# ── Geometry: orient_rect ────────────────────────────────────────────────────

def test_orient_rect_identity():
    assert orient_rect(0.0, 0.0, 1.0, 2.0, "R0") == (0.0, 0.0, 1.0, 2.0)


def test_orient_rect_mx_flips_y():
    # Mirror across X: (0..1, 0..2) → (0..1, -2..0)
    assert orient_rect(0.0, 0.0, 1.0, 2.0, "MX") == (0.0, -2.0, 1.0, 0.0)


def test_orient_rect_my_flips_x():
    assert orient_rect(0.0, 0.0, 1.0, 2.0, "MY") == (-1.0, 0.0, 0.0, 2.0)


def test_orient_rect_r180_flips_both():
    assert orient_rect(0.0, 0.0, 1.0, 2.0, "R180") == (-1.0, -2.0, 0.0, 0.0)


# ── place_device ─────────────────────────────────────────────────────────────

def test_place_device_materialises_transistor(cache):
    state = LayoutState()
    dev = DeviceNode(name="N", device_type="nmos", template="planar_mosfet",
                     w_um=0.5, l_um=0.15, fingers=0, in_nwell=False)
    placed = place_device(state, dev, x_um=1.0, y_um=2.0,
                           orientation="R0", cache=cache)
    assert len(placed) > 0
    # Layers expected from the standard nmos transistor.
    layers = {r.layer for r in placed}
    assert {"diff", "poly", "li1"}.issubset(layers)
    # Translation applied: every rect should have x0 ≥ -2 (poly extends
    # to negative coords pre-translate; +1 makes it ≥ -2).
    assert all(r.x0 >= -2.0 for r in placed)


def test_place_device_cache_hit(cache):
    """Second placement with same params → cache size stays at 1 entry."""
    s1 = LayoutState()
    s2 = LayoutState()
    dev = DeviceNode(name="N", device_type="nmos", template="planar_mosfet",
                     w_um=0.5, l_um=0.15, fingers=0, in_nwell=False)
    place_device(s1, dev, x_um=0.0, y_um=0.0, orientation="R0", cache=cache)
    size_after_first = len(cache._cache)
    place_device(s2, dev, x_um=2.0, y_um=2.0, orientation="MY", cache=cache)
    assert len(cache._cache) == size_after_first


def test_place_device_orientation_mirrors_geometry(cache):
    """R0 vs R180 of the same device must produce different bboxes."""
    dev = DeviceNode(name="N", device_type="nmos", template="planar_mosfet",
                     w_um=0.5, l_um=0.15, fingers=0, in_nwell=False)
    s_r0 = LayoutState()
    s_r180 = LayoutState()
    place_device(s_r0,   dev, x_um=0.0, y_um=0.0,
                  orientation="R0",   cache=cache)
    place_device(s_r180, dev, x_um=0.0, y_um=0.0,
                  orientation="R180", cache=cache)
    bb_r0   = (min(r.x0 for r in s_r0),   min(r.y0 for r in s_r0))
    bb_r180 = (min(r.x0 for r in s_r180), min(r.y0 for r in s_r180))
    assert bb_r0 != bb_r180


# ── ActionSpace with enable_place ────────────────────────────────────────────

def test_action_space_shape_with_place():
    helper = ActionSpace(target_cap=32, mag_bins=8, enable_place=True,
                         device_cap=8, x_bins=12, y_bins=12)
    assert tuple(helper.gym_space.nvec) == (
        len(REPAIR_KINDS) + len(PLACE_KINDS),  # 7
        32, 4, 2, 2, 8,                         # repair dims
        8, 12, 12, N_ORIENTATIONS,              # place dims
    )


def test_action_space_decode_place_kind():
    helper = ActionSpace(target_cap=32, mag_bins=8, enable_place=True,
                         device_cap=4, x_bins=4, y_bins=4,
                         cell_width_um=4.0, cell_height_um=2.0)
    # kind=6 (place_device), target=0, edge=0, sx=0, sy=0, mag=0,
    # device=2, x_bin=1, y_bin=2, orient=1 (MX)
    raw = [6, 0, 0, 0, 0, 0, 2, 1, 2, 1]
    act = helper.decode(raw, idx_to_rid={})
    assert act.is_place
    assert act.device_idx == 2
    # x_bin=1 of 4 over 4.0 µm → centre = (1.5/4)*4.0 = 1.5
    assert act.x_um == pytest.approx(1.5)
    # y_bin=2 of 4 over 2.0 µm → centre = (2.5/4)*2.0 = 1.25
    assert act.y_um == pytest.approx(1.25)
    assert act.orientation == ORIENTATIONS[1]   # "MX"


# ── action_mask_for: phase-aware ─────────────────────────────────────────────

def test_action_mask_place_phase_disables_repair_kinds():
    state = LayoutState()
    rid_to_idx: dict[int, int] = {}
    mask = action_mask_for(
        state, rid_to_idx,
        target_cap=8, mag_bins=4,
        enable_place=True, phase="place",
        device_cap=4, n_devices=2,
        x_bins=4, y_bins=4,
    )
    n_kinds_total = len(REPAIR_KINDS) + len(PLACE_KINDS)
    kind_mask = mask[:n_kinds_total]
    # First N_REPAIR_KINDS should be False; PLACE kinds True.
    assert not kind_mask[: len(REPAIR_KINDS)].any()
    assert kind_mask[len(REPAIR_KINDS):].all()


def test_action_mask_repair_phase_disables_place_kinds():
    state = LayoutState()
    rid_to_idx: dict[int, int] = {}
    mask = action_mask_for(
        state, rid_to_idx,
        target_cap=8, mag_bins=4,
        enable_place=True, phase="repair",
        device_cap=4, n_devices=2,
        x_bins=4, y_bins=4,
    )
    n_kinds_total = len(REPAIR_KINDS) + len(PLACE_KINDS)
    kind_mask = mask[:n_kinds_total]
    assert kind_mask[: len(REPAIR_KINDS)].all()
    assert not kind_mask[len(REPAIR_KINDS):].any()


def test_action_mask_skips_already_placed_devices():
    state = LayoutState()
    placed = np.array([True, False, True, False], dtype=bool)
    mask = action_mask_for(
        state, {},
        target_cap=8, mag_bins=4,
        enable_place=True, phase="place",
        device_cap=4, n_devices=4,
        placed_mask=placed,
        x_bins=4, y_bins=4,
    )
    # Locate the device-dim slice. Layout:
    #   [kinds (7), target (8), edge (4), sx (2), sy (2), mag (4),
    #    device (4), x_bin (4), y_bin (4), orient (4)]
    base = 7 + 8 + 4 + 2 + 2 + 4
    device_mask = mask[base:base + 4]
    np.testing.assert_array_equal(device_mask, [False, True, False, True])


# ── LayoutEnv phase transitions ──────────────────────────────────────────────

class _FakeDRC:
    """Returns no violations regardless of state — simplifies PLACE tests."""

    def run(self, state):
        return []

    def count(self, state) -> int:
        return 0

    def stats(self) -> dict:
        return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}

    def clear(self) -> None:
        pass


def _inverter_env(rules, cache, *, max_place_steps: int = 4,
                  max_steps: int = 6) -> LayoutEnv:
    g = graph_from_template(load_template("inverter"))
    # Provide cell_params defaults so devices have non-zero w/l.
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
    )


def test_env_starts_in_place_phase_when_enabled(rules, cache):
    env = _inverter_env(rules, cache)
    obs, info = env.reset()
    assert env.phase == "place"
    assert info["phase"] == "place"
    assert info["n_polygons"] == 0   # PLACE starts with empty layout


def test_env_place_action_grows_state(rules, cache):
    env = _inverter_env(rules, cache)
    obs, info = env.reset()

    # Build a PLACE action manually: kind=6, device=0, mid x/y, R0.
    raw = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
    raw[0] = 6        # kind = place_device
    raw[6] = 0        # device idx 0
    raw[7] = 4        # x bin 4 of 8
    raw[8] = 4        # y bin 4 of 8
    raw[9] = 0        # orient R0
    obs2, reward, terminated, truncated, info2 = env.step(raw)
    assert info2["phase"] in ("place", "repair")
    assert info2["n_polygons"] > 0
    assert info2["n_devices_placed"] == 1
    assert info2["action"]["valid"] is True


def test_env_transitions_to_repair_when_all_placed(rules, cache):
    env = _inverter_env(rules, cache, max_place_steps=8, max_steps=12)
    env.reset()
    # Place both devices at *distinct* y bins so the no-stacking guard
    # in _apply_place doesn't reject the second one (canonical inverter
    # has gate-aligned X but different rows).
    for d_idx, y_bin in zip((0, 1), (1, 5)):
        raw = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
        raw[0] = 6
        raw[6] = d_idx
        raw[7] = 4
        raw[8] = y_bin
        raw[9] = 0
        obs, _, terminated, truncated, info = env.step(raw)
        if info.get("phase_transitioned"):
            break
    assert env.phase == "repair", \
        f"Expected REPAIR after placing both devices; got {env.phase}"


def test_env_repeated_device_placement_invalid(rules, cache):
    env = _inverter_env(rules, cache)
    env.reset()
    raw = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
    raw[0] = 6
    raw[6] = 0
    raw[7] = 4
    raw[8] = 4
    raw[9] = 0
    env.step(raw)
    # Second placement of the same device should be invalid (mask
    # would normally prevent it; we ignore the mask to test the
    # env's defensive guard).
    obs, reward, terminated, truncated, info = env.step(raw)
    assert info["action"]["valid"] is False


# ── MaskableLayoutPolicy with PLACE ──────────────────────────────────────────

def test_maskable_policy_place_logits_width():
    cfg = LayoutPolicyConfig(
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
    )
    obs_space = spaces.Dict({
        "poly_feats":   spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(cfg.poly_cap,
                                           __import__("layout_gen.repair.features",
                                                      fromlist=["POLY_FEAT_DIM"]).POLY_FEAT_DIM),
                                    dtype=np.float32),
        "poly_mask":    spaces.Box(0.0, 1.0, (cfg.poly_cap,), np.float32),
        "viol_feats":   spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(cfg.viol_cap,
                                           __import__("layout_gen.rl.env.observation",
                                                      fromlist=["V_FEAT_DIM"]).V_FEAT_DIM),
                                    dtype=np.float32),
        "viol_mask":    spaces.Box(0.0, 1.0, (cfg.viol_cap,), np.float32),
        "global_feats": spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(__import__("layout_gen.rl.env.observation",
                                                       fromlist=["N_GLOBAL"]).N_GLOBAL,),
                                    dtype=np.float32),
    })
    nvec = [
        len(REPAIR_KINDS) + len(PLACE_KINDS),
        cfg.target_cap, 4, 2, 2, cfg.mag_bins,
        cfg.device_cap, cfg.x_bins, cfg.y_bins, N_ORIENTATIONS,
    ]
    act_space = spaces.MultiDiscrete(nvec)
    policy = MaskableLayoutPolicy(
        observation_space=obs_space, action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        layout_config=cfg,
    )
    expected_width = sum(nvec)
    assert policy.action_dist.action_dims == nvec
    # Smoke a forward to confirm widths line up with the distribution.
    from layout_gen.rl.env.observation import build_observation
    s = LayoutState()
    s.add(layer="met1", x0=0.0, y0=0.0, x1=0.10, y1=0.10)
    obs_struct = build_observation(s, [], poly_cap=cfg.poly_cap, viol_cap=cfg.viol_cap)
    obs_dict = obs_struct.to_dict()
    batched = {k: torch.from_numpy(np.stack([v])) for k, v in obs_dict.items()}
    mask = action_mask_for(
        s, obs_struct.rid_to_idx,
        target_cap=cfg.target_cap, mag_bins=cfg.mag_bins,
        enable_place=True, phase="repair",
        device_cap=cfg.device_cap, n_devices=2,
        x_bins=cfg.x_bins, y_bins=cfg.y_bins,
    )
    batched_mask = torch.from_numpy(np.stack([mask]))
    actions, values, log_probs = policy.forward(batched, action_masks=batched_mask)
    assert actions.shape == (1, len(nvec))


def test_ppo_with_place_phase_does_not_crash(rules, cache):
    """End-to-end smoke: PPO trains briefly on a PLACE-enabled env."""
    cfg = LayoutPolicyConfig(
        poly_cap=64, viol_cap=8, target_cap=64, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
        enable_place=True, device_cap=8, x_bins=8, y_bins=8,
    )

    def _factory():
        return _inverter_env(rules, cache, max_place_steps=4, max_steps=6)

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
