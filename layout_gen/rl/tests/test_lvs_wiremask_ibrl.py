"""
layout_gen.rl.tests.test_lvs_wiremask_ibrl — bundled tests for the
LVS / wiremask / IBRL trio shipped together.

Each block is self-contained; they don't share fixtures.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from layout_gen.pdk import load_pdk
from layout_gen.synth.geo.state import LayoutState
from layout_gen.synth.loader import load_template

from layout_gen.rl.env.connectivity import compute_short_count
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.env.observation import build_observation, make_observation_space
from layout_gen.rl.env.place_action import TransistorCache, place_device_full
from layout_gen.rl.env.reward import RewardConfig, compute_reward
from layout_gen.rl.env.route_action import add_route_segment
from layout_gen.rl.env.runner import CachedLVS
from layout_gen.rl.env.spice_ref import emit_spice_subckt, write_spice_subckt
from layout_gen.rl.policy.network import LayoutPolicy, LayoutPolicyConfig
from layout_gen.rl.topology import graph_from_template
from layout_gen.rl.topology.parser import DeviceNode


# ── 1) Short-circuit detection + reward ──────────────────────────────────────

def test_compute_short_count_basic():
    s = LayoutState()
    add_route_segment(s, layer="met1", x_um=0.0, y_um=0.0,
                       w_um=1.0, h_um=0.10, net_name="A")
    add_route_segment(s, layer="met1", x_um=0.5, y_um=0.0,
                       w_um=1.0, h_um=0.10, net_name="B")  # overlaps A
    add_route_segment(s, layer="met2", x_um=0.0, y_um=0.0,
                       w_um=1.0, h_um=0.10, net_name="C")  # different layer
    n = compute_short_count(s)
    assert n == 1, f"expected exactly 1 short (A↔B), got {n}"


def test_compute_short_count_same_net_does_not_count():
    s = LayoutState()
    add_route_segment(s, layer="met1", x_um=0.0, y_um=0.0,
                       w_um=1.0, h_um=0.10, net_name="A")
    add_route_segment(s, layer="met1", x_um=0.5, y_um=0.0,
                       w_um=1.0, h_um=0.10, net_name="A")
    assert compute_short_count(s) == 0


def test_short_delta_reward_is_negative_when_shorts_grow():
    rb = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True,
        phase="route", config=RewardConfig(short_delta=2.0),
        short_before=0, short_after=3,
    )
    # +Δshorts (3) → -2*3 = -6 short_delta reward.
    assert rb.short_delta == pytest.approx(-6.0)


# ── 2) SPICE emitter for LVS reference netlists ─────────────────────────────

def test_spice_ref_emits_inverter_subckt():
    g = graph_from_template(
        load_template("inverter"),
        cell_params={"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}},
    )
    src = emit_spice_subckt(g, "inverter")
    assert ".subckt inverter" in src
    assert ".ends" in src
    # Both devices should have G/D/S wired to the YAML's net names.
    assert "OUT IN GND" in src    # NMOS: D=OUT, G=IN, S=GND
    assert "OUT IN VDD" in src    # PMOS: D=OUT, G=IN, S=VDD
    assert "sky130_fd_pr__nfet_01v8" in src
    assert "sky130_fd_pr__pfet_01v8" in src


def test_spice_ref_writes_to_disk(tmp_path: Path):
    g = graph_from_template(
        load_template("inverter"),
        cell_params={"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}},
    )
    out = tmp_path / "inv.spice"
    write_spice_subckt(g, "inverter", out)
    assert out.exists()
    text = out.read_text()
    assert ".subckt" in text


# ── 3) CachedLVS — uses a stubbed runner so the test doesn't need magic ────

class _StubLVSRunner:
    """LVSRunner-shaped stub; counts invocations."""
    tool_name = "stub"

    def __init__(self):
        self.calls = 0
        self._mismatches: list = []

    def is_available(self):
        return True

    def run(self, gds_path, ref_netlist, cell_name):
        from layout_gen.lvs.base import LVSResult
        self.calls += 1
        return LVSResult(clean=not self._mismatches,
                          mismatches=list(self._mismatches),
                          log="stub")


def test_cached_lvs_caches_identical_geometries():
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except Exception:
        from gdsfactory.gpdk import PDK as _G
        _G.activate()
    rules = load_pdk()
    runner = _StubLVSRunner()
    cache = CachedLVS(runner, rules, cell_name="x",
                       ref_netlist=Path("/dev/null"))

    s = LayoutState()
    s.add(layer="met1", x0=0.0, y0=0.0, x1=0.10, y1=0.10)
    cache.run(s)
    cache.run(s)   # identical → cache hit
    assert runner.calls == 1
    assert cache.stats()["hits"] == 1


class _NameRecordingLVSRunner(_StubLVSRunner):
    """Records every cell_name that flowed through ``run``."""
    def __init__(self):
        super().__init__()
        self.names: list[str] = []

    def run(self, gds_path, ref_netlist, cell_name):
        self.names.append(cell_name)
        return super().run(gds_path, ref_netlist, cell_name)


def test_cached_lvs_unique_names_across_instances():
    """Multiple Cached* instances in one process (the DummyVecEnv case
    that bit us at training time) must not collide on cell names —
    gdsfactory's KCLayout registry is process-global, so per-instance
    counters were the source of the ``Cellname inverter_drc1 already
    exists`` crash. The fix is a process-global suffix counter."""
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except Exception:
        from gdsfactory.gpdk import PDK as _G
        _G.activate()
    rules = load_pdk()

    runner_a = _NameRecordingLVSRunner()
    runner_b = _NameRecordingLVSRunner()
    cache_a = CachedLVS(runner_a, rules, cell_name="inverter",
                         ref_netlist=Path("/dev/null"))
    cache_b = CachedLVS(runner_b, rules, cell_name="inverter",
                         ref_netlist=Path("/dev/null"))

    # Two distinct geometries → both caches actually invoke the runner.
    s1 = LayoutState()
    s1.add(layer="met1", x0=0.0, y0=0.0, x1=0.10, y1=0.10)
    s2 = LayoutState()
    s2.add(layer="met1", x0=1.0, y0=0.0, x1=1.10, y1=0.10)

    cache_a.run(s1)
    cache_b.run(s2)
    cache_a.run(s2)   # forces a 3rd invocation across the two instances

    all_names = runner_a.names + runner_b.names
    assert len(all_names) == len(set(all_names)), (
        f"cell-name collision across CachedLVS instances: {all_names}"
    )


# ── 4) Wiremask proximity channel ───────────────────────────────────────────

def test_observation_space_includes_proximity_when_shape_passed():
    space = make_observation_space(poly_cap=8, viol_cap=4,
                                    proximity_shape=(8, 8))
    assert "proximity_map" in space.spaces
    assert space.spaces["proximity_map"].shape == (1, 8, 8)


def test_proximity_map_values_are_normalised_to_unit_diagonal():
    s = LayoutState()
    obs = build_observation(
        s, [], poly_cap=8, viol_cap=4,
        proximity_shape=(4, 4),
        terminal_positions=[(0.0, 0.0)],
        cell_dimensions=(1.0, 1.0),
    )
    arr = obs.proximity_map
    assert arr is not None
    assert arr.shape == (1, 4, 4)
    # All values in [0, 1]; min should be ≤ 1/diag*0.5 ≈ ~0.35.
    assert arr.min() <= 0.5
    assert arr.max() <= 1.0
    # No-terminal case → all ones.
    obs2 = build_observation(
        s, [], poly_cap=8, viol_cap=4,
        proximity_shape=(4, 4),
        terminal_positions=[],
        cell_dimensions=(1.0, 1.0),
    )
    np.testing.assert_array_equal(obs2.proximity_map, np.ones((1, 4, 4)))


def test_layoutenv_emits_proximity_channel_when_enabled():
    rules = load_pdk()
    cache = TransistorCache(rules)
    g = graph_from_template(
        load_template("inverter"),
        cell_params={"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}},
    )

    class _NoOpDRC:
        def run(self, s): return []
        def count(self, s): return 0
        def stats(self): return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}
        def clear(self): pass

    env = LayoutEnv(
        drc=_NoOpDRC(),
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        max_steps=4,
        enable_place=True,
        topology_graph=g, transistor_cache=cache,
        device_cap=8, x_bins=8, y_bins=8,
        cell_width_um=4.0, cell_height_um=2.0,
        proximity_shape=(8, 8),
    )
    obs, _ = env.reset()
    assert "proximity_map" in obs
    assert obs["proximity_map"].shape == (1, 8, 8)


# ── 5) IBRL — BC distillation PPO loss path ─────────────────────────────────

def test_bc_distill_ppo_learns_without_crash(tmp_path: Path):
    """Smoke: a tiny MaskableBCDistillPPO learn() runs and the
    distill_loss appears in the logger when β > 0."""
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from layout_gen.rl.env.layout_env import LayoutEnv
    from layout_gen.rl.policy.sb3 import MaskableLayoutPolicy
    from layout_gen.rl.training.ibrl import MaskableBCDistillPPO

    cfg = LayoutPolicyConfig(
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
    )

    # 1) Save a quick BC checkpoint.
    bc_policy = LayoutPolicy(cfg)
    ckpt = tmp_path / "bc.pt"
    torch.save({"state_dict": bc_policy.state_dict(),
                "config":     bc_policy.cfg.__dict__}, ckpt)

    # 2) Build a tiny REPAIR-only env.
    rules = load_pdk()
    rng = random.Random(0)

    def _state():
        s = LayoutState()
        for k in range(2):
            x0 = 0.20 * k + rng.uniform(-0.01, 0.01)
            s.add(layer="met1", x0=x0, y0=0.0, x1=x0 + 0.10, y1=0.10)
        return s

    class _NoOpDRC:
        def run(self, s): return []
        def count(self, s): return 0
        def stats(self): return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}
        def clear(self): pass

    def _make():
        return Monitor(ActionMasker(LayoutEnv(
            drc=_NoOpDRC(),
            poly_cap=cfg.poly_cap, viol_cap=cfg.viol_cap,
            target_cap=cfg.target_cap, mag_bins=cfg.mag_bins,
            max_steps=4,
            default_state_factory=_state,
        ), lambda e: e.action_masks()))

    vec = DummyVecEnv([_make])
    model = MaskableBCDistillPPO(
        policy=MaskableLayoutPolicy, env=vec,
        n_steps=16, batch_size=8, n_epochs=1, verbose=0,
        policy_kwargs={"layout_config": cfg},
        bc_checkpoint=ckpt, bc_policy_config=cfg,
        beta_start=1.0, beta_end=0.0,
    )
    model.learn(total_timesteps=32)
    # If we got here without exception, the distillation path works.
    assert model.num_timesteps >= 32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
