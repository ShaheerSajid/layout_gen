"""
layout_gen.rl.tests.test_env_topology — env + policy + PPO with topology.

Verifies the full topology pipeline:
  * LayoutEnv with topology_global exposes the right obs_space + obs.
  * MaskableLayoutPolicy(use_topology=True) consumes topology from obs.
  * MaskablePPO trains for a few steps without crashing on a
    topology-conditioned env.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gymnasium import spaces

from layout_gen.synth.geo.state import LayoutState
from layout_gen.synth.loader import load_template

from layout_gen.rl.env.action_space import N_KINDS, action_mask_for
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.env.observation import build_observation, make_observation_space
from layout_gen.rl.policy.network import LayoutPolicyConfig
from layout_gen.rl.policy.sb3 import MaskableLayoutPolicy
from layout_gen.rl.topology import (
    TopologyEncoder, TopologyEncoderConfig, graph_from_template,
)
from layout_gen.rl.training.ppo_train import PPOConfig, PPOTrainer


# ── Fake DRC + small env ─────────────────────────────────────────────────────

class _DirectFakeDRC:
    def __init__(self, threshold_um: float = 0.20):
        self._threshold = threshold_um

    def run(self, state):
        from layout_gen.drc.base import DRCViolation
        out = []
        rects = state.rects
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                if a.layer != b.layer:
                    continue
                d = ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5
                if 0 < d < self._threshold:
                    out.append(DRCViolation(
                        rule=f"{a.layer}.spacing",
                        description=f"min spacing: {self._threshold}",
                        layer=a.layer, x=(a.cx + b.cx) / 2,
                        y=(a.cy + b.cy) / 2, value=d,
                    ))
        return out

    def count(self, state) -> int:
        return len(self.run(state))

    def stats(self) -> dict:
        return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}

    def clear(self) -> None:
        pass


def _seed_factory(rng: random.Random):
    def _make() -> LayoutState:
        s = LayoutState()
        for k in range(4):
            x0 = 0.25 * k + rng.uniform(-0.04, 0.04)
            s.add(layer="met1", x0=x0, y0=0.0, x1=x0 + 0.10, y1=0.10)
        return s
    return _make


def _topology_global_for(cell_name: str, dim: int = 32) -> np.ndarray:
    """Encode a real cell topology to a ``(dim,)`` numpy vector."""
    g = graph_from_template(load_template(cell_name))
    enc = TopologyEncoder(TopologyEncoderConfig(
        d_token=dim, n_layers=1, max_devices=16, max_nets=16,
    )).eval()
    with torch.no_grad():
        out = enc.encode_graphs([g])
    return out.global_embedding[0].cpu().numpy()


# ── Tests ────────────────────────────────────────────────────────────────────

def test_env_obs_space_includes_topology_when_provided():
    topo = _topology_global_for("inverter", dim=16)
    env = LayoutEnv(
        drc=_DirectFakeDRC(threshold_um=0.20),
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        max_steps=4,
        default_state_factory=_seed_factory(random.Random(0)),
        topology_global=topo,
    )
    assert "topology_global" in env.observation_space.spaces
    space = env.observation_space.spaces["topology_global"]
    assert isinstance(space, spaces.Box)
    assert space.shape == (16,)


def test_env_reset_emits_topology_global_in_obs():
    topo = _topology_global_for("inverter", dim=16)
    env = LayoutEnv(
        drc=_DirectFakeDRC(threshold_um=0.20),
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        max_steps=4,
        default_state_factory=_seed_factory(random.Random(0)),
        topology_global=topo,
    )
    obs, info = env.reset()
    assert "topology_global" in obs
    assert obs["topology_global"].shape == (16,)
    np.testing.assert_allclose(obs["topology_global"], topo, atol=1e-6)
    # Step should also keep the topology key.
    obs2, _, _, _, _ = env.step(env.action_space.sample())
    assert "topology_global" in obs2
    np.testing.assert_allclose(obs2["topology_global"], topo, atol=1e-6)


def test_env_without_topology_unchanged():
    """Default behavior (no topology arg) must keep Phase 1–3 obs schema."""
    env = LayoutEnv(
        drc=_DirectFakeDRC(threshold_um=0.20),
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        max_steps=4,
        default_state_factory=_seed_factory(random.Random(0)),
    )
    assert "topology_global" not in env.observation_space.spaces
    obs, _ = env.reset()
    assert "topology_global" not in obs


def test_maskable_policy_accepts_topology_via_obs_dict():
    cfg = LayoutPolicyConfig(
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
        use_topology=True, topology_dim=16,
    )
    obs_space = make_observation_space(
        poly_cap=cfg.poly_cap, viol_cap=cfg.viol_cap,
        topology_dim=cfg.topology_dim,
    )
    nvec = [N_KINDS, cfg.target_cap, 4, 2, 2, cfg.mag_bins]
    act_space = spaces.MultiDiscrete(nvec)
    policy = MaskableLayoutPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        layout_config=cfg,
    )

    # Build a small batch of obs that includes topology_global.
    s = LayoutState()
    s.add(layer="met1", x0=0.0, y0=0.0, x1=0.10, y1=0.10)
    s.add(layer="met1", x0=0.20, y0=0.0, x1=0.30, y1=0.10)
    obs_struct = build_observation(
        s, [], poly_cap=cfg.poly_cap, viol_cap=cfg.viol_cap,
        topology_global=np.ones(cfg.topology_dim, dtype=np.float32),
    )
    obs_dict = obs_struct.to_dict()
    batched = {k: torch.from_numpy(np.stack([v, v])) for k, v in obs_dict.items()}
    mask = action_mask_for(s, obs_struct.rid_to_idx,
                            target_cap=cfg.target_cap, mag_bins=cfg.mag_bins)
    batched_mask = torch.from_numpy(np.stack([mask, mask]))

    actions, values, log_probs = policy.forward(batched, action_masks=batched_mask)
    assert actions.shape == (2, 6)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()


def test_ppo_with_topology_does_not_crash():
    """Smoke: full PPOTrainer.learn() with a topology-conditioned env."""
    cfg = LayoutPolicyConfig(
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
        use_topology=True, topology_dim=16,
    )
    topo = _topology_global_for("inverter", dim=16)

    rng = random.Random(0)

    def _env_factory():
        return LayoutEnv(
            drc=_DirectFakeDRC(threshold_um=0.20),
            poly_cap=cfg.poly_cap, viol_cap=cfg.viol_cap,
            target_cap=cfg.target_cap, mag_bins=cfg.mag_bins,
            max_steps=4,
            default_state_factory=_seed_factory(rng),
            topology_global=topo,
        )

    trainer = PPOTrainer(
        env_factory=_env_factory,
        config=PPOConfig(
            n_envs=1, n_steps=64, batch_size=32, n_epochs=1,
            learning_rate=3e-4, seed=0, verbose=0,
        ),
        layout_config=cfg,
    )
    trainer.learn(total_timesteps=64)

    env = trainer.model.env
    obs = env.reset()
    masks = trainer.model.env.env_method("action_masks")
    actions, _ = trainer.model.predict(
        obs, action_masks=masks, deterministic=True,
    )
    assert actions.shape == (1, 6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
