"""
layout_gen.rl.tests.test_sb3_policy — sanity tests for MaskableLayoutPolicy.

Verifies the SB3 wrapper plumbs LayoutPolicy correctly:
  * forward returns (actions, values, log_probs) of correct shape.
  * Action masking respects the per-dim mask (masked-out kinds never sampled).
  * BC checkpoint loads into the wrapped actor with no missing keys.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gymnasium import spaces

from layout_gen.synth.geo.state import LayoutState

from layout_gen.rl.env.action_space import (
    DEFAULT_MAG_BINS, DEFAULT_TARGET_CAP, N_EDGES, N_KINDS,
    action_mask_for,
)
from layout_gen.rl.env.observation import (
    DEFAULT_POLY_CAP, DEFAULT_VIOL_CAP, build_observation,
    make_observation_space,
)
from layout_gen.rl.policy import (
    LayoutPolicy, LayoutPolicyConfig, MaskableLayoutPolicy,
    load_bc_into_sb3_policy,
)
from layout_gen.rl.training import (
    BCTrainer, BCTrainerConfig, TrajectoryDataset,
    mine_synthetic_trajectories,
)
from layout_gen.rl.training.synthetic import SyntheticMineConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _small_cfg() -> LayoutPolicyConfig:
    """Tiny policy for fast tests."""
    return LayoutPolicyConfig(
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
    )


def _make_obs_and_mask(cfg: LayoutPolicyConfig, batch: int = 2):
    """Build a small observation with two live polygons."""
    s = LayoutState()
    s.add(layer="met1", x0=0.0, y0=0.0, x1=0.10, y1=0.10)
    s.add(layer="met1", x0=0.20, y0=0.0, x1=0.30, y1=0.10)

    obs = build_observation(s, [], poly_cap=cfg.poly_cap, viol_cap=cfg.viol_cap)
    obs_dict = obs.to_dict()

    mask = action_mask_for(s, obs.rid_to_idx,
                           target_cap=cfg.target_cap, mag_bins=cfg.mag_bins)

    # Add a batch dim (B=2 by repeating)
    batched_obs = {
        k: torch.from_numpy(np.stack([v] * batch)) for k, v in obs_dict.items()
    }
    batched_mask = torch.from_numpy(np.stack([mask] * batch))
    return batched_obs, batched_mask


def _build_policy(cfg: LayoutPolicyConfig) -> MaskableLayoutPolicy:
    obs_space = make_observation_space(poly_cap=cfg.poly_cap, viol_cap=cfg.viol_cap)
    nvec = [N_KINDS, cfg.target_cap, N_EDGES, 2, 2, cfg.mag_bins]
    act_space = spaces.MultiDiscrete(nvec)
    return MaskableLayoutPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        layout_config=cfg,
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_forward_returns_well_typed_tuple():
    cfg = _small_cfg()
    policy = _build_policy(cfg)
    obs, mask = _make_obs_and_mask(cfg, batch=2)

    actions, values, log_probs = policy.forward(obs, action_masks=mask)
    assert actions.shape == (2, 6)         # 6 MultiDiscrete dims
    assert values.shape  == (2,)
    assert log_probs.shape == (2,)
    assert torch.all(torch.isfinite(values))
    assert torch.all(torch.isfinite(log_probs))


def test_evaluate_actions_consistent_with_forward():
    cfg = _small_cfg()
    policy = _build_policy(cfg)
    obs, mask = _make_obs_and_mask(cfg, batch=2)

    actions, _, log_probs_fwd = policy.forward(obs, action_masks=mask,
                                                deterministic=True)
    values, log_probs_eval, entropy = policy.evaluate_actions(
        obs, actions, action_masks=mask,
    )
    assert torch.allclose(log_probs_fwd, log_probs_eval, atol=1e-5)
    assert entropy.shape == (2,)
    assert torch.all(torch.isfinite(entropy))


def test_action_mask_excludes_kinds():
    """If a kind is masked off, deterministic forward must not pick it."""
    cfg = _small_cfg()
    policy = _build_policy(cfg)
    obs, mask = _make_obs_and_mask(cfg, batch=4)

    # Force shift_edge (idx 0) to be the only allowed kind.
    flat_mask = mask.clone()
    flat_mask[:, :N_KINDS] = False
    flat_mask[:, 0] = True

    actions, _, _ = policy.forward(obs, action_masks=flat_mask,
                                    deterministic=True)
    assert torch.all(actions[:, 0] == 0), \
        f"Expected all kinds to be 0, got {actions[:, 0]}"


def test_bc_warmstart_loads_cleanly(tmp_path: Path):
    cfg = _small_cfg()

    # 1) BC-train a tiny policy on synthetic trajectories.
    rng = random.Random(0)
    traj_dir = tmp_path / "trajs"

    def _seed():
        s = LayoutState()
        for k in range(4):
            x0 = 0.25 * k + rng.uniform(-0.005, 0.005)
            s.add(layer="met1", x0=x0, y0=0.0, x1=x0 + 0.10, y1=0.10)
        return s

    counts = mine_synthetic_trajectories(
        state_factory=_seed, out_dir=traj_dir,
        config=SyntheticMineConfig(
            n_trajectories=64, depths=(1,),
            forbid_kinds=frozenset({"delete_rect", "shrink_rect", "grow_rect"}),
        ),
        rng=rng,
    )
    if counts["kept"] < 4:
        pytest.skip(f"insufficient synthetic trajectories: {counts}")

    dataset = TrajectoryDataset(traj_dir,
                                 poly_cap=cfg.poly_cap,
                                 viol_cap=cfg.viol_cap,
                                 target_cap=cfg.target_cap,
                                 mag_bins=cfg.mag_bins)
    bc_policy = LayoutPolicy(cfg)
    trainer = BCTrainer(bc_policy, BCTrainerConfig(epochs=2, batch_size=8))
    trainer.fit(dataset)

    bc_path = tmp_path / "bc.pt"
    trainer.save(bc_path)

    # 2) Build PPO policy with matching layout_config; load BC weights.
    ppo_policy = _build_policy(cfg)
    loaded, missing = load_bc_into_sb3_policy(bc_path, ppo_policy, strict=True)
    assert missing == 0, f"BC checkpoint missed {missing} keys"
    assert loaded > 0, "No params loaded from BC checkpoint"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
