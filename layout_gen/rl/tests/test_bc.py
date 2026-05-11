"""
layout_gen.rl.tests.test_bc — BC trainer end-to-end smoke test.

Mines synthetic trajectories, trains for a few epochs, and verifies:
  * Loss decreases meaningfully (≥ 25% drop train→end).
  * Validation accuracy on the kind dim climbs above random chance.
  * Save → load roundtrip preserves the policy's outputs.

Runs in <30 s on CPU — no klayout / GPU required.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from layout_gen.synth.geo.state import LayoutState

from layout_gen.rl.env.action_space import N_KINDS
from layout_gen.rl.policy import LayoutPolicy, LayoutPolicyConfig
from layout_gen.rl.training import (
    BCTrainer, BCTrainerConfig, TrajectoryDataset,
    mine_synthetic_trajectories,
)
from layout_gen.rl.training.synthetic import SyntheticMineConfig


def _seed_factory(rng: random.Random):
    """Seed centres are 0.25 µm apart; perturbations of ≥ 0.06 µm push them
    below the 0.20 µm fake-DRC threshold reliably."""
    def _make() -> LayoutState:
        s = LayoutState()
        for k in range(6):
            x0 = 0.25 * k + rng.uniform(-0.005, 0.005)
            s.add(layer="met1", x0=x0, y0=0.0,
                  x1=x0 + 0.10, y1=0.10)
        return s
    return _make


def test_bc_pretrain_decreases_loss(tmp_path: Path):
    torch.manual_seed(0)
    rng = random.Random(0)

    # 1) Mine a synthetic corpus.
    traj_dir = tmp_path / "trajs"
    counts = mine_synthetic_trajectories(
        state_factory=_seed_factory(rng),
        out_dir=traj_dir,
        config=SyntheticMineConfig(
            n_trajectories=256,
            depths=(1,),
            forbid_kinds=frozenset({
                "delete_rect", "nudge_offgrid",
                "shrink_rect", "grow_rect",
            }),
        ),
        rng=rng,
    )
    # Random translates produce a violation ~25% of the time given the
    # seed spacing; 256 attempts → ~60+ kept comfortably exceeds the floor.
    assert counts["kept"] >= 32, f"Need at least 32 trajectories: {counts}"

    # 2) Build dataset + small policy + trainer.
    dataset = TrajectoryDataset(traj_dir)
    policy  = LayoutPolicy(LayoutPolicyConfig(
        d_token=32, d_trunk=64, n_layers=1, n_heads=4, dim_ff=64,
    ))
    trainer = BCTrainer(policy, BCTrainerConfig(
        epochs=8, batch_size=16, lr=1e-3, val_fraction=0.2,
        log_every=10, num_workers=0,
    ))

    metrics = trainer.fit(dataset)

    # 3) Verify training is doing something.
    assert len(metrics.train_loss) >= 2
    first  = metrics.train_loss[0]
    last   = metrics.train_loss[-1]
    assert last < 0.75 * first, (
        f"Train loss did not decrease enough: {first:.4f} -> {last:.4f}"
    )

    # Kind-head validation accuracy should beat random (1 / N_KINDS).
    if metrics.accuracy:
        last_acc = metrics.accuracy[-1].get("kind", 0.0)
        random_baseline = 1.0 / N_KINDS
        assert last_acc > random_baseline, (
            f"Kind acc {last_acc:.3f} not above random baseline {random_baseline:.3f}"
        )

    # 4) Save → load roundtrip.
    ckpt = tmp_path / "bc.pt"
    trainer.save(ckpt)

    obs_keys = ["poly_feats", "poly_mask", "viol_feats", "viol_mask",
                "global_feats"]
    sample = dataset[0]["obs"]
    obs_batch = {k: sample[k].unsqueeze(0) for k in obs_keys}

    policy.eval()
    with torch.no_grad():
        out_a = policy(obs_batch)

    trainer2 = BCTrainer.load(ckpt)
    trainer2.policy.eval()
    with torch.no_grad():
        out_b = trainer2.policy(obs_batch)
    for a, b in zip(out_a, out_b):
        assert torch.allclose(a, b, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
