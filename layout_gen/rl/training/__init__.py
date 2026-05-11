"""
layout_gen.rl.training — BC pretrain + (Phase 3) PPO fine-tune trainers.

Phase 2 exposes:
  * :class:`TrajectoryDataset` — load mined (perturbed_state,
    inverse_action) trajectories and yield (obs, action, validity) samples.
  * :func:`mine_synthetic_trajectories` — generate trajectories on-the-fly
    using a fake (or any) DRC runner; useful for tests and bring-up
    when no real klayout corpus is available.
  * :class:`BCTrainer` — cross-entropy trainer over the action dims.
"""
from __future__ import annotations

from layout_gen.rl.training.dataset import (
    TrajectoryDataset, TrajectorySample, encode_action_dict,
)
from layout_gen.rl.training.synthetic import (
    mine_synthetic_trajectories,
)
from layout_gen.rl.training.bc_pretrain import (
    BCTrainer, BCTrainerConfig, BCMetrics,
)
from layout_gen.rl.training.ppo_train import (
    PPOTrainer, PPOConfig, make_masked_env,
)

__all__ = [
    "TrajectoryDataset", "TrajectorySample", "encode_action_dict",
    "mine_synthetic_trajectories",
    "BCTrainer", "BCTrainerConfig", "BCMetrics",
    "PPOTrainer", "PPOConfig", "make_masked_env",
]
