"""
layout_gen.rl.env — gymnasium-compatible layout repair / generation env.

Exports
-------
:class:`LayoutEnv`     — the gymnasium.Env (reset, step, render).
:class:`ActionSpace`   — composite (kind, target, edge, magnitude) action.
:class:`Observation`   — per-step observation (poly tensor + violation context).
:class:`RewardConfig`  — weights for the composite reward.
:class:`CachedDRC`     — DRC runner with geometry-hash caching.
"""
from __future__ import annotations

from layout_gen.rl.env.action_space import ActionSpace, EnvAction, action_mask_for
from layout_gen.rl.env.observation  import Observation, build_observation
from layout_gen.rl.env.reward       import RewardConfig, compute_reward
from layout_gen.rl.env.runner       import CachedDRC
from layout_gen.rl.env.layout_env   import LayoutEnv, EpisodeConfig

__all__ = [
    "ActionSpace", "EnvAction", "action_mask_for",
    "Observation", "build_observation",
    "RewardConfig", "compute_reward",
    "CachedDRC",
    "LayoutEnv", "EpisodeConfig",
]
