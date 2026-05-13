"""
layout_gen.rl.training.ppo_train — MaskablePPO trainer wrapped around LayoutEnv.

Phase 3: turn the BC-pretrained policy into an RL agent that improves
under the env's DRC reward. Curriculum is intentionally minimal in v1
(fixed perturbation depth from the env's state factory); deeper
curricula are added once the basic loop is validated.

Usage::

    trainer = PPOTrainer(
        env_factory=lambda: LayoutEnv(drc=..., default_state_factory=...),
        bc_init="checkpoints/bc.pt",
    )
    trainer.learn(total_timesteps=100_000)
    trainer.save("checkpoints/ppo.zip")

The trainer is a thin wrapper around ``MaskablePPO``. Behaviour the user
typically wants to tune (n_steps, batch_size, learning_rate, …) is
exposed via :class:`PPOConfig`; everything else falls back to MaskablePPO
defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from sb3_contrib import MaskablePPO

from layout_gen.rl.training.ibrl import MaskableBCDistillPPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.policy.network import LayoutPolicyConfig
from layout_gen.rl.policy.sb3 import (
    MaskableLayoutPolicy, load_bc_into_sb3_policy,
)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    n_envs:             int   = 1
    n_steps:            int   = 256
    batch_size:         int   = 64
    n_epochs:           int   = 4
    learning_rate:      float = 3e-4
    gamma:              float = 0.99
    gae_lambda:         float = 0.95
    clip_range:         float = 0.2
    ent_coef:           float = 0.01
    vf_coef:            float = 0.5
    max_grad_norm:      float = 0.5
    target_kl:          float | None = None
    seed:               int   = 0
    device:             str   = "cpu"
    verbose:            int   = 1


# ── Helpers ──────────────────────────────────────────────────────────────────

def _action_masks_for(env: LayoutEnv):
    """Function ActionMasker calls on each step to retrieve the mask."""
    return env.action_masks()


def make_masked_env(env_factory: Callable[[], LayoutEnv]) -> Callable[[], Monitor]:
    """Wrap *env_factory* output in ActionMasker (inner) + Monitor (outer).

    Order matters:
      * ActionMasker is innermost so its ``_action_mask_fn`` receives the
        raw :class:`LayoutEnv` (which has ``action_masks()``).
      * Monitor wraps the masker so SB3 logs ``ep_rew_mean`` /
        ``ep_len_mean`` per rollout. ``get_wrapper_attr`` drills through
        Monitor to find ``action_masks`` on the inner ActionMasker.
    """
    def _make() -> Monitor:
        return Monitor(ActionMasker(env_factory(), _action_masks_for))
    return _make


# ── Trainer ──────────────────────────────────────────────────────────────────

class PPOTrainer:
    """MaskablePPO + LayoutEnv glue.

    Parameters
    ----------
    env_factory :
        Either a single callable returning a fresh :class:`LayoutEnv`
        (single-cell training) **or** a list of such callables —
        one per cell — for multi-topology training. With a list, the
        N PPO vec-env workers round-robin through the factories so
        every cell gets at least one worker. ``n_envs`` should be
        a multiple of ``len(factories)`` for clean per-cell
        reporting; if smaller, some cells go unseen per rollout
        (fine, just slower convergence on those).
    config :
        Hyper-parameters (:class:`PPOConfig`).
    layout_config :
        Override for the underlying :class:`LayoutPolicyConfig`. Must
        match the env's poly_cap / viol_cap / target_cap / mag_bins
        — when training across cells, use the **max** of each cap
        across all cells so the action-space shape is uniform.
    bc_init :
        Optional path to a BC checkpoint (saved by :class:`BCTrainer`).
        Loaded into the actor before training begins.
    tensorboard_log :
        Optional dir for TensorBoard logs. Useful for tracking reward
        and DRC-clean rate over time.
    """

    def __init__(
        self,
        env_factory:  Callable[[], LayoutEnv] | Sequence[Callable[[], LayoutEnv]],
        *,
        config:       PPOConfig | None = None,
        layout_config: LayoutPolicyConfig | None = None,
        bc_init:      str | Path | None = None,
        tensorboard_log: str | Path | None = None,
        # ── IBRL via BC distillation (arXiv 2311.02198 spirit) ───────
        # When set, PPO's loss adds β·KL(π_PPO || π_BC) with β decaying
        # linearly from ``ibrl_beta_start`` to ``ibrl_beta_end`` over
        # the full training run. Pairs naturally with ``bc_init`` —
        # use the same checkpoint for both.
        ibrl_bc_checkpoint: str | Path | None = None,
        ibrl_beta_start:    float = 1.0,
        ibrl_beta_end:      float = 0.0,
    ) -> None:
        self.cfg = config or PPOConfig()
        self.layout_config = layout_config or LayoutPolicyConfig()

        factories = (
            list(env_factory) if not callable(env_factory) else [env_factory]
        )
        # Round-robin assignment: vec-env worker i gets factory i % len.
        masked_factories = [
            make_masked_env(factories[i % len(factories)])
            for i in range(self.cfg.n_envs)
        ]
        vec_env = DummyVecEnv(masked_factories)

        ppo_kwargs = dict(
            policy=MaskableLayoutPolicy,
            env=vec_env,
            learning_rate=self.cfg.learning_rate,
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_range=self.cfg.clip_range,
            ent_coef=self.cfg.ent_coef,
            vf_coef=self.cfg.vf_coef,
            max_grad_norm=self.cfg.max_grad_norm,
            target_kl=self.cfg.target_kl,
            seed=self.cfg.seed,
            device=self.cfg.device,
            verbose=self.cfg.verbose,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            policy_kwargs={"layout_config": self.layout_config},
        )
        if ibrl_bc_checkpoint is not None:
            self.model = MaskableBCDistillPPO(
                **ppo_kwargs,
                bc_checkpoint=ibrl_bc_checkpoint,
                bc_policy_config=self.layout_config,
                beta_start=ibrl_beta_start,
                beta_end=ibrl_beta_end,
            )
            if self.cfg.verbose:
                print(f"[ibrl] BC distillation enabled "
                      f"(β: {ibrl_beta_start} → {ibrl_beta_end})")
        else:
            self.model = MaskablePPO(**ppo_kwargs)

        if bc_init is not None:
            loaded, missing = load_bc_into_sb3_policy(
                bc_init, self.model.policy, strict=True,
            )
            if self.cfg.verbose:
                print(f"[ppo] loaded {loaded} BC params (missing={missing})")

    # ── Public API ───────────────────────────────────────────────────────────

    def learn(
        self,
        total_timesteps: int,
        *,
        callback: BaseCallback | None = None,
        log_interval: int = 1,
        progress_bar: bool = False,
    ) -> None:
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            progress_bar=progress_bar,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path,
             env_factory: Callable[[], LayoutEnv] | Sequence[Callable[[], LayoutEnv]],
             config: PPOConfig | None = None,
             layout_config: LayoutPolicyConfig | None = None,
             ) -> "PPOTrainer":
        # Manually construct a trainer skeleton, then load weights.
        trainer = cls.__new__(cls)
        trainer.cfg = config or PPOConfig()
        trainer.layout_config = layout_config or LayoutPolicyConfig()
        factories = (
            list(env_factory) if not callable(env_factory) else [env_factory]
        )
        masked_factories = [
            make_masked_env(factories[i % len(factories)])
            for i in range(trainer.cfg.n_envs)
        ]
        vec_env = DummyVecEnv(masked_factories)
        trainer.model = MaskablePPO.load(
            str(path), env=vec_env, device=trainer.cfg.device,
            custom_objects={"policy_kwargs": {"layout_config": trainer.layout_config}},
        )
        return trainer


__all__ = ["PPOConfig", "PPOTrainer", "make_masked_env"]
