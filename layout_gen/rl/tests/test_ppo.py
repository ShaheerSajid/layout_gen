"""
layout_gen.rl.tests.test_ppo — end-to-end PPO smoke test.

Builds a tiny LayoutEnv with a fake DRC, wraps it in PPOTrainer, runs
a short ``learn()`` loop, and verifies:
  * Training does not crash.
  * The model's predictions are well-typed after training.
  * Save/load roundtrip works.

This is intentionally very small (a few thousand timesteps on a tiny
network) so it runs in <60 s on CPU.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from layout_gen.synth.geo.state import LayoutState

from layout_gen.rl.env.action_space import N_EDGES, N_KINDS
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.policy.network import LayoutPolicyConfig
from layout_gen.rl.training.ppo_train import PPOConfig, PPOTrainer


# ── Fake DRC + env factory ───────────────────────────────────────────────────

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


def _make_layout_env_factory(seed: int):
    rng = random.Random(seed)
    def _make():
        return LayoutEnv(
            drc=_DirectFakeDRC(threshold_um=0.20),
            poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
            max_steps=8,
            default_state_factory=_seed_factory(rng),
        )
    return _make


def _small_cfg() -> LayoutPolicyConfig:
    return LayoutPolicyConfig(
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_ppo_learn_does_not_crash(tmp_path: Path):
    torch.manual_seed(0)
    trainer = PPOTrainer(
        env_factory=_make_layout_env_factory(seed=0),
        config=PPOConfig(
            n_envs=1, n_steps=64, batch_size=32, n_epochs=2,
            learning_rate=3e-4, seed=0, verbose=0,
        ),
        layout_config=_small_cfg(),
    )
    trainer.learn(total_timesteps=128)

    # Predict an action — make sure the API surface is functional.
    env = trainer.model.env
    obs = env.reset()
    masks = trainer.model.env.env_method("action_masks")
    actions, _ = trainer.model.predict(obs, action_masks=masks, deterministic=True)
    assert actions.shape == (1, 6)


def test_ppo_save_load_roundtrip(tmp_path: Path):
    torch.manual_seed(0)
    trainer = PPOTrainer(
        env_factory=_make_layout_env_factory(seed=1),
        config=PPOConfig(
            n_envs=1, n_steps=64, batch_size=32, n_epochs=1,
            learning_rate=3e-4, seed=1, verbose=0,
        ),
        layout_config=_small_cfg(),
    )
    trainer.learn(total_timesteps=64)

    out = tmp_path / "ppo.zip"
    trainer.save(out)
    assert out.exists()

    loaded = PPOTrainer.load(
        out, env_factory=_make_layout_env_factory(seed=1),
        config=PPOConfig(n_envs=1, verbose=0),
        layout_config=_small_cfg(),
    )
    # Quick smoke: the loaded model can predict.
    env = loaded.model.env
    obs = env.reset()
    masks = loaded.model.env.env_method("action_masks")
    actions, _ = loaded.model.predict(obs, action_masks=masks, deterministic=True)
    assert actions.shape == (1, 6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
