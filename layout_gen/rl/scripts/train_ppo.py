"""
layout_gen.rl.scripts.train_ppo — CLI for the MaskablePPO trainer.

Run from the project root::

    .venv/bin/python -m layout_gen.rl.scripts.train_ppo \\
        --bc-init checkpoints/bc.pt \\
        --total-timesteps 100000 \\
        --n-envs 4 \\
        --out checkpoints/ppo.zip

Without ``--bc-init``, PPO trains from scratch (much slower; you almost
certainly want a BC warm-start).

If you don't have klayout/magic available yet, pass ``--synthetic`` to
use the fake DRC checker — useful for end-to-end smoke testing.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from layout_gen.synth.geo.state import LayoutState

from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.policy.network import LayoutPolicyConfig
from layout_gen.rl.training.ppo_train import PPOConfig, PPOTrainer
from layout_gen.rl.training.synthetic import fake_same_layer_spacing_check


# ── Synthetic env factory ────────────────────────────────────────────────────

class _DirectFakeDRC:
    """CachedDRC-shaped facade over the fake spacing check."""

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
                        description=f"min spacing: {self._threshold} um",
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


def _synth_state_factory(rng: random.Random):
    def _make() -> LayoutState:
        s = LayoutState()
        for k in range(6):
            x0 = 0.25 * k + rng.uniform(-0.04, 0.04)
            s.add(layer="met1", x0=x0, y0=0.0, x1=x0 + 0.10, y1=0.10)
        return s
    return _make


def _build_env_factory(args, env_seed: int):
    """Construct a LayoutEnv factory matching the CLI's chosen flavor."""
    if args.synthetic:
        rng = random.Random(env_seed)
        drc = _DirectFakeDRC(threshold_um=0.20)
        def _make():
            return LayoutEnv(
                drc=drc,
                poly_cap=args.poly_cap,
                viol_cap=args.viol_cap,
                target_cap=args.target_cap,
                mag_bins=args.mag_bins,
                max_steps=args.max_steps,
                default_state_factory=_synth_state_factory(rng),
            )
        return _make
    else:
        raise NotImplementedError(
            "Real-DRC env factory is wired in Phase 4 — pass --synthetic for now."
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                   help="Use the fake DRC checker (no klayout needed).")
    p.add_argument("--bc-init", type=Path, default=None,
                   help="Path to a BC checkpoint to warm-start the actor.")
    p.add_argument("--total-timesteps", type=int, default=20000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--poly-cap", type=int, default=64)
    p.add_argument("--viol-cap", type=int, default=16)
    p.add_argument("--target-cap", type=int, default=64)
    p.add_argument("--mag-bins", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("checkpoints/ppo.zip"))
    p.add_argument("--tb-log", type=Path, default=None,
                   help="TensorBoard log directory (optional).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)

    layout_cfg = LayoutPolicyConfig(
        poly_cap=args.poly_cap,
        viol_cap=args.viol_cap,
        target_cap=args.target_cap,
        mag_bins=args.mag_bins,
    )
    ppo_cfg = PPOConfig(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    env_factory = _build_env_factory(args, env_seed=args.seed)

    trainer = PPOTrainer(
        env_factory=env_factory,
        config=ppo_cfg,
        layout_config=layout_cfg,
        bc_init=args.bc_init,
        tensorboard_log=args.tb_log,
    )
    trainer.learn(total_timesteps=args.total_timesteps)
    trainer.save(args.out)
    print(f"[save] checkpoint -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
