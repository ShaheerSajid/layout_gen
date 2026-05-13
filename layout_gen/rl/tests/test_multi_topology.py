"""
layout_gen.rl.tests.test_multi_topology — multi-cell PPO + train_ppo CLI tests.

Verifies:
  * PPOTrainer accepts a list of env factories and round-robins them
    across vec-env workers.
  * Each worker actually calls a *different* factory (the easiest way
    to confirm round-robin: count factory invocations).
  * `train_ppo --topologies inv,nand2` runs end-to-end without crash
    and the topology GNN produces distinct ``topology_global`` vectors
    per cell (otherwise multi-cell training is no different from
    single-cell).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from layout_gen.rl.policy.network import LayoutPolicyConfig
from layout_gen.rl.scripts import train_ppo as train_ppo_cli
from layout_gen.rl.training.ppo_train import PPOConfig, PPOTrainer


# ── Stub env factory: counts invocations ────────────────────────────────────

class _CountingFactory:
    """Returns a tiny LayoutEnv instance and records each call."""

    def __init__(self, label: str, factory):
        self.label = label
        self._factory = factory
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self._factory()


def _tiny_factory():
    """Smallest possible self-contained env: REPAIR-only, no topology,
    no PLACE, no ROUTE — just enough for PPOTrainer to spin up."""
    import random
    from layout_gen.synth.geo.state import LayoutState
    from layout_gen.rl.env.layout_env import LayoutEnv
    from layout_gen.rl.scripts.train_ppo import _DirectFakeDRC

    rng = random.Random(0)

    def _state():
        s = LayoutState()
        for k in range(2):
            x0 = 0.20 * k + rng.uniform(-0.01, 0.01)
            s.add(layer="met1", x0=x0, y0=0.0, x1=x0 + 0.10, y1=0.10)
        return s

    def _make():
        return LayoutEnv(
            drc=_DirectFakeDRC(threshold_um=0.05),
            poly_cap=8, viol_cap=4, target_cap=8, mag_bins=4,
            max_steps=4,
            default_state_factory=_state,
        )
    return _make


def test_ppo_trainer_accepts_factory_list_and_round_robins():
    f1 = _CountingFactory("a", _tiny_factory())
    f2 = _CountingFactory("b", _tiny_factory())

    cfg = PPOConfig(n_envs=4, n_steps=16, batch_size=8, n_epochs=1, verbose=0)
    PPOTrainer(env_factory=[f1, f2], config=cfg)

    # n_envs=4, len(factories)=2 → each factory called twice (one per
    # worker assigned to it).
    assert f1.calls == 2, f"factory 'a' called {f1.calls}, expected 2"
    assert f2.calls == 2, f"factory 'b' called {f2.calls}, expected 2"


def test_ppo_trainer_single_callable_unchanged():
    """A single callable still works exactly as before."""
    f = _CountingFactory("single", _tiny_factory())
    cfg = PPOConfig(n_envs=3, n_steps=16, batch_size=8, n_epochs=1, verbose=0)
    PPOTrainer(env_factory=f, config=cfg)
    assert f.calls == 3


def test_train_ppo_cli_multi_topology_smoke(tmp_path: Path):
    """End-to-end: --topologies inverter,nand2 runs a few PPO updates."""
    out = tmp_path / "ppo_multi.zip"
    rc = train_ppo_cli.main([
        "--topologies", "inverter,nand2",
        "--enable-place", "--enable-route", "--no-drc",
        "--total-timesteps", "128",
        "--n-envs", "2",
        "--n-steps", "32", "--batch-size", "16", "--n-epochs", "1",
        "--max-place-steps", "4", "--max-route-steps", "4",
        "--max-steps", "10",
        "--device-cap", "8", "--net-cap", "8", "--position-bins", "8",
        "--route-size-bins", "4", "--mag-bins", "8",
        "--out", str(out), "--seed", "0",
    ])
    assert rc == 0, f"train_ppo CLI returned {rc}"
    assert out.exists(), f"expected checkpoint at {out}"


def test_topology_encoder_distinguishes_cells():
    """Multi-cell training is only useful if each cell produces a
    distinct ``topology_global``. Verify on inverter vs nand2."""
    from layout_gen.synth.loader import load_template
    from layout_gen.rl.topology import (
        TopologyEncoder, TopologyEncoderConfig, graph_from_template,
    )

    cell_params = {"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}}
    g_inv = graph_from_template(
        load_template("inverter"), cell_params=cell_params,
    )
    g_nand = graph_from_template(
        load_template("nand2"), cell_params=cell_params,
    )

    enc = TopologyEncoder(TopologyEncoderConfig(
        d_token=32, n_layers=2, max_devices=8, max_nets=8,
    )).eval()
    with torch.no_grad():
        topo_inv = enc.encode_graphs([g_inv]).global_embedding[0].cpu().numpy()
        topo_nand = enc.encode_graphs([g_nand]).global_embedding[0].cpu().numpy()
    diff = np.linalg.norm(topo_inv - topo_nand)
    assert diff > 1e-3, (
        f"inverter and nand2 produced near-identical topology embeddings "
        f"(diff={diff:.6f}); multi-cell training would be a no-op"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
