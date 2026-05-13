"""
layout_gen.rl.tests.test_ablation — smoke test for the ablation harness.

Verifies the full train→eval→diff loop runs end-to-end on the
``ibrl`` preset with a tiny budget. Each variant trains for 64 steps
and evals 2 episodes; the test asserts the harness produces a CSV
with one row per variant and a printed comparison table.
"""
from __future__ import annotations

import csv
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

from layout_gen.rl.scripts import ablation as ablation_cli


def test_ablation_ibrl_preset_runs_end_to_end(tmp_path: Path):
    # The IBRL preset references {bc_init} — synthesise a tiny BC
    # checkpoint so the variant can substitute it in.
    from layout_gen.rl.policy.network import LayoutPolicy, LayoutPolicyConfig
    # Match train_ppo.py's default LayoutPolicyConfig so the BC
    # checkpoint loads cleanly into the PPO policy.
    cfg = LayoutPolicyConfig(
        poly_cap=128, viol_cap=32, target_cap=128, mag_bins=8,
        use_topology=True, topology_dim=64,
        enable_place=True, enable_route=True,
        device_cap=8, x_bins=8, y_bins=8,
        net_cap=8, route_x_bins=8, route_y_bins=8,
        route_w_bins=4, route_h_bins=4,
    )
    bc_policy = LayoutPolicy(cfg)
    bc_path = tmp_path / "bc.pt"
    torch.save({"state_dict": bc_policy.state_dict(),
                "config":     bc_policy.cfg.__dict__}, bc_path)

    out_dir = tmp_path / "ablation_runs"
    out_csv = tmp_path / "report.csv"

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ablation_cli.main([
            "--variants", "ibrl",
            "--bc-init", str(bc_path),
            "--topology", "inverter",
            "--no-drc",
            "--total-timesteps", "64",
            "--n-envs", "1",
            "--n-steps", "32",
            "--batch-size", "16",
            "--n-epochs", "1",
            "--max-place-steps", "4",
            "--max-route-steps", "4",
            "--max-steps", "10",
            "--device-cap", "8", "--net-cap", "8",
            "--position-bins", "8", "--route-size-bins", "4",
            "--mag-bins", "8",
            "--episodes", "2",
            "--out-dir", str(out_dir),
            "--out-csv", str(out_csv),
        ])
    assert rc == 0, f"ablation main returned {rc}"
    text = buf.getvalue()

    # CSV exists and has both variants.
    assert out_csv.exists()
    rows = list(csv.DictReader(out_csv.open()))
    assert {r["name"] for r in rows} == {"bc_only", "bc_distill"}
    # Comparison table appeared in stdout.
    assert "ablation comparison" in text
    assert "bc_only" in text
    assert "bc_distill" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
