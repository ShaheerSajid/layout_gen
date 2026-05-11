"""
layout_gen.rl.tests.test_policy — unit tests for LayoutPolicy.

Verifies:
  * Forward pass produces logits with the expected per-dim shapes.
  * All-pad rows (zero polygons or zero violations) don't crash or NaN.
  * The pointer-style target head respects the polygon mask (masked
    positions get -inf logits).
  * Saving and reloading a checkpoint reproduces the same outputs.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from layout_gen.repair.features import POLY_FEAT_DIM
from layout_gen.rl.env.action_space import (
    DEFAULT_MAG_BINS, DEFAULT_TARGET_CAP, N_EDGES, N_KINDS,
)
from layout_gen.rl.env.observation import (
    DEFAULT_POLY_CAP, DEFAULT_VIOL_CAP, N_GLOBAL, V_FEAT_DIM,
)
from layout_gen.rl.policy import LayoutPolicy, LayoutPolicyConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_obs(batch: int = 2,
              n_poly_per: int = 6,
              n_viol_per: int = 2) -> dict[str, torch.Tensor]:
    poly_feats = torch.zeros(batch, DEFAULT_POLY_CAP, POLY_FEAT_DIM)
    poly_mask  = torch.zeros(batch, DEFAULT_POLY_CAP)
    viol_feats = torch.zeros(batch, DEFAULT_VIOL_CAP, V_FEAT_DIM)
    viol_mask  = torch.zeros(batch, DEFAULT_VIOL_CAP)
    global_feats = torch.zeros(batch, N_GLOBAL)

    rng = np.random.default_rng(0)
    poly_feats.normal_(generator=torch.Generator().manual_seed(0))
    viol_feats.normal_(generator=torch.Generator().manual_seed(1))
    global_feats.uniform_(0.0, 1.0,
                          generator=torch.Generator().manual_seed(2))
    poly_mask[:, :n_poly_per] = 1.0
    viol_mask[:, :n_viol_per] = 1.0
    return {
        "poly_feats":   poly_feats,
        "poly_mask":    poly_mask,
        "viol_feats":   viol_feats,
        "viol_mask":    viol_mask,
        "global_feats": global_feats,
    }


# ── Tests ────────────────────────────────────────────────────────────────────

def test_forward_logits_shapes():
    cfg = LayoutPolicyConfig()
    policy = LayoutPolicy(cfg).eval()
    out = policy(_make_obs(batch=3))
    assert out.kind.shape   == (3, N_KINDS)
    assert out.target.shape == (3, cfg.target_cap)
    assert out.edge.shape   == (3, N_EDGES)
    assert out.sign_x.shape == (3, 2)
    assert out.sign_y.shape == (3, 2)
    assert out.mag.shape    == (3, cfg.mag_bins)
    for tensor in (out.kind, out.target, out.edge, out.sign_x,
                   out.sign_y, out.mag):
        assert torch.all(torch.isfinite(tensor) | torch.isinf(tensor))


def test_all_pad_row_does_not_nan():
    """A row with zero valid polygons and zero violations must not crash
    or produce NaN logits in the kind/edge/sign/mag heads."""
    obs = _make_obs(batch=2, n_poly_per=0, n_viol_per=0)
    policy = LayoutPolicy().eval()
    out = policy(obs)
    for name in ("kind", "edge", "sign_x", "sign_y", "mag"):
        t = getattr(out, name)
        assert torch.isfinite(t).all(), f"{name} contains NaN/Inf"


def test_target_head_respects_polygon_mask():
    """Masked polygon positions must receive -inf logits so they're never
    sampled by the policy."""
    obs = _make_obs(batch=1, n_poly_per=3, n_viol_per=1)
    policy = LayoutPolicy().eval()
    out = policy(obs)
    # Positions 3..POLY_CAP should be -inf (masked-out).
    masked = out.target[0, 3:]
    assert torch.all(torch.isinf(masked) & (masked < 0)), (
        f"Expected -inf at masked positions, got: {masked[:5]}"
    )
    # Positions 0..2 should be finite logits.
    assert torch.all(torch.isfinite(out.target[0, :3]))


def test_save_and_reload_roundtrip(tmp_path: Path):
    cfg = LayoutPolicyConfig(d_token=32, d_trunk=64, n_layers=1)
    policy = LayoutPolicy(cfg).eval()

    obs = _make_obs(batch=1)
    with torch.no_grad():
        out_a = policy(obs)

    ckpt = tmp_path / "p.pt"
    torch.save({"state_dict": policy.state_dict(),
                "config":     policy.cfg.__dict__}, ckpt)

    raw = torch.load(ckpt, weights_only=False, map_location="cpu")
    policy2 = LayoutPolicy(LayoutPolicyConfig(**raw["config"]))
    policy2.load_state_dict(raw["state_dict"])
    policy2.eval()

    with torch.no_grad():
        out_b = policy2(obs)

    for a, b in zip(out_a, out_b):
        assert torch.allclose(a, b, atol=1e-6), \
            "Reload produced different logits."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
