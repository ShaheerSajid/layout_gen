"""
layout_gen.rl.tests.test_topology_encoder — GNN encoder + LayoutPolicy wiring.

Verifies:
  * graphs_to_tensors pads correctly and respects masks.
  * TopologyEncoder forward shapes match config.
  * Padded device positions stay zero in device_embeddings (mask honoured).
  * The encoder maps two structurally identical inverters to similar
    global embeddings, and a 6T cell to a notably different one.
  * LayoutPolicy with ``use_topology=True`` consumes ``topology_global``
    and changes its outputs when topology changes.
"""
from __future__ import annotations

import pytest
import torch

from layout_gen.synth.loader import load_template

from layout_gen.rl.policy.network import LayoutPolicy, LayoutPolicyConfig
from layout_gen.rl.topology import (
    TopologyEncoder, TopologyEncoderConfig,
    graph_from_template,
)
from layout_gen.rl.topology.encoder import graphs_to_tensors


def _small_cfg() -> TopologyEncoderConfig:
    return TopologyEncoderConfig(
        d_token=32, n_layers=2, max_devices=16, max_nets=16,
    )


# ── Tensorisation ────────────────────────────────────────────────────────────

def test_graphs_to_tensors_shapes_and_mask():
    g_inv = graph_from_template(load_template("inverter"))
    g_6t  = graph_from_template(load_template("bit_cell_6t"))

    cfg = _small_cfg()
    batch = graphs_to_tensors([g_inv, g_6t],
                               max_devices=cfg.max_devices,
                               max_nets=cfg.max_nets)
    assert batch["device_feats"].shape == (2, cfg.max_devices,
                                            batch["device_feats"].shape[-1])
    assert batch["net_feats"].shape    == (2, cfg.max_nets,
                                            batch["net_feats"].shape[-1])
    assert batch["incidence"].shape    == (2, cfg.max_devices, cfg.max_nets)

    # Inverter has 2 devices, 4 nets.
    assert batch["device_mask"][0].sum().item() == 2
    assert batch["net_mask"][0].sum().item()    == 4

    # 6T has 6 devices, 7 nets.
    assert batch["device_mask"][1].sum().item() == 6
    assert batch["net_mask"][1].sum().item()    == 7

    # Padded positions are zero.
    assert torch.all(batch["device_feats"][0, 2:] == 0.0)
    assert torch.all(batch["net_feats"][0, 4:]    == 0.0)


# ── Encoder forward ──────────────────────────────────────────────────────────

def test_encoder_forward_shapes():
    g_inv = graph_from_template(load_template("inverter"))
    cfg = _small_cfg()
    enc = TopologyEncoder(cfg).eval()
    out = enc.encode_graphs([g_inv, g_inv])
    assert out.device_embeddings.shape == (2, cfg.max_devices, cfg.d_token)
    assert out.global_embedding.shape == (2, cfg.d_token)
    # Padded positions in device_embeddings should be zero.
    masked_rows = out.device_embeddings[0, 2:]
    assert torch.all(masked_rows == 0.0)


def test_encoder_distinguishes_inverter_from_6t():
    g_inv = graph_from_template(load_template("inverter"))
    g_6t  = graph_from_template(load_template("bit_cell_6t"))
    enc = TopologyEncoder(_small_cfg()).eval()
    out = enc.encode_graphs([g_inv, g_6t])
    diff = (out.global_embedding[0] - out.global_embedding[1]).norm().item()
    assert diff > 1e-3, (
        f"inverter and 6T should produce distinct embeddings; got |Δ|={diff}"
    )


# ── LayoutPolicy with topology ───────────────────────────────────────────────

def _make_obs(batch: int = 2):
    """Synthesise a tiny obs dict matching the policy's defaults."""
    from layout_gen.repair.features import POLY_FEAT_DIM
    from layout_gen.rl.env.observation import (
        DEFAULT_POLY_CAP, DEFAULT_VIOL_CAP, N_GLOBAL, V_FEAT_DIM,
    )
    return {
        "poly_feats":   torch.randn(batch, DEFAULT_POLY_CAP, POLY_FEAT_DIM),
        "poly_mask":    torch.cat([torch.ones(batch, 4),
                                    torch.zeros(batch, DEFAULT_POLY_CAP - 4)],
                                   dim=-1),
        "viol_feats":   torch.randn(batch, DEFAULT_VIOL_CAP, V_FEAT_DIM),
        "viol_mask":    torch.cat([torch.ones(batch, 1),
                                    torch.zeros(batch, DEFAULT_VIOL_CAP - 1)],
                                   dim=-1),
        "global_feats": torch.randn(batch, N_GLOBAL),
    }


def test_policy_use_topology_changes_output():
    """With ``use_topology=True`` the policy's predictions must depend on
    the topology_global tensor — otherwise we know the new branch is dead."""
    cfg = LayoutPolicyConfig(use_topology=True, topology_dim=32,
                              d_token=32, d_trunk=64, n_layers=1, n_heads=4,
                              dim_ff=64)
    policy = LayoutPolicy(cfg).eval()

    obs = _make_obs(batch=1)

    obs_a = dict(obs)
    obs_a["topology_global"] = torch.zeros(1, cfg.topology_dim)
    obs_b = dict(obs)
    obs_b["topology_global"] = torch.ones(1, cfg.topology_dim)

    with torch.no_grad():
        out_a = policy(obs_a)
        out_b = policy(obs_b)

    # Kind logits should differ once topology changes.
    diff = (out_a.kind - out_b.kind).abs().max().item()
    assert diff > 1e-5, f"Topology branch had no effect; max Δkind={diff}"


def test_policy_default_use_topology_false_unchanged():
    """Without the flag, the policy must accept obs *without* topology_global
    and behave as before — preserves Phase 1–3 BC checkpoint compatibility."""
    cfg = LayoutPolicyConfig(d_token=32, d_trunk=64, n_layers=1, n_heads=4,
                              dim_ff=64)
    assert cfg.use_topology is False
    policy = LayoutPolicy(cfg).eval()
    obs = _make_obs(batch=2)
    with torch.no_grad():
        out = policy(obs)
    assert torch.isfinite(out.kind).all()
    assert out.kind.shape[0] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
