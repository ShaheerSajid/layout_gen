"""
layout_gen.rl.tests.test_rgcn — typed-edge R-GCN extension to the topology encoder.

Covers:
  * `_extract_typed_edges` pulls (align_gate / abut_x / shared_diffusion)
    edges from placement_logic + placement_relations.
  * `graphs_to_tensors` emits a (B, T, D_max, D_max) device_adj tensor
    with the right per-type sparsity.
  * The encoder's R-GCN branch actually consumes those edges: an
    inverter with its align_gate edge present produces a different
    global embedding than the same inverter with all typed edges
    stripped.
  * Per-type weights are distinct (the encoder treats align_gate and
    abut_x differently — sanity check on the parametrisation).
  * Existing 'no typed edges' callers still work (backward-compat).
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from layout_gen.synth.loader import load_template

from layout_gen.rl.topology import (
    DEVICE_EDGE_TYPES, TopologyEncoder, TopologyEncoderConfig,
    graph_from_template,
)
from layout_gen.rl.topology.encoder import (
    _DEVICE_EDGE_TYPE_TO_IDX, graphs_to_tensors,
)


def _cfg() -> TopologyEncoderConfig:
    return TopologyEncoderConfig(
        d_token=32, n_layers=2, max_devices=8, max_nets=8,
    )


# ── Edge extraction ──────────────────────────────────────────────────────────

def test_inverter_yields_one_align_gate_edge():
    g = graph_from_template(load_template("inverter"))
    # Inverter's placement_logic has one PMOS → align_gate(NMOS) directive.
    types = [t for _, _, t in g.device_edges]
    assert types == ["align_gate"]
    # Sorted canonical (i<j); inverter device order is N(0), P(1).
    assert g.device_edges == [(0, 1, "align_gate")]


def test_6t_yields_align_gate_and_abut_x_edges():
    g = graph_from_template(load_template("bit_cell_6t"))
    # The 6T YAML has 3 abut_x rows and 2 align_gate cross-row pairs.
    kinds = sorted(t for _, _, t in g.device_edges)
    assert kinds.count("align_gate") >= 1
    assert kinds.count("abut_x")     >= 1
    # Every edge canonicalised i<j and known type.
    for (a, b, kind) in g.device_edges:
        assert a < b
        assert kind in DEVICE_EDGE_TYPES


# ── Tensorisation ───────────────────────────────────────────────────────────

def test_graphs_to_tensors_emits_device_adj_with_right_sparsity():
    g_inv = graph_from_template(load_template("inverter"))
    batch = graphs_to_tensors([g_inv], max_devices=8, max_nets=8)
    adj = batch["device_adj"]
    T = len(DEVICE_EDGE_TYPES)
    assert adj.shape == (1, T, 8, 8)

    t = _DEVICE_EDGE_TYPE_TO_IDX["align_gate"]
    # Symmetric and exactly the (0,1)/(1,0) entries set.
    assert adj[0, t, 0, 1].item() == 1.0
    assert adj[0, t, 1, 0].item() == 1.0
    # No self-loops.
    for i in range(8):
        assert adj[0, t, i, i].item() == 0.0
    # Other types are all zero for the inverter.
    for other in (k for k in range(T) if k != t):
        assert adj[0, other].sum().item() == 0.0


def test_device_adj_clipped_at_max_devices():
    """When max_devices is smaller than the graph, edges referencing
    out-of-range devices must not be written."""
    g = graph_from_template(load_template("bit_cell_6t"))
    batch = graphs_to_tensors([g], max_devices=3, max_nets=8)
    adj = batch["device_adj"]
    # All entries must be within [:3, :3]; nothing should bleed past.
    assert adj.shape == (1, len(DEVICE_EDGE_TYPES), 3, 3)
    # 6T edges that reach indices ≥3 are dropped; the surviving subset
    # has at most C(3,2)=3 unique pairs × 2 symmetric entries = 6 ones.
    assert adj.sum().item() <= 6


# ── Encoder behaviour: R-GCN branch is live ─────────────────────────────────

def _strip_typed_edges(g):
    g2 = deepcopy(g)
    g2.device_edges = []
    return g2


def test_encoder_responds_to_typed_edges():
    """Encoder output must change when typed edges are present vs
    absent — otherwise the R-GCN projections are dead weight."""
    g_full   = graph_from_template(load_template("bit_cell_6t"))
    g_naked  = _strip_typed_edges(g_full)
    assert len(g_full.device_edges) > 0
    assert g_naked.device_edges == []

    enc = TopologyEncoder(_cfg()).eval()
    out = enc.encode_graphs([g_full, g_naked])
    diff = (out.global_embedding[0] - out.global_embedding[1]).norm().item()
    assert diff > 1e-3, (
        f"R-GCN branch had no effect on the global embedding; |Δ|={diff}"
    )


def test_encoder_distinguishes_align_gate_from_abut_x():
    """Replace a graph's edges with all-align_gate vs all-abut_x — the
    encoder must produce different embeddings (per-type weights are
    actually distinct)."""
    g = graph_from_template(load_template("inverter"))
    g_align = deepcopy(g)
    g_align.device_edges = [(0, 1, "align_gate")]
    g_abut  = deepcopy(g)
    g_abut.device_edges  = [(0, 1, "abut_x")]

    enc = TopologyEncoder(_cfg()).eval()
    out = enc.encode_graphs([g_align, g_abut])
    diff = (out.global_embedding[0] - out.global_embedding[1]).norm().item()
    assert diff > 1e-4, (
        f"per-type weights collapsed to the same function; |Δ|={diff}"
    )


# ── Backward compatibility ──────────────────────────────────────────────────

def test_encoder_runs_when_batch_lacks_device_adj_key():
    """A caller that built a batch with the old keys (no device_adj)
    must still get a valid forward — the R-GCN branch must skip
    silently in that case."""
    g = graph_from_template(load_template("inverter"))
    batch = graphs_to_tensors([g], max_devices=8, max_nets=8)
    batch.pop("device_adj")
    enc = TopologyEncoder(_cfg()).eval()
    with torch.no_grad():
        out = enc.forward(batch)
    assert out.global_embedding.shape == (1, 32)
    assert torch.isfinite(out.global_embedding).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
