"""
layout_gen.rl.topology.encoder — bipartite GNN over (devices, nets).

Two rounds of message passing on the bipartite graph
(``devices ↔ nets``) followed by a masked mean over devices for the
global cell embedding. Devices and nets each maintain their own
embedding tensor; one round = ``net ← devices`` then ``device ← nets``.

Outputs
-------
:class:`TopologyEncoderOutput` — held as plain attributes:
  ``device_embeddings``  (B, D_max, d_token)
  ``device_mask``        (B, D_max)  — pass-through for downstream masking
  ``global_embedding``   (B, d_token)

Why a tiny custom GNN instead of pulling in PyTorch Geometric?
  * The cells we target are small (≤ ~30 devices); dense bipartite
    aggregation in plain PyTorch is faster than PyG's sparse path here
    and adds zero deps.
  * Keeps Phase 4 self-contained for CPU-only training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from layout_gen.rl.topology.parser import (
    DEVICE_FEAT_DIM, NET_FEAT_DIM,
    TopologyGraph, encode_device, encode_net,
)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TopologyEncoderConfig:
    d_token:      int = 64
    n_layers:     int = 2
    max_devices:  int = 32
    max_nets:     int = 32
    dropout:      float = 0.0


class TopologyEncoderOutput(NamedTuple):
    device_embeddings: torch.Tensor   # (B, D_max, d)
    device_mask:       torch.Tensor   # (B, D_max)
    global_embedding:  torch.Tensor   # (B, d)


# ── Batching ─────────────────────────────────────────────────────────────────

def graphs_to_tensors(
    graphs:      Sequence[TopologyGraph],
    *,
    max_devices: int,
    max_nets:    int,
) -> dict[str, torch.Tensor]:
    """Pad a batch of :class:`TopologyGraph` into fixed-shape tensors.

    Returns a dict with:
      * ``device_feats`` (B, D_max, DEVICE_FEAT_DIM)
      * ``net_feats``    (B, N_max, NET_FEAT_DIM)
      * ``device_mask``  (B, D_max) float in {0, 1}
      * ``net_mask``     (B, N_max) float in {0, 1}
      * ``incidence``    (B, D_max, N_max) float in {0, 1}; 1 iff device d
                         is connected to net n in the original graph.
    """
    B = len(graphs)
    device_feats = torch.zeros(B, max_devices, DEVICE_FEAT_DIM)
    net_feats    = torch.zeros(B, max_nets,    NET_FEAT_DIM)
    device_mask  = torch.zeros(B, max_devices)
    net_mask     = torch.zeros(B, max_nets)
    incidence    = torch.zeros(B, max_devices, max_nets)

    for b, g in enumerate(graphs):
        n_d = min(g.n_devices, max_devices)
        n_n = min(g.n_nets,    max_nets)
        for i in range(n_d):
            device_feats[b, i] = torch.as_tensor(
                encode_device(g.devices[i]), dtype=torch.float32,
            )
            device_mask[b, i] = 1.0
        for j in range(n_n):
            net_feats[b, j] = torch.as_tensor(
                encode_net(g.nets[j]), dtype=torch.float32,
            )
            net_mask[b, j] = 1.0
            for (d_idx, _term) in g.nets[j].connections:
                if d_idx < n_d:
                    incidence[b, d_idx, j] = 1.0

    return {
        "device_feats": device_feats,
        "net_feats":    net_feats,
        "device_mask":  device_mask,
        "net_mask":     net_mask,
        "incidence":    incidence,
    }


# ── Encoder ──────────────────────────────────────────────────────────────────

class _GraphConv(nn.Module):
    """One round of bipartite message passing.

    Updates net embeddings by aggregating incident devices, then updates
    device embeddings by aggregating incident nets. Aggregation = mean
    over masked + incident neighbours, normalised by per-row degree.
    """

    def __init__(self, d_token: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dev_in_net = nn.Linear(d_token, d_token)
        self.net_to_net = nn.Linear(d_token, d_token)
        self.net_in_dev = nn.Linear(d_token, d_token)
        self.dev_to_dev = nn.Linear(d_token, d_token)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        device_emb: torch.Tensor,   # (B, D, d)
        net_emb:    torch.Tensor,   # (B, N, d)
        incidence:  torch.Tensor,   # (B, D, N)
        device_mask: torch.Tensor,  # (B, D)
        net_mask:    torch.Tensor,  # (B, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ── nets ← devices ─────────────────────────────────────────────────
        # Mask out device-side rows that are padded.
        masked_dev = device_emb * device_mask.unsqueeze(-1)            # (B, D, d)
        # For each net n: sum_d incidence[d, n] * dev[d]               (B, N, d)
        # incidence^T @ dev = (N, D) @ (D, d) = (N, d)
        incid_t = incidence.transpose(1, 2)                            # (B, N, D)
        agg_to_net = torch.bmm(incid_t, masked_dev)                    # (B, N, d)
        deg_n = incidence.sum(dim=1).clamp(min=1.0).unsqueeze(-1)      # (B, N, 1)
        agg_to_net = agg_to_net / deg_n
        new_net = F.gelu(self.net_to_net(net_emb) + self.dev_in_net(agg_to_net))
        new_net = new_net * net_mask.unsqueeze(-1)
        new_net = self.dropout(new_net)

        # ── devices ← nets ────────────────────────────────────────────────
        masked_net = new_net * net_mask.unsqueeze(-1)                  # (B, N, d)
        agg_to_dev = torch.bmm(incidence, masked_net)                  # (B, D, d)
        deg_d = incidence.sum(dim=2).clamp(min=1.0).unsqueeze(-1)      # (B, D, 1)
        agg_to_dev = agg_to_dev / deg_d
        new_dev = F.gelu(self.dev_to_dev(device_emb) + self.net_in_dev(agg_to_dev))
        new_dev = new_dev * device_mask.unsqueeze(-1)
        new_dev = self.dropout(new_dev)

        return new_dev, new_net


class TopologyEncoder(nn.Module):
    """Bipartite GNN producing per-device + global cell embeddings."""

    def __init__(self, config: TopologyEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or TopologyEncoderConfig()
        d = self.cfg.d_token

        self.dev_in = nn.Linear(DEVICE_FEAT_DIM, d)
        self.net_in = nn.Linear(NET_FEAT_DIM,    d)
        self.layers = nn.ModuleList([
            _GraphConv(d, dropout=self.cfg.dropout)
            for _ in range(self.cfg.n_layers)
        ])
        self.dev_out_norm = nn.LayerNorm(d)

    # ── Public API ───────────────────────────────────────────────────────────

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> TopologyEncoderOutput:
        """Run the GNN.

        Parameters
        ----------
        batch :
            Dict from :func:`graphs_to_tensors` with keys
            ``device_feats``, ``net_feats``, ``device_mask``, ``net_mask``,
            ``incidence``.
        """
        device_feats = batch["device_feats"]
        net_feats    = batch["net_feats"]
        device_mask  = batch["device_mask"]
        net_mask     = batch["net_mask"]
        incidence    = batch["incidence"]

        device_emb = self.dev_in(device_feats) * device_mask.unsqueeze(-1)
        net_emb    = self.net_in(net_feats)    * net_mask.unsqueeze(-1)

        for layer in self.layers:
            device_emb, net_emb = layer(
                device_emb, net_emb, incidence,
                device_mask, net_mask,
            )

        device_emb = self.dev_out_norm(device_emb) * device_mask.unsqueeze(-1)
        global_emb = _masked_mean(device_emb, device_mask)
        return TopologyEncoderOutput(
            device_embeddings=device_emb,
            device_mask=device_mask,
            global_embedding=global_emb,
        )

    @torch.no_grad()
    def encode_graphs(
        self, graphs: Sequence[TopologyGraph],
        *, device: torch.device | str = "cpu",
    ) -> TopologyEncoderOutput:
        """Convenience: tensorise + run the encoder, no gradient."""
        batch = graphs_to_tensors(
            graphs,
            max_devices=self.cfg.max_devices,
            max_nets=self.cfg.max_nets,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        return self.forward(batch)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


__all__ = [
    "TopologyEncoderConfig", "TopologyEncoderOutput",
    "TopologyEncoder", "graphs_to_tensors",
]
