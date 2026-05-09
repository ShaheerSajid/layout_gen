"""
layout_gen.repair.model — diffusion-style DRC denoiser.

A small attention-based network that consumes a layout's polygon-feature
tensor and a noise-level (perturbation-depth) embedding, and emits the
predicted single-step inverse action.

Output heads
------------
* ``kind_logits``    — (B, N_ACTION_KINDS)
* ``target_logits``  — (B, N_max)             which polygon to act on
* ``edge_logits``    — (B, N_EDGES)           only meaningful for shift_edge
* ``magnitude``      — (B, 3)                 (delta, dx, dy)

This is the *minimal* architecture that matches the diffusion analogy:
no world model, no critic, no MCTS.  Inference iterates the network until
DRC reports clean.

Sized for CPU training on the corpus we have today (~hundreds of
trajectories).  Designed so that swapping in a larger transformer / GNN
later is a one-line change.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layout_gen.repair.features import (
    POLY_FEAT_DIM, N_ACTION_KINDS, N_EDGES, N_RULE_CATEGORIES,
)


# ── Sinusoidal position embedding for k (perturbation depth) ─────────────────

class StepEmbedding(nn.Module):
    """Sinusoidal embedding of an integer noise-level k, à la DDPM."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        device = k.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        args = k.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


# ── Model ────────────────────────────────────────────────────────────────────

class DRCDenoiser(nn.Module):
    """Polygon-set transformer with action-prediction heads.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the per-polygon embedding stream.
    n_layers : int
        Number of transformer-encoder layers.
    n_heads : int
        Multi-head attention heads per layer.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers:   int = 2,
        n_heads:    int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.poly_in = nn.Sequential(
            nn.Linear(POLY_FEAT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.k_emb = StepEmbedding(hidden_dim)
        self.k_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Violation conditioning: (violation_xy, rule_cat_onehot) → hidden_dim.
        # Tells the model "fix the violation at THIS position of THIS kind".
        # At inference this is populated from real DRC reports; at
        # training the proxy is the perturbed-target's centroid + the
        # perturbation's rule classification.
        self.violation_proj = nn.Sequential(
            nn.Linear(2 + N_RULE_CATEGORIES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Pooled representation (mean over valid polygons + global k)
        # feeds the kind / edge / magnitude / target_xy heads.  The
        # target head is a pointer over polygons — a per-token logit.
        self.kind_head      = nn.Linear(hidden_dim, N_ACTION_KINDS)
        self.edge_head      = nn.Linear(hidden_dim, N_EDGES)
        self.magnitude_head = nn.Linear(hidden_dim, 3)
        self.target_head    = nn.Linear(hidden_dim, 1)   # per-token logit
        # Centroid regression: predict the target polygon's (cx, cy) in
        # [0, 1]^2 (cell-bbox normalised).  This is a much smoother
        # learning signal than the per-polygon pointer; with limited
        # training data the model converges on a "where is the target"
        # estimate even when it can't pick the exact polygon.  At
        # inference we snap the predicted centroid to the nearest polygon.
        self.target_xy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),                     # output in [0, 1]
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        poly_feats:   torch.Tensor,    # (B, N, POLY_FEAT_DIM)
        poly_mask:    torch.Tensor,    # (B, N) bool
        k:            torch.Tensor,    # (B,) long
        violation_xy: torch.Tensor | None = None,  # (B, 2) in [0,1]
        rule_cat:     torch.Tensor | None = None,  # (B,) long
    ) -> dict[str, torch.Tensor]:
        B, N, _ = poly_feats.shape

        # 1) embed polygons + add k embedding to every token
        h   = self.poly_in(poly_feats)                   # (B, N, H)
        k_h = self.k_proj(self.k_emb(k))                 # (B, H)
        h   = h + k_h.unsqueeze(1)

        # 2) violation conditioning (optional for back-compat with
        #    pre-v8 checkpoints).  When provided, inject as a global
        #    token added to every position.
        if violation_xy is not None and rule_cat is not None:
            rule_oh = torch.zeros(
                B, N_RULE_CATEGORIES,
                dtype=poly_feats.dtype, device=poly_feats.device,
            )
            rule_oh.scatter_(1, rule_cat.clamp(min=0).unsqueeze(1), 1.0)
            cond_in = torch.cat([violation_xy, rule_oh], dim=-1)   # (B, 2+R)
            cond_h  = self.violation_proj(cond_in)                 # (B, H)
            h       = h + cond_h.unsqueeze(1)

        # 3) attention encoder.  Mask padding tokens.
        attn_pad_mask = ~poly_mask                       # True = padding
        h = self.encoder(h, src_key_padding_mask=attn_pad_mask)

        # 3) pooled representation (mean over valid tokens)
        m = poly_mask.unsqueeze(-1).float()              # (B, N, 1)
        h_pool = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        # 4) heads
        kind_logits   = self.kind_head(h_pool)           # (B, n_kinds)
        edge_logits   = self.edge_head(h_pool)           # (B, n_edges)
        magnitude     = self.magnitude_head(h_pool)      # (B, 3)
        target_logits = self.target_head(h).squeeze(-1)  # (B, N)
        target_xy     = self.target_xy_head(h_pool)      # (B, 2) in [0,1]
        # Mask invalid polygons in target logits
        target_logits = target_logits.masked_fill(~poly_mask, float("-inf"))

        return {
            "kind_logits":   kind_logits,
            "edge_logits":   edge_logits,
            "magnitude":     magnitude,
            "target_logits": target_logits,
            "target_xy":     target_xy,
        }


# ── Loss ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _kind_uses_edge(action_kind: torch.Tensor) -> torch.Tensor:
    """1 where the action kind has a meaningful 'edge' field."""
    # Index 0 = shift_edge — only kind that uses edge.
    return (action_kind == 0).float()


def denoiser_loss(
    pred:            dict[str, torch.Tensor],
    action_kind:     torch.Tensor,   # (B,) long
    target_idx:     torch.Tensor,   # (B,) long
    edge_idx:       torch.Tensor,   # (B,) long
    magnitude:      torch.Tensor,   # (B, 3) float
    target_xy:      torch.Tensor | None = None,   # (B, 2) in [0,1]
    *,
    kind_weights:   torch.Tensor | None = None,    # (N_KINDS,) class weights
    lambda_target:  float = 0.2,
    lambda_target_xy: float = 1.0,
    lambda_edge:    float = 0.2,
    lambda_mag:     float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Aggregate denoiser loss with per-head reporting.

    Default lambda weights tilt the loss toward the *kind* head — that's
    the most informative single decision and the only one whose chance
    baseline (1/6) is high enough that a small model can learn it.
    Target prediction is ~1/N polygons, dominates raw loss and gradient
    if not down-weighted; we therefore set lambda_target=0.2.

    Pass *kind_weights* (shape ``(N_ACTION_KINDS,)``) to apply
    class-balanced cross-entropy on the kind head — counters dataset
    imbalance where one kind is plurality (e.g. ``translate``).
    """
    # Kind loss: cross-entropy over all examples, ignoring -1 labels
    kind_mask = action_kind >= 0
    if kind_mask.any():
        kind_loss = F.cross_entropy(
            pred["kind_logits"][kind_mask], action_kind[kind_mask],
            weight=kind_weights,
        )
    else:
        kind_loss = torch.zeros((), device=pred["kind_logits"].device)

    # Target pointer: same idea, ignore -1 labels
    target_mask = target_idx >= 0
    if target_mask.any():
        target_loss = F.cross_entropy(
            pred["target_logits"][target_mask], target_idx[target_mask],
        )
    else:
        target_loss = torch.zeros((), device=pred["kind_logits"].device)

    # Edge loss applies only to shift_edge actions
    edge_mask = (action_kind == 0) & (edge_idx >= 0)
    if edge_mask.any():
        edge_loss = F.cross_entropy(
            pred["edge_logits"][edge_mask], edge_idx[edge_mask],
        )
    else:
        edge_loss = torch.zeros((), device=pred["kind_logits"].device)

    # Magnitude regression — weight components by which kind uses them.
    # delta is used by shift_edge / shrink / grow; (dx,dy) by translate /
    # nudge_offgrid.  Mask the irrelevant components per example.
    delta_mask = ((action_kind == 0) | (action_kind == 1) | (action_kind == 2)).float().unsqueeze(-1)
    xy_mask    = ((action_kind == 3) | (action_kind == 5)).float().unsqueeze(-1)
    mag_mask   = torch.cat([delta_mask, xy_mask, xy_mask], dim=-1)   # (B, 3)
    mag_loss = ((pred["magnitude"] - magnitude) ** 2 * mag_mask).sum() / (
        mag_mask.sum().clamp(min=1.0)
    )

    # Centroid regression loss: model predicts (x, y) in [0, 1]^2;
    # target_xy = [-1, -1] when invalid.  We mask those out.
    target_xy_loss = torch.zeros((), device=pred["kind_logits"].device)
    if target_xy is not None and "target_xy" in pred:
        valid_xy = (target_xy[:, 0] >= 0.0) & (target_xy[:, 1] >= 0.0)
        if valid_xy.any():
            target_xy_loss = F.mse_loss(
                pred["target_xy"][valid_xy], target_xy[valid_xy],
            )

    total = (kind_loss
             + lambda_target    * target_loss
             + lambda_target_xy * target_xy_loss
             + lambda_edge      * edge_loss
             + lambda_mag       * mag_loss)
    return total, {
        "kind_loss":      kind_loss.detach(),
        "target_loss":    target_loss.detach(),
        "target_xy_loss": target_xy_loss.detach(),
        "edge_loss":      edge_loss.detach(),
        "mag_loss":       mag_loss.detach(),
    }


__all__ = ["DRCDenoiser", "StepEmbedding", "denoiser_loss"]
