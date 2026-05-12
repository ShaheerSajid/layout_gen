"""
layout_gen.rl.policy.network — neural policy module.

Architecture
------------
::

    poly_feats (B, P, F_p)         viol_feats (B, V, F_v)
            │                              │
        Linear→d                       Linear→d
            │                              │
        TransformerEncoder             TransformerEncoder
        (poly_mask)                    (viol_mask)
            │                              │
       masked-mean-pool             masked-mean-pool
            │                              │
            └────────────┬─────────────────┘
                         │
        global_feats (B, G) ──┐
                              ▼
                    concat → Trunk MLP → ctx (B, d_trunk)
                              │
        ┌──────────┬──────────┼──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼          ▼
       kind      target     edge      sign_x     sign_y       mag
       (B,6)   (B,P_cap)   (B,4)     (B,2)      (B,2)      (B,M)

The **target head** is a pointer-style head — it dot-products a query
projection of the trunk against per-polygon embeddings. This gives the
policy a natural way to "point at the bad polygon" rather than learning
a fixed mapping from trunk features to polygon indices.

All other heads are plain Linear projections; they predict
kind / edge / sign / magnitude given the pooled context.

The module is PDK-agnostic: the encoder consumes the observation tensors
already produced by :mod:`layout_gen.rl.env.observation`, which only
encode layer roles (not vendor layer names) and rule categories (not
vendor rule constants).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layout_gen.repair.features import POLY_FEAT_DIM
from layout_gen.rl.env.action_space import (
    DEFAULT_DEVICE_CAP, DEFAULT_MAG_BINS, DEFAULT_POSITION_BINS,
    DEFAULT_TARGET_CAP,
    N_EDGES, N_PLACE_KINDS, N_REPAIR_KINDS,
)
from layout_gen.rl.env.place_action import N_ORIENTATIONS
from layout_gen.rl.env.observation import (
    DEFAULT_POLY_CAP, DEFAULT_VIOL_CAP, N_GLOBAL, V_FEAT_DIM,
)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class LayoutPolicyConfig:
    poly_cap:    int = DEFAULT_POLY_CAP
    viol_cap:    int = DEFAULT_VIOL_CAP
    target_cap:  int = DEFAULT_TARGET_CAP
    mag_bins:    int = DEFAULT_MAG_BINS

    d_token:     int = 64           # per-token embed (poly + viol)
    d_trunk:     int = 128          # post-pool MLP width
    n_layers:    int = 2            # transformer layers per encoder
    n_heads:     int = 4
    dim_ff:      int = 128
    dropout:     float = 0.0

    # ── Topology conditioning (Phase 4 part 1) ─────────────────────────
    # When True, forward() expects ``obs["topology_global"]`` (B, topology_dim)
    # — a fixed-per-episode embedding of the cell's netlist graph.
    # Default False keeps Phase 1–3 BC checkpoints loadable as-is.
    use_topology:  bool = False
    topology_dim:  int  = 64

    # ── PLACE action heads (Phase 4 part 2) ────────────────────────────
    # When True, the policy emits four additional heads
    # (device / x_bin / y_bin / orient) and the kind head's output
    # grows by N_PLACE_KINDS. Default False keeps Phase 1–3 + Phase 4
    # part 1 checkpoints loadable as-is.
    enable_place:  bool = False
    device_cap:    int  = DEFAULT_DEVICE_CAP
    x_bins:        int  = DEFAULT_POSITION_BINS
    y_bins:        int  = DEFAULT_POSITION_BINS

    # Loss weights for BC. Keyed by action dim name; defaults are
    # equal-weight on the dims that always matter (kind, target) and
    # half-weight on conditional dims (edge / sign / mag).
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "kind":   1.0,
        "target": 1.0,
        "edge":   0.5,
        "sign_x": 0.5,
        "sign_y": 0.5,
        "mag":    0.5,
        "device": 1.0,
        "x_bin":  0.5,
        "y_bin":  0.5,
        "orient": 0.5,
    })


# ── Logits container ─────────────────────────────────────────────────────────

class ActionLogits(NamedTuple):
    """Per-dim logits.

    ``device``, ``x_bin``, ``y_bin``, ``orient`` are populated only when
    the policy was built with ``enable_place=True``. Otherwise they hold
    zero-element placeholders (so the NamedTuple shape stays stable
    across configs but contributes nothing to a flat-concat).
    """
    kind:   torch.Tensor   # (B, n_kinds_total)
    target: torch.Tensor   # (B, target_cap)
    edge:   torch.Tensor   # (B, N_EDGES)
    sign_x: torch.Tensor   # (B, 2)
    sign_y: torch.Tensor   # (B, 2)
    mag:    torch.Tensor   # (B, mag_bins)
    device: torch.Tensor   # (B, device_cap) or (B, 0) when PLACE disabled
    x_bin:  torch.Tensor   # (B, x_bins)     or (B, 0)
    y_bin:  torch.Tensor   # (B, y_bins)     or (B, 0)
    orient: torch.Tensor   # (B, N_ORIENTATIONS) or (B, 0)


# ── Module ───────────────────────────────────────────────────────────────────

class LayoutPolicy(nn.Module):
    """Per-token transformer encoder + per-dim action heads.

    Parameters
    ----------
    config :
        Hyper-parameters; defaults match the env defaults so a fresh
        env + fresh policy connect without manual config plumbing.
    """

    def __init__(self, config: LayoutPolicyConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or LayoutPolicyConfig()

        d = self.cfg.d_token

        self.poly_in = nn.Linear(POLY_FEAT_DIM, d)
        self.viol_in = nn.Linear(V_FEAT_DIM,    d)

        enc_layer_p = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.dim_ff,
            dropout=self.cfg.dropout,
            batch_first=True, norm_first=True,
        )
        enc_layer_v = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.dim_ff,
            dropout=self.cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.poly_enc = nn.TransformerEncoder(enc_layer_p, num_layers=self.cfg.n_layers)
        self.viol_enc = nn.TransformerEncoder(enc_layer_v, num_layers=self.cfg.n_layers)

        self.global_in = nn.Linear(N_GLOBAL, d // 4 if d >= 4 else 1)

        trunk_in = d + d + (d // 4 if d >= 4 else 1)
        if self.cfg.use_topology:
            # Project the cell-global topology embedding into the trunk.
            # Kept as a separate Linear so the topology weights only exist
            # when the flag is on — old BC checkpoints stay compatible.
            self.topology_in = nn.Linear(self.cfg.topology_dim, d)
            trunk_in += d
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, self.cfg.d_trunk),
            nn.GELU(),
            nn.Linear(self.cfg.d_trunk, self.cfg.d_trunk),
        )

        # Pointer query for target head — dotted against polygon embeddings.
        self.target_query = nn.Linear(self.cfg.d_trunk, d)

        # Plain heads
        kind_out_dim = N_REPAIR_KINDS + (N_PLACE_KINDS if self.cfg.enable_place else 0)
        self.kind_head   = nn.Linear(self.cfg.d_trunk, kind_out_dim)
        self.edge_head   = nn.Linear(self.cfg.d_trunk, N_EDGES)
        self.signx_head  = nn.Linear(self.cfg.d_trunk, 2)
        self.signy_head  = nn.Linear(self.cfg.d_trunk, 2)
        self.mag_head    = nn.Linear(self.cfg.d_trunk, self.cfg.mag_bins)

        # PLACE heads — only instantiated when enabled, so Phase 1–3
        # checkpoints stay shape-compatible.
        if self.cfg.enable_place:
            self.device_head = nn.Linear(self.cfg.d_trunk, self.cfg.device_cap)
            self.x_bin_head  = nn.Linear(self.cfg.d_trunk, self.cfg.x_bins)
            self.y_bin_head  = nn.Linear(self.cfg.d_trunk, self.cfg.y_bins)
            self.orient_head = nn.Linear(self.cfg.d_trunk, N_ORIENTATIONS)

    # ── Forward ──────────────────────────────────────────────────────────────

    def encode_state(
        self, obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the encoder portion only.

        Returns
        -------
        ctx : (B, d_trunk)         pooled trunk context (used by value head)
        poly_emb : (B, P, d_token) per-polygon transformer embeddings
        poly_pad : (B, P) bool     True at masked-out positions (for the
                                   pointer head's -inf masking)
        """
        poly_feats   = obs["poly_feats"]
        poly_mask    = obs["poly_mask"]
        viol_feats   = obs["viol_feats"]
        viol_mask    = obs["viol_mask"]
        global_feats = obs["global_feats"]

        poly_pad = (poly_mask <= 0.5)
        viol_pad = (viol_mask <= 0.5)
        poly_pad = _avoid_all_pad(poly_pad)
        viol_pad = _avoid_all_pad(viol_pad)

        poly_emb = self.poly_in(poly_feats)
        poly_emb = self.poly_enc(poly_emb, src_key_padding_mask=poly_pad)

        viol_emb = self.viol_in(viol_feats)
        viol_emb = self.viol_enc(viol_emb, src_key_padding_mask=viol_pad)

        poly_pool = _masked_mean(poly_emb, poly_mask)
        viol_pool = _masked_mean(viol_emb, viol_mask)
        glob      = self.global_in(global_feats)

        parts = [poly_pool, viol_pool, glob]
        if self.cfg.use_topology:
            topo = obs.get("topology_global")
            if topo is None:
                # Falling back to zeros lets a topology-aware policy run on
                # an obs that doesn't carry topology yet (e.g. early bring-up
                # before the env wires it in). The model still processes a
                # well-formed input; predictions just won't be topology-aware.
                topo = poly_pool.new_zeros(poly_pool.shape[0], self.cfg.topology_dim)
            parts.append(self.topology_in(topo))

        ctx = self.trunk(torch.cat(parts, dim=-1))
        return ctx, poly_emb, poly_pad

    def heads(
        self, ctx: torch.Tensor,
        poly_emb: torch.Tensor,
        poly_pad: torch.Tensor,
    ) -> ActionLogits:
        """Apply the per-dim action heads to a precomputed encoder state."""
        q = self.target_query(ctx).unsqueeze(1)            # (B, 1, d)
        target_logits = (poly_emb * q).sum(dim=-1)         # (B, P)
        target_logits = target_logits.masked_fill(poly_pad, float("-inf"))
        target_logits = _resize_logits(target_logits, self.cfg.target_cap)

        if self.cfg.enable_place:
            device_logits = self.device_head(ctx)
            x_bin_logits  = self.x_bin_head(ctx)
            y_bin_logits  = self.y_bin_head(ctx)
            orient_logits = self.orient_head(ctx)
        else:
            empty = ctx.new_zeros((ctx.shape[0], 0))
            device_logits = empty
            x_bin_logits  = empty
            y_bin_logits  = empty
            orient_logits = empty

        return ActionLogits(
            kind   = self.kind_head(ctx),
            target = target_logits,
            edge   = self.edge_head(ctx),
            sign_x = self.signx_head(ctx),
            sign_y = self.signy_head(ctx),
            mag    = self.mag_head(ctx),
            device = device_logits,
            x_bin  = x_bin_logits,
            y_bin  = y_bin_logits,
            orient = orient_logits,
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> ActionLogits:
        ctx, poly_emb, poly_pad = self.encode_state(obs)
        return self.heads(ctx, poly_emb, poly_pad)


# ── Loss ─────────────────────────────────────────────────────────────────────

def masked_cross_entropy(
    logits:        ActionLogits,
    targets:       dict[str, torch.Tensor],
    dim_validity:  dict[str, torch.Tensor],
    weights:       dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-dim cross-entropy with sample-level validity masking.

    Parameters
    ----------
    logits :
        Output of :class:`LayoutPolicy.forward`.
    targets :
        Per-dim long tensors of shape (B,). Keys: kind, target, edge,
        sign_x, sign_y, mag.
    dim_validity :
        Per-dim bool tensors of shape (B,). Some action dims don't
        apply to every sample (e.g. ``edge`` is meaningless when
        ``kind != shift_edge``); those samples contribute zero to
        that dim's loss.
    weights :
        Per-dim scalar weights. Defaults to 1.0 each.

    Returns
    -------
    loss : scalar tensor
        Sum of per-dim weighted losses.
    breakdown : dict[str, float]
        Per-dim mean loss for logging.
    """
    weights = weights or {}
    parts: dict[str, torch.Tensor] = {}

    for name, lg in logits._asdict().items():
        if name not in targets:
            continue
        tgt = targets[name].long()
        valid = dim_validity[name].bool()
        if valid.sum() == 0:
            parts[name] = torch.zeros((), device=lg.device, dtype=lg.dtype)
            continue
        # Replace target_cap-out-of-range labels (which appear when the
        # true target rid wasn't found in the observation) with 0 to
        # keep cross-entropy happy; valid mask zeroes them out anyway.
        tgt = tgt.clamp(min=0, max=lg.size(-1) - 1)
        per_sample = F.cross_entropy(lg, tgt, reduction="none")
        per_sample = per_sample * valid.float()
        parts[name] = per_sample.sum() / valid.float().sum().clamp(min=1.0)

    total = torch.zeros((), device=next(iter(parts.values())).device,
                        dtype=next(iter(parts.values())).dtype)
    breakdown: dict[str, float] = {}
    for name, val in parts.items():
        w = weights.get(name, 1.0)
        total = total + w * val
        breakdown[name] = float(val.detach().cpu().item())

    return total, breakdown


# ── Helpers ──────────────────────────────────────────────────────────────────

def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the second dim, weighted by *mask* (B, T).

    Rows whose mask sums to zero return all-zero vectors.
    """
    m = mask.unsqueeze(-1)                                  # (B, T, 1)
    s = (x * m).sum(dim=1)                                  # (B, D)
    n = m.sum(dim=1).clamp(min=1.0)                         # (B, 1)
    return s / n


def _avoid_all_pad(pad: torch.Tensor) -> torch.Tensor:
    """If every position in a row is pad, un-pad the first position.

    Prevents NaN in :class:`nn.MultiheadAttention` softmax.
    """
    if pad.dtype != torch.bool:
        pad = pad.bool()
    all_pad = pad.all(dim=-1)
    if not bool(all_pad.any()):
        return pad
    pad = pad.clone()
    pad[all_pad, 0] = False
    return pad


def _resize_logits(logits: torch.Tensor, cap: int) -> torch.Tensor:
    """Pad or truncate the last-dim of *logits* to *cap*. Padding uses -inf."""
    cur = logits.size(-1)
    if cur == cap:
        return logits
    if cur > cap:
        return logits[..., :cap]
    pad = logits.new_full((*logits.shape[:-1], cap - cur), float("-inf"))
    return torch.cat([logits, pad], dim=-1)


__all__ = [
    "LayoutPolicy", "LayoutPolicyConfig", "ActionLogits",
    "masked_cross_entropy",
]
