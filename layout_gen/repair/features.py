"""
layout_gen.repair.features — tensor encoding of layouts and actions.

Converts a layout state (a list of polygons + violations) into the fixed-
shape tensors a diffusion-style denoiser consumes, and decodes a network
prediction back into a :class:`PerturbAction`.

PDK-agnostic invariants
-----------------------
* Per-polygon features are **layer-role one-hot + dimensionless geometry**
  (centre relative to cell bbox, dimensions normalised by deficit at
  inference time).  No raw µm constants encoded as features.
* Violation features are **rule-category one-hot + position + deficit**.
  No PDK rule values; the deficit comes from the runtime DRC report.
* Action kind is a small finite set (6).  Magnitudes are predicted in
  *units of deficit* — at inference the reported deficit scales the model's
  output to the correct µm distance.

This module is the boundary between the symbolic / geometric world of
layout_gen and the tensor world of PyTorch.  Everything below it stays
PDK-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


# ── Layer role vocabulary ────────────────────────────────────────────────────
#
# Stable across PDKs.  Each PDK YAML maps GDS (layer, datatype) → one of
# these role names; new PDKs add to the registry, never to this list.

LAYER_ROLES: tuple[str, ...] = (
    "nwell",      # n-well
    "pwell",      # p-well / substrate
    "diff",       # active diffusion (n+ or p+ — distinguished by implant overlap)
    "tap",        # well/substrate tap diffusion
    "poly",       # gate poly
    "nsdm",       # n+ implant select
    "psdm",       # p+ implant select
    "licon1",     # diffusion / poly contact
    "li1",        # local interconnect
    "mcon",       # li1 → met1 via
    "met1",       # metal 1
    "via1",       # met1 → met2 via
    "met2",       # metal 2
    "via2",
    "met3",
    "via3",
    "met4",
    "via4",
    "met5",
    "npc",        # nitride poly cut
    "rpo",        # poly resistor protect
    "other",      # catch-all (unknown layer)
)

LAYER_ROLE_INDEX: dict[str, int] = {r: i for i, r in enumerate(LAYER_ROLES)}
N_LAYER_ROLES = len(LAYER_ROLES)


def role_index(layer_name: str) -> int:
    """Look up a layer's role index, defaulting to ``"other"``."""
    return LAYER_ROLE_INDEX.get(layer_name, LAYER_ROLE_INDEX["other"])


# ── Action kind vocabulary ──────────────────────────────────────────────────
#
# Mirrors :class:`~layout_gen.repair.perturb.PerturbAction`.  Order matters
# — these indices are part of the trained model's output schema.

ACTION_KINDS: tuple[str, ...] = (
    "shift_edge",
    "shrink_rect",
    "grow_rect",
    "translate",
    "delete_rect",
    "nudge_offgrid",
)
ACTION_KIND_INDEX: dict[str, int] = {k: i for i, k in enumerate(ACTION_KINDS)}
N_ACTION_KINDS = len(ACTION_KINDS)

# Edge for shift_edge: encoded as a one-hot {left,right,bottom,top}
EDGE_NAMES: tuple[str, ...] = ("left", "right", "bottom", "top")
EDGE_INDEX: dict[str, int] = {e: i for i, e in enumerate(EDGE_NAMES)}
N_EDGES = len(EDGE_NAMES)


# ── Polygon feature width ────────────────────────────────────────────────────

# Per-polygon features (in order):
#   - layer role one-hot                       (N_LAYER_ROLES)
#   - 4 geometry: cx, cy, w, h                 (relative to cell bbox)
#   - 4 anomaly hints:
#       d_same_layer    distance to nearest same-layer neighbour
#       d_other_layer   distance to nearest other-layer neighbour
#       n_same_layer_nb count of same-layer neighbours within a small radius
#       sd_norm         scalar 0..1 — how "small-or-large" this rect is vs
#                       its same-layer cohort (z-score of its longer edge)
#
# These hint at "this polygon is anomalous" — the perturbed polygon will
# typically have an unusual nearest-neighbour distance, an unusual size
# vs. its peers, or both.  All four are PDK-agnostic (computed from
# geometry alone), and all are dimensionless when normalised by the
# cell's diagonal — they carry no PDK rule constants.
POLY_FEAT_DIM = N_LAYER_ROLES + 4 + 4


# Rule categories used as the conditioning vocabulary.  Mirrors the
# small taxonomy in :mod:`layout_gen.repair.catalog`.  Index ``0`` is
# the "no information" fallback so the model can still operate when
# the rule isn't classified.
RULE_CATEGORIES: tuple[str, ...] = (
    "unknown", "width", "spacing_same", "spacing_cross",
    "enclosure", "extension", "area", "merge",
    "density", "overlap", "antenna",
)
RULE_CAT_INDEX: dict[str, int] = {c: i for i, c in enumerate(RULE_CATEGORIES)}
N_RULE_CATEGORIES = len(RULE_CATEGORIES)


@dataclass
class FeaturizedSample:
    """A single training example as torch tensors.

    Attributes
    ----------
    poly_feats : (N, POLY_FEAT_DIM) float
        Per-polygon feature matrix.
    poly_mask : (N,) bool
        True for valid polygons (used for batching with padding).
    rid_to_idx : list[int]
        Maps row index in ``poly_feats`` back to the original rectangle ID.
    k : int
        Perturbation depth (noise level).
    violation_xy : (2,) float
        **Conditioning input**: location of the DRC violation we are
        being asked to fix, in cell-bbox-normalised coordinates.  At
        training time we use the perturbed-polygon centroid as a proxy;
        at inference the DRC tool reports an actual (x, y).
    rule_cat : int
        **Conditioning input**: index into :data:`RULE_CATEGORIES`.  Tells
        the model the *kind* of fix needed (spacing vs width vs merge…).
    action_kind : int
        Output label — index into :data:`ACTION_KINDS`.
    target_idx : int
        Auxiliary label — row index of the target polygon (-1 if N/A).
    target_xy : (2,) float
        Auxiliary regression label — centroid of the target polygon.
        With :attr:`violation_xy` now also given as INPUT, this output
        is approximately redundant; we keep it as an auxiliary head to
        keep the encoder spatially aware.
    edge_idx : int
        Output label — index into :data:`EDGE_NAMES`.
    magnitude : (3,) float
        Output label — ``(delta, dx, dy)`` µm.
    """
    poly_feats:   torch.Tensor
    poly_mask:    torch.Tensor
    rid_to_idx:   list[int]
    k:            int
    violation_xy: torch.Tensor
    rule_cat:     int
    action_kind:  int
    target_idx:   int
    target_xy:    torch.Tensor
    edge_idx:     int
    magnitude:    torch.Tensor


# ── Polygon featurizer ───────────────────────────────────────────────────────

def encode_polygons(
    rects: list[dict],
    *,
    cell_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Encode a list of rectangle dicts into a polygon feature tensor.

    Parameters
    ----------
    rects :
        Each dict has keys ``rid``, ``layer``, ``x0``, ``y0``, ``x1``, ``y1``.
        Format is what :func:`mine_trajectories._serialise_state` produces.
    cell_bbox :
        Optional ``(x0, y0, x1, y1)`` of the cell.  Defaults to the bbox
        of *rects* themselves.  Geometry features are normalised against
        this.

    Returns
    -------
    feats : (N, POLY_FEAT_DIM) float tensor
    rid_to_idx : list[int]
        Row index → original rectangle ID.
    """
    if not rects:
        return torch.zeros((0, POLY_FEAT_DIM), dtype=torch.float32), []

    # Cell bbox → normalisation reference
    if cell_bbox is None:
        x0c = min(r["x0"] for r in rects)
        y0c = min(r["y0"] for r in rects)
        x1c = max(r["x1"] for r in rects)
        y1c = max(r["y1"] for r in rects)
    else:
        x0c, y0c, x1c, y1c = cell_bbox
    w = max(x1c - x0c, 1e-6)
    h = max(y1c - y0c, 1e-6)

    feats = torch.zeros((len(rects), POLY_FEAT_DIM), dtype=torch.float32)
    rid_to_idx: list[int] = []

    # Pre-compute centroids in absolute units once, then normalise.
    centroids: list[tuple[float, float]] = []
    layers:    list[str] = []
    long_edges: list[float] = []
    for r in rects:
        centroids.append(((r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2))
        layers.append(r["layer"])
        long_edges.append(max(r["x1"] - r["x0"], r["y1"] - r["y0"]))

    diag = (w * w + h * h) ** 0.5
    diag = max(diag, 1e-6)

    # Same-layer cohort statistics (mean, std of long edges per layer)
    by_layer: dict[str, list[float]] = {}
    for lyr, le in zip(layers, long_edges):
        by_layer.setdefault(lyr, []).append(le)
    layer_mean: dict[str, float] = {}
    layer_std:  dict[str, float] = {}
    for lyr, vals in by_layer.items():
        m = sum(vals) / len(vals)
        sq = sum((v - m) ** 2 for v in vals) / len(vals)
        layer_mean[lyr] = m
        layer_std[lyr]  = max(sq ** 0.5, 1e-6)

    # Density radius: 5% of cell diagonal — small but not pixel-tight.
    nb_radius = 0.05 * diag

    for i, r in enumerate(rects):
        feats[i, role_index(r["layer"])] = 1.0
        cxi, cyi = centroids[i]
        feats[i, N_LAYER_ROLES + 0] = (cxi - x0c) / w
        feats[i, N_LAYER_ROLES + 1] = (cyi - y0c) / h
        feats[i, N_LAYER_ROLES + 2] = (r["x1"] - r["x0"]) / w
        feats[i, N_LAYER_ROLES + 3] = (r["y1"] - r["y0"]) / h
        rid_to_idx.append(int(r["rid"]))

        # Anomaly hints
        d_same  = float("inf")
        d_other = float("inf")
        n_same_nb = 0
        for j, (cxj, cyj) in enumerate(centroids):
            if i == j:
                continue
            dx = cxi - cxj
            dy = cyi - cyj
            d  = (dx * dx + dy * dy) ** 0.5
            if layers[j] == layers[i]:
                if d < d_same:
                    d_same = d
                if d < nb_radius:
                    n_same_nb += 1
            else:
                if d < d_other:
                    d_other = d

        # Convert to dimensionless (cell-diag relative).  Cap at 1.0 so
        # isolated polygons map to the full-cell distance.
        d_same_n  = min(d_same  / diag, 1.0) if d_same  != float("inf") else 1.0
        d_other_n = min(d_other / diag, 1.0) if d_other != float("inf") else 1.0
        # Long-edge z-score within same-layer cohort
        z = (long_edges[i] - layer_mean[layers[i]]) / layer_std[layers[i]]
        # Squash the z-score into [-1, 1] so the network sees a bounded value
        z_norm = max(-1.0, min(1.0, z / 3.0))

        feats[i, N_LAYER_ROLES + 4] = d_same_n
        feats[i, N_LAYER_ROLES + 5] = d_other_n
        feats[i, N_LAYER_ROLES + 6] = min(float(n_same_nb), 8.0) / 8.0
        feats[i, N_LAYER_ROLES + 7] = z_norm
    return feats, rid_to_idx


# ── Action encoder / decoder ─────────────────────────────────────────────────

def encode_action(
    action_dict: dict,
    rid_to_idx:  dict[int, int],
) -> tuple[int, int, int, torch.Tensor]:
    """Encode a serialised :class:`~layout_gen.repair.perturb.PerturbAction`
    into ``(kind_idx, target_idx, edge_idx, magnitude)``.

    *rid_to_idx* maps the rectangle ID stored in the action to its row in
    the polygon feature tensor.  Returns ``target_idx == -1`` when the
    action's target rid is not present (``add_rect`` re-issues rids).
    """
    kind     = action_dict["kind"]
    kind_idx = ACTION_KIND_INDEX.get(kind, -1)
    target_id = action_dict.get("target", -1)
    target_idx = rid_to_idx.get(target_id, -1)

    edge_idx = -1
    mag = torch.zeros(3, dtype=torch.float32)
    params = action_dict.get("params") or {}

    if kind == "shift_edge":
        edge_idx = EDGE_INDEX.get(params.get("side", "left"), 0)
        mag[0] = float(params.get("delta", 0.0))
    elif kind in ("shrink_rect", "grow_rect"):
        mag[0] = float(params.get("delta", 0.0))
    elif kind in ("translate", "nudge_offgrid"):
        mag[1] = float(params.get("dx", 0.0))
        mag[2] = float(params.get("dy", 0.0))
    elif kind == "delete_rect":
        pass   # no magnitude
    elif kind == "add_rect":
        # Snapshot defines the new rect; for now we do not learn add_rect
        # via this head.  The diffusion network is trained only on edits to
        # existing geometry.
        pass

    return kind_idx, target_idx, edge_idx, mag


def target_centroid(rects: list[dict], target_idx: int) -> torch.Tensor:
    """Return the (cx, cy) of the rectangle at ``target_idx``, normalised
    to [0, 1] using the cell bbox of *rects*.  Returns ``[-1, -1]`` when
    the index is invalid.
    """
    if target_idx < 0 or target_idx >= len(rects):
        return torch.tensor([-1.0, -1.0], dtype=torch.float32)
    x0c = min(r["x0"] for r in rects)
    y0c = min(r["y0"] for r in rects)
    x1c = max(r["x1"] for r in rects)
    y1c = max(r["y1"] for r in rects)
    w = max(x1c - x0c, 1e-6)
    h = max(y1c - y0c, 1e-6)
    r  = rects[target_idx]
    cx = ((r["x0"] + r["x1"]) / 2 - x0c) / w
    cy = ((r["y0"] + r["y1"]) / 2 - y0c) / h
    return torch.tensor([cx, cy], dtype=torch.float32)


def featurize_record(record: dict) -> FeaturizedSample:
    """Convert a mined trajectory JSON record into a :class:`FeaturizedSample`.

    Uses the *first inverse action* as the supervised label — the model's
    job is to denoise one step.  Multi-step inverse sequences are produced
    only for richer training; the first action is sufficient to learn the
    score / reverse-step direction at noise level ``k``.
    """
    rects        = record["perturbed_state"]
    feats, rid_to_idx_list = encode_polygons(rects)
    rid_lookup = {rid: i for i, rid in enumerate(rid_to_idx_list)}

    inverse = record.get("inverse_action_sequence", [])
    if not inverse:
        kind_idx, target_idx, edge_idx, mag = -1, -1, -1, torch.zeros(3)
    else:
        kind_idx, target_idx, edge_idx, mag = encode_action(inverse[0], rid_lookup)

    # Conditioning: at training, the violation centroid is approximated
    # by the perturbed-target's centroid (we know exactly which polygon
    # was perturbed); the rule category is derived from the first rule
    # in violation_rules (catalogued classification).
    tgt_xy = target_centroid(rects, target_idx)
    rule_cat = _infer_rule_cat_for_record(record)

    return FeaturizedSample(
        poly_feats=feats,
        poly_mask=torch.ones(len(rects), dtype=torch.bool),
        rid_to_idx=rid_to_idx_list,
        k=int(record.get("k", 1)),
        violation_xy=tgt_xy,
        rule_cat=rule_cat,
        action_kind=kind_idx,
        target_idx=target_idx,
        target_xy=tgt_xy,
        edge_idx=edge_idx,
        magnitude=mag,
    )


def _infer_rule_cat_for_record(record: dict) -> int:
    """Infer the rule category index for a trajectory record using the
    first violation rule + the catalog classifier.  Falls back to
    ``unknown`` when no rule list is present.
    """
    rules = record.get("violation_rules") or []
    if not rules:
        return RULE_CAT_INDEX["unknown"]
    rule = str(rules[0]).strip().strip("'\"")
    # Use catalog's classifier with empty description / layers — same
    # heuristic, just on rule name.  Catalog classification is good
    # enough for a coarse one-hot conditioning.
    from layout_gen.repair.catalog import classify_rule
    cat = classify_rule(rule, "", [])
    return RULE_CAT_INDEX.get(cat, RULE_CAT_INDEX["unknown"])


# ── Batch collation ──────────────────────────────────────────────────────────

@dataclass
class FeaturizedBatch:
    """Padded batch ready for the model."""
    poly_feats:   torch.Tensor   # (B, N_max, POLY_FEAT_DIM)
    poly_mask:    torch.Tensor   # (B, N_max) bool
    k:            torch.Tensor   # (B,) long
    violation_xy: torch.Tensor   # (B, 2) float — conditioning
    rule_cat:     torch.Tensor   # (B,) long  — conditioning
    action_kind:  torch.Tensor   # (B,) long
    target_idx:   torch.Tensor   # (B,) long
    target_xy:    torch.Tensor   # (B, 2) float
    edge_idx:     torch.Tensor   # (B,) long
    magnitude:    torch.Tensor   # (B, 3) float


def collate(samples: list[FeaturizedSample]) -> FeaturizedBatch:
    B = len(samples)
    N_max = max((s.poly_feats.shape[0] for s in samples), default=0)
    feats = torch.zeros((B, N_max, POLY_FEAT_DIM), dtype=torch.float32)
    mask  = torch.zeros((B, N_max), dtype=torch.bool)
    for i, s in enumerate(samples):
        n = s.poly_feats.shape[0]
        feats[i, :n] = s.poly_feats
        mask[i, :n]  = True
    return FeaturizedBatch(
        poly_feats=feats,
        poly_mask=mask,
        k=torch.tensor([s.k for s in samples], dtype=torch.long),
        violation_xy=torch.stack([s.violation_xy for s in samples]),
        rule_cat=torch.tensor([s.rule_cat for s in samples], dtype=torch.long),
        action_kind=torch.tensor([s.action_kind for s in samples], dtype=torch.long),
        target_idx=torch.tensor([s.target_idx for s in samples], dtype=torch.long),
        target_xy=torch.stack([s.target_xy for s in samples]),
        edge_idx=torch.tensor([s.edge_idx for s in samples], dtype=torch.long),
        magnitude=torch.stack([s.magnitude for s in samples]),
    )


__all__ = [
    "LAYER_ROLES", "LAYER_ROLE_INDEX", "N_LAYER_ROLES",
    "ACTION_KINDS", "ACTION_KIND_INDEX", "N_ACTION_KINDS",
    "EDGE_NAMES", "EDGE_INDEX", "N_EDGES",
    "RULE_CATEGORIES", "RULE_CAT_INDEX", "N_RULE_CATEGORIES",
    "POLY_FEAT_DIM",
    "FeaturizedSample", "FeaturizedBatch",
    "role_index", "encode_polygons", "encode_action",
    "target_centroid",
    "featurize_record", "collate",
]
