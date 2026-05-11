"""
layout_gen.rl.env.observation — fixed-shape observation builder.

Converts a (LayoutState, list[DRCViolation]) pair into the padded
tensors a neural policy consumes. The shape is constant across episodes
so PPO / SAC / any SB3 algo can wrap the env without custom plumbing.

Components
----------
``poly_feats``  (POLY_CAP, POLY_FEAT_DIM)   per-polygon features (encode_polygons)
``poly_mask``   (POLY_CAP,)                 1.0 for live polygons, 0.0 for padding
``viol_feats``  (VIOL_CAP, V_FEAT_DIM)      per-violation features
``viol_mask``   (VIOL_CAP,)                 1.0 for active violations, 0.0 padding
``global_feats``(N_GLOBAL,)                 episode-level scalars

All PDK-specific knowledge enters through:
  * The layer-role table in :mod:`layout_gen.repair.features` (22 stable
    abstract roles; new PDKs map their layer names onto this table).
  * The rule classifier in :mod:`layout_gen.repair.catalog` (heuristic
    string match against rule name / description; falls back to
    "unknown").

Both are PDK-agnostic in their interface — the policy never sees raw
DRC rule constants or vendor layer names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from gymnasium import spaces

from layout_gen.drc.base import DRCViolation
from layout_gen.repair.catalog import classify_rule, extract_layers
from layout_gen.repair.features import (
    LAYER_ROLES, N_LAYER_ROLES, POLY_FEAT_DIM,
    RULE_CATEGORIES, N_RULE_CATEGORIES,
    encode_polygons, role_index,
)
from layout_gen.synth.geo.state import LayoutState


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_POLY_CAP = 256
DEFAULT_VIOL_CAP = 64

# Per-violation features:
#   2  : xy normalised to cell bbox [0, 1]
#   N_RULE_CATEGORIES : one-hot category from classify_rule
#   1  : normalised "value" (measured deficit, clipped to [0, 1])
#   N_LAYER_ROLES : one-hot of the violation's primary layer role (or zero)
V_FEAT_DIM = 2 + N_RULE_CATEGORIES + 1 + N_LAYER_ROLES

# Global features:
#   n_violations_norm : len(violations) / DEFAULT_VIOL_CAP, clipped to [0, 1]
#   n_polygons_norm   : len(state)      / DEFAULT_POLY_CAP, clipped to [0, 1]
#   mean_value_norm   : mean of normalised violation values
#   step_progress     : current_step / max_steps, supplied by the env
N_GLOBAL = 4

# Cap for normalising the violation's "value" (measured µm). Values
# above this are saturated; this is just for feature scaling and does
# not enforce any DRC rule constant.
VALUE_NORM_CAP_UM = 0.5


@dataclass
class Observation:
    """Convenient container; the env returns a dict that gymnasium accepts."""
    poly_feats:   np.ndarray   # (POLY_CAP, POLY_FEAT_DIM)
    poly_mask:    np.ndarray   # (POLY_CAP,)
    viol_feats:   np.ndarray   # (VIOL_CAP, V_FEAT_DIM)
    viol_mask:    np.ndarray   # (VIOL_CAP,)
    global_feats: np.ndarray   # (N_GLOBAL,)
    rid_to_idx:   dict[int, int]   # not part of gym obs; used by action decode

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "poly_feats":   self.poly_feats,
            "poly_mask":    self.poly_mask,
            "viol_feats":   self.viol_feats,
            "viol_mask":    self.viol_mask,
            "global_feats": self.global_feats,
        }


def make_observation_space(*,
                            poly_cap: int = DEFAULT_POLY_CAP,
                            viol_cap: int = DEFAULT_VIOL_CAP) -> spaces.Dict:
    """Build the gymnasium Dict space matching :func:`build_observation`."""
    return spaces.Dict({
        "poly_feats":   spaces.Box(low=-np.inf, high=np.inf,
                                   shape=(poly_cap, POLY_FEAT_DIM),
                                   dtype=np.float32),
        "poly_mask":    spaces.Box(low=0.0, high=1.0,
                                   shape=(poly_cap,), dtype=np.float32),
        "viol_feats":   spaces.Box(low=-np.inf, high=np.inf,
                                   shape=(viol_cap, V_FEAT_DIM),
                                   dtype=np.float32),
        "viol_mask":    spaces.Box(low=0.0, high=1.0,
                                   shape=(viol_cap,), dtype=np.float32),
        "global_feats": spaces.Box(low=-np.inf, high=np.inf,
                                   shape=(N_GLOBAL,), dtype=np.float32),
    })


# ── Builder ──────────────────────────────────────────────────────────────────

def build_observation(
    state:       LayoutState,
    violations:  Sequence[DRCViolation],
    *,
    poly_cap:    int = DEFAULT_POLY_CAP,
    viol_cap:    int = DEFAULT_VIOL_CAP,
    cell_bbox:   tuple[float, float, float, float] | None = None,
    step_progress: float = 0.0,
) -> Observation:
    """Build a padded observation from the current env state."""
    rects = [
        {"rid": r.rid, "layer": r.layer,
         "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}
        for r in state
    ]

    # ── Polygons ─────────────────────────────────────────────────────────────
    feats_t, rid_list = encode_polygons(rects, cell_bbox=cell_bbox)
    n_live = min(feats_t.shape[0], poly_cap)
    poly_feats = np.zeros((poly_cap, POLY_FEAT_DIM), dtype=np.float32)
    poly_mask  = np.zeros((poly_cap,), dtype=np.float32)
    if n_live > 0:
        poly_feats[:n_live] = feats_t[:n_live].cpu().numpy()
        poly_mask[:n_live]  = 1.0
    rid_to_idx = {rid: i for i, rid in enumerate(rid_list[:n_live])}

    # Cell bbox in absolute µm — needed to normalise violation positions.
    if cell_bbox is None and rects:
        x0c = min(r["x0"] for r in rects)
        y0c = min(r["y0"] for r in rects)
        x1c = max(r["x1"] for r in rects)
        y1c = max(r["y1"] for r in rects)
    elif cell_bbox is not None:
        x0c, y0c, x1c, y1c = cell_bbox
    else:
        x0c = y0c = 0.0
        x1c = y1c = 1.0
    w = max(x1c - x0c, 1e-6)
    h = max(y1c - y0c, 1e-6)

    # ── Violations ───────────────────────────────────────────────────────────
    viol_feats = np.zeros((viol_cap, V_FEAT_DIM), dtype=np.float32)
    viol_mask  = np.zeros((viol_cap,), dtype=np.float32)
    n_viol_active = min(len(violations), viol_cap)
    sum_value_norm = 0.0
    for i in range(n_viol_active):
        v = violations[i]
        viol_feats[i] = _encode_violation(v, x0c, y0c, w, h)
        viol_mask[i]  = 1.0
        if v.value is not None:
            sum_value_norm += min(max(v.value, 0.0) / VALUE_NORM_CAP_UM, 1.0)

    # ── Global ───────────────────────────────────────────────────────────────
    global_feats = np.zeros((N_GLOBAL,), dtype=np.float32)
    global_feats[0] = min(len(violations) / max(viol_cap, 1), 1.0)
    global_feats[1] = min(len(state)      / max(poly_cap, 1), 1.0)
    global_feats[2] = (sum_value_norm / n_viol_active) if n_viol_active else 0.0
    global_feats[3] = float(np.clip(step_progress, 0.0, 1.0))

    return Observation(
        poly_feats=poly_feats,
        poly_mask=poly_mask,
        viol_feats=viol_feats,
        viol_mask=viol_mask,
        global_feats=global_feats,
        rid_to_idx=rid_to_idx,
    )


def _encode_violation(v: DRCViolation,
                      x0c: float, y0c: float,
                      w: float,   h: float) -> np.ndarray:
    """Featurise one DRCViolation. PDK-agnostic — derives category + layer
    role from the violation's rule string and description text."""
    out = np.zeros((V_FEAT_DIM,), dtype=np.float32)

    # xy normalised
    out[0] = float(np.clip((v.x - x0c) / w, 0.0, 1.0))
    out[1] = float(np.clip((v.y - y0c) / h, 0.0, 1.0))

    # rule category one-hot
    layers = [v.layer] if v.layer else extract_layers(v.rule, v.description)
    cat = classify_rule(v.rule, v.description, layers)
    cat_idx = RULE_CATEGORIES.index(cat) if cat in RULE_CATEGORIES else 0
    out[2 + cat_idx] = 1.0

    # value normalised
    if v.value is not None:
        out[2 + N_RULE_CATEGORIES] = float(
            np.clip(max(v.value, 0.0) / VALUE_NORM_CAP_UM, 0.0, 1.0)
        )

    # primary layer role one-hot (zero if the violation didn't name a layer)
    base = 2 + N_RULE_CATEGORIES + 1
    if layers:
        out[base + role_index(layers[0])] = 1.0

    return out


__all__ = [
    "DEFAULT_POLY_CAP", "DEFAULT_VIOL_CAP",
    "V_FEAT_DIM", "N_GLOBAL",
    "Observation",
    "build_observation",
    "make_observation_space",
]
