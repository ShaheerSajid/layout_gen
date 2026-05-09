"""
layout_gen.repair.augment — D4 symmetry data augmentation.

CMOS layouts are symmetric under the 8 elements of the dihedral group
D4: identity, three rotations (90°/180°/270°), horizontal/vertical flip,
and the two diagonal flips.  Applying any of these to a (state, action)
training pair produces a *new* valid (state, action) pair the model
should also fit.  This is "free" data multiplication — the symmetries
preserve every DRC rule (rules are translation- and rotation-invariant
in CMOS) and preserve label correctness because the action transforms
the same way as the state.

Apply via :func:`augment_sample`.  The symmetry index ``s ∈ [0, 8)`` is
sampled uniformly.

Mapping table (rotation/flip on (x, y)):

    0 identity      (x, y)          shifted_left↔shifted_left
    1 R90           (-y, x)         left → top
    2 R180          (-x, -y)        left → right
    3 R270          (y, -x)         left → bottom
    4 flip_x        (-x, y)         left ↔ right
    5 flip_y        (x, -y)         bottom ↔ top
    6 flip_diag     (y, x)          left → bottom
    7 flip_antidiag (-y, -x)        left → top  (and reversed)

The geometry transforms are isometries — rectangles stay rectangles,
just with edges remapped.  The action transforms accordingly.
"""
from __future__ import annotations

import torch

from layout_gen.repair.features import (
    POLY_FEAT_DIM, N_LAYER_ROLES, EDGE_INDEX, EDGE_NAMES,
    FeaturizedSample,
)


# ── Edge remapping under each symmetry ───────────────────────────────────────

# Mapping[s][original_edge_idx] = new_edge_idx
# Edge order: 0=left, 1=right, 2=bottom, 3=top
_EDGE_MAP: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),   # 0 identity
    (3, 2, 0, 1),   # 1 R90 ccw  : left→top, right→bottom, bottom→left, top→right
    (1, 0, 3, 2),   # 2 R180     : left↔right, bottom↔top
    (2, 3, 1, 0),   # 3 R270 ccw : left→bottom, right→top, bottom→right, top→left
    (1, 0, 2, 3),   # 4 flip_x   : left↔right, bottom/top unchanged
    (0, 1, 3, 2),   # 5 flip_y   : bottom↔top, left/right unchanged
    (2, 3, 0, 1),   # 6 flip_diag (y=x)  : left↔bottom, right↔top
    (3, 2, 1, 0),   # 7 flip_antidiag (y=-x)  : left↔top, right↔bottom
)


def _xy_transform(s: int, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply symmetry *s* to (x, y) coordinates centred at origin."""
    if s == 0: return  x,  y
    if s == 1: return -y,  x
    if s == 2: return -x, -y
    if s == 3: return  y, -x
    if s == 4: return -x,  y
    if s == 5: return  x, -y
    if s == 6: return  y,  x
    if s == 7: return -y, -x
    raise ValueError(f"Bad symmetry index {s}")


def augment_sample(sample: FeaturizedSample, s: int) -> FeaturizedSample:
    """Return a new :class:`FeaturizedSample` with symmetry *s* applied."""
    if s == 0:
        return sample

    feats = sample.poly_feats.clone()

    # Geometry features (cx, cy, w, h) are at indices N..N+3.  We
    # transform the centre (cx, cy) around (0.5, 0.5) — the cell-bbox
    # centre after the unit-square normalisation done in encode_polygons.
    # Width and height are signed-magnitude under flips/rotations, but
    # since all rectangles are axis-aligned and the flips swap (w, h)
    # under rotations, we update them too.
    cx = feats[:, N_LAYER_ROLES + 0] - 0.5
    cy = feats[:, N_LAYER_ROLES + 1] - 0.5
    w  = feats[:, N_LAYER_ROLES + 2]
    h  = feats[:, N_LAYER_ROLES + 3]

    nx, ny = _xy_transform(s, cx, cy)
    feats[:, N_LAYER_ROLES + 0] = nx + 0.5
    feats[:, N_LAYER_ROLES + 1] = ny + 0.5
    # Rotations / diagonal flips swap width and height
    if s in (1, 3, 6, 7):
        feats[:, N_LAYER_ROLES + 2] = h
        feats[:, N_LAYER_ROLES + 3] = w
    # else: w/h unchanged

    # ── Action transform ────────────────────────────────────────────────────
    new_edge_idx = sample.edge_idx
    if sample.edge_idx >= 0:
        new_edge_idx = _EDGE_MAP[s][sample.edge_idx]

    mag = sample.magnitude.clone()
    if mag.numel() >= 3:
        # mag[0] = delta (scalar, sign-invariant under reflection); leave it.
        # mag[1], mag[2] = (dx, dy); transform them like (cx, cy).
        dx, dy = mag[1].clone(), mag[2].clone()
        ndx, ndy = _xy_transform(s, dx, dy)
        mag[1] = ndx
        mag[2] = ndy

    def _xy_around_centre(xy: torch.Tensor) -> torch.Tensor:
        out = xy.clone()
        if out.numel() == 2 and out[0].item() >= 0.0:
            tx = out[0] - 0.5
            ty = out[1] - 0.5
            ntx, nty = _xy_transform(s, tx, ty)
            out[0] = ntx + 0.5
            out[1] = nty + 0.5
        return out

    new_target_xy    = _xy_around_centre(sample.target_xy)
    new_violation_xy = _xy_around_centre(sample.violation_xy)

    return FeaturizedSample(
        poly_feats=feats,
        poly_mask=sample.poly_mask,
        rid_to_idx=sample.rid_to_idx,
        k=sample.k,
        violation_xy=new_violation_xy,
        rule_cat=sample.rule_cat,             # symmetry-invariant
        action_kind=sample.action_kind,
        target_idx=sample.target_idx,         # row index unchanged — we don't reorder polys
        target_xy=new_target_xy,
        edge_idx=new_edge_idx,
        magnitude=mag,
    )


__all__ = ["augment_sample"]
