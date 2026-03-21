"""
layout_gen.synth.ml.features — analytical DRC margin computation.

All functions are pure Python + numpy — no KLayout, no gdsfactory needed.
Geometry is computed by calling ``transistor_geom()`` directly (single source
of truth for the layout engine), so the features exactly reflect what the
synthesizer will produce.

Constants
---------
FEATURE_NAMES : list[str]
    Column names for :func:`cell_features` output.
MARGIN_NAMES : list[str]
    Column names for :func:`drc_margins` output.

Functions
---------
cell_features(w_N, w_P, l, rules) -> np.ndarray  shape (22,)
drc_margins(w_N, w_P, l, rules)   -> np.ndarray  shape (10,)
margin_vector(...)                 -> alias for drc_margins

DRC margins are *signed distances* from each rule threshold.
Positive  = constraint satisfied (slack from the boundary).
Negative  = constraint violated (extent of violation).
"""
from __future__ import annotations

import numpy as np

from layout_gen.pdk        import PDKRules
from layout_gen.transistor import transistor_geom
from layout_gen.cells.standard import _inter_cell_gap

# ── Column name lists ─────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Explicit params (3)
    "w_N", "w_P", "l",
    # Computed geometry (12)
    "n_fingers_N", "n_fingers_P",
    "sd_length",
    "w_finger_N", "w_finger_P",
    "total_x_N", "total_x_P",
    "total_y_N", "total_y_P",
    "inter_cell_gap",
    # PDK rules baked in as features for multi-tech generalisation (9)
    "poly_width_min",
    "diff_width_min",
    "diff_spacing_min",
    "contact_size",
    "contact_enc_diff",
    "contact_space_to_poly",
    "li1_width_min",
    "nwell_enc_pdiff",
    "nwell_width_min",
]  # 22 total

MARGIN_NAMES: list[str] = [
    "poly.1",       # gate length vs poly_width_min
    "diff.1_N",     # NMOS per-finger width vs diff_width_min
    "diff.1_P",     # PMOS per-finger width vs diff_width_min
    "licon.5a_N",   # NMOS S/D fits contact (enclosure)
    "licon.5a_P",   # PMOS S/D fits contact
    "licon.7_N",    # NMOS contact-to-gate-edge clearance
    "licon.7_P",    # PMOS contact-to-gate-edge clearance
    "diff.2",       # inter-cell diff-to-diff spacing >= 0
    "nwell.1",      # nwell wide enough to enclose PMOS diff
    "li1.1",        # drain bridge li1 width >= li1_width_min
]  # 10 total


# ── Feature extraction ────────────────────────────────────────────────────────

def cell_features(
    w_N:   float,
    w_P:   float,
    l:     float,
    rules: PDKRules,
) -> np.ndarray:
    """Build a feature vector from sizing params and PDK rules.

    Parameters
    ----------
    w_N, w_P :
        NMOS and PMOS channel widths (µm).
    l :
        Gate length for both devices (µm).
    rules :
        PDK rules.

    Returns
    -------
    np.ndarray, shape (22,)
        See :data:`FEATURE_NAMES` for column order.
    """
    ng  = transistor_geom(w_N, l, "nmos", rules)
    pg  = transistor_geom(w_P, l, "pmos", rules)
    gap = _inter_cell_gap(rules)

    p   = rules.poly
    d   = rules.diff
    c   = rules.contacts
    li  = rules.li1
    nw  = rules.nwell

    return np.array([
        # params
        w_N, w_P, l,
        # geometry
        float(ng.n_fingers), float(pg.n_fingers),
        ng.sd_length_um,
        ng.w_finger_um, pg.w_finger_um,
        ng.total_x_um,  pg.total_x_um,
        ng.total_y_um,  pg.total_y_um,
        gap,
        # PDK rule constants
        p["width_min_um"],
        d["width_min_um"],
        d["spacing_min_um"],
        c["size_um"],
        c["enclosure_in_diff_um"],
        c["space_to_poly_um"],
        li["width_min_um"],
        nw["enclosure_of_pdiff_um"],
        nw["width_min_um"],
    ], dtype=np.float64)


# ── DRC margin computation ────────────────────────────────────────────────────

def drc_margins(
    w_N:   float,
    w_P:   float,
    l:     float,
    rules: PDKRules,
) -> np.ndarray:
    """Compute analytical DRC margins for a CMOS inverter cell.

    Each margin is the signed distance from the rule threshold:
    - Positive → constraint satisfied (distance to violation).
    - Negative → constraint violated (extent of violation).

    Parameters
    ----------
    w_N, w_P :
        NMOS and PMOS channel widths (µm).
    l :
        Gate length (µm).
    rules :
        PDK rules.

    Returns
    -------
    np.ndarray, shape (10,)
        See :data:`MARGIN_NAMES` for column order.
    """
    ng  = transistor_geom(w_N, l, "nmos", rules)
    pg  = transistor_geom(w_P, l, "pmos", rules)
    gap = _inter_cell_gap(rules)

    p   = rules.poly
    d   = rules.diff
    c   = rules.contacts
    li  = rules.li1
    nw  = rules.nwell

    poly_wmin  = p["width_min_um"]
    diff_wmin  = d["width_min_um"]
    c_size     = c["size_um"]
    c_enc      = c["enclosure_in_diff_um"]
    stp        = c["space_to_poly_um"]
    li1_wmin   = li["width_min_um"]
    nw_enc     = nw["enclosure_of_pdiff_um"]
    nw_wmin    = nw["width_min_um"]

    sd_N = ng.sd_length_um
    sd_P = pg.sd_length_um

    # poly.1 — gate length must be >= poly width minimum
    m_poly1 = l - poly_wmin

    # diff.1 — per-finger diffusion width must be >= diff width minimum
    m_diff1_N = ng.w_finger_um - diff_wmin
    m_diff1_P = pg.w_finger_um - diff_wmin

    # licon.5a — S/D region must accommodate contact + enclosure on each side
    #   sd_length >= c_size + 2 * c_enc
    m_licon5a_N = sd_N - (c_size + 2 * c_enc)
    m_licon5a_P = sd_P - (c_size + 2 * c_enc)

    # licon.7 — contact must clear the adjacent poly gate edge
    #   Contact is centred in the S/D rail; its right edge is at sd/2 + c_size/2.
    #   The poly gate left edge is at sd (local X).
    #   Required clearance from contact right edge to gate edge >= stp.
    #   Margin = sd - (sd/2 + c_size/2) - stp = sd/2 - c_size/2 - stp
    m_licon7_N = sd_N / 2 - c_size / 2 - stp
    m_licon7_P = sd_P / 2 - c_size / 2 - stp

    # diff.2 — NMOS and PMOS diffs must not overlap (inter_cell_gap >= 0)
    m_diff2 = gap

    # nwell.1 — nwell must enclose PMOS diff by nw_enc on all sides in X;
    #   the resulting nwell X width = total_x_P + 2*nw_enc must be >= nw_wmin.
    nwell_x_width = pg.total_x_um + 2 * nw_enc
    m_nwell1 = nwell_x_width - nw_wmin

    # li1.1 — drain bridge li1 width = sd_length must be >= li1_width_min
    m_li1_1 = sd_N - li1_wmin

    return np.array([
        m_poly1,
        m_diff1_N, m_diff1_P,
        m_licon5a_N, m_licon5a_P,
        m_licon7_N, m_licon7_P,
        m_diff2,
        m_nwell1,
        m_li1_1,
    ], dtype=np.float64)


# Alias used in dataset.py
margin_vector = drc_margins
