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
cell_features(w_N, w_P, l, rules, *, gap_y, finger_N, finger_P) -> np.ndarray  shape (23,)
drc_margins(w_N, w_P, l, rules, *, gap_y, finger_N, finger_P)   -> np.ndarray  shape (12,)
margin_vector(...)                                                -> alias for drc_margins

The three optional keyword arguments extend the optimisation space:

gap_y : float | None
    Y spacing between NMOS poly top and PMOS poly bottom (µm).
    Defaults to ``_inter_cell_gap(rules)`` (PDK minimum).
finger_N, finger_P : float | None
    Number of gate fingers for NMOS / PMOS (continuous relaxation —
    rounded to the nearest integer before geometry is computed).
    Defaults to the finger count auto-derived by ``transistor_geom``.

DRC margins are *signed distances* from each rule threshold.
Positive  = constraint satisfied (slack from the boundary).
Negative  = constraint violated (extent of violation).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from layout_gen.pdk        import PDKRules
from layout_gen.transistor import transistor_geom, TransistorGeom
from layout_gen.cells.standard import _inter_cell_gap

# ── Column name lists ─────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Explicit optimisation params (6)
    "w_N", "w_P", "l",
    "gap_y", "finger_N", "finger_P",
    # Computed geometry (8)
    "sd_length",
    "w_finger_N", "w_finger_P",
    "total_x_N", "total_x_P",
    "total_y_N", "total_y_P",
    "inter_cell_gap_min",   # PDK minimum gap (constant for a given PDK)
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
]  # 23 total

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
    "li1.2",        # li1 spacing across gate (S/D li1 gap >= li1_spacing_min)
    "poly.2",       # poly spacing between adjacent devices
]  # 12 total


# ── Geometry helper ───────────────────────────────────────────────────────────

def _geom_override_fingers(
    geom:     TransistorGeom,
    n_fingers: int,
    rules:    PDKRules,
) -> TransistorGeom:
    """Return a copy of *geom* with the finger count overridden.

    Recomputes ``w_finger_um``, ``total_x_um``, ``total_y_um``, and
    ``n_contacts_y`` from the new finger count.
    """
    n   = max(1, n_fingers)
    w_f = geom.w_um / n
    endcap = rules.poly["endcap_over_diff_um"]
    return replace(
        geom,
        n_fingers    = n,
        w_finger_um  = w_f,
        total_x_um   = (n + 1) * geom.sd_length_um + n * geom.l_um,
        total_y_um   = w_f + 2 * endcap,
        n_contacts_y = rules.sd_contact_columns(w_f),
    )


# ── Feature extraction ────────────────────────────────────────────────────────

def cell_features(
    w_N:      float,
    w_P:      float,
    l:        float,
    rules:    PDKRules,
    *,
    gap_y:    float | None = None,
    finger_N: float | None = None,
    finger_P: float | None = None,
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
    gap_y :
        Y spacing between NMOS poly top and PMOS poly bottom (µm).
        Defaults to ``_inter_cell_gap(rules)`` (PDK minimum).
    finger_N, finger_P :
        Number of gate fingers (continuous relaxation; rounded to nearest
        integer).  Defaults to the auto-derived finger count.

    Returns
    -------
    np.ndarray, shape (23,)
        See :data:`FEATURE_NAMES` for column order.
    """
    ng_base = transistor_geom(w_N, l, "nmos", rules)
    pg_base = transistor_geom(w_P, l, "pmos", rules)

    ng = (_geom_override_fingers(ng_base, int(round(finger_N)), rules)
          if finger_N is not None else ng_base)
    pg = (_geom_override_fingers(pg_base, int(round(finger_P)), rules)
          if finger_P is not None else pg_base)

    _gap_y   = gap_y if gap_y is not None else _inter_cell_gap(rules)
    _gap_min = _inter_cell_gap(rules)

    p  = rules.poly
    d  = rules.diff
    c  = rules.contacts
    li = rules.li1
    nw = rules.nwell

    return np.array([
        # explicit optimisation params
        w_N, w_P, l,
        _gap_y, float(ng.n_fingers), float(pg.n_fingers),
        # computed geometry
        ng.sd_length_um,
        ng.w_finger_um, pg.w_finger_um,
        ng.total_x_um,  pg.total_x_um,
        ng.total_y_um,  pg.total_y_um,
        _gap_min,
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
    w_N:      float,
    w_P:      float,
    l:        float,
    rules:    PDKRules,
    *,
    gap_y:    float | None = None,
    finger_N: float | None = None,
    finger_P: float | None = None,
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
    gap_y :
        Y spacing between NMOS poly top and PMOS poly bottom (µm).
        Defaults to ``_inter_cell_gap(rules)`` (PDK minimum).
    finger_N, finger_P :
        Number of gate fingers (continuous relaxation; rounded to nearest
        integer).  Defaults to the auto-derived finger count.

    Returns
    -------
    np.ndarray, shape (12,)
        See :data:`MARGIN_NAMES` for column order.
    """
    ng_base = transistor_geom(w_N, l, "nmos", rules)
    pg_base = transistor_geom(w_P, l, "pmos", rules)

    ng = (_geom_override_fingers(ng_base, int(round(finger_N)), rules)
          if finger_N is not None else ng_base)
    pg = (_geom_override_fingers(pg_base, int(round(finger_P)), rules)
          if finger_P is not None else pg_base)

    gap_min = _inter_cell_gap(rules)
    _gap_y  = gap_y if gap_y is not None else gap_min

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

    # diff.2 — gap between NMOS and PMOS diff rows must meet PDK minimum.
    #   margin = gap_y - gap_min  (positive when gap_y >= minimum required)
    m_diff2 = _gap_y - gap_min

    # nwell.1 — nwell must enclose PMOS diff by nw_enc on all sides in X;
    #   the resulting nwell X width = total_x_P + 2*nw_enc must be >= nw_wmin.
    nwell_x_width = pg.total_x_um + 2 * nw_enc
    m_nwell1 = nwell_x_width - nw_wmin

    # li1.1 — drain bridge li1 width = sd_length must be >= li1_width_min
    m_li1_1 = sd_N - li1_wmin

    # li1.2 — li1 spacing across gate: the gap between adjacent S/D li1 rails
    #   equals the gate length after pullback.  With pullback applied on both
    #   sides (pullback = max(0, (li1_sp - l) / 2)), the effective gap is:
    #   gap = l + 2 * pullback = max(l, li1_sp).
    #   Without pullback the gap = l.  The analytical margin measures the raw
    #   gap so the model can learn when pullback is needed:
    #   margin = l - li1_spacing_min   (negative when l < li1_sp)
    li1_sp  = li["spacing_min_um"]
    m_li1_2 = l - li1_sp

    # poly.2 — poly spacing between adjacent devices (e.g. INV drain gate vs
    #   adjacent PG gate in a bit cell).  The minimum gap = diff.spacing_min
    #   (device-to-device spacing).  Margin = diff_spacing - poly_spacing_min.
    poly_sp = p["spacing_min_um"]
    diff_sp = d["spacing_min_um"]
    m_poly2 = diff_sp - poly_sp

    return np.array([
        m_poly1,
        m_diff1_N, m_diff1_P,
        m_licon5a_N, m_licon5a_P,
        m_licon7_N, m_licon7_P,
        m_diff2,
        m_nwell1,
        m_li1_1,
        m_li1_2,
        m_poly2,
    ], dtype=np.float64)


# Alias used in dataset.py
margin_vector = drc_margins
