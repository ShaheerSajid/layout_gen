"""
layout_gen.cells.bit_cell — 6T SRAM bit-cell layout.

Topology
--------
Cross-coupled inverters (INV_L, INV_R) with two access pass-gate NMOS:

    BL──[PG_L: WL gate]──Q──[PD_L+PU_L (INV_L)]──Q
                              (Q_ in, Q out)         ╲ cross (TODO: polycontact)
    BL_─[PG_R: WL gate]──Q_─[PD_R+PU_R (INV_R)]──Q_╱

Default sizing (optimizer-recommended for sky130A, W in µm, L=0.15):
  W_PD = 0.80  (pull-down NMOS cross-coupled)
  W_PU = 0.42  (pull-up  PMOS cross-coupled)
  W_PG = 0.60  (access   NMOS pass transistors)

Phase-1 connections made
------------------------
- Q  node: INV_L drain li1 ↔ PG_L left S/D li1 (bridge)
- Q_ node: INV_R drain li1 ↔ PG_R left S/D li1 (bridge)
- Within each draw_inverter sub-cell NMOS drain ↔ PMOS drain (li1 bridge)

Phase-2 TODO (requires polycontact infrastructure)
---------------------------------------------------
- Cross-coupling: Q  li1 → PD_R / PU_R gate poly via polycontact
- Cross-coupling: Q_ li1 → PD_L / PU_L gate poly via polycontact
- WL: horizontal poly connecting PG_L gate and PG_R gate
"""
from __future__ import annotations

import math
from layout_gen.pdk        import PDKRules, RULES
from layout_gen.transistor import draw_transistor, transistor_geom
from layout_gen.cells.standard import (
    draw_inverter, _uname, _activate_pdk, _rect,
    _routing_gap, _diff_y, _sd_x, _gate_x,
)


def draw_bit_cell(
    w_pd:  float = 0.80,
    w_pu:  float = 0.42,
    w_pg:  float = 0.60,
    l:     float = 0.15,
    rules: PDKRules = RULES,
) -> "gf.Component":
    """
    6T SRAM bit cell.

    Parameters
    ----------
    w_pd, w_pu, w_pg :
        Channel widths (µm) for pull-down NMOS, pull-up PMOS, access NMOS.
    l :
        Gate length for all devices (µm).
    rules :
        PDK rules.

    Returns
    -------
    gf.Component
        Ports — WL_A (poly, PG_L gate), WL_B (poly, PG_R gate),
                BL (li1), BL_ (li1), Q (li1), Q_ (li1),
                VDD (met1), GND (met1).
    """
    import gdsfactory as gf
    _activate_pdk()

    # ── Geometry ──────────────────────────────────────────────────────────────
    inv_geom = transistor_geom(w_pd, l, "nmos", rules)   # NMOS of inverter
    pg_geom  = transistor_geom(w_pg, l, "nmos", rules)   # access transistor

    # all 1-finger devices have the same total_x (same sd_len and l)
    inv_tx = inv_geom.total_x_um
    pg_tx  = pg_geom.total_x_um
    sd      = inv_geom.sd_length_um          # 0.29 µm (same for all)
    spacing = rules.diff["spacing_min_um"]   # 0.27 µm

    # Cross-coupling gap must satisfy nwell spacing: the two inverter nwells
    # each extend nw_enc beyond the PMOS diff, and must be ≥ nwell.spacing_min apart.
    nw_enc    = rules.nwell["enclosure_of_pdiff_um"]   # 0.18 µm
    nw_sp     = rules.nwell["spacing_min_um"]           # 1.27 µm
    # gap ≥ nw_sp - 2*nw_enc  (between INV_L.right and INV_R.left)
    cross_gap = max(_routing_gap(rules), nw_sp - 2 * nw_enc)

    # X offsets
    # [INV_L] [Q-bridge] [PG_L]  [cross_gap]  [INV_R] [Q_-bridge] [PG_R]
    x_pg_l  = inv_tx + spacing         # PG_L starts here
    x_inv_r = x_pg_l + pg_tx + cross_gap
    x_pg_r  = x_inv_r + inv_tx + spacing
    cell_x1 = x_pg_r + pg_tx

    # ── Sub-components ────────────────────────────────────────────────────────
    inv_l = draw_inverter(w_pd, w_pu, l, rules)
    inv_r = draw_inverter(w_pd, w_pu, l, rules)
    pg_l  = draw_transistor(w_pg, l, "nmos", rules)
    pg_r  = draw_transistor(w_pg, l, "nmos", rules)

    c = gf.Component(name=_uname(
        f"bit_cell_Wpd{w_pd:.2f}_Wpu{w_pu:.2f}_Wpg{w_pg:.2f}_L{l:.3f}"))

    inv_l_ref = c.add_ref(inv_l)            # at (0, 0)
    inv_r_ref = c.add_ref(inv_r)
    inv_r_ref.movex(x_inv_r)
    pg_l_ref  = c.add_ref(pg_l)
    pg_l_ref.movex(x_pg_l)
    pg_r_ref  = c.add_ref(pg_r)
    pg_r_ref.movex(x_pg_r)

    # ── Layers ────────────────────────────────────────────────────────────────
    lyr_g   = rules.layer("poly")
    lyr_li1 = rules.layer("li1")
    lyr_m1  = rules.layer("met1")

    # NMOS diff Y bounds (local, same for inv and pg since same l)
    nd_y0, nd_y1 = _diff_y(inv_geom, rules)

    # ── Q node: bridge INV_L drain (j=1 right) → PG_L j0 left ───────────────
    # INV_L drain li1 right edge = inv_tx - sd (= x of gate right + sd ... )
    # From draw_inverter, drain = j=1 of 1-finger: x=[sd+l, 2*sd+l]
    inv_drain_x1 = 2 * sd + l      # = inv_tx = right edge of INV_L
    pg_l_j0_x0   = x_pg_l          # left edge of PG_L (j=0)
    _rect(c, inv_drain_x1, pg_l_j0_x0, nd_y0, nd_y1, lyr_li1)   # Q bridge

    # PG_L drain (j=1, right) = BL direction (no extra connection needed here)

    # ── Q_ node: bridge INV_R drain (j=1 right) → PG_R j0 left ─────────────
    inv_r_drain_x1 = x_inv_r + 2 * sd + l   # right edge of INV_R drain li1
    pg_r_j0_x0     = x_pg_r
    _rect(c, inv_r_drain_x1, pg_r_j0_x0, nd_y0, nd_y1, lyr_li1)  # Q_ bridge

    # ── Met1 power rails (span full cell width) ───────────────────────────────
    # draw_inverter already adds rails; extend them to span full bit-cell width.
    # Determine VDD rail Y from the taller PMOS of the two inverters.
    # inv_l and inv_r have the same height — grab from geometry.
    from layout_gen.transistor import transistor_geom as _tg
    pu_geom  = _tg(w_pu, l, "pmos", rules)
    gap_inv  = max(0.0, rules.diff["spacing_min_um"] - 2 * rules.poly["endcap_over_diff_um"]
                    + 2 * rules.diff["extension_past_poly_um"])
    pmos_y   = inv_geom.total_y_um + gap_inv
    rail_h   = 0.17
    cell_ytop = pmos_y + pu_geom.total_y_um
    _rect(c, 0, cell_x1, -rail_h, 0, lyr_m1)                       # GND
    _rect(c, 0, cell_x1, cell_ytop, cell_ytop + rail_h, lyr_m1)    # VDD

    # ── Ports ─────────────────────────────────────────────────────────────────
    # Q port: at the bridge between INV_L and PG_L
    q_x  = (inv_drain_x1 + pg_l_j0_x0) / 2
    q__x = (inv_r_drain_x1 + pg_r_j0_x0) / 2
    nd_ymid = (nd_y0 + nd_y1) / 2
    cx = cell_x1 / 2

    # WL_A and WL_B: at the gate poly of PG_L and PG_R
    pg_gate_x0, pg_gate_x1 = _gate_x(0, pg_geom)
    pg_l_gate_mid_x = x_pg_l + (pg_gate_x0 + pg_gate_x1) / 2
    pg_r_gate_mid_x = x_pg_r + (pg_gate_x0 + pg_gate_x1) / 2
    gate_mid_y      = pg_geom.total_y_um / 2

    c.add_port("WL_A", center=(pg_l_gate_mid_x, gate_mid_y),
               width=pg_geom.total_y_um, orientation=90,  layer=lyr_g)
    c.add_port("WL_B", center=(pg_r_gate_mid_x, gate_mid_y),
               width=pg_geom.total_y_um, orientation=90,  layer=lyr_g)
    c.add_port("BL",   center=(x_pg_l + (sd + l + sd * 1.5), nd_ymid),
               width=nd_y1 - nd_y0, orientation=0,   layer=lyr_li1)
    c.add_port("BL_",  center=(x_pg_r + (sd + l + sd * 1.5), nd_ymid),
               width=nd_y1 - nd_y0, orientation=0,   layer=lyr_li1)
    c.add_port("Q",    center=(q_x,   nd_ymid),
               width=spacing,       orientation=90,  layer=lyr_li1)
    c.add_port("Q_",   center=(q__x,  nd_ymid),
               width=spacing,       orientation=90,  layer=lyr_li1)
    c.add_port("GND",  center=(cx, -rail_h / 2),
               width=cell_x1,       orientation=270, layer=lyr_m1)
    c.add_port("VDD",  center=(cx, cell_ytop + rail_h / 2),
               width=cell_x1,       orientation=90,  layer=lyr_m1)
    return c
