"""
layout_gen.cells.bit_cell — 6T SRAM bit-cell layout.

Topology
--------
Cross-coupled inverters (INV_L, INV_R) with two access pass-gate NMOS:

    BL──[PG_L: WL gate]──Q──[PD_L+PU_L (INV_L)]──Q_
                                                      ╲ cross (Phase-2)
    BL_─[PG_R: WL gate]──Q_─[PD_R+PU_R (INV_R)]──Q ╱

Default sizing (optimizer-recommended for sky130A, W in µm, L=0.15):
  W_PD = 0.80  (pull-down NMOS cross-coupled)
  W_PU = 0.42  (pull-up  PMOS cross-coupled)
  W_PG = 0.60  (access   NMOS pass transistors)

Phase-1 connections (implemented)
----------------------------------
- Q  node: INV_L drain li1 ↔ PG_L source li1 (li1 bridge)
- Q_ node: INV_R drain li1 ↔ PG_R source li1 (li1 bridge)
- WL:  PG_L gate and PG_R gate tied via polycontact stubs → mcon → met1 bus

WL routing detail
-----------------
L=0.15 µm gate poly is narrower than the 0.17 µm licon1 contact.  The
source/drain li1 leaves only 0.15 µm clear between them — not enough for
contact (0.17 µm) plus 0.17 µm li1 spacing on both sides.  Solution: extend
the PG gate poly upward past the transistor body (above total_y_um), where no
S/D li1 exists, widen to a 0.27 µm pad (0.05 µm poly enclosure each side),
and drop a licon1 polycontact + mcon there.  A met1 bus then connects both PG
gate polycontacts across the full cell width.

Phase-2 TODO
------------
Cross-coupling (Q → INV_R gate, Q_ → INV_L gate) has the same fundamental
geometry constraint: any li1 track reaching the INV gate poly in the
inter-cell gap conflicts with the adjacent OUT drain bridge li1 (same x-column,
different net, < 0.17 µm spacing).  Needs met1 routing via a polycontact stub
placed above the PMOS body (y > pmos_y + pg.total_y_um), mirroring the WL
approach used here.
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
    lyr_g       = rules.layer("poly")
    lyr_li1     = rules.layer("li1")
    lyr_m1      = rules.layer("met1")
    lyr_contact = rules.layer("licon1")
    lyr_mcon    = rules.layer("mcon")

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

    # ── WL polycontact stubs + met1 bus ───────────────────────────────────────
    # The PG gate poly (l=0.15 µm) is too narrow to fit a 0.17 µm licon1 with
    # 0.17 µm li1 spacing to source AND drain.  Instead we extend the gate poly
    # vertically above the PG transistor body (above pg_ty = pg_geom.total_y_um)
    # where no S/D li1 exists, widen to a 0.27 µm pad, and drop a polycontact.
    pg_ty       = pg_geom.total_y_um               # PG body top (0.86 µm)
    c_size      = rules.contacts["size_um"]         # licon1/mcon size: 0.17 µm
    enc_poly    = rules.contacts.get("poly_enclosure_um", 0.05)   # poly→licon1
    enc_m1      = rules.mcon.get("enclosure_in_met1_um", 0.03)
    pad_half    = (c_size + 2 * enc_poly) / 2       # 0.135 µm
    ch          = c_size / 2                         # 0.085 µm

    pg_gate_x0, pg_gate_x1 = _gate_x(0, pg_geom)
    pg_gate_cx  = (pg_gate_x0 + pg_gate_x1) / 2    # 0.365 µm (local)
    stub_cy     = pg_ty + pad_half                   # polycontact Y centre (0.995 µm)

    for pg_ref_x in (x_pg_l, x_pg_r):
        gcx = pg_ref_x + pg_gate_cx
        # Widened poly pad extending above PG body (same gate net → just widens)
        _rect(c, gcx - pad_half, gcx + pad_half,
                 pg_ty, pg_ty + 2 * pad_half, lyr_g)
        # Polycontact (licon1 on poly)
        _rect(c, gcx - ch, gcx + ch, stub_cy - ch, stub_cy + ch, lyr_contact)
        # Li1 landing covering the licon1 (enc = 0)
        _rect(c, gcx - ch, gcx + ch, stub_cy - ch, stub_cy + ch, lyr_li1)
        # mcon via (li1 → met1)
        _rect(c, gcx - ch, gcx + ch, stub_cy - ch, stub_cy + ch, lyr_mcon)

    # WL met1 bus spanning PG_L to PG_R (met1 encloses mcon by enc_m1)
    wl_x0 = x_pg_l + pg_gate_cx - ch - enc_m1
    wl_x1 = x_pg_r + pg_gate_cx + ch + enc_m1
    wl_y0 = stub_cy - ch - enc_m1
    wl_y1 = stub_cy + ch + enc_m1
    _rect(c, wl_x0, wl_x1, wl_y0, wl_y1, lyr_m1)

    # ── Ports ─────────────────────────────────────────────────────────────────
    q_x     = (inv_drain_x1   + pg_l_j0_x0) / 2
    q__x    = (inv_r_drain_x1 + pg_r_j0_x0) / 2
    nd_ymid = (nd_y0 + nd_y1) / 2
    cx      = cell_x1 / 2

    # BL and BL_: rightmost S/D (drain) of each PG transistor
    pg_d_x0, pg_d_x1 = _sd_x(1, pg_geom)    # j=1 (drain/BL side, local)
    bl_x_mid  = x_pg_l + (pg_d_x0 + pg_d_x1) / 2
    bl__x_mid = x_pg_r + (pg_d_x0 + pg_d_x1) / 2

    c.add_port("WL",  center=((wl_x0 + wl_x1) / 2, (wl_y0 + wl_y1) / 2),
               width=wl_x1 - wl_x0, orientation=90,  layer=lyr_m1)
    c.add_port("BL",  center=(bl_x_mid,  nd_ymid),
               width=nd_y1 - nd_y0,  orientation=0,   layer=lyr_li1)
    c.add_port("BL_", center=(bl__x_mid, nd_ymid),
               width=nd_y1 - nd_y0,  orientation=0,   layer=lyr_li1)
    c.add_port("Q",   center=(q_x,   nd_ymid),
               width=spacing,        orientation=90,  layer=lyr_li1)
    c.add_port("Q_",  center=(q__x,  nd_ymid),
               width=spacing,        orientation=90,  layer=lyr_li1)
    c.add_port("GND", center=(cx, -rail_h / 2),
               width=cell_x1,        orientation=270, layer=lyr_m1)
    c.add_port("VDD", center=(cx, cell_ytop + rail_h / 2),
               width=cell_x1,        orientation=90,  layer=lyr_m1)
    return c
