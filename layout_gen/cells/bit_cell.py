"""
layout_gen.cells.bit_cell — 6T SRAM bit-cell layout.

Topology
--------
Cross-coupled inverters (INV_L, INV_R) with two access pass-gate NMOS:

    BL──[PG_L: WL gate]──Q──[PD_L+PU_L (INV_L)]──Q_
                                                      ╲ cross-coupled
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

Phase-2 cross-coupling (implemented)
-------------------------------------
Cross-coupling faces the same geometry constraint AND must cross the VDD met1
rail.  Solution: route via met2 (above met1).

Layer stack per coupling path:
  Q/Q_ li1 bridge → mcon → met1 → via1 → met2 (L-shaped route) → via1
  → met1 → mcon → li1 → licon1 → INV gate poly stub above PMOS body

The gate poly stub is placed above cell_ytop (PMOS top), at minimum distance
from the VDD rail to satisfy met1 spacing rules.  The met2 L-shape routes
at one level for Q→INV_R and one level higher for Q_→INV_L to avoid
overlap where the two paths cross in the horizontal X span.
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
        Ports — WL  (met1, orientation 180° West — row decoder side),
                BL  (li1,  orientation  90° North — bitline column up),
                BL_ (li1,  orientation  90° North — bitline column up),
                Q / Q_ (li1, orientation 90° North — internal, debug),
                VDD (met1, orientation  90° North — top power rail),
                GND (met1, orientation 270° South — bottom power rail).
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
    lyr_via1    = rules.layer("via1")
    lyr_m2      = rules.layer("met2")

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
    enc_li_2adj = rules.contacts.get("enclosure_in_li1_2adj_um", 0.08)  # li.5
    enc_m1_2adj = rules.met1.get("enclosure_of_mcon_2adj_um", 0.06)     # m1.5
    pad_half    = (c_size + 2 * enc_poly) / 2       # 0.135 µm
    ch          = c_size / 2                         # 0.085 µm
    li1_lh      = ch + enc_li_2adj                   # li1 landing half (li.5)
    m1_lh       = ch + enc_m1_2adj                   # met1 landing half (m1.5)
    li1_sp      = rules.li1.get("spacing_min_um", 0.17)

    pg_gate_x0, pg_gate_x1 = _gate_x(0, pg_geom)
    pg_gate_cx  = (pg_gate_x0 + pg_gate_x1) / 2    # 0.365 µm (local)

    # Place WL stub high enough to maintain li1 spacing from PG S/D li1.
    # PG S/D li1 top edge = diff_y1 = endcap + w_finger + ext
    _, pg_diff_y1 = _diff_y(pg_geom, rules)
    # Constraint: stub li1 bottom (stub_cy - li1_lh) >= pg_diff_y1 + li1_sp
    stub_cy_min  = pg_diff_y1 + li1_sp + li1_lh
    # Also must be above PG body top + pad_half (for poly pad geometry)
    stub_cy     = max(pg_ty + pad_half, stub_cy_min)

    for pg_ref_x in (x_pg_l, x_pg_r):
        gcx = pg_ref_x + pg_gate_cx
        # Widened poly pad extending above PG body to enclose polycontact
        poly_pad_top = stub_cy + ch + enc_poly
        _rect(c, gcx - pad_half, gcx + pad_half,
                 pg_ty, poly_pad_top, lyr_g)
        # Polycontact (licon1 on poly)
        _rect(c, gcx - ch, gcx + ch, stub_cy - ch, stub_cy + ch, lyr_contact)
        # Li1 landing covering the licon1 (li.5: 2-adj-edge enc)
        _rect(c, gcx - li1_lh, gcx + li1_lh, stub_cy - li1_lh, stub_cy + li1_lh, lyr_li1)
        # mcon via (li1 → met1)
        _rect(c, gcx - ch, gcx + ch, stub_cy - ch, stub_cy + ch, lyr_mcon)

    # WL met1 bus spanning PG_L to PG_R (m1.5: 2-adj-edge enc of mcon)
    wl_x0 = x_pg_l + pg_gate_cx - m1_lh
    wl_x1 = x_pg_r + pg_gate_cx + m1_lh
    wl_y0 = stub_cy - m1_lh
    wl_y1 = stub_cy + m1_lh
    _rect(c, wl_x0, wl_x1, wl_y0, wl_y1, lyr_m1)

    # ── Phase-2: Cross-coupling via met2 ──────────────────────────────────────
    # Q → INV_R gate  (PD_R.G + PU_R.G = storage node Q connects to right inv gate)
    # Q_ → INV_L gate (PD_L.G + PU_L.G = storage node Q_ connects to left inv gate)
    #
    # Layer stack: Q/Q_ li1 → mcon → met1 → via1 → met2 (L-route) → via1
    #              → met1 → mcon → li1 → licon1 → INV gate poly stub above PMOS
    #
    # The gate poly stub cannot be placed alongside the PMOS S/D li1 (same X
    # column, l < licon1 contact size → no room for li1 spacing).  Instead it
    # is placed above cell_ytop where no S/D li1 exists.  A via1+met2 stack
    # jumps over the VDD met1 rail connecting Q at NMOS level to the gate stub.

    # Geometry constants for gate stubs
    enc_poly  = rules.contacts.get("poly_enclosure_um", 0.05)
    enc_li_licon_2adj = rules.contacts.get("enclosure_in_li1_2adj_um", 0.08)
    enc_m1_mcon_2adj = rules.met1.get("enclosure_of_mcon_2adj_um", 0.06)
    _via1       = getattr(rules, "via1", None) or {}
    via1_size   = _via1.get("size_um", 0.15)
    enc_m1_via_2adj = _via1.get("enclosure_in_met1_2adj_um", 0.085)
    _met2       = getattr(rules, "met2", None) or {}
    enc_m2_via_2adj = _met2.get("enclosure_of_via1_2adj_um", 0.085)
    vh          = via1_size / 2                             # 0.075 µm (half of via1)
    met1_sp     = rules.met1.get("spacing_min_um", 0.14)
    met2_w      = _met2.get("width_min_um",   0.14)
    met2_sp     = _met2.get("spacing_min_um", 0.14)
    pad_half_inv = (c_size + 2 * enc_poly) / 2            # 0.135 µm half-width of poly pad

    # Landing half-extents that satisfy 2-adjacent-edge enclosure rules
    li1_land_half = ch + enc_li_licon_2adj            # li.5
    m1_land_half  = max(ch + enc_m1_mcon_2adj,        # m1.5
                        vh + enc_m1_via_2adj)          # via.5a
    m2_land_half  = vh + enc_m2_via_2adj              # m2.5

    # Gate stub centre Y: met1 bottom must clear VDD rail top + met1_spacing
    gsc_y = cell_ytop + rail_h + met1_sp + m1_land_half

    # INV gate centre X (same for INV_L and INV_R, offset by x_inv_r for INV_R)
    ig_x0, ig_x1 = _gate_x(0, inv_geom)
    inv_gcx  = (ig_x0 + ig_x1) / 2          # INV_L gate centre (global, INV_L at x=0)
    inv_r_gcx = x_inv_r + inv_gcx            # INV_R gate centre (global)

    # ── Gate poly stubs + full via stack for both INV_L and INV_R ─────────────
    for gcx in (inv_gcx, inv_r_gcx):
        # Widen gate poly above PMOS body to accept a polycontact
        _rect(c, gcx - pad_half_inv, gcx + pad_half_inv,
                 cell_ytop, gsc_y + ch + enc_poly, lyr_g)
        # Licon1 (polycontact: poly → li1)
        _rect(c, gcx - ch, gcx + ch, gsc_y - ch, gsc_y + ch, lyr_contact)
        # Li1 landing (li.5: 2-adj-edge enc of licon)
        _rect(c, gcx - li1_land_half, gcx + li1_land_half,
                 gsc_y - li1_land_half, gsc_y + li1_land_half, lyr_li1)
        # Mcon (li1 → met1)
        _rect(c, gcx - ch, gcx + ch, gsc_y - ch, gsc_y + ch, lyr_mcon)
        # Met1 landing (m1.5 mcon enc + via.5a via1 enc, 2-adj)
        _rect(c, gcx - m1_land_half, gcx + m1_land_half,
                 gsc_y - m1_land_half, gsc_y + m1_land_half, lyr_m1)
        # Via1 (met1 → met2)
        _rect(c, gcx - vh, gcx + vh, gsc_y - vh, gsc_y + vh, lyr_via1)
        # Met2 landing (m2.5: 2-adj-edge enc of via1)
        _rect(c, gcx - m2_land_half, gcx + m2_land_half,
                 gsc_y - m2_land_half, gsc_y + m2_land_half, lyr_m2)

    # ── Compute Q/Q_ bridge centres (used for port placement too) ─────────────
    q_x     = (inv_drain_x1   + pg_l_j0_x0) / 2
    q__x    = (inv_r_drain_x1 + pg_r_j0_x0) / 2
    nd_ymid = (nd_y0 + nd_y1) / 2

    # ── Via stacks at Q and Q_ li1 nodes → met2 ───────────────────────────────
    for nx in (q_x, q__x):
        # Mcon (existing Q/Q_ li1 → met1)
        _rect(c, nx - ch, nx + ch, nd_ymid - ch, nd_ymid + ch, lyr_mcon)
        # Met1 landing (m1.5 + via.5a, 2-adj)
        _rect(c, nx - m1_land_half, nx + m1_land_half,
                 nd_ymid - m1_land_half, nd_ymid + m1_land_half, lyr_m1)
        # Via1 (met1 → met2)
        _rect(c, nx - vh, nx + vh, nd_ymid - vh, nd_ymid + vh, lyr_via1)
        # Met2 landing (m2.5, 2-adj)
        _rect(c, nx - m2_land_half, nx + m2_land_half,
                 nd_ymid - m2_land_half, nd_ymid + m2_land_half, lyr_m2)

    # ── Q → INV_R gate: L-shaped met2 route ──────────────────────────────────
    hw = met2_w / 2   # met2 wire half-width
    # Extend horizontal wires to cover via landing edges to avoid narrow notches.
    q_xlo = min(q_x, inv_r_gcx) - hw
    q_xhi = max(q_x, inv_r_gcx) + hw
    # Vertical at X = q_x from nd_ymid up to gsc_y (extended into horizontal)
    _rect(c, q_x - hw, q_x + hw, nd_ymid, gsc_y + hw, lyr_m2)
    # Horizontal at Y = gsc_y from q_x to inv_r_gcx
    _rect(c, q_xlo, q_xhi, gsc_y - hw, gsc_y + hw, lyr_m2)

    # ── Q_ → INV_L gate: U-shaped met2 route (one track higher than Q route) ──
    # Track pitch must clear gate-stub met2 landing (m2_land_half above gsc_y)
    # plus met2 spacing, plus wire half-width.
    track_pitch = m2_land_half + met2_sp + hw
    qb_route_y = gsc_y + track_pitch   # horizontal level for Q_ route
    qb_xlo = min(inv_gcx, q__x) - hw
    qb_xhi = max(inv_gcx, q__x) + hw
    # Vertical at X = q__x from nd_ymid up to qb_route_y
    _rect(c, q__x - hw, q__x + hw, nd_ymid, qb_route_y + hw, lyr_m2)
    # Horizontal at Y = qb_route_y from inv_gcx to q__x
    _rect(c, qb_xlo, qb_xhi, qb_route_y - hw, qb_route_y + hw, lyr_m2)
    # Vertical at X = inv_gcx from gsc_y down to qb_route_y (connects to INV_L gate stub)
    _rect(c, inv_gcx - hw, inv_gcx + hw, gsc_y - hw, qb_route_y + hw, lyr_m2)

    # ── Ports ─────────────────────────────────────────────────────────────────
    cx      = cell_x1 / 2

    # BL and BL_: rightmost S/D (drain) of each PG transistor
    pg_d_x0, pg_d_x1 = _sd_x(1, pg_geom)    # j=1 (drain/BL side, local)
    bl_x_mid  = x_pg_l + (pg_d_x0 + pg_d_x1) / 2
    bl__x_mid = x_pg_r + (pg_d_x0 + pg_d_x1) / 2

    # WL: horizontal met1 wordline bus — exits West toward the row decoder
    c.add_port("WL",  center=(wl_x0, (wl_y0 + wl_y1) / 2),
               width=wl_y1 - wl_y0, orientation=180, layer=lyr_m1)
    # BL/BL_: vertical li1 bitlines — exit North through the array column
    bl_w = pg_geom.sd_length_um   # S/D li1 X-width (perpendicular to North exit)
    c.add_port("BL",  center=(bl_x_mid,  nd_ymid),
               width=bl_w, orientation=90,  layer=lyr_li1)
    c.add_port("BL_", center=(bl__x_mid, nd_ymid),
               width=bl_w, orientation=90,  layer=lyr_li1)
    c.add_port("Q",   center=(q_x,   nd_ymid),
               width=spacing,        orientation=90,  layer=lyr_li1)
    c.add_port("Q_",  center=(q__x,  nd_ymid),
               width=spacing,        orientation=90,  layer=lyr_li1)
    c.add_port("GND", center=(cx, -rail_h / 2),
               width=cell_x1,        orientation=270, layer=lyr_m1)
    c.add_port("VDD", center=(cx, cell_ytop + rail_h / 2),
               width=cell_x1,        orientation=90,  layer=lyr_m1)
    return c
