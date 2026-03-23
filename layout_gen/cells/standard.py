"""
layout_gen.cells — CMOS standard cell primitives.

Builds parametric standard cells on top of
:func:`~layout_gen.transistor.draw_transistor`.

Floor-plan (all cells)
-----------------------
- NMOS occupies the lower body (substrate).
- PMOS occupies the upper body, inside an N-well.
- Met1 GND rail runs below NMOS; met1 VDD rail runs above PMOS.
- Shared poly gate fingers run continuously through the inter-device gap.
- Li1 carries all internal signal routing.

Gate-input ports are placed at the left edge of the poly bridge, mid-gap Y.
OUT port is placed at the right edge of the drain li1 column.

Current cells
-------------
draw_inverter   INV   — 1 NMOS + 1 PMOS
draw_nand2      NAND2 — series NMOS pair + parallel PMOS pair
draw_nor2       NOR2  — parallel NMOS pair + series PMOS pair

All dimensions in µm, all rules from the PDK YAML.
"""
from __future__ import annotations

import math
from layout_gen.pdk        import PDKRules, RULES
from layout_gen.transistor import draw_transistor, transistor_geom, TransistorGeom

# ── cell-name uniquifier (same pattern as transistor.py) ─────────────────────
_CELL_COUNTER: dict[str, int] = {}

def _uname(base: str) -> str:
    _CELL_COUNTER[base] = _CELL_COUNTER.get(base, 0) + 1
    n = _CELL_COUNTER[base]
    return base if n == 1 else f"{base}${n}"


def _activate_pdk() -> None:
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except ValueError:
        from gdsfactory.generic_tech import PDK as _GENERIC
        _GENERIC.activate()


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _sd_x(
    j: int,
    geom: TransistorGeom,
    rules: PDKRules | None = None,
) -> tuple[float, float]:
    """(x0, x1) of the j-th source/drain li1 region in local transistor coords.

    When *rules* is provided, the edges adjacent to poly gate fingers are
    pulled back to maintain ``li1.spacing_min_um`` (mirrors the pullback
    applied in :func:`~layout_gen.transistor.draw_transistor`).
    """
    x0 = j * (geom.sd_length_um + geom.l_um)
    x1 = x0 + geom.sd_length_um

    if rules is not None:
        li1_sp   = rules.li1.get("spacing_min_um", 0.17)
        c_size   = rules.contacts["size_um"]
        li_enc   = rules.contacts.get("enclosure_in_li1_um", 0.0)
        half_sd  = geom.sd_length_um / 2
        pullback = max(0.0, (li1_sp - geom.l_um) / 2)
        pullback = min(pullback, max(0.0, half_sd - c_size / 2 - li_enc))

        has_poly_left  = (j > 0)
        has_poly_right = (j < geom.n_fingers)
        if has_poly_left:
            x0 += pullback
        if has_poly_right:
            x1 -= pullback

    return x0, x1


def _gate_x(i: int, geom: TransistorGeom) -> tuple[float, float]:
    """(x0, x1) of the i-th poly gate finger in local transistor coords."""
    x0 = (i + 1) * geom.sd_length_um + i * geom.l_um
    return x0, x0 + geom.l_um


def _diff_y(geom: TransistorGeom, rules: PDKRules) -> tuple[float, float]:
    """(y0, y1) of diffusion in local transistor Y coords."""
    endcap = rules.poly["endcap_over_diff_um"]
    ext    = rules.diff["extension_past_poly_um"]
    return endcap - ext, endcap + geom.w_finger_um + ext


def _inter_cell_gap(rules: PDKRules) -> float:
    """
    Minimum Y gap between NMOS poly top edge and PMOS poly bottom edge
    such that the diff-to-diff spacing rule is satisfied.
    """
    endcap  = rules.poly["endcap_over_diff_um"]
    ext     = rules.diff["extension_past_poly_um"]
    min_sep = rules.diff["spacing_min_um"]
    # With gap G: separation = G + 2*endcap - 2*ext >= min_sep
    return max(0.0, min_sep - 2 * endcap + 2 * ext)


def _routing_gap(rules: PDKRules) -> float:
    """
    Y gap large enough to fit one horizontal li1 routing track between NMOS
    and PMOS (needed for multi-input cells).
    """
    li1_sp = rules.li1["spacing_min_um"]
    li1_w  = rules.li1["width_min_um"]
    # track needs li1_sp from NMOS diff top + li1_w + li1_sp from PMOS diff bot
    endcap = rules.poly["endcap_over_diff_um"]
    ext    = rules.diff["extension_past_poly_um"]
    min_sep = rules.diff["spacing_min_um"]
    # pd_y0 - nd_y1 = pmos_y - w_n - 2*ext + endcap - endcap - ext ... simplifies:
    # pd_y0 - nd_y1 = pmos_y - w_n - 2*ext + gap_adj
    # we need >= 2*li1_sp + li1_w  (spacing from NMOS li1, track width, spacing to PMOS li1)
    needed = 2 * li1_sp + li1_w   # 0.51 for sky130A
    # pd_y0 - nd_y1 = gap + 2*endcap - 2*ext - diff_min_sep (from diff spacing equation)
    #   = gap + 2*endcap - 2*ext when gap is the raw total gap
    # Simpler: gap = needed + 2*ext - 2*endcap + diff spacing tweak
    base = max(0.0, min_sep - 2 * endcap + 2 * ext)  # _inter_cell_gap
    # The track needs pd_y0 - nd_y1 >= 2*sp + w. pd_y0-nd_y1 = gap - (min_sep-base).
    # Solve: gap >= needed - (-(min_sep - base)) ... let's just iterate.
    # pd_y0 - nd_y1 = (gap + 2*endcap - 2*ext) - (2*endcap - 2*ext) ... ugh.
    # Direct formula: pd_y0 - nd_y1 = gap + 2*endcap - 2*ext - diff_sep + diff_sep
    #                               = gap - (2*ext - 2*endcap + min_sep - min_sep)
    # From the diff y definition: pd_y0 = pmos_y + endcap - ext,
    #   nd_y1 = endcap + w_n + ext.  ng.total_y = w_n + 2*endcap.
    #   pd_y0 - nd_y1 = (pmos_y + endcap - ext) - (endcap + w_n + ext)
    #                 = pmos_y - w_n - 2*ext
    #                 = (ng.total_y + gap) - w_n - 2*ext
    #                 = (w_n + 2*endcap + gap) - w_n - 2*ext
    #                 = 2*endcap - 2*ext + gap
    #   Need: 2*endcap - 2*ext + gap >= 2*li1_sp + li1_w
    #   => gap >= 2*li1_sp + li1_w - 2*endcap + 2*ext
    gap_needed = needed - 2 * endcap + 2 * ext
    return max(base, gap_needed)


def _rect(c, x0: float, x1: float, y0: float, y1: float, layer) -> None:
    """Add a rectangle polygon to component c."""
    c.add_polygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        layer=layer,
    )


# ── Inverter ──────────────────────────────────────────────────────────────────

def draw_inverter(
    w_n:   float,
    w_p:   float,
    l:     float,
    rules: PDKRules = RULES,
    gap:   float | None = None,
) -> "gf.Component":
    """
    CMOS inverter: 1 NMOS + 1 PMOS.

    Parameters
    ----------
    w_n, w_p :
        NMOS / PMOS channel width (µm).
    l :
        Gate length for both devices (µm).
    rules :
        PDK rules.
    gap :
        Y gap between NMOS poly top and PMOS poly bottom (µm).
        Defaults to the minimum gap from PDK rules.  Pass a larger value
        (e.g. 0.90) when the caller needs extra space in the inter-cell region
        for routing tracks or polycontacts.

    Returns
    -------
    gf.Component
        Ports — IN (poly), OUT (li1), VDD (met1), GND (met1).
    """
    import gdsfactory as gf
    _activate_pdk()

    ng = transistor_geom(w_n, l, "nmos", rules)
    pg = transistor_geom(w_p, l, "pmos", rules)

    if gap is None:
        gap = _inter_cell_gap(rules)
    pmos_y  = ng.total_y_um + gap   # Y offset of PMOS bottom edge

    nmos_c  = draw_transistor(w_n, l, "nmos", rules)
    pmos_c  = draw_transistor(w_p, l, "pmos", rules)

    c = gf.Component(name=_uname(f"inv_Wn{w_n:.3f}_Wp{w_p:.3f}_L{l:.3f}"))
    nr = c.add_ref(nmos_c)    # placed at (0, 0) — no movey needed
    pr = c.add_ref(pmos_c)
    pr.movey(pmos_y)

    lyr_g   = rules.layer("poly")
    lyr_li1 = rules.layer("li1")
    lyr_m1  = rules.layer("met1")

    nd_y0, nd_y1 = _diff_y(ng, rules)
    pd_y0 = pmos_y + _diff_y(pg, rules)[0]

    # Gate 0: x range is same for NMOS and PMOS (sd_length identical)
    gx0, gx1 = _gate_x(0, ng)

    # Poly bridge in the inter-cell gap
    if gap > 0:
        _rect(c, gx0, gx1, ng.total_y_um, pmos_y, lyr_g)

    # OUT: drain = j=1 for a 1-finger device
    dx0, dx1 = _sd_x(1, ng, rules)
    # Li1 bridge from NMOS drain top to PMOS drain bottom
    _rect(c, dx0, dx1, nd_y1, pd_y0, lyr_li1)

    # Met1 power rails (above/below the cell)
    rail_h   = 0.17    # met1 min width
    cell_x1  = ng.total_x_um
    cell_ytop = pmos_y + pg.total_y_um
    _rect(c, 0, cell_x1, -rail_h, 0,          lyr_m1)   # GND
    _rect(c, 0, cell_x1, cell_ytop, cell_ytop + rail_h, lyr_m1)   # VDD

    # Ports
    cx = cell_x1 / 2
    gate_mid_y = (ng.total_y_um + pmos_y) / 2
    out_mid_y  = (nd_y1 + pd_y0) / 2
    c.add_port("IN",  center=(gx0, gate_mid_y),    width=max(gap, l),      orientation=180, layer=lyr_g)
    c.add_port("OUT", center=(dx1, out_mid_y),      width=max(pd_y0-nd_y1, l), orientation=0, layer=lyr_li1)
    c.add_port("GND", center=(cx, -rail_h / 2),     width=cell_x1,         orientation=270, layer=lyr_m1)
    c.add_port("VDD", center=(cx, cell_ytop + rail_h / 2), width=cell_x1, orientation=90,  layer=lyr_m1)
    return c


# ── NAND2 ─────────────────────────────────────────────────────────────────────

def draw_nand2(
    w_n:   float,
    w_p:   float,
    l:     float,
    rules: PDKRules = RULES,
) -> "gf.Component":
    """
    CMOS NAND2: series NMOS pair + parallel PMOS pair.

    Truth table: OUT = ~(IN_A & IN_B)

    Layout::

        GND — [N_A: GND→node] — [N_B: node→OUT] — (OUT)
        VDD — [P_A: VDD→OUT ] (parallel)
        VDD — [P_B: VDD→OUT ]

    Ports — IN_A (poly), IN_B (poly), OUT (li1), VDD (met1), GND (met1).
    """
    import gdsfactory as gf
    _activate_pdk()

    ng = transistor_geom(w_n, l, "nmos", rules)
    pg = transistor_geom(w_p, l, "pmos", rules)

    dev_spacing = rules.diff["spacing_min_um"]   # diff-to-diff spacing between device pairs
    x_off       = ng.total_x_um + dev_spacing    # X offset of N_B / P_B

    gap         = _routing_gap(rules)            # Y gap (large enough for li1 track)
    pmos_y      = ng.total_y_um + gap

    nmos_c = draw_transistor(w_n, l, "nmos", rules)
    pmos_c = draw_transistor(w_p, l, "pmos", rules)

    c = gf.Component(name=_uname(f"nand2_Wn{w_n:.3f}_Wp{w_p:.3f}_L{l:.3f}"))
    na = c.add_ref(nmos_c)                 # N_A at (0,0)
    nb = c.add_ref(nmos_c); nb.movex(x_off)
    pa = c.add_ref(pmos_c); pa.movey(pmos_y)
    pb = c.add_ref(pmos_c); pb.move((x_off, pmos_y))

    lyr_g   = rules.layer("poly")
    lyr_li1 = rules.layer("li1")
    lyr_m1  = rules.layer("met1")

    nd_y0, nd_y1 = _diff_y(ng, rules)
    pd_y0, pd_y1 = _diff_y(pg, rules)
    pd_y0 += pmos_y;  pd_y1 += pmos_y   # globalise

    sd      = ng.sd_length_um
    gx0, gx1 = _gate_x(0, ng)           # gate 0 local x

    li1_sp = rules.li1["spacing_min_um"]
    li1_w  = rules.li1["width_min_um"]
    # Routing track Y (in the inter-cell gap, above NMOS li1)
    rt_y0 = nd_y1 + li1_sp
    rt_y1 = rt_y0 + li1_w

    # ── Gate A bridge (x = gx0..gx1, same column as gate 0) ─────────────────
    _rect(c, gx0,        gx1,        ng.total_y_um, pmos_y, lyr_g)
    # ── Gate B bridge (x = x_off + gx0 .. x_off + gx1) ──────────────────────
    _rect(c, x_off+gx0,  x_off+gx1,  ng.total_y_um, pmos_y, lyr_g)

    # ── Series NMOS internal node: N_A drain → N_B source ────────────────────
    # N_A drain li1 right edge = sd+l+sd (= total_x); N_B source li1 left = x_off
    # Bridge li1 from N_A drain right to N_B source left (width = dev_spacing)
    na_d_x0, na_d_x1 = _sd_x(1, ng, rules)    # N_A drain x (local = global since x_off=0)
    nb_s_x0, nb_s_x1 = _sd_x(0, ng, rules)
    nb_s_x0 += x_off; nb_s_x1 += x_off
    _rect(c, na_d_x1, nb_s_x0, nd_y0, nd_y1, lyr_li1)

    # ── OUT connection ────────────────────────────────────────────────────────
    # N_B drain (j=1 of N_B) → P_B drain (j=1 of P_B): vertical bridge at same X
    nb_d_x0, nb_d_x1 = _sd_x(1, ng, rules)
    nb_d_x0 += x_off; nb_d_x1 += x_off
    _rect(c, nb_d_x0, nb_d_x1, nd_y1, pd_y0, lyr_li1)   # vertical bridge

    # P_A drain (j=1 of P_A) → routing track → vertical bridge
    pa_d_x0, pa_d_x1 = _sd_x(1, pg, rules)  # local = global (P_A at x_off=0)
    # Routing track: horizontal li1 from P_A drain x to vertical bridge
    _rect(c, pa_d_x0, nb_d_x1, rt_y0, rt_y1, lyr_li1)
    # Vertical connector: P_A drain bottom (pd_y0) down to routing track top (rt_y1)
    _rect(c, pa_d_x0, pa_d_x1, rt_y1, pd_y0, lyr_li1)

    # ── Met1 power rails ──────────────────────────────────────────────────────
    rail_h    = 0.17
    cell_x1   = x_off + ng.total_x_um
    cell_ytop = pmos_y + pg.total_y_um
    _rect(c, 0, cell_x1, -rail_h, 0,               lyr_m1)  # GND
    _rect(c, 0, cell_x1, cell_ytop, cell_ytop+rail_h, lyr_m1)  # VDD

    # ── Ports ─────────────────────────────────────────────────────────────────
    gap_mid_y = (ng.total_y_um + pmos_y) / 2
    out_mid_y = (nd_y1 + pd_y0) / 2
    cx = cell_x1 / 2
    c.add_port("IN_A", center=(gx0,       gap_mid_y), width=gap, orientation=180, layer=lyr_g)
    c.add_port("IN_B", center=(x_off+gx0, gap_mid_y), width=gap, orientation=180, layer=lyr_g)
    c.add_port("OUT",  center=(cell_x1,   out_mid_y), width=max(pd_y0-nd_y1, l), orientation=0, layer=lyr_li1)
    c.add_port("GND",  center=(cx, -rail_h/2),         width=cell_x1, orientation=270, layer=lyr_m1)
    c.add_port("VDD",  center=(cx, cell_ytop+rail_h/2), width=cell_x1, orientation=90,  layer=lyr_m1)
    return c


# ── NOR2 ──────────────────────────────────────────────────────────────────────

def draw_nor2(
    w_n:   float,
    w_p:   float,
    l:     float,
    rules: PDKRules = RULES,
) -> "gf.Component":
    """
    CMOS NOR2: parallel NMOS pair + series PMOS pair.

    Truth table: OUT = ~(IN_A | IN_B)

    Layout::

        GND — [N_A: GND→OUT ] (parallel)
        GND — [N_B: GND→OUT ]
        VDD — [P_A: VDD→node] — [P_B: node→OUT] — (OUT)

    Ports — IN_A (poly), IN_B (poly), OUT (li1), VDD (met1), GND (met1).
    """
    import gdsfactory as gf
    _activate_pdk()

    ng = transistor_geom(w_n, l, "nmos", rules)
    pg = transistor_geom(w_p, l, "pmos", rules)

    dev_spacing = rules.diff["spacing_min_um"]
    x_off       = ng.total_x_um + dev_spacing

    gap         = _routing_gap(rules)
    pmos_y      = ng.total_y_um + gap

    nmos_c = draw_transistor(w_n, l, "nmos", rules)
    pmos_c = draw_transistor(w_p, l, "pmos", rules)

    c = gf.Component(name=_uname(f"nor2_Wn{w_n:.3f}_Wp{w_p:.3f}_L{l:.3f}"))
    na = c.add_ref(nmos_c)
    nb = c.add_ref(nmos_c); nb.movex(x_off)
    pa = c.add_ref(pmos_c); pa.movey(pmos_y)
    pb = c.add_ref(pmos_c); pb.move((x_off, pmos_y))

    lyr_g   = rules.layer("poly")
    lyr_li1 = rules.layer("li1")
    lyr_m1  = rules.layer("met1")

    nd_y0, nd_y1 = _diff_y(ng, rules)
    pd_y0, pd_y1 = _diff_y(pg, rules)
    pd_y0 += pmos_y;  pd_y1 += pmos_y

    sd      = ng.sd_length_um
    gx0, gx1 = _gate_x(0, ng)

    li1_sp = rules.li1["spacing_min_um"]
    li1_w  = rules.li1["width_min_um"]
    # Routing track: above NMOS li1, used to connect N_A drain to OUT
    rt_y0 = nd_y1 + li1_sp
    rt_y1 = rt_y0 + li1_w

    # ── Gate A and B bridges ──────────────────────────────────────────────────
    _rect(c, gx0,       gx1,       ng.total_y_um, pmos_y, lyr_g)
    _rect(c, x_off+gx0, x_off+gx1, ng.total_y_um, pmos_y, lyr_g)

    # ── Series PMOS internal node: P_A drain → P_B source ────────────────────
    # P_A drain (j=1) right edge = pa_d_x1 (global = local since x_off=0 for P_A)
    # P_B source (j=0) left edge = x_off
    pa_d_x0, pa_d_x1 = _sd_x(1, pg, rules)
    pb_s_x0, pb_s_x1 = _sd_x(0, pg, rules)
    pb_s_x0 += x_off; pb_s_x1 += x_off
    _rect(c, pa_d_x1, pb_s_x0, pd_y0, pd_y1, lyr_li1)

    # ── OUT connection ────────────────────────────────────────────────────────
    # N_B drain (j=1 of N_B) and P_B drain (j=1 of P_B) share the same X column
    nb_d_x0, nb_d_x1 = _sd_x(1, ng, rules)
    nb_d_x0 += x_off; nb_d_x1 += x_off
    # Vertical bridge N_B drain → P_B drain
    _rect(c, nb_d_x0, nb_d_x1, nd_y1, pd_y0, lyr_li1)

    # N_A drain (j=1 of N_A) → routing track → merges into vertical bridge
    na_d_x0, na_d_x1 = _sd_x(1, ng, rules)
    # Routing track: horizontal li1 from N_A drain to vertical bridge
    _rect(c, na_d_x0, nb_d_x1, rt_y0, rt_y1, lyr_li1)
    # Vertical connector: N_A drain top (nd_y1) → routing track bottom (rt_y0)
    _rect(c, na_d_x0, na_d_x1, nd_y1, rt_y0, lyr_li1)

    # ── Met1 power rails ──────────────────────────────────────────────────────
    rail_h    = 0.17
    cell_x1   = x_off + ng.total_x_um
    cell_ytop = pmos_y + pg.total_y_um
    _rect(c, 0, cell_x1, -rail_h, 0,                lyr_m1)
    _rect(c, 0, cell_x1, cell_ytop, cell_ytop+rail_h, lyr_m1)

    # ── Ports ─────────────────────────────────────────────────────────────────
    gap_mid_y = (ng.total_y_um + pmos_y) / 2
    out_mid_y = (nd_y1 + pd_y0) / 2
    cx = cell_x1 / 2
    c.add_port("IN_A", center=(gx0,       gap_mid_y), width=gap, orientation=180, layer=lyr_g)
    c.add_port("IN_B", center=(x_off+gx0, gap_mid_y), width=gap, orientation=180, layer=lyr_g)
    c.add_port("OUT",  center=(cell_x1,   out_mid_y), width=max(pd_y0-nd_y1, l), orientation=0, layer=lyr_li1)
    c.add_port("GND",  center=(cx, -rail_h/2),          width=cell_x1, orientation=270, layer=lyr_m1)
    c.add_port("VDD",  center=(cx, cell_ytop+rail_h/2),  width=cell_x1, orientation=90,  layer=lyr_m1)
    return c
