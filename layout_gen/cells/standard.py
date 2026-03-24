"""
layout_gen.cells.standard — geometry helpers for CMOS standard cells.

Shared helper functions used by the synthesizer, placer, and router
for computing transistor geometry extents and cell layout primitives.

All dimensions in µm, all rules from the PDK YAML.
"""
from __future__ import annotations

from layout_gen.pdk        import PDKRules
from layout_gen.transistor import TransistorGeom


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _sd_x(
    j: int,
    geom: TransistorGeom,
    rules: PDKRules | None = None,
) -> tuple[float, float]:
    """(x0, x1) of the j-th source/drain li1 region in local transistor coords.

    When *rules* is provided, returns the licon-width li1 strip extent
    (zero X enclosure, matching sky130 reference transistor geometry).
    """
    cx = j * (geom.sd_length_um + geom.l_um) + geom.sd_length_um / 2

    if rules is not None:
        c_half = rules.contacts["size_um"] / 2
        return cx - c_half, cx + c_half

    # Without rules, return full S/D region (legacy fallback)
    x0 = j * (geom.sd_length_um + geom.l_um)
    return x0, x0 + geom.sd_length_um


def _gate_x(i: int, geom: TransistorGeom) -> tuple[float, float]:
    """(x0, x1) of the i-th poly gate finger in local transistor coords."""
    x0 = (i + 1) * geom.sd_length_um + i * geom.l_um
    return x0, x0 + geom.l_um


def _diff_y(geom: TransistorGeom, rules: PDKRules) -> tuple[float, float]:
    """(y0, y1) of diffusion in local transistor Y coords.

    Diff is contained within the poly gate in Y — poly overhangs diff
    by ``endcap_over_diff_um`` on each side (poly.8).
    ``extension_past_poly_um`` applies in X only (S/D regions).
    """
    endcap = rules.poly["endcap_over_diff_um"]
    return endcap, endcap + geom.w_finger_um


def _inter_cell_gap(rules: PDKRules) -> float:
    """
    Minimum Y gap between NMOS poly top edge and PMOS poly bottom edge
    such that the diff-to-diff spacing rule is satisfied.
    """
    endcap  = rules.poly["endcap_over_diff_um"]
    min_sep = rules.diff["spacing_min_um"]
    # With gap G: separation = G + 2*endcap >= min_sep
    return max(0.0, min_sep - 2 * endcap)


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


def _snap(value: float, grid: float = 0.005) -> float:
    """Snap *value* to nearest manufacturing grid point (default 5 nm)."""
    if grid <= 0:
        return value
    return round(round(value / grid) * grid, 6)


def _rect(c, x0: float, x1: float, y0: float, y1: float, layer,
           snap_grid: float = 0.005) -> None:
    """Add a rectangle polygon to component c, snapped to mfg grid."""
    x0, x1 = _snap(x0, snap_grid), _snap(x1, snap_grid)
    y0, y1 = _snap(y0, snap_grid), _snap(y1, snap_grid)
    c.add_polygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        layer=layer,
    )

