"""
layout_gen.cells.tap — standalone well/substrate tap cell generator.

Generates a tap cell containing P+ substrate tap (VSS) and N+ nwell tap (VDD).
Designed to tile alongside logic cells at regular intervals for latch-up
prevention and well biasing.

All dimensions are derived from PDK rules — nothing is hardcoded.

Usage::

    from layout_gen.cells.tap import draw_tap_cell
    from layout_gen.pdk import load_pdk

    rules = load_pdk()
    comp = draw_tap_cell(cell_height=2.72, rules=rules)
"""
from __future__ import annotations

import pathlib
from typing import Any

from layout_gen.pdk import PDKRules, RULES


# ── YAML loader ──────────────────────────────────────────────────────────────

_TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "templates" / "cells" / "tap_cell.yaml"


def _load_tap_template() -> dict:
    import yaml
    with open(_TEMPLATE) as f:
        return yaml.safe_load(f)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _rect(comp: Any, x0: float, x1: float, y0: float, y1: float, layer: Any):
    comp.add_polygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        layer=layer,
    )


# ── Main generator ──────────────────────────────────────────────────────────

def draw_tap_cell(
    cell_height: float | None = None,
    rules: PDKRules = RULES,
) -> Any:
    """Generate a standalone tap cell with substrate and nwell taps.

    Parameters
    ----------
    cell_height :
        Total cell height in µm.  If ``None``, computed from minimum
        tap geometry + spacing requirements.
    rules :
        PDK rules (defaults to sky130A).

    Returns
    -------
    gdsfactory.Component
        The generated tap cell.
    """
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except ValueError:
        from gdsfactory.generic_tech import PDK as _GENERIC
        _GENERIC.activate()

    template = _load_tap_template()
    comp = gf.Component("tap_cell")

    # ── Extract PDK rules ────────────────────────────────────────────────
    c_size   = rules.contacts["size_um"]
    c_space  = rules.contacts["spacing_um"]
    impl_enc = rules.implant.get("enclosure_of_diff_um", 0.125)

    tap_rules = rules.tap if rules.tap else {}
    tap_enc      = tap_rules.get("enclosure_of_licon_um", c_size * 0.7)
    tap_to_diff  = tap_rules.get("spacing_to_diff_um", rules.diff["spacing_min_um"])

    mcon_sz  = (rules.mcon or {}).get("size_um", c_size)
    met1_enc = (rules.met1 or {}).get("enclosure_of_mcon_2adj_um",
               (rules.mcon or {}).get("enclosure_in_met1_um", 0.0))

    enc_li_2adj = rules.contacts.get("enclosure_in_li1_2adj_um",
                  rules.contacts.get("enclosure_in_li1_um", 0.0))

    nw_enc   = rules.nwell.get("enclosure_of_pdiff_um", 0.18)
    nw_min_w = rules.nwell.get("width_min_um", 0.84)

    rail_h = rules.met1["width_min_um"]

    # Tap diffusion size (square, enclosing one licon)
    tap_w = c_size + 2 * tap_enc
    tap_h = tap_w

    # Li1 pad half-width (must enclose licon per li.5)
    li_hw = max(c_size / 2 + enc_li_2adj,
                rules.li1.get("width_min_um", c_size) / 2)

    # ── Resolve cell height ──────────────────────────────────────────────
    # Minimum height: two taps stacked with spacing between them
    min_height = 2 * (tap_h + impl_enc) + tap_to_diff
    if cell_height is None:
        cell_height = max(min_height, nw_min_w + tap_h + tap_to_diff)

    # ── Layer lookups ────────────────────────────────────────────────────
    stack = template.get("stack", {})
    lyr_tap   = rules.layer(stack.get("diffusion", "tap"))
    lyr_licon = rules.layer(stack.get("contact", "licon1"))
    lyr_li1   = rules.layer(stack.get("li", "li1"))
    lyr_met1  = rules.layer(stack.get("metal", "met1"))

    lyr_mcon = None
    try:
        lyr_mcon = rules.layer(stack.get("via", "mcon"))
    except KeyError:
        pass

    lyr_nsdm  = rules.layer("nsdm")
    lyr_psdm  = rules.layer("psdm")

    lyr_nwell = None
    try:
        lyr_nwell = rules.layer("nwell")
    except KeyError:
        pass

    # ── Cell width: enough for one tap + implant enclosure ───────────────
    cell_w = max(tap_w + 2 * impl_enc, nw_min_w)

    cx = cell_w / 2  # centre X for all taps

    # ── Helper: draw one complete tap contact stack ──────────────────────
    def _draw_tap(tap_cx: float, tap_cy: float, implant_lyr: Any) -> None:
        half = tap_w / 2
        ch   = c_size / 2
        mch  = mcon_sz / 2

        # 1. Tap diffusion
        _rect(comp, tap_cx - half, tap_cx + half,
              tap_cy - half, tap_cy + half, lyr_tap)

        # 2. Licon contact (centred)
        _rect(comp, tap_cx - ch, tap_cx + ch,
              tap_cy - ch, tap_cy + ch, lyr_licon)

        # 3. Li1 pad
        _rect(comp, tap_cx - li_hw, tap_cx + li_hw,
              tap_cy - li_hw, tap_cy + li_hw, lyr_li1)

        # 4. Mcon (li1 → met1)
        if lyr_mcon is not None:
            _rect(comp, tap_cx - mch, tap_cx + mch,
                  tap_cy - mch, tap_cy + mch, lyr_mcon)

        # 5. Met1 pad
        if lyr_mcon is not None:
            met1_hw = mch + met1_enc
            _rect(comp, tap_cx - met1_hw, tap_cx + met1_hw,
                  tap_cy - met1_hw, tap_cy + met1_hw, lyr_met1)

        # 6. Implant enclosure
        _rect(comp, tap_cx - half - impl_enc, tap_cx + half + impl_enc,
              tap_cy - half - impl_enc, tap_cy + half + impl_enc,
              implant_lyr)

    # ── P+ substrate tap (bottom, VSS bias) ──────────────────────────────
    ptap_cy = rail_h + tap_h / 2
    _draw_tap(cx, ptap_cy, lyr_psdm)

    # Met1 bottom rail (GND)
    _rect(comp, 0, cell_w, 0, rail_h, lyr_met1)

    # ── N+ nwell tap (top, VDD bias) ────────────────────────────────────
    ntap_cy = cell_height - rail_h - tap_h / 2
    _draw_tap(cx, ntap_cy, lyr_nsdm)

    # Met1 top rail (VDD)
    _rect(comp, 0, cell_w, cell_height - rail_h, cell_height, lyr_met1)

    # ── Nwell (enclose ntap + nsdm implant) ─────────────────────────────
    if lyr_nwell is not None:
        nw_y0 = ntap_cy - tap_h / 2 - impl_enc - nw_enc
        nw_y1 = cell_height
        nw_x0 = cx - max(tap_w / 2 + impl_enc + nw_enc, nw_min_w / 2)
        nw_x1 = cx + max(tap_w / 2 + impl_enc + nw_enc, nw_min_w / 2)
        # Enforce minimum nwell width in Y
        if nw_y1 - nw_y0 < nw_min_w:
            nw_y0 = nw_y1 - nw_min_w
        _rect(comp, nw_x0, nw_x1, nw_y0, nw_y1, lyr_nwell)

    # ── Ports ────────────────────────────────────────────────────────────
    comp.add_port(
        name="GND", center=(cell_w / 2, rail_h / 2),
        width=cell_w, orientation=270, layer=lyr_met1,
    )
    comp.add_port(
        name="VDD", center=(cell_w / 2, cell_height - rail_h / 2),
        width=cell_w, orientation=90, layer=lyr_met1,
    )

    return comp
