"""
layout_gen.cells.vias — via/contact cell generators.

Each function returns a gdsfactory Component representing one atomic
contact or via stack.  All dimensions are derived from PDK rules.

Using cells (instead of raw polygons) lets the geo fixer identify and
move entire via stacks as atomic units without breaking internal alignment.

Via types
---------
licon_poly  — poly contact: poly pad + licon1 + li1
licon_diff  — diff contact: licon1 + li1
mcon_stack  — li1 → met1: mcon + met1 pad
via1_stack  — met1 → met2: via1 + met1 pad + met2 pad

Composite stacks
----------------
poly_contact_to_met1  — poly pad + licon1 + li1 + mcon + met1
poly_contact_to_met2  — poly pad + licon1 + li1 + mcon + met1 + via1 + met2
diff_contact_to_met1  — licon1 + li1 + mcon + met1
li1_to_met2           — mcon + met1 + via1 + met2
"""
from __future__ import annotations

import itertools
from typing import Any

from layout_gen.pdk import PDKRules

_counter = itertools.count()


def _uname(base: str) -> str:
    return f"{base}_{next(_counter)}"


def _activate_pdk():
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except ValueError:
        from gdsfactory.generic_tech import PDK as _GENERIC
        _GENERIC.activate()


def _rect(comp: Any, x0: float, x1: float, y0: float, y1: float, layer: Any):
    comp.add_polygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        layer=layer,
    )


# ── Primitive via cells ────────────────────────────────────────────────────────

def licon_poly(rules: PDKRules) -> Any:
    """Poly contact: poly enclosure + licon1 + li1 pad.

    Centred at (0, 0).  The poly pad provides the required enclosure
    of the licon, and the li1 pad provides li1 enclosure.
    """
    _activate_pdk()
    import gdsfactory as gf

    c_size = rules.contacts["size_um"]
    ch = c_size / 2

    # Poly enclosure of licon (asymmetric: 2adj > all-sides)
    poly_enc = rules.contacts.get("poly_enclosure_um", 0.05)
    poly_enc_2adj = rules.contacts.get("poly_enclosure_2adj_um", 0.08)

    # Li1 enclosure of licon
    li1_enc = rules.contacts.get("enclosure_in_li1_um", 0.0)
    li1_enc_2adj = rules.contacts.get("enclosure_in_li1_2adj_um", 0.08)

    lyr_poly = rules.layer("poly")
    lyr_licon = rules.layer("licon1")
    lyr_li1 = rules.layer("li1")

    comp = gf.Component(_uname("licon_poly"))

    # Poly pad (2adj enclosure on Y, min enclosure on X)
    _rect(comp, -ch - poly_enc, ch + poly_enc,
                -ch - poly_enc_2adj, ch + poly_enc_2adj, lyr_poly)

    # Licon1
    _rect(comp, -ch, ch, -ch, ch, lyr_licon)

    # Li1 pad (2adj enclosure on X — wider along gate, opp enclosure on Y)
    li1_hx = max(ch + li1_enc_2adj, rules.li1.get("width_min_um", c_size) / 2)
    li1_hy = max(ch + li1_enc, rules.li1.get("width_min_um", c_size) / 2)
    _rect(comp, -li1_hx, li1_hx, -li1_hy, li1_hy, lyr_li1)

    return comp


def licon_diff(rules: PDKRules) -> Any:
    """Diff contact: licon1 + li1 pad (no diff drawn — that's the transistor's job).

    Centred at (0, 0).
    """
    _activate_pdk()
    import gdsfactory as gf

    c_size = rules.contacts["size_um"]
    ch = c_size / 2

    li1_enc = rules.contacts.get("enclosure_in_li1_um", 0.0)
    li1_enc_2adj = rules.contacts.get("enclosure_in_li1_2adj_um", 0.08)

    lyr_licon = rules.layer("licon1")
    lyr_li1 = rules.layer("li1")

    comp = gf.Component(_uname("licon_diff"))

    # Licon1
    _rect(comp, -ch, ch, -ch, ch, lyr_licon)

    # Li1 pad
    li1_hx = max(ch + li1_enc, rules.li1.get("width_min_um", c_size) / 2)
    li1_hy = max(ch + li1_enc_2adj, rules.li1.get("width_min_um", c_size) / 2)
    _rect(comp, -li1_hx, li1_hx, -li1_hy, li1_hy, lyr_li1)

    return comp


def mcon_stack(rules: PDKRules) -> Any:
    """Mcon via: mcon + met1 pad.  Centred at (0, 0).

    When li1 and met1 are the same GDS layer (e.g. GF180MCU) mcon is a
    no-op — only the met1 landing pad is drawn, since no physical via
    connects li1 to met1.
    """
    _activate_pdk()
    import gdsfactory as gf

    mcon_rules = rules.mcon or {}
    mcon_sz = mcon_rules.get("size_um", rules.contacts["size_um"])
    mch = mcon_sz / 2

    met1_enc = (rules.met1 or {}).get("enclosure_of_mcon_2adj_um",
                mcon_rules.get("enclosure_in_met1_um", 0.03))
    m1h = mch + met1_enc

    lyr_met1 = rules.layer("met1")

    comp = gf.Component(_uname("mcon_stack"))

    if not rules.li1_is_met1:
        lyr_mcon = rules.layer("mcon")
        _rect(comp, -mch, mch, -mch, mch, lyr_mcon)
    _rect(comp, -m1h, m1h, -m1h, m1h, lyr_met1)

    return comp


def via1_stack(rules: PDKRules) -> Any:
    """Via1: via1 + met1 pad + met2 pad.  Centred at (0, 0)."""
    _activate_pdk()
    import gdsfactory as gf

    via1_rules = rules.via1 or {}
    via_sz = via1_rules.get("size_um", 0.15)
    vh = via_sz / 2

    m1_enc = via1_rules.get("enclosure_in_met1_2adj_um",
             via1_rules.get("enclosure_in_met1_um", 0.055))
    m2_enc = (rules.met2 or {}).get("enclosure_of_via1_2adj_um",
              via1_rules.get("enclosure_in_met2_um", 0.055))
    m1h = vh + m1_enc
    m2h = vh + m2_enc

    lyr_via1 = rules.layer("via1")
    lyr_met1 = rules.layer("met1")
    lyr_met2 = rules.layer("met2")

    comp = gf.Component(_uname("via1_stack"))

    _rect(comp, -vh, vh, -vh, vh, lyr_via1)
    _rect(comp, -m1h, m1h, -m1h, m1h, lyr_met1)
    _rect(comp, -m2h, m2h, -m2h, m2h, lyr_met2)

    return comp


# ── Composite via stacks ──────────────────────────────────────────────────────

def poly_contact_to_met1(rules: PDKRules) -> Any:
    """Full poly contact stack: poly pad + licon1 + li1 + mcon + met1.

    Centred at (0, 0).  Used for WL gate connections.
    """
    _activate_pdk()
    import gdsfactory as gf

    comp = gf.Component(_uname("poly_contact_to_met1"))

    # Add licon_poly (poly pad + licon + li1)
    lp = licon_poly(rules)
    comp.add_ref(lp)

    # Add mcon stack (mcon + met1)
    mc = mcon_stack(rules)
    comp.add_ref(mc)

    return comp


def poly_contact_to_met2(rules: PDKRules) -> Any:
    """Full poly contact stack up to met2.

    Centred at (0, 0).  Used for cross-couple gate connections.
    """
    _activate_pdk()
    import gdsfactory as gf

    comp = gf.Component(_uname("poly_contact_to_met2"))

    comp.add_ref(licon_poly(rules))
    comp.add_ref(mcon_stack(rules))
    comp.add_ref(via1_stack(rules))

    return comp


def li1_to_met2(rules: PDKRules) -> Any:
    """Li1 to met2 via stack: mcon + met1 + via1 + met2.

    Centred at (0, 0).  Used for bitline and signal routing.
    """
    _activate_pdk()
    import gdsfactory as gf

    comp = gf.Component(_uname("li1_to_met2"))

    comp.add_ref(mcon_stack(rules))
    comp.add_ref(via1_stack(rules))

    return comp
