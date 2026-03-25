"""
layout_gen.transistor — single-transistor layout primitive.

Given a channel width W, channel length L, and a logical device type
(``"nmos"`` or ``"pmos"``), this module computes the optimal finger count,
draws every physical layer (diff, poly, contacts, li1, implant, nwell),
and returns a gdsfactory Component with named ports at the li1 level.

Layer names and design rules come entirely from the PDK YAML — no numbers
are hardcoded here.

Orientation convention
----------------------
- X axis : channel length (L) direction — poly fingers run vertically
- Y axis : channel width  (W) direction — diffusion runs horizontally

Multi-finger layout (n=2 shown, shared source/drain)::

    ←  sd  →← L →←  sd  →← L →←  sd  →
    ┌───────┐┌────┐┌───────┐┌────┐┌───────┐   ─ poly_endcap
    │  S/D  ││poly││  S/D  ││poly││  S/D  │
    │ (li1) ││    ││ (li1) ││    ││ (li1) │
    └───────┘└────┘└───────┘└────┘└───────┘   ─ poly_endcap
              ↑                ↑
              G0               G1

Ports returned (at li1 level, centred on each terminal):
  G   — gate  (on poly top edge, mid-X of each finger; exported as one port at
               the first finger for single-net gate connection)
  D   — drain (right-most S/D rail for finger 0 = drain; li1 rect)
  S   — source (left-most S/D rail; li1 rect)
  B   — bulk tap placeholder (no geometry here; caller handles well tap)

Typical use::

    from layout_gen.transistor import draw_transistor, finger_count
    from layout_gen.pdk import load_pdk

    rules = load_pdk()                              # sky130A default
    comp  = draw_transistor(0.52, 0.15, "nmos", rules)
    comp.write_gds("nmos_0p52.gds")

    n = finger_count(8.0, rules, "nmos")            # → 4 fingers
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from layout_gen.pdk import PDKRules, RULES


# ── Geometry dataclass ────────────────────────────────────────────────────────

@dataclass
class TransistorGeom:
    """Computed transistor geometry — all dimensions in µm.

    Produced by :func:`transistor_geom`; consumed by :func:`draw_transistor`.

    Attributes
    ----------
    w_um, l_um :
        Drawn channel width and length.
    device_type :
        Logical device name (``"nmos"`` / ``"pmos"``).
    n_fingers :
        Number of parallel gate fingers.
    w_finger_um :
        Width of each individual gate finger = ``w_um / n_fingers``.
    sd_length_um :
        Length of each source/drain contact region in X direction.
    n_contacts_y :
        Number of contact rows in the Y (width) direction per S/D column.
    total_x_um :
        Total device width in X = ``(n_fingers+1)*sd_length_um + n_fingers*l_um``.
    total_y_um :
        Total device height in Y = ``w_finger_um + 2*poly.endcap_over_diff_um``.
    """
    w_um:          float
    l_um:          float
    device_type:   str
    n_fingers:     int
    w_finger_um:   float
    sd_length_um:  float
    n_contacts_y:  int
    total_x_um:    float
    total_y_um:    float


# ── Public helpers ─────────────────────────────────────────────────────────────

def _min_channel_width(rules: PDKRules, device_type: str) -> float:
    """Return the PDK minimum channel width (µm) for *device_type*."""
    dev = rules.device(device_type)
    # Prefer explicit channel_width_min; fall back to diff width_min
    w = dev.get("channel_width_min_um", 0.0)
    if w > 0:
        return w
    return rules.diff.get("width_min_um", 0.0)


def finger_count(w_um: float, rules: PDKRules, device_type: str = "nmos") -> int:
    """Return the minimum number of gate fingers for channel width *w_um*.

    Fingers are added when ``w_um`` exceeds ``w_finger_max_um`` from the PDK
    device rules.  Always returns at least 1.

    Parameters
    ----------
    w_um :
        Total drawn channel width (µm).
    rules :
        PDK rules object from :func:`~layout_gen.pdk.load_pdk`.
    device_type :
        Logical device type (``"nmos"`` or ``"pmos"``).
    """
    w_max = rules.device(device_type).get("w_finger_max_um", 2.0)
    n = max(1, math.ceil(w_um / w_max))
    # Clamp so w_finger never drops below minimum channel width
    w_min = _min_channel_width(rules, device_type)
    if w_min > 0:
        n = min(n, max(1, int(w_um / w_min)))
    return n


def transistor_geom(
    w_um:        float,
    l_um:        float,
    device_type: str,
    rules:       PDKRules = RULES,
) -> TransistorGeom:
    """Compute all transistor geometry parameters from PDK rules.

    Parameters
    ----------
    w_um, l_um :
        Drawn channel width and length (µm).
    device_type :
        Logical device type (``"nmos"`` or ``"pmos"``).
    rules :
        PDK rules.  Defaults to the module-level sky130A rules.

    Returns
    -------
    TransistorGeom
        Fully computed geometry — pass to :func:`draw_transistor`.
    """
    dev    = rules.device(device_type)
    n      = finger_count(w_um, rules, device_type)
    w_f    = w_um / n

    # S/D contact region: long enough for at least one contact plus enclosure
    c_size = rules.contacts["size_um"]
    c_enc  = rules.contacts["enclosure_in_diff_um"]
    sd     = max(dev.get("sd_length_min_um", 0.29), c_size + 2 * c_enc)

    n_cy = rules.sd_contact_columns(w_f)

    endcap  = rules.poly["endcap_over_diff_um"]
    total_x = (n + 1) * sd + n * l_um
    total_y = w_f + 2 * endcap

    return TransistorGeom(
        w_um         = w_um,
        l_um         = l_um,
        device_type  = device_type,
        n_fingers    = n,
        w_finger_um  = w_f,
        sd_length_um = sd,
        n_contacts_y = n_cy,
        total_x_um   = total_x,
        total_y_um   = total_y,
    )


# ── Drawing ────────────────────────────────────────────────────────────────────

_CELL_COUNTER: dict[str, int] = {}   # tracks per-base-name instance count

def draw_transistor(
    w_um:        float,
    l_um:        float,
    device_type: str,
    rules:       PDKRules = RULES,
    *,
    n_fingers:   int | None = None,
    skip_sd:     set[int] | None = None,
) -> "gf.Component":
    """Draw a single transistor and return a gdsfactory Component.

    The component origin (0, 0) is at the lower-left corner of the poly
    bounding box.  All coordinates are in µm.

    Parameters
    ----------
    w_um, l_um :
        Drawn channel width and length (µm).
    device_type :
        Logical device type — must match a key in the PDK YAML ``devices``
        section (e.g. ``"nmos"``, ``"pmos"``).
    rules :
        PDK rules.  Defaults to the module-level sky130A rules.

    Returns
    -------
    gf.Component
        Component with ports ``G``, ``D``, ``S`` at the li1 level.
    """
    import gdsfactory as gf

    # Ensure a PDK is active so layer tuples resolve correctly.
    # The generic PDK accepts arbitrary (layer, datatype) integer pairs.
    try:
        gf.get_active_pdk()
    except ValueError:
        from gdsfactory.generic_tech import PDK as _GENERIC
        _GENERIC.activate()

    geom = transistor_geom(w_um, l_um, device_type, rules)
    if n_fingers is not None and n_fingers != geom.n_fingers:
        from dataclasses import replace as _replace
        n   = max(1, int(n_fingers))
        w_f = w_um / n
        geom = _replace(
            geom,
            n_fingers    = n,
            w_finger_um  = w_f,
            total_x_um   = (n + 1) * geom.sd_length_um + n * l_um,
            total_y_um   = w_f + 2 * rules.poly["endcap_over_diff_um"],
            n_contacts_y = rules.sd_contact_columns(w_f),
        )
    dev  = rules.device(device_type)

    # Unique cell name so repeated calls in the same Python session don't clash
    # in gdsfactory's global layout library (kfactory raises on duplicate names).
    _base = f"{device_type}_W{w_um:.3f}_L{l_um:.3f}_f{geom.n_fingers}"
    _CELL_COUNTER[_base] = _CELL_COUNTER.get(_base, 0) + 1
    _n = _CELL_COUNTER[_base]
    _name = _base if _n == 1 else f"{_base}${_n}"
    c = gf.Component(name=_name)

    endcap  = rules.poly["endcap_over_diff_um"]
    ext_y   = rules.diff["extension_past_poly_um"]
    c_size  = rules.contacts["size_um"]
    c_space = rules.contacts["spacing_um"]
    c_enc   = rules.contacts["enclosure_in_diff_um"]
    stp     = rules.contacts["space_to_poly_um"]
    li_w    = rules.li1["width_min_um"]

    # Layer tuples
    lyr_diff    = rules.layer(dev["diff_layer"])
    lyr_gate    = rules.layer(dev["gate_layer"])
    lyr_contact = rules.layer("licon1")
    lyr_li1     = rules.layer("li1")
    lyr_implant = rules.layer(dev["implant_layer"])
    lyr_bulk    = rules.layer(dev["bulk_layer"])

    # ── Diffusion rectangle (covers all fingers) ──────────────────────────────
    # Y: diff spans the channel width, contained within poly so poly overhangs
    # by endcap (poly.8).  extension_past_poly is X-only (S/D extends past gate).
    diff_y0 = endcap
    diff_y1 = endcap + geom.w_finger_um
    diff_x0 = 0.0
    diff_x1 = geom.total_x_um
    c.add_polygon(
        [(diff_x0, diff_y0), (diff_x1, diff_y0),
         (diff_x1, diff_y1), (diff_x0, diff_y1)],
        layer=lyr_diff,
    )

    # ── Implant (S/D select — encloses diff) ─────────────────────────────────
    impl_enc = rules.implant["enclosure_of_diff_um"]
    c.add_polygon(
        [(diff_x0 - impl_enc, diff_y0 - impl_enc),
         (diff_x1 + impl_enc, diff_y0 - impl_enc),
         (diff_x1 + impl_enc, diff_y1 + impl_enc),
         (diff_x0 - impl_enc, diff_y1 + impl_enc)],
        layer=lyr_implant,
    )

    # ── N-well (PMOS only — encloses diff with PDK enclosure rule) ───────────
    if dev.get("nwell", False):
        nw_enc = rules.nwell["enclosure_of_pdiff_um"]
        c.add_polygon(
            [(diff_x0 - nw_enc, diff_y0 - nw_enc),
             (diff_x1 + nw_enc, diff_y0 - nw_enc),
             (diff_x1 + nw_enc, diff_y1 + nw_enc),
             (diff_x0 - nw_enc, diff_y1 + nw_enc)],
            layer=rules.layer("nwell"),
        )

    # ── Poly gate fingers ─────────────────────────────────────────────────────
    gate_port_x: list[float] = []
    for i in range(geom.n_fingers):
        gx0 = (i + 1) * geom.sd_length_um + i * geom.l_um
        gx1 = gx0 + geom.l_um
        c.add_polygon(
            [(gx0, 0.0), (gx1, 0.0), (gx1, geom.total_y_um), (gx0, geom.total_y_um)],
            layer=lyr_gate,
        )
        gate_port_x.append((gx0 + gx1) / 2)

    # ── NPC (Nitride Poly Cut) on poly endcaps ────────────────────────────────
    # Prevents silicide on poly stubs extending beyond diffusion.
    # Only generated if the PDK defines an 'npc' layer.
    try:
        lyr_npc = rules.layer("npc")
        npc_rules = rules.npc if rules.npc else {}
        npc_enc = npc_rules.get("enclosure_of_poly_um", 0.10)
        for i in range(geom.n_fingers):
            gx0 = (i + 1) * geom.sd_length_um + i * geom.l_um
            gx1 = gx0 + geom.l_um
            # Bottom endcap stub: poly from y=0 to diff_y0
            c.add_polygon(
                [(gx0 - npc_enc, 0.0 - npc_enc),
                 (gx1 + npc_enc, 0.0 - npc_enc),
                 (gx1 + npc_enc, diff_y0 + npc_enc),
                 (gx0 - npc_enc, diff_y0 + npc_enc)],
                layer=lyr_npc,
            )
            # Top endcap stub: poly from diff_y1 to total_y_um
            c.add_polygon(
                [(gx0 - npc_enc, diff_y1 - npc_enc),
                 (gx1 + npc_enc, diff_y1 - npc_enc),
                 (gx1 + npc_enc, geom.total_y_um + npc_enc),
                 (gx0 - npc_enc, geom.total_y_um + npc_enc)],
                layer=lyr_npc,
            )
    except KeyError:
        pass  # PDK doesn't define NPC — skip

    # ── Contacts + li1 rails per S/D region ──────────────────────────────────
    # Contacts are centred in Y within the diff (respecting enclosure).
    # Spacing them evenly: first contact at diff_y0 + c_enc + c_size/2
    n_cy  = geom.n_contacts_y
    c_mid = (diff_y0 + diff_y1) / 2

    # Y positions of contact centres (symmetric about mid)
    total_span = n_cy * c_size + (n_cy - 1) * c_space
    cy_start   = c_mid - total_span / 2 + c_size / 2
    c_y_centres = [cy_start + k * (c_size + c_space) for k in range(n_cy)]

    # S/D X centres (n_fingers+1 regions, alternating drain/source for finger 0)
    sd_x_centres: list[float] = []
    for j in range(geom.n_fingers + 1):
        cx = j * (geom.sd_length_um + geom.l_um) + geom.sd_length_um / 2
        sd_x_centres.append(cx)

    # Contact X centre within each S/D (single column, centred)
    # Guard: ensure contact clears poly by stp on both sides
    contact_x_offset = 0.0  # centred in sd region

    drain_li1_x0, source_li1_x0 = None, None

    _skip = skip_sd or set()

    for j, cx in enumerate(sd_x_centres):
        is_drain = (j % 2 == 1)    # finger 0: left=source, right=drain

        # Always track port positions even for skipped S/D
        if j == 0:
            source_li1_x0 = cx
        if j == geom.n_fingers:
            drain_li1_x0 = cx

        # Skip contacts + li1 on S/D strips that connect only via shared
        # diffusion at abutment (e.g. internal nets in NAND gates).
        if j in _skip:
            continue

        # Place one column of contacts
        for cy in c_y_centres:
            c.add_polygon(
                [(cx - c_size / 2 + contact_x_offset, cy - c_size / 2),
                 (cx + c_size / 2 + contact_x_offset, cy - c_size / 2),
                 (cx + c_size / 2 + contact_x_offset, cy + c_size / 2),
                 (cx - c_size / 2 + contact_x_offset, cy + c_size / 2)],
                layer=lyr_contact,
            )

        # li1 rail covering S/D region, pulled back from poly edge
        # to maintain li1.2 spacing (li1 spacing >= li1_spacing_min).
        # Without pullback, adjacent li1 rails are separated by only
        # l_um (gate length), which violates li1.2 when l < li1_spacing.
        #
        # li.5 asymmetric enclosure: li1 must enclose licon by
        #   enc_li_2adj (0.08) on north+south (Y),
        #   enc_li_opp  (0.00) on east+west   (X).
        li1_sp  = rules.li1.get("spacing_min_um", 0.17)
        enc_li_2adj, enc_li_opp = rules.enclosure("contacts", "enclosure_in_li1")
        half_sd = geom.sd_length_um / 2

        # Li1 strip width: max of contact size and li1 min width.
        li1_half_w = max(c_size / 2, li_w / 2)
        li_x0 = cx - li1_half_w + contact_x_offset
        li_x1 = cx + li1_half_w + contact_x_offset

        # Y extent: ensure enc_li_2adj (0.08) above/below outermost contacts
        li_y0 = min(diff_y0, c_y_centres[0]  - c_size / 2 - enc_li_2adj)
        li_y1 = max(diff_y1, c_y_centres[-1] + c_size / 2 + enc_li_2adj)

        c.add_polygon(
            [(li_x0, li_y0), (li_x1, li_y0),
             (li_x1, li_y1), (li_x0, li_y1)],
            layer=lyr_li1,
        )

        # Refine port X to actual li1 centre when li1 is drawn
        if j == 0:
            source_li1_x0 = (li_x0 + li_x1) / 2
        if j == geom.n_fingers:
            drain_li1_x0 = (li_x0 + li_x1) / 2

    # ── Ports ─────────────────────────────────────────────────────────────────
    diff_y_mid = (diff_y0 + diff_y1) / 2

    # G — gate: port on the top edge of the first poly finger (li1 not on gate;
    #     port placed at poly level so the caller can connect a gate rail)
    c.add_port(
        name="G",
        center=(gate_port_x[0], geom.total_y_um),
        width=geom.l_um,
        orientation=90,
        layer=lyr_gate,
    )

    # S — source (leftmost S/D region)
    c.add_port(
        name="S",
        center=(source_li1_x0, diff_y_mid),
        width=geom.w_finger_um,
        orientation=180,
        layer=lyr_li1,
    )

    # D — drain (rightmost S/D region)
    c.add_port(
        name="D",
        center=(drain_li1_x0, diff_y_mid),
        width=geom.w_finger_um,
        orientation=0,
        layer=lyr_li1,
    )

    return c
