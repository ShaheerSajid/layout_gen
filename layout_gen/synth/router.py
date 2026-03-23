"""
layout_gen.synth.router — routing style dispatch for synthesised cells.

Each routing style maps a ``RoutingSpec`` (from the topology template YAML)
to a set of polygon rectangles drawn on the output ``gf.Component``.
Style handlers are registered by name and called by :func:`route`.

Adding a new style
------------------
Define a function with the signature::

    def _my_style(
        comp:   gf.Component,
        spec:   RoutingSpec,
        placed: dict[str, PlacedDevice],
        rules:  PDKRules,
    ) -> list[PortCandidate]:
        ...

Then register it::

    register_style("my_style", _my_style)

Or use the decorator form (at module level):

    @_style("my_style")
    def _my_style(...): ...
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Any

from layout_gen.pdk        import PDKRules
from layout_gen.cells.standard import _rect, _gate_x, _sd_x, _diff_y
from layout_gen.synth.loader   import RoutingSpec
from layout_gen.synth.placer   import (
    PlacedDevice,
    TerminalGeom,
    resolve_terminal,
    global_gate_x,
    global_sd_x,
    global_diff_y,
    global_poly_top,
    global_poly_bottom,
)


# ── PortCandidate ─────────────────────────────────────────────────────────────

@dataclass
class PortCandidate:
    """Port location hint emitted by a routing style handler.

    The synthesizer matches ``location_key`` against each port's
    ``PortSpec.location`` to know where to place the output port.
    """
    net:          str
    location_key: str   # must match the port's location: field in the YAML
    x:            float
    y:            float
    layer:        str   # logical layer name
    width:        float
    orientation:  int   # gdsfactory port orientation (degrees)


# ── Style registry ────────────────────────────────────────────────────────────

# Handler type alias
_Handler = Callable[
    [Any, RoutingSpec, dict[str, PlacedDevice], PDKRules],
    list[PortCandidate],
]
_REGISTRY: dict[str, _Handler] = {}


def register_style(name: str, fn: _Handler) -> None:
    """Register *fn* as the handler for routing style *name*."""
    _REGISTRY[name] = fn


def _style(name: str):
    """Decorator: ``@_style("name")`` registers the decorated function."""
    def _dec(fn: _Handler) -> _Handler:
        _REGISTRY[name] = fn
        return fn
    return _dec


# ── Router ─────────────────────────────────────────────────────────────────────

class Router:
    """Applies all routing specs from a template to a ``gf.Component``.

    Parameters
    ----------
    rules :
        PDK rules.
    """

    def __init__(self, rules: PDKRules):
        self.rules = rules

    def route(
        self,
        comp:     Any,                       # gf.Component
        routing:  list[RoutingSpec],
        placed:   dict[str, PlacedDevice],
    ) -> list[PortCandidate]:
        """Route all specs and return collected port candidates."""
        candidates: list[PortCandidate] = []
        for spec in routing:
            handler = _REGISTRY.get(spec.style)
            if handler is None:
                warnings.warn(
                    f"No handler registered for routing style {spec.style!r} "
                    f"(net={spec.net!r}); skipping.",
                    stacklevel=2,
                )
                continue
            result = handler(comp, spec, placed, self.rules)
            if result:
                candidates.extend(result)
        return candidates


# ── Style handlers ─────────────────────────────────────────────────────────────

@_style("shared_gate_poly")
def _shared_gate_poly(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Vertical poly bridge from NMOS gate top to PMOS gate bottom.

    Expected path: ``[N.G, P.G]``
    """
    if len(spec.path) < 2:
        return []
    n_name = spec.path[0].split(".")[0]
    p_name = spec.path[1].split(".")[0]
    dev_n = placed[n_name]
    dev_p = placed[p_name]

    gx0, gx1 = global_gate_x(dev_n, 0)
    y_bot = global_poly_top(dev_n)
    y_top = global_poly_bottom(dev_p)

    lyr = rules.layer("poly")
    if y_top > y_bot:
        _rect(comp, gx0, gx1, y_bot, y_top, lyr)

    gap = max(y_top - y_bot, rules.poly.get("width_min_um", 0.15))
    gate_mid_y = (y_bot + y_top) / 2

    # Emit both a net-specific key (e.g. "A_gate_left_edge_mid_y") and the
    # generic key ("gate_left_edge_mid_y") so that single-gate cells (inverter)
    # still resolve with the old location string and multi-gate cells (NAND2)
    # can address each gate individually.
    net_key = f"{spec.net}_gate_left_edge_mid_y"
    return [
        PortCandidate(
            net=spec.net, location_key=net_key,
            x=gx0, y=gate_mid_y,
            layer="poly", width=gap, orientation=180,
        ),
        PortCandidate(
            net=spec.net, location_key="gate_left_edge_mid_y",
            x=gx0, y=gate_mid_y,
            layer="poly", width=gap, orientation=180,
        ),
    ]


@_style("drain_bridge")
def _drain_bridge(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Vertical li1 bridge from NMOS drain top to PMOS drain bottom.

    Expected path: ``[N.D, P.D]``
    """
    if len(spec.path) < 2:
        return []
    n_name = spec.path[0].split(".")[0]
    p_name = spec.path[1].split(".")[0]
    dev_n = placed[n_name]
    dev_p = placed[p_name]

    j_n = dev_n.geom.n_fingers   # drain index for n_fingers=1 is j=1
    dx0, dx1 = global_sd_x(dev_n, j_n, rules)

    nd_y0, nd_y1 = global_diff_y(dev_n, rules)
    pd_y0, pd_y1 = global_diff_y(dev_p, rules)

    lyr = rules.layer("li1")
    if pd_y0 > nd_y1:
        _rect(comp, dx0, dx1, nd_y1, pd_y0, lyr)

    bridge_height = max(pd_y0 - nd_y1, dev_n.geom.l_um)
    out_mid_y = (nd_y1 + pd_y0) / 2

    return [PortCandidate(
        net=spec.net,
        location_key="drain_bridge_right_edge_mid_y",
        x=dx1, y=out_mid_y,
        layer="li1",
        width=bridge_height,
        orientation=0,
    )]


@_style("horizontal_power_rail")
def _horizontal_power_rail(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Met1 power rail spanning the full cell width.

    Uses ``spec.edge`` to select ``"bottom"`` (GND) or ``"top"`` (VDD).
    """
    if not placed:
        return []

    # Use max(met1_min, li1_width) so the rail covers li1 contacts cleanly —
    # same logic as the hand-coded draw_inverter / draw_nand2 cells.
    rail_h = max(
        (rules.met1 or {}).get("width_min_um", 0.14),
        rules.li1.get("width_min_um", 0.17),
    )
    lyr    = rules.layer("met1")

    cell_x0  = min(dev.x for dev in placed.values())
    cell_x1  = max(dev.x + dev.geom.total_x_um for dev in placed.values())
    cell_ytop = max(dev.y + dev.geom.total_y_um for dev in placed.values())
    cell_w   = cell_x1 - cell_x0
    cell_cx  = (cell_x0 + cell_x1) / 2

    edge = spec.edge or "bottom"

    if edge == "bottom":
        _rect(comp, cell_x0, cell_x1, -rail_h, 0.0, lyr)
        return [PortCandidate(
            net=spec.net,
            location_key="bottom_rail_center",
            x=cell_cx, y=-rail_h / 2,
            layer="met1",
            width=cell_w,
            orientation=270,
        )]
    else:  # top
        _rect(comp, cell_x0, cell_x1, cell_ytop, cell_ytop + rail_h, lyr)
        return [PortCandidate(
            net=spec.net,
            location_key="top_rail_center",
            x=cell_cx, y=cell_ytop + rail_h / 2,
            layer="met1",
            width=cell_w,
            orientation=90,
        )]


@_style("li1_bridge")
def _li1_bridge(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Horizontal li1 bridge between two S/D terminals.

    Expected path: ``[DevA.Term, DevB.Term]`` where terms are at the same Y band.
    Used for Q and Q_ node connections in the 6T bit cell.
    """
    if len(spec.path) < 2:
        return []

    t0 = resolve_terminal(spec.path[0], placed, rules)
    t1 = resolve_terminal(spec.path[1], placed, rules)

    lyr = rules.layer("li1")

    # Bridge spans between the two terminals in X
    bridge_x0 = min(t0.x1, t1.x0) if t0.x1 < t1.x0 else min(t1.x1, t0.x0)
    bridge_x1 = max(t0.x1, t1.x0) if t0.x1 < t1.x0 else max(t1.x1, t0.x0)

    # Y band: intersection of the two terminal diffs
    y0 = max(t0.y0, t1.y0)
    y1 = min(t0.y1, t1.y1)

    if bridge_x1 > bridge_x0 and y1 > y0:
        _rect(comp, bridge_x0, bridge_x1, y0, y1, lyr)

    mid_x = (bridge_x0 + bridge_x1) / 2
    mid_y = (y0 + y1) / 2

    return [PortCandidate(
        net=spec.net,
        location_key=f"{spec.net}_bridge_center",
        x=mid_x, y=mid_y,
        layer="li1",
        width=max(y1 - y0, rules.li1.get("width_min_um", 0.17)),
        orientation=90,
    )]


# ── Phase-2 routing styles ────────────────────────────────────────────────────

@_style("poly_stub_met1_bus")
def _poly_stub_met1_bus(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """WL wordline: polycontact stub above each PG gate body + met1 horizontal bus.

    Expected path: ``[PG_L.G, PG_R.G]``

    Because the PG gate poly (L=0.15 µm) is narrower than the 0.17 µm licon1
    contact, contacts cannot be placed inside the channel.  Instead, the gate
    poly is widened to a 0.27 µm pad above the transistor body, and a
    polycontact (licon1) + mcon via is dropped there.  A met1 horizontal bus
    then ties both stubs together.

    Emits location_key ``"wl_bus_center"`` at the left edge of the met1 bus,
    orientation 180° (West — toward the row decoder).
    """
    c_size   = rules.contacts["size_um"]                       # licon1/mcon: 0.17 µm
    enc_poly = rules.contacts.get("poly_enclosure_um", 0.05)   # poly enc of licon1
    enc_li_2adj = rules.contacts.get("enclosure_in_li1_2adj_um", 0.08)  # li.5
    enc_m1_2adj = rules.met1.get("enclosure_of_mcon_2adj_um", 0.06)     # m1.5
    ch       = c_size / 2                                       # 0.085 µm
    li1_lh   = ch + enc_li_2adj                                 # li1 landing half
    m1_lh    = ch + enc_m1_2adj                                 # met1 landing half
    pad_half = (c_size + 2 * enc_poly) / 2                     # 0.135 µm
    li1_sp   = rules.li1.get("spacing_min_um", 0.17)

    lyr_g       = rules.layer("poly")
    lyr_li1     = rules.layer("li1")
    lyr_contact = rules.layer("licon1")
    lyr_mcon    = rules.layer("mcon")
    lyr_m1      = rules.layer("met1")

    stub_locs: list[tuple[float, float]] = []   # (gcx, stub_cy) per gate stub

    for ref in spec.path:
        parts = ref.split(".", 1)
        if len(parts) != 2 or parts[1] != "G":
            continue
        dev = placed.get(parts[0])
        if dev is None:
            continue

        gx0, gx1 = global_gate_x(dev, 0)
        gcx      = (gx0 + gx1) / 2
        pg_ty    = global_poly_top(dev)        # top Y of transistor body

        # Place stub high enough to maintain li1 spacing from S/D li1 rails
        _, diff_y1 = global_diff_y(dev, rules)
        stub_cy_min = diff_y1 + li1_sp + li1_lh   # li1 landing bottom >= diff_y1 + li1_sp
        stub_cy     = max(pg_ty + pad_half, stub_cy_min)

        # Widened poly pad above transistor body (same net as gate poly)
        poly_pad_top = stub_cy + ch + enc_poly
        _rect(comp, gcx - pad_half, gcx + pad_half,
                    pg_ty, poly_pad_top, lyr_g)
        # Polycontact (licon1 on poly)
        _rect(comp, gcx - ch, gcx + ch, stub_cy - ch, stub_cy + ch, lyr_contact)
        # Li1 landing over licon1 (li.5: 2-adj-edge enc)
        _rect(comp, gcx - li1_lh, gcx + li1_lh,
                    stub_cy - li1_lh, stub_cy + li1_lh, lyr_li1)
        # Mcon (li1 → met1)
        _rect(comp, gcx - ch, gcx + ch, stub_cy - ch, stub_cy + ch, lyr_mcon)

        stub_locs.append((gcx, stub_cy))

    if not stub_locs:
        return []

    # Met1 bus spanning all stubs (m1.5: enclosing each mcon, 2-adj-edge)
    xs        = [gcx for gcx, _ in stub_locs]
    cy        = stub_locs[0][1]                # all stubs at same Y (same device type)
    wl_x0    = min(xs) - m1_lh
    wl_x1    = max(xs) + m1_lh
    wl_y0    = cy - m1_lh
    wl_y1    = cy + m1_lh
    _rect(comp, wl_x0, wl_x1, wl_y0, wl_y1, lyr_m1)

    # Port at left edge of WL bus, facing West (row decoder direction)
    return [PortCandidate(
        net=spec.net,
        location_key="wl_bus_center",
        x=wl_x0,
        y=(wl_y0 + wl_y1) / 2,
        layer="met1",
        width=wl_y1 - wl_y0,
        orientation=180,
    )]


@_style("cross_couple_gate")
def _cross_couple_gate(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Met2 cross-coupling: Q/Q_ li1 node → opposite inverter gate.

    Geometry
    --------
    - Source via stack (Q/Q_ li1 → met1 → via1 → met2) at the li1 bridge
      centre between the INV drain and the PG source terminal.
    - Gate poly stub widened above PMOS body (same X as gate, above cell_ytop)
      with full licon1→li1→mcon→met1→via1→met2 stack.
    - L-shape (track=0) or U-shape (track=1) met2 wire connects source to gate.

    Template fields
    ---------------
    path  : ``[<ignored_source_label>, PD_X.G, PU_X.G]``
            (first element is a semantic hint; actual source is derived from
            the placed device terminal→net mapping)
    extra : ``track`` — int, 0 or 1.  Controls met2 horizontal Y level.
            track=0 → horizontal at ``gsc_y`` (Q→INV_R)
            track=1 → one track higher (Q_→INV_L, avoids overlap)
    """
    # ── Geometry constants ────────────────────────────────────────────────────
    c_size   = rules.contacts["size_um"]                       # 0.17 µm (licon1/mcon)
    enc_poly = rules.contacts.get("poly_enclosure_um", 0.05)
    enc_li_licon_2adj = rules.contacts.get("enclosure_in_li1_2adj_um", 0.08)
    enc_m1_mcon = rules.mcon.get("enclosure_in_met1_um", 0.03)  # met1 enc of mcon (all sides)
    enc_m1_mcon_2adj = rules.met1.get("enclosure_of_mcon_2adj_um", 0.06)  # met1 enc of mcon (2 adj)
    _met2    = rules.met2 if rules.met2 else rules.met1  # fallback for older PDKs
    _via1    = rules.via1 if rules.via1 else {}
    via1_size = _via1.get("size_um", c_size)                    # via1 cut size (0.15 µm)
    enc_m1_via = _via1.get("enclosure_in_met1_um", 0.055)      # met1 enc of via1 (all sides)
    enc_m1_via_2adj = _via1.get("enclosure_in_met1_2adj_um", 0.085)  # met1 enc of via1 (2 adj)
    enc_m2_via = _via1.get("enclosure_in_met2_um", 0.055)      # met2 enc of via1 (all sides)
    enc_m2_via_2adj = _met2.get("enclosure_of_via1_2adj_um", 0.085)  # met2 enc of via1 (2 adj)
    met1_sp  = rules.met1.get("spacing_min_um", 0.14)
    met2_w   = _met2.get("width_min_um",   0.14)
    met2_sp  = _met2.get("spacing_min_um", 0.14)
    ch       = c_size / 2                                      # 0.085 µm (half of licon1)
    vh       = via1_size / 2                                   # 0.075 µm (half of via1)
    pad_half = (c_size + 2 * enc_poly) / 2                    # 0.135 µm
    rail_h   = max(rules.met1.get("width_min_um", 0.14),
                   rules.li1.get("width_min_um", 0.17))
    # Landing half-extents that satisfy both minimum and 2-adjacent-edge enclosure
    li1_land_half = ch + enc_li_licon_2adj      # li1 landing around licon1
    m1_land_half  = max(ch + enc_m1_mcon_2adj,  # met1 landing: enclose mcon (2adj)
                        vh + enc_m1_via_2adj)   #   AND via1 (2adj)
    m2_land_half  = vh + enc_m2_via_2adj        # met2 landing around via1

    lyr_g       = rules.layer("poly")
    lyr_li1     = rules.layer("li1")
    lyr_contact = rules.layer("licon1")
    lyr_mcon    = rules.layer("mcon")
    lyr_m1      = rules.layer("met1")
    lyr_via1    = rules.layer("via1")
    lyr_m2      = rules.layer("met2")

    # ── Cell top Y: highest PMOS body top ─────────────────────────────────────
    pmos_devs = [d for d in placed.values() if d.spec.device_type == "pmos"]
    if not pmos_devs:
        return []
    cell_ytop = max(d.y + d.geom.total_y_um for d in pmos_devs)

    # Gate stub contact centre Y — must clear VDD rail top + met1 spacing
    gsc_y = cell_ytop + rail_h + met1_sp + m1_land_half

    # ── Source: Q/Q_ li1 bridge centre ────────────────────────────────────────
    # Derive from placed device terminal→net mapping.
    # For net "Q":  NMOS with S=Q → PG_L,  NMOS with D=Q → PD_L
    # For net "Q_": NMOS with S=Q_ → PG_R, NMOS with D=Q_ → PD_R
    pg_dev = pd_dev = None
    for dev in placed.values():
        if dev.spec.device_type != "nmos":
            continue
        if dev.spec.terminals.get("S") == spec.net:
            pg_dev = dev   # pass-gate: Q on source
        if dev.spec.terminals.get("D") == spec.net:
            pd_dev = dev   # pull-down: Q on drain

    if pg_dev is None or pd_dev is None:
        warnings.warn(
            f"cross_couple_gate: cannot locate Q node devices for net {spec.net!r}; "
            f"skipped.",
            stacklevel=3,
        )
        return []

    t_drain  = resolve_terminal(f"{pd_dev.name}.D", placed, rules)
    t_source = resolve_terminal(f"{pg_dev.name}.S", placed, rules)
    q_x      = (t_drain.x1 + t_source.x0) / 2   # li1 bridge centre X
    nd_ymid  = (t_drain.y0 + t_drain.y1) / 2

    # Via stack at Q/Q_ source node (li1 → mcon → met1 → via1 → met2)
    _rect(comp, q_x - ch, q_x + ch,
                nd_ymid - ch, nd_ymid + ch, lyr_mcon)
    _rect(comp, q_x - m1_land_half, q_x + m1_land_half,
                nd_ymid - m1_land_half, nd_ymid + m1_land_half, lyr_m1)
    _rect(comp, q_x - vh, q_x + vh,
                nd_ymid - vh, nd_ymid + vh, lyr_via1)
    _rect(comp, q_x - m2_land_half, q_x + m2_land_half,
                nd_ymid - m2_land_half, nd_ymid + m2_land_half, lyr_m2)

    # ── Target gate stubs (one per unique gate X position) ────────────────────
    gate_xs: list[float] = []
    seen_x:  set[int]   = set()   # keyed on rounded-to-nm int to avoid float issues

    for ref in spec.path:
        parts = ref.split(".", 1)
        if len(parts) != 2 or parts[1] != "G":
            continue
        dev = placed.get(parts[0])
        if dev is None:
            continue

        gx0, gx1 = global_gate_x(dev, 0)
        gcx      = (gx0 + gx1) / 2
        gcx_nm   = round(gcx * 1000)   # nm integer key for dedup
        if gcx_nm in seen_x:
            continue
        seen_x.add(gcx_nm)
        gate_xs.append(gcx)

        # Poly stub widened above PMOS body (at gate poly X, above cell_ytop)
        _rect(comp, gcx - pad_half, gcx + pad_half,
                    cell_ytop, gsc_y + ch + enc_poly, lyr_g)
        # Licon1 (polycontact: poly → li1)
        _rect(comp, gcx - ch, gcx + ch, gsc_y - ch, gsc_y + ch, lyr_contact)
        # Li1 landing (must satisfy li.5: 2-adj-edge enclosure of licon)
        _rect(comp, gcx - li1_land_half, gcx + li1_land_half,
                    gsc_y - li1_land_half, gsc_y + li1_land_half, lyr_li1)
        # Mcon (li1 → met1)
        _rect(comp, gcx - ch, gcx + ch, gsc_y - ch, gsc_y + ch, lyr_mcon)
        # Met1 landing (satisfies both m1.5 mcon enc and via.5a via1 enc, 2-adj)
        _rect(comp, gcx - m1_land_half, gcx + m1_land_half,
                    gsc_y - m1_land_half, gsc_y + m1_land_half, lyr_m1)
        # Via1 (met1 → met2)
        _rect(comp, gcx - vh, gcx + vh, gsc_y - vh, gsc_y + vh, lyr_via1)
        # Met2 landing (satisfies m2.5: 2-adj-edge enc of via1)
        _rect(comp, gcx - m2_land_half, gcx + m2_land_half,
                    gsc_y - m2_land_half, gsc_y + m2_land_half, lyr_m2)

    if not gate_xs:
        return []

    gcx_target = gate_xs[0]
    track      = int(spec.extra.get("track", 0))
    # Track pitch must clear the gate-stub met2 landing (m2_land_half above gsc_y)
    # plus met2 spacing, plus the wire half-width of the next track.
    track_pitch = m2_land_half + met2_sp + met2_w / 2
    route_y    = gsc_y + track * track_pitch

    # ── Met2 route: source → gate stub ────────────────────────────────────────
    # Use met2_w for wire width.  Extend horizontal wires to fully overlap
    # the met2 via-landings at both endpoints to avoid narrow notches that
    # trip the width DRC check.
    hw = met2_w / 2                        # wire half-width
    x_lo = min(q_x, gcx_target) - hw      # left edge of left-most landing
    x_hi = max(q_x, gcx_target) + hw      # right edge of right-most landing

    if track == 0:
        # L-shape: vertical at q_x from nd_ymid to route_y, then horizontal
        _rect(comp, q_x - hw, q_x + hw, nd_ymid, route_y + hw, lyr_m2)
        _rect(comp, x_lo, x_hi, route_y - hw, route_y + hw, lyr_m2)
    else:
        # U-shape: vertical up to route_y, horizontal, vertical down to gsc_y
        _rect(comp, q_x - hw, q_x + hw, nd_ymid, route_y + hw, lyr_m2)
        _rect(comp, x_lo, x_hi, route_y - hw, route_y + hw, lyr_m2)
        _rect(comp, gcx_target - hw, gcx_target + hw,
                    gsc_y - hw, route_y + hw, lyr_m2)

    # No new port — ports for Q/Q_ come from li1_bridge candidates
    return []


@_style("expose_terminal")
def _expose_terminal(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Expose a device terminal as a port without drawing any routing geometry.

    Use this to make BL/BL_ and other unconnected-internally terminals
    accessible as cell ports.

    Expected path: ``[Dev.Terminal]`` (single element).

    Extra fields
    ------------
    orientation : int
        Port orientation in degrees (default 90 = North).
    location_key : str
        Explicit location key to emit (default ``"{net}_terminal_center"``).
    """
    if not spec.path:
        return []

    try:
        t = resolve_terminal(spec.path[0], placed, rules)
    except (KeyError, ValueError) as exc:
        warnings.warn(
            f"expose_terminal (net={spec.net!r}): {exc}; skipped.",
            stacklevel=3,
        )
        return []

    mid_x = (t.x0 + t.x1) / 2
    mid_y = (t.y0 + t.y1) / 2

    orientation = int(spec.extra.get("orientation", 90))
    # Width is perpendicular to orientation direction
    if orientation in (90, 270):   # North/South: width spans X
        width = t.x1 - t.x0
    else:                          # East/West:   width spans Y
        width = t.y1 - t.y0

    width = max(width, rules.li1.get("width_min_um", 0.17))

    location_key = spec.extra.get(
        "location_key",
        f"{spec.path[0].replace('.', '_')}_center",
    )

    return [PortCandidate(
        net=spec.net,
        location_key=location_key,
        x=mid_x,
        y=mid_y,
        layer=t.layer,
        width=width,
        orientation=orientation,
    )]


# ── Phase-3 routing styles: cross-row connections ─────────────────────────────

# ── Shared via-stack helper ──────────────────────────────────────────────────

def _via_stack_up(comp, rules, cx, cy, from_level, to_level):
    """Draw via stack from *from_level* to *to_level* at (cx, cy).

    Levels: 0=li1, 1=met1, 2=met2.  Draws all intermediate contacts/vias
    and metal landings.

    Returns the half-extent of the topmost landing (for wire sizing).
    """
    c_size    = rules.contacts["size_um"]
    mcon_sz   = (rules.mcon or {}).get("size_um", c_size)
    enc_m1_mc = rules.met1.get("enclosure_of_mcon_2adj_um", 0.06)
    met1_w    = rules.met1.get("width_min_um", 0.14)
    _via1     = rules.via1 if rules.via1 else {}
    via1_sz   = _via1.get("size_um", 0.15)
    enc_m1_v1 = _via1.get("enclosure_in_met1_2adj_um", 0.085)
    _met2     = rules.met2 if rules.met2 else rules.met1
    enc_m2_v1 = _met2.get("enclosure_of_via1_2adj_um", 0.085)

    mch  = mcon_sz / 2
    vh   = via1_sz / 2
    m1_lh = max(mch + enc_m1_mc, met1_w / 2, vh + enc_m1_v1)
    m2_lh = vh + enc_m2_v1

    lyr_mcon = rules.layer("mcon")
    lyr_m1   = rules.layer("met1")
    lyr_via1 = rules.layer("via1")
    lyr_m2   = rules.layer("met2")

    top_half = 0.0

    if from_level <= 0 and to_level >= 1:
        # mcon (li1 → met1)
        _rect(comp, cx - mch, cx + mch, cy - mch, cy + mch, lyr_mcon)
        _rect(comp, cx - m1_lh, cx + m1_lh, cy - m1_lh, cy + m1_lh, lyr_m1)
        top_half = m1_lh

    if from_level <= 1 and to_level >= 2:
        # Ensure met1 landing exists (needed when from_level=1)
        if from_level == 1:
            _rect(comp, cx - m1_lh, cx + m1_lh, cy - m1_lh, cy + m1_lh, lyr_m1)
        # via1 (met1 → met2)
        _rect(comp, cx - vh, cx + vh, cy - vh, cy + vh, lyr_via1)
        _rect(comp, cx - m2_lh, cx + m2_lh, cy - m2_lh, cy + m2_lh, lyr_m2)
        top_half = m2_lh

    return top_half


@_style("cross_row_connect")
def _cross_row_connect(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Connect a source terminal to gate(s) in other row pairs via L-route.

    The source (first element in path) is an S/D terminal whose li1 is already
    placed by ``drain_bridge`` or the transistor primitive.  Each target
    (remaining path elements, typically ``Dev.G``) gets a poly-contact stub
    with a via stack, and a metal bus is routed in an L-shape from source to
    all targets.

    Expected path: ``[source_dev.term, target_dev1.G, target_dev2.G, ...]``

    Extra fields
    ------------
    track_x : float
        X position for the vertical trunk.  Defaults to source X.
    via_level : int
        Metal level for the trunk.  1 = met1 (default, for short adjacent-row
        connections).  2 = met2 (for long-distance connections that would
        create met1 shorts).
    """
    if len(spec.path) < 2:
        return []

    via_level = int(spec.extra.get("via_level", 1))

    # ── Geometry constants ─────────────────────────────────────────────────
    c_size    = rules.contacts["size_um"]
    enc_poly  = rules.contacts.get("poly_enclosure_um", 0.05)
    enc_li_2a = rules.contacts.get("enclosure_in_li1_2adj_um", 0.08)
    enc_m1_mc = rules.met1.get("enclosure_of_mcon_2adj_um", 0.06)
    mcon_sz   = (rules.mcon or {}).get("size_um", c_size)
    li1_sp    = rules.li1.get("spacing_min_um", 0.17)
    met1_w    = rules.met1.get("width_min_um", 0.14)
    _via1     = rules.via1 if rules.via1 else {}
    via1_sz   = _via1.get("size_um", 0.15)
    enc_m1_v1 = _via1.get("enclosure_in_met1_2adj_um", 0.085)
    _met2     = rules.met2 if rules.met2 else rules.met1
    met2_w    = _met2.get("width_min_um", 0.14)
    enc_m2_v1 = _met2.get("enclosure_of_via1_2adj_um", 0.085)

    ch        = c_size / 2
    mch       = mcon_sz / 2
    vh        = via1_sz / 2
    pad_half  = (c_size + 2 * enc_poly) / 2
    li1_lh    = ch + enc_li_2a
    m1_lh     = max(mch + enc_m1_mc, met1_w / 2)

    # Trunk wire half-width depends on routing level.
    # Via landing pads (wider, for enclosure) are drawn by _via_stack_up;
    # the trunk wire itself only needs minimum metal width.
    if via_level >= 2:
        trunk_hw = met2_w / 2
        lyr_trunk = rules.layer("met2")
    else:
        trunk_hw = met1_w / 2
        lyr_trunk = rules.layer("met1")

    lyr_g     = rules.layer("poly")
    lyr_li1   = rules.layer("li1")
    lyr_licon = rules.layer("licon1")
    lyr_m1    = rules.layer("met1")

    # ── Resolve source terminal ────────────────────────────────────────────
    try:
        t_src = resolve_terminal(spec.path[0], placed, rules)
    except (KeyError, ValueError):
        return []
    src_cx = (t_src.x0 + t_src.x1) / 2
    src_cy = (t_src.y0 + t_src.y1) / 2

    # Via stack at source: li1 → met1 (→ met2 if via_level=2)
    _via_stack_up(comp, rules, src_cx, src_cy, 0, via_level)

    # ── Resolve target gates and build poly-contact stubs ──────────────────
    target_locs: list[tuple[float, float]] = []

    for ref in spec.path[1:]:
        parts = ref.split(".", 1)
        if len(parts) != 2:
            continue
        dev = placed.get(parts[0])
        if dev is None:
            continue

        term = parts[1]
        if term == "G":
            gx0, gx1 = global_gate_x(dev, 0)
            gcx      = (gx0 + gx1) / 2

            if src_cy < dev.y + dev.geom.total_y_um / 2:
                stub_cy = global_poly_bottom(dev) - li1_sp - li1_lh
            else:
                stub_cy = global_poly_top(dev) + li1_sp + li1_lh

            poly_body_top = global_poly_top(dev)
            poly_body_bot = global_poly_bottom(dev)
            if stub_cy > poly_body_top:
                _rect(comp, gcx - pad_half, gcx + pad_half,
                            poly_body_top, stub_cy + ch + enc_poly, lyr_g)
            else:
                _rect(comp, gcx - pad_half, gcx + pad_half,
                            stub_cy - ch - enc_poly, poly_body_bot, lyr_g)

            # Licon1 + li1 + mcon (+ via1 if met2)
            _rect(comp, gcx - ch, gcx + ch,
                        stub_cy - ch, stub_cy + ch, lyr_licon)
            _rect(comp, gcx - li1_lh, gcx + li1_lh,
                        stub_cy - li1_lh, stub_cy + li1_lh, lyr_li1)
            _via_stack_up(comp, rules, gcx, stub_cy, 0, via_level)

            target_locs.append((gcx, stub_cy))
        else:
            try:
                t_tgt = resolve_terminal(ref, placed, rules)
            except (KeyError, ValueError):
                continue
            tgt_cx = (t_tgt.x0 + t_tgt.x1) / 2
            tgt_cy = (t_tgt.y0 + t_tgt.y1) / 2
            _via_stack_up(comp, rules, tgt_cx, tgt_cy, 0, via_level)
            target_locs.append((tgt_cx, tgt_cy))

    if not target_locs:
        return []

    # ── Trunk L-route on the routing metal ─────────────────────────────────
    track_x = float(spec.extra.get("track_x", src_cx))

    all_ys = [src_cy] + [y for _, y in target_locs]
    y_min  = min(all_ys)
    y_max  = max(all_ys)

    # Vertical trunk
    if y_max > y_min:
        _rect(comp, track_x - trunk_hw, track_x + trunk_hw,
                    y_min - trunk_hw, y_max + trunk_hw, lyr_trunk)

    # Horizontal jog from trunk to source
    if abs(src_cx - track_x) > 0.001:
        jx0 = min(src_cx, track_x) - trunk_hw
        jx1 = max(src_cx, track_x) + trunk_hw
        _rect(comp, jx0, jx1,
                    src_cy - trunk_hw, src_cy + trunk_hw, lyr_trunk)

    # Horizontal jog from trunk to each target
    for tgt_x, tgt_y in target_locs:
        if abs(tgt_x - track_x) > 0.001:
            jx0 = min(tgt_x, track_x) - trunk_hw
            jx1 = max(tgt_x, track_x) + trunk_hw
            _rect(comp, jx0, jx1,
                        tgt_y - trunk_hw, tgt_y + trunk_hw, lyr_trunk)

    return []


@_style("vertical_bus")
def _vertical_bus(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Vertical metal bus connecting S/D terminals across multiple row pairs.

    Used for BL/BL_ bitlines that span the full cell height.

    Expected path: ``[Dev1.term, Dev2.term, ...]``

    Extra fields
    ------------
    bus_x : float
        Override X position for the bus.
    via_level : int
        Metal level for the bus trunk.  1 = met1 (default).  2 = met2.
    """
    if len(spec.path) < 2:
        return []

    via_level = int(spec.extra.get("via_level", 1))

    # ── Geometry constants ─────────────────────────────────────────────────
    mcon_sz   = (rules.mcon or {}).get("size_um", 0.17)
    enc_m1_mc = rules.met1.get("enclosure_of_mcon_2adj_um", 0.06)
    met1_w    = rules.met1.get("width_min_um", 0.14)
    _via1     = rules.via1 if rules.via1 else {}
    via1_sz   = _via1.get("size_um", 0.15)
    _met2     = rules.met2 if rules.met2 else rules.met1
    met2_w    = _met2.get("width_min_um", 0.14)
    enc_m2_v1 = _met2.get("enclosure_of_via1_2adj_um", 0.085)
    vh        = via1_sz / 2
    mch       = mcon_sz / 2

    if via_level >= 2:
        trunk_hw  = met2_w / 2
        lyr_trunk = rules.layer("met2")
    else:
        trunk_hw  = met1_w / 2
        lyr_trunk = rules.layer("met1")

    # ── Resolve all terminal positions ─────────────────────────────────────
    taps: list[tuple[float, float]] = []
    for ref in spec.path:
        try:
            t = resolve_terminal(ref, placed, rules)
        except (KeyError, ValueError):
            continue
        cx = (t.x0 + t.x1) / 2
        cy = (t.y0 + t.y1) / 2
        taps.append((cx, cy))

    if len(taps) < 2:
        return []

    # ── Bus X position ─────────────────────────────────────────────────────
    bus_x = float(spec.extra.get(
        "bus_x",
        sum(x for x, _ in taps) / len(taps),
    ))

    # ── Via stacks and horizontal taps at each terminal ────────────────────
    for tap_x, tap_y in taps:
        # Full via stack from li1 up to routing level
        _via_stack_up(comp, rules, tap_x, tap_y, 0, via_level)
        # Horizontal jog on routing metal from terminal to bus
        if abs(tap_x - bus_x) > 0.001:
            jx0 = min(tap_x, bus_x) - trunk_hw
            jx1 = max(tap_x, bus_x) + trunk_hw
            _rect(comp, jx0, jx1,
                        tap_y - trunk_hw, tap_y + trunk_hw, lyr_trunk)

    # ── Vertical trunk ────────────────────────────────────────────────────
    y_min = min(y for _, y in taps)
    y_max = max(y for _, y in taps)
    _rect(comp, bus_x - trunk_hw, bus_x + trunk_hw,
                y_min - trunk_hw, y_max + trunk_hw, lyr_trunk)

    return []
