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


# ── Poly spacing helper ───────────────────────────────────────────────────────

def _collect_gate_poly_ranges(
    placed: dict[str, PlacedDevice],
) -> list[tuple[float, float, str]]:
    """Return (x0, x1, device_name) for every gate poly in the cell."""
    ranges = []
    for name, dev in placed.items():
        for j in range(dev.geom.n_fingers):
            gx0, gx1 = global_gate_x(dev, j)
            ranges.append((gx0, gx1, name))
    return ranges


def _nudge_for_poly_spacing(
    cx:              float,
    pad_half_x:      float,
    own_gate_range:  tuple[float, float],
    all_gate_ranges: list[tuple[float, float, str]],
    poly_sp:         float,
) -> float:
    """Shift contact centre X so its poly pad maintains *poly_sp* from all gates.

    The pad must still overlap *own_gate_range* for electrical connectivity.
    PDK-agnostic: works with any poly spacing rule.
    """
    own_x0, own_x1 = own_gate_range
    eps = 0.005  # 5 nm extra clearance

    for gx0, gx1, _ in all_gate_ranges:
        # Skip the gate this contact belongs to
        if abs(gx0 - own_x0) < 0.001 and abs(gx1 - own_x1) < 0.001:
            continue

        pad_left  = cx - pad_half_x
        pad_right = cx + pad_half_x

        # Pad is to the left of this gate
        if pad_right <= gx0:
            gap = gx0 - pad_right
            if gap < poly_sp:
                cx -= (poly_sp - gap + eps)
        # Pad is to the right of this gate
        elif pad_left >= gx1:
            gap = pad_left - gx1
            if gap < poly_sp:
                cx += (poly_sp - gap + eps)

    # Clamp: pad must still overlap own gate for connectivity
    # pad_right > own_x0 → cx > own_x0 - pad_half_x
    # pad_left  < own_x1 → cx < own_x1 + pad_half_x
    cx = max(cx, own_x0 - pad_half_x + eps)
    cx = min(cx, own_x1 + pad_half_x - eps)

    return cx


# ── Power rail gap helper ──────────────────────────────────────────────────────

def _power_rail_gap(rules: PDKRules) -> float:
    """Extra Y gap between power rail and transistor body when li1 = met1.

    When li1 and met1 share the same GDS layer (e.g. GF180MCU), the S/D li1
    strips and the power rail met1 interact in DRC.  The natural gap
    (poly endcap) may be smaller than the met1 spacing rule, so we add the
    shortfall plus a small margin.
    """
    if not rules.li1_is_met1:
        return 0.0
    met1_sp = rules.met1.get("spacing_min_um", 0.14)
    endcap  = rules.poly.get("endcap_over_diff_um", 0.0)
    return max(0.0, met1_sp - endcap) + 0.01   # 10 nm margin


# ── General-purpose via stack drawing ──────────────────────────────────────────

def draw_via_stack(
    comp:       Any,
    rules:      PDKRules,
    cx:         float,
    cy:         float,
    from_layer: str,
    to_layer:   str,
) -> float:
    """Draw all vias and metal landings needed to connect *from_layer* to *to_layer*.

    Uses the PDK ``metal_stack`` to determine which vias to insert.
    Returns the half-extent of the topmost landing pad (for wire sizing).
    If the layers resolve to the same physical layer, nothing is drawn and
    0.0 is returned.
    """
    transitions = rules.via_stack_between(from_layer, to_layer)
    if not transitions:
        return 0.0

    top_half = 0.0
    for t in transitions:
        vh = t.via_size / 2

        # Lower metal landing
        lower_w_min = getattr(rules, t.lower_metal, {})
        if isinstance(lower_w_min, dict):
            lower_w_min = lower_w_min.get("width_min_um", 0.0)
        else:
            lower_w_min = 0.0
        lower_lh = max(vh + t.enc_lower, lower_w_min / 2)

        # Upper metal landing
        upper_w_min = getattr(rules, t.upper_metal, {})
        if isinstance(upper_w_min, dict):
            upper_w_min = upper_w_min.get("width_min_um", 0.0)
        else:
            upper_w_min = 0.0
        upper_lh = max(vh + t.enc_upper, upper_w_min / 2)

        lyr_via   = rules.layer(t.via_layer)
        lyr_lower = rules.layer(t.lower_metal)
        lyr_upper = rules.layer(t.upper_metal)

        _rect(comp, cx - vh, cx + vh, cy - vh, cy + vh, lyr_via)
        _rect(comp, cx - lower_lh, cx + lower_lh,
                    cy - lower_lh, cy + lower_lh, lyr_lower)
        _rect(comp, cx - upper_lh, cx + upper_lh,
                    cy - upper_lh, cy + upper_lh, lyr_upper)

        top_half = upper_lh

    return top_half


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

    # Drain is at j=n_fingers (right) by default; sd_flip puts it at j=0 (left)
    j_n = 0 if dev_n.spec.sd_flip else dev_n.geom.n_fingers
    dx0, dx1 = global_sd_x(dev_n, j_n, rules)

    # When li1=met1, ensure the bridge meets met1 min width
    if rules.li1_is_met1:
        met1_w = rules.met1.get("width_min_um", 0.14)
        bridge_w = dx1 - dx0
        if bridge_w < met1_w:
            cx = (dx0 + dx1) / 2
            dx0 = cx - met1_w / 2
            dx1 = cx + met1_w / 2

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
    Alternatively, ``spec.extra["y_pos"]`` places an intermediate rail
    centred at the given Y coordinate (for stacked multi-row layouts).
    """
    if not placed:
        return []

    rail_h = max(
        (rules.met1 or {}).get("width_min_um", 0.14),
        rules.li1.get("width_min_um", 0.17),
    )
    lyr    = rules.layer("met1")

    dev_x0  = min(dev.x for dev in placed.values())
    dev_x1  = max(dev.x + dev.geom.total_x_um for dev in placed.values())
    cell_ytop = max(dev.y + dev.geom.total_y_um for dev in placed.values())

    # Use fixed cell width if provided by auto-router
    fixed_w = spec.extra.get("cell_width", 0) if spec.extra else 0
    if fixed_w > 0:
        dev_cx = (dev_x0 + dev_x1) / 2
        cell_x0 = dev_cx - fixed_w / 2
        cell_x1 = dev_cx + fixed_w / 2
    else:
        cell_x0 = dev_x0
        cell_x1 = dev_x1
    cell_w  = cell_x1 - cell_x0
    cell_cx = (cell_x0 + cell_x1) / 2

    # ── Intermediate rail at explicit Y position ──────────────────
    y_pos = spec.extra.get("y_pos") if spec.extra else None
    if y_pos is not None:
        y_center = float(y_pos)
        y0 = y_center
        y1 = y_center + rail_h
        _rect(comp, cell_x0, cell_x1, y0, y1, lyr)
        return [PortCandidate(
            net=spec.net,
            location_key=f"rail_{spec.net}_{y_center:.3f}",
            x=cell_cx, y=(y0 + y1) / 2,
            layer="met1",
            width=cell_w,
            orientation=90,
        )]

    # ── Edge rails (top / bottom) ─────────────────────────────────
    edge = spec.edge or "bottom"
    gap  = _power_rail_gap(rules)

    if edge == "bottom":
        y0, y1 = -rail_h - gap, -gap
        _rect(comp, cell_x0, cell_x1, y0, y1, lyr)
        return [PortCandidate(
            net=spec.net,
            location_key="bottom_rail_center",
            x=cell_cx, y=(y0 + y1) / 2,
            layer="met1",
            width=cell_w,
            orientation=270,
        )]
    else:  # top
        y0, y1 = cell_ytop + gap, cell_ytop + gap + rail_h
        _rect(comp, cell_x0, cell_x1, y0, y1, lyr)
        return [PortCandidate(
            net=spec.net,
            location_key="top_rail_center",
            x=cell_cx, y=(y0 + y1) / 2,
            layer="met1",
            width=cell_w,
            orientation=90,
        )]


@_style("source_to_rail")
def _source_to_rail(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Connect source terminals to a power rail via li1 extension + mcon.

    Expected path: ``[Dev.S, ...]`` — one or more source terminals.
    ``spec.edge`` selects ``"bottom"`` (GND) or ``"top"`` (VDD).

    For each terminal, draws:
    - li1 strap from terminal edge to the rail Y boundary
    - mcon at the rail boundary to connect li1 → met1
    """
    if not spec.path:
        return []

    edge = spec.edge or "bottom"

    c_size = rules.contacts.get("size_um", 0.17)
    ch     = c_size / 2
    mcon_rules = rules.mcon or {}
    mcon_sz = mcon_rules.get("size_um", c_size)
    mch    = mcon_sz / 2
    enc_m1_2adj, _ = rules.enclosure("met1", "enclosure_of_mcon")
    m1_land = mch + enc_m1_2adj    # met1 landing half-extent around mcon
    li1_w  = rules.li1.get("width_min_um", 0.17)
    rail_h = max(rules.met1.get("width_min_um", 0.14), li1_w)

    lyr_li1  = rules.layer("li1")
    lyr_mcon = rules.layer("mcon")
    lyr_m1   = rules.layer("met1")

    # Rail Y boundaries (shifted by gap when li1=met1)
    gap = _power_rail_gap(rules)
    if edge == "bottom":
        rail_y0, rail_y1 = -rail_h - gap, -gap
    else:
        cell_ytop = max(d.y + d.geom.total_y_um for d in placed.values())
        rail_y0, rail_y1 = cell_ytop + gap, cell_ytop + gap + rail_h

    for ref in spec.path:
        try:
            t = resolve_terminal(ref, placed, rules)
        except (KeyError, ValueError):
            continue

        tx_mid = (t.x0 + t.x1) / 2
        ty_mid = (t.y0 + t.y1) / 2

        # Li1 strap: extend from terminal toward rail
        li1_hx = max(li1_w / 2, (t.x1 - t.x0) / 2)
        if edge == "bottom":
            li1_y0 = rail_y0
            li1_y1 = t.y1
        else:
            li1_y0 = t.y0
            li1_y1 = rail_y1
        _rect(comp, tx_mid - li1_hx, tx_mid + li1_hx,
                    li1_y0, li1_y1, lyr_li1)

        # Via + landing only when li1 and met1 are different layers
        if not rules.li1_is_met1:
            mcon_cy = (rail_y0 + rail_y1) / 2
            _rect(comp, tx_mid - mch, tx_mid + mch,
                        mcon_cy - mch, mcon_cy + mch, lyr_mcon)
            _rect(comp, tx_mid - m1_land, tx_mid + m1_land,
                        mcon_cy - m1_land, mcon_cy + m1_land, lyr_m1)

    return []


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

    lyr    = rules.layer("li1")
    li1_w  = rules.li1.get("width_min_um", 0.17)

    # Bridge spans between the two terminals in X
    bridge_x0 = min(t0.x1, t1.x0) if t0.x1 < t1.x0 else min(t1.x1, t0.x0)
    bridge_x1 = max(t0.x1, t1.x0) if t0.x1 < t1.x0 else max(t1.x1, t0.x0)

    # Narrow horizontal strip: li1_min_width in Y, centred on diff midpoint
    y_mid = (max(t0.y0, t1.y0) + min(t0.y1, t1.y1)) / 2
    y0 = y_mid - li1_w / 2
    y1 = y_mid + li1_w / 2

    if bridge_x1 > bridge_x0:
        _rect(comp, bridge_x0, bridge_x1, y0, y1, lyr)

    mid_x = (bridge_x0 + bridge_x1) / 2

    return [PortCandidate(
        net=spec.net,
        location_key=f"{spec.net}_bridge_center",
        x=mid_x, y=y_mid,
        layer="li1",
        width=li1_w,
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
    poly is widened to a pad above the transistor body, and a via cell
    (poly_contact_to_met1) is placed there.  A met1 horizontal bus then ties
    both stubs together.

    The via cell is added as a Component reference (not flattened) so the
    geo fixer can move it as an atomic unit.

    Emits location_key ``"wl_bus_center"`` at the left edge of the met1 bus,
    orientation 180° (West — toward the row decoder).
    """
    from layout_gen.cells.vias import poly_contact_to_met1

    c_size   = rules.contacts["size_um"]
    enc_poly_2adj, enc_poly_opp = rules.enclosure("contacts", "poly_enclosure")
    enc_li_2adj, _              = rules.enclosure("contacts", "enclosure_in_li1")
    enc_m1_2adj, _              = rules.enclosure("met1", "enclosure_of_mcon")
    ch       = c_size / 2
    li1_lh_2adj = ch + enc_li_2adj
    m1_lh    = ch + enc_m1_2adj
    pad_half_x = (c_size + 2 * enc_poly_2adj) / 2
    pad_half_y = ch + enc_poly_opp
    # Via cell (licon_poly) poly Y half-extent — uses 2adj on Y, opp on X
    via_poly_half_y = ch + enc_poly_2adj
    li1_sp   = rules.li1.get("spacing_min_um", 0.17)

    lyr_g    = rules.layer("poly")
    poly_sp  = rules.poly.get("spacing_min_um", 0.21)

    # Build via cell once, reuse per gate stub
    via_cell = poly_contact_to_met1(rules)

    # Collect all gate poly X ranges for spacing checks
    all_gates = _collect_gate_poly_ranges(placed)

    stub_locs: list[tuple[float, float]] = []

    for ref in spec.path:
        parts = ref.split(".", 1)
        if len(parts) != 2 or parts[1] != "G":
            continue
        dev = placed.get(parts[0])
        if dev is None:
            continue

        gx0, gx1 = global_gate_x(dev, 0)
        gcx      = (gx0 + gx1) / 2
        pg_ty    = global_poly_top(dev)

        # Nudge contact X to maintain poly spacing with adjacent gates
        gcx = _nudge_for_poly_spacing(gcx, pad_half_x, (gx0, gx1),
                                      all_gates, poly_sp)

        # Place stub high enough to maintain li1 spacing from S/D li1 rails
        _, diff_y1 = global_diff_y(dev, rules)
        stub_cy_min = diff_y1 + li1_sp + li1_lh_2adj
        stub_cy     = max(pg_ty + pad_half_y, stub_cy_min)

        # Poly extension from gate top to via cell centre
        # (via cell draws its own poly enclosure around the licon;
        #  this strip just connects the gate to the via cell's poly)
        if stub_cy > pg_ty:
            _rect(comp, gcx - pad_half_x, gcx + pad_half_x,
                        pg_ty, stub_cy, lyr_g)

        # Via cell (licon_poly + mcon_stack) — placed as cell ref, not flattened
        ref_cell = comp.add_ref(via_cell)
        ref_cell.move((gcx, stub_cy))

        stub_locs.append((gcx, stub_cy))

    if not stub_locs:
        return []

    # Target bus layer — defaults to met1, but template can specify higher
    bus_layer = spec.layer or "met1"

    # Via stacks from met1 up to bus layer at each stub
    bus_half = m1_lh
    for gcx, scy in stub_locs:
        lh = draw_via_stack(comp, rules, gcx, scy, "met1", bus_layer)
        if lh > bus_half:
            bus_half = lh

    # Ensure bus meets target layer min width
    bus_layer_rules = getattr(rules, bus_layer, None) or {}
    if isinstance(bus_layer_rules, dict):
        bus_w_min = bus_layer_rules.get("width_min_um", 0.0)
        bus_half = max(bus_half, bus_w_min / 2)

    # Horizontal bus spanning full cell width on the target layer
    lyr_bus = rules.layer(bus_layer)
    xs     = [gcx for gcx, _ in stub_locs]
    cy     = stub_locs[0][1]
    wl_x0  = spec.extra.get("cell_x0", min(xs) - bus_half) if spec.extra else min(xs) - bus_half
    wl_x1  = spec.extra.get("cell_x1", max(xs) + bus_half) if spec.extra else max(xs) + bus_half
    wl_y0  = cy - bus_half
    wl_y1  = cy + bus_half
    _rect(comp, wl_x0, wl_x1, wl_y0, wl_y1, lyr_bus)


    # Port at left edge of WL bus, facing West (row decoder direction)
    return [PortCandidate(
        net=spec.net,
        location_key="wl_bus_center",
        x=wl_x0,
        y=(wl_y0 + wl_y1) / 2,
        layer=bus_layer,
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
    """Cross-coupling: Q/Q_ li1 node → opposite inverter gate.

    Geometry
    --------
    - Source via stack (Q/Q_ li1 → target layer) at the li1 bridge
      centre between the INV drain and the PG source terminal.
    - Gate poly stub widened above PMOS body (same X as gate, above cell_ytop)
      with licon1→li1 poly contact, then via stack from li1 to target layer.
    - L-shape (track=0) or U-shape (track=1) wire on target layer connects
      source to gate.

    Template fields
    ---------------
    path  : ``[<ignored_source_label>, PD_X.G, PU_X.G]``
            (first element is a semantic hint; actual source is derived from
            the placed device terminal→net mapping)
    extra : ``track`` — int, 0 or 1.  Controls horizontal Y level.
            track=0 → horizontal at ``gsc_y`` (Q→INV_R)
            track=1 → one track higher (Q_→INV_L, avoids overlap)
    """
    # ── Target routing layer ───────────────────────────────────────────────────
    target_layer = spec.layer or "met2"
    target_rules_d = getattr(rules, target_layer, None) or {}
    if not isinstance(target_rules_d, dict):
        target_rules_d = {}
    target_w  = target_rules_d.get("width_min_um", 0.14)
    target_sp = target_rules_d.get("spacing_min_um", 0.14)
    lyr_target = rules.layer(target_layer)

    # ── Geometry constants ────────────────────────────────────────────────────
    c_size   = rules.contacts["size_um"]
    enc_poly_2adj, enc_poly_opp = rules.enclosure("contacts", "poly_enclosure")
    enc_li_licon_2adj, enc_li_licon_opp = rules.enclosure("contacts", "enclosure_in_li1")
    enc_m1_mcon_2adj, _ = rules.enclosure("met1", "enclosure_of_mcon")
    met1_sp  = rules.met1.get("spacing_min_um", 0.14)
    ch       = c_size / 2
    cc_pad_half_x = (c_size + 2 * enc_poly_2adj) / 2
    cc_pad_half_y = ch + enc_poly_opp
    rail_h   = max(rules.met1.get("width_min_um", 0.14),
                   rules.li1.get("width_min_um", 0.17))
    li1_land_half_2adj = ch + enc_li_licon_2adj
    li1_land_half_opp  = ch + enc_li_licon_opp

    # Pre-compute maximum landing half from li1→target via stack for spacing calc
    transitions = rules.via_stack_between("li1", target_layer)
    max_land_half = ch + enc_m1_mcon_2adj  # at least the mcon landing
    for t in transitions:
        vh = t.via_size / 2
        for metal_name, enc in [(t.lower_metal, t.enc_lower), (t.upper_metal, t.enc_upper)]:
            m_rules = getattr(rules, metal_name, {})
            m_w = m_rules.get("width_min_um", 0.0) if isinstance(m_rules, dict) else 0.0
            max_land_half = max(max_land_half, vh + enc, m_w / 2)

    poly_sp  = rules.poly.get("spacing_min_um", 0.21)

    lyr_g       = rules.layer("poly")
    lyr_li1     = rules.layer("li1")
    lyr_contact = rules.layer("licon1")

    # Collect all gate poly X ranges for spacing checks
    all_gates = _collect_gate_poly_ranges(placed)

    # ── Cell top Y: highest PMOS body top ─────────────────────────────────────
    pmos_devs = [d for d in placed.values() if d.spec.device_type == "pmos"]
    if not pmos_devs:
        return []
    cell_ytop = max(d.y + d.geom.total_y_um for d in pmos_devs)

    # Gate stub contact centre Y — must clear VDD rail top + met1 spacing
    gsc_y = cell_ytop + rail_h + met1_sp + max_land_half

    # ── Source: Q/Q_ li1 bridge centre ────────────────────────────────────────
    pg_dev = pd_dev = None
    for dev in placed.values():
        if dev.spec.device_type != "nmos":
            continue
        if dev.spec.terminals.get("S") == spec.net:
            pg_dev = dev
        if dev.spec.terminals.get("D") == spec.net:
            pd_dev = dev

    if pg_dev is None or pd_dev is None:
        warnings.warn(
            f"cross_couple_gate: cannot locate Q node devices for net {spec.net!r}; "
            f"skipped.",
            stacklevel=3,
        )
        return []

    t_drain  = resolve_terminal(f"{pd_dev.name}.D", placed, rules)
    t_source = resolve_terminal(f"{pg_dev.name}.S", placed, rules)
    q_x      = (t_drain.x1 + t_source.x0) / 2
    nd_ymid  = (t_drain.y0 + t_drain.y1) / 2

    # Via stack at Q/Q_ source node: li1 → target layer
    src_top_half = draw_via_stack(comp, rules, q_x, nd_ymid, "li1", target_layer)

    # ── Target gate stubs (one per unique gate X position) ────────────────────
    gate_xs: list[float] = []
    seen_x:  set[int]   = set()
    gate_top_half = 0.0

    for ref in spec.path:
        parts = ref.split(".", 1)
        if len(parts) != 2 or parts[1] != "G":
            continue
        dev = placed.get(parts[0])
        if dev is None:
            continue

        gx0, gx1 = global_gate_x(dev, 0)
        gcx      = (gx0 + gx1) / 2
        gcx_nm   = round(gcx * 1000)
        if gcx_nm in seen_x:
            continue
        seen_x.add(gcx_nm)

        gcx = _nudge_for_poly_spacing(gcx, cc_pad_half_x, (gx0, gx1),
                                      all_gates, poly_sp)
        gate_xs.append(gcx)

        # Poly stub widened above PMOS body
        _rect(comp, gcx - cc_pad_half_x, gcx + cc_pad_half_x,
                    cell_ytop, gsc_y + cc_pad_half_y, lyr_g)
        # Licon1 (polycontact: poly → li1)
        _rect(comp, gcx - ch, gcx + ch, gsc_y - ch, gsc_y + ch, lyr_contact)
        # Li1 landing
        _rect(comp, gcx - li1_land_half_2adj, gcx + li1_land_half_2adj,
                    gsc_y - li1_land_half_opp, gsc_y + li1_land_half_opp, lyr_li1)
        # Via stack from li1 → target layer
        lh = draw_via_stack(comp, rules, gcx, gsc_y, "li1", target_layer)
        gate_top_half = max(gate_top_half, lh)

    if not gate_xs:
        return []

    gcx_target = gate_xs[0]
    track      = int(spec.extra.get("track", 0))
    # Track pitch: clear the gate-stub landing + target layer spacing + wire half-width
    landing_half = max(gate_top_half, src_top_half, target_w / 2)
    track_pitch = landing_half + target_sp + target_w / 2
    route_y    = gsc_y + track * track_pitch

    # ── Route on target layer: source → gate stub ────────────────────────────
    hw = target_w / 2
    x_lo = min(q_x, gcx_target) - hw
    x_hi = max(q_x, gcx_target) + hw

    if track == 0:
        # L-shape
        _rect(comp, q_x - hw, q_x + hw, nd_ymid, route_y + hw, lyr_target)
        _rect(comp, x_lo, x_hi, route_y - hw, route_y + hw, lyr_target)
    else:
        # U-shape
        _rect(comp, q_x - hw, q_x + hw, nd_ymid, route_y + hw, lyr_target)
        _rect(comp, x_lo, x_hi, route_y - hw, route_y + hw, lyr_target)
        _rect(comp, gcx_target - hw, gcx_target + hw,
                    gsc_y - hw, route_y + hw, lyr_target)

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


@_style("vertical_met2_bus")
def _vertical_met2_bus(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Full-height vertical stripe at a S/D terminal (BL/BL_ bitline).

    Draws a via stack from li1 up to the target layer specified in
    ``spec.layer`` (defaults to ``"met2"``) at the terminal, and a
    full-height stripe on the target layer.

    Expected path: ``[Dev.Terminal]`` (single element).
    Extra: ``cell_y0``, ``cell_y1`` — cell Y bounds.
    """
    if not spec.path:
        return []

    try:
        t = resolve_terminal(spec.path[0], placed, rules)
    except (KeyError, ValueError) as exc:
        warnings.warn(
            f"vertical_met2_bus (net={spec.net!r}): {exc}; skipped.",
            stacklevel=3,
        )
        return []

    cx = (t.x0 + t.x1) / 2
    cy = (t.y0 + t.y1) / 2

    # Target layer from template (defaults to met2 for backward compat)
    target_layer = spec.layer or "met2"

    # Via stack from li1 up to target layer
    top_half = draw_via_stack(comp, rules, cx, cy, "li1", target_layer)

    # Target layer min width
    target_rules = getattr(rules, target_layer, None) or {}
    if isinstance(target_rules, dict):
        target_w_min = target_rules.get("width_min_um", 0.14)
    else:
        target_w_min = 0.14
    stripe_hw = max(target_w_min / 2, top_half)

    # Full-height stripe on target layer
    lyr_target = rules.layer(target_layer)
    cell_y0 = spec.extra.get("cell_y0", t.y0) if spec.extra else t.y0
    cell_y1 = spec.extra.get("cell_y1", t.y1) if spec.extra else t.y1
    # Extend beyond cell bounds by rail height so stripe reaches power rails
    rail_h = max(rules.met1.get("width_min_um", 0.14),
                 rules.li1.get("width_min_um", 0.17))
    stripe_y0 = cell_y0 - rail_h
    stripe_y1 = cell_y1 + rail_h
    _rect(comp, cx - stripe_hw, cx + stripe_hw, stripe_y0, stripe_y1, lyr_target)

    return [PortCandidate(
        net=spec.net,
        location_key=f"{spec.net}_bitline_center",
        x=cx,
        y=(stripe_y0 + stripe_y1) / 2,
        layer=target_layer,
        width=stripe_hw * 2,
        orientation=90,
    )]


# ── Phase-3 routing styles: cross-row connections ─────────────────────────────

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
    """
    if len(spec.path) < 2:
        return []

    # Target routing layer from template
    target_layer = spec.layer or "met1"
    target_rules_d = getattr(rules, target_layer, None) or {}
    if not isinstance(target_rules_d, dict):
        target_rules_d = {}
    target_w = target_rules_d.get("width_min_um", 0.14)
    trunk_hw = target_w / 2
    lyr_trunk = rules.layer(target_layer)

    # ── Geometry constants ─────────────────────────────────────────────────
    c_size    = rules.contacts["size_um"]
    enc_poly_2adj, enc_poly_opp = rules.enclosure("contacts", "poly_enclosure")
    enc_li_2adj, enc_li_opp     = rules.enclosure("contacts", "enclosure_in_li1")
    li1_sp    = rules.li1.get("spacing_min_um", 0.17)

    ch        = c_size / 2
    cr_pad_half_x = (c_size + 2 * enc_poly_2adj) / 2
    cr_pad_half_y = ch + enc_poly_opp
    li1_lh_2adj   = ch + enc_li_2adj
    li1_lh_opp    = ch + enc_li_opp

    lyr_g     = rules.layer("poly")
    lyr_li1   = rules.layer("li1")
    lyr_licon = rules.layer("licon1")

    # ── Resolve source terminal ────────────────────────────────────────────
    try:
        t_src = resolve_terminal(spec.path[0], placed, rules)
    except (KeyError, ValueError):
        return []
    src_cx = (t_src.x0 + t_src.x1) / 2
    src_cy = (t_src.y0 + t_src.y1) / 2

    # Via stack at source: li1 → target layer
    lh = draw_via_stack(comp, rules, src_cx, src_cy, "li1", target_layer)
    trunk_hw = max(trunk_hw, lh)

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
                stub_cy = global_poly_bottom(dev) - li1_sp - li1_lh_2adj
            else:
                stub_cy = global_poly_top(dev) + li1_sp + li1_lh_2adj

            poly_body_top = global_poly_top(dev)
            poly_body_bot = global_poly_bottom(dev)
            if stub_cy > poly_body_top:
                _rect(comp, gcx - cr_pad_half_x, gcx + cr_pad_half_x,
                            poly_body_top, stub_cy + cr_pad_half_y, lyr_g)
            else:
                _rect(comp, gcx - cr_pad_half_x, gcx + cr_pad_half_x,
                            stub_cy - cr_pad_half_y, poly_body_bot, lyr_g)

            # Licon1 + li1 landing + via stack to target layer
            _rect(comp, gcx - ch, gcx + ch,
                        stub_cy - ch, stub_cy + ch, lyr_licon)
            _rect(comp, gcx - li1_lh_2adj, gcx + li1_lh_2adj,
                        stub_cy - li1_lh_opp, stub_cy + li1_lh_opp, lyr_li1)
            lh = draw_via_stack(comp, rules, gcx, stub_cy, "li1", target_layer)
            trunk_hw = max(trunk_hw, lh)

            target_locs.append((gcx, stub_cy))
        else:
            try:
                t_tgt = resolve_terminal(ref, placed, rules)
            except (KeyError, ValueError):
                continue
            tgt_cx = (t_tgt.x0 + t_tgt.x1) / 2
            tgt_cy = (t_tgt.y0 + t_tgt.y1) / 2
            lh = draw_via_stack(comp, rules, tgt_cx, tgt_cy, "li1", target_layer)
            trunk_hw = max(trunk_hw, lh)
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
    """
    if len(spec.path) < 2:
        return []

    # Target layer from template (defaults to met1)
    target_layer = spec.layer or "met1"

    # Trunk width from target layer rules
    target_rules = getattr(rules, target_layer, None) or {}
    if isinstance(target_rules, dict):
        trunk_hw = target_rules.get("width_min_um", 0.14) / 2
    else:
        trunk_hw = 0.14 / 2
    lyr_trunk = rules.layer(target_layer)

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
        # Via stack from li1 up to target layer
        lh = draw_via_stack(comp, rules, tap_x, tap_y, "li1", target_layer)
        trunk_hw = max(trunk_hw, lh)
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
