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

    return [PortCandidate(
        net=spec.net,
        location_key="gate_left_edge_mid_y",
        x=gx0, y=gate_mid_y,
        layer="poly",
        width=gap,
        orientation=180,
    )]


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
    dx0, dx1 = global_sd_x(dev_n, j_n)

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


# ── Phase-2 stubs ─────────────────────────────────────────────────────────────

@_style("cross_couple_gate")
def _cross_couple_gate(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Phase-2 TODO: met1 cross-coupling via polycontact stub above PMOS body."""
    warnings.warn(
        f"cross_couple_gate (net={spec.net!r}) not yet implemented "
        f"in synthesizer (Phase-2); skipped.",
        stacklevel=3,
    )
    return []


@_style("poly_stub_met1_bus")
def _poly_stub_met1_bus(
    comp:   Any,
    spec:   RoutingSpec,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> list[PortCandidate]:
    """Phase-2 TODO: WL polycontact stub + met1 horizontal bus.

    The implementation is in ``bit_cell.py`` but not yet abstracted into
    this synthesizer.  The hand-coded bit cell still works; this stub allows
    template-driven synthesis to at least load and warn.
    """
    warnings.warn(
        f"poly_stub_met1_bus (net={spec.net!r}) not yet implemented "
        f"in synthesizer (Phase-2); skipped.",
        stacklevel=3,
    )
    return []
