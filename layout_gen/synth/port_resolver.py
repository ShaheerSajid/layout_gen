"""
layout_gen.synth.port_resolver — compass-side port placement.

Replaces the old location-key port matching with a geometric approach:
users specify a compass side (north/south/east/west) and optionally a
specific terminal, and the resolver computes exact (x, y) coordinates.

Also generates ``expose_terminal`` :class:`RoutingSpec` entries for
ports that need external access (Phase E of the auto-router).
"""
from __future__ import annotations

import warnings
from typing import Any

from layout_gen.pdk import PDKRules
from layout_gen.synth.loader import CellTemplate, RoutingSpec
from layout_gen.synth.placer import PlacedDevice, resolve_terminal
from layout_gen.synth.router import PortCandidate
from layout_gen.synth.netlist import NetGraph


# ── Side → orientation mapping ────────────────────────────────────────────────

_SIDE_ORIENTATION = {
    "west":  180,
    "east":  0,
    "north": 90,
    "south": 270,
    "left":  180,
    "right": 0,
    "top":   90,
    "bottom": 270,
}


# ── Port resolver ────────────────────────────────────────────────────────────

def resolve_ports(
    comp:       Any,               # gf.Component
    template:   CellTemplate,
    net_graph:  NetGraph,
    placed:     dict[str, PlacedDevice],
    candidates: list[PortCandidate],
    rules:      PDKRules,
) -> None:
    """Add ports to *comp* using compass-side declarations.

    Falls back to matching routing candidates by net name if the
    compass-side resolution doesn't find a direct terminal match.
    """
    if not template.ports:
        return

    # Cell bounding box
    cell_x0 = min(d.x for d in placed.values())
    cell_x1 = max(d.x + d.geom.total_x_um for d in placed.values())
    cell_y0 = min(d.y for d in placed.values())
    cell_y1 = max(d.y + d.geom.total_y_um for d in placed.values())

    # Index candidates by net
    cand_by_net: dict[str, list[PortCandidate]] = {}
    for c in candidates:
        cand_by_net.setdefault(c.net, []).append(c)

    for port_name, pspec in template.ports.items():
        side = pspec.side
        orientation = _SIDE_ORIENTATION.get(side, 0)
        terminal_ref = pspec.terminal

        # Determine layer, x, y, width
        x = y = width = 0.0
        layer = ""

        net_info = net_graph.nets.get(port_name)

        if terminal_ref:
            # User specified which terminal to expose
            try:
                t = resolve_terminal(terminal_ref, placed, rules)
                x = (t.x0 + t.x1) / 2
                y = (t.y0 + t.y1) / 2
                layer = t.layer
                if orientation in (90, 270):
                    width = t.x1 - t.x0
                else:
                    width = t.y1 - t.y0
            except (KeyError, ValueError) as exc:
                warnings.warn(
                    f"Port {port_name!r}: cannot resolve terminal "
                    f"{terminal_ref!r}: {exc}. Skipped.",
                    stacklevel=3,
                )
                continue

        elif net_info and net_info.is_power:
            # Power net — find matching rail candidate
            rail_cands = cand_by_net.get(port_name, [])
            if rail_cands:
                c = rail_cands[0]
                x, y = c.x, c.y
                layer = c.layer
                width = c.width
                orientation = c.orientation
            else:
                continue

        elif net_info and net_info.gate_terminals:
            # Signal net with gate connections — use gate candidate
            gate_cands = cand_by_net.get(port_name, [])
            if gate_cands:
                # Pick the candidate that best matches the requested side
                c = _best_candidate_for_side(gate_cands, side,
                                             cell_x0, cell_x1,
                                             cell_y0, cell_y1)
                x, y = c.x, c.y
                layer = c.layer
                width = c.width
            else:
                # Fallback: find first gate terminal for this net
                gt = net_info.gate_terminals[0]
                try:
                    t = resolve_terminal(gt.ref, placed, rules)
                    x = (t.x0 + t.x1) / 2
                    y = (t.y0 + t.y1) / 2
                    layer = t.layer
                    width = t.y1 - t.y0 if orientation in (0, 180) else t.x1 - t.x0
                except (KeyError, ValueError):
                    continue

        else:
            # Try matching any candidate by net name
            net_cands = cand_by_net.get(port_name, [])
            if net_cands:
                c = _best_candidate_for_side(net_cands, side,
                                             cell_x0, cell_x1,
                                             cell_y0, cell_y1)
                x, y = c.x, c.y
                layer = c.layer
                width = c.width
            else:
                warnings.warn(
                    f"Port {port_name!r}: no candidate or terminal found. Skipped.",
                    stacklevel=3,
                )
                continue

        width = max(width, rules.li1.get("width_min_um", 0.17))

        # Explicit port layer from YAML overrides auto-detected layer
        if pspec.layer:
            layer = pspec.layer

        try:
            lyr = rules.layer(layer) if isinstance(layer, str) else layer
        except (KeyError, TypeError):
            lyr = (1, 0)

        comp.add_port(
            port_name,
            center=(x, y),
            width=width,
            orientation=orientation,
            layer=lyr,
        )


def generate_expose_specs(
    template:  CellTemplate,
    net_graph: NetGraph,
    placed:    dict[str, PlacedDevice],
) -> list[RoutingSpec]:
    """Generate expose_terminal RoutingSpecs for ports that reference
    specific terminals (Phase E of auto-routing)."""
    specs: list[RoutingSpec] = []

    if not template.ports:
        return specs

    for port_name, pspec in template.ports.items():
        if not pspec.terminal:
            continue  # no specific terminal — resolved from candidates

        orientation = _SIDE_ORIENTATION.get(pspec.side, 0)
        specs.append(RoutingSpec(
            net=port_name,
            style="expose_terminal",
            layer="li1",
            path=[pspec.terminal],
            extra={
                "orientation": orientation,
                "location_key": f"{port_name}_port",
            },
        ))

    return specs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_candidate_for_side(
    cands:   list[PortCandidate],
    side:    str,
    cell_x0: float,
    cell_x1: float,
    cell_y0: float,
    cell_y1: float,
) -> PortCandidate:
    """Pick the candidate closest to the requested cell edge."""
    if len(cands) == 1:
        return cands[0]

    def _dist(c: PortCandidate) -> float:
        if side in ("west", "left"):
            return abs(c.x - cell_x0)
        elif side in ("east", "right"):
            return abs(c.x - cell_x1)
        elif side in ("north", "top"):
            return abs(c.y - cell_y1)
        elif side in ("south", "bottom"):
            return abs(c.y - cell_y0)
        return 0.0

    return min(cands, key=_dist)
