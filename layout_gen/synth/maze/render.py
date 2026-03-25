"""Render maze-routed paths to gdsfactory polygons."""
from __future__ import annotations

from typing import Any

from layout_gen.pdk import PDKRules
from layout_gen.cells.standard import _rect
from layout_gen.synth.maze.types import NetRoute, RouteSegment, ViaLocation


def render_route(
    comp: Any,
    route: NetRoute,
    rules: PDKRules,
) -> None:
    """Draw all segments and vias of a routed net onto *comp*."""
    for seg in route.segments:
        _render_segment(comp, seg, rules)
    for via in route.vias:
        _render_via(comp, via, rules)


def _render_segment(
    comp: Any,
    seg: RouteSegment,
    rules: PDKRules,
) -> None:
    """Draw one wire segment as a rectangle."""
    lyr_rules = getattr(rules, seg.layer, None)
    if isinstance(lyr_rules, dict):
        w_min = lyr_rules.get("width_min_um", 0.17)
    else:
        w_min = 0.17
    w = max(seg.width, w_min)
    hw = w / 2

    lyr = rules.layer(seg.layer)

    # Determine segment orientation
    if abs(seg.x1 - seg.x0) > abs(seg.y1 - seg.y0):
        # Horizontal segment
        x0 = min(seg.x0, seg.x1)
        x1 = max(seg.x0, seg.x1)
        cy = (seg.y0 + seg.y1) / 2
        _rect(comp, x0, x1, cy - hw, cy + hw, lyr)
    else:
        # Vertical segment
        y0 = min(seg.y0, seg.y1)
        y1 = max(seg.y0, seg.y1)
        cx = (seg.x0 + seg.x1) / 2
        _rect(comp, cx - hw, cx + hw, y0, y1, lyr)


def _render_via(
    comp: Any,
    via: ViaLocation,
    rules: PDKRules,
) -> None:
    """Draw a via stack between two layers."""
    # Reuse the existing draw_via_stack from router.py
    from layout_gen.synth.router import draw_via_stack
    draw_via_stack(comp, rules, via.cx, via.cy, via.from_layer, via.to_layer)
