"""
layout_gen.synth.geo.agent — Geometric DRC fix agents.

:class:`GeoFixAgent` is the abstract base.  :class:`RuleGeoAgent`
uses deterministic heuristics to fix DRC violations.

Design principle
----------------
Every agent receives:

1. The :class:`~.state.LayoutState` (mutable polygon set)
2. A :class:`~.violations.ViolationInfo` to fix
3. (optionally) PDK rules for dimensional constants

And returns a list of :class:`~.actions.Action` objects.  The caller
applies them and re-runs DRC.  The agent never calls DRC itself — the
:class:`~.loop.GeoFixLoop` handles the outer loop.

The rule-based agent works by pattern-matching the violation *category*:

- **spacing** — find the two closest same-layer shapes near the violation,
  move the smaller one away by ``deficit``.
- **width** — find the narrow shape, stretch its short edge by ``deficit/2``
  on each side.
- **enclosure** — find the inner shape (via/contact), stretch the outer
  shape's nearest edge by ``deficit``.
- **area** — find the small shape, scale its shorter dimension up until
  ``area >= required``.
- **overlap** — find the two overlapping shapes, move the smaller one.

These operations are universal across CMOS technologies.
"""
from __future__ import annotations

import abc
from typing import Any

from layout_gen.synth.geo.state      import LayoutState, Rect
from layout_gen.synth.geo.actions    import (
    Action, StretchEdge, MoveShape, MoveGroup, AddRect, MergeShapes,
    SnapToGrid, ResizeContact, LayerPromote,
)
from layout_gen.synth.geo.violations import ViolationInfo


_EPS = 0.005  # 5 nm headroom beyond exact deficit


class GeoFixAgent(abc.ABC):
    """Abstract base for geometric DRC fix agents.

    Subclasses implement :meth:`propose_fix` — given a layout state and a
    parsed violation, return one or more actions that should fix it.
    """

    @abc.abstractmethod
    def propose_fix(
        self,
        state:     LayoutState,
        violation: ViolationInfo,
    ) -> list[Action]:
        """Propose geometric actions to fix *violation*.

        Returns an empty list if no fix is known.
        """

    def fix_batch(
        self,
        state:      LayoutState,
        violations: list[ViolationInfo],
    ) -> list[Action]:
        """Propose fixes for all violations (default: one at a time)."""
        actions: list[Action] = []
        for v in violations:
            actions.extend(self.propose_fix(state, v))
        return actions


class RuleGeoAgent(GeoFixAgent):
    """Phase 2: deterministic geometric fixer.

    Reads the violation category and applies the obvious geometric
    transformation — the same thing a human layout engineer would do
    when they see a DRC error marker in KLayout.

    Parameters
    ----------
    rules :
        PDK rules (optional, for dimensional lookups).
    search_radius :
        How far from the violation centroid to search for shapes (µm).
    """

    def __init__(self, rules: Any = None, search_radius: float = 2.0):
        self.rules = rules
        self.search_radius = search_radius

    def propose_fix(
        self,
        state:     LayoutState,
        violation: ViolationInfo,
    ) -> list[Action]:
        cat = violation.category

        # Detect contact/via size violations by rule name pattern
        if self._is_contact_size_violation(violation):
            return self._fix_contact_size(state, violation)

        if cat == "spacing":
            actions = self._fix_spacing(state, violation)
            return self._check_connectivity(state, actions)
        elif cat == "width":
            return self._fix_width(state, violation)
        elif cat == "enclosure" or cat == "extension":
            return self._fix_enclosure(state, violation)
        elif cat == "area":
            return self._fix_area(state, violation)
        elif cat == "overlap":
            return self._fix_overlap(state, violation)
        elif cat == "offgrid":
            return self._fix_offgrid(state, violation)
        else:
            return self._fix_unknown(state, violation)

    # ── Contact/via size detection ───────────────────────────────────────────

    @staticmethod
    def _is_contact_size_violation(v: ViolationInfo) -> bool:
        """True if this violation is about a contact/via being the wrong size."""
        rule_lower = v.rule.lower()
        desc_lower = (v.raw.description or "").lower() if v.raw else ""
        # sky130: licon.1, mcon.1;  TSMC: CO.W.1, VIA.W.1, etc.
        size_keywords = ("size", "licon.1", "mcon.1", "co.w", "via.w")
        if any(kw in rule_lower for kw in size_keywords):
            return True
        if v.category == "width" and v.layer in ("licon1", "mcon", "via1", "via2"):
            return True
        if "size" in desc_lower and any(
            cl in v.layer for cl in ("licon", "mcon", "via", "contact")
        ):
            return True
        return False

    # ── Contact size fix ────────────────────────────────────────────────────

    def _fix_contact_size(
        self, state: LayoutState, v: ViolationInfo,
    ) -> list[Action]:
        """Fix contact/via size violation by resizing to correct PDK dimension.

        Contacts have fixed sizes — stretching them is wrong.  Instead, we
        delete and redraw at the correct size, centred on the same location.
        """
        # Determine correct contact size from PDK rules
        target_size = v.required
        if target_size <= 0 and self.rules is not None:
            target_size = self.rules.contacts.get("size_um", 0.17)
        if target_size <= 0:
            target_size = 0.17  # safe fallback

        # Find the undersized contact(s) near the violation
        shapes = state.near(v.x, v.y, self.search_radius, layer=v.layer)
        if not shapes:
            shapes = state.on_layer(v.layer)

        actions: list[Action] = []
        for s in shapes:
            dim = min(s.width, s.height)
            if dim < target_size - _EPS:
                actions.append(ResizeContact(rid=s.rid,
                                             target_size=target_size))
        return actions

    # ── Connectivity-aware action filter ──────────────────────────────────

    def _check_connectivity(
        self, state: LayoutState, actions: list[Action],
    ) -> list[Action]:
        """Filter out MoveShape actions that would disconnect a net.

        If moving a shape would break all connections to shapes on the same
        net, fall back to StretchEdge on the facing edge instead.
        """
        safe_actions: list[Action] = []
        for action in actions:
            if not isinstance(action, MoveShape):
                safe_actions.append(action)
                continue

            r = state[action.rid]
            if not r.net:
                # No net info — can't check connectivity, allow the move
                safe_actions.append(action)
                continue

            # Check how many same-net shapes this rect touches
            connected = [s for s in state.connected_shapes(r.rid)
                         if s.net == r.net]
            if len(connected) <= 1:
                # Only one connection — moving would isolate it.
                # Fall back to StretchEdge: shrink the facing edge instead.
                # Determine which edge faces the conflict direction
                if abs(action.dx) >= abs(action.dy):
                    edge = "right" if action.dx < 0 else "left"
                    delta = abs(action.dx)
                else:
                    edge = "top" if action.dy < 0 else "bottom"
                    delta = abs(action.dy)
                safe_actions.append(StretchEdge(r.rid, edge, -delta))
            else:
                safe_actions.append(action)
        return safe_actions

    # ── Spacing ──────────────────────────────────────────────────────────────

    def _fix_spacing(self, state: LayoutState, v: ViolationInfo) -> list[Action]:
        """Spacing violation: two shapes too close → shrink facing edges.

        Prefers StretchEdge (shrinking the facing edge inward) over MoveShape,
        because moving an entire shape breaks contact/via alignment and causes
        cascading violations.  Falls back to MoveShape only when shrinking
        would make a shape narrower than the layer's minimum width.
        """
        layer = v.layer
        shapes = state.near(v.x, v.y, self.search_radius, layer=layer)
        if len(shapes) < 2:
            # Widen search
            shapes = state.on_layer(layer)

        if len(shapes) < 2:
            return []

        # Find the closest pair
        best_pair = None
        best_dist = float("inf")
        for i, a in enumerate(shapes):
            for b in shapes[i + 1:]:
                d = a.edge_dist(b)
                if 0 < d < best_dist:
                    best_dist = d
                    best_pair = (a, b)

        if best_pair is None:
            return []

        a, b = best_pair
        # Total gap increase needed
        deficit = v.deficit + _EPS if v.deficit > 0 else v.required - best_dist + _EPS
        if deficit <= 0:
            deficit = 0.01

        # ── Via group handling: if one shape is a via_pad, move its group ──
        via_pad = None
        gate = None
        if a.shape_type == "via_pad" and a.group_id >= 0:
            via_pad, gate = a, b
        elif b.shape_type == "via_pad" and b.group_id >= 0:
            via_pad, gate = b, a

        if via_pad is not None:
            # Push the via group away from the gate poly
            dx, dy = self._repulsion_vector(via_pad, gate, v.required + _EPS)
            return [MoveGroup(group_id=via_pad.group_id, dx=dx, dy=dy)]

        # Determine separation axis: which edges face each other?
        # X-separated: a.x1 ≤ b.x0 or b.x1 ≤ a.x0
        # Y-separated: a.y1 ≤ b.y0 or b.y1 ≤ a.y0
        gap_x = max(a.x0 - b.x1, b.x0 - a.x1)  # positive = separated in X
        gap_y = max(a.y0 - b.y1, b.y0 - a.y1)  # positive = separated in Y

        # Pick the axis with the actual spacing gap (positive = separated)
        if gap_x > 0 and (gap_y <= 0 or gap_x <= gap_y):
            # Separated in X — shrink facing edges
            if a.cx < b.cx:
                left, right = a, b  # a is left, b is right
            else:
                left, right = b, a
            return self._stretch_facing_edges(
                state, left, "right", right, "left", deficit, "width")
        elif gap_y > 0:
            # Separated in Y — shrink facing edges
            if a.cy < b.cy:
                bot, top = a, b
            else:
                bot, top = b, a
            return self._stretch_facing_edges(
                state, bot, "top", top, "bottom", deficit, "height")
        else:
            # Overlapping — fall back to MoveShape
            mover, anchor = (a, b) if a.area <= b.area else (b, a)
            needed = v.required + _EPS if v.required > 0 else deficit
            dx, dy = self._repulsion_vector(mover, anchor, needed)
            return [MoveShape(rid=mover.rid, dx=dx, dy=dy)]

    def _stretch_facing_edges(
        self,
        state:      LayoutState,
        shape_a:    Rect,
        edge_a:     str,       # edge of shape_a that faces shape_b
        shape_b:    Rect,
        edge_b:     str,       # edge of shape_b that faces shape_a
        deficit:    float,
        dim_attr:   str,       # "width" or "height" — the dimension being shrunk
    ) -> list[Action]:
        """Shrink facing edges to increase gap by *deficit*.

        Splits the shrink between both shapes when possible.  Falls back to
        MoveShape on the smaller shape if shrinking would make either shape
        narrower than a safe minimum (contact size + 2× enclosure).
        """
        min_dim = 0.0
        if self.rules is not None:
            c_size = self.rules.contacts.get("size_um", 0.17)
            c_enc  = self.rules.contacts.get("enclosure_in_diff_um", 0.06)
            min_dim = c_size + 2 * c_enc  # minimum safe dimension

        dim_a = getattr(shape_a, dim_attr)
        dim_b = getattr(shape_b, dim_attr)

        # How much each shape can safely shrink
        slack_a = max(0.0, dim_a - min_dim)
        slack_b = max(0.0, dim_b - min_dim)
        total_slack = slack_a + slack_b

        if total_slack >= deficit:
            # Split deficit proportionally to available slack
            if slack_a >= deficit and slack_b >= deficit:
                # Both can absorb it — split evenly
                half = deficit / 2
                return [
                    StretchEdge(shape_a.rid, edge_a, -half),
                    StretchEdge(shape_b.rid, edge_b, -half),
                ]
            elif slack_a >= deficit:
                # Only A can absorb the full deficit
                return [StretchEdge(shape_a.rid, edge_a, -deficit)]
            elif slack_b >= deficit:
                return [StretchEdge(shape_b.rid, edge_b, -deficit)]
            else:
                # Split: each absorbs what it can
                return [
                    StretchEdge(shape_a.rid, edge_a, -slack_a),
                    StretchEdge(shape_b.rid, edge_b, -(deficit - slack_a)),
                ]
        else:
            # Not enough slack — fall back to MoveShape on the smaller shape
            mover, anchor = ((shape_a, shape_b) if shape_a.area <= shape_b.area
                             else (shape_b, shape_a))
            needed = deficit
            dx, dy = self._repulsion_vector(mover, anchor, needed)
            return [MoveShape(rid=mover.rid, dx=dx, dy=dy)]

    # ── Width ────────────────────────────────────────────────────────────────

    def _fix_width(self, state: LayoutState, v: ViolationInfo) -> list[Action]:
        """Width violation: a shape is too narrow → widen it."""
        shapes = state.near(v.x, v.y, self.search_radius, layer=v.layer)
        if not shapes:
            shapes = state.on_layer(v.layer)

        # Find the narrowest shape
        target = None
        min_dim = float("inf")
        for s in shapes:
            d = min(s.width, s.height)
            if d < min_dim:
                min_dim = d
                target = s

        if target is None:
            return []

        needed = v.deficit + _EPS
        if needed <= 0:
            needed = v.required - min_dim + _EPS if v.required > 0 else 0.01

        # Stretch the narrow dimension symmetrically
        actions: list[Action] = []
        half = needed / 2
        if target.width <= target.height:
            # Narrow in X
            actions.append(StretchEdge(target.rid, "left", half))
            actions.append(StretchEdge(target.rid, "right", half))
        else:
            # Narrow in Y
            actions.append(StretchEdge(target.rid, "bottom", half))
            actions.append(StretchEdge(target.rid, "top", half))

        return actions

    # ── Enclosure ────────────────────────────────────────────────────────────

    def _fix_enclosure(self, state: LayoutState, v: ViolationInfo) -> list[Action]:
        """Enclosure violation: outer layer doesn't cover inner layer enough."""
        inner_layer = v.inner_layer or ""
        outer_layer = v.layer

        # Find the inner shape (via/contact) near the violation
        inner_shapes = state.near(v.x, v.y, self.search_radius,
                                  layer=inner_layer) if inner_layer else []

        if not inner_shapes:
            # Try all layers near the point
            inner_shapes = state.at_point(v.x, v.y)

        if not inner_shapes:
            return []

        # Pick inner shape closest to violation point (not smallest area)
        inner = min(inner_shapes,
                    key=lambda s: (s.cx - v.x)**2 + (s.cy - v.y)**2)

        # Find the outer shape
        outer_shapes = state.near(inner.cx, inner.cy, self.search_radius,
                                  layer=outer_layer)
        if not outer_shapes:
            return []

        outer = min(outer_shapes,
                    key=lambda s: ((s.cx - inner.cx)**2 + (s.cy - inner.cy)**2))

        needed = v.deficit + _EPS
        if needed <= 0:
            needed = 0.01

        # Check each edge of outer: if it doesn't enclose inner by enough,
        # stretch it outward
        actions: list[Action] = []

        # Left edge: outer.x0 must be <= inner.x0 - required
        left_gap = inner.x0 - outer.x0
        if left_gap < v.required:
            actions.append(StretchEdge(outer.rid, "left", v.required - left_gap + _EPS))

        # Right edge: outer.x1 must be >= inner.x1 + required
        right_gap = outer.x1 - inner.x1
        if right_gap < v.required:
            actions.append(StretchEdge(outer.rid, "right", v.required - right_gap + _EPS))

        # Bottom edge
        bot_gap = inner.y0 - outer.y0
        if bot_gap < v.required:
            actions.append(StretchEdge(outer.rid, "bottom", v.required - bot_gap + _EPS))

        # Top edge
        top_gap = outer.y1 - inner.y1
        if top_gap < v.required:
            actions.append(StretchEdge(outer.rid, "top", v.required - top_gap + _EPS))

        return actions

    # ── Area ─────────────────────────────────────────────────────────────────

    def _fix_area(self, state: LayoutState, v: ViolationInfo) -> list[Action]:
        """Area violation: shape is too small → enlarge it."""
        shapes = state.near(v.x, v.y, self.search_radius, layer=v.layer)
        if not shapes:
            shapes = state.on_layer(v.layer)

        # Find the smallest shape
        target = min(shapes, key=lambda s: s.area) if shapes else None
        if target is None:
            return []

        required_area = v.required
        if required_area <= 0:
            return []

        current_area = target.area
        if current_area >= required_area:
            return []

        # Scale up the shorter dimension
        ratio = (required_area / current_area) ** 0.5
        actions: list[Action] = []
        if target.width <= target.height:
            extra = target.width * (ratio - 1) / 2 + _EPS
            actions.append(StretchEdge(target.rid, "left", extra))
            actions.append(StretchEdge(target.rid, "right", extra))
        else:
            extra = target.height * (ratio - 1) / 2 + _EPS
            actions.append(StretchEdge(target.rid, "bottom", extra))
            actions.append(StretchEdge(target.rid, "top", extra))

        return actions

    # ── Overlap ──────────────────────────────────────────────────────────────

    def _fix_overlap(self, state: LayoutState, v: ViolationInfo) -> list[Action]:
        """Overlap/short violation: two shapes touch that shouldn't."""
        shapes = state.near(v.x, v.y, self.search_radius, layer=v.layer)

        # Find overlapping pair
        for i, a in enumerate(shapes):
            for b in shapes[i + 1:]:
                if a.overlaps(b) and a.edge_dist(b) == 0:
                    # Move the smaller one away
                    mover = a if a.area <= b.area else b
                    anchor = b if mover is a else a
                    dx, dy = self._repulsion_vector(mover, anchor, 0.01)
                    return [MoveShape(mover.rid, dx, dy)]
        return []

    # ── Off-grid ────────────────────────────────────────────────────────────

    def _fix_offgrid(self, state: LayoutState, v: ViolationInfo) -> list[Action]:
        """Off-grid violation: snap all shapes near the violation to grid."""
        shapes = state.near(v.x, v.y, self.search_radius, layer=v.layer)
        if not shapes:
            shapes = state.near(v.x, v.y, self.search_radius)
        if not shapes:
            # Snap everything
            return [SnapToGrid(rid=-1)]
        return [SnapToGrid(rid=s.rid) for s in shapes]

    # ── Layer promotion for persistent spacing violations ───────────────────

    def _try_layer_promote(
        self, state: LayoutState, v: ViolationInfo,
    ) -> list[Action]:
        """Suggest promoting a met1 wire to met2 if spacing can't be fixed.

        Only applies to met1 spacing violations where StretchEdge/MoveShape
        have already been tried (caller should check).
        """
        if v.layer not in ("met1", "m1", "metal1"):
            return []

        shapes = state.near(v.x, v.y, self.search_radius, layer=v.layer)
        wires = [s for s in shapes if s.shape_type == "wire"]
        if not wires:
            wires = shapes  # fall back to any shape

        if len(wires) < 2:
            return []

        # Promote the smaller wire
        target = min(wires, key=lambda s: s.area)

        via_size = 0.15
        via_enc  = 0.085
        if self.rules is not None:
            via_size = self.rules.contacts.get("via1_size_um",
                       self.rules.contacts.get("size_um", 0.15))
            via_enc  = self.rules.contacts.get("via1_enclosure_um", 0.085)

        return [LayerPromote(
            rid=target.rid,
            from_layer="met1",
            to_layer="met2",
            via_layer="via1",
            via_size=via_size,
            via_enclosure=via_enc,
        )]

    # ── Unknown ──────────────────────────────────────────────────────────────

    def _fix_unknown(self, state: LayoutState, v: ViolationInfo) -> list[Action]:
        """Fallback: try spacing fix, then width fix, then layer promote."""
        actions = self._fix_spacing(state, v)
        if actions:
            return actions
        actions = self._fix_width(state, v)
        if actions:
            return actions
        return self._try_layer_promote(state, v)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _repulsion_vector(
        mover: Rect, anchor: Rect, min_gap: float,
    ) -> tuple[float, float]:
        """Compute (dx, dy) to push *mover* away from *anchor* by *min_gap*.

        Moves in the direction of shortest separation.
        """
        # Overlaps in X and Y
        ox = min(mover.x1, anchor.x1) - max(mover.x0, anchor.x0)
        oy = min(mover.y1, anchor.y1) - max(mover.y0, anchor.y0)

        # Gaps in X and Y
        gx_left  = anchor.x0 - mover.x1  # gap if mover is left
        gx_right = mover.x0 - anchor.x1  # gap if mover is right
        gy_below = anchor.y0 - mover.y1
        gy_above = mover.y0 - anchor.y1

        # Choose the direction with smallest displacement needed
        candidates = []
        if gx_left >= 0 or ox > 0:
            # mover is left of anchor (or overlapping)
            delta = min_gap - gx_left if gx_left >= 0 else min_gap + abs(gx_left)
            candidates.append((-delta, 0.0, abs(delta)))
        if gx_right >= 0 or ox > 0:
            delta = min_gap - gx_right if gx_right >= 0 else min_gap + abs(gx_right)
            candidates.append((delta, 0.0, abs(delta)))
        if gy_below >= 0 or oy > 0:
            delta = min_gap - gy_below if gy_below >= 0 else min_gap + abs(gy_below)
            candidates.append((0.0, -delta, abs(delta)))
        if gy_above >= 0 or oy > 0:
            delta = min_gap - gy_above if gy_above >= 0 else min_gap + abs(gy_above)
            candidates.append((0.0, delta, abs(delta)))

        if not candidates:
            # Default: push right
            return (min_gap, 0.0)

        # Pick the move with the smallest displacement
        candidates.sort(key=lambda c: c[2])
        return (candidates[0][0], candidates[0][1])
