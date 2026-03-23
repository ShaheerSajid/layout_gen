"""
layout_gen.synth.geo.actions — Geometric layout operations.

Every action the agent can take is represented as a dataclass.  The
:func:`apply_action` function mutates a :class:`~.state.LayoutState`
accordingly.

The action set is **technology-agnostic** — the same operations fix DRC
violations in sky130, TSMC 65 nm, or any other CMOS process:

- :class:`StretchEdge` — move one edge of a rectangle (the fundamental fix)
- :class:`MoveShape` — translate a rectangle
- :class:`AddRect` — insert a new rectangle on a layer
- :class:`RemoveShape` — delete a rectangle
- :class:`MergeShapes` — replace overlapping rects with their bounding box

Phase 3 learned agents choose from this same action set — no new actions
are needed, just a better *policy* for selecting which action to apply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from layout_gen.synth.geo.state import LayoutState, Rect


# ── Action types ─────────────────────────────────────────────────────────────

@dataclass
class StretchEdge:
    """Move one edge of a rectangle by *delta* µm.

    The fundamental DRC-fix operation: spacing violation → stretch the
    near edge outward; width violation → stretch the short edge outward;
    enclosure violation → stretch the enclosing shape's edge.

    Parameters
    ----------
    rid : int
        Rectangle ID.
    edge : str
        Which edge: ``"left"`` (x0), ``"right"`` (x1),
        ``"bottom"`` (y0), ``"top"`` (y1).
    delta : float
        Distance to move the edge.  Positive = outward (makes shape larger),
        negative = inward (makes shape smaller).
    """
    rid:   int
    edge:  str      # "left", "right", "bottom", "top"
    delta: float    # µm, positive = outward

    def describe(self) -> str:
        direction = "outward" if self.delta > 0 else "inward"
        return (f"Stretch {self.edge} edge of rect {self.rid} "
                f"by {abs(self.delta):.3f} µm {direction}")


@dataclass
class MoveShape:
    """Translate a rectangle by ``(dx, dy)`` µm."""
    rid:  int
    dx:   float
    dy:   float

    def describe(self) -> str:
        return f"Move rect {self.rid} by ({self.dx:.3f}, {self.dy:.3f}) µm"


@dataclass
class AddRect:
    """Insert a new rectangle on a given layer."""
    layer: str
    x0:    float
    y0:    float
    x1:    float
    y1:    float

    def describe(self) -> str:
        return (f"Add {self.layer} rect "
                f"({self.x0:.3f},{self.y0:.3f})-({self.x1:.3f},{self.y1:.3f})")


@dataclass
class RemoveShape:
    """Delete a rectangle."""
    rid: int

    def describe(self) -> str:
        return f"Remove rect {self.rid}"


@dataclass
class MergeShapes:
    """Replace a set of overlapping rectangles with their bounding box.

    Useful when adjacent same-layer shapes should be one continuous region
    (e.g. implant merging, power rail stitching).
    """
    rids:  list[int]
    layer: str | None = None  # override layer (or keep from first rect)

    def describe(self) -> str:
        return f"Merge rects {self.rids} into bounding box"


@dataclass
class SnapToGrid:
    """Snap all edges of a rectangle to the manufacturing grid.

    Off-grid coordinates cause DRC violations in production decks.
    Standard CMOS grid is 0.005 µm (5 nm); sky130 uses 0.001 µm (1 nm).

    Parameters
    ----------
    rid : int
        Rectangle ID.  If -1, snaps ALL rectangles in the state.
    grid : float
        Grid pitch in µm (default 0.005).
    """
    rid:   int    = -1
    grid:  float  = 0.005   # µm

    def describe(self) -> str:
        if self.rid == -1:
            return f"Snap all rects to {self.grid*1000:.0f} nm grid"
        return f"Snap rect {self.rid} to {self.grid*1000:.0f} nm grid"


# Union of all action types (for type annotations)
Action = Union[StretchEdge, MoveShape, AddRect, RemoveShape, MergeShapes, SnapToGrid]


# ── Apply ────────────────────────────────────────────────────────────────────

def apply_action(state: LayoutState, action: Action) -> Rect | None:
    """Apply *action* to *state* in place.  Returns the affected Rect."""

    if isinstance(action, StretchEdge):
        r = state[action.rid]
        edge_map = {
            "left":   ("x0", -1),
            "right":  ("x1",  1),
            "bottom": ("y0", -1),
            "top":    ("y1",  1),
        }
        attr, sign = edge_map[action.edge]
        new_val = getattr(r, attr) + sign * action.delta
        state.update(action.rid, **{attr: new_val})
        return state[action.rid]

    elif isinstance(action, MoveShape):
        r = state[action.rid]
        state.update(
            action.rid,
            x0=r.x0 + action.dx, x1=r.x1 + action.dx,
            y0=r.y0 + action.dy, y1=r.y1 + action.dy,
        )
        return state[action.rid]

    elif isinstance(action, AddRect):
        return state.add(action.layer, action.x0, action.y0,
                         action.x1, action.y1)

    elif isinstance(action, RemoveShape):
        return state.remove(action.rid)

    elif isinstance(action, MergeShapes):
        rects = [state[rid] for rid in action.rids if rid in state]
        if not rects:
            return None
        layer = action.layer or rects[0].layer
        x0 = min(r.x0 for r in rects)
        y0 = min(r.y0 for r in rects)
        x1 = max(r.x1 for r in rects)
        y1 = max(r.y1 for r in rects)
        for r in rects:
            state.remove(r.rid)
        return state.add(layer, x0, y0, x1, y1)

    elif isinstance(action, SnapToGrid):
        g = action.grid
        def _snap(v: float) -> float:
            return round(round(v / g) * g, 6)

        targets = [state[action.rid]] if action.rid >= 0 else list(state)
        last = None
        for r in targets:
            state.update(r.rid,
                         x0=_snap(r.x0), y0=_snap(r.y0),
                         x1=_snap(r.x1), y1=_snap(r.y1))
            last = state[r.rid]
        return last

    else:
        raise TypeError(f"Unknown action type: {type(action)}")
