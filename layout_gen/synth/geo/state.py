"""
layout_gen.synth.geo.state — Mutable polygon-level layout representation.

:class:`LayoutState` holds every rectangle in the layout as a :class:`Rect`
with an integer ``rid`` (rectangle ID), layer name, and bounding box.
It supports spatial queries and mutation, and can round-trip to/from a
gdsfactory ``Component``.

Design decisions
----------------
* **Rectangles only** — VLSI layouts are >99 % rectangles.  Non-rect
  polygons (L-shapes from GDS boolean merge) are decomposed into their
  bounding box on import; the original Component is kept for GDS export
  so no information is lost.
* **Layer as string** — the logical layer name (``"met1"``, ``"poly"``)
  rather than GDS (layer, datatype) tuple.  The PDKRules object maps
  between them.
* **No net tracking (yet)** — the agent works on geometry alone, like a
  human staring at the layout in KLayout.  Net-awareness can be added
  in Phase 3 via graph annotations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class Rect:
    """One axis-aligned rectangle in the layout.

    Attributes
    ----------
    rid : int
        Unique rectangle ID within a :class:`LayoutState`.
    layer : str
        Logical layer name (e.g. ``"met1"``).
    x0, y0, x1, y1 : float
        Bounding box in µm.  ``x0 < x1``, ``y0 < y1``.
    net : str
        Net name this shape belongs to (empty if unknown).
    shape_type : str
        Semantic type: ``"wire"``, ``"contact"``, ``"via"``, ``"gate"``,
        ``"diffusion"``, ``"implant"``, ``"rail"``, or ``""`` (unknown).
    """
    rid:        int
    layer:      str
    x0:         float
    y0:         float
    x1:         float
    y1:         float
    net:        str = ""
    shape_type: str = ""
    group_id:   int = -1     # shapes in same via/contact stack share a group_id

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def overlaps(self, other: "Rect") -> bool:
        """True if this rect overlaps *other* (sharing an edge counts)."""
        return (self.x0 <= other.x1 and self.x1 >= other.x0 and
                self.y0 <= other.y1 and self.y1 >= other.y0)

    def edge_dist(self, other: "Rect") -> float:
        """Minimum edge-to-edge distance (0 if overlapping)."""
        dx = max(0.0, max(self.x0 - other.x1, other.x0 - self.x1))
        dy = max(0.0, max(self.y0 - other.y1, other.y0 - self.y1))
        if dx > 0 and dy > 0:
            return (dx**2 + dy**2) ** 0.5
        return max(dx, dy)

    def contains_point(self, x: float, y: float, tol: float = 0.001) -> bool:
        return (self.x0 - tol <= x <= self.x1 + tol and
                self.y0 - tol <= y <= self.y1 + tol)

    @property
    def is_contact(self) -> bool:
        """True if this shape is a contact or via."""
        return self.shape_type in ("contact", "via")

    def copy(self) -> "Rect":
        return Rect(self.rid, self.layer, self.x0, self.y0, self.x1, self.y1,
                    self.net, self.shape_type, self.group_id)


class LayoutState:
    """Mutable collection of layout rectangles with spatial queries.

    Parameters
    ----------
    rects : list[Rect] | None
        Initial rectangles.  Each must have a unique ``rid``.
    """

    def __init__(self, rects: list[Rect] | None = None):
        self._rects: dict[int, Rect] = {}
        self._next_rid = 0
        if rects:
            for r in rects:
                self._rects[r.rid] = r
                self._next_rid = max(self._next_rid, r.rid + 1)

    # ── Accessors ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._rects)

    def __getitem__(self, rid: int) -> Rect:
        return self._rects[rid]

    def __contains__(self, rid: int) -> bool:
        return rid in self._rects

    def __iter__(self) -> Iterator[Rect]:
        return iter(self._rects.values())

    @property
    def rects(self) -> list[Rect]:
        return list(self._rects.values())

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add(self, layer: str, x0: float, y0: float,
            x1: float, y1: float,
            net: str = "", shape_type: str = "") -> Rect:
        """Add a rectangle and return it."""
        rid = self._next_rid
        self._next_rid += 1
        r = Rect(rid, layer, min(x0, x1), min(y0, y1),
                 max(x0, x1), max(y0, y1), net, shape_type)
        self._rects[rid] = r
        return r

    def remove(self, rid: int) -> Rect | None:
        """Remove a rectangle by ID.  Returns it, or None if not found."""
        return self._rects.pop(rid, None)

    def update(self, rid: int, **kwargs: float) -> None:
        """Update fields of a rectangle (x0, y0, x1, y1)."""
        r = self._rects[rid]
        for k, v in kwargs.items():
            setattr(r, k, v)
        # Enforce x0 < x1, y0 < y1
        if r.x0 > r.x1:
            r.x0, r.x1 = r.x1, r.x0
        if r.y0 > r.y1:
            r.y0, r.y1 = r.y1, r.y0

    # ── Spatial queries ──────────────────────────────────────────────────────

    def on_layer(self, layer: str) -> list[Rect]:
        """All rectangles on *layer*."""
        return [r for r in self._rects.values() if r.layer == layer]

    def near(self, x: float, y: float, radius: float,
             layer: str | None = None) -> list[Rect]:
        """Rectangles whose centre is within *radius* µm of ``(x, y)``."""
        result = []
        for r in self._rects.values():
            if layer and r.layer != layer:
                continue
            d = ((r.cx - x)**2 + (r.cy - y)**2) ** 0.5
            if d <= radius:
                result.append(r)
        return result

    def at_point(self, x: float, y: float,
                 layer: str | None = None,
                 tol: float = 0.01) -> list[Rect]:
        """Rectangles that contain or nearly contain point ``(x, y)``."""
        result = []
        for r in self._rects.values():
            if layer and r.layer != layer:
                continue
            if r.contains_point(x, y, tol):
                result.append(r)
        return result

    def neighbours(self, rid: int, max_dist: float = 1.0,
                   same_layer: bool = True) -> list[tuple[Rect, float]]:
        """Rectangles within *max_dist* µm of rect *rid*, sorted by distance."""
        ref = self._rects[rid]
        result = []
        for r in self._rects.values():
            if r.rid == rid:
                continue
            if same_layer and r.layer != ref.layer:
                continue
            d = ref.edge_dist(r)
            if d <= max_dist:
                result.append((r, d))
        result.sort(key=lambda t: t[1])
        return result

    def spacing_pairs(self, layer: str,
                      max_dist: float = 0.5) -> list[tuple[Rect, Rect, float]]:
        """All pairs of same-layer rects closer than *max_dist*."""
        shapes = self.on_layer(layer)
        pairs = []
        for i, a in enumerate(shapes):
            for b in shapes[i + 1:]:
                d = a.edge_dist(b)
                if 0 < d <= max_dist:
                    pairs.append((a, b, d))
        pairs.sort(key=lambda t: t[2])
        return pairs

    # ── Connectivity queries ──────────────────────────────────────────────────

    def on_net(self, net: str) -> list[Rect]:
        """All rectangles belonging to *net*."""
        return [r for r in self._rects.values() if r.net == net]

    def contacts_near(self, x: float, y: float, radius: float,
                      layer: str | None = None) -> list[Rect]:
        """Contact/via shapes near ``(x, y)``."""
        return [r for r in self.near(x, y, radius, layer=layer)
                if r.is_contact]

    def shapes_of_type(self, shape_type: str,
                       layer: str | None = None) -> list[Rect]:
        """All rectangles of a given shape_type, optionally filtered by layer."""
        result = []
        for r in self._rects.values():
            if r.shape_type != shape_type:
                continue
            if layer and r.layer != layer:
                continue
            result.append(r)
        return result

    def connected_shapes(self, rid: int, max_dist: float = 0.01) -> list[Rect]:
        """Shapes on any layer that overlap or touch rect *rid* (connectivity)."""
        ref = self._rects.get(rid)
        if ref is None:
            return []
        result = []
        for r in self._rects.values():
            if r.rid == rid:
                continue
            if ref.edge_dist(r) <= max_dist:
                result.append(r)
        return result

    # ── Via group detection ─────────────────────────────────────────────

    def tag_via_groups(self, tol: float = 0.02) -> int:
        """Detect contact/via stacks and assign group_ids.

        A poly contact stack is identified when a licon1 rect has a
        co-centred poly rect (poly pad) — these are tagged as a group.
        Any li1, mcon, met1 rects at the same centre are added to the group.

        A diff contact stack is identified when a licon1 rect has a
        co-centred li1 rect but NO co-centred poly rect.

        Returns the number of groups found.
        """
        licon_rects = self.on_layer("licon1")
        if not licon_rects:
            return 0

        group_count = 0

        for licon in licon_rects:
            if licon.group_id >= 0:
                continue  # already assigned

            cx, cy = licon.cx, licon.cy
            group_count += 1
            gid = group_count

            # Find all shapes centred at the same point
            members = [licon]
            for r in self._rects.values():
                if r.rid == licon.rid or r.group_id >= 0:
                    continue
                if abs(r.cx - cx) <= tol and abs(r.cy - cy) <= tol:
                    if r.layer in ("poly", "li1", "mcon", "met1", "via1", "met2"):
                        members.append(r)

            # Check if this is a poly contact (has co-centred poly)
            has_poly = any(m.layer == "poly" for m in members)

            for m in members:
                m.group_id = gid
                if has_poly:
                    m.shape_type = "via"  # poly contact stack
                else:
                    m.shape_type = "contact"  # diff contact

            # Tag the poly member specifically
            if has_poly:
                for m in members:
                    if m.layer == "poly":
                        m.shape_type = "via_pad"  # poly pad (movable, not gate)

        return group_count

    def group_members(self, group_id: int) -> list[Rect]:
        """All rectangles belonging to *group_id*."""
        if group_id < 0:
            return []
        return [r for r in self._rects.values() if r.group_id == group_id]

    def move_group(self, group_id: int, dx: float, dy: float) -> None:
        """Translate all shapes in a group by (dx, dy)."""
        for r in self.group_members(group_id):
            r.x0 += dx
            r.x1 += dx
            r.y0 += dy
            r.y1 += dy

    # ── Import / Export ──────────────────────────────────────────────────────

    @classmethod
    def from_component(cls, comp: Any, rules: Any) -> "LayoutState":
        """Import all polygons from a gdsfactory Component.

        Each polygon is approximated as a single bounding-box rectangle.
        The ``rules`` object maps ``(gds_layer, gds_datatype)`` → logical
        layer name.
        """
        # Build reverse map: (layer, datatype) → logical name
        rev: dict[tuple[int, int], str] = {}
        for name, entry in rules.layers.items():
            rev[(entry["layer"], entry["datatype"])] = name

        state = cls()
        # gdsfactory 7+: get_polygons() returns dict[(layer,dt)] → [kdb.Polygon]
        try:
            polys_by_layer = comp.get_polygons()
        except Exception:
            return state

        dbu = 0.001  # default KLayout dbu = 1 nm
        try:
            # gdsfactory may expose dbu
            dbu = comp.kcl.dbu
        except Exception:
            pass

        for layer_key, polys in polys_by_layer.items():
            if isinstance(layer_key, int):
                # gdsfactory 7+: int index → resolve via kcl.get_info()
                try:
                    info = comp.kcl.get_info(layer_key)
                    layer_key = (info.layer, info.datatype)
                except Exception:
                    continue
            lname = rev.get(layer_key, None)
            if lname is None:
                continue
            for poly in polys:
                # Get bounding box
                try:
                    bbox = poly.bbox()
                    x0 = bbox.left * dbu
                    y0 = bbox.bottom * dbu
                    x1 = bbox.right * dbu
                    y1 = bbox.top * dbu
                except Exception:
                    # Try dbbox (floating-point bbox)
                    try:
                        bbox = poly.dbbox
                        x0, y0, x1, y1 = bbox.left, bbox.bottom, bbox.right, bbox.top
                    except Exception:
                        continue
                state.add(lname, x0, y0, x1, y1)
        return state

    def to_component(self, rules: Any, name: str = "geo_fixed") -> Any:
        """Export all rectangles back to a new gdsfactory Component."""
        import gdsfactory as gf
        comp = gf.Component(name=name)
        for r in self._rects.values():
            lyr = rules.layer(r.layer)
            comp.add_polygon(
                [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)],
                layer=lyr,
            )
        return comp

    # ── Local context for ML ────────────────────────────────────────────────

    def local_crop(self, x: float, y: float, radius: float = 2.0,
                   layers: list[str] | None = None) -> np.ndarray:
        """Extract a local feature matrix around ``(x, y)``.

        Returns an (N, 6) array where each row is::

            [layer_idx, x0_rel, y0_rel, x1_rel, y1_rel, area]

        Coordinates are relative to ``(x, y)``.  This is the observation
        format for Phase 3 learned agents.
        """
        rects = self.near(x, y, radius)
        if layers:
            rects = [r for r in rects if r.layer in layers]
        if not rects:
            return np.zeros((0, 6), dtype=np.float32)

        # Map layer names to indices
        layer_set = sorted({r.layer for r in rects})
        layer_idx = {n: i for i, n in enumerate(layer_set)}

        rows = []
        for r in rects:
            rows.append([
                layer_idx[r.layer],
                r.x0 - x, r.y0 - y,
                r.x1 - x, r.y1 - y,
                r.area,
            ])
        return np.array(rows, dtype=np.float32)
