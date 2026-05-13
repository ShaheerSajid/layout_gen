"""
layout_gen.rl.scripts.inspect_gds — read a GDS and report what's inside.

Diagnostic tool: takes a GDS file, applies the project's PDK layer
mapping (so raw (layer, datatype) tuples turn back into logical names
like ``nwell`` / ``psdm`` / ``poly``), then prints:

  * Per-layer polygon counts and bounding-box ranges.
  * **Device clusters** — heuristic grouping of (diff + poly + implant)
    polygons by spatial proximity. For each cluster we report the
    transistor type (nmos vs pmos by implant + nwell presence) and
    whether any expected layer is missing.
  * **Routing summary** — counts of polygons on metal layers, with
    bbox bounds.
  * A simple ASCII top-down sketch of the cell so you can eyeball
    overlaps without opening KLayout.

Usage::

    .venv/bin/python -m layout_gen.rl.scripts.inspect_gds /path/to/cell.gds

Add ``--ascii`` to print a denser ASCII map; add ``--strict`` to exit
non-zero when the heuristic flags any missing-layer issue.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import gdsfactory as gf

from layout_gen.pdk import load_pdk


# ── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class Poly:
    layer:    str          # logical name; "?" if unknown
    raw_lt:   tuple[int, int]
    x0:       float
    y0:       float
    x1:       float
    y1:       float

    @property
    def cx(self) -> float: return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float: return (self.y0 + self.y1) / 2

    @property
    def area(self) -> float: return (self.x1 - self.x0) * (self.y1 - self.y0)


def _activate_pdk():
    try:
        gf.get_active_pdk()
    except Exception:
        from gdsfactory.gpdk import PDK as _G   # noqa: I001
        _G.activate()


def _read_polys(gds_path: Path, rules) -> tuple[str, list[Poly]]:
    _activate_pdk()
    comp = gf.import_gds(str(gds_path))
    rev = {(e["layer"], e["datatype"]): name for name, e in rules.layers.items()}

    out: list[Poly] = []
    raw = comp.get_polygons()
    for k, vs in raw.items():
        if isinstance(k, int):
            try:
                info = comp.kcl.get_info(k)
                ltup = (info.layer, info.datatype)
            except Exception:
                ltup = k if isinstance(k, tuple) else (-1, -1)
        else:
            ltup = k
        name = rev.get(ltup, "?")
        for p in vs:
            try:
                bb = p.bbox()
                # KLayout dbu is 1 nm by default → convert to µm.
                dbu = 0.001
                try: dbu = comp.kcl.dbu
                except Exception: pass
                out.append(Poly(
                    layer=name, raw_lt=ltup,
                    x0=bb.left*dbu, y0=bb.bottom*dbu,
                    x1=bb.right*dbu, y1=bb.top*dbu,
                ))
            except Exception:
                continue
    return comp.name, out


# ── Layer summary ────────────────────────────────────────────────────────────

def print_layer_summary(polys: list[Poly]) -> None:
    by_layer: dict[str, list[Poly]] = {}
    for p in polys:
        by_layer.setdefault(p.layer, []).append(p)
    print("─── Polygons per logical layer ───────────────────────────────────")
    for name in sorted(by_layer):
        ps = by_layer[name]
        x0 = min(p.x0 for p in ps); y0 = min(p.y0 for p in ps)
        x1 = max(p.x1 for p in ps); y1 = max(p.y1 for p in ps)
        print(f"  {name:8s} count={len(ps):3d}   bbox=({x0:+.3f}, {y0:+.3f}) "
              f"→ ({x1:+.3f}, {y1:+.3f}) µm")


# ── Device clustering ────────────────────────────────────────────────────────

# Layers that must appear in any transistor; used as cluster anchors.
_DEVICE_ANCHORS = ("diff",)
# Layers we expect alongside each device, by type.
_NMOS_EXPECTED = {"diff", "poly", "nsdm", "li1", "licon1"}
_PMOS_EXPECTED = {"diff", "poly", "psdm", "nwell", "li1", "licon1"}


def _overlaps(a: Poly, b: Poly, tol: float = 0.05) -> bool:
    return (a.x0 <= b.x1 + tol and a.x1 + tol >= b.x0 and
            a.y0 <= b.y1 + tol and a.y1 + tol >= b.y0)


def _cluster_devices(polys: list[Poly]) -> list[list[Poly]]:
    """Group polys spatially around each ``diff`` rect.

    Each diff anchors a cluster; every other poly that overlaps the
    diff (with a small tolerance) is added. Two diffs that touch end
    up in the same cluster — that's intentional, since adjacent
    devices share diffusion in real cells.
    """
    diffs = [p for p in polys if p.layer in _DEVICE_ANCHORS]
    others = [p for p in polys if p.layer not in _DEVICE_ANCHORS]

    # Union-find over diffs by overlap.
    parent = list(range(len(diffs)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i in range(len(diffs)):
        for j in range(i + 1, len(diffs)):
            if _overlaps(diffs[i], diffs[j]):
                union(i, j)

    groups: dict[int, list[Poly]] = {}
    for i, d in enumerate(diffs):
        groups.setdefault(find(i), []).append(d)

    clusters: list[list[Poly]] = []
    for _, group_diffs in groups.items():
        # Compute the cluster's hull from its diff members.
        hx0 = min(p.x0 for p in group_diffs)
        hy0 = min(p.y0 for p in group_diffs)
        hx1 = max(p.x1 for p in group_diffs)
        hy1 = max(p.y1 for p in group_diffs)
        anchor = Poly("hull", (-1, -1), hx0, hy0, hx1, hy1)
        members = list(group_diffs)
        for p in others:
            if _overlaps(anchor, p, tol=0.10):
                members.append(p)
        clusters.append(members)
    return clusters


def _classify_cluster(members: list[Poly]) -> tuple[str, set[str], set[str]]:
    """Return (device_type, present_layers, missing_layers)."""
    present = {p.layer for p in members}
    has_nwell = "nwell" in present
    has_nsdm  = "nsdm"  in present
    has_psdm  = "psdm"  in present

    if has_nwell or has_psdm:
        kind = "pmos"
        expected = _PMOS_EXPECTED
    elif has_nsdm:
        kind = "nmos"
        expected = _NMOS_EXPECTED
    else:
        kind = "unknown"
        expected = _NMOS_EXPECTED  # default to nmos for missing-layer report
    missing = expected - present
    return kind, present, missing


def _stacked_device_count(members: list[Poly]) -> int:
    """Estimate how many distinct devices the cluster contains by counting
    poly columns whose centroids are >0.05 µm apart in X. A correctly-
    placed pair of devices has two poly columns at different X; a
    "stack" (two devices at the same coords) has only one — flagged by
    the caller."""
    polys_x = sorted(round(p.cx, 3) for p in members if p.layer == "poly")
    if not polys_x:
        return 0
    distinct = [polys_x[0]]
    for x in polys_x[1:]:
        if abs(x - distinct[-1]) > 0.05:
            distinct.append(x)
    return len(distinct)


def print_device_summary(polys: list[Poly]) -> tuple[int, int, int]:
    """Returns (n_nmos, n_pmos, n_issues)."""
    clusters = _cluster_devices(polys)
    print(f"─── Device clusters (heuristic, by diff overlap) ─────────────────")
    n_nmos = n_pmos = n_issues = 0
    for i, members in enumerate(clusters):
        kind, present, missing = _classify_cluster(members)
        diffs = [p for p in members if p.layer == "diff"]
        cx = sum(p.cx for p in diffs) / len(diffs)
        cy = sum(p.cy for p in diffs) / len(diffs)
        status = "OK"
        if missing:
            status = f"MISSING: {sorted(missing)}"
            n_issues += 1
        # Heuristic stacked-device check: more diffs in the cluster
        # than poly columns ⇒ devices stacked at the same coords.
        n_diffs = len(diffs)
        n_poly_cols = _stacked_device_count(members)
        if n_diffs > 1 and n_poly_cols < n_diffs:
            status = (status + " | "
                      if status != "OK" else "") + (
                f"STACKED: {n_diffs} diffs but only {n_poly_cols} poly column(s)"
            )
            if "STACKED" in status and "OK" in status:
                status = status.replace("OK", "").strip()
            n_issues += 1
        print(f"  device #{i}  {kind:7s}  centre=({cx:+.3f}, {cy:+.3f}) "
              f"layers={sorted(present)}  {status}")
        if kind == "nmos": n_nmos += 1
        elif kind == "pmos": n_pmos += 1
    return n_nmos, n_pmos, n_issues


# ── Routing summary ──────────────────────────────────────────────────────────

_METAL_LAYERS = ("li1", "met1", "met2", "met3", "met4", "met5")


def print_routing_summary(polys: list[Poly]) -> None:
    print("─── Routing (interconnect layers) ────────────────────────────────")
    saw_any = False
    for layer in _METAL_LAYERS:
        ps = [p for p in polys if p.layer == layer]
        if not ps:
            continue
        saw_any = True
        x0 = min(p.x0 for p in ps); y0 = min(p.y0 for p in ps)
        x1 = max(p.x1 for p in ps); y1 = max(p.y1 for p in ps)
        print(f"  {layer:6s} count={len(ps):3d}   bbox=({x0:+.3f}, {y0:+.3f}) "
              f"→ ({x1:+.3f}, {y1:+.3f}) µm")
    if not saw_any:
        print("  (no interconnect polygons found)")


# ── ASCII sketch ─────────────────────────────────────────────────────────────

# Compact one-char glyph per layer family; multiple polys overlapping in
# one cell pick the highest-priority glyph (priority = order in this list).
_LAYER_GLYPHS = [
    ("poly",   "P"),
    ("diff",   "D"),
    ("li1",    "l"),
    ("met1",   "1"),
    ("met2",   "2"),
    ("met3",   "3"),
    ("nwell",  "N"),
    ("nsdm",   "n"),
    ("psdm",   "p"),
    ("licon1", "·"),
    ("npc",    " "),
]


def print_ascii_sketch(polys: list[Poly], cols: int = 60, rows: int = 18) -> None:
    if not polys:
        print("(empty layout)")
        return
    x0 = min(p.x0 for p in polys); y0 = min(p.y0 for p in polys)
    x1 = max(p.x1 for p in polys); y1 = max(p.y1 for p in polys)
    sx = (x1 - x0) / max(cols, 1)
    sy = (y1 - y0) / max(rows, 1)
    if sx <= 0 or sy <= 0:
        return
    glyph_priority = {name: i for i, (name, _) in enumerate(_LAYER_GLYPHS)}
    grid = [["·"] * cols for _ in range(rows)]
    grid_priority = [[len(_LAYER_GLYPHS)] * cols for _ in range(rows)]
    for p in polys:
        prio = glyph_priority.get(p.layer, len(_LAYER_GLYPHS))
        # Find the glyph for this layer (default '?' if unknown).
        glyph = next((g for n, g in _LAYER_GLYPHS if n == p.layer), "?")
        c0 = max(0, int((p.x0 - x0) / sx))
        c1 = min(cols, int((p.x1 - x0) / sx) + 1)
        r0 = max(0, int((p.y0 - y0) / sy))
        r1 = min(rows, int((p.y1 - y0) / sy) + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                if prio < grid_priority[r][c]:
                    grid_priority[r][c] = prio
                    grid[r][c] = glyph
    print(f"─── ASCII sketch ({cols}×{rows} cells, "
          f"x∈[{x0:+.3f}, {x1:+.3f}]µm  y∈[{y0:+.3f}, {y1:+.3f}]µm) ─────")
    # Print top-down (y descending).
    for row in reversed(grid):
        print("  " + "".join(row))
    print("  Legend: " + "  ".join(f"{g}={n}" for n, g in _LAYER_GLYPHS))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("gds", type=Path)
    p.add_argument("--ascii", action="store_true",
                   help="Also print a top-down ASCII sketch.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any device cluster is missing "
                        "an expected layer.")
    args = p.parse_args(argv)

    rules = load_pdk()
    cell, polys = _read_polys(args.gds, rules)
    print(f"GDS: {args.gds}")
    print(f"Cell: {cell}    polygons: {len(polys)}")
    if polys:
        x0 = min(p.x0 for p in polys); y0 = min(p.y0 for p in polys)
        x1 = max(p.x1 for p in polys); y1 = max(p.y1 for p in polys)
        print(f"Cell bbox: ({x0:+.3f}, {y0:+.3f}) → ({x1:+.3f}, {y1:+.3f}) µm")
    print()
    print_layer_summary(polys)
    print()
    n_nmos, n_pmos, n_issues = print_device_summary(polys)
    print(f"  → totals: {n_nmos} NMOS, {n_pmos} PMOS, {n_issues} cluster(s) with missing layers")
    print()
    print_routing_summary(polys)
    if args.ascii:
        print()
        print_ascii_sketch(polys)

    if args.strict and n_issues:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
