"""
layout_gen.rl.training.mine_scl_demos — sky130 SCL → PLACE-action demos.

Reverse-engineers PLACE actions from real standard-cell layouts.
Walks ``$PDK_ROOT/<pdk>/libs.ref/<scl>/gds/<scl>.gds``, classifies
each cell against our primitive vocabulary
(:func:`layout_gen.repair.seeds.classify_primitive`), pulls the cell's
polygons, clusters them into device groups via the same logic
``inspect_gds`` uses, and emits one demo JSON per (SCL cell, matching
YAML template) pair.

Why
---
The synth-derived demo corpus is tiny (≈12 PLACE actions across our
3 starter templates). Sky130 ships hundreds of canonical cells; each
maps onto one of our YAML topologies. Mining them produces an order
of magnitude more demos at zero additional design cost.

What gets emitted
-----------------
For every matching SCL cell:

  * Cluster all polygons into device groups (diff-overlap union-find,
    same as :func:`inspect_gds._cluster_devices`).
  * Classify each cluster as nmos / pmos via implant + nwell.
  * Sort: NMOS clusters (sorted left-to-right), then PMOS clusters
    (sorted left-to-right). This matches the convention every YAML
    in ``layout_gen/templates/cells/`` uses for its ``devices`` order
    (N_A,N_B,...,P_A,P_B,...).
  * Map cluster i → topology device i. Drop the demo if the cluster
    count doesn't match the topology's device count (avoids feeding
    misaligned data to BC).
  * Emit one PLACE action per device with the cluster centroid as
    ``(x_um, y_um)``. Orientation is defaulted to ``R0``; recovering
    actual orientation from raw GDS would need a per-template
    geometric match that's out of scope here.
  * No ROUTE actions (the SCL routing isn't a clean per-net match
    to our action space; rely on synth-derived demos for ROUTE).

Coordinate frame
----------------
The cluster centroids are in the SCL cell's *raw* coordinate frame
(typically lower-left origin). The cell's bbox dimensions go into
``cell_width_um`` / ``cell_height_um`` so the dataset's
``_coord_to_bin`` puts them in the right discretisation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gdstk

from layout_gen.pdk import load_pdk
from layout_gen.repair.seeds import scl_seeds, classify_primitive
from layout_gen.synth.loader import load_template

from layout_gen.rl.scripts.inspect_gds import (
    Poly, _classify_cluster, _cluster_devices,
)
from layout_gen.rl.topology import graph_from_template
from layout_gen.rl.training.demo_extract import PlacementDemo, write_demo


# Map repair/seeds.py primitive bucket → matching YAML template name.
_PRIMITIVE_TO_TEMPLATE: dict[str, str] = {
    "inv":     "inverter",
    "nand":    "nand2",        # use nand2 for any nand cell; mismatch detected later
    "nor":     "nor2",
    "buf":     "buffer",
    # NB: oai / aoi / dff / mux skipped — multi-stage cells where the
    # cluster-to-topology mapping isn't 1:1.
}


def _read_polys(gds_path: Path, cell_name: str, rules) -> list[Poly]:
    """Pull every polygon of *cell_name* out of *gds_path* into Poly tuples."""
    rev = {(e["layer"], e["datatype"]): name
           for name, e in rules.layers.items()}
    lib = gdstk.read_gds(str(gds_path))
    cell = next((c for c in lib.cells if c.name == cell_name), None)
    if cell is None:
        return []
    out: list[Poly] = []
    for poly in cell.polygons:
        ltup = (poly.layer, poly.datatype)
        name = rev.get(ltup, "?")
        if name == "?":
            continue   # vendor-only layer (boundary, prBoundary, …)
        bb = poly.bounding_box()
        if bb is None:
            continue
        (x0, y0), (x1, y1) = bb
        out.append(Poly(layer=name, raw_lt=ltup,
                        x0=x0, y0=y0, x1=x1, y1=y1))
    return out


def _sort_clusters_by_row_then_x(
    clusters: list[list[Poly]],
) -> tuple[list[list[Poly]], list[list[Poly]]]:
    """Returns (nmos_clusters, pmos_clusters), each sorted left-to-right."""
    nmos: list[tuple[float, list[Poly]]] = []
    pmos: list[tuple[float, list[Poly]]] = []
    for c in clusters:
        kind, _present, _missing = _classify_cluster(c)
        diffs = [p for p in c if p.layer == "diff"]
        if not diffs:
            continue
        cx = sum(p.cx for p in diffs) / len(diffs)
        if kind == "pmos":
            pmos.append((cx, c))
        elif kind == "nmos":
            nmos.append((cx, c))
        # unknown clusters dropped — usually just well taps
    nmos.sort(key=lambda pair: pair[0])
    pmos.sort(key=lambda pair: pair[0])
    return [c for _, c in nmos], [c for _, c in pmos]


def _cluster_centroid(cluster: list[Poly]) -> tuple[float, float]:
    diffs = [p for p in cluster if p.layer == "diff"]
    if not diffs:
        diffs = cluster
    cx = sum(p.cx for p in diffs) / len(diffs)
    cy = sum(p.cy for p in diffs) / len(diffs)
    # The cluster centroid sits roughly at the device's mid-Y / mid-X.
    # Our place_device origin convention is the device's local (0, 0)
    # which sits at the lower-left of the poly bounding box. Subtract
    # half the diff height to approximate the lower-left origin.
    if diffs:
        half_h = (max(p.y1 for p in diffs) - min(p.y0 for p in diffs)) / 2
        half_w = (max(p.x1 for p in diffs) - min(p.x0 for p in diffs)) / 2
        return cx - half_w, cy - half_h
    return cx, cy


def _cell_bbox(polys: list[Poly]) -> tuple[float, float]:
    if not polys:
        return 4.0, 2.0
    w = max(p.x1 for p in polys) - min(p.x0 for p in polys)
    h = max(p.y1 for p in polys) - min(p.y0 for p in polys)
    return max(w, 0.001), max(h, 0.001)


def mine(
    pdk:               str = "sky130A",
    *,
    template_names:    list[str] | None = None,
    max_per_primitive: int = 5,
    out_dir:           Path = Path("demos/scl/"),
    rules = None,
) -> list[Path]:
    """Mine PLACE-action demos from the SCL.

    Parameters
    ----------
    pdk :
        PDK name (default ``sky130A``).
    template_names :
        Restrict to demos that match one of these YAML templates. ``None``
        means "all primitives we know how to map" (currently inv / nand /
        nor / buf via :data:`_PRIMITIVE_TO_TEMPLATE`).
    max_per_primitive :
        Cap on cells per primitive bucket (so we don't drown in inv_1
        through inv_16).
    out_dir :
        Where to write ``*.demo.json``.
    """
    rules = rules or load_pdk()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = scl_seeds(pdk, max_per_primitive=max_per_primitive)
    written: list[Path] = []
    skipped = {"unknown_primitive": 0, "no_template_match": 0,
               "device_count_mismatch": 0, "no_polys": 0}

    for s in seeds:
        primitive = s.primitive
        tmpl_name = _PRIMITIVE_TO_TEMPLATE.get(primitive)
        if tmpl_name is None:
            skipped["unknown_primitive"] += 1
            continue
        if template_names is not None and tmpl_name not in template_names:
            continue

        try:
            template = load_template(tmpl_name)
        except FileNotFoundError:
            skipped["no_template_match"] += 1
            continue

        polys = _read_polys(s.gds_path, s.name, rules)
        if not polys:
            skipped["no_polys"] += 1
            continue
        clusters = _cluster_devices(polys)
        nmos_cls, pmos_cls = _sort_clusters_by_row_then_x(clusters)

        # Match to topology device order: NMOS first, then PMOS, in the
        # YAML's iteration order. If the SCL's cluster count doesn't
        # match the YAML's device count, drop — feeding misaligned
        # demos to BC would be worse than no demo.
        cell_params = {"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}}
        graph = graph_from_template(template, cell_params=cell_params)
        topo_nmos = [d for d in graph.devices if d.device_type == "nmos"]
        topo_pmos = [d for d in graph.devices if d.device_type == "pmos"]
        if (len(nmos_cls) != len(topo_nmos)
                or len(pmos_cls) != len(topo_pmos)):
            skipped["device_count_mismatch"] += 1
            continue

        cell_w, cell_h = _cell_bbox(polys)

        # Compute the rebase offset so the leftmost NMOS lands at (0, 0).
        # Synth-derived demos use that convention (first NMOS at origin),
        # so SCL-mined demos have to match or BC sees conflicting
        # examples and the policy averages the two frames into garbage.
        nmos_pairs = list(zip(nmos_cls, topo_nmos))
        pmos_pairs = list(zip(pmos_cls, topo_pmos))
        di = graph.device_index()

        if nmos_pairs:
            base_x, base_y = _cluster_centroid(nmos_pairs[0][0])
        elif pmos_pairs:
            base_x, base_y = _cluster_centroid(pmos_pairs[0][0])
        else:
            base_x, base_y = 0.0, 0.0

        actions: list[dict] = []
        for cluster, dev in nmos_pairs + pmos_pairs:
            x_um, y_um = _cluster_centroid(cluster)
            actions.append({
                "kind":         "place_device",
                "device_name":  dev.name,
                "device_idx":   di[dev.name],
                "x_um":         float(x_um - base_x),
                "y_um":         float(y_um - base_y),
                "orientation":  "R0",
            })

        demo = PlacementDemo(
            template=tmpl_name,
            cell_width_um=cell_w,
            cell_height_um=cell_h,
            cell_params=cell_params["_defaults"],
            actions=actions,
        )
        # Append the SCL cell name so multiple SCL variants of the
        # same primitive don't overwrite each other.
        out_path = out_dir / f"{tmpl_name}__{s.name}.demo.json"
        write_demo(demo, out_path)
        written.append(out_path)

    return written


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Mine PLACE-action demos from sky130 standard cells."
    )
    p.add_argument("--pdk", default="sky130A")
    p.add_argument("--templates", default=None,
                   help="Comma-separated YAML template names to mine for "
                        "(e.g. inverter,nand2,nor2). Default: all "
                        "supported (inv/nand/nor/buf).")
    p.add_argument("--max-per-primitive", type=int, default=5,
                   help="Cap on SCL cells per primitive bucket.")
    p.add_argument("--out", type=Path, default=Path("demos/scl/"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    os.environ.setdefault("PDK_ROOT", "/usr/local/share/pdk")
    template_names = (
        [t.strip() for t in args.templates.split(",") if t.strip()]
        if args.templates else None
    )
    paths = mine(
        pdk=args.pdk,
        template_names=template_names,
        max_per_primitive=args.max_per_primitive,
        out_dir=args.out,
    )
    if not args.quiet:
        print(f"[scl-mine] wrote {len(paths)} demos to {args.out}")
        for p_ in paths:
            print(f"  {p_.name}")
    return 0


__all__ = ["mine"]


if __name__ == "__main__":
    sys.exit(_main())
