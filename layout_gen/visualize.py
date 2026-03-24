"""
layout_gen.visualize — GDS → SVG renderer with per-polygon layer labels.

Renders a gdsfactory Component (or a GDS file path) to a standalone SVG.
Each polygon is filled with the layer colour defined in the PDK YAML and
labelled with its logical layer name in the lower-left corner of its own
bounding box.

Coordinate conventions
----------------------
- GDS / layout_gen use µm with Y increasing upward.
- SVG has Y increasing downward, so the Y axis is flipped.
- Coordinates are scaled by *scale* pixels-per-µm (default 800).

Typical use::

    from layout_gen.visualize import write_svg
    from layout_gen          import draw_transistor, load_pdk

    rules = load_pdk()
    comp  = draw_transistor(0.52, 0.15, "nmos", rules)
    write_svg(comp, "out/nmos_0p52.svg", rules=rules)

    # From an existing GDS file
    write_svg("out/nmos_0p52.gds", "out/nmos_0p52.svg", rules=rules)
"""
from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
from typing import Union

from layout_gen.pdk import PDKRules, RULES, _load_yaml, PDK_YAML


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reverse_layer_map(rules: PDKRules) -> dict[tuple[int, int], str]:
    """Return (gds_layer, gds_datatype) → logical_name mapping from PDK rules."""
    return {(v["layer"], v["datatype"]): k for k, v in rules.layers.items()}


def _layer_color(logical: str, pdk_data: dict) -> tuple[str, float]:
    """Return (fill_hex, opacity) for a logical layer name.

    Falls back to ``("#888888", 0.5)`` for layers not in the PDK colors section.
    """
    colors = pdk_data.get("colors", {})
    entry  = colors.get(logical, {})
    return entry.get("fill", "#888888"), entry.get("opacity", 0.5)


def _font_size(scale: float) -> float:
    """A readable font size relative to the zoom scale (pixels per µm)."""
    return max(6.0, min(14.0, scale * 0.018))


# ── Public API ─────────────────────────────────────────────────────────────────

def write_svg(
    component:   Union["gf.Component", str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    rules:       PDKRules = RULES,
    pdk_yaml:    pathlib.Path = PDK_YAML,
    scale:       float = 800.0,
    margin_um:   float = 0.15,
) -> pathlib.Path:
    """Render *component* to an SVG file and return the output path.

    Parameters
    ----------
    component :
        A gdsfactory ``Component`` object **or** a path to an existing GDS file.
        If a GDS path is given the top cell is used.
    output_path :
        Destination ``.svg`` file.  Parent directories are created if needed.
    rules :
        PDK rules (used for the layer → logical-name reverse map).
    pdk_yaml :
        Path to the PDK YAML (used to read the ``colors`` section).
    scale :
        Pixels per micrometre.  800 works well for individual transistors;
        use lower values (e.g. 200) for full cells.
    margin_um :
        White-space border around the design (µm).

    Returns
    -------
    pathlib.Path
        Absolute path of the written SVG.
    """
    import gdsfactory as gf

    # ── Load component ────────────────────────────────────────────────────────
    if isinstance(component, (str, pathlib.Path)):
        component = gf.import_gds(str(component))

    # ── Build layer maps ──────────────────────────────────────────────────────
    rev_map  = _reverse_layer_map(rules)
    # Prefer colors from rules object; fall back to pdk_yaml file
    if rules.colors:
        pdk_data = {"colors": rules.colors}
    else:
        pdk_data = _load_yaml(pdk_yaml)

    # ── Collect all polygons with their layers ────────────────────────────────
    # gdsfactory 9 / kfactory API:
    #   get_polygons(by='tuple') → dict[(layer, dt): list[kdb.Polygon]]
    # Polygon points via each_point_hull() in database units (dbu = 0.001 µm).
    import gdsfactory as gf
    dbu = gf.kcl.dbu   # µm per database unit (0.001 for sky130A / generic)

    polys_by_layer: dict = component.get_polygons(by="tuple")

    if not polys_by_layer:
        raise ValueError(f"Component '{component.name}' contains no polygons.")

    def _pts_um(poly) -> list[tuple[float, float]]:
        """Extract polygon hull vertices in µm."""
        return [(pt.x * dbu, pt.y * dbu) for pt in poly.each_point_hull()]

    # ── Compute overall bounding box (µm) ─────────────────────────────────────
    all_pts: list[tuple[float, float]] = []
    for poly_list in polys_by_layer.values():
        for poly in poly_list:
            all_pts.extend(_pts_um(poly))

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min_d, x_max_d = min(xs) - margin_um, max(xs) + margin_um
    y_min_d, y_max_d = min(ys) - margin_um, max(ys) + margin_um

    width_um  = x_max_d - x_min_d
    height_um = y_max_d - y_min_d
    svg_w = width_um  * scale
    svg_h = height_um * scale

    def to_svg(x_um: float, y_um: float) -> tuple[float, float]:
        """Convert layout µm coords to SVG pixel coords (Y flipped)."""
        sx = (x_um - x_min_d) * scale
        sy = (y_max_d - y_um) * scale   # flip Y
        return sx, sy

    # ── Build SVG ─────────────────────────────────────────────────────────────
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    svg = ET.Element("svg", {
        "xmlns":   "http://www.w3.org/2000/svg",
        "width":   f"{svg_w:.1f}",
        "height":  f"{svg_h:.1f}",
        "viewBox": f"0 0 {svg_w:.1f} {svg_h:.1f}",
    })

    # White background
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": f"{svg_w:.1f}", "height": f"{svg_h:.1f}",
        "fill": "white",
    })

    fs = _font_size(scale)

    # ── Draw polygons layer by layer (nwell first, contacts last) ─────────────
    # Sort by GDS layer number so higher layers (metals) appear on top.
    for (gds_layer, gds_dt), pts_list in sorted(polys_by_layer.items(),
                                                  key=lambda kv: kv[0][0]):
        logical = rev_map.get((gds_layer, gds_dt),
                               f"L{gds_layer}/{gds_dt}")
        fill, opacity = _layer_color(logical, pdk_data)

        grp = ET.SubElement(svg, "g", {
            "id":      logical,
            "opacity": f"{opacity:.2f}",
        })

        for poly in pts_list:
            pts = _pts_um(poly)
            # ── Polygon shape ─────────────────────────────────────────────────
            svg_pts = [to_svg(float(p[0]), float(p[1])) for p in pts]
            pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in svg_pts)
            ET.SubElement(grp, "polygon", {
                "points": pts_str,
                "fill":   fill,
                "stroke": "#000000",
                "stroke-width": f"{max(0.3, scale * 0.0005):.2f}",
            })

            # ── Layer label at lower-left of this polygon's bbox ──────────────
            poly_xs = [p[0] for p in pts]
            poly_ys = [p[1] for p in pts]
            lx_um = min(poly_xs)   # lower-left in layout coords = min X, min Y
            ly_um = min(poly_ys)

            lx_svg, ly_svg = to_svg(lx_um, ly_um)

            # Nudge slightly inside the polygon
            nudge = fs * 0.25
            ET.SubElement(grp, "text", {
                "x":           f"{lx_svg + nudge:.2f}",
                "y":           f"{ly_svg - nudge:.2f}",   # SVG Y goes down → subtract
                "font-family": "monospace",
                "font-size":   f"{fs:.1f}",
                "fill":        "#000000",
                "fill-opacity": "1",
                "dominant-baseline": "text-after-edge",
            }).text = logical

    # ── Title ─────────────────────────────────────────────────────────────────
    title = ET.SubElement(svg, "text", {
        "x": f"{svg_w / 2:.1f}",
        "y": f"{fs * 1.5:.1f}",
        "text-anchor": "middle",
        "font-family": "monospace",
        "font-size":   f"{fs * 1.1:.1f}",
        "fill":        "#333333",
    })
    title.text = component.name

    # ── Write file ────────────────────────────────────────────────────────────
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="unicode", xml_declaration=False)

    return out.resolve()
