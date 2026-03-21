"""
layout_gen — parametric SRAM cell layout generation with DRC-aware repair.

Generates DRC-clean GDS layouts from parameterized cell descriptions.
Works alongside spice_gen / liberty_gen / verilog_gen in the fabram pipeline.

Pipeline
--------
Phase 1 — Morphable templates
    Parametric gdsfactory cells built from PDK rules.
    Entry points: ``draw_transistor``, and per-cell ``draw_<cell>`` functions.
    Topology templates in ``layout_gen/templates/`` describe spatial topology
    independently of any tech node.

Phase 2 — DRC repair / ML synthesis
    DRC runner (``layout_gen.drc``) validates GDS via KLayout or Magic.
    ML synthesizer reads topology templates + PDK rules → routes polygons
    → iterates with DRC until clean.

Typical use::

    from layout_gen import draw_transistor, load_pdk, run_drc

    rules      = load_pdk()
    comp       = draw_transistor(0.52, 0.15, "nmos", rules)
    comp.write_gds("out/nmos_0p52.gds")
    violations = run_drc("out/nmos_0p52.gds", rules, tool="klayout")
    print(f"{len(violations)} violations")
"""
from layout_gen.pdk         import load_pdk, PDKRules, PDK_YAML, RULES
from layout_gen.transistor  import (
    draw_transistor,
    transistor_geom,
    finger_count,
    TransistorGeom,
)
from layout_gen.visualize   import write_svg
from layout_gen.cells       import draw_inverter, draw_nand2, draw_nor2, draw_bit_cell
from layout_gen.drc         import run_drc, DRCViolation, available_tools

__all__ = [
    # PDK
    "load_pdk",
    "PDKRules",
    "PDK_YAML",
    "RULES",
    # Transistor primitive
    "draw_transistor",
    "transistor_geom",
    "finger_count",
    "TransistorGeom",
    # Standard cells
    "draw_inverter",
    "draw_nand2",
    "draw_nor2",
    # SRAM cells
    "draw_bit_cell",
    # Visualisation
    "write_svg",
    # DRC
    "run_drc",
    "DRCViolation",
    "available_tools",
]
