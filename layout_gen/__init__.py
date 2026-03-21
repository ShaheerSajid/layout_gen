"""
layout_gen — parametric SRAM cell layout generation with DRC-aware repair.

Generates DRC-clean GDS layouts from parameterized cell descriptions.
Works alongside spice_gen / liberty_gen / verilog_gen in the fabram pipeline.

Pipeline
--------
Phase 1 — Morphable templates
    Parametric gdsfactory cells built from PDK rules.
    Entry points: ``draw_transistor``, and per-cell ``draw_<cell>`` functions.

Phase 2 — DRC repair graph  (planned)
    Typed violation → repair action → converge to clean layout.

Typical use::

    from layout_gen import draw_transistor, load_pdk

    rules = load_pdk()                              # sky130A default
    comp  = draw_transistor(0.52, 0.15, "nmos", rules)
    comp.write_gds("out/nmos_0p52.gds")

    # Different PDK
    from pathlib import Path
    rules = load_pdk(Path("pdks/gf180.yaml"))
    comp  = draw_transistor(0.40, 0.18, "nmos", rules)
"""
from layout_gen.pdk         import load_pdk, PDKRules, PDK_YAML, RULES
from layout_gen.transistor  import (
    draw_transistor,
    transistor_geom,
    finger_count,
    TransistorGeom,
)
from layout_gen.visualize   import write_svg
from layout_gen.cells       import draw_inverter, draw_nand2, draw_nor2  # standard cell primitives

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
    # Visualisation
    "write_svg",
]
