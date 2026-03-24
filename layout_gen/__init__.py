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
    from layout_gen.synth import load_template, Synthesizer

    rules    = load_pdk()
    template = load_template("inverter")
    result   = Synthesizer(rules).synthesize(
        template, params={"w_N": 0.52, "w_P": 0.42, "l": 0.15}
    )
    result.component.write_gds("inv_synth.gds")
"""
from layout_gen.pdk         import load_pdk, PDKRules, PDK_YAML, RULES
from layout_gen.transistor  import (
    draw_transistor,
    transistor_geom,
    finger_count,
    TransistorGeom,
)
from layout_gen.visualize   import write_svg
from layout_gen.cells       import draw_tap_cell
from layout_gen.drc         import run_drc, DRCViolation, available_tools
from layout_gen.synth       import load_template, Synthesizer, SynthResult, MLAgent, ModelNotTrainedError

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
    # Tap cell
    "draw_tap_cell",
    # Visualisation
    "write_svg",
    # DRC
    "run_drc",
    "DRCViolation",
    "available_tools",
    # ML synthesizer
    "load_template",
    "Synthesizer",
    "SynthResult",
    "MLAgent",
    "ModelNotTrainedError",
]
