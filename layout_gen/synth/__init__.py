"""
layout_gen.synth — topology-template-driven layout synthesis.

Provides a technology-agnostic pipeline that reads a YAML cell topology
template and synthesizes a DRC-clean GDS layout by:

1. Loading the template (device connectivity + floorplan + routing specs)
2. Resolving symbolic constraints using PDK rules
3. Placing transistor primitives at computed coordinates
4. Routing nets using registered style handlers
5. Iterating with DRC feedback (optional ML model or built-in heuristic)

Quick start
-----------
::

    from layout_gen       import load_pdk
    from layout_gen.synth import load_template, Synthesizer

    rules    = load_pdk()
    template = load_template("inverter")   # reads templates/cells/inverter.yaml
    synth    = Synthesizer(rules)

    result = synth.synthesize(
        template,
        params={"w_N": 0.52, "w_P": 0.42, "l": 0.15},
    )
    result.component.write_gds("inv.gds")
    print(f"Synthesized in {result.iterations} iteration(s); "
          f"DRC {'clean' if result.converged else f'{len(result.violations)} violations'}")

ML-guided synthesis
-------------------
::

    def my_model(template, rules, violations, params):
        # Analyse DRC violations → return updated params dict
        return params

    synth = Synthesizer(rules, drc_runner=runner, ml_model=my_model)
    result = synth.synthesize(template, params, max_iter=20)

Built-in templates
------------------
- ``"inverter"``       — single-stage CMOS inverter (N + P)
- ``"nand2"``          — 2-input NAND (coming in Phase-2)
- ``"bit_cell_6t"``    — 6T SRAM bit cell (routing stubs for Phase-2)
"""
from layout_gen.synth.loader      import (
    load_template,
    CellTemplate,
    DeviceSpec,
    RoutingSpec,
    PortSpec,
)
from layout_gen.synth.constraints import (
    eval_expr,
    resolve_named_constraints,
    build_namespace,
)
from layout_gen.synth.placer      import (
    Placer,
    PlacedDevice,
    TerminalGeom,
    resolve_terminal,
    global_gate_x,
    global_sd_x,
    global_diff_y,
    global_poly_top,
    global_poly_bottom,
)
from layout_gen.synth.router      import (
    Router,
    PortCandidate,
    register_style,
)
from layout_gen.synth.synthesizer import (
    Synthesizer,
    SynthResult,
    MLModel,
    PortResolutionError,
)

__all__ = [
    # Template loading
    "load_template",
    "CellTemplate",
    "DeviceSpec",
    "RoutingSpec",
    "PortSpec",
    # Constraint evaluation
    "eval_expr",
    "resolve_named_constraints",
    "build_namespace",
    # Placement
    "Placer",
    "PlacedDevice",
    "TerminalGeom",
    "resolve_terminal",
    "global_gate_x",
    "global_sd_x",
    "global_diff_y",
    "global_poly_top",
    "global_poly_bottom",
    # Routing
    "Router",
    "PortCandidate",
    "register_style",
    # Synthesis
    "Synthesizer",
    "SynthResult",
    "MLModel",
    "PortResolutionError",
]
