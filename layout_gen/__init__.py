"""layout_gen — parametric cell layout generation with DRC-aware repair.

Generates DRC-clean GDS layouts from parameterized cell descriptions.
Designed to work alongside spice_gen / liberty_gen / verilog_gen in the
fabram SRAM compiler pipeline.

Pipeline
--------
1. Morphable templates  — parametric gdsfactory cells, one per SRAM building block
2. DRC repair graph     — typed violation → repair action → converge to clean layout

Typical use::

    from layout_gen import CellLayout, generate_gds

    layout = generate_gds("bit_cell", W_PD=0.52, W_PU=0.42, W_PG=0.48)
    layout.write_gds("out/bit_cell.gds")
"""
