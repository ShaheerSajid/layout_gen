# layout_gen

Parametric SRAM cell layout generation with DRC-aware repair. Part of the
[fabram](https://github.com/ShaheerSajid/fabram) compiler pipeline.

## Install

```bash
pip install -e .          # core (gdsfactory + klayout)
pip install -e ".[ml]"    # + PyTorch + torch-geometric for the repair agent
```

## Pipeline

```
Optimizer W/L values
        ↓
Phase 1 — Morphable templates
  Parametric gdsfactory cells, finger solver, DRC-by-construction
        ↓
Phase 2 — DRC repair graph  (WIP)
  Typed violation → repair action → converge to clean layout
        ↓
GDS output
```
