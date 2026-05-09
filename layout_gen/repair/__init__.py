"""
layout_gen.repair — autonomous DRC repair engine.

Closed-domain (CMOS memory) layout repair. The engine consumes a layout
+ a DRC tool's violation report and produces an action sequence that
drives the layout to DRC-clean.

Sub-modules
-----------
- :mod:`~layout_gen.repair.catalog`   — DRC rule taxonomy (rule -> category,
  layer roles, fix-zone radius, sample violations).  Built empirically
  from real DRC runs across PDKs; it is *data*, not policy.
- :mod:`~layout_gen.repair.primitives` — registry of canonical memory
  primitives (bitcell, sense amp, write driver, …) that the engine knows.
- :mod:`~layout_gen.repair.perturb`    — procedural perturbation library:
  given a clean layout, produce labelled (problem, fix) pairs.
- :mod:`~layout_gen.repair.zones`      — conflict-graph + zone extractor.
- :mod:`~layout_gen.repair.skills`     — per-zone repair skills (rule-based
  initially, ML-replaceable later).

Design principles
-----------------
1. **PDK-agnostic.**  No PDK rule values appear in code.  The only numerical
   inputs to repair logic come from the DRC tool's violation report at runtime
   (measured value, required value parsed from description text, layer name).
2. **Memory-domain scoped.**  We exploit the fact that memory uses a finite,
   well-known set of cell topologies — there is no need to generalise to
   arbitrary IC layout.
3. **Catalog-driven.**  Every supported rule is classified once into a
   universal category (spacing/width/enclosure/area/extension/merge/overlap/
   density) at runtime.  Adding a new PDK = updating the catalog, not code.
"""
from __future__ import annotations
