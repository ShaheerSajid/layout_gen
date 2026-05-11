"""
layout_gen.rl — reinforcement-learning layout generator.

PDK-agnostic RL system that takes a topology YAML (devices + nets +
placement_logic + routing hints) and produces a DRC/LVS-clean layout.
SRAM cells are the starting point; the same policy generalises across
cell types because all PDK-specific knowledge is hidden behind the
DRC/LVS runner interface.

Phases of the action space (built incrementally):

1. **Repair** — given a (broken) layout, edit polygons to clear DRC.
   Uses the 6 ``perturb`` primitives (shift_edge, shrink, grow,
   translate, delete, nudge_offgrid).
2. **Place** — pick devices from the topology and lay them down.
3. **Route** — connect placed devices to satisfy the netlist.

Phase 1 of the build (this checkpoint) implements only the **repair**
action set, sufficient to train a policy that matches/exceeds the
deprecated diffusion baseline. Place/route action heads are added
later without changing the env's outer interface.
"""
from __future__ import annotations
