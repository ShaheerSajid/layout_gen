"""
layout_gen.synth.geo — Geometric DRC fix agent.

Technology-agnostic layout repair via polygon-level geometric operations.

Architecture
------------
Phase 1 (existing): ``fix_policy.DRCFixPredictor`` adjusts *parameters*
    (w, l, gap) to avoid violations.  Works for initial layout generation.

Phase 2 (this package): ``RuleGeoAgent`` applies *geometric actions*
    (stretch edge, move shape, merge) directly on layout polygons.
    It parses DRC violation descriptions to decide what to do — no
    hardcoded rule→fix mapping per technology.

Phase 3 (future): ``LearnedGeoAgent`` replaces rule-based heuristics
    with a trained policy network (GNN/Transformer + RL).  Same action
    space, same violation parser — just a learned decision function.

Usage::

    from layout_gen.synth.geo import GeoFixLoop, RuleGeoAgent

    agent = RuleGeoAgent(rules)
    loop  = GeoFixLoop(agent, drc_runner, rules)
    state, violations = loop.run(component, max_iter=20)
    if not violations:
        state.to_component()  # DRC-clean layout
"""

from layout_gen.synth.geo.state      import LayoutState, Rect
from layout_gen.synth.geo.actions    import (
    Action, StretchEdge, MoveShape, AddRect, RemoveShape, MergeShapes,
    apply_action,
)
from layout_gen.synth.geo.violations import ViolationInfo, parse_violation
from layout_gen.synth.geo.agent      import GeoFixAgent, RuleGeoAgent
from layout_gen.synth.geo.loop       import GeoFixLoop

__all__ = [
    "LayoutState", "Rect",
    "Action", "StretchEdge", "MoveShape", "AddRect", "RemoveShape", "MergeShapes",
    "apply_action",
    "ViolationInfo", "parse_violation",
    "GeoFixAgent", "RuleGeoAgent",
    "GeoFixLoop",
]
