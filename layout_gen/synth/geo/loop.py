"""
layout_gen.synth.geo.loop — Iterative geometric DRC fix loop.

Orchestrates the agent → DRC → agent cycle:

1. Import layout into :class:`~.state.LayoutState`
2. Run DRC → parse violations
3. Ask agent to propose fixes
4. Apply fixes to LayoutState
5. Export to GDS → re-run DRC
6. Repeat until clean or max iterations

Usage::

    from layout_gen.synth.geo import GeoFixLoop, RuleGeoAgent

    agent = RuleGeoAgent(rules)
    loop  = GeoFixLoop(agent, drc_runner, rules)
    state, remaining = loop.run(component, max_iter=20)
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from layout_gen.drc.base import DRCRunner, DRCViolation
from layout_gen.pdk     import PDKRules
from layout_gen.synth.geo.state      import LayoutState
from layout_gen.synth.geo.actions    import Action, apply_action
from layout_gen.synth.geo.violations import ViolationInfo, parse_violations
from layout_gen.synth.geo.agent      import GeoFixAgent


log = logging.getLogger(__name__)

_geo_counter = 0


@dataclass
class FixRecord:
    """One iteration of the fix loop (for Phase 3 replay / training)."""
    iteration:  int
    violations: list[ViolationInfo]
    actions:    list[Action]
    remaining:  int   # violations after applying actions


@dataclass
class GeoFixResult:
    """Result of a geometric fix loop run."""
    state:      LayoutState
    violations: list[DRCViolation]
    converged:  bool
    iterations: int
    history:    list[FixRecord] = field(default_factory=list)


class GeoFixLoop:
    """Iterative geometric DRC repair loop.

    Parameters
    ----------
    agent :
        The geometric fix agent (rule-based or learned).
    drc_runner :
        DRC backend (KLayout, Magic, etc.).
    rules :
        PDK rules.
    """

    def __init__(
        self,
        agent:      GeoFixAgent,
        drc_runner: DRCRunner,
        rules:      PDKRules,
    ):
        self.agent      = agent
        self.drc_runner = drc_runner
        self.rules      = rules

    def run(
        self,
        component: Any,
        max_iter:  int = 20,
        state:     LayoutState | None = None,
    ) -> GeoFixResult:
        """Run the fix loop on *component*.

        Parameters
        ----------
        component :
            Initial gdsfactory Component.  DRC is always run on the full
            component (not the simplified LayoutState export) so that
            transistor primitives and other subcell geometry are included.
        max_iter :
            Maximum repair iterations.
        state :
            Pre-built LayoutState (skips import from component).

        Returns
        -------
        GeoFixResult
            Contains the final LayoutState, remaining violations,
            convergence status, and full fix history (for RL training).
        """
        if state is None:
            state = LayoutState.from_component(component, self.rules)

        # Detect via/contact stacks so the agent can move them as groups
        n_groups = state.tag_via_groups()
        if n_groups:
            log.info("GeoFixLoop: tagged %d via/contact group(s)", n_groups)

        history: list[FixRecord] = []
        violations: list[DRCViolation] = []

        # Use the full component for DRC when provided (preserves transistor
        # cells, implants, nwell, etc. that LayoutState doesn't capture)
        use_full_drc = component is not None

        for iteration in range(1, max_iter + 1):
            # ── DRC on the current state ─────────────────────────────────
            if use_full_drc:
                violations = self._run_drc_full(component)
            else:
                violations = self._run_drc(state)
            if not violations:
                log.info("GeoFixLoop: DRC clean at iteration %d", iteration)
                return GeoFixResult(state, [], True, iteration, history)

            # ── Parse ───────────────────────────────────────────────────
            parsed = parse_violations(violations)
            log.info("GeoFixLoop iter %d: %d violation(s)", iteration, len(parsed))

            # ── Agent proposes fixes ────────────────────────────────────
            actions = self.agent.fix_batch(state, parsed)
            if not actions:
                log.warning("GeoFixLoop: agent proposed no fixes; stopping")
                break

            # ── Apply ───────────────────────────────────────────────────
            for action in actions:
                log.debug("  %s", action.describe() if hasattr(action, 'describe') else action)
                try:
                    apply_action(state, action)
                except KeyError:
                    # Shape was already removed by a prior fix in this batch
                    log.debug("  (skipped — target shape no longer exists)")
                    continue

            # ── Record ──────────────────────────────────────────────────
            history.append(FixRecord(
                iteration=iteration,
                violations=parsed,
                actions=actions,
                remaining=0,  # updated below after DRC
            ))

            # Feed experience to learned agent if applicable
            from layout_gen.synth.geo.learned_agent import LearnedGeoAgent
            if isinstance(self.agent, LearnedGeoAgent):
                # remaining will be updated after next DRC run
                self.agent.record_outcome(
                    state, parsed, actions, len(parsed),
                )

        # Final DRC check
        if use_full_drc:
            violations = self._run_drc_full(component)
        else:
            violations = self._run_drc(state)
        if history:
            history[-1].remaining = len(violations)

        return GeoFixResult(
            state, violations,
            converged=(len(violations) == 0),
            iterations=len(history),
            history=history,
        )

    def _run_drc_full(self, component: Any) -> list[DRCViolation]:
        """Run DRC on the full component (with all subcells)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gds = Path(tmpdir) / "check.gds"
            component.write_gds(str(gds))
            return self.drc_runner.run(gds, component.name)

    def _run_drc(self, state: LayoutState) -> list[DRCViolation]:
        """Export state to GDS, run DRC, return violations (simplified)."""
        global _geo_counter
        _geo_counter += 1
        comp = state.to_component(self.rules, name=f"geo_fix_check_{_geo_counter}")
        with tempfile.TemporaryDirectory() as tmpdir:
            gds = Path(tmpdir) / "check.gds"
            comp.write_gds(str(gds))
            return self.drc_runner.run(gds)
