"""
layout_gen.rl.env.runner — cached DRC adapter for the RL environment.

Wraps a :class:`~layout_gen.drc.base.DRCRunner` with an LRU cache keyed
on the layout's geometry hash. Identical states are checked once; the
RL policy can probe many candidate edits without paying for redundant
DRC tool invocations.

Implementation notes
--------------------
* The cache key is the sorted tuple of ``(layer, x0, y0, x1, y1)`` rounded
  to 1 nm — this is finer than every PDK manufacturing grid we ship.
* The runner writes the layout to a tempfile, invokes DRC, and discards
  the file. Long-lived caching of the GDS itself is unnecessary because
  the violations dominate the cost (DRC tool startup + execution).
* LVS support is deferred to Phase 2 — :meth:`run_lvs` is a stub that
  raises so we don't silently return wrong values.
"""
from __future__ import annotations

import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layout_gen.drc.base import DRCRunner, DRCViolation
from layout_gen.synth.geo.state import LayoutState


# ── Geometry hashing ─────────────────────────────────────────────────────────

def geometry_key(state: LayoutState, *, ndigits: int = 6) -> tuple:
    """Stable, hashable digest of *state*'s geometry.

    Two states with identical rounded ``(layer, x0, y0, x1, y1)`` tuples
    map to the same key regardless of insertion order or rectangle IDs.
    """
    rows = sorted(
        (r.layer,
         round(r.x0, ndigits), round(r.y0, ndigits),
         round(r.x1, ndigits), round(r.y1, ndigits))
        for r in state
    )
    return tuple(rows)


# ── Cached runner ────────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    violations: list[DRCViolation]
    hits:       int = 0


class CachedDRC:
    """LRU-cached front-end for a DRCRunner.

    Parameters
    ----------
    runner :
        Any concrete :class:`DRCRunner` (KLayout, Magic, …).
    rules :
        :class:`PDKRules` — required to convert :class:`LayoutState` to a
        gdsfactory Component for GDS writing.
    cell_name :
        Name written into the top GDS cell. Must match what the DRC
        runner is asked to check.
    capacity :
        Maximum number of distinct geometries to remember. The repair
        policy frequently revisits states; caching even ~1k entries
        cuts wall time by orders of magnitude vs. cold DRC runs.
    """

    def __init__(
        self,
        runner:    DRCRunner,
        rules:     Any,
        *,
        cell_name: str,
        capacity:  int = 4096,
    ) -> None:
        self._runner    = runner
        self._rules     = rules
        self._cell_name = cell_name
        self._capacity  = capacity
        self._cache: OrderedDict[tuple, _CacheEntry] = OrderedDict()
        self._misses = 0
        self._hits   = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, state: LayoutState) -> list[DRCViolation]:
        """Return DRC violations for *state*, served from cache when possible."""
        key = geometry_key(state)
        entry = self._cache.get(key)
        if entry is not None:
            entry.hits += 1
            self._hits += 1
            self._cache.move_to_end(key)
            return entry.violations

        violations = self._invoke_tool(state)
        self._cache[key] = _CacheEntry(violations=violations)
        self._misses += 1
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return violations

    def count(self, state: LayoutState) -> int:
        return len(self.run(state))

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses,
                "size": len(self._cache), "capacity": self._capacity}

    def clear(self) -> None:
        self._cache.clear()
        self._hits = self._misses = 0

    # ── Internals ────────────────────────────────────────────────────────────

    def _invoke_tool(self, state: LayoutState) -> list[DRCViolation]:
        with tempfile.TemporaryDirectory(prefix="rl_drc_") as td:
            gds_path = Path(td) / f"{self._cell_name}.gds"
            comp = state.to_component(self._rules, name=self._cell_name)
            comp.write_gds(str(gds_path), with_metadata=False)
            return self._runner.run(gds_path, self._cell_name)


__all__ = ["CachedDRC", "geometry_key"]
