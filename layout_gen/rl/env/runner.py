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

import itertools
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layout_gen.drc.base import DRCRunner, DRCViolation
from layout_gen.lvs.base import LVSResult, LVSRunner
from layout_gen.synth.geo.state import LayoutState


# Process-global, thread-safe counter for DRC/LVS cell-name suffixes.
# gdsfactory's KCLayout cell registry is process-global, so per-instance
# counters collide when multiple CachedDRC instances live in one process
# (e.g. DummyVecEnv with n_envs>1). itertools.count is GIL-atomic in
# CPython; the lock guards future-proofing against free-threading builds.
_CELL_NAME_COUNTER = itertools.count(1)
_CELL_NAME_LOCK = threading.Lock()


def _next_cell_suffix() -> int:
    with _CELL_NAME_LOCK:
        return next(_CELL_NAME_COUNTER)


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
        # Unique per-call name keeps gdsfactory's process-global cell
        # registry collision-free across vectorised envs (DummyVecEnv
        # with n_envs>1 puts multiple CachedDRC instances in the same
        # process). The DRC tool only ever sees one cell per tempfile,
        # so the suffix is harmless to the violation report.
        unique_name = f"{self._cell_name}_drc{_next_cell_suffix()}"
        with tempfile.TemporaryDirectory(prefix="rl_drc_") as td:
            gds_path = Path(td) / f"{unique_name}.gds"
            comp = state.to_component(self._rules, name=unique_name)
            comp.write_gds(str(gds_path), with_metadata=False)
            return self._runner.run(gds_path, unique_name)


# ── Cached LVS ─────────────────────────────────────────────────────────────

@dataclass
class _LVSCacheEntry:
    result: LVSResult
    hits:   int = 0


class CachedLVS:
    """LRU-cached front-end for an :class:`LVSRunner`.

    Same caching contract as :class:`CachedDRC`: identical layout
    geometries hit the cache, novel ones invoke magic+netgen.

    Parameters
    ----------
    runner :
        Concrete :class:`LVSRunner` (today: :class:`MagicNetgenLVSRunner`).
    rules :
        :class:`PDKRules` for ``LayoutState.to_component`` GDS export.
    cell_name :
        Top cell name; must match the reference netlist's ``.subckt``.
    ref_netlist :
        Path to the reference SPICE netlist
        (auto-emitted from the topology graph by
        :func:`layout_gen.rl.env.spice_ref.write_spice_subckt`).
    capacity :
        Max distinct layouts to remember.
    """

    def __init__(
        self,
        runner:       LVSRunner,
        rules:        Any,
        *,
        cell_name:    str,
        ref_netlist:  Path,
        capacity:     int = 1024,
    ) -> None:
        self._runner       = runner
        self._rules        = rules
        self._cell_name    = cell_name
        self._ref_netlist  = Path(ref_netlist)
        self._capacity     = capacity
        self._cache: OrderedDict[tuple, _LVSCacheEntry] = OrderedDict()
        self._misses = 0
        self._hits   = 0

    def run(self, state: LayoutState) -> LVSResult:
        key = geometry_key(state)
        entry = self._cache.get(key)
        if entry is not None:
            entry.hits += 1
            self._hits += 1
            self._cache.move_to_end(key)
            return entry.result
        result = self._invoke_tool(state)
        self._cache[key] = _LVSCacheEntry(result=result)
        self._misses += 1
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return result

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses,
                "size": len(self._cache), "capacity": self._capacity}

    def clear(self) -> None:
        self._cache.clear()
        self._hits = self._misses = 0

    def _invoke_tool(self, state: LayoutState) -> LVSResult:
        unique_name = f"{self._cell_name}_lvs{_next_cell_suffix()}"
        with tempfile.TemporaryDirectory(prefix="rl_lvs_") as td:
            gds_path = Path(td) / f"{unique_name}.gds"
            comp = state.to_component(self._rules, name=unique_name)
            comp.write_gds(str(gds_path), with_metadata=False)
            try:
                return self._runner.run(gds_path, self._ref_netlist, unique_name)
            except Exception as exc:
                # LVS tool errored (e.g. magic crashed on degenerate
                # geometry). Return a failed-but-non-crashing result so
                # the env keeps training.
                return LVSResult(clean=False,
                                  mismatches=[], log=f"runner exception: {exc}")


__all__ = ["CachedDRC", "CachedLVS", "geometry_key"]
