"""
layout_gen.drc — tool-agnostic DRC runner framework.

Quick start::

    from layout_gen import load_pdk, run_drc

    rules      = load_pdk()
    violations = run_drc("out/bit_cell.gds", rules, tool="klayout")
    print(f"{len(violations)} violations")
    for v in violations:
        print(v)

Adding a new tool backend
--------------------------
1. Create ``layout_gen/drc/<tool>_runner.py`` with a subclass of
   :class:`~layout_gen.drc.base.DRCRunner`.
2. Register it below with :func:`registry.register`.
3. Users pass ``tool="<name>"`` to :func:`run_drc`.

Public API
----------
:class:`DRCViolation`   — one violation (rule, description, x, y, value)
:class:`DRCRunner`      — ABC for all backends
:func:`run_drc`         — convenience entry-point
:func:`available_tools` — list registered tool names
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from layout_gen.drc.base            import DRCViolation, DRCRunner
from layout_gen.drc.klayout_runner  import KLayoutDRCRunner
from layout_gen.drc.magic_runner    import MagicDRCRunner
from layout_gen.drc                 import registry

# ── Register built-in backends ────────────────────────────────────────────────
registry.register("klayout", KLayoutDRCRunner)
registry.register("magic",   MagicDRCRunner)


def run_drc(
    gds_path,
    rules,
    tool: str = "klayout",
    cell_name: str | None = None,
    **runner_kwargs,
) -> List[DRCViolation]:
    """Run DRC on *gds_path* with the specified *tool* backend.

    Parameters
    ----------
    gds_path :
        Path to the GDS file.
    rules :
        :class:`~layout_gen.pdk.PDKRules` (from :func:`~layout_gen.pdk.load_pdk`).
    tool :
        Backend name — ``"klayout"`` (default) or ``"magic"``.
        Extend via :func:`layout_gen.drc.registry.register`.
    cell_name :
        Top-cell name to check.  ``None`` → tool picks automatically.
    **runner_kwargs :
        Extra keyword arguments forwarded to the runner's ``__init__``
        (e.g. ``klayout_exe="/opt/klayout/bin/klayout"``).

    Returns
    -------
    list[DRCViolation]
        Empty list = clean.
    """
    runner = registry.get(tool, rules=rules, **runner_kwargs)
    return runner.run(Path(gds_path), cell_name)


def available_tools() -> list[str]:
    """Return tool names that have a registered DRC backend."""
    return registry.available()


__all__ = [
    "DRCViolation",
    "DRCRunner",
    "KLayoutDRCRunner",
    "MagicDRCRunner",
    "run_drc",
    "available_tools",
]
