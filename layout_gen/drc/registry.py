"""
layout_gen.drc.registry — tool name → DRCRunner class mapping.

New backends register here so the rest of the codebase can stay
tool-agnostic::

    from layout_gen.drc import registry
    registry.register("calibre", CalibreDRCRunner)
    runner = registry.get("calibre", rules=my_rules)

Backends registered at import time (in ``layout_gen/drc/__init__.py``):
  - ``"klayout"`` → :class:`~layout_gen.drc.klayout_runner.KLayoutDRCRunner`
  - ``"magic"``   → :class:`~layout_gen.drc.magic_runner.MagicDRCRunner`
"""
from __future__ import annotations

from typing import Type
from layout_gen.drc.base import DRCRunner

_RUNNERS: dict[str, Type[DRCRunner]] = {}


def register(name: str, cls: Type[DRCRunner]) -> None:
    """Register *cls* under *name*.  Overwrites any existing entry."""
    _RUNNERS[name] = cls


def get(name: str, **kwargs) -> DRCRunner:
    """Return an instantiated runner for *name*, passing *kwargs* to ``__init__``.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in _RUNNERS:
        raise KeyError(
            f"No DRC runner registered for tool {name!r}. "
            f"Available: {sorted(_RUNNERS)}"
        )
    return _RUNNERS[name](**kwargs)


def available() -> list[str]:
    """Return sorted list of registered tool names."""
    return sorted(_RUNNERS)
