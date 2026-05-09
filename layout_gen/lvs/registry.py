"""layout_gen.lvs.registry — tool name → LVSRunner class mapping."""
from __future__ import annotations

from typing import Type
from layout_gen.lvs.base import LVSRunner

_RUNNERS: dict[str, Type[LVSRunner]] = {}


def register(name: str, cls: Type[LVSRunner]) -> None:
    """Register *cls* under *name*.  Overwrites any existing entry."""
    _RUNNERS[name] = cls


def get(name: str, **kwargs) -> LVSRunner:
    """Instantiate the runner registered under *name*."""
    if name not in _RUNNERS:
        raise KeyError(
            f"No LVS runner registered for tool {name!r}. "
            f"Available: {sorted(_RUNNERS)}"
        )
    return _RUNNERS[name](**kwargs)


def available() -> list[str]:
    return sorted(_RUNNERS)
