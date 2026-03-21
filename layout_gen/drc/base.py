"""
layout_gen.drc.base — tool-agnostic DRC interface.

Every DRC backend (KLayout, Magic, Calibre, ICV, …) implements
:class:`DRCRunner` and returns a list of :class:`DRCViolation` objects.
Downstream code (ML synthesizer, CLI reporter) only ever sees this interface.

Adding a new tool
-----------------
1. Subclass :class:`DRCRunner`.
2. Implement :meth:`tool_name` and :meth:`run`.
3. Register with :func:`layout_gen.drc.registry.register`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DRCViolation:
    """One DRC violation instance returned by any backend.

    Attributes
    ----------
    rule :
        Rule identifier, e.g. ``"poly.2"``, ``"li1.1"``.
    description :
        Human-readable description from the DRC deck.
    layer :
        Logical PDK layer name involved (empty if not determinable).
    severity :
        ``"error"`` (blocks signoff) or ``"warning"`` (informational).
    x, y :
        Centroid of the violating geometry in µm (cell coordinates).
    value :
        Measured value that caused the violation (µm), or ``None`` if
        the backend does not report it.
    """

    rule:        str
    description: str
    layer:       str   = ""
    severity:    str   = "error"
    x:           float = 0.0
    y:           float = 0.0
    value:       float | None = None

    def __repr__(self) -> str:
        loc = f"({self.x:.3f}, {self.y:.3f})"
        val = f" measured={self.value:.4f}" if self.value is not None else ""
        return f"DRCViolation({self.rule!r}{val} @ {loc})"


class DRCRunner(ABC):
    """Abstract base for DRC tool backends.

    Parameters
    ----------
    rules :
        PDK rules object.  The runner uses it to generate the DRC deck
        and to resolve logical layer names from violation reports.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Short identifier for this backend, e.g. ``"klayout"``."""

    @abstractmethod
    def run(
        self,
        gds_path: Path,
        cell_name: str | None = None,
    ) -> List[DRCViolation]:
        """Run DRC on *gds_path* and return all violations.

        Parameters
        ----------
        gds_path :
            Path to the GDS file.
        cell_name :
            Top-cell name to check.  If ``None`` the runner picks the
            last cell in the file (tool-dependent behaviour).
        """

    def is_available(self) -> bool:
        """Return ``True`` if the tool executable is reachable."""
        return True

    def count(
        self,
        gds_path: Path,
        cell_name: str | None = None,
    ) -> int:
        """Convenience: return the number of DRC violations."""
        return len(self.run(gds_path, cell_name))
