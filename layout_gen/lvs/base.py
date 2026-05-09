"""
layout_gen.lvs.base — tool-agnostic LVS interface.

LVS (Layout vs Schematic) compares the netlist extracted from a synthesized
GDS against a reference netlist derived from the cell topology template.
Backends (Magic+Netgen, KLayout) implement :class:`LVSRunner`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class LVSMismatch:
    """One LVS comparison mismatch.

    Attributes
    ----------
    kind :
        ``"net"``, ``"device"``, ``"port"``, ``"property"``, or ``"unknown"``.
    description :
        Human-readable text from the LVS tool.
    layout_obj :
        Object name on the layout side (empty if not parseable).
    schema_obj :
        Object name on the schematic side (empty if not parseable).
    """
    kind:        str
    description: str
    layout_obj:  str = ""
    schema_obj:  str = ""

    def __repr__(self) -> str:
        return f"LVSMismatch({self.kind!r}: {self.description[:60]!r})"


@dataclass
class LVSResult:
    """Outcome of one LVS run.

    Attributes
    ----------
    clean :
        True iff the layout and reference netlist match.
    mismatches :
        Parsed mismatch list (empty when clean).
    layout_devices, schema_devices :
        Device counts on each side (0 if not reported by the tool).
    layout_nets, schema_nets :
        Net counts on each side (0 if not reported).
    log :
        Raw tool log (for debugging).
    """
    clean:           bool
    mismatches:      List[LVSMismatch] = field(default_factory=list)
    layout_devices:  int  = 0
    schema_devices:  int  = 0
    layout_nets:     int  = 0
    schema_nets:     int  = 0
    log:             str  = ""

    def __bool__(self) -> bool:
        return self.clean


class LVSRunner(ABC):
    """Abstract base for LVS tool backends."""

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Short identifier, e.g. ``"magic_netgen"``."""

    @abstractmethod
    def run(
        self,
        gds_path:     Path,
        ref_netlist:  Path,
        cell_name:    str,
    ) -> LVSResult:
        """Compare *gds_path* against *ref_netlist* for *cell_name*.

        Both netlists must use the same cell name as their top subckt.
        """

    def is_available(self) -> bool:
        """True when the underlying tool is reachable."""
        return True
