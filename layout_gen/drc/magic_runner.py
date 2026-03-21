"""
layout_gen.drc.magic_runner — Magic DRC backend (stub).

Implement :meth:`MagicDRCRunner.run` to support Magic.  The expected
invocation pattern::

    magic -dnull -noconsole -rcfile $PDK_ROOT/sky130A/libs.tech/magic/sky130A.magicrc \\
          << EOF
    gds read cell.gds
    load CELL_NAME
    drc catchup
    drc listall count
    quit
    EOF

Parse the text output for violation counts and locations.

The :class:`~layout_gen.drc.base.DRCRunner` interface is unchanged —
callers never need to know which tool is running.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from layout_gen.pdk import PDKRules
from layout_gen.drc.base import DRCRunner, DRCViolation


class MagicDRCRunner(DRCRunner):
    """Magic DRC backend — not yet implemented."""

    def __init__(self, rules: PDKRules, magic_exe: str = "magic"):
        self.rules = rules
        self.magic_exe = magic_exe

    @property
    def tool_name(self) -> str:
        return "magic"

    def is_available(self) -> bool:
        import subprocess
        try:
            subprocess.run(
                [self.magic_exe, "--version"],
                capture_output=True,
                timeout=10,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run(
        self,
        gds_path: Path,
        cell_name: str | None = None,
    ) -> List[DRCViolation]:
        raise NotImplementedError(
            "Magic DRC runner is not yet implemented. "
            "Use tool='klayout' for now."
        )
