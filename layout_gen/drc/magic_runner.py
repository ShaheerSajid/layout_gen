"""
layout_gen.drc.magic_runner — Magic DRC backend.

Runs Magic in batch mode with the sky130A DRC deck, parses the
structured output, and returns :class:`DRCViolation` objects.

Invocation pattern::

    magic -dnull -noconsole -rcfile <pdk>.magicrc <script>.tcl

The Tcl script reads the GDS, runs ``drc catchup``, then
``drc listall why`` to collect all violations with coordinates.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List

from layout_gen.pdk import PDKRules
from layout_gen.drc.base import DRCRunner, DRCViolation


# ── Layer name heuristics ────────────────────────────────────────────────────

_LAYER_FROM_RULE = re.compile(
    r"(nwell|diff|tap|poly|licon|li1|mcon|met[1-5]|via[1-4]|npc|nsdm|psdm)",
    re.IGNORECASE,
)


def _guess_layer(rule_text: str) -> str:
    """Extract a logical layer name from a DRC rule description."""
    m = _LAYER_FROM_RULE.search(rule_text)
    return m.group(1).lower() if m else ""


def _guess_rule_id(rule_text: str) -> str:
    """Extract a compact rule ID like 'poly.2' from the description.

    Magic DRC descriptions look like:
      "P-diff distance to N-tap must be >= 0.130um (difftap.2)"
    The parenthesised suffix is the rule code.
    """
    m = re.search(r"\(([a-zA-Z0-9_.]+)\)\s*$", rule_text)
    if m:
        return m.group(1)
    # Fallback: first word + number-ish pattern
    m2 = re.search(r"([a-zA-Z]+[\d.]+)", rule_text)
    return m2.group(1) if m2 else "unknown"


class MagicDRCRunner(DRCRunner):
    """Magic DRC backend.

    Parameters
    ----------
    rules :
        PDK rules (used to locate the ``.magicrc`` file).
    magic_exe :
        Path to the ``magic`` executable.
    """

    def __init__(self, rules: PDKRules, magic_exe: str = "magic"):
        self.rules = rules
        self.magic_exe = magic_exe

    @property
    def tool_name(self) -> str:
        return "magic"

    def is_available(self) -> bool:
        try:
            subprocess.run(
                [self.magic_exe, "--version"],
                capture_output=True,
                timeout=10,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── rcfile discovery ─────────────────────────────────────────────────

    def _find_rcfile(self) -> str:
        """Locate the Magic ``.magicrc`` for the active PDK."""
        pdk_root = os.environ.get("PDK_ROOT", "/usr/local/share/pdk")
        pdk_name = getattr(self.rules, "name", "sky130A")

        # Try standard PDK path
        rc = Path(pdk_root) / pdk_name / "libs.tech" / "magic" / f"{pdk_name}.magicrc"
        if rc.exists():
            return str(rc)

        # Try DRC section in PDK YAML
        drc_cfg = getattr(self.rules, "drc", None)
        if isinstance(drc_cfg, dict):
            tech = drc_cfg.get("magic", "")
            if tech:
                tech = os.path.expandvars(tech)
                # .tech file lives next to .magicrc
                rc2 = Path(tech).parent / f"{pdk_name}.magicrc"
                if rc2.exists():
                    return str(rc2)

        raise FileNotFoundError(
            f"Cannot find Magic rcfile for PDK {pdk_name!r}. "
            f"Set PDK_ROOT or provide the path via the PDK YAML drc.magic field."
        )

    # ── DRC execution ────────────────────────────────────────────────────

    def run(
        self,
        gds_path: Path,
        cell_name: str | None = None,
    ) -> List[DRCViolation]:
        """Run full DRC via Magic and return violations."""
        gds_path = Path(gds_path).resolve()
        if not gds_path.exists():
            raise FileNotFoundError(f"GDS file not found: {gds_path}")

        rcfile = self._find_rcfile()

        # Determine cell name — prefer user-supplied, then scan the GDS
        if cell_name is None:
            cell_name = gds_path.stem

        # Flatten into a clean single-cell GDS so Magic always finds it.
        # gdsfactory wraps in $$$CONTEXT_INFO$$$ which Magic can't load.
        with tempfile.TemporaryDirectory(prefix="magic_drc_") as tmpdir:
            flat_gds = Path(tmpdir) / "flat.gds"
            cell_name = self._flatten_gds(gds_path, flat_gds, cell_name)
            gds_path = flat_gds
            output_file = Path(tmpdir) / "drc_results.txt"
            tcl_script = Path(tmpdir) / "run_drc.tcl"

            # Write Tcl DRC script
            tcl_script.write_text(self._generate_tcl(
                gds_path=str(gds_path),
                cell_name=cell_name,
                output_file=str(output_file),
            ))

            env = os.environ.copy()
            env["MAGTYPE"] = "mag"

            proc = subprocess.run(
                [self.magic_exe, "-dnull", "-noconsole", "-rcfile", rcfile,
                 str(tcl_script)],
                env=env,
                cwd=tmpdir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=300,
            )

            if proc.returncode != 0:
                # Magic may still produce partial output — try to parse
                import warnings
                warnings.warn(
                    f"Magic exited with code {proc.returncode}.\n"
                    f"stderr: {proc.stderr[:500]}",
                    stacklevel=2,
                )

            if not output_file.exists():
                # No output file — return empty or raise
                import warnings
                warnings.warn(
                    f"Magic DRC produced no output file. "
                    f"stdout: {proc.stdout[:500]}",
                    stacklevel=2,
                )
                return []

            return self._parse_results(output_file.read_text())

    # ── GDS flattening ─────────────────────────────────────────────────

    @staticmethod
    def _flatten_gds(src: Path, dst: Path, cell_name: str) -> str:
        """Flatten the GDS into a single cell for Magic.

        gdsfactory often wraps the real cell in ``$$$CONTEXT_INFO$$$``.
        This method finds the target cell, flattens it, and writes a
        clean single-cell GDS that Magic can load reliably.

        Returns the actual cell name used.
        """
        import gdstk

        lib = gdstk.read_gds(str(src))

        # Find the target cell
        target = None
        for c in lib.cells:
            if c.name == cell_name:
                target = c
                break

        if target is None:
            # Try partial match (strip synth_ prefix etc.)
            for c in lib.cells:
                if "$$$" not in c.name and c.polygons:
                    target = c
                    break

        if target is None:
            # Fall back to first top-level cell
            tops = lib.top_level()
            target = tops[0] if tops else lib.cells[0]

        # Flatten all references into the target cell
        target.flatten()

        # Sanitise cell name for Magic (no dots, dollar signs)
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", target.name)
        target.name = safe_name

        # Write single-cell library
        out_lib = gdstk.Library()
        out_lib.add(target)
        out_lib.write_gds(str(dst))

        return safe_name

    # ── Tcl script generation ────────────────────────────────────────────

    @staticmethod
    def _generate_tcl(
        gds_path: str,
        cell_name: str,
        output_file: str,
    ) -> str:
        """Generate the Magic DRC batch Tcl script."""
        return f"""\
# Auto-generated Magic DRC script
crashbackups stop
drc euclidean on
drc style drc(full)
drc on
snap internal
gds flatglob *__example_*
gds flatten true
gds read {gds_path}
load {cell_name}
select top cell
expand
drc catchup
set allerrors [drc listall why]
set oscale [cif scale out]
set ofile [open {output_file} w]
puts $ofile "DRC errors for cell {cell_name}"
puts $ofile "--------------------------------------------"
foreach {{whytext rectlist}} $allerrors {{
   puts $ofile ""
   puts $ofile $whytext
   foreach rect $rectlist {{
       set llx [format "%.3f" [expr $oscale * [lindex $rect 0]]]
       set lly [format "%.3f" [expr $oscale * [lindex $rect 1]]]
       set urx [format "%.3f" [expr $oscale * [lindex $rect 2]]]
       set ury [format "%.3f" [expr $oscale * [lindex $rect 3]]]
       puts $ofile "$llx $lly $urx $ury"
   }}
}}
close $ofile
puts "DRC complete: [llength $allerrors] rule(s) with violations"
quit
"""

    # ── Output parsing ───────────────────────────────────────────────────

    @staticmethod
    def _parse_results(text: str) -> List[DRCViolation]:
        """Parse Magic DRC output text into DRCViolation objects.

        Format::

            DRC errors for cell <name>
            --------------------------------------------

            <rule description text>
            x0 y0 x1 y1
            x0 y0 x1 y1
            ...

            <next rule description>
            ...
        """
        violations: list[DRCViolation] = []
        current_rule = ""
        coord_re = re.compile(
            r"^\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$"
        )

        for line in text.splitlines():
            line = line.strip()

            # Skip header lines
            if not line or line.startswith("DRC errors") or line.startswith("---"):
                continue

            # Try to parse as coordinates
            m = coord_re.match(line)
            if m and current_rule:
                x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                violations.append(DRCViolation(
                    rule=_guess_rule_id(current_rule),
                    description=current_rule,
                    layer=_guess_layer(current_rule),
                    severity="error",
                    x=cx,
                    y=cy,
                ))
            else:
                # This is a rule description line
                current_rule = line

        return violations
