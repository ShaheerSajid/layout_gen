"""
layout_gen.lvs.magic_runner — Magic (extract) + Netgen (compare) LVS backend.

Two-tool flow:

1. **Magic** loads the GDS, runs ``extract all``, then ``ext2spice`` to write
   a layout-side SPICE netlist.
2. **Netgen** compares that netlist against the reference netlist using the
   PDK's ``netgen_setup.tcl`` device-equivalence rules.

Required environment
--------------------
- ``magic`` and ``netgen-lvs`` on ``PATH`` (override with ``magic_exe`` /
  ``netgen_exe`` arguments).
- ``PDK_ROOT`` set, *or* ``lvs.netgen_setup`` and ``drc.magic`` paths
  populated in the PDK YAML.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List

from layout_gen.pdk      import PDKRules
from layout_gen.lvs.base import LVSRunner, LVSResult, LVSMismatch


class MagicNetgenLVSRunner(LVSRunner):
    """Magic + Netgen LVS backend.

    Parameters
    ----------
    rules :
        PDK rules — locates ``.magicrc`` (via ``drc.magic`` or ``PDK_ROOT``)
        and the netgen setup file (via ``lvs.netgen_setup`` or ``PDK_ROOT``).
    magic_exe, netgen_exe :
        Override executable names / paths.
    timeout :
        Per-tool subprocess timeout in seconds (default 300).
    """

    def __init__(
        self,
        rules:      PDKRules,
        magic_exe:  str   = "magic",
        netgen_exe: str   = "netgen-lvs",
        timeout:    int   = 300,
    ):
        self.rules      = rules
        self.magic_exe  = magic_exe
        self.netgen_exe = netgen_exe
        self.timeout    = timeout

    @property
    def tool_name(self) -> str:
        return "magic_netgen"

    def is_available(self) -> bool:
        # `magic --version` exits cleanly; `netgen --version` does NOT —
        # it opens an interactive session.  Probe netgen with `-batch quit`
        # which exits immediately whether or not the LVS sub-language matches.
        probes = [
            (self.magic_exe,  ["--version"]),
            (self.netgen_exe, ["-batch", "quit"]),
        ]
        for exe, args in probes:
            try:
                subprocess.run(
                    [exe, *args],
                    capture_output=True, timeout=10,
                    stdin=subprocess.DEVNULL,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return False
        return True

    # ── Setup file discovery ──────────────────────────────────────────────

    def _find_magicrc(self) -> str:
        """Locate the Magic ``.magicrc`` for the active PDK."""
        pdk_root = os.environ.get("PDK_ROOT", "/usr/local/share/pdk")
        pdk_name = getattr(self.rules, "name", "sky130A")
        rc = Path(pdk_root) / pdk_name / "libs.tech" / "magic" / f"{pdk_name}.magicrc"
        if rc.exists():
            return str(rc)
        drc_cfg = getattr(self.rules, "drc", {}) or {}
        tech = drc_cfg.get("magic", "")
        if tech:
            tech = os.path.expandvars(tech)
            rc2 = Path(tech).parent / f"{pdk_name}.magicrc"
            if rc2.exists():
                return str(rc2)
        raise FileNotFoundError(
            f"Cannot find Magic rcfile for PDK {pdk_name!r}. "
            f"Set PDK_ROOT or populate drc.magic in the PDK YAML."
        )

    def _find_netgen_setup(self) -> str | None:
        """Locate the netgen ``_setup.tcl``.  Returns None if not found.

        Without a setup file, netgen still runs but device-name aliasing
        and bulk-net mappings won't be applied — fine for cells that use
        only generic ``nmos`` / ``pmos`` model names on both sides.
        """
        lvs_cfg = getattr(self.rules, "lvs", {}) or {}
        setup = lvs_cfg.get("netgen_setup", "")
        if setup:
            setup = os.path.expandvars(setup)
            if Path(setup).is_file():
                return setup

        pdk_root = os.environ.get("PDK_ROOT")
        pdk_name = getattr(self.rules, "name", "sky130A")
        if pdk_root:
            for cand in (
                Path(pdk_root) / pdk_name / "libs.tech" / "netgen" /
                                            f"{pdk_name}_setup.tcl",
                Path(pdk_root) / pdk_name / "libs.tech" / "netgen" / "setup.tcl",
            ):
                if cand.is_file():
                    return str(cand)
        return None

    # ── Main entry point ──────────────────────────────────────────────────

    def run(
        self,
        gds_path:    Path,
        ref_netlist: Path,
        cell_name:   str,
    ) -> LVSResult:
        gds_path    = Path(gds_path).resolve()
        ref_netlist = Path(ref_netlist).resolve()
        if not gds_path.exists():
            raise FileNotFoundError(f"GDS not found: {gds_path}")
        if not ref_netlist.exists():
            raise FileNotFoundError(f"Reference netlist not found: {ref_netlist}")

        rcfile = self._find_magicrc()
        setup  = self._find_netgen_setup()

        with tempfile.TemporaryDirectory(prefix="layoutgen_lvs_") as tmpdir:
            tmp = Path(tmpdir)
            flat_gds = tmp / "flat.gds"
            safe_name = self._flatten_gds(gds_path, flat_gds, cell_name)
            ext_spice = tmp / f"{safe_name}.spice"
            log_file  = tmp / "netgen.log"

            # 1) Magic extraction
            self._run_magic_extract(
                flat_gds, safe_name, ext_spice, rcfile, tmp,
            )

            if not ext_spice.exists():
                magic_stdout = getattr(self, "_last_magic_stdout", "") or ""
                magic_stderr = getattr(self, "_last_magic_stderr", "") or ""
                return LVSResult(
                    clean=False,
                    mismatches=[LVSMismatch(
                        kind="extraction",
                        description="Magic produced no .spice output",
                    )],
                    log=("--- magic stdout ---\n" + magic_stdout +
                         "\n--- magic stderr ---\n" + magic_stderr),
                )

            # 2) Netgen compare
            return self._run_netgen(
                ext_spice, ref_netlist, safe_name, cell_name,
                setup, log_file, tmp,
            )

    # ── Magic extraction ──────────────────────────────────────────────────

    def _run_magic_extract(
        self,
        gds:       Path,
        cell:      str,
        out_spice: Path,
        rcfile:    str,
        cwd:       Path,
    ) -> None:
        tcl = cwd / "extract.tcl"
        tcl.write_text(self._magic_tcl(gds, cell, out_spice))
        env = os.environ.copy()
        env["MAGTYPE"] = "mag"
        proc = subprocess.run(
            [self.magic_exe, "-dnull", "-noconsole", "-rcfile", rcfile,
             str(tcl)],
            env=env, cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=self.timeout,
        )
        # Stash logs so callers can inspect them on failure
        self._last_magic_stdout = proc.stdout
        self._last_magic_stderr = proc.stderr
        if proc.returncode != 0:
            import warnings
            warnings.warn(
                f"Magic extract exited {proc.returncode}: {proc.stderr[:400]}",
                stacklevel=3,
            )

    @staticmethod
    def _magic_tcl(gds: Path, cell: str, out: Path) -> str:
        # Note:
        # - `port makeall` promotes every cell label into a top-level port,
        #   which is what netgen needs to align port-net pairs across sides.
        # - `ext2spice lvs` enables the standard LVS preset (subcircuit off,
        #   hierarchical on, scale set, parasitics dropped) — do *not* set
        #   `ext2spice subcircuit on` afterwards or each device gets wrapped
        #   in its own per-size subckt and netgen can't match them.
        # Labels on the PDK's *pin*-purpose metal layer (e.g. (68,16) MET1PIN
        # in sky130) are auto-flagged as ports by magic on GDS read — that's
        # what makes them appear on the .subckt header line.  We don't need
        # any explicit `port make` here as long as the synthesizer wrote
        # labels on the pin-purpose layer.
        return f"""\
crashbackups stop
gds read {gds}
load {cell}
select top cell
expand
extract do local
extract no capacitance
extract no coupling
extract no resistance
extract no adjust
extract no length
extract all
ext2spice lvs
ext2spice cthresh infinite
ext2spice rthresh infinite
ext2spice format ngspice
ext2spice -o {out} {cell}
quit -noprompt
"""

    # ── GDS flattening (mirrors DRC magic backend) ────────────────────────

    @staticmethod
    def _flatten_gds(src: Path, dst: Path, cell_name: str) -> str:
        """Flatten the GDS into a single cell named *cell_name*.

        Picks the top cell that matches *cell_name* (exact, suffix match
        like ``synth_<cell_name>``, or substring), flattens its hierarchy,
        renames it to *cell_name* (sanitised), and writes a single-cell GDS.
        """
        import gdstk
        lib = gdstk.read_gds(str(src))

        tops = [c for c in lib.top_level() if "$$$" not in c.name]
        if not tops:
            tops = [c for c in lib.cells if "$$$" not in c.name]

        # Match priority: exact > endswith > contains > first
        target = None
        for c in tops:
            if c.name == cell_name:
                target = c
                break
        if target is None:
            for c in tops:
                if c.name.endswith(cell_name) or c.name.endswith(f"_{cell_name}"):
                    target = c
                    break
        if target is None:
            for c in tops:
                if cell_name in c.name:
                    target = c
                    break
        if target is None:
            target = tops[0] if tops else lib.cells[0]

        target.flatten()
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", cell_name)
        target.name = safe
        out = gdstk.Library()
        out.add(target)
        out.write_gds(str(dst))
        return safe

    # ── Netgen compare ────────────────────────────────────────────────────

    def _run_netgen(
        self,
        ext_spice:    Path,
        ref_spice:    Path,
        ext_cell:     str,
        ref_cell:     str,
        setup:        str | None,
        log_file:     Path,
        cwd:          Path,
    ) -> LVSResult:
        # The netgen LVS command wants:
        #   lvs "<file1> <cell1>" "<file2> <cell2>" <setup> <log>
        cmd = [
            self.netgen_exe, "-batch", "lvs",
            f"{ext_spice} {ext_cell}",
            f"{ref_spice} {ref_cell}",
        ]
        if setup:
            cmd.append(setup)
        cmd.append(str(log_file))

        proc = subprocess.run(
            cmd, cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=self.timeout,
        )

        log_text = log_file.read_text() if log_file.exists() else proc.stdout
        return _parse_netgen_log(log_text)


# ── Netgen log parser ────────────────────────────────────────────────────────

_OK_PATTERNS = (
    "Circuits match uniquely",
    "Circuits match exactly",
)


def _parse_netgen_log(text: str) -> LVSResult:
    """Parse a netgen LVS report into an :class:`LVSResult`.

    Netgen output isn't a strict format, but a small set of marker phrases
    determines the verdict and we extract counts where possible.
    """
    clean = any(p in text for p in _OK_PATTERNS)
    mismatches: list[LVSMismatch] = []

    if not clean:
        for pattern, kind in (
            (r"Net mismatch.*",            "net"),
            (r"Property error.*",          "property"),
            (r"Cell.*are different\.",     "device"),
            (r"Subcircuits do not match", "device"),
            (r"\*\*Mismatch\*\*.*",        "unknown"),
        ):
            for m in re.finditer(pattern, text):
                mismatches.append(LVSMismatch(
                    kind=kind, description=m.group(0).strip(),
                ))

        if not mismatches and "Failed" in text:
            mismatches.append(LVSMismatch(
                kind="unknown",
                description=(
                    re.search(r"Failed.*", text).group(0).strip()
                    if re.search(r"Failed.*", text) else
                    "netgen reported failure (see log)"
                ),
            ))

    # Best-effort counts
    def _count(pat: str) -> int:
        m = re.search(pat, text)
        return int(m.group(1)) if m else 0

    return LVSResult(
        clean=clean,
        mismatches=mismatches,
        layout_devices=_count(r"Circuit 1.*?(\d+) device"),
        schema_devices=_count(r"Circuit 2.*?(\d+) device"),
        layout_nets=_count(r"Circuit 1.*?(\d+) net"),
        schema_nets=_count(r"Circuit 2.*?(\d+) net"),
        log=text,
    )
