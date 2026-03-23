"""
layout_gen.drc.klayout_runner — KLayout batch-mode DRC backend.

How it works
------------
1. :func:`_generate_drc_script` builds a KLayout Ruby DRC script from
   the PDKRules object (no hardcoded numbers — everything comes from the
   PDK YAML).
2. :class:`KLayoutDRCRunner` invokes ``klayout -b -r <script>`` via
   subprocess and writes a ``.lyrdb`` report file.
3. :func:`_parse_lyrdb` reads the KLayout rule-database XML and converts
   each violation to a :class:`~layout_gen.drc.base.DRCViolation`.

Adding another tool (Calibre, ICV, Magic)
------------------------------------------
Implement a parallel module with the same two public pieces:
  - A subclass of :class:`~layout_gen.drc.base.DRCRunner`
  - A ``_generate_*_script`` / ``_parse_*_report`` pair
Then register it in ``layout_gen/drc/registry.py``.
"""
from __future__ import annotations

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from layout_gen.pdk import PDKRules
from layout_gen.drc.base import DRCRunner, DRCViolation


class KLayoutDRCRunner(DRCRunner):
    """DRC via KLayout batch mode.

    Generates a KLayout Ruby DRC script from :class:`~layout_gen.pdk.PDKRules`
    and invokes::

        klayout -b -r <script.drc>
                -rd input=<cell.gds>
                -rd topcell=<CELL>
                -rd report=<violations.lyrdb>

    Parameters
    ----------
    rules :
        PDK rules.
    klayout_exe :
        Path / name of the KLayout executable (default ``"klayout"``).
    """

    def __init__(self, rules: PDKRules, klayout_exe: str = "klayout"):
        self.rules = rules
        self.klayout_exe = klayout_exe

    @property
    def tool_name(self) -> str:
        return "klayout"

    def is_available(self) -> bool:
        try:
            subprocess.run(
                [self.klayout_exe, "-v"],
                capture_output=True,
                timeout=10,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _resolve_deck(self) -> Path | None:
        """Return the path to a real PDK DRC deck, or None to use auto-generated."""
        drc_cfg = getattr(self.rules, "drc", None) or {}
        deck_path = drc_cfg.get("klayout")
        if deck_path and Path(deck_path).is_file():
            return Path(deck_path)
        return None

    def run(
        self,
        gds_path: Path,
        cell_name: str | None = None,
    ) -> List[DRCViolation]:
        gds_path = Path(gds_path).resolve()
        deck     = self._resolve_deck()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir      = Path(tmpdir)
            report_path = tmpdir / "violations.lyrdb"

            if deck is not None:
                # Use the real PDK DRC deck (e.g. sky130A_mr.drc)
                script_path = deck
                cmd = [
                    self.klayout_exe, "-b",
                    "-r", str(script_path),
                    "-rd", f"input={gds_path}",
                    "-rd", f"report={report_path}",
                    # Enable all relevant check groups
                    "-rd", "feol=true",
                    "-rd", "beol=true",
                    "-rd", "offgrid=true",
                    "-rd", "seal=false",
                    "-rd", "floating_met=false",
                ]
                if cell_name:
                    cmd += ["-rd", f"top_cell={cell_name}"]
            else:
                # Fall back to auto-generated script from PDK YAML rules
                script  = _generate_drc_script(self.rules)
                script_path = tmpdir / "drc.drc"
                script_path.write_text(script, encoding="utf-8")
                cmd = [
                    self.klayout_exe, "-b",
                    "-r", str(script_path),
                    "-rd", f"input={gds_path}",
                    "-rd", f"report={report_path}",
                ]
                if cell_name:
                    cmd += ["-rd", f"topcell={cell_name}"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"KLayout DRC failed (exit {result.returncode}):\n"
                    f"{result.stderr}"
                )

            if not report_path.exists():
                return []
            return _parse_lyrdb(report_path)


# ── DRC script generation ──────────────────────────────────────────────────────

def _generate_drc_script(rules: PDKRules) -> str:
    """Return a KLayout Ruby DRC script derived entirely from *rules*.

    Every numeric constant comes from the PDK YAML — no tech-specific
    hardcoding here.  To support a new tech node, update the YAML only.
    """
    R   = rules
    lyr = rules.layers   # {name: {layer: int, datatype: int}}

    def inp(name: str) -> str:
        """Return 'input(layer, dt)' string for a logical layer."""
        e = lyr[name]
        return f"input({e['layer']}, {e['datatype']})"

    # ── Header ────────────────────────────────────────────────────────────────
    lines = [
        f"# Auto-generated KLayout DRC — PDK: {R.name}",
        "# Generated by layout_gen.drc.klayout_runner",
        "",
        "source($input, $topcell)",
        "report('DRC', $report)",
        "",
        "# ── Layer inputs ────────────────────────────────────────────────────",
        f"poly   = {inp('poly')}",
        f"diff   = {inp('diff')}",
        f"licon1 = {inp('licon1')}",
        f"li1    = {inp('li1')}",
        f"met1   = {inp('met1')}",
        f"nwell  = {inp('nwell')}",
        f"nsdm   = {inp('nsdm')}",
        f"psdm   = {inp('psdm')}",
    ]
    if "mcon" in lyr:
        lines.append(f"mcon   = {inp('mcon')}")

    # ── Poly ──────────────────────────────────────────────────────────────────
    p = R.poly
    pw, ps = p['width_min_um'], p['spacing_min_um']
    lines += [
        "",
        "# ── Poly ────────────────────────────────────────────────────────────",
        f"poly.width({pw}.um).output('poly.1', 'Poly width < {pw} um')",
        f"poly.space({ps}.um).output('poly.2', 'Poly spacing < {ps} um')",
    ]

    # ── Diff ──────────────────────────────────────────────────────────────────
    d = R.diff
    dw, ds = d['width_min_um'], d['spacing_min_um']
    lines += [
        "",
        "# ── Diff ────────────────────────────────────────────────────────────",
        f"diff.width({dw}.um).output('diff.1', 'Diff width < {dw} um')",
        f"diff.space({ds}.um).output('diff.2', 'Diff spacing < {ds} um')",
    ]

    # ── Licon1 (contacts) ─────────────────────────────────────────────────────
    c = R.contacts
    csz, csp = c['size_um'], c['spacing_um']
    cenc_d, cenc_li = c['enclosure_in_diff_um'], c['enclosure_in_li1_um']
    lines += [
        "",
        "# ── Licon1 ──────────────────────────────────────────────────────────",
        f"licon1.width({csz}.um).output('licon.1', 'Licon1 size < {csz} um')",
        f"licon1.space({csp}.um).output('licon.2', 'Licon1 spacing < {csp} um')",
        f"diff.enclosing(licon1.and(diff), {cenc_d}.um)"
        f".output('licon.5a', 'Diff must enclose licon1 by {cenc_d} um')",
        f"li1.enclosing(licon1, {cenc_li}.um).output('licon.3', 'Li1 must enclose licon1')",
    ]
    if "space_to_poly_um" in c:
        stp = c['space_to_poly_um']
        lines.append(
            f"licon1.and(diff).separation(poly, {stp}.um)"
            f".output('licon.7', 'Diff contact to poly spacing < {stp} um')"
        )
    if "poly_enclosure_um" in c:
        penc = c['poly_enclosure_um']
        lines.append(
            f"poly.enclosing(licon1.and(poly), {penc}.um)"
            f".output('licon.10', 'Poly must enclose polycontact by {penc} um')"
        )

    # ── Li1 ───────────────────────────────────────────────────────────────────
    li = R.li1
    liw, lis = li['width_min_um'], li['spacing_min_um']
    lines += [
        "",
        "# ── Li1 ─────────────────────────────────────────────────────────────",
        f"li1.width({liw}.um).output('li1.1', 'Li1 width < {liw} um')",
        f"li1.space({lis}.um).output('li1.2', 'Li1 spacing < {lis} um')",
    ]

    # ── Met1 ──────────────────────────────────────────────────────────────────
    m1 = R.met1
    if m1:
        m1w, m1s = m1['width_min_um'], m1['spacing_min_um']
        lines += [
            "",
            "# ── Met1 ────────────────────────────────────────────────────────",
            f"met1.width({m1w}.um).output('met1.1', 'Met1 width < {m1w} um')",
            f"met1.space({m1s}.um).output('met1.2', 'Met1 spacing < {m1s} um')",
        ]

    # ── Mcon ──────────────────────────────────────────────────────────────────
    mc = R.mcon
    if mc and "mcon" in lyr:
        mcsz, mcsp = mc['size_um'], mc['spacing_um']
        mcli, mcm1 = mc['enclosure_in_li1_um'], mc['enclosure_in_met1_um']
        lines += [
            "",
            "# ── Mcon ────────────────────────────────────────────────────────",
            f"mcon.width({mcsz}.um).output('mcon.1', 'Mcon size < {mcsz} um')",
            f"mcon.space({mcsp}.um).output('mcon.2', 'Mcon spacing < {mcsp} um')",
            f"li1.enclosing(mcon, {mcli}.um).output('mcon.3', 'Li1 must enclose mcon')",
            f"met1.enclosing(mcon, {mcm1}.um).output('mcon.5', 'Met1 must enclose mcon by {mcm1} um')",
        ]

    # ── Nwell ─────────────────────────────────────────────────────────────────
    nw = R.nwell
    nww, nws = nw['width_min_um'], nw['spacing_min_um']
    lines += [
        "",
        "# ── Nwell ───────────────────────────────────────────────────────────",
        f"nwell.width({nww}.um).output('nwell.1', 'Nwell width < {nww} um')",
        f"nwell.space({nws}.um).output('nwell.2', 'Nwell spacing < {nws} um')",
    ]
    if "enclosure_of_pdiff_um" in nw:
        nwenc = nw['enclosure_of_pdiff_um']
        lines.append(
            f"nwell.enclosing(diff.and(psdm), {nwenc}.um)"
            f".output('nwell.5', 'Nwell must enclose PMOS diff by {nwenc} um')"
        )

    # ── Met2 ──────────────────────────────────────────────────────────────────
    m2 = R.met2 if R.met2 else None
    if m2 and "met2" in lyr:
        m2w, m2s = m2['width_min_um'], m2['spacing_min_um']
        lines += [
            "",
            f"met2   = {inp('met2')}",
            "# ── Met2 ────────────────────────────────────────────────────────",
            f"met2.width({m2w}.um).output('met2.1', 'Met2 width < {m2w} um')",
            f"met2.space({m2s}.um).output('met2.2', 'Met2 spacing < {m2s} um')",
        ]

    # ── Via1 (met1 → met2) ────────────────────────────────────────────────────
    v1 = R.via1 if R.via1 else None
    if v1 and "via1" in lyr:
        v1sz = v1['size_um']
        v1sp = v1['spacing_um']
        v1em1 = v1.get('enclosure_in_met1_um', 0.055)
        v1em2 = v1.get('enclosure_in_met2_um', 0.055)
        lines += [
            "",
            f"via1   = {inp('via1')}",
            "# ── Via1 ────────────────────────────────────────────────────────",
            f"via1.width({v1sz}.um).output('via.1a', 'Via1 size < {v1sz} um')",
            f"via1.space({v1sp}.um).output('via.2', 'Via1 spacing < {v1sp} um')",
            f"met1.enclosing(via1, {v1em1}.um)"
            f".output('via.4a', 'Met1 must enclose via1 by {v1em1} um')",
        ]
        if m2 and "met2" in lyr:
            lines.append(
                f"met2.enclosing(via1, {v1em2}.um)"
                f".output('via.5a', 'Met2 must enclose via1 by {v1em2} um')"
            )

    # ── Poly endcap ───────────────────────────────────────────────────────────
    endcap = R.poly.get("endcap_over_diff_um")
    if endcap:
        lines += [
            "",
            "# ── Poly endcap ─────────────────────────────────────────────────",
            f"# poly.4: poly must extend {endcap} um past diff edge",
            f"gate_poly = poly.and(diff).extents",
            f"# (Informational — edge-based endcap checks require custom logic)",
        ]

    # ── Implant ───────────────────────────────────────────────────────────────
    impl = R.implant
    if "enclosure_of_diff_um" in impl:
        ienc = impl['enclosure_of_diff_um']
        lines += [
            "",
            "# ── Implant ─────────────────────────────────────────────────────",
            f"nsdm.enclosing(diff.and(nsdm), {ienc}.um)"
            f".output('nsdm.3', 'Nsdm must enclose NMOS diff by {ienc} um')",
            f"psdm.enclosing(diff.and(psdm), {ienc}.um)"
            f".output('psdm.3', 'Psdm must enclose PMOS diff by {ienc} um')",
        ]

    # ── Antenna ───────────────────────────────────────────────────────────────
    lines += [
        "",
        "# ── Cross-layer overlap checks ─────────────────────────────────",
        "# nsdm and psdm must not overlap (implant exclusivity)",
        "nsdm.and(psdm).output('implant.overlap', 'Nsdm/Psdm overlap')",
    ]

    return "\n".join(lines) + "\n"


# ── RDB report parser ──────────────────────────────────────────────────────────

_DBU = 0.001   # KLayout default: 1 dbu = 1 nm = 0.001 µm


def _parse_lyrdb(path: Path) -> List[DRCViolation]:
    """Parse a KLayout ``.lyrdb`` XML report → :class:`DRCViolation` list."""
    tree = ET.parse(path)
    root = tree.getroot()

    # Build category → description map
    cat_desc: dict[str, str] = {}
    for cat in root.findall(".//category"):
        name = cat.findtext("name", "")
        desc = cat.findtext("description", "")
        cat_desc[name] = desc

    violations: List[DRCViolation] = []
    for item in root.findall(".//item"):
        rule = item.findtext("category", "")
        desc = cat_desc.get(rule, "")
        x, y = _centroid_from_item(item)

        value: float | None = None
        val_el = item.find("values/value")
        if val_el is not None and val_el.text:
            try:
                value = float(val_el.text)
            except ValueError:
                pass

        violations.append(DRCViolation(
            rule=rule,
            description=desc,
            x=x,
            y=y,
            value=value,
        ))

    return violations


def _centroid_from_item(item: ET.Element) -> tuple[float, float]:
    """Return (x_um, y_um) centroid for an RDB item's geometry."""
    # Polygon:  (x1,y1;x2,y2;...)
    poly_el = item.find("polygon")
    if poly_el is not None and poly_el.text:
        pts = _parse_pts(poly_el.text)
        if pts:
            return _centroid(pts)

    # Edge-pair: (x1,y1;x2,y2)/(x3,y3;x4,y4)
    ep_el = item.find("edge-pair")
    if ep_el is not None and ep_el.text:
        pts = []
        for seg in ep_el.text.split("/"):
            pts.extend(_parse_pts(seg))
        if pts:
            return _centroid(pts)

    return 0.0, 0.0


def _parse_pts(s: str) -> list[tuple[float, float]]:
    """Parse '(x1,y1;x2,y2;...)' → list of (x_um, y_um).

    KLayout may output coordinates as integer dbu (multiply by _DBU)
    or as floating-point µm values.  We try int first, then float.
    """
    s = s.strip().strip("()")
    pts = []
    for token in s.split(";"):
        token = token.strip()
        if "," in token:
            xs, ys = token.split(",", 1)
            try:
                pts.append((int(xs) * _DBU, int(ys) * _DBU))
            except ValueError:
                try:
                    pts.append((float(xs), float(ys)))
                except ValueError:
                    pass
    return pts


def _centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(pts)
    return sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
