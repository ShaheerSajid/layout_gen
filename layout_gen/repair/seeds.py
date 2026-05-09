"""
layout_gen.repair.seeds — discover and ingest DRC-clean seed layouts.

Three sources of seed layouts feed the catalog generator and (later) the
perturbation library:

1. **Reference memory primitives** — hand-curated DRC-clean GDS files that
   cover the cells our synthesizer should target (bitcell, sense amp,
   write driver, …).
2. **Open standard cell libraries** shipped with the PDK
   (``${PDK_ROOT}/<pdk>/libs.ref/<scl>/gds/<scl>.gds``).  These contain
   hundreds of variants (sizes / drive strengths) of the universal CMOS
   primitives — invaluable for cataloguing every rule the PDK exercises
   and as seeds for inverse-perturbation training.
3. **Synthesizer output** — our own cells, kept here for completeness;
   the synthesizer pipeline records its own emissions.

The ingester does *not* extract cells into per-cell GDS files unless asked
— DRC tools accept a top-cell name argument so the SCL container GDS can
be passed directly with each cell name.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gdstk


# ── Seed entry ────────────────────────────────────────────────────────────────

@dataclass
class SeedCell:
    """One seed layout the catalog generator will run DRC on."""
    name:       str          # cell name inside the GDS
    gds_path:   Path         # GDS file containing the cell
    pdk:        str          # PDK whose rules to apply
    primitive:  str          # logical primitive (inv, nand2, dfxtp, bitcell, …)
    source:     str          # "reference" | "scl" | "synthesizer"
    library:    str = ""     # SCL name (sky130_fd_sc_hd, …) or "" for refs


# ── Primitive regex ──────────────────────────────────────────────────────────

# Maps an SCL cell name → the canonical primitive bucket the repair engine
# cares about.  Stripped of size suffixes (_1, _2, _4 …) and library prefix.
#
# Memory-domain relevance: every entry below is something that appears in
# either bitcell-style structures, decoders, latches, or compound logic
# we would synthesize.  Buffers, clock gates, and IO are excluded.
_PRIMITIVE_PATTERNS: list[tuple[str, str]] = [
    # exact-base patterns first
    (r"^inv$",                    "inv"),
    (r"^buf$",                    "buf"),
    (r"^nand[2-4]$",              "nand"),
    (r"^nor[2-4]$",               "nor"),
    (r"^and[2-4]$",               "and"),
    (r"^or[2-4]$",                "or"),
    (r"^xor[2-3]$",               "xor"),
    (r"^xnor[2-3]$",              "xnor"),
    # AOI / OAI families (a21o, a21oi, o21a, o21ai, a22o, …)
    (r"^a\d+o[i]?$",              "aoi"),
    (r"^o\d+a[i]?$",              "oai"),
    (r"^a\d+\d+o[i]?$",           "aoi"),
    (r"^o\d+\d+a[i]?$",           "oai"),
    # Multiplexers
    (r"^mux[2-4]$",               "mux"),
    # D flip-flops & latches (sky130 + gf180 conventions)
    (r"^dfxtp$",                  "dff"),
    (r"^dfrtp$",                  "dff_r"),    # async reset
    (r"^dfstp$",                  "dff_s"),    # async set
    (r"^dlxtp$",                  "dlatch"),
    (r"^sdfrtp$",                 "sdff_r"),
    (r"^dff[a-z]*$",              "dff"),
    (r"^dlat[a-z]*$",             "dlatch"),
]


def classify_primitive(cell_name: str) -> str | None:
    """Map a cell name (with or without library prefix and size suffix) to
    a canonical primitive bucket, or ``None`` if it is not one we target.
    """
    base = cell_name
    # Strip library prefix like 'sky130_fd_sc_hd__' or 'gf180mcu_fd_sc_…__'
    if "__" in base:
        base = base.split("__", 1)[1]
    # Strip trailing _<size> (digits only)
    m = re.match(r"^([a-zA-Z]+\d*[a-zA-Z]*)(?:_\d+)?$", base)
    if m:
        base = m.group(1)
    base = base.lower()

    for pat, bucket in _PRIMITIVE_PATTERNS:
        if re.match(pat, base):
            return bucket
    return None


# ── Reference cells ──────────────────────────────────────────────────────────

# Default location of the FabRAM reference cells the user supplied.
_DEFAULT_REF_DIR = Path(
    os.environ.get(
        "FABRAM_REFCELLS_DIR",
        str(Path.home() / "Downloads" / "FabRAM_v1-main" / "FE" / "gds"),
    )
)

# Reference filename → primitive bucket
_REFERENCE_PRIMITIVES: dict[str, str] = {
    "not.gds":              "inv",
    "nand2.gds":            "nand",
    "nand3.gds":            "nand",
    "nand4.gds":            "nand",
    "bit_cell.gds":         "bitcell",
    "ms_ff.gds":            "dff",
    "row_driver_f.gds":     "row_driver",
    "sense_amplifier.gds":  "sense_amp",
    "write_driver.gds":     "write_driver",
    "control.gds":          "control",
    "dec_2to4.gds":         "decoder",
    "dec_3to6.gds":         "decoder",
    "col_dec4.gds":         "col_mux",
    "col_dec8.gds":         "col_mux",
    "dido.gds":             "dido",
    "dummy_cell.gds":       "dummy",
}


def reference_seeds(
    ref_dir:  Path | None = None,
    pdk:      str         = "sky130A",
) -> list[SeedCell]:
    """Discover reference memory primitives in *ref_dir*."""
    ref_dir = Path(ref_dir or _DEFAULT_REF_DIR)
    seeds: list[SeedCell] = []
    if not ref_dir.is_dir():
        return seeds
    for fname, primitive in _REFERENCE_PRIMITIVES.items():
        path = ref_dir / fname
        if not path.is_file():
            continue
        # Cell name: read the top cell from the file
        try:
            lib = gdstk.read_gds(str(path))
            tops = [c for c in lib.top_level()]
            if not tops:
                continue
            cell_name = tops[0].name
        except Exception:
            cell_name = path.stem
        seeds.append(SeedCell(
            name=cell_name, gds_path=path, pdk=pdk,
            primitive=primitive, source="reference",
        ))
    return seeds


# ── Standard cell library ingestion ──────────────────────────────────────────

# Default SCLs to walk when ``scl_seeds`` is called without ``libraries``.
DEFAULT_SCLS: dict[str, list[str]] = {
    "sky130A": [
        "sky130_fd_sc_hd",
        # Other libraries follow the same pattern when needed:
        # "sky130_fd_sc_hs", "sky130_fd_sc_lp", "sky130_fd_sc_ms",
        # "sky130_fd_sc_hdll", "sky130_fd_sc_ls",
    ],
    "gf180mcuD": [
        "gf180mcu_fd_sc_mcu7t5v0",
        "gf180mcu_fd_sc_mcu9t5v0",
    ],
}


def scl_seeds(
    pdk:        str,
    libraries:  list[str] | None = None,
    pdk_root:   str | os.PathLike | None = None,
    max_per_primitive: int | None = None,
) -> list[SeedCell]:
    """Walk an SCL container GDS and emit one :class:`SeedCell` per cell that
    maps to a memory-relevant primitive.

    Parameters
    ----------
    pdk :
        PDK name (``"sky130A"``, ``"gf180mcuD"``, …) — used both as the
        PDK identifier on each SeedCell and to construct the libs.ref path.
    libraries :
        SCL names to scan.  Defaults to :data:`DEFAULT_SCLS` for the PDK.
    pdk_root :
        Override for ``$PDK_ROOT``.  Defaults to the env var or
        ``/usr/local/share/pdk``.
    max_per_primitive :
        If given, keep at most N cells per primitive bucket per library
        (useful when downsampling for fast iteration).
    """
    pdk_root = Path(pdk_root or os.environ.get("PDK_ROOT", "/usr/local/share/pdk"))
    libs = libraries or DEFAULT_SCLS.get(pdk, [])
    seeds: list[SeedCell] = []

    for lib_name in libs:
        gds_path = pdk_root / pdk / "libs.ref" / lib_name / "gds" / f"{lib_name}.gds"
        if not gds_path.is_file():
            continue
        try:
            lib = gdstk.read_gds(str(gds_path))
        except Exception:
            continue

        per_prim_count: dict[str, int] = {}
        for cell in lib.cells:
            cname = cell.name
            if not cname.startswith(f"{lib_name}__"):
                continue
            primitive = classify_primitive(cname)
            if primitive is None:
                continue
            if max_per_primitive is not None:
                per_prim_count.setdefault(primitive, 0)
                if per_prim_count[primitive] >= max_per_primitive:
                    continue
                per_prim_count[primitive] += 1
            seeds.append(SeedCell(
                name=cname, gds_path=gds_path, pdk=pdk,
                primitive=primitive, source="scl", library=lib_name,
            ))
    return seeds


# ── Combined entry point ─────────────────────────────────────────────────────

def all_seeds_for(
    pdk:        str,
    *,
    include_references: bool = True,
    include_scl:        bool = True,
    include_synth:      bool = True,
    max_per_primitive:  int | None = None,
) -> list[SeedCell]:
    """One-call helper: every memory-relevant seed layout we know about for
    *pdk*.

    Order: reference cells → SCL cells → synthesizer-emitted cells.
    The synthesizer cells are typically *not* clean — that's intentional;
    they surface the failure modes the catalog must cover.
    """
    seeds: list[SeedCell] = []
    if include_references and pdk == "sky130A":
        seeds.extend(reference_seeds(pdk=pdk))
    if include_scl:
        seeds.extend(scl_seeds(pdk, max_per_primitive=max_per_primitive))
    if include_synth:
        seeds.extend(synth_seeds(pdk))
    return seeds


# ── Synthesizer-emitted cells ────────────────────────────────────────────────

# Cell template name → primitive bucket.  Drives the synthesize-and-save loop.
_SYNTH_TEMPLATES: dict[str, str] = {
    "inverter":   "inv",
    "buffer":     "buf",
    "nand2":      "nand",
    "nand3":      "nand",
    "nor2":       "nor",
    "nor3":       "nor",
    "aoi21":      "aoi",
    "oai21":      "oai",
    "row_driver": "row_driver",
    "bit_cell_6t": "bitcell",
}

# Default sizing per template — broadly tuned to give reasonable nor/nand
# pull-up:pull-down ratios.  Override by passing your own dict to
# :func:`synth_seeds`.
_DEFAULT_SYNTH_PARAMS: dict[str, dict[str, float]] = {
    "inverter":    {"w_N": 0.52, "w_P": 0.42, "l": 0.15},
    "buffer":      {"w_N": 0.52, "w_P": 0.42, "l": 0.15},
    "nand2":       {"w_N": 0.52, "w_P": 0.42, "l": 0.15},
    "nand3":       {"w_N": 0.52, "w_P": 0.42, "l": 0.15},
    "nor2":        {"w_N": 0.52, "w_P": 0.84, "l": 0.15},
    "nor3":        {"w_N": 0.52, "w_P": 1.26, "l": 0.15},
    "aoi21":       {"w_N": 0.52, "w_P": 0.84, "l": 0.15},
    "oai21":       {"w_N": 0.52, "w_P": 0.84, "l": 0.15},
    "row_driver":  {"w_N": 0.52, "w_P": 0.42, "l": 0.15},
    "bit_cell_6t": {"w_N": 0.52, "w_P": 0.42, "l": 0.15},
}


def synth_seeds(
    pdk:         str,
    *,
    out_dir:     Path | None = None,
    templates:   dict[str, str] | None = None,
    params:      dict[str, dict[str, float]] | None = None,
) -> list[SeedCell]:
    """Synthesize a fixed set of templates and emit one :class:`SeedCell` per
    cell.  GDS files are written to *out_dir* (defaults to a per-PDK cache
    directory) and reused on subsequent calls.

    These cells are *expected* to have DRC violations — that is the point.
    Cataloguing what fires here is what tells the engine which rules it
    must learn to repair.
    """
    from layout_gen import load_pdk
    from layout_gen.synth.loader import load_template
    from layout_gen.synth.synthesizer import Synthesizer

    templates = templates or _SYNTH_TEMPLATES
    params    = params    or _DEFAULT_SYNTH_PARAMS

    if out_dir is None:
        out_dir = Path(__file__).parent / "data" / "synth_cache" / pdk
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the bundled PDK YAML
    import layout_gen
    pdk_yaml = Path(layout_gen.__file__).parent / "pdks" / f"{pdk}.yaml"
    if not pdk_yaml.is_file():
        return []
    rules = load_pdk(pdk_yaml)
    synth = Synthesizer(rules)   # no DRC runner — synthesize once, no iteration

    seeds: list[SeedCell] = []
    for tmpl_name, primitive in templates.items():
        try:
            tmpl   = load_template(tmpl_name)
        except FileNotFoundError:
            continue
        try:
            cell_params = params.get(tmpl_name, {"w": 0.42, "l": 0.15})
            result = synth.synthesize(tmpl, cell_params)
        except Exception:
            continue

        # Stable canonical name: synthesize() may emit "synth_<X>$2" when
        # the global cell counter increments across calls — we don't want
        # that leaking into the cache filename or the SeedCell name.
        canonical_name = f"synth_{tmpl_name}"
        gds_path = out_dir / f"{tmpl_name}.gds"
        try:
            tmp_gds = gds_path.with_suffix(".tmp.gds")
            result.component.write_gds(str(tmp_gds), with_metadata=False)
            # Rename the top cell to canonical_name on disk so DRC tool
            # lookups are stable across runs.
            import gdstk
            lib = gdstk.read_gds(str(tmp_gds))
            tops = [c for c in lib.top_level()]
            if tops:
                tops[0].name = canonical_name
                out = gdstk.Library()
                out.add(tops[0])
                # Bring in any referenced sub-cells too (transistor primitives)
                for c in lib.cells:
                    if c.name != canonical_name and c.name not in (t.name for t in out.cells):
                        # Skip $$$CONTEXT_INFO$$$ markers
                        if "$$$" in c.name:
                            continue
                        out.add(c)
                out.write_gds(str(gds_path))
            else:
                tmp_gds.replace(gds_path)
            tmp_gds.unlink(missing_ok=True)
        except Exception:
            continue

        seeds.append(SeedCell(
            name=canonical_name,
            gds_path=gds_path, pdk=pdk,
            primitive=primitive, source="synthesizer",
        ))
    return seeds


__all__ = [
    "SeedCell",
    "classify_primitive",
    "reference_seeds",
    "scl_seeds",
    "synth_seeds",
    "all_seeds_for",
    "DEFAULT_SCLS",
]
