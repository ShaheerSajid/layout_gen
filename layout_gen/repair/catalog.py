"""
layout_gen.repair.catalog — empirical DRC rule taxonomy.

This module builds and consumes a *catalog* of DRC rules: every distinct
rule observed across the PDKs and cells we test, mapped onto a small set
of universal categories.  The catalog is data (YAML on disk), not code —
adding a new PDK or new cells re-runs the generator and produces a richer
catalog without touching the engine.

The categories are PDK-independent.  CMOS DRCs collapse into:

  ┌──────────────────────────┬─────────────────────────────────────────┐
  │  category                │  fix primitive (universal)              │
  ├──────────────────────────┼─────────────────────────────────────────┤
  │  width                   │  stretch one edge by deficit            │
  │  spacing_same            │  push one same-layer shape away         │
  │  spacing_cross           │  push one shape away from another layer │
  │  enclosure               │  grow outer shape OR shrink inner       │
  │  extension               │  extend longer-direction edge           │
  │  area                    │  grow shape until area >= required      │
  │  merge                   │  union shapes when too close            │
  │  density                 │  add fill / remove polygon (global)     │
  │  overlap                 │  exclude one of two overlapping layers  │
  │  antenna                 │  add tie-down / break route (global)    │
  │  unknown                 │  needs human classification             │
  └──────────────────────────┴─────────────────────────────────────────┘

The catalog records, per rule:
  - PDK that emitted the rule
  - category (from above)
  - primary / secondary layer (when parseable)
  - required value (parsed from the DRC tool's description text)
  - typical measured deficit
  - sample violation locations
  - cells/primitives where it has been observed
  - fix-zone radius (worst rule distance in the active PDK; bounds side-effects)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ── Categories ────────────────────────────────────────────────────────────────

CATEGORIES = (
    "width", "spacing_same", "spacing_cross", "enclosure", "extension",
    "area", "merge", "density", "overlap", "antenna", "unknown",
)


# ── Catalog entry ────────────────────────────────────────────────────────────

@dataclass
class RuleEntry:
    """One rule in the catalog.

    Attributes
    ----------
    rule :
        DRC tool's rule identifier (e.g. ``"licon.13"``, ``"npc.2"``).
    pdk :
        PDK name that emitted the rule.
    category :
        One of :data:`CATEGORIES`.
    layers :
        Layer names involved (best-effort parse from rule name + description).
        First entry is primary, second (if any) is secondary for cross-layer.
    required_um :
        Required value in µm (parsed from description text).  ``None`` when
        the description lacks a numeric value (e.g. overlap rules).
    description :
        Verbatim description text from the DRC tool.
    typical_deficit_um :
        Median (required − measured) across observed violations.
    n_observations :
        How many violations of this rule have been seen across all runs.
    seen_in :
        List of cell names where this rule has fired.
    sample :
        One representative violation: ``{cell, x, y, measured, deficit}``.
    """
    rule:               str
    pdk:                str
    category:           str = "unknown"
    layers:             list[str] = field(default_factory=list)
    required_um:        float | None = None
    description:        str = ""
    typical_deficit_um: float | None = None
    n_observations:     int  = 0
    seen_in:            list[str] = field(default_factory=list)
    sample:             dict[str, Any] = field(default_factory=dict)


# ── Required-value parser ────────────────────────────────────────────────────

_REQ_RE = re.compile(
    r"(?:>=|<=|<|>|:\s*)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:um|µm|micron|nm)?",
    re.IGNORECASE,
)


def parse_required_um(description: str) -> float | None:
    """Best-effort parse of the required value from a DRC description.

    Many DRC decks emit text like ``"min. licon spacing : 0.17um"`` or
    ``"Diff width < 0.15 um"``.  We pull the first numeric value with a
    µm/nm/unit-less suffix.

    Returns ``None`` if no numeric value is present (e.g. ``"Implant
    overlap"``).
    """
    if not description:
        return None
    m = _REQ_RE.search(description)
    if not m:
        return None
    val = float(m.group(1))
    # Convert nm if the unit is explicit
    if "nm" in (description[m.start():m.end()].lower()):
        val /= 1000.0
    return val


# ── Rule → category classifier ───────────────────────────────────────────────

# Order matters: more specific patterns first.
_CATEGORY_RULES: list[tuple[str, str]] = [
    # rule_name patterns + description heuristics
    (r"_OFFGRID|off.?grid",                               "unknown"),
    # licon.13 / licon.13_a / licon.17 — sky130 cross-layer spacing
    (r"licon\.1[37](?:_[a-z])?",                          "spacing_cross"),
    (r"\.area\b|min[._\s]*area|min[._\s]*hole",           "area"),
    (r"\.enc(losure)?\b|enclosur|must enclose",           "enclosure"),
    (r"\.endcap\b|endcap|extension|extend(s|ed)?\s+(past|over|by)", "extension"),
    (r"\.merge\b|merged?\s+if|spacing.*merge|manually merged",      "merge"),
    (r"density|\bden\b",                                  "density"),
    (r"antenna|\bant\b",                                  "antenna"),
    (r"overlap|exclus|coincid",                           "overlap"),
    (r"\.width\b|\bwidth\b|min[._\s]*width|min[._\s]*size|\bsize\b", "width"),
    # Explicit "spacing" markers in DRC rule names (e.g. MR_*.SP.*)
    (r"\.SP\.|\.SP\b",                                    "spacing_same"),
    # spacing — same-layer vs cross-layer is decided by layer-count parsed below
    (r"spac|sep(aration)?|distance|\.s\d|\.\d+s\b",       "spacing_same"),
    # sky130 enclosure rule numbers (.4 / .5 of typical decks)
    (r"\.4(?:_[a-z])?$|\.5(?:_[a-z])?$",                  "enclosure"),
    # sky130 spacing rule numbers (.2 / .3)
    (r"\.2(?:_[a-z])?$|\.3(?:_[a-z])?$",                  "spacing_same"),
    # Trailing .1 / .1a / .Na / .0N typically marks a width/size rule
    (r"\.1(?:_[a-z])?$|\.\d+a$|\.0\d",                    "width"),
]


def classify_rule(rule: str, description: str, layers: list[str]) -> str:
    """Heuristically classify a rule into one of :data:`CATEGORIES`."""
    rule_l = rule.lower().strip()
    desc_l = description.lower().strip()

    # Merge has a very specific text signature in descriptions
    if ("merged" in desc_l or "merge if" in desc_l
            or "should be manually merged" in desc_l):
        return "merge"

    # First match against the rule name alone (so $-anchored patterns
    # work).  Fall through to combined text on no match.
    combined = f"{rule_l} {desc_l}".strip()
    for source in (rule_l, combined):
        for pattern, category in _CATEGORY_RULES:
            if re.search(pattern, source, re.IGNORECASE):
                # Promote spacing_same → spacing_cross when ≥2 distinct layers
                if category == "spacing_same" and len(layers) >= 2:
                    return "spacing_cross"
                return category
    return "unknown"


# ── Layer extraction ─────────────────────────────────────────────────────────

# Logical layer names (PDK-agnostic) we look for in rule names + descriptions.
_LAYER_NAMES = (
    "nwell", "pwell", "diff", "tap", "poly", "licon1", "licon", "li1",
    "mcon", "met1", "met2", "met3", "met4", "met5", "met6", "met7",
    "via1", "via2", "via3", "via4", "via5", "via6",
    "nsdm", "psdm", "npc", "rpo", "hvi",
)


def extract_layers(rule: str, description: str) -> list[str]:
    """Pull layer names out of a rule identifier + description text."""
    text = f"{rule} {description}".lower()
    found: list[str] = []
    for name in _LAYER_NAMES:
        # word-boundary match, but allow leading dot for rule prefixes
        if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", text):
            if name not in found:
                found.append(name)
    return found


# ── Builder ──────────────────────────────────────────────────────────────────

class CatalogBuilder:
    """Accumulate rule observations into a :class:`RuleEntry` map.

    Use :meth:`record` once per ``DRCViolation`` from any DRC run, then
    :meth:`finalise` to compute typical deficits and emit YAML.
    """

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], RuleEntry] = {}
        # all observed deficits per (pdk, rule), used for median deficit
        self._deficits: dict[tuple[str, str], list[float]] = {}

    def record(
        self,
        violation: Any,           # DRCViolation
        pdk:       str,
        cell:      str,
    ) -> None:
        rule = (violation.rule or "").strip()
        if not rule:
            return
        key = (pdk, rule)

        if key not in self.entries:
            layers = extract_layers(rule, violation.description or "")
            req    = parse_required_um(violation.description or "")
            cat    = classify_rule(rule, violation.description or "", layers)
            self.entries[key] = RuleEntry(
                rule=rule, pdk=pdk, category=cat,
                layers=layers,
                required_um=req,
                description=violation.description or "",
            )

        e = self.entries[key]
        e.n_observations += 1
        if cell not in e.seen_in:
            e.seen_in.append(cell)
        # First-seen sample is good enough for inspection
        if not e.sample:
            e.sample = {
                "cell":     cell,
                "x":        round(violation.x, 4),
                "y":        round(violation.y, 4),
                "measured": violation.value,
            }
        # Track deficit when both required and measured are known
        meas = violation.value
        req  = e.required_um
        if isinstance(meas, (int, float)) and meas >= 0 and req is not None:
            self._deficits.setdefault(key, []).append(req - float(meas))

    def finalise(self) -> dict[tuple[str, str], RuleEntry]:
        for key, entry in self.entries.items():
            ds = self._deficits.get(key, [])
            if ds:
                ds = sorted(ds)
                entry.typical_deficit_um = round(ds[len(ds) // 2], 5)
        return self.entries

    def to_yaml(self) -> str:
        self.finalise()
        # Group by PDK for readability
        by_pdk: dict[str, list[dict]] = {}
        for (pdk, rule), entry in sorted(self.entries.items()):
            by_pdk.setdefault(pdk, []).append(asdict(entry))
        return yaml.safe_dump({"pdks": by_pdk}, sort_keys=False)


# ── Convenience: catalog from synthesised cells + reference GDS files ────────

def build_catalog_for(
    drc_runner:     Any,
    pdk_name:       str,
    seed_layouts:   list[tuple[str, Path]],
    builder:        CatalogBuilder | None = None,
) -> CatalogBuilder:
    """Run *drc_runner* on each ``(cell_name, gds_path)`` pair and accumulate
    observations into *builder* (creating one if not provided).
    """
    builder = builder or CatalogBuilder()
    for cell_name, gds_path in seed_layouts:
        viols = drc_runner.run(Path(gds_path), cell_name)
        for v in viols:
            builder.record(v, pdk=pdk_name, cell=cell_name)
    return builder


__all__ = [
    "CATEGORIES",
    "RuleEntry",
    "CatalogBuilder",
    "parse_required_um",
    "classify_rule",
    "extract_layers",
    "build_catalog_for",
]
