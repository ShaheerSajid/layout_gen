"""
layout_gen.synth.geo.violations — Technology-agnostic DRC violation parser.

Converts raw :class:`~layout_gen.drc.base.DRCViolation` objects (rule name +
description string) into structured :class:`ViolationInfo` records that the
geometric agent can reason about *without knowing the PDK*.

The parser extracts:

- **category** — spacing, width, enclosure, area, overlap, or unknown
- **layer(s)** — which metal/poly/diff layer is involved
- **measured** / **required** — the actual vs. minimum dimension (µm)
- **deficit** — how much geometry must change (required − measured)
- **location** — violation centroid (x, y) from the DRC tool

This is the bridge between PDK-specific DRC output and technology-agnostic
geometric fixing: a "met1 spacing < 0.14 µm" violation in sky130 and a
"M2 spacing < 0.065 µm" violation in TSMC 65 nm both produce a
``ViolationInfo(category="spacing", deficit=...)`` that the agent handles
identically — find the two closest edges, move them apart by ``deficit``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from layout_gen.drc.base import DRCViolation


# ── Structured violation ─────────────────────────────────────────────────────

@dataclass
class ViolationInfo:
    """Parsed, technology-agnostic DRC violation.

    Attributes
    ----------
    category :
        One of ``"spacing"``, ``"width"``, ``"enclosure"``, ``"area"``,
        ``"overlap"``, ``"unknown"``.
    layer :
        Primary layer name (e.g. ``"met1"``).  For enclosure violations,
        this is the *outer* layer.
    inner_layer :
        For enclosure violations, the *inner* layer (e.g. ``"via1"``).
        Empty string otherwise.
    measured :
        Actual measured dimension (µm) or 0 if not available.
    required :
        Minimum required dimension (µm) or 0 if not parseable.
    deficit :
        ``required - measured`` (µm) — how much geometry must change.
    x, y :
        Violation centroid (µm).
    rule :
        Original rule name from the DRC tool.
    raw :
        Original DRCViolation object.
    """
    category:    str
    layer:       str
    inner_layer: str   = ""
    measured:    float = 0.0
    required:    float = 0.0
    deficit:     float = 0.0
    x:           float = 0.0
    y:           float = 0.0
    rule:        str   = ""
    raw:         DRCViolation | None = None

    def __repr__(self) -> str:
        d = f" deficit={self.deficit:.3f}" if self.deficit else ""
        return (f"ViolationInfo({self.category}/{self.layer}{d} "
                f"@ ({self.x:.2f},{self.y:.2f}))")


# ── Layer name normalization ─────────────────────────────────────────────────

_LAYER_ALIASES: dict[str, str] = {
    "metal1": "met1", "metal2": "met2",
    "li": "li1", "local_interconnect": "li1",
    "ct": "licon1", "contact": "licon1", "licon": "licon1",
    "polycontact": "licon1", "polycon": "licon1",
    "via": "via1",
    "nsd": "nsdm", "psd": "psdm", "nplus": "nsdm", "pplus": "psdm",
    "active": "diff", "od": "diff", "diffusion": "diff",
    "gate": "poly", "pc": "poly",
    "nw": "nwell", "n_well": "nwell",
}
# Generate m1→met1 .. m10→met10 and v1→via1 .. v9→via9 dynamically
for _i in range(1, 11):
    _LAYER_ALIASES[f"m{_i}"] = f"met{_i}"
for _i in range(1, 10):
    _LAYER_ALIASES[f"v{_i}"] = f"via{_i}"


def _normalize_layer(s: str) -> str:
    """Map common layer aliases to canonical names."""
    s = s.strip().lower().replace("-", "").replace("_", "")
    # Direct canonical names — static set + dynamic met/via patterns
    if s in ("poly", "diff", "li1", "licon1", "mcon", "nwell", "nsdm", "psdm"):
        return s
    # met1..met10, via1..via9 are canonical
    if (s.startswith("met") or s.startswith("via")) and s[3:].isdigit():
        return s
    return _LAYER_ALIASES.get(s, s)


# ── Category detection ───────────────────────────────────────────────────────

# Patterns in rule names/descriptions → category.  Order matters — more
# specific categories match first.  ``merge`` precedes ``spacing`` because
# rules like "min spacing, should be manually merged if less than 0.27µm"
# are *spacing* rules whose preferred fix is union, not push.
_CATEGORY_PATTERNS: list[tuple[str, str]] = [
    (r"off.?grid|offgrid|grid",            "offgrid"),
    (r"merged?\s+if|manually\s+merged|spacing.*merge", "merge"),
    (r"spacing|space|spac\b|\.sp\b",       "spacing"),
    (r"width|wid\b|\.w\b|minimum width",   "width"),
    (r"enclos|ovlp|overlap.*encl",         "enclosure"),
    (r"area\b|min.*area",                  "area"),
    (r"overlap|short|bridg",               "overlap"),
    (r"extend|extension|endcap",           "extension"),
    (r"antenna",                           "antenna"),
    (r"density|fill",                      "density"),
]

def _detect_category(rule: str, desc: str) -> str:
    """Infer violation category from rule name and description."""
    text = f"{rule} {desc}".lower()
    for pat, cat in _CATEGORY_PATTERNS:
        if re.search(pat, text):
            return cat
    return "unknown"


# ── Value extraction ─────────────────────────────────────────────────────────

# Match patterns like "< 0.14 um", ": 0.38um", "= 0.17 µm", "by 0.05 um"
_VALUE_RE = re.compile(
    r'(?:[:<>=]|by)\s*(\d+\.?\d*)\s*(?:um|µm|micron)',
    re.IGNORECASE,
)

# Match "enclosure of X by Y um" or "enclose via1 by 0.085"
_ENCLOSURE_LAYER_RE = re.compile(
    r'enclos(?:ure|ing|e)?\s+(?:of\s+)?(\w+)',
    re.IGNORECASE,
)

# Match "Poly must enclose X" → outer layer = "Poly"
_ENCLOSURE_OUTER_RE = re.compile(
    r'(\w+)\s+must\s+enclos',
    re.IGNORECASE,
)

def _extract_required(desc: str) -> float:
    """Extract the required dimension from the violation description."""
    m = _VALUE_RE.search(desc)
    if m:
        return float(m.group(1))
    # Try bare numbers at end of description
    m = re.search(r'(\d+\.?\d*)\s*$', desc.strip())
    if m:
        return float(m.group(1))
    return 0.0


def _extract_layer_from_rule(rule: str) -> str:
    """Extract the primary layer from a rule name like 'met2.1' or 'li.5'."""
    rule = rule.strip().strip("'\"")
    # "met2.1" → "met2", "li.5" → "li", "via.5a" → "via"
    parts = rule.split(".", 1)
    if parts:
        return _normalize_layer(parts[0])
    return ""


def _extract_inner_layer(desc: str) -> str:
    """For enclosure violations, extract the inner layer name."""
    m = _ENCLOSURE_LAYER_RE.search(desc)
    if m:
        return _normalize_layer(m.group(1))
    return ""


# ── Public API ───────────────────────────────────────────────────────────────

def parse_violation(v: DRCViolation) -> ViolationInfo:
    """Parse a raw DRC violation into a structured :class:`ViolationInfo`."""
    rule = v.rule.strip().strip("'\"")
    desc = v.description or ""

    category = _detect_category(rule, desc)
    layer    = _extract_layer_from_rule(rule)
    inner    = _extract_inner_layer(desc) if category == "enclosure" else ""

    # For enclosure: description "X must enclose Y" → outer=X, inner=Y
    # Rule name often gives the inner layer (e.g. licon.10 = poly encloses licon)
    if category == "enclosure" and desc:
        m_outer = _ENCLOSURE_OUTER_RE.search(desc)
        if m_outer:
            outer = _normalize_layer(m_outer.group(1))
            if outer != layer:
                # Description overrides rule-name layer for outer
                inner = inner or layer
                layer = outer

    # For cross-layer spacing rules: pull every layer name in the description
    # so handlers can see both sides.  E.g. "licon on diff spacing to npc"
    # → primary=licon, secondary=npc.
    if category in ("spacing", "merge") and desc and not inner:
        _KNOWN = {"poly", "diff", "li1", "licon1", "mcon",
                  "nwell", "nsdm", "psdm", "npc", "rpo", "tap"}
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", desc):
            cand = _normalize_layer(tok)
            if not cand or cand == layer:
                continue
            is_known = cand in _KNOWN
            is_metal = (cand.startswith(("met", "via"))
                        and len(cand) > 3 and cand[3:].isdigit())
            if is_known or is_metal:
                inner = cand
                break
    required = _extract_required(desc)
    measured = v.value if v.value is not None else 0.0
    deficit  = max(0.0, required - measured) if required > 0 else 0.0

    # If DRC flagged a violation but deficit rounds to 0 (precision issue
    # or no measured value), apply a minimum nudge
    if deficit == 0.0 and required > 0:
        deficit = 0.01  # 10 nm nudge

    return ViolationInfo(
        category=category,
        layer=layer,
        inner_layer=inner,
        measured=measured,
        required=required,
        deficit=deficit,
        x=v.x,
        y=v.y,
        rule=rule,
        raw=v,
    )


def parse_violations(vs: Sequence[DRCViolation]) -> list[ViolationInfo]:
    """Parse a batch of DRC violations."""
    return [parse_violation(v) for v in vs]
