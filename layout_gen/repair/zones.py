"""
layout_gen.repair.zones — conflict graph and zone extraction.

A *fix zone* is a bounded geometric region within which an edit may
introduce side effects.  The locality property of CMOS DRC (every rule's
horizon is bounded — typically < 1–2 µm) means side effects don't
propagate arbitrarily far.  We exploit that to factor whole-layout
repair into zone-local repair.

Two violations belong to the same zone iff their *dilated* bounding
boxes overlap, where the dilation radius is the rule's *fix-zone
radius* — derived from the catalog or from the active PDK's worst-rule
horizon.

Connected components of the resulting conflict graph are zones.  Each
zone is the natural reasoning unit for the skill library: a small CSP
over a handful of polygons and rules, solvable by a short action
sequence.

Usage::

    from layout_gen.repair.zones import extract_zones
    zones = extract_zones(violations, rules=pdk_rules, default_radius_um=0.5)
    for z in zones:
        print(z.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from layout_gen.drc.base import DRCViolation


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _bbox_overlap(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> bool:
    """True if two AABB rectangles (x0, y0, x1, y1) overlap (inclusive)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_union(a: tuple[float, float, float, float],
                b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


# ── Zone ─────────────────────────────────────────────────────────────────────

@dataclass
class Zone:
    """A connected cluster of DRC violations sharing a fix region.

    Attributes
    ----------
    id :
        Sequential index within the layout (0-based).
    bbox :
        Tight axis-aligned bounding box around all violations + their
        dilation radii: ``(x0, y0, x1, y1)``.
    violations :
        DRC violations clustered into this zone.
    radius_um :
        Maximum dilation radius among the violations in this zone.
        Bounds the geometric region that any fix in this zone can
        affect.
    """
    id:         int
    bbox:       tuple[float, float, float, float]
    violations: list[DRCViolation] = field(default_factory=list)
    radius_um:  float              = 0.0

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    @property
    def size_um(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0, y1 - y0)

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    def rule_counts(self) -> dict[str, int]:
        """Return ``{rule_name: count}`` for the violations in this zone."""
        out: dict[str, int] = {}
        for v in self.violations:
            out[v.rule] = out.get(v.rule, 0) + 1
        return out

    @property
    def n_distinct_rules(self) -> int:
        return len({v.rule for v in self.violations})

    def dominant_rule(self) -> str:
        """Rule contributing the most violations to this zone."""
        rc = self.rule_counts()
        if not rc:
            return ""
        return max(rc.items(), key=lambda kv: kv[1])[0]

    def is_homogeneous(self) -> bool:
        """True when every violation in the zone is the same rule.

        Homogeneous zones are the strongest candidates for a single
        coarse abstract action ("merge all NPCs", "stretch all li1
        edges by deficit").
        """
        return self.n_distinct_rules == 1

    def summary(self) -> str:
        rc = self.rule_counts()
        rule_str = ", ".join(f"{r}x{n}" for r, n in sorted(rc.items()))
        cx, cy = self.center
        w, h   = self.size_um
        return (
            f"Zone#{self.id}  center=({cx:.2f},{cy:.2f})  "
            f"size=({w:.2f}x{h:.2f})µm  "
            f"viols={self.n_violations}  rules=[{rule_str}]"
        )


# ── Default radius lookup ────────────────────────────────────────────────────

# A rule's fix-zone radius is the maximum geometric horizon over which an
# edit anywhere inside the rule's reported region can plausibly cascade.
# It depends on the PDK and on the rule category.  When we have catalog
# data with a numeric required value, we use that as the radius.  When we
# don't, we fall back to a *PDK-wide worst-rule horizon* (well-spacing).

def pdk_worst_rule_um(rules) -> float:
    """Conservative upper bound on rule-horizon for the active PDK.

    Pulls the largest spacing/width number we can find in the rules dict.
    PDK YAML values, not training data — runtime lookup, allowed.
    """
    candidates = []
    for sec_name in ("nwell", "implant", "diff", "li1", "met1", "met2"):
        sec = getattr(rules, sec_name, {}) or {}
        for key, val in sec.items():
            if not isinstance(val, (int, float)):
                continue
            if "spacing" in key or "width" in key:
                candidates.append(float(val))
    return max(candidates) if candidates else 1.5


# Rules that are global / array-level rather than cell-local: they describe
# requirements that can only be satisfied by structures *outside* the cell
# (tap rows for latchup, well-spacing across cells, antenna across nets).
# We exclude them from zone extraction because their fix is structural
# (insert tap row, add tie-down) and not a local edit.
GLOBAL_RULES: tuple[str, ...] = (
    "LU.1", "LU.2", "LU.3", "LU.4", "LU.5",
    "nwell.4",
    "nsd.7", "psd.7",                   # well-region long-range
    "antenna.", "ant.",                 # antenna effects
    "density.", "den.",                 # density windows
)


def is_global_rule(rule_name: str) -> bool:
    rl = (rule_name or "").lower()
    return any(rl.startswith(g.lower()) or rl == g.lower().rstrip(".")
               for g in GLOBAL_RULES)


def default_zone_radius(violation: DRCViolation, rules) -> float:
    """Pick a zone radius for *violation*.

    Priority:
    1. The required-value parsed from the description (the rule's own
       horizon — directly bounds side effects).
    2. The PDK's worst-rule horizon (always safe).
    """
    from layout_gen.repair.catalog import parse_required_um
    req = parse_required_um(violation.description or "")
    if req is not None and req > 0:
        return req
    return pdk_worst_rule_um(rules)


# ── Union-Find ───────────────────────────────────────────────────────────────

class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[ri] = rj


# ── Public API ───────────────────────────────────────────────────────────────

def extract_zones(
    violations:        list[DRCViolation],
    *,
    rules:             object | None = None,
    default_radius_um: float | None  = None,
    drop_global:       bool          = True,
) -> list[Zone]:
    """Cluster *violations* into fix zones.

    Each violation's bounding box is its centroid dilated by the
    rule-specific radius (or *default_radius_um*).  Violations whose
    boxes overlap go into the same zone.

    Parameters
    ----------
    violations :
        DRC violations from one cell.  Empty input → empty output.
    rules :
        :class:`PDKRules` for radius fallback.  Optional when every
        violation's description has a parseable required value.
    default_radius_um :
        Override the per-violation lookup with a fixed radius (useful for
        debugging / zone-size sensitivity analysis).
    drop_global :
        Skip violations of rules in :data:`GLOBAL_RULES`.  These describe
        array-level / structural requirements (latchup, density, antenna)
        whose fix is not a local edit; they belong on a separate
        non-zone agent.  Default True.
    """
    if drop_global:
        violations = [v for v in violations if not is_global_rule(v.rule)]
    if not violations:
        return []

    # 1. Compute a dilated bbox per violation.
    boxes:    list[tuple[float, float, float, float]] = []
    radii:    list[float] = []
    for v in violations:
        if default_radius_um is not None:
            r = float(default_radius_um)
        elif rules is not None:
            r = default_zone_radius(v, rules)
        else:
            r = 1.5
        radii.append(r)
        x, y = float(v.x), float(v.y)
        boxes.append((x - r, y - r, x + r, y + r))

    # 2. Pairwise overlap → union-find.  O(N²) but N is small per cell
    #    (hundreds at most).
    uf = _UF(len(violations))
    for i in range(len(violations)):
        for j in range(i + 1, len(violations)):
            if _bbox_overlap(boxes[i], boxes[j]):
                uf.union(i, j)

    # 3. Aggregate per-component → Zone objects.
    components: dict[int, list[int]] = {}
    for i in range(len(violations)):
        components.setdefault(uf.find(i), []).append(i)

    zones: list[Zone] = []
    for zid, members in enumerate(sorted(components.values(), key=lambda m: m[0])):
        agg_bbox = boxes[members[0]]
        for m in members[1:]:
            agg_bbox = _bbox_union(agg_bbox, boxes[m])
        zones.append(Zone(
            id=zid,
            bbox=agg_bbox,
            violations=[violations[m] for m in members],
            radius_um=max(radii[m] for m in members),
        ))
    return zones


# ── Per-cell statistics ──────────────────────────────────────────────────────

@dataclass
class ZoneStats:
    """Aggregated zone statistics for one cell."""
    cell:               str
    n_violations:       int
    n_zones:            int
    median_violations_per_zone: int
    max_violations_per_zone:    int
    median_zone_size_um:        float
    max_zone_size_um:           float
    n_homogeneous_zones:        int = 0   # zones with one rule type
    median_distinct_rules_per_zone: int = 0

    def __repr__(self) -> str:
        return (f"ZoneStats({self.cell}: {self.n_violations} viol → "
                f"{self.n_zones} zones, "
                f"med_v/z={self.median_violations_per_zone}, "
                f"max_v/z={self.max_violations_per_zone}, "
                f"med_size={self.median_zone_size_um:.2f}, "
                f"max_size={self.max_zone_size_um:.2f}, "
                f"homog={self.n_homogeneous_zones}/{self.n_zones})")


def zone_stats(cell: str, zones: list[Zone]) -> ZoneStats:
    if not zones:
        return ZoneStats(cell=cell, n_violations=0, n_zones=0,
                         median_violations_per_zone=0,
                         max_violations_per_zone=0,
                         median_zone_size_um=0.0,
                         max_zone_size_um=0.0)
    n_viols   = sum(z.n_violations for z in zones)
    v_per_z   = sorted(z.n_violations for z in zones)
    sizes     = sorted(max(z.size_um) for z in zones)
    distinct  = sorted(z.n_distinct_rules for z in zones)
    n_homog   = sum(1 for z in zones if z.is_homogeneous())
    return ZoneStats(
        cell=cell,
        n_violations=n_viols,
        n_zones=len(zones),
        median_violations_per_zone=v_per_z[len(v_per_z) // 2],
        max_violations_per_zone=v_per_z[-1],
        median_zone_size_um=round(sizes[len(sizes) // 2], 4),
        max_zone_size_um=round(sizes[-1], 4),
        n_homogeneous_zones=n_homog,
        median_distinct_rules_per_zone=distinct[len(distinct) // 2],
    )


__all__ = [
    "Zone",
    "ZoneStats",
    "extract_zones",
    "zone_stats",
    "default_zone_radius",
    "pdk_worst_rule_um",
]
