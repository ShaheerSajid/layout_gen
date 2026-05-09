"""
layout_gen.repair.analyze_zones — measure zone-decomposition statistics.

Validates the locality assumption from PLAN.md: zones should be small
(few violations, few µm wide) and bounded.  Produces a per-cell summary
plus aggregate statistics across the seed corpus.

Run::

    PDK_ROOT=/usr/local/share/pdk python -m layout_gen.repair.analyze_zones \\
        --pdks sky130A gf180mcuD --max-per-primitive 2
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from layout_gen import load_pdk
from layout_gen.drc import registry, available_tools

from layout_gen.repair.seeds import all_seeds_for
from layout_gen.repair.zones import extract_zones, zone_stats


def _pdk_yaml(name: str) -> Path:
    import layout_gen
    return Path(layout_gen.__file__).parent / "pdks" / f"{name}.yaml"


def analyse(
    pdks:              list[str],
    max_per_primitive: int | None = None,
    drc_tools:         list[str] | None = None,
    only_failing:      bool = True,
) -> None:
    """Walk seeds, run DRC with each available tool, extract zones, print stats."""
    all_stats: list = []

    for pdk_name in pdks:
        rules   = load_pdk(_pdk_yaml(pdk_name))
        seeds   = all_seeds_for(pdk_name, max_per_primitive=max_per_primitive)
        tools   = drc_tools if drc_tools else available_tools()
        runners = []
        for t in tools:
            try:
                r = registry.get(t, rules=rules)
                if r.is_available():
                    runners.append((t, r))
            except Exception:
                continue
        if not runners:
            print(f"[{pdk_name}] no DRC runner available; skipping", file=sys.stderr)
            continue

        print(f"\n[{pdk_name}] {len(seeds)} seeds, "
              f"tools={[t for t, _ in runners]}")
        print("-" * 96)
        print(f"  {'cell':35s}  {'viol':>5s}  {'zones':>6s}  "
              f"{'med v/z':>8s}  {'max v/z':>8s}  "
              f"{'med size':>9s}  {'max size':>9s}  "
              f"{'homog':>6s}  {'rules/z':>8s}")
        print("-" * 116)

        for seed in seeds:
            all_viols = []
            for _, runner in runners:
                try:
                    all_viols.extend(runner.run(seed.gds_path, seed.name))
                except Exception:
                    continue
            if only_failing and not all_viols:
                continue
            zones = extract_zones(all_viols, rules=rules)
            s = zone_stats(seed.name, zones)
            all_stats.append((pdk_name, seed.primitive, s))
            print(f"  {seed.name:35s}  "
                  f"{s.n_violations:5d}  {s.n_zones:6d}  "
                  f"{s.median_violations_per_zone:8d}  "
                  f"{s.max_violations_per_zone:8d}  "
                  f"{s.median_zone_size_um:9.3f}  "
                  f"{s.max_zone_size_um:9.3f}  "
                  f"{s.n_homogeneous_zones:>3d}/{s.n_zones:<2d}  "
                  f"{s.median_distinct_rules_per_zone:8d}")

    if not all_stats:
        return

    print("\n" + "=" * 96)
    print("Aggregate (cells with ≥1 violation)")
    print("=" * 96)
    n_viols   = [s.n_violations for _, _, s in all_stats]
    n_zones   = [s.n_zones      for _, _, s in all_stats]
    v_per_z   = [s.median_violations_per_zone for _, _, s in all_stats]
    max_v_per_z = [s.max_violations_per_zone for _, _, s in all_stats]
    sizes     = [s.median_zone_size_um for _, _, s in all_stats]
    max_sizes = [s.max_zone_size_um    for _, _, s in all_stats]

    def fmt(name, xs):
        if not xs: return f"  {name}: (no data)"
        return (f"  {name}:  median={statistics.median(xs):.2f}  "
                f"mean={statistics.mean(xs):.2f}  "
                f"max={max(xs):.2f}  count={len(xs)}")

    print(fmt("violations per cell        ", n_viols))
    print(fmt("zones per cell             ", n_zones))
    print(fmt("median violations per zone ", v_per_z))
    print(fmt("max violations per zone    ", max_v_per_z))
    print(fmt("median zone size (µm)      ", sizes))
    print(fmt("max zone size (µm)         ", max_sizes))


def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pdks", nargs="+", default=["sky130A"])
    p.add_argument("--max-per-primitive", type=int, default=2)
    p.add_argument("--drc-tools", nargs="*", default=None)
    p.add_argument("--all-cells", action="store_true",
                   help="Include cells with zero violations (default: skip).")
    args = p.parse_args()
    analyse(
        pdks=args.pdks,
        max_per_primitive=args.max_per_primitive,
        drc_tools=args.drc_tools,
        only_failing=not args.all_cells,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
