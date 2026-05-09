"""
layout_gen.repair.build_catalog — build the DRC rule catalog from seeds.

Run as a script::

    PDK_ROOT=/usr/local/share/pdk python -m layout_gen.repair.build_catalog \
        --pdks sky130A gf180mcuD \
        --max-per-primitive 2 \
        --out layout_gen/repair/data/catalog.yaml

For each (PDK, seed cell) pair this:

1. Loads the PDK rules (and DRC tool).
2. Runs DRC on the seed's GDS top cell.
3. Records every violation into the :class:`CatalogBuilder`.

The result is the empirical catalog of every rule observed across the
seed corpus, classified by category and annotated with sample violations.
This data is what the repair engine, the perturbation library, and the
zone extractor will consume — none of them encode any PDK constants
themselves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from layout_gen import load_pdk
from layout_gen.drc import get_runner

from layout_gen.repair.catalog import CatalogBuilder
from layout_gen.repair.seeds   import all_seeds_for, SeedCell


def _pdk_yaml_for(name: str) -> Path:
    """Resolve PDK YAML by name (uses bundled YAMLs in layout_gen/pdks/)."""
    import layout_gen
    return Path(layout_gen.__file__).parent / "pdks" / f"{name}.yaml"


def _build_runners(rules, requested_tools: list[str] | None):
    """Build the list of (tool_name, runner) pairs to run.

    When ``requested_tools`` is None, every available tool is used.  This
    is desirable because different DRC decks emit different rule names
    (Magic catches LU/well rules; KLayout's sky130A_mr deck catches
    licon.13 / npc.2 / etc.) and the catalog should union them all.
    """
    from layout_gen.drc import registry, available_tools as _avail
    pairs: list = []
    candidates = requested_tools if requested_tools else _avail()
    for name in candidates:
        try:
            r = registry.get(name, rules=rules)
            if r.is_available():
                pairs.append((name, r))
        except Exception:
            continue
    return pairs


def build(
    pdks:               list[str],
    max_per_primitive:  int | None = None,
    drc_tools:          list[str] | None = None,
    out_path:           Path | None = None,
    verbose:            bool = True,
) -> CatalogBuilder:
    builder = CatalogBuilder()

    for pdk_name in pdks:
        rules   = load_pdk(_pdk_yaml_for(pdk_name))
        runners = _build_runners(rules, drc_tools)
        if not runners:
            print(f"[{pdk_name}] no DRC runner available; skipping", file=sys.stderr)
            continue

        seeds = all_seeds_for(pdk_name, max_per_primitive=max_per_primitive)
        tool_names = ", ".join(t for t, _ in runners)
        if verbose:
            print(f"[{pdk_name}] {len(seeds)} seeds, DRC tools=[{tool_names}]")

        for i, seed in enumerate(seeds, 1):
            total = 0
            tool_breakdown: list[str] = []
            for tool_name, runner in runners:
                try:
                    viols = runner.run(seed.gds_path, seed.name)
                except Exception as exc:
                    tool_breakdown.append(f"{tool_name}=ERR")
                    continue
                for v in viols:
                    builder.record(v, pdk=pdk_name, cell=seed.name)
                tool_breakdown.append(f"{tool_name}={len(viols)}")
                total += len(viols)
            if verbose:
                print(f"  [{i:2d}/{len(seeds)}] {seed.name:35s}  "
                      f"{seed.primitive:14s}  total={total:4d}  ({', '.join(tool_breakdown)})")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(builder.to_yaml(), encoding="utf-8")
        if verbose:
            print(f"\nWrote {out_path}")
    return builder


# ── CLI ──────────────────────────────────────────────────────────────────────

def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdks", nargs="+", default=["sky130A"],
                   help="PDKs to walk (default: sky130A)")
    p.add_argument("--max-per-primitive", type=int, default=None,
                   help="Cap SCL cells per primitive bucket (default: no cap)")
    p.add_argument(
        "--drc-tools", nargs="*", default=None,
        help="DRC tools to run (default: every available tool, unioned)",
    )
    p.add_argument("--out", type=Path, default=None,
                   help="Output YAML path (omit for stdout summary)")
    args = p.parse_args()

    builder = build(
        pdks=args.pdks,
        max_per_primitive=args.max_per_primitive,
        drc_tools=args.drc_tools,
        out_path=args.out,
    )

    if args.out is None:
        # Print a compact summary to stdout
        builder.finalise()
        print()
        print("=" * 78)
        print(f"Catalog summary: {len(builder.entries)} distinct rules across "
              f"{len(set(p for p, _ in builder.entries))} PDK(s)")
        print("=" * 78)
        from collections import Counter
        cats = Counter(e.category for e in builder.entries.values())
        for cat, n in cats.most_common():
            print(f"  {cat:15s}  x{n}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
