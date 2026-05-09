"""
layout_gen.repair.test_repair — end-to-end repair smoke test.

Loads a checkpointed denoiser, applies it iteratively to a synthesizer-
emitted cell, and reports per-step DRC violation counts.

Run::

    PDK_ROOT=/usr/local/share/pdk python -m layout_gen.repair.test_repair \\
        --checkpoint layout_gen/repair/data/denoiser_v7.pt \\
        --cell       inverter \\
        --pdk        sky130A \\
        --max-iter   30
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

from layout_gen import load_pdk
from layout_gen.synth.geo.state import LayoutState
from layout_gen.drc import get_runner
from layout_gen.repair.infer    import load_denoiser, repair
from layout_gen.repair.seeds    import synth_seeds


def _pdk_yaml(name: str) -> Path:
    import layout_gen
    return Path(layout_gen.__file__).parent / "pdks" / f"{name}.yaml"


def _load_synth_state(cell: str, pdk_name: str, rules) -> LayoutState | None:
    """Synthesize the named template and return its LayoutState."""
    seeds = synth_seeds(pdk_name)
    target = next((s for s in seeds if s.name == f"synth_{cell}"), None)
    if target is None:
        print(f"No synth seed named 'synth_{cell}'", file=sys.stderr)
        return None
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except Exception:
        from gdsfactory.gpdk import PDK as _GPDK
        _GPDK.activate()
    comp = gf.import_gds(str(target.gds_path), cellname=target.name)
    return LayoutState.from_component(comp, rules)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cell",       default="inverter",
                   help="Synth template name (matches templates/cells/<cell>.yaml)")
    p.add_argument("--pdk",        default="sky130A")
    p.add_argument("--max-iter",   type=int, default=30)
    args = p.parse_args()

    rules  = load_pdk(_pdk_yaml(args.pdk))
    runner = get_runner(rules)
    if runner is None:
        print("No DRC runner available; install Magic or KLayout.",
              file=sys.stderr)
        return 1

    state = _load_synth_state(args.cell, args.pdk, rules)
    if state is None:
        return 2
    print(f"Loaded {args.cell}: {len(state)} polygons")

    model = load_denoiser(args.checkpoint)
    print(f"Loaded {args.checkpoint}")

    print(f"\nRunning iterative repair (max {args.max_iter} steps)...")
    result = repair(state, model, runner, rules, max_iter=args.max_iter)

    # ── Summary ──
    print("\n" + "=" * 72)
    print(f"Initial violations:  {result.initial_violations}")
    print(f"Final violations:    {result.final_violations}")
    print(f"Iterations taken:    {result.iterations}")
    print(f"Converged DRC-clean: {result.converged}")
    print("=" * 72)

    if result.history:
        print("\nPer-step trace:")
        print(f"  {'iter':>4s}  {'before':>6s}  {'after':>6s}  "
              f"{'Δ':>4s}  action")
        for i, h in enumerate(result.history, 1):
            d = h.n_violations_after - h.n_violations_before
            sign = '+' if d > 0 else ('=' if d == 0 else '')
            print(f"  {i:>4d}  {h.n_violations_before:>6d}  "
                  f"{h.n_violations_after:>6d}  {sign}{d:>3d}  "
                  f"{h.action.kind} target={h.action.target} "
                  f"params={h.action.params}")
    return 0 if result.converged else 3


if __name__ == "__main__":
    sys.exit(main())
