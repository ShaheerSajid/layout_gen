"""
layout_gen.repair.mine_trajectories — build training data for the
diffusion-style DRC denoiser.

For every (clean seed) × (perturbation depth k ∈ [1, K]) × (random seed),
this script:

1. Loads the seed layout into a :class:`LayoutState`.
2. Applies k random perturbations, recording the inverse trajectory.
3. Verifies via DRC that violations were actually created (otherwise the
   perturbation was a no-op — discarded).
4. Saves the resulting record::

       {
         "seed_pdk":       "sky130A",
         "seed_cell":      "sky130_fd_sc_hd__inv_1",
         "seed_primitive": "inv",
         "k":              3,
         "rng_seed":       17,
         "perturbed_state": <serialised LayoutState>,
         "inverse_action_sequence": [<PerturbAction>, ...],
         "n_violations":   12,
         "violation_rules": ["licon.13", "npc.2", ...]
       }

   to ``layout_gen/repair/data/trajectories/<pdk>/<seed>_k<k>_s<seed>.json``.

This is the diffusion training set: each record is a (state, inverse_step)
pair at known noise level k, with a one-shot reverse step labelled
unambiguously.  The model trained on these learns the score / denoising
direction of the layout-DRC distribution.

Key invariants
--------------
* **PDK-agnostic by construction.**  Layer roles + deficit values feed the
  model, never raw µm constants.
* **Clean seeds only.**  We start from layouts that pass DRC, so any
  violations on the perturbed state are caused by the perturbation alone.
* **Inverse-action labels are exact.**  The forward perturbation knows
  its own inverse — no human or rule-agent labelling needed.

Run::

    PDK_ROOT=/usr/local/share/pdk python -m layout_gen.repair.mine_trajectories \\
        --pdks sky130A --max-per-primitive 1 --depths 1 2 3 \\
        --seeds-per-config 5 --out layout_gen/repair/data/trajectories
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

from layout_gen import load_pdk
from layout_gen.synth.geo.state import LayoutState
from layout_gen.drc import get_runner

from layout_gen.repair.perturb import (
    PerturbConfig, generate_trajectory, snapshot_state,
)
from layout_gen.repair.seeds import all_seeds_for, SeedCell


def _pdk_yaml(name: str) -> Path:
    import layout_gen
    return Path(layout_gen.__file__).parent / "pdks" / f"{name}.yaml"


def _serialise_state(state: LayoutState) -> list[dict]:
    """Compact per-rect dump for the perturbed state."""
    return [
        {"rid": r.rid, "layer": r.layer,
         "x0": round(r.x0, 6), "y0": round(r.y0, 6),
         "x1": round(r.x1, 6), "y1": round(r.y1, 6),
         "net": r.net, "shape_type": r.shape_type, "group_id": r.group_id}
        for r in state
    ]


def _seed_to_state(seed: SeedCell, rules) -> LayoutState | None:
    """Load seed GDS into a :class:`LayoutState` via gdsfactory."""
    try:
        import gdsfactory as gf
        try:
            gf.get_active_pdk()
        except Exception:
            from gdsfactory.gpdk import PDK as _GPDK
            _GPDK.activate()
        comp = gf.import_gds(str(seed.gds_path), cellname=seed.name)
        return LayoutState.from_component(comp, rules)
    except Exception:
        return None


_DRC_COUNTER = [0]


def _drc_count(state: LayoutState, rules, runner) -> tuple[int, list[str]]:
    """Run DRC on *state* via a temp GDS, return (violation_count, rule_names)."""
    _DRC_COUNTER[0] += 1
    cell_name = f"mine_check_{_DRC_COUNTER[0]}"
    try:
        comp = state.to_component(rules, name=cell_name)
    except Exception:
        return -1, []
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        gds = Path(f.name)
    try:
        comp.write_gds(str(gds))
        viols = runner.run(gds, cell_name)
    except Exception:
        return -1, []
    finally:
        gds.unlink(missing_ok=True)
    return len(viols), [v.rule for v in viols]


def mine(
    pdks:                list[str],
    depths:              list[int],
    seeds_per_config:    int,
    max_per_primitive:   int | None,
    out_dir:             Path,
    require_violations:  bool = True,
    verbose:             bool = True,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"total": 0, "kept": 0, "no_viol": 0, "errors": 0,
              "round_trip_failed": 0}

    for pdk_name in pdks:
        rules  = load_pdk(_pdk_yaml(pdk_name))
        runner = get_runner(rules)
        if runner is None:
            print(f"[{pdk_name}] no DRC runner available; skipping",
                  file=sys.stderr)
            continue
        seeds = all_seeds_for(pdk_name,
                              max_per_primitive=max_per_primitive,
                              include_synth=False)   # synth seeds aren't clean
        if verbose:
            print(f"[{pdk_name}] {len(seeds)} clean seeds, depths={depths}, "
                  f"seeds/config={seeds_per_config}")

        pdk_dir = out_dir / pdk_name
        pdk_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            for k in depths:
                for s in range(seeds_per_config):
                    counts["total"] += 1
                    state = _seed_to_state(seed, rules)
                    if state is None:
                        counts["errors"] += 1
                        continue

                    init = snapshot_state(state)
                    rng_seed = hash((seed.name, k, s)) % (2**31)
                    rng = random.Random(rng_seed)
                    cfg = PerturbConfig(
                        delta_min_um=0.02,
                        delta_max_um=0.08,
                        forbid_kinds=frozenset({"delete_rect"}),
                    )
                    try:
                        traj = generate_trajectory(state, n_steps=k,
                                                   config=cfg, rng=rng)
                    except Exception:
                        counts["errors"] += 1
                        continue

                    if require_violations:
                        n_viol, rule_list = _drc_count(state, rules, runner)
                        if n_viol <= 0:
                            counts["no_viol"] += 1
                            continue
                    else:
                        n_viol, rule_list = -1, []

                    record = {
                        "schema":          1,
                        "seed_pdk":        pdk_name,
                        "seed_cell":       seed.name,
                        "seed_primitive":  seed.primitive,
                        "seed_source":     seed.source,
                        "k":               k,
                        "rng_seed":        rng_seed,
                        "perturbed_state": _serialise_state(state),
                        "forward_action_sequence": [a.to_dict() for a in traj.forward],
                        "inverse_action_sequence": [a.to_dict() for a in traj.inverse],
                        "n_violations":    n_viol,
                        "violation_rules": rule_list,
                    }
                    safe_name = seed.name.replace("/", "_").replace("$", "_")
                    out_path = pdk_dir / f"{safe_name}_k{k}_s{s}.json"
                    out_path.write_text(json.dumps(record), encoding="utf-8")
                    counts["kept"] += 1

                    if verbose and counts["kept"] % 20 == 0:
                        print(f"  ...kept={counts['kept']} "
                              f"no_viol={counts['no_viol']} "
                              f"err={counts['errors']}")

    if verbose:
        print(f"\nSummary: {counts}")
    return counts


def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pdks", nargs="+", default=["sky130A"])
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--seeds-per-config", type=int, default=3,
                   help="Random samples per (cell × depth)")
    p.add_argument("--max-per-primitive", type=int, default=2,
                   help="Cap SCL cells per primitive bucket (default 2)")
    p.add_argument("--out", type=Path,
                   default=Path("layout_gen/repair/data/trajectories"))
    p.add_argument("--no-require-violations", action="store_true",
                   help="Keep records even when the perturbation didn't "
                        "actually trigger DRC.  Useful for fast iteration "
                        "but lower-quality data.")
    args = p.parse_args()

    mine(
        pdks=args.pdks,
        depths=args.depths,
        seeds_per_config=args.seeds_per_config,
        max_per_primitive=args.max_per_primitive,
        out_dir=args.out,
        require_violations=not args.no_require_violations,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
