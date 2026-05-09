"""
layout_gen.repair.augment_trajectories — enrich existing JSON trajectories
with per-violation (rule, x, y, deficit, layer).

Original miner saved only ``violation_rules`` (a list of rule names).  For
violation-conditioned training we need the **position** of each violation
to use as the model's conditioning input.  This script re-runs DRC on
the perturbed state of each trajectory and adds a richer ``violations``
field to the JSON.  ``violation_rules`` stays for backward compat.

Idempotent: trajectories that already carry the new field are skipped.

Run::

    PDK_ROOT=/usr/local/share/pdk python -m layout_gen.repair.augment_trajectories \\
        --root layout_gen/repair/data/trajectories
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from layout_gen import load_pdk
from layout_gen.synth.geo.state import LayoutState
from layout_gen.drc import get_runner
from layout_gen.repair.catalog import classify_rule, parse_required_um


def _pdk_yaml(name: str) -> Path:
    import layout_gen
    return Path(layout_gen.__file__).parent / "pdks" / f"{name}.yaml"


def _state_from_record(rec: dict) -> LayoutState:
    state = LayoutState()
    for r in rec["perturbed_state"]:
        new = state.add(layer=r["layer"],
                        x0=r["x0"], y0=r["y0"],
                        x1=r["x1"], y1=r["y1"],
                        net=r.get("net", ""),
                        shape_type=r.get("shape_type", ""))
        new.group_id = r.get("group_id", -1)
    return state


def _bbox(rec: dict) -> tuple[float, float, float, float]:
    rs = rec["perturbed_state"]
    return (min(r["x0"] for r in rs), min(r["y0"] for r in rs),
            max(r["x1"] for r in rs), max(r["y1"] for r in rs))


_DRC_COUNTER = [0]


def _enrich_record(rec: dict, rules, runner) -> dict | None:
    """Return a new record with a populated ``violations`` field, or
    ``None`` on failure.  Each violation is::

        {"rule": ..., "category": ..., "x_norm": ..., "y_norm": ...,
         "deficit_um": ..., "measured_um": ..., "required_um": ...}

    where (x_norm, y_norm) are cell-bbox-normalised coordinates in
    [0, 1]².
    """
    state = _state_from_record(rec)
    _DRC_COUNTER[0] += 1
    cell_name = f"augment_{_DRC_COUNTER[0]}"
    try:
        comp = state.to_component(rules, name=cell_name)
    except Exception as exc:
        import sys; print(f"  to_component failed: {exc}", file=sys.stderr)
        return None
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        gds_path = Path(f.name)
    try:
        comp.write_gds(str(gds_path))
        viols = runner.run(gds_path, cell_name)
    except Exception as exc:
        import sys; print(f"  drc failed: {exc}", file=sys.stderr)
        return None
    finally:
        gds_path.unlink(missing_ok=True)

    x0c, y0c, x1c, y1c = _bbox(rec)
    w = max(x1c - x0c, 1e-6)
    h = max(y1c - y0c, 1e-6)

    out_v: list[dict] = []
    for v in viols:
        cat = classify_rule(v.rule, v.description or "", [])
        req = parse_required_um(v.description or "")
        meas = float(v.value) if v.value is not None and v.value >= 0 else None
        deficit = (req - meas) if (req is not None and meas is not None) else None
        out_v.append({
            "rule":      v.rule,
            "category":  cat,
            "x_norm":    round((float(v.x) - x0c) / w, 6),
            "y_norm":    round((float(v.y) - y0c) / h, 6),
            "x_um":      round(float(v.x), 4),
            "y_um":      round(float(v.y), 4),
            "measured_um": meas,
            "required_um": req,
            "deficit_um":  deficit,
        })

    enriched = dict(rec)
    enriched["violations"] = out_v
    enriched["schema"]     = max(int(rec.get("schema", 1)), 2)
    return enriched


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path("layout_gen/repair/data/trajectories"))
    p.add_argument("--pdk", default="sky130A")
    p.add_argument("--max", type=int, default=0,
                   help="Stop after this many records (0 = all)")
    args = p.parse_args()

    files = sorted((args.root / args.pdk).glob("*.json"))
    if not files:
        print(f"No trajectory JSONs under {args.root / args.pdk}",
              file=sys.stderr)
        return 1

    # Activate gdsfactory PDK (needed by LayoutState.to_component)
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except Exception:
        from gdsfactory.gpdk import PDK as _GPDK
        _GPDK.activate()

    rules  = load_pdk(_pdk_yaml(args.pdk))
    runner = get_runner(rules)
    if runner is None:
        print("No DRC runner; install KLayout or Magic.", file=sys.stderr)
        return 2

    n_total = len(files)
    n_done   = 0
    n_skipped = 0
    n_errored = 0
    n_emptyv  = 0
    print(f"Augmenting {n_total} trajectories with violation positions…")

    for i, fpath in enumerate(files, 1):
        if args.max and n_done >= args.max:
            break
        try:
            rec = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            n_errored += 1
            continue

        if "violations" in rec and rec["violations"]:
            n_skipped += 1
            continue

        enriched = _enrich_record(rec, rules, runner)
        if enriched is None:
            n_errored += 1
            continue
        if not enriched["violations"]:
            n_emptyv += 1
            continue

        fpath.write_text(json.dumps(enriched), encoding="utf-8")
        n_done += 1
        if n_done % 50 == 0:
            print(f"  [{i}/{n_total}] done={n_done}  "
                  f"skipped={n_skipped}  emptyV={n_emptyv}  err={n_errored}")

    print(f"\nFinished: done={n_done}, skipped={n_skipped}, "
          f"emptyV={n_emptyv}, err={n_errored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
