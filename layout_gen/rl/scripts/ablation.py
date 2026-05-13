"""
layout_gen.rl.scripts.ablation — automate train+eval pairs for A/B comparisons.

Without this script, the question "did adding the wiremask channel
actually help?" requires manually running ``train_ppo`` twice with
different flags, then ``eval`` twice, then squinting at numbers. With
this script:

    .venv/bin/python -m layout_gen.rl.scripts.ablation \\
        --topology inverter --episodes 8 --total-timesteps 2000 \\
        --variants ablation_ibrl

prints a side-by-side table over the variants defined in
:data:`PRESETS`, plus a CSV in ``--out-csv``.

Each preset is a list of ``Variant`` dicts; each variant overlays a
small set of ``train_ppo`` argv overrides on the base command. The
script trains each variant under identical env / seed / step budget,
then evals all variants on a held-out episode set with the new
:mod:`layout_gen.rl.scripts.eval` harness.

Intended use is **fast iteration on the design space** — keep
``--total-timesteps`` small (1–5k) for tight feedback; bump up only
when narrowing in on the variant you want to publish.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from layout_gen.rl.scripts import eval as eval_cli
from layout_gen.rl.scripts import train_ppo as train_ppo_cli


# ── Preset definitions ──────────────────────────────────────────────────────

@dataclass
class Variant:
    name:           str
    train_extra:    list[str] = field(default_factory=list)
    bc_init:        str | None = None
    description:    str = ""


PRESETS: dict[str, list[Variant]] = {
    # IBRL ablation: BC init alone vs BC init + KL distillation.
    # Pass --bc-init <path> to ablation.py to populate bc_init for both.
    "ibrl": [
        Variant(
            name="bc_only",
            train_extra=[],
            description="BC checkpoint loads weights once, then plain PPO.",
        ),
        Variant(
            name="bc_distill",
            train_extra=["--ibrl-bc-init", "{bc_init}",
                          "--ibrl-beta-start", "1.0",
                          "--ibrl-beta-end",   "0.0"],
            description="BC init + KL(π_PPO || π_BC) added to the PPO loss "
                        "with β decaying linearly to 0.",
        ),
    ],

    # Reward-shaping ablation: the new HPWL / electrical / short / LVS terms
    # vs disabling them one at a time.
    "rewards": [
        Variant(name="full",  description="All reward terms at default weights."),
        Variant(
            name="no_short", train_extra=[],   # placeholder
            description="short_delta zero (no short-circuit penalty).",
        ),
    ],
}


# ── Variant runner ──────────────────────────────────────────────────────────

@dataclass
class VariantResult:
    name:                 str
    checkpoint:           str
    n_episodes:           int
    drc_clean_rate:       float
    inspector_pass_rate:  float
    ep_reward_mean:       float
    ep_len_mean:          float
    electrical_mean:      float
    connectivity_mean:    float
    alignment_mean:       float
    hpwl_mean:            float
    per_topology:         dict[str, dict] = field(default_factory=dict)


def _run_variant(
    variant:  Variant,
    *,
    base_train_args: list[str],
    base_eval_args:  list[str],
    out_dir:         Path,
    bc_init:         str | None,
    verbose:         bool,
) -> VariantResult:
    """Train + eval one variant. Returns aggregated metrics."""
    ckpt = out_dir / f"{variant.name}.zip"

    # Substitute any ``{bc_init}`` placeholder in the variant's extra args.
    extra: list[str] = []
    for tok in variant.train_extra:
        if "{bc_init}" in tok:
            if not bc_init:
                raise SystemExit(
                    f"variant '{variant.name}' references {{bc_init}} but "
                    "no --bc-init was passed to ablation.py"
                )
            tok = tok.replace("{bc_init}", bc_init)
        extra.append(tok)

    train_argv = list(base_train_args) + extra + ["--out", str(ckpt)]
    if bc_init and "--bc-init" not in train_argv:
        train_argv += ["--bc-init", bc_init]

    if verbose:
        print(f"\n[ablation] === train variant '{variant.name}' ===")
        print(f"  argv: {' '.join(shlex.quote(a) for a in train_argv)}")
    rc = train_ppo_cli.main(train_argv)
    if rc != 0:
        raise SystemExit(f"variant '{variant.name}' training failed (rc={rc})")
    assert ckpt.exists(), f"checkpoint not written for {variant.name}"

    # Eval the resulting checkpoint.
    json_path = out_dir / f"{variant.name}.eval.json"
    eval_argv = list(base_eval_args) + [
        "--checkpoint", str(ckpt),
        "--out-json", str(json_path),
    ]
    if verbose:
        print(f"\n[ablation] --- eval variant '{variant.name}' ---")
    rc = eval_cli.main(eval_argv)
    if rc != 0:
        raise SystemExit(f"variant '{variant.name}' eval failed (rc={rc})")
    rep = json.loads(json_path.read_text())
    return VariantResult(
        name=variant.name, checkpoint=str(ckpt),
        n_episodes=rep["n_episodes"],
        drc_clean_rate=rep["drc_clean_rate"],
        inspector_pass_rate=rep["inspector_pass_rate"],
        ep_reward_mean=rep["ep_reward_mean"],
        ep_len_mean=rep["ep_len_mean"],
        electrical_mean=rep["electrical_mean"],
        connectivity_mean=rep["connectivity_mean"],
        alignment_mean=rep["alignment_mean"],
        hpwl_mean=rep["hpwl_mean"],
        per_topology=rep["per_topology"],
    )


# ── Reporting ───────────────────────────────────────────────────────────────

def _print_table(results: list[VariantResult]) -> None:
    print("\n─── ablation comparison ─────────────────────────────────────────")
    headers = ("variant", "ep_rew", "drc_clean%", "inspect%",
               "electrical", "alignment", "hpwl")
    print(f"  {headers[0]:14s} {headers[1]:>8s} {headers[2]:>10s} "
          f"{headers[3]:>9s} {headers[4]:>10s} {headers[5]:>9s} "
          f"{headers[6]:>8s}")
    print("  " + "─" * 70)
    for r in results:
        print(
            f"  {r.name:14s} {r.ep_reward_mean:+8.3f} "
            f"{r.drc_clean_rate*100:9.1f}% "
            f"{r.inspector_pass_rate*100:8.1f}% "
            f"{r.electrical_mean:10.3f} "
            f"{r.alignment_mean:9.3f} "
            f"{r.hpwl_mean:+8.3f}"
        )

    if len(results) >= 2:
        # Diff against the first variant for quick eye-balling.
        baseline = results[0]
        print("\n─── Δ vs first variant ──────────────────────────────────────────")
        for r in results[1:]:
            d_rew     = r.ep_reward_mean      - baseline.ep_reward_mean
            d_drc     = (r.drc_clean_rate     - baseline.drc_clean_rate) * 100
            d_inspect = (r.inspector_pass_rate - baseline.inspector_pass_rate) * 100
            d_elec    = r.electrical_mean      - baseline.electrical_mean
            d_align   = r.alignment_mean       - baseline.alignment_mean
            d_hpwl    = r.hpwl_mean            - baseline.hpwl_mean
            print(
                f"  {r.name:14s} {d_rew:+8.3f} "
                f"{d_drc:+9.1f}p "
                f"{d_inspect:+8.1f}p "
                f"{d_elec:+10.3f} "
                f"{d_align:+9.3f} "
                f"{d_hpwl:+8.3f}"
            )


def _write_csv(results: list[VariantResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "checkpoint", "n_episodes",
        "ep_reward_mean", "ep_len_mean",
        "drc_clean_rate", "inspector_pass_rate",
        "electrical_mean", "connectivity_mean",
        "alignment_mean", "hpwl_mean",
    ]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fields}
            w.writerow(row)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", required=True,
                   help=f"Preset name from PRESETS: {sorted(PRESETS)}")
    p.add_argument("--bc-init", default=None,
                   help="Optional BC checkpoint applied to every variant. "
                        "Variants that reference {bc_init} substitute this.")

    # Train args (the base; variants extend with their own).
    p.add_argument("--topology", default=None)
    p.add_argument("--topologies", default=None)
    p.add_argument("--total-timesteps", type=int, default=2000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=2)
    p.add_argument("--max-place-steps", type=int, default=4)
    p.add_argument("--max-route-steps", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--device-cap", type=int, default=8)
    p.add_argument("--net-cap", type=int, default=8)
    p.add_argument("--position-bins", type=int, default=8)
    p.add_argument("--route-size-bins", type=int, default=4)
    p.add_argument("--mag-bins", type=int, default=8)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--no-drc", action="store_true",
                   help="Skip real DRC during training (fast).")

    # Eval args.
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--no-route-eval", action="store_true",
                   help="Disable route in eval (matches a --no-route train).")

    p.add_argument("--out-dir", type=Path,
                   default=Path("ablation_runs"),
                   help="Where to write per-variant checkpoints + JSON evals.")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional CSV path for the comparison table.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.variants not in PRESETS:
        raise SystemExit(
            f"unknown --variants '{args.variants}'; available: {sorted(PRESETS)}"
        )
    variants = PRESETS[args.variants]
    if not args.topology and not args.topologies:
        raise SystemExit("error: --topology or --topologies required")

    # Common train args reused by every variant.
    base_train: list[str] = [
        "--enable-place", "--enable-route",
        "--total-timesteps", str(args.total_timesteps),
        "--n-envs",     str(args.n_envs),
        "--n-steps",    str(args.n_steps),
        "--batch-size", str(args.batch_size),
        "--n-epochs",   str(args.n_epochs),
        "--max-place-steps", str(args.max_place_steps),
        "--max-route-steps", str(args.max_route_steps),
        "--max-steps",       str(args.max_steps),
        "--device-cap",      str(args.device_cap),
        "--net-cap",         str(args.net_cap),
        "--position-bins",   str(args.position_bins),
        "--route-size-bins", str(args.route_size_bins),
        "--mag-bins",        str(args.mag_bins),
        "--ent-coef",        str(args.ent_coef),
    ]
    if args.no_drc:
        base_train.append("--no-drc")
    if args.topology:
        base_train += ["--topology", args.topology]
    if args.topologies:
        base_train += ["--topologies", args.topologies]

    base_eval: list[str] = [
        "--episodes",        str(args.episodes),
        "--device-cap",      str(args.device_cap),
        "--net-cap",         str(args.net_cap),
        "--position-bins",   str(args.position_bins),
        "--route-size-bins", str(args.route_size_bins),
        "--mag-bins",        str(args.mag_bins),
        "--max-place-steps", str(args.max_place_steps),
        "--max-route-steps", str(args.max_route_steps),
        "--max-steps",       str(args.max_steps),
        "--routing-mode",    "std_cell",
    ]
    if args.no_drc:
        base_eval.append("--no-drc")
    if args.no_route_eval:
        base_eval.append("--no-route")
    if args.topologies:
        base_eval += ["--topologies", args.topologies]
    elif args.topology:
        base_eval += ["--topology", args.topology]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"[ablation] preset={args.variants}  variants="
              f"{[v.name for v in variants]}")
        print(f"[ablation] {args.total_timesteps} train steps × "
              f"{args.episodes} eval episodes per variant")

    results: list[VariantResult] = []
    for v in variants:
        results.append(_run_variant(
            v,
            base_train_args=base_train,
            base_eval_args=base_eval,
            out_dir=out_dir,
            bc_init=args.bc_init,
            verbose=not args.quiet,
        ))

    if not args.quiet:
        _print_table(results)
    if args.out_csv:
        _write_csv(results, args.out_csv)
        if not args.quiet:
            print(f"\n[ablation] CSV → {args.out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
