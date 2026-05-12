"""
layout_gen.rl.scripts.extract_demos — bulk-extract BC demos from synth.

Runs the rule-based synthesizer on a list of cell templates, writes one
demo JSON per template (PLACE-action sequence) to ``--out``.

Usage::

    .venv/bin/python -m layout_gen.rl.scripts.extract_demos \\
        --templates inverter,nand2,nor2 \\
        --out demos/

Then plug those demos into the BC trainer via the new
``PlacementDemoDataset`` and pretrain the policy before PPO finetune.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from layout_gen.rl.training.demo_extract import extract_many


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--templates", default="inverter",
                   help="Comma-separated list of cell template names "
                        "(e.g. 'inverter,nand2,nor2'). Each must resolve "
                        "via layout_gen.synth.loader.load_template.")
    p.add_argument("--out", type=Path, default=Path("demos/"),
                   help="Output directory. One '<template>.demo.json' per "
                        "input template.")
    p.add_argument("--w-n", type=float, default=0.5)
    p.add_argument("--w-p", type=float, default=0.5)
    p.add_argument("--l",   type=float, default=0.15)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    names = [n.strip() for n in args.templates.split(",") if n.strip()]
    if not names:
        raise SystemExit("error: --templates must list at least one name")

    cell_params = {"w_N": args.w_n, "w_P": args.w_p, "l": args.l}
    paths = extract_many(names, args.out, cell_params=cell_params)

    if not args.quiet:
        for p_ in paths:
            print(f"[demo] wrote {p_}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
