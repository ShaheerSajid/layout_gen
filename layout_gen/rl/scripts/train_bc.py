"""
layout_gen.rl.scripts.train_bc — CLI entry point for the BC trainer.

Run from the project root::

    .venv/bin/python -m layout_gen.rl.scripts.train_bc \\
        --trajectories layout_gen/repair/data/trajectories \\
        --epochs 5 --batch-size 64 \\
        --out checkpoints/bc.pt

If no trajectories are mined yet, pass ``--synthetic`` to bootstrap a
small synthetic corpus on the fly (uses a fake DRC checker; suitable
for smoke testing only).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from layout_gen.synth.geo.state import LayoutState

from layout_gen.rl.policy import LayoutPolicy, LayoutPolicyConfig
from layout_gen.rl.training import (
    BCTrainer, BCTrainerConfig, PlacementDemoDataset, TrajectoryDataset,
    mine_synthetic_trajectories,
)
from layout_gen.rl.training.synthetic import SyntheticMineConfig


def _build_synthetic_state(rng: random.Random) -> LayoutState:
    """Tiny PDK-agnostic seed: 6 met1 rects whose centres are 0.25 µm apart.

    Spacing is tuned so the default fake-DRC threshold (0.20 µm) and
    perturbation range (0.02–0.10 µm) reliably produce violations on
    a fraction of trajectories.
    """
    s = LayoutState()
    for k in range(6):
        x0 = 0.25 * k + rng.uniform(-0.005, 0.005)
        s.add(layer="met1", x0=x0, y0=0.0, x1=x0 + 0.10, y1=0.10)
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", type=Path, required=False,
                   help="Directory of mined trajectory JSONs. Required "
                        "unless --synthetic is set.")
    p.add_argument("--synthetic", action="store_true",
                   help="Mine a small synthetic corpus on the fly using a "
                        "fake DRC checker (no klayout needed).")
    p.add_argument("--synthetic-out", type=Path,
                   default=Path("layout_gen/rl/data/synthetic_trajectories"),
                   help="Where to write synthetic trajectories.")
    p.add_argument("--synthetic-n", type=int, default=128,
                   help="Number of synthetic trajectories to mine.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints/bc.pt"))
    p.add_argument("--max-trajectories", type=int, default=None,
                   help="Cap on number of trajectory files to load.")

    # PLACE-action demos (synth-derived BC corpus)
    p.add_argument("--demos", type=Path, default=None,
                   help="Directory or glob of *.demo.json files written "
                        "by extract_demos. Switches the dataset to "
                        "PlacementDemoDataset (PLACE-action BC) instead "
                        "of the perturb-trajectory dataset.")
    p.add_argument("--device-cap",      type=int, default=16)
    p.add_argument("--position-bins",   type=int, default=16,
                   help="Must match the PPO env that consumes the BC "
                        "checkpoint. 16 over a 4 µm cell separates "
                        "adjacent gate columns; 8 collides nand2/nor2.")
    p.add_argument("--enable-place",    action="store_true",
                   help="When training with --demos, build the policy "
                        "with enable_place=True so the PLACE heads exist.")
    p.add_argument("--enable-route",    action="store_true",
                   help="Build the policy with enable_route=True so the "
                        "ROUTE heads (and their flat-logit slots) exist; "
                        "ROUTE labels stay masked-off in the loss.")
    p.add_argument("--use-topology",    action="store_true",
                   help="Build the policy with use_topology=True so the "
                        "topology_global slot exists. Doesn't add the "
                        "vector to demos (zeros at train time); needed "
                        "for shape-compatibility with PPO checkpoints.")
    p.add_argument("--topology-dim",    type=int, default=64)
    p.add_argument("--mag-bins",        type=int, default=8)
    p.add_argument("--poly-cap",        type=int, default=128)
    p.add_argument("--viol-cap",        type=int, default=32)
    p.add_argument("--target-cap",      type=int, default=128)
    p.add_argument("--net-cap",         type=int, default=8)
    p.add_argument("--route-size-bins", type=int, default=4)

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)

    # ── PLACE demo path (Phase 5) ────────────────────────────────────────
    if args.demos is not None:
        demo_paths = sorted(Path(args.demos).glob("*.demo.json"))
        if not demo_paths:
            print(f"error: no *.demo.json under {args.demos}", file=sys.stderr)
            return 2
        print(f"[demos] loading {len(demo_paths)} demo file(s) from {args.demos}")
        dataset = PlacementDemoDataset(
            demo_paths,
            poly_cap=args.poly_cap,
            viol_cap=args.viol_cap,
            device_cap=args.device_cap,
            x_bins=args.position_bins,
            y_bins=args.position_bins,
            net_cap=args.net_cap,
            route_x_bins=args.position_bins,
            route_y_bins=args.position_bins,
            route_w_bins=args.route_size_bins,
            route_h_bins=args.route_size_bins,
        )
        print(f"[demos] {len(dataset)} (obs, action) samples (PLACE + ROUTE)")
        cfg = LayoutPolicyConfig(
            poly_cap=args.poly_cap,
            viol_cap=args.viol_cap,
            target_cap=args.target_cap,
            mag_bins=args.mag_bins,
            use_topology=args.use_topology,
            topology_dim=args.topology_dim,
            enable_place=args.enable_place,
            device_cap=args.device_cap,
            x_bins=args.position_bins, y_bins=args.position_bins,
            enable_route=args.enable_route,
            net_cap=args.net_cap,
            route_x_bins=args.position_bins,
            route_y_bins=args.position_bins,
            route_w_bins=args.route_size_bins,
            route_h_bins=args.route_size_bins,
        )
        policy = LayoutPolicy(cfg)
        trainer = BCTrainer(
            policy,
            BCTrainerConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                device=args.device,
            ),
        )
        metrics = trainer.fit(dataset)
        if metrics.train_loss:
            print(f"[done] final train loss = {metrics.train_loss[-1]:.4f}")
        if metrics.val_loss:
            print(f"[done] best val loss   = {metrics.best_val():.4f}")
        if metrics.accuracy:
            print(f"[done] last val accs   = {metrics.accuracy[-1]}")
        trainer.save(args.out)
        print(f"[save] checkpoint -> {args.out}")
        return 0

    if args.synthetic:
        out = args.synthetic_out
        cfg = SyntheticMineConfig(
            n_trajectories=args.synthetic_n,
            depths=(1,),
            forbid_kinds=frozenset({
                "delete_rect", "nudge_offgrid",
                "shrink_rect", "grow_rect",
            }),
            require_violations=True,
        )
        counts = mine_synthetic_trajectories(
            state_factory=lambda: _build_synthetic_state(rng),
            out_dir=out, config=cfg, rng=rng,
        )
        print(f"[synthetic] mined {counts}")
        traj_dir = out
    else:
        if args.trajectories is None:
            print("error: --trajectories or --synthetic required", file=sys.stderr)
            return 2
        traj_dir = args.trajectories

    print(f"[load] reading trajectories from {traj_dir}")
    dataset = TrajectoryDataset(
        traj_dir,
        max_trajectories=args.max_trajectories,
    )
    print(f"[load] {len(dataset)} samples")

    policy = LayoutPolicy(LayoutPolicyConfig())
    trainer = BCTrainer(
        policy,
        BCTrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            device=args.device,
        ),
    )
    metrics = trainer.fit(dataset)

    if metrics.train_loss:
        print(f"[done] final train loss = {metrics.train_loss[-1]:.4f}")
    if metrics.val_loss:
        print(f"[done] best val loss   = {metrics.best_val():.4f}")
    if metrics.accuracy:
        print(f"[done] last val accs   = {metrics.accuracy[-1]}")

    trainer.save(args.out)
    print(f"[save] checkpoint -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
