"""
layout_gen.rl.scripts.train_ppo — CLI for the MaskablePPO trainer.

Two modes:

  * ``--synthetic`` — env uses a fake DRC checker on a tiny seed
    layout. No klayout required. Good for verifying the training
    loop runs.

  * ``--topology <name>`` — env loads a real cell topology YAML,
    runs the topology GNN to compute a ``topology_global``
    conditioning vector, and (with ``--real-drc``) dispatches to
    klayout/magic for the violation count. Optional
    ``--enable-place`` / ``--enable-route`` turn on the generative
    phases; without them the env starts in REPAIR phase from
    whatever ``--seed-state`` callable you provide (default: empty).

BC warm-start (``--bc-init``) loads a previously-trained
:class:`LayoutPolicy` checkpoint into the actor.

Tip: real-DRC training is bottlenecked by klayout invocations. Start
with small ``--total-timesteps`` and short episodes to validate the
loop, then scale up.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from layout_gen.pdk import load_pdk
from layout_gen.synth.geo.state import LayoutState
from layout_gen.synth.loader import load_template

from layout_gen.rl.env.action_space import (
    derive_metal_directions, derive_metal_pitches_um, derive_poly_pitch_um,
)
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.env.place_action import TransistorCache
from layout_gen.rl.policy.network import LayoutPolicyConfig
from layout_gen.rl.topology import (
    TopologyEncoder, TopologyEncoderConfig, graph_from_template,
)
from layout_gen.rl.training.ppo_train import PPOConfig, PPOTrainer


# ── Synthetic env factory ────────────────────────────────────────────────────

class _DirectFakeDRC:
    """CachedDRC-shaped facade over the fake spacing check."""

    def __init__(self, threshold_um: float = 0.20):
        self._threshold = threshold_um

    def run(self, state):
        from layout_gen.drc.base import DRCViolation
        out = []
        rects = state.rects
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                if a.layer != b.layer:
                    continue
                d = ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5
                if 0 < d < self._threshold:
                    out.append(DRCViolation(
                        rule=f"{a.layer}.spacing",
                        description=f"min spacing: {self._threshold} um",
                        layer=a.layer, x=(a.cx + b.cx) / 2,
                        y=(a.cy + b.cy) / 2, value=d,
                    ))
        return out

    def count(self, state) -> int:
        return len(self.run(state))

    def stats(self) -> dict:
        return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}

    def clear(self) -> None:
        pass


def _synth_state_factory(rng: random.Random):
    def _make() -> LayoutState:
        s = LayoutState()
        for k in range(6):
            x0 = 0.25 * k + rng.uniform(-0.04, 0.04)
            s.add(layer="met1", x0=x0, y0=0.0, x1=x0 + 0.10, y1=0.10)
        return s
    return _make


# ── Real (topology) env factory ──────────────────────────────────────────────

def _build_real_env_factory(args, env_seed: int):
    """Topology-aware env: parse the YAML, build a TopologyGraph, encode
    it once, then return a closure that constructs LayoutEnv on demand
    (one per vec-env worker)."""
    from layout_gen.rl.topology import (
        TopologyEncoder, TopologyEncoderConfig, graph_from_template,
    )

    rules = load_pdk()
    template = load_template(args.topology)
    cell_w = float(template.cell_dimensions.width  or 4.0)
    cell_h = float(template.cell_dimensions.height or 2.0)
    cell_params = {"_defaults": {"w_N": args.w_n, "w_P": args.w_p, "l": args.l}}
    graph = graph_from_template(template, cell_params=cell_params)

    enc = TopologyEncoder(TopologyEncoderConfig(
        d_token=args.topology_dim,
        n_layers=2,
        max_devices=max(args.device_cap, graph.n_devices),
        max_nets=max(graph.n_nets, args.net_cap),
    )).eval()
    with torch.no_grad():
        topo_global = enc.encode_graphs([graph]).global_embedding[0].cpu().numpy()
    topo_global = topo_global.astype(np.float32)

    if args.no_drc:
        print("[warn] --no-drc set; using fake spacing-only DRC. PPO "
              "will train against zero-violation lies for most steps.",
              file=sys.stderr)
        def _drc_factory():
            return _DirectFakeDRC(threshold_um=0.20)
    else:
        from layout_gen.drc import get_runner
        from layout_gen.rl.env.runner import CachedDRC
        runner = get_runner(rules)
        if runner is None:
            raise SystemExit(
                "error: no DRC tool found. Install klayout or magic, or "
                "set KLAYOUT_BIN / MAGIC_BIN. Pass --no-drc to bypass "
                "(not recommended)."
            )
        def _drc_factory():
            return CachedDRC(runner, rules, cell_name=args.topology)

    cache = TransistorCache(rules)
    poly_pitch_um, metal_pitches, metal_dirs = _resolve_pitches(args, rules)

    def _make():
        return LayoutEnv(
            drc=_drc_factory(),
            poly_cap=args.poly_cap,
            viol_cap=args.viol_cap,
            target_cap=args.target_cap,
            mag_bins=args.mag_bins,
            max_steps=args.max_steps,
            topology_global=topo_global,
            enable_place=args.enable_place,
            topology_graph=(graph
                             if (args.enable_place or args.enable_route)
                             else None),
            transistor_cache=(cache if args.enable_place else None),
            device_cap=max(args.device_cap, graph.n_devices),
            x_bins=args.position_bins, y_bins=args.position_bins,
            cell_width_um=cell_w, cell_height_um=cell_h,
            max_place_steps=args.max_place_steps,
            enable_route=args.enable_route,
            net_cap=max(args.net_cap, graph.n_nets),
            route_x_bins=args.position_bins,
            route_y_bins=args.position_bins,
            route_w_bins=args.route_size_bins,
            route_h_bins=args.route_size_bins,
            max_route_steps=args.max_route_steps,
            placement_directives=template.placement_directives,
            poly_pitch_um=poly_pitch_um,
            metal_pitch_um_per_layer=metal_pitches,
            metal_direction_per_layer=metal_dirs,
        )
    return _make, graph


def _resolve_pitches(args, rules):
    """Translate ``--routing-mode`` + ``--poly-pitch-um`` into the
    ``(poly_pitch_um, metal_pitch_um_per_layer)`` pair the env consumes.

    Modes
    -----
    * ``std_cell`` (default) — poly pitch ON, metal pitches ON. Snaps
      both placement and routing to PDK-derived track grids, giving
      the policy a maze-router-like action substrate.
    * ``analog`` — poly pitch ON (gates must be on a grid), metal
      pitches OFF (routes stay on the manufacturing grid). Matches
      typical analog layout practice.
    * ``off`` — both OFF; the policy can output any (mfg-grid) value.
    """
    mode = args.routing_mode
    if args.no_pitch_snap:
        return None, None, None

    if args.poly_pitch_um is not None:
        poly_pitch = float(args.poly_pitch_um) if args.poly_pitch_um > 0 else None
    else:
        poly_pitch = derive_poly_pitch_um(rules)

    if mode == "off":
        return None, None, None
    if mode == "analog":
        return poly_pitch, None, None
    # std_cell: snap routes too, and honour the PDK's preferred-direction
    # so a horizontal layer only quantises its y (track index) and a
    # vertical layer only quantises x.
    return (
        poly_pitch,
        derive_metal_pitches_um(rules) or None,
        derive_metal_directions(rules) or None,
    )


def _build_env_factory(args, env_seed: int):
    """Pick the right factory (or list of factories) based on CLI flags.

    Returns ``(factory_or_list, graphs)`` where ``graphs`` is None for
    synthetic mode, a single graph for ``--topology``, or a list of
    graphs (one per cell) for ``--topologies``.
    """
    if args.synthetic:
        rng = random.Random(env_seed)
        drc = _DirectFakeDRC(threshold_um=0.20)
        def _make():
            return LayoutEnv(
                drc=drc,
                poly_cap=args.poly_cap,
                viol_cap=args.viol_cap,
                target_cap=args.target_cap,
                mag_bins=args.mag_bins,
                max_steps=args.max_steps,
                default_state_factory=_synth_state_factory(rng),
            )
        return _make, None
    if args.topologies:
        return _build_multi_topology_factories(args, env_seed)
    if args.topology:
        return _build_real_env_factory(args, env_seed)
    raise SystemExit(
        "error: pass --synthetic, --topology <name>, or --topologies "
        "<name1,name2,...>"
    )


def _build_multi_topology_factories(args, env_seed: int):
    """Build one env factory per cell in ``args.topologies``.

    Each factory closes over its own (template, graph, topology_global,
    cell_w, cell_h, transistor_cache). The returned list is consumed by
    ``PPOTrainer`` which round-robins across vec-env workers.

    The action-space caps in ``args`` (device_cap, net_cap) are bumped
    upward to the max needed across all cells; the *original* values
    are honoured as a floor only.
    """
    cell_names = [n.strip() for n in args.topologies.split(",") if n.strip()]
    if not cell_names:
        raise SystemExit("error: --topologies must list at least one name")

    # Precompute everything per cell so we can size the policy + env once
    # at the union of caps before any LayoutEnv is constructed.
    rules = load_pdk()
    cache = TransistorCache(rules)
    poly_pitch_um, metal_pitches, metal_dirs = _resolve_pitches(args, rules)

    if args.no_drc:
        print("[warn] --no-drc set; using fake spacing-only DRC. PPO "
              "will train against zero-violation lies for most steps.",
              file=sys.stderr)
    else:
        from layout_gen.drc import get_runner
        runner = get_runner(rules)
        if runner is None:
            raise SystemExit(
                "error: no DRC tool found. Install klayout or magic, or "
                "set KLAYOUT_BIN / MAGIC_BIN. Pass --no-drc to bypass "
                "(not recommended)."
            )

    cell_specs = []
    for name in cell_names:
        template = load_template(name)
        cell_w = float(template.cell_dimensions.width  or 4.0)
        cell_h = float(template.cell_dimensions.height or 2.0)
        cell_params = {"_defaults": {"w_N": args.w_n, "w_P": args.w_p, "l": args.l}}
        graph = graph_from_template(template, cell_params=cell_params)
        cell_specs.append((name, template, graph, cell_w, cell_h))

    max_devices = max((g.n_devices for _, _, g, _, _ in cell_specs),
                      default=args.device_cap)
    max_nets    = max((g.n_nets    for _, _, g, _, _ in cell_specs),
                      default=args.net_cap)
    args.device_cap = max(args.device_cap, max_devices)
    args.net_cap    = max(args.net_cap,    max_nets)

    # One topology encoder shared across all cells (params unused once
    # we've extracted the constant per-cell embedding).
    enc = TopologyEncoder(TopologyEncoderConfig(
        d_token=args.topology_dim,
        n_layers=2,
        max_devices=args.device_cap,
        max_nets=args.net_cap,
    )).eval()

    factories = []
    graphs = []
    for (name, template, graph, cell_w, cell_h) in cell_specs:
        with torch.no_grad():
            topo_global = enc.encode_graphs([graph]).global_embedding[0]
        topo_global = topo_global.cpu().numpy().astype(np.float32)

        if args.no_drc:
            def _drc_factory(_n=name):
                return _DirectFakeDRC(threshold_um=0.20)
        else:
            from layout_gen.rl.env.runner import CachedDRC
            def _drc_factory(_n=name, _r=rules, _runner=runner):
                return CachedDRC(_runner, _r, cell_name=_n)

        # Bind the per-cell vars at definition time via default args so
        # the factory closure isn't fooled by the loop iteration.
        def _make(_g=graph, _t=template, _w=cell_w, _h=cell_h,
                  _topo=topo_global, _drc=_drc_factory) -> LayoutEnv:
            return LayoutEnv(
                drc=_drc(),
                poly_cap=args.poly_cap,
                viol_cap=args.viol_cap,
                target_cap=args.target_cap,
                mag_bins=args.mag_bins,
                max_steps=args.max_steps,
                topology_global=_topo,
                enable_place=args.enable_place,
                topology_graph=(_g
                                 if (args.enable_place or args.enable_route)
                                 else None),
                transistor_cache=(cache if args.enable_place else None),
                device_cap=args.device_cap,
                x_bins=args.position_bins, y_bins=args.position_bins,
                cell_width_um=_w, cell_height_um=_h,
                max_place_steps=args.max_place_steps,
                enable_route=args.enable_route,
                net_cap=args.net_cap,
                route_x_bins=args.position_bins,
                route_y_bins=args.position_bins,
                route_w_bins=args.route_size_bins,
                route_h_bins=args.route_size_bins,
                max_route_steps=args.max_route_steps,
                placement_directives=_t.placement_directives,
                poly_pitch_um=poly_pitch_um,
                metal_pitch_um_per_layer=metal_pitches,
                metal_direction_per_layer=metal_dirs,
            )
        factories.append(_make)
        graphs.append(graph)

    return factories, graphs


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    mode = p.add_argument_group("mode (pick one)")
    mode.add_argument("--synthetic", action="store_true",
                      help="Use the fake DRC checker on a tiny synthetic "
                           "seed (no klayout, no real cell).")
    mode.add_argument("--topology", default=None,
                      help="Cell template name (e.g. 'inverter') for "
                           "topology-aware training.")
    mode.add_argument("--topologies", default=None,
                      help="Comma-separated list of cell template names "
                           "(e.g. 'inverter,nand2,nor2') for multi-cell "
                           "training. Each cell gets its own vec-env "
                           "worker; n_envs is bumped to len(topologies) "
                           "if smaller. The topology GNN's conditioning "
                           "vector finally pulls its weight here.")

    p.add_argument("--bc-init", type=Path, default=None,
                   help="Path to a BC checkpoint to warm-start the actor.")
    p.add_argument("--ibrl-bc-init", type=Path, default=None,
                   help="Path to a BC checkpoint to use as the IBRL "
                        "distillation reference. When set, PPO's loss "
                        "adds β·KL(π_PPO || π_BC) with β decaying linearly "
                        "from --ibrl-beta-start to --ibrl-beta-end. Pairs "
                        "naturally with --bc-init (use the same path).")
    p.add_argument("--ibrl-beta-start", type=float, default=1.0,
                   help="Initial KL-to-BC weight at training start.")
    p.add_argument("--ibrl-beta-end",   type=float, default=0.0,
                   help="Final KL-to-BC weight at training end.")
    p.add_argument("--total-timesteps", type=int, default=20000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--poly-cap",   type=int, default=128)
    p.add_argument("--viol-cap",   type=int, default=32)
    p.add_argument("--target-cap", type=int, default=128)
    p.add_argument("--mag-bins",   type=int, default=8)

    # Real / topology mode
    p.add_argument("--no-drc", action="store_true",
                   help="OPT OUT of real DRC and use the fake spacing "
                        "checker. Default is to dispatch to the auto-"
                        "detected klayout/magic runner — fake DRC "
                        "trains the policy against zero-violation lies "
                        "and is a smoke-test convenience only.")
    p.add_argument("--enable-place", action="store_true",
                   help="Turn on the PLACE phase.")
    p.add_argument("--enable-route", action="store_true",
                   help="Turn on the ROUTE phase.")
    p.add_argument("--topology-dim", type=int, default=64)
    p.add_argument("--device-cap",   type=int, default=16)
    p.add_argument("--net-cap",      type=int, default=16)
    p.add_argument("--position-bins", type=int, default=8)
    p.add_argument("--route-size-bins", type=int, default=4)
    p.add_argument("--max-place-steps", type=int, default=4)
    p.add_argument("--max-route-steps", type=int, default=4)
    p.add_argument("--w-n", type=float, default=0.5)
    p.add_argument("--w-p", type=float, default=0.5)
    p.add_argument("--l",   type=float, default=0.15)

    # Pitch quantisation (track-aligned action space).
    p.add_argument(
        "--routing-mode", choices=["std_cell", "analog", "off"],
        default="std_cell",
        help="std_cell: snap PLACE x to poly pitch AND ROUTE x/y to "
             "per-layer metal pitch (track-aligned grid/maze router). "
             "analog: snap only PLACE x to poly pitch; routes stay on "
             "the mfg grid. off: no pitch snapping (legacy free-bin "
             "behaviour). Default std_cell.")
    p.add_argument(
        "--poly-pitch-um", type=float, default=None,
        help="Override the poly pitch in µm. Default: auto-detect from "
             "rules.poly['pitch_um'], else width_min + spacing_min.")
    p.add_argument(
        "--no-pitch-snap", action="store_true",
        help="Disable all pitch snapping (equivalent to --routing-mode off).")

    p.add_argument("--out", type=Path, default=Path("checkpoints/ppo.zip"))
    p.add_argument("--tb-log", type=Path, default=None,
                   help="TensorBoard log directory (optional).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)

    # Real DRC needs a real cell topology to run on.
    if not args.synthetic and not args.topology and not args.topologies:
        raise SystemExit(
            "error: pass --synthetic, --topology <name>, or "
            "--topologies <list>"
        )
    if args.topology and args.topologies:
        raise SystemExit("error: pass either --topology or --topologies, not both")

    # Build the env factory (or list) FIRST so we can read the
    # auto-bumped device_cap / net_cap before sizing the policy.
    env_factory, graphs = _build_env_factory(args, env_seed=args.seed)

    # Multi-topology: bump n_envs to at least len(factories) so every
    # cell sees at least one worker per rollout.
    if isinstance(env_factory, list) and args.n_envs < len(env_factory):
        if args.n_envs > 1:
            print(f"[multi] bumping n_envs from {args.n_envs} to "
                  f"{len(env_factory)} so every cell gets a worker")
        args.n_envs = len(env_factory)

    layout_cfg = LayoutPolicyConfig(
        poly_cap=args.poly_cap,
        viol_cap=args.viol_cap,
        target_cap=args.target_cap,
        mag_bins=args.mag_bins,
        use_topology=bool(args.topology or args.topologies),
        topology_dim=args.topology_dim,
        enable_place=args.enable_place,
        enable_route=args.enable_route,
        device_cap=args.device_cap,
        x_bins=args.position_bins, y_bins=args.position_bins,
        net_cap=args.net_cap,
        route_x_bins=args.position_bins, route_y_bins=args.position_bins,
        route_w_bins=args.route_size_bins, route_h_bins=args.route_size_bins,
    )
    # Build the PPOConfig AFTER env construction so n_envs reflects
    # the multi-cell bump above.
    ppo_cfg = PPOConfig(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    if isinstance(graphs, list):
        names = args.topologies.split(",")
        for n, g in zip(names, graphs):
            print(f"[topology] cell={n.strip()} devices={g.n_devices} "
                  f"nets={g.n_nets}  enable_place={args.enable_place} "
                  f"enable_route={args.enable_route}  "
                  f"drc={'fake' if args.no_drc else 'real'}")
        print(f"[multi] training across {len(graphs)} cells, "
              f"n_envs={args.n_envs}")
    elif graphs is not None:
        print(f"[topology] cell={args.topology} devices={graphs.n_devices} "
              f"nets={graphs.n_nets}  enable_place={args.enable_place} "
              f"enable_route={args.enable_route}  "
              f"drc={'fake' if args.no_drc else 'real'}")

    trainer = PPOTrainer(
        env_factory=env_factory,
        config=ppo_cfg,
        layout_config=layout_cfg,
        bc_init=args.bc_init,
        tensorboard_log=args.tb_log,
        ibrl_bc_checkpoint=args.ibrl_bc_init,
        ibrl_beta_start=args.ibrl_beta_start,
        ibrl_beta_end=args.ibrl_beta_end,
    )
    trainer.learn(total_timesteps=args.total_timesteps)
    trainer.save(args.out)
    print(f"[save] checkpoint -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
