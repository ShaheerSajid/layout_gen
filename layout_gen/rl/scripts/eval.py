"""
layout_gen.rl.scripts.eval — quantitative eval harness for trained policies.

Runs N rollouts of a checkpoint against one or more cell topologies and
reports aggregate quality metrics. The intended workflow is:

    train_ppo (or BC + train_ppo) → eval → compare numbers

Without this script every "did change X help?" question becomes a
visual GDS comparison. With it, we have:

  * **DRC-clean rate** — fraction of episodes ending with 0 violations
    (only meaningful with --no-drc disabled).
  * **Inspector pass rate** — fraction of final layouts where every
    device cluster has all its expected layers (i.e. inspect_gds
    --strict would exit 0). Reuses the same logic as the inspector CLI
    so the in-memory shortcut produces identical verdicts.
  * **Mean / p50 / p10 / p90 ep_rew_mean** — aggregate reward.
  * **Mean ep_len** — episode length (lower = policy more efficient).
  * **Mean electrical / connectivity / hpwl / alignment scores** — env
    metrics already exposed in info.

Per-topology breakdown when ``--topologies`` lists more than one cell.

Usage::

    .venv/bin/python -m layout_gen.rl.scripts.eval \\
        --topologies inverter,nand2,nor2 \\
        --checkpoint checkpoints/ppo_inv_full.zip \\
        --episodes 8 \\
        --no-drc

Drop ``--no-drc`` to score against real klayout (slow). Pass
``--out-json`` to dump the full metrics for downstream analysis.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

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
from layout_gen.rl.policy.sb3 import MaskableLayoutPolicy
from layout_gen.rl.scripts.generate import _NoOpDRC, _resolve_pitches
from layout_gen.rl.scripts.inspect_gds import (
    Poly, _classify_cluster, _cluster_devices,
)
from layout_gen.rl.topology import (
    TopologyEncoder, TopologyEncoderConfig, graph_from_template,
)


# ── Metrics container ───────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    topology:        str
    seed:            int
    ep_reward:       float
    ep_len:          int
    n_violations:    int
    n_polygons:      int
    electrical:      float
    connectivity:    float
    hpwl:            float
    alignment:       float
    n_devices_placed: int
    n_nets_routed:   int
    inspector_clean: bool
    n_missing_layer_clusters: int


@dataclass
class AggregateReport:
    n_episodes:      int
    drc_clean_rate:  float
    inspector_pass_rate: float
    ep_reward_mean:  float
    ep_reward_p10:   float
    ep_reward_p50:   float
    ep_reward_p90:   float
    ep_len_mean:     float
    n_violations_mean: float
    electrical_mean: float
    connectivity_mean: float
    hpwl_mean:       float
    alignment_mean:  float
    per_topology:    dict[str, dict] = field(default_factory=dict)


# ── Inspector shortcut (no temp GDS required) ───────────────────────────────

def _inspect_state_in_memory(state: LayoutState) -> tuple[bool, int]:
    """Run the same cluster-classification logic as `inspect_gds.main` on
    a live :class:`LayoutState`. Returns ``(passed_strict, n_missing_clusters)``."""
    polys = [
        Poly(layer=r.layer, raw_lt=(-1, -1),
             x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
        for r in state
    ]
    clusters = _cluster_devices(polys)
    if not clusters:
        return False, 0   # no devices = a definitive fail
    n_missing = 0
    for members in clusters:
        _kind, _present, missing = _classify_cluster(members)
        if missing:
            n_missing += 1
    return n_missing == 0, n_missing


# ── Per-episode rollout ─────────────────────────────────────────────────────

def _rollout_episode(env: LayoutEnv,
                     policy: MaskableLayoutPolicy,
                     *,
                     deterministic: bool,
                     max_total_steps: int,
                     forbid_kinds: frozenset[str],
                     ) -> tuple[float, dict, int]:
    """Returns (cumulative_reward, last_info, ep_len)."""
    obs, info = env.reset(options={"forbid_kinds": forbid_kinds})
    cum_reward = 0.0
    last_info = info
    ep_len = 0
    for step in range(max_total_steps):
        obs_b = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        masks = np.expand_dims(env.action_masks(), 0)
        with torch.no_grad():
            actions, _ = policy.predict(
                obs_b, action_masks=masks, deterministic=deterministic,
            )
        action = np.asarray(actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += float(reward)
        last_info = info
        ep_len = step + 1
        if terminated or truncated:
            break
    return cum_reward, last_info, ep_len


# ── Eval driver ─────────────────────────────────────────────────────────────

def evaluate(
    *,
    topologies:   list[str],
    checkpoint:   Path | None,
    episodes:     int,
    args,
) -> AggregateReport:
    rules = load_pdk()
    cache = TransistorCache(rules)
    poly_pitch_um, metal_pitches, metal_dirs = _resolve_pitches(args, rules)

    layout_cfg = LayoutPolicyConfig(
        poly_cap=args.poly_cap,
        viol_cap=args.viol_cap,
        target_cap=args.target_cap,
        mag_bins=args.mag_bins,
        use_topology=True,
        topology_dim=args.topology_dim,
        enable_place=True,
        couple_device_position=args.couple_device_position,
        enable_route=not args.no_route,
        device_cap=args.device_cap,
        x_bins=args.position_bins, y_bins=args.position_bins,
        net_cap=args.net_cap,
        route_x_bins=args.position_bins, route_y_bins=args.position_bins,
        route_w_bins=args.route_size_bins, route_h_bins=args.route_size_bins,
    )

    results: list[EpisodeResult] = []

    for topology_name in topologies:
        template = load_template(topology_name)
        cell_w = float(template.cell_dimensions.width  or 4.0)
        cell_h = float(template.cell_dimensions.height or 2.0)
        cell_params = {"_defaults": {"w_N": args.w_n, "w_P": args.w_p, "l": args.l}}
        graph = graph_from_template(template, cell_params=cell_params)

        enc = TopologyEncoder(TopologyEncoderConfig(
            d_token=args.topology_dim, n_layers=2,
            max_devices=max(args.device_cap, graph.n_devices),
            max_nets=max(graph.n_nets, args.net_cap),
        )).eval()
        with torch.no_grad():
            topo_global = enc.encode_graphs([graph]).global_embedding[0].cpu().numpy()
        topo_global = topo_global.astype(np.float32)

        # DRC
        if args.no_drc:
            drc_factory = lambda: _NoOpDRC()  # noqa: E731
        else:
            from layout_gen.drc import get_runner
            from layout_gen.rl.env.runner import CachedDRC
            runner = get_runner(rules)
            if runner is None:
                raise SystemExit(
                    "error: no DRC tool found. Install klayout/magic or "
                    "pass --no-drc."
                )
            drc_factory = lambda: CachedDRC(runner, rules,
                                              cell_name=topology_name)

        # LVS factory (optional): only built when --lvs is set and
        # magic+netgen are available. Each cell gets its own auto-
        # emitted SPICE reference netlist.
        lvs_factory = None
        if getattr(args, "lvs", False):
            from layout_gen.lvs           import get_runner as get_lvs_runner
            from layout_gen.rl.env.runner import CachedLVS
            from layout_gen.rl.env.spice_ref import write_spice_subckt
            _lvs_runner = get_lvs_runner(rules)
            if _lvs_runner is not None and _lvs_runner.is_available():
                _ref_dir = Path(args.out_json).parent if args.out_json else Path("out")
                _ref_path = _ref_dir / "lvs_refs" / f"{topology_name}.ref.spice"
                write_spice_subckt(graph, topology_name, _ref_path)
                lvs_factory = lambda: CachedLVS(
                    _lvs_runner, rules,
                    cell_name=topology_name, ref_netlist=_ref_path,
                )
            else:
                print(f"[warn] --lvs set but no usable backend for "
                      f"{topology_name}; skipping.", file=sys.stderr)

        def _make_env() -> LayoutEnv:
            return LayoutEnv(
                drc=drc_factory(),
                poly_cap=args.poly_cap, viol_cap=args.viol_cap,
                target_cap=args.target_cap, mag_bins=args.mag_bins,
                max_steps=args.max_steps,
                topology_global=topo_global,
                enable_place=True,
                topology_graph=graph, transistor_cache=cache,
                device_cap=max(args.device_cap, graph.n_devices),
                x_bins=args.position_bins, y_bins=args.position_bins,
                cell_width_um=cell_w, cell_height_um=cell_h,
                max_place_steps=args.max_place_steps,
                enable_route=not args.no_route,
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
                lvs=lvs_factory() if lvs_factory is not None else None,
                strict_row_alignment=getattr(args, "strict_row_alignment", False),
            )

        env = _make_env()

        # Policy: load from checkpoint, or fresh.
        if checkpoint is not None:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.wrappers import ActionMasker
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv

            wrapped = DummyVecEnv([
                lambda: Monitor(ActionMasker(_make_env(),
                                              lambda e: e.action_masks()))
            ])
            model = MaskablePPO.load(
                str(checkpoint), env=wrapped, device=args.device,
                custom_objects={"policy_kwargs": {"layout_config": layout_cfg}},
            )
            policy = model.policy
        else:
            policy = MaskableLayoutPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=lambda _: 3e-4,
                layout_config=layout_cfg,
            ).to(args.device)

        deterministic = (args.deterministic == "yes" or
                          (args.deterministic == "auto" and checkpoint is not None))
        forbid = frozenset({"delete_rect"}) if checkpoint is None else frozenset()

        # Run N episodes per topology.
        for ep in range(episodes):
            seed = args.seed_base + ep
            torch.manual_seed(seed)
            np.random.seed(seed)
            cum_r, info, ep_len = _rollout_episode(
                env, policy,
                deterministic=deterministic,
                max_total_steps=args.max_steps,
                forbid_kinds=forbid,
            )
            inspector_clean, n_missing = _inspect_state_in_memory(env.state)
            results.append(EpisodeResult(
                topology=topology_name,
                seed=seed,
                ep_reward=cum_r,
                ep_len=ep_len,
                n_violations=int(info.get("n_violations", 0)),
                n_polygons=int(info.get("n_polygons", 0)),
                electrical=float(info.get("electrical", 0.0)),
                connectivity=float(info.get("connectivity", 0.0)),
                hpwl=float(info.get("hpwl", 0.0)),
                alignment=float(info.get("alignment", 0.0)),
                n_devices_placed=int(info.get("n_devices_placed", 0)),
                n_nets_routed=int(info.get("n_nets_routed", 0)),
                inspector_clean=inspector_clean,
                n_missing_layer_clusters=n_missing,
            ))

    return _aggregate(results)


def _aggregate(results: list[EpisodeResult]) -> AggregateReport:
    n = len(results)
    if n == 0:
        return AggregateReport(
            n_episodes=0, drc_clean_rate=0.0, inspector_pass_rate=0.0,
            ep_reward_mean=0.0, ep_reward_p10=0.0,
            ep_reward_p50=0.0, ep_reward_p90=0.0, ep_len_mean=0.0,
            n_violations_mean=0.0, electrical_mean=0.0,
            connectivity_mean=0.0, hpwl_mean=0.0, alignment_mean=0.0,
        )
    rewards = sorted(r.ep_reward for r in results)
    def _q(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return rewards[idx]
    per_topo: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        per_topo[r.topology].append(r)

    return AggregateReport(
        n_episodes=n,
        drc_clean_rate=sum(1 for r in results if r.n_violations == 0) / n,
        inspector_pass_rate=sum(1 for r in results if r.inspector_clean) / n,
        ep_reward_mean=statistics.mean(r.ep_reward for r in results),
        ep_reward_p10=_q(0.10), ep_reward_p50=_q(0.50), ep_reward_p90=_q(0.90),
        ep_len_mean=statistics.mean(r.ep_len for r in results),
        n_violations_mean=statistics.mean(r.n_violations for r in results),
        electrical_mean=statistics.mean(r.electrical for r in results),
        connectivity_mean=statistics.mean(r.connectivity for r in results),
        hpwl_mean=statistics.mean(r.hpwl for r in results),
        alignment_mean=statistics.mean(r.alignment for r in results),
        per_topology={
            name: {
                "n_episodes":      len(rs),
                "drc_clean_rate":  sum(1 for r in rs if r.n_violations == 0) / len(rs),
                "inspector_pass_rate": sum(1 for r in rs if r.inspector_clean) / len(rs),
                "ep_reward_mean":  statistics.mean(r.ep_reward for r in rs),
                "ep_len_mean":     statistics.mean(r.ep_len for r in rs),
                "electrical_mean": statistics.mean(r.electrical for r in rs),
            }
            for name, rs in per_topo.items()
        },
    )


# ── Pretty-printer ──────────────────────────────────────────────────────────

def print_report(rep: AggregateReport) -> None:
    print(f"─── eval over {rep.n_episodes} episodes ─────────────────────────")
    print(f"  DRC-clean rate       : {rep.drc_clean_rate * 100:5.1f}%")
    print(f"  Inspector pass rate  : {rep.inspector_pass_rate * 100:5.1f}%")
    print(f"  ep_reward mean       : {rep.ep_reward_mean:+.3f}")
    print(f"  ep_reward p10/p50/p90: {rep.ep_reward_p10:+.3f} / "
          f"{rep.ep_reward_p50:+.3f} / {rep.ep_reward_p90:+.3f}")
    print(f"  ep_len mean          : {rep.ep_len_mean:.2f} steps")
    print(f"  n_violations mean    : {rep.n_violations_mean:.2f}")
    print(f"  electrical mean      : {rep.electrical_mean:.3f}")
    print(f"  connectivity mean    : {rep.connectivity_mean:.3f}")
    print(f"  alignment mean       : {rep.alignment_mean:.3f}")
    print(f"  hpwl mean            : {rep.hpwl_mean:.3f}")
    if len(rep.per_topology) > 1:
        print("─── per-topology breakdown ─────────────────────────────────")
        for name, m in rep.per_topology.items():
            print(f"  {name:14s}  n={m['n_episodes']:3d}  "
                  f"drc_clean={m['drc_clean_rate']*100:5.1f}%  "
                  f"inspect={m['inspector_pass_rate']*100:5.1f}%  "
                  f"ep_rew={m['ep_reward_mean']:+.2f}  "
                  f"electrical={m['electrical_mean']:.2f}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--topology",   default=None,
                   help="Single topology name (alternative to --topologies).")
    p.add_argument("--topologies", default=None,
                   help="Comma-separated list of topology names. "
                        "Alternative to --topology.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Optional MaskablePPO checkpoint. Without it, evaluates "
                        "an untrained policy as a baseline.")
    p.add_argument("--episodes",   type=int, default=4,
                   help="Episodes per topology.")
    p.add_argument("--out-json",   type=Path, default=None,
                   help="Optional path to dump the AggregateReport as JSON.")
    p.add_argument("--seed-base",  type=int, default=0)

    # Mirror generate.py / train_ppo.py knobs so the env can be built.
    p.add_argument("--device-cap",      type=int, default=16)
    p.add_argument("--position-bins",   type=int, default=16,
                   help="Must match training. 16 over a 4 µm cell "
                        "separates adjacent gate columns; 8 collides "
                        "nand2/nor2.")
    p.add_argument("--mag-bins",        type=int, default=8)
    p.add_argument("--poly-cap",        type=int, default=128)
    p.add_argument("--viol-cap",        type=int, default=32)
    p.add_argument("--target-cap",      type=int, default=128)
    p.add_argument("--net-cap",         type=int, default=8)
    p.add_argument("--route-size-bins", type=int, default=4)
    p.add_argument("--max-place-steps", type=int, default=4)
    p.add_argument("--max-route-steps", type=int, default=6)
    p.add_argument("--max-steps",       type=int, default=16)
    p.add_argument("--no-route",        action="store_true")
    p.add_argument("--w-n", type=float, default=0.5)
    p.add_argument("--w-p", type=float, default=0.5)
    p.add_argument("--l",   type=float, default=0.15)
    p.add_argument("--routing-mode", choices=["std_cell", "analog", "off"],
                   default="std_cell")
    p.add_argument("--poly-pitch-um", type=float, default=None)
    p.add_argument("--no-pitch-snap", action="store_true")
    p.add_argument("--topology-dim",  type=int, default=64)

    p.add_argument("--no-drc", action="store_true",
                   help="Use the no-op DRC stub. Default dispatches to "
                        "klayout/magic; eval should normally NOT use this.")
    p.add_argument("--lvs", action="store_true",
                   help="Enable the magic+netgen LVS truth signal during "
                        "rollouts. Auto-emits a SPICE reference netlist "
                        "per topology. Slow (~1s/distinct geometry) but "
                        "the only signal that verifies device connectivity.")
    p.add_argument("--strict-row-alignment", action="store_true",
                   help="Reject PLACE actions whose device type disagrees "
                        "with the row implied by y. Should match training.")
    p.add_argument("--couple-device-position", action="store_true",
                   help="Build the eval policy with autoregressive PLACE "
                        "coupling (RL_GUIDE §9.1 option C). MUST match the "
                        "flag the loaded checkpoint was trained with.")
    p.add_argument("--deterministic", choices=("auto", "yes", "no"),
                   default="auto")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    if args.topology and args.topologies:
        raise SystemExit("error: pass either --topology or --topologies, not both")
    if args.topologies:
        topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    elif args.topology:
        topologies = [args.topology]
    else:
        raise SystemExit("error: --topology or --topologies required")

    rep = evaluate(
        topologies=topologies,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        args=args,
    )
    print_report(rep)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        # Convert dataclass + nested dicts to a JSON-friendly form.
        from dataclasses import asdict
        args.out_json.write_text(json.dumps(asdict(rep), indent=2))
        print(f"[json] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
