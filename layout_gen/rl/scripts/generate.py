"""
layout_gen.rl.scripts.generate — end-to-end CLI: topology YAML → GDS.

Loads a cell topology, runs a (trained or untrained) MaskablePPO policy
through one PLACE → REPAIR episode, and writes the resulting layout to
GDS. This is the visible artifact of the RL pipeline — what a user
would invoke after training to actually generate a cell.

Usage::

    .venv/bin/python -m layout_gen.rl.scripts.generate \\
        --topology bit_cell_6t \\
        --pdk sky130A \\
        --checkpoint checkpoints/ppo.zip \\
        --out out/bit_cell.gds

Without ``--checkpoint`` the script uses an untrained policy — useful
for sanity-checking the pipeline (the resulting layout will be
geometrically correct per-device but won't satisfy DRC/LVS).

Without ``--real-drc`` the env uses a stub DRC that returns no
violations. Pass ``--real-drc`` to invoke the auto-detected klayout /
magic backend (gated on ``KLAYOUT_BIN`` / ``MAGIC_BIN`` being on
PATH or set as env vars).
"""
from __future__ import annotations

import argparse
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
from layout_gen.rl.policy.sb3 import MaskableLayoutPolicy
from layout_gen.rl.topology import (
    TopologyEncoder, TopologyEncoderConfig, graph_from_template,
)


# ── Stub DRC (when --real-drc is not requested) ──────────────────────────────

class _NoOpDRC:
    """CachedDRC-shaped facade that always reports zero violations.

    Useful for first-pass generation runs where you only want to see
    the policy place + emit geometry, without paying for klayout.
    """

    def run(self, state):
        return []

    def count(self, state) -> int:
        return 0

    def stats(self) -> dict:
        return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}

    def clear(self) -> None:
        pass


def _build_drc(args, rules):
    if args.no_drc:
        print("[warn] --no-drc set; using no-op DRC. Generated geometry "
              "will not be checked against any rules.", file=sys.stderr)
        return _NoOpDRC()
    from layout_gen.drc import get_runner
    from layout_gen.rl.env.runner import CachedDRC
    runner = get_runner(rules)
    if runner is None:
        raise SystemExit(
            "error: no DRC tool found. Install klayout or magic, or set "
            "KLAYOUT_BIN / MAGIC_BIN. Pass --no-drc to bypass (not "
            "recommended; fake DRC produces misleading runs)."
        )
    return CachedDRC(runner, rules, cell_name=args.cell_name or "synth")


# ── Cell-bbox helper ─────────────────────────────────────────────────────────

def _cell_dimensions(template) -> tuple[float, float]:
    """Pull (width, height) from CellTemplate.cell_dimensions, with sane
    defaults when the YAML leaves them at zero."""
    w = template.cell_dimensions.width or 4.0
    h = template.cell_dimensions.height or 2.0
    return float(w), float(h)


# ── Rollout ──────────────────────────────────────────────────────────────────

def rollout(
    env:         LayoutEnv,
    policy:      MaskableLayoutPolicy,
    *,
    deterministic: bool = True,
    max_total_steps: int = 64,
    verbose:     bool = True,
    forbid_kinds: frozenset[str] = frozenset(),
) -> LayoutState:
    """Run one episode and return the env's final :class:`LayoutState`.

    ``forbid_kinds`` is forwarded into the env via ``reset(options=...)``
    so the action mask suppresses those kinds for the whole episode.
    Useful for untrained rollouts where (e.g.) ``delete_rect`` would
    otherwise vandalise the PLACE phase's output.
    """
    obs, info = env.reset(options={"forbid_kinds": forbid_kinds})
    if verbose:
        print(f"[rollout] start  phase={info['phase']} polys={info['n_polygons']}")
    for step in range(max_total_steps):
        # MaskablePPO obs is a single env's dict; SB3's predict expects a
        # batched dict. Add a batch dim.
        obs_batched = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        masks = np.expand_dims(env.action_masks(), 0)
        with torch.no_grad():
            actions, _ = policy.predict(
                obs_batched, action_masks=masks, deterministic=deterministic,
            )
        action = np.asarray(actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            kind = info["action"]["kind"]
            valid = info["action"]["valid"]
            print(f"  step {step + 1:3d} phase={info['phase']:6s} "
                  f"kind={kind:14s} valid={valid} polys={info['n_polygons']:3d} "
                  f"viol={info['n_violations']:3d}")
        if terminated or truncated:
            if verbose:
                why = "DRC-clean" if terminated else "max_steps"
                print(f"[rollout] done   ({why}) at step {step + 1}")
            break
    return env.state


# ── GDS writing ──────────────────────────────────────────────────────────────

def write_gds(state: LayoutState, rules, *, out_path: Path,
              cell_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(state) == 0:
        # gdsfactory rejects empty Components; emit a single tiny rect
        # on a no-op layer so the file is still well-formed.
        state.add(layer="met1", x0=0.0, y0=0.0, x1=0.001, y1=0.001)
    comp = state.to_component(rules, name=cell_name)
    comp.write_gds(str(out_path), with_metadata=False)


# ── Pitch snapping resolution ────────────────────────────────────────────────

def _resolve_pitches(args, rules):
    """``--routing-mode`` + ``--poly-pitch-um`` → ``(poly_pitch_um,
    metal_pitch_um_per_layer, metal_direction_per_layer)``. See
    train_ppo.py for the semantics."""
    if args.no_pitch_snap:
        return None, None, None
    if args.poly_pitch_um is not None:
        poly_pitch = float(args.poly_pitch_um) if args.poly_pitch_um > 0 else None
    else:
        poly_pitch = derive_poly_pitch_um(rules)
    mode = args.routing_mode
    if mode == "off":
        return None, None, None
    if mode == "analog":
        return poly_pitch, None, None
    return (
        poly_pitch,
        derive_metal_pitches_um(rules) or None,
        derive_metal_directions(rules) or None,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--topology", required=True,
                   help="Cell template name (e.g. 'inverter') or path to a "
                        "topology YAML.")
    p.add_argument("--pdk", default="sky130A",
                   help="PDK YAML name under layout_gen/pdks/. Default sky130A.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Optional MaskablePPO checkpoint (.zip). When omitted "
                        "the policy is untrained — useful for pipeline "
                        "smoke-testing only.")
    p.add_argument("--out", type=Path, default=Path("out/generated.gds"))
    p.add_argument("--cell-name", default="generated_cell",
                   help="Top-cell name written to the GDS.")

    # Topology / cell sizing
    p.add_argument("--device-cap",  type=int, default=16)
    p.add_argument("--position-bins", type=int, default=16)
    p.add_argument("--mag-bins", type=int, default=16,
                   help="REPAIR-phase magnitude bin count. Must match the "
                        "value used at training time when loading a "
                        "checkpoint, otherwise MaskablePPO.load will reject "
                        "the action-space mismatch.")
    p.add_argument("--poly-cap",   type=int, default=128)
    p.add_argument("--viol-cap",   type=int, default=32)
    p.add_argument("--target-cap", type=int, default=128)
    p.add_argument("--max-place-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=32)

    # Routing
    p.add_argument("--no-route", action="store_true",
                   help="Skip the ROUTE phase entirely (devices ship "
                        "disconnected). Useful for debugging PLACE.")
    p.add_argument("--net-cap", type=int, default=16)
    p.add_argument("--route-size-bins", type=int, default=8)
    p.add_argument("--max-route-steps", type=int, default=8)

    # Default sizing fallbacks for devices whose YAML omits w / l
    p.add_argument("--w-n", type=float, default=0.5)
    p.add_argument("--w-p", type=float, default=0.5)
    p.add_argument("--l",   type=float, default=0.15)

    # Pitch quantisation (track-aligned action space).
    p.add_argument(
        "--routing-mode", choices=["std_cell", "analog", "off"],
        default="std_cell",
        help="std_cell: snap PLACE x to poly pitch AND ROUTE x/y to "
             "per-layer metal pitch. analog: snap only PLACE x to "
             "poly pitch. off: no snapping. Must match the training run.")
    p.add_argument(
        "--poly-pitch-um", type=float, default=None,
        help="Override the poly pitch in µm. Default: auto-detect from "
             "rules.poly['pitch_um'], else width_min + spacing_min.")
    p.add_argument(
        "--no-pitch-snap", action="store_true",
        help="Disable all pitch snapping.")

    # Topology encoder / policy sizing
    p.add_argument("--topology-dim", type=int, default=64)
    p.add_argument("--d-token", type=int, default=64)
    p.add_argument("--d-trunk", type=int, default=128)

    p.add_argument("--lvs-check", action="store_true",
                   help="After writing the GDS, run a one-shot "
                        "magic+netgen LVS check against an auto-"
                        "emitted SPICE reference netlist and report "
                        "clean/dirty + mismatch count.")
    p.add_argument("--no-drc", action="store_true",
                   help="OPT OUT of real DRC and use the no-op stub. "
                        "Default behaviour is to dispatch to the auto-"
                        "detected klayout/magic runner — fake DRC "
                        "produces misleading zero-violation runs and "
                        "should only be used for pipeline smoke testing.")
    p.add_argument("--deterministic", choices=("auto", "yes", "no"),
                   default="auto",
                   help="Sampling mode for the policy. 'auto' = "
                        "deterministic when a --checkpoint is loaded, "
                        "stochastic otherwise (an untrained policy with "
                        "deterministic argmax tends to stack every device "
                        "at the same bin, producing visually-overlapping "
                        "geometry).")
    p.add_argument("--forbid-delete", choices=("auto", "yes", "no"),
                   default="auto",
                   help="Forbid the REPAIR-phase delete_rect kind. "
                        "'auto' = forbid when no --checkpoint is loaded "
                        "(untrained policies tend to randomly delete the "
                        "PLACE/ROUTE work otherwise).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--quiet",  action="store_true")
    args = p.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rules = load_pdk()    # default PDK YAML (sky130A); --pdk override TBD

    # ── Topology ──────────────────────────────────────────────────────────
    template = load_template(args.topology)
    cell_w, cell_h = _cell_dimensions(template)
    cell_params = {"_defaults": {"w_N": args.w_n, "w_P": args.w_p, "l": args.l}}
    graph = graph_from_template(template, cell_params=cell_params)
    if not args.quiet:
        print(f"[topology] cell={template.name} devices={graph.n_devices} "
              f"nets={graph.n_nets} bbox={cell_w}x{cell_h}um")

    # ── Topology encoder ─────────────────────────────────────────────────
    enc_cfg = TopologyEncoderConfig(
        d_token=args.topology_dim,
        n_layers=2,
        max_devices=max(args.device_cap, graph.n_devices),
        max_nets=max(graph.n_nets, 16),
    )
    encoder = TopologyEncoder(enc_cfg).eval()
    with torch.no_grad():
        topo_out = encoder.encode_graphs([graph])
    topology_global = topo_out.global_embedding[0].cpu().numpy().astype(np.float32)

    # ── DRC + transistor cache ───────────────────────────────────────────
    drc = _build_drc(args, rules)
    cache = TransistorCache(rules)

    enable_route = not args.no_route
    poly_pitch_um, metal_pitches, metal_dirs = _resolve_pitches(args, rules)

    # ── Env ──────────────────────────────────────────────────────────────
    env = LayoutEnv(
        drc=drc,
        poly_cap=args.poly_cap, viol_cap=args.viol_cap,
        target_cap=args.target_cap, mag_bins=args.mag_bins,
        max_steps=args.max_steps,
        enable_place=True,
        topology_graph=graph, transistor_cache=cache,
        device_cap=args.device_cap,
        x_bins=args.position_bins, y_bins=args.position_bins,
        cell_width_um=cell_w, cell_height_um=cell_h,
        max_place_steps=args.max_place_steps,
        topology_global=topology_global,
        enable_route=enable_route,
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

    # ── Policy ───────────────────────────────────────────────────────────
    layout_cfg = LayoutPolicyConfig(
        poly_cap=args.poly_cap, viol_cap=args.viol_cap,
        target_cap=args.target_cap, mag_bins=args.mag_bins,
        d_token=args.d_token, d_trunk=args.d_trunk,
        n_layers=2, n_heads=4, dim_ff=128,
        use_topology=True, topology_dim=args.topology_dim,
        enable_place=True, device_cap=args.device_cap,
        x_bins=args.position_bins, y_bins=args.position_bins,
        enable_route=enable_route,
        net_cap=max(args.net_cap, graph.n_nets),
        route_x_bins=args.position_bins, route_y_bins=args.position_bins,
        route_w_bins=args.route_size_bins, route_h_bins=args.route_size_bins,
    )

    if args.checkpoint is not None:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        def _action_mask_fn(env_inner):
            return env_inner.action_masks()

        # MaskablePPO.load wants an env to bind to. Wrap our env once.
        wrapped = DummyVecEnv([
            lambda: Monitor(ActionMasker(env, _action_mask_fn))
        ])
        model = MaskablePPO.load(
            str(args.checkpoint), env=wrapped, device=args.device,
            custom_objects={"policy_kwargs": {"layout_config": layout_cfg}},
        )
        policy = model.policy
        if not args.quiet:
            print(f"[policy] loaded {args.checkpoint}")
    else:
        from gymnasium import spaces
        # Build a fresh, untrained policy that matches the env's spaces.
        policy = MaskableLayoutPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 3e-4,
            layout_config=layout_cfg,
        ).to(args.device)
        if not args.quiet:
            print("[policy] using fresh untrained policy "
                  "(pass --checkpoint to load weights)")

    if args.deterministic == "yes":
        det = True
    elif args.deterministic == "no":
        det = False
    else:  # "auto"
        det = args.checkpoint is not None

    if args.forbid_delete == "yes":
        forbid = frozenset({"delete_rect"})
    elif args.forbid_delete == "no":
        forbid = frozenset()
    else:  # "auto"
        forbid = (frozenset({"delete_rect"})
                   if args.checkpoint is None else frozenset())
    if forbid and not args.quiet:
        print(f"[forbid] {sorted(forbid)} — protects PLACE/ROUTE work "
              "from an untrained REPAIR head")

    final_state = rollout(env, policy,
                           deterministic=det,
                           max_total_steps=args.max_steps,
                           verbose=not args.quiet,
                           forbid_kinds=forbid)

    write_gds(final_state, rules,
              out_path=args.out, cell_name=args.cell_name)
    if not args.quiet:
        print(f"[done] wrote {args.out}  ({len(final_state)} polygons)")

    if args.lvs_check:
        _run_lvs_check(args, rules, graph)
    return 0


def _run_lvs_check(args, rules, graph) -> None:
    """One-shot magic+netgen LVS check on the written GDS. Auto-emits a
    SPICE reference netlist from the topology graph and reports the
    verdict + mismatch count. Failures don't raise — they print and
    return so the user can still inspect the GDS."""
    from layout_gen.lvs           import get_runner as get_lvs_runner
    from layout_gen.rl.env.spice_ref import write_spice_subckt

    lvs_runner = get_lvs_runner(rules)
    if lvs_runner is None or not lvs_runner.is_available():
        print("[lvs] no usable LVS backend (need magic + netgen); skipping.",
              file=sys.stderr)
        return

    ref_path = args.out.with_suffix(".ref.spice")
    write_spice_subckt(graph, args.cell_name, ref_path)
    try:
        result = lvs_runner.run(args.out, ref_path, args.cell_name)
    except Exception as exc:
        print(f"[lvs] runner crashed: {exc}", file=sys.stderr)
        return

    if result.clean:
        print(f"[lvs] CLEAN  ref={ref_path.name}")
    else:
        n_mm = len(result.mismatches)
        print(f"[lvs] DIRTY  {n_mm} mismatch(es) vs {ref_path.name}",
              file=sys.stderr)
        for mm in result.mismatches[:10]:
            print(f"        {mm}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
