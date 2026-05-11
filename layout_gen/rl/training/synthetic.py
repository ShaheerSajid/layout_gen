"""
layout_gen.rl.training.synthetic — generate trajectories without klayout.

Wraps :func:`layout_gen.repair.perturb.generate_trajectory` with an
arbitrary ``state_factory`` and an arbitrary DRC checker. Writes JSONs
in the same format as :mod:`layout_gen.repair.mine_trajectories` so
:class:`TrajectoryDataset` consumes them transparently.

The default DRC checker is **fake** — it flags any pair of same-layer
rects whose centres are within a configurable threshold. This lets the
RL pipeline be smoke-tested end-to-end with zero external dependencies.
Real DRC mining still uses :mod:`layout_gen.repair.mine_trajectories`
once a klayout/magic backend is available.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from layout_gen.repair.perturb import (
    PerturbConfig, generate_trajectory, snapshot_state,
)
from layout_gen.synth.geo.state import LayoutState


# ── Default fake DRC ─────────────────────────────────────────────────────────

def fake_same_layer_spacing_check(
    state: LayoutState, *, threshold_um: float = 0.20,
) -> tuple[int, list[str]]:
    """Returns (n_violations, rule_names) for the fake spacing rule."""
    n = 0
    rules: list[str] = []
    rects = state.rects
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            if a.layer != b.layer:
                continue
            d = ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5
            if 0 < d < threshold_um:
                n += 1
                rules.append(f"{a.layer}.spacing")
    return n, rules


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SyntheticMineConfig:
    n_trajectories:    int = 64
    depths:            Sequence[int] = (1, 2)
    delta_min_um:      float = 0.02
    delta_max_um:      float = 0.10
    forbid_kinds:      frozenset[str] = frozenset({"delete_rect"})
    drc_threshold_um:  float = 0.20
    require_violations: bool = True
    seed_pdk:          str = "synthetic"
    seed_cell_prefix:  str = "synth_seed"
    seed_primitive:    str = "synthetic"
    seed_source:       str = "synthetic"


# ── Miner ────────────────────────────────────────────────────────────────────

def mine_synthetic_trajectories(
    state_factory: Callable[[], LayoutState],
    out_dir:       Path,
    *,
    config:        SyntheticMineConfig | None = None,
    drc_checker:   Callable[[LayoutState], tuple[int, list[str]]] | None = None,
    rng:           random.Random | None = None,
) -> dict[str, int]:
    """Mine *n_trajectories* trajectories using *state_factory* + a fake DRC.

    Each output file matches the schema produced by
    :mod:`layout_gen.repair.mine_trajectories` so the same
    :class:`TrajectoryDataset` reads both.
    """
    cfg = config or SyntheticMineConfig()
    rng = rng or random.Random(0)
    drc_checker = drc_checker or (
        lambda s: fake_same_layer_spacing_check(
            s, threshold_um=cfg.drc_threshold_um
        )
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdk_dir = out_dir / cfg.seed_pdk
    pdk_dir.mkdir(parents=True, exist_ok=True)

    counts = {"total": 0, "kept": 0, "no_viol": 0, "errors": 0}

    for i in range(cfg.n_trajectories):
        counts["total"] += 1
        try:
            state = state_factory()
            depth = rng.choice(list(cfg.depths))
            init_snap = snapshot_state(state)
            traj = generate_trajectory(
                state,
                n_steps=depth,
                config=PerturbConfig(
                    delta_min_um=cfg.delta_min_um,
                    delta_max_um=cfg.delta_max_um,
                    forbid_kinds=cfg.forbid_kinds,
                    seed=rng.randint(0, 2**31 - 1),
                ),
                rng=rng,
            )
        except Exception:
            counts["errors"] += 1
            continue

        n_viol, rule_list = drc_checker(state)
        if cfg.require_violations and n_viol <= 0:
            counts["no_viol"] += 1
            continue

        record = {
            "schema":          1,
            "seed_pdk":        cfg.seed_pdk,
            "seed_cell":       f"{cfg.seed_cell_prefix}_{i:04d}",
            "seed_primitive":  cfg.seed_primitive,
            "seed_source":     cfg.seed_source,
            "k":               depth,
            "rng_seed":        rng.randint(0, 2**31 - 1),
            "perturbed_state": [
                {"rid": r.rid, "layer": r.layer,
                 "x0": round(r.x0, 6), "y0": round(r.y0, 6),
                 "x1": round(r.x1, 6), "y1": round(r.y1, 6),
                 "net": r.net, "shape_type": r.shape_type,
                 "group_id": r.group_id}
                for r in state
            ],
            "forward_action_sequence": [a.to_dict() for a in traj.forward],
            "inverse_action_sequence": [a.to_dict() for a in traj.inverse],
            "n_violations":    n_viol,
            "violation_rules": rule_list,
        }
        out_path = pdk_dir / f"{cfg.seed_cell_prefix}_{i:04d}_k{depth}.json"
        out_path.write_text(json.dumps(record), encoding="utf-8")
        counts["kept"] += 1

    return counts


__all__ = [
    "SyntheticMineConfig",
    "mine_synthetic_trajectories",
    "fake_same_layer_spacing_check",
]
