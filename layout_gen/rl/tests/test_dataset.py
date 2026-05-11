"""
layout_gen.rl.tests.test_dataset — unit tests for TrajectoryDataset.

Verifies the action encoder, state reconstruction, and end-to-end
trajectory loading using the synthetic miner (no klayout needed).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from layout_gen.repair.perturb import PerturbAction
from layout_gen.synth.geo.state import LayoutState

from layout_gen.rl.env.action_space import (
    ACTION_KINDS, EDGE_NAMES, magnitude_bins,
)
from layout_gen.rl.training.dataset import (
    TrajectoryDataset, encode_action_dict, _state_from_serialised,
    _action_from_dict, synthetic_violation_at_target,
)
from layout_gen.rl.training.synthetic import (
    SyntheticMineConfig, mine_synthetic_trajectories,
)


# ── Action encoding ──────────────────────────────────────────────────────────

def test_encode_shift_edge():
    rid_to_idx = {7: 3}
    action = PerturbAction(
        "shift_edge", target=7,
        params={"side": "right", "delta": 0.05},
    )
    labels, validity = encode_action_dict(action.to_dict(), rid_to_idx,
                                          mag_table=magnitude_bins(8))
    assert labels["kind"]   == ACTION_KINDS.index("shift_edge")
    assert labels["target"] == 3
    assert labels["edge"]   == EDGE_NAMES.index("right")
    assert labels["sign_y"] == 1   # positive delta = outward
    assert validity["edge"]
    assert validity["sign_y"]
    assert validity["mag"]
    assert not validity["sign_x"]


def test_encode_translate_signs():
    rid_to_idx = {2: 0}
    pos = encode_action_dict(
        {"kind": "translate", "target": 2,
         "params": {"dx": 0.04, "dy": -0.03}},
        rid_to_idx, mag_table=magnitude_bins(8))[0]
    assert pos["sign_x"] == 1
    assert pos["sign_y"] == 0


def test_encode_delete_only_kind_target_valid():
    rid_to_idx = {0: 0}
    labels, validity = encode_action_dict(
        {"kind": "delete_rect", "target": 0, "params": {}},
        rid_to_idx, mag_table=magnitude_bins(8))
    assert labels["kind"] == ACTION_KINDS.index("delete_rect")
    assert validity["kind"]
    assert validity["target"]
    assert not validity["edge"]
    assert not validity["sign_x"]
    assert not validity["sign_y"]
    assert not validity["mag"]


def test_encode_unresolvable_target_invalid():
    """When the target rid isn't in rid_to_idx, target validity is False."""
    labels, validity = encode_action_dict(
        {"kind": "translate", "target": 999,
         "params": {"dx": 0.01, "dy": 0.0}},
        rid_to_idx={0: 0}, mag_table=magnitude_bins(8),
    )
    assert not validity["target"]
    assert validity["kind"]


# ── State reconstruction ─────────────────────────────────────────────────────

def test_state_from_serialised_preserves_rids():
    rects = [
        {"rid": 5, "layer": "met1", "x0": 0.0, "y0": 0.0, "x1": 0.1, "y1": 0.1},
        {"rid": 11, "layer": "poly", "x0": 0.2, "y0": 0.0, "x1": 0.3, "y1": 0.1},
    ]
    state = _state_from_serialised(rects)
    assert 5 in state
    assert 11 in state
    assert state[5].layer == "met1"
    assert state[11].layer == "poly"


def test_action_from_dict_roundtrip():
    a = PerturbAction("shift_edge", target=4,
                      params={"side": "left", "delta": -0.03})
    a2 = _action_from_dict(a.to_dict())
    assert a2.kind == a.kind
    assert a2.target == a.target
    assert a2.params == a.params


# ── Synthetic violation source ───────────────────────────────────────────────

def test_synthetic_violation_at_target():
    s = LayoutState()
    s.add(layer="met1", x0=0.10, y0=0.20, x1=0.30, y1=0.40)
    rid = list(s)[0].rid
    viols = synthetic_violation_at_target(
        s, {"kind": "translate", "target": rid, "params": {}},
    )
    assert len(viols) == 1
    assert viols[0].layer == "met1"
    assert viols[0].x == pytest.approx(0.20)
    assert viols[0].y == pytest.approx(0.30)


# ── End-to-end loading ──────────────────────────────────────────────────────

def _seed_factory(rng: random.Random):
    """Seed centres are 0.25 µm apart; with the 0.20 µm fake-DRC threshold
    and the default 0.02–0.10 µm perturbation range, a translate/shift_edge
    by ≥ 0.06 µm reliably produces a spacing violation."""
    def _make() -> LayoutState:
        s = LayoutState()
        for k in range(4):
            x0 = 0.25 * k + rng.uniform(-0.005, 0.005)
            s.add(layer="met1", x0=x0, y0=0.0,
                  x1=x0 + 0.10, y1=0.10)
        return s
    return _make


def test_dataset_loads_synthetic_trajectories(tmp_path: Path):
    rng = random.Random(42)
    out = tmp_path / "trajs"
    counts = mine_synthetic_trajectories(
        state_factory=_seed_factory(rng),
        out_dir=out,
        config=SyntheticMineConfig(
            n_trajectories=24, depths=(1,),
            forbid_kinds=frozenset({
                "delete_rect", "shrink_rect", "grow_rect",
            }),
        ),
        rng=rng,
    )
    assert counts["kept"] > 0, f"No trajectories mined: {counts}"

    dataset = TrajectoryDataset(out)
    assert len(dataset) > 0

    sample = dataset[0]
    assert "obs" in sample and "action" in sample and "validity" in sample
    assert sample["obs"]["poly_feats"].ndim == 2
    assert sample["obs"]["poly_mask"].ndim == 1
    # The trajectory has at least one valid action — kind must be valid.
    assert bool(sample["validity"]["kind"].item())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
