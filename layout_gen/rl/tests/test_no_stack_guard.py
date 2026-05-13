"""
layout_gen.rl.tests.test_no_stack_guard — verifies the env's no-stacking
guard rejects placements that collide with an already-placed device's
origin.

Catches the multi-cell failure mode where the trained policy placed
two NMOSes at the same (x_bin, y_bin) bin and the old inspector hid
the bug because the merged cluster still had every expected layer.
"""
from __future__ import annotations

import pytest

from layout_gen.pdk import load_pdk
from layout_gen.synth.loader import load_template

from layout_gen.rl.env.action_space import REPAIR_KINDS
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.env.place_action import TransistorCache
from layout_gen.rl.topology import graph_from_template

import numpy as np


class _NoOpDRC:
    def run(self, s): return []
    def count(self, s): return 0
    def stats(self): return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}
    def clear(self): pass


def _nand2_env() -> LayoutEnv:
    rules = load_pdk()
    cache = TransistorCache(rules)
    g = graph_from_template(
        load_template("nand2"),
        cell_params={"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}},
    )
    return LayoutEnv(
        drc=_NoOpDRC(),
        poly_cap=64, viol_cap=8, target_cap=64, mag_bins=8,
        max_steps=10,
        enable_place=True,
        topology_graph=g, transistor_cache=cache,
        device_cap=8, x_bins=8, y_bins=8,
        cell_width_um=4.0, cell_height_um=2.0,
        max_place_steps=4,
    )


def _place_action(env: LayoutEnv, *, device_idx: int,
                  x_bin: int, y_bin: int) -> np.ndarray:
    """Build a PLACE action targeting (device_idx, x_bin, y_bin)."""
    raw = np.zeros(env.action_space.nvec.shape, dtype=np.int64)
    raw[0] = len(REPAIR_KINDS)   # place_device kind index
    raw[6] = device_idx
    raw[7] = x_bin
    raw[8] = y_bin
    raw[9] = 0                   # R0 orientation
    return raw


def test_no_stack_guard_rejects_second_device_at_same_bin():
    env = _nand2_env()
    env.reset()
    # Place device 0 (N_A) at (x_bin=2, y_bin=2).
    obs, _, _, _, info = env.step(_place_action(env, device_idx=0,
                                                  x_bin=2, y_bin=2))
    assert info["action"]["valid"] is True

    # Place device 1 (N_B) at the SAME (x_bin=2, y_bin=2) — must be rejected.
    obs, _, _, _, info = env.step(_place_action(env, device_idx=1,
                                                  x_bin=2, y_bin=2))
    assert info["action"]["valid"] is False, (
        "no-stacking guard should reject identical-bin placement"
    )
    assert info["n_devices_placed"] == 1


def test_no_stack_guard_allows_distinct_bins():
    env = _nand2_env()
    env.reset()
    # Distinct bins → both succeed.
    _, _, _, _, info1 = env.step(_place_action(env, device_idx=0,
                                                 x_bin=2, y_bin=2))
    _, _, _, _, info2 = env.step(_place_action(env, device_idx=1,
                                                 x_bin=4, y_bin=2))
    assert info1["action"]["valid"]
    assert info2["action"]["valid"]
    assert info2["n_devices_placed"] == 2


def test_no_stack_guard_allows_same_x_different_y():
    """Gate-aligned devices in different rows MUST still be allowed —
    that's the canonical inverter / nand pattern."""
    env = _nand2_env()
    env.reset()
    _, _, _, _, info1 = env.step(_place_action(env, device_idx=0,
                                                 x_bin=2, y_bin=2))
    # Same X but the PMOS row (y_bin=6) is far enough away in µm.
    _, _, _, _, info2 = env.step(_place_action(env, device_idx=2,
                                                 x_bin=2, y_bin=6))
    assert info1["action"]["valid"]
    assert info2["action"]["valid"], (
        "gate-aligned devices in different rows must remain placeable"
    )
    assert info2["n_devices_placed"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
