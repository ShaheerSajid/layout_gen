"""
layout_gen.rl.tests.test_eval — eval harness smoke + sanity tests.

Verifies the script runs end-to-end on a tiny config and produces a
well-shaped AggregateReport, including the per-topology breakdown when
multiple topologies are passed. Uses an untrained policy + no-op DRC +
no-route to keep the test fast (a few seconds).
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from layout_gen.rl.scripts import eval as eval_cli
from layout_gen.rl.scripts.eval import AggregateReport


def _common_args() -> list[str]:
    return [
        "--episodes", "2",
        "--no-drc", "--no-route",
        "--device-cap", "8",
        "--net-cap", "8",
        "--position-bins", "8",
        "--route-size-bins", "4",
        "--mag-bins", "8",
        "--max-place-steps", "4",
        "--max-steps", "8",
        "--routing-mode", "off",
        "--seed-base", "0",
    ]


def test_eval_single_topology(tmp_path: Path):
    out_json = tmp_path / "report.json"
    rc = eval_cli.main([
        "--topology", "inverter",
        "--out-json", str(out_json),
        *_common_args(),
    ])
    assert rc == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    # Aggregate fields the harness must always produce.
    for key in ("n_episodes", "drc_clean_rate", "inspector_pass_rate",
                "ep_reward_mean", "ep_reward_p10", "ep_reward_p50",
                "ep_reward_p90", "ep_len_mean", "n_violations_mean",
                "electrical_mean", "connectivity_mean", "hpwl_mean",
                "alignment_mean", "per_topology"):
        assert key in data, f"missing aggregate field: {key}"
    assert data["n_episodes"] == 2
    assert "inverter" in data["per_topology"]


def test_eval_multi_topology_breakdown(tmp_path: Path):
    out_json = tmp_path / "multi.json"
    rc = eval_cli.main([
        "--topologies", "inverter,nand2",
        "--out-json", str(out_json),
        *_common_args(),
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert data["n_episodes"] == 4   # 2 episodes × 2 topologies
    assert set(data["per_topology"]) == {"inverter", "nand2"}
    for topo in ("inverter", "nand2"):
        sub = data["per_topology"][topo]
        assert sub["n_episodes"] == 2
        assert 0.0 <= sub["drc_clean_rate"] <= 1.0
        assert 0.0 <= sub["inspector_pass_rate"] <= 1.0


def test_eval_print_report_does_not_crash(tmp_path: Path):
    """The pretty-printer is the user-facing surface — make sure it
    handles the per-topology branch without crashing."""
    rep = AggregateReport(
        n_episodes=4, drc_clean_rate=0.5, inspector_pass_rate=0.75,
        ep_reward_mean=3.14, ep_reward_p10=1.0, ep_reward_p50=3.0,
        ep_reward_p90=5.0, ep_len_mean=8.0, n_violations_mean=2.5,
        electrical_mean=1.5, connectivity_mean=0.8,
        hpwl_mean=-2.0, alignment_mean=0.6,
        per_topology={
            "inverter": {"n_episodes": 2, "drc_clean_rate": 0.5,
                          "inspector_pass_rate": 1.0,
                          "ep_reward_mean": 4.2, "ep_len_mean": 8.0,
                          "electrical_mean": 2.0},
            "nand2": {"n_episodes": 2, "drc_clean_rate": 0.5,
                       "inspector_pass_rate": 0.5,
                       "ep_reward_mean": 2.1, "ep_len_mean": 8.0,
                       "electrical_mean": 1.0},
        },
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        eval_cli.print_report(rep)
    text = buf.getvalue()
    # Spot-checks on the printed output.
    assert "DRC-clean rate" in text
    assert "Inspector pass rate" in text
    assert "per-topology breakdown" in text
    assert "inverter" in text
    assert "nand2" in text


def test_eval_inspector_in_memory_matches_inspect_strict(tmp_path: Path):
    """`_inspect_state_in_memory` is the eval shortcut for the
    inspector's --strict verdict. Run it on a state with two complete
    transistors and confirm both clean."""
    from layout_gen.pdk import load_pdk
    from layout_gen.synth.geo.state import LayoutState
    from layout_gen.rl.env.place_action import TransistorCache, place_device_full
    from layout_gen.rl.topology.parser import DeviceNode
    from layout_gen.rl.scripts.eval import _inspect_state_in_memory

    rules = load_pdk()
    cache = TransistorCache(rules)
    s = LayoutState()
    place_device_full(
        s, DeviceNode("N", "nmos", "planar_mosfet", 0.5, 0.15, 0, False),
        x_um=0.0, y_um=0.0, orientation="R0", cache=cache,
    )
    place_device_full(
        s, DeviceNode("P", "pmos", "planar_mosfet", 0.5, 0.15, 0, False),
        x_um=2.0, y_um=0.0, orientation="R0", cache=cache,
    )
    clean, missing = _inspect_state_in_memory(s)
    assert clean is True
    assert missing == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
