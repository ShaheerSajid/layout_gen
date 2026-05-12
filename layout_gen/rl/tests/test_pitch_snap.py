"""
layout_gen.rl.tests.test_pitch_snap — pitch quantisation in ActionSpace.

Covers:
  * `_snap_to_pitch` static helper rounds to k*pitch + pitch/2 and
    clamps within [pitch/2, span - pitch/2].
  * `ActionSpace(poly_pitch_um=...)` snaps PLACE x_um but NOT y_um.
  * `ActionSpace(metal_pitch_um_per_layer={...})` snaps ROUTE x/y on
    the targeted layer; layers absent from the dict are not snapped.
  * `derive_poly_pitch_um` and `derive_metal_pitches_um` read the
    expected fields from a real PDK YAML (sky130A).
  * `LayoutEnv` forwards pitch params to its inner ActionSpace.
"""
from __future__ import annotations

import pytest

from layout_gen.pdk import load_pdk

from layout_gen.rl.env.action_space import (
    ActionSpace, REPAIR_KINDS,
    derive_metal_directions, derive_metal_pitches_um, derive_poly_pitch_um,
)
from layout_gen.rl.env.layout_env import LayoutEnv
from layout_gen.rl.env.place_action import TransistorCache
from layout_gen.rl.topology import graph_from_template
from layout_gen.synth.loader import load_template


# ── _snap_to_pitch ───────────────────────────────────────────────────────────

def test_snap_to_pitch_rounds_to_half_pitch_grid():
    # Grid lines: 0.23, 0.69, 1.15, 1.61, 2.07, 2.53, 2.99, 3.45, 3.91
    # (k*0.46 + 0.23 for k in 0..8)
    snap = ActionSpace._snap_to_pitch
    assert snap(0.20, 4.0, 0.46) == pytest.approx(0.23)
    assert snap(0.50, 4.0, 0.46) == pytest.approx(0.69)
    assert snap(1.10, 4.0, 0.46) == pytest.approx(1.15)
    assert snap(2.00, 4.0, 0.46) == pytest.approx(2.07)


def test_snap_to_pitch_clamps_below_half_pitch():
    snap = ActionSpace._snap_to_pitch
    # Anything < pitch/2 clamps up to pitch/2.
    assert snap(0.0,  4.0, 0.46) == pytest.approx(0.23)
    assert snap(-1.0, 4.0, 0.46) == pytest.approx(0.23)


def test_snap_to_pitch_clamps_above_span_minus_half():
    snap = ActionSpace._snap_to_pitch
    # 4.0 - 0.23 = 3.77 is the largest grid line that fits.
    assert snap(3.99, 4.0, 0.46) == pytest.approx(3.77)
    assert snap(5.00, 4.0, 0.46) == pytest.approx(3.77)


def test_snap_to_pitch_zero_pitch_returns_raw():
    snap = ActionSpace._snap_to_pitch
    assert snap(1.234, 4.0, 0.0) == pytest.approx(1.234)


# ── ActionSpace: PLACE x snapping (poly pitch) ───────────────────────────────

def _place_action(helper: ActionSpace, *, x_bin: int, y_bin: int) -> tuple:
    """Build a minimal PLACE-kind raw action and return decoded (x, y)."""
    n_repair = len(REPAIR_KINDS)
    raw = [n_repair, 0, 0, 0, 0, 0, 0, x_bin, y_bin, 0]
    act = helper.decode(raw, idx_to_rid={})
    assert act.is_place
    return act.x_um, act.y_um


def test_place_x_snapped_when_poly_pitch_set():
    helper = ActionSpace(
        target_cap=8, mag_bins=4, enable_place=True,
        device_cap=4, x_bins=8, y_bins=8,
        cell_width_um=4.0, cell_height_um=4.0,
        poly_pitch_um=0.46,
    )
    # x_bin=3 of 8 over 4.0 → raw centre = 3.5/8*4.0 = 1.75
    # nearest 0.46-pitch grid line: 1.61 (k=3: 3*0.46+0.23=1.61) or
    # 2.07 (k=4). |1.75-1.61|=0.14, |1.75-2.07|=0.32. Snap to 1.61.
    x_um, y_um = _place_action(helper, x_bin=3, y_bin=3)
    assert x_um == pytest.approx(1.61)
    # y is NOT snapped (poly pitch only applies to x).
    assert y_um == pytest.approx(1.75)


def test_place_x_not_snapped_when_poly_pitch_none():
    helper = ActionSpace(
        target_cap=8, mag_bins=4, enable_place=True,
        device_cap=4, x_bins=8, y_bins=8,
        cell_width_um=4.0, cell_height_um=4.0,
        # poly_pitch_um omitted → no snapping
    )
    x_um, y_um = _place_action(helper, x_bin=3, y_bin=3)
    assert x_um == pytest.approx(1.75)
    assert y_um == pytest.approx(1.75)


# ── ActionSpace: ROUTE x/y snapping (metal pitch) ────────────────────────────

def _route_action(helper: ActionSpace, *, layer_idx: int,
                   rxb: int, ryb: int) -> tuple:
    """Build a ROUTE-kind raw action and return (x_um, y_um, layer)."""
    n_repair = len(REPAIR_KINDS)
    # repair(6) + place(4) + route(6)
    raw = [n_repair + 1,            # kind = route_segment (after PLACE)
           0, 0, 0, 0, 0,
           0, 0, 0, 0,              # PLACE dims, unused
           0, layer_idx, rxb, ryb, 0, 0]
    act = helper.decode(raw, idx_to_rid={})
    assert act.is_route
    return act.route_x_um, act.route_y_um, act.route_layer


def test_route_xy_snapped_when_metal_pitch_set():
    """ROUTE_LAYERS = (li1, met1, met2, met3); index 1 = met1."""
    helper = ActionSpace(
        target_cap=8, mag_bins=4,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
        cell_width_um=4.0, cell_height_um=4.0,
        enable_route=True, net_cap=4,
        route_x_bins=8, route_y_bins=8, route_w_bins=4, route_h_bins=4,
        metal_pitch_um_per_layer={"met1": 0.34, "met2": 0.46},
    )
    # rxb=2 of 8 over 4.0 → raw centre = 2.5/8*4 = 1.25
    # met1 pitch 0.34 grid: 0.17, 0.51, 0.85, 1.19, 1.53, ... → snap to 1.19
    x_um, y_um, layer = _route_action(helper, layer_idx=1, rxb=2, ryb=2)
    assert layer == "met1"
    assert x_um == pytest.approx(1.19)
    assert y_um == pytest.approx(1.19)


def test_route_xy_not_snapped_for_unmapped_layer():
    helper = ActionSpace(
        target_cap=8, mag_bins=4,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
        cell_width_um=4.0, cell_height_um=4.0,
        enable_route=True, net_cap=4,
        route_x_bins=8, route_y_bins=8, route_w_bins=4, route_h_bins=4,
        # Only met2 in the dict. met1 routes should pass through raw.
        metal_pitch_um_per_layer={"met2": 0.46},
    )
    # layer_idx=1 → met1 (not in dict)
    x_um, y_um, layer = _route_action(helper, layer_idx=1, rxb=2, ryb=2)
    assert layer == "met1"
    assert x_um == pytest.approx(1.25)
    assert y_um == pytest.approx(1.25)


def test_route_horizontal_layer_snaps_only_y():
    """Horizontal layer (met1 in sky130 = power rails) — routes run
    along x, so only the y track-index should be quantised."""
    helper = ActionSpace(
        target_cap=8, mag_bins=4,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
        cell_width_um=4.0, cell_height_um=4.0,
        enable_route=True, net_cap=4,
        route_x_bins=8, route_y_bins=8, route_w_bins=4, route_h_bins=4,
        metal_pitch_um_per_layer={"met1": 0.34},
        metal_direction_per_layer={"met1": "horizontal"},
    )
    # rxb=ryb=2 → raw centre = 1.25 on both axes.
    # Horizontal met1: y snapped to 0.34 grid → 1.19; x stays raw at 1.25.
    x_um, y_um, layer = _route_action(helper, layer_idx=1, rxb=2, ryb=2)
    assert layer == "met1"
    assert x_um == pytest.approx(1.25)   # NOT snapped
    assert y_um == pytest.approx(1.19)   # snapped


def test_route_vertical_layer_snaps_only_x():
    """Vertical layer (met2 in sky130) — routes run along y; only x
    track-index should be quantised."""
    helper = ActionSpace(
        target_cap=8, mag_bins=4,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
        cell_width_um=4.0, cell_height_um=4.0,
        enable_route=True, net_cap=4,
        route_x_bins=8, route_y_bins=8, route_w_bins=4, route_h_bins=4,
        metal_pitch_um_per_layer={"met2": 0.46},
        metal_direction_per_layer={"met2": "vertical"},
    )
    # layer_idx=2 → met2; raw centre = 1.25 on both axes.
    # Vertical met2: x snapped to 0.46 grid → 1.15; y stays raw at 1.25.
    x_um, y_um, layer = _route_action(helper, layer_idx=2, rxb=2, ryb=2)
    assert layer == "met2"
    assert x_um == pytest.approx(1.15)   # snapped
    assert y_um == pytest.approx(1.25)   # NOT snapped


def test_route_empty_direction_snaps_both_axes():
    """Layer with no preferred direction (li1) — both x and y snap.
    Matches the original direction-agnostic behaviour."""
    helper = ActionSpace(
        target_cap=8, mag_bins=4,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
        cell_width_um=4.0, cell_height_um=4.0,
        enable_route=True, net_cap=4,
        route_x_bins=8, route_y_bins=8, route_w_bins=4, route_h_bins=4,
        metal_pitch_um_per_layer={"li1": 0.34},
        metal_direction_per_layer={"li1": ""},   # empty = both
    )
    # layer_idx=0 → li1; both axes snap to 0.34 grid → 1.19.
    x_um, y_um, layer = _route_action(helper, layer_idx=0, rxb=2, ryb=2)
    assert layer == "li1"
    assert x_um == pytest.approx(1.19)
    assert y_um == pytest.approx(1.19)


def test_derive_metal_directions_from_sky130():
    """sky130A.yaml declares met1=horizontal, met2=vertical, li1=""."""
    rules = load_pdk()
    dirs = derive_metal_directions(rules)
    assert dirs.get("met1") == "horizontal"
    assert dirs.get("met2") == "vertical"
    assert dirs.get("li1") == ""


def test_route_xy_no_snap_when_dict_empty():
    helper = ActionSpace(
        target_cap=8, mag_bins=4,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
        cell_width_um=4.0, cell_height_um=4.0,
        enable_route=True, net_cap=4,
        route_x_bins=8, route_y_bins=8, route_w_bins=4, route_h_bins=4,
        # metal_pitch_um_per_layer omitted
    )
    x_um, y_um, _ = _route_action(helper, layer_idx=1, rxb=2, ryb=2)
    assert x_um == pytest.approx(1.25)
    assert y_um == pytest.approx(1.25)


# ── PDK-derivation helpers ──────────────────────────────────────────────────

def test_derive_poly_pitch_prefers_yaml_pitch_um():
    """sky130A.yaml declares `poly.pitch_um: 0.46` (stdcell CPP); the
    helper must return that value rather than the DRC-minimum
    width_min + spacing_min = 0.36 µm fallback."""
    rules = load_pdk()
    p = derive_poly_pitch_um(rules)
    assert p == pytest.approx(0.46)


def test_derive_poly_pitch_falls_back_when_pitch_um_missing():
    """When the YAML omits pitch_um, the helper computes
    width_min + spacing_min (the DRC lower bound)."""
    class _Stub:
        poly = {"width_min_um": 0.15, "spacing_min_um": 0.21}
    assert derive_poly_pitch_um(_Stub()) == pytest.approx(0.36)


def test_derive_metal_pitches_includes_li1_met1_met2():
    rules = load_pdk()
    pitches = derive_metal_pitches_um(rules)
    # sky130A defines width/spacing for all three; rest may be absent.
    for layer in ("li1", "met1", "met2"):
        assert layer in pitches, (
            f"derive_metal_pitches_um missed {layer}: {sorted(pitches)}"
        )
        assert pitches[layer] > 0


# ── LayoutEnv forwarding ─────────────────────────────────────────────────────

def test_layout_env_forwards_pitch_args():
    rules = load_pdk()
    cache = TransistorCache(rules)
    g = graph_from_template(
        load_template("inverter"),
        cell_params={"_defaults": {"w_N": 0.5, "w_P": 0.5, "l": 0.15}},
    )

    class _NoOpDRC:
        def run(self, state): return []
        def count(self, state): return 0
        def stats(self): return {"hits": 0, "misses": 0, "size": 0, "capacity": 0}
        def clear(self): pass

    env = LayoutEnv(
        drc=_NoOpDRC(),
        poly_cap=64, viol_cap=8, target_cap=64, mag_bins=4,
        max_steps=8,
        enable_place=True,
        topology_graph=g, transistor_cache=cache,
        device_cap=8, x_bins=8, y_bins=8,
        cell_width_um=4.0, cell_height_um=4.0,
        max_place_steps=4,
        enable_route=True, net_cap=8,
        route_x_bins=8, route_y_bins=8,
        route_w_bins=4, route_h_bins=4,
        max_route_steps=4,
        poly_pitch_um=0.46,
        metal_pitch_um_per_layer={"met1": 0.34},
    )
    assert env._action_helper.poly_pitch_um == pytest.approx(0.46)
    assert env._action_helper.metal_pitch_um_per_layer == {"met1": 0.34}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
