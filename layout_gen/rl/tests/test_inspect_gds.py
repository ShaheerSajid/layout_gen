"""
layout_gen.rl.tests.test_inspect_gds — smoke test for the inspector CLI.

Generates an inverter via generate.main, then runs inspect_gds.main on
the resulting file and verifies the inspector reports both an NMOS
and a PMOS device with no missing-layer issues.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from layout_gen.rl.scripts import generate as generate_cli
from layout_gen.rl.scripts import inspect_gds as inspect_cli


def test_inspector_reports_nmos_and_pmos_after_generate(tmp_path: Path):
    out = tmp_path / "inv.gds"
    rc = generate_cli.main([
        "--topology", "inverter",
        "--out", str(out),
        "--cell-name", "inspector_smoke_cell",   # avoid global-name collision
        "--max-place-steps", "4",
        "--max-route-steps", "4",
        "--max-steps", "16",
        "--device-cap", "8",
        "--position-bins", "8",
        "--net-cap", "8",
        "--route-size-bins", "4",
        "--seed", "42",
        "--no-drc",         # tests use the no-op DRC for speed
        "--quiet",
    ])
    assert rc == 0, f"generate.main failed: {rc}"
    assert out.exists()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc2 = inspect_cli.main([str(out)])
    text = buf.getvalue()
    assert rc2 == 0
    # Both transistor types should appear in the device summary.
    assert "nmos" in text.lower()
    assert "pmos" in text.lower()
    # Layer summary should include the PMOS-only nwell.
    assert "nwell" in text.lower()


def test_inspector_strict_flag_returns_nonzero_on_missing_layer(tmp_path: Path):
    """Build a synthetic GDS that's missing PMOS poly, then verify
    --strict flags it as an error."""
    from layout_gen.synth.geo.state import LayoutState
    from layout_gen.pdk import load_pdk

    rules = load_pdk()
    state = LayoutState()
    # A PMOS-shaped cluster missing its poly gate.
    state.add(layer="diff",   x0=0.0, y0=0.0, x1=0.5, y1=0.3)
    state.add(layer="nwell",  x0=-0.1, y0=-0.1, x1=0.6, y1=0.4)
    state.add(layer="psdm",   x0=-0.05, y0=-0.05, x1=0.55, y1=0.35)
    state.add(layer="li1",    x0=0.05, y0=0.05, x1=0.15, y1=0.25)
    state.add(layer="licon1", x0=0.05, y0=0.10, x1=0.15, y1=0.20)

    out = tmp_path / "broken.gds"
    comp = state.to_component(rules, name="broken_pmos")
    comp.write_gds(str(out), with_metadata=False)

    rc = inspect_cli.main([str(out), "--strict"])
    assert rc != 0, "Expected --strict to fail on a PMOS missing poly"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
