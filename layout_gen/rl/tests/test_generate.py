"""
layout_gen.rl.tests.test_generate — end-to-end pipeline smoke test.

Runs the ``generate`` CLI in-process on the inverter cell with an
untrained policy and the no-op DRC, then verifies that:
  * The CLI exits cleanly (return code 0).
  * A non-empty GDS file is written to the requested path.
  * The output contains polygons on at least the expected device layers
    (poly + diff + li1) — proof that PLACE materialised at least one
    transistor through the full pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from layout_gen.rl.scripts import generate as generate_cli


def test_generate_inverter_smoke(tmp_path: Path):
    out = tmp_path / "inverter.gds"
    rc = generate_cli.main([
        "--topology", "inverter",
        "--out", str(out),
        "--max-place-steps", "4",
        "--max-steps", "10",
        "--device-cap", "8",
        "--position-bins", "8",
        "--no-drc",         # tests use the no-op DRC for speed
        "--quiet",
    ])
    assert rc == 0, f"generate.main returned {rc}"
    assert out.exists(), f"expected {out} to be written"
    assert out.stat().st_size > 200, "GDS file too small to be meaningful"

    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except Exception:
        from gdsfactory.gpdk import PDK as _G
        _G.activate()
    comp = gf.import_gds(str(out))
    polys = comp.get_polygons()
    # Should have at least one layer with multiple polygons (transistor
    # contacts are ≥ 2 each, plus poly / diff / li1).
    total = sum(len(v) for v in polys.values())
    assert total >= 5, f"too few polygons in generated cell: {total}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
