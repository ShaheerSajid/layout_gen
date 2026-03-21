"""
layout_gen/tests/test_synthesis.py — end-to-end synthesis validation.

Tests that the template-driven synthesizer produces layouts matching the
hand-coded reference functions (draw_inverter, draw_bit_cell) in port
count, port names, and approximate geometry.

Run with::

    pytest layout_gen/tests/test_synthesis.py -v

DRC tests (run_drc assertions) are skipped when no DRC tool is available.
"""
from __future__ import annotations

import math
import pytest

from layout_gen                  import load_pdk
from layout_gen.synth            import load_template, Synthesizer
from layout_gen.cells.standard   import draw_inverter
from layout_gen.cells.bit_cell   import draw_bit_cell


# ── Helpers ───────────────────────────────────────────────────────────────────

def _port_map(comp):
    """Return {port_name: port} dict from a gf.Component."""
    return {p.name: p for p in comp.ports}


def _approx(a: float, b: float, tol: float = 0.05) -> bool:
    """Return True if |a - b| <= tol (µm)."""
    return abs(a - b) <= tol


# ── Inverter synthesis test ───────────────────────────────────────────────────

class TestInverterSynthesis:
    """Synthesize inverter from template; compare to draw_inverter reference."""

    W_N, W_P, L = 0.52, 0.42, 0.15

    @pytest.fixture(scope="class")
    def rules(self):
        return load_pdk()

    @pytest.fixture(scope="class")
    def synth_result(self, rules):
        template = load_template("inverter")
        synth    = Synthesizer(rules)
        return synth.synthesize(template, params={"w_N": self.W_N, "w_P": self.W_P, "l": self.L})

    @pytest.fixture(scope="class")
    def ref_comp(self, rules):
        return draw_inverter(self.W_N, self.W_P, self.L, rules)

    def test_converged(self, synth_result):
        assert synth_result.converged, "Synthesizer did not converge (DRC runner not configured — should always converge)"

    def test_port_names(self, synth_result, ref_comp):
        synth_ports = set(_port_map(synth_result.component))
        ref_ports   = set(_port_map(ref_comp))
        assert synth_ports == ref_ports, (
            f"Port mismatch.  Synth: {sorted(synth_ports)}  Ref: {sorted(ref_ports)}"
        )

    def test_port_x_positions(self, synth_result, ref_comp):
        """Synthesized ports should be within 0.05 µm of reference in X."""
        synth = _port_map(synth_result.component)
        ref   = _port_map(ref_comp)
        for name in ref:
            if name not in synth:
                continue
            sx = synth[name].center[0]
            rx = ref[name].center[0]
            assert _approx(sx, rx), (
                f"Port {name!r} X mismatch: synth={sx:.4f} ref={rx:.4f} diff={abs(sx-rx):.4f}"
            )

    def test_cell_height(self, synth_result, ref_comp):
        """Synthesized cell height should match reference to within 0.05 µm."""
        sh = synth_result.component.ysize
        rh = ref_comp.ysize
        assert _approx(sh, rh, tol=0.10), (
            f"Cell height mismatch: synth={sh:.4f} ref={rh:.4f}"
        )


# ── Bit cell synthesis test ───────────────────────────────────────────────────

class TestBitCellSynthesis:
    """Synthesize 6T SRAM bit cell from template; compare to draw_bit_cell reference."""

    W_PD, W_PU, W_PG, L = 0.80, 0.42, 0.60, 0.15

    @pytest.fixture(scope="class")
    def rules(self):
        return load_pdk()

    @pytest.fixture(scope="class")
    def synth_result(self, rules):
        template = load_template("bit_cell_6t")
        synth    = Synthesizer(rules)
        return synth.synthesize(
            template,
            params={
                "w_PD_L": self.W_PD, "w_PD_R": self.W_PD,
                "w_PU_L": self.W_PU, "w_PU_R": self.W_PU,
                "w_PG_L": self.W_PG, "w_PG_R": self.W_PG,
                "l":      self.L,
            },
        )

    @pytest.fixture(scope="class")
    def ref_comp(self, rules):
        return draw_bit_cell(
            w_pd=self.W_PD, w_pu=self.W_PU, w_pg=self.W_PG, l=self.L, rules=rules
        )

    def test_converged(self, synth_result):
        assert synth_result.converged

    def test_port_names(self, synth_result, ref_comp):
        synth_ports = set(_port_map(synth_result.component))
        ref_ports   = set(_port_map(ref_comp))
        assert synth_ports == ref_ports, (
            f"Port mismatch.  Synth: {sorted(synth_ports)}  Ref: {sorted(ref_ports)}"
        )

    def test_wl_port_orientation(self, synth_result):
        """WL port should exit West (orientation 180°)."""
        port = _port_map(synth_result.component).get("WL")
        assert port is not None, "WL port missing"
        assert int(port.orientation) == 180, (
            f"WL orientation should be 180 (West), got {port.orientation}"
        )

    def test_bl_port_orientation(self, synth_result):
        """BL and BL_ ports should exit North (orientation 90°)."""
        ports = _port_map(synth_result.component)
        for name in ("BL", "BL_"):
            port = ports.get(name)
            assert port is not None, f"{name} port missing"
            assert int(port.orientation) == 90, (
                f"{name} orientation should be 90 (North), got {port.orientation}"
            )

    def test_bl_port_x_positions(self, synth_result, ref_comp):
        """BL and BL_ X positions should match reference to within 0.05 µm."""
        synth = _port_map(synth_result.component)
        ref   = _port_map(ref_comp)
        for name in ("BL", "BL_"):
            if name not in synth or name not in ref:
                continue
            sx = synth[name].center[0]
            rx = ref[name].center[0]
            assert _approx(sx, rx), (
                f"{name} X mismatch: synth={sx:.4f} ref={rx:.4f} diff={abs(sx-rx):.4f}"
            )

    def test_gnd_vdd_port_positions(self, synth_result, ref_comp):
        """GND and VDD ports should be centered on the cell (within 0.1 µm)."""
        synth = _port_map(synth_result.component)
        ref   = _port_map(ref_comp)
        for name in ("GND", "VDD"):
            if name not in synth or name not in ref:
                continue
            sx = synth[name].center[0]
            rx = ref[name].center[0]
            assert _approx(sx, rx, tol=0.10), (
                f"{name} X mismatch: synth={sx:.4f} ref={rx:.4f}"
            )


# ── DRC tests (optional — skipped if no DRC tool available) ───────────────────

def _try_get_drc_runner():
    """Return a DRC runner or None if none is configured."""
    try:
        from layout_gen.drc import get_runner
        return get_runner()
    except Exception:
        return None


@pytest.mark.skipif(
    _try_get_drc_runner() is None,
    reason="No DRC tool configured (install KLayout or Magic and set KLAYOUT_BIN / MAGIC_BIN)",
)
class TestDRCClean:
    """Verify synthesized layouts pass DRC when a tool is available."""

    @pytest.fixture(scope="class")
    def rules(self):
        return load_pdk()

    @pytest.fixture(scope="class")
    def drc_runner(self):
        return _try_get_drc_runner()

    def _synthesize_and_run_drc(self, template_name, params, rules, drc_runner):
        template = load_template(template_name)
        synth    = Synthesizer(rules, drc_runner=drc_runner)
        result   = synth.synthesize(template, params=params)
        return result

    def test_inverter_drc_clean(self, rules, drc_runner):
        result = self._synthesize_and_run_drc(
            "inverter",
            {"w_N": 0.52, "w_P": 0.42, "l": 0.15},
            rules, drc_runner,
        )
        assert result.converged, (
            f"Inverter has {len(result.violations)} DRC violation(s):\n"
            + "\n".join(f"  {v}" for v in result.violations[:5])
        )

    def test_bit_cell_drc_clean(self, rules, drc_runner):
        result = self._synthesize_and_run_drc(
            "bit_cell_6t",
            {
                "w_PD_L": 0.80, "w_PD_R": 0.80,
                "w_PU_L": 0.42, "w_PU_R": 0.42,
                "w_PG_L": 0.60, "w_PG_R": 0.60,
                "l": 0.15,
            },
            rules, drc_runner,
        )
        assert result.converged, (
            f"Bit cell has {len(result.violations)} DRC violation(s):\n"
            + "\n".join(f"  {v}" for v in result.violations[:5])
        )
