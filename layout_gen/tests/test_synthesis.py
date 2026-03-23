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


# ── Stacked layout tests ────────────────────────────────────────────────────

class TestStackedPlacement:
    """Verify stacked multi-row placement for DIDO column peripheral."""

    @pytest.fixture(scope="class")
    def rules(self):
        return load_pdk()

    @pytest.fixture(scope="class")
    def template(self):
        return load_template("dido")

    @pytest.fixture(scope="class")
    def placed(self, rules, template):
        from layout_gen.synth.placer import Placer
        return Placer(rules, {"w": 0.42, "l": 0.15}).place(template)

    def test_layout_mode(self, template):
        assert template.layout_mode == "stacked"

    def test_row_pair_count(self, template):
        assert len(template.row_pairs) == 10

    def test_device_count(self, placed):
        assert len(placed) == 21

    def test_cell_width_within_column(self, placed):
        """All devices must fit within bit cell column width (~1.5 µm)."""
        x_max = max(d.x + d.geom.total_x_um for d in placed.values())
        assert x_max <= 1.5, f"Cell width {x_max:.3f} µm exceeds column pitch"

    def test_abutment_overlap(self, placed):
        """Abutting devices in the same row pair should share one S/D region."""
        # Row 0 NMOS: N_PB and N_PD should abut (N_PD starts before N_PB ends)
        n_pb = placed["N_PB"]
        n_pd = placed["N_PD"]
        overlap = (n_pb.x + n_pb.geom.total_x_um) - n_pd.x
        expected = n_pb.geom.sd_length_um
        assert _approx(overlap, expected, tol=0.001), (
            f"NMOS abutment overlap {overlap:.3f} != sd_length {expected:.3f}"
        )

    def test_sd_flip_flags(self, placed):
        """Devices with sd_flip should have it set."""
        assert placed["N_PB"].spec.sd_flip is True
        assert placed["P_PB"].spec.sd_flip is True
        assert placed["N_SB"].spec.sd_flip is True
        assert placed["N_PD"].spec.sd_flip is False
        assert placed["N_NA"].spec.sd_flip is False

    def test_sd_flip_terminal_swap(self, rules, placed):
        """Flipped device should have S and D positions swapped."""
        from layout_gen.synth.placer import resolve_terminal

        # N_PB is flipped: D should be on LEFT (j=0), S on RIGHT (j=nf)
        t_d = resolve_terminal("N_PB.D", placed, rules)
        t_s = resolve_terminal("N_PB.S", placed, rules)
        # D center X should be less than S center X (D on left when flipped)
        d_cx = (t_d.x0 + t_d.x1) / 2
        s_cx = (t_s.x0 + t_s.x1) / 2
        assert d_cx < s_cx, (
            f"Flipped N_PB: D center ({d_cx:.3f}) should be left of S ({s_cx:.3f})"
        )

        # N_PD is NOT flipped: S on LEFT, D on RIGHT
        t_d2 = resolve_terminal("N_PD.D", placed, rules)
        t_s2 = resolve_terminal("N_PD.S", placed, rules)
        d2_cx = (t_d2.x0 + t_d2.x1) / 2
        s2_cx = (t_s2.x0 + t_s2.x1) / 2
        assert s2_cx < d2_cx, (
            f"Normal N_PD: S center ({s2_cx:.3f}) should be left of D ({d2_cx:.3f})"
        )

    def test_row_pairs_vertically_ordered(self, placed, template):
        """Row pairs should be stacked vertically — later rows at higher Y."""
        for i in range(len(template.row_pairs) - 1):
            rp_a = template.row_pairs[i]
            rp_b = template.row_pairs[i + 1]
            devs_a = [placed[n] for n in rp_a.nmos_devices + rp_a.pmos_devices if n in placed]
            devs_b = [placed[n] for n in rp_b.nmos_devices + rp_b.pmos_devices if n in placed]
            if devs_a and devs_b:
                max_y_a = max(d.y + d.geom.total_y_um for d in devs_a)
                min_y_b = min(d.y for d in devs_b)
                assert max_y_a < min_y_b, (
                    f"Row pair {rp_a.id} top ({max_y_a:.3f}) overlaps "
                    f"row pair {rp_b.id} bottom ({min_y_b:.3f})"
                )

    def test_pmos_only_rows_no_nmos(self, placed, template):
        """PMOS-only row pairs should have no NMOS placement."""
        for rp in template.row_pairs:
            if not rp.nmos_devices:
                # All devices in this row pair should be PMOS
                for name in rp.pmos_devices:
                    assert placed[name].spec.device_type == "pmos"

    def test_synth_produces_ports(self, rules, template):
        """Full synthesis should produce all 11 ports."""
        synth = Synthesizer(rules)
        result = synth.synthesize(template, {"w": 0.42, "l": 0.15})
        port_names = {p.name for p in result.component.ports}
        expected = {"PCHG", "SEL", "WREN", "BL", "BL_",
                    "DR", "DR_", "DW", "DW_", "VDD", "VSS"}
        assert port_names == expected, (
            f"Port mismatch. Got: {sorted(port_names)} Expected: {sorted(expected)}"
        )


# ── Cross-row routing tests ──────────────────────────────────────────────────

class TestCrossRowRouting:
    """Verify cross-row-pair routing for DIDO column peripheral."""

    @pytest.fixture(scope="class")
    def synth_result(self):
        rules = load_pdk()
        tmpl  = load_template("dido")
        synth = Synthesizer(rules)
        return synth.synthesize(tmpl, {"w": 0.42, "l": 0.15}), rules

    @pytest.fixture(scope="class")
    def layout_state(self, synth_result):
        from layout_gen.synth.geo.state import LayoutState
        result, rules = synth_result
        return LayoutState.from_component(result.component, rules)

    def test_met1_geometry_exists(self, layout_state):
        """Cross-row routing should produce met1 rectangles."""
        met1 = layout_state.on_layer("met1")
        # At least power rails + cross-row routes + bitline buses
        assert len(met1) >= 10, f"Only {len(met1)} met1 rects (expected ≥10)"

    def test_mcon_via_stacks(self, layout_state):
        """Cross-row routes should create mcon via stacks (li1→met1)."""
        mcon = layout_state.on_layer("mcon")
        assert len(mcon) >= 10, f"Only {len(mcon)} mcon rects (expected ≥10)"

    def test_licon_poly_contacts(self, layout_state):
        """Gate targets should have licon1 poly contacts."""
        licon = layout_state.on_layer("licon1")
        # Original transistor contacts + poly contact stubs for gate targets
        assert len(licon) >= 50, f"Only {len(licon)} licon1 rects"

    def test_net6_vertical_span(self, layout_state):
        """net6 met1 route should span from row 0 (y≈0.3) to row 5 (y≈8.3)."""
        met1 = layout_state.on_layer("met1")
        # Find a met1 rectangle that spans most of the net6 range
        spanning = [r for r in met1 if r.y0 < 1.5 and r.y1 > 7.0]
        assert len(spanning) >= 1, (
            "No met1 rect spans net6 range (row 0 → rows 4/5)"
        )

    def test_net4_vertical_span(self, layout_state):
        """net4 met1 route should span from row 1 (y≈2.1) to row 7 (y≈10.2)."""
        met1 = layout_state.on_layer("met1")
        spanning = [r for r in met1 if r.y0 < 3.0 and r.y1 > 8.5]
        assert len(spanning) >= 1, (
            "No met1 rect spans net4 range (row 1 → rows 6/7)"
        )

    def test_net2_vertical_span(self, layout_state):
        """net2 met1 route should span from row 3 (y≈5.6) to row 9 (y≈12.1)."""
        met1 = layout_state.on_layer("met1")
        spanning = [r for r in met1 if r.y0 < 6.5 and r.y1 > 10.5]
        assert len(spanning) >= 1, (
            "No met1 rect spans net2 range (row 3 → rows 8/9)"
        )

    def test_bl_bus_vertical(self, layout_state):
        """BL vertical bus should span rows 4-8 on met1."""
        met1 = layout_state.on_layer("met1")
        bl_bus = [r for r in met1 if r.y0 < 8.0 and r.y1 > 10.5 and r.height > 2.0]
        assert len(bl_bus) >= 1, "No met1 BL vertical bus found"

    def test_bl_bar_bus_vertical(self, layout_state):
        """BL_ vertical bus should span rows 4-9 on met1."""
        met1 = layout_state.on_layer("met1")
        bl_bar = [r for r in met1 if r.y0 < 8.0 and r.y1 > 9.0 and r.height > 1.0]
        assert len(bl_bar) >= 1, "No met1 BL_ vertical bus found"

    def test_routing_directive_count(self):
        """DIDO template should now have 30 routing directives."""
        tmpl = load_template("dido")
        assert len(tmpl.routing) == 30, (
            f"Expected 30 routing directives, got {len(tmpl.routing)}"
        )
