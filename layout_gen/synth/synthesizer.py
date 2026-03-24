"""
layout_gen.synth.synthesizer — orchestrates template → GDS layout.

The :class:`Synthesizer` drives the full pipeline:

1. **Placer** — resolves floorplan constraints → device (x, y) positions
2. **Router** — applies routing style handlers → net polygons
3. **Ports** — resolves location keywords → port objects
4. **DRC** — validates the layout (optional)
5. **ML loop** — calls an ML model on violations, adjusts params, repeats

ML hooks
--------
Pass an ``ml_model`` callable to enable ML-guided parameter adjustment::

    def my_model(
        template:   CellTemplate,
        rules:      PDKRules,
        violations: list[DRCViolation],
        params:     dict,
    ) -> dict:
        # analyse violation geometry → propose new sizing/spacing params
        return new_params

    synth = Synthesizer(rules, drc_runner=runner, ml_model=my_model)

When no model is provided the built-in heuristic widens device widths by a
small margin on each failed iteration, which tends to increase S/D contact
region sizes and relax spacing constraints.
"""
from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

from layout_gen.pdk        import PDKRules
from layout_gen.transistor import draw_transistor
from layout_gen.synth.loader      import CellTemplate
from layout_gen.synth.placer      import Placer, PlacedDevice
from layout_gen.synth.router      import Router, PortCandidate
from layout_gen.synth.netlist     import build_net_graph
from layout_gen.synth.auto_router import AutoRouter
from layout_gen.synth.port_resolver import resolve_ports, generate_expose_specs
from layout_gen.synth.geo.agent   import GeoFixAgent
from layout_gen.synth.geo.loop    import GeoFixLoop
from layout_gen.synth.ml.fix_policy import DRCFixPredictor


# ── Public types ──────────────────────────────────────────────────────────────

#: Type alias for an ML model callable used by :class:`Synthesizer`.
MLModel = Callable[
    [CellTemplate, PDKRules, list, dict],   # template, rules, violations, params
    dict,                                   # → adjusted params
]


@dataclass
class SynthResult:
    """Result of one :meth:`Synthesizer.synthesize` call.

    Attributes
    ----------
    component :
        Final ``gf.Component`` (may not be DRC-clean if ``converged=False``).
    placed :
        Map of device name → :class:`~layout_gen.synth.placer.PlacedDevice`.
    params :
        The parameter dict used for the final iteration.
    violations :
        DRC violations from the last run (empty list if DRC was not run or
        the layout was clean).
    iterations :
        Number of synthesis attempts made.
    converged :
        ``True`` if DRC was not run or the layout passed DRC.
    """
    component:  Any                        # gf.Component
    placed:     dict[str, PlacedDevice]
    params:     dict
    violations: list = field(default_factory=list)
    iterations: int  = 1
    converged:  bool = True


# ── Synthesizer ────────────────────────────────────────────────────────────────

class Synthesizer:
    """Synthesizes a GDS layout from a :class:`~layout_gen.synth.loader.CellTemplate`.

    Parameters
    ----------
    rules :
        PDK rules.
    drc_runner :
        Optional :class:`~layout_gen.drc.base.DRCRunner`.  When provided the
        synthesizer runs DRC after each attempt and iterates on failure.
    ml_model :
        Optional callable ``(template, rules, violations, params) → params``.
        When ``None`` the built-in heuristic is used (widen device widths by a
        small per-iteration margin).
    max_iter :
        Maximum DRC-fix iterations (default 10).  Ignored when no DRC runner
        is configured.
    """

    def __init__(
        self,
        rules:          PDKRules,
        drc_runner:     Any | None              = None,
        ml_model:       MLModel | None          = None,
        fix_predictor:  DRCFixPredictor | None  = None,
        max_iter:       int                     = 10,
        geo_agent:      GeoFixAgent | None      = None,
        geo_max_iter:   int                     = 10,
    ):
        self.rules          = rules
        self.drc_runner     = drc_runner
        self.ml_model       = ml_model or _heuristic_ml_model
        self.fix_predictor  = fix_predictor
        self.max_iter       = max_iter
        self.geo_agent      = geo_agent
        self.geo_max_iter   = geo_max_iter

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(
        self,
        template: CellTemplate,
        params:   dict[str, Any] | None = None,
    ) -> SynthResult:
        """Synthesize a layout from *template*.

        Parameters
        ----------
        template :
            Cell topology template from
            :func:`~layout_gen.synth.loader.load_template`.
        params :
            Device sizing.  Keys: ``"w_<DevName>"`` (µm), ``"l"`` (gate
            length µm), ``"w"`` (default width for all devices).
            Example: ``{"w_N": 0.52, "w_P": 0.42, "l": 0.15}``.
            Falls back to PDK minimum sizes for missing keys.

        Returns
        -------
        SynthResult
        """
        _activate_pdk()
        current_params = dict(params or {})
        violations: list = []

        for iteration in range(1, self.max_iter + 1):
            # ── Placement ─────────────────────────────────────────────────────
            placer = Placer(self.rules, current_params)
            placed = placer.place(template)

            # ── Build component skeleton (add device refs) ─────────────────
            import gdsfactory as gf
            comp = gf.Component(name=_cell_name(template, iteration))
            for dev in placed.values():
                tc = draw_transistor(
                    dev.geom.w_um,
                    dev.geom.l_um,
                    dev.spec.device_type,
                    self.rules,
                    n_fingers=dev.geom.n_fingers,
                )
                ref = comp.add_ref(tc)
                ref.move((dev.x, dev.y))
                dev.component = tc

            # ── Merge implant/nwell regions ──
            _merge_implants(comp, placed, self.rules)
            if template.layout_mode == "stacked":
                _merge_nwells_stacked(comp, placed, self.rules)

            # ── Routing ────────────────────────────────────────────────────
            router = Router(self.rules)

            net_graph     = build_net_graph(template)
            auto_router   = AutoRouter(self.rules)
            routing_specs = auto_router.plan(net_graph, placed, template)
            # Add expose_terminal specs for ports with explicit terminals
            routing_specs.extend(
                generate_expose_specs(template, net_graph, placed)
            )
            candidates = router.route(comp, routing_specs, placed)
            resolve_ports(
                comp, template, net_graph,
                placed, candidates, self.rules,
            )

            # ── DRC (optional) ─────────────────────────────────────────────
            if self.drc_runner is None:
                return SynthResult(comp, placed, current_params,
                                   [], iteration, True)

            violations = _run_drc(self.drc_runner, comp)
            if not violations:
                return SynthResult(comp, placed, current_params,
                                   [], iteration, True)

            # ── ML-guided param adjustment for next iteration ────────────
            if self.fix_predictor is not None and self.fix_predictor.is_trained:
                delta = self.fix_predictor.predict(
                    violations, current_params, self.rules)
                new_params = {
                    k: current_params.get(k, 0) + delta.get(k, 0)
                    for k in set(current_params) | set(delta)
                }
                # Clamp to PDK minimums
                new_params = _clamp_params(new_params, self.rules)
            else:
                new_params = self.ml_model(
                    template, self.rules, violations, current_params
                )
            if new_params == current_params:
                # Model proposed no change — avoid spinning
                break
            current_params = new_params

        # ── Geometric fix fallback (Phase 2) ─────────────────────────────
        if violations and self.geo_agent is not None and self.drc_runner is not None:
            geo_loop = GeoFixLoop(self.geo_agent, self.drc_runner, self.rules)
            geo_result = geo_loop.run(comp, max_iter=self.geo_max_iter)
            if geo_result.converged:
                geo_comp = geo_result.state.to_component(
                    self.rules, name=_cell_name(template, 0))
                # Carry over ports from the original component
                for port in comp.ports:
                    try:
                        geo_comp.add_port(
                            port.name,
                            center=port.center,
                            width=port.width,
                            orientation=port.orientation,
                            layer=port.layer,
                        )
                    except Exception:
                        pass
                return SynthResult(geo_comp, placed, current_params,
                                   [], iteration + geo_result.iterations, True)
            violations = geo_result.violations

        # Return best result even if DRC is not clean
        return SynthResult(comp, placed, current_params,
                           violations, self.max_iter, False)


# ── Implant merging ────────────────────────────────────────────────────────────

def _merge_implants(
    comp:   Any,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> None:
    """Draw merged implant rectangles for devices that are close enough to
    cause implant spacing violations.

    In standard mode: one bounding implant per device type (original behaviour).
    In stacked mode: group by implant layer and merge clusters of devices whose
    individual implant regions are closer than ``implant.spacing_min_um``.
    """
    from layout_gen.cells.standard import _diff_y

    impl_enc = rules.implant.get("enclosure_of_diff_um", 0.125)
    impl_sp  = rules.implant.get("spacing_min_um", 0.38)

    # Collect per-device implant bounding boxes, grouped by implant layer
    layer_devboxes: dict[str, list[tuple[PlacedDevice, tuple]]] = {}
    for dev in placed.values():
        dev_rules = rules.device(dev.spec.device_type)
        impl_layer = dev_rules["implant_layer"]
        dy0, dy1 = _diff_y(dev.geom, rules)
        bbox = (
            dev.x - impl_enc,
            dev.x + dev.geom.total_x_um + impl_enc,
            dy0 + dev.y - impl_enc,
            dy1 + dev.y + impl_enc,
        )
        layer_devboxes.setdefault(impl_layer, []).append((dev, bbox))

    for impl_layer_name, devboxes in layer_devboxes.items():
        if len(devboxes) < 2:
            continue

        lyr = rules.layer(impl_layer_name)

        # Sort by Y bottom edge, then cluster devices whose implant boxes
        # are closer than impl_sp in Y (they need to merge).
        devboxes.sort(key=lambda db: db[1][2])  # sort by y0

        clusters: list[list[tuple]] = [[devboxes[0][1]]]
        for _, bbox in devboxes[1:]:
            prev_cluster = clusters[-1]
            prev_y1 = max(b[3] for b in prev_cluster)
            if bbox[2] - prev_y1 < impl_sp:
                prev_cluster.append(bbox)
            else:
                clusters.append([bbox])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            x0 = min(b[0] for b in cluster)
            x1 = max(b[1] for b in cluster)
            y0 = min(b[2] for b in cluster)
            y1 = max(b[3] for b in cluster)
            comp.add_polygon(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                layer=lyr,
            )


def _merge_nwells_stacked(
    comp:   Any,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> None:
    """Draw one merged nwell per cluster of PMOS devices in stacked layouts.

    Individual PMOS transistors draw their own nwell, but adjacent nwells
    closer than ``nwell.spacing_min_um`` (1.27 µm) violate nwell.2.  In a
    stacked layout the safest fix is to merge all PMOS nwells into one
    continuous region (or per-cluster when gaps are large enough).
    """
    from layout_gen.cells.standard import _diff_y

    nw_enc = rules.nwell.get("enclosure_of_pdiff_um", 0.18)
    nw_sp  = rules.nwell.get("spacing_min_um", 1.27)

    pmos_devs = [d for d in placed.values() if d.spec.device_type == "pmos"]
    if len(pmos_devs) < 2:
        return

    # Compute nwell bbox per device
    boxes: list[tuple[float, float, float, float]] = []
    for dev in pmos_devs:
        dy0, dy1 = _diff_y(dev.geom, rules)
        boxes.append((
            dev.x - nw_enc,
            dev.x + dev.geom.total_x_um + nw_enc,
            dy0 + dev.y - nw_enc,
            dy1 + dev.y + nw_enc,
        ))

    # Sort by Y bottom, cluster nwells that are closer than nw_sp
    boxes.sort(key=lambda b: b[2])
    clusters: list[list[tuple]] = [[boxes[0]]]
    for box in boxes[1:]:
        prev_y1 = max(b[3] for b in clusters[-1])
        if box[2] - prev_y1 < nw_sp:
            clusters[-1].append(box)
        else:
            clusters.append([box])

    lyr = rules.layer("nwell")
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        x0 = min(b[0] for b in cluster)
        x1 = max(b[1] for b in cluster)
        y0 = min(b[2] for b in cluster)
        y1 = max(b[3] for b in cluster)
        comp.add_polygon(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            layer=lyr,
        )


# ── Parameter clamping ─────────────────────────────────────────────────────────

def _clamp_params(params: dict, rules: PDKRules) -> dict:
    """Clamp synthesis params to PDK minimum values."""
    clamped = dict(params)
    poly_min = rules.poly.get("width_min_um", 0.15)
    diff_min = rules.diff.get("width_min_um", 0.15)
    if "l" in clamped:
        clamped["l"] = max(float(clamped["l"]), poly_min)
    if "w_N" in clamped:
        clamped["w_N"] = max(float(clamped["w_N"]), diff_min)
    if "w_P" in clamped:
        clamped["w_P"] = max(float(clamped["w_P"]), diff_min)
    if "gap_y" in clamped:
        clamped["gap_y"] = max(float(clamped["gap_y"]), 0.0)
    for k in ("finger_N", "finger_P"):
        if k in clamped:
            clamped[k] = max(1, int(round(float(clamped[k]))))
    return clamped


# ── DRC helper ────────────────────────────────────────────────────────────────

def _run_drc(runner: Any, comp: Any) -> list:
    """Write *comp* to a temp GDS and run DRC; return violations list."""
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        gds_path = Path(f.name)
    try:
        comp.write_gds(str(gds_path))
        return runner.run(gds_path, comp.name)
    except RuntimeError as exc:
        warnings.warn(f"DRC runner error: {exc}", stacklevel=3)
        return []
    finally:
        gds_path.unlink(missing_ok=True)


# ── ML heuristic ──────────────────────────────────────────────────────────────

_MARGIN_UM = 0.02   # width increase per iteration


def _heuristic_ml_model(
    template:   CellTemplate,
    rules:      PDKRules,
    violations: list,
    params:     dict,
) -> dict:
    """Default (non-ML) parameter adjuster: widen all device widths by a small margin.

    This increases S/D contact-region heights, which can relieve diff-enclosure
    and contact-spacing violations near the cell boundary.

    A real ML model would analyse violation geometry (rule name, x/y centroid,
    measured value) to make targeted adjustments — e.g. increase only the
    specific device whose drain contact is violating.
    """
    new_params = dict(params)
    changed = False
    for key in list(new_params):
        if key.startswith("w") and isinstance(new_params[key], (int, float)):
            new_params[key] = round(float(new_params[key]) + _MARGIN_UM, 4)
            changed = True
    if not changed:
        # No width keys in params — try adding a global width
        new_params["w"] = round(
            float(new_params.get("w", 0.52)) + _MARGIN_UM, 4
        )
    return new_params


# ── Utilities ──────────────────────────────────────────────────────────────────

_SYNTH_COUNTER: dict[str, int] = {}


def _cell_name(template: CellTemplate, iteration: int) -> str:
    base = f"synth_{template.name}"
    _SYNTH_COUNTER[base] = _SYNTH_COUNTER.get(base, 0) + 1
    n = _SYNTH_COUNTER[base]
    suffix = "" if n == 1 else f"${n}"
    return f"{base}{suffix}"


def _activate_pdk() -> None:
    import gdsfactory as gf
    try:
        gf.get_active_pdk()
    except ValueError:
        from gdsfactory.generic_tech import PDK as _GENERIC
        _GENERIC.activate()
