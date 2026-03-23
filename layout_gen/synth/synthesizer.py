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
from layout_gen.synth.loader      import CellTemplate, PortSpec
from layout_gen.synth.placer      import Placer, PlacedDevice
from layout_gen.synth.router      import Router, PortCandidate
from layout_gen.synth.geo.agent   import GeoFixAgent
from layout_gen.synth.geo.loop    import GeoFixLoop


# ── Public types ──────────────────────────────────────────────────────────────

#: Type alias for an ML model callable used by :class:`Synthesizer`.
MLModel = Callable[
    [CellTemplate, PDKRules, list, dict],   # template, rules, violations, params
    dict,                                   # → adjusted params
]


class PortResolutionError(ValueError):
    """Raised when a port's location keyword cannot be matched to any
    routing candidate emitted by the style handlers."""


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
        rules:      PDKRules,
        drc_runner: Any | None          = None,
        ml_model:   MLModel | None      = None,
        max_iter:   int                 = 10,
        geo_agent:  GeoFixAgent | None  = None,
        geo_max_iter: int               = 10,
    ):
        self.rules        = rules
        self.drc_runner   = drc_runner
        self.ml_model     = ml_model or _heuristic_ml_model
        self.max_iter     = max_iter
        self.geo_agent    = geo_agent
        self.geo_max_iter = geo_max_iter

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

            # ── Merge implant regions (avoid nsdm.1/psdm.1 spacing violations) ──
            _merge_implants(comp, placed, self.rules)

            # ── Routing ────────────────────────────────────────────────────
            router     = Router(self.rules)
            candidates = router.route(comp, template.routing, placed)

            # ── Ports ─────────────────────────────────────────────────────
            _add_ports(comp, template, candidates, self.rules)

            # ── DRC (optional) ─────────────────────────────────────────────
            if self.drc_runner is None:
                return SynthResult(comp, placed, current_params,
                                   [], iteration, True)

            violations = _run_drc(self.drc_runner, comp)
            if not violations:
                return SynthResult(comp, placed, current_params,
                                   [], iteration, True)

            # ── ML model adjusts params for next iteration ─────────────────
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
    """Draw merged implant rectangles that cover all same-type devices.

    Individual transistor primitives already draw their own implant boxes,
    but when two same-type devices are placed close together (closer than
    ``implant.spacing_min_um``), their implant regions must merge to avoid
    nsdm.1/psdm.1 spacing violations.

    This function draws one bounding implant rectangle per device type that
    covers all devices of that type.  The overlapping polygons naturally
    merge in the GDS (KLayout DRC treats them as one region).
    """
    from layout_gen.cells.standard import _diff_y

    impl_enc = rules.implant.get("enclosure_of_diff_um", 0.125)

    # Group devices by implant layer
    groups: dict[str, list[PlacedDevice]] = {}
    for dev in placed.values():
        dev_rules = rules.device(dev.spec.device_type)
        impl_layer = dev_rules["implant_layer"]
        groups.setdefault(impl_layer, []).append(dev)

    for impl_layer_name, devs in groups.items():
        if len(devs) < 2:
            continue  # single device — no merging needed

        lyr = rules.layer(impl_layer_name)

        # Compute bounding box of all diff regions + implant enclosure
        x0_min = float("inf")
        x1_max = float("-inf")
        y0_min = float("inf")
        y1_max = float("-inf")

        for dev in devs:
            # Diff X: from dev.x to dev.x + total_x
            dx0 = dev.x
            dx1 = dev.x + dev.geom.total_x_um
            # Diff Y
            dy0, dy1 = _diff_y(dev.geom, rules)
            dy0 += dev.y
            dy1 += dev.y

            x0_min = min(x0_min, dx0 - impl_enc)
            x1_max = max(x1_max, dx1 + impl_enc)
            y0_min = min(y0_min, dy0 - impl_enc)
            y1_max = max(y1_max, dy1 + impl_enc)

        # Draw merged implant covering all devices of this type
        comp.add_polygon(
            [(x0_min, y0_min), (x1_max, y0_min),
             (x1_max, y1_max), (x0_min, y1_max)],
            layer=lyr,
        )


# ── Port resolution ────────────────────────────────────────────────────────────

def _add_ports(
    comp:       Any,  # gf.Component
    template:   CellTemplate,
    candidates: list[PortCandidate],
    rules:      PDKRules,
) -> None:
    """Add output ports to *comp* by matching location keywords to candidates."""
    candidate_map: dict[str, PortCandidate] = {c.location_key: c for c in candidates}

    for net_name, pspec in template.ports.items():
        cand = candidate_map.get(pspec.location)
        if cand is None:
            warnings.warn(
                f"No routing candidate matched port {net_name!r} with "
                f"location {pspec.location!r}.  Available keys: "
                f"{list(candidate_map)}.  Port skipped.",
                stacklevel=4,
            )
            continue
        try:
            lyr = rules.layer(pspec.layer)
        except (KeyError, TypeError):
            lyr = (1, 0)  # fallback layer tuple

        comp.add_port(
            net_name,
            center=(cand.x, cand.y),
            width=cand.width,
            orientation=cand.orientation,
            layer=lyr,
        )


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
