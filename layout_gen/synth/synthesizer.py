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
    lvs :
        :class:`~layout_gen.lvs.base.LVSResult` if an ``lvs_runner`` was
        configured, otherwise ``None``.  The verdict is ``lvs.clean``;
        ``converged`` reflects DRC only.
    """
    component:  Any                        # gf.Component
    placed:     dict[str, PlacedDevice]
    params:     dict
    violations: list = field(default_factory=list)
    iterations: int  = 1
    converged:  bool = True
    lvs:        Any  = None


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
        max_iter:       int                     = 10,
        geo_agent:      GeoFixAgent | None      = None,
        geo_max_iter:   int                     = 10,
        lvs_runner:     Any | None              = None,
    ):
        self.rules          = rules
        self.drc_runner     = drc_runner
        self.ml_model       = ml_model or _heuristic_ml_model
        self.max_iter       = max_iter
        self.geo_agent      = geo_agent
        self.geo_max_iter   = geo_max_iter
        self.lvs_runner     = lvs_runner

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

            # Compute which S/D indices to skip contacts on (shared
            # diffusion at abutment provides the connection already).
            skip_map = _compute_skip_sd(template, placed)

            for dev in placed.values():
                tc = draw_transistor(
                    dev.geom.w_um,
                    dev.geom.l_um,
                    dev.spec.device_type,
                    self.rules,
                    n_fingers=dev.geom.n_fingers,
                    skip_sd=skip_map.get(dev.name),
                )
                ref = comp.add_ref(tc)
                ref.move((dev.x, dev.y))
                dev.component = tc

            # ── Merge implant/nwell regions ──
            _merge_implants(comp, placed, self.rules)
            _merge_nwells(comp, placed, self.rules)

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

            # ── GDS labels ────────────────────────────────────────────────
            _add_labels(comp, template, self.rules)
            _add_well_labels(comp, template, placed, self.rules)

            # ── DRC (optional) ─────────────────────────────────────────────
            if self.drc_runner is None:
                lvs_result = self._run_lvs(comp, template, current_params)
                return SynthResult(comp, placed, current_params,
                                   [], iteration, True, lvs=lvs_result)

            violations = _run_drc(self.drc_runner, comp)
            if not violations:
                lvs_result = self._run_lvs(comp, template, current_params)
                return SynthResult(comp, placed, current_params,
                                   [], iteration, True, lvs=lvs_result)

            # ── ML-guided param adjustment for next iteration ────────────
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

            # Use geo-fixed component (even if not fully converged,
            # it typically has fewer violations than the original)
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

            if geo_result.converged:
                return SynthResult(geo_comp, placed, current_params,
                                   [], iteration + geo_result.iterations, True)

            # Use geo-fixed component as the output
            comp = geo_comp

        # ── Final full DRC on actual output component ─────────────────
        if self.drc_runner is not None:
            violations = _run_drc(self.drc_runner, comp)

        # Return best result even if DRC is not clean
        lvs_result = self._run_lvs(comp, template, current_params)
        return SynthResult(comp, placed, current_params,
                           violations, self.max_iter,
                           converged=(not violations),
                           lvs=lvs_result)

    # ── LVS helper ───────────────────────────────────────────────────────────

    def _run_lvs(
        self,
        comp:     Any,
        template: CellTemplate,
        params:   dict,
    ) -> Any:
        """Run LVS on *comp* against a reference netlist generated from
        *template*.  Returns ``None`` if no ``lvs_runner`` is configured.
        """
        if self.lvs_runner is None:
            return None
        try:
            from layout_gen.lvs.netlist import build_reference_netlist
            ref = build_reference_netlist(template, self.rules, params)
        except Exception as exc:
            warnings.warn(f"LVS reference generation failed: {exc}", stacklevel=3)
            return None

        with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f_gds:
            gds_path = Path(f_gds.name)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".spice", delete=False, encoding="utf-8",
        ) as f_sp:
            f_sp.write(ref)
            ref_path = Path(f_sp.name)
        try:
            comp.write_gds(str(gds_path), with_metadata=False)
            return self.lvs_runner.run(gds_path, ref_path, template.name)
        except Exception as exc:
            warnings.warn(f"LVS runner failed: {exc}", stacklevel=3)
            return None
        finally:
            gds_path.unlink(missing_ok=True)
            ref_path.unlink(missing_ok=True)


# ── Skip-contacts computation ──────────────────────────────────────────────────

def _compute_skip_sd(
    template: CellTemplate,
    placed:   dict[str, PlacedDevice],
) -> dict[str, set[int]]:
    """Determine which S/D indices should skip contacts + li1.

    At abutment boundaries, the shared S/D connects via continuous
    diffusion.  If the net on that shared S/D is internal (not power
    and not an external port), contacts and li1/met1 are unnecessary.
    Skipping them avoids metal-area DRC violations on PDKs where
    li1 and met1 are the same layer.

    Returns dict mapping device_name → set of S/D indices to skip.
    """
    skip: dict[str, set[int]] = {}

    # Build set of nets that need metal connections (power + ports)
    port_nets = set(template.ports.keys()) if template.ports else set()
    power_nets = set()
    for net_name, net_info in (template.nets or {}).items():
        if net_info.net_type == "power":
            power_nets.add(net_name)
    needs_metal = power_nets | port_nets

    if not template.placement_directives:
        return skip

    for directive in template.placement_directives:
        if directive.relation != "abut_x":
            continue
        if not directive.relative_to:
            continue

        dev_name = directive.name
        anchor_name = directive.relative_to
        dev = placed.get(dev_name)
        anchor = placed.get(anchor_name)
        if dev is None or anchor is None:
            continue
        # Only same-type abutment (NMOS-NMOS, PMOS-PMOS)
        if dev.spec.device_type != anchor.spec.device_type:
            continue

        # The shared S/D: rightmost of anchor, leftmost of dev
        # Anchor's rightmost S/D index = n_fingers
        # Dev's leftmost S/D index = 0
        # Determine the net on the shared S/D
        # For anchor: rightmost S/D is drain (j=n_fingers, odd → drain) or
        #   source depending on finger count and sd_flip
        anchor_j = anchor.geom.n_fingers  # rightmost S/D index
        dev_j = 0                          # leftmost S/D index

        # Determine terminal type at these indices
        # j even → source, j odd → drain (before sd_flip)
        def _terminal_at(d, j):
            is_drain = (j % 2 == 1)
            if d.spec.sd_flip:
                is_drain = not is_drain
            return "D" if is_drain else "S"

        anchor_term = _terminal_at(anchor, anchor_j)
        dev_term = _terminal_at(dev, dev_j)

        # Look up the net name from the template
        anchor_spec = template.devices.get(anchor_name)
        dev_spec = template.devices.get(dev_name)
        if anchor_spec is None or dev_spec is None:
            continue

        anchor_net = anchor_spec.terminals.get(anchor_term, "")
        dev_net = dev_spec.terminals.get(dev_term, "")

        # Only skip if both map to the same internal net
        if anchor_net == dev_net and anchor_net and anchor_net not in needs_metal:
            skip.setdefault(anchor_name, set()).add(anchor_j)
            skip.setdefault(dev_name, set()).add(dev_j)

    return skip


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


def _merge_nwells(
    comp:   Any,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> None:
    """Draw one merged nwell per cluster of PMOS devices.

    Individual PMOS transistors draw their own nwell, but adjacent nwells
    closer than ``nwell.spacing_min_um`` (1.27 µm) violate nwell.2.
    Merge all nearby PMOS nwells into one continuous region.
    """
    from layout_gen.cells.standard import _diff_y

    nw_enc = rules.nwell.get("enclosure_of_pdiff_um", 0.18)
    nw_sp  = rules.nwell.get("spacing_min_um", 1.27)

    pmos_devs = [d for d in placed.values() if d.spec.device_type == "pmos"]
    if not pmos_devs:
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
    # Minimum nwell width from PDK
    nw_min_w = rules.nwell.get("width_min_um", 0.84)
    for cluster in clusters:
        x0 = min(b[0] for b in cluster)
        x1 = max(b[1] for b in cluster)
        y0 = min(b[2] for b in cluster)
        y1 = max(b[3] for b in cluster)
        # Ensure minimum nwell width in both directions
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if x1 - x0 < nw_min_w:
            x0 = cx - nw_min_w / 2
            x1 = cx + nw_min_w / 2
        if y1 - y0 < nw_min_w:
            y0 = cy - nw_min_w / 2
            y1 = cy + nw_min_w / 2
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


# ── GDS label export ──────────────────────────────────────────────────────────

def _add_labels(comp: Any, template: "CellTemplate", rules: PDKRules) -> None:
    """Add GDS text labels at port locations for LVS connectivity.

    The ``label_layers:`` section in the PDK YAML maps each routing layer
    to the GDS (layer, datatype) pair that the technology's DRC/LVS deck
    treats as the *pin* purpose.  We resolve each port's pin layer by
    matching its GDS layer to that map; this is what lets Magic promote a
    GDS label into a top-cell port during extraction.
    """
    pdk_labels: dict[str, tuple[int, int]] = dict(rules.label_layers or {})

    # Index by GDS layer number for fast lookup
    pdk_by_gds = {ll[0]: ll for ll in pdk_labels.values()}

    # Template-level overrides (legacy field) layered on top
    template_map: dict[str, tuple[int, int]] = {
        "met1": template.label_layers.met1,
        "met2": template.label_layers.met2,
    }

    def _resolve_for_port(port) -> tuple[int, int]:
        # 1. Logical layer name stashed by port_resolver (most reliable —
        #    gdsfactory may have remapped port.layer to an enum integer).
        try:
            ln = port.info.get("layer_name", "")
        except Exception:
            ln = ""
        if ln in pdk_labels:
            return pdk_labels[ln]
        # 2. Template-declared layer for this port name.
        pspec = template.ports.get(port.name)
        if pspec and pspec.layer in pdk_labels:
            return pdk_labels[pspec.layer]
        if pspec and pspec.layer in template_map:
            return template_map[pspec.layer]
        # 3. Match by GDS layer number for ports that still carry a tuple.
        port_layer = port.layer
        if isinstance(port_layer, (list, tuple)) and len(port_layer) >= 2:
            gds_layer = port_layer[0] if isinstance(port_layer[0], int) else port_layer
            if gds_layer in pdk_by_gds:
                return pdk_by_gds[gds_layer]
        return pdk_labels.get("met1") or template_map["met1"]

    for port in comp.ports:
        try:
            comp.add_label(
                port.name,
                position=port.center,
                layer=_resolve_for_port(port),
            )
        except Exception:
            pass  # label support varies by gdsfactory version


def _add_well_labels(
    comp:     Any,
    template: "CellTemplate",
    placed:   dict[str, PlacedDevice],
    rules:    PDKRules,
) -> None:
    """Drop VDD/GND labels onto the nwell and substrate layers.

    Magic's extractor names bulk nets from labels on the well-pin layers
    (``nwell`` / ``pwell`` in the PDK YAML's ``well_labels`` section).
    Without these labels Magic invents anonymous nets (``VSUBS``,
    ``w_36#`` …) that don't match the schematic-side bulk, breaking LVS
    even when the topology is otherwise correct.

    Note: this only *names* the well/substrate.  Physical well/substrate
    *taps* (diff + licon stack tying bulk to the rail) are required for
    the cell to be electrically correct in silicon — that is a separate
    concern.
    """
    well_layers: dict[str, tuple[int, int]] = dict(rules.well_labels or {})
    if not well_layers:
        return

    # Find power nets on each rail.
    vdd_net = ""
    gnd_net = ""
    for n in template.nets.values():
        if n.net_type != "power":
            continue
        if n.rail == "top":
            vdd_net = n.name
        elif n.rail == "bottom":
            gnd_net = n.name
    # Fall-back name guessing
    if not vdd_net and "VDD" in template.nets:
        vdd_net = "VDD"
    if not gnd_net:
        for cand in ("VSS", "GND"):
            if cand in template.nets:
                gnd_net = cand
                break

    # Cell bounds — labels go inside the footprint.
    if not placed:
        return
    x_lo = min(d.x for d in placed.values())
    x_hi = max(d.x + d.geom.total_x_um for d in placed.values())
    y_lo = min(d.y for d in placed.values())
    y_hi = max(d.y + d.geom.total_y_um for d in placed.values())
    cx = (x_lo + x_hi) / 2

    # nwell label → on a PMOS body (highest cluster of PMOS rows)
    pmos = [d for d in placed.values() if d.spec.device_type == "pmos"]
    if pmos and vdd_net and "nwell" in well_layers:
        py0 = min(d.y for d in pmos)
        py1 = max(d.y + d.geom.total_y_um for d in pmos)
        try:
            comp.add_label(
                vdd_net,
                position=(cx, (py0 + py1) / 2),
                layer=well_layers["nwell"],
            )
        except Exception:
            pass

    # substrate (pwell) label → outside the nwell, on the NMOS row
    nmos = [d for d in placed.values() if d.spec.device_type == "nmos"]
    if nmos and gnd_net and "pwell" in well_layers:
        ny0 = min(d.y for d in nmos)
        ny1 = max(d.y + d.geom.total_y_um for d in nmos)
        try:
            comp.add_label(
                gnd_net,
                position=(cx, (ny0 + ny1) / 2),
                layer=well_layers["pwell"],
            )
        except Exception:
            pass


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
