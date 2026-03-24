"""
layout_gen.synth.loader — topology template YAML → Python dataclasses.

Parses a cell topology template into strongly-typed objects.  No numeric
evaluation happens here; every symbolic string stays a string so the
constraint evaluator and placer can process it later.

The template format is declarative: the user specifies devices,
connectivity (nets with types), placement (row pairs or standard pairs),
and ports (compass side).  Routing is inferred automatically by the
auto-router from the connectivity graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Default search directories (in priority order)
_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class NetSpec:
    """Net declaration from a topology template."""
    name:     str
    net_type: str        # "power" | "signal" | "internal"
    rail:     str = ""   # "top" | "bottom" (power nets only)


@dataclass
class PortSpec:
    """Port declaration from a topology template (compass-side)."""
    name:     str
    side:     str              # "north" | "south" | "east" | "west"
    terminal: str = ""         # optional "Dev.Term" for disambiguation
    layer:    str = ""         # optional explicit layer ("met1", "met2")


@dataclass
class DeviceSpec:
    """A single device instance inside a cell template."""
    name:         str
    template:     str              # "planar_mosfet"
    device_type:  str              # "nmos" | "pmos"
    terminals:    dict[str, str]   # {terminal: net}, e.g. {"G": "IN", "D": "OUT"}
    fingers:      int  = 0         # 0 = auto (ceil(w / w_finger_max_um)); >0 = explicit
    w:            float = 0.0      # per-device width (µm); 0 = use params/defaults
    l:            float = 0.0      # per-device gate length (µm); 0 = use params/defaults
    # ── Placement fields (populated from placement section) ────────────────
    region:       str  = "bottom"  # "bottom", "top", "bottom_only"
    in_nwell:     bool = False
    y_offset_expr: Any = 0         # int/float 0 or a symbolic string
    x_spec:        Any = None      # None=left/0, or "right_of: X", "between(A,B)", etc.
    # ── Stacked layout fields (populated from row_pairs section) ────────────
    row_pair:     int  = -1        # index into CellTemplate.row_pairs (-1 = unassigned)
    sd_flip:      bool = False     # swap S/D terminals for diffusion sharing abutment


@dataclass
class PlacementDirective:
    """Explicit per-device placement instruction.

    Each device gets one directive that fully specifies its position
    relative to another device (or at an absolute origin).

    Fields
    ------
    name : device name (must match a key in ``devices``)
    relative_to : device this is placed relative to (empty = absolute)
    relation : how to position relative to the anchor device:
        ``"abut_x"``  — shared-diffusion abutment (overlap one S/D)
        ``"space_x"`` — place with ``spacing_rule`` gap in X
        ``"align_gate"`` — align gate poly vertically (X match)
    spacing_rule : named PDK-derived spacing formula:
        ``"min_diff_spacing"``, ``"cross_couple_wiring"``,
        ``"min_well_separation"``
    alignment : vertical positioning relative to anchor:
        ``"bottom"`` — align bottom edges
        ``"gate"``   — align gate poly centres
    orientation : device orientation:
        ``"R0"`` — no rotation (default)
        ``"MX"`` — mirror across X axis (flip vertically)
        ``"MY"`` — mirror across Y axis (flip horizontally)
        ``"R180"`` — rotate 180°
    origin : absolute (x, y) origin — only for the first device
    """
    name:          str
    relative_to:   str = ""
    relation:      str = ""          # "abut_x", "space_x", "align_gate"
    spacing_rule:  str = ""          # "min_diff_spacing", "cross_couple_wiring", etc.
    alignment:     str = "bottom"    # "bottom", "gate", "top", "center"
    orientation:   str = "R0"        # "R0", "MX", "MY", "R180"
    sd_flip:       bool = False     # swap S/D so internal net faces shared contact
    origin:        tuple[float, float] | None = None   # absolute (x, y)


@dataclass
class RoutingSpec:
    """One routing connection specification (auto-router output).

    This is an internal type produced by the auto-router and consumed by
    the Router's style handlers.  Users never write these in YAML.
    """
    net:   str
    style: str           # "shared_gate_poly", "drain_bridge", "horizontal_power_rail", …
    layer: str  = ""
    path:  list[str] = field(default_factory=list)   # terminal refs: ["N.G", "P.G"]
    edge:  str  = ""     # for horizontal_power_rail: "bottom" | "top"
    extra: dict = field(default_factory=dict)  # via_level, track_x, bus_x, …


@dataclass
class AbutmentSpec:
    """Abutment/tiling rules for array integration."""
    pitch_x:     float = 0.0    # column pitch (0 = auto)
    pitch_y:     float = 0.0    # row pitch (0 = auto)
    mirror_x:    bool  = False  # mirror for adjacent columns
    rail_align:  bool  = True   # align power rails across abutment


@dataclass
class CellDimensions:
    """Fixed cell dimensions for array pitch matching."""
    width:  float = 0.0    # 0 = auto-compute from device placement
    height: float = 0.0    # 0 = auto-compute


@dataclass
class LabelLayerSpec:
    """GDS layer/datatype for port labels per metal layer."""
    met1: tuple[int, int] = (68, 5)
    met2: tuple[int, int] = (69, 5)


@dataclass
class RowPairSpec:
    """One NMOS/PMOS row pair in a stacked layout.

    Devices in ``nmos_devices`` and ``pmos_devices`` are placed left-to-right
    with adjacent devices sharing diffusion (abutment).
    """
    id:            int
    nmos_devices:  list[str] = field(default_factory=list)
    pmos_devices:  list[str] = field(default_factory=list)
    rail_top:      str = ""    # power net name for rail above this pair (e.g. "VDD")
    rail_bottom:   str = ""    # power net name for rail below this pair (e.g. "VSS")


@dataclass
class RoutingHint:
    """Per-net routing hint from the topology YAML.

    Tells the auto-router the *preferred* layer and style for a net.
    These are hints, not hard constraints — the engine may override if
    the preferred choice is infeasible.

    Fields
    ------
    layer : preferred metal layer (``"met1"``, ``"met2"``)
    style : ``"full_width"`` | ``"full_height"`` | ``"local"``
    strategy : ``"local"`` = keep on lower metals
    path_type : ``"trunk"`` | ``"rail"`` | ``"bridge"``  (maps to style handler)
    coverage : ``"full_width"`` | ``"full_height"``  (alternative to style)
    port_side : list of sides where net must be accessible (``["west", "east"]``)
    width_type : ``"fixed_min"`` (PDK minimum) | ``"wide"`` (double width)
    mergeable : whether adjacent cells' copies of this net can merge
    """
    net:         str
    layer:       str = ""
    style:       str = ""
    strategy:    str = ""
    path_type:   str = ""        # "trunk", "rail", "bridge"
    coverage:    str = ""        # "full_width", "full_height"
    port_side:   list[str] = field(default_factory=list)
    width_type:  str = ""        # "fixed_min", "wide"
    mergeable:   bool = False


@dataclass
class CellTemplate:
    """Parsed cell topology template."""
    name:               str
    description:        str
    devices:            dict[str, DeviceSpec]
    nets:               dict[str, NetSpec]
    # --- added fields below are populated during _load_template ---
    ports:              dict[str, PortSpec]
    named_constraints:  dict[str, Any]   # {name: {min: expr} or expr}
    source_path:        Path | None = None
    layout_mode:        str = "standard"   # "standard" or "stacked"
    row_pairs:          list[RowPairSpec] = field(default_factory=list)
    cell_dimensions:    CellDimensions = field(default_factory=CellDimensions)
    abutment:           AbutmentSpec = field(default_factory=AbutmentSpec)
    label_layers:       LabelLayerSpec = field(default_factory=LabelLayerSpec)
    device_params:      dict[str, dict[str, Any]] = field(default_factory=dict)
    routing_hints:      dict[str, RoutingHint] = field(default_factory=dict)
    placement_relations: dict[str, list[list[str]]] = field(default_factory=dict)
    placement_directives: list[PlacementDirective] = field(default_factory=list)
    diffusion_merges: list[tuple[str, str]] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def load_template(name_or_path: str | Path) -> CellTemplate:
    """Load a cell topology template from a YAML file.

    Parameters
    ----------
    name_or_path :
        Either an absolute path to a ``.yaml`` file, or a template name
        (e.g. ``"inverter"``).  Template names are resolved against the
        built-in template directories.

    Returns
    -------
    CellTemplate
    """
    path = _resolve_path(name_or_path)
    raw  = yaml.safe_load(path.read_text(encoding="utf-8"))
    devices = _parse_devices(raw)
    return _load_template(raw, devices, path)


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _resolve_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".yaml" and p.exists():
        return p
    candidates = [
        _TEMPLATE_DIR / "cells" / f"{name_or_path}.yaml",
        _TEMPLATE_DIR / f"{name_or_path}.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Template {name_or_path!r} not found.  "
        f"Searched: {[str(c) for c in candidates]}"
    )


def _parse_devices(raw: dict) -> dict[str, DeviceSpec]:
    devices: dict[str, DeviceSpec] = {}
    for name, spec in raw.get("devices", {}).items():
        devices[name] = DeviceSpec(
            name        = name,
            template    = spec.get("template", "planar_mosfet"),
            device_type = spec.get("type", "nmos"),
            terminals   = dict(spec.get("terminals", {})),
            fingers     = int(spec.get("fingers", 0)),
            w           = float(spec.get("w", 0.0)),
            l           = float(spec.get("l", 0.0)),
        )
    return devices


def _normalize_layer(layer_str: str) -> str:
    """Normalize layer names: ``M1`` → ``met1``, ``M2`` → ``met2``, etc."""
    s = layer_str.strip()
    # Accept both "M1" and "met1" conventions
    import re
    m = re.match(r'^[Mm](\d+)$', s)
    if m:
        return f"met{m.group(1)}"
    return s.lower() if s else ""


def _parse_routing_hint(net_name: str, spec: dict) -> RoutingHint:
    """Parse a single routing hint dict into a RoutingHint object."""
    layer = _normalize_layer(str(spec.get("layer", "")))
    style = str(spec.get("style", ""))
    strategy = str(spec.get("strategy", ""))
    path_type = str(spec.get("path_type", ""))
    coverage = str(spec.get("coverage", ""))
    width_type = str(spec.get("width_type", ""))
    mergeable = bool(spec.get("mergeable", False))

    # Normalize port_side: accept string or list
    raw_ps = spec.get("port_side", [])
    if isinstance(raw_ps, str):
        port_side = [raw_ps]
    elif isinstance(raw_ps, list):
        port_side = [str(s) for s in raw_ps]
    else:
        port_side = []

    # Map path_type/coverage to style if style not explicitly set
    if not style:
        if path_type == "rail" or coverage == "full_width":
            style = "full_width"
        elif coverage == "full_height":
            style = "full_height"
        elif path_type == "trunk":
            style = "full_width"
        elif strategy == "local" or path_type == "bridge":
            strategy = strategy or "local"

    return RoutingHint(
        net=net_name,
        layer=layer,
        style=style,
        strategy=strategy,
        path_type=path_type,
        coverage=coverage,
        port_side=port_side,
        width_type=width_type,
        mergeable=mergeable,
    )


def _load_template(
    raw:     dict,
    devices: dict[str, DeviceSpec],
    path:    Path,
) -> CellTemplate:
    """Parse a declarative topology template."""
    placement = raw.get("placement", {})
    # placement can be a dict (standard/stacked modes) or a list (directives mode)
    if isinstance(placement, dict):
        layout_mode = placement.get("mode", "standard")
    else:
        layout_mode = "directives"

    # ── Nets ──────────────────────────────────────────────────────────────
    nets: dict[str, NetSpec] = {}
    raw_nets = raw.get("nets", {})
    if isinstance(raw_nets, dict):
        for name, spec in raw_nets.items():
            if isinstance(spec, dict):
                nets[name] = NetSpec(
                    name=name,
                    net_type=spec.get("type", "signal"),
                    rail=spec.get("rail", ""),
                )
            else:
                nets[name] = NetSpec(name=name, net_type="signal")
    elif isinstance(raw_nets, list):
        for name in raw_nets:
            if name in ("VDD", "VSS", "GND"):
                rail = "top" if name == "VDD" else "bottom"
                nets[name] = NetSpec(name=name, net_type="power", rail=rail)
            else:
                nets[name] = NetSpec(name=name, net_type="signal")

    # Auto-add nets from device terminals that aren't declared
    for dev in devices.values():
        for term, net_name in dev.terminals.items():
            if term == "B":
                continue
            if net_name not in nets:
                nets[net_name] = NetSpec(name=net_name, net_type="internal")

    # ── Ports ─────────────────────────────────────────────────────────────
    ports: dict[str, PortSpec] = {}
    for name, spec in raw.get("ports", {}).items():
        if isinstance(spec, dict):
            ports[name] = PortSpec(
                name=name,
                side=spec.get("side", "east"),
                terminal=spec.get("terminal", ""),
                layer=spec.get("layer", ""),
            )

    # ── Placement → row_pairs or standard pairs ──────────────────────────
    row_pairs: list[RowPairSpec] = []
    named_constraints: dict[str, Any] = {}
    raw_relations = placement.get("relations", {}) if isinstance(placement, dict) else {}

    if layout_mode == "stacked":
        raw_pairs = placement.get("row_pairs", [])
        for i, rp in enumerate(raw_pairs):
            nmos = list(rp.get("nmos", []))
            pmos = list(rp.get("pmos", []))
            sd_flip = rp.get("sd_flip", {})

            pair = RowPairSpec(
                id=int(rp.get("id", i)),
                nmos_devices=nmos,
                pmos_devices=pmos,
                rail_top=str(rp.get("rail_top", "")),
                rail_bottom=str(rp.get("rail_bottom", "")),
            )
            row_pairs.append(pair)

            for name in nmos:
                if name in devices:
                    devices[name].row_pair = pair.id
                    devices[name].region = "bottom"
                    devices[name].sd_flip = bool(sd_flip.get(name, False))
            for name in pmos:
                if name in devices:
                    devices[name].row_pair = pair.id
                    devices[name].region = "top"
                    devices[name].in_nwell = True
                    devices[name].sd_flip = bool(sd_flip.get(name, False))
    else:
        # Standard mode: pairs section (or rows section)
        raw_pairs = placement.get("pairs", [])
        if not raw_pairs:
            # New format: placement.rows
            raw_rows = placement.get("rows", {})
            if raw_rows:
                raw_pairs = [{"nmos": raw_rows.get("nmos", []),
                              "pmos": raw_rows.get("pmos", [])}]

        # ── Parse placement.relations ─────────────────────────────────
        shared_diff_set: set[tuple[str, str]] = set()
        cross_gap_set: set[tuple[str, str]] = set()
        gate_align_map: dict[str, str] = {}  # pmos_name → nmos_name

        for pair_list in raw_relations.get("shared_diffusion", []):
            if len(pair_list) >= 2:
                shared_diff_set.add((pair_list[0], pair_list[1]))
                shared_diff_set.add((pair_list[1], pair_list[0]))
        for pair_list in raw_relations.get("cross_couple_gap", []):
            if len(pair_list) >= 2:
                cross_gap_set.add((pair_list[0], pair_list[1]))
                cross_gap_set.add((pair_list[1], pair_list[0]))
        for pair_list in raw_relations.get("gate_align", []):
            if len(pair_list) >= 2:
                # Convention: [nmos_dev, pmos_dev]
                gate_align_map[pair_list[1]] = pair_list[0]

        if raw_pairs:
            pair = raw_pairs[0]
            nmos_names = list(pair.get("nmos", []))
            pmos_names = list(pair.get("pmos", []))

            # ── NMOS row: cumulative X with relation-aware spacing ────
            for idx, name in enumerate(nmos_names):
                if name not in devices:
                    continue
                devices[name].region = "bottom"
                devices[name].y_offset_expr = 0
                if idx == 0:
                    devices[name].x_spec = "left"
                else:
                    prev = nmos_names[idx - 1]
                    if (prev, name) in shared_diff_set:
                        # Abutment: overlap one shared S/D region
                        devices[name].x_spec = (
                            f"{prev}_x + {prev}.total_x - {prev}.sd"
                        )
                    elif (prev, name) in cross_gap_set:
                        # Cross-couple gap for internal wiring
                        devices[name].x_spec = (
                            f"{prev}_x + {prev}.total_x + cross_gap"
                        )
                    else:
                        # Default: place right after previous device
                        devices[name].x_spec = (
                            f"{prev}_x + {prev}.total_x"
                        )

            # ── PMOS row: gate-aligned or cumulative X ─────────────────
            for idx, name in enumerate(pmos_names):
                if name not in devices:
                    continue
                devices[name].region = "top"
                devices[name].in_nwell = True

                first_nmos = nmos_names[0] if nmos_names else name
                devices[name].y_offset_expr = (
                    f"{first_nmos}.total_y + inter_cell_gap"
                )

                aligned_nmos = gate_align_map.get(name)
                if aligned_nmos:
                    # Gate-align: place at same X as the aligned NMOS device
                    devices[name].x_spec = f"{aligned_nmos}_x"
                elif idx == 0:
                    devices[name].x_spec = "left"
                else:
                    prev = pmos_names[idx - 1]
                    if (prev, name) in shared_diff_set:
                        devices[name].x_spec = (
                            f"{prev}_x + {prev}.total_x - {prev}.sd"
                        )
                    else:
                        devices[name].x_spec = (
                            f"{prev}_x + {prev}.total_x"
                        )

    # ── Constraints ───────────────────────────────────────────────────────
    raw_constraints = placement.get("constraints", {}) if isinstance(placement, dict) else {}
    for key, val in raw_constraints.items():
        named_constraints[key] = val

    # ── Cell dimensions ───────────────────────────────────────────────
    raw_dims = raw.get("cell_dimensions", {})
    cell_dims = CellDimensions(
        width=float(raw_dims.get("width", 0)),
        height=float(raw_dims.get("height", 0)),
    ) if raw_dims else CellDimensions()

    # ── Abutment ──────────────────────────────────────────────────────
    raw_abut = raw.get("abutment", {})
    abutment = AbutmentSpec(
        pitch_x=float(raw_abut.get("pitch_x", 0)),
        pitch_y=float(raw_abut.get("pitch_y", 0)),
        mirror_x=bool(raw_abut.get("mirror_x", False)),
        rail_align=bool(raw_abut.get("rail_align", True)),
    ) if raw_abut else AbutmentSpec()

    # ── Label layers ──────────────────────────────────────────────────
    raw_labels = raw.get("label_layers", {})
    label_layers = LabelLayerSpec()
    if raw_labels:
        if "met1" in raw_labels:
            label_layers.met1 = tuple(raw_labels["met1"])
        if "met2" in raw_labels:
            label_layers.met2 = tuple(raw_labels["met2"])

    # ── Per-device parameter overrides ────────────────────────────────
    raw_params = raw.get("params", {})
    device_params: dict[str, dict[str, Any]] = {}
    if isinstance(raw_params, dict):
        overrides = raw_params.get("overrides", {})
        if isinstance(overrides, dict):
            device_params = {k: dict(v) for k, v in overrides.items()
                           if isinstance(v, dict)}
        # Store defaults in named_constraints for the placer to pick up
        defaults = raw_params.get("defaults", {})
        if isinstance(defaults, dict):
            for dk, dv in defaults.items():
                named_constraints.setdefault(f"param_{dk}", dv)

    # ── Merge per-device w/l from device spec into device_params ─────
    for dname, dspec in devices.items():
        if dspec.w > 0 or dspec.l > 0:
            dp = device_params.setdefault(dname, {})
            if dspec.w > 0:
                dp.setdefault("w", dspec.w)
            if dspec.l > 0:
                dp.setdefault("l", dspec.l)

    # ── Routing hints ─────────────────────────────────────────────────
    routing_hints: dict[str, RoutingHint] = {}
    raw_routing = raw.get("routing", raw.get("routing_instructions", {}))
    # Support both dict format ({net: {layer:...}}) and list format ([{net:..., layer:...}])
    if isinstance(raw_routing, list):
        # List-of-dicts format: [{net: "WL", layer: "met1", path_type: "trunk"}, ...]
        for hint_spec in raw_routing:
            if not isinstance(hint_spec, dict) or "net" not in hint_spec:
                continue
            net_name = str(hint_spec["net"])
            routing_hints[net_name] = _parse_routing_hint(net_name, hint_spec)
    elif isinstance(raw_routing, dict):
        for net_name, hint_spec in raw_routing.items():
            if isinstance(hint_spec, dict):
                routing_hints[net_name] = _parse_routing_hint(net_name, hint_spec)

    # ── Placement relations (store raw for downstream use) ────────────
    placement_relations: dict[str, list[list[str]]] = {}
    if isinstance(raw_relations, dict):
        for rel_type, pairs in raw_relations.items():
            if isinstance(pairs, list):
                placement_relations[rel_type] = [
                    list(p) for p in pairs if isinstance(p, list)
                ]

    # ── Placement directives (explicit per-device placement) ──────────
    placement_directives: list[PlacementDirective] = []
    raw_directives = raw.get("placement_logic", [])
    if not raw_directives and isinstance(placement, list):
        # Also accept placement: [{name: ...}, ...] as a list of directives
        raw_directives = placement
    for pd in raw_directives:
        if not isinstance(pd, dict) or "name" not in pd:
            continue
        origin = None
        raw_origin = pd.get("origin")
        if raw_origin and isinstance(raw_origin, (list, tuple)) and len(raw_origin) >= 2:
            origin = (float(raw_origin[0]), float(raw_origin[1]))
        placement_directives.append(PlacementDirective(
            name=str(pd["name"]),
            relative_to=str(pd.get("relative_to", "")),
            relation=str(pd.get("relation", "")),
            spacing_rule=str(pd.get("spacing_rule", "")),
            alignment=str(pd.get("alignment", "bottom")),
            orientation=str(pd.get("orientation", "R0")),
            sd_flip=bool(pd.get("sd_flip", False)),
            origin=origin,
        ))

    # ── Diffusion merge directives ────────────────────────────────────
    diffusion_merges: list[tuple[str, str]] = []
    raw_merges = raw.get("diffusion_merge", [])
    if isinstance(raw_merges, list):
        for pair in raw_merges:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                diffusion_merges.append((str(pair[0]), str(pair[1])))

    return CellTemplate(
        name                = raw.get("name", path.stem),
        description         = str(raw.get("description", "")),
        devices             = devices,
        nets                = nets,
        ports               = ports,
        named_constraints   = named_constraints,
        source_path         = path,
        layout_mode         = layout_mode,
        row_pairs           = row_pairs,
        cell_dimensions     = cell_dims,
        abutment            = abutment,
        label_layers        = label_layers,
        device_params       = device_params,
        routing_hints       = routing_hints,
        placement_relations = placement_relations,
        placement_directives = placement_directives,
        diffusion_merges    = diffusion_merges,
    )
