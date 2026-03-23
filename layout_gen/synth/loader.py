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


@dataclass
class DeviceSpec:
    """A single device instance inside a cell template."""
    name:         str
    template:     str              # "planar_mosfet"
    device_type:  str              # "nmos" | "pmos"
    terminals:    dict[str, str]   # {terminal: net}, e.g. {"G": "IN", "D": "OUT"}
    fingers:      int  = 0         # 0 = auto (ceil(w / w_finger_max_um)); >0 = explicit
    # ── Placement fields (populated from placement section) ────────────────
    region:       str  = "bottom"  # "bottom", "top", "bottom_only"
    in_nwell:     bool = False
    y_offset_expr: Any = 0         # int/float 0 or a symbolic string
    x_spec:        Any = None      # None=left/0, or "right_of: X", "between(A,B)", etc.
    # ── Stacked layout fields (populated from row_pairs section) ────────────
    row_pair:     int  = -1        # index into CellTemplate.row_pairs (-1 = unassigned)
    sd_flip:      bool = False     # swap S/D terminals for diffusion sharing abutment


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
class RowPairSpec:
    """One NMOS/PMOS row pair in a stacked layout.

    Devices in ``nmos_devices`` and ``pmos_devices`` are placed left-to-right
    with adjacent devices sharing diffusion (abutment).
    """
    id:            int
    nmos_devices:  list[str] = field(default_factory=list)
    pmos_devices:  list[str] = field(default_factory=list)


@dataclass
class CellTemplate:
    """Parsed cell topology template."""
    name:               str
    description:        str
    devices:            dict[str, DeviceSpec]
    nets:               dict[str, NetSpec]
    ports:              dict[str, PortSpec]
    named_constraints:  dict[str, Any]   # {name: {min: expr} or expr}
    source_path:        Path | None = None
    layout_mode:        str = "standard"   # "standard" or "stacked"
    row_pairs:          list[RowPairSpec] = field(default_factory=list)


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
        )
    return devices


def _load_template(
    raw:     dict,
    devices: dict[str, DeviceSpec],
    path:    Path,
) -> CellTemplate:
    """Parse a declarative topology template."""
    placement = raw.get("placement", {})
    layout_mode = placement.get("mode", "standard")

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
            )

    # ── Placement → row_pairs or standard pairs ──────────────────────────
    row_pairs: list[RowPairSpec] = []
    named_constraints: dict[str, Any] = {}

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
        # Standard mode: pairs section
        raw_pairs = placement.get("pairs", [])
        if raw_pairs:
            pair = raw_pairs[0]
            nmos_names = list(pair.get("nmos", []))
            pmos_names = list(pair.get("pmos", []))

            # Auto-derive floorplan for standard single-pair layouts
            for idx, name in enumerate(nmos_names):
                if name in devices:
                    devices[name].region = "bottom"
                    if idx == 0:
                        devices[name].x_spec = "left"
                    else:
                        prev = nmos_names[idx - 1]
                        devices[name].x_spec = f"{prev}.total_x"
                    devices[name].y_offset_expr = 0
            for idx, name in enumerate(pmos_names):
                if name in devices:
                    devices[name].region = "top"
                    devices[name].in_nwell = True
                    if idx == 0:
                        devices[name].x_spec = "left"
                    else:
                        prev = pmos_names[idx - 1]
                        devices[name].x_spec = f"{prev}.total_x"
                    first_nmos = nmos_names[0] if nmos_names else name
                    devices[name].y_offset_expr = (
                        f"{first_nmos}.total_y + inter_cell_gap"
                    )

    # ── Constraints ───────────────────────────────────────────────────────
    raw_constraints = placement.get("constraints", {})
    for key, val in raw_constraints.items():
        named_constraints[key] = val

    return CellTemplate(
        name               = raw.get("name", path.stem),
        description        = str(raw.get("description", "")),
        devices            = devices,
        nets               = nets,
        ports              = ports,
        named_constraints  = named_constraints,
        source_path        = path,
        layout_mode        = layout_mode,
        row_pairs          = row_pairs,
    )
