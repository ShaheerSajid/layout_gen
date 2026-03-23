"""
layout_gen.synth.loader — topology template YAML → Python dataclasses.

Parses a cell topology template into strongly-typed objects.  No numeric
evaluation happens here; every symbolic string stays a string so the
constraint evaluator and placer can process it later.
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
class PortSpec:
    """Output port declaration from a cell topology template."""
    name:     str
    layer:    str   # logical layer name: "poly", "li1", "met1"
    location: str   # location keyword, e.g. "gate_left_edge_mid_y"


@dataclass
class DeviceSpec:
    """A single device instance inside a cell template."""
    name:         str
    template:     str              # "planar_mosfet"
    device_type:  str              # "nmos" | "pmos"
    terminals:    dict[str, str]   # {terminal: net}, e.g. {"G": "IN", "D": "OUT"}
    fingers:      int  = 0         # 0 = auto (ceil(w / w_finger_max_um)); >0 = explicit
    # ── Floorplan fields (populated from floorplan section) ──────────────────
    region:       str  = "bottom"  # "bottom", "top", "bottom_only"
    in_nwell:     bool = False
    y_offset_expr: Any = 0         # int/float 0 or a symbolic string
    x_spec:        Any = None      # None=left/0, or "right_of: X", "between(A,B)", etc.
    # ── Stacked layout fields (populated from row_pairs section) ────────────
    row_pair:     int  = -1        # index into CellTemplate.row_pairs (-1 = unassigned)
    sd_flip:      bool = False     # swap S/D terminals for diffusion sharing abutment


@dataclass
class RoutingSpec:
    """One routing connection specification."""
    net:   str
    style: str           # "shared_gate_poly", "drain_bridge", "horizontal_power_rail", …
    layer: str
    path:  list[str]     # terminal refs: ["N.G", "P.G"]
    edge:  str  = ""     # for horizontal_power_rail: "bottom" | "top"
    extra: dict = field(default_factory=dict)  # via_stack, note, width, …


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
    """Parsed cell topology template (device + floorplan + routing + ports)."""
    name:               str
    version:            str
    description:        str
    devices:            dict[str, DeviceSpec]
    nets:               list[str]
    routing:            list[RoutingSpec]
    ports:              dict[str, PortSpec]
    named_constraints:  dict[str, Any]   # {name: {min: expr} or expr}
    source_path:        Path | None = None
    layout_mode:        str = "standard"                          # "standard" or "stacked"
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
    fp = raw.get("floorplan", {})
    layout_mode = fp.get("layout_mode", "standard")

    _apply_floorplan(fp, devices)

    row_pairs: list[RowPairSpec] = []
    if layout_mode == "stacked":
        row_pairs = _parse_row_pairs(fp, devices)

    return CellTemplate(
        name               = raw.get("name", path.stem),
        version            = str(raw.get("version", "1.0")),
        description        = str(raw.get("description", "")),
        devices            = devices,
        nets               = list(raw.get("nets", [])),
        routing            = _parse_routing(raw.get("routing", [])),
        ports              = _parse_ports(raw.get("ports", {})),
        named_constraints  = _parse_named_constraints(fp, devices),
        source_path        = path,
        layout_mode        = layout_mode,
        row_pairs          = row_pairs,
    )


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


def _apply_floorplan(fp: dict, devices: dict[str, DeviceSpec]) -> None:
    """Populate floorplan fields on DeviceSpec objects from the floorplan section."""
    if fp.get("layout_mode") == "stacked":
        # Stacked mode: device placement comes from row_pairs, not per-device entries.
        return
    for key, val in fp.items():
        if key in devices and isinstance(val, dict):
            _apply_device_fp(devices[key], val)
        elif isinstance(val, dict):
            # Nested group (e.g. INV_L: {PD_L: {...}, PU_L: {...}})
            for subkey, subval in val.items():
                if subkey in devices and isinstance(subval, dict):
                    _apply_device_fp(devices[subkey], subval)


def _apply_device_fp(spec: DeviceSpec, fp: dict) -> None:
    spec.region       = fp.get("region", "bottom")
    spec.in_nwell     = bool(fp.get("in_nwell", False))
    spec.y_offset_expr = fp.get("y_offset", 0)
    spec.x_spec        = fp.get("x", None)


def _parse_routing(raw_list: list) -> list[RoutingSpec]:
    out = []
    for r in raw_list:
        known = {"net", "style", "layer", "path", "edge"}
        out.append(RoutingSpec(
            net   = r.get("net", ""),
            style = r.get("style", ""),
            layer = r.get("layer", ""),
            path  = list(r.get("path", [])),
            edge  = r.get("edge", ""),
            extra = {k: v for k, v in r.items() if k not in known},
        ))
    return out


def _parse_ports(raw: dict) -> dict[str, PortSpec]:
    out: dict[str, PortSpec] = {}
    for name, spec in raw.items():
        out[name] = PortSpec(
            name     = name,
            layer    = spec.get("layer", ""),
            location = spec.get("location", ""),
        )
    return out


def _parse_row_pairs(
    fp: dict, devices: dict[str, DeviceSpec],
) -> list[RowPairSpec]:
    """Parse ``row_pairs`` list from the floorplan section (stacked mode)."""
    raw_pairs = fp.get("row_pairs", [])
    pairs: list[RowPairSpec] = []
    for i, rp in enumerate(raw_pairs):
        nmos = list(rp.get("nmos", []))
        pmos = list(rp.get("pmos", []))
        sd_flip = rp.get("sd_flip", {})

        pair = RowPairSpec(
            id=int(rp.get("id", i)),
            nmos_devices=nmos,
            pmos_devices=pmos,
        )
        pairs.append(pair)

        # Propagate placement info onto each DeviceSpec
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

    return pairs


# Floorplan keys that are structural directives, not named constraints.
_STRUCTURAL_FP_KEYS = frozenset({"layout_mode", "row_pairs"})


def _parse_named_constraints(fp: dict, devices: dict[str, DeviceSpec]) -> dict[str, Any]:
    """Extract named constraint entries (non-device, non-group entries)."""
    out: dict[str, Any] = {}
    for key, val in fp.items():
        if key in devices or key in _STRUCTURAL_FP_KEYS:
            continue
        # Check if it's a pure device-group (all sub-keys are device names)
        if isinstance(val, dict) and all(k in devices for k in val if k not in ("note",)):
            continue
        out[key] = val
    return out
