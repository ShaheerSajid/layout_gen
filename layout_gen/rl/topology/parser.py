"""
layout_gen.rl.topology.parser — CellTemplate → TopologyGraph.

Strips the placement/routing schema down to the netlist graph the policy
needs as conditioning:
  * **Device nodes** carry type (nmos/pmos), sizing (w, l, fingers),
    well placement (in_nwell), and a one-hot template id.
  * **Net hyperedges** carry type (power/signal/internal), rail position
    (top/bottom for power), and a hint layer (one-hot over LAYER_ROLES
    when supplied; zero otherwise).

Sizing fallbacks: when a device YAML omits ``w``/``l``/``fingers``, we
look up ``cell_params[device_name]`` (the dict the synth pipeline already
threads through), then fall back to the cell-wide defaults
``{w_N, w_P, l}`` if those keys exist.

PDK-agnostic invariants
-----------------------
* Layer hints map to the abstract :data:`LAYER_ROLES` table — no vendor
  layer names enter the parser's output.
* Rail position uses a small categorical (``""``, ``"top"``, ``"bottom"``);
  no rail Y coordinate or PDK-specific pitch.
* Device features are dimensionless where possible (fingers as int,
  w/l in µm only because the YAML uses them too — the encoder
  normalises before they reach a tensor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from layout_gen.repair.features import LAYER_ROLES, role_index
from layout_gen.synth.loader import CellTemplate, DeviceSpec, NetSpec


# ── Vocabulary ───────────────────────────────────────────────────────────────

DEVICE_TYPES: tuple[str, ...] = ("nmos", "pmos", "other")
NET_TYPES:    tuple[str, ...] = ("power", "signal", "internal", "other")
RAIL_POSITIONS: tuple[str, ...] = ("none", "top", "bottom")
TEMPLATES:    tuple[str, ...] = ("planar_mosfet", "other")

_DEV_TYPE_IDX  = {t: i for i, t in enumerate(DEVICE_TYPES)}
_NET_TYPE_IDX  = {t: i for i, t in enumerate(NET_TYPES)}
_RAIL_IDX      = {r: i for i, r in enumerate(RAIL_POSITIONS)}
_TEMPLATE_IDX  = {t: i for i, t in enumerate(TEMPLATES)}


# ── Graph dataclasses ────────────────────────────────────────────────────────

@dataclass
class DeviceNode:
    """One device in the topology graph."""
    name:        str
    device_type: str         # one of DEVICE_TYPES
    template:    str         # one of TEMPLATES
    w_um:        float
    l_um:        float
    fingers:     int
    in_nwell:    bool


@dataclass
class NetEdge:
    """One net (hyperedge) in the topology graph.

    ``connections`` is a list of ``(device_idx, terminal_name)`` pairs
    — the GNN expands this into a clique among the device endpoints.
    """
    name:        str
    net_type:    str         # one of NET_TYPES
    rail:        str         # one of RAIL_POSITIONS
    layer_hint:  str         # logical layer name or "" if unspecified
    connections: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class TopologyGraph:
    """Devices + nets ready for the GNN encoder.

    Attributes
    ----------
    cell_name :
        Name of the source cell (for logging only).
    devices :
        Ordered list of :class:`DeviceNode`. Index = device_idx.
    nets :
        Ordered list of :class:`NetEdge`. Index = net_idx.
    """
    cell_name: str
    devices:   list[DeviceNode]
    nets:      list[NetEdge]

    @property
    def n_devices(self) -> int:
        return len(self.devices)

    @property
    def n_nets(self) -> int:
        return len(self.nets)

    def device_index(self) -> dict[str, int]:
        return {d.name: i for i, d in enumerate(self.devices)}

    def adjacency(self) -> list[list[int]]:
        """For each device, the list of net indices it touches."""
        adj: list[list[int]] = [[] for _ in range(self.n_devices)]
        for j, net in enumerate(self.nets):
            for (i, _term) in net.connections:
                adj[i].append(j)
        return adj


# ── Builder ──────────────────────────────────────────────────────────────────

def graph_from_template(
    template:    CellTemplate,
    cell_params: dict[str, dict[str, Any]] | None = None,
) -> TopologyGraph:
    """Build a :class:`TopologyGraph` from a :class:`CellTemplate`.

    Parameters
    ----------
    template :
        The parsed YAML.
    cell_params :
        Optional ``{device_name: {w, l, fingers}}`` overrides for devices
        whose YAML left those fields at zero. Cell-wide defaults
        (``w_N``, ``w_P``, ``l``) under the ``"_defaults"`` key are also
        consulted.
    """
    cell_params = cell_params or {}
    defaults = cell_params.get("_defaults", {})

    devices: list[DeviceNode] = []
    name_to_idx: dict[str, int] = {}
    for i, (name, spec) in enumerate(template.devices.items()):
        w = _resolve_sizing(spec, "w", cell_params, defaults)
        l = _resolve_sizing(spec, "l", cell_params, defaults)
        fingers = int(_resolve_field(spec, "fingers",
                                      cell_params, defaults, default=0))
        devices.append(DeviceNode(
            name=name,
            device_type=_canon_device_type(spec.device_type),
            template=_canon_template(spec.template),
            w_um=float(w),
            l_um=float(l),
            fingers=fingers,
            in_nwell=bool(getattr(spec, "in_nwell", False)),
        ))
        name_to_idx[name] = i

    nets: list[NetEdge] = []
    net_name_to_idx: dict[str, int] = {}
    for j, (net_name, net_spec) in enumerate(template.nets.items()):
        nets.append(NetEdge(
            name=net_name,
            net_type=_canon_net_type(net_spec.net_type),
            rail=_canon_rail(net_spec.rail),
            layer_hint=_canon_layer(net_spec.layer)
                       or _hint_layer_for(net_name, template),
            connections=[],
        ))
        net_name_to_idx[net_name] = j

    # Wire device terminals into nets.
    for d_idx, (_name, spec) in enumerate(template.devices.items()):
        for term, net_name in (spec.terminals or {}).items():
            j = net_name_to_idx.get(net_name)
            if j is None:
                # Net mentioned by a device but not declared — skip silently;
                # downstream code already tolerates missing nets.
                continue
            nets[j].connections.append((d_idx, term))

    return TopologyGraph(
        cell_name=template.name,
        devices=devices,
        nets=nets,
    )


# ── Sizing / canonicalisation helpers ────────────────────────────────────────

def _resolve_sizing(
    spec:        DeviceSpec,
    field_name:  str,
    cell_params: dict[str, dict[str, Any]],
    defaults:    dict[str, Any],
) -> float:
    """w / l fallbacks: device YAML > cell_params[name][field] > device-type
    default (``w_N`` for nmos, ``w_P`` for pmos, ``l`` for both) > 0."""
    raw = getattr(spec, field_name, 0.0) or 0.0
    if raw:
        return float(raw)
    name_params = cell_params.get(spec.name) if hasattr(spec, "name") else None
    if name_params and name_params.get(field_name):
        return float(name_params[field_name])
    if field_name == "w":
        if spec.device_type == "nmos" and defaults.get("w_N"):
            return float(defaults["w_N"])
        if spec.device_type == "pmos" and defaults.get("w_P"):
            return float(defaults["w_P"])
    if field_name == "l" and defaults.get("l"):
        return float(defaults["l"])
    return 0.0


def _resolve_field(spec, field_name, cell_params, defaults, *, default=0):
    raw = getattr(spec, field_name, default) or default
    if raw:
        return raw
    name_params = cell_params.get(getattr(spec, "name", ""))
    if name_params and name_params.get(field_name):
        return name_params[field_name]
    return default


def _canon_device_type(t: str) -> str:
    return t if t in DEVICE_TYPES else "other"


def _canon_net_type(t: str) -> str:
    return t if t in NET_TYPES else "other"


def _canon_rail(r: str) -> str:
    if not r:
        return "none"
    return r if r in RAIL_POSITIONS else "none"


def _canon_template(t: str) -> str:
    return t if t in TEMPLATES else "other"


def _canon_layer(layer: str) -> str:
    """Return *layer* iff it's a known LAYER_ROLE; else empty string."""
    if not layer:
        return ""
    return layer if layer in LAYER_ROLES else ""


def _hint_layer_for(net_name: str, template: CellTemplate) -> str:
    """Pull the routing-hint layer for *net_name* if present in the YAML."""
    hint = template.routing_hints.get(net_name)
    if hint and hint.layer in LAYER_ROLES:
        return hint.layer
    return ""


# ── Featurisation (numpy/torch-free; encoder consumes the dataclasses) ──────

# Per-device feature width: device_type one-hot + template one-hot +
# (w, l, fingers, in_nwell) numerics.
DEVICE_FEAT_DIM = len(DEVICE_TYPES) + len(TEMPLATES) + 4

# Per-net feature width: net_type one-hot + rail one-hot + layer-role one-hot.
NET_FEAT_DIM = len(NET_TYPES) + len(RAIL_POSITIONS) + len(LAYER_ROLES)


def device_feature_indices() -> dict[str, int]:
    """Index map for unit-testing the device feature vector layout."""
    base = 0
    out = {}
    for i, t in enumerate(DEVICE_TYPES):
        out[f"type_{t}"] = base + i
    base += len(DEVICE_TYPES)
    for i, t in enumerate(TEMPLATES):
        out[f"template_{t}"] = base + i
    base += len(TEMPLATES)
    out["w_um"]     = base + 0
    out["l_um"]     = base + 1
    out["fingers"]  = base + 2
    out["in_nwell"] = base + 3
    return out


def net_feature_indices() -> dict[str, int]:
    """Index map for unit-testing the net feature vector layout."""
    base = 0
    out = {}
    for i, t in enumerate(NET_TYPES):
        out[f"type_{t}"] = base + i
    base += len(NET_TYPES)
    for i, r in enumerate(RAIL_POSITIONS):
        out[f"rail_{r}"] = base + i
    base += len(RAIL_POSITIONS)
    for i, lyr in enumerate(LAYER_ROLES):
        out[f"layer_{lyr}"] = base + i
    return out


def encode_device(node: DeviceNode) -> Sequence[float]:
    """Plain-Python feature vector for *node* — encoder converts to tensor."""
    out = [0.0] * DEVICE_FEAT_DIM
    out[_DEV_TYPE_IDX.get(node.device_type, _DEV_TYPE_IDX["other"])] = 1.0
    base = len(DEVICE_TYPES)
    out[base + _TEMPLATE_IDX.get(node.template, _TEMPLATE_IDX["other"])] = 1.0
    base += len(TEMPLATES)
    out[base + 0] = float(node.w_um)
    out[base + 1] = float(node.l_um)
    out[base + 2] = float(node.fingers)
    out[base + 3] = 1.0 if node.in_nwell else 0.0
    return out


def encode_net(net: NetEdge) -> Sequence[float]:
    """Plain-Python feature vector for *net* — encoder converts to tensor."""
    out = [0.0] * NET_FEAT_DIM
    out[_NET_TYPE_IDX.get(net.net_type, _NET_TYPE_IDX["other"])] = 1.0
    base = len(NET_TYPES)
    out[base + _RAIL_IDX.get(net.rail, _RAIL_IDX["none"])] = 1.0
    base += len(RAIL_POSITIONS)
    if net.layer_hint:
        out[base + role_index(net.layer_hint)] = 1.0
    return out


__all__ = [
    "DEVICE_TYPES", "NET_TYPES", "RAIL_POSITIONS", "TEMPLATES",
    "DEVICE_FEAT_DIM", "NET_FEAT_DIM",
    "DeviceNode", "NetEdge", "TopologyGraph",
    "graph_from_template",
    "device_feature_indices", "net_feature_indices",
    "encode_device", "encode_net",
]
