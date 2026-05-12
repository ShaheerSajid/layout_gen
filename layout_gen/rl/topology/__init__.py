"""
layout_gen.rl.topology — netlist + placement-intent encoder.

Phase 4 module. Converts a :class:`layout_gen.synth.loader.CellTemplate`
(parsed from a topology YAML) into a graph the policy can condition on,
then encodes it with a small GNN.

The graph is the conditioning signal that lets the policy generate
*different cells* — without it, the policy only knows what's currently
on the layout, not what cell it's supposed to be building.

Public API
----------
:class:`TopologyGraph`     — netlist as devices (nodes) + nets (hyperedges).
:func:`graph_from_template` — CellTemplate → TopologyGraph.
:class:`TopologyEncoder`   — GNN producing per-device + global embeddings.
:class:`TopologyEncoderConfig`
"""
from __future__ import annotations

from layout_gen.rl.topology.parser import (
    DeviceNode, NetEdge, TopologyGraph,
    DEVICE_TYPES, NET_TYPES, RAIL_POSITIONS,
    graph_from_template,
)
from layout_gen.rl.topology.encoder import (
    TopologyEncoder, TopologyEncoderConfig,
)

__all__ = [
    "DeviceNode", "NetEdge", "TopologyGraph",
    "DEVICE_TYPES", "NET_TYPES", "RAIL_POSITIONS",
    "graph_from_template",
    "TopologyEncoder", "TopologyEncoderConfig",
]
