"""
layout_gen.rl.training.demo_extract — synth pipeline → BC demo trajectory.

Runs the existing rule-based :class:`layout_gen.synth.synthesizer.Synthesizer`
on a cell template and converts its ``placed`` dict into a sequence of
PLACE actions, then computes one ROUTE action per net (covering all
placed terminals of that net) so the BC trainer can pretrain both the
PLACE and ROUTE heads of the policy.

Why bother
----------
The synthesizer produces DRC-clean, gate-aligned, conventional-axis
layouts of every cell template we have. Without demos, the RL policy
has to discover these conventions from scratch under sparse reward
shaping; with demos it starts from a near-correct policy and PPO
only has to refine.

Output format
-------------
Each demo is one JSON file with shape::

    {
      "schema":   "demo-place-route-1",
      "template": "inverter",
      "cell_width_um":  4.0,
      "cell_height_um": 2.0,
      "cell_params":    {"w_N": 0.5, "w_P": 0.5, "l": 0.15},
      "actions": [
        {"kind": "place_device",
         "device_name": "N",
         "device_idx":  0,
         "x_um":   0.0,
         "y_um":   0.0,
         "orientation": "R0"},
        ...
        {"kind":      "route_segment",
         "net_name":  "OUT",
         "net_idx":   2,
         "layer":     "met1",
         "x_um":      0.615,
         "y_um":      0.505,
         "w_um":      0.20,
         "h_um":      1.45},
        ...
      ]
    }

The action's ``device_idx`` / ``net_idx`` match the topology graph's
ordering (first key in the YAML's ``devices`` / ``nets`` mapping), so
the RL action space's ``device`` / ``net`` dims can be encoded
directly.

Schema versions
---------------
* ``demo-place-1`` (legacy)        — PLACE actions only.
* ``demo-place-route-1`` (current) — PLACE + per-net ROUTE actions.
  Older PLACE-only demos still load via :func:`read_demo`.

ROUTE-action policy
-------------------
For each net with at least one placed terminal, the extractor emits
one ROUTE action whose rectangle is the bbox of those terminals on
the net's preferred layer (from the topology graph's ``layer_hint``
when set, else a sensible default: ``met1`` for power rails, ``li1``
for everything else). This is a *skeleton* — synth's actual route
geometry is generally more elaborate, but the bbox is sufficient to
land all terminals of a net inside the segment, which is what the
``electrical_delta`` reward checks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from layout_gen.pdk import load_pdk
from layout_gen.synth.geo.state import LayoutState
from layout_gen.synth.loader import CellTemplate, load_template
from layout_gen.synth.synthesizer import Synthesizer

from layout_gen.rl.env.place_action import TransistorCache, place_device_full
from layout_gen.rl.topology.parser import TopologyGraph, graph_from_template


# ── Demo container ──────────────────────────────────────────────────────────

@dataclass
class PlacementDemo:
    """One synth-derived demo — a sequence of PLACE then ROUTE actions
    for a cell. Each action dict has a ``kind`` of ``"place_device"`` or
    ``"route_segment"``; see the module docstring for the schema."""
    template:        str
    cell_width_um:   float
    cell_height_um:  float
    cell_params:     dict
    actions:         list[dict]

    def to_dict(self) -> dict:
        return {
            "schema":         "demo-place-route-1",
            "template":       self.template,
            "cell_width_um":  self.cell_width_um,
            "cell_height_um": self.cell_height_um,
            "cell_params":    self.cell_params,
            "actions":        self.actions,
        }


# ── Helpers ─────────────────────────────────────────────────────────────────

def _orientation_for(template: CellTemplate, device_name: str) -> str:
    """Look up the device's orientation from the template's placement
    directives. Defaults to R0 (the placer's default for unannotated
    devices)."""
    for d in template.placement_directives:
        if d.name == device_name:
            return d.orientation or "R0"
    return "R0"


def _cell_dimensions(template: CellTemplate,
                      *, default_w: float = 4.0,
                      default_h: float = 2.0) -> tuple[float, float]:
    w = float(template.cell_dimensions.width  or default_w)
    h = float(template.cell_dimensions.height or default_h)
    return w, h


# Route-layer order must match :data:`layout_gen.rl.env.route_action.ROUTE_LAYERS`.
_ROUTE_LAYERS_ORDER: tuple[str, ...] = ("li1", "met1", "met2", "met3")
_DEFAULT_POWER_LAYER  = "met1"
_DEFAULT_SIGNAL_LAYER = "li1"


def _route_layer_for_net(net) -> str:
    """Pick a representative routing layer for *net*.

    Preference order:
      1. Net's ``layer_hint`` if it's in :data:`_ROUTE_LAYERS_ORDER`.
      2. ``met1`` for power rails (top/bottom rail position).
      3. ``li1`` for everything else (the synth pipeline's default for
         signal nets in sky130).
    """
    hint = (net.layer_hint or "").strip()
    if hint in _ROUTE_LAYERS_ORDER:
        return hint
    if net.rail in ("top", "bottom"):
        return _DEFAULT_POWER_LAYER
    return _DEFAULT_SIGNAL_LAYER


def _build_route_actions(
    graph:     TopologyGraph,
    terminals: dict[tuple[int, str], tuple[float, float, str]],
) -> list[dict]:
    """For each net with ≥1 placed terminal, return one ROUTE action
    whose rectangle covers all of that net's terminal positions on the
    net's preferred layer.

    Single-terminal nets get a small fixed rect centred on the
    terminal — :func:`add_route_segment` won't error on it and the
    BC label is still meaningful (the policy learns "land a small
    pad on the terminal").
    """
    out: list[dict] = []
    for n_idx, net in enumerate(graph.nets):
        pts = [terminals[(d_idx, term)]
               for (d_idx, term) in net.connections
               if (d_idx, term) in terminals]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max(max_x - min_x, 0.20)   # at least ROUTE_SIZE_MIN_UM = 0.10 µm
        h = max(max_y - min_y, 0.20)
        out.append({
            "kind":     "route_segment",
            "net_name": net.name,
            "net_idx":  n_idx,
            "layer":    _route_layer_for_net(net),
            "x_um":     float(min_x),
            "y_um":     float(min_y),
            "w_um":     float(w),
            "h_um":     float(h),
        })
    return out


def _simulate_place_for_terminals(
    graph:      TopologyGraph,
    actions:    list[dict],
    cache:      TransistorCache,
) -> dict[tuple[int, str], tuple[float, float, str]]:
    """Replay the PLACE actions through :func:`place_device_full` to
    recover each terminal's (x, y, layer). Used by the ROUTE-action
    builder so its rects match what the env's terminal-tracking
    machinery would see during BC training."""
    state = LayoutState()
    terminals: dict[tuple[int, str], tuple[float, float, str]] = {}
    for action in actions:
        if action.get("kind") != "place_device":
            continue
        d_idx = int(action["device_idx"])
        if not 0 <= d_idx < graph.n_devices:
            continue
        device = graph.devices[d_idx]
        orient = action.get("orientation", "R0")
        try:
            _, ports = place_device_full(
                state, device,
                x_um=float(action["x_um"]),
                y_um=float(action["y_um"]),
                orientation=orient,
                cache=cache,
            )
        except Exception:
            continue
        for term, (px, py, layer) in ports.items():
            terminals[(d_idx, term)] = (float(px), float(py), str(layer))
    return terminals


# ── Extractor ───────────────────────────────────────────────────────────────

def extract_placement_demo(
    template_name: str,
    *,
    cell_params:   dict | None = None,
    rules = None,
) -> PlacementDemo:
    """Run synth on *template_name* and record the resulting PLACE actions.

    Parameters
    ----------
    template_name :
        Cell template name resolvable via :func:`load_template`.
    cell_params :
        Sizing override dict (e.g. ``{"w_N": 0.5, "w_P": 0.5, "l": 0.15}``).
        Defaults to a sensible w_N / w_P / l that lets the synth complete
        without DRC retries.
    rules :
        :class:`PDKRules`. Defaults to ``load_pdk()``.

    Returns
    -------
    PlacementDemo :
        One PLACE action per device in the topology graph's order.
    """
    rules = rules or load_pdk()
    template = load_template(template_name)
    params   = cell_params or {"w_N": 0.5, "w_P": 0.5, "l": 0.15}

    synth = Synthesizer(rules)
    result = synth.synthesize(template, params)

    cell_w, cell_h = _cell_dimensions(template)
    graph = graph_from_template(template, cell_params={"_defaults": params})

    actions: list[dict] = []
    # Iterate in topology-graph order so device_idx matches the action
    # space's device dim during BC training.
    for d_idx, d_node in enumerate(graph.devices):
        placed = result.placed.get(d_node.name)
        if placed is None:
            # Synth didn't place this device — skip it. The demo is
            # still useful for the devices that did make it.
            continue
        orient = _orientation_for(template, d_node.name)
        actions.append({
            "kind":         "place_device",
            "device_name":  d_node.name,
            "device_idx":   d_idx,
            "x_um":         float(placed.x),
            "y_um":         float(placed.y),
            "orientation":  orient,
        })

    # ROUTE actions are computed from the *replayed* terminal positions,
    # not from synth's wire geometry. That keeps the BC labels aligned
    # with whatever ``place_device_full`` will produce at training
    # time — synth's actual wires can take more circuitous paths that
    # the single-segment action space wouldn't reproduce exactly.
    cache = TransistorCache(rules)
    terminals = _simulate_place_for_terminals(graph, actions, cache)
    actions.extend(_build_route_actions(graph, terminals))

    return PlacementDemo(
        template=template_name,
        cell_width_um=cell_w,
        cell_height_um=cell_h,
        cell_params=params,
        actions=actions,
    )


def write_demo(demo: PlacementDemo, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(demo.to_dict(), indent=2))


def read_demo(path: Path) -> PlacementDemo:
    raw = json.loads(Path(path).read_text())
    return PlacementDemo(
        template=raw["template"],
        cell_width_um=raw["cell_width_um"],
        cell_height_um=raw["cell_height_um"],
        cell_params=raw["cell_params"],
        actions=list(raw["actions"]),
    )


def extract_many(
    template_names: Iterable[str],
    out_dir:        Path,
    *,
    cell_params:    dict | None = None,
    rules = None,
) -> list[Path]:
    """Bulk-extract: one JSON per template, returned as a list of paths."""
    out_paths: list[Path] = []
    for name in template_names:
        demo = extract_placement_demo(name, cell_params=cell_params, rules=rules)
        path = out_dir / f"{name}.demo.json"
        write_demo(demo, path)
        out_paths.append(path)
    return out_paths


__all__ = [
    "PlacementDemo",
    "extract_placement_demo",
    "extract_many",
    "write_demo", "read_demo",
]
