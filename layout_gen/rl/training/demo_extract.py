"""
layout_gen.rl.training.demo_extract — synth pipeline → BC demo trajectory.

Runs the existing rule-based :class:`layout_gen.synth.synthesizer.Synthesizer`
on a cell template and converts its ``placed`` dict (one
:class:`PlacedDevice` per topology device) into a PLACE-action sequence
the RL policy can be behaviour-cloned on.

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
      "schema":   "demo-place-1",
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
      ]
    }

The action's ``device_idx`` matches the topology graph's ordering
(first key in the YAML's ``devices`` mapping), so the RL action space's
``device`` dim can be encoded directly from it.

Phase 4 part 6 keeps this PLACE-only — ROUTE demos require attributing
synth's wire output to specific nets, which is fiddlier and the new
``electrical_delta`` reward already gives ROUTE good signal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from layout_gen.pdk import load_pdk
from layout_gen.synth.loader import CellTemplate, load_template
from layout_gen.synth.synthesizer import Synthesizer

from layout_gen.rl.topology.parser import graph_from_template


# ── Demo container ──────────────────────────────────────────────────────────

@dataclass
class PlacementDemo:
    """One synth-derived demo — a list of PLACE actions for a cell."""
    template:        str
    cell_width_um:   float
    cell_height_um:  float
    cell_params:     dict
    actions:         list[dict]

    def to_dict(self) -> dict:
        return {
            "schema":         "demo-place-1",
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
    name_to_idx = graph.device_index()

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
