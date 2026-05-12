"""
layout_gen.rl.training.demo_dataset — BC dataset over PLACE-action demos.

Walks each demo (a JSON written by :mod:`layout_gen.rl.training.demo_extract`)
through a simulated env and yields one ``(observation, action_label,
validity)`` sample per PLACE step. The result plugs into the existing
:class:`~layout_gen.rl.training.bc_pretrain.BCTrainer` directly — no
trainer changes needed for v1.

Design choices
--------------
* The simulated env uses a no-op DRC. Demos are already DRC-clean, so
  step-by-step DRC runs would only slow training without adding signal.
* Action labels go through the same MultiDiscrete encoding the live env
  uses, so BC-trained checkpoints load straight into MaskablePPO.
* Coordinate quantisation: demos store continuous (x_um, y_um); we
  round to the nearest x_bin / y_bin per the env's discretisation.
  Quantisation error ≤ ``cell_dim / (2 · n_bins)``.
* Per-sample validity flags mark which action dims contribute to the
  loss for that sample. PLACE samples mark
  ``{kind, device, x_bin, y_bin, orient}`` valid; everything else
  (REPAIR-only fields) is masked off so the BC objective doesn't
  pretend to know what an inappropriate dim should be.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from layout_gen.pdk import load_pdk
from layout_gen.synth.geo.state import LayoutState
from layout_gen.synth.loader import load_template

from layout_gen.rl.env.action_space import (
    N_PLACE_KINDS, N_REPAIR_KINDS,
)
from layout_gen.rl.env.observation import (
    DEFAULT_POLY_CAP, DEFAULT_VIOL_CAP, build_observation,
)
from layout_gen.rl.env.place_action import (
    N_ORIENTATIONS, ORIENTATIONS, TransistorCache, place_device_full,
)
from layout_gen.rl.env.route_action import (
    DEFAULT_SIZE_BINS, N_ROUTE_LAYERS, ROUTE_LAYERS, add_route_segment,
    size_bins,
)
from layout_gen.rl.training.demo_extract import PlacementDemo, read_demo
from layout_gen.rl.topology import graph_from_template


def _coord_to_bin(value: float, span: float, n_bins: int) -> int:
    """Inverse of ``ActionSpace._bin_to_coord``: nearest bin index."""
    if span <= 0 or n_bins <= 0:
        return 0
    raw = value / span * n_bins - 0.5
    return max(0, min(n_bins - 1, int(round(raw))))


def _size_to_bin(value: float, n_bins: int) -> int:
    """Inverse of :func:`size_bins`: nearest log-spaced size bin."""
    if n_bins <= 0:
        return 0
    table = size_bins(n_bins)
    diffs = np.abs(table - float(value))
    return int(diffs.argmin())


# Default-disabled validity entry used to mask off action dims that
# don't apply to a given sample.
_REPAIR_DIM_NAMES = ("target", "edge", "sign_x", "sign_y", "mag")
_PLACE_DIM_NAMES  = ("device", "x_bin", "y_bin", "orient")
_ROUTE_DIM_NAMES  = ("net", "route_layer",
                     "route_x_bin", "route_y_bin",
                     "route_w_bin", "route_h_bin")


def _default_action_labels() -> dict[str, int]:
    return {
        "kind":   0, "target": 0, "edge":   0, "sign_x": 0, "sign_y": 0,
        "mag":    0, "device": 0, "x_bin":  0, "y_bin":  0, "orient": 0,
        "net":    0, "route_layer":  0,
        "route_x_bin": 0, "route_y_bin": 0,
        "route_w_bin": 0, "route_h_bin": 0,
    }


def _default_validity() -> dict[str, bool]:
    return {
        "kind":   False, "target": False, "edge":   False,
        "sign_x": False, "sign_y": False, "mag":    False,
        "device": False, "x_bin":  False, "y_bin":  False, "orient": False,
        "net":    False, "route_layer":  False,
        "route_x_bin": False, "route_y_bin": False,
        "route_w_bin": False, "route_h_bin": False,
    }


# ── Dataset ─────────────────────────────────────────────────────────────────

class PlacementDemoDataset(Dataset):
    """Per-step BC dataset over a list of demo JSONs.

    Constructor walks each demo once, simulating the env forward and
    recording one sample per PLACE step. ``__getitem__`` returns those
    cached samples — no env calls during training.

    Parameters
    ----------
    demo_paths :
        Iterable of paths to ``*.demo.json`` files written by
        :mod:`layout_gen.rl.training.demo_extract`.
    poly_cap, viol_cap :
        Observation-space caps (must match the policy used downstream).
    device_cap, x_bins, y_bins :
        Action-space PLACE discretisation (must match the policy).
    net_cap, route_x_bins, route_y_bins, route_w_bins, route_h_bins :
        Action-space ROUTE discretisation. Set to match the policy /
        env you'll train against. When all ROUTE bin counts are zero
        the dataset behaves as a PLACE-only dataset (legacy
        ``demo-place-1`` demos are silently truncated to their PLACE
        actions).
    rules :
        :class:`PDKRules` (defaults to ``load_pdk()``).
    """

    def __init__(
        self,
        demo_paths:   Sequence[Path] | Sequence[str],
        *,
        poly_cap:     int = DEFAULT_POLY_CAP,
        viol_cap:     int = DEFAULT_VIOL_CAP,
        device_cap:   int = 16,
        x_bins:       int = 8,
        y_bins:       int = 8,
        net_cap:      int = 16,
        route_x_bins: int = 8,
        route_y_bins: int = 8,
        route_w_bins: int = DEFAULT_SIZE_BINS,
        route_h_bins: int = DEFAULT_SIZE_BINS,
        rules = None,
    ) -> None:
        self.poly_cap     = poly_cap
        self.viol_cap     = viol_cap
        self.device_cap   = device_cap
        self.x_bins       = x_bins
        self.y_bins       = y_bins
        self.net_cap      = net_cap
        self.route_x_bins = route_x_bins
        self.route_y_bins = route_y_bins
        self.route_w_bins = route_w_bins
        self.route_h_bins = route_h_bins
        self._rules       = rules or load_pdk()
        self._cache       = TransistorCache(self._rules)

        self._samples: list[dict] = []
        for path in demo_paths:
            self._ingest(read_demo(Path(path)))

    def _ingest(self, demo: PlacementDemo) -> None:
        template = load_template(demo.template)
        defaults = dict(demo.cell_params)
        graph = graph_from_template(
            template, cell_params={"_defaults": defaults},
        )
        # Simulate forward.
        state = LayoutState()
        terminals: dict[tuple[int, str], tuple[float, float, str]] = {}

        for action in demo.actions:
            kind = action.get("kind")
            sample = None
            if kind == "place_device":
                sample = self._build_place_sample(
                    action, demo, state, graph,
                )
                if sample is None:
                    break
                # Advance state.
                d_idx = int(action["device_idx"])
                device = graph.devices[d_idx]
                orient = action.get("orientation", "R0")
                try:
                    _, ports = place_device_full(
                        state, device,
                        x_um=float(action["x_um"]),
                        y_um=float(action["y_um"]),
                        orientation=orient if orient in ORIENTATIONS else "R0",
                        cache=self._cache,
                    )
                    for term_name, pos in ports.items():
                        terminals[(d_idx, term_name)] = pos
                except Exception:
                    break
            elif kind == "route_segment":
                sample = self._build_route_sample(action, demo, state, graph)
                if sample is None:
                    # Unknown net / layer / bins not configured — skip
                    # the sample but keep ingesting the rest.
                    continue
                # Advance state with the route geometry.
                try:
                    add_route_segment(
                        state,
                        layer=str(action["layer"]),
                        x_um=float(action["x_um"]),
                        y_um=float(action["y_um"]),
                        w_um=float(action["w_um"]),
                        h_um=float(action["h_um"]),
                        net_name=str(action.get("net_name", "")),
                    )
                except Exception:
                    break
            else:
                continue

            self._samples.append(sample)

    def _build_place_sample(
        self,
        action: dict,
        demo:   PlacementDemo,
        state:  LayoutState,
        graph,
    ) -> dict | None:
        d_idx = int(action["device_idx"])
        if d_idx < 0 or d_idx >= graph.n_devices:
            return None

        obs_struct = build_observation(
            state, [], poly_cap=self.poly_cap, viol_cap=self.viol_cap,
        )
        labels   = _default_action_labels()
        validity = _default_validity()

        # kind: 'place_device' is the first PLACE kind, sitting at
        # index N_REPAIR_KINDS in the combined kinds tuple.
        labels["kind"]   = N_REPAIR_KINDS
        validity["kind"] = True

        labels["device"]   = min(d_idx, self.device_cap - 1)
        validity["device"] = True

        labels["x_bin"] = _coord_to_bin(
            float(action["x_um"]),
            demo.cell_width_um, self.x_bins,
        )
        labels["y_bin"] = _coord_to_bin(
            float(action["y_um"]),
            demo.cell_height_um, self.y_bins,
        )
        validity["x_bin"] = True
        validity["y_bin"] = True

        orient = action.get("orientation", "R0")
        if orient in ORIENTATIONS:
            labels["orient"]   = ORIENTATIONS.index(orient)
            validity["orient"] = True

        return {
            "obs":      obs_struct.to_dict(),
            "action":   labels,
            "validity": validity,
        }

    def _build_route_sample(
        self,
        action: dict,
        demo:   PlacementDemo,
        state:  LayoutState,
        graph,
    ) -> dict | None:
        """Build a BC sample for one ``route_segment`` action.

        Returns None when the route action can't be mapped to the
        configured action-space dims (e.g. an unknown layer or a
        net_idx outside ``net_cap``); those samples are dropped
        instead of producing a noisy label.
        """
        if self.route_x_bins <= 0 or self.route_y_bins <= 0:
            return None
        n_idx = int(action.get("net_idx", -1))
        if n_idx < 0 or n_idx >= graph.n_nets or n_idx >= self.net_cap:
            return None
        layer = str(action.get("layer", ""))
        if layer not in ROUTE_LAYERS:
            return None

        obs_struct = build_observation(
            state, [], poly_cap=self.poly_cap, viol_cap=self.viol_cap,
        )
        labels   = _default_action_labels()
        validity = _default_validity()

        # kind: ROUTE kinds sit after REPAIR + PLACE.
        labels["kind"]   = N_REPAIR_KINDS + N_PLACE_KINDS
        validity["kind"] = True

        labels["net"]   = n_idx
        validity["net"] = True

        labels["route_layer"]   = ROUTE_LAYERS.index(layer)
        validity["route_layer"] = True

        labels["route_x_bin"] = _coord_to_bin(
            float(action["x_um"]),
            demo.cell_width_um, self.route_x_bins,
        )
        labels["route_y_bin"] = _coord_to_bin(
            float(action["y_um"]),
            demo.cell_height_um, self.route_y_bins,
        )
        validity["route_x_bin"] = True
        validity["route_y_bin"] = True

        labels["route_w_bin"] = _size_to_bin(
            float(action["w_um"]), self.route_w_bins,
        )
        labels["route_h_bin"] = _size_to_bin(
            float(action["h_um"]), self.route_h_bins,
        )
        validity["route_w_bin"] = True
        validity["route_h_bin"] = True

        return {
            "obs":      obs_struct.to_dict(),
            "action":   labels,
            "validity": validity,
        }

    # ── PyTorch Dataset API ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        return {
            "obs":      {k: torch.from_numpy(v) for k, v in s["obs"].items()},
            "action":   {k: torch.tensor(v, dtype=torch.long)
                         for k, v in s["action"].items()},
            "validity": {k: torch.tensor(v, dtype=torch.bool)
                         for k, v in s["validity"].items()},
        }


__all__ = ["PlacementDemoDataset"]
