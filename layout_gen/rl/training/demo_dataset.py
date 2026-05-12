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
    N_REPAIR_KINDS,
)
from layout_gen.rl.env.observation import (
    DEFAULT_POLY_CAP, DEFAULT_VIOL_CAP, build_observation,
)
from layout_gen.rl.env.place_action import (
    N_ORIENTATIONS, ORIENTATIONS, TransistorCache, place_device_full,
)
from layout_gen.rl.training.demo_extract import PlacementDemo, read_demo
from layout_gen.rl.topology import graph_from_template


def _coord_to_bin(value: float, span: float, n_bins: int) -> int:
    """Inverse of ``ActionSpace._bin_to_coord``: nearest bin index."""
    if span <= 0 or n_bins <= 0:
        return 0
    raw = value / span * n_bins - 0.5
    return max(0, min(n_bins - 1, int(round(raw))))


# Default-disabled validity entry used to mask off action dims that
# don't apply to a PLACE-only sample.
_REPAIR_DIM_NAMES   = ("target", "edge", "sign_x", "sign_y", "mag")
_PLACE_DIM_NAMES    = ("device", "x_bin", "y_bin", "orient")


def _default_action_labels() -> dict[str, int]:
    return {
        "kind":   0, "target": 0, "edge":   0, "sign_x": 0, "sign_y": 0,
        "mag":    0, "device": 0, "x_bin":  0, "y_bin":  0, "orient": 0,
    }


def _default_validity() -> dict[str, bool]:
    return {
        "kind":   False, "target": False, "edge":   False,
        "sign_x": False, "sign_y": False, "mag":    False,
        "device": False, "x_bin":  False, "y_bin":  False, "orient": False,
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
        Action-space discretisation (must match the policy).
    rules :
        :class:`PDKRules` (defaults to ``load_pdk()``).
    """

    def __init__(
        self,
        demo_paths:  Sequence[Path] | Sequence[str],
        *,
        poly_cap:    int = DEFAULT_POLY_CAP,
        viol_cap:    int = DEFAULT_VIOL_CAP,
        device_cap:  int = 16,
        x_bins:      int = 8,
        y_bins:      int = 8,
        rules = None,
    ) -> None:
        self.poly_cap   = poly_cap
        self.viol_cap   = viol_cap
        self.device_cap = device_cap
        self.x_bins     = x_bins
        self.y_bins     = y_bins
        self._rules     = rules or load_pdk()
        self._cache     = TransistorCache(self._rules)

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
            if action["kind"] != "place_device":
                continue
            d_idx = int(action["device_idx"])
            if d_idx < 0 or d_idx >= graph.n_devices:
                continue
            device = graph.devices[d_idx]

            # Build the BC sample BEFORE applying the action.
            obs_struct = build_observation(
                state, [], poly_cap=self.poly_cap, viol_cap=self.viol_cap,
            )
            obs_dict = obs_struct.to_dict()

            labels   = _default_action_labels()
            validity = _default_validity()

            # kind: 'place_device' is the first PLACE kind, sitting at
            # index N_REPAIR_KINDS in the combined kinds tuple.
            labels["kind"]   = N_REPAIR_KINDS
            validity["kind"] = True

            # device pointer
            labels["device"]   = min(d_idx, self.device_cap - 1)
            validity["device"] = True

            # x_bin / y_bin
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

            # orientation
            orient = action.get("orientation", "R0")
            if orient in ORIENTATIONS:
                labels["orient"]   = ORIENTATIONS.index(orient)
                validity["orient"] = True

            self._samples.append({
                "obs":      obs_dict,
                "action":   labels,
                "validity": validity,
            })

            # Apply the action to advance the simulated env state.
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
                # Skip the rest of this demo on a bad action — it's
                # better to drop a noisy demo than to corrupt the
                # remaining samples in the corpus.
                break

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
