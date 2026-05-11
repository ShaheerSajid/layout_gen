"""
layout_gen.rl.training.dataset — load mined trajectories for BC pretrain.

A trajectory file (as written by :mod:`layout_gen.repair.mine_trajectories`)
has this shape::

    {
      "schema":          1,
      "seed_pdk":        "sky130A",
      "seed_cell":       "sky130_fd_sc_hd__inv_1",
      "k":               3,
      "perturbed_state": [{"rid": …, "layer": …, "x0": …, …}, …],
      "forward_action_sequence":  [<perturb action>, …],
      "inverse_action_sequence":  [<repair action>, …],
      "n_violations":   12,
      "violation_rules": ["licon.13", "npc.2"]
    }

The dataset expands each k-step trajectory into k samples. Sample *i*'s
state is the original ``perturbed_state`` with ``inverse[0..i-1]`` applied
in order; its target action is ``inverse[i]``.

Violation context
-----------------
The mined JSON does not include per-violation positions (just rule names
and count). For BC we synthesise one violation at the centroid of the
action's target polygon — this matches the diffusion baseline's
"violation-XY-as-conditioning" trick. A custom callable can be plugged
in via ``violation_source`` to use a real DRC runner instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from layout_gen.drc.base import DRCViolation
from layout_gen.repair import perturb as perturb_lib
from layout_gen.repair.perturb import PerturbAction
from layout_gen.synth.geo.state import LayoutState, Rect

from layout_gen.rl.env.action_space import (
    ACTION_KINDS, EDGE_NAMES, DEFAULT_MAG_BINS, DEFAULT_TARGET_CAP,
    OFFGRID_SCALE, magnitude_bins,
)
from layout_gen.rl.env.observation import (
    DEFAULT_POLY_CAP, DEFAULT_VIOL_CAP, build_observation,
)


ViolationSource = Callable[[LayoutState, dict], list[DRCViolation]]


# ── Sample dataclass (held only inside the dataset; loader returns dicts) ────

@dataclass
class TrajectorySample:
    obs:      dict[str, np.ndarray]
    action:   dict[str, int]
    validity: dict[str, bool]


# ── Action encoder ───────────────────────────────────────────────────────────

def encode_action_dict(
    action: dict,
    rid_to_idx: dict[int, int],
    *,
    mag_table: np.ndarray | None = None,
) -> tuple[dict[str, int], dict[str, bool]]:
    """Encode a serialised :class:`PerturbAction` into per-dim labels.

    Returns
    -------
    labels : dict[str, int]
        Per-dim integer label. Don't-care fields are filled with 0.
    validity : dict[str, bool]
        True iff the dim's label should contribute to the loss for this
        sample.
    """
    if mag_table is None:
        mag_table = magnitude_bins()

    kind = action["kind"]
    params = action.get("params") or {}
    target_rid = int(action.get("target", -1))
    target_idx = rid_to_idx.get(target_rid, -1)

    labels: dict[str, int] = {
        "kind":   ACTION_KINDS.index(kind) if kind in ACTION_KINDS else 0,
        "target": max(target_idx, 0),
        "edge":   0,
        "sign_x": 0,
        "sign_y": 0,
        "mag":    0,
    }
    validity: dict[str, bool] = {
        "kind":   kind in ACTION_KINDS,
        "target": target_idx >= 0,
        "edge":   False,
        "sign_x": False,
        "sign_y": False,
        "mag":    False,
    }

    if kind == "shift_edge":
        side  = params.get("side", "left")
        delta = float(params.get("delta", 0.0))
        labels["edge"]   = EDGE_NAMES.index(side) if side in EDGE_NAMES else 0
        validity["edge"] = side in EDGE_NAMES
        labels["sign_y"] = 1 if delta >= 0 else 0
        validity["sign_y"] = True
        labels["mag"]    = int(np.argmin(np.abs(mag_table - abs(delta))))
        validity["mag"]  = True

    elif kind in ("shrink_rect", "grow_rect"):
        delta = float(params.get("delta", 0.0))
        labels["mag"]   = int(np.argmin(np.abs(mag_table - abs(delta))))
        validity["mag"] = True

    elif kind in ("translate", "nudge_offgrid"):
        dx = float(params.get("dx", 0.0))
        dy = float(params.get("dy", 0.0))
        # The action space encodes dx/dy as separate signs + a single
        # magnitude bin (axis-aligned). Use the dominant component for
        # the magnitude target; the sign labels cover both axes.
        mag = max(abs(dx), abs(dy))
        if kind == "nudge_offgrid":
            mag = mag / OFFGRID_SCALE   # un-rescale the offgrid mag
        labels["sign_x"] = 1 if dx >= 0 else 0
        labels["sign_y"] = 1 if dy >= 0 else 0
        labels["mag"]    = int(np.argmin(np.abs(mag_table - mag)))
        validity["sign_x"] = True
        validity["sign_y"] = True
        validity["mag"]    = True

    # delete_rect: only kind+target are valid.

    return labels, validity


# ── State reconstruction ─────────────────────────────────────────────────────

def _state_from_serialised(rects: list[dict]) -> LayoutState:
    """Rebuild a LayoutState while preserving the original rectangle IDs
    so action targets resolve against the same rid space as the trajectory."""
    return LayoutState([
        Rect(
            rid=int(r["rid"]), layer=r["layer"],
            x0=r["x0"], y0=r["y0"], x1=r["x1"], y1=r["y1"],
            net=r.get("net", ""),
            shape_type=r.get("shape_type", ""),
            group_id=int(r.get("group_id", -1)),
        )
        for r in rects
    ])


def _action_from_dict(d: dict) -> PerturbAction:
    return PerturbAction(
        kind=d["kind"],
        target=int(d.get("target", -1)),
        params=dict(d.get("params") or {}),
        snapshot=d.get("snapshot"),
    )


# ── Default violation source ─────────────────────────────────────────────────

def synthetic_violation_at_target(
    state: LayoutState, action_dict: dict,
) -> list[DRCViolation]:
    """One pseudo-violation centred on the action's target rect.

    The diffusion baseline showed that conditioning the policy on the
    violation's (x, y) is the single biggest signal for predicting the
    next repair action; this helper makes that conditioning available
    when the trajectory JSON doesn't carry per-violation positions.
    """
    rid = int(action_dict.get("target", -1))
    if rid < 0 or rid not in state:
        return []
    r = state[rid]
    return [DRCViolation(
        rule="synth.target",
        description="synthetic violation at action target centroid",
        layer=r.layer,
        x=r.cx, y=r.cy,
        value=0.05,
    )]


# ── Dataset ──────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """Dataset of (obs, action, validity) tuples expanded from trajectory JSONs.

    Parameters
    ----------
    trajectory_dir :
        Directory containing ``*.json`` trajectory files (recursive).
    poly_cap, viol_cap, target_cap, mag_bins :
        Must match the ``LayoutEnv`` they will be trained against.
    violation_source :
        Callable ``(state, action_dict) -> list[DRCViolation]``.
        Defaults to :func:`synthetic_violation_at_target`.
    max_trajectories :
        Optional upper bound on the number of JSON files to load.
        Useful for fast smoke tests.
    """

    def __init__(
        self,
        trajectory_dir: str | Path,
        *,
        poly_cap:        int = DEFAULT_POLY_CAP,
        viol_cap:        int = DEFAULT_VIOL_CAP,
        target_cap:      int = DEFAULT_TARGET_CAP,
        mag_bins:        int = DEFAULT_MAG_BINS,
        violation_source: ViolationSource | None = None,
        max_trajectories: int | None = None,
    ) -> None:
        self.trajectory_dir = Path(trajectory_dir)
        self.poly_cap   = poly_cap
        self.viol_cap   = viol_cap
        self.target_cap = target_cap
        self.mag_bins   = mag_bins
        self._mag_table = magnitude_bins(mag_bins)
        self._viol_src  = violation_source or synthetic_violation_at_target

        self._samples: list[TrajectorySample] = []
        self._build(max_trajectories)

    def _build(self, cap: int | None) -> None:
        files = sorted(self.trajectory_dir.rglob("*.json"))
        if cap is not None:
            files = files[:cap]
        for path in files:
            try:
                rec = json.loads(path.read_text())
            except Exception:
                continue
            self._expand_record(rec)

    def _expand_record(self, rec: dict) -> None:
        try:
            state = _state_from_serialised(rec["perturbed_state"])
            inverse = [_action_from_dict(a) for a in rec["inverse_action_sequence"]]
        except KeyError:
            return

        for i, act in enumerate(inverse):
            sample = self._sample_from_state(state, act)
            if sample is not None:
                self._samples.append(sample)
            try:
                perturb_lib.apply(state, act)
            except Exception:
                # Step couldn't apply; skip the rest of this trajectory.
                break

    def _sample_from_state(self,
                           state: LayoutState,
                           action: PerturbAction) -> TrajectorySample | None:
        # Build the observation. Synthetic violations populate viol_feats so
        # the policy gets a "where to look" hint at training time.
        action_dict = action.to_dict()
        violations = self._viol_src(state, action_dict)
        obs_struct = build_observation(
            state, violations,
            poly_cap=self.poly_cap,
            viol_cap=self.viol_cap,
        )

        labels, validity = encode_action_dict(
            action_dict, obs_struct.rid_to_idx,
            mag_table=self._mag_table,
        )

        # Truncate target labels into [0, target_cap).
        labels["target"] = min(labels["target"], self.target_cap - 1)

        return TrajectorySample(
            obs=obs_struct.to_dict(),
            action=labels,
            validity=validity,
        )

    # ── PyTorch Dataset API ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        return {
            "obs":      {k: torch.from_numpy(v) for k, v in s.obs.items()},
            "action":   {k: torch.tensor(v, dtype=torch.long) for k, v in s.action.items()},
            "validity": {k: torch.tensor(v, dtype=torch.bool) for k, v in s.validity.items()},
        }


# ── Collate ──────────────────────────────────────────────────────────────────

def collate_samples(batch: Iterable[dict]) -> dict:
    """Stack samples into batched dicts. Default collate_fn for DataLoader."""
    batch = list(batch)
    obs_keys      = batch[0]["obs"].keys()
    action_keys   = batch[0]["action"].keys()
    validity_keys = batch[0]["validity"].keys()
    out = {
        "obs":      {k: torch.stack([b["obs"][k] for b in batch])      for k in obs_keys},
        "action":   {k: torch.stack([b["action"][k] for b in batch])   for k in action_keys},
        "validity": {k: torch.stack([b["validity"][k] for b in batch]) for k in validity_keys},
    }
    return out


__all__ = [
    "TrajectoryDataset", "TrajectorySample",
    "encode_action_dict",
    "synthetic_violation_at_target",
    "collate_samples",
]
