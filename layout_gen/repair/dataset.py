"""
layout_gen.repair.dataset — torch Dataset for mined diffusion trajectories.

Loads JSON records produced by :mod:`mine_trajectories` and yields
:class:`~.features.FeaturizedSample` instances.  Use with the standard
``torch.utils.data.DataLoader``.

Filtering options keep the batches sane:

* ``min_violations`` / ``max_violations`` — drop trivial or pathological
  perturbations.
* ``allowed_kinds`` — restrict labels to learnable inverse-action kinds
  (excludes ``add_rect`` which has no fixed-target representation).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from layout_gen.repair.augment  import augment_sample
from layout_gen.repair.features import (
    FeaturizedSample, ACTION_KIND_INDEX, featurize_record, collate,
    encode_polygons, encode_action, target_centroid,
    RULE_CAT_INDEX,
)
from layout_gen.repair.features import _infer_rule_cat_for_record


def _pick_conditioning(
    rec:    dict,
    fallback_xy: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Choose one violation from a trajectory record to condition on.

    Picks the violation whose centroid is closest to the perturbed
    polygon's centroid (most likely to be the *cause* of that
    perturbation's broken layout).  Returns ``(xy_norm, rule_cat_index)``.

    Falls back to ``(fallback_xy, classify(first_rule_name))`` when the
    record predates schema=2 (no ``violations`` field) or the list is
    empty.
    """
    viols = rec.get("violations") or []
    if not viols:
        return fallback_xy, _infer_rule_cat_for_record(rec)

    fb = fallback_xy.tolist()
    best, best_d = viols[0], float("inf")
    for v in viols:
        try:
            dx = float(v["x_norm"]) - fb[0]
            dy = float(v["y_norm"]) - fb[1]
            d  = dx * dx + dy * dy
        except Exception:
            d = float("inf")
        if d < best_d:
            best_d = d
            best   = v

    xy = torch.tensor(
        [float(best.get("x_norm", fb[0])), float(best.get("y_norm", fb[1]))],
        dtype=torch.float32,
    )
    cat_name = best.get("category") or "unknown"
    cat_idx  = RULE_CAT_INDEX.get(cat_name, RULE_CAT_INDEX["unknown"])
    return xy, cat_idx
from layout_gen.synth.geo.state import LayoutState
from layout_gen.repair.perturb import (
    PerturbAction, apply as apply_perturbation,
)


# Action kinds the model is trained to predict.  ``add_rect`` is excluded
# because its target rid doesn't exist in the perturbed state — a separate
# head would be needed.
_DEFAULT_ALLOWED_KINDS = frozenset({
    "shift_edge", "shrink_rect", "grow_rect", "translate", "nudge_offgrid",
})


class TrajectoryDataset(Dataset):
    """Dataset over mined trajectory JSONs.

    By default, each k-step trajectory is *expanded* into k separate
    training samples — one per inverse step.  This gives the model
    intermediate states at every noise level instead of just the deepest.
    With ``expand_steps=False`` only the perturbed state + first inverse
    action are used (legacy behaviour).

    Parameters
    ----------
    root :
        Directory containing ``.json`` files (recursively).  Typically
        ``layout_gen/repair/data/trajectories``.
    pdks :
        Restrict to records whose ``seed_pdk`` matches.  Empty = all.
    primitives :
        Restrict to records whose ``seed_primitive`` matches.  Empty = all.
    depths :
        Restrict to records with these ``k`` values.  Empty = all.
    min_violations / max_violations :
        Filter records by perturbed-state violation count (use ``-1`` /
        ``1e9`` to disable).
    allowed_kinds :
        Filter records by inverse-action kind.
    expand_steps :
        When True (default), expand each trajectory into one sample per
        inverse step — generates ~2× to ~4× more training examples for
        the same mined data.
    """

    def __init__(
        self,
        root:           str | Path,
        *,
        pdks:           Iterable[str]      = (),
        primitives:     Iterable[str]      = (),
        depths:         Iterable[int]      = (),
        min_violations: int                = 1,
        max_violations: int                = 200,
        allowed_kinds:  Iterable[str]      = _DEFAULT_ALLOWED_KINDS,
        expand_steps:   bool               = True,
        d4_augment:     bool               = False,
    ):
        self.d4_augment = d4_augment
        self.root = Path(root)
        self._files: list[Path] = sorted(self.root.rglob("*.json"))
        if not self._files:
            raise FileNotFoundError(f"No trajectory JSONs under {self.root}")

        self._pdks       = set(pdks)
        self._primitives = set(primitives)
        self._depths     = set(depths)
        self._min_v      = min_violations
        self._max_v      = max_violations
        self._allowed    = set(allowed_kinds)
        self._expand     = expand_steps

        # Pre-load (parsed JSON) but defer featurisation to __getitem__
        # so the dataset doesn't blow up RAM.  JSONs are small.
        self._records: list[dict] = []
        for f in self._files:
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not self._keep(rec):
                continue
            self._records.append(rec)

        # Pre-compute the per-step expansion: (record_idx, step_idx).
        # step_idx = 0 means the perturbed (deepest) state with the first
        # inverse action; step_idx = k-1 means the state one step from
        # clean with the last inverse action.
        self._index: list[tuple[int, int]] = []
        for ri, rec in enumerate(self._records):
            inv = rec.get("inverse_action_sequence") or []
            n_steps = len(inv) if self._expand else min(1, len(inv))
            for si in range(n_steps):
                if inv[si].get("kind") in self._allowed:
                    self._index.append((ri, si))

    # ── Filtering ────────────────────────────────────────────────────────────

    def _keep(self, rec: dict) -> bool:
        if self._pdks and rec.get("seed_pdk", "") not in self._pdks:
            return False
        if self._primitives and rec.get("seed_primitive", "") not in self._primitives:
            return False
        if self._depths and rec.get("k", -1) not in self._depths:
            return False
        nv = int(rec.get("n_violations", 0))
        if nv < self._min_v or nv > self._max_v:
            return False
        inv = rec.get("inverse_action_sequence") or []
        if not inv:
            return False
        if inv[0].get("kind") not in self._allowed:
            return False
        return True

    # ── Conditioning: pick which DRC violation drives this sample ──────────

    @staticmethod
    def _pick_conditioning_for_record(rec: dict, tgt_xy: torch.Tensor):
        return _pick_conditioning(rec, tgt_xy)

    # ── Per-step state replay ────────────────────────────────────────────────

    @staticmethod
    def _replay_state(rec: dict, step: int) -> tuple[list[dict], dict]:
        """Return (state_at_step, inverse_action_dict) where ``step=0`` is
        the perturbed state (deepest noise) and the action is the first
        inverse step; ``step=k-1`` is one inverse step from clean.

        State is replayed by applying the first ``step`` inverse actions
        to the perturbed state.
        """
        inv = rec["inverse_action_sequence"]
        rects = rec["perturbed_state"]
        if step == 0:
            return rects, inv[0]

        # Replay step inverse actions in order to advance state forward
        # in the denoising direction.
        state = LayoutState()
        for r in rects:
            new = state.add(layer=r["layer"],
                            x0=r["x0"], y0=r["y0"],
                            x1=r["x1"], y1=r["y1"],
                            net=r.get("net", ""),
                            shape_type=r.get("shape_type", ""))
            new.group_id = r.get("group_id", -1)

        for s in range(step):
            action = PerturbAction(
                kind=inv[s]["kind"],
                target=inv[s].get("target", -1),
                params=inv[s].get("params") or {},
                snapshot=inv[s].get("snapshot"),
            )
            try:
                apply_perturbation(state, action)
            except Exception:
                # Mid-replay failure — return what we have
                break
        snapshot = [{"rid": r.rid, "layer": r.layer,
                     "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1,
                     "net": r.net, "shape_type": r.shape_type,
                     "group_id": r.group_id}
                    for r in state]
        return snapshot, inv[step]

    # ── torch.utils.data.Dataset API ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> FeaturizedSample:
        rec_idx, step = self._index[idx]
        rec = self._records[rec_idx]
        rects, action_dict = self._replay_state(rec, step)
        feats, rid_to_idx_list = encode_polygons(rects)
        rid_lookup = {rid: i for i, rid in enumerate(rid_to_idx_list)}
        kind_idx, target_idx, edge_idx, mag = encode_action(action_dict, rid_lookup)
        # Effective noise level at THIS step: number of inverse actions
        # remaining (including this one).  step=0 is deepest, step=k-1 is
        # shallowest.
        k_remaining = len(rec["inverse_action_sequence"]) - step
        tgt_xy = target_centroid(rects, target_idx)

        # If the trajectory has the new ``violations`` field (schema≥2),
        # pick a real DRC-reported violation as conditioning.  Otherwise
        # fall back to the perturbed-target centroid + first rule_name
        # (legacy proxy).
        viol_xy, rule_cat = _pick_conditioning(rec, tgt_xy)

        sample = FeaturizedSample(
            poly_feats=feats,
            poly_mask=torch.ones(len(rects), dtype=torch.bool),
            rid_to_idx=rid_to_idx_list,
            k=k_remaining,
            violation_xy=viol_xy,
            rule_cat=rule_cat,
            action_kind=kind_idx,
            target_idx=target_idx,
            target_xy=tgt_xy,
            edge_idx=edge_idx,
            magnitude=mag,
        )
        if self.d4_augment:
            # Random D4 symmetry per access (8 elements).  Stratified would
            # be better; uniform is fine for our purposes.
            import random
            s = random.randint(0, 7)
            sample = augment_sample(sample, s)
        return sample

    # ── Class statistics for loss weighting ─────────────────────────────────

    def kind_counts(self) -> torch.Tensor:
        """Return ``(N_ACTION_KINDS,)`` count tensor over the index."""
        from layout_gen.repair.features import N_ACTION_KINDS
        counts = torch.zeros(N_ACTION_KINDS, dtype=torch.float32)
        for ri, si in self._index:
            inv = self._records[ri]["inverse_action_sequence"]
            kind = inv[si]["kind"]
            ki = ACTION_KIND_INDEX.get(kind, -1)
            if ki >= 0:
                counts[ki] += 1
        return counts

    def kind_class_weights(self, smoothing: float = 1.0) -> torch.Tensor:
        """Return inverse-frequency class weights for the kind head, with
        Laplace smoothing.  Use directly as ``weight=`` in
        :func:`F.cross_entropy`.
        """
        counts = self.kind_counts()
        weights = (counts.sum() + smoothing * counts.shape[0]) / (counts + smoothing)
        # Normalise so the average weight is 1
        weights = weights / weights.mean()
        return weights

    # ── Convenience ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Coverage breakdown — useful for sanity-checking the corpus."""
        from collections import Counter
        pdks  = Counter(r.get("seed_pdk", "?") for r in self._records)
        prims = Counter(r.get("seed_primitive", "?") for r in self._records)
        ks    = Counter(r.get("k", -1) for r in self._records)
        # For kind stats use the *expanded* index, not just first-action
        kinds: Counter = Counter()
        for ri, si in self._index:
            inv = self._records[ri]["inverse_action_sequence"]
            kinds[inv[si]["kind"]] += 1
        ks_expanded = Counter(
            len(self._records[ri]["inverse_action_sequence"]) - si
            for ri, si in self._index
        )
        return {
            "n_records":     len(self._records),
            "n_samples":     len(self._index),
            "expand_steps":  self._expand,
            "by_pdk":        dict(pdks),
            "by_prim":       dict(prims),
            "by_record_k":   dict(ks),
            "by_sample_k":   dict(ks_expanded),
            "by_kind":       dict(kinds),
        }


def make_dataloader(
    dataset:    TrajectoryDataset,
    batch_size: int  = 16,
    shuffle:    bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Default DataLoader using our padded :func:`collate`."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


__all__ = ["TrajectoryDataset", "make_dataloader"]
