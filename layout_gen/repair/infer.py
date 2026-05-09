"""
layout_gen.repair.infer — apply a trained denoiser to a layout iteratively.

The diffusion-style inference loop:

1. Run DRC on the layout.  If clean, done.
2. Featurise the current state + use ``n_violations`` as the noise-level k.
3. Run the network → predicted (kind, target, edge, magnitude).
4. Decode into a :class:`PerturbAction` (here we use the same dataclass
   the perturbation library uses — the inverse of a perturbation IS a fix).
5. Apply the action to the layout state.
6. Loop.

The network is trained on **inverse-perturbation** trajectories.  At
inference time, applying its predicted action *to a violating state*
moves the state one step closer to clean — exactly as DDPM's reverse
sampling moves a noisy image toward the data manifold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from layout_gen.synth.geo.state import LayoutState
from layout_gen.repair.features import (
    encode_polygons, ACTION_KINDS, EDGE_NAMES, POLY_FEAT_DIM,
)
from layout_gen.repair.model    import DRCDenoiser
from layout_gen.repair.perturb  import (
    PerturbAction, apply as apply_perturbation,
)


@dataclass
class InferenceStep:
    """One repair step's record (for diagnostics)."""
    n_violations_before: int
    n_violations_after:  int
    action:              PerturbAction
    kind_logits_top:     list[tuple[str, float]]


@dataclass
class InferenceResult:
    """Outcome of a denoising pass."""
    converged:    bool
    iterations:   int
    final_state:  LayoutState
    history:      list[InferenceStep]
    initial_violations: int
    final_violations:   int


# ── Loading ──────────────────────────────────────────────────────────────────

def load_denoiser(checkpoint: Path | str) -> DRCDenoiser:
    """Reconstruct a :class:`DRCDenoiser` from a saved checkpoint."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})
    # Defaults match the v5/v7 winning architecture (small).  Older
    # checkpoints (v0–v4) override these via cfg.
    model = DRCDenoiser(
        hidden_dim=cfg.get("hidden_dim", 32),
        n_layers=cfg.get("n_layers", 1),
        n_heads=cfg.get("n_heads", 2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ── State encoding ───────────────────────────────────────────────────────────

def _state_to_rectdicts(state: LayoutState) -> list[dict]:
    return [{"rid": r.rid, "layer": r.layer,
             "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}
            for r in state]


# ── Action decoding ──────────────────────────────────────────────────────────

def _decode_action(
    pred:        dict[str, torch.Tensor],
    rid_to_idx:  list[int],
    poly_feats:  torch.Tensor,
    *,
    sample:      bool   = False,
    temperature: float  = 1.0,
    rng:         torch.Generator | None = None,
) -> PerturbAction:
    """Decode network output into a :class:`PerturbAction`.

    Target selection: we use **centroid-snap**.  The model regresses the
    target polygon's (cx, cy) in cell-bbox-normalised [0, 1]² space; we
    pick the polygon whose centroid is nearest.  This was the head that
    actually learned spatial structure (target top-10 = 40% on v7); the
    raw target-pointer logits stayed near chance.

    Parameters
    ----------
    sample :
        When True, sample kind/edge from softmax distributions and add
        Gaussian jitter to the centroid prediction so iterative inference
        explores more than one fix.  Greedy argmax (sample=False) tends
        to deadlock when the same input leads to the same prediction.
    temperature :
        Softmax temperature for sampling (>1 → more diverse, <1 → more
        peaked).  Ignored when sample=False.
    rng :
        Optional torch generator for reproducible sampling.
    """
    from layout_gen.repair.features import N_LAYER_ROLES

    if sample:
        kind_probs = torch.softmax(pred["kind_logits"][0] / temperature, dim=-1)
        kind_idx   = int(torch.multinomial(kind_probs, 1, generator=rng).item())
        edge_probs = torch.softmax(pred["edge_logits"][0] / temperature, dim=-1)
        edge_idx   = int(torch.multinomial(edge_probs, 1, generator=rng).item())
    else:
        kind_idx   = int(torch.argmax(pred["kind_logits"][0]).item())
        edge_idx   = int(torch.argmax(pred["edge_logits"][0]).item())
    kind = ACTION_KINDS[kind_idx]
    edge = EDGE_NAMES[edge_idx]

    # Centroid-snap target selection (with optional jitter).
    poly_xy   = poly_feats[..., N_LAYER_ROLES:N_LAYER_ROLES+2]    # (N, 2)
    pred_xy   = pred["target_xy"][0]                              # (2,)
    if sample:
        # Jitter the predicted centroid by ~5% of the cell extent so we
        # don't always snap to the same polygon when several are near.
        jitter = torch.randn(2, generator=rng) * 0.05
        pred_xy = (pred_xy + jitter).clamp(0.0, 1.0)
    dists     = torch.norm(poly_xy - pred_xy.unsqueeze(0), dim=-1)
    target_idx_int = int(torch.argmin(dists).item())
    target_rid = (rid_to_idx[target_idx_int]
                  if target_idx_int < len(rid_to_idx) else -1)

    delta_pred = float(pred["magnitude"][0, 0].item())
    dx_pred    = float(pred["magnitude"][0, 1].item())
    dy_pred    = float(pred["magnitude"][0, 2].item())

    if kind == "shift_edge":
        return PerturbAction(kind, target=target_rid,
                             params={"side": edge, "delta": delta_pred})
    if kind in ("shrink_rect", "grow_rect"):
        # Take absolute value to keep the "direction" implied by the kind
        return PerturbAction(kind, target=target_rid,
                             params={"delta": abs(delta_pred)})
    if kind in ("translate", "nudge_offgrid"):
        return PerturbAction(kind, target=target_rid,
                             params={"dx": dx_pred, "dy": dy_pred})
    if kind == "delete_rect":
        return PerturbAction(kind, target=target_rid, params={})
    # Fallback
    return PerturbAction(kind, target=target_rid, params={"delta": delta_pred})


def _topk_kinds(pred: dict[str, torch.Tensor], k: int = 3) -> list[tuple[str, float]]:
    logits = pred["kind_logits"][0]
    probs  = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k=min(k, probs.shape[0]))
    return [(ACTION_KINDS[int(i)], float(p))
            for p, i in zip(top.values, top.indices)]


# ── Public entry point ──────────────────────────────────────────────────────

def repair(
    state:       LayoutState,
    model:       DRCDenoiser,
    drc_runner:  Any,
    rules:       Any,
    *,
    max_iter:    int = 20,
    record_history: bool = True,
    sample:      bool = True,
    temperature: float = 1.0,
    seed:        int | None = None,
    k_max:       int = 6,
) -> InferenceResult:
    """Iterate the denoiser on *state* until DRC reports clean or we hit
    *max_iter*.

    The DRC runner is consulted at every iteration — it is the only signal
    grounding the denoiser's progress in real PDK rules.  The denoiser
    itself never sees PDK constants.
    """
    import tempfile

    history: list[InferenceStep] = []
    _ctr = [0]
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    def _drc_run(st: LayoutState):
        """Return the full list of DRCViolation objects, not just the count."""
        _ctr[0] += 1
        comp = st.to_component(rules, name=f"infer_check_{id(st)}_{_ctr[0]}")
        with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
            gds = Path(f.name)
        try:
            comp.write_gds(str(gds))
            return drc_runner.run(gds, comp.name)
        except Exception:
            return []
        finally:
            gds.unlink(missing_ok=True)

    def _violation_xy_norm(v, comp_bbox):
        """Map a DRC violation (x, y) into [0, 1]² cell-bbox space."""
        x0c, y0c, x1c, y1c = comp_bbox
        w = max(x1c - x0c, 1e-6)
        h = max(y1c - y0c, 1e-6)
        return ((float(v.x) - x0c) / w,
                (float(v.y) - y0c) / h)

    def _bbox_of_state(st: LayoutState):
        rs = list(st)
        if not rs:
            return (0.0, 0.0, 1.0, 1.0)
        return (min(r.x0 for r in rs), min(r.y0 for r in rs),
                max(r.x1 for r in rs), max(r.y1 for r in rs))

    initial_viols = _drc_run(state)
    n_before = len(initial_viols)
    initial_violations = n_before
    if n_before == 0:
        return InferenceResult(converged=True, iterations=0,
                               final_state=state, history=[],
                               initial_violations=0, final_violations=0)

    current_viols = initial_viols
    for it in range(1, max_iter + 1):
        rect_dicts = _state_to_rectdicts(state)
        feats, rid_to_idx = encode_polygons(rect_dicts)
        if feats.shape[0] == 0:
            break

        # Pick one violation to fix this iteration: cycle through the
        # violations stochastically (sample=True) or deterministically
        # in order (sample=False).  Each iteration repairs one violation,
        # then DRC re-runs to get an updated list.
        if not current_viols:
            break
        if sample:
            v_idx = int(torch.randint(0, len(current_viols), (1,),
                                      generator=rng).item())
        else:
            v_idx = 0
        target_v = current_viols[v_idx]
        bbox = _bbox_of_state(state)
        v_x, v_y = _violation_xy_norm(target_v, bbox)
        v_xy = torch.tensor([[v_x, v_y]], dtype=torch.float32)

        # Map rule name → category index using the catalog classifier.
        from layout_gen.repair.catalog import classify_rule
        from layout_gen.repair.features import RULE_CAT_INDEX
        cat_name = classify_rule(target_v.rule, target_v.description or "", [])
        rule_cat = torch.tensor(
            [RULE_CAT_INDEX.get(cat_name, RULE_CAT_INDEX["unknown"])],
            dtype=torch.long,
        )

        # Cap noise level at k_max — training corpus saw k ≤ 6, anything
        # higher is OOD and the model's behaviour gets unstable.
        k_in = min(max(n_before, 1), k_max)
        with torch.no_grad():
            pred = model(
                feats.unsqueeze(0),
                torch.ones(1, feats.shape[0], dtype=torch.bool),
                torch.tensor([k_in], dtype=torch.long),
                violation_xy=v_xy,
                rule_cat=rule_cat,
            )

        # Override target_xy with the actual violation position so the
        # centroid-snap in _decode_action picks the polygon nearest the
        # *real* violation instead of the model's predicted xy.
        pred = dict(pred)
        pred["target_xy"] = v_xy

        action = _decode_action(
            pred, rid_to_idx, feats,
            sample=sample, temperature=temperature, rng=rng,
        )
        topk   = _topk_kinds(pred, k=3)

        try:
            apply_perturbation(state, action)
        except (KeyError, ValueError):
            break

        current_viols = _drc_run(state)
        n_after = len(current_viols)

        if record_history:
            history.append(InferenceStep(
                n_violations_before=n_before,
                n_violations_after=n_after,
                action=action,
                kind_logits_top=topk,
            ))

        if n_after == 0:
            return InferenceResult(converged=True, iterations=it,
                                   final_state=state, history=history,
                                   initial_violations=initial_violations,
                                   final_violations=0)
        # No-progress stop: only abort if the last K steps STRICTLY made
        # things worse (stochastic sampling means flat steps are normal).
        K = 6
        if (record_history and len(history) >= K
                and all(h.n_violations_after > h.n_violations_before
                        for h in history[-K:])):
            break
        n_before = n_after

    return InferenceResult(
        converged=False, iterations=len(history),
        final_state=state, history=history,
        initial_violations=initial_violations,
        final_violations=n_before,
    )


__all__ = [
    "load_denoiser", "repair",
    "InferenceStep", "InferenceResult",
]
