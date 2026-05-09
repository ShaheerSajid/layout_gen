"""
layout_gen.repair.perturb — procedural perturbations for self-supervised data.

Given a clean layout, apply a random destructive geometric edit to make it
DRC-fail.  The *inverse* of that edit is, by construction, a labelled fix.
This produces (problem, solution) trajectories without any human in the
loop.

Design rules
------------
* **Pure geometry.**  Perturbations operate on rectangles and edges.  No
  PDK rule values appear here — the perturbation magnitude is a
  configurable parameter, expressed in µm.
* **Invertible.**  Every perturbation records the rectangle ID(s) and
  parameters needed to reverse it.  Applying ``inverse`` restores the
  original layout exactly.
* **Layer-role-aware (optional).**  When a layer-role filter is provided
  (e.g. "perturb only metal"), the picker respects it.  This lets us
  generate curricula that target specific rule classes without ever
  encoding rule values.

Action set
----------
* ``shift_edge``   — push one edge of a rectangle by ±δ.
* ``shrink_rect``  — inset all four edges by δ.
* ``grow_rect``    — outset all four edges by δ.
* ``translate``    — move a rectangle by (dx, dy).
* ``delete_rect``  — remove a rectangle.
* ``add_rect``     — insert a small extraneous rectangle.
* ``nudge_offgrid``— translate by a sub-grid amount.

Each is a thin function on a :class:`~layout_gen.synth.geo.state.LayoutState`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from layout_gen.synth.geo.state import LayoutState, Rect


# ── Perturbation actions ─────────────────────────────────────────────────────

@dataclass
class PerturbAction:
    """One perturbation step.  Carries enough state to compute its inverse."""
    kind:    str                # "shift_edge" / "shrink_rect" / ...
    target:  int                # rectangle ID it operated on (-1 for add)
    params:  dict[str, Any] = field(default_factory=dict)
    # For ``add_rect`` / ``delete_rect`` we need the full Rect snapshot
    snapshot: dict[str, Any] | None = None

    def inverse(self) -> "PerturbAction":
        """Return the action that undoes this one."""
        if self.kind == "shift_edge":
            return PerturbAction(
                kind="shift_edge",
                target=self.target,
                params={"side": self.params["side"],
                        "delta": -self.params["delta"]},
            )
        if self.kind == "shrink_rect":
            return PerturbAction(
                kind="grow_rect",
                target=self.target,
                params={"delta": self.params["delta"]},
            )
        if self.kind == "grow_rect":
            return PerturbAction(
                kind="shrink_rect",
                target=self.target,
                params={"delta": self.params["delta"]},
            )
        if self.kind == "translate":
            return PerturbAction(
                kind="translate",
                target=self.target,
                params={"dx": -self.params["dx"], "dy": -self.params["dy"]},
            )
        if self.kind == "nudge_offgrid":
            return PerturbAction(
                kind="translate",
                target=self.target,
                params={"dx": -self.params["dx"], "dy": -self.params["dy"]},
            )
        if self.kind == "delete_rect":
            assert self.snapshot is not None
            return PerturbAction(
                kind="add_rect", target=-1,
                params={}, snapshot=dict(self.snapshot),
            )
        if self.kind == "add_rect":
            return PerturbAction(
                kind="delete_rect",
                target=self.target,
                snapshot=self.snapshot,
            )
        raise ValueError(f"Unknown perturbation kind: {self.kind!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# ── Forward operations ───────────────────────────────────────────────────────
#
# All operations route through :class:`LayoutState`'s public mutators
# (``add`` / ``update`` / ``remove``).  The ``rects`` property returns a
# copy, so mutating that list is silently lost — never do that.

def shift_edge(
    state:  LayoutState,
    rid:    int,
    side:   str,                # "left" | "right" | "bottom" | "top"
    delta:  float,              # signed; positive moves outward
) -> PerturbAction:
    r = state[rid]
    nx0, ny0, nx1, ny1 = r.x0, r.y0, r.x1, r.y1
    if side == "left":   nx0 -= delta
    elif side == "right":  nx1 += delta
    elif side == "bottom": ny0 -= delta
    elif side == "top":    ny1 += delta
    else:
        raise ValueError(f"Bad side: {side!r}")
    state.update(rid, x0=nx0, y0=ny0, x1=nx1, y1=ny1)
    return PerturbAction("shift_edge", target=rid,
                          params={"side": side, "delta": delta})


def shrink_rect(state: LayoutState, rid: int, delta: float) -> PerturbAction:
    r = state[rid]
    state.update(rid,
                 x0=r.x0 + delta, y0=r.y0 + delta,
                 x1=r.x1 - delta, y1=r.y1 - delta)
    return PerturbAction("shrink_rect", target=rid, params={"delta": delta})


def grow_rect(state: LayoutState, rid: int, delta: float) -> PerturbAction:
    r = state[rid]
    state.update(rid,
                 x0=r.x0 - delta, y0=r.y0 - delta,
                 x1=r.x1 + delta, y1=r.y1 + delta)
    return PerturbAction("grow_rect", target=rid, params={"delta": delta})


def translate(
    state: LayoutState, rid: int, dx: float, dy: float,
) -> PerturbAction:
    r = state[rid]
    state.update(rid,
                 x0=r.x0 + dx, y0=r.y0 + dy,
                 x1=r.x1 + dx, y1=r.y1 + dy)
    return PerturbAction("translate", target=rid,
                         params={"dx": dx, "dy": dy})


def nudge_offgrid(
    state: LayoutState, rid: int, dx: float, dy: float,
) -> PerturbAction:
    """Same as :func:`translate` but the inverse is recorded as a
    *snap-to-grid* fix rather than a raw translate.  Useful for the
    off-grid violation category."""
    act = translate(state, rid, dx, dy)
    act.kind = "nudge_offgrid"
    return act


def delete_rect(state: LayoutState, rid: int) -> PerturbAction:
    r = state[rid]
    snap = {"rid": r.rid, "layer": r.layer,
            "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1,
            "net": r.net, "shape_type": r.shape_type,
            "group_id": r.group_id}
    state.remove(rid)
    return PerturbAction("delete_rect", target=rid, snapshot=snap)


def add_rect(state: LayoutState, snapshot: dict[str, Any]) -> PerturbAction:
    """Insert a rectangle described by *snapshot*.  Used by the inverse of
    :func:`delete_rect` and as a perturbation that adds spurious
    geometry."""
    new = state.add(
        layer=snapshot["layer"],
        x0=snapshot["x0"], y0=snapshot["y0"],
        x1=snapshot["x1"], y1=snapshot["y1"],
        net=snapshot.get("net", ""),
        shape_type=snapshot.get("shape_type", ""),
    )
    if snapshot.get("group_id", -1) != -1:
        new.group_id = snapshot["group_id"]
    return PerturbAction("add_rect", target=new.rid,
                         snapshot={**snapshot, "rid": new.rid})


def apply(state: LayoutState, action: PerturbAction) -> None:
    """Apply *action* to *state* in-place.  Used for replaying inverses."""
    if action.kind == "shift_edge":
        shift_edge(state, action.target, action.params["side"],
                   action.params["delta"])
    elif action.kind == "shrink_rect":
        shrink_rect(state, action.target, action.params["delta"])
    elif action.kind == "grow_rect":
        grow_rect(state, action.target, action.params["delta"])
    elif action.kind in ("translate", "nudge_offgrid"):
        translate(state, action.target, action.params["dx"],
                  action.params["dy"])
    elif action.kind == "delete_rect":
        delete_rect(state, action.target)
    elif action.kind == "add_rect":
        add_rect(state, action.snapshot or {})
    else:
        raise ValueError(f"Unknown action kind: {action.kind!r}")


# ── Random sampler ───────────────────────────────────────────────────────────

@dataclass
class PerturbConfig:
    """Knobs controlling a random perturbation episode.

    Attributes
    ----------
    delta_min_um, delta_max_um :
        Range of geometric perturbation magnitudes.
    layer_filter :
        If non-empty, only rectangles on these layers are eligible.  Use
        a *layer-role* set (e.g. ``{"poly", "li1", "met1"}``) — the
        picker treats them as logical layer names.
    forbid_kinds :
        Perturbation kinds to skip (e.g. exclude ``delete_rect`` for
        small cells where deleting any rect leaves the layout unrepairable).
    seed :
        RNG seed for reproducibility.
    """
    delta_min_um:  float            = 0.005
    delta_max_um:  float            = 0.05
    layer_filter:  frozenset[str]   = frozenset()
    forbid_kinds:  frozenset[str]   = frozenset()
    seed:          int | None       = None


# Probabilities are unweighted by default; tune via *forbid_kinds* in
# PerturbConfig if a particular kind is unwanted.
_KINDS = (
    "shift_edge", "shrink_rect", "grow_rect",
    "translate", "delete_rect", "nudge_offgrid",
)


def random_perturbation(
    state:  LayoutState,
    config: PerturbConfig | None = None,
    rng:    random.Random | None = None,
) -> PerturbAction:
    """Apply one random perturbation in-place and return the action.

    The caller should run DRC on the modified *state* to confirm a
    violation was actually created (perturbations occasionally produce
    geometrically valid layouts — those should be discarded).
    """
    config = config or PerturbConfig()
    rng    = rng    or random.Random(config.seed)

    eligible: list[Rect] = list(state)   # iterates _rects.values()
    if config.layer_filter:
        eligible = [r for r in eligible if r.layer in config.layer_filter]
    if not eligible:
        raise ValueError("No rectangles match the layer filter")

    available_kinds = [k for k in _KINDS if k not in config.forbid_kinds]
    if not available_kinds:
        raise ValueError("All perturbation kinds were forbidden")

    kind  = rng.choice(available_kinds)
    rect  = rng.choice(eligible)
    delta = rng.uniform(config.delta_min_um, config.delta_max_um)

    if kind == "shift_edge":
        side = rng.choice(("left", "right", "bottom", "top"))
        signed = delta * rng.choice((-1.0, 1.0))
        return shift_edge(state, rect.rid, side, signed)
    if kind == "shrink_rect":
        # Don't shrink past zero
        max_shrink = min(rect.width, rect.height) / 2 - 1e-3
        delta = min(delta, max_shrink) if max_shrink > 0 else 0.001
        return shrink_rect(state, rect.rid, delta)
    if kind == "grow_rect":
        return grow_rect(state, rect.rid, delta)
    if kind == "translate":
        dx = delta * rng.choice((-1.0, 1.0))
        dy = delta * rng.choice((-1.0, 1.0))
        return translate(state, rect.rid, dx, dy)
    if kind == "delete_rect":
        return delete_rect(state, rect.rid)
    if kind == "nudge_offgrid":
        # Sub-grid nudges (~ 1 nm) — break manufacturing-grid alignment
        sub = max(0.001, delta / 10)
        dx = sub * rng.choice((-1.0, 1.0))
        dy = sub * rng.choice((-1.0, 1.0))
        return nudge_offgrid(state, rect.rid, dx, dy)
    raise AssertionError("unreachable")


# ── Trajectory generation ────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """One self-supervised training example.

    ``forward`` is the destructive sequence applied to the clean layout;
    ``inverse`` (in the same order, reversed and inverted) is the
    labelled fix sequence the model should produce.
    """
    forward:  list[PerturbAction] = field(default_factory=list)
    inverse:  list[PerturbAction] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.forward)

    def to_dict(self) -> dict:
        return {
            "forward": [a.to_dict() for a in self.forward],
            "inverse": [a.to_dict() for a in self.inverse],
        }


def generate_trajectory(
    state:  LayoutState,
    *,
    n_steps: int = 1,
    config:  PerturbConfig | None = None,
    rng:     random.Random | None = None,
) -> Trajectory:
    """Apply *n_steps* random perturbations to *state* in-place.

    Returns a :class:`Trajectory` whose ``inverse`` field, when applied
    in order, restores *state* to its starting geometry.  This is the
    label for self-supervised training: given the perturbed *state*, the
    model should produce ``inverse``.

    Multi-step trajectories automatically forbid ``delete_rect`` and
    ``add_rect`` to keep rectangle IDs stable across steps — otherwise
    a later perturbation might reference a rid that an earlier one has
    removed, breaking invertibility.
    """
    config = config or PerturbConfig()
    if n_steps > 1 and "delete_rect" not in config.forbid_kinds:
        # Don't mutate the caller's config — make a local copy with
        # delete forbidden.
        config = PerturbConfig(
            delta_min_um=config.delta_min_um,
            delta_max_um=config.delta_max_um,
            layer_filter=config.layer_filter,
            forbid_kinds=config.forbid_kinds | {"delete_rect"},
            seed=config.seed,
        )
    rng = rng or random.Random(config.seed)

    forward: list[PerturbAction] = []
    used_rids: set[int] = set()
    for _ in range(n_steps):
        # Avoid touching the same rect twice in one trajectory: the
        # inverse of action #2 on rid X would undo what we want, then
        # action #1's inverse would re-apply.  Either is geometrically
        # consistent but adds noise to training.
        attempts = 0
        while True:
            act = random_perturbation(state, config=config, rng=rng)
            if act.target in used_rids and n_steps > 1 and attempts < 8:
                # Roll back this attempt and try again
                apply(state, act.inverse())
                attempts += 1
                continue
            forward.append(act)
            used_rids.add(act.target)
            break
    # Inverse: reversed forward, with each action inverted
    inverse = [a.inverse() for a in reversed(forward)]
    return Trajectory(forward=forward, inverse=inverse)


def round_trip_check(
    initial_snapshot: list[tuple],
    final_state:      LayoutState,
) -> bool:
    """Confirm that *final_state*'s rectangles match *initial_snapshot*.

    *initial_snapshot* must be captured *before* any perturbation is
    applied (it's a list of ``(layer, x0, y0, x1, y1)`` tuples — rids may
    differ if the trajectory included add/delete operations).
    """
    final = sorted((r.layer, round(r.x0, 6), round(r.y0, 6),
                    round(r.x1, 6), round(r.y1, 6)) for r in final_state)
    initial = sorted(initial_snapshot)
    return final == initial


def snapshot_state(state: LayoutState) -> list[tuple]:
    """Capture *state*'s geometry as a list of (layer, x0, y0, x1, y1)."""
    return [(r.layer, round(r.x0, 6), round(r.y0, 6),
             round(r.x1, 6), round(r.y1, 6)) for r in state]


__all__ = [
    "PerturbAction",
    "PerturbConfig",
    "Trajectory",
    "shift_edge",
    "shrink_rect",
    "grow_rect",
    "translate",
    "nudge_offgrid",
    "delete_rect",
    "add_rect",
    "apply",
    "random_perturbation",
    "generate_trajectory",
    "round_trip_check",
    "snapshot_state",
]
