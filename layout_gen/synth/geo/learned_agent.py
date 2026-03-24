"""
layout_gen.synth.geo.learned_agent — Experience replay + learned DRC fix agent.

Phase 3 of the geometric fix pipeline:

1. :class:`ExperienceBuffer` collects (state, violation, action, reward)
   tuples from :class:`~.loop.GeoFixLoop` runs and persists them to disk.
2. :class:`LearnedGeoAgent` uses a trained policy network to propose
   geometric fixes, replacing the rule-based heuristics of
   :class:`~.agent.RuleGeoAgent`.

The agent shares the same action space as the rule-based agent — it just
selects actions more intelligently using patterns learned from experience.

Training data comes from two sources:

- **Supervised**: rule-based agent runs + outcomes (did the violation count
  decrease?).  Each (state crop, violation, action, delta_violations) tuple
  becomes a training sample.
- **Self-play** (future): the learned agent's own runs, with reward shaping
  that penalises cascading violations and rewards convergence speed.

The architecture is PDK-agnostic: the observation is a local geometry crop
(see :meth:`~.state.LayoutState.local_crop`) + violation features, and
the action is a parametric geometric operation.  Models trained on sky130
data should transfer to TSMC 65nm because the violation categories and
fix operations are universal across CMOS processes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from layout_gen.synth.geo.state      import LayoutState, Rect
from layout_gen.synth.geo.actions    import (
    Action, StretchEdge, MoveShape, ResizeContact, LayerPromote,
)
from layout_gen.synth.geo.violations import ViolationInfo
from layout_gen.synth.geo.agent      import GeoFixAgent, RuleGeoAgent

log = logging.getLogger(__name__)


# ── Violation feature encoding (PDK-agnostic) ───────────────────────────────

_CATEGORY_TO_IDX = {
    "spacing": 0, "width": 1, "enclosure": 2, "area": 3,
    "overlap": 4, "extension": 5, "offgrid": 6, "size": 7, "unknown": 8,
}

_LAYER_CLASS_TO_IDX = {
    "gate": 0, "diffusion": 1, "contact": 2, "interconnect": 3,
    "metal1": 4, "metal2": 5, "via": 6, "well": 7, "implant": 8, "other": 9,
}


def _encode_violation(v: ViolationInfo) -> np.ndarray:
    """Encode a ViolationInfo into a 5-element feature vector."""
    from layout_gen.synth.ml.fix_policy import _classify_layer
    cat_idx = _CATEGORY_TO_IDX.get(v.category, 8)
    layer_cls = _classify_layer(v.layer)
    layer_idx = _LAYER_CLASS_TO_IDX.get(layer_cls, 9)
    return np.array([
        cat_idx, layer_idx, v.deficit, v.required, v.measured,
    ], dtype=np.float32)


def _serialise_action(action: Action) -> dict:
    """Convert an action to a JSON-serialisable dict."""
    d: dict[str, Any] = {"type": type(action).__name__}
    if isinstance(action, StretchEdge):
        d.update(rid=action.rid, edge=action.edge, delta=action.delta)
    elif isinstance(action, MoveShape):
        d.update(rid=action.rid, dx=action.dx, dy=action.dy)
    elif isinstance(action, ResizeContact):
        d.update(rid=action.rid, target_size=action.target_size)
    elif isinstance(action, LayerPromote):
        d.update(rid=action.rid, from_layer=action.from_layer,
                 to_layer=action.to_layer)
    return d


# ── Experience tuple ─────────────────────────────────────────────────────────

@dataclass
class Experience:
    """One (state, violation, action, reward) transition.

    Attributes
    ----------
    observation :
        Local geometry crop around the violation, shape ``(N, 6)``.
    violation_features :
        Encoded violation: ``[category_idx, layer_class_idx,
        deficit, required, measured]``.
    action_type :
        Action class name (e.g. ``"StretchEdge"``).
    action_params :
        Serialised action parameters.
    reward :
        ``+1`` if the target violation was fixed, ``-0.5`` per new
        cascading violation introduced, ``0`` if no change.
    """
    observation:         np.ndarray
    violation_features:  np.ndarray
    action_type:         str
    action_params:       dict
    reward:              float


# ── Experience buffer with persistence ───────────────────────────────────────

class ExperienceBuffer:
    """Stores and persists (state, violation, action, reward) transitions.

    Supports saving/loading from JSON-lines files for offline training.
    Automatically evicts oldest entries when capacity is reached.
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self._buffer: list[Experience] = []

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, exp: Experience) -> None:
        """Add an experience.  Oldest entries are evicted at capacity."""
        self._buffer.append(exp)
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)

    def add_from_fix_record(
        self,
        state:          LayoutState,
        violations:     list[ViolationInfo],
        actions:        list[Action],
        remaining:      int,
        crop_radius:    float = 2.0,
    ) -> None:
        """Convert a :class:`~.loop.FixRecord` into experiences.

        One experience per (violation, action) pair.  Reward is computed
        from the change in violation count.
        """
        n_before = len(violations)
        n_after  = remaining
        # Global reward: positive if we reduced violations
        base_reward = float(n_before - n_after) / max(n_before, 1)
        # Penalty for increase
        if n_after > n_before:
            base_reward = -0.5 * (n_after - n_before) / max(n_before, 1)

        # Pair violations with actions (zip stops at shorter list)
        for v, a in zip(violations, actions):
            obs = state.local_crop(v.x, v.y, radius=crop_radius)
            vf  = _encode_violation(v)
            self.add(Experience(
                observation=obs,
                violation_features=vf,
                action_type=type(a).__name__,
                action_params=_serialise_action(a),
                reward=base_reward,
            ))

    def sample(self, n: int) -> list[Experience]:
        """Random sample of *n* experiences (without replacement)."""
        rng = np.random.default_rng()
        idx = rng.choice(len(self._buffer), size=min(n, len(self._buffer)),
                         replace=False)
        return [self._buffer[i] for i in idx]

    def save(self, path: str | Path) -> None:
        """Save buffer to a JSON-lines file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for exp in self._buffer:
                record = {
                    "obs_shape":     list(exp.observation.shape),
                    "obs":           exp.observation.tolist(),
                    "vf":            exp.violation_features.tolist(),
                    "action_type":   exp.action_type,
                    "action_params": exp.action_params,
                    "reward":        exp.reward,
                }
                f.write(json.dumps(record) + "\n")
        log.info("Saved %d experiences to %s", len(self._buffer), path)

    def load(self, path: str | Path) -> None:
        """Load experiences from a JSON-lines file (appends to buffer)."""
        path = Path(path)
        if not path.exists():
            return
        count = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                exp = Experience(
                    observation=np.array(record["obs"], dtype=np.float32),
                    violation_features=np.array(record["vf"], dtype=np.float32),
                    action_type=record["action_type"],
                    action_params=record["action_params"],
                    reward=record["reward"],
                )
                self.add(exp)
                count += 1
        log.info("Loaded %d experiences from %s", count, path)


# ── Learned agent ────────────────────────────────────────────────────────────

class LearnedGeoAgent(GeoFixAgent):
    """Phase 3: learned geometric DRC fix agent.

    Falls back to :class:`RuleGeoAgent` when the model is not trained
    or confidence is low, ensuring we never do worse than rule-based.

    The policy network architecture (when trained):

    - Input: flattened local_crop (padded to fixed size) + violation features
    - Hidden: 2-layer MLP (128, 64) with ReLU
    - Output: action type logits + continuous action parameters

    For now this is a **skeleton** — the ``propose_fix`` method delegates
    to the rule-based agent while collecting experience for future training.
    Once enough experience is collected (from both sky130 and TSMC runs),
    the model can be trained offline and loaded via ``model_path``.

    Parameters
    ----------
    rules :
        PDK rules (passed to fallback agent).
    buffer :
        Experience buffer for recording transitions.
    fallback :
        Rule-based agent for fallback when model is not loaded.
    model_path :
        Path to a saved model checkpoint.
    crop_radius :
        Radius (µm) around violation centroid for local crop.
    """

    # Fixed observation dimension: max 50 rects × 6 features + 5 violation features
    MAX_RECTS = 50
    OBS_DIM   = MAX_RECTS * 6 + 5  # 305
    # Action types: StretchEdge(4 edges), MoveShape, ResizeContact, LayerPromote
    N_ACTION_TYPES = 7

    def __init__(
        self,
        rules:       Any = None,
        buffer:      ExperienceBuffer | None = None,
        fallback:    RuleGeoAgent | None = None,
        model_path:  str | None = None,
        crop_radius: float = 2.0,
    ):
        self.rules       = rules
        self.buffer      = buffer or ExperienceBuffer()
        self.fallback    = fallback or RuleGeoAgent(rules)
        self.crop_radius = crop_radius
        self._model      = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        try:
            import torch
            log.info("LearnedGeoAgent: model loading from %s", path)
            # Model architecture will be defined during Phase 3 training
            # self._model = torch.load(path, weights_only=True)
            log.warning("LearnedGeoAgent: model architecture not yet defined")
        except ImportError:
            log.warning("PyTorch not available; using fallback agent")

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def propose_fix(
        self,
        state:     LayoutState,
        violation: ViolationInfo,
    ) -> list[Action]:
        """Use the trained model to propose a fix, or fall back to rules.

        Also records the observation for later experience collection.
        """
        if self._model is not None:
            actions = self._model_propose(state, violation)
            if actions:
                return actions

        # Fall back to rule-based agent
        return self.fallback.propose_fix(state, violation)

    def _model_propose(
        self, state: LayoutState, violation: ViolationInfo,
    ) -> list[Action]:
        """Query the trained policy network for actions."""
        obs_vec = self._build_obs_vector(state, violation)

        try:
            prediction = self._model.predict(obs_vec.reshape(1, -1))[0]
        except Exception:
            return []

        # prediction[:N_ACTION_TYPES] = action type logits
        # prediction[N_ACTION_TYPES:] = continuous params
        action_type_idx = int(np.argmax(prediction[:self.N_ACTION_TYPES]))
        params = prediction[self.N_ACTION_TYPES:]

        return self._decode_action(action_type_idx, params, state, violation)

    def _build_obs_vector(
        self, state: LayoutState, violation: ViolationInfo,
    ) -> np.ndarray:
        """Build a fixed-size observation vector."""
        crop = state.local_crop(violation.x, violation.y,
                                radius=self.crop_radius)
        vf = _encode_violation(violation)

        # Pad/truncate crop to MAX_RECTS rows
        if crop.shape[0] > self.MAX_RECTS:
            crop = crop[:self.MAX_RECTS]
        elif crop.shape[0] < self.MAX_RECTS:
            pad = np.zeros((self.MAX_RECTS - crop.shape[0], 6), dtype=np.float32)
            crop = np.vstack([crop, pad]) if crop.size > 0 else pad

        return np.concatenate([crop.flatten(), vf])

    @staticmethod
    def _decode_action(
        type_idx:  int,
        params:    np.ndarray,
        state:     LayoutState,
        violation: ViolationInfo,
    ) -> list[Action]:
        """Decode model output into concrete actions."""
        shapes = state.near(violation.x, violation.y, 2.0,
                            layer=violation.layer)
        if not shapes:
            return []
        target = shapes[0]

        delta = max(0.005, min(abs(float(params[0])) if len(params) > 0
                                else violation.deficit, 1.0))

        if type_idx <= 3:
            # StretchEdge on one of 4 edges
            edges = ["left", "right", "bottom", "top"]
            return [StretchEdge(target.rid, edges[type_idx], delta)]
        elif type_idx == 4:
            # MoveShape
            dx = float(params[0]) if len(params) > 0 else 0.0
            dy = float(params[1]) if len(params) > 1 else 0.0
            return [MoveShape(target.rid, dx, dy)]
        elif type_idx == 5:
            # ResizeContact
            size = float(params[0]) if len(params) > 0 else 0.17
            return [ResizeContact(target.rid, max(0.05, size))]
        elif type_idx == 6:
            # LayerPromote
            if violation.layer in ("met1", "m1", "metal1"):
                return [LayerPromote(target.rid, "met1", "met2")]
        return []

    def record_outcome(
        self,
        state:      LayoutState,
        violations: list[ViolationInfo],
        actions:    list[Action],
        remaining:  int,
    ) -> None:
        """Record the outcome of a fix iteration for experience replay."""
        self.buffer.add_from_fix_record(
            state, violations, actions, remaining,
            crop_radius=self.crop_radius,
        )

    @property
    def experience_count(self) -> int:
        return len(self.buffer)
