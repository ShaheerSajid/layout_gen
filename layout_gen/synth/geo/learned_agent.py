"""
layout_gen.synth.geo.learned_agent — Phase 3: ML-based geometric fix agent.

Replaces :class:`~.agent.RuleGeoAgent` with a trained policy that selects
geometric actions from the same action space.

Architecture
------------
The model sees:
- A local crop of the layout around the violation (set of rectangles
  with layer/coordinate features)
- The parsed violation info (category, layer, deficit, required value)

And outputs:
- An action type (spacing/width/enclosure fix)
- Action parameters (which rect, which edge, how much to stretch)

Training
--------
Phase 3a — **Imitation learning**: record the rule-based agent's decisions
    via :class:`~.loop.FixRecord` and train a policy to mimic them.  This
    provides a warm start.

Phase 3b — **Reinforcement learning**: fine-tune with reward =
    ``(violations_before − violations_after) − area_penalty``.
    The replay buffer collects (state, action, reward, next_state) tuples.

Phase 3c — **Cross-technology transfer**: train on sky130, fine-tune on
    TSMC 65nm with a small dataset.  The geometric intuitions transfer —
    "spacing violation → move edges apart" is universal.

Model options (to be evaluated)
-------------------------------
1. **Graph Neural Network** — layout as a graph where nodes = rects,
   edges = adjacency.  Captures spatial relationships naturally.
   Pro: equivariant to translation.  Con: harder to train.

2. **Set Transformer** — layout crop as an unordered set of rect features.
   Pro: simple, proven architecture.  Con: doesn't capture topology.

3. **CNN on rasterized crop** — render the local area as a multi-channel
   image (one channel per layer).  Pro: standard CV pipeline.  Con: fixed
   resolution, loses precision.

4. **Hybrid** — Set Transformer for the layout + MLP for violation features,
   fused to predict action parameters.  This is the recommended starting
   point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from layout_gen.synth.geo.state      import LayoutState
from layout_gen.synth.geo.actions    import Action, StretchEdge, MoveShape
from layout_gen.synth.geo.violations import ViolationInfo
from layout_gen.synth.geo.agent      import GeoFixAgent


log = logging.getLogger(__name__)


# ── Feature encoding ─────────────────────────────────────────────────────────

# Category → integer encoding
_CAT_IDX = {
    "spacing": 0, "width": 1, "enclosure": 2,
    "area": 3, "overlap": 4, "unknown": 5,
}

# Action type → integer encoding
_ACT_IDX = {
    "stretch_left": 0, "stretch_right": 1,
    "stretch_bottom": 2, "stretch_top": 3,
    "move_x": 4, "move_y": 5,
}
N_ACTION_TYPES = len(_ACT_IDX)


@dataclass
class Observation:
    """What the learned agent sees at each step.

    Attributes
    ----------
    layout_crop : np.ndarray
        (N, 6) array from ``LayoutState.local_crop()`` — nearby rects.
    violation_feat : np.ndarray
        (8,) vector: [category_idx, layer_hash, inner_layer_hash,
        measured, required, deficit, x, y].
    """
    layout_crop:    np.ndarray   # (N, 6) float32
    violation_feat: np.ndarray   # (8,) float32


def encode_observation(
    state:     LayoutState,
    violation: ViolationInfo,
    crop_radius: float = 2.0,
) -> Observation:
    """Build an observation from the layout state and a violation."""
    crop = state.local_crop(violation.x, violation.y, radius=crop_radius)

    vfeat = np.array([
        _CAT_IDX.get(violation.category, 5),
        hash(violation.layer) % 1000,
        hash(violation.inner_layer) % 1000 if violation.inner_layer else 0,
        violation.measured,
        violation.required,
        violation.deficit,
        violation.x,
        violation.y,
    ], dtype=np.float32)

    return Observation(layout_crop=crop, violation_feat=vfeat)


# ── Replay buffer ────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """One step of experience for RL training."""
    obs:        Observation
    action_idx: int
    action_param: float   # magnitude (delta for stretch, distance for move)
    target_rid: int       # which rect was acted on
    reward:     float
    done:       bool


class ReplayBuffer:
    """Fixed-size circular buffer for RL experience."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._buffer: list[Transition] = []
        self._pos = 0

    def push(self, transition: Transition) -> None:
        if len(self._buffer) < self.capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        rng = np.random.default_rng()
        indices = rng.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


# ── Learned agent (stub — Phase 3 implementation) ────────────────────────────

class LearnedGeoAgent(GeoFixAgent):
    """Phase 3: neural-network-based geometric fix agent.

    This is a skeleton implementation.  The actual model training and
    inference will be implemented when Phase 3 begins.

    The agent architecture:
    1. Encode local layout crop → Set Transformer → layout embedding
    2. Encode violation features → MLP → violation embedding
    3. Concatenate → MLP → action type logits + action parameter regression
    4. Sample action from policy (training) or take argmax (inference)

    Parameters
    ----------
    model_path : str | None
        Path to a saved model checkpoint.  If None, falls back to the
        rule-based agent.
    crop_radius : float
        Radius (µm) around violation centroid for local crop.
    fallback :
        Agent to use when model is not loaded.
    """

    def __init__(
        self,
        model_path:  str | None = None,
        crop_radius: float = 2.0,
        fallback:    GeoFixAgent | None = None,
    ):
        self.model_path  = model_path
        self.crop_radius = crop_radius
        self.fallback    = fallback
        self._model      = None
        self._replay     = ReplayBuffer()

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        try:
            import torch
            # TODO: Define and load the actual model architecture
            log.info("LearnedGeoAgent: model loading not yet implemented")
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
        """Use the trained model to propose a fix, or fall back."""
        if not self.is_trained:
            if self.fallback:
                return self.fallback.propose_fix(state, violation)
            return []

        # ── Inference (Phase 3 implementation) ───────────────────────────
        obs = encode_observation(state, violation, self.crop_radius)

        # TODO: Forward pass through model
        # action_logits, param_pred = self._model(obs)
        # action_type = action_logits.argmax()
        # delta = param_pred.item()

        # For now, fall back
        if self.fallback:
            return self.fallback.propose_fix(state, violation)
        return []

    def record_experience(
        self,
        obs:        Observation,
        action_idx: int,
        action_param: float,
        target_rid: int,
        reward:     float,
        done:       bool,
    ) -> None:
        """Store a transition in the replay buffer for training."""
        self._replay.push(Transition(
            obs=obs,
            action_idx=action_idx,
            action_param=action_param,
            target_rid=target_rid,
            reward=reward,
            done=done,
        ))

    def train_step(self, batch_size: int = 64) -> float:
        """One gradient step from replay buffer.

        Returns the training loss.  Requires PyTorch.
        """
        if len(self._replay) < batch_size:
            return 0.0

        # TODO: Implement actual training
        # batch = self._replay.sample(batch_size)
        # ... compute loss, backprop ...
        return 0.0

    @property
    def replay_size(self) -> int:
        return len(self._replay)
