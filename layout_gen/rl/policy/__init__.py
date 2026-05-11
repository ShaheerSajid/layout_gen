"""
layout_gen.rl.policy — neural-network components for the layout RL agent.

The :class:`LayoutPolicy` module consumes the env's observation dict
and emits per-dim logits over the MultiDiscrete action space. It can
be trained standalone (BC pretrain) and later wrapped as a
``stable_baselines3`` custom features extractor for PPO fine-tuning.

The module is intentionally simple — a small transformer over polygon
tokens, a small transformer over violation tokens, masked mean
pooling, then per-dim Linear heads. Phase 4 (place / route actions)
adds new heads without changing the encoder.
"""
from __future__ import annotations

from layout_gen.rl.policy.network import (
    LayoutPolicy, LayoutPolicyConfig, ActionLogits,
    masked_cross_entropy,
)
from layout_gen.rl.policy.sb3 import (
    MaskableLayoutPolicy, load_bc_into_sb3_policy,
)

__all__ = [
    "LayoutPolicy", "LayoutPolicyConfig", "ActionLogits",
    "masked_cross_entropy",
    "MaskableLayoutPolicy", "load_bc_into_sb3_policy",
]
