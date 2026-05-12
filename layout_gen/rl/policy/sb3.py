"""
layout_gen.rl.policy.sb3 — sb3-contrib MaskablePPO integration.

Wraps :class:`LayoutPolicy` as a custom :class:`MaskableActorCriticPolicy`
so PPO can train it under the env's DRC reward. The wrapper also adds a
small value head that consumes the same trunk context as the action heads.

Why a custom subclass and not the default ``MaskableActorCriticPolicy``?
  * The target action head is **pointer-style** — its logits depend on
    per-polygon embeddings, not just a pooled feature vector. The default
    ``action_net`` is a single ``Linear`` and can't express that.
  * Sharing the encoder between actor and critic is more sample-efficient
    than duplicating it.

BC checkpoints saved by :class:`BCTrainer` can be loaded with
:func:`load_bc_into_sb3_policy` to warm-start PPO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import (
    MaskableMultiCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from layout_gen.rl.policy.network import (
    ActionLogits, LayoutPolicy, LayoutPolicyConfig,
)


# ── Custom policy ────────────────────────────────────────────────────────────

class MaskableLayoutPolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy backed by :class:`LayoutPolicy`.

    Parameters
    ----------
    observation_space, action_space, lr_schedule :
        Standard SB3 policy constructor args.
    layout_config :
        Override for the underlying :class:`LayoutPolicyConfig`. Must
        match the env's poly_cap / viol_cap / target_cap / mag_bins.
    value_hidden :
        Width of the value-head hidden layer.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space:      spaces.MultiDiscrete,
        lr_schedule,
        *,
        layout_config:     LayoutPolicyConfig | None = None,
        value_hidden:      int = 64,
        **kwargs: Any,
    ) -> None:
        # We don't use the default features_extractor / mlp_extractor /
        # action_net / value_net pipeline. Pass `net_arch=[]` so the
        # parent doesn't waste params building them. They still get
        # created but are never called by our overrides.
        kwargs.setdefault("net_arch", [])
        super().__init__(
            observation_space, action_space, lr_schedule, **kwargs,
        )

        self.layout_config = layout_config or LayoutPolicyConfig()
        self.layout_policy = LayoutPolicy(self.layout_config)
        self.value_head = nn.Sequential(
            nn.Linear(self.layout_config.d_trunk, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 1),
        )

        # Replace the parent's distribution with one keyed on this env's
        # MultiDiscrete nvec. Doing this in __init__ avoids any chance of
        # action_net / value_net being instantiated with the wrong shapes
        # before our overrides take effect.
        self.action_dist = MaskableMultiCategoricalDistribution(
            list(action_space.nvec),
        )

        # Re-build the optimizer so it sees our new params.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1.0),
            **self.optimizer_kwargs,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _flatten_logits(self, logits: ActionLogits) -> torch.Tensor:
        """Concat per-dim logits in MultiDiscrete order → (B, sum(nvec)).

        PLACE- and ROUTE-only heads carry zero-width tensors when their
        phase is disabled, so the concat order is stable regardless of
        config. The order matches :class:`ActionSpace`'s nvec layout:
        REPAIR base → PLACE block → ROUTE block.
        """
        return torch.cat([
            logits.kind, logits.target, logits.edge,
            logits.sign_x, logits.sign_y, logits.mag,
            logits.device, logits.x_bin, logits.y_bin, logits.orient,
            logits.net, logits.route_layer,
            logits.route_x_bin, logits.route_y_bin,
            logits.route_w_bin, logits.route_h_bin,
        ], dim=-1)

    def _logits_and_value(
        self, obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx, poly_emb, poly_pad = self.layout_policy.encode_state(obs)
        logits = self.layout_policy.heads(ctx, poly_emb, poly_pad)
        flat   = self._flatten_logits(logits)
        value  = self.value_head(ctx).squeeze(-1)
        return flat, value

    # ── MaskableActorCriticPolicy overrides ──────────────────────────────────

    def forward(  # type: ignore[override]
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
        action_masks:  torch.Tensor | None = None,
    ):
        flat_logits, value = self._logits_and_value(obs)
        distribution = self.action_dist.proba_distribution(action_logits=flat_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions   = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        return actions, value, log_probs

    def evaluate_actions(  # type: ignore[override]
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ):
        flat_logits, value = self._logits_and_value(obs)
        distribution = self.action_dist.proba_distribution(action_logits=flat_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_probs = distribution.log_prob(actions)
        entropy   = distribution.entropy()
        return value, log_probs, entropy

    def predict_values(  # type: ignore[override]
        self, obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ctx, _, _ = self.layout_policy.encode_state(obs)
        return self.value_head(ctx).squeeze(-1)

    def _predict(  # type: ignore[override]
        self,
        observation: dict[str, torch.Tensor],
        deterministic: bool = False,
        action_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_logits, _ = self._logits_and_value(observation)
        distribution = self.action_dist.proba_distribution(action_logits=flat_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution.get_actions(deterministic=deterministic)


# ── BC warm-start ────────────────────────────────────────────────────────────

def load_bc_into_sb3_policy(
    bc_ckpt:    str | Path,
    sb3_policy: MaskableLayoutPolicy,
    *,
    strict:     bool = True,
) -> tuple[int, int]:
    """Copy BC-pretrained LayoutPolicy weights into ``sb3_policy.layout_policy``.

    The BC checkpoint stores ``state_dict`` for the bare LayoutPolicy.
    We load it into the wrapped policy's ``.layout_policy`` submodule;
    everything else (value_head, optimizer state) is left at its
    randomly-initialised values.

    Returns
    -------
    (loaded, missing) :
        Number of params loaded vs. missing. Useful for sanity-checking
        that the BC checkpoint matches the PPO policy's layout config.
    """
    ckpt = torch.load(bc_ckpt, map_location="cpu", weights_only=False)
    bc_state = ckpt["state_dict"]

    sb3_state = sb3_policy.layout_policy.state_dict()
    matched: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for k, v in bc_state.items():
        if k in sb3_state and sb3_state[k].shape == v.shape:
            matched[k] = v
        else:
            missing.append(k)

    if strict and missing:
        raise RuntimeError(
            f"BC checkpoint has {len(missing)} keys that don't match the "
            f"PPO policy's layout config (sample: {missing[:3]}). "
            "Make sure layout_config matches the one used during BC."
        )

    sb3_policy.layout_policy.load_state_dict(matched, strict=False)
    return len(matched), len(missing)


__all__ = ["MaskableLayoutPolicy", "load_bc_into_sb3_policy"]
