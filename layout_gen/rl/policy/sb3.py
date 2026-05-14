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
        device_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode + flatten heads into a single (B, sum(nvec)) logit tensor.

        ``device_idx`` is a passthrough to :meth:`LayoutPolicy.heads` for
        the device-conditioned PLACE regime; ignored when
        ``couple_device_position`` is False. Callers that need the
        autoregressive PLACE sample should use :meth:`_sample_device`
        first and feed the result back here.
        """
        ctx, poly_emb, poly_pad = self.layout_policy.encode_state(obs)
        logits = self.layout_policy.heads(
            ctx, poly_emb, poly_pad, device_idx=device_idx,
        )
        flat   = self._flatten_logits(logits)
        value  = self.value_head(ctx).squeeze(-1)
        return flat, value

    # ── Autoregressive PLACE coupling ────────────────────────────────────────

    @property
    def _couples_device_position(self) -> bool:
        return bool(
            self.layout_config.enable_place
            and self.layout_config.couple_device_position
        )

    def _device_dim_index(self) -> int:
        """Index of the ``device`` slot in the MultiDiscrete action vector.

        REPAIR base block (kind/target/edge/sx/sy/mag) is always 6 dims;
        the device slot is the first PLACE-only dim.
        """
        return 6  # kind, target, edge, sign_x, sign_y, mag

    def _device_logit_slice(self) -> tuple[int, int]:
        """[start, stop) into the flat-logit / action-mask vector for the
        device dim. Used to pull just the device-portion of the mask
        when sampling the device in pass 1 of the autoregressive forward."""
        nvec = list(self.action_dist.action_dims)
        device_dim = self._device_dim_index()
        start = sum(nvec[:device_dim])
        stop  = start + nvec[device_dim]
        return start, stop

    def _sample_device(
        self,
        obs:           dict[str, torch.Tensor],
        action_masks:  torch.Tensor | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the encoder + device head to autoregressively sample the
        device dim.

        Returns
        -------
        device_idx : (B,) long
            The sampled (or argmaxed) device index, masked by the
            device-portion of ``action_masks``.
        ctx, poly_emb, poly_pad :
            The encoded state — handed back so the caller doesn't pay
            for a second encode_state pass.
        """
        ctx, poly_emb, poly_pad = self.layout_policy.encode_state(obs)
        device_logits = self.layout_policy.device_head(ctx)   # (B, device_cap)

        if action_masks is not None:
            d_lo, d_hi = self._device_logit_slice()
            # action_masks arrives as np.ndarray during PPO rollout
            # collection and as torch.Tensor at evaluate-actions time;
            # normalise to a bool tensor on the policy's device so the
            # downstream torch.where is happy under both call paths.
            dev_mask = torch.as_tensor(
                action_masks[:, d_lo:d_hi],
                device=device_logits.device,
            ).bool()
            # Replace masked-off slots with a very negative value rather
            # than -inf so the categorical never sees an all-(-inf) row
            # (which would NaN the log-softmax).
            neg_inf = torch.full_like(device_logits, -1e9)
            device_logits = torch.where(dev_mask, device_logits, neg_inf)

        if deterministic:
            device_idx = device_logits.argmax(dim=-1)
        else:
            device_idx = torch.distributions.Categorical(
                logits=device_logits,
            ).sample()
        return device_idx, ctx, poly_emb, poly_pad

    def _coupled_logits_and_value(
        self,
        obs:           dict[str, torch.Tensor],
        action_masks:  torch.Tensor | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two-pass forward: sample device first, then build flat logits
        with position heads conditioned on the sampled device.

        Returns
        -------
        flat_logits : (B, sum(nvec))
            Flat logits whose position slots are conditioned on
            ``device_idx``. Other slots (kind / target / etc.) are
            identical to the un-coupled forward.
        value : (B,)
            Trunk value estimate.
        device_idx : (B,) long
            The device sampled in pass 1; the caller must overwrite the
            device dim of the final action with this value to keep the
            autoregressive contract (``p(x|d)`` uses the actual ``d``).
        """
        device_idx, ctx, poly_emb, poly_pad = self._sample_device(
            obs, action_masks, deterministic,
        )
        logits = self.layout_policy.heads(
            ctx, poly_emb, poly_pad, device_idx=device_idx,
        )
        flat  = self._flatten_logits(logits)
        value = self.value_head(ctx).squeeze(-1)
        return flat, value, device_idx

    # ── MaskableActorCriticPolicy overrides ──────────────────────────────────

    def forward(  # type: ignore[override]
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
        action_masks:  torch.Tensor | None = None,
    ):
        if self._couples_device_position:
            flat_logits, value, device_idx = self._coupled_logits_and_value(
                obs, action_masks, deterministic,
            )
        else:
            flat_logits, value = self._logits_and_value(obs)
            device_idx = None
        distribution = self.action_dist.proba_distribution(action_logits=flat_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        if device_idx is not None:
            # The position dims were drawn from a distribution conditioned
            # on this device, so the action's device dim must agree.
            actions[:, self._device_dim_index()] = device_idx
        log_probs = distribution.log_prob(actions)
        return actions, value, log_probs

    def evaluate_actions(  # type: ignore[override]
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ):
        if self._couples_device_position:
            # Condition position heads on the device that was actually
            # taken in this transition (PPO is on-policy w.r.t. its own
            # rollouts, so this matches the sampling-time conditioning).
            device_idx = actions[:, self._device_dim_index()].long()
            flat_logits, value = self._logits_and_value(obs, device_idx=device_idx)
        else:
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
        if self._couples_device_position:
            flat_logits, _, device_idx = self._coupled_logits_and_value(
                observation, action_masks, deterministic,
            )
        else:
            flat_logits, _ = self._logits_and_value(observation)
            device_idx = None
        distribution = self.action_dist.proba_distribution(action_logits=flat_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        if device_idx is not None:
            actions[:, self._device_dim_index()] = device_idx
        return actions


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
