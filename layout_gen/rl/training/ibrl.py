"""
layout_gen.rl.training.ibrl — IBRL via behaviour-cloning distillation.

The classical IBRL (`arXiv 2311.02198`) keeps a frozen BC policy
alongside the RL policy and **mixes BC-proposed actions into the
rollout buffer** with a decaying mixing-rate β. That requires
overriding SB3's collect_rollouts, which is invasive.

This module ships a more SB3-friendly variant: a **distillation
penalty** added to the PPO policy loss. The auxiliary term is

    L_distill(t) = β(t) · KL( π_PPO(·|s) || π_BC(·|s) )

with β decaying linearly from ``beta_start`` (early in training, where
the BC policy is the better proposer) to 0 (late training, where the
RL policy should be free to outperform BC). The policy weights stay
close to BC for the first ~half of training and then specialise.

Implementation
--------------
:class:`MaskableBCDistillPPO` subclasses ``MaskablePPO`` and overrides
``train()`` to add the KL term. The BC policy is loaded from a saved
checkpoint at construction; its weights are frozen
(``requires_grad=False``).

Caveat: this is a **regulariser**, not strict IBRL — there's no
mixing-into-rollouts, just a soft pull toward BC. In practice both
flavours achieve the same headline goal (don't drift catastrophically
from the BC checkpoint) and the distillation flavour costs ~5% extra
training time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.distributions import (
    MaskableMultiCategoricalDistribution,
)

from layout_gen.rl.policy.network import LayoutPolicy, LayoutPolicyConfig
from layout_gen.rl.policy.sb3 import MaskableLayoutPolicy


class MaskableBCDistillPPO(MaskablePPO):
    """MaskablePPO with an added KL-to-BC penalty in the policy loss.

    Parameters
    ----------
    bc_checkpoint :
        Path to a BC checkpoint (saved by :class:`BCTrainer`).
    bc_policy_config :
        :class:`LayoutPolicyConfig` matching the BC checkpoint. Must
        agree with the live PPO policy's config so the action spaces
        line up.
    beta_start :
        Initial KL weight. Decays linearly to ``beta_end`` over training.
    beta_end :
        Final KL weight (default 0 = full release of BC influence).
    """

    def __init__(
        self,
        *args,
        bc_checkpoint:    str | Path,
        bc_policy_config: LayoutPolicyConfig,
        beta_start:       float = 1.0,
        beta_end:         float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._bc_policy = LayoutPolicy(bc_policy_config)
        ckpt = torch.load(str(bc_checkpoint), map_location=self.device,
                          weights_only=False)
        self._bc_policy.load_state_dict(ckpt["state_dict"], strict=False)
        self._bc_policy.to(self.device).eval()
        for p in self._bc_policy.parameters():
            p.requires_grad = False
        self._beta_start = float(beta_start)
        self._beta_end   = float(beta_end)
        self._distill_dist = MaskableMultiCategoricalDistribution(
            list(self.action_space.nvec),
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _current_beta(self) -> float:
        """Linear decay from beta_start → beta_end over total_timesteps."""
        if self._total_timesteps <= 0:
            return self._beta_start
        progress = min(1.0, max(0.0, self.num_timesteps / self._total_timesteps))
        return self._beta_start + (self._beta_end - self._beta_start) * progress

    def _flatten_bc_logits(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the frozen BC policy + concat its per-dim logits in nvec
        order. Shares the flatten convention with
        :class:`MaskableLayoutPolicy._flatten_logits`."""
        with torch.no_grad():
            ctx, poly_emb, poly_pad = self._bc_policy.encode_state(obs)
            logits = self._bc_policy.heads(ctx, poly_emb, poly_pad)
        return torch.cat([
            logits.kind, logits.target, logits.edge,
            logits.sign_x, logits.sign_y, logits.mag,
            logits.device, logits.x_bin, logits.y_bin, logits.orient,
            logits.net, logits.route_layer,
            logits.route_x_bin, logits.route_y_bin,
            logits.route_w_bin, logits.route_h_bin,
        ], dim=-1)

    # ── Loss override ────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run one PPO epoch; same as MaskablePPO.train but adds a
        KL-to-BC term to each minibatch's policy loss.

        We compute the KL between the live PPO policy's per-dim
        categoricals and the BC policy's per-dim categoricals on the
        same observations, then add ``β * KL`` to the policy loss
        before backprop.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        beta = self._current_beta()

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space.shape, tuple):
                    actions = actions.long().flatten()
                else:
                    actions = actions.long()

                # MaskablePPO requires masks during evaluate_actions.
                action_masks = None
                if hasattr(rollout_data, "action_masks"):
                    action_masks = rollout_data.action_masks

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    action_masks=action_masks,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf, clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                if entropy is None:
                    entropy_loss = -log_prob.mean()
                else:
                    entropy_loss = -entropy.mean()

                # ── Distillation term ──────────────────────────────
                if beta > 0:
                    bc_flat = self._flatten_bc_logits(rollout_data.observations)
                    # Build the same MaskableMultiCategorical the policy uses,
                    # mask it identically, then compute KL per dim.
                    distill_loss = self._kl_to_bc(
                        rollout_data.observations,
                        action_masks, bc_flat,
                    )
                else:
                    distill_loss = torch.zeros((), device=self.device)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef  * value_loss
                    + beta          * distill_loss
                )

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm,
                )
                self.policy.optimizer.step()

        self._n_updates += self.n_epochs
        # Logger surface
        self.logger.record("train/distill_beta",     float(beta))
        self.logger.record("train/policy_loss",      float(policy_loss.item()))
        self.logger.record("train/value_loss",       float(value_loss.item()))
        if beta > 0:
            self.logger.record("train/distill_loss", float(distill_loss.item()))

    def _kl_to_bc(
        self,
        observations:  dict[str, torch.Tensor],
        action_masks:  torch.Tensor | None,
        bc_flat:       torch.Tensor,
    ) -> torch.Tensor:
        """KL( π_PPO(·|s) || π_BC(·|s) ) summed across MultiDiscrete dims."""
        # Live policy's logits (gradient flows through these).
        if isinstance(self.policy, MaskableLayoutPolicy):
            ppo_flat, _ = self.policy._logits_and_value(observations)
        else:
            # Fallback path — should not be reached in our setup but
            # keeps the class generic for testing.
            actions, _, _ = self.policy.forward(
                observations, action_masks=action_masks,
            )
            ppo_flat = self.policy.action_dist.distribution[0].logits
            for d in self.policy.action_dist.distribution[1:]:
                ppo_flat = torch.cat([ppo_flat, d.logits], dim=-1)

        # Apply identical masking so KL ignores impossible actions.
        if action_masks is not None:
            mask = action_masks.bool()
            ppo_flat = ppo_flat.masked_fill(~mask, -1e8)
            bc_flat  = bc_flat.masked_fill(~mask, -1e8)

        # Split into per-dim categorical pieces.
        nvec = list(self.action_space.nvec)
        offsets = [0]
        for n in nvec:
            offsets.append(offsets[-1] + int(n))
        kl_total = torch.zeros((), device=ppo_flat.device, dtype=ppo_flat.dtype)
        for i, n in enumerate(nvec):
            a, b = offsets[i], offsets[i + 1]
            log_p = F.log_softmax(ppo_flat[:, a:b], dim=-1)
            log_q = F.log_softmax(bc_flat[:,  a:b], dim=-1)
            p = log_p.exp()
            kl = (p * (log_p - log_q)).sum(dim=-1).mean()
            kl_total = kl_total + kl
        return kl_total


__all__ = ["MaskableBCDistillPPO"]
