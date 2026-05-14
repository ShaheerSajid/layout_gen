"""
layout_gen.rl.training.bc_pretrain — behaviour-cloning trainer.

Cross-entropy training of :class:`LayoutPolicy` on the (obs, action,
validity) tuples produced by :class:`TrajectoryDataset`. Each action
dim contributes its own loss term, weighted by the policy config's
``loss_weights`` and masked per-sample by ``validity`` so dims that
don't apply to a given sample contribute zero.

The trainer is deliberately minimal — no LR scheduler, no AMP, no
distributed support. Phase 3 wraps the trained policy as a
``MaskablePPO`` features extractor and continues training under
RL reward; that's the loop where infrastructure complexity belongs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from layout_gen.rl.policy.network import (
    ActionLogits, LayoutPolicy, LayoutPolicyConfig, masked_cross_entropy,
)
from layout_gen.rl.training.dataset import collate_samples


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class BCTrainerConfig:
    epochs:       int   = 5
    batch_size:   int   = 32
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    grad_clip:    float = 1.0
    num_workers:  int   = 0
    device:       str   = "cpu"
    log_every:    int   = 50


@dataclass
class BCMetrics:
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    per_dim:    list[dict[str, float]] = field(default_factory=list)
    accuracy:   list[dict[str, float]] = field(default_factory=list)

    def best_val(self) -> float:
        return min(self.val_loss) if self.val_loss else float("inf")


# ── Trainer ──────────────────────────────────────────────────────────────────

class BCTrainer:
    def __init__(
        self,
        policy:  LayoutPolicy,
        config:  BCTrainerConfig | None = None,
    ) -> None:
        self.policy = policy
        self.cfg    = config or BCTrainerConfig()
        self.device = torch.device(self.cfg.device)
        self.policy.to(self.device)

        self.opt = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.metrics = BCMetrics()

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(self, dataset: Dataset) -> BCMetrics:
        if len(dataset) == 0:
            raise ValueError("Empty dataset; mine some trajectories first.")

        train_set, val_set = self._split(dataset)
        train_loader = DataLoader(
            train_set, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, collate_fn=collate_samples,
        )
        val_loader = DataLoader(
            val_set, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, collate_fn=collate_samples,
        ) if len(val_set) > 0 else None

        for epoch in range(self.cfg.epochs):
            self._run_train_epoch(train_loader, epoch)
            if val_loader is not None:
                self._run_val_epoch(val_loader)
        return self.metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.policy.state_dict(),
            "config":     self.policy.cfg.__dict__,
        }, path)

    @classmethod
    def load(cls, path: str | Path,
             config: BCTrainerConfig | None = None) -> "BCTrainer":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        policy = LayoutPolicy(LayoutPolicyConfig(**ckpt["config"]))
        policy.load_state_dict(ckpt["state_dict"])
        return cls(policy, config)

    # ── Internals ────────────────────────────────────────────────────────────

    def _split(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        n = len(dataset)
        n_val = int(round(n * self.cfg.val_fraction))
        n_train = max(n - n_val, 1)
        n_val   = n - n_train
        if n_val == 0:
            return dataset, []
        gen = torch.Generator().manual_seed(0)
        return random_split(dataset, [n_train, n_val], generator=gen)

    def _run_train_epoch(self, loader: Iterable[dict], epoch: int) -> None:
        self.policy.train()
        running_loss = 0.0
        running_n    = 0
        for step, batch in enumerate(loader):
            obs = self._to_device(batch["obs"])
            actions  = self._to_device(batch["action"])
            validity = self._to_device(batch["validity"])

            logits = self.policy(obs, device_idx=self._target_device(actions))
            loss, breakdown = masked_cross_entropy(
                logits, actions, validity,
                weights=self.policy.cfg.loss_weights,
            )

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                            self.cfg.grad_clip)
            self.opt.step()

            running_loss += float(loss.detach().cpu().item()) * _bsz(batch)
            running_n    += _bsz(batch)

            if (step + 1) % self.cfg.log_every == 0:
                avg = running_loss / max(running_n, 1)
                self.metrics.train_loss.append(avg)
                self.metrics.per_dim.append(breakdown)

        avg = running_loss / max(running_n, 1)
        self.metrics.train_loss.append(avg)

    @torch.no_grad()
    def _run_val_epoch(self, loader: Iterable[dict]) -> None:
        self.policy.eval()
        total_loss = 0.0
        total_n    = 0
        correct: dict[str, int] = {}
        seen:    dict[str, int] = {}
        for batch in loader:
            obs = self._to_device(batch["obs"])
            actions  = self._to_device(batch["action"])
            validity = self._to_device(batch["validity"])

            logits = self.policy(obs, device_idx=self._target_device(actions))
            loss, _ = masked_cross_entropy(
                logits, actions, validity,
                weights=self.policy.cfg.loss_weights,
            )
            total_loss += float(loss.cpu().item()) * _bsz(batch)
            total_n    += _bsz(batch)

            for name, lg in logits._asdict().items():
                if name not in actions:
                    continue
                tgt = actions[name]
                v   = validity[name]
                pred = lg.argmax(dim=-1)
                hit  = ((pred == tgt) & v).sum().item()
                correct[name] = correct.get(name, 0) + int(hit)
                seen[name]    = seen.get(name, 0) + int(v.sum().item())

        self.metrics.val_loss.append(total_loss / max(total_n, 1))
        self.metrics.accuracy.append({
            k: (correct[k] / seen[k]) if seen[k] > 0 else float("nan")
            for k in correct
        })

    def _to_device(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in d.items()}

    def _target_device(
        self, actions: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Return the per-sample ground-truth device for conditioning the
        position heads, or None when the policy isn't in coupled mode.

        BC samples that lack a device label (REPAIR-only trajectories,
        zero-padded device dim) get a 0 here — those samples will have
        ``validity['x_bin'] == False`` so the conditioned logits never
        contribute to the loss. The dummy device just keeps the
        one-hot-concat math well-formed across the batch.
        """
        if not self.policy.cfg.couple_device_position:
            return None
        if "device" not in actions:
            return None
        return actions["device"].long()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bsz(batch: dict) -> int:
    sample_dim = next(iter(batch["obs"].values()))
    return int(sample_dim.shape[0])


__all__ = ["BCTrainer", "BCTrainerConfig", "BCMetrics"]
