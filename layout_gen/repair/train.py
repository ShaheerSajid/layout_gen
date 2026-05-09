"""
layout_gen.repair.train — train the diffusion-style DRC denoiser.

Run::

    PDK_ROOT=/usr/local/share/pdk python -m layout_gen.repair.train \\
        --data layout_gen/repair/data/trajectories \\
        --epochs 50 --batch-size 16 \\
        --out layout_gen/repair/data/denoiser.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Be polite about CPU: small models like ours don't benefit from saturating
# every core, and the user explicitly asked us to stay moderate.  Default
# to 2 threads; override with TORCH_NUM_THREADS env var.
_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))
torch.set_num_threads(_THREADS)

from layout_gen.repair.dataset import TrajectoryDataset, make_dataloader
from layout_gen.repair.model   import DRCDenoiser, denoiser_loss


def train(
    data_dir:    Path,
    out_path:    Path,
    *,
    epochs:      int   = 50,
    batch_size:  int   = 16,
    lr:          float = 3e-4,
    warmup_frac: float = 0.1,
    val_split:   float = 0.1,
    seed:        int   = 0,
    hidden_dim:  int   = 64,
    n_layers:    int   = 2,
    n_heads:     int   = 4,
    pdks:        tuple[str, ...] = (),
    primitives:  tuple[str, ...] = (),
    depths:      tuple[int, ...] = (),
    expand_steps: bool = True,
    d4_augment:   bool = True,
    use_class_weights: bool = True,
    device:      str   = "cpu",
    verbose:     bool  = True,
) -> dict:
    torch.manual_seed(seed)

    full_dataset = TrajectoryDataset(
        data_dir, pdks=pdks, primitives=primitives, depths=depths,
        expand_steps=expand_steps, d4_augment=d4_augment,
    )
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = make_dataloader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = make_dataloader(val_set,   batch_size=batch_size, shuffle=False)

    if verbose:
        s = full_dataset.stats()
        print(f"Dataset: {s['n_records']} records → {s['n_samples']} samples "
              f"(train={n_train}, val={n_val})  expand={s['expand_steps']}")
        print(f"  by_pdk:      {s['by_pdk']}")
        print(f"  by_sample_k: {s['by_sample_k']}")
        print(f"  by_kind:     {s['by_kind']}")

    # Class weights: counter the imbalance on the kind head
    kind_weights = None
    if use_class_weights:
        kind_weights = full_dataset.kind_class_weights().to(device)
        if verbose:
            from layout_gen.repair.features import ACTION_KINDS
            print("  kind_weights: " + ", ".join(
                f"{k}={w:.2f}" for k, w in zip(ACTION_KINDS, kind_weights.tolist())
            ))

    model = DRCDenoiser(
        hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model: {n_params:,} parameters "
              f"(hidden={hidden_dim} layers={n_layers} heads={n_heads})")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # LR schedule: linear warmup → cosine decay to 1/10 of base lr
    steps_per_epoch = max(1, len(train_loader))
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = max(1, int(total_steps * warmup_frac))
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay to 0.1
        import math
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)

    history: list[dict] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_total = 0.0
        train_n     = 0
        for batch in train_loader:
            batch_t = _move(batch, device)
            pred = model(
                batch_t.poly_feats, batch_t.poly_mask, batch_t.k,
                violation_xy=batch_t.violation_xy,
                rule_cat=batch_t.rule_cat,
            )
            loss, _ = denoiser_loss(
                pred,
                action_kind=batch_t.action_kind,
                target_idx=batch_t.target_idx,
                edge_idx=batch_t.edge_idx,
                magnitude=batch_t.magnitude,
                target_xy=batch_t.target_xy,
                kind_weights=kind_weights,
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            scheduler.step()
            train_total += float(loss.item()) * batch_t.k.shape[0]
            train_n     += batch_t.k.shape[0]
        train_avg = train_total / max(train_n, 1)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_total = 0.0
        val_n     = 0
        kind_correct = 0
        kind_n       = 0
        target_correct = 0
        target_n       = 0
        snap_correct = 0
        snap_n       = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_t = _move(batch, device)
                pred = model(
                batch_t.poly_feats, batch_t.poly_mask, batch_t.k,
                violation_xy=batch_t.violation_xy,
                rule_cat=batch_t.rule_cat,
            )
                loss, _ = denoiser_loss(
                    pred,
                    action_kind=batch_t.action_kind,
                    target_idx=batch_t.target_idx,
                    edge_idx=batch_t.edge_idx,
                    magnitude=batch_t.magnitude,
                    target_xy=batch_t.target_xy,
                    kind_weights=kind_weights,
                )
                val_total += float(loss.item()) * batch_t.k.shape[0]
                val_n     += batch_t.k.shape[0]

                # Top-1 accuracies
                kind_pred = pred["kind_logits"].argmax(dim=-1)
                kmask = batch_t.action_kind >= 0
                kind_correct += int((kind_pred[kmask] == batch_t.action_kind[kmask]).sum())
                kind_n       += int(kmask.sum())

                # Original "pointer" target accuracy
                target_pred = pred["target_logits"].argmax(dim=-1)
                tmask = batch_t.target_idx >= 0
                target_correct += int((target_pred[tmask] == batch_t.target_idx[tmask]).sum())
                target_n       += int(tmask.sum())

                # Centroid-snap accuracy: predicted (x, y) snapped to
                # the nearest valid polygon by centroid distance — this
                # is what the inference path will use.
                xy_pred = pred["target_xy"]   # (B, 2)
                # Polygon centroids from poly_feats: indices N_LAYER_ROLES + 0/1
                # are cx, cy in [0, 1]
                from layout_gen.repair.features import N_LAYER_ROLES
                poly_xy = batch_t.poly_feats[..., N_LAYER_ROLES:N_LAYER_ROLES+2]
                # (B, N, 2) – (B, 1, 2) → (B, N, 2) → (B, N) distances
                dists = torch.norm(poly_xy - xy_pred.unsqueeze(1), dim=-1)
                # Mask out padding polys with +inf
                dists = dists.masked_fill(~batch_t.poly_mask, float("inf"))
                snap_pred = dists.argmin(dim=-1)
                snap_correct += int((snap_pred[tmask] == batch_t.target_idx[tmask]).sum())
                snap_n       += int(tmask.sum())

        val_avg = val_total / max(val_n, 1)
        kind_acc   = kind_correct   / max(kind_n,   1)
        target_acc = target_correct / max(target_n, 1)
        snap_acc   = snap_correct   / max(snap_n,   1)
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": train_avg, "val_loss": val_avg,
            "kind_acc":   kind_acc,  "target_acc": target_acc,
            "snap_acc":   snap_acc,
            "elapsed":    elapsed,
        })
        if verbose:
            print(f"epoch {epoch:3d}/{epochs}  "
                  f"train={train_avg:.4f}  val={val_avg:.4f}  "
                  f"kind_acc={kind_acc:.2%}  "
                  f"target_acc={target_acc:.2%}  snap_acc={snap_acc:.2%}  "
                  f"({elapsed:.1f}s)")

        if val_avg < best_val:
            best_val = val_avg
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}

    # Save best model
    if best_state is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":  best_state,
            "history":     history,
            "best_val":    best_val,
            "config":      {
                "hidden_dim": hidden_dim,
                "n_layers":   n_layers,
                "n_heads":    n_heads,
            },
        }, out_path)
        if verbose:
            print(f"\nBest val={best_val:.4f}; saved to {out_path}")

    return {"history": history, "best_val": best_val,
            "out_path": str(out_path)}


def _move(batch, device):
    return type(batch)(
        poly_feats   = batch.poly_feats.to(device),
        poly_mask    = batch.poly_mask.to(device),
        k            = batch.k.to(device),
        violation_xy = batch.violation_xy.to(device),
        rule_cat     = batch.rule_cat.to(device),
        action_kind  = batch.action_kind.to(device),
        target_idx   = batch.target_idx.to(device),
        target_xy    = batch.target_xy.to(device),
        edge_idx     = batch.edge_idx.to(device),
        magnitude    = batch.magnitude.to(device),
    )


def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path("layout_gen/repair/data/trajectories"))
    p.add_argument("--out",  type=Path,
                   default=Path("layout_gen/repair/data/denoiser.pt"))
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--val-split",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--hidden-dim",  type=int,   default=64)
    p.add_argument("--n-layers",    type=int,   default=2)
    p.add_argument("--n-heads",     type=int,   default=4)
    p.add_argument("--no-expand",   action="store_true",
                   help="Disable per-step trajectory expansion.")
    p.add_argument("--no-class-weights", action="store_true",
                   help="Disable inverse-frequency class weighting.")
    p.add_argument("--no-d4", action="store_true",
                   help="Disable D4 symmetry augmentation.")
    p.add_argument("--pdks",        nargs="*",  default=())
    p.add_argument("--primitives",  nargs="*",  default=())
    p.add_argument("--depths",      nargs="*",  type=int, default=())
    p.add_argument("--device",      default="cpu")
    args = p.parse_args()

    train(
        data_dir=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_frac=args.warmup_frac,
        val_split=args.val_split,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        expand_steps=not args.no_expand,
        d4_augment=not args.no_d4,
        use_class_weights=not args.no_class_weights,
        pdks=tuple(args.pdks),
        primitives=tuple(args.primitives),
        depths=tuple(args.depths),
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
