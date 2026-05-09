"""
layout_gen.repair.diagnose — inspect what a trained denoiser actually
predicts.

Run::

    python -m layout_gen.repair.diagnose \\
        --checkpoint layout_gen/repair/data/denoiser_v1.pt \\
        --data       layout_gen/repair/data/trajectories \\
        --n          50

Reports:
* Confusion matrix: predicted kind × true kind
* Top-K target accuracy (1, 5, 10) — proxy for "did the model at least
  point near the right polygon"?
* Magnitude error distribution
* Whether the model is collapsing to a single class
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from layout_gen.repair.dataset  import TrajectoryDataset, make_dataloader
from layout_gen.repair.features import ACTION_KINDS, EDGE_NAMES
from layout_gen.repair.model    import DRCDenoiser


def diagnose(
    checkpoint: Path,
    data_dir:   Path,
    n_samples:  int = 50,
    seed:       int = 0,
) -> None:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})
    print(f"Checkpoint: {checkpoint}")
    print(f"  config: {cfg}")
    print(f"  best_val: {ckpt.get('best_val', '?')}")

    model = DRCDenoiser(
        hidden_dim=cfg.get("hidden_dim", 64),
        n_layers=cfg.get("n_layers", 2),
        n_heads=cfg.get("n_heads", 4),
    )
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError as e:
        print(f"  ⚠ state_dict mismatch: {e}")
        print(f"  (likely feature-dim change; cannot diagnose this checkpoint)")
        return
    model.eval()

    ds = TrajectoryDataset(data_dir, expand_steps=True, d4_augment=False)
    loader = make_dataloader(ds, batch_size=8, shuffle=True)

    # ── Walk a few batches ──────────────────────────────────────────────────
    n_kinds = len(ACTION_KINDS)
    confusion = torch.zeros(n_kinds, n_kinds, dtype=torch.long)
    top1 = top5 = top10 = total = 0
    mag_errors: list[float] = []
    pred_kind_counts: dict[int, int] = {}
    seen = 0

    with torch.no_grad():
        for batch in loader:
            pred = model(batch.poly_feats, batch.poly_mask, batch.k)
            kind_pred = pred["kind_logits"].argmax(dim=-1)
            tgt_logits = pred["target_logits"]

            for i in range(batch.k.shape[0]):
                if seen >= n_samples:
                    break
                seen += 1
                tk, pk = int(batch.action_kind[i]), int(kind_pred[i])
                if tk >= 0:
                    confusion[tk, pk] += 1
                pred_kind_counts[pk] = pred_kind_counts.get(pk, 0) + 1

                # Top-K target — only when the target is valid
                if int(batch.target_idx[i]) >= 0:
                    total += 1
                    valid = batch.poly_mask[i]
                    logits = tgt_logits[i].clone()
                    logits[~valid] = float("-inf")
                    top = torch.topk(logits, k=min(10, int(valid.sum().item())))
                    rank_set = set(int(t) for t in top.indices.tolist())
                    target = int(batch.target_idx[i])
                    if target == int(top.indices[0]):
                        top1 += 1
                    if target in [int(t) for t in top.indices[:5]]:
                        top5 += 1
                    if target in rank_set:
                        top10 += 1

                # Magnitude error
                mag_p = pred["magnitude"][i]
                mag_t = batch.magnitude[i]
                mag_errors.append(float(torch.norm(mag_p - mag_t).item()))
            if seen >= n_samples:
                break

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\nDiagnosed {seen} samples\n")

    print("=== KIND CONFUSION (rows=true, cols=predicted) ===")
    header = "true\\pred  " + "  ".join(f"{k[:10]:>10s}" for k in ACTION_KINDS)
    print(header)
    for ti, kt in enumerate(ACTION_KINDS):
        row = f"{kt[:10]:10s}  "
        for pi in range(n_kinds):
            row += f"{int(confusion[ti, pi]):>10d}  "
        print(row)
    n_correct = int(confusion.diag().sum().item())
    n_kind = int(confusion.sum().item())
    print(f"\nKind top-1 acc: {n_correct}/{n_kind} = "
          f"{100 * n_correct / max(n_kind, 1):.1f}%")

    # Is the model collapsing to a single prediction?
    print("\nPredicted kind distribution:")
    total_pred = sum(pred_kind_counts.values()) or 1
    for k_idx, cnt in sorted(pred_kind_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {ACTION_KINDS[k_idx]:>14s}  {cnt:>3d}  "
              f"({100 * cnt / total_pred:.1f}%)")

    # Top-K target
    print(f"\n=== TARGET POINTING ===")
    print(f"  top-1   : {top1}/{total} = {100 * top1 / max(total, 1):.1f}%")
    print(f"  top-5   : {top5}/{total} = {100 * top5 / max(total, 1):.1f}%")
    print(f"  top-10  : {top10}/{total} = {100 * top10 / max(total, 1):.1f}%")

    # Magnitude
    if mag_errors:
        mag_errors.sort()
        print(f"\n=== MAGNITUDE L2 ERROR ===")
        print(f"  median: {mag_errors[len(mag_errors)//2]:.4f}")
        print(f"  p90   : {mag_errors[int(len(mag_errors)*0.9)]:.4f}")
        print(f"  max   : {mag_errors[-1]:.4f}")


def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data",       type=Path,
                   default=Path("layout_gen/repair/data/trajectories"))
    p.add_argument("--n",          type=int, default=50)
    p.add_argument("--seed",       type=int, default=0)
    args = p.parse_args()
    diagnose(args.checkpoint, args.data, args.n, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
