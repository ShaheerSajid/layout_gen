"""
layout_gen.synth.ml.train — CLI training script for the margin predictor.

Generates a synthetic dataset (Sobol + boundary-enriched), trains an
ensemble :class:`MarginPredictor` (MLP + GradientBoosting), reports
per-model and per-margin metrics, then saves the model to disk.

Reports include:
- Per-sub-model MAE and sign accuracy (MLP vs GBR comparison)
- Ensemble MAE, sign accuracy, and uncertainty calibration
- Per-margin breakdown

Usage::

    python -m layout_gen.synth.ml.train [--samples N] [--out PATH] [--seed S]

Options
-------
--samples N     Number of training samples (default: 20 000).
--out PATH      Where to save the model pickle (default: model.pkl).
--seed S        NumPy / sklearn random seed (default: 42).
--no-gbr        Disable GradientBoosting (MLP only, faster but no uncertainty).
--quiet         Suppress per-margin table; print only final summary.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from layout_gen                    import load_pdk
from layout_gen.synth.ml.dataset   import generate_dataset
from layout_gen.synth.ml.features  import MARGIN_NAMES
from layout_gen.synth.ml.model     import MarginPredictor


# ── Helpers ───────────────────────────────────────────────────────────────

def _sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the same sign as ground truth."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _print_report(
    y_test:    np.ndarray,
    y_pred:    np.ndarray,
    label:     str  = "Ensemble",
    quiet:     bool = False,
    y_std:     np.ndarray | None = None,
) -> None:
    mae_total  = float(np.mean(np.abs(y_test - y_pred)))
    sacc_total = _sign_accuracy(y_test, y_pred)

    print(f"\n  [{label}]  MAE: {mae_total:.4f} um | Sign acc: {sacc_total:.1%}")

    if not quiet:
        col_w = max(len(n) for n in MARGIN_NAMES) + 2
        header = f"    {'Margin':<{col_w}}  {'MAE (um)':>10}  {'Sign acc':>10}"
        if y_std is not None:
            header += f"  {'Mean std':>10}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for i, name in enumerate(MARGIN_NAMES):
            mae  = float(np.mean(np.abs(y_test[:, i] - y_pred[:, i])))
            sacc = _sign_accuracy(y_test[:, i], y_pred[:, i])
            line = f"    {name:<{col_w}}  {mae:>10.4f}  {sacc:>9.1%}"
            if y_std is not None:
                mean_s = float(np.mean(y_std[:, i]))
                line += f"  {mean_s:>10.4f}"
            print(line)

    # Uncertainty calibration (if available)
    if y_std is not None:
        errors = np.abs(y_test - y_pred)
        within_1std = float(np.mean(errors <= y_std))
        within_2std = float(np.mean(errors <= 2 * y_std))
        print(f"\n    Uncertainty calibration:")
        print(f"      Errors within 1*std: {within_1std:.1%}  (ideal: ~68%)")
        print(f"      Errors within 2*std: {within_2std:.1%}  (ideal: ~95%)")


# ── Main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the layout_gen DRC margin predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int,   default=20_000,    metavar="N",
                        help="Number of training samples")
    parser.add_argument("--out",     type=str,   default="model.pkl", metavar="PATH",
                        help="Output model path")
    parser.add_argument("--seed",    type=int,   default=42,         metavar="S",
                        help="Random seed")
    parser.add_argument("--no-gbr",  action="store_true",
                        help="Disable GradientBoosting (MLP only)")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress per-margin table")
    args = parser.parse_args(argv)

    # ── Load PDK ──────────────────────────────────────────────────────────
    print("Loading PDK rules ...")
    rules = load_pdk()

    # ── Generate dataset ──────────────────────────────────────────────────
    print(f"Generating {args.samples:,} samples (Sobol + boundary-enriched, "
          f"seed={args.seed}) ...")
    t0 = time.time()
    ds = generate_dataset(rules, n_samples=args.samples, seed=args.seed)
    dt = time.time() - t0
    n_neg = int((ds.y < 0).any(axis=1).sum())
    print(f"  Dataset: {ds.X.shape[0]:,} samples in {dt:.1f}s")
    print(f"  Violation samples: {n_neg:,} ({n_neg/max(len(ds.y),1):.0%})")

    # ── Train/test split (80/20) ──────────────────────────────────────────
    rng    = np.random.default_rng(args.seed)
    idx    = rng.permutation(len(ds.X))
    split  = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    X_train, y_train = ds.X[tr], ds.y[tr]
    X_test,  y_test  = ds.X[te], ds.y[te]

    # ── Fit model ─────────────────────────────────────────────────────────
    use_gbr = not args.no_gbr
    model_desc = "MLP + GradientBoosting" if use_gbr else "MLP only"
    print(f"\nTraining ensemble ({model_desc}, {len(X_train):,} samples) ...")
    t0 = time.time()
    mp = MarginPredictor(random_state=args.seed, use_gbr=use_gbr)
    mp.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"  Training complete in {dt:.1f}s")
    print(f"  Sub-models: {[name for name, _ in mp._models]}")

    # ── Evaluate each sub-model individually ──────────────────────────────
    print(f"\nTest-set evaluation ({len(X_test):,} samples):")
    for name, pipe in mp._models:
        y_sub = pipe.predict(X_test)
        _print_report(y_test, y_sub, label=name.upper(), quiet=args.quiet)

    # ── Evaluate ensemble ─────────────────────────────────────────────────
    result = mp.predict(X_test, return_std=True)
    if isinstance(result, tuple):
        y_pred, y_std = result
    else:
        y_pred, y_std = result, None

    _print_report(y_test, y_pred, label="ENSEMBLE", quiet=args.quiet, y_std=y_std)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    mp.save(out_path)
    print(f"\nModel saved -> {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
