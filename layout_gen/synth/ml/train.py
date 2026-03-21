"""
layout_gen.synth.ml.train — CLI training script for the margin predictor.

Generates a synthetic dataset, trains a :class:`MarginPredictor`, reports
per-margin MAE and sign accuracy, then saves the model to disk.

Usage::

    python -m layout_gen.synth.ml.train [--samples N] [--out PATH] [--seed S]

Options
-------
--samples N     Number of training samples (default: 20 000).
--out PATH      Where to save the model pickle (default: model.pkl).
--seed S        NumPy / sklearn random seed (default: 42).
--quiet         Suppress per-margin table; print only final summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from layout_gen                    import load_pdk
from layout_gen.synth.ml.dataset   import generate_dataset
from layout_gen.synth.ml.features  import MARGIN_NAMES
from layout_gen.synth.ml.model     import MarginPredictor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the same sign as ground truth."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _print_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    quiet:  bool = False,
) -> None:
    mae_total = float(np.mean(np.abs(y_test - y_pred)))
    sacc_total = _sign_accuracy(y_test, y_pred)

    if not quiet:
        col_w = max(len(n) for n in MARGIN_NAMES) + 2
        header = f"{'Margin':<{col_w}}  {'MAE (µm)':>10}  {'Sign acc':>10}"
        print(header)
        print("-" * len(header))
        for i, name in enumerate(MARGIN_NAMES):
            mae  = float(np.mean(np.abs(y_test[:, i] - y_pred[:, i])))
            sacc = _sign_accuracy(y_test[:, i], y_pred[:, i])
            print(f"{name:<{col_w}}  {mae:>10.4f}  {sacc:>9.1%}")
        print("-" * len(header))

    print(f"\nOverall MAE : {mae_total:.4f} µm")
    print(f"Sign acc    : {sacc_total:.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

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
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress per-margin table")
    args = parser.parse_args(argv)

    # ── Load PDK ──────────────────────────────────────────────────────────────
    print("Loading PDK rules …")
    rules = load_pdk()

    # ── Generate dataset ──────────────────────────────────────────────────────
    print(f"Generating {args.samples:,} samples (seed={args.seed}) …")
    ds = generate_dataset(rules, n_samples=args.samples, seed=args.seed)
    print(f"  Dataset size: {ds.X.shape[0]:,}  (skipped "
          f"{args.samples - ds.X.shape[0]:,} degenerate combos)")

    # ── Train/test split (80/20) ──────────────────────────────────────────────
    rng    = np.random.default_rng(args.seed)
    idx    = rng.permutation(len(ds.X))
    split  = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    X_train, y_train = ds.X[tr], ds.y[tr]
    X_test,  y_test  = ds.X[te], ds.y[te]

    # ── Fit model ─────────────────────────────────────────────────────────────
    print(f"Training MLP ({len(X_train):,} samples) …")
    mp = MarginPredictor(random_state=args.seed)
    mp.fit(X_train, y_train)
    print("  Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = mp.predict(X_test)
    print(f"\nTest-set evaluation ({len(X_test):,} samples):\n")
    _print_report(y_test, y_pred, quiet=args.quiet)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    mp.save(out_path)
    print(f"\nModel saved → {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
