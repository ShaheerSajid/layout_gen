"""
layout_gen.synth.ml.dataset — training data generator.

Generates synthetic training datasets by sampling the 6D parameter space
(w_N, w_P, l, gap_y, finger_N, finger_P) and computing analytical DRC
margins.  No KLayout or gdsfactory rendering is needed — all computation
is pure Python through :func:`~layout_gen.synth.ml.features.cell_features`
and :func:`~layout_gen.synth.ml.features.drc_margins`.

Sampling strategies
-------------------
- **Sobol** (default): quasi-random low-discrepancy sequence for uniform
  coverage of the parameter space.  Better than pseudo-random for the same
  sample budget — fills gaps instead of clumping.
- **Boundary-enriched**: extra samples concentrated near each DRC rule
  boundary (the decision surface the model must learn).  Improves sign
  accuracy where it matters most.
- **Uniform random** fallback when scipy is not installed.

Usage::

    from layout_gen import load_pdk
    from layout_gen.synth.ml.dataset import generate_dataset

    rules = load_pdk()
    ds = generate_dataset(rules, n_samples=20_000)
    print(ds.X.shape, ds.y.shape)   # (N, 23)  (N, 10)
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from layout_gen.pdk            import PDKRules
from layout_gen.cells.standard import _inter_cell_gap
from layout_gen.synth.ml.features import (
    cell_features, drc_margins, FEATURE_NAMES, MARGIN_NAMES,
)


class Dataset(NamedTuple):
    """Training dataset returned by :func:`generate_dataset`.

    Attributes
    ----------
    X :
        Feature matrix, shape ``(N, 23)``.
    y :
        Margin matrix, shape ``(N, 10)``.
    df :
        ``pandas.DataFrame`` with named columns, or ``None`` if pandas
        is not installed.
    """
    X:  np.ndarray
    y:  np.ndarray
    df: object   # pd.DataFrame | None


# ── Sampling strategies ──────────────────────────────────────────────────

def _sobol_samples(
    n_samples: int,
    bounds:    list[tuple[float, float]],
    seed:      int,
) -> np.ndarray:
    """Generate quasi-random samples using Sobol sequences.

    Falls back to uniform random if scipy.stats.qmc is unavailable.

    Returns array of shape (n_samples, len(bounds)).
    """
    ndim = len(bounds)
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=ndim, scramble=True, seed=seed)
        # Sobol requires power-of-2 samples; draw enough then truncate
        m = int(np.ceil(np.log2(max(n_samples, 2))))
        raw = sampler.random_base2(m)[:n_samples]
    except ImportError:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(0, 1, (n_samples, ndim))

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return raw * (hi - lo) + lo


def _boundary_samples(
    n_samples: int,
    rules:     PDKRules,
    seed:      int,
) -> np.ndarray:
    """Generate samples concentrated near DRC rule boundaries.

    For each of the 10 DRC rules, we know the analytical threshold.
    We sample parameters such that the margin is near zero (both sides).
    This gives the model more training signal exactly where it needs it.
    """
    rng = np.random.default_rng(seed + 1000)
    gap_min = _inter_cell_gap(rules)

    poly_wmin = rules.poly["width_min_um"]
    diff_wmin = rules.diff["width_min_um"]

    samples_per_rule = max(1, n_samples // 10)
    rows = []

    for _ in range(samples_per_rule):
        # poly.1 boundary: l ≈ poly_wmin
        l = poly_wmin + rng.normal(0, 0.02)
        w_N = rng.uniform(0.15, 2.0)
        w_P = rng.uniform(0.15, 2.0)
        rows.append([w_N, w_P, l, gap_min, 1.0, 1.0])

        # diff.1_N boundary: w_N/fingers ≈ diff_wmin
        fn = rng.integers(1, 4)
        w_N = diff_wmin * fn + rng.normal(0, 0.03) * fn
        rows.append([w_N, rng.uniform(0.15, 2.0), 0.15, gap_min, float(fn), 1.0])

        # diff.1_P boundary
        fp = rng.integers(1, 4)
        w_P = diff_wmin * fp + rng.normal(0, 0.03) * fp
        rows.append([rng.uniform(0.15, 2.0), w_P, 0.15, gap_min, 1.0, float(fp)])

        # diff.2 boundary: gap_y ≈ gap_min
        gap_y = gap_min + rng.normal(0, 0.02)
        rows.append([rng.uniform(0.3, 1.5), rng.uniform(0.3, 1.5), 0.15,
                      gap_y, 1.0, 1.0])

    return np.array(rows, dtype=np.float64)


# ── Vectorized feature/margin computation ────────────────────────────────

def _compute_batch(
    params: np.ndarray,
    rules:  PDKRules,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute features and margins for a batch of parameter vectors.

    Parameters
    ----------
    params : ndarray, shape (N, 6)
        Columns: [w_N, w_P, l, gap_y, finger_N, finger_P].
    rules :
        PDK rules.

    Returns
    -------
    X : ndarray, shape (M, 23)
    y : ndarray, shape (M, 10)
    valid_mask : ndarray, shape (N,), bool
    """
    n = len(params)
    X_rows = []
    y_rows = []
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        w_N, w_P, l, gap_y, fn, fp = params[i]
        try:
            feat = cell_features(
                float(w_N), float(w_P), float(l), rules,
                gap_y=float(gap_y),
                finger_N=float(fn),
                finger_P=float(fp),
            )
            marg = drc_margins(
                float(w_N), float(w_P), float(l), rules,
                gap_y=float(gap_y),
                finger_N=float(fn),
                finger_P=float(fp),
            )
        except Exception:
            continue
        X_rows.append(feat)
        y_rows.append(marg)
        valid[i] = True

    X = np.array(X_rows, dtype=np.float64) if X_rows else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(y_rows, dtype=np.float64) if y_rows else np.empty((0, len(MARGIN_NAMES)))
    return X, y, valid


# ── Public API ───────────────────────────────────────────────────────────

def generate_dataset(
    rules:         PDKRules,
    n_samples:     int                 = 20_000,
    w_range:       tuple[float, float] = (0.10, 3.0),
    l_range:       tuple[float, float] = (0.08, 0.50),
    gap_y_range:   tuple[float, float] = (0.0,  0.50),
    finger_range:  tuple[int,   int]   = (1,    4),
    seed:          int                 = 42,
    use_sobol:     bool                = True,
    boundary_frac: float               = 0.15,
) -> Dataset:
    """Generate a synthetic training dataset.

    Combines Sobol quasi-random sampling with boundary-enriched samples
    near DRC rule thresholds for better model accuracy where it matters.

    Parameters
    ----------
    rules :
        PDK rules.
    n_samples :
        Total target sample count (actual may differ slightly due to
        Sobol power-of-2 rounding and degenerate sample removal).
    w_range :
        ``(min, max)`` for ``w_N`` and ``w_P`` (µm).
    l_range :
        ``(min, max)`` for ``l`` (µm).
    gap_y_range :
        ``(min, max)`` for ``gap_y`` (µm).
    finger_range :
        ``(min, max)`` inclusive for ``finger_N`` and ``finger_P``.
    seed :
        Random seed for reproducibility.
    use_sobol :
        Use Sobol quasi-random sequences (requires scipy).
    boundary_frac :
        Fraction of samples to concentrate near rule boundaries.

    Returns
    -------
    Dataset
    """
    n_boundary = int(n_samples * boundary_frac) if boundary_frac > 0 else 0
    n_bulk     = n_samples - n_boundary

    # ── Bulk samples (Sobol or uniform) ──────────────────────────────────
    bounds = [
        w_range,
        w_range,
        l_range,
        gap_y_range,
        (float(finger_range[0]), float(finger_range[1])),
        (float(finger_range[0]), float(finger_range[1])),
    ]

    if use_sobol:
        bulk_params = _sobol_samples(n_bulk, bounds, seed)
    else:
        rng = np.random.default_rng(seed)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        bulk_params = rng.uniform(lo, hi, (n_bulk, 6))

    # Round finger counts to integers
    bulk_params[:, 4] = np.round(bulk_params[:, 4]).clip(finger_range[0], finger_range[1])
    bulk_params[:, 5] = np.round(bulk_params[:, 5]).clip(finger_range[0], finger_range[1])

    # ── Boundary-enriched samples ────────────────────────────────────────
    if n_boundary > 0:
        bnd_params = _boundary_samples(n_boundary, rules, seed)
        # Clip to valid ranges
        for col, (lo, hi) in enumerate(bounds):
            bnd_params[:, col] = np.clip(bnd_params[:, col], lo, hi)
        bnd_params[:, 4] = np.round(bnd_params[:, 4]).clip(finger_range[0], finger_range[1])
        bnd_params[:, 5] = np.round(bnd_params[:, 5]).clip(finger_range[0], finger_range[1])
        all_params = np.vstack([bulk_params, bnd_params])
    else:
        all_params = bulk_params

    # ── Compute features + margins ───────────────────────────────────────
    X, y, _ = _compute_batch(all_params, rules)

    # ── Build DataFrame (optional) ───────────────────────────────────────
    df = None
    try:
        import pandas as pd
        df = pd.DataFrame(
            np.hstack([X, y]),
            columns=FEATURE_NAMES + MARGIN_NAMES,
        )
    except ImportError:
        pass

    return Dataset(X=X, y=y, df=df)
