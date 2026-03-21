"""
layout_gen.synth.ml.dataset — training data generator.

Samples (w_N, w_P, l) uniformly and computes analytical DRC margins.
No KLayout or gdsfactory rendering is needed — all computation is pure
Python through :func:`~layout_gen.synth.ml.features.cell_features` and
:func:`~layout_gen.synth.ml.features.drc_margins`.

Usage::

    from layout_gen import load_pdk
    from layout_gen.synth.ml.dataset import generate_dataset

    rules = load_pdk()
    ds = generate_dataset(rules, n_samples=20_000)
    print(ds.X.shape, ds.y.shape)   # (N, 22)  (N, 10)
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from layout_gen.pdk            import PDKRules
from layout_gen.synth.ml.features import (
    cell_features, drc_margins, FEATURE_NAMES, MARGIN_NAMES,
)


class Dataset(NamedTuple):
    """Training dataset returned by :func:`generate_dataset`.

    Attributes
    ----------
    X :
        Feature matrix, shape ``(N, 22)``.  See :data:`~layout_gen.synth.ml.features.FEATURE_NAMES`.
    y :
        Margin matrix, shape ``(N, 10)``.  See :data:`~layout_gen.synth.ml.features.MARGIN_NAMES`.
    df :
        ``pandas.DataFrame`` with named columns (``X`` + ``y`` side by side),
        or ``None`` if pandas is not installed.
    """
    X:  np.ndarray
    y:  np.ndarray
    df: object   # pd.DataFrame | None


def generate_dataset(
    rules:     PDKRules,
    n_samples: int              = 20_000,
    w_range:   tuple[float, float] = (0.10, 3.0),
    l_range:   tuple[float, float] = (0.08, 0.50),
    seed:      int              = 42,
) -> Dataset:
    """Generate a synthetic training dataset by uniformly sampling params.

    The lower bounds of *w_range* (0.10) and *l_range* (0.08) are set
    **below** the sky130A PDK minimums (diff_width_min=0.15, poly_width_min=0.15)
    so the dataset contains samples with negative DRC margins.  This is
    intentional: the model must learn both sides of each rule boundary.

    Parameters
    ----------
    rules :
        PDK rules.  The same object must be used during inference.
    n_samples :
        Number of ``(w_N, w_P, l)`` triples to attempt.  Degenerate
        combinations (very small w that prevent contact placement) are
        silently skipped, so the actual dataset size may be slightly smaller.
    w_range :
        ``(min, max)`` for uniform sampling of ``w_N`` and ``w_P`` (µm).
    l_range :
        ``(min, max)`` for uniform sampling of ``l`` (µm).
    seed :
        NumPy random seed for reproducibility.

    Returns
    -------
    Dataset
        Named tuple of ``(X, y, df)``.
    """
    rng = np.random.default_rng(seed)

    w_N_arr = rng.uniform(w_range[0], w_range[1], n_samples)
    w_P_arr = rng.uniform(w_range[0], w_range[1], n_samples)
    l_arr   = rng.uniform(l_range[0], l_range[1], n_samples)

    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for w_N, w_P, l in zip(w_N_arr, w_P_arr, l_arr):
        try:
            feat = cell_features(w_N, w_P, l, rules)
            marg = drc_margins(w_N, w_P, l, rules)
        except Exception:
            # Extreme combinations (w much smaller than contact size) can raise
            # inside transistor_geom.  Skip silently.
            continue
        X_rows.append(feat)
        y_rows.append(marg)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.float64)

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
