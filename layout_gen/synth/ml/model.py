"""
layout_gen.synth.ml.model — sklearn MLP margin predictor.

Predicts the 10-element DRC margin vector from a 22-element feature vector.
Wraps an sklearn ``Pipeline`` (``StandardScaler`` + ``MLPRegressor``) so
prediction is a single normalised forward pass regardless of feature scale.

Requires
--------
scikit-learn >= 1.3   (``pip install scikit-learn``)

Usage::

    from layout_gen.synth.ml.model import MarginPredictor

    mp = MarginPredictor()
    mp.fit(X_train, y_train)
    mp.save("model.pkl")

    mp2     = MarginPredictor.load("model.pkl")
    margins = mp2.predict(X_new)   # np.ndarray (N, 10)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from layout_gen.synth.ml.features import FEATURE_NAMES, MARGIN_NAMES


# ── Dependency guard ──────────────────────────────────────────────────────────

def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ML model training and inference.\n"
            "Install it with:  pip install scikit-learn\n"
            "Or install the ML extras:  pip install 'layout_gen[ml]'"
        ) from exc


# ── MarginPredictor ────────────────────────────────────────────────────────────

class MarginPredictor:
    """Predicts DRC margins from cell params + geometry features.

    The model is a ``StandardScaler → MLPRegressor`` sklearn Pipeline.
    Scaling is learned from training data so inference is scale-invariant.

    Parameters
    ----------
    hidden_layer_sizes :
        MLP hidden layer topology (default: ``(64, 32)``).
    max_iter :
        Maximum adam iterations (default: 500; ``early_stopping=True``
        typically exits well before this).
    random_state :
        Seed for the MLP weight initialisation.
    tol :
        Loss tolerance for the early-stopping criterion.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter:           int             = 500,
        random_state:       int             = 42,
        tol:                float           = 1e-4,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter           = max_iter
        self.random_state       = random_state
        self.tol                = tol
        self.pipeline_          = None
        self.feature_names_     = list(FEATURE_NAMES)
        self.margin_names_      = list(MARGIN_NAMES)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MarginPredictor":
        """Fit the StandardScaler + MLPRegressor pipeline.

        Parameters
        ----------
        X : np.ndarray, shape (N, 22)
        y : np.ndarray, shape (N, 10)

        Returns
        -------
        self
        """
        _require_sklearn()
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor

        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes = self.hidden_layer_sizes,
                activation         = "relu",
                solver             = "adam",
                max_iter           = self.max_iter,
                random_state       = self.random_state,
                tol                = self.tol,
                early_stopping     = True,
                n_iter_no_change   = 20,
                validation_fraction = 0.1,
                verbose            = False,
            )),
        ])
        self.pipeline_.fit(X, y)
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict DRC margins.

        Parameters
        ----------
        X : np.ndarray, shape (N, 22) or (22,)
            Feature vectors.

        Returns
        -------
        np.ndarray, shape (N, 10) or (10,)
            DRC margin predictions.  Positive = DRC pass; negative = fail.
        """
        if self.pipeline_ is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() or load() first."
            )
        scalar_input = (X.ndim == 1)
        if scalar_input:
            X = X[np.newaxis, :]
        out = self.pipeline_.predict(X)
        return out[0] if scalar_input else out

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the fitted pipeline to *path*.

        Parameters
        ----------
        path :
            Destination file path (created with parent directories if needed).
        """
        if self.pipeline_ is None:
            raise RuntimeError(
                "Nothing to save — model has not been fitted."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "pipeline":           self.pipeline_,
            "feature_names":      self.feature_names_,
            "margin_names":       self.margin_names_,
            "hidden_layer_sizes": self.hidden_layer_sizes,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "MarginPredictor":
        """Load a previously saved :class:`MarginPredictor` from *path*.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        _require_sklearn()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model file at {path!r}.  "
                "Train one first:  python -m layout_gen.synth.ml.train"
            )
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.pipeline_           = state["pipeline"]
        obj.feature_names_      = state["feature_names"]
        obj.margin_names_       = state["margin_names"]
        obj.hidden_layer_sizes  = state.get("hidden_layer_sizes", (64, 32))
        obj.max_iter            = 500
        obj.random_state        = 42
        obj.tol                 = 1e-4
        return obj

    def __repr__(self) -> str:
        fitted = self.pipeline_ is not None
        return (
            f"MarginPredictor(hidden_layer_sizes={self.hidden_layer_sizes}, "
            f"fitted={fitted})"
        )
