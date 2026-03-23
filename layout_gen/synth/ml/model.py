"""
layout_gen.synth.ml.model — ensemble DRC margin predictor with uncertainty.

Predicts the 10-element DRC margin vector from a 23-element feature vector
using an ensemble of heterogeneous regressors (MLP + Gradient Boosting).
Ensemble disagreement provides calibrated uncertainty estimates — critical
for safe parameter exploration in the optimizer.

Architecture
------------
The ``MarginPredictor`` trains *N* diverse sub-models:

1. **MLPRegressor** — neural network, good at smooth non-linear boundaries.
2. **GradientBoostingRegressor** — tree ensemble, excellent on tabular data
   with heterogeneous feature scales, naturally captures rule-like thresholds.

Final prediction = mean of sub-model predictions.
Uncertainty       = std-dev across sub-models (per-margin).

When ``predict_with_uncertainty=True``, the caller receives both the mean
prediction and a standard-deviation vector.  The optimizer uses this to
implement *Lower Confidence Bound* (LCB) acquisition — preferring parameter
regions where the model is both confident and predicts large margins.

Requires
--------
scikit-learn >= 1.3   (``pip install scikit-learn``)

Usage::

    from layout_gen.synth.ml.model import MarginPredictor

    mp = MarginPredictor()
    mp.fit(X_train, y_train)
    mp.save("model.pkl")

    mp2 = MarginPredictor.load("model.pkl")

    # Point prediction
    margins = mp2.predict(X_new)              # np.ndarray (N, 10)

    # With uncertainty
    mean, std = mp2.predict(X_new, return_std=True)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from layout_gen.synth.ml.features import FEATURE_NAMES, MARGIN_NAMES


# ── Dependency guard ──────────────────────────────────────────────────────

def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ML model training and inference.\n"
            "Install it with:  pip install scikit-learn\n"
            "Or install the ML extras:  pip install 'layout_gen[ml]'"
        ) from exc


# ── MarginPredictor ────────────────────────────────────────────────────────

class MarginPredictor:
    """Ensemble DRC margin predictor with uncertainty quantification.

    Trains both an MLP and a Gradient Boosting regressor on the same data.
    Prediction = ensemble mean; uncertainty = ensemble standard deviation.

    Parameters
    ----------
    hidden_layer_sizes :
        MLP hidden layer topology (default: ``(128, 64, 32)``).
    n_estimators :
        Number of boosting trees per output (default: 200).
    max_iter :
        Maximum adam iterations for MLP (default: 600).
    random_state :
        Seed for all sub-model weight initialisation.
    tol :
        Loss tolerance for MLP early-stopping.
    use_gbr :
        Whether to include GradientBoostingRegressor in the ensemble.
        Set to ``False`` for faster training at the cost of no uncertainty.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
        n_estimators:       int             = 200,
        max_iter:           int             = 600,
        random_state:       int             = 42,
        tol:                float           = 1e-4,
        use_gbr:            bool            = True,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_estimators       = n_estimators
        self.max_iter           = max_iter
        self.random_state       = random_state
        self.tol                = tol
        self.use_gbr            = use_gbr
        self._models: list      = []   # fitted (name, pipeline) pairs
        self.feature_names_     = list(FEATURE_NAMES)
        self.margin_names_      = list(MARGIN_NAMES)
        self._n_outputs         = len(MARGIN_NAMES)

    @property
    def pipeline_(self):
        """Backward-compatible: returns the first fitted pipeline or None."""
        if not self._models:
            return None
        return self._models[0][1]

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MarginPredictor":
        """Fit the ensemble on training data.

        Parameters
        ----------
        X : np.ndarray, shape (N, 23)
        y : np.ndarray, shape (N, 10)

        Returns
        -------
        self
        """
        _require_sklearn()
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        from sklearn.multioutput import MultiOutputRegressor

        self._models = []
        self._n_outputs = y.shape[1] if y.ndim > 1 else 1

        # ── Sub-model 1: MLP ──────────────────────────────────────────────
        mlp_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes = self.hidden_layer_sizes,
                activation         = "relu",
                solver             = "adam",
                max_iter           = self.max_iter,
                random_state       = self.random_state,
                tol                = self.tol,
                early_stopping     = True,
                n_iter_no_change   = 25,
                validation_fraction = 0.1,
                verbose            = False,
            )),
        ])
        mlp_pipe.fit(X, y)
        self._models.append(("mlp", mlp_pipe))

        # ── Sub-model 2: Gradient Boosting (per-output) ───────────────────
        if self.use_gbr:
            try:
                from sklearn.ensemble import GradientBoostingRegressor

                gbr_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("gbr", MultiOutputRegressor(
                        GradientBoostingRegressor(
                            n_estimators    = self.n_estimators,
                            max_depth       = 5,
                            learning_rate   = 0.1,
                            subsample       = 0.8,
                            random_state    = self.random_state,
                            validation_fraction = 0.1,
                            n_iter_no_change    = 15,
                        ),
                    )),
                ])
                gbr_pipe.fit(X, y)
                self._models.append(("gbr", gbr_pipe))
            except Exception:
                pass  # GBR failed — continue with MLP only

        return self

    # ── Predict ───────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict DRC margins, optionally with uncertainty.

        Parameters
        ----------
        X : np.ndarray, shape (N, 23) or (23,)
            Feature vectors.
        return_std : bool
            If True, return ``(mean, std)`` tuple.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            DRC margin predictions.  Positive = DRC pass; negative = fail.
            When ``return_std=True``, returns ``(mean, std)`` where std is
            the per-margin ensemble standard deviation.
        """
        if not self._models:
            raise RuntimeError(
                "Model is not fitted. Call fit() or load() first."
            )
        scalar_input = (X.ndim == 1)
        if scalar_input:
            X = X[np.newaxis, :]

        preds = np.stack([pipe.predict(X) for _, pipe in self._models], axis=0)
        mean = np.mean(preds, axis=0)
        std  = np.std(preds, axis=0) if len(self._models) > 1 else np.zeros_like(mean)

        if scalar_input:
            mean = mean[0]
            std  = std[0]

        return (mean, std) if return_std else mean

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the fitted ensemble to *path*."""
        if not self._models:
            raise RuntimeError("Nothing to save — model has not been fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "models":             self._models,
            "feature_names":      self.feature_names_,
            "margin_names":       self.margin_names_,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "n_estimators":       self.n_estimators,
            "use_gbr":            self.use_gbr,
            "n_outputs":          self._n_outputs,
            "version":            2,  # ensemble format
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "MarginPredictor":
        """Load a previously saved :class:`MarginPredictor` from *path*."""
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
        obj.feature_names_      = state["feature_names"]
        obj.margin_names_       = state["margin_names"]
        obj.hidden_layer_sizes  = state.get("hidden_layer_sizes", (128, 64, 32))
        obj.n_estimators        = state.get("n_estimators", 200)
        obj.max_iter            = 600
        obj.random_state        = 42
        obj.tol                 = 1e-4
        obj.use_gbr             = state.get("use_gbr", True)
        obj._n_outputs          = state.get("n_outputs", len(MARGIN_NAMES))

        # Handle both v1 (single pipeline) and v2 (ensemble) formats
        if "models" in state:
            obj._models = state["models"]
        elif "pipeline" in state:
            # v1 backward compatibility
            obj._models = [("mlp", state["pipeline"])]
        else:
            obj._models = []

        return obj

    def __repr__(self) -> str:
        model_names = [name for name, _ in self._models] if self._models else []
        fitted = bool(self._models)
        return (
            f"MarginPredictor(ensemble={model_names}, "
            f"fitted={fitted})"
        )
