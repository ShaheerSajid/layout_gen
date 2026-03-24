"""
layout_gen.synth.ml.agent — ML-guided parameter optimiser.

Wraps a trained :class:`~layout_gen.synth.ml.model.MarginPredictor` and
optimises cell parameters to maximise DRC margin while minimising area.

Optimisation strategy
---------------------
1. **Multi-start L-BFGS-B** — gradient-free bounds are natively respected
   (unlike Nelder-Mead which only clips post-hoc).  Multiple initial points
   avoid local-minima traps.

2. **Lower Confidence Bound (LCB)** acquisition — when the model provides
   uncertainty estimates (ensemble disagreement), the objective becomes::

       LCB(x) = mean_min_margin(x) - kappa * std_min_margin(x)

   This balances exploitation (large predicted margin) with exploration
   (high uncertainty → worth investigating).

3. **Area-aware objective** — a penalty term discourages unnecessarily large
   cells.  The combined objective is::

       obj(x) = -min_margin(x) + alpha * area(x) / area_ref

   where ``alpha`` controls the area penalty weight (default 0.1).

Usage::

    from layout_gen import load_pdk, load_template, Synthesizer
    from layout_gen.synth.ml import MLAgent

    rules  = load_pdk()
    agent  = MLAgent.load("model.pkl")
    result = Synthesizer(rules, ml_model=agent).synthesize(
        load_template("inverter"),
        params={"w_N": 0.52, "w_P": 0.42, "l": 0.15},
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from layout_gen.pdk              import PDKRules
from layout_gen.synth.ml.features import cell_features
from layout_gen.synth.ml.model    import MarginPredictor
from layout_gen.transistor        import transistor_geom
from layout_gen.cells.standard    import _inter_cell_gap


# ── Exceptions ────────────────────────────────────────────────────────────

class ModelNotTrainedError(RuntimeError):
    """Raised when :class:`MLAgent` is asked to optimise without a model."""


# ── Bounds ────────────────────────────────────────────────────────────────

_W_MAX      = 5.0   # total channel width (µm)
_L_MAX      = 2.0   # gate length (µm)
_GAP_MIN,    _GAP_MAX    = 0.00, 0.50  # Y gap between NMOS/PMOS rows
_FINGER_MIN, _FINGER_MAX = 1.0,  4.0   # finger count (continuous relaxation)


def _get_bounds(rules: PDKRules) -> tuple[list, np.ndarray, np.ndarray]:
    """Return (bounds_list, lo_array, hi_array) derived from PDK rules."""
    w_min = rules.diff["width_min_um"]
    l_min = rules.poly["width_min_um"]
    bounds = [
        (w_min,       _W_MAX),
        (w_min,       _W_MAX),
        (l_min,       _L_MAX),
        (_GAP_MIN,    _GAP_MAX),
        (_FINGER_MIN, _FINGER_MAX),
        (_FINGER_MIN, _FINGER_MAX),
    ]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return bounds, lo, hi


# ── Area computation ─────────────────────────────────────────────────────

def _cell_area(x: np.ndarray, rules: PDKRules) -> float:
    """Estimate cell area (µm²) from parameter vector."""
    w_N, w_P, l, gap_y, fn, fp = (
        float(x[0]), float(x[1]), float(x[2]),
        float(x[3]), float(x[4]), float(x[5]),
    )
    try:
        from layout_gen.synth.ml.features import _geom_override_fingers
        ng = transistor_geom(w_N, l, "nmos", rules)
        pg = transistor_geom(w_P, l, "pmos", rules)
        ng = _geom_override_fingers(ng, max(1, int(round(fn))), rules)
        pg = _geom_override_fingers(pg, max(1, int(round(fp))), rules)
        width  = max(ng.total_x_um, pg.total_x_um)
        height = ng.total_y_um + gap_y + pg.total_y_um
        return width * height
    except Exception:
        return 1e6


# ── MLAgent ───────────────────────────────────────────────────────────────

class MLAgent:
    """ML-guided cell parameter optimiser.

    Parameters
    ----------
    model :
        A fitted :class:`~layout_gen.synth.ml.model.MarginPredictor`.
    n_restarts :
        Number of random restarts for multi-start optimization.
    alpha :
        Area penalty weight in the objective (0 = ignore area).
    kappa :
        Exploration weight for LCB acquisition (0 = pure exploitation).
    step :
        Coordinate-descent step size (fallback when scipy is absent).
    """

    def __init__(
        self,
        model:      MarginPredictor,
        n_restarts: int   = 5,
        alpha:      float = 0.1,
        kappa:      float = 1.0,
        step:       float = 0.01,
    ):
        self.model      = model
        self.n_restarts = n_restarts
        self.alpha      = alpha
        self.kappa      = kappa
        self.step       = step

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "MLAgent":
        """Load a saved model and wrap it in an MLAgent."""
        model = MarginPredictor.load(path)
        return cls(model, **kwargs)

    # ── Objective ─────────────────────────────────────────────────────────

    def _objective(
        self,
        x:     np.ndarray,
        rules: PDKRules,
        area_ref: float,
    ) -> float:
        """Combined objective: -min_margin + area_penalty + uncertainty_penalty.

        Lower is better (minimisation target).
        """
        w_N, w_P, l = float(x[0]), float(x[1]), float(x[2])
        gap_y, fn, fp = float(x[3]), float(x[4]), float(x[5])
        try:
            feat = cell_features(
                w_N, w_P, l, rules,
                gap_y=gap_y, finger_N=fn, finger_P=fp,
            )
        except Exception:
            return 1e6

        result = self.model.predict(feat, return_std=True)
        if isinstance(result, tuple):
            margins, std = result
        else:
            margins, std = result, np.zeros_like(result)

        min_margin = float(np.min(margins))

        # LCB: penalise uncertainty (explore regions where model is unsure)
        min_std = float(np.max(std))  # worst-case uncertainty
        lcb = min_margin - self.kappa * min_std

        # Area penalty: normalised by reference area
        area = _cell_area(x, rules)
        area_term = self.alpha * (area / area_ref) if area_ref > 0 else 0

        return -lcb + area_term

    # ── Optimisers ────────────────────────────────────────────────────────

    def _optimise_scipy(
        self,
        x0:       np.ndarray,
        rules:    PDKRules,
        area_ref: float,
    ) -> tuple[np.ndarray, float]:
        """Multi-start L-BFGS-B optimisation."""
        from scipy.optimize import minimize

        bounds, lo, hi = _get_bounds(rules)
        best_x = x0.copy()
        best_f = self._objective(x0, rules, area_ref)

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = [x0]
        for _ in range(self.n_restarts - 1):
            x_rand = rng.uniform(lo, hi)
            # Round finger counts
            x_rand[4] = round(x_rand[4])
            x_rand[5] = round(x_rand[5])
            starts.append(x_rand)

        for x_start in starts:
            try:
                res = minimize(
                    self._objective,
                    x_start,
                    args=(rules, area_ref),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 200, "ftol": 1e-8},
                )
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                continue

        return best_x, best_f

    def _optimise_coord(
        self,
        x0:       np.ndarray,
        rules:    PDKRules,
        area_ref: float,
    ) -> tuple[np.ndarray, float]:
        """Coordinate-descent fallback (no scipy required)."""
        _, lo, hi = _get_bounds(rules)
        x = x0.copy()
        f = self._objective(x, rules, area_ref)

        improved = True
        while improved:
            improved = False
            for i in range(len(x)):
                for delta in (self.step, -self.step):
                    x_try    = x.copy()
                    x_try[i] = np.clip(x_try[i] + delta, lo[i], hi[i])
                    f_try    = self._objective(x_try, rules, area_ref)
                    if f_try < f - 1e-8:
                        x, f     = x_try, f_try
                        improved = True
        return x, f

    # ── MLModel callable ──────────────────────────────────────────────────

    def __call__(
        self,
        template: Any,
        rules:    PDKRules,
        violations: list,
        params:   dict,
    ) -> dict:
        """Suggest improved cell parameters.

        Matches the ``MLModel`` protocol expected by
        :class:`~layout_gen.synth.synthesizer.Synthesizer`.
        """
        if not self.model._models:
            raise ModelNotTrainedError(
                "MLAgent has no fitted model.  "
                "Train one with: python -m layout_gen.synth.ml.train"
            )

        w_N      = float(params.get("w_N",      0.52))
        w_P      = float(params.get("w_P",      0.42))
        l        = float(params.get("l",         0.15))
        gap_y    = float(params.get("gap_y",    _GAP_MIN))
        finger_N = float(params.get("finger_N", 1.0))
        finger_P = float(params.get("finger_P", 1.0))

        x0 = np.array([w_N, w_P, l, gap_y, finger_N, finger_P])

        # Reference area for normalisation (area at initial params)
        area_ref = max(_cell_area(x0, rules), 0.01)

        try:
            import scipy  # noqa: F401
            x_opt, _ = self._optimise_scipy(x0, rules, area_ref)
        except ImportError:
            x_opt, _ = self._optimise_coord(x0, rules, area_ref)

        return {
            "w_N":      round(float(np.clip(x_opt[0], _W_MIN,      _W_MAX)), 2),
            "w_P":      round(float(np.clip(x_opt[1], _W_MIN,      _W_MAX)), 2),
            "l":        round(float(np.clip(x_opt[2], _L_MIN,      _L_MAX)), 2),
            "gap_y":    round(float(np.clip(x_opt[3], _GAP_MIN,    _GAP_MAX)), 3),
            "finger_N": int(round(np.clip(x_opt[4], _FINGER_MIN, _FINGER_MAX))),
            "finger_P": int(round(np.clip(x_opt[5], _FINGER_MIN, _FINGER_MAX))),
        }

    # ── Standalone optimization (for scripts like run_ml_bitcell) ────────

    def optimize(self, rules: PDKRules, x0: np.ndarray | None = None) -> dict:
        """Run standalone optimization without the Synthesizer loop.

        Parameters
        ----------
        rules :
            PDK rules.
        x0 :
            Initial parameter vector [w_N, w_P, l, gap_y, fn, fp].
            Defaults to ``[0.52, 0.42, 0.15, gap_min, 1, 1]``.

        Returns
        -------
        dict
            Optimised parameters.
        """
        if x0 is None:
            gap_min = _inter_cell_gap(rules)
            x0 = np.array([0.52, 0.42, 0.15, gap_min, 1.0, 1.0])

        area_ref = max(_cell_area(x0, rules), 0.01)

        try:
            import scipy  # noqa: F401
            x_opt, _ = self._optimise_scipy(x0, rules, area_ref)
        except ImportError:
            x_opt, _ = self._optimise_coord(x0, rules, area_ref)

        return {
            "w_N":      round(float(np.clip(x_opt[0], _W_MIN,      _W_MAX)), 2),
            "w_P":      round(float(np.clip(x_opt[1], _W_MIN,      _W_MAX)), 2),
            "l":        round(float(np.clip(x_opt[2], _L_MIN,      _L_MAX)), 2),
            "gap_y":    round(float(np.clip(x_opt[3], _GAP_MIN,    _GAP_MAX)), 3),
            "finger_N": int(round(np.clip(x_opt[4], _FINGER_MIN, _FINGER_MAX))),
            "finger_P": int(round(np.clip(x_opt[5], _FINGER_MIN, _FINGER_MAX))),
        }

    def __repr__(self) -> str:
        return (
            f"MLAgent(model={self.model!r}, n_restarts={self.n_restarts}, "
            f"alpha={self.alpha}, kappa={self.kappa})"
        )
