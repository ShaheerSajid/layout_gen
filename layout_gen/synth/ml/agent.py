"""
layout_gen.synth.ml.agent — ML-guided parameter optimiser.

Wraps a trained :class:`~layout_gen.synth.ml.model.MarginPredictor` and exposes
it as an ``MLModel`` callable compatible with
:class:`~layout_gen.synth.synthesizer.Synthesizer`.

The optimiser finds the ``(w_N, w_P, l, gap_y, finger_N, finger_P)`` vector
that maximises the minimum predicted DRC margin across all 10 rules (i.e. the
most room from any violation).  scipy Nelder-Mead is used when available; a
simple coordinate-descent fallback is used otherwise.

``finger_N`` and ``finger_P`` use continuous relaxation during optimisation and
are rounded to the nearest integer in the returned parameter dict.

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


# ── Exceptions ────────────────────────────────────────────────────────────────

class ModelNotTrainedError(RuntimeError):
    """Raised when :class:`MLAgent` is asked to optimise without a model."""


# ── Bounds ────────────────────────────────────────────────────────────────────

# Physically reasonable search bounds for sky130 (and similar nodes).
# Widths and lengths in µm; finger counts are continuous relaxations of integers.
_W_MIN,      _W_MAX      = 0.15, 5.0   # per-finger total channel width
_L_MIN,      _L_MAX      = 0.15, 2.0   # gate length
_GAP_MIN,    _GAP_MAX    = 0.00, 0.50  # Y gap between NMOS/PMOS rows (µm)
_FINGER_MIN, _FINGER_MAX = 1.0,  4.0   # gate finger count (rounded to int at output)


# ── MLAgent ───────────────────────────────────────────────────────────────────

class MLAgent:
    """ML-guided cell parameter optimiser.

    Parameters
    ----------
    model :
        A fitted :class:`~layout_gen.synth.ml.model.MarginPredictor`.
    step :
        Coordinate-descent step size in µm (used only when scipy is absent).
    """

    def __init__(
        self,
        model: MarginPredictor,
        step:  float = 0.01,
    ):
        self.model = model
        self.step  = step

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "MLAgent":
        """Load a saved :class:`MarginPredictor` and wrap it in an MLAgent.

        Parameters
        ----------
        path :
            Path to the ``.pkl`` file produced by
            :meth:`~layout_gen.synth.ml.model.MarginPredictor.save`.
        **kwargs :
            Forwarded to :class:`MLAgent.__init__` (e.g. ``step``).
        """
        model = MarginPredictor.load(path)
        return cls(model, **kwargs)

    # ── Objective ─────────────────────────────────────────────────────────────

    def _objective(
        self,
        x:     np.ndarray,
        rules: PDKRules,
    ) -> float:
        """Negative minimum DRC margin (to be minimised by the optimiser)."""
        w_N, w_P, l      = float(x[0]), float(x[1]), float(x[2])
        gap_y, fn, fp     = float(x[3]), float(x[4]), float(x[5])
        try:
            feat = cell_features(
                w_N, w_P, l, rules,
                gap_y=gap_y, finger_N=fn, finger_P=fp,
            )
        except Exception:
            return 1e6   # degenerate geometry → high penalty
        margins = self.model.predict(feat)
        return -float(np.min(margins))

    # ── Optimisers ────────────────────────────────────────────────────────────

    def _optimise_scipy(
        self,
        x0:    np.ndarray,
        rules: PDKRules,
    ) -> np.ndarray:
        """Nelder-Mead optimisation via scipy."""
        from scipy.optimize import minimize, Bounds

        bounds = Bounds(
            lb=[_W_MIN, _W_MIN, _L_MIN, _GAP_MIN, _FINGER_MIN, _FINGER_MIN],
            ub=[_W_MAX, _W_MAX, _L_MAX, _GAP_MAX, _FINGER_MAX, _FINGER_MAX],
        )
        res = minimize(
            self._objective,
            x0,
            args=(rules,),
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-6},
        )
        return res.x

    def _optimise_coord(
        self,
        x0:    np.ndarray,
        rules: PDKRules,
    ) -> np.ndarray:
        """Simple coordinate-descent fallback (no scipy required)."""
        x = x0.copy()
        f = self._objective(x, rules)
        lo = np.array([_W_MIN, _W_MIN, _L_MIN, _GAP_MIN, _FINGER_MIN, _FINGER_MIN])
        hi = np.array([_W_MAX, _W_MAX, _L_MAX, _GAP_MAX, _FINGER_MAX, _FINGER_MAX])

        improved = True
        while improved:
            improved = False
            for i in range(len(x)):
                for delta in (self.step, -self.step):
                    x_try    = x.copy()
                    x_try[i] = np.clip(x_try[i] + delta, lo[i], hi[i])
                    f_try    = self._objective(x_try, rules)
                    if f_try < f - 1e-8:
                        x, f     = x_try, f_try
                        improved = True
        return x

    # ── MLModel callable ──────────────────────────────────────────────────────

    def __call__(
        self,
        template: Any,           # CellTemplate (not imported to avoid circulars)
        rules:    PDKRules,
        violations: list,        # list[DRCViolation]
        params:   dict,
    ) -> dict:
        """Suggest improved cell parameters.

        This method matches the ``MLModel`` protocol expected by
        :class:`~layout_gen.synth.synthesizer.Synthesizer`.

        Parameters
        ----------
        template :
            Current cell template (unused; kept for protocol compatibility).
        rules :
            PDK rules used for feature extraction.
        violations :
            Current DRC violations (unused; ML objective drives optimisation).
        params :
            Current parameter dict.  Recognised keys: ``"w_N"``, ``"w_P"``,
            ``"l"``, ``"gap_y"``, ``"finger_N"``, ``"finger_P"``.

        Returns
        -------
        dict
            Updated parameter dict with keys ``"w_N"``, ``"w_P"``, ``"l"``,
            ``"gap_y"``, ``"finger_N"`` (int), ``"finger_P"`` (int).
        """
        if self.model.pipeline_ is None:
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

        try:
            import scipy  # noqa: F401
            x_opt = self._optimise_scipy(x0, rules)
        except ImportError:
            x_opt = self._optimise_coord(x0, rules)

        w_N_opt  = float(np.clip(x_opt[0], _W_MIN,      _W_MAX))
        w_P_opt  = float(np.clip(x_opt[1], _W_MIN,      _W_MAX))
        l_opt    = float(np.clip(x_opt[2], _L_MIN,      _L_MAX))
        gap_opt  = float(np.clip(x_opt[3], _GAP_MIN,    _GAP_MAX))
        fn_opt   = int(round(np.clip(x_opt[4], _FINGER_MIN, _FINGER_MAX)))
        fp_opt   = int(round(np.clip(x_opt[5], _FINGER_MIN, _FINGER_MAX)))

        # Round continuous params to 2 decimal places — keeps GDS coordinates clean.
        return {
            "w_N":      round(w_N_opt, 2),
            "w_P":      round(w_P_opt, 2),
            "l":        round(l_opt,   2),
            "gap_y":    round(gap_opt, 3),
            "finger_N": fn_opt,
            "finger_P": fp_opt,
        }

    def __repr__(self) -> str:
        return (
            f"MLAgent(model={self.model!r}, step={self.step})"
        )
