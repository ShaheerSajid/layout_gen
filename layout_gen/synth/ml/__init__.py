"""
layout_gen.synth.ml — ML-guided DRC margin prediction and parameter optimisation.

Sub-modules
-----------
features    Analytical DRC margin computation (pure Python + numpy).
dataset     Synthetic training dataset generator.
model       sklearn MLP margin predictor (``MarginPredictor``).
agent       scipy-based parameter optimiser (``MLAgent``).
fix_policy  DRC violation encoder + fix-delta predictor (``DRCFixPredictor``).
train       CLI training script (``python -m layout_gen.synth.ml.train``).

Quick start::

    # 1. Train margin predictor
    python -m layout_gen.synth.ml.train --samples 20000 --out model.pkl

    # 2. Use in synthesis
    from layout_gen import load_pdk, load_template, Synthesizer
    from layout_gen.synth.ml import MLAgent

    rules  = load_pdk()
    agent  = MLAgent.load("model.pkl")
    result = Synthesizer(rules, ml_model=agent).synthesize(
        load_template("inverter"),
        params={"w_N": 0.52, "w_P": 0.42, "l": 0.15},
    )

    # 3. Train DRC fix predictor
    from layout_gen.synth.ml import DRCFixPredictor, generate_fix_dataset

    dataset = generate_fix_dataset(rules, n_samples=10_000)
    pred    = DRCFixPredictor()
    pred.fit(dataset.X, dataset.y)
"""
from layout_gen.synth.ml.agent      import MLAgent, ModelNotTrainedError
from layout_gen.synth.ml.model      import MarginPredictor
from layout_gen.synth.ml.fix_policy import (
    DRCFixPredictor,
    ViolationEncoder,
    generate_fix_dataset,
    FixDataset,
)

__all__ = [
    "MLAgent",
    "ModelNotTrainedError",
    "MarginPredictor",
    "DRCFixPredictor",
    "ViolationEncoder",
    "generate_fix_dataset",
    "FixDataset",
]
