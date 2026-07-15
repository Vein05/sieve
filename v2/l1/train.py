"""Model training for L1 sufficiency estimator.

Three models: stem-overlap baseline (threshold), logistic regression, GBM.
Training is offline, deterministic, no network access.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v2.l1.features import FeatureVector, feature_names, to_array
from v2.l1.labels import LabeledExample

# Stem-overlap threshold that mirrors the incumbent guard.
# The guarded_structured packager uses COVERAGE_MIN_FRACTION = 0.5 on per-unit
# overlap; here we use the same threshold on the pack-level stem_coverage feature.
BASELINE_THRESHOLD = 0.5

# GBM regularization: keep small to avoid overfit on ~1-4K rows.
_GBM_MAX_ITER = 200
_GBM_MAX_LEAF_NODES = 15
_GBM_L2_REG = 1.0
_GBM_LEARNING_RATE = 0.05

# LR regularization.
_LR_C = 0.1  # inverse regularization strength (smaller = stronger)

_FEATURE_STEM_COVERAGE_IDX = 0  # position of stem_coverage in FeatureVector


@dataclass
class TrainedModels:
    """Container for all three fitted models."""

    baseline_threshold: float
    lr_model: Any          # sklearn LogisticRegression
    gbm_model: Any         # sklearn HistGradientBoostingClassifier
    feature_names: list[str]


def _to_matrix(examples: list[LabeledExample]) -> tuple[np.ndarray, np.ndarray]:
    """Convert labeled examples to (X, y) numpy arrays."""
    X = np.array([to_array(ex_to_fv(e)) for e in examples], dtype=np.float64)
    y = np.array([e.label for e in examples], dtype=np.float64)
    return X, y


def ex_to_fv(example: LabeledExample) -> FeatureVector:
    """Extract features from a LabeledExample."""
    from v2.l1.features import extract

    return extract(example.query, example.pack_texts)


def train(train_examples: list[LabeledExample]) -> TrainedModels:
    """Fit all three models on train_examples."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X, y = _to_matrix(train_examples)

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=_LR_C, max_iter=1000, random_state=42)),
    ])
    lr_model.fit(X, y)

    gbm_model = HistGradientBoostingClassifier(
        max_iter=_GBM_MAX_ITER,
        max_leaf_nodes=_GBM_MAX_LEAF_NODES,
        l2_regularization=_GBM_L2_REG,
        learning_rate=_GBM_LEARNING_RATE,
        random_state=42,
    )
    gbm_model.fit(X, y)

    return TrainedModels(
        baseline_threshold=BASELINE_THRESHOLD,
        lr_model=lr_model,
        gbm_model=gbm_model,
        feature_names=feature_names(),
    )


def predict_proba(models: TrainedModels, examples: list[LabeledExample]) -> dict[str, np.ndarray]:
    """Return predicted probabilities for each model on the given examples.

    Returns dict with keys 'baseline', 'lr', 'gbm' each mapping to a 1-D array.
    """
    X, _ = _to_matrix(examples)
    stem_scores = X[:, _FEATURE_STEM_COVERAGE_IDX]
    baseline_proba = (stem_scores >= models.baseline_threshold).astype(np.float64)

    lr_proba = models.lr_model.predict_proba(X)[:, 1]
    gbm_proba = models.gbm_model.predict_proba(X)[:, 1]

    return {"baseline": baseline_proba, "lr": lr_proba, "gbm": gbm_proba}


def save_models(models: TrainedModels, output_path: Path) -> None:
    """Persist TrainedModels to disk via pickle."""
    with open(output_path, "wb") as fh:
        pickle.dump(models, fh)


def load_models(model_path: Path) -> TrainedModels:
    """Load TrainedModels from disk."""
    with open(model_path, "rb") as fh:
        return pickle.load(fh)
