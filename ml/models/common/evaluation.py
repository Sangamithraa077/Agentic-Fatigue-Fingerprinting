"""Shared model evaluation: identical for classical and deep learning
models, since both are wrapped to expose the same `.predict(X)` /
`.predict_proba(X)` interface (see `deep_learning/inference.py`'s
`TorchClassifierAdapter` for how DL models conform to this).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from ml.models.common.config import DatasetConfig
from ml.models.common.metrics import ClassificationMetrics, TimingMetrics, compute_classification_metrics, measure_time_and_memory

logger = logging.getLogger(__name__)


class PredictorProtocol(Protocol):
    """The minimal interface `evaluate_model` requires of any model."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation output for one (model, dataset) pair.

    Attributes:
        model_name: Registry name of the evaluated model.
        dataset_config: The `DatasetConfig` used to build the evaluated data.
        fold_index: Cross-validation fold index, or `None` for a single
            held-out test-set evaluation.
        metrics: Standard classification metrics.
        timing: Training/inference timing (training_seconds is 0 here --
            populated by the caller after combining with fit timing; see
            `evaluate_model`'s `training_seconds` argument).
        y_true, y_pred: True/predicted labels for this evaluation.
        y_proba: Predicted class probabilities, if the model supports it.
    """

    model_name: str
    dataset_config: DatasetConfig
    fold_index: Optional[int]
    metrics: ClassificationMetrics
    timing: TimingMetrics
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]


def evaluate_model(
    model: PredictorProtocol,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    dataset_config: DatasetConfig,
    training_seconds: float = 0.0,
    fold_index: Optional[int] = None,
) -> EvaluationResult:
    """Evaluate a fitted model on held-out data, computing metrics and timing.

    Args:
        model: Any object exposing `.predict(X)` and `.predict_proba(X)`
            (both classical sklearn-style estimators and
            `TorchClassifierAdapter`-wrapped deep learning models qualify).
        X_test, y_test: Held-out evaluation data.
        model_name: For labeling the result (e.g. "random_forest").
        dataset_config: The `DatasetConfig` that produced `X_test`/`y_test`.
        training_seconds: Wall-clock training time, passed through from the
            caller (measured at fit time, not here).
        fold_index: Cross-validation fold index, if applicable.

    Returns:
        A fully populated `EvaluationResult`.
    """
    with measure_time_and_memory() as timing_info:
        y_pred = model.predict(X_test)
        y_proba: Optional[np.ndarray] = None
        try:
            y_proba = model.predict_proba(X_test)
        except (AttributeError, NotImplementedError) as exc:
            logger.warning("Model '%s' does not support predict_proba: %s", model_name, exc)

    n_samples = len(X_test)
    timing = TimingMetrics(
        training_seconds=training_seconds,
        inference_seconds_total=timing_info["seconds"],
        inference_seconds_per_sample=timing_info["seconds"] / n_samples if n_samples else float("nan"),
        peak_memory_mb=timing_info["peak_memory_mb"],
    )

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    return EvaluationResult(
        model_name=model_name,
        dataset_config=dataset_config,
        fold_index=fold_index,
        metrics=metrics,
        timing=timing,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )
