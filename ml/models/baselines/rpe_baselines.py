"""Module 5: Continuous Borg RPE Baselines & Evaluation Models.

Implements:
- MeanPredictorBaseline: Always predicts the training set mean RPE.
- ElapsedTimeBaseline: Predicts RPE monotonically from normalized elapsed time in a session.
- Single-Sensor and Fused Ridge / Random Forest Regressors for Borg RPE (6-20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from ml.models.common.metrics import RegressionMetrics, compute_regression_metrics


class MeanPredictorBaseline(BaseEstimator, RegressorMixin):
    """Dummy baseline: always predicts the mean of the training target."""

    def __init__(self):
        self.mean_value_: float = 6.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanPredictorBaseline":
        self.mean_value_ = float(np.mean(y)) if len(y) > 0 else 6.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full((n_samples,), self.mean_value_, dtype=float)


class ElapsedTimeBaseline(BaseEstimator, RegressorMixin):
    """Dummy baseline: predicts RPE monotonically based solely on normalized elapsed time.

    Assumes fatigue builds progressively throughout a workout trial from baseline (RPE ~6.0)
    to near-exhaustion (RPE ~17.0).
    """

    def __init__(self, start_rpe: float = 6.0, end_rpe: float = 17.0):
        self.start_rpe = start_rpe
        self.end_rpe = end_rpe
        self.fitted_slope_: float = 0.0
        self.fitted_intercept_: float = 6.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElapsedTimeBaseline":
        # If time is not explicitly in X, we fit a linear ramp over window sequence indices
        n_samples = len(y)
        if n_samples > 1:
            t_norm = np.linspace(0.0, 1.0, n_samples)
            # Fit least-squares slope and intercept
            A = np.vstack([t_norm, np.ones(n_samples)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            self.fitted_slope_ = float(slope)
            self.fitted_intercept_ = float(intercept)
        else:
            self.fitted_slope_ = self.end_rpe - self.start_rpe
            self.fitted_intercept_ = self.start_rpe
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        if n_samples <= 1:
            return np.array([self.fitted_intercept_], dtype=float)
        t_norm = np.linspace(0.0, 1.0, n_samples)
        pred = self.fitted_slope_ * t_norm + self.fitted_intercept_
        return np.clip(pred, 6.0, 20.0)


@dataclass
class BaselineEvaluationResult:
    """Evaluation summary for a single baseline regressor."""

    model_name: str
    modality: str
    metrics: RegressionMetrics


def evaluate_rpe_baseline(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    modality: str = "Fused",
) -> BaselineEvaluationResult:
    """Fits an RPE regressor and evaluates it against test ground-truth RPE."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 6.0, 20.0)
    metrics = compute_regression_metrics(y_test, y_pred)
    return BaselineEvaluationResult(
        model_name=model_name,
        modality=modality,
        metrics=metrics,
    )
