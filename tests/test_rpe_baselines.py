"""Unit tests for Module 5: Continuous Borg RPE Baselines & Regression Metrics."""

import numpy as np
import pytest

from ml.models.baselines.rpe_baselines import (
    ElapsedTimeBaseline,
    MeanPredictorBaseline,
    evaluate_rpe_baseline,
)
from ml.models.common.metrics import compute_regression_metrics


def test_compute_regression_metrics():
    y_true = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    y_pred = np.array([10.2, 11.8, 14.5, 15.5, 18.1])

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics.n_samples == 5
    assert metrics.mae < 0.4
    assert metrics.rmse < 0.5
    assert metrics.pearson_r > 0.95
    assert metrics.within_1_rpe == 1.0  # All errors are <= 0.5 <= 1.0


def test_mean_predictor_baseline():
    X_train = np.ones((10, 5))
    y_train = np.array([12.0, 14.0, 16.0, 18.0, 20.0, 10.0, 12.0, 14.0, 16.0, 18.0])
    mean_val = float(np.mean(y_train))

    X_test = np.ones((4, 5))
    model = MeanPredictorBaseline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == 4
    assert np.allclose(preds, mean_val)


def test_elapsed_time_baseline():
    X_train = np.zeros((10, 4))
    y_train = np.linspace(6.0, 18.0, 10)

    model = ElapsedTimeBaseline()
    model.fit(X_train, y_train)

    X_test = np.zeros((5, 4))
    preds = model.predict(X_test)
    assert len(preds) == 5
    # Should be strictly monotonic increasing
    assert preds[0] < preds[-1]
    assert np.all(preds >= 6.0)
    assert np.all(preds <= 20.0)


def test_evaluate_rpe_baseline_helper():
    X_train = np.random.randn(20, 6)
    y_train = np.linspace(7.0, 17.0, 20)
    X_test = np.random.randn(8, 6)
    y_test = np.linspace(8.0, 16.0, 8)

    res = evaluate_rpe_baseline(
        model=MeanPredictorBaseline(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="MeanDummy",
        modality="EMG",
    )
    assert res.model_name == "MeanDummy"
    assert res.modality == "EMG"
    assert res.metrics.n_samples == 8
