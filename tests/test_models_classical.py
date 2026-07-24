"""Unit tests for ml.models.classical (logistic regression, SVM, random
forest, XGBoost-with-fallback)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.models.classical.logistic_regression import LogisticRegressionConfig, build_logistic_regression
from ml.models.classical.random_forest import RandomForestConfig, build_random_forest
from ml.models.classical.svm import SVMConfig, build_svm
from ml.models.classical.xgboost import XGBoostConfig, build_xgboost, check_xgboost_availability

RNG = np.random.default_rng(3)


def _toy_classification_data():
    X = RNG.normal(0, 1, (100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.mark.parametrize(
    "builder",
    [build_logistic_regression, build_svm, build_random_forest, build_xgboost],
)
def test_classical_model_fit_predict_proba(builder):
    X, y = _toy_classification_data()
    model = builder()
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert preds.shape == (100,)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_logistic_regression_config_applied():
    config = LogisticRegressionConfig(C=0.5, max_iter=500, class_weight=None, random_state=1)
    model = build_logistic_regression(config)
    assert model.C == 0.5
    assert model.max_iter == 500
    assert model.class_weight is None


def test_svm_config_applied():
    config = SVMConfig(C=2.0, kernel="linear", class_weight=None)
    model = build_svm(config)
    assert model.C == 2.0
    assert model.kernel == "linear"


def test_random_forest_config_applied():
    config = RandomForestConfig(n_estimators=50, max_depth=5, class_weight=None)
    model = build_random_forest(config)
    assert model.n_estimators == 50
    assert model.max_depth == 5


def test_xgboost_config_applied():
    config = XGBoostConfig(n_estimators=50, max_depth=3, learning_rate=0.2)
    model = build_xgboost(config)
    # Works whether the real xgboost or the sklearn fallback was returned.
    assert model.n_estimators == 50
    assert model.max_depth == 3


def test_check_xgboost_availability_returns_tuple():
    is_available, reason = check_xgboost_availability()
    assert isinstance(is_available, bool)
    if not is_available:
        assert reason is not None
    else:
        assert reason is None


def test_default_configs_are_usable_without_arguments():
    for builder in (build_logistic_regression, build_svm, build_random_forest, build_xgboost):
        model = builder(None)
        assert model is not None
