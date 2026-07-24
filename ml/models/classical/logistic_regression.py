"""Logistic Regression baseline: a linear, L2-regularized classifier.

Purpose in this benchmark: the simplest possible baseline -- if a more
complex model can't beat this, the complexity isn't earning its keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Hyperparameters for `build_logistic_regression`.

    Attributes:
        C: Inverse regularization strength (smaller = stronger regularization).
        max_iter: Maximum solver iterations.
        class_weight: `"balanced"` reweights classes inversely to frequency,
            appropriate given this project's class imbalance (e.g. few
            high-RPE windows relative to low-RPE ones); `None` disables this.
        random_state: Seed for the solver's internal randomness.
    """

    C: float = 1.0
    max_iter: int = 2000
    class_weight: Optional[str] = "balanced"
    random_state: int = 42


def build_logistic_regression(config: Optional[LogisticRegressionConfig] = None) -> LogisticRegression:
    """Construct a configured `LogisticRegression` estimator."""
    config = config or LogisticRegressionConfig()
    return LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
        random_state=config.random_state,
    )
