"""Support Vector Machine baseline (RBF kernel).

Purpose in this benchmark: a non-linear classical baseline that can capture
feature interactions a purely linear model (logistic regression) cannot,
without the sample-inefficiency of a full deep network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.svm import SVC


@dataclass(frozen=True)
class SVMConfig:
    """Hyperparameters for `build_svm`.

    Attributes:
        C: Regularization strength (larger = less regularization).
        kernel: SVM kernel; RBF is the standard non-linear default.
        gamma: Kernel coefficient; `"scale"` (1 / (n_features * X.var()))
            is a robust, scale-aware default.
        class_weight: `"balanced"` reweights classes inversely to frequency.
        probability: Must be `True` to expose `predict_proba` for
            ROC-AUC/PR-AUC (adds a computational cost via internal
            cross-validation, but is required by this benchmark's shared
            evaluation interface).
        random_state: Seed (used internally by `probability=True`'s CV).
    """

    C: float = 1.0
    kernel: str = "rbf"
    gamma: str = "scale"
    class_weight: Optional[str] = "balanced"
    probability: bool = True
    random_state: int = 42


def build_svm(config: Optional[SVMConfig] = None) -> SVC:
    """Construct a configured `SVC` estimator."""
    config = config or SVMConfig()
    return SVC(
        C=config.C,
        kernel=config.kernel,
        gamma=config.gamma,
        class_weight=config.class_weight,
        probability=config.probability,
        random_state=config.random_state,
    )
