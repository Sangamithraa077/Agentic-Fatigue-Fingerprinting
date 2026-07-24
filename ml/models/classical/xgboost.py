"""XGBoost baseline (gradient-boosted trees), with a graceful, documented
fallback to `sklearn.ensemble.GradientBoostingClassifier` if the `xgboost`
package is not installed.

Purpose in this benchmark: gradient boosting typically represents the
strongest classical baseline for tabular data, giving the deep learning
models a genuinely competitive bar to clear.

Note: this module is named `xgboost.py` (matching the phase's requested
file layout) yet imports the real third-party `xgboost` package by the
same name -- verified safe: Python 3's absolute-import resolution finds the
installed top-level package correctly from within
`ml.models.classical.xgboost`, since this project is never run in a way
that adds `ml/models/classical/` itself to `sys.path`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
    _UNAVAILABLE_REASON: Optional[str] = None
except ImportError as exc:  # pragma: no cover - exercised only when xgboost is absent
    _XGBOOST_AVAILABLE = False
    _UNAVAILABLE_REASON = str(exc)
    logger.warning(
        "xgboost is not installed (%s) -- build_xgboost() will fall back to "
        "sklearn.ensemble.GradientBoostingClassifier with a matched hyperparameter mapping.",
        _UNAVAILABLE_REASON,
    )


@dataclass(frozen=True)
class XGBoostConfig:
    """Hyperparameters for `build_xgboost` (also used for the fallback model).

    Attributes:
        n_estimators: Number of boosting rounds/trees.
        max_depth: Maximum tree depth.
        learning_rate: Boosting shrinkage rate.
        random_state: Seed for reproducibility.
    """

    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.1
    random_state: int = 42


def check_xgboost_availability() -> Tuple[bool, Optional[str]]:
    """Return `(is_available, reason_if_not)` for the real `xgboost` package."""
    return _XGBOOST_AVAILABLE, _UNAVAILABLE_REASON


def build_xgboost(config: Optional[XGBoostConfig] = None):
    """Construct a gradient-boosted-trees classifier.

    Returns an `xgboost.XGBClassifier` if the package is installed;
    otherwise falls back to `sklearn.ensemble.GradientBoostingClassifier`
    with an equivalent hyperparameter mapping, logging a warning. Both
    expose the same `.fit`/`.predict`/`.predict_proba` interface, so callers
    never need to branch on which one they received.
    """
    config = config or XGBoostConfig()

    if _XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
            eval_metric="logloss",
        )

    logger.warning(
        "Falling back to GradientBoostingClassifier for this run because xgboost is unavailable (%s).",
        _UNAVAILABLE_REASON,
    )
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
    )
