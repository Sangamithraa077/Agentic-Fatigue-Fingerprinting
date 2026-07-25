"""Global feature ranking from a matrix of per-sample SHAP values.

Deliberately separate from `global_explainer.py`: ranking a
(n_samples, n_features) array of importances by mean absolute value is a
generic operation with no SHAP- or model-specific knowledge, so it is kept
as its own reusable, independently testable function rather than inlined
into the SHAP orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ml.explainability.config import FeatureRankingConfig


@dataclass(frozen=True)
class RankedFeature:
    """One feature's position in a global importance ranking.

    Attributes:
        feature_name: Column name.
        importance: Mean absolute SHAP value across the explained samples
            (higher = more influential on the model's output, on average).
        rank: 0-indexed rank, 0 = most important.
    """

    feature_name: str
    importance: float
    rank: int


def rank_features(
    shap_values: np.ndarray, feature_names: Sequence[str], config: Optional[FeatureRankingConfig] = None
) -> Tuple[RankedFeature, ...]:
    """Rank features by mean absolute SHAP value, descending.

    Args:
        shap_values: (n_samples, n_features) per-sample SHAP contributions
            (e.g. `GlobalExplanation.explanation.values`, already reduced to
            a single output/class -- see `shap_explainer.py`).
        feature_names: One name per column of `shap_values`.
        config: Unused for the ranking computation itself (every feature is
            always ranked); accepted for API symmetry with the rest of this
            package and so callers can carry a single config through.

    Returns:
        Every feature, ranked descending by mean(|shap_value|) -- not just
        the top `config.top_k` (that slicing is a rendering-time concern
        for callers, e.g. `feature_ranking[: config.top_k]`).
    """
    del config  # not needed for ranking itself; see docstring
    shap_values = np.asarray(shap_values)
    if shap_values.ndim != 2:
        raise ValueError(f"shap_values must be 2D (n_samples, n_features), got shape {shap_values.shape}")
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {shap_values.shape[1]} feature columns but {len(feature_names)} feature_names were given."
        )

    mean_abs_importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_importance)[::-1]

    return tuple(
        RankedFeature(feature_name=feature_names[idx], importance=float(mean_abs_importance[idx]), rank=rank)
        for rank, idx in enumerate(order)
    )
