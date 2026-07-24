"""Configurable feature selection RECOMMENDATIONS.

Every function here reports what it would suggest at a given, explicit
threshold -- **none of them remove or modify any feature**. `FeatureSelector.recommend`
returns a plain list of `SelectionRecommendation`s for a human (or a later,
explicitly-configured phase) to review and act on. To actually drop
columns, call `apply_recommendations` yourself; it is never called
automatically anywhere in this pipeline.

Three methods are implemented, matching the phase requirements:

- **Variance Threshold** -- flags constant and near-zero-variance features
  (recomputed here from the raw matrix using this module's own configured
  threshold, independent of whatever threshold `feature_validation.py`'s
  diagnostic report used -- see module note below).
- **Correlation Filtering** -- flags exact duplicates (`remove`) and highly
  correlated pairs (`review`, since which of a correlated pair to drop is
  a judgment call this module does not make for you).
- **Mutual Information Ranking** -- flags the lowest-scoring fraction of
  features against a chosen target as `review` candidates.

Recommendation functions here recompute variance/duplicate/correlation
checks directly from the raw matrix (reusing `feature_validation.py`'s
pure check functions) rather than consuming a pre-built
`FeatureValidationResult`, so this module's thresholds are independently
configurable from whatever thresholds were used for the separate
diagnostic validation report -- selection and reporting are deliberately
decoupled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from ml.feature_engineering.feature_importance import FeatureImportanceResult
from ml.feature_engineering.feature_validation import (
    check_constant_features,
    check_duplicate_features,
    check_highly_correlated,
    check_near_zero_variance,
    compute_correlation_matrix,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Configuration for `FeatureSelector`.

    Attributes:
        variance_threshold: Features with variance below this are
            recommended for removal.
        correlation_threshold: Feature pairs at or above this absolute
            correlation are flagged for review.
        mutual_info_bottom_fraction: The lowest-scoring this-fraction of
            features by mutual information are flagged for review.
    """

    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.95
    mutual_info_bottom_fraction: float = 0.1


@dataclass(frozen=True)
class SelectionRecommendation:
    """One recommendation for one feature."""

    feature_name: str
    method: str
    recommendation: str  # "remove" | "review"
    reason: str


def recommend_variance_threshold(matrix: pd.DataFrame, threshold: float) -> List[SelectionRecommendation]:
    """Recommend removing constant and near-zero-variance features."""
    constant_features = check_constant_features(matrix)
    recommendations = [
        SelectionRecommendation(name, "variance_threshold", "remove", "constant feature (single unique value)")
        for name in constant_features
    ]
    near_zero = set(check_near_zero_variance(matrix, threshold)) - set(constant_features)
    recommendations.extend(
        SelectionRecommendation(name, "variance_threshold", "remove", f"variance below threshold {threshold}")
        for name in near_zero
    )
    return recommendations


def recommend_correlation_filter(
    matrix: pd.DataFrame, threshold: float, correlation_matrix: Optional[pd.DataFrame] = None
) -> List[SelectionRecommendation]:
    """Recommend removing exact duplicates and reviewing highly correlated pairs."""
    recommendations = [
        SelectionRecommendation(duplicate, "correlation_filter", "remove", f"identical to '{original}'")
        for original, duplicate in check_duplicate_features(matrix)
    ]
    corr = correlation_matrix if correlation_matrix is not None else compute_correlation_matrix(matrix)
    for col_a, col_b, value in check_highly_correlated(corr, threshold):
        recommendations.append(
            SelectionRecommendation(
                col_b, "correlation_filter", "review", f"correlation {value:.3f} with '{col_a}' (threshold {threshold})"
            )
        )
    return recommendations


def recommend_mutual_info_ranking(
    importance_result: Optional[FeatureImportanceResult], bottom_fraction: float
) -> List[SelectionRecommendation]:
    """Recommend reviewing the lowest-MI-scoring `bottom_fraction` of
    features against `importance_result`'s target.

    Returns an empty list if `importance_result` is None (e.g. the target
    column was unavailable for this modality/subset).
    """
    if importance_result is None or importance_result.mutual_information.empty:
        return []

    scores = importance_result.mutual_information.sort_values(ascending=True)
    n_flag = max(1, int(round(len(scores) * bottom_fraction)))
    flagged = scores.iloc[:n_flag]

    return [
        SelectionRecommendation(
            name,
            "mutual_info_ranking",
            "review",
            f"low mutual information ({value:.4f}) with target '{importance_result.target_column}' -- "
            f"among the bottom {bottom_fraction:.0%} of {importance_result.modality} features (exploratory only)",
        )
        for name, value in flagged.items()
    ]


class FeatureSelector:
    """Combines all three selection methods into one recommendation list.

    Example:
        >>> selector = FeatureSelector()
        >>> recommendations = selector.recommend(dataset.matrix, importance_result)
        >>> to_remove = [r for r in recommendations if r.recommendation == "remove"]
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None) -> None:
        self.config = config or FeatureSelectionConfig()

    def recommend(
        self,
        matrix: pd.DataFrame,
        importance_result: Optional[FeatureImportanceResult] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> List[SelectionRecommendation]:
        """Run all three configured methods and return every recommendation.

        Does not remove or modify `matrix`. Call `apply_recommendations`
        explicitly if you want to act on the result.
        """
        recommendations: List[SelectionRecommendation] = []
        recommendations.extend(recommend_variance_threshold(matrix, self.config.variance_threshold))
        recommendations.extend(recommend_correlation_filter(matrix, self.config.correlation_threshold, correlation_matrix))
        recommendations.extend(recommend_mutual_info_ranking(importance_result, self.config.mutual_info_bottom_fraction))
        return recommendations


def apply_recommendations(
    matrix: pd.DataFrame, recommendations: Sequence[SelectionRecommendation], actions: Sequence[str] = ("remove",)
) -> pd.DataFrame:
    """Explicitly drop columns recommended under any of `actions`.

    This is an opt-in utility the caller must invoke deliberately -- it is
    never called automatically anywhere in this pipeline, per the
    requirement that features are never removed automatically.

    Args:
        matrix: The feature matrix to filter.
        recommendations: Recommendations from `FeatureSelector.recommend`.
        actions: Which recommendation types to act on (default: only
            "remove"; pass `("remove", "review")` to also drop review-flagged
            features).

    Returns:
        A new DataFrame with the selected columns dropped.
    """
    to_drop = {r.feature_name for r in recommendations if r.recommendation in actions}
    dropped = [c for c in to_drop if c in matrix.columns]
    if dropped:
        logger.info("Dropping %d feature column(s) per applied recommendations: %s", len(dropped), dropped)
    return matrix.drop(columns=dropped)
