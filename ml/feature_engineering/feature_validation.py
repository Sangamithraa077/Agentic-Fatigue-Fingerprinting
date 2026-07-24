"""Feature quality validation: missing/infinite values, constant and
near-zero-variance features, duplicates, highly correlated pairs, and
descriptive statistics.

Every check here is a pure function over a `pandas.DataFrame` (a
`FeatureDataset.matrix`), so each is independently unit testable and
reusable outside the full `validate_features` pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureValidationConfig:
    """Configuration for `validate_features`.

    Attributes:
        near_zero_variance_threshold: A feature with variance below this
            value is flagged as near-zero-variance.
        correlation_threshold: Absolute Pearson correlation at or above
            this value flags a feature pair as highly correlated.
    """

    near_zero_variance_threshold: float = 1e-6
    correlation_threshold: float = 0.95


@dataclass(frozen=True)
class FeatureIssue:
    """One validation finding for one feature."""

    feature_name: str
    issue_type: str
    detail: str
    related_feature: Optional[str] = None


@dataclass(frozen=True)
class FeatureValidationResult:
    """Full validation result for one modality's feature matrix."""

    modality: str
    n_samples: int
    n_features: int
    statistics: pd.DataFrame
    correlation_matrix: pd.DataFrame
    issues: Tuple[FeatureIssue, ...] = field(default_factory=tuple)

    def issues_by_type(self, issue_type: str) -> List[FeatureIssue]:
        return [issue for issue in self.issues if issue.issue_type == issue_type]


def check_missing_values(matrix: pd.DataFrame) -> Dict[str, int]:
    """Count NaN entries per feature column."""
    return matrix.isna().sum().to_dict()


def check_infinite_values(matrix: pd.DataFrame) -> Dict[str, int]:
    """Count +/-inf entries per feature column."""
    numeric = matrix.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}
    return np.isinf(numeric).sum().to_dict()


def check_constant_features(matrix: pd.DataFrame) -> List[str]:
    """Return feature names with at most one distinct non-null value."""
    return [column for column in matrix.columns if matrix[column].nunique(dropna=True) <= 1]


def check_near_zero_variance(matrix: pd.DataFrame, threshold: float) -> List[str]:
    """Return feature names whose variance is below `threshold`."""
    numeric = matrix.select_dtypes(include=[np.number])
    if numeric.empty:
        return []
    variances = numeric.var(skipna=True)
    return variances[variances < threshold].index.tolist()


def check_duplicate_features(matrix: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return (original, duplicate) pairs of columns that are element-wise
    identical (including matching NaN positions)."""
    pairs: List[Tuple[str, str]] = []
    columns = list(matrix.columns)
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            if matrix[col_a].equals(matrix[col_b]):
                pairs.append((col_a, col_b))
    return pairs


def check_highly_correlated(correlation_matrix: pd.DataFrame, threshold: float) -> List[Tuple[str, str, float]]:
    """Return (feature_a, feature_b, correlation) triples for every pair
    whose absolute correlation is at or above `threshold`."""
    pairs: List[Tuple[str, str, float]] = []
    columns = list(correlation_matrix.columns)
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            value = correlation_matrix.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((col_a, col_b, float(value)))
    return pairs


def compute_feature_statistics(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std/min/max/variance plus missing/infinite counts, one
    row per feature."""
    numeric = matrix.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame(columns=["mean", "std", "min", "max", "variance", "n_missing", "n_infinite"])
    return pd.DataFrame(
        {
            "mean": numeric.mean(),
            "std": numeric.std(),
            "min": numeric.min(),
            "max": numeric.max(),
            "variance": numeric.var(),
            "n_missing": numeric.isna().sum(),
            "n_infinite": np.isinf(numeric).sum(),
        }
    )


def compute_correlation_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation matrix over the numeric feature columns.

    Infinite values are replaced with NaN before computing the matrix, then
    handled by pandas' default pairwise-complete-observations logic.
    `pandas.DataFrame.corr()` does *not* reliably propagate `inf` to `NaN`
    on its own -- verified it can silently report a spurious 1.0
    correlation between two otherwise-identical columns when one contains
    an infinite value, which would be a dangerously misleading validation
    result. Replacing `inf`/`-inf` with NaN first ensures affected pairs are
    correctly treated as missing data rather than perfectly correlated.
    """
    numeric = matrix.select_dtypes(include=[np.number])
    cleaned = numeric.replace([np.inf, -np.inf], np.nan)
    return cleaned.corr()


def validate_features(
    matrix: pd.DataFrame, modality: str, config: Optional[FeatureValidationConfig] = None
) -> FeatureValidationResult:
    """Run the full validation battery on one modality's feature matrix.

    Args:
        matrix: `FeatureDataset.matrix` (one row per window, one column per feature).
        modality: Label for the result (e.g. "EMG").
        config: Thresholds for near-zero-variance and correlation checks.

    Returns:
        A `FeatureValidationResult` with statistics, a correlation matrix,
        and every issue found. This function only *reports* issues; it
        never removes or modifies any feature (see `feature_selection.py`
        for opt-in recommendations).
    """
    config = config or FeatureValidationConfig()
    statistics = compute_feature_statistics(matrix)
    correlation_matrix = compute_correlation_matrix(matrix)

    issues: List[FeatureIssue] = []

    for name, count in check_missing_values(matrix).items():
        if count > 0:
            issues.append(FeatureIssue(name, "missing_values", f"{count} missing value(s) ({count / len(matrix):.1%})"))

    for name, count in check_infinite_values(matrix).items():
        if count > 0:
            issues.append(FeatureIssue(name, "infinite_values", f"{count} infinite value(s)"))

    constant_features = check_constant_features(matrix)
    for name in constant_features:
        issues.append(FeatureIssue(name, "constant", "single unique value across all samples"))

    near_zero = set(check_near_zero_variance(matrix, config.near_zero_variance_threshold)) - set(constant_features)
    for name in near_zero:
        issues.append(FeatureIssue(name, "near_zero_variance", f"variance below threshold {config.near_zero_variance_threshold}"))

    for original, duplicate in check_duplicate_features(matrix):
        issues.append(FeatureIssue(duplicate, "duplicate", f"identical to '{original}'", related_feature=original))

    for col_a, col_b, corr in check_highly_correlated(correlation_matrix, config.correlation_threshold):
        issues.append(
            FeatureIssue(
                col_b, "highly_correlated", f"correlation {corr:.3f} with '{col_a}' (threshold {config.correlation_threshold})",
                related_feature=col_a,
            )
        )

    if issues:
        logger.info("%s feature validation found %d issue(s) across %d features.", modality, len(issues), matrix.shape[1])

    return FeatureValidationResult(
        modality=modality,
        n_samples=len(matrix),
        n_features=matrix.shape[1],
        statistics=statistics,
        correlation_matrix=correlation_matrix,
        issues=tuple(issues),
    )
