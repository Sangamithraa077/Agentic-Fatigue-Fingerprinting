"""Feature importance PROXIES: Mutual Information and ANOVA F-score.

## These are exploratory diagnostics, not final model importance

The scores computed here rank features by their univariate statistical
association with a single, simple metadata/label column (e.g.
`intensity_band` or `rpe_nearest`) computed independently for each
feature. This is standard, well-understood feature-screening practice, but
it is **not** the importance a trained multimodal fatigue model would
assign to these features:

- It ignores interactions between features (a feature can be highly
  predictive jointly with another while scoring low individually, or vice
  versa).
- It is computed against a simple proxy target (metadata label or a
  nearest-neighbor Borg RPE lookup -- see `feature_dataset.nearest_rpe`),
  not the actual modeling target/formulation a later phase will define.
- It says nothing about performance in the personalized (per-subject) or
  temporal (fatigue-trajectory) modeling this project ultimately targets.

Use these scores to flag candidates for review (very low scores) or to
sanity-check that known-relevant modalities (e.g. EMG amplitude features
vs. fatigue intensity) show the expected association -- not as a
substitute for model-based feature importance or SHAP (a later phase).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)

# Metadata columns that are categorical labels rather than continuous
# targets, used to decide which sklearn estimator family applies.
CATEGORICAL_TARGET_COLUMNS = frozenset({"intensity_band", "condition_type", "direction", "source_sensor"})


@dataclass(frozen=True)
class FeatureImportanceConfig:
    """Configuration for `compute_feature_importance`.

    Attributes:
        target_column: Which metadata column (see
            `feature_dataset.METADATA_COLUMNS`) to compute importance
            against. Categorical columns (see `CATEGORICAL_TARGET_COLUMNS`)
            use `mutual_info_classif`/`f_classif`; anything else (e.g.
            `rpe_nearest`) is treated as continuous and uses
            `mutual_info_regression`/`f_regression`.
        random_state: Passed to sklearn's mutual-information estimators
            (they use a randomized nearest-neighbor procedure internally).
    """

    target_column: str = "intensity_band"
    random_state: int = 42


@dataclass(frozen=True)
class FeatureImportanceResult:
    """Exploratory importance scores for one modality against one target."""

    modality: str
    target_column: str
    target_type: str
    n_samples_used: int
    n_samples_excluded: int
    mutual_information: pd.Series
    anova_f_score: pd.Series
    anova_p_value: pd.Series
    notes: Tuple[str, ...] = field(default_factory=tuple)


def compute_feature_importance(
    matrix: pd.DataFrame, metadata: pd.DataFrame, modality: str, config: Optional[FeatureImportanceConfig] = None
) -> Optional[FeatureImportanceResult]:
    """Compute Mutual Information and ANOVA F-score for every feature
    against one target metadata column.

    MI/F-score are computed on **complete cases only**: rows with any NaN
    among the features being scored or a NaN/inapplicable target value are
    excluded (this is a diagnostic-scope restriction, not imputation --
    see `notes` on the returned result for exactly how many rows that was).

    Args:
        matrix: `FeatureDataset.matrix`.
        metadata: `FeatureDataset.metadata` (same row order as `matrix`).
        modality: Label for the result (e.g. "EMG").
        config: Which target column to use, and sklearn's random state.

    Returns:
        A `FeatureImportanceResult`, or `None` if the target column is
        missing or no complete rows are available (e.g. an all-MVIC subset
        scored against `rpe_nearest`, which is NaN for every MVIC window).
    """
    config = config or FeatureImportanceConfig()
    target_column = config.target_column

    if target_column not in metadata.columns:
        logger.warning("Target column '%s' not found in metadata -- skipping importance for %s.", target_column, modality)
        return None
    if matrix.empty:
        logger.warning("Empty feature matrix -- skipping importance for %s.", modality)
        return None

    combined = matrix.reset_index(drop=True).copy()
    combined["__target__"] = metadata[target_column].reset_index(drop=True).values
    # Treat +/-inf as missing for this diagnostic: sklearn's estimators below
    # do not handle infinite inputs gracefully, and a row containing one
    # should be excluded from "complete cases" exactly like a NaN would be.
    combined = combined.replace([np.inf, -np.inf], np.nan)
    complete = combined.dropna()

    n_excluded = len(combined) - len(complete)
    notes = [
        f"{n_excluded} of {len(combined)} row(s) excluded (missing/infinite feature value or missing/inapplicable "
        f"target '{target_column}') -- scores computed on complete cases only, not imputed."
    ]

    if complete.empty:
        logger.warning("No complete rows available for importance computation (%s, target=%s).", modality, target_column)
        return None

    feature_columns = list(matrix.columns)
    X = complete[feature_columns].to_numpy()
    is_categorical = target_column in CATEGORICAL_TARGET_COLUMNS

    if is_categorical:
        y = complete["__target__"].astype(str).to_numpy()
        mi_scores = mutual_info_classif(X, y, random_state=config.random_state)
        f_scores, p_values = f_classif(X, y)
        target_type = "categorical"
        notes.append(f"Target '{target_column}' treated as categorical -- used mutual_info_classif/f_classif.")
    else:
        y = complete["__target__"].to_numpy(dtype=float)
        mi_scores = mutual_info_regression(X, y, random_state=config.random_state)
        f_scores, p_values = f_regression(X, y)
        target_type = "continuous"
        notes.append(f"Target '{target_column}' treated as continuous -- used mutual_info_regression/f_regression.")

    mutual_information = pd.Series(mi_scores, index=feature_columns, name="mutual_information").sort_values(ascending=False)
    anova_f_score = pd.Series(f_scores, index=feature_columns, name="anova_f_score").reindex(mutual_information.index)
    anova_p_value = pd.Series(p_values, index=feature_columns, name="anova_p_value").reindex(mutual_information.index)

    return FeatureImportanceResult(
        modality=modality,
        target_column=target_column,
        target_type=target_type,
        n_samples_used=len(complete),
        n_samples_excluded=n_excluded,
        mutual_information=mutual_information,
        anova_f_score=anova_f_score,
        anova_p_value=anova_p_value,
        notes=tuple(notes),
    )
