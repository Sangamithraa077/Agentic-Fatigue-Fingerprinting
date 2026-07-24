"""Builds (X, y, groups) modeling arrays from a Phase 8.5 `FeatureDataset`,
given a `DatasetConfig`.

## Why deep learning models consume the same tabular feature matrix

Phase 9's models are compared "using identical datasets" -- so every model
(classical or deep learning) is trained on the exact same per-window
feature matrix that Phase 8 extracted (MAV, RMS, per-axis IMU statistics,
etc.), not on raw signal windows. The 1D CNN/BiLSTM/CNN-BiLSTM/TCN
architectures therefore treat one sample's feature vector as a length-N
"sequence" (one step per feature) rather than a true time series -- a
standard tabular-deep-learning technique, and the only way to honor "use
the validated feature matrices produced in Phase 8.5" for every model type.
This is documented again in `ml/models/deep_learning/*.py`.

## Target derivation (exact rules)

All three targets are derived from `FeatureDataset.metadata["rpe_nearest"]`
(nearest-in-time Borg RPE reading per window; Phase 8.5
`feature_dataset.nearest_rpe`). Rows where `rpe_nearest` is NaN (every
MVIC-calibration window -- no Borg data was collected during MVIC trials)
are excluded from all three targets, since a fatigue label is not
meaningful for a calibration trial.

1. **binary_fatigue**: `1` if `rpe_nearest >= binary_fatigue_threshold`
   (default 13 -- Borg's own "somewhat hard" verbal anchor, roughly the
   midpoint of the 6-20 scale), else `0`.
2. **fatigue_level**: 3 ordinal classes ("low"/"moderate"/"high") from
   `fatigue_level_bins` (default edges (6, 11, 16, 21)).
3. **borg_rpe_class**: the rounded integer RPE value itself, label-encoded
   to a contiguous 0..k-1 range (up to 15 possible classes, 6-20) -- the
   finest-grained target, not binned.

## Feature subset derivation

- **all**: every column of `FeatureDataset.matrix`.
- **variance_filtered**: drops "remove"-flagged columns from Phase 8.5's
  `recommend_variance_threshold` (constant / near-zero-variance).
- **correlation_filtered**: additionally drops "remove"-flagged (exact
  duplicate) columns from `recommend_correlation_filter`. Merely "review"
  -flagged (highly correlated but not identical) columns are kept --
  choosing which of a correlated pair to drop is a judgment call this
  automatic subset does not make.
- **mutual_info_ranked**: keeps the top `mi_top_fraction` of columns by
  mutual information computed directly against *this experiment's actual
  target* (not a generic metadata proxy) via `sklearn.mutual_info_classif`.

## Data cleaning: imputation and scaling (documented simplification)

Phase 8.5 identified a small number of extreme-outlier / corrupted raw
values (see `reports/feature_validation_report.md`). `build_modeling_dataset`
replaces +/-inf with NaN and imputes remaining NaNs with each feature's
column median computed **once, over the whole selected dataset** (not
separately per train/test split) -- a deliberate, documented simplification
for this baseline phase (a stricter pipeline would fit imputation
statistics on the training fold only). `fit_scaler`/`apply_scaler` use a
`RobustScaler` (median/IQR-based) rather than a mean/std `StandardScaler`,
since standard scaling would be badly distorted by the known extreme
outliers; scaling itself *is* fit train-only by callers (see
`common/evaluation.py`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler

from ml.feature_engineering.feature_dataset import FeatureDataset
from ml.feature_engineering.feature_selection import recommend_correlation_filter, recommend_variance_threshold
from ml.models.common.config import DatasetConfig

logger = logging.getLogger(__name__)

FATIGUE_LEVEL_LABELS: Tuple[str, ...] = ("low", "moderate", "high")


@dataclass(frozen=True)
class ModelingDataset:
    """The (X, y, groups) triple ready for modeling, plus provenance.

    Attributes:
        X: (n_samples, n_features) array -- imputed/cleaned but *not*
            scaled (scaling is fit per train split; see `fit_scaler`).
        y: (n_samples,) integer class labels, 0-indexed and contiguous.
        groups: (n_samples,) subject_id per row, for group-aware splitting.
        feature_names: Column names corresponding to `X`'s columns.
        class_names: Human-readable label for each class index in `y`.
        dataset_config: The config that produced this dataset.
        n_rows_excluded: Rows dropped for having no applicable target.
        n_values_imputed: Individual NaN/inf feature values that were imputed.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: Tuple[str, ...]
    class_names: Tuple[str, ...]
    dataset_config: DatasetConfig
    n_rows_excluded: int
    n_values_imputed: int


def derive_target(metadata: pd.DataFrame, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Derive the target array and a validity mask from `rpe_nearest`.

    Returns:
        `(y, valid_mask, class_names)`. `y` has one entry per row of
        `metadata`; entries where `valid_mask` is False are placeholders
        and must not be used -- callers apply the mask themselves.
    """
    rpe = metadata["rpe_nearest"].to_numpy(dtype=float)
    valid_mask = ~np.isnan(rpe)

    if config.target_type == "binary_fatigue":
        y = (rpe >= config.binary_fatigue_threshold).astype(int)
        class_names = ("not_fatigued", "fatigued")

    elif config.target_type == "fatigue_level":
        edges = config.fatigue_level_bins
        y = np.digitize(rpe, edges[1:-1], right=False)
        y = np.clip(y, 0, len(FATIGUE_LEVEL_LABELS) - 1)
        class_names = FATIGUE_LEVEL_LABELS

    elif config.target_type == "borg_rpe_class":
        # Fill invalid (NaN) entries with a placeholder *before* casting to
        # int -- casting NaN directly to int is undefined behavior and
        # raises a RuntimeWarning; the placeholder is masked out by
        # valid_mask everywhere it's used, so its exact value is irrelevant.
        rpe_filled = np.where(valid_mask, rpe, -1.0)
        rounded = np.round(rpe_filled).astype(int)
        unique_values = sorted(set(rounded[valid_mask].tolist()))
        value_to_index = {value: index for index, value in enumerate(unique_values)}
        y = np.array([value_to_index.get(value, -1) for value in rounded])
        class_names = tuple(str(value) for value in unique_values)

    else:
        raise ValueError(f"Unknown target_type: {config.target_type!r}")

    return y, valid_mask, class_names


def select_feature_subset(
    matrix: pd.DataFrame, config: DatasetConfig, y_valid: np.ndarray, valid_mask: np.ndarray
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Select feature columns per `config.feature_subset` and return the
    resulting (full-length, not yet row-filtered) array plus column names."""
    if config.feature_subset == "all":
        selected_columns = list(matrix.columns)

    elif config.feature_subset == "variance_filtered":
        recs = recommend_variance_threshold(matrix, config.variance_threshold)
        drop = {r.feature_name for r in recs if r.recommendation == "remove"}
        selected_columns = [c for c in matrix.columns if c not in drop]

    elif config.feature_subset == "correlation_filtered":
        variance_recs = recommend_variance_threshold(matrix, config.variance_threshold)
        correlation_recs = recommend_correlation_filter(matrix, config.correlation_threshold)
        drop = {r.feature_name for r in (*variance_recs, *correlation_recs) if r.recommendation == "remove"}
        selected_columns = [c for c in matrix.columns if c not in drop]

    elif config.feature_subset == "mutual_info_ranked":
        selected_columns = _rank_by_mutual_information(matrix, y_valid, valid_mask, config.mi_top_fraction)

    else:
        raise ValueError(f"Unknown feature_subset: {config.feature_subset!r}")

    if not selected_columns:
        logger.warning("Feature subset '%s' selected zero columns -- falling back to all features.", config.feature_subset)
        selected_columns = list(matrix.columns)

    return matrix[selected_columns].to_numpy(dtype=float), tuple(selected_columns)


def _rank_by_mutual_information(matrix: pd.DataFrame, y_valid: np.ndarray, valid_mask: np.ndarray, top_fraction: float) -> list:
    matrix_valid = matrix.loc[valid_mask].reset_index(drop=True)
    cleaned = matrix_valid.replace([np.inf, -np.inf], np.nan)
    complete_mask = cleaned.notna().all(axis=1).to_numpy()

    if complete_mask.sum() < 10:
        logger.warning(
            "Only %d complete rows available for mutual-information ranking -- falling back to all features.",
            int(complete_mask.sum()),
        )
        return list(matrix.columns)

    mi_scores = mutual_info_classif(cleaned.loc[complete_mask], y_valid[complete_mask], random_state=42)
    ranked = pd.Series(mi_scores, index=matrix.columns).sort_values(ascending=False)
    n_keep = max(1, int(round(len(ranked) * top_fraction)))
    return ranked.head(n_keep).index.tolist()


_EXTREME_OUTLIER_ROBUST_Z_THRESHOLD = 1000.0


def _impute_and_clean(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """Replace +/-inf with NaN, then impute NaN (and confirmed extreme
    outliers) with each column's median.

    Phase 8.5 independently confirmed a small number of raw EMG values on
    the order of 1e28-1e34 -- physiologically impossible, yet finite (not
    NaN/inf), so they are untouched by a plain inf/NaN check and instead
    trip a downstream `RobustScaler` (dividing by a tiny IQR can push such
    a value past float32's representable range, producing a fresh `inf`
    that then breaks sklearn/torch). This function therefore also clips
    per-column values whose robust z-score (deviation from the column
    median, scaled by median absolute deviation) exceeds
    `_EXTREME_OUTLIER_ROBUST_Z_THRESHOLD` (1000 -- deliberately very high,
    so only the kind of many-orders-of-magnitude corruption Phase 8.5
    documented is affected, not ordinary high-but-real variance).

    Returns the cleaned array and the total count of individually
    imputed/clipped values.
    """
    X = np.asarray(X, dtype=float).copy()
    X[np.isinf(X)] = np.nan

    with np.errstate(invalid="ignore"):
        column_medians = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - column_medians), axis=0)
        safe_mad = np.where(mad > 0, mad, 1.0)
        robust_z = np.abs(X - column_medians) / safe_mad
    extreme_mask = np.nan_to_num(robust_z, nan=0.0) > _EXTREME_OUTLIER_ROBUST_Z_THRESHOLD
    X[extreme_mask] = np.nan

    nan_mask = np.isnan(X)
    n_imputed = int(nan_mask.sum())
    if n_imputed:
        column_medians = np.nan_to_num(column_medians, nan=0.0)
        rows, cols = np.where(nan_mask)
        X[rows, cols] = column_medians[cols]
    return X, n_imputed


def build_modeling_dataset(dataset: FeatureDataset, config: DatasetConfig) -> ModelingDataset:
    """Build a complete `ModelingDataset` from one modality's `FeatureDataset`.

    Raises:
        ValueError: If `dataset.matrix` is empty, or the dataset's modality
            does not match `config.modality`.
    """
    if dataset.matrix.empty:
        raise ValueError(f"FeatureDataset for modality '{dataset.modality}' is empty -- cannot build a modeling dataset.")
    if dataset.modality != config.modality:
        raise ValueError(f"Dataset modality '{dataset.modality}' does not match config modality '{config.modality}'.")

    y_full, valid_mask, class_names = derive_target(dataset.metadata, config)
    n_excluded = int((~valid_mask).sum())
    if n_excluded:
        logger.info(
            "%s/%s: excluding %d/%d row(s) with no applicable '%s' target (e.g. MVIC trials).",
            dataset.modality, config.feature_subset, n_excluded, len(y_full), config.target_type,
        )

    X_all_rows, feature_names = select_feature_subset(dataset.matrix, config, y_full[valid_mask], valid_mask)

    X = X_all_rows[valid_mask]
    y = y_full[valid_mask]
    groups = dataset.metadata["subject_id"].to_numpy()[valid_mask]

    X_clean, n_imputed = _impute_and_clean(X)
    if n_imputed:
        logger.info("%s: imputed %d NaN/infinite feature value(s) with column medians.", dataset.modality, n_imputed)

    return ModelingDataset(
        X=X_clean,
        y=y,
        groups=groups,
        feature_names=feature_names,
        class_names=class_names,
        dataset_config=config,
        n_rows_excluded=n_excluded,
        n_values_imputed=n_imputed,
    )


def fit_scaler(X_train: np.ndarray) -> RobustScaler:
    """Fit a median/IQR-based `RobustScaler` on the training split only."""
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: RobustScaler, X: np.ndarray) -> np.ndarray:
    """Apply an already-fit scaler to `X` (train, val, or test)."""
    return scaler.transform(X)
