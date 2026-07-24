"""Aligns per-modality Phase 8.5 `FeatureDataset`s (EMG/IMU/PPG) into shared
multimodal samples for Phase 10's fusion model.

## Why this exists

Phases 7/8/8.5 deliberately kept each modality's windows on its own grid:
EMG and IMU default to 1.0s windows (50% overlap), while PPG uses 10.0s
windows, since a 1-second window can't contain enough heartbeats for
meaningful pulse-rate features (Phase 8's own documented rationale). That
was the right call for single-modality feature extraction, but a fusion
model needs one row per multimodal sample -- solving that alignment is
exactly Phase 10's job, not a further deferral.

## Two-step alignment

1. **Combine across sensors within a modality.** Each `FeatureDataset` is
   long-format: one row per (subject, condition, *sensor*, window_index)
   -- e.g. EMG has 6 rows per window (one per muscle), IMU has 6 (one per
   location). `combine_sensors_per_window` pivots this into one row per
   (subject, condition, window_index), with every sensor's features as
   separate columns prefixed by sensor name (e.g. `anterior_deltoid__mav`).
2. **Align across modalities by nearest window-center time.** EMG and IMU
   share the same window config by default, so their window indices
   already correspond to the same nominal time span (verified against real
   data: identical window-index sets per condition) -- but this module
   does not *assume* that; it matches purely by computed center time via
   `pandas.merge_asof(..., direction="nearest")`, grouped by
   (subject_id, condition_id), which is correct regardless of whether the
   window configs happen to coincide. PPG's much coarser windows are
   matched the same way: every EMG/IMU window is paired with whichever PPG
   window's center time is closest, so multiple EMG/IMU windows
   legitimately share one PPG window's features -- an intentional,
   documented many-to-one mapping, not a bug.

MVIC-calibration conditions have no PPG recording at all, so they get no
PPG match (`merge_asof` leaves those rows/columns absent); combined with
`derive_target` (reused from `dataset_builder.py`) excluding any row with
no Borg RPE, MVIC rows never survive into a trainable multimodal sample --
consistent with every prior phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ml.feature_engineering.feature_dataset import METADATA_COLUMNS, FeatureDataset
from ml.models.common.config import DatasetConfig
from ml.models.common.dataset_builder import impute_and_clean, derive_target

logger = logging.getLogger(__name__)

_JOIN_KEYS = ("subject_id", "condition_id")
_SHARED_METADATA_COLUMNS = tuple(c for c in METADATA_COLUMNS if c != "source_sensor")


@dataclass(frozen=True)
class ModalityDims:
    """Per-modality feature-column counts, in the fixed order the combined
    array's columns are concatenated (EMG, then IMU, then PPG)."""

    emg: int
    imu: int
    ppg: int

    @property
    def total(self) -> int:
        return self.emg + self.imu + self.ppg

    def split(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a horizontally-concatenated (n, total) array back into
        its three per-modality (n, emg|imu|ppg) parts, in column order."""
        emg_end = self.emg
        imu_end = emg_end + self.imu
        return X[:, :emg_end], X[:, emg_end:imu_end], X[:, imu_end:]


@dataclass(frozen=True)
class MultimodalDataset:
    """The aligned, concatenated (X, y, groups) triple for the fusion model.

    Attributes:
        X: (n_samples, emg_dim + imu_dim + ppg_dim) array -- EMG, IMU, and
            PPG features horizontally concatenated in that order. Split
            back into per-modality arrays via `dims.split(X)`.
        y, groups, class_names: Same meaning as `dataset_builder.ModelingDataset`.
        dims: Column counts needed to split `X` back into per-modality arrays.
        feature_names: Per-modality feature (column) names, in the same
            order as `X`'s columns.
        dataset_config: The config used to derive the target.
        n_rows_excluded: Rows dropped for having no applicable target
            (includes all MVIC-calibration rows, which have no Borg RPE).
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    dims: ModalityDims
    feature_names: Dict[str, Tuple[str, ...]]
    class_names: Tuple[str, ...]
    dataset_config: DatasetConfig
    n_rows_excluded: int


def combine_sensors_per_window(dataset: FeatureDataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot one modality's long-format `FeatureDataset` into one row per
    (subject_id, condition_id, window_index), with every sensor's features
    as separate, sensor-prefixed columns.

    Returns:
        (matrix, metadata) -- `matrix` has one column per
        `"{sensor}__{feature}"`; `metadata` carries the shared per-window
        fields (everything in `METADATA_COLUMNS` except `source_sensor`,
        which no longer identifies a single row once sensors are combined).
    """
    if dataset.matrix.empty:
        return pd.DataFrame(), pd.DataFrame(columns=_SHARED_METADATA_COLUMNS)

    combined = pd.concat([dataset.metadata.reset_index(drop=True), dataset.matrix.reset_index(drop=True)], axis=1)
    key_cols = ["subject_id", "condition_id", "window_index"]

    rows = []
    for _, group in combined.groupby(key_cols, sort=False):
        row = {col: group.iloc[0][col] for col in _SHARED_METADATA_COLUMNS}
        for _, sensor_row in group.iterrows():
            sensor_name = sensor_row["source_sensor"]
            for feature_name in dataset.feature_names:
                row[f"{sensor_name}__{feature_name}"] = sensor_row[feature_name]
        rows.append(row)

    result = pd.DataFrame(rows)
    feature_columns = [c for c in result.columns if c not in _SHARED_METADATA_COLUMNS]
    return result[feature_columns], result[list(_SHARED_METADATA_COLUMNS)]


def _window_center_seconds(metadata: pd.DataFrame, window_seconds: float, overlap_ratio: float) -> pd.Series:
    """Compute each row's window center time from `window_index` and the
    (window_seconds, overlap_ratio) config it was extracted with -- Phase
    8.5's `FeatureDataset.metadata` records `window_index` but not raw
    sample positions, so center time must be reconstructed from the known
    windowing config rather than read directly."""
    step_seconds = window_seconds * (1.0 - overlap_ratio)
    start_seconds = metadata["window_index"].astype(float) * step_seconds
    return start_seconds + window_seconds / 2.0


def _merge_by_center_seconds(
    left_combined: pd.DataFrame, right_combined: pd.DataFrame, right_feature_columns: Tuple[str, ...]
) -> pd.DataFrame:
    """`merge_asof`-based nearest-time join, grouped by subject/condition.

    Both inputs must already carry a `center_seconds` column (added by the
    caller using each modality's own sampling rate) and be sorted by it.
    """
    left_sorted = left_combined.sort_values("center_seconds").reset_index(drop=True)
    right_sorted = right_combined.sort_values("center_seconds").reset_index(drop=True)

    merged = pd.merge_asof(
        left_sorted,
        right_sorted[["subject_id", "condition_id", "center_seconds", *right_feature_columns]],
        on="center_seconds",
        by=["subject_id", "condition_id"],
        direction="nearest",
    )
    return merged


def build_multimodal_dataset(
    emg_dataset: FeatureDataset,
    imu_dataset: FeatureDataset,
    ppg_dataset: FeatureDataset,
    config: DatasetConfig,
    emg_window_seconds: float = 1.0,
    emg_overlap_ratio: float = 0.5,
    imu_window_seconds: float = 1.0,
    imu_overlap_ratio: float = 0.5,
    ppg_window_seconds: float = 10.0,
    ppg_overlap_ratio: float = 0.5,
) -> MultimodalDataset:
    """Build the aligned (X, y, groups) multimodal dataset for the fusion model.

    Args:
        emg_dataset, imu_dataset, ppg_dataset: Per-modality `FeatureDataset`s
            from `ml.feature_engineering.feature_dataset.build_feature_datasets`
            (must all come from the same set of subjects).
        config: `DatasetConfig` used only for its `target_type` and target
            derivation parameters (`feature_subset` is not applicable here --
            the fusion model always uses every feature, per-modality feature
            selection is a later-phase concern).
        emg/imu/ppg_window_seconds, *_overlap_ratio: The `WindowConfig`
            each modality's `FeaturePipeline` used to produce its windows
            (Phase 8 defaults: EMG/IMU 1.0s/50%, PPG 10.0s/50%), needed to
            reconstruct each row's window center time from its
            `window_index` for nearest-time alignment.

    Returns:
        A `MultimodalDataset` ready for the proposed model.
    """
    emg_matrix, emg_meta = combine_sensors_per_window(emg_dataset)
    imu_matrix, imu_meta = combine_sensors_per_window(imu_dataset)
    ppg_matrix, ppg_meta = combine_sensors_per_window(ppg_dataset)

    emg_meta = emg_meta.assign(center_seconds=_window_center_seconds(emg_meta, emg_window_seconds, emg_overlap_ratio))
    imu_meta = imu_meta.assign(center_seconds=_window_center_seconds(imu_meta, imu_window_seconds, imu_overlap_ratio))
    if not ppg_meta.empty:
        ppg_meta = ppg_meta.assign(center_seconds=_window_center_seconds(ppg_meta, ppg_window_seconds, ppg_overlap_ratio))

    emg_feature_names = tuple(emg_matrix.columns)
    imu_feature_names = tuple(imu_matrix.columns)
    ppg_feature_names = tuple(ppg_matrix.columns)

    left_combined = pd.concat([emg_meta.reset_index(drop=True), emg_matrix.reset_index(drop=True)], axis=1)
    imu_combined = pd.concat([imu_meta.reset_index(drop=True), imu_matrix.reset_index(drop=True)], axis=1)

    merged = _merge_by_center_seconds(left_combined, imu_combined, imu_feature_names)

    if not ppg_meta.empty:
        ppg_combined = pd.concat([ppg_meta.reset_index(drop=True), ppg_matrix.reset_index(drop=True)], axis=1)
        merged = _merge_by_center_seconds(merged, ppg_combined, ppg_feature_names)
    else:
        for name in ppg_feature_names:
            merged[name] = np.nan
        logger.warning("PPG dataset is empty -- every PPG feature column will be NaN (imputed away downstream).")

    y_full, valid_mask, class_names = derive_target(merged, config)
    n_excluded = int((~valid_mask).sum())
    if n_excluded:
        logger.info(
            "Multimodal dataset: excluding %d/%d row(s) with no applicable '%s' target (e.g. MVIC trials).",
            n_excluded, len(y_full), config.target_type,
        )

    all_feature_names = (*emg_feature_names, *imu_feature_names, *ppg_feature_names)
    X_all_rows = merged[list(all_feature_names)].to_numpy(dtype=float)
    X = X_all_rows[valid_mask]
    y = y_full[valid_mask]
    groups = merged["subject_id"].to_numpy()[valid_mask]

    X_clean, n_imputed = impute_and_clean(X)
    if n_imputed:
        logger.info("Multimodal dataset: imputed %d NaN/infinite/extreme feature value(s) with column medians.", n_imputed)

    dims = ModalityDims(emg=len(emg_feature_names), imu=len(imu_feature_names), ppg=len(ppg_feature_names))

    return MultimodalDataset(
        X=X_clean,
        y=y,
        groups=groups,
        dims=dims,
        feature_names={"emg": emg_feature_names, "imu": imu_feature_names, "ppg": ppg_feature_names},
        class_names=class_names,
        dataset_config=config,
        n_rows_excluded=n_excluded,
    )
