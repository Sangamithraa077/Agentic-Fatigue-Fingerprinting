"""Assembles the complete feature matrix (with full metadata) from Phase 8
`FeaturePipeline` output, ready for validation and selection.

## Why one matrix per modality, not one fused matrix

EMG, IMU, and PPG windows are independently timed: each modality's
`FeaturePipeline` window/overlap is configured separately (see
`feature_pipeline.FeaturePipelineConfig`) because the physiologically
appropriate window length differs (EMG/IMU ~1s, PPG ~10s). Their windows
therefore do not align 1:1 in time or count, so stacking their features
into one row-per-sample matrix would require a resampling/alignment
decision -- exactly the cross-modality fusion step Phase 7 and Phase 8
both explicitly deferred. This module preserves that boundary: it builds
one independent `FeatureDataset` per modality (EMG, IMU, PPG), each with
its own rows (one per window) and its own feature columns. Fusion remains
a decision for a later phase.

## Metadata and the `rpe_nearest` target

Each row carries full identifying metadata (subject, condition, source
sensor, window index, ...) plus `rpe_nearest`: the Borg RPE reading whose
timestamp is closest to that window's center time, taken from the
originating `Condition.labels.rpe_timeseries` (via the corresponding
`ProcessedCondition`, since Phase 8's `ConditionFeatures` does not itself
carry labels). This is `NaN` for MVIC-calibration windows (no Borg data
was collected during MVIC, per Phase 5/6/7) and is intended purely as an
optional exploratory target column for Phase 8.5's importance diagnostics
-- it is a nearest-neighbor lookup against a sparse (every-10-seconds)
reading, not an interpolated or validated ground-truth label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml.data_models.labels import Labels
from ml.feature_engineering.emg_features import EMG_FREQUENCY_DOMAIN_FEATURE_NAMES
from ml.feature_engineering.feature_vector import ConditionFeatures, SubjectFeatures, WindowedFeatures
from ml.preprocessing.emg_preprocessor import EMG_SAMPLING_RATE_HZ
from ml.preprocessing.imu_preprocessor import IMU_SAMPLING_RATE_HZ
from ml.preprocessing.ppg_preprocessor import PPG_SAMPLING_RATE_HZ
from ml.preprocessing.preprocessing_results import ProcessedCondition, ProcessedSubject

logger = logging.getLogger(__name__)

METADATA_COLUMNS: Tuple[str, ...] = (
    "subject_id",
    "condition_id",
    "condition_type",
    "direction",
    "intensity_band",
    "trial_number",
    "source_sensor",
    "window_index",
    "rpe_nearest",
)


@dataclass(frozen=True)
class FeatureDataset:
    """One modality's complete feature matrix, with parallel metadata.

    Attributes:
        modality: "EMG", "IMU", or "PPG".
        feature_names: The feature columns present in `matrix` (derived
            from the data itself, so it reflects whatever `FeaturePipeline`
            configuration actually produced -- e.g. fewer EMG columns if
            frequency-domain features were disabled).
        matrix: One row per window, one column per feature. Row order
            matches `metadata` exactly (same index).
        metadata: One row per window (aligned with `matrix`), with columns
            `METADATA_COLUMNS`.
    """

    modality: str
    feature_names: Tuple[str, ...]
    matrix: pd.DataFrame
    metadata: pd.DataFrame


def window_center_seconds(windowed: WindowedFeatures, sampling_rate_hz: float) -> float:
    """Compute a window's center time, in seconds, within its source signal."""
    center_sample = (windowed.window_start_sample + windowed.window_end_sample) / 2.0
    return center_sample / sampling_rate_hz


def nearest_rpe(labels: Optional[Labels], time_seconds: float) -> float:
    """Find the Borg RPE reading nearest in time to `time_seconds`.

    Combines `rpe_before` (t=0), every `rpe_timeseries` entry, and
    `rpe_at_end_of_trial` (at `trial_duration_seconds`, if known) into one
    sorted set of (time, RPE) points, then returns the value of whichever
    point is closest to `time_seconds`.

    Returns:
        The nearest RPE value, or NaN if `labels` is None or has no usable
        RPE readings at all (e.g. an MVIC-calibration condition).
    """
    if labels is None:
        return float("nan")

    points: List[Tuple[float, int]] = []
    if labels.rpe_before is not None:
        points.append((0.0, labels.rpe_before))
    points.extend((float(t), v) for t, v in labels.rpe_timeseries.items())
    if labels.rpe_at_end_of_trial is not None and labels.trial_duration_seconds is not None:
        points.append((labels.trial_duration_seconds, labels.rpe_at_end_of_trial))

    if not points:
        return float("nan")

    times = np.array([p[0] for p in points])
    values = np.array([p[1] for p in points], dtype=float)
    nearest_index = int(np.argmin(np.abs(times - time_seconds)))
    return float(values[nearest_index])


def _metadata_row(
    condition_features: ConditionFeatures,
    source_sensor: str,
    windowed: WindowedFeatures,
    labels: Optional[Labels],
    sampling_rate_hz: float,
) -> dict:
    center_seconds = window_center_seconds(windowed, sampling_rate_hz)
    return {
        "subject_id": condition_features.subject_id,
        "condition_id": condition_features.condition_id,
        "condition_type": condition_features.condition_type.value,
        "direction": condition_features.direction.value,
        "intensity_band": condition_features.intensity_band.value if condition_features.intensity_band else None,
        "trial_number": condition_features.trial_number,
        "source_sensor": source_sensor,
        "window_index": windowed.window_index,
        "rpe_nearest": nearest_rpe(labels, center_seconds),
    }


def _condition_rows(condition_features: ConditionFeatures, processed_condition: Optional[ProcessedCondition]) -> Tuple[List[dict], List[dict], List[dict]]:
    labels = processed_condition.labels if processed_condition is not None else None

    emg_rows = []
    for muscle, channel_features in condition_features.emg.items():
        for windowed in channel_features.windows:
            row = _metadata_row(condition_features, muscle.value, windowed, labels, EMG_SAMPLING_RATE_HZ)
            row.update(windowed.features.as_dict())
            emg_rows.append(row)

    imu_rows = []
    for location, location_features in condition_features.imu.items():
        for windowed in location_features.windows:
            row = _metadata_row(condition_features, location.value, windowed, labels, IMU_SAMPLING_RATE_HZ)
            row.update(windowed.features.as_dict())
            imu_rows.append(row)

    ppg_rows = []
    if condition_features.ppg is not None:
        for windowed in condition_features.ppg.windows:
            row = _metadata_row(condition_features, "ppg", windowed, labels, PPG_SAMPLING_RATE_HZ)
            row.update(windowed.features.as_dict())
            ppg_rows.append(row)

    return emg_rows, imu_rows, ppg_rows


def _rows_to_dataset(modality: str, rows: List[dict]) -> FeatureDataset:
    if not rows:
        logger.warning("No %s windows available -- returning an empty FeatureDataset.", modality)
        return FeatureDataset(modality=modality, feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame(columns=METADATA_COLUMNS))

    df = pd.DataFrame(rows)
    feature_names = tuple(c for c in df.columns if c not in METADATA_COLUMNS)
    metadata = df[list(METADATA_COLUMNS)].reset_index(drop=True)
    matrix = df[list(feature_names)].reset_index(drop=True)
    return FeatureDataset(modality=modality, feature_names=feature_names, matrix=matrix, metadata=metadata)


def build_feature_datasets(
    subjects_features: Sequence[SubjectFeatures], processed_subjects: Sequence[ProcessedSubject]
) -> Dict[str, FeatureDataset]:
    """Build the complete per-modality feature matrices across all given subjects.

    Args:
        subjects_features: `SubjectFeatures` from `FeaturePipeline.extract_subject`.
        processed_subjects: The corresponding `ProcessedSubject` objects
            (Phase 7 output) used only to look up each condition's Borg
            labels for `rpe_nearest`. Matched to `subjects_features` by
            `subject_id`, not by list position.

    Returns:
        `{"EMG": FeatureDataset, "IMU": FeatureDataset, "PPG": FeatureDataset}`.
    """
    processed_by_id = {ps.subject_id: ps for ps in processed_subjects}

    all_emg_rows: List[dict] = []
    all_imu_rows: List[dict] = []
    all_ppg_rows: List[dict] = []

    for subject_features in subjects_features:
        processed_subject = processed_by_id.get(subject_features.subject_id)
        if processed_subject is None:
            logger.warning(
                "No matching ProcessedSubject for '%s' -- rpe_nearest will be NaN for all its windows.",
                subject_features.subject_id,
            )

        for condition_id, condition_features in subject_features.conditions.items():
            processed_condition = processed_subject.conditions.get(condition_id) if processed_subject else None
            emg_rows, imu_rows, ppg_rows = _condition_rows(condition_features, processed_condition)
            all_emg_rows.extend(emg_rows)
            all_imu_rows.extend(imu_rows)
            all_ppg_rows.extend(ppg_rows)

    return {
        "EMG": _rows_to_dataset("EMG", all_emg_rows),
        "IMU": _rows_to_dataset("IMU", all_imu_rows),
        "PPG": _rows_to_dataset("PPG", all_ppg_rows),
    }


def categorize_feature(modality: str, feature_name: str) -> str:
    """Assign a human-meaningful category to a feature name, for the catalog."""
    if modality == "EMG":
        return "frequency_domain" if feature_name in EMG_FREQUENCY_DOMAIN_FEATURE_NAMES else "time_domain"
    if modality == "IMU":
        return "resultant" if "resultant" in feature_name else "per_axis"
    if modality == "PPG":
        return "amplitude" if feature_name == "signal_energy" else "peak_based"
    return "unknown"


def build_feature_catalog(feature_datasets: Dict[str, FeatureDataset]) -> pd.DataFrame:
    """Build a one-row-per-feature catalog: feature_name, modality, category.

    This is the "feature name" + "modality" + "feature category" metadata
    requested for interpretability; it is deliberately kept separate from
    the per-window `metadata` in each `FeatureDataset` (which carries
    `source_sensor`/`window_index`), since one feature *name* (e.g. "mav")
    applies uniformly across many source sensors (e.g. all 6 EMG muscles).
    """
    rows = []
    for modality, dataset in feature_datasets.items():
        for name in dataset.feature_names:
            rows.append({"feature_name": name, "modality": modality, "category": categorize_feature(modality, name)})
    return pd.DataFrame(rows)
