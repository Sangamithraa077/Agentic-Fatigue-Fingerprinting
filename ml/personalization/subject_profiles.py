"""`SubjectProfile`: one participant's complete personalization profile,
assembled from data Phases 5 (raw `Subject` demographics/anthropometrics)
and 8.5 (`FeatureDataset` baselines/fatigue history) already produced.

This module only *assembles*; all the underlying statistics are computed by
`baseline_builder.py` (kept separate per single-responsibility -- baseline
computation is reusable on its own, e.g. directly by
`personalization_model.py`, without needing a full profile).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ml.data_models.anthropometry import AnthropometricData
from ml.data_models.demographics import DemographicData
from ml.data_models.subject import Subject
from ml.feature_engineering.feature_dataset import FeatureDataset
from ml.personalization.baseline_builder import FatigueHistory, ModalityBaseline, build_fatigue_history, build_modality_baseline
from ml.personalization.config import SubjectProfileConfig

SUMMARY_FEATURE_NAMES: Tuple[str, ...] = (
    "age", "height_cm", "mass_kg", "bmi", "body_fat_percent", "muscle_percent",
    "mean_rpe", "emg_baseline_mean", "imu_baseline_mean", "ppg_baseline_mean",
)


@dataclass(frozen=True)
class SubjectProfile:
    """One participant's complete personalization profile.

    Attributes:
        subject_id: Canonical identifier, e.g. `"P01"`.
        demographic: Phase 5's `DemographicData`, or `None` if unavailable.
        anthropometric: Phase 5's `AnthropometricData`, or `None` if
            unavailable.
        emg_baseline, imu_baseline, ppg_baseline: This subject's own
            baseline feature statistics per modality (see `baseline_builder`).
        fatigue_history: This subject's own Borg RPE history summary.
        summary_vector: A fixed-length, fixed-order numeric summary
            (`SUMMARY_FEATURE_NAMES`) combining demographics, anthropometrics,
            fatigue history, and each modality's overall baseline level --
            the representation `similarity.find_similar_subjects` compares
            across subjects. Missing inputs (e.g. no demographic record, no
            RPE readings) are filled with `0.0` rather than dropped, so
            every profile's vector has the same shape.
        summary_feature_names: Column labels for `summary_vector`, in order.
    """

    subject_id: str
    demographic: Optional[DemographicData]
    anthropometric: Optional[AnthropometricData]
    emg_baseline: ModalityBaseline
    imu_baseline: ModalityBaseline
    ppg_baseline: ModalityBaseline
    fatigue_history: FatigueHistory
    summary_vector: np.ndarray
    summary_feature_names: Tuple[str, ...] = SUMMARY_FEATURE_NAMES


def _mean_of_means(baseline: ModalityBaseline) -> float:
    """A single scalar summarizing a modality baseline's overall level,
    for `summary_vector` -- the per-feature detail stays in `feature_means`
    for callers (e.g. `personalization_model`) that need it."""
    if not baseline.feature_means:
        return 0.0
    return float(np.mean(list(baseline.feature_means.values())))


def build_subject_profile(
    subject: Subject, feature_datasets: Dict[str, FeatureDataset], config: Optional[SubjectProfileConfig] = None
) -> SubjectProfile:
    """Assemble `subject`'s complete `SubjectProfile`.

    Args:
        subject: A Phase 5 `Subject` (for demographics/anthropometrics).
        feature_datasets: `{"EMG": FeatureDataset, "IMU": ..., "PPG": ...}`,
            e.g. `ml.feature_engineering.feature_dataset.build_feature_datasets`'s
            output -- reused unmodified, not rebuilt.
        config: Profile-building configuration; defaults applied if omitted.
    """
    config = config or SubjectProfileConfig()

    emg_baseline = build_modality_baseline(feature_datasets["EMG"], subject.subject_id)
    imu_baseline = build_modality_baseline(feature_datasets["IMU"], subject.subject_id)
    ppg_baseline = build_modality_baseline(feature_datasets["PPG"], subject.subject_id)
    fatigue_history = build_fatigue_history(feature_datasets["EMG"], subject.subject_id, config)

    demographic = subject.demographic
    anthropometric = subject.anthropometric

    summary_vector = np.array(
        [
            float(demographic.age) if demographic else 0.0,
            float(demographic.height_cm) if demographic else 0.0,
            float(anthropometric.mass_kg) if anthropometric else 0.0,
            float(anthropometric.bmi) if anthropometric else 0.0,
            float(anthropometric.body_fat_percent) if anthropometric else 0.0,
            float(anthropometric.muscle_percent) if anthropometric else 0.0,
            fatigue_history.mean_rpe if fatigue_history.mean_rpe is not None else 0.0,
            _mean_of_means(emg_baseline),
            _mean_of_means(imu_baseline),
            _mean_of_means(ppg_baseline),
        ],
        dtype=float,
    )

    return SubjectProfile(
        subject_id=subject.subject_id,
        demographic=demographic,
        anthropometric=anthropometric,
        emg_baseline=emg_baseline,
        imu_baseline=imu_baseline,
        ppg_baseline=ppg_baseline,
        fatigue_history=fatigue_history,
        summary_vector=summary_vector,
    )
