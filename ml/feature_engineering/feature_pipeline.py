"""Top-level feature-engineering pipeline.

Accepts the `Processed*` objects produced by Phase 7
(`ml.preprocessing.preprocessing_results`) and produces windowed,
named `FeatureVector`s for every EMG channel, IMU location, and PPG
signal, via `windowing.segment_signal` + the modality-specific extractors.

This module performs orchestration and windowing only -- all feature
formulas live in `emg_features.py` / `imu_features.py` / `ppg_features.py`.
No model training, explainability, or cross-modality fusion happens here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ml.feature_engineering.emg_features import (
    EMG_FREQUENCY_DOMAIN_FEATURE_NAMES,
    EMG_TIME_DOMAIN_FEATURE_NAMES,
    EMGFeatureConfig,
    EMGFeatureExtractor,
)
from ml.feature_engineering.feature_vector import (
    ConditionFeatures,
    EMGChannelFeatures,
    FeatureExtractionSummary,
    IMULocationFeatures,
    PPGFeatures,
    SubjectFeatures,
    WindowedFeatures,
)
from ml.feature_engineering.imu_features import IMUFeatureConfig, IMUFeatureExtractor, imu_feature_names
from ml.feature_engineering.ppg_features import PPG_FEATURE_NAMES, SKIPPED_HRV_FEATURES, PPGFeatureConfig, PPGFeatureExtractor
from ml.feature_engineering.windowing import WindowConfig, compute_step_samples, compute_window_samples, segment_signal
from ml.preprocessing.preprocessing_results import ProcessedCondition, ProcessedSubject

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeaturePipelineConfig:
    """Bundles per-modality feature/window configuration.

    Window sizes are configured independently per modality because the
    physiologically appropriate window length differs substantially:
    EMG/IMU activity is well characterized by short (~0.5-1s) windows,
    while PPG heart-rate/pulse-interval features need enough window length
    to contain several heartbeats (default 10s, comfortably covering
    several beats even at a resting ~50 bpm).
    """

    emg: EMGFeatureConfig = field(default_factory=EMGFeatureConfig)
    imu: IMUFeatureConfig = field(default_factory=IMUFeatureConfig)
    ppg: PPGFeatureConfig = field(default_factory=PPGFeatureConfig)
    emg_window: WindowConfig = field(default_factory=lambda: WindowConfig(window_seconds=1.0, overlap_ratio=0.5))
    imu_window: WindowConfig = field(default_factory=lambda: WindowConfig(window_seconds=1.0, overlap_ratio=0.5))
    ppg_window: WindowConfig = field(default_factory=lambda: WindowConfig(window_seconds=10.0, overlap_ratio=0.5))


class FeaturePipeline:
    """Applies configurable sliding-window feature extraction across a
    `ProcessedCondition` or `ProcessedSubject`.

    Example:
        >>> pipeline = FeaturePipeline()
        >>> condition_features = pipeline.extract_condition(processed_condition)
        >>> subject_features = pipeline.extract_subject(processed_subject)
        >>> print(pipeline.summary.as_text())
    """

    def __init__(self, config: Optional[FeaturePipelineConfig] = None) -> None:
        self.config = config or FeaturePipelineConfig()
        self._emg_extractor = EMGFeatureExtractor(self.config.emg)
        self._imu_extractor = IMUFeatureExtractor(self.config.imu)
        self._ppg_extractor = PPGFeatureExtractor(self.config.ppg)

        self._n_conditions = 0
        self._n_emg_windows = 0
        self._n_imu_windows = 0
        self._n_ppg_windows = 0
        self._warnings: List[str] = []

    def extract(self, target: Union[ProcessedSubject, ProcessedCondition]) -> Union[SubjectFeatures, ConditionFeatures]:
        """Dispatch to `extract_subject` or `extract_condition` by type."""
        if isinstance(target, ProcessedSubject):
            return self.extract_subject(target)
        if isinstance(target, ProcessedCondition):
            return self.extract_condition(target)
        raise TypeError(f"Expected a ProcessedSubject or ProcessedCondition, got {type(target)!r}")

    def extract_subject(self, subject: ProcessedSubject) -> SubjectFeatures:
        """Extract features for every condition belonging to one subject."""
        logger.info("Extracting features for subject '%s' (%d conditions)", subject.subject_id, len(subject.conditions))
        conditions = {}
        for condition_id, condition in subject.conditions.items():
            try:
                conditions[condition_id] = self.extract_condition(condition)
            except Exception:
                logger.exception(
                    "Failed to extract features for condition '%s' in subject '%s' -- skipping",
                    condition_id,
                    subject.subject_id,
                )
        return SubjectFeatures(subject_id=subject.subject_id, conditions=conditions)

    def extract_condition(self, condition: ProcessedCondition) -> ConditionFeatures:
        """Extract features for every sensor recording within one condition."""
        self._n_conditions += 1

        emg_features = {}
        for muscle, emg_data in condition.emg.items():
            try:
                emg_features[muscle] = self._extract_emg_channel(emg_data)
            except Exception:
                logger.exception(
                    "Failed to extract EMG features for muscle '%s' in condition '%s' -- skipping",
                    muscle.value,
                    condition.condition_id,
                )

        imu_features = {}
        for location, imu_data in condition.imu.items():
            try:
                imu_features[location] = self._extract_imu_location(imu_data)
            except Exception:
                logger.exception(
                    "Failed to extract IMU features for location '%s' in condition '%s' -- skipping",
                    location.value,
                    condition.condition_id,
                )

        ppg_features = None
        if condition.ppg is not None:
            try:
                ppg_features = self._extract_ppg(condition.ppg)
            except Exception:
                logger.exception("Failed to extract PPG features in condition '%s' -- skipping", condition.condition_id)

        return ConditionFeatures(
            condition_id=condition.condition_id,
            subject_id=condition.subject_id,
            condition_type=condition.condition_type,
            direction=condition.direction,
            intensity_band=condition.intensity_band,
            trial_number=condition.trial_number,
            emg=emg_features,
            imu=imu_features,
            ppg=ppg_features,
        )

    def _extract_emg_channel(self, emg_data) -> EMGChannelFeatures:
        signal = emg_data.signal
        window_samples = compute_window_samples(self.config.emg_window.window_seconds, signal.sampling_rate_hz)
        step_samples = compute_step_samples(window_samples, self.config.emg_window.overlap_ratio)
        windows = segment_signal(
            signal.values, window_samples, step_samples, self.config.emg_window.drop_last_incomplete
        )

        windowed_results = []
        for index, window in enumerate(windows):
            start = index * step_samples
            features = self._emg_extractor.extract(window, signal.sampling_rate_hz)
            windowed_results.append(
                WindowedFeatures(
                    window_index=index, window_start_sample=start, window_end_sample=start + len(window), features=features
                )
            )
        self._n_emg_windows += len(windowed_results)
        return EMGChannelFeatures(muscle=emg_data.muscle, windows=windowed_results)

    def _extract_imu_location(self, imu_data) -> IMULocationFeatures:
        rate = imu_data.accelerometer.sampling_rate_hz
        window_samples = compute_window_samples(self.config.imu_window.window_seconds, rate)
        step_samples = compute_step_samples(window_samples, self.config.imu_window.overlap_ratio)

        acc_windows = segment_signal(imu_data.accelerometer.values, window_samples, step_samples, self.config.imu_window.drop_last_incomplete)
        gyr_windows = segment_signal(imu_data.gyroscope.values, window_samples, step_samples, self.config.imu_window.drop_last_incomplete)
        mag_windows = segment_signal(imu_data.magnetometer.values, window_samples, step_samples, self.config.imu_window.drop_last_incomplete)

        n_windows = min(len(acc_windows), len(gyr_windows), len(mag_windows))
        if len({len(acc_windows), len(gyr_windows), len(mag_windows)}) > 1:
            logger.warning(
                "IMU stream window-count mismatch for location '%s' (acc=%d, gyr=%d, mag=%d) -- using the shortest.",
                imu_data.location.value, len(acc_windows), len(gyr_windows), len(mag_windows),
            )

        windowed_results = []
        for index in range(n_windows):
            start = index * step_samples
            features = self._imu_extractor.extract_location(acc_windows[index], gyr_windows[index], mag_windows[index])
            windowed_results.append(
                WindowedFeatures(
                    window_index=index,
                    window_start_sample=start,
                    window_end_sample=start + len(acc_windows[index]),
                    features=features,
                )
            )
        self._n_imu_windows += len(windowed_results)
        return IMULocationFeatures(location=imu_data.location, windows=windowed_results)

    def _extract_ppg(self, ppg_data) -> PPGFeatures:
        signal = ppg_data.signal
        window_samples = compute_window_samples(self.config.ppg_window.window_seconds, signal.sampling_rate_hz)
        step_samples = compute_step_samples(window_samples, self.config.ppg_window.overlap_ratio)
        windows = segment_signal(
            signal.values, window_samples, step_samples, self.config.ppg_window.drop_last_incomplete
        )

        windowed_results = []
        for index, window in enumerate(windows):
            start = index * step_samples
            features = self._ppg_extractor.extract(window, signal.sampling_rate_hz)
            windowed_results.append(
                WindowedFeatures(
                    window_index=index, window_start_sample=start, window_end_sample=start + len(window), features=features
                )
            )
        self._n_ppg_windows += len(windowed_results)
        if not windowed_results:
            self._warnings.append(
                f"No PPG windows produced (signal shorter than the configured {self.config.ppg_window.window_seconds}s window)."
            )
        return PPGFeatures(windows=windowed_results)

    @property
    def summary(self) -> FeatureExtractionSummary:
        """Aggregate summary of everything extracted by this pipeline
        instance so far, including the feature catalog and documented
        skipped features (see `ppg_features.SKIPPED_HRV_FEATURES`)."""
        feature_names_by_category = {
            "emg_time_domain": EMG_TIME_DOMAIN_FEATURE_NAMES,
            "emg_frequency_domain": EMG_FREQUENCY_DOMAIN_FEATURE_NAMES if self.config.emg.compute_frequency_domain else (),
            "imu": imu_feature_names(),
            "ppg": PPG_FEATURE_NAMES,
        }
        warnings = list(self._warnings)
        if self._emg_extractor.detected_rectified_input:
            warnings.append(
                "One or more EMG windows appeared to already be rectified (non-negative) input -- "
                "zero_crossings/slope_sign_changes/willison_amplitude are degenerate for those windows. "
                "See emg_features.py module docstring; set EMGPreprocessingConfig(rectify=False) upstream "
                "in Phase 7 to get meaningful values for these three features."
            )
        return FeatureExtractionSummary(
            n_conditions_processed=self._n_conditions,
            n_emg_windows=self._n_emg_windows,
            n_imu_windows=self._n_imu_windows,
            n_ppg_windows=self._n_ppg_windows,
            feature_names_by_category=feature_names_by_category,
            skipped_features=dict(SKIPPED_HRV_FEATURES),
            warnings=warnings,
        )
