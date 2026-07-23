"""Top-level preprocessing pipeline.

Orchestrates the per-modality preprocessors (`EMGPreprocessor`,
`IMUPreprocessor`, `PPGPreprocessor`) over a `Condition` or `Subject` object
from `ml.data_models` and produces the corresponding `Processed*` result
objects (see `preprocessing_results.py`) plus an aggregate summary.

This module performs orchestration and cross-signal validation only -- all
actual signal transforms live in `emg_preprocessor.py` / `imu_preprocessor.py`
/ `ppg_preprocessor.py`. No feature extraction, segmentation, cross-modality
resampling/fusion, or model training happens anywhere in this phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ml.data_models.condition import Condition
from ml.data_models.sensors import EMGData, IMUData, PPGData
from ml.data_models.subject import Subject
from ml.preprocessing.emg_preprocessor import EMGPreprocessingConfig, EMGPreprocessor
from ml.preprocessing.imu_preprocessor import IMUPreprocessingConfig, IMUPreprocessor, check_stream_consistency
from ml.preprocessing.ppg_preprocessor import PPGPreprocessingConfig, PPGPreprocessor
from ml.preprocessing.preprocessing_results import (
    PreprocessingSummary,
    ProcessedCondition,
    ProcessedEMGData,
    ProcessedIMUData,
    ProcessedPPGData,
    ProcessedSubject,
)
from ml.preprocessing.quality_checks import check_length_consistency
from ml.preprocessing.signal_preprocessor import ProcessedSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingPipelineConfig:
    """Bundles the per-modality configuration used by `PreprocessingPipeline`."""

    emg: EMGPreprocessingConfig = field(default_factory=EMGPreprocessingConfig)
    imu: IMUPreprocessingConfig = field(default_factory=IMUPreprocessingConfig)
    ppg: PPGPreprocessingConfig = field(default_factory=PPGPreprocessingConfig)


class PreprocessingPipeline:
    """Applies modality-specific preprocessing across a `Condition` or `Subject`.

    Example:
        >>> pipeline = PreprocessingPipeline()
        >>> processed_condition = pipeline.process_condition(condition)
        >>> processed_subject = pipeline.process_subject(subject)
        >>> print(pipeline.summary.as_text())

    A single pipeline instance accumulates summary statistics across every
    call to `process`/`process_subject`/`process_condition`; create a new
    instance to start a fresh summary.
    """

    def __init__(self, config: Optional[PreprocessingPipelineConfig] = None) -> None:
        self.config = config or PreprocessingPipelineConfig()
        self._emg_preprocessor = EMGPreprocessor(self.config.emg)
        self._imu_preprocessor = IMUPreprocessor(self.config.imu)
        self._ppg_preprocessor = PPGPreprocessor(self.config.ppg)

        self._n_conditions = 0
        self._n_emg = 0
        self._n_imu = 0
        self._n_ppg = 0
        self._n_quality_failures_input = 0
        self._n_quality_failures_output = 0
        self._issues: List[str] = []

    def process(self, target: Union[Subject, Condition]) -> Union[ProcessedSubject, ProcessedCondition]:
        """Dispatch to `process_subject` or `process_condition` by type.

        Raises:
            TypeError: If `target` is neither a `Subject` nor a `Condition`.
        """
        if isinstance(target, Subject):
            return self.process_subject(target)
        if isinstance(target, Condition):
            return self.process_condition(target)
        raise TypeError(f"Expected a Subject or Condition, got {type(target)!r}")

    def process_subject(self, subject: Subject) -> ProcessedSubject:
        """Preprocess every condition belonging to one subject."""
        logger.info("Preprocessing subject '%s' (%d conditions)", subject.subject_id, len(subject.conditions))
        processed_conditions = {}
        for condition_id, condition in subject.conditions.items():
            try:
                processed_conditions[condition_id] = self.process_condition(condition)
            except Exception:
                logger.exception(
                    "Failed to preprocess condition '%s' for subject '%s' -- skipping",
                    condition_id,
                    subject.subject_id,
                )
        return ProcessedSubject(subject_id=subject.subject_id, conditions=processed_conditions)

    def process_condition(self, condition: Condition) -> ProcessedCondition:
        """Preprocess every sensor recording within one condition trial."""
        self._n_conditions += 1

        processed_emg = {}
        for muscle, emg in condition.emg.items():
            try:
                processed_emg[muscle] = self._process_emg(emg)
            except Exception:
                logger.exception(
                    "Failed to preprocess EMG channel '%s' in condition '%s' -- skipping",
                    muscle.value,
                    condition.condition_id,
                )
        self._check_condition_wide_consistency(
            condition.condition_id, "EMG", [data.signal.values.shape[0] for data in processed_emg.values()]
        )

        processed_imu = {}
        for location, imu in condition.imu.items():
            try:
                processed_imu[location] = self._process_imu(imu)
            except Exception:
                logger.exception(
                    "Failed to preprocess IMU location '%s' in condition '%s' -- skipping",
                    location.value,
                    condition.condition_id,
                )
        self._check_condition_wide_consistency(
            condition.condition_id, "IMU", [data.accelerometer.values.shape[0] for data in processed_imu.values()]
        )

        processed_ppg = None
        if condition.ppg is not None:
            try:
                processed_ppg = self._process_ppg(condition.ppg)
            except Exception:
                logger.exception("Failed to preprocess PPG in condition '%s' -- skipping", condition.condition_id)

        return ProcessedCondition(
            condition_id=condition.condition_id,
            subject_id=condition.subject_id,
            condition_type=condition.condition_type,
            direction=condition.direction,
            intensity_band=condition.intensity_band,
            trial_number=condition.trial_number,
            emg=processed_emg,
            imu=processed_imu,
            ppg=processed_ppg,
            labels=condition.labels,
        )

    def _process_emg(self, emg: EMGData) -> ProcessedEMGData:
        raw_values = emg.signal.to_numpy().reshape(-1)
        result = self._emg_preprocessor.process(raw_values, emg.sampling_rate_hz)
        self._n_emg += 1
        self._track_quality(result)
        return ProcessedEMGData(muscle=emg.muscle, signal=result)

    def _process_imu(self, imu: IMUData) -> ProcessedIMUData:
        acc_result = self._imu_preprocessor.process(imu.accelerometer.to_numpy(), imu.sampling_rate_hz)
        gyr_result = self._imu_preprocessor.process(imu.gyroscope.to_numpy(), imu.sampling_rate_hz)
        mag_result = self._imu_preprocessor.process(imu.magnetometer.to_numpy(), imu.sampling_rate_hz)
        self._n_imu += 1
        for result in (acc_result, gyr_result, mag_result):
            self._track_quality(result)

        consistent = check_stream_consistency(
            {
                "accelerometer": acc_result.values.shape[0],
                "gyroscope": gyr_result.values.shape[0],
                "magnetometer": mag_result.values.shape[0],
            }
        )
        if not consistent:
            issue = (
                f"IMU stream length mismatch for location '{imu.location.value}' "
                f"in condition '{imu.condition_id}'"
            )
            logger.warning(issue)
            self._issues.append(issue)

        return ProcessedIMUData(
            location=imu.location,
            accelerometer=acc_result,
            gyroscope=gyr_result,
            magnetometer=mag_result,
            stream_consistent=consistent,
        )

    def _process_ppg(self, ppg: PPGData) -> ProcessedPPGData:
        raw_values = ppg.signal.to_numpy().reshape(-1)
        result = self._ppg_preprocessor.process(raw_values, ppg.sampling_rate_hz)
        self._n_ppg += 1
        self._track_quality(result)
        return ProcessedPPGData(signal=result)

    def _track_quality(self, result: ProcessedSignal) -> None:
        if not result.input_quality.passed:
            self._n_quality_failures_input += 1
            self._issues.extend(result.input_quality.issues)
        if not result.output_quality.passed:
            self._n_quality_failures_output += 1
            self._issues.extend(result.output_quality.issues)

    def _check_condition_wide_consistency(self, condition_id: str, modality: str, lengths: List[int]) -> None:
        """Check that all channels/locations of one modality within a
        condition share the same sample count (they were recorded
        simultaneously and should remain aligned in length)."""
        if lengths and not check_length_consistency(lengths):
            issue = f"{modality} channel/location length mismatch within condition '{condition_id}': {lengths}"
            logger.warning(issue)
            self._issues.append(issue)

    @property
    def summary(self) -> PreprocessingSummary:
        """Aggregate summary of everything processed by this pipeline
        instance so far."""
        return PreprocessingSummary(
            n_conditions_processed=self._n_conditions,
            n_emg_signals_processed=self._n_emg,
            n_imu_signals_processed=self._n_imu,
            n_ppg_signals_processed=self._n_ppg,
            n_quality_failures_input=self._n_quality_failures_input,
            n_quality_failures_output=self._n_quality_failures_output,
            issues=list(self._issues),
        )
