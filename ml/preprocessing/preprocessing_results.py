"""Result types produced by `ml.preprocessing.pipeline.PreprocessingPipeline`.

These mirror the raw `ml.data_models` structure (Condition -> Subject) but
hold `ProcessedSignal` outputs in place of raw sensor DataFrames, so the raw
object graph built by the Phase 5 loader is never touched -- preprocessing
always produces a new, parallel set of objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ml.data_models.enums import ConditionType, Direction, EMGMuscle, IMULocation, IntensityBand
from ml.data_models.labels import Labels
from ml.preprocessing.signal_preprocessor import ProcessedSignal


@dataclass(frozen=True)
class ProcessedEMGData:
    """Preprocessed EMG recording for one muscle."""

    muscle: EMGMuscle
    signal: ProcessedSignal


@dataclass(frozen=True)
class ProcessedIMUData:
    """Preprocessed IMU recording for one body location.

    Attributes:
        stream_consistent: True if the accelerometer, gyroscope, and
            magnetometer streams had matching sample counts after
            processing (see `imu_preprocessor.check_stream_consistency`).
    """

    location: IMULocation
    accelerometer: ProcessedSignal
    gyroscope: ProcessedSignal
    magnetometer: ProcessedSignal
    stream_consistent: bool


@dataclass(frozen=True)
class ProcessedPPGData:
    """Preprocessed PPG recording."""

    signal: ProcessedSignal


@dataclass(frozen=True)
class ProcessedCondition:
    """Preprocessed counterpart of `ml.data_models.condition.Condition`.

    `labels` is passed through unchanged (a reference to the same `Labels`
    instance) since Borg RPE labels are not a signal and this phase does
    not modify them.
    """

    condition_id: str
    subject_id: str
    condition_type: ConditionType
    direction: Direction
    intensity_band: Optional[IntensityBand]
    trial_number: Optional[int]
    emg: Dict[EMGMuscle, ProcessedEMGData] = field(default_factory=dict)
    imu: Dict[IMULocation, ProcessedIMUData] = field(default_factory=dict)
    ppg: Optional[ProcessedPPGData] = None
    labels: Optional[Labels] = None


@dataclass(frozen=True)
class ProcessedSubject:
    """Preprocessed counterpart of `ml.data_models.subject.Subject`."""

    subject_id: str
    conditions: Dict[str, ProcessedCondition] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingSummary:
    """Aggregate summary of everything a `PreprocessingPipeline` instance
    has processed since it was created.

    Attributes:
        n_conditions_processed: Number of `Condition` objects processed.
        n_emg_signals_processed: Number of individual EMG channel signals
            processed (across all conditions).
        n_imu_signals_processed: Number of IMU *locations* processed (each
            covering 3 streams: accelerometer, gyroscope, magnetometer).
        n_ppg_signals_processed: Number of PPG signals processed.
        n_quality_failures_input: Number of signals whose *input* quality
            check failed (NaNs, infinities, empty, or invalid sampling rate).
        n_quality_failures_output: Number of signals whose *output* quality
            check failed after processing.
        issues: Every individual issue message encountered, in order.
    """

    n_conditions_processed: int
    n_emg_signals_processed: int
    n_imu_signals_processed: int
    n_ppg_signals_processed: int
    n_quality_failures_input: int
    n_quality_failures_output: int
    issues: List[str] = field(default_factory=list)

    def as_text(self) -> str:
        """Render this summary as a short human-readable report."""
        lines = [
            "Preprocessing Summary",
            f"Conditions processed: {self.n_conditions_processed}",
            f"EMG signals processed: {self.n_emg_signals_processed}",
            f"IMU locations processed: {self.n_imu_signals_processed}",
            f"PPG signals processed: {self.n_ppg_signals_processed}",
            f"Quality failures (input): {self.n_quality_failures_input}",
            f"Quality failures (output): {self.n_quality_failures_output}",
        ]
        if self.issues:
            lines.append(f"Issues ({len(self.issues)}):")
            lines.extend(f"  - {issue}" for issue in self.issues)
        else:
            lines.append("Issues: none")
        return "\n".join(lines)
