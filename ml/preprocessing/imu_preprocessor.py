"""IMU-specific signal preprocessing.

Steps applied, in order, to each of the accelerometer/gyroscope/
magnetometer streams for one IMU location:

1. **Timestamp/sampling-rate validation** -- the raw WSD4FEDSRM files carry
   no timestamp column (samples are implicitly spaced at 1/sampling_rate;
   see the Phase 5 loader), so "timestamp validity" here means confirming
   the declared sampling rate is positive/finite and the implied recording
   duration is sane -- catching corrupt or truncated files early.
2. **Missing-value handling** -- linearly interpolates (or drops) NaN
   samples so later windowing/segmentation sees a continuous series;
   brief inertial-sensor dropouts are conventionally handled by
   interpolation rather than by leaving gaps.
3. **Optional smoothing** -- a Savitzky-Golay filter reduces high-frequency
   sensor noise while preserving the underlying motion trend, without the
   phase distortion a simple moving average would introduce.
4. **Sensor consistency checks** -- accelerometer, gyroscope, and
   magnetometer streams for one location are recorded in lockstep by the
   same physical device and must have matching sample counts; a mismatch
   indicates a corrupted or partially written file.

Every step is individually toggleable via `IMUPreprocessingConfig`, and each
transform is also exposed as a standalone, unit-testable function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import savgol_filter

from ml.preprocessing.quality_checks import check_length_consistency, check_sampling_rate, validate_signal
from ml.preprocessing.resampling import resample_signal
from ml.preprocessing.signal_preprocessor import ProcessedSignal, SignalPreprocessor

logger = logging.getLogger(__name__)

IMU_SAMPLING_RATE_HZ = 100.0


@dataclass(frozen=True)
class IMUPreprocessingConfig:
    """Configuration for `IMUPreprocessor`.

    Attributes:
        validate_timestamps: Check implied sample timing is sane (step 1).
        handle_missing: Fill or drop NaN samples (step 2).
        missing_value_strategy: `"interpolate"` (linear) or `"drop"`.
        apply_smoothing: Apply Savitzky-Golay smoothing (step 3).
        smoothing_window: Window length (samples) for smoothing; adjusted
            to the nearest valid odd value internally.
        smoothing_polyorder: Polynomial order for the Savitzky-Golay filter.
        target_sampling_rate_hz: If set, resample the final output to this
            rate. `None` by default -- no resampling is performed unless
            explicitly requested.
    """

    validate_timestamps: bool = True
    handle_missing: bool = True
    missing_value_strategy: str = "interpolate"
    apply_smoothing: bool = False
    smoothing_window: int = 5
    smoothing_polyorder: int = 2
    target_sampling_rate_hz: Optional[float] = None


def validate_timestamps(n_samples: int, sampling_rate_hz: float) -> bool:
    """Validate a recording's implied timing: positive sample count,
    positive/finite sampling rate, and a finite positive implied duration.

    Why: without an explicit timestamp column, this is the only available
    signal that the declared sampling rate and sample count are consistent
    (e.g. would catch a sampling rate of 0 or a truncated/corrupt file).
    """
    if n_samples <= 0:
        return False
    if not check_sampling_rate(sampling_rate_hz):
        return False
    duration = n_samples / sampling_rate_hz
    return bool(np.isfinite(duration) and duration > 0)


def handle_missing_values(signal_values: np.ndarray, strategy: str = "interpolate") -> np.ndarray:
    """Fill or drop NaN samples in a 1D or 2D (n_samples, n_channels) array.

    Why: IMU dropouts are typically brief; interpolation preserves signal
    continuity for downstream windowing without discarding surrounding
    valid samples, which is the conventional approach for inertial data.

    Args:
        signal_values: The signal to clean.
        strategy: `"interpolate"` (linear interpolation per channel) or
            `"drop"` (remove any row containing a NaN in any channel).

    Returns:
        A new array with missing values handled. Returned unchanged if
        there are no NaNs.
    """
    if signal_values.size == 0:
        return signal_values
    values = np.asarray(signal_values, dtype=float).copy()
    if not np.isnan(values).any():
        return values

    if strategy == "drop":
        mask = ~np.isnan(values).any(axis=-1) if values.ndim > 1 else ~np.isnan(values)
        return values[mask]

    if strategy == "interpolate":
        squeeze = values.ndim == 1
        if squeeze:
            values = values.reshape(-1, 1)
        index = np.arange(values.shape[0])
        for col in range(values.shape[1]):
            column = values[:, col]
            nan_mask = np.isnan(column)
            if not nan_mask.any() or nan_mask.all():
                continue
            values[nan_mask, col] = np.interp(index[nan_mask], index[~nan_mask], column[~nan_mask])
        return values.reshape(-1) if squeeze else values

    raise ValueError(f"Unknown missing-value strategy: {strategy!r}")


def smooth_signal(signal_values: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along the time axis.

    Why: reduces high-frequency sensor noise while preserving the shape of
    the underlying motion trend, unlike a plain moving average which can
    distort peaks and introduce lag.
    """
    if signal_values.size == 0:
        return signal_values
    n = signal_values.shape[0]
    win = window if window % 2 == 1 else window + 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win <= polyorder or win < 3:
        logger.warning("Smoothing window too small for signal length %d -- skipping smoothing.", n)
        return signal_values
    return savgol_filter(signal_values, window_length=win, polyorder=polyorder, axis=0)


def check_stream_consistency(stream_lengths: Dict[str, int]) -> bool:
    """Check that the accelerometer/gyroscope/magnetometer streams for one
    IMU location have matching sample counts (they are recorded in
    lockstep by the same physical device)."""
    return check_length_consistency(list(stream_lengths.values()))


class IMUPreprocessor(SignalPreprocessor):
    """Applies the configured IMU preprocessing chain to one 3-axis stream
    (accelerometer, gyroscope, or magnetometer) for one IMU location.

    Cross-stream consistency checking (step 4) spans all three streams for
    a location and is therefore performed by the caller (see
    `ml.preprocessing.pipeline.PreprocessingPipeline`), which calls
    `process` once per stream and then `check_stream_consistency` on the
    resulting lengths.
    """

    expected_sampling_rate_hz = IMU_SAMPLING_RATE_HZ

    def __init__(self, config: Optional[IMUPreprocessingConfig] = None) -> None:
        self.config = config or IMUPreprocessingConfig()

    def process(self, signal_values: np.ndarray, sampling_rate_hz: float) -> ProcessedSignal:
        input_quality = self._validate(signal_values, sampling_rate_hz)
        if input_quality.is_empty:
            logger.warning("IMU stream is empty -- returning as-is with no processing steps applied.")
            return ProcessedSignal(
                values=signal_values,
                sampling_rate_hz=sampling_rate_hz,
                steps_applied=(),
                input_quality=input_quality,
                output_quality=input_quality,
            )

        values = np.asarray(signal_values, dtype=float).copy()
        steps: List[str] = []
        rate = sampling_rate_hz

        if self.config.validate_timestamps and not validate_timestamps(values.shape[0], rate):
            logger.warning(
                "IMU stream failed timestamp/sampling-rate validation (n=%d, rate=%s Hz).", values.shape[0], rate
            )

        if self.config.handle_missing and np.isnan(values).any():
            values = handle_missing_values(values, self.config.missing_value_strategy)
            steps.append(f"handle_missing_values({self.config.missing_value_strategy})")

        if self.config.apply_smoothing:
            values = smooth_signal(values, self.config.smoothing_window, self.config.smoothing_polyorder)
            steps.append(f"smooth(window={self.config.smoothing_window})")

        if self.config.target_sampling_rate_hz and self.config.target_sampling_rate_hz != rate:
            values = resample_signal(values, rate, self.config.target_sampling_rate_hz)
            rate = self.config.target_sampling_rate_hz
            steps.append(f"resample(to={rate}Hz)")

        output_quality = validate_signal(values, sampling_rate_hz=rate)
        logger.info("IMU preprocessing applied: %s", steps)
        return ProcessedSignal(
            values=values,
            sampling_rate_hz=rate,
            steps_applied=tuple(steps),
            input_quality=input_quality,
            output_quality=output_quality,
        )
