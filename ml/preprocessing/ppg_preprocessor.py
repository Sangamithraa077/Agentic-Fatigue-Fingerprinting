"""PPG-specific signal preprocessing.

Steps applied, in order:

1. **Baseline drift removal** -- a high-pass Butterworth filter (default
   0.5 Hz cutoff) removes slow drift caused by motion, respiration, and
   finger pressure changes, which otherwise dominates the raw PPG trace
   and obscures the pulsatile cardiac waveform.
2. **Optional smoothing** -- a low-pass filter (default 8 Hz cutoff, above
   the highest physiologically relevant pulse-waveform harmonic) reduces
   high-frequency sensor noise while preserving pulse shape.

Every step is individually toggleable via `PPGPreprocessingConfig`, and each
transform is also exposed as a standalone, unit-testable function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import butter, filtfilt

from ml.preprocessing.quality_checks import validate_signal
from ml.preprocessing.resampling import resample_signal
from ml.preprocessing.signal_preprocessor import ProcessedSignal, SignalPreprocessor

logger = logging.getLogger(__name__)

PPG_SAMPLING_RATE_HZ = 200.0


@dataclass(frozen=True)
class PPGPreprocessingConfig:
    """Configuration for `PPGPreprocessor`.

    Attributes:
        remove_baseline_drift: Apply the high-pass drift-removal filter
            (step 1).
        drift_highpass_hz: High-pass cutoff frequency (Hz) for drift
            removal.
        drift_filter_order: Butterworth filter order for drift removal.
        apply_smoothing: Apply the low-pass smoothing filter (step 2).
        smoothing_lowpass_hz: Low-pass cutoff frequency (Hz) for smoothing.
        smoothing_order: Butterworth filter order for smoothing.
        target_sampling_rate_hz: If set, resample the final output to this
            rate. `None` by default -- no resampling is performed unless
            explicitly requested.
    """

    remove_baseline_drift: bool = True
    drift_highpass_hz: float = 0.5
    drift_filter_order: int = 2
    apply_smoothing: bool = True
    smoothing_lowpass_hz: float = 8.0
    smoothing_order: int = 4
    target_sampling_rate_hz: Optional[float] = None


def remove_baseline_drift(
    signal_values: np.ndarray, sampling_rate_hz: float, cutoff_hz: float, order: int = 2
) -> np.ndarray:
    """Zero-phase high-pass Butterworth filter removing slow baseline drift.

    Why: motion, respiration, and finger-pressure changes introduce slow
    trends that dominate the raw PPG amplitude and obscure the much
    smaller, faster pulsatile cardiac component of interest.
    """
    if signal_values.size == 0:
        return signal_values
    nyquist = sampling_rate_hz / 2.0
    cutoff = min(cutoff_hz, nyquist * 0.99)
    b, a = butter(order, cutoff / nyquist, btype="high")
    try:
        return filtfilt(b, a, signal_values, axis=0)
    except ValueError as exc:
        logger.warning("Baseline drift removal failed (%s) -- returning original signal.", exc)
        return signal_values


def smooth_ppg(signal_values: np.ndarray, sampling_rate_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass Butterworth filter for PPG noise reduction.

    Why: attenuates high-frequency sensor/electronic noise while preserving
    the shape of the pulse waveform, whose relevant energy sits well below
    this cutoff.
    """
    if signal_values.size == 0:
        return signal_values
    nyquist = sampling_rate_hz / 2.0
    cutoff = min(cutoff_hz, nyquist * 0.99)
    b, a = butter(order, cutoff / nyquist, btype="low")
    try:
        return filtfilt(b, a, signal_values, axis=0)
    except ValueError as exc:
        logger.warning("PPG smoothing failed (%s) -- returning unsmoothed signal.", exc)
        return signal_values


class PPGPreprocessor(SignalPreprocessor):
    """Applies the configured PPG preprocessing chain to one PPG recording."""

    expected_sampling_rate_hz = PPG_SAMPLING_RATE_HZ

    def __init__(self, config: Optional[PPGPreprocessingConfig] = None) -> None:
        self.config = config or PPGPreprocessingConfig()

    def process(self, signal_values: np.ndarray, sampling_rate_hz: float) -> ProcessedSignal:
        input_quality = self._validate(signal_values, sampling_rate_hz)
        if input_quality.is_empty:
            logger.warning("PPG signal is empty -- returning as-is with no processing steps applied.")
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

        if self.config.remove_baseline_drift:
            values = remove_baseline_drift(values, rate, self.config.drift_highpass_hz, self.config.drift_filter_order)
            steps.append(f"remove_baseline_drift(highpass={self.config.drift_highpass_hz}Hz)")

        if self.config.apply_smoothing:
            values = smooth_ppg(values, rate, self.config.smoothing_lowpass_hz, self.config.smoothing_order)
            steps.append(f"smooth(lowpass={self.config.smoothing_lowpass_hz}Hz)")

        if self.config.target_sampling_rate_hz and self.config.target_sampling_rate_hz != rate:
            values = resample_signal(values, rate, self.config.target_sampling_rate_hz)
            rate = self.config.target_sampling_rate_hz
            steps.append(f"resample(to={rate}Hz)")

        output_quality = validate_signal(values, sampling_rate_hz=rate)
        logger.info("PPG preprocessing applied: %s", steps)
        return ProcessedSignal(
            values=values,
            sampling_rate_hz=rate,
            steps_applied=tuple(steps),
            input_quality=input_quality,
            output_quality=output_quality,
        )
