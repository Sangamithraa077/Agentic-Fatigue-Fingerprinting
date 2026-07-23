"""EMG-specific signal preprocessing.

Standard surface EMG preprocessing chain, applied in order:

1. **DC offset removal** -- surface EMG electrodes and amplifiers introduce
   a slowly varying baseline offset from the electrode-skin interface;
   subtracting the mean recenters the signal around zero before filtering.
2. **Band-pass filtering** (default 20-450 Hz) -- rejects low-frequency
   motion artifact/baseline wander below ~20 Hz and high-frequency noise
   near the Nyquist limit. 20-450 Hz is the standard SENIAM-recommended
   bandwidth for surface EMG at a 1000 Hz sampling rate.
3. **Full-wave rectification** -- raw EMG oscillates around zero; taking
   the absolute value is required before any amplitude/envelope analysis,
   producing a unipolar signal proportional to instantaneous muscle
   activation magnitude.
4. **Optional linear envelope** -- low-pass filtering the rectified signal
   (default 6 Hz cutoff) smooths it into a slowly varying envelope that
   approximates the underlying muscle activation/force profile -- the
   conventional "linear envelope" technique in EMG analysis.

Every step is individually toggleable via `EMGPreprocessingConfig`, and each
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

EMG_SAMPLING_RATE_HZ = 1000.0


@dataclass(frozen=True)
class EMGPreprocessingConfig:
    """Configuration for `EMGPreprocessor`.

    Attributes:
        remove_dc_offset: Subtract the signal mean (step 1).
        apply_bandpass: Apply the band-pass filter (step 2).
        bandpass_low_hz: Band-pass low cutoff frequency (Hz).
        bandpass_high_hz: Band-pass high cutoff frequency (Hz).
        bandpass_order: Butterworth filter order for the band-pass filter.
        rectify: Apply full-wave rectification (step 3).
        compute_envelope: Apply low-pass envelope smoothing to the
            rectified signal (step 4). Ignored with a warning if
            `rectify=False`.
        envelope_lowpass_hz: Envelope low-pass cutoff frequency (Hz).
        envelope_order: Butterworth filter order for the envelope filter.
        target_sampling_rate_hz: If set, resample the final output to this
            rate. `None` by default -- no resampling is performed unless
            explicitly requested.
    """

    remove_dc_offset: bool = True
    apply_bandpass: bool = True
    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 450.0
    bandpass_order: int = 4
    rectify: bool = True
    compute_envelope: bool = False
    envelope_lowpass_hz: float = 6.0
    envelope_order: int = 4
    target_sampling_rate_hz: Optional[float] = None


def remove_dc_offset(signal_values: np.ndarray) -> np.ndarray:
    """Subtract the mean to remove DC offset/baseline bias.

    Why: surface EMG amplifiers and the electrode-skin interface introduce
    a near-constant baseline voltage unrelated to muscle activity; removing
    it centers the signal at zero, which subsequent filtering and
    rectification assume.
    """
    if signal_values.size == 0:
        return signal_values
    return signal_values - np.nanmean(signal_values)


def bandpass_filter(
    signal_values: np.ndarray,
    sampling_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (via `filtfilt`).

    Why: rejects sub-20Hz motion artifact/baseline wander and noise near
    the Nyquist frequency, isolating the frequency band that carries
    genuine muscle electrical activity.

    The high cutoff is clamped just below the Nyquist frequency so the
    filter design remains valid regardless of the input sampling rate.
    """
    if signal_values.size == 0:
        return signal_values
    nyquist = sampling_rate_hz / 2.0
    high = min(high_hz, nyquist * 0.99)
    low = max(low_hz, 1e-6)
    if low >= high:
        logger.warning(
            "Band-pass range [%s, %s] Hz invalid for sampling rate %s Hz -- skipping filter.",
            low_hz,
            high_hz,
            sampling_rate_hz,
        )
        return signal_values

    b, a = butter(order, [low / nyquist, high / nyquist], btype="bandpass")
    try:
        return filtfilt(b, a, signal_values, axis=0)
    except ValueError as exc:
        logger.warning("Band-pass filtering failed (%s) -- returning unfiltered signal.", exc)
        return signal_values


def rectify(signal_values: np.ndarray) -> np.ndarray:
    """Full-wave rectification (absolute value).

    Why: raw EMG is a zero-mean oscillating signal; amplitude/envelope
    analysis requires a unipolar signal proportional to activation
    magnitude.
    """
    return np.abs(signal_values)


def lowpass_envelope(
    signal_values: np.ndarray,
    sampling_rate_hz: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase low-pass filter deriving the linear envelope from a
    rectified EMG signal.

    Why: smooths the rectified signal's high-frequency oscillation into a
    slowly varying trace that approximates the muscle's activation/force
    profile over time.

    Note: because this is a zero-phase (non-causal) filter applied to a
    non-negative rectified signal, small negative excursions (filter
    ringing) of a few thousandths of the signal's scale can appear at sharp
    transitions. This is an expected characteristic of linear envelope
    filtering, not a defect.
    """
    if signal_values.size == 0:
        return signal_values
    nyquist = sampling_rate_hz / 2.0
    cutoff = min(cutoff_hz, nyquist * 0.99)
    b, a = butter(order, cutoff / nyquist, btype="low")
    try:
        return filtfilt(b, a, signal_values, axis=0)
    except ValueError as exc:
        logger.warning("Envelope filtering failed (%s) -- returning rectified signal.", exc)
        return signal_values


class EMGPreprocessor(SignalPreprocessor):
    """Applies the configured EMG preprocessing chain to one EMG channel."""

    expected_sampling_rate_hz = EMG_SAMPLING_RATE_HZ

    def __init__(self, config: Optional[EMGPreprocessingConfig] = None) -> None:
        self.config = config or EMGPreprocessingConfig()

    def process(self, signal_values: np.ndarray, sampling_rate_hz: float) -> ProcessedSignal:
        input_quality = self._validate(signal_values, sampling_rate_hz)
        if input_quality.is_empty:
            logger.warning("EMG signal is empty -- returning as-is with no processing steps applied.")
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

        if self.config.remove_dc_offset:
            values = remove_dc_offset(values)
            steps.append("remove_dc_offset")

        if self.config.apply_bandpass:
            values = bandpass_filter(
                values, rate, self.config.bandpass_low_hz, self.config.bandpass_high_hz, self.config.bandpass_order
            )
            steps.append(f"bandpass_filter({self.config.bandpass_low_hz}-{self.config.bandpass_high_hz}Hz)")

        if self.config.rectify:
            values = rectify(values)
            steps.append("rectify")

            if self.config.compute_envelope:
                values = lowpass_envelope(values, rate, self.config.envelope_lowpass_hz, self.config.envelope_order)
                steps.append(f"envelope(lowpass={self.config.envelope_lowpass_hz}Hz)")
        elif self.config.compute_envelope:
            logger.warning("compute_envelope=True requires rectify=True -- skipping envelope step.")

        if self.config.target_sampling_rate_hz and self.config.target_sampling_rate_hz != rate:
            values = resample_signal(values, rate, self.config.target_sampling_rate_hz)
            rate = self.config.target_sampling_rate_hz
            steps.append(f"resample(to={rate}Hz)")

        output_quality = validate_signal(values, sampling_rate_hz=rate)
        logger.info("EMG preprocessing applied: %s", steps)
        return ProcessedSignal(
            values=values,
            sampling_rate_hz=rate,
            steps_applied=tuple(steps),
            input_quality=input_quality,
            output_quality=output_quality,
        )
