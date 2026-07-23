"""PPG feature extraction over one window of a (preprocessed) PPG signal.

Implemented features:

- **Peak Count** -- number of detected pulse peaks in the window.
- **Heart Rate (bpm)** -- 60 / mean inter-peak interval.
- **Pulse Interval Statistics** -- mean/std/min/max of the inter-peak
  intervals (seconds) within the window.
- **Signal Energy** -- sum of squared sample values.

Peak detection uses `scipy.signal.find_peaks` with a minimum-distance
constraint derived from a plausible maximum heart rate, so implausibly
close "peaks" (noise) are rejected without needing a PPG-specific systolic-
peak algorithm.

## Explicitly NOT implemented: full HRV metrics

Per this phase's requirement to document rather than force an unreliable
implementation: **RMSSD, pNN50, frequency-domain HRV (LF/HF power ratio),
and nonlinear/Poincare-based HRV measures are intentionally not
implemented.** Justification:

1. **Trial duration.** WSD4FEDSRM's PPG recordings range from ~20s to
   ~252s (mean ~72s per the Phase 6 EDA). Frequency-domain HRV
   conventionally requires >=2-5 minutes of stable recording for adequate
   spectral resolution in the LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz) bands
   -- most trials here are shorter than that requirement outright, and
   window-level segmentation (seconds to tens of seconds) makes this far
   worse.
2. **Peak-detection reliability.** Peaks here are found with a generic
   amplitude/distance-based detector on a signal that has only been
   band-pass filtered (Phase 7), not processed by a validated PPG-specific
   systolic-peak algorithm. Higher-order HRV metrics (RMSSD, pNN50) are
   highly sensitive to peak-detection jitter of a few milliseconds, which
   this detector is not validated to guarantee.

Given these two constraints, reporting RMSSD/pNN50/LF-HF from this pipeline
would imply a precision the underlying signal and detection method cannot
support. The simpler, more robust pulse-interval statistics and mean heart
rate implemented here are reported instead, and this limitation is
propagated into `FeatureExtractionSummary.skipped_features`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from ml.feature_engineering.feature_extractor import FeatureExtractor, compute_energy
from ml.feature_engineering.feature_vector import FeatureVector

logger = logging.getLogger(__name__)

PPG_FEATURE_NAMES: Tuple[str, ...] = (
    "peak_count",
    "heart_rate_bpm",
    "mean_pulse_interval_s",
    "std_pulse_interval_s",
    "min_pulse_interval_s",
    "max_pulse_interval_s",
    "signal_energy",
)

SKIPPED_HRV_FEATURES = {
    "rmssd": "Requires reliably jitter-free inter-beat intervals; not guaranteed by the generic peak detector used here.",
    "pnn50": "Same reliability requirement as RMSSD; not implemented for the same reason.",
    "lf_hf_ratio": (
        "Frequency-domain HRV requires >=2-5 minutes of stable recording for adequate spectral resolution; "
        "WSD4FEDSRM PPG trials (~20-252s, further reduced by windowing) do not meet this."
    ),
    "poincare_sd1_sd2": "Nonlinear HRV measures share the same duration and peak-detection-reliability limitations above.",
}


@dataclass(frozen=True)
class PPGFeatureConfig:
    """Configuration for `PPGFeatureExtractor`.

    Attributes:
        min_heart_rate_bpm: Lower plausible heart rate bound, used to cap
            the maximum accepted inter-peak interval (currently informational;
            peak spacing is primarily governed by `max_heart_rate_bpm`).
        max_heart_rate_bpm: Upper plausible heart rate bound, used to set
            the minimum allowed distance between detected peaks (higher
            bound -> smaller minimum distance).
        peak_prominence_ratio: Minimum peak prominence, as a fraction of
            the window's standard deviation, required for a peak to count
            -- suppresses small noise fluctuations from being detected as
            pulses.
    """

    min_heart_rate_bpm: float = 40.0
    max_heart_rate_bpm: float = 200.0
    peak_prominence_ratio: float = 0.3


def detect_peaks(
    window: np.ndarray, sampling_rate_hz: float, max_heart_rate_bpm: float = 200.0, peak_prominence_ratio: float = 0.3
) -> np.ndarray:
    """Detect pulse peaks in a PPG window.

    Args:
        window: 1D PPG signal.
        sampling_rate_hz: Sampling rate of `window`.
        max_heart_rate_bpm: Used to compute the minimum allowed sample
            distance between consecutive peaks.
        peak_prominence_ratio: Minimum peak prominence as a fraction of
            the window's standard deviation.

    Returns:
        Sample indices (into `window`) of detected peaks.
    """
    if window.size < 2:
        return np.array([], dtype=int)
    min_distance_seconds = 60.0 / max_heart_rate_bpm
    min_distance_samples = max(1, int(round(min_distance_seconds * sampling_rate_hz)))
    prominence = peak_prominence_ratio * float(np.std(window))
    peaks, _ = find_peaks(window, distance=min_distance_samples, prominence=prominence if prominence > 0 else None)
    return peaks


def pulse_intervals_seconds(peak_indices: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Convert peak sample indices to inter-peak intervals in seconds."""
    if peak_indices.size < 2:
        return np.array([])
    return np.diff(peak_indices) / sampling_rate_hz


class PPGFeatureExtractor(FeatureExtractor):
    """Computes peak-based pulse features for one PPG window.

    Heart rate and pulse-interval statistics are `nan` when fewer than 2
    peaks are detected in the window (too short/noisy to estimate an
    interval) -- this is logged, not silently defaulted to zero.
    """

    def __init__(self, config: Optional[PPGFeatureConfig] = None) -> None:
        self.config = config or PPGFeatureConfig()

    def extract(self, window: np.ndarray, sampling_rate_hz: float) -> FeatureVector:
        window = np.asarray(window, dtype=float).reshape(-1)

        peaks = detect_peaks(
            window, sampling_rate_hz, self.config.max_heart_rate_bpm, self.config.peak_prominence_ratio
        )
        intervals = pulse_intervals_seconds(peaks, sampling_rate_hz)

        if intervals.size == 0:
            logger.warning(
                "Fewer than 2 pulse peaks detected in a %d-sample PPG window -- "
                "heart rate and pulse interval statistics are unavailable (nan).",
                window.size,
            )
            heart_rate = mean_interval = std_interval = min_interval = max_interval = float("nan")
        else:
            mean_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals))
            min_interval = float(np.min(intervals))
            max_interval = float(np.max(intervals))
            heart_rate = 60.0 / mean_interval if mean_interval > 0 else float("nan")

        values = (
            float(len(peaks)),
            heart_rate,
            mean_interval,
            std_interval,
            min_interval,
            max_interval,
            compute_energy(window),
        )
        return FeatureVector(names=PPG_FEATURE_NAMES, values=values)
