"""EMG feature extraction: time-domain and frequency-domain features
computed over one window of a (preprocessed) EMG channel.

Time-domain features (the classic Hudgins EMG feature set plus IEMG):

- **Mean Absolute Value (MAV)** -- average rectified amplitude; the
  simplest, most common estimator of muscle activation level.
- **Root Mean Square (RMS)** -- amplitude estimator proportional to
  contraction intensity and (approximately) motor unit recruitment.
- **Variance** -- signal power around the mean; sensitive to amplitude
  changes associated with fatigue and effort level.
- **Waveform Length (WL)** -- cumulative length of the waveform; reflects
  both amplitude and frequency content in a single value.
- **Zero Crossings (ZC)** -- count of sign changes; a simple frequency-
  content proxy.
- **Slope Sign Changes (SSC)** -- count of changes in the sign of the
  first derivative; another frequency-content proxy, sensitive to the
  number of turning points in the waveform.
- **Willison Amplitude (WAMP)** -- count of consecutive-sample amplitude
  differences exceeding a threshold; correlates with the firing of motor
  unit action potentials.
- **Integrated EMG (IEMG)** -- summed rectified amplitude; a cumulative
  activation measure over the window.

Frequency-domain features, derived from a Welch power-spectral-density
estimate:

- **Mean Frequency (MNF)**, **Median Frequency (MDF)**, **Peak Frequency**,
  and **Spectral Entropy** -- MNF/MDF are the standard EMG fatigue
  indicators (both shift lower as a muscle fatigues); peak frequency
  locates the dominant spectral component; spectral entropy summarizes how
  concentrated vs. spread out the power spectrum is.

**Important caveat on ZC/SSC/WAMP:** these three are classically computed
on a band-pass-*filtered but not rectified* EMG signal (they rely on the
signal crossing/changing sign around zero). If the upstream
`ml.preprocessing.emg_preprocessor.EMGPreprocessor` was run with its
default `rectify=True`, the input to this extractor is already non-negative
and these three features become degenerate (near-zero ZC/SSC counts, an
over-counted WAMP). `EMGFeatureExtractor` detects this case (via
`is_likely_rectified`) and logs a warning rather than silently returning
misleading numbers; the caveat is also recorded in the feature-extraction
report. To get meaningful ZC/SSC/WAMP, preprocess with `rectify=False`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import welch

from ml.feature_engineering.feature_extractor import (
    FeatureExtractor,
    compute_mean_absolute_value,
    compute_rms,
    compute_variance,
)
from ml.feature_engineering.feature_vector import FeatureVector

logger = logging.getLogger(__name__)

EMG_TIME_DOMAIN_FEATURE_NAMES: Tuple[str, ...] = (
    "mav",
    "rms",
    "variance",
    "waveform_length",
    "zero_crossings",
    "slope_sign_changes",
    "willison_amplitude",
    "integrated_emg",
)
EMG_FREQUENCY_DOMAIN_FEATURE_NAMES: Tuple[str, ...] = (
    "mean_frequency",
    "median_frequency",
    "peak_frequency",
    "spectral_entropy",
)


@dataclass(frozen=True)
class EMGFeatureConfig:
    """Configuration for `EMGFeatureExtractor`.

    Attributes:
        zc_threshold: Minimum amplitude change required around a zero
            crossing to count (suppresses noise-level crossings). 0.0
            reproduces the simplest textbook definition.
        ssc_threshold: Minimum amplitude change required around a slope
            sign change to count.
        wamp_threshold: Minimum consecutive-sample amplitude difference
            required to count for Willison Amplitude. If `None`, an
            adaptive threshold of `0.1 * std(window)` is used, since a
            fixed absolute default is not meaningful across differently
            scaled/preprocessed signals (raw filtered mV vs. a rectified
            envelope).
        welch_nperseg: `nperseg` for `scipy.signal.welch`. If `None`,
            uses `min(len(window), 256)`.
        compute_frequency_domain: Whether to compute MNF/MDF/peak
            frequency/spectral entropy in addition to the time-domain set.
    """

    zc_threshold: float = 0.0
    ssc_threshold: float = 0.0
    wamp_threshold: Optional[float] = None
    welch_nperseg: Optional[int] = None
    compute_frequency_domain: bool = True


def is_likely_rectified(window: np.ndarray) -> bool:
    """Heuristic: True if the window contains (almost) no negative values,
    suggesting it has already been full-wave rectified upstream."""
    if window.size == 0:
        return False
    negative_fraction = float(np.mean(window < -1e-9))
    return negative_fraction < 0.01


def waveform_length(window: np.ndarray) -> float:
    """Cumulative length of the waveform: sum of absolute first differences."""
    if window.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(window))))


def zero_crossings(window: np.ndarray, threshold: float = 0.0) -> int:
    """Count sign changes between consecutive samples, requiring the
    amplitude change across the crossing to exceed `threshold`."""
    if window.size < 2:
        return 0
    sign_change = (window[:-1] * window[1:]) < 0
    amplitude_ok = np.abs(np.diff(window)) >= threshold
    return int(np.sum(sign_change & amplitude_ok))


def slope_sign_changes(window: np.ndarray, threshold: float = 0.0) -> int:
    """Count changes in the sign of the first derivative, requiring the
    combined amplitude change to exceed `threshold`."""
    if window.size < 3:
        return 0
    diffs = np.diff(window)
    sign_change = (diffs[:-1] * diffs[1:]) < 0
    amplitude_ok = (np.abs(diffs[:-1]) + np.abs(diffs[1:])) >= threshold
    return int(np.sum(sign_change & amplitude_ok))


def willison_amplitude(window: np.ndarray, threshold: Optional[float] = None) -> int:
    """Count consecutive-sample absolute differences exceeding `threshold`.

    If `threshold` is None, uses an adaptive `0.1 * std(window)` (see
    `EMGFeatureConfig.wamp_threshold`).
    """
    if window.size < 2:
        return 0
    effective_threshold = threshold if threshold is not None else 0.1 * float(np.std(window))
    return int(np.sum(np.abs(np.diff(window)) > effective_threshold))


def integrated_emg(window: np.ndarray) -> float:
    """Sum of absolute sample values."""
    return float(np.sum(np.abs(window))) if window.size else 0.0


def power_spectrum(window: np.ndarray, sampling_rate_hz: float, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the power spectral density via Welch's method.

    Returns:
        (frequencies, power) arrays.
    """
    if window.size < 2:
        return np.array([]), np.array([])
    effective_nperseg = min(window.size, nperseg or 256)
    freqs, power = welch(window, fs=sampling_rate_hz, nperseg=effective_nperseg)
    return freqs, power


def mean_frequency(freqs: np.ndarray, power: np.ndarray) -> float:
    """Power-weighted average frequency."""
    total_power = np.sum(power)
    if freqs.size == 0 or total_power <= 0:
        return float("nan")
    return float(np.sum(freqs * power) / total_power)


def median_frequency(freqs: np.ndarray, power: np.ndarray) -> float:
    """Frequency at which cumulative power reaches half the total power."""
    total_power = np.sum(power)
    if freqs.size == 0 or total_power <= 0:
        return float("nan")
    cumulative = np.cumsum(power)
    idx = int(np.searchsorted(cumulative, total_power / 2.0))
    idx = min(idx, freqs.size - 1)
    return float(freqs[idx])


def peak_frequency(freqs: np.ndarray, power: np.ndarray) -> float:
    """Frequency of maximum spectral power."""
    if freqs.size == 0:
        return float("nan")
    return float(freqs[int(np.argmax(power))])


def spectral_entropy(power: np.ndarray) -> float:
    """Normalized Shannon entropy of the power spectrum, in [0, 1].

    0 indicates all power concentrated in a single frequency bin; 1
    indicates power spread uniformly across all bins.
    """
    total_power = np.sum(power)
    if power.size == 0 or total_power <= 0:
        return float("nan")
    probabilities = power / total_power
    probabilities = probabilities[probabilities > 0]
    raw_entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(power.size) if power.size > 1 else 1.0
    return float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0


class EMGFeatureExtractor(FeatureExtractor):
    """Computes the configured EMG time- and frequency-domain feature set
    for one window of one EMG channel."""

    def __init__(self, config: Optional[EMGFeatureConfig] = None) -> None:
        self.config = config or EMGFeatureConfig()
        self._has_warned_rectified = False

    @property
    def detected_rectified_input(self) -> bool:
        """True if this extractor has processed at least one window that
        appeared to already be rectified (see `is_likely_rectified`)."""
        return self._has_warned_rectified

    def extract(self, window: np.ndarray, sampling_rate_hz: float) -> FeatureVector:
        window = np.asarray(window, dtype=float).reshape(-1)

        if not self._has_warned_rectified and is_likely_rectified(window):
            logger.warning(
                "EMG window(s) appear to already be rectified (no negative values) -- "
                "zero_crossings/slope_sign_changes/willison_amplitude will be degenerate for "
                "all such windows processed by this extractor instance. "
                "See emg_features.py module docstring. (This warning is logged once per extractor instance.)"
            )
            self._has_warned_rectified = True

        names = list(EMG_TIME_DOMAIN_FEATURE_NAMES)
        values = [
            compute_mean_absolute_value(window),
            compute_rms(window),
            compute_variance(window),
            waveform_length(window),
            float(zero_crossings(window, self.config.zc_threshold)),
            float(slope_sign_changes(window, self.config.ssc_threshold)),
            float(willison_amplitude(window, self.config.wamp_threshold)),
            integrated_emg(window),
        ]

        if self.config.compute_frequency_domain:
            freqs, power = power_spectrum(window, sampling_rate_hz, self.config.welch_nperseg)
            names.extend(EMG_FREQUENCY_DOMAIN_FEATURE_NAMES)
            values.extend(
                [
                    mean_frequency(freqs, power),
                    median_frequency(freqs, power),
                    peak_frequency(freqs, power),
                    spectral_entropy(power),
                ]
            )

        return FeatureVector(names=tuple(names), values=tuple(values))
