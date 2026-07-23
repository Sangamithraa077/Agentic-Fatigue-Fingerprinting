"""Reusable single-signal resampling utilities.

These exist so an individual modality's preprocessor can optionally change
its own working sample rate (e.g. downsampling a smoothed EMG envelope to
reduce storage). They are opt-in (off by default in every preprocessing
config) and operate on exactly one signal at a time.

Explicitly out of scope here: aligning multiple modalities onto a shared
timebase for fusion. That is a Phase 8 (multimodal fusion) concern -- using
these utilities to resample EMG/IMU/PPG onto one common rate as a step
towards combining them is intentionally not done anywhere in this phase.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


def compute_resampled_length(n_samples: int, original_rate_hz: float, target_rate_hz: float) -> int:
    """Compute the number of samples `n_samples` at `original_rate_hz`
    corresponds to at `target_rate_hz`."""
    if original_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("Sampling rates must be positive.")
    return max(1, round(n_samples * target_rate_hz / original_rate_hz))


def resample_signal(signal_values: np.ndarray, original_rate_hz: float, target_rate_hz: float) -> np.ndarray:
    """Resample a 1D or 2D (n_samples, n_channels) signal to a new rate.

    Uses FFT-based resampling (`scipy.signal.resample`), appropriate for
    offline, non-causal resampling of an already-captured recording.

    Args:
        signal_values: The signal to resample, time axis first.
        original_rate_hz: The signal's current sampling rate.
        target_rate_hz: The desired sampling rate.

    Returns:
        The resampled signal. Returned unchanged if the signal is empty or
        the rates are already equal.
    """
    if signal_values.size == 0:
        return signal_values
    if original_rate_hz == target_rate_hz:
        return signal_values

    target_length = compute_resampled_length(signal_values.shape[0], original_rate_hz, target_rate_hz)
    try:
        return scipy_signal.resample(signal_values, target_length, axis=0)
    except ValueError as exc:
        logger.warning("Resampling failed (%s) -- returning original signal.", exc)
        return signal_values


def align_signal_length(signal_values: np.ndarray, target_length: int, pad_value: float = np.nan) -> np.ndarray:
    """Truncate or pad a signal along its time axis to exactly `target_length`
    samples.

    Padding uses `pad_value` (default NaN, so padded regions are explicit
    and visible to later quality checks rather than silently looking like
    real zero-valued data).
    """
    current_length = signal_values.shape[0]
    if current_length == target_length:
        return signal_values
    if current_length > target_length:
        return signal_values[:target_length]

    pad_shape = (target_length - current_length,) + signal_values.shape[1:]
    pad_block = np.full(pad_shape, pad_value, dtype=float)
    return np.concatenate([signal_values, pad_block], axis=0)
