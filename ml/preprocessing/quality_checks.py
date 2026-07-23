"""Signal quality validation utilities.

These are small, pure functions operating on plain NumPy arrays -- no
dependency on the dataset object model -- so they are directly unit
testable in isolation and reusable by every modality-specific preprocessor.

Every preprocessing step in this phase runs its input and output through
`validate_signal` so that data quality problems (NaNs, infinities, empty
signals, invalid sampling rates) are surfaced explicitly rather than
silently propagating into downstream phases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalQualityReport:
    """Result of running the standard validation battery on one signal.

    Attributes:
        n_samples: Number of samples in the signal.
        n_nan: Count of NaN entries.
        n_infinite: Count of +/-inf entries.
        is_empty: True if the signal has zero samples.
        sampling_rate_hz: The sampling rate that was checked.
        expected_sampling_rate_hz: The expected rate it was checked against,
            if any.
        sampling_rate_valid: True if the sampling rate is positive, finite,
            and (when an expectation was given) within tolerance of it.
        passed: True if no issues were found at all.
        issues: Human-readable description of every problem found.
    """

    n_samples: int
    n_nan: int
    n_infinite: int
    is_empty: bool
    sampling_rate_hz: Optional[float]
    expected_sampling_rate_hz: Optional[float]
    sampling_rate_valid: bool
    passed: bool
    issues: List[str] = field(default_factory=list)


def count_nan(signal_values: np.ndarray) -> int:
    """Count NaN entries in a signal array."""
    if signal_values.size == 0:
        return 0
    return int(np.isnan(signal_values).sum())


def count_infinite(signal_values: np.ndarray) -> int:
    """Count infinite (+inf or -inf) entries in a signal array."""
    if signal_values.size == 0:
        return 0
    return int(np.isinf(signal_values).sum())


def is_empty_signal(signal_values: np.ndarray) -> bool:
    """True if the signal has zero samples."""
    return signal_values.size == 0


def check_sampling_rate(
    actual_hz: Optional[float], expected_hz: Optional[float] = None, tolerance_hz: float = 1e-3
) -> bool:
    """Validate a sampling rate is positive and finite, and optionally
    matches an expected value within `tolerance_hz`."""
    if actual_hz is None or not np.isfinite(actual_hz) or actual_hz <= 0:
        return False
    if expected_hz is not None and abs(actual_hz - expected_hz) > tolerance_hz:
        return False
    return True


def check_length_consistency(lengths: Sequence[int]) -> bool:
    """True if every length in `lengths` is equal.

    Used to check that streams recorded in lockstep by the same device
    (e.g. an IMU's accelerometer/gyroscope/magnetometer, or the 6 EMG
    channels within one trial) have matching sample counts.
    """
    if not lengths:
        return True
    return len(set(lengths)) == 1


def validate_signal(
    signal_values: np.ndarray,
    sampling_rate_hz: Optional[float],
    expected_sampling_rate_hz: Optional[float] = None,
) -> SignalQualityReport:
    """Run the full standard validation battery on a single signal array.

    Checks (in order): emptiness, NaNs, infinities, sampling-rate validity.
    None of these checks modify `signal_values` -- this function is purely
    diagnostic.

    Args:
        signal_values: 1D or 2D (n_samples, n_channels) array.
        sampling_rate_hz: The signal's sampling rate.
        expected_sampling_rate_hz: If given, the sampling rate is also
            checked against this expected value.

    Returns:
        A `SignalQualityReport` summarizing every issue found (empty list
        of issues if the signal passed every check).
    """
    issues: List[str] = []
    n_samples = int(signal_values.shape[0]) if signal_values.ndim else int(signal_values.size)
    empty = is_empty_signal(signal_values)
    if empty:
        issues.append("Signal is empty (zero samples).")

    n_nan = count_nan(signal_values) if not empty else 0
    if n_nan:
        issues.append(f"Signal contains {n_nan} NaN sample(s).")

    n_inf = count_infinite(signal_values) if not empty else 0
    if n_inf:
        issues.append(f"Signal contains {n_inf} infinite sample(s).")

    rate_valid = check_sampling_rate(sampling_rate_hz, expected_sampling_rate_hz)
    if not rate_valid:
        detail = f" (expected ~{expected_sampling_rate_hz} Hz)" if expected_sampling_rate_hz else ""
        issues.append(f"Sampling rate {sampling_rate_hz} Hz is invalid{detail}.")

    passed = not issues
    if not passed:
        logger.warning("Signal quality check failed: %s", "; ".join(issues))

    return SignalQualityReport(
        n_samples=n_samples,
        n_nan=n_nan,
        n_infinite=n_inf,
        is_empty=empty,
        sampling_rate_hz=sampling_rate_hz,
        expected_sampling_rate_hz=expected_sampling_rate_hz,
        sampling_rate_valid=rate_valid,
        passed=passed,
        issues=issues,
    )
