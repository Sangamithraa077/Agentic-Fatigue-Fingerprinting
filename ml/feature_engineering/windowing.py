"""Configurable sliding-window segmentation.

Windows are configured in seconds rather than raw sample counts so that the
same `WindowConfig` intent ("1-second windows, 50% overlap") can be applied
consistently to signals recorded at different sampling rates (EMG 1000 Hz,
IMU 100 Hz, PPG 200 Hz) -- each modality converts the same time-based
configuration to its own sample count via `compute_window_samples`.

These are small, pure functions operating on plain NumPy arrays / integers,
so they are directly unit testable in isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window segmentation configuration.

    Attributes:
        window_seconds: Window length in seconds.
        overlap_ratio: Fraction of a window that overlaps with the next
            window, in [0, 1). 0.5 means consecutive windows share half
            their samples (a step of half the window length).
        drop_last_incomplete: If True, a final window shorter than the
            configured length is discarded rather than zero-padded, so
            every returned window has an identical, known sample count.
    """

    window_seconds: float = 1.0
    overlap_ratio: float = 0.5
    drop_last_incomplete: bool = True

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {self.window_seconds}")
        if not (0.0 <= self.overlap_ratio < 1.0):
            raise ValueError(f"overlap_ratio must be in [0, 1), got {self.overlap_ratio}")


def compute_window_samples(window_seconds: float, sampling_rate_hz: float) -> int:
    """Convert a window length in seconds to a sample count at the given rate."""
    if sampling_rate_hz <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {sampling_rate_hz}")
    return max(1, round(window_seconds * sampling_rate_hz))


def compute_step_samples(window_samples: int, overlap_ratio: float) -> int:
    """Convert a window length and overlap ratio to a step size in samples."""
    step = round(window_samples * (1.0 - overlap_ratio))
    return max(1, step)


def generate_window_indices(
    n_samples: int, window_samples: int, step_samples: int, drop_last_incomplete: bool = True
) -> List[Tuple[int, int]]:
    """Compute (start, end) sample-index pairs for every window.

    Args:
        n_samples: Total number of samples in the signal being segmented.
        window_samples: Number of samples per window.
        step_samples: Number of samples to advance between window starts.
        drop_last_incomplete: If True, omit a final window that would run
            past `n_samples`.

    Returns:
        A list of (start, end) index pairs, usable as `signal[start:end]`.
        Empty if the signal is shorter than one window and
        `drop_last_incomplete` is True.
    """
    if n_samples <= 0 or window_samples <= 0 or step_samples <= 0:
        return []

    indices: List[Tuple[int, int]] = []
    start = 0
    while start < n_samples:
        end = start + window_samples
        if end > n_samples:
            if drop_last_incomplete:
                break
            end = n_samples
        indices.append((start, end))
        if end >= n_samples:
            break
        start += step_samples
    return indices


def segment_signal(
    signal_values: np.ndarray, window_samples: int, step_samples: int, drop_last_incomplete: bool = True
) -> List[np.ndarray]:
    """Segment a 1D or 2D (n_samples, n_channels) signal into overlapping windows.

    Args:
        signal_values: Signal to segment, time axis first.
        window_samples: Number of samples per window.
        step_samples: Number of samples to advance between window starts.
        drop_last_incomplete: If True, omit a final undersized window.

    Returns:
        A list of array slices, one per window (views into `signal_values`,
        not copies).
    """
    n_samples = signal_values.shape[0]
    indices = generate_window_indices(n_samples, window_samples, step_samples, drop_last_incomplete)
    if not indices:
        logger.warning(
            "No windows produced: signal has %d samples, window requires %d.", n_samples, window_samples
        )
    return [signal_values[start:end] for start, end in indices]
