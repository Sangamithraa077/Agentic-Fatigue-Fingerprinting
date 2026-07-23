"""Shared feature-extraction abstraction and generic numeric primitives.

`compute_*` functions here are low-level statistics reused by more than one
modality (e.g. RMS is a core EMG time-domain feature and also one of the
required IMU per-axis features) -- defined once to avoid duplicating the
same formula across `emg_features.py` and `imu_features.py`.

`FeatureExtractor` is the abstract base every modality-specific extractor
implements, mirroring `ml.preprocessing.signal_preprocessor.SignalPreprocessor`
from Phase 7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ml.feature_engineering.feature_vector import FeatureVector


def compute_mean(window: np.ndarray) -> float:
    """Arithmetic mean."""
    return float(np.mean(window)) if window.size else float("nan")


def compute_std(window: np.ndarray) -> float:
    """Standard deviation."""
    return float(np.std(window)) if window.size else float("nan")


def compute_variance(window: np.ndarray) -> float:
    """Variance."""
    return float(np.var(window)) if window.size else float("nan")


def compute_rms(window: np.ndarray) -> float:
    """Root mean square."""
    return float(np.sqrt(np.mean(np.square(window)))) if window.size else float("nan")


def compute_energy(window: np.ndarray) -> float:
    """Signal energy: sum of squared sample values."""
    return float(np.sum(np.square(window))) if window.size else float("nan")


def compute_max(window: np.ndarray) -> float:
    """Maximum value."""
    return float(np.max(window)) if window.size else float("nan")


def compute_min(window: np.ndarray) -> float:
    """Minimum value."""
    return float(np.min(window)) if window.size else float("nan")


def compute_range(window: np.ndarray) -> float:
    """Peak-to-peak range (max - min)."""
    return compute_max(window) - compute_min(window) if window.size else float("nan")


def compute_mean_absolute_value(window: np.ndarray) -> float:
    """Mean of the absolute sample values."""
    return float(np.mean(np.abs(window))) if window.size else float("nan")


class FeatureExtractor(ABC):
    """Base contract for a single-modality feature extractor.

    Subclasses compute a fixed, named set of numerical features from one
    windowed signal array and return a `FeatureVector` (see
    `feature_vector.py`) so feature names travel with their values.
    """

    @abstractmethod
    def extract(self, window: np.ndarray, sampling_rate_hz: float) -> "FeatureVector":
        """Compute this extractor's configured features for one window."""
