"""IMU feature extraction: per-axis and resultant-vector features computed
over one window of one (preprocessed) IMU stream (accelerometer, gyroscope,
or magnetometer -- each a 3-column x/y/z array).

Per-axis features (computed independently for x, y, and z):

- **Mean**, **Std**, **RMS**, **Max**, **Min**, **Range** -- standard
  descriptive statistics of the axis signal over the window.
- **Energy** -- sum of squared samples; reflects overall movement
  intensity on that axis.
- **Signal Magnitude Area (per-axis)** -- mean absolute value of the axis,
  i.e. that axis's own contribution to the classical tri-axial SMA (see
  below). Reported per-axis for completeness, but the conventional,
  physically meaningful SMA is the *combined* tri-axial quantity computed
  by `compute_resultant_features`.

Resultant-vector features, computed on the per-sample vector magnitude
`r = sqrt(x^2 + y^2 + z^2)`:

- **Resultant Mean/Std/RMS/Max/Min/Energy** -- the same descriptive
  statistics applied to the orientation-independent magnitude signal,
  useful because it is unaffected by sensor orientation.
- **Resultant Signal Magnitude Area** -- `mean(|x| + |y| + |z|)`, the
  standard tri-axial SMA used in activity-intensity literature.

Feature names are prefixed per axis/resultant (e.g. `x_mean`, `resultant_sma`)
and, at the `IMUFeatureExtractor` level, further prefixed by stream
(`acc_`, `gyr_`, `mag_`) when combined into one window's flat feature set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ml.feature_engineering.feature_extractor import (
    compute_energy,
    compute_max,
    compute_mean,
    compute_mean_absolute_value,
    compute_min,
    compute_range,
    compute_rms,
    compute_std,
)
from ml.feature_engineering.feature_vector import FeatureVector

IMU_AXIS_FEATURE_NAMES: Tuple[str, ...] = ("mean", "std", "rms", "max", "min", "range", "energy", "sma")
IMU_RESULTANT_FEATURE_NAMES: Tuple[str, ...] = (
    "resultant_mean",
    "resultant_std",
    "resultant_rms",
    "resultant_max",
    "resultant_min",
    "resultant_energy",
    "resultant_sma",
)
_AXIS_LABELS: Tuple[str, ...] = ("x", "y", "z")
_STREAM_PREFIXES: Tuple[str, ...] = ("acc", "gyr", "mag")


@dataclass(frozen=True)
class IMUFeatureConfig:
    """Configuration for `IMUFeatureExtractor`. No tunable parameters are
    currently needed -- present for interface symmetry with the other
    modalities and to hold future options without changing call sites."""


def compute_axis_features(axis_values: np.ndarray) -> Dict[str, float]:
    """Compute the per-axis feature set for one 1D axis array."""
    return {
        "mean": compute_mean(axis_values),
        "std": compute_std(axis_values),
        "rms": compute_rms(axis_values),
        "max": compute_max(axis_values),
        "min": compute_min(axis_values),
        "range": compute_range(axis_values),
        "energy": compute_energy(axis_values),
        "sma": compute_mean_absolute_value(axis_values),
    }


def compute_resultant_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """Compute resultant-vector (tri-axial magnitude) features.

    Args:
        x, y, z: The three axis arrays for one window (equal length).
    """
    magnitude = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    resultant_sma = float(np.mean(np.abs(x) + np.abs(y) + np.abs(z))) if x.size else float("nan")
    return {
        "resultant_mean": compute_mean(magnitude),
        "resultant_std": compute_std(magnitude),
        "resultant_rms": compute_rms(magnitude),
        "resultant_max": compute_max(magnitude),
        "resultant_min": compute_min(magnitude),
        "resultant_energy": compute_energy(magnitude),
        "resultant_sma": resultant_sma,
    }


def extract_stream_features(window: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute per-axis + resultant features for one 3-axis window, with
    every feature name prefixed by `prefix` (e.g. "acc_x_mean",
    "acc_resultant_sma").

    Args:
        window: (n_samples, 3) array, columns ordered (x, y, z).
        prefix: Stream identifier, e.g. "acc", "gyr", "mag".
    """
    window = np.asarray(window, dtype=float)
    features: Dict[str, float] = {}

    for axis_index, axis_label in enumerate(_AXIS_LABELS):
        axis_values = window[:, axis_index] if window.ndim == 2 else window
        for name, value in compute_axis_features(axis_values).items():
            features[f"{prefix}_{axis_label}_{name}"] = value

    if window.ndim == 2 and window.shape[1] == 3:
        resultant = compute_resultant_features(window[:, 0], window[:, 1], window[:, 2])
        for name, value in resultant.items():
            features[f"{prefix}_{name}"] = value

    return features


class IMUFeatureExtractor:
    """Computes the combined per-axis + resultant feature set for all three
    streams (accelerometer, gyroscope, magnetometer) of one IMU location.

    Unlike `EMGFeatureExtractor`/`PPGFeatureExtractor`, this does not
    implement the single-array `FeatureExtractor.extract` contract, since
    one IMU location's feature set spans three 3-axis streams rather than
    one 1D signal -- the same deliberate deviation Phase 7's
    `IMUPreprocessor` made from the single-stream `SignalPreprocessor`
    contract, for the same reason.
    """

    def __init__(self, config: IMUFeatureConfig = None) -> None:
        self.config = config or IMUFeatureConfig()

    def extract_location(
        self, accelerometer: np.ndarray, gyroscope: np.ndarray, magnetometer: np.ndarray
    ) -> FeatureVector:
        """Compute the full feature set for one IMU location's window.

        Args:
            accelerometer, gyroscope, magnetometer: (n_samples, 3) windows
                for the same time span, same location.
        """
        features: Dict[str, float] = {}
        for stream_prefix, stream_window in zip(_STREAM_PREFIXES, (accelerometer, gyroscope, magnetometer)):
            features.update(extract_stream_features(stream_window, stream_prefix))

        return FeatureVector(names=tuple(features.keys()), values=tuple(features.values()))


def imu_feature_names() -> Tuple[str, ...]:
    """Every feature name `IMUFeatureExtractor` can produce, for reporting."""
    names = []
    for stream_prefix in _STREAM_PREFIXES:
        for axis_label in _AXIS_LABELS:
            names.extend(f"{stream_prefix}_{axis_label}_{name}" for name in IMU_AXIS_FEATURE_NAMES)
        names.extend(f"{stream_prefix}_{name}" for name in IMU_RESULTANT_FEATURE_NAMES)
    return tuple(names)
