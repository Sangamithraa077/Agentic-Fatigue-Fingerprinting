"""Sensor data classes: the SensorData base contract and its three concrete
modalities (EMG, IMU, PPG).

These classes hold raw, unmodified wearable sensor recordings for exactly
one subject-condition trial. They perform no signal processing themselves;
that is the responsibility of the (future) ``ml/preprocessing`` and
``ml/feature_engineering`` modules, which consume these objects and produce
new ones rather than mutating them in place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.data_models.enums import EMGMuscle, IMULocation


@dataclass(frozen=True)
class SensorData(ABC):
    """Abstract base for any wearable sensor recording tied to one trial.

    Attributes:
        subject_id: Canonical subject identifier (e.g. "P01").
        condition_id: Canonical condition identifier this recording belongs to.
        sampling_rate_hz: Fixed sampling rate for this modality.
        source_file: Path to the exact raw file this recording was read from,
            kept for provenance and auditability.
    """

    subject_id: str
    condition_id: str
    sampling_rate_hz: float
    source_file: Path

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Number of samples in this recording."""

    @property
    def duration_seconds(self) -> float:
        """Recording duration, derived from sample count and sampling rate."""
        if self.sampling_rate_hz <= 0:
            return 0.0
        return self.n_samples / self.sampling_rate_hz


@dataclass(frozen=True)
class EMGData(SensorData):
    """Surface EMG recording for one muscle during one trial.

    Attributes:
        muscle: Which of the 6 instrumented muscles this is.
        signal: Single-column DataFrame of raw EMG samples (mV).
        unit: Fixed physical unit of ``signal``.
    """

    muscle: EMGMuscle
    signal: pd.DataFrame
    unit: str = "mV"

    @property
    def n_samples(self) -> int:
        return len(self.signal)


@dataclass(frozen=True)
class IMUData(SensorData):
    """Full 9-axis IMU recording for one body location during one trial.

    Represented as a single object per location (not three) because the
    accelerometer, gyroscope, and magnetometer streams come from the same
    physical sensor, sampled in lockstep, and must not drift out of sync.

    Attributes:
        location: Which of the 6 instrumented body locations this is.
        accelerometer: DataFrame with columns acc_x, acc_y, acc_z (m/s^2).
        gyroscope: DataFrame with columns gyr_x, gyr_y, gyr_z (rad/s).
        magnetometer: DataFrame with columns mag_x, mag_y, mag_z (a.u.).
    """

    location: IMULocation
    accelerometer: pd.DataFrame
    gyroscope: pd.DataFrame
    magnetometer: pd.DataFrame

    @property
    def n_samples(self) -> int:
        return len(self.accelerometer)


@dataclass(frozen=True)
class PPGData(SensorData):
    """Photoplethysmogram recording during one trial.

    Only present for fatigue-task conditions; MVIC calibration trials were
    not instrumented with PPG.

    Attributes:
        signal: Single-column DataFrame of raw PPG samples (Volt).
        unit: Fixed physical unit of ``signal``.
        sensor_site: Fixed anatomical placement of the PPG sensor.
    """

    signal: pd.DataFrame
    unit: str = "Volt"
    sensor_site: str = "non_dominant_index_finger"

    @property
    def n_samples(self) -> int:
        return len(self.signal)
