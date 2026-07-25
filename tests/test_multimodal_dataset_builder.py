"""Unit tests for ml.models.common.multimodal_dataset_builder (Phase 10's
EMG/IMU/PPG alignment infrastructure)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.feature_engineering.feature_dataset import FeatureDataset
from ml.models.common.config import DatasetConfig
from ml.models.common.multimodal_dataset_builder import (
    ModalityDims,
    build_multimodal_dataset,
    combine_sensors_per_window,
)

RNG = np.random.default_rng(21)


def _make_long_format_dataset(modality: str, sensors: list, n_windows: int, n_features: int, subject_id: str = "P01", condition_id: str = "cond1", rpe: float = 15.0) -> FeatureDataset:
    rows_matrix = []
    rows_meta = []
    for sensor in sensors:
        for window_index in range(n_windows):
            rows_matrix.append({f"feat_{i}": RNG.normal(0, 1) for i in range(n_features)})
            rows_meta.append(
                {
                    "subject_id": subject_id,
                    "condition_id": condition_id,
                    "condition_type": "fatigue_task",
                    "direction": "external_rotation",
                    "intensity_band": "30-40",
                    "trial_number": None,
                    "source_sensor": sensor,
                    "window_index": window_index,
                    "rpe_nearest": rpe,
                }
            )
    matrix = pd.DataFrame(rows_matrix)
    metadata = pd.DataFrame(rows_meta)
    return FeatureDataset(modality=modality, feature_names=tuple(matrix.columns), matrix=matrix, metadata=metadata)


def test_combine_sensors_per_window_shapes():
    dataset = _make_long_format_dataset("EMG", ["m1", "m2", "m3"], n_windows=5, n_features=4)
    matrix, metadata = combine_sensors_per_window(dataset)
    assert matrix.shape == (5, 3 * 4)  # 3 sensors x 4 features
    assert set(metadata.columns) >= {"subject_id", "condition_id", "window_index"}
    assert len(metadata) == 5
    assert "m1__feat_0" in matrix.columns
    assert "m3__feat_3" in matrix.columns


def test_combine_sensors_per_window_empty_dataset():
    empty = FeatureDataset(modality="EMG", feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame())
    matrix, metadata = combine_sensors_per_window(empty)
    assert matrix.empty


def test_modality_dims_split_roundtrip():
    dims = ModalityDims(emg=3, imu=5, ppg=2)
    X = np.arange(100).reshape(10, 10).astype(float)
    x_emg, x_imu, x_ppg = dims.split(X)
    assert x_emg.shape == (10, 3)
    assert x_imu.shape == (10, 5)
    assert x_ppg.shape == (10, 2)
    assert np.array_equal(np.concatenate([x_emg, x_imu, x_ppg], axis=1), X)
    assert dims.total == 10


def test_build_multimodal_dataset_basic_alignment():
    emg = _make_long_format_dataset("EMG", ["m1", "m2"], n_windows=10, n_features=3)
    imu = _make_long_format_dataset("IMU", ["l1", "l2"], n_windows=10, n_features=4)
    ppg = _make_long_format_dataset("PPG", ["ppg"], n_windows=1, n_features=2)  # 1 coarse PPG window covers all 10 EMG/IMU windows

    config = DatasetConfig(modality="EMG", target_type="binary_fatigue", binary_fatigue_threshold=13.0)
    result = build_multimodal_dataset(emg, imu, ppg, config)

    assert result.dims.emg == 2 * 3
    assert result.dims.imu == 2 * 4
    assert result.dims.ppg == 1 * 2
    assert result.X.shape == (10, result.dims.total)
    assert len(result.y) == 10
    assert not np.isnan(result.X).any()
    assert not np.isinf(result.X).any()
    # rpe=15 >= threshold 13 -> all "fatigued"
    assert (result.y == 1).all()


def test_build_multimodal_dataset_excludes_mvic_rows_without_ppg():
    # Fatigue-task condition (with PPG) + MVIC condition (no PPG, no RPE)
    emg_fatigue = _make_long_format_dataset("EMG", ["m1"], n_windows=5, n_features=2, condition_id="fatigue1", rpe=15.0)
    imu_fatigue = _make_long_format_dataset("IMU", ["l1"], n_windows=5, n_features=2, condition_id="fatigue1", rpe=15.0)
    ppg_fatigue = _make_long_format_dataset("PPG", ["ppg"], n_windows=1, n_features=2, condition_id="fatigue1", rpe=15.0)

    emg_mvic = _make_long_format_dataset("EMG", ["m1"], n_windows=3, n_features=2, condition_id="mvic1", rpe=np.nan)
    imu_mvic = _make_long_format_dataset("IMU", ["l1"], n_windows=3, n_features=2, condition_id="mvic1", rpe=np.nan)

    emg = FeatureDataset(
        modality="EMG", feature_names=emg_fatigue.feature_names,
        matrix=pd.concat([emg_fatigue.matrix, emg_mvic.matrix], ignore_index=True),
        metadata=pd.concat([emg_fatigue.metadata, emg_mvic.metadata], ignore_index=True),
    )
    imu = FeatureDataset(
        modality="IMU", feature_names=imu_fatigue.feature_names,
        matrix=pd.concat([imu_fatigue.matrix, imu_mvic.matrix], ignore_index=True),
        metadata=pd.concat([imu_fatigue.metadata, imu_mvic.metadata], ignore_index=True),
    )

    config = DatasetConfig(modality="EMG", target_type="binary_fatigue")
    result = build_multimodal_dataset(emg, imu, ppg_fatigue, config)

    # only the 5 fatigue-task windows should survive (3 MVIC windows excluded -- no RPE)
    assert len(result.y) == 5
    assert result.n_rows_excluded == 3


def test_build_multimodal_dataset_empty_ppg_is_handled_gracefully():
    # A fully empty PPG FeatureDataset carries no sensor labels to prefix
    # columns with, so it correctly contributes zero PPG feature columns
    # (an honest reflection of "no PPG data available"), rather than
    # raising -- this is the graceful-degradation path being tested.
    emg = _make_long_format_dataset("EMG", ["m1"], n_windows=5, n_features=2)
    imu = _make_long_format_dataset("IMU", ["l1"], n_windows=5, n_features=2)
    empty_ppg = FeatureDataset(modality="PPG", feature_names=("f1", "f2"), matrix=pd.DataFrame(), metadata=pd.DataFrame())

    config = DatasetConfig(modality="EMG", target_type="binary_fatigue")
    result = build_multimodal_dataset(emg, imu, empty_ppg, config)
    assert result.dims.ppg == 0
    assert result.X.shape[1] == result.dims.total == result.dims.emg + result.dims.imu
    assert not np.isnan(result.X).any()
    assert not np.isinf(result.X).any()
