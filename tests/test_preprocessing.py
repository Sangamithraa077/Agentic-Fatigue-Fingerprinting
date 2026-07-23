"""Unit tests for ml.preprocessing (Phase 7 signal preprocessing pipeline).

These exercise the small, pure helper functions directly (quality checks,
resampling, and each modality's individual transforms), plus an end-to-end
check that the `PreprocessingPipeline` correctly builds `Processed*` objects
from synthetic `Condition`/`Subject` instances without mutating the raw
input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data_models.condition import Condition
from ml.data_models.enums import ConditionType, Direction, EMGMuscle, IMULocation, IntensityBand
from ml.data_models.sensors import EMGData, IMUData, PPGData
from ml.data_models.subject import Subject
from ml.preprocessing.emg_preprocessor import (
    EMGPreprocessingConfig,
    EMGPreprocessor,
    bandpass_filter,
    rectify,
    remove_dc_offset,
)
from ml.preprocessing.imu_preprocessor import (
    IMUPreprocessingConfig,
    IMUPreprocessor,
    check_stream_consistency,
    handle_missing_values,
    validate_timestamps,
)
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.preprocessing.ppg_preprocessor import PPGPreprocessingConfig, PPGPreprocessor, remove_baseline_drift
from ml.preprocessing.quality_checks import (
    check_length_consistency,
    check_sampling_rate,
    count_infinite,
    count_nan,
    is_empty_signal,
    validate_signal,
)
from ml.preprocessing.resampling import align_signal_length, compute_resampled_length, resample_signal

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# quality_checks
# ---------------------------------------------------------------------------


def test_count_nan_and_infinite():
    signal = np.array([1.0, np.nan, np.inf, -np.inf, 2.0, np.nan])
    assert count_nan(signal) == 2
    assert count_infinite(signal) == 2


def test_is_empty_signal():
    assert is_empty_signal(np.array([]))
    assert not is_empty_signal(np.array([1.0]))


def test_check_sampling_rate():
    assert check_sampling_rate(100.0)
    assert not check_sampling_rate(0.0)
    assert not check_sampling_rate(-5.0)
    assert not check_sampling_rate(float("nan"))
    assert check_sampling_rate(100.0, expected_hz=100.0)
    assert not check_sampling_rate(100.0, expected_hz=200.0)


def test_check_length_consistency():
    assert check_length_consistency([10, 10, 10])
    assert not check_length_consistency([10, 9, 10])
    assert check_length_consistency([])


def test_validate_signal_clean():
    report = validate_signal(np.ones(50), sampling_rate_hz=100.0, expected_sampling_rate_hz=100.0)
    assert report.passed
    assert report.n_nan == 0
    assert report.n_infinite == 0
    assert not report.is_empty


def test_validate_signal_flags_nan_and_infinite():
    report = validate_signal(np.array([1.0, np.nan, np.inf]), sampling_rate_hz=100.0)
    assert not report.passed
    assert report.n_nan == 1
    assert report.n_infinite == 1
    assert len(report.issues) >= 2


def test_validate_signal_flags_empty():
    report = validate_signal(np.array([]), sampling_rate_hz=100.0)
    assert not report.passed
    assert report.is_empty


def test_validate_signal_flags_invalid_sampling_rate():
    report = validate_signal(np.ones(10), sampling_rate_hz=0.0)
    assert not report.passed
    assert not report.sampling_rate_valid


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------


def test_compute_resampled_length():
    assert compute_resampled_length(1000, 1000.0, 100.0) == 100
    assert compute_resampled_length(100, 100.0, 1000.0) == 1000


def test_compute_resampled_length_rejects_invalid_rates():
    with pytest.raises(ValueError):
        compute_resampled_length(100, 0.0, 100.0)


def test_resample_signal_changes_length():
    signal = np.sin(np.linspace(0, 10, 1000))
    resampled = resample_signal(signal, 1000.0, 100.0)
    assert len(resampled) == 100


def test_resample_signal_noop_when_rates_equal():
    signal = np.arange(10, dtype=float)
    assert np.array_equal(resample_signal(signal, 100.0, 100.0), signal)


def test_align_signal_length_pads_and_truncates():
    signal = np.arange(5, dtype=float)
    padded = align_signal_length(signal, 8)
    assert len(padded) == 8
    assert np.isnan(padded[5:]).all()

    truncated = align_signal_length(signal, 3)
    assert len(truncated) == 3
    assert np.array_equal(truncated, signal[:3])


# ---------------------------------------------------------------------------
# EMG preprocessing
# ---------------------------------------------------------------------------


def test_remove_dc_offset_centers_signal():
    signal = 5.0 + RNG.normal(0, 0.1, 500)
    centered = remove_dc_offset(signal)
    assert abs(centered.mean()) < abs(signal.mean())


def test_rectify_is_non_negative():
    signal = RNG.normal(0, 1, 200)
    assert (rectify(signal) >= 0).all()


def test_bandpass_filter_preserves_length():
    signal = RNG.normal(0, 1, 2000)
    filtered = bandpass_filter(signal, sampling_rate_hz=1000.0, low_hz=20.0, high_hz=450.0)
    assert filtered.shape == signal.shape


def test_bandpass_filter_invalid_range_returns_unchanged():
    signal = RNG.normal(0, 1, 200)
    # low >= high is invalid -- should log a warning and return input unchanged
    filtered = bandpass_filter(signal, sampling_rate_hz=1000.0, low_hz=400.0, high_hz=100.0)
    assert np.array_equal(filtered, signal)


def test_emg_preprocessor_full_chain_is_non_negative_with_envelope():
    signal = 5.0 + RNG.normal(0, 0.5, 2000)
    preprocessor = EMGPreprocessor(EMGPreprocessingConfig(compute_envelope=True))
    result = preprocessor.process(signal, 1000.0)
    # Zero-phase (filtfilt) low-pass filtering of a rectified signal can
    # ripple slightly below zero right at the start/end edges -- a known,
    # expected artifact of non-causal filtering, not a defect. Excluding a
    # small edge margin, the envelope must be non-negative.
    interior = result.values[50:-50]
    assert interior.min() > -1e-6
    assert "rectify" in result.steps_applied
    assert result.output_quality.passed


def test_emg_preprocessor_handles_empty_signal():
    preprocessor = EMGPreprocessor()
    result = preprocessor.process(np.array([]), 1000.0)
    assert result.steps_applied == ()
    assert result.input_quality.is_empty


def test_emg_preprocessor_skips_envelope_without_rectify():
    preprocessor = EMGPreprocessor(EMGPreprocessingConfig(rectify=False, compute_envelope=True))
    result = preprocessor.process(RNG.normal(0, 1, 500), 1000.0)
    assert not any("envelope" in step for step in result.steps_applied)


# ---------------------------------------------------------------------------
# IMU preprocessing
# ---------------------------------------------------------------------------


def test_handle_missing_values_interpolate_removes_nans():
    signal = RNG.normal(0, 1, (100, 3))
    signal[40:43, 1] = np.nan
    cleaned = handle_missing_values(signal, strategy="interpolate")
    assert not np.isnan(cleaned).any()
    assert cleaned.shape == signal.shape


def test_handle_missing_values_drop_removes_rows():
    signal = RNG.normal(0, 1, (10, 3))
    signal[3, 0] = np.nan
    cleaned = handle_missing_values(signal, strategy="drop")
    assert cleaned.shape[0] == 9


def test_handle_missing_values_unknown_strategy_raises():
    signal = np.array([[1.0, 2.0], [np.nan, 4.0]])
    with pytest.raises(ValueError):
        handle_missing_values(signal, strategy="bogus")


def test_validate_timestamps():
    assert validate_timestamps(100, 100.0)
    assert not validate_timestamps(0, 100.0)
    assert not validate_timestamps(100, 0.0)


def test_check_stream_consistency():
    assert check_stream_consistency({"acc": 100, "gyr": 100, "mag": 100})
    assert not check_stream_consistency({"acc": 100, "gyr": 99, "mag": 100})


def test_imu_preprocessor_interpolates_and_preserves_shape():
    signal = RNG.normal(0, 1, (100, 3))
    signal[10, 0] = np.nan
    preprocessor = IMUPreprocessor(IMUPreprocessingConfig(apply_smoothing=True))
    result = preprocessor.process(signal, 100.0)
    assert not np.isnan(result.values).any()
    assert result.values.shape == signal.shape


# ---------------------------------------------------------------------------
# PPG preprocessing
# ---------------------------------------------------------------------------


def test_remove_baseline_drift_reduces_low_frequency_power():
    t = np.arange(2000) / 200.0
    drift = 3.0 * np.sin(2 * np.pi * 0.05 * t)
    pulse = 0.3 * np.sin(2 * np.pi * 1.2 * t)
    signal = drift + pulse
    cleaned = remove_baseline_drift(signal, sampling_rate_hz=200.0, cutoff_hz=0.5)
    assert cleaned.std() < signal.std()


def test_ppg_preprocessor_full_chain_runs():
    t = np.arange(1000) / 200.0
    signal = 2.0 * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 1.2 * t)
    preprocessor = PPGPreprocessor(PPGPreprocessingConfig())
    result = preprocessor.process(signal, 200.0)
    assert result.values.shape == signal.shape
    assert result.output_quality.passed


# ---------------------------------------------------------------------------
# Pipeline (synthetic Condition/Subject objects)
# ---------------------------------------------------------------------------


def _make_synthetic_emg(subject_id: str, condition_id: str, muscle: EMGMuscle, n: int = 500) -> EMGData:
    return EMGData(
        subject_id=subject_id,
        condition_id=condition_id,
        sampling_rate_hz=1000.0,
        source_file=Path("synthetic_emg.csv"),
        muscle=muscle,
        signal=pd.DataFrame({"EMG(mV)": RNG.normal(0, 1, n)}),
    )


def _make_synthetic_imu(subject_id: str, condition_id: str, location: IMULocation, n: int = 100) -> IMUData:
    return IMUData(
        subject_id=subject_id,
        condition_id=condition_id,
        sampling_rate_hz=100.0,
        source_file=Path("synthetic_imu"),
        location=location,
        accelerometer=pd.DataFrame(RNG.normal(0, 1, (n, 3)), columns=["acc_x", "acc_y", "acc_z"]),
        gyroscope=pd.DataFrame(RNG.normal(0, 1, (n, 3)), columns=["gyr_x", "gyr_y", "gyr_z"]),
        magnetometer=pd.DataFrame(RNG.normal(0, 1, (n, 3)), columns=["mag_x", "mag_y", "mag_z"]),
    )


def _make_synthetic_condition(subject_id: str, with_ppg: bool) -> Condition:
    condition_id = f"{subject_id}_external_rotation_30-40"
    emg = {muscle: _make_synthetic_emg(subject_id, condition_id, muscle) for muscle in EMGMuscle}
    imu = {location: _make_synthetic_imu(subject_id, condition_id, location) for location in IMULocation}
    ppg = None
    if with_ppg:
        ppg = PPGData(
            subject_id=subject_id,
            condition_id=condition_id,
            sampling_rate_hz=200.0,
            source_file=Path("synthetic_ppg.csv"),
            signal=pd.DataFrame({"PPG(Volt)": RNG.normal(0, 1, 200)}),
        )
    return Condition(
        condition_id=condition_id,
        subject_id=subject_id,
        condition_type=ConditionType.FATIGUE_TASK,
        direction=Direction.EXTERNAL_ROTATION,
        intensity_band=IntensityBand.BAND_30_40,
        trial_number=None,
        task_order_code="task4_35e",
        emg=emg,
        imu=imu,
        ppg=ppg,
        labels=None,
    )


def test_pipeline_process_condition_preserves_structure_and_raw_data():
    condition = _make_synthetic_condition("P99", with_ppg=True)
    raw_emg_signal_ref = condition.emg[EMGMuscle.ANTERIOR_DELTOID].signal

    pipeline = PreprocessingPipeline()
    processed = pipeline.process_condition(condition)

    assert processed.condition_id == condition.condition_id
    assert set(processed.emg.keys()) == set(condition.emg.keys())
    assert set(processed.imu.keys()) == set(condition.imu.keys())
    assert processed.ppg is not None
    assert processed.labels is None

    # raw data must be untouched
    assert condition.emg[EMGMuscle.ANTERIOR_DELTOID].signal is raw_emg_signal_ref

    summary = pipeline.summary
    assert summary.n_conditions_processed == 1
    assert summary.n_emg_signals_processed == len(EMGMuscle)
    assert summary.n_imu_signals_processed == len(IMULocation)
    assert summary.n_ppg_signals_processed == 1


def test_pipeline_process_condition_without_ppg():
    condition = _make_synthetic_condition("P99", with_ppg=False)
    pipeline = PreprocessingPipeline()
    processed = pipeline.process_condition(condition)
    assert processed.ppg is None


def test_pipeline_process_subject_aggregates_all_conditions():
    condition = _make_synthetic_condition("P99", with_ppg=True)
    subject = Subject(
        subject_id="P99",
        demographic=None,
        anthropometric=None,
        mvic_force=None,
        conditions={condition.condition_id: condition},
    )
    pipeline = PreprocessingPipeline()
    processed_subject = pipeline.process_subject(subject)
    assert processed_subject.subject_id == "P99"
    assert len(processed_subject.conditions) == 1


def test_pipeline_process_dispatches_by_type():
    condition = _make_synthetic_condition("P99", with_ppg=False)
    pipeline = PreprocessingPipeline()
    result = pipeline.process(condition)
    assert type(result).__name__ == "ProcessedCondition"


def test_pipeline_process_rejects_invalid_type():
    pipeline = PreprocessingPipeline()
    with pytest.raises(TypeError):
        pipeline.process("not a subject or condition")
