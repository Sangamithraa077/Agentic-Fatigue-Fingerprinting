"""Unit tests for ml.feature_engineering (Phase 8 feature extraction pipeline).

These exercise the small, pure helper functions directly (windowing,
generic statistics, and each modality's individual feature formulas), plus
an end-to-end check that `FeaturePipeline` correctly builds windowed
`FeatureVector`s from synthetic `ProcessedCondition`/`ProcessedSubject`
instances (built directly from Phase 7 result types, without needing the
real dataset on disk).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.data_models.enums import ConditionType, Direction, EMGMuscle, IMULocation, IntensityBand
from ml.feature_engineering.emg_features import (
    EMGFeatureConfig,
    EMGFeatureExtractor,
    integrated_emg,
    is_likely_rectified,
    mean_frequency,
    median_frequency,
    peak_frequency,
    power_spectrum,
    slope_sign_changes,
    spectral_entropy,
    waveform_length,
    willison_amplitude,
    zero_crossings,
)
from ml.feature_engineering.feature_extractor import (
    compute_energy,
    compute_max,
    compute_mean,
    compute_mean_absolute_value,
    compute_min,
    compute_range,
    compute_rms,
    compute_std,
    compute_variance,
)
from ml.feature_engineering.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from ml.feature_engineering.feature_vector import ConditionFeatures, FeatureVector, condition_features_to_dataframe
from ml.feature_engineering.imu_features import (
    IMUFeatureExtractor,
    compute_axis_features,
    compute_resultant_features,
    extract_stream_features,
    imu_feature_names,
)
from ml.feature_engineering.ppg_features import PPGFeatureConfig, PPGFeatureExtractor, detect_peaks, pulse_intervals_seconds
from ml.feature_engineering.windowing import (
    WindowConfig,
    compute_step_samples,
    compute_window_samples,
    generate_window_indices,
    segment_signal,
)
from ml.preprocessing.preprocessing_results import ProcessedCondition, ProcessedEMGData, ProcessedIMUData, ProcessedPPGData, ProcessedSubject
from ml.preprocessing.quality_checks import validate_signal
from ml.preprocessing.signal_preprocessor import ProcessedSignal

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------


def test_window_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        WindowConfig(window_seconds=0)
    with pytest.raises(ValueError):
        WindowConfig(overlap_ratio=1.0)


def test_compute_window_samples():
    assert compute_window_samples(1.0, 1000.0) == 1000
    assert compute_window_samples(0.5, 100.0) == 50


def test_compute_step_samples():
    assert compute_step_samples(1000, 0.5) == 500
    assert compute_step_samples(1000, 0.0) == 1000


def test_generate_window_indices_basic():
    indices = generate_window_indices(1000, 200, 100)
    assert indices[0] == (0, 200)
    assert indices[1] == (100, 300)
    assert all(end - start == 200 for start, end in indices)


def test_generate_window_indices_drops_incomplete_by_default():
    indices = generate_window_indices(250, 200, 100, drop_last_incomplete=True)
    assert all(end <= 250 for _, end in indices)
    assert all(end - start == 200 for start, end in indices)


def test_generate_window_indices_keeps_incomplete_when_requested():
    indices = generate_window_indices(250, 200, 100, drop_last_incomplete=False)
    assert indices[-1][1] == 250


def test_segment_signal_matches_indices():
    signal = np.arange(1000, dtype=float)
    windows = segment_signal(signal, 200, 100)
    assert len(windows) == len(generate_window_indices(1000, 200, 100))
    assert np.array_equal(windows[0], signal[0:200])


# ---------------------------------------------------------------------------
# feature_extractor (shared generic primitives)
# ---------------------------------------------------------------------------


def test_generic_primitives():
    x = np.array([1.0, -2.0, 3.0, -4.0])
    assert compute_mean(x) == pytest.approx(-0.5)
    assert compute_std(x) > 0
    assert compute_variance(x) == pytest.approx(compute_std(x) ** 2)
    assert compute_rms(x) == pytest.approx(np.sqrt(np.mean(x**2)))
    assert compute_energy(x) == pytest.approx(np.sum(x**2))
    assert compute_max(x) == 3.0
    assert compute_min(x) == -4.0
    assert compute_range(x) == 7.0
    assert compute_mean_absolute_value(x) == pytest.approx(2.5)


def test_generic_primitives_empty_array():
    empty = np.array([])
    assert np.isnan(compute_mean(empty))
    assert np.isnan(compute_rms(empty))


# ---------------------------------------------------------------------------
# feature_vector
# ---------------------------------------------------------------------------


def test_feature_vector_as_dict():
    fv = FeatureVector(names=("a", "b"), values=(1.0, 2.0))
    assert fv.as_dict() == {"a": 1.0, "b": 2.0}


def test_feature_vector_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        FeatureVector(names=("a", "b"), values=(1.0,))


def test_condition_features_to_dataframe_empty():
    cf = ConditionFeatures(
        condition_id="P00_x",
        subject_id="P00",
        condition_type=ConditionType.FATIGUE_TASK,
        direction=Direction.EXTERNAL_ROTATION,
        intensity_band=IntensityBand.BAND_30_40,
        trial_number=None,
    )
    df = condition_features_to_dataframe(cf)
    assert df.empty


# ---------------------------------------------------------------------------
# EMG features
# ---------------------------------------------------------------------------


def test_waveform_length_positive_for_varying_signal():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    assert waveform_length(x) == pytest.approx(4.0)


def test_zero_crossings_counts_sign_changes():
    x = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    assert zero_crossings(x) == 4


def test_zero_crossings_respects_threshold():
    x = np.array([0.01, -0.01, 0.01, -0.01])
    assert zero_crossings(x, threshold=1.0) == 0


def test_slope_sign_changes_basic():
    x = np.array([0.0, 2.0, 1.0, 3.0, 0.0])
    assert slope_sign_changes(x) > 0


def test_willison_amplitude_counts_large_jumps():
    x = np.array([0.0, 5.0, 0.0, 5.0, 0.0])
    assert willison_amplitude(x, threshold=1.0) == 4


def test_integrated_emg_is_sum_of_abs():
    x = np.array([1.0, -2.0, 3.0])
    assert integrated_emg(x) == pytest.approx(6.0)


def test_is_likely_rectified():
    assert is_likely_rectified(np.abs(RNG.normal(0, 1, 500)))
    assert not is_likely_rectified(RNG.normal(0, 1, 500))


def test_power_spectrum_and_frequency_features():
    fs = 1000.0
    t = np.arange(2000) / fs
    x = np.sin(2 * np.pi * 100 * t)  # pure 100 Hz tone
    freqs, power = power_spectrum(x, fs)
    assert freqs.size > 0
    peak = peak_frequency(freqs, power)
    assert peak == pytest.approx(100.0, abs=10.0)
    mnf = mean_frequency(freqs, power)
    mdf = median_frequency(freqs, power)
    assert mnf > 0 and mdf > 0
    entropy = spectral_entropy(power)
    assert 0.0 <= entropy <= 1.0


def test_emg_feature_extractor_produces_all_configured_names():
    x = RNG.normal(0, 1, 2000)
    extractor = EMGFeatureExtractor(EMGFeatureConfig())
    fv = extractor.extract(x, 1000.0)
    assert set(fv.names) == {
        "mav", "rms", "variance", "waveform_length", "zero_crossings", "slope_sign_changes",
        "willison_amplitude", "integrated_emg", "mean_frequency", "median_frequency",
        "peak_frequency", "spectral_entropy",
    }


def test_emg_feature_extractor_flags_rectified_input_once():
    extractor = EMGFeatureExtractor(EMGFeatureConfig())
    assert not extractor.detected_rectified_input
    extractor.extract(np.abs(RNG.normal(0, 1, 500)), 1000.0)
    assert extractor.detected_rectified_input


def test_emg_feature_extractor_skips_frequency_domain_when_disabled():
    extractor = EMGFeatureExtractor(EMGFeatureConfig(compute_frequency_domain=False))
    fv = extractor.extract(RNG.normal(0, 1, 500), 1000.0)
    assert "mean_frequency" not in fv.names


# ---------------------------------------------------------------------------
# IMU features
# ---------------------------------------------------------------------------


def test_compute_axis_features_keys():
    x = RNG.normal(0, 1, 100)
    features = compute_axis_features(x)
    assert set(features.keys()) == {"mean", "std", "rms", "max", "min", "range", "energy", "sma"}


def test_compute_resultant_features_nonnegative_magnitude_stats():
    x = RNG.normal(0, 1, 100)
    y = RNG.normal(0, 1, 100)
    z = RNG.normal(0, 1, 100)
    features = compute_resultant_features(x, y, z)
    assert features["resultant_min"] >= 0
    assert features["resultant_max"] >= features["resultant_min"]


def test_extract_stream_features_naming():
    window = RNG.normal(0, 1, (50, 3))
    features = extract_stream_features(window, "acc")
    assert "acc_x_mean" in features
    assert "acc_resultant_sma" in features
    assert len(features) == 3 * 8 + 7


def test_imu_feature_extractor_total_feature_count_matches_catalog():
    acc = RNG.normal(0, 1, (100, 3))
    gyr = RNG.normal(0, 1, (100, 3))
    mag = RNG.normal(0, 1, (100, 3))
    fv = IMUFeatureExtractor().extract_location(acc, gyr, mag)
    assert set(fv.names) == set(imu_feature_names())


# ---------------------------------------------------------------------------
# PPG features
# ---------------------------------------------------------------------------


def test_detect_peaks_finds_approximately_correct_count():
    fs = 200.0
    duration_s = 10.0
    hr_hz = 75.0 / 60.0
    t = np.arange(int(fs * duration_s)) / fs
    signal = np.sin(2 * np.pi * hr_hz * t)
    peaks = detect_peaks(signal, fs)
    expected = int(duration_s * hr_hz)
    assert abs(len(peaks) - expected) <= 1


def test_pulse_intervals_seconds():
    peaks = np.array([0, 100, 200, 300])
    intervals = pulse_intervals_seconds(peaks, 100.0)
    assert np.allclose(intervals, [1.0, 1.0, 1.0])


def test_ppg_feature_extractor_returns_nan_when_insufficient_peaks():
    flat_signal = np.zeros(50)
    extractor = PPGFeatureExtractor()
    fv = extractor.extract(flat_signal, 200.0)
    values = fv.as_dict()
    assert values["peak_count"] == 0
    assert np.isnan(values["heart_rate_bpm"])


def test_ppg_feature_extractor_realistic_signal():
    fs = 200.0
    t = np.arange(int(fs * 10)) / fs
    hr_hz = 70.0 / 60.0
    signal = np.sin(2 * np.pi * hr_hz * t)
    fv = PPGFeatureExtractor(PPGFeatureConfig()).extract(signal, fs)
    values = fv.as_dict()
    assert values["peak_count"] >= 5
    assert 60.0 < values["heart_rate_bpm"] < 80.0


# ---------------------------------------------------------------------------
# FeaturePipeline (synthetic ProcessedCondition/ProcessedSubject objects)
# ---------------------------------------------------------------------------


def _make_processed_signal(n: int, sampling_rate_hz: float, rectified: bool = False) -> ProcessedSignal:
    values = np.abs(RNG.normal(0, 1, n)) if rectified else RNG.normal(0, 1, n)
    quality = validate_signal(values, sampling_rate_hz=sampling_rate_hz)
    return ProcessedSignal(
        values=values, sampling_rate_hz=sampling_rate_hz, steps_applied=(), input_quality=quality, output_quality=quality
    )


def _make_processed_imu_signal(n: int, sampling_rate_hz: float) -> ProcessedSignal:
    values = RNG.normal(0, 1, (n, 3))
    quality = validate_signal(values, sampling_rate_hz=sampling_rate_hz)
    return ProcessedSignal(
        values=values, sampling_rate_hz=sampling_rate_hz, steps_applied=(), input_quality=quality, output_quality=quality
    )


def _make_processed_condition(subject_id: str, with_ppg: bool, emg_seconds: float = 3.0) -> ProcessedCondition:
    condition_id = f"{subject_id}_external_rotation_30-40"
    emg = {
        muscle: ProcessedEMGData(muscle=muscle, signal=_make_processed_signal(int(1000 * emg_seconds), 1000.0))
        for muscle in EMGMuscle
    }
    imu = {
        location: ProcessedIMUData(
            location=location,
            accelerometer=_make_processed_imu_signal(int(100 * emg_seconds), 100.0),
            gyroscope=_make_processed_imu_signal(int(100 * emg_seconds), 100.0),
            magnetometer=_make_processed_imu_signal(int(100 * emg_seconds), 100.0),
            stream_consistent=True,
        )
        for location in IMULocation
    }
    ppg = None
    if with_ppg:
        ppg = ProcessedPPGData(signal=_make_processed_signal(int(200 * 15), 200.0))
    return ProcessedCondition(
        condition_id=condition_id,
        subject_id=subject_id,
        condition_type=ConditionType.FATIGUE_TASK,
        direction=Direction.EXTERNAL_ROTATION,
        intensity_band=IntensityBand.BAND_30_40,
        trial_number=None,
        emg=emg,
        imu=imu,
        ppg=ppg,
        labels=None,
    )


def test_feature_pipeline_extract_condition_produces_windows_for_every_modality():
    condition = _make_processed_condition("P99", with_ppg=True)
    pipeline = FeaturePipeline()
    result = pipeline.extract_condition(condition)

    assert set(result.emg.keys()) == set(condition.emg.keys())
    assert set(result.imu.keys()) == set(condition.imu.keys())
    assert result.ppg is not None

    # 3 seconds @ 1s window, 0.5 overlap -> multiple windows expected
    any_muscle_windows = next(iter(result.emg.values())).windows
    assert len(any_muscle_windows) >= 3

    summary = pipeline.summary
    assert summary.n_conditions_processed == 1
    assert summary.n_emg_windows == len(any_muscle_windows) * len(EMGMuscle)


def test_feature_pipeline_extract_condition_without_ppg():
    condition = _make_processed_condition("P99", with_ppg=False)
    pipeline = FeaturePipeline()
    result = pipeline.extract_condition(condition)
    assert result.ppg is None


def test_feature_pipeline_extract_subject_aggregates_conditions():
    condition = _make_processed_condition("P99", with_ppg=True)
    subject = ProcessedSubject(subject_id="P99", conditions={condition.condition_id: condition})
    pipeline = FeaturePipeline()
    result = pipeline.extract_subject(subject)
    assert result.subject_id == "P99"
    assert len(result.conditions) == 1


def test_feature_pipeline_dispatch_by_type():
    condition = _make_processed_condition("P99", with_ppg=False)
    pipeline = FeaturePipeline()
    result = pipeline.extract(condition)
    assert type(result).__name__ == "ConditionFeatures"


def test_feature_pipeline_rejects_invalid_type():
    pipeline = FeaturePipeline()
    with pytest.raises(TypeError):
        pipeline.extract("not processed data")


def test_feature_pipeline_summary_reports_skipped_hrv_features():
    pipeline = FeaturePipeline()
    summary = pipeline.summary
    assert "rmssd" in summary.skipped_features
    assert "lf_hf_ratio" in summary.skipped_features


def test_feature_pipeline_flattens_to_dataframe():
    condition = _make_processed_condition("P99", with_ppg=True)
    pipeline = FeaturePipeline()
    result = pipeline.extract_condition(condition)
    df = condition_features_to_dataframe(result)
    assert not df.empty
    assert "modality" in df.columns
    assert set(df["modality"].unique()) <= {"EMG", "IMU", "PPG"}


def test_feature_pipeline_custom_window_config():
    condition = _make_processed_condition("P99", with_ppg=False, emg_seconds=2.0)
    config = FeaturePipelineConfig(emg_window=WindowConfig(window_seconds=0.5, overlap_ratio=0.0))
    pipeline = FeaturePipeline(config)
    result = pipeline.extract_condition(condition)
    any_muscle_windows = next(iter(result.emg.values())).windows
    # 2s signal, 0.5s non-overlapping windows -> 4 windows
    assert len(any_muscle_windows) == 4
