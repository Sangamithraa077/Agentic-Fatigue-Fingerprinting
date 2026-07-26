"""Unit tests for backend.services.replay_engine.

The full session-build pipeline (`get_or_build_session`) replays a real
subject's entire recording through Phases 5-8.5 and is exercised by the
manual/browser verification pass documented in `docs/replay_dashboard.md`,
not here -- it takes on the order of tens of seconds to minutes per
subject, which is inappropriate for a fast unit test run on every change.
This suite covers the parts of the module that are cheap to test in
isolation: real subject discovery, session-cache lookup semantics, and the
pure signal-slicing helpers (given small, hand-built `Processed*`/
`*Features` objects instead of a full pipeline run).
"""

from __future__ import annotations

import numpy as np

from backend.services import replay_engine
from ml.data_models.enums import EMGMuscle, IMULocation
from ml.feature_engineering.feature_vector import EMGChannelFeatures, FeatureVector, IMULocationFeatures, WindowedFeatures
from ml.preprocessing.preprocessing_results import ProcessedEMGData, ProcessedIMUData
from ml.preprocessing.quality_checks import validate_signal
from ml.preprocessing.signal_preprocessor import ProcessedSignal


def _processed_signal(values: np.ndarray, rate: float) -> ProcessedSignal:
    quality = validate_signal(values, rate)
    return ProcessedSignal(values=values, sampling_rate_hz=rate, steps_applied=(), input_quality=quality, output_quality=quality)


def test_list_available_subjects_returns_all_real_discoverable_subjects():
    subjects = replay_engine.list_available_subjects()
    assert len(subjects) == 34
    assert subjects == sorted(subjects)
    assert "P01" in subjects and "P34" in subjects


def test_get_cached_session_returns_none_when_not_built():
    assert replay_engine.get_cached_session("A_SUBJECT_NEVER_REQUESTED") is None


def test_signal_slice_emg_uses_real_window_bounds():
    emg_values = np.arange(10, dtype=float)
    processed_signal = _processed_signal(emg_values, 1000.0)
    windowed = WindowedFeatures(window_index=0, window_start_sample=2, window_end_sample=6, features=FeatureVector(names=("mav",), values=(0.5,)))

    condition_features_emg = {EMGMuscle.ANTERIOR_DELTOID: EMGChannelFeatures(muscle=EMGMuscle.ANTERIOR_DELTOID, windows=[windowed])}
    processed_condition_emg = {EMGMuscle.ANTERIOR_DELTOID: ProcessedEMGData(muscle=EMGMuscle.ANTERIOR_DELTOID, signal=processed_signal)}

    class _FakeFeaturesCondition:
        emg = condition_features_emg
        imu = {}

    class _FakeProcessedCondition:
        emg = processed_condition_emg
        imu = {}

    result = replay_engine._signal_slice(_FakeProcessedCondition(), _FakeFeaturesCondition(), EMGMuscle.ANTERIOR_DELTOID, False, 0)
    assert result == [2.0, 3.0, 4.0, 5.0]  # exactly the window_start_sample:window_end_sample slice


def test_signal_slice_emg_missing_channel_returns_none():
    class _Empty:
        emg = {}
        imu = {}

    result = replay_engine._signal_slice(_Empty(), _Empty(), EMGMuscle.PECTORALIS_MAJOR, False, 0)
    assert result is None


def test_signal_slice_imu_returns_all_three_axes():
    acc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    processed_signal = _processed_signal(acc, 100.0)
    windowed = WindowedFeatures(window_index=0, window_start_sample=1, window_end_sample=3, features=FeatureVector(names=("mean",), values=(1.0,)))

    condition_features_imu = {IMULocation.UPPER_ARM: IMULocationFeatures(location=IMULocation.UPPER_ARM, windows=[windowed])}
    processed_condition_imu = {IMULocation.UPPER_ARM: ProcessedIMUData(location=IMULocation.UPPER_ARM, accelerometer=processed_signal, gyroscope=processed_signal, magnetometer=processed_signal, stream_consistent=True)}

    class _FakeFeaturesCondition:
        emg = {}
        imu = condition_features_imu

    class _FakeProcessedCondition:
        emg = {}
        imu = processed_condition_imu

    result = replay_engine._signal_slice(_FakeProcessedCondition(), _FakeFeaturesCondition(), IMULocation.UPPER_ARM, True, 0)
    assert result["acc_x"] == [4.0, 7.0]
    assert result["acc_y"] == [5.0, 8.0]
    assert result["acc_z"] == [6.0, 9.0]


def test_ppg_slice_none_when_no_ppg():
    class _NoPpg:
        ppg = None

    assert replay_engine._ppg_slice(_NoPpg(), 5.0) is None


def test_ppg_slice_returns_recent_window_ending_at_elapsed_time():
    values = np.arange(0, 1000, dtype=float)  # 5 seconds at 200 Hz
    signal = _processed_signal(values, 200.0)

    class _PpgSignal:
        signal = None

    ppg_holder = _PpgSignal()
    ppg_holder.signal = signal

    class _FakeCondition:
        ppg = ppg_holder

    result = replay_engine._ppg_slice(_FakeCondition(), elapsed_seconds=2.0)
    assert result["sampling_rate_hz"] == 200.0
    # end_sample = int(2.0*200)+1 = 401; start_sample = max(0, 401 - 4*200) = 0
    assert len(result["values"]) == 401
    assert result["values"][-1] == 400.0
