"""Unit tests for Phase 8.5 feature validation & selection
(ml.feature_engineering.feature_dataset / feature_validation /
feature_importance / feature_selection).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data_models.condition import Condition
from ml.data_models.enums import ConditionType, Direction, EMGMuscle, IMULocation, IntensityBand
from ml.data_models.labels import Labels
from ml.feature_engineering.feature_dataset import (
    METADATA_COLUMNS,
    build_feature_catalog,
    build_feature_datasets,
    categorize_feature,
    nearest_rpe,
    window_center_seconds,
)
from ml.feature_engineering.feature_importance import FeatureImportanceConfig, compute_feature_importance
from ml.feature_engineering.feature_pipeline import FeaturePipeline
from ml.feature_engineering.feature_selection import (
    FeatureSelectionConfig,
    FeatureSelector,
    apply_recommendations,
    recommend_correlation_filter,
    recommend_mutual_info_ranking,
    recommend_variance_threshold,
)
from ml.feature_engineering.feature_vector import WindowedFeatures, FeatureVector
from ml.feature_engineering.feature_validation import (
    FeatureValidationConfig,
    check_constant_features,
    check_duplicate_features,
    check_highly_correlated,
    check_infinite_values,
    check_missing_values,
    check_near_zero_variance,
    compute_correlation_matrix,
    compute_feature_statistics,
    validate_features,
)
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.preprocessing.preprocessing_results import ProcessedSubject
from ml.data_models.sensors import EMGData, IMUData, PPGData
from ml.data_models.subject import Subject

RNG = np.random.default_rng(11)


# ---------------------------------------------------------------------------
# feature_validation
# ---------------------------------------------------------------------------


def test_check_missing_values():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    counts = check_missing_values(df)
    assert counts["a"] == 1
    assert counts["b"] == 0


def test_check_infinite_values():
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf], "b": [1.0, 2.0, 3.0]})
    counts = check_infinite_values(df)
    assert counts["a"] == 2
    assert counts["b"] == 0


def test_check_constant_features():
    df = pd.DataFrame({"const": [1.0, 1.0, 1.0], "varying": [1.0, 2.0, 3.0]})
    assert check_constant_features(df) == ["const"]


def test_check_near_zero_variance():
    df = pd.DataFrame({"flat": np.full(50, 1.0) + RNG.normal(0, 1e-10, 50), "normal": RNG.normal(0, 1, 50)})
    flagged = check_near_zero_variance(df, threshold=1e-6)
    assert "flat" in flagged
    assert "normal" not in flagged


def test_check_duplicate_features_exact_and_non_duplicate():
    a = RNG.normal(0, 1, 100)
    df = pd.DataFrame({"a": a, "dup": a.copy(), "different": RNG.normal(0, 1, 100)})
    pairs = check_duplicate_features(df)
    assert ("a", "dup") in pairs
    assert not any("different" in pair for pair in pairs)


def test_check_highly_correlated():
    a = RNG.normal(0, 1, 200)
    df = pd.DataFrame({"a": a, "scaled": a * 3 + 1, "unrelated": RNG.normal(0, 1, 200)})
    corr = compute_correlation_matrix(df)
    pairs = check_highly_correlated(corr, threshold=0.95)
    pair_names = [(p[0], p[1]) for p in pairs]
    assert ("a", "scaled") in pair_names
    assert not any("unrelated" in (p[0], p[1]) for p in pairs)


def test_compute_correlation_matrix_handles_infinite_values_correctly():
    """Regression test: raw pandas .corr() can give an incorrect (NaN)
    result when a column contains inf, instead of the mathematically
    correct value obtained by treating that row as missing. Verified this
    exact case: a=[inf,2,3,4,5], b=[1,2,3,4,5] -- excluding row 0 the two
    are perfectly correlated (1.0), but raw df.corr() gives NaN."""
    a = pd.Series([np.inf, 2.0, 3.0, 4.0, 5.0])
    b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    df = pd.DataFrame({"a": a, "b": b})

    correct_value = a[1:].corr(b[1:])
    assert correct_value == pytest.approx(1.0)

    fixed = compute_correlation_matrix(df)
    assert fixed.loc["a", "b"] == pytest.approx(1.0)


def test_compute_feature_statistics():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    stats = compute_feature_statistics(df)
    assert stats.loc["a", "mean"] == pytest.approx(2.5)
    assert stats.loc["a", "min"] == 1.0
    assert stats.loc["a", "max"] == 4.0


def test_validate_features_end_to_end():
    a = RNG.normal(0, 1, 100)
    df = pd.DataFrame(
        {
            "a": a,
            "dup": a.copy(),
            "const": np.ones(100),
            "with_nan": np.concatenate([[np.nan] * 5, RNG.normal(0, 1, 95)]),
        }
    )
    result = validate_features(df, "TEST", FeatureValidationConfig(correlation_threshold=0.95))
    assert result.n_features == 4
    assert result.n_samples == 100
    issue_types = {issue.issue_type for issue in result.issues}
    assert "constant" in issue_types
    assert "duplicate" in issue_types
    assert "missing_values" in issue_types
    assert len(result.issues_by_type("missing_values")) == 1


# ---------------------------------------------------------------------------
# feature_importance
# ---------------------------------------------------------------------------


def test_compute_feature_importance_categorical_target():
    n = 200
    group = np.array(["30-40"] * 100 + ["50-60"] * 100)
    # feature 'relevant' clearly separates the two groups; 'noise' does not
    relevant = np.concatenate([RNG.normal(0, 1, 100), RNG.normal(10, 1, 100)])
    noise = RNG.normal(0, 1, n)
    matrix = pd.DataFrame({"relevant": relevant, "noise": noise})
    metadata = pd.DataFrame({"intensity_band": group})

    result = compute_feature_importance(matrix, metadata, "TEST", FeatureImportanceConfig(target_column="intensity_band"))
    assert result is not None
    assert result.target_type == "categorical"
    assert result.mutual_information["relevant"] > result.mutual_information["noise"]
    assert result.anova_f_score["relevant"] > result.anova_f_score["noise"]


def test_compute_feature_importance_continuous_target():
    n = 200
    target = RNG.normal(0, 1, n)
    relevant = target * 2 + RNG.normal(0, 0.1, n)
    noise = RNG.normal(0, 1, n)
    matrix = pd.DataFrame({"relevant": relevant, "noise": noise})
    metadata = pd.DataFrame({"rpe_nearest": target})

    result = compute_feature_importance(matrix, metadata, "TEST", FeatureImportanceConfig(target_column="rpe_nearest"))
    assert result is not None
    assert result.target_type == "continuous"
    assert result.mutual_information["relevant"] > result.mutual_information["noise"]


def test_compute_feature_importance_missing_target_column_returns_none():
    matrix = pd.DataFrame({"a": RNG.normal(0, 1, 10)})
    metadata = pd.DataFrame({"other_column": range(10)})
    result = compute_feature_importance(matrix, metadata, "TEST", FeatureImportanceConfig(target_column="intensity_band"))
    assert result is None


def test_compute_feature_importance_excludes_infinite_rows():
    n = 100
    matrix = pd.DataFrame({"a": RNG.normal(0, 1, n)})
    matrix.loc[0, "a"] = np.inf
    metadata = pd.DataFrame({"intensity_band": ["30-40"] * 50 + ["50-60"] * 50})
    result = compute_feature_importance(matrix, metadata, "TEST", FeatureImportanceConfig(target_column="intensity_band"))
    assert result is not None
    assert result.n_samples_excluded == 1
    assert result.n_samples_used == n - 1


# ---------------------------------------------------------------------------
# feature_selection
# ---------------------------------------------------------------------------


def test_recommend_variance_threshold():
    df = pd.DataFrame({"const": np.ones(50), "normal": RNG.normal(0, 1, 50)})
    recs = recommend_variance_threshold(df, threshold=1e-6)
    assert any(r.feature_name == "const" and r.recommendation == "remove" for r in recs)
    assert not any(r.feature_name == "normal" for r in recs)


def test_recommend_correlation_filter():
    a = RNG.normal(0, 1, 100)
    df = pd.DataFrame({"a": a, "dup": a.copy(), "scaled": a * 2 + 0.001, "unrelated": RNG.normal(0, 1, 100)})
    recs = recommend_correlation_filter(df, threshold=0.95)
    assert any(r.feature_name == "dup" and r.recommendation == "remove" for r in recs)
    assert any(r.feature_name == "scaled" and r.recommendation == "review" for r in recs)


def test_recommend_mutual_info_ranking_flags_lowest_scores():
    n = 200
    group = np.array(["a"] * 100 + ["b"] * 100)
    matrix = pd.DataFrame(
        {
            "strong": np.concatenate([RNG.normal(0, 1, 100), RNG.normal(10, 1, 100)]),
            "weak": RNG.normal(0, 1, n),
        }
    )
    metadata = pd.DataFrame({"intensity_band": group})
    importance = compute_feature_importance(matrix, metadata, "TEST", FeatureImportanceConfig(target_column="intensity_band"))
    recs = recommend_mutual_info_ranking(importance, bottom_fraction=0.5)
    flagged_names = {r.feature_name for r in recs}
    assert "weak" in flagged_names


def test_recommend_mutual_info_ranking_handles_none():
    assert recommend_mutual_info_ranking(None, 0.1) == []


def test_feature_selector_combines_all_methods():
    a = RNG.normal(0, 1, 100)
    df = pd.DataFrame({"a": a, "dup": a.copy(), "const": np.ones(100)})
    selector = FeatureSelector(FeatureSelectionConfig(correlation_threshold=0.95))
    recs = selector.recommend(df)
    methods = {r.method for r in recs}
    assert "variance_threshold" in methods
    assert "correlation_filter" in methods


def test_apply_recommendations_only_drops_specified_actions():
    a = RNG.normal(0, 1, 100)
    df = pd.DataFrame({"a": a, "dup": a.copy(), "const": np.ones(100), "keep_me": RNG.normal(0, 1, 100)})
    selector = FeatureSelector()
    recs = selector.recommend(df)
    filtered = apply_recommendations(df, recs, actions=("remove",))
    assert "const" not in filtered.columns
    assert "dup" not in filtered.columns
    assert "keep_me" in filtered.columns
    assert "a" in filtered.columns


def test_apply_recommendations_does_not_mutate_original():
    df = pd.DataFrame({"const": np.ones(20), "normal": RNG.normal(0, 1, 20)})
    original_columns = list(df.columns)
    recs = FeatureSelector().recommend(df)
    apply_recommendations(df, recs)
    assert list(df.columns) == original_columns


# ---------------------------------------------------------------------------
# feature_dataset
# ---------------------------------------------------------------------------


def test_window_center_seconds():
    windowed = WindowedFeatures(window_index=0, window_start_sample=0, window_end_sample=1000, features=FeatureVector((), ()))
    assert window_center_seconds(windowed, sampling_rate_hz=1000.0) == pytest.approx(0.5)


def test_nearest_rpe_none_labels():
    assert np.isnan(nearest_rpe(None, 10.0))


def test_nearest_rpe_picks_closest_point():
    labels = Labels(
        task_order_code="task1_35i",
        rpe_before=6,
        rpe_timeseries={10: 8, 20: 10, 30: 12},
        rpe_at_end_of_trial=20,
        trial_duration_seconds=35.0,
    )
    assert nearest_rpe(labels, 0.0) == 6.0
    assert nearest_rpe(labels, 19.0) == 10.0
    assert nearest_rpe(labels, 35.0) == 20.0


def test_categorize_feature():
    assert categorize_feature("EMG", "mean_frequency") == "frequency_domain"
    assert categorize_feature("EMG", "mav") == "time_domain"
    assert categorize_feature("IMU", "acc_resultant_mean") == "resultant"
    assert categorize_feature("IMU", "acc_x_mean") == "per_axis"
    assert categorize_feature("PPG", "signal_energy") == "amplitude"
    assert categorize_feature("PPG", "heart_rate_bpm") == "peak_based"


def _make_synthetic_condition_with_labels(subject_id: str) -> Condition:
    condition_id = f"{subject_id}_external_rotation_30-40"
    emg = {
        muscle: EMGData(
            subject_id=subject_id, condition_id=condition_id, sampling_rate_hz=1000.0, source_file=Path("x.csv"),
            muscle=muscle, signal=pd.DataFrame({"EMG(mV)": RNG.normal(0, 1, 3000)}),
        )
        for muscle in EMGMuscle
    }
    imu = {
        location: IMUData(
            subject_id=subject_id, condition_id=condition_id, sampling_rate_hz=100.0, source_file=Path("x"),
            location=location,
            accelerometer=pd.DataFrame(RNG.normal(0, 1, (300, 3)), columns=["acc_x", "acc_y", "acc_z"]),
            gyroscope=pd.DataFrame(RNG.normal(0, 1, (300, 3)), columns=["gyr_x", "gyr_y", "gyr_z"]),
            magnetometer=pd.DataFrame(RNG.normal(0, 1, (300, 3)), columns=["mag_x", "mag_y", "mag_z"]),
        )
        for location in IMULocation
    }
    ppg = PPGData(
        subject_id=subject_id, condition_id=condition_id, sampling_rate_hz=200.0, source_file=Path("x.csv"),
        signal=pd.DataFrame({"PPG(Volt)": RNG.normal(0, 1, 3000)}),
    )
    labels = Labels(task_order_code="task4_35e", rpe_before=6, rpe_timeseries={10: 8, 20: 12}, rpe_at_end_of_trial=20, trial_duration_seconds=25.0)
    return Condition(
        condition_id=condition_id, subject_id=subject_id, condition_type=ConditionType.FATIGUE_TASK,
        direction=Direction.EXTERNAL_ROTATION, intensity_band=IntensityBand.BAND_30_40, trial_number=None,
        task_order_code="task4_35e", emg=emg, imu=imu, ppg=ppg, labels=labels,
    )


def test_build_feature_datasets_end_to_end():
    condition = _make_synthetic_condition_with_labels("P99")
    subject = Subject(subject_id="P99", demographic=None, anthropometric=None, mvic_force=None, conditions={condition.condition_id: condition})

    preprocessing_pipeline = PreprocessingPipeline()
    processed_subject: ProcessedSubject = preprocessing_pipeline.process_subject(subject)

    feature_pipeline = FeaturePipeline()
    subject_features = feature_pipeline.extract_subject(processed_subject)

    datasets = build_feature_datasets([subject_features], [processed_subject])

    assert set(datasets.keys()) == {"EMG", "IMU", "PPG"}
    assert not datasets["EMG"].matrix.empty
    assert not datasets["IMU"].matrix.empty
    assert not datasets["PPG"].matrix.empty

    for modality, dataset in datasets.items():
        assert list(dataset.metadata.columns) == list(METADATA_COLUMNS)
        assert len(dataset.matrix) == len(dataset.metadata)
        assert (dataset.metadata["subject_id"] == "P99").all()

    # Fatigue-task condition: rpe_nearest should be populated (not all NaN)
    assert datasets["EMG"].metadata["rpe_nearest"].notna().any()

    catalog = build_feature_catalog(datasets)
    assert set(catalog["modality"].unique()) == {"EMG", "IMU", "PPG"}
    assert len(catalog) == sum(len(d.feature_names) for d in datasets.values())


def test_build_feature_datasets_mvic_condition_has_nan_rpe():
    condition_id = "P99_mvic_external_rotation_trial1"
    emg = {
        muscle: EMGData(
            subject_id="P99", condition_id=condition_id, sampling_rate_hz=1000.0, source_file=Path("x.csv"),
            muscle=muscle, signal=pd.DataFrame({"EMG(mV)": RNG.normal(0, 1, 2000)}),
        )
        for muscle in EMGMuscle
    }
    condition = Condition(
        condition_id=condition_id, subject_id="P99", condition_type=ConditionType.MVIC_CALIBRATION,
        direction=Direction.EXTERNAL_ROTATION, intensity_band=None, trial_number=1, task_order_code=None,
        emg=emg, imu={}, ppg=None, labels=None,
    )
    subject = Subject(subject_id="P99", demographic=None, anthropometric=None, mvic_force=None, conditions={condition_id: condition})
    processed_subject = PreprocessingPipeline().process_subject(subject)
    subject_features = FeaturePipeline().extract_subject(processed_subject)

    datasets = build_feature_datasets([subject_features], [processed_subject])
    assert datasets["EMG"].metadata["rpe_nearest"].isna().all()
