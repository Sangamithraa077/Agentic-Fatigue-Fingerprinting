"""Unit tests for ml.personalization (Phase 12 personalization framework):
config, baseline_builder, subject_profiles, loso, personalization_model,
similarity, evaluation, report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data_models.anthropometry import AnthropometricData
from ml.data_models.demographics import DemographicData
from ml.data_models.enums import Sex
from ml.data_models.subject import Subject
from ml.feature_engineering.feature_dataset import METADATA_COLUMNS, FeatureDataset
from ml.models.common.config import DatasetConfig
from ml.models.common.dataset_builder import ModelingDataset
from ml.personalization.baseline_builder import FatigueHistory, ModalityBaseline, build_fatigue_history, build_modality_baseline
from ml.personalization.config import LosoConfig, PersonalizationConfig, SimilarityConfig
from ml.personalization.evaluation import compare_generic_vs_personalized, summarize_personalization_evaluation
from ml.personalization.loso import evaluate_loso
from ml.personalization.personalization_model import compute_baseline_deviation, predict_personalized
from ml.personalization.report import (
    format_comparison_table,
    format_loso_table,
    format_similarity_table,
    format_subject_profile_table,
    generate_personalization_report,
)
from ml.personalization.similarity import compute_similarity_matrix, find_similar_subjects
from ml.personalization.subject_profiles import SubjectProfile, build_subject_profile

RNG = np.random.default_rng(23)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row(subject_id: str, condition_id: str, condition_type: str, window_index: int, rpe_nearest: float, **features) -> dict:
    row = {
        "subject_id": subject_id, "condition_id": condition_id, "condition_type": condition_type,
        "direction": "internal_rotation", "intensity_band": "30-40", "trial_number": None,
        "source_sensor": "anterior_deltoid", "window_index": window_index, "rpe_nearest": rpe_nearest,
    }
    row.update(features)
    return row


def _make_feature_dataset(modality: str, rows, feature_names) -> FeatureDataset:
    df = pd.DataFrame(rows)
    metadata = df[list(METADATA_COLUMNS)].reset_index(drop=True)
    matrix = df[list(feature_names)].reset_index(drop=True)
    return FeatureDataset(modality=modality, feature_names=tuple(feature_names), matrix=matrix, metadata=metadata)


def _synthetic_emg_dataset() -> FeatureDataset:
    rows = [
        _row("P01", "P01_c1", "fatigue_task", 0, 10.0, mav=1.0, rms=2.0),
        _row("P01", "P01_c1", "fatigue_task", 1, 12.0, mav=1.2, rms=2.2),
        _row("P01", "P01_c2", "fatigue_task", 0, 16.0, mav=1.4, rms=2.4),
        _row("P01", "P01_mvic", "mvic_calibration", 0, float("nan"), mav=5.0, rms=6.0),
        _row("P02", "P02_c1", "fatigue_task", 0, 8.0, mav=3.0, rms=4.0),
        _row("P02", "P02_c1", "fatigue_task", 1, 20.0, mav=3.2, rms=4.2),
    ]
    return _make_feature_dataset("EMG", rows, ("mav", "rms"))


def _demographic(age=30, height_cm=170.0) -> DemographicData:
    return DemographicData(
        age=age, height_cm=height_cm, sex=Sex.FEMALE, dominant_hand="right", execution_order_group=1,
        exercise_type="running", exercise_frequency_per_week=3,
    )


def _anthropometric(mass_kg=65.0, bmi=22.0) -> AnthropometricData:
    return AnthropometricData(mass_kg=mass_kg, body_fat_percent=20.0, muscle_percent=35.0, visceral_fat=5.0, bmi=bmi)


def _subject(subject_id: str, with_demographics: bool = True) -> Subject:
    return Subject(
        subject_id=subject_id,
        demographic=_demographic() if with_demographics else None,
        anthropometric=_anthropometric() if with_demographics else None,
        mvic_force=None,
    )


def _make_profile(subject_id: str, summary_vector) -> SubjectProfile:
    empty_baseline = ModalityBaseline(modality="EMG", feature_means={}, feature_stds={}, n_windows=0)
    empty_history = FatigueHistory(n_trials=0, n_windows=0, mean_rpe=None, max_rpe=None, min_rpe=None)
    return SubjectProfile(
        subject_id=subject_id, demographic=None, anthropometric=None,
        emg_baseline=empty_baseline, imu_baseline=empty_baseline, ppg_baseline=empty_baseline,
        fatigue_history=empty_history, summary_vector=np.array(summary_vector, dtype=float),
    )


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_personalization_config_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        PersonalizationConfig(alpha=1.5)


def test_personalization_config_rejects_non_positive_deviation_scale():
    with pytest.raises(ValueError):
        PersonalizationConfig(deviation_scale=0.0)


def test_similarity_config_rejects_unknown_metric():
    with pytest.raises(ValueError):
        SimilarityConfig(metric="manhattan")


def test_similarity_config_rejects_non_positive_top_k():
    with pytest.raises(ValueError):
        SimilarityConfig(top_k=0)


def test_loso_config_default_dataset_config_is_emg_binary_fatigue():
    config = LosoConfig()
    assert config.dataset_config.modality == "EMG"
    assert config.dataset_config.target_type == "binary_fatigue"


# ---------------------------------------------------------------------------
# baseline_builder
# ---------------------------------------------------------------------------


def test_build_modality_baseline_computes_correct_mean_and_std():
    dataset = _synthetic_emg_dataset()
    baseline = build_modality_baseline(dataset, "P01")
    assert baseline.n_windows == 4
    expected_mav_mean = np.mean([1.0, 1.2, 1.4, 5.0])
    assert baseline.feature_means["mav"] == pytest.approx(expected_mav_mean)
    assert baseline.feature_stds["mav"] == pytest.approx(np.std([1.0, 1.2, 1.4, 5.0], ddof=1))


def test_build_modality_baseline_unknown_subject_is_empty():
    dataset = _synthetic_emg_dataset()
    baseline = build_modality_baseline(dataset, "P99")
    assert baseline.n_windows == 0
    assert baseline.feature_means == {}


def test_build_modality_baseline_empty_dataset():
    empty = FeatureDataset(modality="PPG", feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame(columns=METADATA_COLUMNS))
    baseline = build_modality_baseline(empty, "P01")
    assert baseline.n_windows == 0


def test_build_modality_baseline_clips_extreme_outlier():
    # Regression test: Phase 8.5 confirmed some raw EMG values are
    # individually corrupted to ~1e28-1e34 (finite, so plain NaN/inf checks
    # miss them) -- caught during this phase's own report generation, where
    # an uncleaned baseline put one subject's "variance" mean at ~1e62.
    rows = [
        _row("P01", "P01_c1", "fatigue_task", 0, 10.0, mav=1.0, rms=2.0),
        _row("P01", "P01_c1", "fatigue_task", 1, 11.0, mav=1.1, rms=2.1),
        _row("P01", "P01_c1", "fatigue_task", 2, 12.0, mav=1.2, rms=2.2),
        _row("P01", "P01_c1", "fatigue_task", 3, 13.0, mav=1e30, rms=2.3),  # corrupted
    ]
    dataset = _make_feature_dataset("EMG", rows, ("mav", "rms"))
    baseline = build_modality_baseline(dataset, "P01")
    assert baseline.feature_means["mav"] < 10.0  # clipped, not ~1e30
    assert baseline.feature_stds["mav"] < 10.0


def test_build_fatigue_history_excludes_mvic_and_computes_stats():
    dataset = _synthetic_emg_dataset()
    history = build_fatigue_history(dataset, "P01")
    assert history.n_trials == 2  # P01_c1, P01_c2 -- P01_mvic excluded (mvic_calibration)
    assert history.n_windows == 3
    assert history.mean_rpe == pytest.approx(np.mean([10.0, 12.0, 16.0]))
    assert history.max_rpe == 16.0
    assert history.min_rpe == 10.0


def test_build_fatigue_history_unknown_subject_returns_none_stats():
    dataset = _synthetic_emg_dataset()
    history = build_fatigue_history(dataset, "P99")
    assert history.n_trials == 0
    assert history.mean_rpe is None


# ---------------------------------------------------------------------------
# subject_profiles
# ---------------------------------------------------------------------------


def test_build_subject_profile_populates_all_fields():
    emg = _synthetic_emg_dataset()
    imu = FeatureDataset(modality="IMU", feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame(columns=METADATA_COLUMNS))
    ppg = FeatureDataset(modality="PPG", feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame(columns=METADATA_COLUMNS))
    subject = _subject("P01")

    profile = build_subject_profile(subject, {"EMG": emg, "IMU": imu, "PPG": ppg})

    assert profile.subject_id == "P01"
    assert profile.demographic.age == 30
    assert profile.anthropometric.mass_kg == 65.0
    assert profile.emg_baseline.n_windows == 4
    assert profile.fatigue_history.n_trials == 2
    assert profile.summary_vector.shape == (10,)
    assert profile.summary_vector[0] == 30.0  # age
    assert profile.summary_vector[2] == 65.0  # mass_kg


def test_build_subject_profile_handles_missing_demographics():
    emg = _synthetic_emg_dataset()
    imu = FeatureDataset(modality="IMU", feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame(columns=METADATA_COLUMNS))
    ppg = FeatureDataset(modality="PPG", feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame(columns=METADATA_COLUMNS))
    subject = _subject("P02", with_demographics=False)

    profile = build_subject_profile(subject, {"EMG": emg, "IMU": imu, "PPG": ppg})
    assert profile.demographic is None
    assert profile.summary_vector[0] == 0.0  # age falls back to 0.0
    assert not np.isnan(profile.summary_vector).any()


# ---------------------------------------------------------------------------
# loso
# ---------------------------------------------------------------------------


def _synthetic_modeling_dataset(n_per_subject: int = 20) -> ModelingDataset:
    subjects = ["P01", "P02", "P03"]
    X_parts, y_parts, group_parts = [], [], []
    for i, subject_id in enumerate(subjects):
        offset = i * 3.0
        X_subject = RNG.normal(loc=offset, scale=1.0, size=(n_per_subject, 4))
        # Centered around each subject's own offset so every subject (and
        # every combination of subjects pooled together for a LOSO training
        # fold) has a roughly balanced 50/50 label split -- an offset-only
        # threshold instead made P02/P03 almost entirely one class, which
        # occasionally starved a LOSO training fold of a second class and
        # broke predict_proba's (n, 2) shape assumption.
        y_subject = ((X_subject[:, 0] - offset) + (X_subject[:, 1] - offset) > 0).astype(int)
        X_parts.append(X_subject)
        y_parts.append(y_subject)
        group_parts.append(np.full(n_per_subject, subject_id))

    return ModelingDataset(
        X=np.vstack(X_parts), y=np.concatenate(y_parts), groups=np.concatenate(group_parts),
        feature_names=("f0", "f1", "f2", "f3"), class_names=("not_fatigued", "fatigued"),
        dataset_config=DatasetConfig(modality="EMG", feature_subset="all", target_type="binary_fatigue"),
        n_rows_excluded=0, n_values_imputed=0,
    )


def test_evaluate_loso_produces_one_fold_per_subject():
    modeling_dataset = _synthetic_modeling_dataset()
    summary = evaluate_loso(modeling_dataset, LosoConfig(model_name="random_forest"))
    assert [fold.held_out_subject for fold in summary.folds] == ["P01", "P02", "P03"]
    for fold in summary.folds:
        assert fold.held_out_subject not in fold.train_subjects
        assert set(fold.train_subjects) == {"P01", "P02", "P03"} - {fold.held_out_subject}


def test_evaluate_loso_mean_metrics_are_bounded():
    modeling_dataset = _synthetic_modeling_dataset()
    summary = evaluate_loso(modeling_dataset, LosoConfig(model_name="random_forest"))
    assert 0.0 <= summary.mean_accuracy <= 1.0
    assert 0.0 <= summary.mean_f1 <= 1.0


def test_evaluate_loso_rejects_unknown_model_name():
    modeling_dataset = _synthetic_modeling_dataset()
    with pytest.raises(ValueError):
        evaluate_loso(modeling_dataset, LosoConfig(model_name="not_a_model"))


def test_evaluate_loso_rejects_single_subject():
    modeling_dataset = ModelingDataset(
        X=RNG.normal(size=(10, 3)), y=RNG.integers(0, 2, size=10), groups=np.full(10, "P01"),
        feature_names=("f0", "f1", "f2"), class_names=("a", "b"),
        dataset_config=DatasetConfig(modality="EMG", feature_subset="all", target_type="binary_fatigue"),
        n_rows_excluded=0, n_values_imputed=0,
    )
    with pytest.raises(ValueError):
        evaluate_loso(modeling_dataset)


# ---------------------------------------------------------------------------
# personalization_model
# ---------------------------------------------------------------------------


def test_compute_baseline_deviation_zero_when_at_baseline_mean():
    baseline = ModalityBaseline(modality="EMG", feature_means={"mav": 1.0, "rms": 2.0}, feature_stds={"mav": 0.5, "rms": 0.5}, n_windows=10)
    deviation = compute_baseline_deviation(np.array([1.0, 2.0]), ("mav", "rms"), baseline)
    assert deviation == pytest.approx(0.0)


def test_compute_baseline_deviation_matches_manual_z_score():
    baseline = ModalityBaseline(modality="EMG", feature_means={"mav": 1.0, "rms": 2.0}, feature_stds={"mav": 0.5, "rms": 1.0}, n_windows=10)
    deviation = compute_baseline_deviation(np.array([2.0, 4.0]), ("mav", "rms"), baseline)
    expected = np.mean([abs((2.0 - 1.0) / 0.5), abs((4.0 - 2.0) / 1.0)])
    assert deviation == pytest.approx(expected)


def test_compute_baseline_deviation_empty_baseline_returns_zero():
    empty_baseline = ModalityBaseline(modality="EMG", feature_means={}, feature_stds={}, n_windows=0)
    deviation = compute_baseline_deviation(np.array([1.0, 2.0]), ("mav", "rms"), empty_baseline)
    assert deviation == 0.0


def test_predict_personalized_alpha_one_ignores_deviation():
    baseline = ModalityBaseline(modality="EMG", feature_means={"mav": 1.0}, feature_stds={"mav": 0.5}, n_windows=10)
    result = predict_personalized("P01", 0.73, np.array([100.0]), ("mav",), baseline, PersonalizationConfig(alpha=1.0))
    assert result.personalized_probability == pytest.approx(0.73)


def test_predict_personalized_alpha_zero_uses_only_deviation():
    baseline = ModalityBaseline(modality="EMG", feature_means={"mav": 1.0}, feature_stds={"mav": 0.5}, n_windows=10)
    result = predict_personalized("P01", 0.1, np.array([1.0]), ("mav",), baseline, PersonalizationConfig(alpha=0.0))
    assert result.personalized_probability == pytest.approx(result.deviation_probability)


def test_predict_personalized_class_matches_threshold():
    baseline = ModalityBaseline(modality="EMG", feature_means={"mav": 1.0}, feature_stds={"mav": 0.5}, n_windows=10)
    result = predict_personalized("P01", 0.9, np.array([1.0]), ("mav",), baseline, PersonalizationConfig(alpha=1.0))
    assert result.personalized_class == 1
    result_low = predict_personalized("P01", 0.1, np.array([1.0]), ("mav",), baseline, PersonalizationConfig(alpha=1.0))
    assert result_low.personalized_class == 0


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------


def _three_profiles():
    close_a = _make_profile("A", [30, 170, 65, 22, 20, 35, 12, 1.0, 1.0, 1.0])
    close_b = _make_profile("B", [31, 171, 66, 22, 21, 35, 12, 1.0, 1.0, 1.0])
    far_c = _make_profile("C", [55, 190, 95, 30, 35, 20, 18, 5.0, 5.0, 5.0])
    return [close_a, close_b, far_c]


def test_compute_similarity_matrix_is_symmetric_with_self_similarity_one():
    matrix, subject_ids = compute_similarity_matrix(_three_profiles(), SimilarityConfig(metric="cosine"))
    assert subject_ids == ("A", "B", "C")
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-6)


def test_compute_similarity_matrix_euclidean_self_similarity_one():
    matrix, _ = compute_similarity_matrix(_three_profiles(), SimilarityConfig(metric="euclidean"))
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-8)


def test_find_similar_subjects_excludes_target_and_sorts_descending():
    results = find_similar_subjects("A", _three_profiles(), SimilarityConfig(metric="cosine", top_k=5))
    assert all(r.subject_id != "A" for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].subject_id == "B"  # A and B are the closer pair by construction


def test_find_similar_subjects_respects_top_k():
    results = find_similar_subjects("A", _three_profiles(), SimilarityConfig(top_k=1))
    assert len(results) == 1


def test_find_similar_subjects_unknown_target_raises():
    with pytest.raises(ValueError):
        find_similar_subjects("Z", _three_profiles())


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def test_compare_generic_vs_personalized_detects_improvement():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    generic_probability = np.array([0.6, 0.55, 0.4, 0.45, 0.3, 0.6, 0.4, 0.55])  # mostly wrong
    personalized_probability = np.array([0.1, 0.2, 0.9, 0.8, 0.85, 0.15, 0.9, 0.1])  # matches y_true well

    comparison = compare_generic_vs_personalized("P01", y_true, generic_probability, personalized_probability)
    assert comparison.subject_id == "P01"
    assert comparison.n_samples == 8
    assert comparison.accuracy_improvement > 0
    assert comparison.f1_improvement > 0


def test_compare_generic_vs_personalized_identical_probabilities_zero_improvement():
    y_true = np.array([0, 1, 0, 1])
    probability = np.array([0.2, 0.8, 0.3, 0.7])
    comparison = compare_generic_vs_personalized("P01", y_true, probability, probability)
    assert comparison.accuracy_improvement == pytest.approx(0.0)
    assert comparison.f1_improvement == pytest.approx(0.0)


def test_summarize_personalization_evaluation_aggregates_correctly():
    y_true = np.array([0, 1, 0, 1])
    good = np.array([0.1, 0.9, 0.1, 0.9])
    bad = np.array([0.9, 0.1, 0.9, 0.1])
    improved = compare_generic_vs_personalized("P01", y_true, bad, good)
    unchanged = compare_generic_vs_personalized("P02", y_true, good, good)

    summary = summarize_personalization_evaluation([improved, unchanged])
    assert summary.n_subjects_improved_f1 == 1
    assert summary.mean_f1_improvement == pytest.approx((improved.f1_improvement + 0.0) / 2)


def test_summarize_personalization_evaluation_rejects_empty():
    with pytest.raises(ValueError):
        summarize_personalization_evaluation([])


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_generate_personalization_report_writes_file(tmp_path):
    from ml.explainability.explanation_report import ReportSection

    sections = [ReportSection(title="Overview", content="text")]
    output_path = generate_personalization_report(sections, tmp_path / "report.md", report_title="Test")
    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("# Test")
    assert "## Overview" in text


def test_format_subject_profile_table_includes_subject_ids():
    emg = _synthetic_emg_dataset()
    imu = FeatureDataset(modality="IMU", feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame(columns=METADATA_COLUMNS))
    ppg = FeatureDataset(modality="PPG", feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame(columns=METADATA_COLUMNS))
    profile = build_subject_profile(_subject("P01"), {"EMG": emg, "IMU": imu, "PPG": ppg})
    table = format_subject_profile_table([profile])
    assert "P01" in table
    assert "|" in table


def test_format_loso_table_includes_mean_row():
    modeling_dataset = _synthetic_modeling_dataset()
    summary = evaluate_loso(modeling_dataset, LosoConfig(model_name="random_forest"))
    table = format_loso_table(summary)
    assert "**mean**" in table
    for fold in summary.folds:
        assert fold.held_out_subject in table


def test_format_similarity_table_includes_target_and_results():
    results = find_similar_subjects("A", _three_profiles())
    table = format_similarity_table("A", results)
    assert "**A**" in table
    assert "B" in table


def test_format_comparison_table_includes_deltas():
    y_true = np.array([0, 1, 0, 1])
    good = np.array([0.1, 0.9, 0.1, 0.9])
    bad = np.array([0.9, 0.1, 0.9, 0.1])
    comparison = compare_generic_vs_personalized("P01", y_true, bad, good)
    summary = summarize_personalization_evaluation([comparison])
    table = format_comparison_table(summary)
    assert "P01" in table
    assert "**mean**" in table
