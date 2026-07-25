"""Unit tests for ml.models.common (Phase 9 baseline modeling framework):
config, dataset_builder, data_splitter, metrics, callbacks, evaluation,
model_registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering.feature_dataset import FeatureDataset
from ml.models.common.callbacks import EarlyStopping, ModelCheckpoint
from ml.models.common.config import DatasetConfig, SplitConfig
from ml.models.common.data_splitter import stratified_group_kfold_split, train_val_test_group_split
from ml.models.common.dataset_builder import (
    apply_scaler,
    build_modeling_dataset,
    derive_target,
    fit_scaler,
    select_feature_subset,
)
from ml.models.common.evaluation import evaluate_model
from ml.models.common.metrics import compute_classification_metrics, measure_time_and_memory
from ml.models.common.model_registry import ModelRegistry, ModelSpec, build_default_registry

RNG = np.random.default_rng(5)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_dataset_config_rejects_invalid_modality():
    with pytest.raises(ValueError):
        DatasetConfig(modality="BAD")


def test_dataset_config_rejects_invalid_feature_subset():
    with pytest.raises(ValueError):
        DatasetConfig(modality="EMG", feature_subset="bad")


def test_dataset_config_rejects_invalid_target_type():
    with pytest.raises(ValueError):
        DatasetConfig(modality="EMG", target_type="bad")


def test_split_config_requires_ratios_sum_to_one():
    with pytest.raises(ValueError):
        SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_split_config_requires_at_least_two_folds():
    with pytest.raises(ValueError):
        SplitConfig(n_splits=1)


# ---------------------------------------------------------------------------
# dataset_builder
# ---------------------------------------------------------------------------


def _make_metadata(n: int, n_subjects: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": RNG.choice([f"P{i:02d}" for i in range(n_subjects)], n),
            "condition_id": "cond1",
            "condition_type": "fatigue_task",
            "direction": "external_rotation",
            "intensity_band": RNG.choice(["30-40", "40-50", "50-60"], n),
            "trial_number": None,
            "source_sensor": "anterior_deltoid",
            "window_index": np.arange(n),
            "rpe_nearest": RNG.choice([6.0, 10.0, 14.0, 18.0, np.nan], n),
        }
    )


def test_derive_target_binary_fatigue():
    metadata = _make_metadata(200)
    config = DatasetConfig(modality="EMG", target_type="binary_fatigue", binary_fatigue_threshold=13.0)
    y, valid_mask, class_names = derive_target(metadata, config)
    assert class_names == ("not_fatigued", "fatigued")
    rpe = metadata["rpe_nearest"].to_numpy()
    assert np.array_equal(valid_mask, ~np.isnan(rpe))
    assert set(y[valid_mask].tolist()) <= {0, 1}
    assert (y[valid_mask & (rpe >= 13)] == 1).all()
    assert (y[valid_mask & (rpe < 13)] == 0).all()


def test_derive_target_fatigue_level_bins():
    metadata = _make_metadata(200)
    config = DatasetConfig(modality="EMG", target_type="fatigue_level")
    y, valid_mask, class_names = derive_target(metadata, config)
    assert class_names == ("low", "moderate", "high")
    assert set(y[valid_mask].tolist()) <= {0, 1, 2}


def test_derive_target_borg_rpe_class_is_label_encoded():
    metadata = _make_metadata(200)
    config = DatasetConfig(modality="EMG", target_type="borg_rpe_class")
    y, valid_mask, class_names = derive_target(metadata, config)
    assert set(y[valid_mask].tolist()) == set(range(len(class_names)))
    assert class_names == ("6", "10", "14", "18")


def test_select_feature_subset_all_keeps_every_column():
    matrix = pd.DataFrame({"a": RNG.normal(0, 1, 50), "b": RNG.normal(0, 1, 50)})
    config = DatasetConfig(modality="EMG", feature_subset="all")
    X, names = select_feature_subset(matrix, config, np.zeros(50), np.ones(50, dtype=bool))
    assert names == ("a", "b")
    assert X.shape == (50, 2)


def test_select_feature_subset_variance_filtered_drops_constant():
    matrix = pd.DataFrame({"a": RNG.normal(0, 1, 50), "const": np.ones(50)})
    config = DatasetConfig(modality="EMG", feature_subset="variance_filtered")
    X, names = select_feature_subset(matrix, config, np.zeros(50), np.ones(50, dtype=bool))
    assert "const" not in names
    assert "a" in names


def test_select_feature_subset_mutual_info_ranked_keeps_top_fraction():
    n = 200
    strong = np.concatenate([RNG.normal(0, 1, 100), RNG.normal(10, 1, 100)])
    weak = RNG.normal(0, 1, n)
    matrix = pd.DataFrame({"strong": strong, "weak": weak, "weak2": RNG.normal(0, 1, n), "weak3": RNG.normal(0, 1, n)})
    y = np.array([0] * 100 + [1] * 100)
    config = DatasetConfig(modality="EMG", feature_subset="mutual_info_ranked", mi_top_fraction=0.25)
    X, names = select_feature_subset(matrix, config, y, np.ones(n, dtype=bool))
    assert "strong" in names
    assert len(names) == 1


def test_build_modeling_dataset_end_to_end():
    n = 300
    metadata = _make_metadata(n)
    matrix = pd.DataFrame({"a": RNG.normal(0, 1, n), "b": RNG.normal(0, 1, n)})
    dataset = FeatureDataset(modality="EMG", feature_names=("a", "b"), matrix=matrix, metadata=metadata)
    config = DatasetConfig(modality="EMG", target_type="binary_fatigue")

    modeling_dataset = build_modeling_dataset(dataset, config)
    assert not np.isnan(modeling_dataset.X).any()
    assert not np.isinf(modeling_dataset.X).any()
    assert len(modeling_dataset.y) == len(modeling_dataset.X) == len(modeling_dataset.groups)
    assert modeling_dataset.n_rows_excluded == int(metadata["rpe_nearest"].isna().sum())


def test_build_modeling_dataset_rejects_empty_matrix():
    dataset = FeatureDataset(modality="EMG", feature_names=(), matrix=pd.DataFrame(), metadata=pd.DataFrame())
    config = DatasetConfig(modality="EMG")
    with pytest.raises(ValueError):
        build_modeling_dataset(dataset, config)


def test_build_modeling_dataset_rejects_modality_mismatch():
    n = 50
    metadata = _make_metadata(n)
    matrix = pd.DataFrame({"a": RNG.normal(0, 1, n)})
    dataset = FeatureDataset(modality="EMG", feature_names=("a",), matrix=matrix, metadata=metadata)
    config = DatasetConfig(modality="IMU")
    with pytest.raises(ValueError):
        build_modeling_dataset(dataset, config)


def test_build_modeling_dataset_imputes_inf_and_nan():
    n = 200
    metadata = _make_metadata(n)
    values = RNG.normal(0, 1, n)
    values[0] = np.inf
    values[1] = np.nan
    matrix = pd.DataFrame({"a": values, "b": RNG.normal(0, 1, n)})
    dataset = FeatureDataset(modality="EMG", feature_names=("a", "b"), matrix=matrix, metadata=metadata)
    config = DatasetConfig(modality="EMG", target_type="binary_fatigue")
    modeling_dataset = build_modeling_dataset(dataset, config)
    assert modeling_dataset.n_values_imputed >= 2
    assert not np.isnan(modeling_dataset.X).any()
    assert not np.isinf(modeling_dataset.X).any()


def test_fit_and_apply_scaler():
    X_train = RNG.normal(5, 2, (100, 3))
    X_test = RNG.normal(5, 2, (20, 3))
    scaler = fit_scaler(X_train)
    scaled_train = apply_scaler(scaler, X_train)
    scaled_test = apply_scaler(scaler, X_test)
    assert scaled_train.shape == X_train.shape
    assert scaled_test.shape == X_test.shape


# ---------------------------------------------------------------------------
# data_splitter
# ---------------------------------------------------------------------------


def test_stratified_group_kfold_no_group_leakage():
    n_subjects = 10
    groups = np.repeat([f"P{i:02d}" for i in range(n_subjects)], 40)
    y = RNG.integers(0, 2, len(groups))
    folds = stratified_group_kfold_split(y, groups, SplitConfig(n_splits=5))
    assert len(folds) == 5
    for fold in folds:
        assert not (set(groups[fold.train_idx]) & set(groups[fold.test_idx]))


def test_stratified_group_kfold_raises_when_too_few_groups():
    groups = np.array(["P1"] * 10 + ["P2"] * 10)
    y = RNG.integers(0, 2, 20)
    with pytest.raises(ValueError):
        stratified_group_kfold_split(y, groups, SplitConfig(n_splits=5))


def test_train_val_test_group_split_no_leakage_and_correct_proportions():
    n_subjects = 12
    groups = np.repeat([f"P{i:02d}" for i in range(n_subjects)], 50)
    y = RNG.integers(0, 2, len(groups))
    split = train_val_test_group_split(y, groups, SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15))
    train_g, val_g, test_g = set(groups[split.train_idx]), set(groups[split.val_idx]), set(groups[split.test_idx])
    assert not (train_g & val_g) and not (train_g & test_g) and not (val_g & test_g)
    total = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    assert total == len(y)


def test_train_val_test_group_split_raises_when_too_few_groups():
    groups = np.array(["P1"] * 10 + ["P2"] * 10)
    y = RNG.integers(0, 2, 20)
    with pytest.raises(ValueError):
        train_val_test_group_split(y, groups, SplitConfig())


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_compute_classification_metrics_binary():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert 0 <= metrics.accuracy <= 1
    assert metrics.roc_auc is not None
    assert metrics.pr_auc is not None
    assert metrics.confusion_matrix_.shape == (2, 2)
    assert metrics.n_samples == 5
    assert metrics.n_classes == 2


def test_compute_classification_metrics_without_proba():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=None)
    assert metrics.roc_auc is None
    assert metrics.pr_auc is None


def test_compute_classification_metrics_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    y_proba = np.random.default_rng(1).dirichlet(np.ones(3), size=6)
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert metrics.n_classes == 3
    assert metrics.confusion_matrix_.shape == (3, 3)


def test_measure_time_and_memory_context_manager():
    with measure_time_and_memory() as result:
        _ = [i**2 for i in range(1000)]
    assert result["seconds"] >= 0
    assert result["peak_memory_mb"] is None or result["peak_memory_mb"] >= 0


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2, mode="min")
    results = [stopper.step(v) for v in [1.0, 0.9, 0.95, 0.96, 0.97]]
    assert results[-1] is True
    assert stopper.best_value == pytest.approx(0.9)


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper.step(1.0) is False
    assert stopper.step(0.9) is False
    assert stopper.step(0.95) is False  # 1 epoch without improvement
    assert stopper.step(0.5) is False  # improvement resets patience
    assert stopper.best_value == pytest.approx(0.5)


def test_early_stopping_rejects_invalid_mode():
    with pytest.raises(ValueError):
        EarlyStopping(mode="bad")


def test_model_checkpoint_saves_only_on_improvement(tmp_path):
    checkpoint = ModelCheckpoint(tmp_path / "model.pt", mode="min")
    save_calls = []
    assert checkpoint.step(1.0, save_calls.append) is True
    assert checkpoint.step(1.5, save_calls.append) is False
    assert checkpoint.step(0.5, save_calls.append) is True
    assert len(save_calls) == 2


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal predictor for testing evaluate_model without a real fitted model."""

    def predict(self, X):
        return (X[:, 0] > 0).astype(int)

    def predict_proba(self, X):
        p1 = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p1, p1])


def test_evaluate_model_returns_full_result():
    X_test = RNG.normal(0, 1, (50, 3))
    y_test = (X_test[:, 0] > 0).astype(int)
    config = DatasetConfig(modality="EMG")
    result = evaluate_model(_StubModel(), X_test, y_test, "stub_model", config, training_seconds=1.23)
    assert result.model_name == "stub_model"
    assert result.timing.training_seconds == 1.23
    assert result.metrics.accuracy == pytest.approx(1.0)
    assert result.y_proba is not None


# ---------------------------------------------------------------------------
# model_registry
# ---------------------------------------------------------------------------


def test_model_registry_register_and_get():
    registry = ModelRegistry()
    registry.register(ModelSpec("foo", "classical", "test model"))
    assert registry.get("foo").name == "foo"
    with pytest.raises(KeyError):
        registry.get("missing")


def test_model_registry_rejects_duplicate_registration():
    registry = ModelRegistry()
    registry.register(ModelSpec("foo", "classical", "test model"))
    with pytest.raises(ValueError):
        registry.register(ModelSpec("foo", "classical", "duplicate"))


def test_model_registry_list_names_filters_by_category_and_availability():
    registry = ModelRegistry()
    registry.register(ModelSpec("a", "classical", "x", is_available=True))
    registry.register(ModelSpec("b", "classical", "y", is_available=False, unavailable_reason="missing dep"))
    registry.register(ModelSpec("c", "deep_learning", "z"))
    assert registry.list_names("classical") == ["a", "b"]
    assert registry.list_names("classical", available_only=True) == ["a"]
    assert registry.list_names() == ["a", "b", "c"]


def test_build_default_registry_has_all_expected_models():
    registry = build_default_registry()
    assert registry.list_names("classical") == ["logistic_regression", "random_forest", "svm", "xgboost"]
    assert registry.list_names("deep_learning") == ["bilstm", "cnn1d", "cnn_bilstm", "tcn"]
