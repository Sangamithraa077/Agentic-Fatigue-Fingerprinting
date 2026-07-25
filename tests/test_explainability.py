"""Unit tests for ml.explainability (Phase 11 explainable AI framework):
config, shap_explainer, feature_ranker, global_explainer, local_explainer,
attention_visualizer, explanation_report.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from ml.explainability.attention_visualizer import (
    AttentionMap,
    compute_attention_distributions,
    compute_modality_importance,
    generate_attention_map,
    plot_attention_distribution,
    plot_attention_heatmap,
    plot_modality_importance,
)
from ml.explainability.config import FeatureRankingConfig, ShapConfig
from ml.explainability.explanation_report import (
    ReportSection,
    format_feature_ranking_table,
    format_local_explanation_table,
    format_modality_importance_table,
    generate_report,
)
from ml.explainability.feature_ranker import RankedFeature, rank_features
from ml.explainability.global_explainer import explain_global
from ml.explainability.local_explainer import explain_prediction
from ml.explainability.shap_explainer import (
    build_shap_explainer,
    compute_shap_values,
    plot_beeswarm,
    plot_dependence,
    plot_force,
    plot_summary,
    plot_waterfall,
    select_explainer_type,
    subsample_rows,
)
from ml.models.classical.logistic_regression import build_logistic_regression
from ml.models.classical.random_forest import build_random_forest
from ml.models.classical.xgboost import build_xgboost
from ml.models.proposed.config import AttentionConfig, ClassifierConfig, EncoderConfig, FusionConfig, ProposedModelConfig
from ml.models.proposed.model import MODALITY_NAMES, ProposedFatigueModel

RNG = np.random.default_rng(11)
FEATURE_NAMES = tuple(f"feature_{i}" for i in range(6))


def _synthetic_dataset(n_samples: int = 80, n_features: int = 6):
    X = RNG.normal(size=(n_samples, n_features))
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_shap_config_defaults():
    config = ShapConfig()
    assert config.background_samples == 100
    assert config.positive_class_index == 1


def test_shap_config_rejects_non_positive_background_samples():
    with pytest.raises(ValueError):
        ShapConfig(background_samples=0)


def test_feature_ranking_config_rejects_non_positive_top_k():
    with pytest.raises(ValueError):
        FeatureRankingConfig(top_k=0)


# ---------------------------------------------------------------------------
# shap_explainer
# ---------------------------------------------------------------------------


def test_select_explainer_type_dispatch():
    assert select_explainer_type("random_forest") == "tree"
    assert select_explainer_type("xgboost") == "tree"
    assert select_explainer_type("logistic_regression") == "linear"
    assert select_explainer_type("some_future_model") == "kernel"


def test_subsample_rows_no_op_when_small_enough():
    X = RNG.normal(size=(10, 4))
    out = subsample_rows(X, max_rows=20, random_seed=1)
    assert out.shape == X.shape
    np.testing.assert_array_equal(out, X)


def test_subsample_rows_subsamples_when_larger():
    X = RNG.normal(size=(50, 4))
    out = subsample_rows(X, max_rows=10, random_seed=1)
    assert out.shape == (10, 4)


def test_subsample_rows_is_deterministic_given_seed():
    X = RNG.normal(size=(50, 4))
    out1 = subsample_rows(X, max_rows=10, random_seed=7)
    out2 = subsample_rows(X, max_rows=10, random_seed=7)
    np.testing.assert_array_equal(out1, out2)


def test_build_shap_explainer_tree_for_random_forest():
    X, y = _synthetic_dataset()
    model = build_random_forest()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "random_forest", X, ShapConfig(background_samples=20))
    assert bundle.explainer_type == "tree"
    assert bundle.model_name == "random_forest"


def test_build_shap_explainer_tree_for_xgboost():
    X, y = _synthetic_dataset()
    model = build_xgboost()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "xgboost", X, ShapConfig(background_samples=20))
    assert bundle.explainer_type == "tree"


def test_build_shap_explainer_linear_for_logistic_regression():
    X, y = _synthetic_dataset()
    model = build_logistic_regression()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "logistic_regression", X, ShapConfig(background_samples=20))
    assert bundle.explainer_type == "linear"


def test_build_shap_explainer_kernel_fallback_for_unknown_model():
    X, y = _synthetic_dataset(n_samples=30)
    model = LogisticRegression().fit(X, y)
    bundle = build_shap_explainer(model, "unregistered_model", X, ShapConfig(background_samples=10))
    assert bundle.explainer_type == "kernel"


def test_compute_shap_values_normalizes_tree_output_to_2d():
    X, y = _synthetic_dataset()
    model = build_random_forest()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "random_forest", X, ShapConfig(background_samples=20))
    explanation = compute_shap_values(bundle, X[:15], FEATURE_NAMES)
    assert explanation.values.shape == (15, len(FEATURE_NAMES))
    assert list(explanation.feature_names) == list(FEATURE_NAMES)


def test_compute_shap_values_normalizes_linear_output_to_2d():
    X, y = _synthetic_dataset()
    model = build_logistic_regression()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "logistic_regression", X, ShapConfig(background_samples=20))
    explanation = compute_shap_values(bundle, X[:15], FEATURE_NAMES)
    assert explanation.values.shape == (15, len(FEATURE_NAMES))


def test_compute_shap_values_reconstructs_prediction_margin():
    # Sanity check on the SHAP additivity property: base_value + sum(shap
    # values) should reconstruct the tree ensemble's raw per-sample output.
    X, y = _synthetic_dataset()
    model = build_random_forest()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "random_forest", X, ShapConfig(background_samples=20))
    explanation = compute_shap_values(bundle, X[:5], FEATURE_NAMES)
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    predicted_proba_positive = model.predict_proba(X[:5])[:, 1]
    np.testing.assert_allclose(reconstructed, predicted_proba_positive, atol=1e-6)


@pytest.fixture
def rf_shap_setup(tmp_path):
    X, y = _synthetic_dataset(n_samples=100)
    model = build_random_forest()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "random_forest", X, ShapConfig(background_samples=30))
    explanation = compute_shap_values(bundle, X[:25], FEATURE_NAMES)
    return model, bundle, explanation, X, y, tmp_path


def test_plot_summary_saves_file(rf_shap_setup):
    _, _, explanation, _, _, tmp_path = rf_shap_setup
    path = plot_summary(explanation, tmp_path / "summary.png")
    assert path.exists()


def test_plot_beeswarm_saves_file(rf_shap_setup):
    _, _, explanation, _, _, tmp_path = rf_shap_setup
    path = plot_beeswarm(explanation, tmp_path / "beeswarm.png")
    assert path.exists()


def test_plot_waterfall_saves_file(rf_shap_setup):
    _, _, explanation, _, _, tmp_path = rf_shap_setup
    path = plot_waterfall(explanation[0], tmp_path / "waterfall.png")
    assert path.exists()


def test_plot_force_saves_file(rf_shap_setup):
    _, _, explanation, _, _, tmp_path = rf_shap_setup
    path = plot_force(explanation[0], tmp_path / "force.png")
    assert path.exists()


def test_plot_dependence_saves_file(rf_shap_setup):
    _, _, explanation, _, _, tmp_path = rf_shap_setup
    path = plot_dependence(FEATURE_NAMES[0], explanation, tmp_path / "dependence.png")
    assert path.exists()


# ---------------------------------------------------------------------------
# feature_ranker
# ---------------------------------------------------------------------------


def test_rank_features_orders_by_mean_absolute_value_descending():
    shap_values = np.array(
        [
            [0.1, -0.9, 0.0],
            [-0.2, 0.8, 0.1],
            [0.05, 1.0, -0.05],
        ]
    )
    ranking = rank_features(shap_values, ("low", "high", "tiny"))
    assert [f.feature_name for f in ranking] == ["high", "low", "tiny"]
    assert [f.rank for f in ranking] == [0, 1, 2]
    assert ranking[0].importance == pytest.approx(0.9, abs=1e-6)


def test_rank_features_returns_every_feature_not_just_top_k():
    shap_values = RNG.normal(size=(10, 8))
    names = tuple(f"f{i}" for i in range(8))
    ranking = rank_features(shap_values, names, FeatureRankingConfig(top_k=3))
    assert len(ranking) == 8


def test_rank_features_rejects_mismatched_feature_names():
    shap_values = RNG.normal(size=(10, 4))
    with pytest.raises(ValueError):
        rank_features(shap_values, ("only_one_name",))


def test_rank_features_rejects_non_2d_input():
    with pytest.raises(ValueError):
        rank_features(RNG.normal(size=(10,)), ("f0",))


# ---------------------------------------------------------------------------
# global_explainer
# ---------------------------------------------------------------------------


def test_explain_global_returns_full_ranking_and_bounded_sample_count():
    X, y = _synthetic_dataset(n_samples=60)
    model = build_random_forest()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "random_forest", X, ShapConfig(background_samples=20))

    result = explain_global(bundle, X, FEATURE_NAMES, ShapConfig(background_samples=20, max_explain_samples=15))

    assert result.model_name == "random_forest"
    assert result.n_samples_explained == 15
    assert len(result.feature_ranking) == len(FEATURE_NAMES)
    assert result.explanation.values.shape == (15, len(FEATURE_NAMES))


def test_explain_global_does_not_subsample_when_data_is_small():
    X, y = _synthetic_dataset(n_samples=10)
    model = build_logistic_regression()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "logistic_regression", X, ShapConfig(background_samples=8))

    result = explain_global(bundle, X, FEATURE_NAMES, ShapConfig(background_samples=8, max_explain_samples=100))
    assert result.n_samples_explained == 10


# ---------------------------------------------------------------------------
# local_explainer
# ---------------------------------------------------------------------------


def test_explain_prediction_matches_model_predicted_class():
    X, y = _synthetic_dataset(n_samples=60)
    model = build_random_forest()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "random_forest", X, ShapConfig(background_samples=20))

    x_instance = X[0]
    result = explain_prediction(bundle, model, x_instance, FEATURE_NAMES)

    expected_class = int(np.argmax(model.predict_proba(x_instance.reshape(1, -1))[0]))
    assert result.predicted_class == expected_class
    assert result.predicted_probability == pytest.approx(model.predict_proba(x_instance.reshape(1, -1))[0, expected_class])


def test_explain_prediction_contributions_are_sorted_by_absolute_value():
    X, y = _synthetic_dataset(n_samples=60)
    model = build_random_forest()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "random_forest", X, ShapConfig(background_samples=20))

    result = explain_prediction(bundle, model, X[3], FEATURE_NAMES)
    magnitudes = [abs(c.shap_value) for c in result.top_contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert len(result.top_contributions) == len(FEATURE_NAMES)


def test_explain_prediction_reconstructs_probability_via_base_value():
    X, y = _synthetic_dataset(n_samples=60)
    model = build_random_forest()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "random_forest", X, ShapConfig(background_samples=20))

    result = explain_prediction(bundle, model, X[0], FEATURE_NAMES)
    reconstructed = result.base_value + sum(c.shap_value for c in result.top_contributions)
    positive_proba = model.predict_proba(X[0].reshape(1, -1))[0, 1]
    assert reconstructed == pytest.approx(positive_proba, abs=1e-6)


# ---------------------------------------------------------------------------
# attention_visualizer
# ---------------------------------------------------------------------------


def _synthetic_attention_weights(n_layers: int, batch: int, heads: int, n_modalities: int = 3):
    """Softmax-normalized synthetic attention weights, matching the real
    shape `CrossModalAttention` produces: (batch, heads, n_modalities, n_modalities)."""
    weights = []
    for _ in range(n_layers):
        raw = torch.randn(batch, heads, n_modalities, n_modalities)
        weights.append(torch.softmax(raw, dim=-1))
    return weights


def test_generate_attention_map_shapes_and_labels():
    weights = _synthetic_attention_weights(n_layers=2, batch=4, heads=3)
    maps = generate_attention_map(weights)
    assert len(maps) == 2
    for layer_index, attention_map in enumerate(maps):
        assert attention_map.layer_index == layer_index
        assert attention_map.matrix.shape == (3, 3)
        assert attention_map.modality_names == tuple(MODALITY_NAMES)


def test_compute_modality_importance_normalizes_to_one():
    weights = _synthetic_attention_weights(n_layers=2, batch=4, heads=3)
    importances = compute_modality_importance(weights)
    assert len(importances) == 3
    total = sum(item.importance for item in importances)
    assert total == pytest.approx(1.0, abs=1e-5)


def test_compute_attention_distributions_covers_every_layer_and_head():
    weights = _synthetic_attention_weights(n_layers=2, batch=4, heads=3)
    distributions = compute_attention_distributions(weights)
    assert len(distributions) == 2 * 3
    pairs = {(d.layer_index, d.head_index) for d in distributions}
    assert pairs == {(l, h) for l in range(2) for h in range(3)}
    for d in distributions:
        assert 0.0 <= d.min <= d.mean <= d.max <= 1.0


def test_attention_visualizer_integrates_with_real_proposed_model():
    # End-to-end check that ProposedFatigueModel's actual attention_weights
    # output is directly consumable by this module -- not just synthetic
    # tensors of the right shape.
    config = ProposedModelConfig(
        emg_encoder=EncoderConfig(hidden_dims=(8,), output_dim=4),
        imu_encoder=EncoderConfig(hidden_dims=(8,), output_dim=4),
        ppg_encoder=EncoderConfig(hidden_dims=(4,), output_dim=4),
        fusion=FusionConfig(strategy="attention", attention=AttentionConfig(embed_dim=4, num_heads=2, num_layers=2)),
        classifier=ClassifierConfig(hidden_dims=(4,)),
        n_classes=2,
    )
    model = ProposedFatigueModel(emg_dim=10, imu_dim=12, ppg_dim=5, config=config)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(6, 10), torch.randn(6, 12), torch.randn(6, 5))

    assert output.attention_weights is not None
    maps = generate_attention_map(output.attention_weights)
    assert len(maps) == 2
    importances = compute_modality_importance(output.attention_weights)
    assert len(importances) == 3


def test_plot_attention_heatmap_saves_file(tmp_path):
    attention_map = AttentionMap(layer_index=0, matrix=np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]]), modality_names=("emg", "imu", "ppg"))
    path = plot_attention_heatmap(attention_map, tmp_path / "heatmap.png")
    assert path.exists()


def test_plot_modality_importance_saves_file(tmp_path):
    weights = _synthetic_attention_weights(n_layers=1, batch=3, heads=2)
    importances = compute_modality_importance(weights)
    path = plot_modality_importance(importances, tmp_path / "importance.png")
    assert path.exists()


def test_plot_attention_distribution_saves_file(tmp_path):
    weights = _synthetic_attention_weights(n_layers=1, batch=3, heads=2)
    distributions = compute_attention_distributions(weights)
    path = plot_attention_distribution(distributions, tmp_path / "distribution.png")
    assert path.exists()


# ---------------------------------------------------------------------------
# explanation_report
# ---------------------------------------------------------------------------


def test_generate_report_writes_titled_sections(tmp_path):
    sections = [ReportSection(title="Overview", content="Some overview text."), ReportSection(title="Results", content="Some results text.")]
    output_path = generate_report(sections, tmp_path / "report.md", report_title="Test Report", preamble="Reproduce with `x`.")

    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("# Test Report")
    assert "Reproduce with `x`." in text
    assert "## Overview" in text
    assert "Some overview text." in text
    assert "## Results" in text
    assert text.index("## Overview") < text.index("## Results")


def test_format_feature_ranking_table_respects_top_k():
    ranking = tuple(RankedFeature(feature_name=f"f{i}", importance=1.0 - i * 0.1, rank=i) for i in range(5))
    table = format_feature_ranking_table(ranking, top_k=2)
    assert "f0" in table
    assert "f1" in table
    assert "f2" not in table


def test_format_modality_importance_table_sorted_descending():
    from ml.explainability.attention_visualizer import ModalityImportance

    importances = (
        ModalityImportance(modality_name="ppg", importance=0.2),
        ModalityImportance(modality_name="emg", importance=0.5),
        ModalityImportance(modality_name="imu", importance=0.3),
    )
    table = format_modality_importance_table(importances)
    assert table.index("emg") < table.index("imu") < table.index("ppg")


def test_format_local_explanation_table_includes_prediction_and_contributions():
    X, y = _synthetic_dataset(n_samples=60)
    model = build_random_forest()
    model.fit(X, y)
    bundle = build_shap_explainer(model, "random_forest", X, ShapConfig(background_samples=20))
    local = explain_prediction(bundle, model, X[0], FEATURE_NAMES)

    table = format_local_explanation_table(local, top_k=3)
    assert f"**{local.predicted_class}**" in table
    assert local.top_contributions[0].feature_name in table
