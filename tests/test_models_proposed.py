"""Unit tests for ml.models.proposed (Phase 10 multimodal fusion model):
config, encoders, attention, fusion, classifier, model, losses, training,
inference.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.models.common.multimodal_dataset_builder import ModalityDims
from ml.models.common.config import DatasetConfig
from ml.models.common.evaluation import evaluate_model
from ml.models.deep_learning.training import get_device
from ml.models.proposed.attention import CrossModalAttention, CrossModalAttentionLayer
from ml.models.proposed.classifier import ClassificationHead
from ml.models.proposed.config import (
    AttentionConfig,
    ClassifierConfig,
    EncoderConfig,
    FusionConfig,
    LossConfig,
    ProposedModelConfig,
    ProposedTrainingConfig,
)
from ml.models.proposed.encoders import ModalityEncoder
from ml.models.proposed.fusion import FusionModule
from ml.models.proposed.inference import ProposedModelAdapter
from ml.models.proposed.losses import build_loss_function, compute_class_weights
from ml.models.proposed.model import MODALITY_NAMES, ProposedFatigueModel
from ml.models.proposed.training import train_proposed_model

RNG = np.random.default_rng(17)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_attention_config_rejects_non_divisible_heads():
    with pytest.raises(ValueError):
        AttentionConfig(embed_dim=10, num_heads=3)


def test_fusion_config_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        FusionConfig(strategy="bogus")


def test_fusion_config_default_is_attention():
    assert FusionConfig().strategy == "attention"


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------


def test_modality_encoder_output_shape():
    encoder = ModalityEncoder(input_dim=72, config=EncoderConfig(hidden_dims=(64, 32), output_dim=16))
    x = torch.randn(8, 72)
    out = encoder(x)
    assert out.shape == (8, 16)


def test_modality_encoder_without_batch_norm():
    encoder = ModalityEncoder(input_dim=10, config=EncoderConfig(hidden_dims=(8,), output_dim=4, use_batch_norm=False))
    out = encoder(torch.randn(5, 10))
    assert out.shape == (5, 4)
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in encoder.modules())


def test_modality_encoder_single_sample_eval_mode():
    encoder = ModalityEncoder(input_dim=10, config=EncoderConfig(hidden_dims=(8,), output_dim=4))
    encoder.eval()
    with torch.no_grad():
        out = encoder(torch.randn(1, 10))
    assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def test_cross_modal_attention_layer_shapes():
    config = AttentionConfig(embed_dim=16, num_heads=4, num_layers=1)
    layer = CrossModalAttentionLayer(config)
    x = torch.randn(8, 3, 16)  # batch, num_modalities, embed_dim
    out, weights = layer(x)
    assert out.shape == (8, 3, 16)
    assert weights.shape == (8, 4, 3, 3)  # batch, heads, modalities, modalities


def test_cross_modal_attention_stack_returns_per_layer_weights():
    config = AttentionConfig(embed_dim=16, num_heads=2, num_layers=3)
    stack = CrossModalAttention(config)
    x = torch.randn(4, 3, 16)
    result = stack(x)
    assert result.embeddings.shape == (4, 3, 16)
    assert len(result.attention_weights) == 3
    for w in result.attention_weights:
        assert w.shape == (4, 2, 3, 3)


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------


def _make_embeddings(batch: int, embed_dim: int) -> dict:
    return {name: torch.randn(batch, embed_dim) for name in MODALITY_NAMES}


def _uniform_dims(embed_dim: int) -> dict:
    return {name: embed_dim for name in MODALITY_NAMES}


def test_fusion_concatenation_shape_and_no_attention():
    fusion = FusionModule(embed_dims=_uniform_dims(16), modality_names=MODALITY_NAMES, config=FusionConfig(strategy="concatenation"))
    output = fusion(_make_embeddings(8, 16))
    assert output.fused.shape == (8, 48)
    assert output.attention_weights is None
    assert fusion.fused_dim == 48


def test_fusion_concatenation_allows_mismatched_dims():
    dims = {"emg": 16, "imu": 32, "ppg": 8}
    fusion = FusionModule(embed_dims=dims, modality_names=MODALITY_NAMES, config=FusionConfig(strategy="concatenation"))
    embeddings = {name: torch.randn(4, dim) for name, dim in dims.items()}
    output = fusion(embeddings)
    assert output.fused.shape == (4, 56)
    assert fusion.fused_dim == 56


def test_fusion_attention_shape_and_weights():
    config = FusionConfig(strategy="attention", attention=AttentionConfig(embed_dim=16, num_heads=4, num_layers=2))
    fusion = FusionModule(embed_dims=_uniform_dims(16), modality_names=MODALITY_NAMES, config=config)
    output = fusion(_make_embeddings(8, 16))
    assert output.fused.shape == (8, 16)
    assert len(output.attention_weights) == 2
    assert output.contextualized_embeddings.shape == (8, 3, 16)
    assert fusion.fused_dim == 16


def test_fusion_weighted_sum_shape_and_weights_sum_to_one():
    fusion = FusionModule(embed_dims=_uniform_dims(16), modality_names=MODALITY_NAMES, config=FusionConfig(strategy="weighted_sum"))
    output = fusion(_make_embeddings(8, 16))
    assert output.fused.shape == (8, 16)
    assert output.modality_weights.shape == (3,)
    assert torch.allclose(output.modality_weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_fusion_attention_requires_matching_embed_dim():
    config = FusionConfig(strategy="attention", attention=AttentionConfig(embed_dim=8, num_heads=2))
    with pytest.raises(ValueError):
        FusionModule(embed_dims={"emg": 16, "imu": 16, "ppg": 8}, modality_names=MODALITY_NAMES, config=config)


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------


def test_classification_head_binary_and_multiclass():
    for n_classes in (2, 4):
        head = ClassificationHead(input_dim=32, n_classes=n_classes, config=ClassifierConfig(hidden_dims=(16,)))
        out = head(torch.randn(8, 32))
        assert out.shape == (8, n_classes)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["attention", "concatenation", "weighted_sum"])
def test_proposed_model_forward_all_strategies(strategy):
    config = ProposedModelConfig(
        emg_encoder=EncoderConfig(hidden_dims=(32,), output_dim=16),
        imu_encoder=EncoderConfig(hidden_dims=(64,), output_dim=16),
        ppg_encoder=EncoderConfig(hidden_dims=(8,), output_dim=16),
        fusion=FusionConfig(strategy=strategy, attention=AttentionConfig(embed_dim=16, num_heads=4)),
        n_classes=2,
    )
    model = ProposedFatigueModel(emg_dim=72, imu_dim=558, ppg_dim=7, config=config)
    output = model(torch.randn(8, 72), torch.randn(8, 558), torch.randn(8, 7))

    assert output.logits.shape == (8, 2)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(8), atol=1e-5)
    assert output.predictions.shape == (8,)
    assert set(output.modality_embeddings.keys()) == set(MODALITY_NAMES)
    for embedding in output.modality_embeddings.values():
        assert embedding.shape == (8, 16)


def test_proposed_model_rejects_mismatched_encoder_dims_for_attention_fusion():
    config = ProposedModelConfig(
        emg_encoder=EncoderConfig(output_dim=16),
        imu_encoder=EncoderConfig(output_dim=32),  # mismatched
        ppg_encoder=EncoderConfig(output_dim=16),
        fusion=FusionConfig(strategy="attention", attention=AttentionConfig(embed_dim=16, num_heads=4)),
    )
    with pytest.raises(ValueError):
        ProposedFatigueModel(emg_dim=10, imu_dim=10, ppg_dim=10, config=config)


def test_proposed_model_concatenation_allows_mismatched_encoder_dims():
    config = ProposedModelConfig(
        emg_encoder=EncoderConfig(hidden_dims=(16,), output_dim=16),
        imu_encoder=EncoderConfig(hidden_dims=(16,), output_dim=32),
        ppg_encoder=EncoderConfig(hidden_dims=(16,), output_dim=8),
        fusion=FusionConfig(strategy="concatenation"),
    )
    model = ProposedFatigueModel(emg_dim=10, imu_dim=10, ppg_dim=10, config=config)
    output = model(torch.randn(4, 10), torch.randn(4, 10), torch.randn(4, 10))
    assert output.fused_embedding.shape == (4, 16 + 32 + 8)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------


def test_compute_class_weights_inverse_frequency():
    y = np.array([0] * 8 + [1] * 2)
    weights = compute_class_weights(y, n_classes=2)
    assert weights[1] > weights[0]  # rarer class gets larger weight
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-4)


def test_compute_class_weights_handles_absent_class():
    y = np.array([0, 0, 0, 0])  # class 1 never appears
    weights = compute_class_weights(y, n_classes=2)
    assert torch.isfinite(weights).all()


def test_build_loss_function_with_class_weights():
    y = np.array([0] * 8 + [1] * 2)
    loss_fn = build_loss_function(y_train=y, n_classes=2)
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
    assert loss_fn.weight is not None


def test_build_loss_function_without_class_weights():
    loss_fn = build_loss_function(LossConfig(use_class_weights=False))
    assert loss_fn.weight is None


def test_build_loss_function_requires_labels_when_weighting_enabled():
    with pytest.raises(ValueError):
        build_loss_function(LossConfig(use_class_weights=True))


def test_build_loss_function_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_loss_function(LossConfig(name="bogus"))


# ---------------------------------------------------------------------------
# training + inference (integration)
# ---------------------------------------------------------------------------


def test_train_proposed_model_reduces_loss_and_returns_history():
    n, emg_dim, imu_dim, ppg_dim = 200, 10, 15, 5
    x_emg = RNG.normal(0, 1, (n, emg_dim))
    x_imu = RNG.normal(0, 1, (n, imu_dim))
    x_ppg = RNG.normal(0, 1, (n, ppg_dim))
    y = (x_emg[:, 0] + x_imu[:, 0] > 0).astype(int)

    model = ProposedFatigueModel(emg_dim, imu_dim, ppg_dim, ProposedModelConfig())
    history = train_proposed_model(
        model, x_emg[:140], x_imu[:140], x_ppg[:140], y[:140],
        x_emg[140:], x_imu[140:], x_ppg[140:], y[140:],
        ProposedTrainingConfig(max_epochs=10, batch_size=16, early_stopping_patience=10),
    )
    assert history.epochs_trained > 0
    assert history.training_seconds > 0
    assert len(history.train_loss) == history.epochs_trained


def test_train_proposed_model_early_stopping_triggers():
    n, emg_dim, imu_dim, ppg_dim = 150, 8, 8, 4
    x_emg = RNG.normal(0, 1, (n, emg_dim))
    x_imu = RNG.normal(0, 1, (n, imu_dim))
    x_ppg = RNG.normal(0, 1, (n, ppg_dim))
    y = RNG.integers(0, 2, n)  # pure noise -- should not keep improving

    model = ProposedFatigueModel(emg_dim, imu_dim, ppg_dim, ProposedModelConfig())
    history = train_proposed_model(
        model, x_emg[:100], x_imu[:100], x_ppg[:100], y[:100],
        x_emg[100:], x_imu[100:], x_ppg[100:], y[100:],
        ProposedTrainingConfig(max_epochs=200, batch_size=16, early_stopping_patience=3),
    )
    assert history.epochs_trained < 200


def test_train_proposed_model_checkpointing_saves_file(tmp_path):
    n, emg_dim, imu_dim, ppg_dim = 100, 6, 6, 3
    x_emg = RNG.normal(0, 1, (n, emg_dim))
    x_imu = RNG.normal(0, 1, (n, imu_dim))
    x_ppg = RNG.normal(0, 1, (n, ppg_dim))
    y = (x_emg[:, 0] > 0).astype(int)
    checkpoint_path = tmp_path / "proposed_model.pt"

    model = ProposedFatigueModel(emg_dim, imu_dim, ppg_dim, ProposedModelConfig())
    train_proposed_model(
        model, x_emg[:70], x_imu[:70], x_ppg[:70], y[:70], x_emg[70:], x_imu[70:], x_ppg[70:], y[70:],
        ProposedTrainingConfig(max_epochs=5, batch_size=16), checkpoint_path=checkpoint_path,
    )
    assert checkpoint_path.exists()


def test_proposed_model_adapter_predict_and_proba():
    emg_dim, imu_dim, ppg_dim = 10, 15, 5
    model = ProposedFatigueModel(emg_dim, imu_dim, ppg_dim, ProposedModelConfig())
    dims = ModalityDims(emg=emg_dim, imu=imu_dim, ppg=ppg_dim)
    adapter = ProposedModelAdapter(model, dims, device=torch.device("cpu"))

    X = RNG.normal(0, 1, (25, dims.total)).astype(np.float32)
    proba = adapter.predict_proba(X)
    preds = adapter.predict(X)
    assert proba.shape == (25, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(preds, proba.argmax(axis=1))


def test_proposed_model_adapter_predict_with_details_exposes_attention_and_embeddings():
    emg_dim, imu_dim, ppg_dim = 10, 15, 5
    model = ProposedFatigueModel(emg_dim, imu_dim, ppg_dim, ProposedModelConfig())
    dims = ModalityDims(emg=emg_dim, imu=imu_dim, ppg=ppg_dim)
    adapter = ProposedModelAdapter(model, dims, device=torch.device("cpu"))

    X = RNG.normal(0, 1, (10, dims.total)).astype(np.float32)
    details = adapter.predict_with_details(X)
    assert details.attention_weights is not None
    assert set(details.modality_embeddings.keys()) == set(MODALITY_NAMES)
    assert details.fused_embedding.shape[0] == 10


def test_proposed_model_adapter_reuses_shared_evaluate_model():
    """Confirms Phase 9's evaluate_model works unmodified with this
    multi-input model via the adapter's concatenated-array interface."""
    emg_dim, imu_dim, ppg_dim = 10, 15, 5
    model = ProposedFatigueModel(emg_dim, imu_dim, ppg_dim, ProposedModelConfig())
    dims = ModalityDims(emg=emg_dim, imu=imu_dim, ppg=ppg_dim)
    adapter = ProposedModelAdapter(model, dims, device=get_device("cpu"))

    X = RNG.normal(0, 1, (30, dims.total)).astype(np.float32)
    y = RNG.integers(0, 2, 30)
    config = DatasetConfig(modality="EMG", target_type="binary_fatigue")
    result = evaluate_model(adapter, X, y, "proposed_model", config, training_seconds=1.0)

    assert result.model_name == "proposed_model"
    assert result.y_proba.shape == (30, 2)
    assert 0 <= result.metrics.accuracy <= 1
