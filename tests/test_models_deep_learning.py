"""Unit tests for ml.models.deep_learning (CNN1D, BiLSTM, CNN-BiLSTM, TCN,
training, inference)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.models.common.config import TrainingConfig
from ml.models.deep_learning.bilstm import BiLSTMClassifier, BiLSTMConfig
from ml.models.deep_learning.cnn1d import CNN1DClassifier, CNN1DConfig
from ml.models.deep_learning.cnn_bilstm import CNNBiLSTMClassifier, CNNBiLSTMConfig
from ml.models.deep_learning.inference import TorchClassifierAdapter
from ml.models.deep_learning.tcn import TCNClassifier, TCNConfig
from ml.models.deep_learning.training import get_device, set_seed, train_model

RNG = np.random.default_rng(9)

ARCHITECTURES = [
    (CNN1DClassifier, CNN1DConfig),
    (BiLSTMClassifier, BiLSTMConfig),
    (CNNBiLSTMClassifier, CNNBiLSTMConfig),
    (TCNClassifier, TCNConfig),
]


@pytest.mark.parametrize("model_cls,config_cls", ARCHITECTURES)
def test_architecture_forward_pass_shape(model_cls, config_cls):
    n_features, n_classes, batch = 20, 3, 8
    model = model_cls(n_features, n_classes, config_cls())
    x = torch.randn(batch, n_features)
    output = model(x)
    assert output.shape == (batch, n_classes)


@pytest.mark.parametrize("model_cls,config_cls", ARCHITECTURES)
def test_architecture_default_config_works(model_cls, config_cls):
    model = model_cls(n_features=15, n_classes=2)
    output = model(torch.randn(4, 15))
    assert output.shape == (4, 2)


@pytest.mark.parametrize("model_cls,config_cls", ARCHITECTURES)
def test_architecture_handles_single_sample_batch(model_cls, config_cls):
    # BatchNorm1d requires batch size > 1 in train mode; verify eval mode
    # (used at inference time) works fine with a single sample.
    model = model_cls(n_features=15, n_classes=2, config=config_cls())
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 15))
    assert output.shape == (1, 2)


def test_set_seed_reproducibility():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_get_device_cpu_preference():
    assert get_device("cpu").type == "cpu"


def test_get_device_auto_returns_valid_device():
    device = get_device("auto")
    assert device.type in ("cpu", "cuda")


def test_train_model_reduces_training_loss_and_returns_history():
    n_features, n_classes = 10, 2
    X = RNG.normal(0, 1, (200, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, y_train = X[:140], y[:140]
    X_val, y_val = X[140:], y[140:]

    model = CNN1DClassifier(n_features, n_classes)
    config = TrainingConfig(max_epochs=10, batch_size=16, early_stopping_patience=10)
    history = train_model(model, X_train, y_train, X_val, y_val, config)

    assert history.epochs_trained > 0
    assert len(history.train_loss) == history.epochs_trained
    assert history.training_seconds > 0
    assert history.train_loss[-1] < history.train_loss[0] or history.epochs_trained == 1


def test_train_model_early_stopping_triggers_before_max_epochs():
    n_features, n_classes = 8, 2
    X = RNG.normal(0, 1, (150, n_features))
    y = RNG.integers(0, 2, 150)  # pure noise -- val loss should not improve
    X_train, y_train = X[:100], y[:100]
    X_val, y_val = X[100:], y[100:]

    model = CNN1DClassifier(n_features, n_classes)
    config = TrainingConfig(max_epochs=200, batch_size=16, early_stopping_patience=3)
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    assert history.epochs_trained < 200


def test_train_model_checkpointing_saves_file(tmp_path):
    n_features, n_classes = 6, 2
    X = RNG.normal(0, 1, (100, n_features))
    y = (X[:, 0] > 0).astype(int)
    checkpoint_path = tmp_path / "model.pt"
    model = CNN1DClassifier(n_features, n_classes)
    config = TrainingConfig(max_epochs=5, batch_size=16)
    train_model(model, X[:70], y[:70], X[70:], y[70:], config, checkpoint_path=checkpoint_path)
    assert checkpoint_path.exists()


def test_torch_classifier_adapter_predict_and_proba():
    model = CNN1DClassifier(n_features=10, n_classes=2)
    adapter = TorchClassifierAdapter(model, device=torch.device("cpu"))
    X = RNG.normal(0, 1, (25, 10)).astype(np.float32)
    proba = adapter.predict_proba(X)
    preds = adapter.predict(X)
    assert proba.shape == (25, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert preds.shape == (25,)
    assert np.array_equal(preds, proba.argmax(axis=1))


def test_torch_classifier_adapter_batches_large_input():
    model = CNN1DClassifier(n_features=5, n_classes=2)
    adapter = TorchClassifierAdapter(model, device=torch.device("cpu"), batch_size=10)
    X = RNG.normal(0, 1, (37, 5)).astype(np.float32)
    proba = adapter.predict_proba(X)
    assert proba.shape == (37, 2)
