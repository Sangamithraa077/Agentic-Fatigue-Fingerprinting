"""Unit tests for Module 6: Data Augmentation (Training Data Only)."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from ml.preprocessing.augmentation import (
    TrainingDataAugmenter,
    add_gaussian_noise,
    apply_amplitude_scaling,
    apply_time_warping,
    benchmark_augmentations,
)


def test_add_gaussian_noise():
    X = np.ones((20, 5))
    X_noisy = add_gaussian_noise(X, noise_ratio=0.05, seed=42)
    assert X_noisy.shape == X.shape
    assert not np.array_equal(X, X_noisy)
    # Mean should still be close to original mean
    assert np.isclose(np.mean(X), np.mean(X_noisy), atol=0.05)


def test_apply_amplitude_scaling():
    X = np.ones((15, 4))
    X_scaled = apply_amplitude_scaling(X, scale_range=(0.9, 1.1), seed=42)
    assert X_scaled.shape == X.shape
    assert np.all(X_scaled >= 0.89)
    assert np.all(X_scaled <= 1.11)


def test_apply_time_warping():
    # Linear ramp
    X = np.linspace(0.0, 10.0, 30).reshape(30, 1)
    X_warped = apply_time_warping(X, max_warp=0.05, seed=42)
    assert X_warped.shape == X.shape
    # Endpoints preserved
    assert np.isclose(X_warped[0, 0], X[0, 0], atol=1e-3)
    assert np.isclose(X_warped[-1, 0], X[-1, 0], atol=1e-3)


def test_training_data_augmenter():
    X_train = np.random.randn(10, 6)
    y_train = np.linspace(6.0, 16.0, 10)

    augmenter = TrainingDataAugmenter(use_noise=True, use_scaling=True, use_warping=True)
    X_aug, y_aug = augmenter.augment(X_train, y_train, multiplier=2)

    # Multiplier 2 means original + 2 augmented copies = 30 samples
    assert X_aug.shape == (30, 6)
    assert len(y_aug) == 30
    # Original data is preserved in the first 10 rows
    assert np.allclose(X_aug[:10], X_train)


def test_benchmark_augmentations():
    X_train = np.random.randn(25, 4)
    y_train = np.linspace(7.0, 17.0, 25)
    X_test = np.random.randn(10, 4)
    y_test = np.linspace(8.0, 16.0, 10)

    results = benchmark_augmentations(
        estimator_factory=lambda: Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert "unaugmented" in results
    assert "noise_only" in results
    assert "scaling_only" in results
    assert results["unaugmented"].mae >= 0.0
