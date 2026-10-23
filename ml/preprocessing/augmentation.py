"""Module 6: Signal & Feature Augmentation (Training Data Only).

Strict adherence to PDF Rules:
- Rule 2: Augment training data only. Always test on real, untouched recordings.
- Avoid shuffling window order, since muscular fatigue strictly progresses over time.
- Implements: Gaussian jitter/noise, amplitude scaling, small time warping.
- Evaluator to determine which augmentations improve generalization and which do not.

Evidence before use: these transforms perturb *extracted feature tables*. In the
project's leave-one-participant-out study (`research/results/final_synthesis_report.md`)
exactly this kind of perturbation made cross-athlete generalization worse (LSTM MAE
2.48 -> 2.76 Borg points with Gaussian noise), because it breaks relationships between
features (e.g. RMS^2 = mean^2 + std^2) that real signals obey. The variant that helped
in that study augmented the *raw signals* (magnitude warping) and recomputed features.
Nothing in the training code enables this module by default; run
`benchmark_augmentations` on your own data and keep only what it shows helps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from ml.models.common.metrics import RegressionMetrics, compute_regression_metrics


def add_gaussian_noise(
    X: np.ndarray,
    noise_ratio: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Adds zero-mean Gaussian jitter scaled by per-feature standard deviation."""
    rng = np.random.default_rng(seed)
    std = np.std(X, axis=0, keepdims=True)
    std = np.where(std == 0, 1e-6, std)
    noise = rng.normal(0.0, noise_ratio * std, size=X.shape)
    return X + noise


def apply_amplitude_scaling(
    X: np.ndarray,
    scale_range: Tuple[float, float] = (0.92, 1.08),
    seed: Optional[int] = None,
) -> np.ndarray:
    """Scales feature amplitudes by a smooth physiological variation multiplier."""
    rng = np.random.default_rng(seed)
    # Per-sample scaling factor to simulate inter-trial exertion variation
    scales = rng.uniform(scale_range[0], scale_range[1], size=(X.shape[0], 1))
    return X * scales


def apply_time_warping(
    X: np.ndarray,
    max_warp: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Applies monotonic temporal warping across consecutive time windows.

    Preserves chronological order (crucial for fatigue progression) while slightly
    varying exercise tempo / repetition speed.
    """
    n_samples = X.shape[0]
    if n_samples <= 3:
        return X.copy()

    rng = np.random.default_rng(seed)
    orig_indices = np.linspace(0.0, 1.0, n_samples)
    
    # Generate smooth monotonic warped time axis with fixed start (0) and end (1)
    random_shifts = rng.normal(0.0, max_warp / 2.0, size=n_samples)
    random_shifts[0] = 0.0
    random_shifts[-1] = 0.0
    warped_indices = np.sort(np.clip(orig_indices + random_shifts, 0.0, 1.0))

    warped_X = np.empty_like(X)
    for feat_idx in range(X.shape[1]):
        warped_X[:, feat_idx] = np.interp(warped_indices, orig_indices, X[:, feat_idx])
    return warped_X


class TrainingDataAugmenter:
    """Augmentation pipeline applied strictly to training data."""

    def __init__(
        self,
        use_noise: bool = True,
        use_scaling: bool = True,
        use_warping: bool = False,
        noise_ratio: float = 0.02,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        seed: int = 42,
    ):
        self.use_noise = use_noise
        self.use_scaling = use_scaling
        self.use_warping = use_warping
        self.noise_ratio = noise_ratio
        self.scale_range = scale_range
        self.seed = seed

    def augment(
        self, X_train: np.ndarray, y_train: np.ndarray, multiplier: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns concatenated original and augmented training batches."""
        X_out = [X_train]
        y_out = [y_train]

        for m in range(multiplier):
            current_seed = self.seed + m * 101
            X_aug = X_train.copy()

            if self.use_noise:
                X_aug = add_gaussian_noise(X_aug, noise_ratio=self.noise_ratio, seed=current_seed)
            if self.use_scaling:
                X_aug = apply_amplitude_scaling(X_aug, scale_range=self.scale_range, seed=current_seed + 1)
            if self.use_warping:
                X_aug = apply_time_warping(X_aug, seed=current_seed + 2)

            X_out.append(X_aug)
            y_out.append(y_train)

        return np.vstack(X_out), np.concatenate(y_out)


@dataclass
class AugmentationAblationResult:
    """Record of which augmentation methods improved metrics vs baseline."""

    method_name: str
    mae: float
    rmse: float
    within_1_rpe: float
    delta_mae: float  # negative is improvement
    beneficial: bool


def benchmark_augmentations(
    estimator_factory: Callable[[], BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, AugmentationAblationResult]:
    """Tests each augmentation technique individually against the unaugmented baseline.

    Corresponds to Module 6.2 - 6.4:
    'Test each method alone against the Module 5 baseline. Combine only the methods that helped.
    Record what helped and what did not.'
    """
    # 1. Baseline without augmentation
    base_model = estimator_factory()
    base_model.fit(X_train, y_train)
    pred_base = np.clip(base_model.predict(X_test), 6.0, 20.0)
    base_metrics = compute_regression_metrics(y_test, pred_base)

    results: Dict[str, AugmentationAblationResult] = {
        "unaugmented": AugmentationAblationResult(
            method_name="Unaugmented Baseline",
            mae=base_metrics.mae,
            rmse=base_metrics.rmse,
            within_1_rpe=base_metrics.within_1_rpe,
            delta_mae=0.0,
            beneficial=False,
        )
    }

    methods = [
        ("noise_only", TrainingDataAugmenter(use_noise=True, use_scaling=False, use_warping=False)),
        ("scaling_only", TrainingDataAugmenter(use_noise=False, use_scaling=True, use_warping=False)),
        ("warping_only", TrainingDataAugmenter(use_noise=False, use_scaling=False, use_warping=True)),
        ("noise_and_scaling", TrainingDataAugmenter(use_noise=True, use_scaling=True, use_warping=False)),
    ]

    for name, augmenter in methods:
        X_aug, y_aug = augmenter.augment(X_train, y_train, multiplier=1)
        model = estimator_factory()
        model.fit(X_aug, y_aug)
        pred = np.clip(model.predict(X_test), 6.0, 20.0)
        metrics = compute_regression_metrics(y_test, pred)
        delta = metrics.mae - base_metrics.mae

        results[name] = AugmentationAblationResult(
            method_name=name,
            mae=metrics.mae,
            rmse=metrics.rmse,
            within_1_rpe=metrics.within_1_rpe,
            delta_mae=delta,
            beneficial=delta < -0.01,  # Improved MAE by > 0.01 RPE points
        )

    return results
