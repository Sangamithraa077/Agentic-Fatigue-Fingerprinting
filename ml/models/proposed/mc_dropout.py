"""Monte Carlo Dropout uncertainty quantification for fatigue predictions.

Provides `MCDropoutPredictor` which wraps any `ProposedModelAdapter` and estimates
predictive uncertainty by enabling dropout at inference time and running N forward
passes. Returns a point estimate + 95% confidence interval.

Usage:
    mc = MCDropoutPredictor(adapter, n_samples=50)
    result = mc.predict_with_uncertainty(X_single_window)
    print(f"RPE: {result.mean_rpe:.1f} ± {result.ci_half_width:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ml.models.proposed.inference import ProposedModelAdapter


@dataclass(frozen=True)
class UncertaintyPrediction:
    """Probabilistic fatigue prediction with calibrated confidence interval.

    Attributes:
        mean_prob: Mean fatigue probability across MC samples (0–1).
        std_prob: Standard deviation of probability across MC samples.
        ci_lower_prob: Lower bound of 95% confidence interval (probability).
        ci_upper_prob: Upper bound of 95% confidence interval (probability).
        mean_rpe: Mean prediction on Borg 6–20 scale.
        ci_lower_rpe: Lower 95% CI bound on Borg scale.
        ci_upper_rpe: Upper 95% CI bound on Borg scale.
        ci_half_width: Half-width of 95% CI on Borg scale.
        n_samples: Number of MC dropout forward passes used.
        is_high_uncertainty: True if std > HIGH_UNCERTAINTY_THRESHOLD.
    """

    mean_prob: float
    std_prob: float
    ci_lower_prob: float
    ci_upper_prob: float
    mean_rpe: float
    ci_lower_rpe: float
    ci_upper_rpe: float
    ci_half_width: float
    n_samples: int
    is_high_uncertainty: bool

    def format_rpe_string(self) -> str:
        return f"RPE {self.mean_rpe:.1f} ± {self.ci_half_width:.2f} (95% CI: [{self.ci_lower_rpe:.1f}, {self.ci_upper_rpe:.1f}])"


def _prob_to_borg(prob: float) -> float:
    """Map probability [0, 1] → Borg RPE scale [6, 20]."""
    return 6.0 + prob * 14.0


def _enable_dropout(model: torch.nn.Module) -> None:
    """Sets all Dropout layers to training mode, leaving the rest in eval mode.

    This activates MC Dropout without affecting Batch Norm or other modules.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


HIGH_UNCERTAINTY_THRESHOLD = 0.08  # std > 8% of probability range = high uncertainty


class MCDropoutPredictor:
    """Monte Carlo Dropout uncertainty estimator.

    Works with any `ProposedModelAdapter`. Runs `n_samples` stochastic forward
    passes with dropout active and computes the empirical predictive distribution.
    """

    def __init__(
        self,
        adapter: ProposedModelAdapter,
        n_samples: int = 50,
        ci_level: float = 0.95,
    ):
        self.adapter = adapter
        self.n_samples = n_samples
        self.ci_level = ci_level
        self._alpha = 1.0 - ci_level

    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyPrediction:
        """Runs N stochastic forward passes and returns probabilistic prediction.

        Args:
            X: Feature array of shape (n_samples_batch, n_features). For a
               single window, shape should be (1, n_features).

        Returns:
            UncertaintyPrediction dataclass with mean, std, and CI.
        """
        X = np.asarray(X, dtype=np.float32)

        # Switch to eval mode first, then enable dropout only
        self.adapter.model.eval()
        _enable_dropout(self.adapter.model)

        mc_probs: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                proba = self.adapter.predict_proba(X)
                # For multi-class: take probability of highest-fatigue class
                # (last class = most fatigued; probability of class > 0.5 threshold)
                mc_probs.append(proba[:, -1])  # (n_batch,)

        # Restore eval mode
        self.adapter.model.eval()

        stacked = np.stack(mc_probs, axis=0)  # (n_samples, n_batch)

        mean_prob = float(np.mean(stacked))
        std_prob = float(np.std(stacked))

        lower_q = (self._alpha / 2.0) * 100.0
        upper_q = (1.0 - self._alpha / 2.0) * 100.0
        ci_lower_prob = float(np.percentile(stacked, lower_q))
        ci_upper_prob = float(np.percentile(stacked, upper_q))

        mean_rpe = _prob_to_borg(mean_prob)
        ci_lower_rpe = _prob_to_borg(ci_lower_prob)
        ci_upper_rpe = _prob_to_borg(ci_upper_prob)
        ci_half_width = (ci_upper_rpe - ci_lower_rpe) / 2.0

        return UncertaintyPrediction(
            mean_prob=mean_prob,
            std_prob=std_prob,
            ci_lower_prob=ci_lower_prob,
            ci_upper_prob=ci_upper_prob,
            mean_rpe=mean_rpe,
            ci_lower_rpe=ci_lower_rpe,
            ci_upper_rpe=ci_upper_rpe,
            ci_half_width=ci_half_width,
            n_samples=self.n_samples,
            is_high_uncertainty=std_prob > HIGH_UNCERTAINTY_THRESHOLD,
        )
