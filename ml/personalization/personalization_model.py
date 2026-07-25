"""The personalization layer: combines a population-level ("global") model's
prediction with a subject's own baseline profile to produce a personalized
fatigue score.

## Why a baseline-deviation blend, not a second trained model

This layer does not train anything -- it takes whatever probability an
already-fitted model produced (a Phase 9 classical checkpoint, a Phase 10
proposed-model forward pass, or a `loso.LosoFold`'s held-out prediction) and
adjusts it using **how unusual the current window's features are relative
to this specific subject's own historical baseline** (`baseline_builder.
ModalityBaseline`), rather than the population's. Concretely: each feature
is converted to a per-subject z-score against that subject's own baseline
mean/std, averaged in absolute value into one deviation signal, and squashed
through a logistic function into a `[0, 1]` "this looks unusual for this
person" probability. The final personalized probability is a configurable
convex blend of the global probability and this deviation signal
(`PersonalizationConfig.alpha`).

This keeps the layer model-agnostic (works identically for any modality/model
whose prediction is a single probability) and cheap to compute per-sample --
appropriate for a layer meant to be consumed directly by Phase 13's agentic
framework, which needs fast, explainable-by-construction adjustments rather
than another opaque model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ml.personalization.baseline_builder import ModalityBaseline
from ml.personalization.config import PersonalizationConfig


@dataclass(frozen=True)
class PersonalizedPrediction:
    """One sample's personalized fatigue prediction.

    Attributes:
        subject_id: Whose baseline this was scored against.
        global_probability: The population-level model's input probability,
            unchanged.
        baseline_deviation_score: Mean absolute per-feature z-score against
            `subject_id`'s own baseline (0 = exactly at this subject's
            typical value; larger = more unusual for this subject).
        deviation_probability: `baseline_deviation_score` squashed to `[0, 1]`.
        personalized_probability: `alpha * global_probability +
            (1 - alpha) * deviation_probability`.
        personalized_class: `1` if `personalized_probability >= 0.5`, else `0`.
    """

    subject_id: str
    global_probability: float
    baseline_deviation_score: float
    deviation_probability: float
    personalized_probability: float
    personalized_class: int


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def compute_baseline_deviation(x: np.ndarray, feature_names: Sequence[str], baseline: ModalityBaseline) -> float:
    """Mean absolute per-feature z-score of `x` against `baseline`.

    Features present in `x`/`feature_names` but absent from `baseline`
    (e.g. an empty baseline -- zero training windows for this subject) fall
    back to a neutral z-score of 0 for that feature rather than raising, so
    a subject with a sparse baseline still gets a (less informative, not a
    crashing) deviation score.
    """
    x = np.asarray(x, dtype=float)
    if not baseline.feature_means:
        return 0.0

    means = np.array([baseline.feature_means.get(name, x[i]) for i, name in enumerate(feature_names)])
    stds = np.array([baseline.feature_stds.get(name, 0.0) for name in feature_names])
    safe_stds = np.where(stds > 1e-8, stds, 1.0)
    z_scores = (x - means) / safe_stds
    return float(np.mean(np.abs(z_scores)))


def predict_personalized(
    subject_id: str,
    global_probability: float,
    x: np.ndarray,
    feature_names: Sequence[str],
    baseline: ModalityBaseline,
    config: Optional[PersonalizationConfig] = None,
) -> PersonalizedPrediction:
    """Blend a global model's probability with a subject-baseline deviation signal.

    Args:
        subject_id: Whose baseline `x` is compared against.
        global_probability: The population-level model's predicted
            probability of the positive (`fatigued`) class for `x`.
        x: This sample's raw (unscaled) feature vector, shape `(n_features,)`.
        feature_names: One name per entry of `x`, matching `baseline`'s keys.
        baseline: `subject_id`'s own `ModalityBaseline` (see
            `baseline_builder.build_modality_baseline`).
        config: Personalization configuration; defaults applied if omitted.

    Returns:
        A `PersonalizedPrediction`.
    """
    config = config or PersonalizationConfig()

    deviation_score = compute_baseline_deviation(x, feature_names, baseline)
    deviation_probability = _logistic(config.deviation_scale * (deviation_score - config.deviation_midpoint))
    personalized_probability = config.alpha * global_probability + (1.0 - config.alpha) * deviation_probability

    return PersonalizedPrediction(
        subject_id=subject_id,
        global_probability=float(global_probability),
        baseline_deviation_score=deviation_score,
        deviation_probability=float(deviation_probability),
        personalized_probability=float(personalized_probability),
        personalized_class=int(personalized_probability >= 0.5),
    )
