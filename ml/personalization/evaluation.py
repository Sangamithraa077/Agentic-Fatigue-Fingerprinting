"""Generic-model vs. personalized-model comparison.

Reuses `ml.models.common.metrics.compute_classification_metrics` for both
sides of the comparison (rather than re-implementing accuracy/precision/
recall/F1/ROC-AUC/PR-AUC computation) -- this is the same metrics battery
`ml.models.common.evaluation.evaluate_model` and `loso.evaluate_loso` use,
so a subject's generic-model numbers here are directly comparable to their
`loso.LosoFold.evaluation.metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ml.models.common.metrics import ClassificationMetrics, compute_classification_metrics


@dataclass(frozen=True)
class PersonalizationComparison:
    """One subject's generic-vs-personalized comparison.

    Attributes:
        subject_id: The compared subject.
        n_samples: Number of evaluated windows.
        generic_metrics: Metrics for the unmodified global/LOSO-fold
            probability, thresholded at 0.5.
        personalized_metrics: Metrics for `personalization_model.
            predict_personalized`'s output, thresholded at 0.5.
        accuracy_improvement, f1_improvement: `personalized - generic`
            (positive = personalization helped this subject).
        roc_auc_improvement: Same, or `None` if either side's ROC-AUC is
            undefined (e.g. only one class present in this subject's data).
    """

    subject_id: str
    n_samples: int
    generic_metrics: ClassificationMetrics
    personalized_metrics: ClassificationMetrics
    accuracy_improvement: float
    f1_improvement: float
    roc_auc_improvement: Optional[float]


@dataclass(frozen=True)
class PersonalizationEvaluationSummary:
    """Aggregate comparison across every subject.

    Attributes:
        comparisons: One `PersonalizationComparison` per subject.
        mean_accuracy_improvement, mean_f1_improvement: Unweighted mean
            improvement across subjects.
        mean_roc_auc_improvement: Same, over subjects with a defined
            `roc_auc_improvement`; `None` if none are defined.
        n_subjects_improved_f1: How many subjects had `f1_improvement > 0`.
    """

    comparisons: Tuple[PersonalizationComparison, ...]
    mean_accuracy_improvement: float
    mean_f1_improvement: float
    mean_roc_auc_improvement: Optional[float]
    n_subjects_improved_f1: int


def _binary_probability_to_metrics(y_true: np.ndarray, positive_probability: np.ndarray) -> ClassificationMetrics:
    predictions = (positive_probability >= 0.5).astype(int)
    probability_matrix = np.column_stack([1.0 - positive_probability, positive_probability])
    return compute_classification_metrics(y_true, predictions, probability_matrix)


def compare_generic_vs_personalized(
    subject_id: str, y_true: np.ndarray, generic_probability: np.ndarray, personalized_probability: np.ndarray
) -> PersonalizationComparison:
    """Compare a generic model's probabilities against the personalized
    layer's output for one subject's held-out windows.

    Args:
        subject_id: The subject these windows belong to.
        y_true: True binary labels, shape `(n_samples,)`.
        generic_probability: The unmodified global/LOSO-fold model's
            predicted positive-class probability per sample.
        personalized_probability: `personalization_model.predict_personalized`'s
            `personalized_probability` per sample.

    Returns:
        A `PersonalizationComparison`.
    """
    generic_metrics = _binary_probability_to_metrics(y_true, np.asarray(generic_probability, dtype=float))
    personalized_metrics = _binary_probability_to_metrics(y_true, np.asarray(personalized_probability, dtype=float))

    roc_auc_improvement = None
    if generic_metrics.roc_auc is not None and personalized_metrics.roc_auc is not None:
        roc_auc_improvement = personalized_metrics.roc_auc - generic_metrics.roc_auc

    return PersonalizationComparison(
        subject_id=subject_id,
        n_samples=len(y_true),
        generic_metrics=generic_metrics,
        personalized_metrics=personalized_metrics,
        accuracy_improvement=personalized_metrics.accuracy - generic_metrics.accuracy,
        f1_improvement=personalized_metrics.f1_macro - generic_metrics.f1_macro,
        roc_auc_improvement=roc_auc_improvement,
    )


def summarize_personalization_evaluation(comparisons: Sequence[PersonalizationComparison]) -> PersonalizationEvaluationSummary:
    """Aggregate a list of per-subject comparisons into one summary."""
    if not comparisons:
        raise ValueError("Need at least 1 comparison to summarize.")

    accuracy_improvements = [c.accuracy_improvement for c in comparisons]
    f1_improvements = [c.f1_improvement for c in comparisons]
    roc_auc_improvements = [c.roc_auc_improvement for c in comparisons if c.roc_auc_improvement is not None]

    return PersonalizationEvaluationSummary(
        comparisons=tuple(comparisons),
        mean_accuracy_improvement=float(np.mean(accuracy_improvements)),
        mean_f1_improvement=float(np.mean(f1_improvements)),
        mean_roc_auc_improvement=float(np.mean(roc_auc_improvements)) if roc_auc_improvements else None,
        n_subjects_improved_f1=sum(1 for value in f1_improvements if value > 0),
    )
