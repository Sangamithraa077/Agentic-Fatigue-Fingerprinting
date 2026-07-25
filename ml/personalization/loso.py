"""Leave-One-Subject-Out (LOSO) cross-validation: the honest way to measure
how a fatigue model generalizes to a subject it has never seen, and this
phase's basis for comparison against the personalization layer
(`personalization_model.py`) -- a subject's LOSO fold is exactly "the
generic model trained without this subject", the correct "generic model"
baseline for `evaluation.compare_generic_vs_personalized` to personalize.

Splitting reuses `sklearn.model_selection.LeaveOneGroupOut` directly rather
than adding a bespoke grouped-split implementation next to
`ml.models.common.data_splitter`'s existing group-aware splitters -- LOSO is
exactly "K-fold with K = number of groups, one held out per fold", which
`LeaveOneGroupOut` already implements correctly. Per-fold fitting reuses the
same classical model builders, `fit_scaler`/`apply_scaler`, and
`evaluate_model` Phase 9 already established -- only the split strategy is
new.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from ml.models.classical.logistic_regression import build_logistic_regression
from ml.models.classical.random_forest import build_random_forest
from ml.models.classical.svm import build_svm
from ml.models.classical.xgboost import build_xgboost
from ml.models.common.dataset_builder import ModelingDataset, apply_scaler, fit_scaler
from ml.models.common.evaluation import EvaluationResult, evaluate_model
from ml.personalization.config import LosoConfig

logger = logging.getLogger(__name__)

MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "svm": build_svm,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
}


@dataclass(frozen=True)
class LosoFold:
    """One LOSO fold's train/test subjects and evaluation result.

    Attributes:
        held_out_subject: The one subject this fold tested on.
        train_subjects: Every other subject, trained on.
        evaluation: Full `EvaluationResult` (reused from
            `ml.models.common.evaluation`) on the held-out subject's data.
    """

    held_out_subject: str
    train_subjects: Tuple[str, ...]
    evaluation: EvaluationResult


@dataclass(frozen=True)
class LosoSummary:
    """Aggregate LOSO results across every held-out subject.

    Attributes:
        model_name: Which classical model was evaluated.
        folds: One `LosoFold` per subject, sorted by `held_out_subject`.
        mean_accuracy, mean_precision, mean_recall, mean_f1: Unweighted
            mean of each fold's corresponding `ClassificationMetrics` field
            across subjects.
        mean_roc_auc, mean_pr_auc: Same, but `None` if no fold produced a
            value (e.g. a fold's held-out subject has only one class present).
    """

    model_name: str
    folds: Tuple[LosoFold, ...]
    mean_accuracy: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_roc_auc: Optional[float]
    mean_pr_auc: Optional[float]


def evaluate_loso(modeling_dataset: ModelingDataset, config: Optional[LosoConfig] = None) -> LosoSummary:
    """Run Leave-One-Subject-Out cross-validation over `modeling_dataset`.

    For every distinct subject in `modeling_dataset.groups`: fit
    `config.model_name` on every *other* subject's rows (scaler fit on the
    same training rows, per `ml.models.common.dataset_builder`'s train-only
    scaling convention), then evaluate on the held-out subject's rows via
    `ml.models.common.evaluation.evaluate_model`.

    Args:
        modeling_dataset: A `ModelingDataset` (e.g. from
            `ml.models.common.dataset_builder.build_modeling_dataset`)
            spanning at least 2 subjects.
        config: LOSO configuration; defaults applied if omitted.

    Returns:
        A `LosoSummary`.

    Raises:
        ValueError: If `config.model_name` is not registered, or fewer than
            2 distinct subjects are present in `modeling_dataset.groups`.
    """
    config = config or LosoConfig()
    if config.model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown LOSO model_name: {config.model_name!r}. Known: {sorted(MODEL_BUILDERS)}")
    builder = MODEL_BUILDERS[config.model_name]

    X, y, groups = modeling_dataset.X, modeling_dataset.y, modeling_dataset.groups
    unique_subjects = sorted(set(groups.tolist()))
    if len(unique_subjects) < 2:
        raise ValueError(f"LOSO requires at least 2 distinct subjects, found {len(unique_subjects)}.")

    splitter = LeaveOneGroupOut()
    folds: List[LosoFold] = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        held_out_subjects = sorted(set(groups[test_idx].tolist()))
        held_out_subject = held_out_subjects[0]
        train_subjects = tuple(sorted(set(groups[train_idx].tolist())))

        scaler = fit_scaler(X[train_idx])
        model = builder()
        model.fit(apply_scaler(scaler, X[train_idx]), y[train_idx])

        evaluation = evaluate_model(
            model, apply_scaler(scaler, X[test_idx]), y[test_idx],
            model_name=config.model_name, dataset_config=modeling_dataset.dataset_config,
        )
        logger.info(
            "LOSO fold: held out '%s' (trained on %s) -- accuracy=%.3f, f1_macro=%.3f",
            held_out_subject, train_subjects, evaluation.metrics.accuracy, evaluation.metrics.f1_macro,
        )
        folds.append(LosoFold(held_out_subject=held_out_subject, train_subjects=train_subjects, evaluation=evaluation))

    folds.sort(key=lambda fold: fold.held_out_subject)
    return _summarize(config.model_name, tuple(folds))


def _summarize(model_name: str, folds: Tuple[LosoFold, ...]) -> LosoSummary:
    def _mean(attr: str) -> float:
        return float(np.mean([getattr(fold.evaluation.metrics, attr) for fold in folds]))

    def _mean_optional(attr: str) -> Optional[float]:
        values = [getattr(fold.evaluation.metrics, attr) for fold in folds if getattr(fold.evaluation.metrics, attr) is not None]
        return float(np.mean(values)) if values else None

    return LosoSummary(
        model_name=model_name,
        folds=folds,
        mean_accuracy=_mean("accuracy"),
        mean_precision=_mean("precision_macro"),
        mean_recall=_mean("recall_macro"),
        mean_f1=_mean("f1_macro"),
        mean_roc_auc=_mean_optional("roc_auc"),
        mean_pr_auc=_mean_optional("pr_auc"),
    )
