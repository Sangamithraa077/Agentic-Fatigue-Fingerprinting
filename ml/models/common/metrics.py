"""Classification metrics and timing/memory instrumentation, shared by
every classical and deep learning model so results are directly comparable.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegressionMetrics:
    """Continuous fatigue / Borg RPE regression metrics (Module Split v2, Module 5).

    Attributes:
        mae: Mean Absolute Error (in Borg RPE points).
        rmse: Root Mean Squared Error.
        pearson_r: Pearson correlation coefficient with ground-truth RPE.
        spearman_rho: Spearman rank correlation coefficient.
        within_1_rpe: Share of predictions with absolute error <= 1.0 RPE point.
        within_2_rpe: Share of predictions with absolute error <= 2.0 RPE points.
        n_samples: Number of evaluated samples.
    """

    mae: float
    rmse: float
    pearson_r: float
    spearman_rho: float
    within_1_rpe: float
    within_2_rpe: float
    n_samples: int


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standard classification metrics for one evaluation.

    Attributes:
        accuracy: Overall accuracy.
        precision_macro, recall_macro, f1_macro: Macro-averaged (unweighted
            mean across classes) -- treats every class equally regardless
            of support, appropriate given this project's class imbalance.
        roc_auc: Binary: standard AUC. Multiclass: macro-averaged
            one-vs-rest AUC. `None` if probabilities weren't available or
            only one class is present.
        pr_auc: Same averaging convention as `roc_auc`, using average
            precision (area under the precision-recall curve).
        confusion_matrix_: Row-normalized-free raw confusion matrix
            (rows=true, columns=predicted), ordered by sorted class label.
        classification_report_text: sklearn's per-class text report.
        n_samples: Number of evaluated samples.
        n_classes: Number of distinct classes present in `y_true`.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix_: np.ndarray
    classification_report_text: str
    n_samples: int
    n_classes: int


@dataclass(frozen=True)
class TimingMetrics:
    """Timing and memory instrumentation for one train/evaluate cycle."""

    training_seconds: float
    inference_seconds_total: float
    inference_seconds_per_sample: float
    peak_memory_mb: Optional[float]


@contextmanager
def measure_time_and_memory() -> Iterator[dict]:
    """Context manager measuring wall-clock time and peak memory via
    `tracemalloc` (stdlib, cross-platform -- the `resource` module used for
    this on Unix is unavailable on Windows).

    Usage:
        with measure_time_and_memory() as result:
            do_work()
        result["seconds"], result["peak_memory_mb"]
    """
    result: dict = {}
    tracemalloc_was_running = tracemalloc.is_tracing()
    if not tracemalloc_was_running:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        try:
            _, peak_bytes = tracemalloc.get_traced_memory()
            peak_mb = peak_bytes / (1024 * 1024)
        except Exception:  # pragma: no cover - defensive
            peak_mb = None
        if not tracemalloc_was_running:
            tracemalloc.stop()
        result["seconds"] = elapsed
        result["peak_memory_mb"] = peak_mb


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> ClassificationMetrics:
    """Compute the full standard metrics battery for one set of predictions.

    Args:
        y_true: (n_samples,) true integer class labels.
        y_pred: (n_samples,) predicted integer class labels.
        y_proba: Optional (n_samples, n_classes) predicted probabilities,
            required for `roc_auc`/`pr_auc`.

    Returns:
        A `ClassificationMetrics` instance. ROC-AUC/PR-AUC are `None` if
        `y_proba` is not given or only one class is present in `y_true`.
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(classes.tolist()))
    report_text = classification_report(y_true, y_pred, zero_division=0)

    roc_auc = None
    pr_auc = None
    if y_proba is not None and n_classes >= 2:
        try:
            if n_classes == 2:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                pr_auc = average_precision_score(y_true, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=sorted(classes.tolist()))
                pr_auc = average_precision_score(
                    np.eye(y_proba.shape[1])[y_true], y_proba, average="macro"
                )
        except ValueError as exc:
            logger.warning("Could not compute ROC-AUC/PR-AUC: %s", exc)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix_=cm,
        classification_report_text=report_text,
        n_samples=len(y_true),
        n_classes=n_classes,
    )


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> RegressionMetrics:
    """Compute continuous Borg RPE regression metrics (MAE, RMSE, correlation, error bands).

    Args:
        y_true: (n_samples,) true continuous target values (e.g. Borg RPE 6-20).
        y_pred: (n_samples,) predicted continuous target values.

    Returns:
        A RegressionMetrics instance.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n_samples = len(y_true)
    if n_samples == 0:
        return RegressionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Correlation checks (handle zero-variance safely)
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]) or n_samples < 2:
        r_val = 0.0
        rho_val = 0.0
    else:
        try:
            r_val = float(pearsonr(y_true, y_pred)[0])
            rho_val = float(spearmanr(y_true, y_pred)[0])
        except Exception:
            r_val = 0.0
            rho_val = 0.0

    errors = np.abs(y_true - y_pred)
    within_1 = float(np.mean(errors <= 1.0))
    within_2 = float(np.mean(errors <= 2.0))

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        pearson_r=r_val if not np.isnan(r_val) else 0.0,
        spearman_rho=rho_val if not np.isnan(rho_val) else 0.0,
        within_1_rpe=within_1,
        within_2_rpe=within_2,
        n_samples=n_samples,
    )

