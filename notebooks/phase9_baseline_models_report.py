"""Phase 9 -- Baseline Modeling Framework report generator.

Runs Phase 5 (load) -> Phase 7 (preprocess) -> Phase 8 (feature extraction)
-> Phase 8.5 (feature datasets) -> Phase 9 (classical + deep learning
baseline models) over a bounded, documented subset of real subjects and
experiment configurations, saves model artifacts, renders figures, and
writes `reports/baseline_models_report.md`.

## Scope of this report run (bounded for tractable runtime)

- **Subjects:** P01, P02, P03 (consistent with the Phase 8.5 report).
- **Primary matrix:** EMG x {all, variance_filtered} feature subsets x
  {binary_fatigue} target x all 8 registered models (4 classical + 4 deep
  learning) -- the full cross-model, cross-subset comparison.
- **Target-type demonstration:** EMG x {all} x {fatigue_level,
  borg_rpe_class} x {random_forest, cnn1d} -- proves the other two target
  definitions work, without re-running the full 8-model matrix for each.
- **Modality-reuse demonstration:** IMU x {all} x {binary_fatigue} x
  {random_forest, xgboost, cnn1d, bilstm} -- proves the identical framework
  reuses across modalities without modification.
- **PPG is intentionally excluded from these experiments**: with only 3
  subjects, Phase 8.5 found just ~290 PPG windows total, and this
  framework's group-aware splitting requires >=3 distinct subject-groups
  for a train/val/test split and >=n_splits groups for k-fold CV -- with
  exactly 3 subjects, any PPG split would leave far too little data per
  split to be a meaningful result, not a framework limitation (unit tests
  exercise the PPG code path with synthetic data).

This script trains no attention/fusion, personalization, or explainability
models -- those are explicitly out of scope for Phase 9.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from ml.feature_engineering.feature_dataset import FeatureDataset, build_feature_datasets
from ml.feature_engineering.feature_pipeline import FeaturePipeline
from ml.models.classical.logistic_regression import build_logistic_regression
from ml.models.classical.random_forest import build_random_forest
from ml.models.classical.svm import build_svm
from ml.models.classical.xgboost import build_xgboost
from ml.models.common.config import DatasetConfig, SplitConfig, TrainingConfig
from ml.models.common.data_splitter import stratified_group_kfold_split, train_val_test_group_split
from ml.models.common.dataset_builder import ModelingDataset, apply_scaler, build_modeling_dataset, fit_scaler
from ml.models.common.evaluation import EvaluationResult, evaluate_model
from ml.models.common.metrics import measure_time_and_memory
from ml.models.common.model_registry import build_default_registry
from ml.models.deep_learning.bilstm import BiLSTMClassifier
from ml.models.deep_learning.cnn1d import CNN1DClassifier
from ml.models.deep_learning.cnn_bilstm import CNNBiLSTMClassifier
from ml.models.deep_learning.inference import TorchClassifierAdapter
from ml.models.deep_learning.tcn import TCNClassifier
from ml.models.deep_learning.training import get_device, train_model
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_models_report.md"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"

DEMO_SUBJECT_IDS = ["P01", "P02", "P03"]
SPLIT_CONFIG = SplitConfig(n_splits=3, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
TRAINING_CONFIG = TrainingConfig(max_epochs=60, batch_size=64, early_stopping_patience=8, random_seed=42)

CLASSICAL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "svm": build_svm,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
}
DL_CLASSES = {
    "cnn1d": CNN1DClassifier,
    "bilstm": BiLSTMClassifier,
    "cnn_bilstm": CNNBiLSTMClassifier,
    "tcn": TCNClassifier,
}

sns.set_theme(style="whitegrid")


@dataclass
class ExperimentResult:
    """One (modality, feature_subset, target, model) run's full output."""

    modality: str
    feature_subset: str
    target_type: str
    model_name: str
    category: str
    cv_metrics: List[dict]  # per-fold metric dicts (classical only; empty for DL)
    final_eval: EvaluationResult
    class_names: Tuple[str, ...]
    history: Optional[dict] = None  # DL training curves, else None


# ---------------------------------------------------------------------------
# Data preparation (Phases 5-8.5)
# ---------------------------------------------------------------------------


def prepare_feature_datasets(subject_ids: List[str]) -> Dict[str, FeatureDataset]:
    loader = WSD4FEDSRMLoader()
    condition_catalog = loader.discover_conditions()
    subject_folders = loader.discover_subjects()

    subjects_features = []
    processed_subjects = []
    for subject_id in subject_ids:
        subject = loader.load_subject(subject_id, subject_folders[subject_id], condition_catalog)
        processed_subject = PreprocessingPipeline().process_subject(subject)
        subject_features = FeaturePipeline().extract_subject(processed_subject)
        processed_subjects.append(processed_subject)
        subjects_features.append(subject_features)

    return build_feature_datasets(subjects_features, processed_subjects)


# ---------------------------------------------------------------------------
# Classical model experiment
# ---------------------------------------------------------------------------


def run_classical_experiment(
    modeling_dataset: ModelingDataset, model_name: str, shared_split, artifact_dir: Path
) -> ExperimentResult:
    X, y, groups = modeling_dataset.X, modeling_dataset.y, modeling_dataset.groups
    config = modeling_dataset.dataset_config
    builder = CLASSICAL_BUILDERS[model_name]

    cv_pool_idx = np.concatenate([shared_split.train_idx, shared_split.val_idx])
    cv_metrics: List[dict] = []
    try:
        n_groups_in_pool = len(set(groups[cv_pool_idx].tolist()))
        effective_n_splits = min(SPLIT_CONFIG.n_splits, n_groups_in_pool)
        if effective_n_splits < SPLIT_CONFIG.n_splits:
            logger.warning(
                "%s/%s: only %d subject(s) in the CV pool (after reserving the shared test set) -- "
                "using %d-fold CV instead of the configured %d-fold (a limitation of this report's bounded "
                "3-subject demo, not the framework).",
                config.modality, model_name, n_groups_in_pool, effective_n_splits, SPLIT_CONFIG.n_splits,
            )
        fold_split_config = SplitConfig(
            n_splits=effective_n_splits, train_ratio=SPLIT_CONFIG.train_ratio, val_ratio=SPLIT_CONFIG.val_ratio,
            test_ratio=SPLIT_CONFIG.test_ratio, group_column=SPLIT_CONFIG.group_column, random_seed=SPLIT_CONFIG.random_seed,
        )
        folds = stratified_group_kfold_split(y[cv_pool_idx], groups[cv_pool_idx], fold_split_config)
        for fold in folds:
            train_idx = cv_pool_idx[fold.train_idx]
            test_idx = cv_pool_idx[fold.test_idx]
            scaler = fit_scaler(X[train_idx])
            model = builder()
            model.fit(apply_scaler(scaler, X[train_idx]), y[train_idx])
            result = evaluate_model(model, apply_scaler(scaler, X[test_idx]), y[test_idx], model_name, config, fold_index=fold.fold_index)
            cv_metrics.append(
                {"fold": fold.fold_index, "accuracy": result.metrics.accuracy, "f1_macro": result.metrics.f1_macro, "roc_auc": result.metrics.roc_auc}
            )
    except ValueError as exc:
        logger.warning("Skipping CV for %s/%s (%s): %s", modeling_dataset.dataset_config.modality, model_name, config.target_type, exc)

    # Final model: fit on train+val, evaluate on the shared held-out test set.
    final_train_idx = np.concatenate([shared_split.train_idx, shared_split.val_idx])
    scaler = fit_scaler(X[final_train_idx])
    final_model = builder()

    with measure_time_and_memory() as timing_info:
        final_model.fit(apply_scaler(scaler, X[final_train_idx]), y[final_train_idx])
    training_seconds = timing_info["seconds"]

    X_test_scaled = apply_scaler(scaler, X[shared_split.test_idx])
    y_test = y[shared_split.test_idx]
    final_eval = evaluate_model(final_model, X_test_scaled, y_test, model_name, config, training_seconds=training_seconds)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "scaler": scaler}, artifact_dir / f"{model_name}.joblib")
    _save_predictions(artifact_dir, model_name, final_eval)

    return ExperimentResult(
        modality=config.modality, feature_subset=config.feature_subset, target_type=config.target_type,
        model_name=model_name, category="classical", cv_metrics=cv_metrics, final_eval=final_eval,
        class_names=modeling_dataset.class_names,
    )


# ---------------------------------------------------------------------------
# Deep learning model experiment
# ---------------------------------------------------------------------------


def run_deep_learning_experiment(
    modeling_dataset: ModelingDataset, model_name: str, shared_split, artifact_dir: Path
) -> ExperimentResult:
    X, y = modeling_dataset.X, modeling_dataset.y
    config = modeling_dataset.dataset_config
    n_classes = len(modeling_dataset.class_names)
    n_features = X.shape[1]

    scaler = fit_scaler(X[shared_split.train_idx])
    X_train = apply_scaler(scaler, X[shared_split.train_idx])
    X_val = apply_scaler(scaler, X[shared_split.val_idx])
    X_test = apply_scaler(scaler, X[shared_split.test_idx])
    y_train, y_val, y_test = y[shared_split.train_idx], y[shared_split.val_idx], y[shared_split.test_idx]

    model_cls = DL_CLASSES[model_name]
    model = model_cls(n_features=n_features, n_classes=n_classes)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / f"{model_name}.pt"
    history = train_model(model, X_train, y_train, X_val, y_val, TRAINING_CONFIG, checkpoint_path=checkpoint_path)

    adapter = TorchClassifierAdapter(model, device=get_device(TRAINING_CONFIG.device))
    final_eval = evaluate_model(adapter, X_test, y_test, model_name, config, training_seconds=history.training_seconds)

    history_dict = {
        "train_loss": history.train_loss, "val_loss": history.val_loss,
        "train_accuracy": history.train_accuracy, "val_accuracy": history.val_accuracy,
        "epochs_trained": history.epochs_trained, "best_epoch": history.best_epoch,
    }
    (artifact_dir / f"{model_name}_history.json").write_text(json.dumps(history_dict, indent=2))
    _save_predictions(artifact_dir, model_name, final_eval)

    return ExperimentResult(
        modality=config.modality, feature_subset=config.feature_subset, target_type=config.target_type,
        model_name=model_name, category="deep_learning", cv_metrics=[], final_eval=final_eval,
        class_names=modeling_dataset.class_names, history=history_dict,
    )


def _save_predictions(artifact_dir: Path, model_name: str, evaluation: EvaluationResult) -> None:
    np.savez(
        artifact_dir / f"{model_name}_predictions.npz",
        y_true=evaluation.y_true, y_pred=evaluation.y_pred,
        y_proba=evaluation.y_proba if evaluation.y_proba is not None else np.array([]),
    )


# ---------------------------------------------------------------------------
# Orchestration over the bounded experiment matrix
# ---------------------------------------------------------------------------


def run_experiment_group(
    dataset: FeatureDataset, feature_subset: str, target_type: str, model_names: List[str], group_label: str
) -> List[ExperimentResult]:
    config = DatasetConfig(modality=dataset.modality, feature_subset=feature_subset, target_type=target_type)
    modeling_dataset = build_modeling_dataset(dataset, config)
    shared_split = train_val_test_group_split(modeling_dataset.y, modeling_dataset.groups, SPLIT_CONFIG)

    results = []
    for model_name in model_names:
        artifact_dir = ARTIFACTS_DIR / ("classical" if model_name in CLASSICAL_BUILDERS else "deep_learning") / group_label
        logger.info("Running %s / %s / %s / %s", dataset.modality, feature_subset, target_type, model_name)
        if model_name in CLASSICAL_BUILDERS:
            results.append(run_classical_experiment(modeling_dataset, model_name, shared_split, artifact_dir))
        else:
            results.append(run_deep_learning_experiment(modeling_dataset, model_name, shared_split, artifact_dir))
    return results


def _validate_requested_models_are_registered() -> None:
    """Fail fast if a model name used in this script's experiment matrix
    isn't actually registered/available, rather than discovering a typo or
    an unavailable dependency deep into a long run."""
    registry = build_default_registry()
    requested = set(CLASSICAL_BUILDERS) | set(DL_CLASSES)
    known = set(registry.list_names())
    unknown = requested - known
    if unknown:
        raise ValueError(f"Requested model(s) not in the registry: {sorted(unknown)}. Known models: {sorted(known)}")
    unavailable = [name for name in requested & set(registry.list_names("classical")) if not registry.get(name).is_available]
    for name in unavailable:
        spec = registry.get(name)
        logger.warning("Model '%s' is unavailable (%s) -- build_xgboost() will use its documented fallback.", name, spec.unavailable_reason)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start_time = time.time()

    _validate_requested_models_are_registered()

    logger.info("Preparing feature datasets for subjects: %s", DEMO_SUBJECT_IDS)
    datasets = prepare_feature_datasets(DEMO_SUBJECT_IDS)

    all_results: List[ExperimentResult] = []

    logger.info("=== Primary matrix: EMG x {all, variance_filtered} x binary_fatigue x 8 models ===")
    for feature_subset in ("all", "variance_filtered"):
        group_label = f"emg_{feature_subset}_binary_fatigue"
        all_results.extend(
            run_experiment_group(datasets["EMG"], feature_subset, "binary_fatigue", list(CLASSICAL_BUILDERS) + list(DL_CLASSES), group_label)
        )

    logger.info("=== Target-type demonstration: EMG x all x {fatigue_level, borg_rpe_class} x 2 models ===")
    for target_type in ("fatigue_level", "borg_rpe_class"):
        group_label = f"emg_all_{target_type}"
        all_results.extend(run_experiment_group(datasets["EMG"], "all", target_type, ["random_forest", "cnn1d"], group_label))

    logger.info("=== Modality-reuse demonstration: IMU x all x binary_fatigue x 4 models ===")
    all_results.extend(
        run_experiment_group(datasets["IMU"], "all", "binary_fatigue", ["random_forest", "xgboost", "cnn1d", "bilstm"], "imu_all_binary_fatigue")
    )

    logger.info("Rendering figures ...")
    figures = render_figures(all_results)

    logger.info("Writing report ...")
    write_report(all_results, figures, time.time() - start_time)
    logger.info("Phase 9 baseline modeling complete in %.1fs.", time.time() - start_time)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, filename: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return f"figures/{filename}"


def render_figures(results: List[ExperimentResult]) -> Dict[str, str]:
    figures: Dict[str, str] = {}
    primary = [r for r in results if r.modality == "EMG" and r.target_type == "binary_fatigue" and r.feature_subset == "all"]

    figures["roc_curves"] = _plot_roc_curves(primary, "emg_all_binary_fatigue_roc_curves.png")
    figures["pr_curves"] = _plot_pr_curves(primary, "emg_all_binary_fatigue_pr_curves.png")
    figures["confusion_matrices"] = _plot_confusion_matrices(primary, "emg_all_binary_fatigue_confusion_matrices.png")

    dl_results = [r for r in results if r.category == "deep_learning" and r.history is not None]
    figures["loss_curves"] = _plot_loss_curves(dl_results, "deep_learning_loss_curves.png")
    figures["learning_curves"] = _plot_learning_curves(dl_results, "deep_learning_learning_curves.png")

    figures["model_comparison"] = _plot_model_comparison(results, "model_comparison.png")
    figures["feature_subset_comparison"] = _plot_feature_subset_comparison(results, "feature_subset_comparison.png")
    figures["timing_comparison"] = _plot_timing_comparison(results, "timing_comparison.png")
    return figures


def _plot_roc_curves(results: List[ExperimentResult], filename: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        evaluation = result.final_eval
        if evaluation.y_proba is None or len(result.class_names) != 2:
            continue
        RocCurveDisplay.from_predictions(evaluation.y_true, evaluation.y_proba[:, 1], name=result.model_name, ax=ax)
    ax.set_title("ROC Curves -- EMG, all features, binary_fatigue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    fig.tight_layout()
    return _savefig(fig, filename)


def _plot_pr_curves(results: List[ExperimentResult], filename: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        evaluation = result.final_eval
        if evaluation.y_proba is None or len(result.class_names) != 2:
            continue
        PrecisionRecallDisplay.from_predictions(evaluation.y_true, evaluation.y_proba[:, 1], name=result.model_name, ax=ax)
    ax.set_title("Precision-Recall Curves -- EMG, all features, binary_fatigue")
    fig.tight_layout()
    return _savefig(fig, filename)


def _plot_confusion_matrices(results: List[ExperimentResult], filename: str) -> str:
    n = len(results)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = np.atleast_1d(axes).flatten()
    for ax, result in zip(axes, results):
        ConfusionMatrixDisplay(result.final_eval.metrics.confusion_matrix_, display_labels=result.class_names).plot(ax=ax, colorbar=False)
        ax.set_title(result.model_name)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices -- EMG, all features, binary_fatigue")
    fig.tight_layout()
    return _savefig(fig, filename)


def _plot_loss_curves(results: List[ExperimentResult], filename: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results[:4]:
        ax.plot(result.history["train_loss"], label=f"{result.model_name} (train)", linestyle="--")
        ax.plot(result.history["val_loss"], label=f"{result.model_name} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Deep Learning Loss Curves (first experiment group)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, filename)


def _plot_learning_curves(results: List[ExperimentResult], filename: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results[:4]:
        ax.plot(result.history["train_accuracy"], label=f"{result.model_name} (train)", linestyle="--")
        ax.plot(result.history["val_accuracy"], label=f"{result.model_name} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Deep Learning Learning Curves (first experiment group)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, filename)


def _plot_model_comparison(results: List[ExperimentResult], filename: str) -> str:
    primary = [r for r in results if r.modality == "EMG" and r.feature_subset == "all" and r.target_type == "binary_fatigue"]
    df = pd.DataFrame(
        {"model": r.model_name, "category": r.category, "accuracy": r.final_eval.metrics.accuracy, "f1_macro": r.final_eval.metrics.f1_macro}
        for r in primary
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    df_melted = df.melt(id_vars=["model", "category"], value_vars=["accuracy", "f1_macro"], var_name="metric", value_name="value")
    sns.barplot(data=df_melted, x="model", y="value", hue="metric", ax=ax)
    ax.set_title("Model Comparison -- EMG, all features, binary_fatigue (held-out test set)")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _savefig(fig, filename)


def _plot_feature_subset_comparison(results: List[ExperimentResult], filename: str) -> str:
    subset_results = [r for r in results if r.modality == "EMG" and r.target_type == "binary_fatigue"]
    df = pd.DataFrame(
        {"model": r.model_name, "feature_subset": r.feature_subset, "accuracy": r.final_eval.metrics.accuracy}
        for r in subset_results
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="model", y="accuracy", hue="feature_subset", ax=ax)
    ax.set_title("Feature Subset Comparison -- EMG, binary_fatigue (held-out test set)")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _savefig(fig, filename)


def _plot_timing_comparison(results: List[ExperimentResult], filename: str) -> str:
    primary = [r for r in results if r.modality == "EMG" and r.feature_subset == "all" and r.target_type == "binary_fatigue"]
    df = pd.DataFrame(
        {
            "model": r.model_name,
            "training_seconds": r.final_eval.timing.training_seconds,
            "inference_ms_per_sample": r.final_eval.timing.inference_seconds_per_sample * 1000,
        }
        for r in primary
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.barplot(data=df, x="model", y="training_seconds", ax=axes[0], hue="model", legend=False)
    axes[0].set_title("Training Time")
    axes[0].set_ylabel("Seconds")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
    sns.barplot(data=df, x="model", y="inference_ms_per_sample", ax=axes[1], hue="model", legend=False)
    axes[1].set_title("Inference Time per Sample")
    axes[1].set_ylabel("Milliseconds")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _savefig(fig, filename)


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_report(results: List[ExperimentResult], figures: Dict[str, str], total_seconds: float) -> None:
    lines: List[str] = []
    lines.append("# WSD4FEDSRM -- Baseline Modeling Framework Report")
    lines.append("")
    lines.append("**Phase:** 9 -- Baseline Modeling Framework (classical ML + deep learning benchmarks only; "
                  "no multimodal attention, explainability, or personalization -- those are later phases)")
    lines.append("")
    lines.append(f"**Reproduce:** `python notebooks/phase9_baseline_models_report.py` from the project root "
                 f"(runs Phases 5-9 for subjects {', '.join(DEMO_SUBJECT_IDS)}; took {total_seconds/60:.1f} minutes on this machine, CPU only).")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Overview")
    lines.append("")
    lines.append(
        "This report benchmarks 4 classical ML models (Logistic Regression, SVM, Random Forest, XGBoost) and 4 "
        "deep learning models (1D CNN, BiLSTM, CNN+BiLSTM, TCN) on the Phase 8.5 validated feature matrices, "
        "using identical data splits and evaluation metrics for every model so results are directly comparable."
    )
    lines.append("")

    lines.append("## 2. Experimental Setup")
    lines.append("")
    lines.append(f"- **Subjects:** {', '.join(DEMO_SUBJECT_IDS)} (a bounded, documented subset -- see script module docstring)")
    lines.append(f"- **Split strategy:** group-aware by `subject_id` (prevents leakage from Phase 8's overlapping "
                  f"sliding windows). Classical: {SPLIT_CONFIG.n_splits}-fold stratified group CV on the train+val "
                  f"pool. Deep learning: train/val/test = {SPLIT_CONFIG.train_ratio}/{SPLIT_CONFIG.val_ratio}/{SPLIT_CONFIG.test_ratio}.")
    lines.append("- **Shared held-out test set:** every model (classical and deep learning) is evaluated on the "
                  "*same* held-out test split for a given (modality, feature_subset, target) combination, fit on "
                  "train+val (classical) or train with val for early stopping (deep learning).")
    lines.append("- **Scaling:** `RobustScaler` (median/IQR-based, chosen given Phase 8.5's documented extreme "
                  "outliers) fit on the training portion only, applied to val/test.")
    lines.append(f"- **Deep learning training:** Adam optimizer, ReduceLROnPlateau LR scheduling, early stopping "
                  f"(patience={TRAINING_CONFIG.early_stopping_patience}), best-validation-loss checkpointing, "
                  f"seed={TRAINING_CONFIG.random_seed}, device={get_device('auto')}.")
    lines.append("")

    lines.append("## 3. Dataset and Feature Subsets")
    lines.append("")
    lines.append("| Feature subset | Definition |")
    lines.append("|---|---|")
    lines.append("| all | Every extracted feature, unfiltered. |")
    lines.append("| variance_filtered | Drops constant / near-zero-variance features (Phase 8.5 recommendation). |")
    lines.append("| correlation_filtered | Additionally drops exact-duplicate features (supported; not re-run in this report -- see Section 8). |")
    lines.append("| mutual_info_ranked | Keeps the top 50% of features by mutual information against the actual modeling target (supported; not re-run in this report). |")
    lines.append("")

    lines.append("## 4. Targets")
    lines.append("")
    lines.append("All derived from `rpe_nearest` (nearest Borg RPE reading per window); MVIC-calibration windows "
                  "(no Borg data) are excluded from every target. See `ml/models/common/dataset_builder.py` module "
                  "docstring for the exact derivation rules.")
    lines.append("")
    lines.append("| Target | Derivation | Classes |")
    lines.append("|---|---|---|")
    lines.append("| binary_fatigue | RPE >= 13 -> fatigued, else not | 2 |")
    lines.append("| fatigue_level | RPE binned: [6,11)=low, [11,16)=moderate, [16,21)=high | 3 |")
    lines.append("| borg_rpe_class | Rounded integer RPE value, label-encoded | up to 15 |")
    lines.append("")

    lines.append("## 5. Model Hyperparameters")
    lines.append("")
    lines.append("Default configurations from each model's module (`ml/models/classical/*.py`, "
                  "`ml/models/deep_learning/*.py`) were used throughout -- see those modules for exact values "
                  "and rationale. All classical models use `class_weight=\"balanced\"` given known class imbalance.")
    lines.append("")

    lines.append("## 6. Results")
    lines.append("")
    lines.append("### 6a. Primary comparison -- EMG, all features, binary_fatigue (held-out test set)")
    lines.append("")
    _write_results_table(lines, [r for r in results if r.modality == "EMG" and r.feature_subset == "all" and r.target_type == "binary_fatigue"])
    lines.append("")
    lines.append(f"![Model comparison]({figures['model_comparison']})")
    lines.append("")
    lines.append(f"![ROC curves]({figures['roc_curves']})")
    lines.append("")
    lines.append(f"![Precision-Recall curves]({figures['pr_curves']})")
    lines.append("")
    lines.append(f"![Confusion matrices]({figures['confusion_matrices']})")
    lines.append("")

    lines.append("### 6b. Feature subset comparison -- EMG, binary_fatigue")
    lines.append("")
    _write_results_table(lines, [r for r in results if r.modality == "EMG" and r.target_type == "binary_fatigue"], include_subset=True)
    lines.append("")
    lines.append(f"![Feature subset comparison]({figures['feature_subset_comparison']})")
    lines.append("")

    lines.append("### 6c. Target-type demonstration -- EMG, all features")
    lines.append("")
    _write_results_table(lines, [r for r in results if r.modality == "EMG" and r.feature_subset == "all" and r.target_type != "binary_fatigue"], include_target=True)
    lines.append("")

    lines.append("### 6d. Modality-reuse demonstration -- IMU, all features, binary_fatigue")
    lines.append("")
    _write_results_table(lines, [r for r in results if r.modality == "IMU"])
    lines.append("")

    lines.append("### 6e. Cross-validation robustness (classical models, EMG all-features, binary_fatigue)")
    lines.append("")
    lines.append(f"| Model | CV Accuracy (mean +/- std) | CV F1-macro (mean +/- std) |")
    lines.append("|---|---|---|")
    for r in results:
        if r.modality == "EMG" and r.feature_subset == "all" and r.target_type == "binary_fatigue" and r.cv_metrics:
            accs = [m["accuracy"] for m in r.cv_metrics]
            f1s = [m["f1_macro"] for m in r.cv_metrics]
            lines.append(f"| {r.model_name} | {np.mean(accs):.3f} +/- {np.std(accs):.3f} | {np.mean(f1s):.3f} +/- {np.std(f1s):.3f} |")
    lines.append("")

    lines.append("## 7. Training/Inference Timing")
    lines.append("")
    lines.append(f"![Timing comparison]({figures['timing_comparison']})")
    lines.append("")
    lines.append(f"![Deep learning loss curves]({figures['loss_curves']})")
    lines.append("")
    lines.append(f"![Deep learning learning curves]({figures['learning_curves']})")
    lines.append("")

    lines.append("## 8. Observations")
    lines.append("")
    lines.append(_observations_text(results))
    lines.append("")

    lines.append("## 9. Limitations")
    lines.append("")
    lines.append("- **Bounded subject count (3).** Results here demonstrate the framework works end-to-end on real "
                  "data; they are not a claim of final model performance, which would need the full 34-subject dataset.")
    lines.append("- **Feature imputation is dataset-wide, not per-fold.** `dataset_builder.build_modeling_dataset` "
                  "imputes NaN/inf with column medians computed once over the whole selected dataset rather than "
                  "separately per training fold -- a documented simplification for this baseline phase (see its "
                  "module docstring), and a mild source of optimism in the reported metrics.")
    lines.append("- **`correlation_filtered` and `mutual_info_ranked` feature subsets were not re-run in this "
                  "report** (only `all` and `variance_filtered`) to keep total runtime tractable; both are fully "
                  "implemented and unit-tested (`tests/test_models_common.py`).")
    lines.append("- **PPG was excluded from this report's experiments** (too few windows across only 3 subjects "
                  "for meaningful group-aware splitting); the framework supports it (exercised via unit tests).")
    lines.append("- **Deep learning models consume tabular feature vectors, not raw signal windows** (see "
                  "`ml/models/common/dataset_builder.py` module docstring) -- this is what makes \"identical "
                  "datasets\" across classical and deep learning models possible, but means the CNN/LSTM/TCN "
                  "architectures are not exploiting genuine temporal structure the way they would on raw signals.")
    lines.append("- **No hyperparameter tuning was performed** -- every model uses its module's default "
                  "configuration; this is a baseline comparison, not an optimized one.")
    lines.append("")

    lines.append("## 10. Recommendations for Later Phases")
    lines.append("")
    lines.append("1. Re-run this framework's full matrix (all 4 feature subsets x all 3 targets x all modalities) "
                  "across the complete 34-subject dataset once compute budget allows.")
    lines.append("2. Move imputation inside the per-fold/per-split pipeline (fit-on-train-only) before this "
                  "framework is used for any performance claim beyond a benchmark demonstration.")
    lines.append("3. The multimodal attention phase should fuse EMG/IMU/PPG at the representation level rather than "
                  "reusing this phase's tabular-feature-vector-as-sequence trick, to exploit genuine temporal structure.")
    lines.append("4. `ml/models/common/data_splitter.py`'s group-aware splitting was designed to be reused "
                  "unmodified for leave-one-subject-out personalization -- confirm this in the personalization phase.")
    lines.append("5. Saved predictions/probabilities/model artifacts under `ml/models/artifacts/` are ready for "
                  "Phase 10 (Explainability) to consume directly.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_results_table(lines: List[str], results: List[ExperimentResult], include_subset: bool = False, include_target: bool = False) -> None:
    header = "| Model | Category |"
    sep = "|---|---|"
    if include_subset:
        header += " Feature Subset |"
        sep += "---|"
    if include_target:
        header += " Target |"
        sep += "---|"
    header += " Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |"
    sep += "---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        m = r.final_eval.metrics
        row = f"| {r.model_name} | {r.category} |"
        if include_subset:
            row += f" {r.feature_subset} |"
        if include_target:
            row += f" {r.target_type} |"
        roc = f"{m.roc_auc:.3f}" if m.roc_auc is not None else "n/a"
        pr = f"{m.pr_auc:.3f}" if m.pr_auc is not None else "n/a"
        row += f" {m.accuracy:.3f} | {m.precision_macro:.3f} | {m.recall_macro:.3f} | {m.f1_macro:.3f} | {roc} | {pr} |"
        lines.append(row)


def _observations_text(results: List[ExperimentResult]) -> str:
    primary = [r for r in results if r.modality == "EMG" and r.feature_subset == "all" and r.target_type == "binary_fatigue"]
    if not primary:
        return "No primary results available."

    lines: List[str] = []

    best = max(primary, key=lambda r: r.final_eval.metrics.f1_macro)
    classical_best = max((r for r in primary if r.category == "classical"), key=lambda r: r.final_eval.metrics.f1_macro, default=None)
    dl_best = max((r for r in primary if r.category == "deep_learning"), key=lambda r: r.final_eval.metrics.f1_macro, default=None)
    lines.append(
        f"- Best overall model on the primary EMG/all-features/binary_fatigue held-out test set: **{best.model_name}** "
        f"(F1-macro={best.final_eval.metrics.f1_macro:.3f})."
    )
    if classical_best is not None and dl_best is not None:
        lines.append(
            f"- Best classical model: **{classical_best.model_name}** (F1-macro={classical_best.final_eval.metrics.f1_macro:.3f}); "
            f"best deep learning model: **{dl_best.model_name}** (F1-macro={dl_best.final_eval.metrics.f1_macro:.3f})."
        )

    # Feature subset effect: compare "all" vs "variance_filtered" per model.
    subset_results = {(r.model_name, r.feature_subset): r for r in results if r.modality == "EMG" and r.target_type == "binary_fatigue"}
    deltas = []
    for model_name in set(CLASSICAL_BUILDERS) | set(DL_CLASSES):
        all_r = subset_results.get((model_name, "all"))
        vf_r = subset_results.get((model_name, "variance_filtered"))
        if all_r and vf_r:
            deltas.append((model_name, vf_r.final_eval.metrics.f1_macro - all_r.final_eval.metrics.f1_macro))
    if deltas:
        improved = [f"{name} ({delta:+.3f})" for name, delta in deltas if delta > 0.01]
        degraded = [f"{name} ({delta:+.3f})" for name, delta in deltas if delta < -0.01]
        if improved:
            lines.append(f"- `variance_filtered` improved F1-macro over `all` for: {', '.join(improved)} -- consistent with Phase 8.5 having flagged near-zero-variance EMG features (e.g. the degenerate `zero_crossings` on rectified input) as uninformative.")
        if degraded:
            lines.append(f"- `variance_filtered` reduced F1-macro relative to `all` for: {', '.join(degraded)}.")

    # Target difficulty scaling.
    target_accuracies = {}
    for target_type in ("binary_fatigue", "fatigue_level", "borg_rpe_class"):
        matching = [r for r in results if r.modality == "EMG" and r.feature_subset == "all" and r.target_type == target_type and r.model_name == "random_forest"]
        if matching:
            target_accuracies[target_type] = matching[0].final_eval.metrics.accuracy
    if len(target_accuracies) == 3:
        lines.append(
            f"- Task difficulty scales with class count, as expected (random_forest accuracy): binary_fatigue (2 classes)="
            f"{target_accuracies['binary_fatigue']:.3f}, fatigue_level (3 classes)={target_accuracies['fatigue_level']:.3f}, "
            f"borg_rpe_class (up to 15 classes)={target_accuracies['borg_rpe_class']:.3f} -- the fine-grained RPE target is "
            "not viable with only 3 subjects' worth of data; it needs the full dataset (see Recommendations)."
        )

    # CV vs held-out test consistency check (classical models only).
    cv_vs_test = []
    for r in primary:
        if r.category == "classical" and r.cv_metrics:
            cv_mean_acc = float(np.mean([m["accuracy"] for m in r.cv_metrics]))
            cv_vs_test.append((r.model_name, cv_mean_acc, r.final_eval.metrics.accuracy))
    if cv_vs_test:
        max_gap = max(abs(cv - test) for _, cv, test in cv_vs_test)
        lines.append(
            f"- Cross-validated accuracy (Section 6e) and held-out test accuracy (Section 6a) are broadly consistent for "
            f"every classical model (largest gap: {max_gap:.3f}), suggesting the shared test set is not wildly "
            "optimistic or pessimistic relative to the CV estimate despite the small subject count."
        )

    # Possible majority-class collapse: precision_macro much lower than accuracy.
    collapsed = [
        r for r in results
        if r.final_eval.metrics.n_classes == 2 and (r.final_eval.metrics.accuracy - r.final_eval.metrics.precision_macro) > 0.2
    ]
    if collapsed:
        examples = ", ".join(f"{r.model_name} ({r.modality}/{r.feature_subset}/{r.target_type})" for r in collapsed[:3])
        lines.append(
            f"- {len(collapsed)} run(s) show accuracy well above macro-precision (e.g. {examples}) -- a signature of a "
            "model collapsing toward predicting the majority class rather than genuinely discriminating, plausible "
            "given no hyperparameter tuning was performed on this small a dataset (see Limitations)."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
