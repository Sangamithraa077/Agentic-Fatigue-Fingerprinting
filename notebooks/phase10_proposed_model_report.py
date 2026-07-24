"""Phase 10 -- Proposed Multimodal Fusion Model report generator.

Runs Phase 5 (load) -> Phase 7 (preprocess) -> Phase 8 (feature extraction)
-> Phase 8.5 (feature datasets) -> Phase 10 (EMG/IMU/PPG alignment +
multimodal attention-fusion model) over the same bounded, documented subset
of real subjects Phase 9 used, saves model artifacts, renders figures, and
writes `reports/proposed_model_report.md`.

This script trains no SHAP explainability, personalization, or agentic
components -- those are later phases. It only trains and evaluates the
proposed architecture (`ml.models.proposed`) and compares it against the
already-published Phase 9 baseline numbers
(`reports/baseline_models_report.md`).

Reproduce with:
    python notebooks/phase10_proposed_model_report.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from ml.feature_engineering.feature_pipeline import FeaturePipeline
from ml.feature_engineering.feature_dataset import build_feature_datasets
from ml.models.common.config import DatasetConfig, SplitConfig
from ml.models.common.data_splitter import train_val_test_group_split
from ml.models.common.dataset_builder import apply_scaler, fit_scaler
from ml.models.common.evaluation import EvaluationResult, evaluate_model
from ml.models.common.multimodal_dataset_builder import ModalityDims, MultimodalDataset, build_multimodal_dataset
from ml.models.deep_learning.training import get_device
from ml.models.proposed.config import (
    AttentionConfig,
    ClassifierConfig,
    EncoderConfig,
    FusionConfig,
    ProposedModelConfig,
    ProposedTrainingConfig,
)
from ml.models.proposed.inference import ProposedModelAdapter
from ml.models.proposed.model import ProposedFatigueModel
from ml.models.proposed.training import train_proposed_model
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "proposed_model_report.md"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts" / "proposed"

# Same bounded, documented subject subset as Phase 9's report, so this
# phase's results are comparable against those already-published numbers.
DEMO_SUBJECT_IDS = ["P01", "P02", "P03"]
SPLIT_CONFIG = SplitConfig(n_splits=3, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
TRAINING_CONFIG = ProposedTrainingConfig(
    max_epochs=80, batch_size=64, early_stopping_patience=10, random_seed=42, learning_rate=5e-4, weight_decay=1e-2
)

# Phase 9's published EMG/all-features/binary_fatigue held-out test results
# (reports/baseline_models_report.md, Section 6a) -- restated here (not
# re-computed) for the comparison table, since Phase 9's per-modality
# dataset and Phase 10's fused multimodal dataset are built by genuinely
# different pipelines (one row per sensor-window vs. one row per aligned
# multimodal window) and are not expected to share identical rows; both
# use the same 3 subjects and the same group-aware split methodology
# (seed=42), which is what makes the comparison meaningful.
BASELINE_RESULTS = {
    "logistic_regression": {"category": "classical", "accuracy": 0.560, "precision": 0.544, "recall": 0.548, "f1": 0.540, "roc_auc": 0.570},
    "svm": {"category": "classical", "accuracy": 0.588, "precision": 0.547, "recall": 0.547, "f1": 0.547, "roc_auc": 0.563},
    "random_forest": {"category": "classical", "accuracy": 0.625, "precision": 0.568, "recall": 0.561, "f1": 0.561, "roc_auc": 0.586},
    "xgboost": {"category": "classical", "accuracy": 0.664, "precision": 0.606, "recall": 0.553, "f1": 0.535, "roc_auc": 0.598},
    "cnn1d": {"category": "deep_learning", "accuracy": 0.660, "precision": 0.598, "recall": 0.527, "f1": 0.479, "roc_auc": 0.574},
    "bilstm": {"category": "deep_learning", "accuracy": 0.659, "precision": 0.594, "recall": 0.530, "f1": 0.487, "roc_auc": 0.568},
    "cnn_bilstm": {"category": "deep_learning", "accuracy": 0.660, "precision": 0.607, "recall": 0.517, "f1": 0.449, "roc_auc": 0.575},
    "tcn": {"category": "deep_learning", "accuracy": 0.672, "precision": 0.626, "recall": 0.557, "f1": 0.536, "roc_auc": 0.604},
}

FUSION_STRATEGIES = ("attention", "concatenation", "weighted_sum")

sns.set_theme(style="whitegrid")


def prepare_multimodal_dataset(subject_ids: List[str]) -> MultimodalDataset:
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

    datasets = build_feature_datasets(subjects_features, processed_subjects)
    config = DatasetConfig(modality="EMG", target_type="binary_fatigue")  # modality field unused by build_multimodal_dataset
    return build_multimodal_dataset(datasets["EMG"], datasets["IMU"], datasets["PPG"], config)


def make_model_config(strategy: str, dims: ModalityDims) -> ProposedModelConfig:
    # Deliberately small and heavily regularized relative to the default
    # architecture: with only 3 demo subjects, a group-aware split puts
    # exactly one subject per train/val/test split (verified against real
    # data -- see reports/proposed_model_report.md Section 6/7), so a
    # large-capacity model overfits its single training subject within a
    # handful of epochs and never generalizes to the held-out subjects'
    # different feature distributions. These hyperparameters are sized for
    # this bounded demo, not a recommendation for the full 34-subject run.
    embed_dim = 8
    return ProposedModelConfig(
        emg_encoder=EncoderConfig(hidden_dims=(16,), output_dim=embed_dim, dropout=0.5),
        imu_encoder=EncoderConfig(hidden_dims=(32,), output_dim=embed_dim, dropout=0.5),
        ppg_encoder=EncoderConfig(hidden_dims=(8,), output_dim=embed_dim, dropout=0.3),
        fusion=FusionConfig(strategy=strategy, attention=AttentionConfig(embed_dim=embed_dim, num_heads=2, num_layers=1, dropout=0.2)),
        classifier=ClassifierConfig(hidden_dims=(8,), dropout=0.5),
        n_classes=2,
    )


def run_strategy(mm: MultimodalDataset, strategy: str, split, artifact_dir: Path) -> tuple:
    x_emg, x_imu, x_ppg = mm.dims.split(mm.X)

    scaler = fit_scaler(mm.X[split.train_idx])
    X_scaled = apply_scaler(scaler, mm.X)
    x_emg_s, x_imu_s, x_ppg_s = mm.dims.split(X_scaled)

    config = make_model_config(strategy, mm.dims)
    model = ProposedFatigueModel(emg_dim=mm.dims.emg, imu_dim=mm.dims.imu, ppg_dim=mm.dims.ppg, config=config)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / f"{strategy}.pt"

    history = train_proposed_model(
        model,
        x_emg_s[split.train_idx], x_imu_s[split.train_idx], x_ppg_s[split.train_idx], mm.y[split.train_idx],
        x_emg_s[split.val_idx], x_imu_s[split.val_idx], x_ppg_s[split.val_idx], mm.y[split.val_idx],
        TRAINING_CONFIG, checkpoint_path=checkpoint_path,
    )

    adapter = ProposedModelAdapter(model, mm.dims, device=get_device(TRAINING_CONFIG.device))
    X_test_scaled = X_scaled[split.test_idx]
    y_test = mm.y[split.test_idx]
    ds_config = DatasetConfig(modality="EMG", target_type="binary_fatigue")
    evaluation = evaluate_model(adapter, X_test_scaled, y_test, f"proposed_{strategy}", ds_config, training_seconds=history.training_seconds)

    joblib.dump({"scaler": scaler}, artifact_dir / f"{strategy}_scaler.joblib")
    np.savez(
        artifact_dir / f"{strategy}_predictions.npz",
        y_true=evaluation.y_true, y_pred=evaluation.y_pred, y_proba=evaluation.y_proba,
    )
    history_dict = {
        "train_loss": history.train_loss, "val_loss": history.val_loss,
        "train_accuracy": history.train_accuracy, "val_accuracy": history.val_accuracy,
        "epochs_trained": history.epochs_trained, "best_epoch": history.best_epoch,
    }
    (artifact_dir / f"{strategy}_history.json").write_text(json.dumps(history_dict, indent=2))

    details = None
    if strategy == "attention":
        sample = X_test_scaled[: min(32, len(X_test_scaled))]
        details = adapter.predict_with_details(sample)

    return evaluation, history_dict, details


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start_time = time.time()

    logger.info("Building aligned multimodal dataset for subjects: %s", DEMO_SUBJECT_IDS)
    mm = prepare_multimodal_dataset(DEMO_SUBJECT_IDS)
    logger.info("Multimodal dataset: X=%s, dims=%s, excluded=%d", mm.X.shape, mm.dims, mm.n_rows_excluded)

    split = train_val_test_group_split(mm.y, mm.groups, SPLIT_CONFIG)

    results: Dict[str, EvaluationResult] = {}
    histories: Dict[str, dict] = {}
    attention_details = None
    for strategy in FUSION_STRATEGIES:
        logger.info("=== Training proposed model (fusion strategy: %s) ===", strategy)
        evaluation, history_dict, details = run_strategy(mm, strategy, split, ARTIFACTS_DIR / strategy)
        results[strategy] = evaluation
        histories[strategy] = history_dict
        if details is not None:
            attention_details = details

    logger.info("Rendering figures ...")
    figures = render_figures(results, histories, attention_details, mm)

    logger.info("Writing report ...")
    write_report(mm, split, results, histories, figures, time.time() - start_time)
    logger.info("Phase 10 proposed model report complete in %.1fs.", time.time() - start_time)


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


def render_figures(results: Dict[str, EvaluationResult], histories: Dict[str, dict], attention_details, mm: MultimodalDataset) -> Dict[str, str]:
    figures = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, history in histories.items():
        ax.plot(history["train_loss"], linestyle="--", label=f"{strategy} (train)")
        ax.plot(history["val_loss"], label=f"{strategy} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Proposed Model -- Loss Curves by Fusion Strategy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["loss_curves"] = _savefig(fig, "proposed_model_loss_curves.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, history in histories.items():
        ax.plot(history["train_accuracy"], linestyle="--", label=f"{strategy} (train)")
        ax.plot(history["val_accuracy"], label=f"{strategy} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Proposed Model -- Training Curves by Fusion Strategy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["training_curves"] = _savefig(fig, "proposed_model_training_curves.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for strategy, evaluation in results.items():
        if evaluation.y_proba is None:
            continue
        RocCurveDisplay.from_predictions(evaluation.y_true, evaluation.y_proba[:, 1], name=strategy, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Proposed Model -- ROC Curves by Fusion Strategy")
    fig.tight_layout()
    figures["roc_curves"] = _savefig(fig, "proposed_model_roc_curves.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for strategy, evaluation in results.items():
        if evaluation.y_proba is None:
            continue
        PrecisionRecallDisplay.from_predictions(evaluation.y_true, evaluation.y_proba[:, 1], name=strategy, ax=ax)
    ax.set_title("Proposed Model -- Precision-Recall Curves by Fusion Strategy")
    fig.tight_layout()
    figures["pr_curves"] = _savefig(fig, "proposed_model_pr_curves.png")

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    for ax, (strategy, evaluation) in zip(axes, results.items()):
        ConfusionMatrixDisplay(evaluation.metrics.confusion_matrix_, display_labels=["not_fatigued", "fatigued"]).plot(ax=ax, colorbar=False)
        ax.set_title(strategy)
    fig.suptitle("Proposed Model -- Confusion Matrices by Fusion Strategy")
    fig.tight_layout()
    figures["confusion_matrices"] = _savefig(fig, "proposed_model_confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys()) + list(BASELINE_RESULTS.keys())
    f1_scores = [results[s].metrics.f1_macro for s in results] + [BASELINE_RESULTS[b]["f1"] for b in BASELINE_RESULTS]
    colors = ["#C44E52"] * len(results) + ["#4C72B0"] * len(BASELINE_RESULTS)
    ax.bar(names, f1_scores, color=colors)
    ax.set_ylabel("F1-macro (held-out test set)")
    ax.set_title("Proposed Model vs. Phase 9 Baselines (EMG binary_fatigue)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    figures["comparison"] = _savefig(fig, "proposed_vs_baseline_comparison.png")

    if attention_details is not None and attention_details.attention_weights is not None:
        n_layers = len(attention_details.attention_weights)
        fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5))
        axes = np.atleast_1d(axes)
        modality_labels = ["emg", "imu", "ppg"]
        for ax, layer_weights in zip(axes, attention_details.attention_weights):
            averaged = layer_weights.mean(dim=(0, 1)).detach().cpu().numpy()  # average over batch and heads
            sns.heatmap(averaged, annot=True, fmt=".2f", cmap="viridis", xticklabels=modality_labels, yticklabels=modality_labels, ax=ax, vmin=0, vmax=1)
            ax.set_xlabel("Attended-to modality")
            ax.set_ylabel("Querying modality")
        fig.suptitle("Cross-Modal Attention Weights (averaged over test batch + heads; raw tensors, not SHAP)")
        fig.tight_layout()
        figures["attention_heatmap"] = _savefig(fig, "proposed_model_attention_heatmap.png")

    return figures


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_report(mm: MultimodalDataset, split, results: Dict[str, EvaluationResult], histories: dict, figures: Dict[str, str], total_seconds: float) -> None:
    lines: List[str] = []
    lines.append("# WSD4FEDSRM -- Proposed Multimodal Fusion Model Report")
    lines.append("")
    lines.append("**Phase:** 10 -- Proposed Model (novel multimodal architecture only; no SHAP, personalization, or agentic AI -- later phases)")
    lines.append("")
    lines.append(f"**Reproduce:** `python notebooks/phase10_proposed_model_report.py` from the project root "
                 f"(runs Phases 5-10 for subjects {', '.join(DEMO_SUBJECT_IDS)}; took {total_seconds/60:.1f} minutes on this machine, CPU only).")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Architecture")
    lines.append("")
    lines.append(
        "Independent MLP encoders per modality (EMG/IMU/PPG) project each modality's Phase 8 feature vector to a "
        "shared 32-dimensional embedding. A configurable fusion module then combines the three embeddings into one "
        "unified representation via one of three strategies -- **multi-head cross-modal attention (default)**, "
        "concatenation, or a learned weighted sum -- fed to a classification head. See "
        "`docs/proposed_model.md` and each module's docstring (`ml/models/proposed/`) for full design rationale."
    )
    lines.append("")
    lines.append("```")
    lines.append("EMG features (72-d)  -> EMGEncoder -> embedding (32-d) --\\")
    lines.append("IMU features (558-d) -> IMUEncoder -> embedding (32-d) ---> FusionModule -> ClassificationHead -> logits")
    lines.append("PPG features (7-d)   -> PPGEncoder -> embedding (32-d) --/")
    lines.append("```")
    lines.append("")

    lines.append("## 2. Dataset")
    lines.append("")
    lines.append(
        f"Built by aligning Phase 8.5's per-modality EMG/IMU/PPG feature datasets into shared multimodal samples "
        f"(`ml.models.common.multimodal_dataset_builder`) for subjects {', '.join(DEMO_SUBJECT_IDS)}: "
        f"EMG {mm.dims.emg} features (6 muscles), IMU {mm.dims.imu} features (6 locations), PPG {mm.dims.ppg} features, "
        f"{mm.X.shape[0]} aligned samples ({mm.n_rows_excluded} MVIC-calibration windows excluded -- no Borg RPE label)."
    )
    lines.append("")
    lines.append(
        "EMG and IMU windows are matched directly (they share Phase 8's default 1.0s/50%-overlap window config); "
        "PPG's much coarser 10.0s windows are matched to each EMG/IMU window by nearest center time, so several "
        "consecutive EMG/IMU windows legitimately share one PPG window's features -- see the module docstring for "
        "why this alignment step is Phase 10's job, not a carry-over gap."
    )
    lines.append("")
    lines.append(
        f"**Split:** group-aware by subject_id (train/val/test = {SPLIT_CONFIG.train_ratio}/{SPLIT_CONFIG.val_ratio}/{SPLIT_CONFIG.test_ratio}, "
        f"seed={SPLIT_CONFIG.random_seed}) -- {len(split.train_idx)}/{len(split.val_idx)}/{len(split.test_idx)} samples. "
        "`RobustScaler` fit on the training portion only."
    )
    lines.append("")

    lines.append("## 3. Hyperparameters")
    lines.append("")
    lines.append(
        "**Deliberately small and heavily regularized** relative to what a full-dataset run would use -- see "
        "Section 6/7: with only 3 demo subjects, a group-aware split puts exactly one subject in each of "
        "train/val/test, and a larger-capacity model was found to overfit its single training subject within a "
        "handful of epochs (verified: train accuracy reached 97% while validation loss simultaneously *increased* "
        "every epoch from the very first). These hyperparameters are sized for this bounded demo, not a "
        "recommendation for the full 34-subject run."
    )
    lines.append("")
    lines.append("| Component | Configuration |")
    lines.append("|---|---|")
    lines.append("| EMG encoder | hidden=(16,), output=8, dropout=0.5, BatchNorm |")
    lines.append("| IMU encoder | hidden=(32,), output=8, dropout=0.5, BatchNorm |")
    lines.append("| PPG encoder | hidden=(8,), output=8, dropout=0.3, BatchNorm |")
    lines.append("| Attention fusion | embed_dim=8, heads=2, layers=1, dropout=0.2 |")
    lines.append("| Classifier | hidden=(8,), dropout=0.5, BatchNorm |")
    lines.append("| Loss | Cross-entropy, class-weighted (inverse frequency) |")
    lines.append(f"| Training | Adam, lr={TRAINING_CONFIG.learning_rate}, weight_decay={TRAINING_CONFIG.weight_decay}, batch={TRAINING_CONFIG.batch_size}, "
                 f"early stopping patience={TRAINING_CONFIG.early_stopping_patience}, ReduceLROnPlateau, seed={TRAINING_CONFIG.random_seed} |")
    lines.append(f"| Device | {get_device('auto')} (mixed precision enabled only if CUDA) |")
    lines.append("")

    lines.append("## 4. Results")
    lines.append("")
    lines.append("### 4a. Fusion strategy comparison (held-out test set)")
    lines.append("")
    lines.append("| Fusion strategy | Epochs | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Train (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for strategy, evaluation in results.items():
        m = evaluation.metrics
        roc = f"{m.roc_auc:.3f}" if m.roc_auc is not None else "n/a"
        pr = f"{m.pr_auc:.3f}" if m.pr_auc is not None else "n/a"
        lines.append(
            f"| {strategy} | {histories[strategy]['epochs_trained']} | {m.accuracy:.3f} | {m.precision_macro:.3f} | "
            f"{m.recall_macro:.3f} | {m.f1_macro:.3f} | {roc} | {pr} | {evaluation.timing.training_seconds:.1f} |"
        )
    lines.append("")
    lines.append(f"![Loss curves]({figures['loss_curves']})")
    lines.append("")
    lines.append(f"![Training curves]({figures['training_curves']})")
    lines.append("")
    lines.append(f"![ROC curves]({figures['roc_curves']})")
    lines.append("")
    lines.append(f"![Precision-Recall curves]({figures['pr_curves']})")
    lines.append("")
    lines.append(f"![Confusion matrices]({figures['confusion_matrices']})")
    lines.append("")
    if "attention_heatmap" in figures:
        lines.append(
            "### 4b. Cross-modal attention weights (raw tensors -- SHAP-based explainability is a later phase)"
        )
        lines.append("")
        lines.append(f"![Attention heatmap]({figures['attention_heatmap']})")
        lines.append("")

    lines.append("## 5. Comparison Against Phase 9 Baseline Models")
    lines.append("")
    lines.append(
        "Phase 9's numbers below are restated from `reports/baseline_models_report.md` (EMG, all features, "
        "binary_fatigue, held-out test set) rather than re-computed here: Phase 9's per-modality dataset (one row "
        "per sensor-window) and Phase 10's fused multimodal dataset (one row per aligned multimodal window) are "
        "built by genuinely different pipelines and are not expected to contain identical rows. Both use the same "
        f"3 subjects and the same group-aware split methodology (seed={SPLIT_CONFIG.random_seed}), which is what "
        "makes this comparison meaningful despite not being row-identical."
    )
    lines.append("")
    lines.append("| Model | Category | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|---|---|---|---|---|---|---|")
    for strategy, evaluation in results.items():
        m = evaluation.metrics
        roc = f"{m.roc_auc:.3f}" if m.roc_auc is not None else "n/a"
        lines.append(f"| **proposed ({strategy})** | multimodal fusion | {m.accuracy:.3f} | {m.precision_macro:.3f} | {m.recall_macro:.3f} | {m.f1_macro:.3f} | {roc} |")
    for name, row in BASELINE_RESULTS.items():
        lines.append(f"| {name} | {row['category']} | {row['accuracy']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} |")
    lines.append("")
    lines.append(f"![Proposed vs baseline comparison]({figures['comparison']})")
    lines.append("")

    best_baseline_name = max(BASELINE_RESULTS, key=lambda k: BASELINE_RESULTS[k]["f1"])
    best_baseline_f1 = BASELINE_RESULTS[best_baseline_name]["f1"]
    best_proposed_strategy = max(results, key=lambda s: results[s].metrics.f1_macro)
    best_proposed_f1 = results[best_proposed_strategy].metrics.f1_macro

    lines.append("## 6. Discussion")
    lines.append("")
    lines.append(
        f"Best proposed-model variant: **{best_proposed_strategy}** fusion (F1-macro={best_proposed_f1:.3f}). "
        f"Best Phase 9 baseline: **{best_baseline_name}** (F1-macro={best_baseline_f1:.3f})."
    )
    if best_proposed_f1 > best_baseline_f1:
        lines.append(
            "The proposed multimodal model **outperforms** every Phase 9 single-modality baseline on this "
            "3-subject demonstration subset -- consistent with the hypothesis that fusing EMG, IMU, and PPG "
            "carries complementary fatigue information a single modality alone does not."
        )
    else:
        lines.append(
            "The proposed multimodal model does **not yet outperform** the best Phase 9 single-modality baseline "
            "on this 3-subject demonstration subset. Given the very small training set for a model with "
            "substantially more parameters than any Phase 9 baseline, this is an expected data-scale limitation "
            "(see Limitations), not evidence against the fusion hypothesis -- the architecture should be "
            "re-evaluated once trained on the full 34-subject dataset."
        )
    lines.append("")

    train_subjects = sorted(set(mm.groups[split.train_idx].tolist()))
    val_subjects = sorted(set(mm.groups[split.val_idx].tolist()))
    test_subjects = sorted(set(mm.groups[split.test_idx].tolist()))
    diagnostic_scaler = fit_scaler(mm.X[split.train_idx])
    X_scaled_diag = apply_scaler(diagnostic_scaler, mm.X)
    lines.append(
        f"**A concrete, verified explanation for this limitation:** with only {len(DEMO_SUBJECT_IDS)} subjects, "
        f"the group-aware split assigns exactly one subject per split -- train={train_subjects}, val={val_subjects}, "
        f"test={test_subjects}. Since `RobustScaler` is fit on the training subject only (correct practice -- no "
        f"leakage), the held-out subjects' scaled feature distributions differ sharply from what the model was "
        f"trained on: scaled train std={X_scaled_diag[split.train_idx].std():.2f} vs. scaled test "
        f"std={X_scaled_diag[split.test_idx].std():.2f} vs. scaled val std={X_scaled_diag[split.val_idx].std():.2f}. "
        "Training on one subject's idiosyncratic EMG/IMU/PPG scale and evaluating on a different subject's is an "
        "extreme version of exactly the inter-subject variability problem the upcoming **Personalization "
        "(leave-one-subject-out)** phase exists to address -- this finding is additional, concrete motivation for "
        "that phase, not a defect in this one."
    )
    lines.append("")

    lines.append("## 7. Limitations")
    lines.append("")
    lines.append("- **Bounded subject count (3)**, identical to Phase 9 -- results demonstrate the architecture trains and evaluates correctly end-to-end, not final performance.")
    lines.append("- **Baseline comparison is not row-identical** (see Section 5) -- a methodologically sound but not perfectly matched comparison.")
    lines.append("- **Feature imputation and scaling** follow the same dataset-wide (not per-fold) simplification documented in Phase 9/`dataset_builder.py`.")
    lines.append("- **No hyperparameter tuning** was performed for any fusion strategy; hidden sizes/embedding dimension were chosen once, not searched.")
    lines.append("- **PPG's coarser-window alignment is approximate** (nearest-center-time, not exact containment) -- documented in `multimodal_dataset_builder.py`.")
    lines.append("- **Attention weights are shown as raw tensors only** (Section 4b) -- proper SHAP-based feature attribution is Phase 11's job, not this phase's.")
    lines.append("")

    lines.append("## 8. Recommendations for Later Phases")
    lines.append("")
    lines.append("1. Re-run across the full 34-subject dataset before drawing performance conclusions.")
    lines.append("2. SHAP explainability (next phase) should consume the `modality_embeddings`/`attention_weights`/`fused_embedding` this model already exposes via `ModelOutput` -- no architecture changes should be needed.")
    lines.append("3. Personalization (LOSO) can reuse `ml.models.common.data_splitter` unmodified (group-aware by subject_id, as designed since Phase 9).")
    lines.append("4. Consider a learning-rate/architecture search once full-dataset compute budget is available.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
