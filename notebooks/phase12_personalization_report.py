"""Phase 12 -- Personalization Framework report generator.

Runs Phase 5 (load) -> Phase 7 (preprocess) -> Phase 8 (feature extraction)
-> Phase 8.5 (feature datasets) -> Phase 12 (subject profiles, LOSO,
personalized prediction, similarity search) over the same bounded,
documented subset of real subjects Phases 9-11 used, renders figures, and
writes `reports/personalization_report.md`.

This script implements no new modeling framework -- it only exercises
`ml.personalization` (this phase's new code) over real data. It trains
fresh models where the phase genuinely requires new computation
(Leave-One-Subject-Out cross-validation has no equivalent Phase 9
checkpoint -- Phase 9's fixed train/val/test split is a different
partition of the data), and reuses those same LOSO folds' predictions
(rather than refitting) for the personalization comparison, per this
phase's "do not duplicate ... reuse existing pipelines" requirement.

Reproduce with:
    python notebooks/phase12_personalization_report.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from ml.explainability.explanation_report import ReportSection
from ml.feature_engineering.feature_dataset import FeatureDataset, build_feature_datasets
from ml.feature_engineering.feature_pipeline import FeaturePipeline
from ml.models.common.config import DatasetConfig
from ml.models.common.dataset_builder import ModelingDataset, build_modeling_dataset
from ml.personalization.config import LosoConfig, PersonalizationConfig, SimilarityConfig, SubjectProfileConfig
from ml.personalization.evaluation import PersonalizationComparison, compare_generic_vs_personalized, summarize_personalization_evaluation
from ml.personalization.loso import LosoSummary, evaluate_loso
from ml.personalization.personalization_model import PersonalizedPrediction, predict_personalized
from ml.personalization.report import (
    format_comparison_table,
    format_loso_table,
    format_similarity_table,
    format_subject_profile_table,
    generate_personalization_report,
)
from ml.personalization.similarity import compute_similarity_matrix, find_similar_subjects, standardize_rows
from ml.personalization.subject_profiles import SubjectProfile, build_subject_profile
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "personalization_report.md"

# Same bounded, documented subject subset as Phases 9-11, so this phase's
# LOSO folds ("train on all subjects except one") are the smallest
# non-trivial case (2 training subjects per fold) rather than a fully
# realistic-scale run -- see Limitations in the generated report.
DEMO_SUBJECT_IDS = ["P01", "P02", "P03"]
EMG_DATASET_CONFIG = DatasetConfig(modality="EMG", feature_subset="all", target_type="binary_fatigue")
LOSO_CONFIG = LosoConfig(model_name="random_forest", dataset_config=EMG_DATASET_CONFIG)
PERSONALIZATION_CONFIG = PersonalizationConfig(alpha=0.6, deviation_scale=1.0, deviation_midpoint=1.0)
SIMILARITY_CONFIG = SimilarityConfig(metric="cosine", top_k=2)

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Data preparation (Phases 5-8.5) -- reused, not duplicated
# ---------------------------------------------------------------------------


def prepare_feature_pipeline_outputs(subject_ids: List[str]):
    loader = WSD4FEDSRMLoader()
    condition_catalog = loader.discover_conditions()
    subject_folders = loader.discover_subjects()

    subjects = []
    subjects_features = []
    processed_subjects = []
    for subject_id in subject_ids:
        subject = loader.load_subject(subject_id, subject_folders[subject_id], condition_catalog)
        processed_subject = PreprocessingPipeline().process_subject(subject)
        subject_features = FeaturePipeline().extract_subject(processed_subject)
        subjects.append(subject)
        processed_subjects.append(processed_subject)
        subjects_features.append(subject_features)

    feature_datasets = build_feature_datasets(subjects_features, processed_subjects)
    return subjects, feature_datasets


# ---------------------------------------------------------------------------
# Phase 12 computation
# ---------------------------------------------------------------------------


def build_all_profiles(subjects, feature_datasets: Dict[str, FeatureDataset]) -> Dict[str, SubjectProfile]:
    config = SubjectProfileConfig()
    return {subject.subject_id: build_subject_profile(subject, feature_datasets, config) for subject in subjects}


def run_personalization(
    modeling_dataset: ModelingDataset, loso_summary: LosoSummary, profiles: Dict[str, SubjectProfile]
) -> Tuple[List[PersonalizationComparison], Dict[str, List[PersonalizedPrediction]]]:
    """Reuse each LOSO fold's already-computed held-out predictions as the
    "generic model" signal for `personalization_model.predict_personalized`
    -- a LOSO fold's test set *is* one subject's data by construction, so no
    additional model fitting is needed here."""
    comparisons: List[PersonalizationComparison] = []
    sample_predictions: Dict[str, List[PersonalizedPrediction]] = {}

    for fold in loso_summary.folds:
        subject_id = fold.held_out_subject
        mask = modeling_dataset.groups == subject_id
        X_subject = modeling_dataset.X[mask]
        y_subject = modeling_dataset.y[mask]

        if not np.array_equal(y_subject, fold.evaluation.y_true):
            raise RuntimeError(
                f"Row alignment mismatch for subject '{subject_id}' -- LOSO fold's y_true does not match "
                "the modeling dataset's own rows for this subject."
            )

        generic_probability = fold.evaluation.y_proba[:, 1]
        baseline = profiles[subject_id].emg_baseline

        personalized = [
            predict_personalized(
                subject_id, float(generic_probability[i]), X_subject[i], modeling_dataset.feature_names,
                baseline, PERSONALIZATION_CONFIG,
            )
            for i in range(len(X_subject))
        ]
        personalized_probability = np.array([p.personalized_probability for p in personalized])

        comparisons.append(compare_generic_vs_personalized(subject_id, y_subject, generic_probability, personalized_probability))
        sample_predictions[subject_id] = personalized[:5]

    return comparisons, sample_predictions


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


def plot_loso_results(loso_summary: LosoSummary) -> str:
    subjects = [fold.held_out_subject for fold in loso_summary.folds]
    metrics_by_name = {
        "accuracy": [fold.evaluation.metrics.accuracy for fold in loso_summary.folds],
        "f1_macro": [fold.evaluation.metrics.f1_macro for fold in loso_summary.folds],
        "roc_auc": [fold.evaluation.metrics.roc_auc or 0.0 for fold in loso_summary.folds],
    }
    x = np.arange(len(subjects))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, (name, values) in zip((-width, 0, width), metrics_by_name.items()):
        ax.bar(x + offset, values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"LOSO Results by Held-Out Subject ({loso_summary.model_name})")
    ax.legend()
    fig.tight_layout()
    return _savefig(fig, "loso_results.png")


def plot_subject_similarity(profiles: Dict[str, SubjectProfile]) -> str:
    matrix, subject_ids = compute_similarity_matrix(list(profiles.values()), SIMILARITY_CONFIG)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=subject_ids, yticklabels=subject_ids, ax=ax, vmin=-1, vmax=1)
    ax.set_title(f"Subject Similarity ({SIMILARITY_CONFIG.metric})")
    fig.tight_layout()
    return _savefig(fig, "subject_similarity.png")


def plot_personalization_comparison(comparisons: List[PersonalizationComparison]) -> str:
    subjects = [c.subject_id for c in comparisons]
    x = np.arange(len(subjects))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(x - width / 2, [c.generic_metrics.accuracy for c in comparisons], width, label="generic")
    axes[0].bar(x + width / 2, [c.personalized_metrics.accuracy for c in comparisons], width, label="personalized")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subjects)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].bar(x - width / 2, [c.generic_metrics.f1_macro for c in comparisons], width, label="generic")
    axes[1].bar(x + width / 2, [c.personalized_metrics.f1_macro for c in comparisons], width, label="personalized")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subjects)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1-macro")
    axes[1].set_title("F1-macro")
    axes[1].legend()

    fig.suptitle("Generic vs. Personalized Model -- Per-Subject Comparison")
    fig.tight_layout()
    return _savefig(fig, "personalization_comparison.png")


def plot_subject_embeddings(profiles: Dict[str, SubjectProfile]) -> str:
    subject_ids = list(profiles.keys())
    raw = np.vstack([profiles[sid].summary_vector for sid in subject_ids])
    standardized = standardize_rows(raw)
    n_components = min(2, len(subject_ids) - 1) if len(subject_ids) > 1 else 1
    coords = PCA(n_components=n_components, random_state=42).fit_transform(standardized)
    if n_components == 1:
        coords = np.column_stack([coords, np.zeros(len(subject_ids))])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=120, color="#4C72B0")
    for i, subject_id in enumerate(subject_ids):
        ax.annotate(subject_id, (coords[i, 0], coords[i, 1]), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Subject Embeddings (PCA of standardized profile summary vectors)")
    fig.tight_layout()
    return _savefig(fig, "subject_embeddings.png")


def plot_baseline_distribution(profiles: Dict[str, SubjectProfile]) -> str:
    representative_features = ("mav", "rms", "variance", "waveform_length")
    subject_ids = list(profiles.keys())
    available_features = [
        f for f in representative_features if all(f in profiles[sid].emg_baseline.feature_means for sid in subject_ids)
    ]
    if not available_features:
        available_features = sorted(set.intersection(*(set(profiles[sid].emg_baseline.feature_means) for sid in subject_ids)))[:4]

    # Small multiples, one panel per feature: EMG features have wildly
    # different natural magnitudes (e.g. waveform_length integrates over a
    # window and is orders of magnitude larger than variance), so a single
    # shared y-axis made every feature but the largest invisible.
    fig, axes = plt.subplots(1, len(available_features), figsize=(3.2 * len(available_features), 4.5))
    axes = np.atleast_1d(axes)
    colors = sns.color_palette("deep", len(subject_ids))
    for ax, feature in zip(axes, available_features):
        means = [profiles[sid].emg_baseline.feature_means[feature] for sid in subject_ids]
        stds = [profiles[sid].emg_baseline.feature_stds[feature] for sid in subject_ids]
        ax.bar(subject_ids, means, yerr=stds, capsize=4, color=colors)
        ax.set_title(feature)
    axes[0].set_ylabel("Baseline mean feature value (+/- std)")
    fig.suptitle("EMG Baseline Feature Distribution by Subject")
    fig.tight_layout()
    return _savefig(fig, "baseline_distribution.png")


# ---------------------------------------------------------------------------
# Report section builders
# ---------------------------------------------------------------------------


def build_overview_section(n_subjects: int) -> ReportSection:
    content = (
        "This phase implements a modular personalization framework (`ml.personalization/`) that adapts fatigue "
        f"prediction to individual subjects, evaluated over the same bounded, documented {n_subjects}-subject demo "
        "(P01, P02, P03) Phases 9-11 used. It reuses Phase 5's raw `Subject` loader (demographics/anthropometrics), "
        "Phase 8.5's `FeatureDataset`s (EMG baseline statistics, fatigue history), and Phase 9's classical model "
        "builders/evaluation utilities (`evaluate_model`, `ClassificationMetrics`) throughout -- no preprocessing or "
        "feature engineering is duplicated.\n\n"
        "Four capabilities are covered: (1) **subject profiles** combining demographics, anthropometrics, per-"
        "modality baseline statistics, and fatigue history; (2) **Leave-One-Subject-Out (LOSO) cross-validation**, "
        "training on all-but-one subject and testing on the held-out one, repeated for every subject; "
        "(3) a **personalization layer** blending each LOSO fold's \"generic\" (population, minus this subject) "
        "prediction with a subject-baseline deviation signal; and (4) **subject similarity search** over profile "
        "summary vectors."
    )
    return ReportSection(title="1. Overview", content=content)


def build_profile_section(profiles: Dict[str, SubjectProfile]) -> ReportSection:
    content = (
        "Built via `build_subject_profile()`, combining Phase 5's demographic/anthropometric records with "
        "per-subject baseline statistics and fatigue history derived from Phase 8.5's EMG/IMU/PPG feature "
        "matrices.\n\n" + format_subject_profile_table(list(profiles.values()))
    )
    return ReportSection(title="2. Subject Profile Overview", content=content)


def build_loso_section(loso_summary: LosoSummary, figure_path: str) -> ReportSection:
    content = (
        f"`evaluate_loso()` trained a fresh `{loso_summary.model_name}` per fold (2 training subjects, 1 held-out "
        "test subject) via `sklearn.model_selection.LeaveOneGroupOut`, reusing `ml.models.common.evaluation."
        "evaluate_model` for metrics unmodified.\n\n"
        + format_loso_table(loso_summary)
        + f"\n\n![LOSO results]({figure_path})"
    )
    return ReportSection(title="3. LOSO Evaluation", content=content)


def build_personalized_prediction_section(sample_predictions: Dict[str, List[PersonalizedPrediction]], config: PersonalizationConfig) -> ReportSection:
    lines = [
        f"`predict_personalized()` blends each held-out window's generic (LOSO-fold) probability with a "
        f"subject-baseline deviation signal: `personalized = alpha * generic + (1 - alpha) * deviation` "
        f"(alpha={config.alpha}). The deviation signal is the mean absolute z-score of the window's raw EMG "
        "features against *that subject's own* baseline mean/std (`baseline_builder.ModalityBaseline`), squashed "
        f"through a logistic function (scale={config.deviation_scale}, midpoint={config.deviation_midpoint}).",
        "",
        "Representative sample predictions (first 5 held-out windows per subject):",
        "",
        "| Subject | Global probability | Baseline deviation | Deviation probability | Personalized probability | Personalized class |",
        "|---|---|---|---|---|---|",
    ]
    for subject_id, predictions in sample_predictions.items():
        for p in predictions:
            lines.append(
                f"| {subject_id} | {p.global_probability:.3f} | {p.baseline_deviation_score:.3f} | "
                f"{p.deviation_probability:.3f} | {p.personalized_probability:.3f} | {p.personalized_class} |"
            )
    return ReportSection(title="4. Personalized Prediction Results", content="\n".join(lines))


def build_similarity_section(profiles: Dict[str, SubjectProfile], heatmap_path: str, embeddings_path: str) -> ReportSection:
    lines = [
        f"`find_similar_subjects()` over each profile's standardized `summary_vector` "
        f"(metric={SIMILARITY_CONFIG.metric}), covering demographics, anthropometrics, fatigue history, and "
        "per-modality baseline levels.",
        "",
        "*Note on the scores below:* with only 3 subjects, standardizing each column forces it to sum to "
        "(approximately) zero across the population, so any two subjects' standardized vectors are mathematically "
        "likely to disagree in sign on several dimensions -- cosine similarity being negative here is an expected "
        "small-N standardization artifact, not a sign of a broken computation; the *ranking* between candidates "
        "remains meaningful even though the absolute scores are not directly comparable to a larger-population run.",
        "",
    ]
    for subject_id in profiles:
        results = find_similar_subjects(subject_id, list(profiles.values()), SIMILARITY_CONFIG)
        lines.append(format_similarity_table(subject_id, results))
        lines.append("")
    lines.append(f"![Subject similarity]({heatmap_path})")
    lines.append("")
    lines.append(f"![Subject embeddings]({embeddings_path})")
    return ReportSection(title="5. Similarity Analysis", content="\n".join(lines))


def build_comparison_section(evaluation_summary, comparison_figure: str, baseline_figure: str) -> ReportSection:
    content = (
        format_comparison_table(evaluation_summary)
        + f"\n\n![Personalization comparison]({comparison_figure})"
        + f"\n\n![Baseline distribution]({baseline_figure})"
    )
    return ReportSection(title="6. Performance Comparison", content=content)


def build_discussion_section(evaluation_summary) -> ReportSection:
    content = (
        f"Personalization changed mean F1-macro by {evaluation_summary.mean_f1_improvement:+.3f} and mean accuracy "
        f"by {evaluation_summary.mean_accuracy_improvement:+.3f} across the {len(evaluation_summary.comparisons)} "
        f"held-out subjects, improving F1 for {evaluation_summary.n_subjects_improved_f1}/"
        f"{len(evaluation_summary.comparisons)} of them. The personalization layer is deliberately simple (a "
        "baseline-deviation blend, not a second trained model) so its behavior is directly attributable: a subject "
        "whose held-out windows look statistically unlike their own historical baseline gets pulled toward the "
        "positive (fatigued) class regardless of what the generic model says, which only helps when that deviation "
        "signal actually correlates with true fatigue for that subject -- exactly the kind of per-subject behavior "
        "LOSO's already-severe inter-subject variability (Section 3) predicts will vary."
    )
    return ReportSection(title="7. Discussion", content=content)


def build_limitations_section() -> ReportSection:
    items = [
        "**Bounded subject count (3)**, identical to Phases 9-11 -- LOSO with 3 subjects trains each fold on only "
        "2 subjects, the smallest non-degenerate case; this demonstrates the framework end-to-end, not final LOSO "
        "generalization performance.",
        "**The personalization layer is a heuristic blend, not a learned combiner** -- `alpha` was chosen once, not "
        "tuned or validated against a held-out personalization-specific split.",
        "**Baseline statistics use a subject's *entire* window history** (including the very windows LOSO evaluates "
        "on), since Phase 12's 3-subject demo has no spare data for a separate baseline-calibration split -- a "
        "production deployment should calibrate each subject's baseline from an initial period before relying on "
        "personalized predictions for that subject.",
        "**Similarity search compares only the fixed 10-dimensional summary vector**, not the full per-feature "
        "baseline -- a coarser representation chosen for cross-subject comparability, not the richest possible one.",
        "**No agentic reasoning** -- this phase's outputs (profiles, LOSO results, personalized predictions, "
        "similarity) are designed to be consumed by Phase 13's agentic framework, not to implement it.",
    ]
    return ReportSection(title="8. Limitations", content="\n".join(f"- {item}" for item in items))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start_time = time.time()

    logger.info("Rebuilding feature datasets for subjects: %s", DEMO_SUBJECT_IDS)
    subjects, feature_datasets = prepare_feature_pipeline_outputs(DEMO_SUBJECT_IDS)
    modeling_dataset = build_modeling_dataset(feature_datasets["EMG"], EMG_DATASET_CONFIG)
    logger.info("EMG modeling dataset: X=%s, %d feature(s).", modeling_dataset.X.shape, len(modeling_dataset.feature_names))

    logger.info("Building subject profiles ...")
    profiles = build_all_profiles(subjects, feature_datasets)

    logger.info("Running LOSO cross-validation (%s) ...", LOSO_CONFIG.model_name)
    loso_summary = evaluate_loso(modeling_dataset, LOSO_CONFIG)

    logger.info("Running personalized prediction + comparison ...")
    comparisons, sample_predictions = run_personalization(modeling_dataset, loso_summary, profiles)
    evaluation_summary = summarize_personalization_evaluation(comparisons)

    logger.info("Rendering figures ...")
    loso_figure = plot_loso_results(loso_summary)
    similarity_figure = plot_subject_similarity(profiles)
    comparison_figure = plot_personalization_comparison(comparisons)
    embeddings_figure = plot_subject_embeddings(profiles)
    baseline_figure = plot_baseline_distribution(profiles)

    logger.info("Writing report ...")
    total_seconds = time.time() - start_time
    sections = [
        build_overview_section(len(DEMO_SUBJECT_IDS)),
        build_profile_section(profiles),
        build_loso_section(loso_summary, loso_figure),
        build_personalized_prediction_section(sample_predictions, PERSONALIZATION_CONFIG),
        build_similarity_section(profiles, similarity_figure, embeddings_figure),
        build_comparison_section(evaluation_summary, comparison_figure, baseline_figure),
        build_discussion_section(evaluation_summary),
        build_limitations_section(),
    ]
    preamble = (
        "**Phase:** 12 -- Personalization Framework (LOSO evaluation, subject baselines, personalized prediction, "
        "similarity search; consumed directly by Phase 13's agentic framework -- no agentic reasoning here)\n\n"
        f"**Reproduce:** `python notebooks/phase12_personalization_report.py` from the project root (took "
        f"{total_seconds:.1f}s on this machine, CPU only)."
    )
    generate_personalization_report(sections, REPORT_PATH, report_title="WSD4FEDSRM -- Personalization Report", preamble=preamble)
    logger.info("Phase 12 personalization report complete in %.1fs.", total_seconds)


if __name__ == "__main__":
    main()
