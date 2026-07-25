"""Markdown report assembly for the personalization framework.

Reuses `ml.explainability.explanation_report.generate_report`/`ReportSection`
-- the generic "assemble titled markdown sections into one file" logic Phase
11 already built has no explainability-specific knowledge, so re-implementing
it here would be exactly the kind of duplication this project's engineering
standard forbids. `generate_personalization_report` is a thin, phase-scoped
re-export so this module still exposes the exact API name Phase 12 requires.

The `format_*` helpers below are this phase's equivalent of Phase 11's
`format_feature_ranking_table`/etc.: turning this package's typed result
dataclasses into markdown tables, reused verbatim regardless of which
subject/model produced the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ml.explainability.explanation_report import ReportSection, generate_report as _generate_report
from ml.personalization.evaluation import PersonalizationEvaluationSummary
from ml.personalization.loso import LosoSummary
from ml.personalization.similarity import SimilarityResult
from ml.personalization.subject_profiles import SubjectProfile


def generate_personalization_report(
    sections: Sequence[ReportSection], output_path: Path, report_title: str, preamble: str = ""
) -> Path:
    """Assemble titled sections into `reports/personalization_report.md`.

    A direct pass-through to `ml.explainability.explanation_report.generate_report`
    -- see that function's docstring for the exact assembly behavior.
    """
    return _generate_report(sections, output_path, report_title, preamble)


def format_subject_profile_table(profiles: Sequence[SubjectProfile]) -> str:
    """Markdown table: one row per subject's profile overview."""
    lines = [
        "| Subject | Age | Height (cm) | Mass (kg) | BMI | Mean RPE | Fatigue trials | EMG windows | IMU windows | PPG windows |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for profile in profiles:
        age = profile.demographic.age if profile.demographic else "n/a"
        height = f"{profile.demographic.height_cm:.1f}" if profile.demographic else "n/a"
        mass = f"{profile.anthropometric.mass_kg:.1f}" if profile.anthropometric else "n/a"
        bmi = f"{profile.anthropometric.bmi:.1f}" if profile.anthropometric else "n/a"
        mean_rpe = f"{profile.fatigue_history.mean_rpe:.1f}" if profile.fatigue_history.mean_rpe is not None else "n/a"
        lines.append(
            f"| {profile.subject_id} | {age} | {height} | {mass} | {bmi} | {mean_rpe} | "
            f"{profile.fatigue_history.n_trials} | {profile.emg_baseline.n_windows} | "
            f"{profile.imu_baseline.n_windows} | {profile.ppg_baseline.n_windows} |"
        )
    return "\n".join(lines)


def format_loso_table(summary: LosoSummary) -> str:
    """Markdown table: per-fold LOSO metrics plus the overall mean row."""
    lines = [
        "| Held-out subject | Trained on | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fold in summary.folds:
        m = fold.evaluation.metrics
        roc = f"{m.roc_auc:.3f}" if m.roc_auc is not None else "n/a"
        pr = f"{m.pr_auc:.3f}" if m.pr_auc is not None else "n/a"
        lines.append(
            f"| {fold.held_out_subject} | {', '.join(fold.train_subjects)} | {m.accuracy:.3f} | "
            f"{m.precision_macro:.3f} | {m.recall_macro:.3f} | {m.f1_macro:.3f} | {roc} | {pr} |"
        )
    roc_mean = f"{summary.mean_roc_auc:.3f}" if summary.mean_roc_auc is not None else "n/a"
    pr_mean = f"{summary.mean_pr_auc:.3f}" if summary.mean_pr_auc is not None else "n/a"
    lines.append(
        f"| **mean** | -- | **{summary.mean_accuracy:.3f}** | **{summary.mean_precision:.3f}** | "
        f"**{summary.mean_recall:.3f}** | **{summary.mean_f1:.3f}** | **{roc_mean}** | **{pr_mean}** |"
    )
    return "\n".join(lines)


def format_similarity_table(target_subject_id: str, results: Sequence[SimilarityResult]) -> str:
    """Markdown table: subjects most similar to `target_subject_id`."""
    lines = [f"Most similar subjects to **{target_subject_id}**:", "", "| Rank | Subject | Similarity |", "|---|---|---|"]
    for rank, result in enumerate(results, start=1):
        lines.append(f"| {rank} | {result.subject_id} | {result.score:.3f} |")
    return "\n".join(lines)


def format_comparison_table(summary: PersonalizationEvaluationSummary) -> str:
    """Markdown table: per-subject generic-vs-personalized improvement, plus the mean row."""
    lines = [
        "| Subject | Generic Accuracy | Personalized Accuracy | Generic F1 | Personalized F1 | "
        "Accuracy Δ | F1 Δ | ROC-AUC Δ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for comparison in summary.comparisons:
        roc_delta = f"{comparison.roc_auc_improvement:+.3f}" if comparison.roc_auc_improvement is not None else "n/a"
        lines.append(
            f"| {comparison.subject_id} | {comparison.generic_metrics.accuracy:.3f} | "
            f"{comparison.personalized_metrics.accuracy:.3f} | {comparison.generic_metrics.f1_macro:.3f} | "
            f"{comparison.personalized_metrics.f1_macro:.3f} | {comparison.accuracy_improvement:+.3f} | "
            f"{comparison.f1_improvement:+.3f} | {roc_delta} |"
        )
    roc_mean = f"{summary.mean_roc_auc_improvement:+.3f}" if summary.mean_roc_auc_improvement is not None else "n/a"
    lines.append(
        f"| **mean** | -- | -- | -- | -- | **{summary.mean_accuracy_improvement:+.3f}** | "
        f"**{summary.mean_f1_improvement:+.3f}** | **{roc_mean}** |"
    )
    return "\n".join(lines)
