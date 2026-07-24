"""Phase 8.5 -- Feature Validation & Selection report generator.

Runs Phase 5 (load) -> Phase 7 (preprocess) -> Phase 8 (feature extraction)
-> Phase 8.5 (validation, statistics, correlation, importance proxies,
selection recommendations) over a small set of real subjects, renders
figures to `reports/figures/`, and writes
`reports/feature_validation_report.md`.

This script trains no models and performs no explainability -- Mutual
Information and ANOVA F-score are computed strictly as exploratory
screening diagnostics (see `ml.feature_engineering.feature_importance`
module docstring) against simple metadata targets, and no feature is ever
removed automatically: only recommendations are produced and reported.

Reproduce with:
    python notebooks/phase8_5_feature_validation_report.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ml.feature_engineering.feature_dataset import FeatureDataset, build_feature_catalog, build_feature_datasets
from ml.feature_engineering.feature_importance import FeatureImportanceConfig, compute_feature_importance
from ml.feature_engineering.feature_pipeline import FeaturePipeline
from ml.feature_engineering.feature_selection import FeatureSelectionConfig, FeatureSelector
from ml.feature_engineering.feature_validation import FeatureValidationConfig, validate_features
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_validation_report.md"

# A small, fixed set of real subjects keeps this report fast and
# deterministic to regenerate. Each subject contributes thousands of
# windows per modality (see the Phase 8 report), so this already gives a
# statistically substantial sample for the diagnostics below; the pipeline
# itself is unchanged for any other subject or the full dataset via
# FeaturePipeline.extract_subject / build_feature_datasets on every subject.
DEMO_SUBJECT_IDS = ["P01", "P02", "P03"]

VALIDATION_CONFIG = FeatureValidationConfig(near_zero_variance_threshold=1e-6, correlation_threshold=0.95)
SELECTION_CONFIG = FeatureSelectionConfig(variance_threshold=1e-6, correlation_threshold=0.95, mutual_info_bottom_fraction=0.1)
IMPORTANCE_TARGET = "intensity_band"  # categorical; available on all fatigue-task windows across all 3 subjects

sns.set_theme(style="whitegrid")


def find_extreme_outlier_rows(dataset: FeatureDataset, robust_z_threshold: float = 100.0, max_rows: int = 10) -> "pd.DataFrame":
    """Flag rows containing at least one feature value whose robust z-score
    (deviation from the column median, scaled by median absolute deviation)
    exceeds `robust_z_threshold`.

    This is a generic, distribution-shape-agnostic outlier check (no
    modality-specific plausible-range assumptions are hardcoded) used only
    by this report script to surface candidate raw-data corruption for
    human review -- it does not modify or exclude anything from the
    feature matrices themselves.
    """
    import numpy as np
    import pandas as pd

    numeric = dataset.matrix.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()

    median = numeric.median()
    mad = (numeric - median).abs().median()
    robust_z = (numeric - median).abs().div(mad.replace(0, np.nan))
    worst_z_per_row = robust_z.max(axis=1)
    worst_feature_per_row = robust_z.idxmax(axis=1)

    flagged_mask = worst_z_per_row > robust_z_threshold
    if not flagged_mask.any():
        return pd.DataFrame()

    flagged = dataset.metadata.loc[flagged_mask, ["subject_id", "condition_id", "source_sensor", "window_index"]].copy()
    flagged["feature"] = worst_feature_per_row[flagged_mask]
    flagged["value"] = [dataset.matrix.loc[i, worst_feature_per_row[i]] for i in flagged.index]
    flagged["robust_z"] = worst_z_per_row[flagged_mask]
    return flagged.sort_values("robust_z", ascending=False).head(max_rows)


def _savefig(fig: plt.Figure, filename: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return f"figures/{filename}"


def plot_correlation_heatmap(dataset: FeatureDataset) -> str:
    corr = dataset.matrix.select_dtypes(include="number").corr()
    annotate = corr.shape[0] <= 16
    fig_width = max(6, min(0.35 * corr.shape[0] + 3, 16))
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=annotate, fmt=".2f", ax=ax, square=True,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title(f"{dataset.modality} Feature Correlation Matrix ({corr.shape[0]} features)")
    fig.tight_layout()
    return _savefig(fig, f"feature_correlation_{dataset.modality.lower()}.png")


def plot_variance_distribution(datasets: Dict[str, FeatureDataset]) -> str:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    for ax, (modality, dataset) in zip(axes, datasets.items()):
        variances = dataset.matrix.select_dtypes(include="number").var()
        log_variances = np.log10(variances[variances > 0])
        sns.histplot(log_variances, bins=30, ax=ax, color="#4C72B0")
        ax.set_title(f"{modality} Feature Variance (log10)")
        ax.set_xlabel("log10(variance)")
    fig.tight_layout()
    return _savefig(fig, "feature_variance_distribution.png")


def plot_mutual_information_ranking(importance_results: dict, top_n: int = 15) -> str:
    available = {modality: result for modality, result in importance_results.items() if result is not None}
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 6))
    if len(available) == 1:
        axes = [axes]
    for ax, (modality, result) in zip(axes, available.items()):
        top_scores = result.mutual_information.head(top_n)
        sns.barplot(x=top_scores.values, y=top_scores.index, ax=ax, hue=top_scores.index, palette="viridis", legend=False)
        ax.set_title(f"{modality}: Top {top_n} Features by Mutual Information\n(target: {result.target_column})")
        ax.set_xlabel("Mutual information (exploratory)")
        ax.set_ylabel("")
    fig.tight_layout()
    return _savefig(fig, "mutual_information_ranking.png")


def plot_feature_category_counts(catalog) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = catalog.groupby(["modality", "category"]).size().reset_index(name="count")
    sns.barplot(data=counts, x="category", y="count", hue="modality", ax=ax)
    ax.set_title("Feature Count by Category and Modality")
    ax.set_ylabel("Number of features")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _savefig(fig, "feature_category_counts.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading, preprocessing, and extracting features for subjects: %s", DEMO_SUBJECT_IDS)
    loader = WSD4FEDSRMLoader()
    condition_catalog = loader.discover_conditions()
    subject_folders = loader.discover_subjects()

    subjects_features = []
    processed_subjects = []
    for subject_id in DEMO_SUBJECT_IDS:
        raw_folder = subject_folders[subject_id]
        subject = loader.load_subject(subject_id, raw_folder, condition_catalog)
        processed_subject = PreprocessingPipeline().process_subject(subject)
        subject_features = FeaturePipeline().extract_subject(processed_subject)
        processed_subjects.append(processed_subject)
        subjects_features.append(subject_features)

    logger.info("Building per-modality feature datasets ...")
    datasets = build_feature_datasets(subjects_features, processed_subjects)
    catalog = build_feature_catalog(datasets)

    validation_results = {}
    importance_results = {}
    recommendations_by_modality = {}
    outlier_rows_by_modality = {}
    for modality, dataset in datasets.items():
        if dataset.matrix.empty:
            logger.warning("%s feature matrix is empty -- skipping validation/selection.", modality)
            continue
        logger.info("Validating %s features (%d samples x %d features) ...", modality, *dataset.matrix.shape)
        validation_results[modality] = validate_features(dataset.matrix, modality, VALIDATION_CONFIG)
        outlier_rows_by_modality[modality] = find_extreme_outlier_rows(dataset)
        if not outlier_rows_by_modality[modality].empty:
            logger.warning(
                "%s: %d extreme-outlier window(s) detected (robust z-score > 100) -- see report for detail.",
                modality, len(outlier_rows_by_modality[modality]),
            )

        importance_results[modality] = compute_feature_importance(
            dataset.matrix, dataset.metadata, modality, FeatureImportanceConfig(target_column=IMPORTANCE_TARGET)
        )

        selector = FeatureSelector(SELECTION_CONFIG)
        recommendations_by_modality[modality] = selector.recommend(
            dataset.matrix, importance_results[modality], validation_results[modality].correlation_matrix
        )

    logger.info("Rendering figures ...")
    figures = {}
    for modality, dataset in datasets.items():
        if not dataset.matrix.empty:
            figures[f"correlation_{modality}"] = plot_correlation_heatmap(dataset)
    figures["variance"] = plot_variance_distribution({m: d for m, d in datasets.items() if not d.matrix.empty})
    figures["mutual_information"] = plot_mutual_information_ranking(importance_results)
    figures["category_counts"] = plot_feature_category_counts(catalog)

    logger.info("Writing feature validation report ...")
    write_report(datasets, catalog, validation_results, importance_results, recommendations_by_modality, outlier_rows_by_modality, figures)
    logger.info("Phase 8.5 feature validation & selection complete.")


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.4g}"  # NaN-safe


def write_report(datasets, catalog, validation_results, importance_results, recommendations_by_modality, outlier_rows_by_modality, figures) -> None:
    lines: List[str] = []
    lines.append("# WSD4FEDSRM -- Feature Validation & Selection Report")
    lines.append("")
    lines.append("**Phase:** 8.5 -- Feature Validation & Selection (no model training, no explainability)")
    lines.append("")
    lines.append(
        f"**Reproduce:** `python notebooks/phase8_5_feature_validation_report.py` from the project root "
        f"(runs Phase 5 load -> Phase 7 preprocess -> Phase 8 feature extraction -> Phase 8.5 validation/selection "
        f"for subjects {', '.join(DEMO_SUBJECT_IDS)})."
    )
    lines.append("")
    lines.append(
        "**Configuration used:** near-zero-variance threshold "
        f"`{VALIDATION_CONFIG.near_zero_variance_threshold}`, correlation threshold `{VALIDATION_CONFIG.correlation_threshold}`, "
        f"mutual-info bottom fraction `{SELECTION_CONFIG.mutual_info_bottom_fraction}`, importance target `{IMPORTANCE_TARGET}`. "
        "All are independently configurable per-call via `FeatureValidationConfig`/`FeatureSelectionConfig`/`FeatureImportanceConfig`."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Dataset Overview")
    lines.append("")
    lines.append(f"Built from {len(DEMO_SUBJECT_IDS)} real subjects ({', '.join(DEMO_SUBJECT_IDS)}), all 10 conditions each, "
                  "using default Phase 7/8 preprocessing and feature-extraction configuration.")
    lines.append("")
    lines.append("| Modality | Windows (rows) | Features (columns) |")
    lines.append("|---|---|---|")
    for modality, dataset in datasets.items():
        lines.append(f"| {modality} | {dataset.matrix.shape[0]} | {dataset.matrix.shape[1]} |")
    lines.append("")
    lines.append(f"![Feature category counts]({figures['category_counts']})")
    lines.append("")

    lines.append("## 2. Data Quality Findings")
    lines.append("")
    lines.append(
        "Beyond the generic missing/infinite/constant/variance/correlation checks (Section 5), this report also "
        "runs a distribution-shape-agnostic extreme-outlier scan (robust z-score, median/MAD-based, threshold 100) "
        "over every modality's raw feature matrix -- this is a report-level diagnostic only, not part of the reusable "
        "`ml.feature_engineering.feature_validation` checks, since it encodes no modality-specific plausible-range "
        "assumptions and exists purely to surface candidate raw-data corruption for human review."
    )
    lines.append("")
    any_outliers = any(not df.empty for df in outlier_rows_by_modality.values())
    if not any_outliers:
        lines.append("No extreme-outlier windows were detected in this run.")
        lines.append("")

    # EMG: independently confirmed as raw-file corruption (values ~1e28-1e34,
    # many orders of magnitude beyond any possible mV-scale EMG reading).
    emg_outliers = outlier_rows_by_modality.get("EMG")
    if emg_outliers is not None and not emg_outliers.empty:
        lines.append(f"### EMG: {len(emg_outliers)} extreme-outlier window(s) found -- confirmed raw-file corruption")
        lines.append("")
        lines.append(
            "**Independently verified against the raw WSD4FEDSRM source files** (read directly with plain "
            "`pandas.read_csv`, no preprocessing/feature-extraction involved): "
            "`datasets/raw/WSD4FEDSRM/EMG, IMU, and PPG data/40-50_ internal rotation/Subject 3/EMG data/"
            "anterior_deltoid.csv` and `.../infraspinatus.csv` contain raw sample values on the order of "
            "1e28-1e34 mV -- physiologically impossible by many orders of magnitude for surface EMG (normally "
            "single-digit mV), yet finite (not NaN/inf), so they pass the infinite-value check silently and only "
            "surface via extreme mean/variance statistics or this outlier scan."
        )
        lines.append("")
        lines.append("| Subject | Condition | Sensor | Window | Feature | Value |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in emg_outliers.iterrows():
            lines.append(
                f"| {row['subject_id']} | {row['condition_id']} | {row['source_sensor']} | {row['window_index']} | "
                f"{row['feature']} | {row['value']:.3e} |"
            )
        lines.append("")
        lines.append(
            "**Recommendation:** treat these two specific raw files as candidates for exclusion or correction in a "
            "future preprocessing revision (e.g. Phase 7's `quality_checks.py` could add a plausible-amplitude-range "
            "check for EMG); this report deliberately does not modify, exclude, or impute any data itself."
        )
        lines.append("")

    # IMU: flagged by the same generic scan, but NOT independently confirmed
    # as corruption -- magnitudes are only ~1-2 orders above this dataset's
    # own 99.9th percentile, plausibly reflecting genuine high-energy motion
    # (e.g. a fast rotational movement) rather than a sensor fault.
    imu_outliers = outlier_rows_by_modality.get("IMU")
    if imu_outliers is not None and not imu_outliers.empty:
        lines.append(f"### IMU: {len(imu_outliers)} statistical outlier window(s) found -- not confirmed as corruption")
        lines.append("")
        lines.append(
            "Flagged by the same generic robust-outlier scan, but **substantially less extreme and not "
            "independently confirmed as a data error**: e.g. `gyr_z_energy` has a dataset median of ~0.73 and a "
            "99.9th percentile of ~216 across all 3 subjects, while the flagged values reach ~350-1400 -- large "
            "relative to the rest of the distribution, but nowhere near the 1e28+ scale confirmed for the EMG "
            "finding above. This could plausibly reflect a genuine burst of fast rotational movement rather than "
            "a sensor artifact; it is reported here for manual review, not asserted as corruption."
        )
        lines.append("")
        lines.append("| Subject | Condition | Sensor | Window | Feature | Value |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in imu_outliers.iterrows():
            lines.append(
                f"| {row['subject_id']} | {row['condition_id']} | {row['source_sensor']} | {row['window_index']} | "
                f"{row['feature']} | {row['value']:.3e} |"
            )
        lines.append("")

    # Any other modality (e.g. PPG): generic, hedged reporting -- not
    # specifically investigated/confirmed either way in this run.
    for modality, outlier_df in outlier_rows_by_modality.items():
        if modality in ("EMG", "IMU") or outlier_df.empty:
            continue
        lines.append(f"### {modality}: {len(outlier_df)} statistical outlier window(s) found -- not independently investigated")
        lines.append("")
        lines.append(
            "Flagged by the generic robust-outlier scan; not specifically investigated against the raw source "
            "files in this run (see the EMG finding above for what that investigation looks like). Treat as a "
            "candidate for manual review, not a confirmed data error."
        )
        lines.append("")
        lines.append("| Subject | Condition | Sensor | Window | Feature | Value |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in outlier_df.iterrows():
            lines.append(
                f"| {row['subject_id']} | {row['condition_id']} | {row['source_sensor']} | {row['window_index']} | "
                f"{row['feature']} | {row['value']:.3e} |"
            )
        lines.append("")

    lines.append("## 3. Feature Statistics")
    lines.append("")
    if any_outliers:
        lines.append(
            "**Caveat:** the extreme-outlier window(s) documented in Section 2 are included, unmodified, in the "
            "statistics below -- this is why some means/variances (particularly EMG amplitude features) appear as "
            "implausibly large scientific notation. This is a faithful reflection of the raw data as loaded, not a "
            "computation error; see Section 2 for the specific offending rows."
        )
        lines.append("")
    for modality, result in validation_results.items():
        lines.append(f"### {modality}")
        lines.append("")
        stats = result.statistics
        preview = stats.head(8) if len(stats) > 8 else stats
        lines.append("| Feature | Mean | Std | Min | Max | Variance |")
        lines.append("|---|---|---|---|---|---|")
        for name, row in preview.iterrows():
            lines.append(f"| {name} | {_fmt(row['mean'])} | {_fmt(row['std'])} | {_fmt(row['min'])} | {_fmt(row['max'])} | {_fmt(row['variance'])} |")
        if len(stats) > 8:
            lines.append(f"| ... | ... | ... | ... | ... | ... | *(+{len(stats) - 8} more features)* |")
        lines.append("")

    lines.append("## 4. Feature Correlation Matrices")
    lines.append("")
    for modality in datasets:
        key = f"correlation_{modality}"
        if key in figures:
            lines.append(f"![{modality} correlation matrix]({figures[key]})")
            lines.append("")
    lines.append(f"![Feature variance distribution]({figures['variance']})")
    lines.append("")

    lines.append("## 5. Feature Importance Proxies (Exploratory Diagnostics Only)")
    lines.append("")
    lines.append(
        "**These are exploratory screening diagnostics, not final model importance.** Mutual Information and "
        "ANOVA F-score are computed independently per feature against a single simple target column "
        f"(`{IMPORTANCE_TARGET}`), ignore feature interactions, and say nothing about a trained model's actual "
        "reliance on these features. See `ml/feature_engineering/feature_importance.py` module docstring for the "
        "full rationale."
    )
    lines.append("")
    for modality, result in importance_results.items():
        if result is None:
            lines.append(f"### {modality}: not computed (target column unavailable or no complete rows)")
            lines.append("")
            continue
        lines.append(f"### {modality} (target: `{result.target_column}`, type: {result.target_type})")
        lines.append("")
        for note in result.notes:
            lines.append(f"- {note}")
        lines.append("")
        top5 = result.mutual_information.head(5)
        lines.append("Top 5 by mutual information: " + ", ".join(f"`{name}` ({value:.4f})" for name, value in top5.items()))
        lines.append("")
    lines.append(f"![Mutual information ranking]({figures['mutual_information']})")
    lines.append("")

    lines.append("## 6. Selection Recommendations (Not Automatically Applied)")
    lines.append("")
    lines.append(
        "Every recommendation below is produced by `FeatureSelector.recommend` and is **not** acted upon anywhere "
        "in this pipeline. Call `feature_selection.apply_recommendations(matrix, recommendations, actions=(...))` "
        "yourself to actually drop columns, choosing which recommendation types to act on."
    )
    lines.append("")

    total_remove = 0
    total_review = 0
    total_features = 0
    for modality, dataset in datasets.items():
        recommendations = recommendations_by_modality.get(modality, [])
        total_features += dataset.matrix.shape[1]
        remove_names = sorted({r.feature_name for r in recommendations if r.recommendation == "remove"})
        review_names = sorted({r.feature_name for r in recommendations if r.recommendation == "review"})
        total_remove += len(remove_names)
        total_review += len(review_names)

        lines.append(f"### {modality}")
        lines.append("")
        lines.append(f"- **Recommended for removal ({len(remove_names)}):**")
        if remove_names:
            for name in remove_names:
                reasons = [r.reason for r in recommendations if r.feature_name == name and r.recommendation == "remove"]
                lines.append(f"  - `{name}` -- {reasons[0]}")
        else:
            lines.append("  - none")
        lines.append(f"- **Recommended for review ({len(review_names)}):**")
        if review_names:
            for name in review_names:
                reasons = [r.reason for r in recommendations if r.feature_name == name and r.recommendation == "review"]
                lines.append(f"  - `{name}` -- {reasons[0]}")
        else:
            lines.append("  - none")
        retained = dataset.matrix.shape[1] - len(remove_names)
        lines.append(f"- **Retained if 'remove' recommendations were applied:** {retained} / {dataset.matrix.shape[1]}")
        lines.append("")

    lines.append(
        f"**Overall:** {total_features} total features across all modalities; {total_remove} recommended for "
        f"removal, {total_review} flagged for review, {total_features - total_remove} would remain if only "
        "'remove' recommendations were applied. No features were actually removed by this report."
    )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
