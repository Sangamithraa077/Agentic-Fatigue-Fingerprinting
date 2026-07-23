"""Phase 8 -- Feature Extraction Report generator.

Runs the full Phase 5 (load) -> Phase 7 (preprocess) -> Phase 8 (feature
extraction) pipeline over one representative real subject and writes
`reports/feature_engineering_report.md` summarizing: number of extracted
features, feature categories, window/signal counts, and any skipped
features with justification (see `ppg_features.SKIPPED_HRV_FEATURES`).

This script performs no model training or explainability -- it only runs
the existing loading/preprocessing/feature-extraction pipelines and reports
on their output.

Reproduce with:
    python notebooks/phase8_feature_engineering_report.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.feature_engineering.feature_pipeline import FeaturePipeline
from ml.feature_engineering.feature_vector import condition_features_to_dataframe
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_engineering_report.md"

# A single representative subject is used to keep this report fast and
# deterministic to regenerate; the pipeline itself supports the full
# dataset unchanged via FeaturePipeline.extract_subject on every subject.
DEMO_SUBJECT_ID = "P01"
DEMO_SUBJECT_FOLDER = "Subject 1"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading subject %s via WSD4FEDSRMLoader ...", DEMO_SUBJECT_ID)
    loader = WSD4FEDSRMLoader()
    condition_catalog = loader.discover_conditions()
    subject = loader.load_subject(DEMO_SUBJECT_ID, DEMO_SUBJECT_FOLDER, condition_catalog)

    logger.info("Preprocessing subject (Phase 7) ...")
    preprocessing_pipeline = PreprocessingPipeline()
    processed_subject = preprocessing_pipeline.process_subject(subject)

    logger.info("Extracting features (Phase 8) ...")
    feature_pipeline = FeaturePipeline()
    subject_features = feature_pipeline.extract_subject(processed_subject)
    summary = feature_pipeline.summary

    example_condition_id = next(iter(subject_features.conditions))
    example_df = condition_features_to_dataframe(subject_features.conditions[example_condition_id])

    per_condition_rows = []
    for condition_id, condition_features in subject_features.conditions.items():
        n_emg = sum(len(ch.windows) for ch in condition_features.emg.values())
        n_imu = sum(len(loc.windows) for loc in condition_features.imu.values())
        n_ppg = len(condition_features.ppg.windows) if condition_features.ppg else 0
        per_condition_rows.append((condition_id, n_emg, n_imu, n_ppg))

    lines = [
        "# WSD4FEDSRM -- Feature Extraction Report",
        "",
        "**Phase:** 8 -- Feature Engineering (no model training, no explainability, no cross-modality fusion)",
        "",
        f"**Reproduce:** `python notebooks/phase8_feature_engineering_report.py` from the project root "
        f"(runs Phase 5 load -> Phase 7 preprocess -> Phase 8 feature extraction for subject {DEMO_SUBJECT_ID}).",
        "",
        "---",
        "",
        "## 1. What This Report Covers",
        "",
        f"Generated from real data for one representative subject (**{DEMO_SUBJECT_ID}**, all "
        f"{len(condition_catalog)} conditions) using default configuration for every pipeline stage. "
        "The pipeline itself is unchanged for any other subject or the full dataset -- this single-subject "
        "run is chosen only to keep the report fast and deterministic to regenerate.",
        "",
        "## 2. Extraction Summary",
        "",
        f"- **Conditions processed:** {summary.n_conditions_processed}",
        f"- **EMG windows extracted:** {summary.n_emg_windows}",
        f"- **IMU windows extracted:** {summary.n_imu_windows}",
        f"- **PPG windows extracted:** {summary.n_ppg_windows}",
        f"- **Total distinct feature names:** {summary.n_total_feature_names}",
        "",
        "## 3. Feature Categories",
        "",
    ]

    for category, names in summary.feature_names_by_category.items():
        lines.append(f"### {category} ({len(names)} features)")
        lines.append("")
        lines.append(", ".join(f"`{name}`" for name in names))
        lines.append("")

    lines.append("## 4. Windows per Condition")
    lines.append("")
    lines.append("| Condition | EMG windows | IMU windows | PPG windows |")
    lines.append("|---|---|---|---|")
    for condition_id, n_emg, n_imu, n_ppg in per_condition_rows:
        lines.append(f"| {condition_id} | {n_emg} | {n_imu} | {n_ppg} |")
    lines.append("")
    lines.append(
        "PPG windows are 0 for the 4 MVIC-calibration conditions -- PPG was not recorded during MVIC trials "
        "(confirmed in Phase 5/6/7), so this is expected, not a data quality issue."
    )
    lines.append("")

    lines.append("## 5. Skipped Features (documented, not implemented)")
    lines.append("")
    for name, justification in summary.skipped_features.items():
        lines.append(f"- **{name}**: {justification}")
    lines.append("")

    lines.append("## 6. Warnings Encountered")
    lines.append("")
    if summary.warnings:
        for warning in summary.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## 7. Example Flattened Feature Table")
    lines.append("")
    lines.append(
        f"`condition_features_to_dataframe` for condition `{example_condition_id}` produces one long-format "
        f"table (shape: {example_df.shape[0]} rows x {example_df.shape[1]} columns) where each row belongs to "
        "exactly one modality/channel/window, so only that modality's feature columns are populated (the rest "
        "are NaN by construction, not missing data). One example row per modality, showing only its populated "
        "columns:"
    )
    lines.append("")
    id_cols = ["modality", "channel", "window_index"]
    for modality in ("EMG", "IMU", "PPG"):
        subset = example_df[example_df["modality"] == modality]
        if subset.empty:
            continue
        row = subset.iloc[[0]].dropna(axis=1)
        feature_cols = [c for c in row.columns if c not in id_cols]
        lines.append(f"**{modality}** (`{row.iloc[0]['channel']}`, window 0) -- {len(feature_cols)} feature columns:")
        lines.append("")
        lines.append("```")
        lines.append(row[id_cols + feature_cols].to_string(index=False))
        lines.append("```")
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote feature extraction report to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
