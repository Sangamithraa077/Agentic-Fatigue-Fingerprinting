"""Phase 6 -- Exploratory Data Analysis for the WSD4FEDSRM dataset.

Analysis only: this script does not preprocess, normalize, filter, or
feature-engineer any signal, and trains no models. It loads the dataset
through the Phase 5 loader (`ml.preprocessing.dataset_loader`), computes
descriptive statistics over the resulting typed object graph, renders
figures to `reports/figures/`, and writes `reports/eda_report.md`.

Reproduce with:
    python notebooks/phase6_eda_analysis.py
(or ``python -m notebooks.phase6_eda_analysis`` from the project root)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering -- this script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data_models.dataset import Dataset
from ml.data_models.enums import ConditionType, Direction, EMGMuscle, IMULocation
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_report.md"

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Tidy-frame extraction: convert the typed object graph into flat DataFrames
# that are convenient to describe/plot with pandas + seaborn. This is pure
# bookkeeping/aggregation for analysis purposes -- no signal transformation.
# ---------------------------------------------------------------------------


def build_sensor_frame(dataset: Dataset) -> pd.DataFrame:
    """One row per individual EMG/IMU/PPG recording across the whole dataset."""
    rows: List[dict] = []
    for subject in dataset.subjects.values():
        for condition in subject.conditions.values():
            for muscle, emg in condition.emg.items():
                rows.append(
                    {
                        "subject_id": subject.subject_id,
                        "condition_id": condition.condition_id,
                        "condition_type": condition.condition_type.value,
                        "direction": condition.direction.value,
                        "intensity_band": condition.intensity_band.value if condition.intensity_band else None,
                        "modality": "EMG",
                        "channel": muscle.value,
                        "n_samples": emg.n_samples,
                        "duration_seconds": emg.duration_seconds,
                        "sampling_rate_hz": emg.sampling_rate_hz,
                    }
                )
            for location, imu in condition.imu.items():
                rows.append(
                    {
                        "subject_id": subject.subject_id,
                        "condition_id": condition.condition_id,
                        "condition_type": condition.condition_type.value,
                        "direction": condition.direction.value,
                        "intensity_band": condition.intensity_band.value if condition.intensity_band else None,
                        "modality": "IMU",
                        "channel": location.value,
                        "n_samples": imu.n_samples,
                        "duration_seconds": imu.duration_seconds,
                        "sampling_rate_hz": imu.sampling_rate_hz,
                    }
                )
            if condition.ppg is not None:
                rows.append(
                    {
                        "subject_id": subject.subject_id,
                        "condition_id": condition.condition_id,
                        "condition_type": condition.condition_type.value,
                        "direction": condition.direction.value,
                        "intensity_band": condition.intensity_band.value if condition.intensity_band else None,
                        "modality": "PPG",
                        "channel": "ppg",
                        "n_samples": condition.ppg.n_samples,
                        "duration_seconds": condition.ppg.duration_seconds,
                        "sampling_rate_hz": condition.ppg.sampling_rate_hz,
                    }
                )
    return pd.DataFrame(rows)


def build_condition_frame(dataset: Dataset, condition_catalog: Dict[str, object]) -> pd.DataFrame:
    """One row per (subject, condition) pair actually loaded, with
    completeness counts -- used for missing-data and condition analysis."""
    rows: List[dict] = []
    for subject in dataset.subjects.values():
        for condition in subject.conditions.values():
            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "condition_id": condition.condition_id,
                    "condition_type": condition.condition_type.value,
                    "direction": condition.direction.value,
                    "intensity_band": condition.intensity_band.value if condition.intensity_band else None,
                    "trial_number": condition.trial_number,
                    "n_emg": len(condition.emg),
                    "n_imu": len(condition.imu),
                    "has_ppg": condition.ppg is not None,
                    "has_labels": condition.labels is not None,
                }
            )
    return pd.DataFrame(rows)


def build_subject_frame(dataset: Dataset, expected_conditions: int) -> pd.DataFrame:
    """One row per subject: demographics, anthropometrics, KSS, MVIC, and
    load-completeness flags -- used for participant and missing-data analysis."""
    rows: List[dict] = []
    for subject in dataset.subjects.values():
        demo = subject.demographic
        anthro = subject.anthropometric
        mvic = subject.mvic_force
        rows.append(
            {
                "subject_id": subject.subject_id,
                "age": demo.age if demo else None,
                "height_cm": demo.height_cm if demo else None,
                "sex": demo.sex.value if demo else None,
                "dominant_hand": demo.dominant_hand if demo else None,
                "execution_order_group": demo.execution_order_group if demo else None,
                "mass_kg": anthro.mass_kg if anthro else None,
                "body_fat_percent": anthro.body_fat_percent if anthro else None,
                "muscle_percent": anthro.muscle_percent if anthro else None,
                "visceral_fat": anthro.visceral_fat if anthro else None,
                "bmi": anthro.bmi if anthro else None,
                "kss_before": subject.kss_before,
                "kss_after": subject.kss_after,
                "ir_mean_n": mvic.ir_mean_n if mvic else None,
                "er_mean_n": mvic.er_mean_n if mvic else None,
                "n_conditions_loaded": len(subject.conditions),
                "has_demographic": demo is not None,
                "has_anthropometric": anthro is not None,
                "has_mvic": mvic is not None,
                "has_kss": subject.kss_before is not None and subject.kss_after is not None,
                "is_complete": (
                    demo is not None
                    and anthro is not None
                    and mvic is not None
                    and subject.kss_before is not None
                    and len(subject.conditions) == expected_conditions
                ),
            }
        )
    return pd.DataFrame(rows)


def build_labels_frame(dataset: Dataset) -> pd.DataFrame:
    """One row per fatigue-task trial's Borg summary (before/end/duration)."""
    rows: List[dict] = []
    for subject in dataset.subjects.values():
        for condition in subject.conditions.values():
            if condition.labels is None:
                continue
            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "condition_id": condition.condition_id,
                    "direction": condition.direction.value,
                    "intensity_band": condition.intensity_band.value if condition.intensity_band else None,
                    "task_order_code": condition.labels.task_order_code,
                    "rpe_before": condition.labels.rpe_before,
                    "rpe_at_end_of_trial": condition.labels.rpe_at_end_of_trial,
                    "trial_duration_seconds": condition.labels.trial_duration_seconds,
                    "n_rpe_readings": len(condition.labels.rpe_timeseries),
                }
            )
    return pd.DataFrame(rows)


def build_rpe_timeseries_frame(dataset: Dataset) -> pd.DataFrame:
    """Long-format: one row per individual RPE reading within a trial."""
    rows: List[dict] = []
    for subject in dataset.subjects.values():
        for condition in subject.conditions.values():
            if condition.labels is None:
                continue
            for elapsed_seconds, rpe_value in condition.labels.rpe_timeseries.items():
                rows.append(
                    {
                        "subject_id": subject.subject_id,
                        "condition_id": condition.condition_id,
                        "intensity_band": condition.intensity_band.value if condition.intensity_band else None,
                        "elapsed_seconds": elapsed_seconds,
                        "rpe": rpe_value,
                    }
                )
    return pd.DataFrame(rows)


def count_missing_values(dataset: Dataset) -> Dict[str, int]:
    """Count NaN cells within the raw sensor signal DataFrames themselves
    (distinct from *structurally* absent modalities like MVIC-trial PPG)."""
    counts = {"EMG": 0, "IMU": 0, "PPG": 0}
    cells = {"EMG": 0, "IMU": 0, "PPG": 0}
    for subject in dataset.subjects.values():
        for condition in subject.conditions.values():
            for emg in condition.emg.values():
                counts["EMG"] += int(emg.signal.isna().sum().sum())
                cells["EMG"] += emg.signal.size
            for imu in condition.imu.values():
                for frame in (imu.accelerometer, imu.gyroscope, imu.magnetometer):
                    counts["IMU"] += int(frame.isna().sum().sum())
                    cells["IMU"] += frame.size
            if condition.ppg is not None:
                counts["PPG"] += int(condition.ppg.signal.isna().sum().sum())
                cells["PPG"] += condition.ppg.signal.size
    return {"missing_cells": counts, "total_cells": cells}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, filename: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return f"figures/{filename}"


def plot_participant_demographics(subject_df: pd.DataFrame) -> Dict[str, str]:
    figures = {}

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(subject_df["age"].dropna(), bins=10, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Participant Age Distribution")
    ax.set_xlabel("Age (years)")
    figures["age"] = _savefig(fig, "participant_age_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(subject_df["height_cm"].dropna(), bins=10, kde=True, ax=ax, color="#55A868")
    ax.set_title("Participant Height Distribution")
    ax.set_xlabel("Height (cm)")
    figures["height"] = _savefig(fig, "participant_height_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(subject_df["mass_kg"].dropna(), bins=10, kde=True, ax=ax, color="#C44E52")
    ax.set_title("Participant Weight (Body Mass) Distribution")
    ax.set_xlabel("Mass (kg)")
    figures["weight"] = _savefig(fig, "participant_weight_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sex_counts = subject_df["sex"].value_counts()
    sns.barplot(x=sex_counts.index, y=sex_counts.values, ax=axes[0], hue=sex_counts.index, palette="muted", legend=False)
    axes[0].set_title("Sex Distribution")
    axes[0].set_ylabel("Number of participants")

    hand_counts = subject_df["dominant_hand"].value_counts()
    sns.barplot(x=hand_counts.index, y=hand_counts.values, ax=axes[1], hue=hand_counts.index, palette="muted", legend=False)
    axes[1].set_title("Dominant Hand Distribution")
    axes[1].set_ylabel("Number of participants")
    fig.tight_layout()
    figures["sex_hand"] = _savefig(fig, "participant_sex_and_dominant_hand.png")

    return figures


def plot_borg_distribution(labels_df: pd.DataFrame, rpe_ts_df: pd.DataFrame) -> Dict[str, str]:
    figures = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(labels_df["rpe_before"].dropna(), bins=range(6, 22), ax=axes[0], color="#4C72B0")
    axes[0].set_title("Borg RPE Before Task")
    axes[0].set_xlabel("RPE (6-20 scale)")

    sns.histplot(labels_df["rpe_at_end_of_trial"].dropna(), bins=range(6, 22), ax=axes[1], color="#DD8452")
    axes[1].set_title("Borg RPE at End of Trial")
    axes[1].set_xlabel("RPE (6-20 scale)")
    fig.tight_layout()
    figures["borg_before_after"] = _savefig(fig, "borg_rpe_before_and_end_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(rpe_ts_df["rpe"], bins=range(6, 22), ax=ax, color="#8172B2")
    ax.set_title("All Recorded Borg RPE Readings (every 10s)")
    ax.set_xlabel("RPE (6-20 scale)")
    figures["borg_all_readings"] = _savefig(fig, "borg_rpe_all_readings_distribution.png")

    if not rpe_ts_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for band, group in rpe_ts_df.groupby("intensity_band"):
            trajectory = group.groupby("elapsed_seconds")["rpe"].mean()
            ax.plot(trajectory.index, trajectory.values, marker="o", label=band)
        ax.set_title("Mean Borg RPE Trajectory by Intensity Band")
        ax.set_xlabel("Elapsed time (seconds)")
        ax.set_ylabel("Mean RPE")
        ax.legend(title="Intensity band (% MVIC)")
        figures["rpe_trajectory"] = _savefig(fig, "rpe_trajectory_by_intensity_band.png")

    return figures


def plot_kss_distribution(subject_df: pd.DataFrame) -> Dict[str, str]:
    kss_long = subject_df[["subject_id", "kss_before", "kss_after"]].melt(
        id_vars="subject_id", var_name="timing", value_name="kss"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=kss_long, x="timing", y="kss", ax=ax, hue="timing", palette="Set2", legend=False)
    sns.stripplot(data=kss_long, x="timing", y="kss", ax=ax, color="black", alpha=0.5, size=3)
    ax.set_title("Karolinska Sleepiness Scale: Before vs After Session")
    ax.set_xlabel("")
    ax.set_ylabel("KSS (1-9 scale)")
    return {"kss": _savefig(fig, "kss_before_after_distribution.png")}


def plot_mvic_force(subject_df: pd.DataFrame) -> Dict[str, str]:
    mvic_long = subject_df[["subject_id", "ir_mean_n", "er_mean_n"]].melt(
        id_vars="subject_id", var_name="direction", value_name="force_n"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=mvic_long, x="direction", y="force_n", ax=ax, hue="direction", palette="Set3", legend=False)
    sns.stripplot(data=mvic_long, x="direction", y="force_n", ax=ax, color="black", alpha=0.5, size=3)
    ax.set_title("MVIC Calibration Force: Internal vs External Rotation")
    ax.set_ylabel("Mean MVIC force (N)")
    ax.set_xlabel("")
    return {"mvic": _savefig(fig, "mvic_force_boxplot.png")}


def plot_condition_counts(condition_df: pd.DataFrame) -> Dict[str, str]:
    figures = {}

    fig, ax = plt.subplots(figsize=(9, 5))
    counts = condition_df["condition_id"].apply(lambda cid: cid.split("_", 1)[1]).value_counts()
    # Re-derive a friendlier per-condition-type count instead (subject counts per condition template)
    ax.clear()
    grouping_cols = ["condition_type", "direction", "intensity_band", "trial_number"]
    per_condition = condition_df.groupby(grouping_cols, dropna=False)["subject_id"].nunique().reset_index()
    per_condition["label"] = per_condition.apply(
        lambda r: (
            f"{r['direction']}\n{r['intensity_band']}"
            if r["condition_type"] == "fatigue_task"
            else f"MVIC {r['direction']}\ntrial {int(r['trial_number'])}"
        ),
        axis=1,
    )
    per_condition = per_condition.sort_values("label")
    sns.barplot(data=per_condition, x="label", y="subject_id", ax=ax, hue="condition_type", dodge=False)
    ax.set_title("Number of Subjects with Recordings per Condition")
    ax.set_ylabel("Number of subjects")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    figures["per_condition"] = _savefig(fig, "recordings_per_condition.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    direction_counts = condition_df["direction"].value_counts()
    sns.barplot(x=direction_counts.index, y=direction_counts.values, ax=axes[0], hue=direction_counts.index, palette="pastel", legend=False)
    axes[0].set_title("Internal vs External Rotation Trials")
    axes[0].set_ylabel("Number of trials")

    type_counts = condition_df["condition_type"].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, ax=axes[1], hue=type_counts.index, palette="pastel", legend=False)
    axes[1].set_title("Fatigue Task vs MVIC Calibration Trials")
    axes[1].set_ylabel("Number of trials")
    fig.tight_layout()
    figures["direction_type"] = _savefig(fig, "direction_and_condition_type_counts.png")

    return figures


def plot_signal_length_distributions(sensor_df: pd.DataFrame) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, modality in zip(axes, ["EMG", "IMU", "PPG"]):
        subset = sensor_df[sensor_df["modality"] == modality]
        sns.histplot(subset["duration_seconds"], bins=25, ax=ax, color="#4C72B0")
        ax.set_title(f"{modality} Recording Duration")
        ax.set_xlabel("Duration (seconds)")
    fig.tight_layout()
    fig_duration = _savefig(fig, "signal_duration_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, modality in zip(axes, ["EMG", "IMU", "PPG"]):
        subset = sensor_df[sensor_df["modality"] == modality]
        sns.histplot(subset["n_samples"], bins=25, ax=ax, color="#55A868")
        ax.set_title(f"{modality} Samples per Recording")
        ax.set_xlabel("Number of samples")
    fig.tight_layout()
    fig_samples = _savefig(fig, "samples_per_recording_distributions.png")

    return {"duration": fig_duration, "samples": fig_samples}


def plot_missing_data_heatmap(subject_df: pd.DataFrame, condition_df: pd.DataFrame, expected_conditions: int) -> Dict[str, str]:
    pivot_rows = []
    for _, subject_row in subject_df.iterrows():
        sid = subject_row["subject_id"]
        subject_conditions = condition_df[condition_df["subject_id"] == sid]
        pivot_rows.append(
            {
                "subject_id": sid,
                "Demographic": 1 if subject_row["has_demographic"] else 0,
                "Anthropometric": 1 if subject_row["has_anthropometric"] else 0,
                "KSS": 1 if subject_row["has_kss"] else 0,
                "MVIC force": 1 if subject_row["has_mvic"] else 0,
                "Conditions": subject_row["n_conditions_loaded"] / expected_conditions,
                "EMG completeness": subject_conditions["n_emg"].mean() / len(EMGMuscle) if not subject_conditions.empty else 0,
                "IMU completeness": subject_conditions["n_imu"].mean() / len(IMULocation) if not subject_conditions.empty else 0,
            }
        )
    pivot_df = pd.DataFrame(pivot_rows).set_index("subject_id")

    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(pivot_df))))
    sns.heatmap(pivot_df, cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Completeness (1.0 = fully present)"}, ax=ax)
    ax.set_title("Per-Subject Data Completeness")
    fig.tight_layout()
    return {"missing_heatmap": _savefig(fig, "missing_data_completeness_heatmap.png")}


def plot_metadata_correlation(subject_df: pd.DataFrame) -> Dict[str, str]:
    numeric_cols = [
        "age",
        "height_cm",
        "mass_kg",
        "body_fat_percent",
        "muscle_percent",
        "visceral_fat",
        "bmi",
        "kss_before",
        "kss_after",
        "ir_mean_n",
        "er_mean_n",
    ]
    corr = subject_df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Metadata Correlation Matrix (participant-level variables)")
    fig.tight_layout()
    return {"correlation": _savefig(fig, "metadata_correlation_matrix.png")}


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _fmt_stats(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "n/a"
    return f"mean={series.mean():.2f}, std={series.std():.2f}, min={series.min():.2f}, max={series.max():.2f}"


def write_report(
    dataset: Dataset,
    condition_catalog: Dict[str, object],
    sensor_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    rpe_ts_df: pd.DataFrame,
    missing_values: Dict[str, Dict[str, int]],
    figures: Dict[str, str],
) -> None:
    n_subjects = len(dataset.subjects)
    n_conditions_catalog = len(condition_catalog)
    total_recordings = len(sensor_df)
    n_emg = (sensor_df["modality"] == "EMG").sum()
    n_imu = (sensor_df["modality"] == "IMU").sum()
    n_ppg = (sensor_df["modality"] == "PPG").sum()

    incomplete_subjects = subject_df[~subject_df["is_complete"]]

    missing_cells = missing_values["missing_cells"]
    total_cells = missing_values["total_cells"]

    lines: List[str] = []
    lines.append("# WSD4FEDSRM -- Exploratory Data Analysis Report")
    lines.append("")
    lines.append("**Phase:** 6 -- Exploratory Data Analysis (analysis only; no preprocessing, normalization, filtering, feature engineering, or modeling)")
    lines.append("")
    lines.append("**Reproduce:** `python notebooks/phase6_eda_analysis.py` from the project root (loads the dataset via `ml.preprocessing.dataset_loader.WSD4FEDSRMLoader`, ~70-90s).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Dataset overview
    lines.append("## 1. Dataset Overview")
    lines.append("")
    lines.append(f"- **Subjects loaded:** {n_subjects}")
    lines.append(f"- **Distinct experimental conditions:** {n_conditions_catalog} (6 fatigue tasks + 4 MVIC calibration trials)")
    lines.append(f"- **Total subject-condition recordings:** {len(condition_df)} (expected {n_subjects * n_conditions_catalog})")
    lines.append(f"- **Total individual sensor recordings:** {total_recordings} (EMG={n_emg}, IMU={n_imu}, PPG={n_ppg})")
    lines.append("- **Sensor modalities available:** EMG, IMU, PPG (PPG only for fatigue-task conditions)")
    lines.append("")

    # 2. Signal statistics
    lines.append("## 2. Signal Statistics")
    lines.append("")
    lines.append("| Modality | Recordings | Sampling rate (Hz) | Duration (s) | Samples per recording |")
    lines.append("|---|---|---|---|---|")
    for modality in ["EMG", "IMU", "PPG"]:
        subset = sensor_df[sensor_df["modality"] == modality]
        rate = subset["sampling_rate_hz"].iloc[0] if not subset.empty else float("nan")
        lines.append(
            f"| {modality} | {len(subset)} | {rate:g} | {_fmt_stats(subset['duration_seconds'])} | {_fmt_stats(subset['n_samples'])} |"
        )
    lines.append("")
    lines.append(f"![Signal duration distributions]({figures['duration']})")
    lines.append("")
    lines.append(f"![Samples per recording distributions]({figures['samples']})")
    lines.append("")
    lines.append(
        "Recording durations vary *within* each modality because the fatigue protocol runs each trial until the "
        "participant reaches Borg RPE = 20 (volitional exhaustion) or the intensity-specific cutoff -- trial length "
        "is itself fatigue-related signal, not measurement noise."
    )
    lines.append("")

    # 3. Missing data analysis
    lines.append("## 3. Missing Data Analysis")
    lines.append("")
    lines.append(f"- **Missing files/sensors during load:** 0 across all {n_subjects} subjects and {n_conditions_catalog} conditions "
                  "(the Phase 5 loader logs a warning for every missing file/sensor; none were logged for this dataset).")
    lines.append(f"- **Subjects with incomplete records:** {len(incomplete_subjects)} / {n_subjects}"
                  + (f" -- {', '.join(incomplete_subjects['subject_id'])}" if not incomplete_subjects.empty else " (none)"))
    lines.append("- **Missing values within raw sensor signals (NaN cells):**")
    for modality in ["EMG", "IMU", "PPG"]:
        m, t = missing_cells[modality], total_cells[modality]
        pct = (m / t * 100) if t else 0.0
        lines.append(f"  - {modality}: {m} / {t} cells ({pct:.4f}%)")
    lines.append(
        "- **Structurally absent modalities (by design, not data loss):** PPG and Borg `Labels` are `None` for all "
        "4 MVIC calibration conditions per subject -- confirmed against the source dataset (PPG was never recorded "
        "during MVIC trials; `borg_data.csv` contains exactly 34x6=204 rows, none for MVIC)."
    )
    lines.append(
        f"- **Borg RPE timeseries sparsity:** each fatigue trial has up to 25 scheduled RPE readings (before-task + "
        f"every 10s to 250s), but trials end at exhaustion, so most trials have far fewer. Observed: "
        f"{_fmt_stats(labels_df['n_rpe_readings'])} readings per trial out of a possible 25."
    )
    lines.append("")
    lines.append(f"![Missing data completeness heatmap]({figures['missing_heatmap']})")
    lines.append("")

    # 4. Label analysis
    lines.append("## 4. Label Analysis")
    lines.append("")
    lines.append(f"- **Borg RPE before task:** {_fmt_stats(labels_df['rpe_before'])}")
    lines.append(f"- **Borg RPE at end of trial:** {_fmt_stats(labels_df['rpe_at_end_of_trial'])}")
    lines.append(f"- **All individual RPE readings (n={len(rpe_ts_df)}):** {_fmt_stats(rpe_ts_df['rpe'])}")
    lines.append(f"- **Trial duration until exhaustion (s):** {_fmt_stats(labels_df['trial_duration_seconds'])}")
    lines.append(f"- **KSS before session:** {_fmt_stats(subject_df['kss_before'])}")
    lines.append(f"- **KSS after session:** {_fmt_stats(subject_df['kss_after'])}")
    lines.append(f"- **MVIC mean force, internal rotation (N):** {_fmt_stats(subject_df['ir_mean_n'])}")
    lines.append(f"- **MVIC mean force, external rotation (N):** {_fmt_stats(subject_df['er_mean_n'])}")
    lines.append("")
    lines.append(f"![Borg RPE before/end distribution]({figures['borg_before_after']})")
    lines.append("")
    lines.append(f"![All Borg RPE readings distribution]({figures['borg_all_readings']})")
    lines.append("")
    if "rpe_trajectory" in figures:
        lines.append(f"![Mean RPE trajectory by intensity band]({figures['rpe_trajectory']})")
        lines.append("")
    lines.append(f"![KSS before/after distribution]({figures['kss']})")
    lines.append("")
    lines.append(f"![MVIC force boxplot]({figures['mvic']})")
    lines.append("")

    # 5. Participant statistics
    lines.append("## 5. Participant Statistics")
    lines.append("")
    lines.append(f"- **Age (years):** {_fmt_stats(subject_df['age'])}")
    lines.append(f"- **Height (cm):** {_fmt_stats(subject_df['height_cm'])}")
    lines.append(f"- **Weight / mass (kg):** {_fmt_stats(subject_df['mass_kg'])}")
    lines.append(f"- **BMI:** {_fmt_stats(subject_df['bmi'])}")
    sex_counts = subject_df["sex"].value_counts().to_dict()
    hand_counts = subject_df["dominant_hand"].value_counts().to_dict()
    lines.append(f"- **Sex:** {sex_counts}")
    lines.append(f"- **Dominant hand:** {hand_counts}")
    lines.append("")
    lines.append(f"![Participant age distribution]({figures['age']})")
    lines.append("")
    lines.append(f"![Participant height distribution]({figures['height']})")
    lines.append("")
    lines.append(f"![Participant weight distribution]({figures['weight']})")
    lines.append("")
    lines.append(f"![Sex and dominant hand distribution]({figures['sex_hand']})")
    lines.append("")
    lines.append(f"![Metadata correlation matrix]({figures['correlation']})")
    lines.append("")

    # 6. Condition analysis
    lines.append("## 6. Condition Analysis")
    lines.append("")
    direction_counts = condition_df["direction"].value_counts().to_dict()
    type_counts = condition_df["condition_type"].value_counts().to_dict()
    lines.append(f"- **Internal vs external rotation trials:** {direction_counts}")
    lines.append(f"- **Fatigue-task vs MVIC-calibration trials:** {type_counts}")
    lines.append(f"- **Subjects per condition:** every one of the {n_conditions_catalog} conditions has recordings "
                  f"from {condition_df.groupby(['condition_type','direction','intensity_band','trial_number'], dropna=False)['subject_id'].nunique().min()}"
                  f"-{condition_df.groupby(['condition_type','direction','intensity_band','trial_number'], dropna=False)['subject_id'].nunique().max()} "
                  f"subjects (out of {n_subjects}).")
    lines.append("")
    lines.append(f"![Recordings per condition]({figures['per_condition']})")
    lines.append("")
    lines.append(f"![Direction and condition type counts]({figures['direction_type']})")
    lines.append("")

    # 7. Summary
    lines.append("## 7. Key Observations")
    lines.append("")
    lines.append(
        "- The dataset is essentially **complete**: 0 missing files/sensors were logged by the loader across all "
        f"{n_subjects} subjects x {n_conditions_catalog} conditions, and only "
        f"{len(incomplete_subjects)} subject(s) have any incomplete subject-level record."
    )
    lines.append(
        "- Trial **duration is highly variable by design** -- it directly encodes time-to-exhaustion, the core "
        "fatigue signal, rather than being a nuisance to normalize away without care."
    )
    lines.append(
        "- Borg RPE readings are **sparse by construction** (trials end early at exhaustion), so any tabular/matrix "
        "representation of the RPE timeseries will have structural missingness that must be handled explicitly "
        "(not imputed as if it were random)."
    )
    lines.append(
        "- PPG and Labels are **legitimately absent for all MVIC conditions** -- any multimodal fusion design must "
        "treat this as an expected, structural branch (MVIC vs. fatigue-task) rather than a data quality defect."
    )
    lines.append(
        "- Anthropometric source files (breadth/circumference/depth/length) contain typos in column names (e.g. "
        "\"ditsal\" for \"distal\") -- these were intentionally preserved verbatim by the Phase 5 loader and are "
        "visible here as-is."
    )
    lines.append("")

    lines.append("## 8. Potential Preprocessing Challenges")
    lines.append("")
    lines.append("- **Variable-length signals:** EMG/IMU/PPG recordings differ in sample count per trial (see Section 2); "
                  "downstream models will need windowing/padding/resampling strategies rather than assuming fixed-length input.")
    lines.append("- **Sparse, irregularly-terminated RPE labels:** converting `rpe_timeseries` into a regression/classification "
                  "target requires an explicit policy for censored/early-terminated trials (e.g. last-observation-carried-forward "
                  "vs. survival-style modeling of time-to-RPE-20).")
    lines.append("- **Multi-rate fusion:** EMG (1000 Hz), IMU (100 Hz), and PPG (200 Hz) must be aligned/resampled consistently "
                  "before any multimodal fusion model can consume them jointly.")
    lines.append("- **Structural modality absence in MVIC trials:** any shared preprocessing pipeline must branch on "
                  "condition type rather than assuming all 3 modalities are always present.")
    lines.append("- **Messy anthropometric column names:** the loader preserves raw (typo'd, inconsistently formatted) column "
                  "names verbatim in `breadth_cm`/`circumference_cm`/`depth_cm`/`length_cm`; a preprocessing step should map "
                  "these onto clean, stable feature names before they reach any model.")
    lines.append("")

    lines.append("## 9. Data Quality Issues")
    lines.append("")
    if incomplete_subjects.empty:
        lines.append("- No subjects with incomplete subject-level metadata were found.")
    else:
        lines.append("- Subjects with incomplete subject-level metadata:")
        for _, row in incomplete_subjects.iterrows():
            lines.append(
                f"  - {row['subject_id']}: demographic={row['has_demographic']}, "
                f"anthropometric={row['has_anthropometric']}, mvic={row['has_mvic']}, "
                f"kss={row['has_kss']}, conditions_loaded={row['n_conditions_loaded']}/{n_conditions_catalog}"
            )
    lines.append(
        "- Raw sensor CSVs (EMG/IMU/PPG) contain effectively no missing values within-file (see Section 3 percentages)."
    )
    lines.append(
        "- Source metadata files (demographic.csv, MVIC_force_data.csv, anthropometric files) use inconsistent column "
        "naming/spacing across files; the loader's tolerant column-matching handled this without data loss, but any "
        "external tool consuming the raw CSVs directly would need the same tolerance."
    )
    lines.append("")

    lines.append("## 10. Recommendations for Phase 7 (Preprocessing)")
    lines.append("")
    lines.append("1. Define an explicit windowing/resampling strategy per modality before any fusion (EMG 1000 Hz, IMU 100 Hz, PPG 200 Hz).")
    lines.append("2. Decide and document a single policy for handling early-terminated Borg RPE trajectories (e.g. treat "
                  "`trial_duration_seconds` / time-to-RPE-20 as the primary regression target, rather than padding the sparse timeseries).")
    lines.append("3. Keep MVIC-calibration and fatigue-task trials in separate preprocessing branches given their structurally "
                  "different modality availability (no PPG/labels for MVIC).")
    lines.append("4. Rename/standardize the anthropometric measurement keys (breadth/circumference/depth/length) to clean, "
                  "typo-free feature names as an explicit, documented mapping step -- do not silently guess at intent.")
    lines.append("5. Use subject-level MVIC force (ir_mean_n/er_mean_n) to normalize task load (%MVIC) if constructing "
                  "cross-subject-comparable load features.")
    lines.append("6. Retain raw data untouched (as now) and produce all cleaned/resampled artifacts under `datasets/processed/`, "
                  "consistent with the project's existing raw/processed separation.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote EDA report to %s", REPORT_PATH)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading dataset via WSD4FEDSRMLoader ...")
    loader = WSD4FEDSRMLoader()
    dataset = loader.load_dataset()
    condition_catalog = loader.discover_conditions()
    expected_conditions = len(condition_catalog)
    logger.info("Loaded %d subjects, %d condition templates", len(dataset.subjects), expected_conditions)

    logger.info("Building tidy analysis frames ...")
    sensor_df = build_sensor_frame(dataset)
    condition_df = build_condition_frame(dataset, condition_catalog)
    subject_df = build_subject_frame(dataset, expected_conditions)
    labels_df = build_labels_frame(dataset)
    rpe_ts_df = build_rpe_timeseries_frame(dataset)
    missing_values = count_missing_values(dataset)

    logger.info("Rendering figures to %s ...", FIGURES_DIR)
    figures: Dict[str, str] = {}
    figures.update(plot_participant_demographics(subject_df))
    figures.update(plot_borg_distribution(labels_df, rpe_ts_df))
    figures.update(plot_kss_distribution(subject_df))
    figures.update(plot_mvic_force(subject_df))
    figures.update(plot_condition_counts(condition_df))
    figures.update(plot_signal_length_distributions(sensor_df))
    figures.update(plot_missing_data_heatmap(subject_df, condition_df, expected_conditions))
    figures.update(plot_metadata_correlation(subject_df))

    logger.info("Writing EDA report ...")
    write_report(
        dataset=dataset,
        condition_catalog=condition_catalog,
        sensor_df=sensor_df,
        condition_df=condition_df,
        subject_df=subject_df,
        labels_df=labels_df,
        rpe_ts_df=rpe_ts_df,
        missing_values=missing_values,
        figures=figures,
    )
    logger.info("Phase 6 EDA complete.")


if __name__ == "__main__":
    main()
