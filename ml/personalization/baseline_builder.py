"""Per-subject baseline statistics, built directly from Phase 8.5's
`FeatureDataset`s -- no raw signal reprocessing, no re-derivation of
features (per this phase's "do not duplicate preprocessing or feature
engineering" requirement).

Two things are computed here:

1. **Modality baselines** (`ModalityBaseline`): each subject's own
   mean/std per feature, across every window that subject contributed to
   a modality's `FeatureDataset`. This is literally a subject-only slice
   of the same feature matrix Phase 9-11 already build -- no new
   computation beyond `groupby(subject_id).agg(mean, std)`. These are the
   "Baseline EMG/IMU/PPG statistics" required by `SubjectProfile`, and are
   also what `personalization_model.predict_personalized` compares a new
   window against.
2. **Fatigue history** (`FatigueHistory`): summary statistics of a
   subject's own `rpe_nearest` readings (Phase 8.5's nearest-in-time Borg
   RPE lookup, already computed -- not re-derived here) restricted to
   fatigue-task windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from ml.feature_engineering.feature_dataset import FeatureDataset
from ml.models.common.dataset_builder import impute_and_clean
from ml.personalization.config import SubjectProfileConfig


@dataclass(frozen=True)
class ModalityBaseline:
    """One subject's baseline statistics for one modality.

    Attributes:
        modality: `"EMG"`, `"IMU"`, or `"PPG"`.
        feature_means: This subject's mean value per feature, across every
            window of theirs in the modality's `FeatureDataset`.
        feature_stds: This subject's per-feature standard deviation (0.0
            if fewer than 2 windows, rather than NaN).
        n_windows: How many windows this baseline was computed from.
    """

    modality: str
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]
    n_windows: int


@dataclass(frozen=True)
class FatigueHistory:
    """One subject's summary of their own fatigue-task Borg RPE history.

    Attributes:
        n_trials: Distinct fatigue-task conditions this subject performed.
        n_windows: Total feature windows across those trials.
        mean_rpe, max_rpe, min_rpe: Statistics of `rpe_nearest` across this
            subject's fatigue-task windows, or `None` if no RPE readings
            are available (e.g. the subject has zero fatigue-task windows
            in the given `FeatureDataset`).
    """

    n_trials: int
    n_windows: int
    mean_rpe: Optional[float]
    max_rpe: Optional[float]
    min_rpe: Optional[float]


def build_modality_baseline(feature_dataset: FeatureDataset, subject_id: str) -> ModalityBaseline:
    """Compute `subject_id`'s baseline mean/std per feature within one modality.

    Applies `ml.models.common.dataset_builder.impute_and_clean` to this
    subject's raw window rows before computing statistics -- reused
    unmodified, not duplicated, because Phase 8.5 already confirmed some raw
    EMG files contain individually corrupted (physiologically impossible,
    e.g. ~1e28-1e34) values that are finite (so a plain NaN/inf check misses
    them) but wildly distort a plain `mean`/`std` if left in. Without this,
    a single corrupted window can make one subject's "baseline" for a
    feature meaningless by many orders of magnitude -- caught during this
    phase's own report generation (`baseline_distribution.png` showed one
    subject's `variance` baseline at ~1e62 before this fix).
    """
    if feature_dataset.matrix.empty:
        return ModalityBaseline(modality=feature_dataset.modality, feature_means={}, feature_stds={}, n_windows=0)

    mask = (feature_dataset.metadata["subject_id"] == subject_id).to_numpy()
    subject_rows = feature_dataset.matrix.loc[mask]
    if subject_rows.empty:
        return ModalityBaseline(modality=feature_dataset.modality, feature_means={}, feature_stds={}, n_windows=0)

    cleaned, _ = impute_and_clean(subject_rows.to_numpy(dtype=float))
    cleaned_rows = pd.DataFrame(cleaned, columns=subject_rows.columns)

    means = cleaned_rows.mean(axis=0)
    stds = cleaned_rows.std(axis=0).fillna(0.0)
    return ModalityBaseline(
        modality=feature_dataset.modality,
        feature_means={str(name): float(value) for name, value in means.items()},
        feature_stds={str(name): float(value) for name, value in stds.items()},
        n_windows=int(mask.sum()),
    )


def build_fatigue_history(
    feature_dataset: FeatureDataset, subject_id: str, config: Optional[SubjectProfileConfig] = None
) -> FatigueHistory:
    """Summarize `subject_id`'s fatigue-task RPE history from `feature_dataset.metadata`."""
    config = config or SubjectProfileConfig()
    if feature_dataset.metadata.empty:
        return FatigueHistory(n_trials=0, n_windows=0, mean_rpe=None, max_rpe=None, min_rpe=None)

    metadata = feature_dataset.metadata
    mask = (metadata["subject_id"] == subject_id) & (metadata["condition_type"] == config.fatigue_condition_type)
    subject_metadata = metadata.loc[mask]
    if subject_metadata.empty:
        return FatigueHistory(n_trials=0, n_windows=0, mean_rpe=None, max_rpe=None, min_rpe=None)

    rpe = subject_metadata["rpe_nearest"].dropna()
    return FatigueHistory(
        n_trials=int(subject_metadata["condition_id"].nunique()),
        n_windows=int(len(subject_metadata)),
        mean_rpe=float(rpe.mean()) if len(rpe) else None,
        max_rpe=float(rpe.max()) if len(rpe) else None,
        min_rpe=float(rpe.min()) if len(rpe) else None,
    )
