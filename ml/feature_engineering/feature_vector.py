"""Feature-result container types.

`FeatureVector` is the atomic unit returned by every modality-specific
extractor: an ordered set of named feature values for one window of one
signal. Names always travel with values, so a feature can always be traced
back to what it means (requirement: "preserve feature names for
interpretability").

The remaining dataclasses aggregate `FeatureVector`s up through the same
Condition -> Subject hierarchy used by `ml.data_models` and
`ml.preprocessing.preprocessing_results`, so a `SubjectFeatures` mirrors the
shape of a `ProcessedSubject` one level further downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ml.data_models.enums import ConditionType, Direction, EMGMuscle, IMULocation, IntensityBand


@dataclass(frozen=True)
class FeatureVector:
    """Named feature values for one window of one signal.

    Attributes:
        names: Feature names, e.g. ("mav", "rms", "variance", ...).
        values: Feature values, in the same order as `names`.
    """

    names: Tuple[str, ...]
    values: Tuple[float, ...]

    def as_dict(self) -> Dict[str, float]:
        """Return {feature_name: value} for this vector."""
        return dict(zip(self.names, self.values))

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(
                f"names and values must be the same length, got {len(self.names)} and {len(self.values)}"
            )


@dataclass(frozen=True)
class WindowedFeatures:
    """One window's features plus its position within the source signal."""

    window_index: int
    window_start_sample: int
    window_end_sample: int
    features: FeatureVector


@dataclass(frozen=True)
class EMGChannelFeatures:
    """All windows' features for one EMG muscle channel."""

    muscle: EMGMuscle
    windows: List[WindowedFeatures] = field(default_factory=list)


@dataclass(frozen=True)
class IMULocationFeatures:
    """All windows' features for one IMU location.

    Each `WindowedFeatures.features` here already combines per-axis
    (accelerometer/gyroscope/magnetometer x/y/z) and resultant-vector
    features into one flat, prefixed `FeatureVector` (see
    `imu_features.IMUFeatureExtractor`).
    """

    location: IMULocation
    windows: List[WindowedFeatures] = field(default_factory=list)


@dataclass(frozen=True)
class PPGFeatures:
    """All windows' features for the PPG signal of one condition."""

    windows: List[WindowedFeatures] = field(default_factory=list)


@dataclass(frozen=True)
class ConditionFeatures:
    """Feature-engineering result for one condition trial."""

    condition_id: str
    subject_id: str
    condition_type: ConditionType
    direction: Direction
    intensity_band: Optional[IntensityBand]
    trial_number: Optional[int]
    emg: Dict[EMGMuscle, EMGChannelFeatures] = field(default_factory=dict)
    imu: Dict[IMULocation, IMULocationFeatures] = field(default_factory=dict)
    ppg: Optional[PPGFeatures] = None


@dataclass(frozen=True)
class SubjectFeatures:
    """Feature-engineering result for every condition of one subject."""

    subject_id: str
    conditions: Dict[str, ConditionFeatures] = field(default_factory=dict)


def condition_features_to_dataframe(condition_features: ConditionFeatures) -> pd.DataFrame:
    """Flatten a `ConditionFeatures` into one tidy DataFrame.

    One row per (modality, channel/location, window); columns are the
    fully qualified feature names (e.g. "anterior_deltoid__mav") plus
    `modality`, `channel`, and `window_index` identifier columns. Provided
    as a convenience for downstream analysis/modeling phases -- purely a
    reorganization of already-computed features, not new computation.
    """
    rows: List[dict] = []

    for muscle, channel_features in condition_features.emg.items():
        for windowed in channel_features.windows:
            row = {"modality": "EMG", "channel": muscle.value, "window_index": windowed.window_index}
            row.update(windowed.features.as_dict())
            rows.append(row)

    for location, location_features in condition_features.imu.items():
        for windowed in location_features.windows:
            row = {"modality": "IMU", "channel": location.value, "window_index": windowed.window_index}
            row.update(windowed.features.as_dict())
            rows.append(row)

    if condition_features.ppg is not None:
        for windowed in condition_features.ppg.windows:
            row = {"modality": "PPG", "channel": "ppg", "window_index": windowed.window_index}
            row.update(windowed.features.as_dict())
            rows.append(row)

    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FeatureExtractionSummary:
    """Aggregate summary of everything a `FeaturePipeline` instance has
    processed since it was created.

    Attributes:
        n_conditions_processed: Number of `ProcessedCondition` objects
            processed.
        n_emg_windows: Total EMG windows feature-extracted (across all
            muscles and conditions).
        n_imu_windows: Total IMU windows feature-extracted (across all
            locations and conditions).
        n_ppg_windows: Total PPG windows feature-extracted.
        feature_names_by_category: Every distinct feature name produced,
            grouped by category (e.g. "emg_time_domain",
            "emg_frequency_domain", "imu_per_axis", "imu_resultant", "ppg").
        skipped_features: Features considered but intentionally not
            implemented, mapped to a short justification (e.g. frequency-
            domain HRV metrics).
        warnings: Every individual warning/issue encountered, in order.
    """

    n_conditions_processed: int
    n_emg_windows: int
    n_imu_windows: int
    n_ppg_windows: int
    feature_names_by_category: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    skipped_features: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_total_feature_names(self) -> int:
        return sum(len(names) for names in self.feature_names_by_category.values())

    def as_text(self) -> str:
        """Render this summary as a short human-readable report."""
        lines = [
            "Feature Extraction Summary",
            f"Conditions processed: {self.n_conditions_processed}",
            f"EMG windows: {self.n_emg_windows}",
            f"IMU windows: {self.n_imu_windows}",
            f"PPG windows: {self.n_ppg_windows}",
            f"Total distinct feature names: {self.n_total_feature_names}",
            "",
            "Feature categories:",
        ]
        for category, names in self.feature_names_by_category.items():
            lines.append(f"  - {category} ({len(names)}): {', '.join(names)}")

        lines.append("")
        lines.append("Skipped features:")
        if self.skipped_features:
            for name, justification in self.skipped_features.items():
                lines.append(f"  - {name}: {justification}")
        else:
            lines.append("  none")

        lines.append("")
        lines.append(f"Warnings ({len(self.warnings)}):")
        if self.warnings:
            lines.extend(f"  - {warning}" for warning in self.warnings)
        else:
            lines.append("  none")

        return "\n".join(lines)
