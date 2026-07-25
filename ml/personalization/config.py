"""Configuration dataclasses for the personalization framework.

Mirrors `ml.models.common.config` / `ml.models.proposed.config` /
`ml.explainability.config`'s philosophy: every profile/LOSO/personalization/
similarity computation is fully described by a combination of these config
objects, so it is reproducible from its configuration alone. Nothing here
builds a profile, trains a model, or touches data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ml.models.common.config import DatasetConfig


@dataclass(frozen=True)
class SubjectProfileConfig:
    """Configuration for `subject_profiles.build_subject_profile`.

    Attributes:
        fatigue_condition_type: The `FeatureDataset.metadata["condition_type"]`
            value identifying fatigue-task (as opposed to MVIC-calibration)
            windows, used to build `baseline_builder.FatigueHistory`.
    """

    fatigue_condition_type: str = "fatigue_task"


@dataclass(frozen=True)
class LosoConfig:
    """Configuration for `loso.evaluate_loso`.

    Attributes:
        model_name: Which registered classical model to train per fold
            (see `loso.MODEL_BUILDERS`).
        dataset_config: Selects which Phase 8.5 feature matrix/target to
            build the LOSO modeling dataset from.
        random_seed: Seed passed to the per-fold model builder, where
            applicable.
    """

    model_name: str = "random_forest"
    dataset_config: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(modality="EMG", feature_subset="all", target_type="binary_fatigue")
    )
    random_seed: int = 42


@dataclass(frozen=True)
class PersonalizationConfig:
    """Configuration for `personalization_model.predict_personalized`.

    Attributes:
        alpha: Blending weight in `[0, 1]` between the population-level
            (`"global"`) model's probability and the subject-baseline
            deviation signal -- `alpha=1.0` reduces to the pure global
            prediction, `alpha=0.0` to pure baseline-deviation scoring.
        deviation_scale: Logistic steepness applied to the mean absolute
            baseline z-score when converting it to a `[0, 1]` deviation
            probability.
        deviation_midpoint: The mean absolute z-score at which the
            deviation probability crosses 0.5 (higher = more deviation
            required before the personalization layer treats a window as
            fatigue-indicative).
    """

    alpha: float = 0.5
    deviation_scale: float = 1.0
    deviation_midpoint: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.deviation_scale <= 0.0:
            raise ValueError(f"deviation_scale must be > 0, got {self.deviation_scale}")


@dataclass(frozen=True)
class SimilarityConfig:
    """Configuration for `similarity.find_similar_subjects`.

    Attributes:
        metric: `"cosine"` or `"euclidean"`.
        top_k: Maximum number of similar subjects to return.
    """

    metric: str = "cosine"
    top_k: int = 5

    def __post_init__(self) -> None:
        if self.metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unknown similarity metric: {self.metric!r}")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
