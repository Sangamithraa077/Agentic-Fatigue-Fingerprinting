"""Central configuration dataclasses for the Phase 9 baseline modeling
framework.

Every experiment (classical or deep learning) is fully described by a
combination of these config objects, so a run is reproducible from its
configuration alone. Nothing in this module trains a model or touches
data -- it is pure configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    """Selects which Phase 8.5 feature matrix and prediction target to use.

    Attributes:
        modality: "EMG", "IMU", or "PPG" (matches `FeatureDataset.modality`
            from `ml.feature_engineering.feature_dataset`).
        feature_subset: Which feature-selection view to use:
            - "all": every feature, unfiltered.
            - "variance_filtered": drops constant/near-zero-variance
              features (Phase 8.5 `recommend_variance_threshold`).
            - "correlation_filtered": additionally drops exact-duplicate
              features (Phase 8.5 `recommend_correlation_filter`,
              "remove"-level recommendations only -- "review"-level
              correlated pairs are kept, since choosing which of a
              correlated pair to drop is a judgment call, not automated).
            - "mutual_info_ranked": keeps only the top
              `mi_top_fraction` of features by mutual information against
              `target_type`.
        target_type: "binary_fatigue", "fatigue_level", or "borg_rpe_class"
            -- see `dataset_builder.py` module docstring for exactly how
            each is derived from `rpe_nearest`.
        variance_threshold: Passed to the variance-filtered subset.
        correlation_threshold: Passed to the correlation-filtered subset.
        mi_top_fraction: Fraction of features (by MI rank, descending) kept
            for the "mutual_info_ranked" subset.
        binary_fatigue_threshold: Borg RPE cutoff for `binary_fatigue`
            (>= threshold -> "fatigued").
        fatigue_level_bins: Borg RPE bin edges (inclusive lower bounds) for
            `fatigue_level`'s 3 classes.
    """

    modality: str
    feature_subset: str = "all"
    target_type: str = "binary_fatigue"
    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.95
    mi_top_fraction: float = 0.5
    binary_fatigue_threshold: float = 13.0
    fatigue_level_bins: Tuple[float, float, float, float] = (6.0, 11.0, 16.0, 21.0)

    def __post_init__(self) -> None:
        if self.modality not in ("EMG", "IMU", "PPG"):
            raise ValueError(f"Unknown modality: {self.modality!r}")
        if self.feature_subset not in ("all", "variance_filtered", "correlation_filtered", "mutual_info_ranked"):
            raise ValueError(f"Unknown feature_subset: {self.feature_subset!r}")
        if self.target_type not in ("binary_fatigue", "fatigue_level", "borg_rpe_class"):
            raise ValueError(f"Unknown target_type: {self.target_type!r}")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for both validation strategies.

    Splits are **group-aware by `group_column`** (default: `subject_id`) to
    avoid leakage from Phase 8's overlapping (50%) sliding windows: two
    overlapping windows share most of their samples, so a naive random
    row-level split could put near-duplicate windows in both train and
    test. Grouping by subject also means the exact same splitting utilities
    can be reused unmodified for future leave-one-subject-out
    personalization (Phase 11) -- just enumerate one held-out subject per
    fold instead of a random group split.

    Attributes:
        n_splits: Number of folds for classical models' stratified group
            k-fold cross-validation.
        train_ratio, val_ratio, test_ratio: Deep learning train/val/test
            split proportions (must sum to 1.0); also used to carve out the
            single shared held-out test set used for the cross-category
            comparison table.
        group_column: Metadata column defining non-splittable groups.
        random_seed: Seed for every random split performed.
    """

    n_splits: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    group_column: str = "subject_id"
    random_seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")


@dataclass(frozen=True)
class TrainingConfig:
    """Deep learning training loop configuration.

    Attributes:
        batch_size: Mini-batch size.
        max_epochs: Upper bound on training epochs.
        learning_rate: Initial Adam learning rate.
        weight_decay: Adam L2 weight decay.
        early_stopping_patience: Epochs without validation-loss improvement
            before stopping.
        early_stopping_min_delta: Minimum improvement to reset patience.
        lr_scheduler_patience: Epochs without improvement before the LR
            scheduler reduces the learning rate.
        lr_scheduler_factor: Multiplicative LR reduction factor.
        random_seed: Seed for `torch`/`numpy`/`random` (reproducibility).
        device: "auto" (use CUDA if available, else CPU), "cpu", or "cuda".
        checkpoint_dir: Where to save the best-epoch model checkpoint.
            `None` disables checkpointing.
    """

    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    random_seed: int = 42
    device: str = "auto"
    checkpoint_dir: Optional[Path] = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Bundles everything needed to fully reproduce one experiment."""

    dataset: DatasetConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
