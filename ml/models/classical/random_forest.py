"""Random Forest baseline: a bagged ensemble of decision trees.

Purpose in this benchmark: a strong, low-tuning-effort non-linear baseline
that is naturally robust to unscaled/outlier-contaminated features (trees
split on rank order, not magnitude) -- relevant given Phase 8.5's
documented extreme-outlier findings in some raw EMG files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class RandomForestConfig:
    """Hyperparameters for `build_random_forest`.

    Attributes:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (`None` = expand until pure/min leaf).
        min_samples_leaf: Minimum samples required at a leaf node.
        class_weight: `"balanced"` reweights classes inversely to frequency.
        random_state: Seed for bootstrap sampling and feature selection.
        n_jobs: Parallel jobs (`-1` = use all available cores).
    """

    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
    class_weight: Optional[str] = "balanced"
    random_state: int = 42
    n_jobs: int = -1


def build_random_forest(config: Optional[RandomForestConfig] = None) -> RandomForestClassifier:
    """Construct a configured `RandomForestClassifier` estimator."""
    config = config or RandomForestConfig()
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
