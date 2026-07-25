"""Configuration dataclasses for the explainability framework.

Mirrors `ml.models.common.config` / `ml.models.proposed.config`'s
philosophy: every explanation run is fully described by a combination of
these config objects, so it is reproducible from its configuration alone.
Nothing here computes an explanation or touches data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapConfig:
    """Configuration for building and running a SHAP explainer.

    Attributes:
        background_samples: Maximum number of rows kept from the training
            split to serve as the SHAP "background"/reference distribution
            (used by `LinearExplainer` and the `KernelExplainer` fallback;
            `TreeExplainer` does not require one). Subsampled, not
            truncated, when the available data exceeds this.
        max_explain_samples: Maximum number of rows a global explanation is
            computed over -- SHAP's cost scales with this, and
            `KernelExplainer` in particular is expensive per row.
        positive_class_index: Which class index's contribution to explain
            for a multi-output (per-class) SHAP explanation, e.g. tree
            models on binary targets return one SHAP value per class; index
            1 is "the positive/fatigued class" for this project's
            `binary_fatigue` target.
        random_seed: Seed for background/explain-sample subsampling.
    """

    background_samples: int = 100
    max_explain_samples: int = 200
    positive_class_index: int = 1
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.background_samples < 1:
            raise ValueError("background_samples must be >= 1")
        if self.max_explain_samples < 1:
            raise ValueError("max_explain_samples must be >= 1")


@dataclass(frozen=True)
class FeatureRankingConfig:
    """Configuration for `feature_ranker.rank_features`.

    Attributes:
        top_k: How many top-ranked features a caller typically wants
            displayed (rendering-time concern) -- `rank_features` itself
            always ranks every feature; this only bounds what report/plot
            code slices off the top.
    """

    top_k: int = 15

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
