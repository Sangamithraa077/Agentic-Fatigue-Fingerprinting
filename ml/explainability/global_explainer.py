"""Global (dataset-level) explanation: "which features matter most to this
model, on average, across many predictions?"

Bundles a SHAP explanation over a (possibly subsampled) set of rows with
the resulting global feature ranking, so callers get both the raw
per-sample SHAP values (for summary/beeswarm/dependence plots) and the
aggregated ranking (for tables/reports) from one call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import shap

from ml.explainability.config import FeatureRankingConfig, ShapConfig
from ml.explainability.feature_ranker import RankedFeature, rank_features
from ml.explainability.shap_explainer import ShapExplainerBundle, compute_shap_values, subsample_rows


@dataclass
class GlobalExplanation:
    """Complete global explanation for one model over one dataset.

    Attributes:
        model_name: Registry name of the explained model.
        explanation: The normalized (positive-class, 2D) `shap.Explanation`
            covering `explained_data`, suitable for `shap_explainer.plot_summary`/
            `plot_beeswarm`/`plot_dependence`.
        feature_ranking: Every feature ranked by mean absolute SHAP value,
            descending (see `feature_ranker.rank_features`).
        feature_names: Column names, in `explanation`'s column order.
        n_samples_explained: Number of rows `explanation` covers (after any
            subsampling `explain_global` applied).
    """

    model_name: str
    explanation: "shap.Explanation"
    feature_ranking: Tuple[RankedFeature, ...]
    feature_names: Tuple[str, ...]
    n_samples_explained: int


def explain_global(
    bundle: ShapExplainerBundle,
    X: np.ndarray,
    feature_names: Sequence[str],
    shap_config: Optional[ShapConfig] = None,
    ranking_config: Optional[FeatureRankingConfig] = None,
) -> GlobalExplanation:
    """Compute a dataset-level SHAP explanation and feature ranking.

    Args:
        bundle: A `ShapExplainerBundle` from `shap_explainer.build_shap_explainer`.
        X: (n_samples, n_features) data to explain (typically a held-out
            test set). Subsampled to `shap_config.max_explain_samples` rows
            if larger -- global explanation cost scales with this, and the
            ranking is a mean over samples, so subsampling trades a small
            amount of ranking precision for bounded runtime.
        feature_names: One name per column of `X`.
        shap_config: SHAP configuration; defaults applied if omitted.
        ranking_config: Feature-ranking configuration; defaults applied if
            omitted (unused by the ranking computation itself, see
            `feature_ranker.rank_features`).

    Returns:
        A `GlobalExplanation`.
    """
    shap_config = shap_config or ShapConfig()
    X_explain = subsample_rows(np.asarray(X), shap_config.max_explain_samples, shap_config.random_seed)

    explanation = compute_shap_values(bundle, X_explain, feature_names)
    feature_ranking = rank_features(explanation.values, tuple(feature_names), ranking_config)

    return GlobalExplanation(
        model_name=bundle.model_name,
        explanation=explanation,
        feature_ranking=feature_ranking,
        feature_names=tuple(feature_names),
        n_samples_explained=len(X_explain),
    )
