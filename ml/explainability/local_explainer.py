"""Local (single-prediction) explanation: "why did the model predict *this*
for *this* sample?"

Kept separate from `global_explainer.py` because the two answer genuinely
different questions with different outputs: a global explanation aggregates
many samples into one ranking, while a local explanation keeps one sample's
*signed* per-feature contributions (a feature can push a prediction toward
or away from "fatigued", which matters for a single case study but washes
out in a mean-absolute global ranking).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np
import shap

from ml.explainability.shap_explainer import ShapExplainerBundle, compute_shap_values


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's signed contribution to a single prediction.

    Attributes:
        feature_name: Column name.
        feature_value: That feature's actual (scaled) value for this sample.
        shap_value: Signed contribution to the positive class's log-odds/
            probability -- positive pushes the prediction toward the
            positive class, negative pushes away.
    """

    feature_name: str
    feature_value: float
    shap_value: float


@dataclass
class LocalExplanation:
    """Complete local explanation for one sample's prediction.

    Attributes:
        model_name: Registry name of the explained model.
        predicted_class: `argmax(predict_proba(x))`.
        predicted_probability: `predict_proba(x)[predicted_class]`.
        base_value: The explainer's expected-output baseline (the
            prediction if no feature's value were known) for the positive
            class -- `base_value + sum(shap_value for all features)`
            reconstructs the (pre-sigmoid, for tree/linear margin output)
            model output.
        top_contributions: Every feature, ranked by `|shap_value|`
            descending (not just the top-k -- callers slice, e.g.
            `top_contributions[:10]`, matching `feature_ranker.rank_features`'s
            convention).
        explanation_row: The single-row `shap.Explanation` this was derived
            from, for `shap_explainer.plot_waterfall`/`plot_force`.
    """

    model_name: str
    predicted_class: int
    predicted_probability: float
    base_value: float
    top_contributions: Tuple[FeatureContribution, ...]
    explanation_row: "shap.Explanation"


def explain_prediction(
    bundle: ShapExplainerBundle, model: Any, x_instance: np.ndarray, feature_names: Sequence[str]
) -> LocalExplanation:
    """Explain one model's prediction for one sample.

    Args:
        bundle: A `ShapExplainerBundle` from `shap_explainer.build_shap_explainer`.
        model: The same fitted estimator `bundle` was built for -- used
            here only for `predict_proba`, not for SHAP computation.
        x_instance: A single sample's feature vector, shape `(n_features,)`
            or `(1, n_features)`.
        feature_names: One name per feature.

    Returns:
        A `LocalExplanation`.
    """
    x_row = np.asarray(x_instance).reshape(1, -1)
    explanation = compute_shap_values(bundle, x_row, feature_names)[0]

    probabilities = model.predict_proba(x_row)[0]
    predicted_class = int(np.argmax(probabilities))

    order = np.argsort(np.abs(explanation.values))[::-1]
    contributions = tuple(
        FeatureContribution(feature_name=feature_names[i], feature_value=float(x_row[0, i]), shap_value=float(explanation.values[i]))
        for i in order
    )

    return LocalExplanation(
        model_name=bundle.model_name,
        predicted_class=predicted_class,
        predicted_probability=float(probabilities[predicted_class]),
        base_value=float(np.asarray(explanation.base_values).item()),
        top_contributions=contributions,
        explanation_row=explanation,
    )
