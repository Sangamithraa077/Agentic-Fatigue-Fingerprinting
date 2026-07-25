"""SHAP explainer construction, evaluation, and plotting for the Phase 9
classical baseline models (Random Forest, XGBoost, Logistic Regression).

## Automatic explainer selection

SHAP provides model-family-specific explainers that are both faster and
exact (rather than approximate) when the model's internal structure is
known:

- **Tree-based models** (`random_forest`, `xgboost`) use `shap.TreeExplainer`,
  which computes exact Shapley values in polynomial time by exploiting the
  tree structure directly -- no background dataset needed.
- **Linear models** (`logistic_regression`) use `shap.LinearExplainer`,
  exact for linear models given a background distribution to integrate
  over.
- **Anything else** falls back to the model-agnostic `shap.KernelExplainer`
  (sampling-based, works for any `predict_proba`-exposing model, but is
  the slowest of the three) -- included so this module keeps working if a
  future model type is added, not because this phase requires it (only
  the three model families above are in scope).

`select_explainer_type`/`build_shap_explainer` implement this dispatch so
callers never need to know which explainer class a given model needs.

## Normalizing SHAP's per-model output shape

Different explainer/model combinations return differently-shaped
`shap.Explanation.values`: tree ensembles on a binary target return one set
of contributions *per class* (`(n_samples, n_features, n_classes)`), while
linear/kernel explainers on the same target return a single set of
contributions already representing the positive class's log-odds
(`(n_samples, n_features)`). `compute_shap_values` normalizes both cases
down to one 2D `(n_samples, n_features)` `Explanation` representing the
positive class's contribution, so every downstream consumer (feature
ranking, global/local explanation, plotting) can treat all three model
families identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import shap

from ml.explainability.config import ShapConfig

TREE_MODEL_NAMES = frozenset({"random_forest", "xgboost"})
LINEAR_MODEL_NAMES = frozenset({"logistic_regression"})


@dataclass
class ShapExplainerBundle:
    """A fitted SHAP explainer plus the metadata needed to interpret its output.

    Attributes:
        explainer: The underlying `shap.Explainer` (a `TreeExplainer`,
            `LinearExplainer`, or `KernelExplainer` instance).
        explainer_type: `"tree"`, `"linear"`, or `"kernel"` -- which branch
            of `build_shap_explainer` produced `explainer`.
        model_name: Registry name of the explained model (e.g.
            `"random_forest"`).
        positive_class_index: Class index whose contribution
            `compute_shap_values` extracts from multi-output explanations.
    """

    explainer: Any
    explainer_type: str
    model_name: str
    positive_class_index: int


def select_explainer_type(model_name: str) -> str:
    """Return `"tree"`, `"linear"`, or `"kernel"` for a registry model name."""
    if model_name in TREE_MODEL_NAMES:
        return "tree"
    if model_name in LINEAR_MODEL_NAMES:
        return "linear"
    return "kernel"


def subsample_rows(X: np.ndarray, max_rows: int, random_seed: int) -> np.ndarray:
    if len(X) <= max_rows:
        return X
    rng = np.random.default_rng(random_seed)
    idx = np.sort(rng.choice(len(X), size=max_rows, replace=False))
    return X[idx]


def build_shap_explainer(
    model: Any, model_name: str, X_background: np.ndarray, config: Optional[ShapConfig] = None
) -> ShapExplainerBundle:
    """Build the appropriate SHAP explainer for `model`, per `select_explainer_type`.

    Args:
        model: A fitted sklearn-compatible estimator (`RandomForestClassifier`,
            `XGBClassifier`, or `LogisticRegression`, or -- via the
            `"kernel"` fallback -- any estimator exposing `predict_proba`).
        model_name: Registry name (see `TREE_MODEL_NAMES`/`LINEAR_MODEL_NAMES`)
            used to select the explainer type.
        X_background: Reference distribution for `"linear"`/`"kernel"`
            explainers (ignored for `"tree"`, which needs none). Subsampled
            to `config.background_samples` rows if larger.
        config: SHAP configuration; defaults applied if omitted.

    Returns:
        A `ShapExplainerBundle` ready for `compute_shap_values`.
    """
    config = config or ShapConfig()
    explainer_type = select_explainer_type(model_name)
    background = subsample_rows(np.asarray(X_background), config.background_samples, config.random_seed)

    if explainer_type == "tree":
        explainer: Any = shap.TreeExplainer(model)
    elif explainer_type == "linear":
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return ShapExplainerBundle(
        explainer=explainer, explainer_type=explainer_type, model_name=model_name,
        positive_class_index=config.positive_class_index,
    )


def _select_positive_class_output(explanation: "shap.Explanation", positive_class_index: int) -> "shap.Explanation":
    """Collapse a per-class `(n, f, n_classes)` explanation to one class's
    `(n, f)` explanation; a no-op for explainers that already return a
    single output (linear/kernel explainers on a binary target)."""
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        class_index = min(positive_class_index, values.shape[-1] - 1)
        return explanation[:, :, class_index]
    return explanation


def compute_shap_values(
    bundle: ShapExplainerBundle, X: np.ndarray, feature_names: Sequence[str]
) -> "shap.Explanation":
    """Run `bundle.explainer` over `X`, returning a normalized 2D
    `(n_samples, n_features)` `shap.Explanation` for the positive class.

    Callers control how many rows this is run over (e.g. `global_explainer`
    subsamples per `ShapConfig.max_explain_samples`; `local_explainer`
    passes a single row) -- this function does no subsampling itself.
    """
    X = np.asarray(X)
    if bundle.explainer_type == "kernel":
        raw = bundle.explainer(X, nsamples="auto", silent=True)
    else:
        raw = bundle.explainer(X)

    explanation = _select_positive_class_output(raw, bundle.positive_class_index)
    explanation.feature_names = list(feature_names)
    return explanation


# ---------------------------------------------------------------------------
# Plotting (SHAP summary / beeswarm / waterfall / force / dependence)
# ---------------------------------------------------------------------------


def _save_current_figure(save_path: Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return save_path


def plot_summary(explanation: "shap.Explanation", save_path: Path, max_display: int = 20) -> Path:
    """Bar-style SHAP summary plot (mean absolute contribution per feature)."""
    shap.summary_plot(
        explanation.values, explanation.data, feature_names=list(explanation.feature_names),
        plot_type="bar", max_display=max_display, show=False,
    )
    return _save_current_figure(save_path)


def plot_beeswarm(explanation: "shap.Explanation", save_path: Path, max_display: int = 20) -> Path:
    """Beeswarm plot: per-sample SHAP value distribution per feature."""
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    return _save_current_figure(save_path)


def plot_waterfall(explanation_row: "shap.Explanation", save_path: Path, max_display: int = 15) -> Path:
    """Waterfall plot for one sample's prediction (`explanation_row` = a
    single-row slice of a `shap.Explanation`, e.g. `explanation[i]`)."""
    shap.plots.waterfall(explanation_row, max_display=max_display, show=False)
    return _save_current_figure(save_path)


def plot_force(explanation_row: "shap.Explanation", save_path: Path) -> Path:
    """Force plot for one sample's prediction, rendered as a static image
    (`matplotlib=True` -- avoids SHAP's default interactive JS/HTML output,
    which cannot be saved under `reports/figures/`)."""
    shap.plots.force(
        explanation_row.base_values, explanation_row.values, explanation_row.data,
        feature_names=list(explanation_row.feature_names), matplotlib=True, show=False,
    )
    return _save_current_figure(save_path)


def plot_dependence(feature_name: str, explanation: "shap.Explanation", save_path: Path) -> Path:
    """Dependence plot: one feature's value vs. its SHAP contribution,
    colored by SHAP's automatically-chosen most-interacting feature."""
    shap.dependence_plot(
        feature_name, explanation.values, explanation.data, feature_names=list(explanation.feature_names), show=False,
    )
    return _save_current_figure(save_path)
