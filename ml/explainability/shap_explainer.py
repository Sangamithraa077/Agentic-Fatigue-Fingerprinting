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

# Patch SHAP XGBTreeModelLoader for XGBoost 3.x base_score compatibility
try:
    import io
    import scipy.special
    import xgboost as xgb
    import shap.explainers._tree as _tree_mod
    from shap.explainers._tree import _check_xgboost_version, decode_ubjson_buffer

    _orig_loader_init = _tree_mod.XGBTreeModelLoader.__init__

    def _patched_xgb_loader_init(self, xgb_model) -> None:
        try:
            _orig_loader_init(self, xgb_model)
        except ValueError as err:
            if "could not convert string to float" not in str(err):
                raise
            _check_xgboost_version(xgb.__version__)
            model: xgb.Booster = xgb_model
            raw = xgb_model.save_raw(raw_format="ubj")
            with io.BytesIO(raw) as fd:
                jmodel = decode_ubjson_buffer(fd)

            learner = jmodel["learner"]
            learner_model_param = learner["learner_model_param"]
            objective = learner["objective"]
            booster = learner["gradient_booster"]

            n_classes = max(int(learner_model_param["num_class"]), 1)
            n_targets = max(int(learner_model_param["num_target"]), 1)
            n_targets = max(n_targets, n_classes)

            if "gbtree" in booster and "model" not in booster:
                booster = booster["gbtree"]
            if booster["model"].get("iteration_indptr", None) is not None:
                iteration_indptr = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
                diff = np.diff(iteration_indptr)
            else:
                n_parallel_trees = int(booster["model"]["gbtree_model_param"]["num_parallel_tree"])
                diff = np.repeat(n_targets * n_parallel_trees, model.num_boosted_rounds())

            self.n_trees_per_iter = int(diff[0])
            self.n_targets = n_targets

            raw_bs = learner_model_param["base_score"]
            if isinstance(raw_bs, str):
                raw_bs = raw_bs.strip("[]")
            self.base_score = float(raw_bs)

            self.name_obj = objective["name"]
            self.name_gbm = booster["name"]
            base_score = float(raw_bs)
            if self.name_obj in ("binary:logistic", "reg:logistic"):
                self.base_score = scipy.special.logit(base_score)
            elif self.name_obj in (
                "reg:gamma",
                "reg:tweedie",
                "count:poisson",
                "survival:cox",
                "survival:aft",
            ):
                self.base_score = np.log(self.base_score)
            else:
                self.base_score = base_score

            self.num_feature = int(learner_model_param["num_feature"])
            self.num_class = int(learner_model_param["num_class"])

            trees = booster["model"]["trees"]
            self.num_trees = len(trees)

            self.node_parents = []
            self.node_cleft = []
            self.node_cright = []
            self.node_sindex = []
            self.children_default: list[np.ndarray] = []
            self.sum_hess = []
            self.values = []
            self.thresholds = []
            self.threshold_types = []
            self.features = []
            self.split_types = []
            self.categories = []

            feature_types = model.feature_types
            if feature_types is not None:
                cat_indices = np.where(np.asarray(feature_types) == "c")[0]
                self.cat_feature_indices = None if len(cat_indices) == 0 else cat_indices
            else:
                self.cat_feature_indices = None

            def _to_integers(data: list[int]) -> np.ndarray:
                return np.asanyarray(data, dtype=np.uint8)

            for i in range(self.num_trees):
                tree = trees[i]
                parents = np.asarray(tree["parents"])
                self.node_parents.append(parents)
                self.node_cleft.append(np.asarray(tree["left_children"], dtype=np.int32))
                self.node_cright.append(np.asarray(tree["right_children"], dtype=np.int32))
                self.node_sindex.append(np.asarray(tree["split_indices"], dtype=np.uint32))

                base_weight = np.asarray(tree["base_weights"], dtype=np.float32)
                default_left = _to_integers(tree["default_left"])
                default_child = np.where(default_left == 1, self.node_cleft[-1], self.node_cright[-1]).astype(np.int64)
                self.children_default.append(default_child)
                self.sum_hess.append(np.asarray(tree["sum_hessian"], dtype=np.float64))

                is_leaf = self.node_cleft[-1] == -1
                split_cond = np.asarray(tree["split_conditions"], dtype=np.float32)
                leaf_weight = np.where(is_leaf, split_cond, 0.0)
                thresholds = np.where(is_leaf, 0.0, split_cond)
                thresholds = np.where(is_leaf, 0.0, np.nextafter(thresholds, -np.float32(np.inf)))
                threshold_types = np.zeros_like(thresholds, dtype=np.int32)

                self.values.append(leaf_weight.reshape(leaf_weight.size, 1))
                self.thresholds.append(thresholds)
                self.threshold_types.append(threshold_types)
                split_idx = np.asarray(tree["split_indices"], dtype=np.int64)
                self.features.append(split_idx)
                split_types = _to_integers(tree["split_type"])
                self.split_types.append(split_types)
                cat_segments = tree["categories_segments"]
                cat_sizes = tree["categories_sizes"]
                cat_nodes = tree["categories_nodes"]
                cats = tree["categories"]
                tree_categories = self.parse_categories(cat_nodes, cat_segments, cat_sizes, cats, self.node_cleft[-1])
                self.categories.append(tree_categories)

    _tree_mod.XGBTreeModelLoader.__init__ = _patched_xgb_loader_init
except Exception:
    pass

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
