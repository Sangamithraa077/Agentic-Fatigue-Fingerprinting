# Explainable AI Framework -- Phase 11

**Phase:** 11 -- Explainable AI (SHAP + attention analysis over already-trained Phase 9/10 models; no personalization or agentic reasoning -- later phases)

This document explains the design of `ml/explainability/` and why it's structured this way. Full API-level documentation lives in each module's docstrings; this is the human-readable summary.

## Scope

A modular explainability framework that explains predictions from models **already trained** in Phases 9 and 10 -- it trains nothing itself. Two complementary families are covered: SHAP (exact, game-theoretic feature attribution) for the classical models, and cross-modal attention analysis (raw internal-computation inspection) for the proposed multimodal model. Later phases add subject personalization and agentic reasoning on top of this; this phase does not implement either.

## Module Layout

| File | Responsibility |
|---|---|
| `config.py` | `ShapConfig`, `FeatureRankingConfig`. |
| `shap_explainer.py` | Automatic per-model-family explainer selection (`select_explainer_type`/`build_shap_explainer`), SHAP value computation with output-shape normalization (`compute_shap_values`), and every SHAP plot (summary, beeswarm, waterfall, force, dependence). |
| `feature_ranker.py` | `rank_features` -- generic mean-absolute-SHAP-value ranking over any `(n_samples, n_features)` array. |
| `global_explainer.py` | `explain_global` -- bundles a dataset-level SHAP explanation with its feature ranking. |
| `local_explainer.py` | `explain_prediction` -- one sample's signed per-feature SHAP contributions. |
| `attention_visualizer.py` | `generate_attention_map`, `compute_modality_importance`, `compute_attention_distributions`, and their plots -- summarizes the proposed model's raw `nn.MultiheadAttention` weights. |
| `explanation_report.py` | `generate_report` (generic markdown section assembler) plus `format_*` table helpers. |

## Why This Phase Reloads Checkpoints Instead of Retraining

Phases 9/10 already produced fully-trained, saved models (`ml/models/artifacts/classical/*/*.joblib`, `ml/models/artifacts/proposed/*/*.pt`). Explaining a model's predictions requires the trained model and the exact data split it was evaluated on -- not a fresh training run. `notebooks/phase11_explainability_report.py` therefore:

1. Re-runs Phases 5-8 (data loading, preprocessing, feature extraction) -- this is fast, deterministic, and necessary because the saved checkpoints don't carry the feature matrix with them, only the fitted model/scaler.
2. Reconstructs the identical group-aware split Phase 9/10 used (same subjects, same `SplitConfig`, same random seed).
3. Loads the saved model + scaler (`joblib.load`/`torch.load` + `load_state_dict`) rather than calling any `.fit()`/`train_*` function.
4. **Verifies** this reconstruction is faithful by comparing the rebuilt test split's labels against Phase 9's saved `random_forest_predictions.npz` `y_true` array -- if they don't match exactly, the script raises rather than silently explaining the wrong rows.

## Automatic SHAP Explainer Selection

`shap_explainer.select_explainer_type` maps a model's registry name to the appropriate SHAP explainer:

| Model family | Explainer | Why |
|---|---|---|
| `random_forest`, `xgboost` | `shap.TreeExplainer` | Exact Shapley values in polynomial time via the tree structure; no background data needed. |
| `logistic_regression` | `shap.LinearExplainer` | Exact for linear models given a background distribution. |
| anything else | `shap.KernelExplainer` (fallback) | Model-agnostic, sampling-based; slower, included for forward-compatibility rather than because this phase needs it (only the three families above are in scope). |

Different explainer/model combinations return differently-shaped output (tree ensembles on a binary target return one value set *per class*; linear/kernel explainers return one set already representing the positive class). `compute_shap_values` normalizes every case down to one 2D `(n_samples, n_features)` explanation of the positive (`fatigued`) class, so every downstream consumer (ranking, global/local explanation, plotting) is written once and works identically across all three model families.

## Attention Weights Are Not SHAP Values

`attention_visualizer.py` summarizes the proposed model's `CrossModalAttentionOutput.attention_weights` (exposed by Phase 10 for exactly this purpose). This is a useful, cheap view into what the model's attention mechanism computes internally, but it is **not** a rigorous attribution method the way SHAP is: SHAP values are guaranteed to sum to the model's output minus its baseline (verified directly in `tests/test_explainability.py`'s reconstruction tests); attention weights carry no such guarantee, a well-documented limitation of "attention as explanation" in the interpretability literature. The Phase 11 report keeps the two views separate rather than forcing a single ranking between EMG's 72 individual SHAP-attributed features and the proposed model's 3 modality-level attention weights -- they answer different questions at different granularities.

## Testing

`tests/test_explainability.py` (39 tests) covers: config validation, explainer-type dispatch for all three model families plus the kernel fallback, SHAP output-shape normalization (including a direct check that `base_value + sum(shap_values)` reconstructs the model's predicted probability -- SHAP's additivity property), every SHAP plot type, feature ranking correctness, global/local explanation orchestration, attention map/importance/distribution computation (both on synthetic tensors and a real `ProposedFatigueModel` forward pass), attention plots, and the report assembler/formatters. Run with:

```bash
python -m pytest tests/test_explainability.py -v
```

## What's Deliberately Not Here

- Subject personalization (leave-one-subject-out) -- a later phase.
- Agentic reasoning -- a later phase.
- Retraining or hyperparameter changes to any Phase 9/10 model -- this phase only explains what already exists.
- SHAP over the deep-learning baselines or non-attention fusion strategies -- out of this phase's explicit scope (Random Forest, XGBoost, Logistic Regression only).
