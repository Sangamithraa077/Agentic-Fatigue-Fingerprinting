# Baseline Modeling Framework — Phase 9

**Phase:** 9 — Baseline Modeling Framework (benchmark models only; no multimodal attention, explainability, or personalization — those are later phases)

This document explains the design of `ml/models/` and why it's structured this way. Full API-level documentation lives in each module's docstrings; this is the human-readable summary.

## Scope

A reproducible benchmark comparing classical ML and deep learning models on the Phase 8.5 validated feature matrices, using **identical datasets and evaluation protocols** for every model. No model here is the project's novel contribution — that's multimodal attention fusion, personalization (LOSO), explainability (SHAP), and agentic reasoning, all later phases. This phase only establishes a strong, honest bar for them to clear.

## Module Layout

| File | Responsibility |
|---|---|
| `common/config.py` | `DatasetConfig`, `SplitConfig`, `TrainingConfig`, `ExperimentConfig` — every experiment is fully described by these. |
| `common/dataset_builder.py` | Builds `(X, y, groups)` from a Phase 8.5 `FeatureDataset` — target derivation, feature-subset selection, imputation/outlier-clipping, scaling. |
| `common/data_splitter.py` | Group-aware stratified K-fold (classical) and train/val/test (deep learning) splitting. |
| `common/metrics.py` | Classification metrics + timing/memory instrumentation, shared by every model. |
| `common/evaluation.py` | `evaluate_model` — identical evaluation path for classical and deep learning models via a shared `.predict`/`.predict_proba` interface. |
| `common/callbacks.py` | `EarlyStopping`, `ModelCheckpoint` — framework-agnostic. |
| `common/model_registry.py` | Name → availability/metadata catalog (e.g. XGBoost's fallback status). |
| `classical/{logistic_regression,svm,random_forest,xgboost}.py` | Thin, configurable factories over sklearn/xgboost estimators. |
| `deep_learning/{cnn1d,bilstm,cnn_bilstm,tcn}.py` | Four independent `torch.nn.Module` architectures. |
| `deep_learning/training.py` | Reusable mini-batch training loop (seeding, device selection, early stopping, LR scheduling, checkpointing). |
| `deep_learning/inference.py` | `TorchClassifierAdapter` — makes a trained `nn.Module` sklearn-compatible for `evaluate_model`. |

## Why Deep Learning Models Consume the Same Tabular Feature Matrix

Every model in this benchmark trains on the **exact same Phase 8 feature matrix** (MAV, RMS, per-axis IMU statistics, ...) — not raw signal windows — because "identical datasets" is a hard requirement for a fair comparison. The CNN1D/BiLSTM/CNN-BiLSTM/TCN architectures therefore treat one sample's feature vector as a length-N "sequence" (one step per feature), a standard tabular-deep-learning technique. This is a deliberate trade-off, documented again in `common/dataset_builder.py` and flagged in the report's Limitations section: these models are not exploiting genuine temporal structure the way they would on raw signals — that's exactly what the future multimodal attention phase should do differently.

## Group-Aware Splitting (Not Naive Random Splitting)

Phase 8 windows overlap by 50%. A naive row-level random split could put two near-duplicate overlapping windows on both sides of a train/test split — silent leakage that would inflate every metric in this report. `data_splitter.py` groups by `subject_id` instead: `StratifiedGroupKFold` for classical CV, and a two-step `GroupShuffleSplit` (test first, then val from the remainder) for the deep learning train/val/test split. This also means the exact same utilities can be reused unmodified for a future leave-one-subject-out personalization phase — just enumerate one held-out subject per fold instead of a random group split.

## Shared Held-Out Test Set

For a given (modality, feature_subset, target), **one** group-aware train/val/test split is computed and shared: classical models fit their final artifact on train+val (after separately running stratified group K-fold CV *within* train+val for robustness reporting); deep learning models train on train with val for early stopping/checkpointing. Both categories are then evaluated on the **same** held-out test set — this is what makes the cross-category comparison table meaningful.

## Targets (from Phase 8.5's `rpe_nearest`)

| Target | Derivation | Classes |
|---|---|---|
| `binary_fatigue` | RPE ≥ 13 → fatigued (else not) | 2 |
| `fatigue_level` | RPE binned: [6,11)=low, [11,16)=moderate, [16,21)=high | 3 |
| `borg_rpe_class` | Rounded integer RPE, label-encoded | up to 15 |

MVIC-calibration windows (no Borg data) are excluded from all three. See `dataset_builder.py`'s module docstring for exact rules.

## Feature Subsets

`"all"`, `"variance_filtered"`, `"correlation_filtered"`, `"mutual_info_ranked"` map directly onto Phase 8.5's `feature_selection` recommendations — `mutual_info_ranked` computes MI directly against the actual modeling target (not a generic metadata proxy), which is more relevant here than reusing Phase 8.5's diagnostic MI scores.

## A Real Data-Quality Bug This Phase Surfaced and Fixed

Phase 8.5 documented ~10 EMG windows with raw values on the order of 1e28–1e34 (confirmed corrupted raw files, Subject 3). These are *finite* (not NaN/inf), so a plain inf/NaN check misses them — and dividing by a `RobustScaler`'s tiny IQR can push such a value past float32's representable range, producing a **new** `inf` that then crashes sklearn/torch. `dataset_builder._impute_and_clean` now also clips per-column values whose robust z-score (median/MAD-based) exceeds 1000 — deliberately very high, so only many-orders-of-magnitude corruption like this is affected, not ordinary high-but-real variance (e.g. the separate, much milder IMU gyroscope-energy outliers Phase 8.5 flagged are nowhere near this threshold and are left untouched).

## XGBoost's Graceful Fallback

`classical/xgboost.py` wraps the import in a try/except; if unavailable, `build_xgboost()` logs why and returns a `sklearn.ensemble.GradientBoostingClassifier` with an equivalent hyperparameter mapping instead, exposing the identical `.fit`/`.predict`/`.predict_proba` interface so callers never branch on which one they received. (XGBoost was available in this project's environment, so the fallback path is exercised only by its unit test.)

## Testing

`tests/test_models_common.py`, `tests/test_models_classical.py`, `tests/test_models_deep_learning.py` (63 tests) cover config validation, target derivation, feature-subset selection, group-leakage-free splitting, metrics, callbacks, the shared evaluation path, every classical model builder, and every deep learning architecture's forward pass + a real (loss-decreasing, early-stopping-triggering) training run. Run with:

```bash
python -m pytest tests/test_models_common.py tests/test_models_classical.py tests/test_models_deep_learning.py -v
```

## What's Deliberately Not Here

- Multimodal attention / cross-modality fusion — a later phase.
- SHAP or any other explainability — Phase 10.
- Leave-one-subject-out personalization — a later phase (the group-aware splitters are built to support it without modification).
- Hyperparameter tuning — every model uses its module's documented default configuration; this is a baseline comparison, not an optimized one.
