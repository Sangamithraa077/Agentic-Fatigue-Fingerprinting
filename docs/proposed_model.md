# Proposed Multimodal Fusion Model — Phase 10

**Phase:** 10 — The Proposed Model (this project's novel architectural contribution; no SHAP, personalization, or agentic AI — later phases)

This document explains the design of `ml/models/proposed/` and why it's structured this way. Full API-level documentation lives in each module's docstrings; this is the human-readable summary.

## Scope

A modular multimodal neural network that learns a shared fatigue representation from EMG, IMU, and PPG **feature vectors** (Phase 8's output, not raw signals) via independent modality encoders and configurable attention-based fusion. This is the project's proposed contribution — later phases add explainability (SHAP), personalization (leave-one-subject-out), and agentic reasoning on top of it, without changing this architecture.

## Module Layout

| File | Responsibility |
|---|---|
| `config.py` | `EncoderConfig`, `AttentionConfig`, `FusionConfig`, `ClassifierConfig`, `LossConfig`, `ProposedModelConfig`, `ProposedTrainingConfig`. |
| `encoders.py` | `ModalityEncoder` — configurable MLP (Linear → BatchNorm → ReLU → Dropout, repeated) mapping one modality's raw feature vector to a shared embedding size. |
| `attention.py` | `CrossModalAttention` — a configurable stack of `nn.MultiheadAttention`-based self-attention layers over the 3 modality embeddings treated as a length-3 sequence. |
| `fusion.py` | `FusionModule` — combines the 3 modality embeddings via `"attention"` (default), `"concatenation"`, or `"weighted_sum"`. |
| `classifier.py` | `ClassificationHead` — configurable MLP mapping the fused representation to class logits (binary or multiclass, identically). |
| `model.py` | `ProposedFatigueModel` — wires encoders → fusion → classifier; `ModelOutput` carries predictions, probabilities, attention weights, and intermediate embeddings. |
| `losses.py` | `build_loss_function` — class-weighted cross-entropy (weights computed from training labels). |
| `training.py` | `train_proposed_model` — mini-batch training loop reusing Phase 9's shared primitives. |
| `inference.py` | `ProposedModelAdapter` — makes the 3-input model sklearn-compatible for Phase 9's `evaluate_model`. |

Plus one necessary addition outside `ml/models/proposed/`: `ml/models/common/multimodal_dataset_builder.py` (see below) — data-pipeline plumbing, not model architecture, so it lives alongside Phase 9's other shared dataset infrastructure in `common/`.

## Why a New Multimodal Alignment Step Was Needed

Phases 7/8/8.5 deliberately kept EMG, IMU, and PPG on **separate window grids** — EMG/IMU default to 1.0s windows, PPG to 10.0s windows (a 1-second window can't contain enough heartbeats for meaningful pulse-rate features). That was the right call for single-modality feature extraction, but a fusion model needs one row per multimodal sample. Solving that alignment is exactly this phase's job, not a further deferral:

1. **Combine across sensors within a modality.** Each Phase 8.5 `FeatureDataset` is long-format (one row per sensor per window — e.g. 6 rows per EMG window, one per muscle). `combine_sensors_per_window` pivots this into one row per window, with every sensor's features as separate, sensor-prefixed columns (e.g. `anterior_deltoid__mav`).
2. **Align across modalities by nearest window-center time**, via `pandas.merge_asof(..., direction="nearest")` grouped by `(subject_id, condition_id)`. EMG and IMU share the same window config (verified empirically: identical window-index sets per condition), so they align almost exactly; PPG's much coarser windows are matched the same way, so several EMG/IMU windows legitimately share one PPG window's features — an intentional many-to-one mapping, not a bug.

Verified against real data: the row-count arithmetic reconciles exactly (e.g. excluded-row counts match raw MVIC-window counts precisely), confirming the alignment and target-exclusion logic are correct together.

## Why Deep Learning Encoders Consume Tabular Feature Vectors

Consistent with Phase 9: every encoder operates on Phase 8's already-extracted feature vector (EMG 72-d from 6 muscles × 12 features, IMU 558-d from 6 locations × 93 features, PPG 7-d), not raw signal windows. This keeps the whole pipeline (Phases 7 → 8 → 8.5 → 9 → 10) consistent and avoids reloading/reprocessing raw signals, per this phase's explicit requirement.

## Cross-Modal Attention: How It Works

The 3 modality embeddings (each already projected to the same `embed_dim` by their own `ModalityEncoder`) are stacked into a length-3 "sequence" and passed through `nn.MultiheadAttention` as **self-attention over that sequence** — each modality's representation is updated based on (attends to) the other two. This reuses PyTorch's standard, well-tested attention implementation rather than a bespoke mechanism. Attention weights (shape `(batch, num_heads, 3, 3)`, kept per-head and per-layer, not averaged away) are returned alongside the contextualized embeddings specifically so a later phase can inspect *which* modality most influenced a given prediction — ahead of, and feeding into, SHAP-based explainability.

## Fusion Strategies

| Strategy | Mechanism | Constraint |
|---|---|---|
| `attention` (default) | Cross-modal attention, then mean-pool across modalities | All modalities must share `embed_dim` |
| `concatenation` | Simple concatenation of the 3 embeddings | None — the only strategy tolerating different per-modality embedding sizes |
| `weighted_sum` | Learned softmax-normalized scalar weight per modality | All modalities must share `embed_dim` |

The equal-embedding-size constraint is validated inside `FusionModule.__init__` itself (where the constraint actually originates), not by the caller — a real bug caught during testing: an earlier version validated this in `model.py` using a single shared `embed_dim` even for `concatenation`, which incorrectly rejected/mis-sized valid mismatched-dimension concatenation configs. Fixed by having `FusionModule` accept a `{modality: dim}` dict and compute `fused_dim` correctly per strategy.

## Reusing Phase 9's Framework (and Where It Couldn't Be Reused Verbatim)

- `ml.models.common.callbacks.{EarlyStopping,ModelCheckpoint}` and `ml.models.deep_learning.training.{set_seed,get_device}` are imported and used **directly, unmodified** — they're input-shape-agnostic utilities.
- `ml.models.common.evaluation.evaluate_model` is reused **completely unmodified**: `ProposedModelAdapter` accepts a single horizontally-concatenated `(EMG|IMU|PPG)` array and splits it internally via `ModalityDims`, so from `evaluate_model`'s point of view this 3-input model looks exactly like any single-input Phase 9 model.
- `ml/models/proposed/training.py`'s mini-batch loop is a **new implementation**, not a call into Phase 9's single-input `train_model` — that function's signature takes one input tensor per sample; this model takes three. Reusing the shared primitives while writing a thin, input-shape-appropriate loop is what "no duplicated logic" actually calls for here: the loop's structure looks similar to Phase 9's by design (same callback semantics), not because logic was copy-pasted.
- **In-Memory Data Augmentation**: `train_proposed_model` accepts `augment_training: bool = True` (or configured via `ProposedTrainingConfig.augment_training`). When enabled, `TrainingDataAugmenter` applies Gaussian noise and magnitude scaling across the 3 modality arrays (`X_emg`, `X_imu`, `X_ppg`) prior to `DataLoader` construction, doubling training samples in RAM without leaking into validation/test splits.
- Mixed precision (`torch.autocast` + `torch.amp.GradScaler`) is implemented and gated on `device.type == "cuda"`; this project's environment is CPU-only, so the path is implemented and unit-tested but not exercised end-to-end here.

## Testing

`tests/test_multimodal_dataset_builder.py` (6 tests) and `tests/test_models_proposed.py` (31 tests) cover: sensor-combination pivoting, cross-modality nearest-time alignment (including MVIC exclusion and empty-PPG graceful handling), every fusion strategy's shapes/constraints, the classifier, class-weight computation, the training loop (loss reduction, early stopping, checkpointing), and the adapter's reuse of Phase 9's shared `evaluate_model`. Run with:

```bash
python -m pytest tests/test_multimodal_dataset_builder.py tests/test_models_proposed.py -v
```

## What's Deliberately Not Here

- SHAP or any other explainability beyond raw attention-weight tensors — a later phase.
- Leave-one-subject-out personalization — a later phase (this model's outputs are already personalization-ready: per-subject predictions/embeddings/attention weights are all exposed).
- Agentic reasoning — a later phase.
- Hyperparameter tuning — this phase's configuration was chosen once, not searched.
