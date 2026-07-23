# Feature Engineering Pipeline — Phase 8

**Phase:** 8 — Feature Engineering (no model training, no explainability, no cross-modality fusion)

This document explains what `ml/feature_engineering/` computes and why. Full API-level documentation lives in each module's docstrings; this is the human-readable summary.

## Scope

The pipeline takes the `Processed*` objects produced by Phase 7 (`ml.preprocessing.preprocessing_results`) and computes **reusable numerical features** over configurable sliding windows. It does not:

- Train or evaluate any model
- Perform explainability (SHAP, etc.)
- Fuse modalities into one combined feature vector — each modality's features are kept separate, ready for a future fusion phase to combine

## Module Layout

| File | Responsibility |
|---|---|
| `windowing.py` | Configurable sliding-window segmentation (`WindowConfig`, pure index/segmentation functions). |
| `feature_extractor.py` | Shared `FeatureExtractor` ABC + generic numeric primitives (mean/std/rms/variance/energy/...) reused across modalities. |
| `feature_vector.py` | `FeatureVector` (names+values) and the Condition/Subject-level aggregation dataclasses, plus `FeatureExtractionSummary` and a DataFrame-flattening helper. |
| `emg_features.py` | EMG time- and frequency-domain feature formulas + `EMGFeatureExtractor`. |
| `imu_features.py` | IMU per-axis and resultant-vector feature formulas + `IMUFeatureExtractor`. |
| `ppg_features.py` | PPG peak-based feature formulas + `PPGFeatureExtractor`; documents which HRV metrics are intentionally not implemented. |
| `feature_pipeline.py` | `FeaturePipeline` orchestrator: windows every signal and applies the matching extractor, producing `ConditionFeatures`/`SubjectFeatures`. |

## Windowing

Window length and overlap are configured in **seconds**, via `WindowConfig(window_seconds, overlap_ratio)`, and converted to each signal's own sample count via `compute_window_samples`. This lets the same time-based intent apply consistently across EMG (1000 Hz), IMU (100 Hz), and PPG (200 Hz).

`FeaturePipelineConfig` holds **three independent `WindowConfig` instances** (`emg_window`, `imu_window`, `ppg_window`) rather than one shared window size, because the physiologically appropriate window length genuinely differs:

- EMG/IMU default: 1.0s window, 50% overlap — short enough to track fast-changing muscle/motion activity.
- PPG default: 10.0s window, 50% overlap — long enough to contain several heartbeats even at a resting ~50 bpm; a 1-second PPG window could contain zero or one beat, making pulse-interval statistics meaningless.

## EMG Features

**Time domain** (the classic Hudgins feature set + IEMG): Mean Absolute Value, RMS, Variance, Waveform Length, Zero Crossings, Slope Sign Changes, Willison Amplitude, Integrated EMG.

**Frequency domain** (via a Welch power-spectral-density estimate): Mean Frequency, Median Frequency, Peak Frequency, Spectral Entropy. MNF/MDF are the standard EMG fatigue indicators — both are expected to shift lower as a muscle fatigues, directly relevant to this project's fatigue-fingerprinting goal.

**Important caveat, documented in code and here:** Zero Crossings, Slope Sign Changes, and Willison Amplitude are classically computed on a filtered-but-**not**-rectified EMG signal — they rely on the signal crossing/changing sign around zero. Phase 7's `EMGPreprocessor` defaults to `rectify=True`, which makes these three features degenerate on the default preprocessing output (near-zero ZC/SSC, an over-counted WAMP). `EMGFeatureExtractor` detects this (`is_likely_rectified`) and logs a single warning per extractor instance rather than either silently producing misleading numbers or spamming one warning per window. To get meaningful values for these three features, preprocess upstream with `EMGPreprocessingConfig(rectify=False)`.

## IMU Features

**Per axis** (x, y, z independently, for each of accelerometer/gyroscope/magnetometer): Mean, Std, RMS, Max, Min, Range, Energy, and a per-axis SMA contribution (`mean(|axis|)`).

**Resultant-vector features**, computed on the per-sample magnitude `r = sqrt(x²+y²+z²)`: Mean, Std, RMS, Max, Min, Energy, and the conventional tri-axial **Signal Magnitude Area** = `mean(|x|+|y|+|z|)`. The resultant is orientation-independent, which per-axis features are not.

Feature names are prefixed by stream and axis (e.g. `acc_x_mean`, `gyr_resultant_sma`) so every value's meaning is recoverable from its name alone. One IMU location's full feature set (3 streams × (3 axes × 8 + 7 resultant features) = 93 features) is combined into a single flat `FeatureVector` by `IMUFeatureExtractor.extract_location`.

## PPG Features

Implemented: Peak Count, Heart Rate (bpm), Pulse Interval Statistics (mean/std/min/max, seconds), Signal Energy. Peaks are found via `scipy.signal.find_peaks` with a minimum-distance constraint derived from a plausible maximum heart rate (default 200 bpm) and an amplitude-prominence threshold.

**Explicitly not implemented, per this phase's instruction to document rather than force an unreliable result:** RMSSD, pNN50, frequency-domain HRV (LF/HF power ratio), and nonlinear/Poincaré HRV measures. Two independent reasons, both documented in `ppg_features.py` and surfaced in `FeatureExtractionSummary.skipped_features`:

1. **Trial duration.** WSD4FEDSRM PPG recordings run ~20–252s (mean ~72s, Phase 6 EDA) — shorter, in most cases, than the ≥2–5 minutes conventionally required for stable frequency-domain HRV spectral resolution, and windowing reduces this further.
2. **Peak-detection reliability.** Peaks are found with a generic amplitude/distance detector on band-pass-filtered PPG, not a validated PPG-specific systolic-peak algorithm. RMSSD/pNN50 are highly sensitive to a few milliseconds of detection jitter that this detector is not validated to avoid.

When a window has fewer than 2 detected peaks, heart rate and interval statistics are reported as `nan` (logged), never silently defaulted to zero.

## FeaturePipeline and Reporting

`FeaturePipeline.extract_condition` / `.extract_subject` / `.extract` (type-dispatched) window every EMG channel, IMU location, and PPG signal and apply the matching extractor, returning `ConditionFeatures`/`SubjectFeatures`. `condition_features_to_dataframe` flattens a result into one tidy `pandas.DataFrame` (one row per window, one column per named feature) for convenient downstream use — a pure reorganization of already-computed values, not new computation.

`FeaturePipeline.summary` (a `FeatureExtractionSummary`) reports: conditions processed, window counts per modality, every distinct feature name produced (grouped by category), the documented skipped HRV features with justification, and any warnings encountered (including the rectified-EMG-input caveat above, if triggered).

## Usage

```python
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.feature_engineering.feature_pipeline import FeaturePipeline

loader = WSD4FEDSRMLoader()
conditions = loader.discover_conditions()
subject = loader.load_subject("P01", "Subject 1", conditions)

processed_subject = PreprocessingPipeline().process_subject(subject)
subject_features = FeaturePipeline().extract_subject(processed_subject)
```

## Testing

`tests/test_feature_engineering.py` (39 tests) covers windowing, the shared generic primitives, every modality's individual feature formulas (including the rectified-EMG warning behavior and the PPG insufficient-peaks `nan` case), and an end-to-end `FeaturePipeline` check against synthetic `ProcessedCondition`/`ProcessedSubject` objects. Run with:

```bash
python -m pytest tests/test_feature_engineering.py -v
```

## What's Deliberately Not Here

- Model training, evaluation, or hyperparameter tuning — a later phase.
- SHAP or any other explainability — a later phase.
- Cross-modality fusion into one combined feature vector — a later phase; each modality's `FeatureVector`s remain separate here.
- Feature selection/dimensionality reduction — a later phase.
