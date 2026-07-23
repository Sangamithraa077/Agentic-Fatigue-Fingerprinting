# Signal Preprocessing Pipeline — Phase 7

**Phase:** 7 — Signal Preprocessing (no feature extraction, no modeling, no explainability, no cross-modality fusion)

This document explains what the preprocessing pipeline under `ml/preprocessing/` does to each modality, and — more importantly — *why* each step is applied. Full API-level documentation lives in each module's docstrings; this is the human-readable summary.

## Scope

The pipeline prepares raw EMG/IMU/PPG signals (as loaded by the Phase 5 `WSD4FEDSRMLoader` into `ml.data_models` objects) for downstream analysis. It performs **cleaning and conditioning only**:

- No windowing/segmentation
- No scalar feature computation (RMS, mean frequency, etc.)
- No cross-modality resampling to a shared rate or fusion
- No modeling or explainability

Raw `ml.data_models` objects (`Dataset`/`Subject`/`Condition`/`EMGData`/...) are frozen dataclasses and are never mutated. Every preprocessing call returns a **new** parallel object (`ProcessedCondition`, `ProcessedSubject`, etc., defined in `preprocessing_results.py`) built from copies of the underlying arrays.

## Module Layout

| File | Responsibility |
|---|---|
| `quality_checks.py` | Pure validation functions (NaN/Inf/empty/sampling-rate/length checks) + `SignalQualityReport`. |
| `resampling.py` | Reusable single-signal resampling utilities (opt-in, not applied by default). |
| `signal_preprocessor.py` | Shared `ProcessedSignal` result type + `SignalPreprocessor` abstract base class. |
| `emg_preprocessor.py` | EMG-specific transforms + `EMGPreprocessor` + `EMGPreprocessingConfig`. |
| `imu_preprocessor.py` | IMU-specific transforms + `IMUPreprocessor` + `IMUPreprocessingConfig`. |
| `ppg_preprocessor.py` | PPG-specific transforms + `PPGPreprocessor` + `PPGPreprocessingConfig`. |
| `preprocessing_results.py` | Output dataclasses (`ProcessedEMGData`, `ProcessedIMUData`, `ProcessedPPGData`, `ProcessedCondition`, `ProcessedSubject`, `PreprocessingSummary`). |
| `pipeline.py` | `PreprocessingPipeline` orchestrator, accepting a `Subject` or `Condition` and producing the corresponding `Processed*` object plus a running `PreprocessingSummary`. |

## EMG Preprocessing (1000 Hz)

Applied in order, each step individually toggleable via `EMGPreprocessingConfig`:

1. **DC offset removal** (subtract mean) — surface EMG electrodes/amplifiers introduce a slowly varying baseline offset from the electrode-skin interface; this must be removed before filtering.
2. **Band-pass filter** (default 20–450 Hz, zero-phase Butterworth) — rejects sub-20Hz motion artifact/baseline wander and near-Nyquist noise; 20–450 Hz is the SENIAM-standard bandwidth for surface EMG at 1000 Hz.
3. **Full-wave rectification** (absolute value) — raw EMG oscillates around zero; amplitude/envelope analysis requires a unipolar signal proportional to activation magnitude.
4. **Optional linear envelope** (default off; low-pass at 6 Hz) — smooths the rectified signal into a slowly varying trace approximating the muscle activation/force profile — the conventional "linear envelope" technique.

**Note:** zero-phase filtering of a rectified (non-negative) signal can produce tiny negative excursions (filter ringing) right at the trial edges — an expected characteristic of non-causal filtering, not a defect.

## IMU Preprocessing (100 Hz)

Applied per stream (accelerometer, gyroscope, magnetometer), each step toggleable via `IMUPreprocessingConfig`:

1. **Timestamp/sampling-rate validation** — the raw files carry no timestamp column (samples are implicitly spaced at 1/rate), so validity here means: positive sample count, positive/finite sampling rate, and a finite positive implied duration. Catches corrupt/truncated files.
2. **Missing-value handling** (default: linear interpolation) — brief inertial-sensor dropouts are conventionally interpolated rather than left as gaps, preserving continuity for later windowing.
3. **Optional smoothing** (default off; Savitzky-Golay) — reduces high-frequency sensor noise while preserving the motion trend, without the phase distortion of a plain moving average.
4. **Sensor consistency checks** — accelerometer/gyroscope/magnetometer for one location are recorded in lockstep and must share sample counts; a mismatch flags a corrupted/partial file. Performed by `PreprocessingPipeline` across the three streams of each location (and additionally across all 6 locations within a condition).

## PPG Preprocessing (200 Hz)

Applied in order, toggleable via `PPGPreprocessingConfig`:

1. **Baseline drift removal** (default on; high-pass at 0.5 Hz) — motion, respiration, and finger-pressure changes introduce slow drift that dominates the raw trace and obscures the pulsatile cardiac component.
2. **Optional smoothing** (default on; low-pass at 8 Hz) — reduces high-frequency sensor noise while preserving pulse shape.

## Sampling Rates and Resampling

EMG (1000 Hz), IMU (100 Hz), and PPG (200 Hz) are each processed at their native rate. `resampling.py` provides `resample_signal`/`align_signal_length` utilities that a modality config can opt into (`target_sampling_rate_hz`) if a future phase needs a modality resampled for its own purposes — but **no config enables this by default**, and nothing in this phase aligns modalities onto a shared timebase. That alignment is explicitly deferred to the future multimodal fusion phase.

## Validation

Every signal is validated (via `quality_checks.validate_signal`) both **before** (`input_quality`) and **after** (`output_quality`) processing, checking: emptiness, NaNs, infinities, and sampling-rate validity. `PreprocessingPipeline` additionally checks **length consistency** across the 3 IMU streams within one location, across all locations within a condition, and across all 6 EMG channels within a condition. Every failure is logged and collected into the pipeline's `PreprocessingSummary` rather than aborting the run — one bad channel/condition/subject does not stop the rest from being processed.

## Usage

```python
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline, PreprocessingPipelineConfig
from ml.preprocessing.emg_preprocessor import EMGPreprocessingConfig

loader = WSD4FEDSRMLoader()
conditions = loader.discover_conditions()
subject = loader.load_subject("P01", "Subject 1", conditions)

pipeline = PreprocessingPipeline(
    PreprocessingPipelineConfig(emg=EMGPreprocessingConfig(compute_envelope=True))
)
processed_subject = pipeline.process_subject(subject)
print(pipeline.summary.as_text())
```

## Testing

`tests/test_preprocessing.py` covers the pure helper functions (quality checks, resampling, each modality's transforms) and an end-to-end pipeline check against synthetic `Condition`/`Subject` objects, verifying raw inputs are never mutated. Run with:

```bash
python -m pytest tests/test_preprocessing.py -v
```

## What's Deliberately Not Here

- Feature extraction (RMS, MAV, zero-crossing, mean/median frequency, etc.) — Phase 8.
- Windowing/segmentation into fixed-length analysis frames — Phase 8.
- Cross-modality resampling/alignment/fusion — a later fusion phase.
- Model training, SHAP explainability, personalization (LOSO), agents — later phases.
