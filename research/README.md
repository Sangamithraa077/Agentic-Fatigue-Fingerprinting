# Research study: predicting Borg RPE across athletes

This folder is the offline study behind the app's **Ridge Borg-RPE estimator**
(`ml/models/borg_regression/`). It asks one question: *can a model trained on
other athletes estimate a new athlete's Borg RPE from their EMG, IMU and PPG?*
Everything is evaluated leave-one-participant-out (LOPO): each of the 34
athletes is scored by a model that never saw them.

## Headline results (honest version)

| Model (LOPO, 888 sequences) | MAE (Borg points) | Correlation | Within 1 point |
|---|---|---|---|
| Always predict the training median | **2.18** | -- | 34.3% |
| Ridge regression (deployed) | 2.31 | 0.16 | 22.7% |
| LSTM + attention | 2.48 | 0.09 | 25.9% |

- The Ridge model is **not meaningfully better than guessing the median**. Its
  value is a weak but real positive correlation with the athlete's reported
  effort. Treat any single estimate as a research indication, not a measurement.
- Ridge beats the LSTM and is far more stable across athletes (fold standard
  deviation about 0.4 vs 1.4), which is why it is the deployed estimator.
- Noise added to the *extracted feature tables* (Gaussian jitter, amplitude
  scaling) made the LSTM generalize worse: MAE 2.48 -> 2.76 -> 2.85.
  Augmenting the **raw signals** and recomputing features (magnitude warping)
  was the only variant with a benefit: it lowered the LSTM's MAE (2.69 -> 2.57)
  and its variance across athletes by about 38%, and raised Ridge's
  correlation (0.16 -> 0.19) without lowering Ridge's MAE. Full tables and the
  reasoning are in [`results/final_synthesis_report.md`](results/final_synthesis_report.md).

## Pipeline

```
raw dataset ──► preprocessing_cleaning_pipeline.py ──► cleaned_data/
cleaned_data ─► feature_extraction.py ──► fatigue_features_all_subjects.csv
features ─────► prepare_lstm_sequences.py ──► lstm_sequences.npz + lstm_sequence_metadata.csv
sequences ────► nested_lopo_ridge_comparison.py ──► nested_lopo_ridge_*.csv
```

Each sequence is 3 consecutive 5-second windows (55 statistical features each,
165 inputs), labelled with the Borg rating at the end of the last window.
Consecutive sequences overlap, so they are not independent samples.

## Reproducing

The scripts use paths relative to where they run.

```bash
# 1. From the dataset root (datasets/raw/WSD4FEDSRM/): cleaning, then features
python path/to/research/preprocessing_cleaning_pipeline.py
python path/to/research/feature_extraction.py        # writes fatigue_features_all_subjects.csv

# 2. From research/ (inputs are already committed alongside the scripts)
cd research
python prepare_lstm_sequences.py                      # sequences + metadata
python nested_lopo_ridge_comparison.py                # LOPO Ridge vs median vs LSTM
python build_evaluation_metrics.py                    # regenerates reports/evaluation_metrics.json
```

Re-running step 2 overwrites `nested_lopo_ridge_*.csv` in this folder; the
published copies stay in `results/`.

## How it connects to the app

| Piece | Where |
|---|---|
| Packaged sequences (identical to `lstm_sequences.npz`) | `ml/models/borg_regression/data/` |
| Ridge LOPO fit, reproduced exactly | `ml/models/borg_regression/ridge_lopo.py` |
| Live estimate in the replay payload | `backend/services/replay_engine.py` (`prediction.borg_regression`) |
| Served metrics | `GET /api/reports/metrics` (from `reports/evaluation_metrics.json`) |

`tests/test_borg_regression.py` checks that the packaged model reproduces
`results/nested_lopo_ridge_predictions.csv` for every athlete.

## What is deliberately not here

The study also tried personalization (calibration-adjusted and fine-tuned
per-athlete models, `module7_*`). Those results did not beat the population
model on error, and the saved `population_ridge.joblib` there was fitted on
all athletes including its own evaluation sequences, so it is **not** used by
the app; the app refits a fresh held-out model per athlete instead.
