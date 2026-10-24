# Dataset pipeline (Modules 1 to 7)

An audited, reproducible path from the raw WSD4FEDSRM recordings to leave-one-person-out estimates of Borg RPE, built
in `ml/dataset_pipeline/`. It sits next to the earlier study in `research/` and the deployed models in `ml/models/`;
it does not change them. Outputs are in `reports/dataset_modules_1_7/`.

## Run it

```bash
# the dataset must be at datasets/raw/WSD4FEDSRM/  (or set WSD4FEDSRM_DATA to its folder)
python -m ml.dataset_pipeline.m1_audit            # Module 1: inventory, problems, participant table
python -m ml.dataset_pipeline.m3_clean            # Module 3: align, flag, filter, window, label, quality score   (~90 s)
python -m ml.dataset_pipeline.m4_features         # Module 4: feature table and checks
python -m ml.dataset_pipeline.m5_baselines        # Module 5: baselines
python -m ml.dataset_pipeline.m5b_improved        # stronger baselines and the best generalised model
python -m ml.dataset_pipeline.m2_eda              # Module 2: figures and marker correlations
python -m ml.dataset_pipeline.m6_augmentation     # Module 6: augmentation on the Module 5 model        (~10 min)
python -m ml.dataset_pipeline.m6b_augmentation_boosting   # ... and on the best model
python -m ml.dataset_pipeline.m7_personalised     # Module 7: personalisation, saves the models             (~10 min)
python -m ml.dataset_pipeline.explain_windows     # top drivers per window (for the dashboard)
python -m ml.dataset_pipeline.compare_with_research       # like-for-like comparison with research/
python -m ml.dataset_pipeline.common 1 task4_35e  # load one participant and exercise
```

Cleaned signals (about 0.5 GB) go to `datasets/processed/clean_signals/` and are not committed. Tests that need no
dataset: `python -m pytest tests/test_dataset_pipeline.py tests/test_live_pipeline.py tests/test_session_lab.py`.

## What the audit found (Module 1, details in `reports/dataset_modules_1_7/m1_data_dictionary.md`)

- 204 recordings (34 people x 6 exercises), one continuous set each. No repeat sessions on other days, so
  personalisation can only be tested within a session. All files are present; EMG, IMU and PPG durations agree to 0.01 s.
- Participant 3, 45% internal rotation: two EMG channels are corrupt. Participant 27, 30-40% internal rotation: one
  accelerometer spike. Three exercises have a recorded length that disagrees with the sensors (participant 27's, 122 s
  against 80.9 s, is probably a typo). Participant 3's 30-40% internal rotation has IMU files 11 s apart in length.
- No 50 or 60 Hz mains peak in any recording, so no notch filter is applied.

## Design choices

Windows are 10 s long and end on each Borg rating (the label is that rating). Signals are never shuffled. Each
recording's first two windows form the person's baseline; `*_rel` features are the change from it. A channel's features
are left out (NaN) when under 80% of its samples in the window are valid. Magnetometers are not used. Every split is by person.

## Results (leave-one-person-out, 34 people, 1309 windows)

Elapsed time alone is a strong baseline, because effort rises with time:

| Model | MAE (Borg points) | within 1 point |
|---|---|---|
| Train-mean dummy | 3.17 | 16% |
| **Elapsed time only** | **1.79** | 34% |
| Fused sensors, raw features (RF) | 2.72 | 20% |
| Fused sensors, relative to own start (RF) | 1.92 | 31% |
| Time + exercise condition (boosting) | 1.71 | 39% |
| **Sensors + time + condition (boosting), used by the dashboard** | **1.65** | 40% |

- Normalising to each person's own start is essential (2.72 to 1.92).
- No sensor-only model beats elapsed time. Sensors + time + condition beats elapsed time alone (24 of 34 people, p = 0.006), but
  the sensors' own contribution is small: against the same model without them (time + condition, boosting) the error is
  1.65 vs 1.71 and the gain is not significant (better for 19 of 34 people, p = 0.30).
- The EMG trends that look like fatigue (median frequency falling, amplitude rising) mostly track time: after removing
  elapsed time the within-exercise correlations drop to about 0.1.
- The model itself leans on time: elapsed time is the largest single driver in 972 of 1309 windows, then rotator-cuff
  muscle signal speed (221).
- **Augmentation does not help.** Noise, gain drift, time warp and overlapping windows each changed the error by under
  0.01 Borg points on the best model, with no significant gain (`m6b_augmentation_boosting_results.csv`). Train on real windows only.

### Like-for-like with the earlier study (`comparison_with_research.csv`)

The earlier study scores 888 sequences that all end at 30 s or later, so its headline numbers are not comparable to the
tables above. On exactly those windows: the study's training-median baseline 2.18 and deployed Ridge 2.31; elapsed time
alone 1.74; sensors only (random forest) 1.93; sensors + time + condition (boosting) **1.58**. The sensors-only model is
no better than the study's median baseline; the gain comes from elapsed time and the exercise condition, which the
study's sequence models did not use.

### Personalisation (Module 7), with a correction

Two ways of calibrating were tested, and they disagree:

| Calibration | Generalised | Personalised | People better | p |
|---|---|---|---|---|
| First **half** of each recording (not deployable) | 1.37 | 1.18 | 23 of 34 | 0.006 |
| First 30 s (3 windows), deployable | 1.50 | 1.48 | 16 of 34 | 0.89 |
| First 40 s (4 windows), deployable | 1.37 | 1.35 | 16 of 34 | 0.81 |
| First 50 s (5 windows), deployable | 1.26 | 1.27 | 17 of 33 | 0.58 |

**The first-half result should not be relied on.** It calibrates on "the first half of the recording", which a live
system cannot know in advance, and it probably lets the normalisation carry information about how long the exercise lasts
(not proven; the fixed-length test below removes the possibility). With
a fixed calibration period, the only thing a real device can do, personalising brings **no significant gain**. The
dashboard therefore defaults to the generalised estimate, offers the personalised one as an alternative view, and says
so next to the toggle. Fine-tuning the model's output with the calibration windows (bias or linear correction) made
things worse, as did models fitted on one person's windows alone. Cross-day personalisation could not be tested.

## Caveats

- 34 people, one session each; windows within an exercise are correlated, so everything is judged by person.
- The label is self-rated effort, which is close to a function of elapsed time; sensors add a modest amount.
- Heart-rate variability from 10 s windows is rough; PPG-only models are the weakest.
- The estimate is a research guide with a typical error of about 1.7 Borg points, not a measurement.
