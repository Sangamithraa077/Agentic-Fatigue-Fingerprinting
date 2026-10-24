# Dataset pipeline outputs (Modules 1 to 7, plus inputs of the Session lab)

Written by `ml/dataset_pipeline/`; see [`docs/dataset_pipeline_modules_1_7.md`](../../docs/dataset_pipeline_modules_1_7.md)
for what each step does and what the numbers mean. The dashboard's Session lab reads these tables directly.

| File | Module | Content |
|---|---|---|
| `m1_inventory.csv`, `m1_problems.csv`, `m1_participant_summary.csv`, `m1_data_dictionary.md` | 1 | per-recording inventory, flagged problems, per-person table, data dictionary |
| `m2_*.csv`, `figs/` | 2 | marker correlations with effort (also after removing elapsed time), between-person differences, figures |
| `m3_windows.csv`, `m3_cleaning_log.csv`, `m3_quality.csv`, `m3_rejected_windows.csv` | 3 | 1309 labelled windows with channel validity, what was flagged, a quality score per person |
| `m4_features.csv`, `m4_feature_check.txt` | 4 | one row per window: raw and baseline-relative features, effort label, strength, elapsed time |
| `m5_baseline_*.csv`, `m5_splits.csv`, `m5_time_split_dummies.csv` | 5 | baselines and the splits used |
| `m5b_improved_*.csv` | 5b | stronger baselines, sensors + time + condition models, per-window out-of-sample predictions |
| `m6_augmentation_results.csv`, `m6b_augmentation_boosting_results.csv` | 6 | augmentation tests (none helped) |
| `m7_results.csv`, `m7_by_person.csv`, `m7_predictions.csv` | 7 | personalisation with the first-half calibration (an oracle, not deployable) |
| `m7_live_calibration_*.csv` | 7 | personalisation with a fixed 30, 40 or 50 s calibration (deployable): no significant gain |
| `m8_driver_contributions.csv` | 12.7 | top five drivers per window |
| `m11_replay_equality.csv` | 11.7 | live replay versus offline estimates (difference 0.0) |
| `m14_e2e_report.md/.json` | 14 | end-to-end run with the hardware simulator |
| `comparison_with_research.csv` | | like-for-like comparison with `research/` on the same 888 windows |

Not committed (regenerable, large): the cleaned signals in `datasets/processed/clean_signals/` and the augmented
feature tables `m6_aug_features_*.csv`.
