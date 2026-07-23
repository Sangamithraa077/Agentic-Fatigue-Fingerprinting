# WSD4FEDSRM -- Feature Extraction Report

**Phase:** 8 -- Feature Engineering (no model training, no explainability, no cross-modality fusion)

**Reproduce:** `python notebooks/phase8_feature_engineering_report.py` from the project root (runs Phase 5 load -> Phase 7 preprocess -> Phase 8 feature extraction for subject P01).

---

## 1. What This Report Covers

Generated from real data for one representative subject (**P01**, all 10 conditions) using default configuration for every pipeline stage. The pipeline itself is unchanged for any other subject or the full dataset -- this single-subject run is chosen only to keep the report fast and deterministic to regenerate.

## 2. Extraction Summary

- **Conditions processed:** 10
- **EMG windows extracted:** 5610
- **IMU windows extracted:** 5610
- **PPG windows extracted:** 79
- **Total distinct feature names:** 112

## 3. Feature Categories

### emg_time_domain (8 features)

`mav`, `rms`, `variance`, `waveform_length`, `zero_crossings`, `slope_sign_changes`, `willison_amplitude`, `integrated_emg`

### emg_frequency_domain (4 features)

`mean_frequency`, `median_frequency`, `peak_frequency`, `spectral_entropy`

### imu (93 features)

`acc_x_mean`, `acc_x_std`, `acc_x_rms`, `acc_x_max`, `acc_x_min`, `acc_x_range`, `acc_x_energy`, `acc_x_sma`, `acc_y_mean`, `acc_y_std`, `acc_y_rms`, `acc_y_max`, `acc_y_min`, `acc_y_range`, `acc_y_energy`, `acc_y_sma`, `acc_z_mean`, `acc_z_std`, `acc_z_rms`, `acc_z_max`, `acc_z_min`, `acc_z_range`, `acc_z_energy`, `acc_z_sma`, `acc_resultant_mean`, `acc_resultant_std`, `acc_resultant_rms`, `acc_resultant_max`, `acc_resultant_min`, `acc_resultant_energy`, `acc_resultant_sma`, `gyr_x_mean`, `gyr_x_std`, `gyr_x_rms`, `gyr_x_max`, `gyr_x_min`, `gyr_x_range`, `gyr_x_energy`, `gyr_x_sma`, `gyr_y_mean`, `gyr_y_std`, `gyr_y_rms`, `gyr_y_max`, `gyr_y_min`, `gyr_y_range`, `gyr_y_energy`, `gyr_y_sma`, `gyr_z_mean`, `gyr_z_std`, `gyr_z_rms`, `gyr_z_max`, `gyr_z_min`, `gyr_z_range`, `gyr_z_energy`, `gyr_z_sma`, `gyr_resultant_mean`, `gyr_resultant_std`, `gyr_resultant_rms`, `gyr_resultant_max`, `gyr_resultant_min`, `gyr_resultant_energy`, `gyr_resultant_sma`, `mag_x_mean`, `mag_x_std`, `mag_x_rms`, `mag_x_max`, `mag_x_min`, `mag_x_range`, `mag_x_energy`, `mag_x_sma`, `mag_y_mean`, `mag_y_std`, `mag_y_rms`, `mag_y_max`, `mag_y_min`, `mag_y_range`, `mag_y_energy`, `mag_y_sma`, `mag_z_mean`, `mag_z_std`, `mag_z_rms`, `mag_z_max`, `mag_z_min`, `mag_z_range`, `mag_z_energy`, `mag_z_sma`, `mag_resultant_mean`, `mag_resultant_std`, `mag_resultant_rms`, `mag_resultant_max`, `mag_resultant_min`, `mag_resultant_energy`, `mag_resultant_sma`

### ppg (7 features)

`peak_count`, `heart_rate_bpm`, `mean_pulse_interval_s`, `std_pulse_interval_s`, `min_pulse_interval_s`, `max_pulse_interval_s`, `signal_energy`

## 4. Windows per Condition

| Condition | EMG windows | IMU windows | PPG windows |
|---|---|---|---|
| P01_external_rotation_30-40 | 900 | 900 | 14 |
| P01_internal_rotation_30-40 | 1260 | 1260 | 20 |
| P01_external_rotation_40-50 | 762 | 762 | 11 |
| P01_internal_rotation_40-50 | 912 | 912 | 14 |
| P01_external_rotation_50-60 | 582 | 582 | 8 |
| P01_internal_rotation_50-60 | 822 | 822 | 12 |
| P01_mvic_external_rotation_trial1 | 96 | 96 | 0 |
| P01_mvic_external_rotation_trial2 | 96 | 96 | 0 |
| P01_mvic_internal_rotation_trial1 | 90 | 90 | 0 |
| P01_mvic_internal_rotation_trial2 | 90 | 90 | 0 |

PPG windows are 0 for the 4 MVIC-calibration conditions -- PPG was not recorded during MVIC trials (confirmed in Phase 5/6/7), so this is expected, not a data quality issue.

## 5. Skipped Features (documented, not implemented)

- **rmssd**: Requires reliably jitter-free inter-beat intervals; not guaranteed by the generic peak detector used here.
- **pnn50**: Same reliability requirement as RMSSD; not implemented for the same reason.
- **lf_hf_ratio**: Frequency-domain HRV requires >=2-5 minutes of stable recording for adequate spectral resolution; WSD4FEDSRM PPG trials (~20-252s, further reduced by windowing) do not meet this.
- **poincare_sd1_sd2**: Nonlinear HRV measures share the same duration and peak-detection-reliability limitations above.

## 6. Warnings Encountered

- One or more EMG windows appeared to already be rectified (non-negative) input -- zero_crossings/slope_sign_changes/willison_amplitude are degenerate for those windows. See emg_features.py module docstring; set EMGPreprocessingConfig(rectify=False) upstream in Phase 7 to get meaningful values for these three features.

## 7. Example Flattened Feature Table

`condition_features_to_dataframe` for condition `P01_external_rotation_30-40` produces one long-format table (shape: 1814 rows x 115 columns) where each row belongs to exactly one modality/channel/window, so only that modality's feature columns are populated (the rest are NaN by construction, not missing data). One example row per modality, showing only its populated columns:

**EMG** (`anterior_deltoid`, window 0) -- 12 feature columns:

```
modality          channel  window_index      mav      rms  variance  waveform_length  zero_crossings  slope_sign_changes  willison_amplitude  integrated_emg  mean_frequency  median_frequency  peak_frequency  spectral_entropy
     EMG anterior_deltoid             0 0.010728 0.014852  0.000105         4.749353             0.0               464.0               836.0       10.728394       88.366594           70.3125        27.34375          0.814189
```

**IMU** (`forearm`, window 0) -- 93 feature columns:

```
modality channel  window_index  acc_x_mean  acc_x_std  acc_x_rms  acc_x_max  acc_x_min  acc_x_range  acc_x_energy  acc_x_sma  acc_y_mean  acc_y_std  acc_y_rms  acc_y_max  acc_y_min  acc_y_range  acc_y_energy  acc_y_sma  acc_z_mean  acc_z_std  acc_z_rms  acc_z_max  acc_z_min  acc_z_range  acc_z_energy  acc_z_sma  acc_resultant_mean  acc_resultant_std  acc_resultant_rms  acc_resultant_max  acc_resultant_min  acc_resultant_energy  acc_resultant_sma  gyr_x_mean  gyr_x_std  gyr_x_rms  gyr_x_max  gyr_x_min  gyr_x_range  gyr_x_energy  gyr_x_sma  gyr_y_mean  gyr_y_std  gyr_y_rms  gyr_y_max  gyr_y_min  gyr_y_range  gyr_y_energy  gyr_y_sma  gyr_z_mean  gyr_z_std  gyr_z_rms  gyr_z_max  gyr_z_min  gyr_z_range  gyr_z_energy  gyr_z_sma  gyr_resultant_mean  gyr_resultant_std  gyr_resultant_rms  gyr_resultant_max  gyr_resultant_min  gyr_resultant_energy  gyr_resultant_sma  mag_x_mean  mag_x_std  mag_x_rms  mag_x_max  mag_x_min  mag_x_range  mag_x_energy  mag_x_sma  mag_y_mean  mag_y_std  mag_y_rms  mag_y_max  mag_y_min  mag_y_range  mag_y_energy  mag_y_sma  mag_z_mean  mag_z_std  mag_z_rms  mag_z_max  mag_z_min  mag_z_range  mag_z_energy  mag_z_sma  mag_resultant_mean  mag_resultant_std  mag_resultant_rms  mag_resultant_max  mag_resultant_min  mag_resultant_energy  mag_resultant_sma
     IMU forearm             0    3.129327   1.152078   3.334662   5.098918    0.68488     4.414038   1111.996972   3.129327   -7.735229   0.629174   7.760775  -6.256437  -8.911368     2.654931   6022.962483   7.735229    5.173875   0.536292   5.201595   6.426088   3.793704     2.632384   2705.659433   5.173875            9.905066           0.543934           9.919989          10.890613           8.559528           9840.618887          16.038431    -0.07422   0.241293    0.25245   0.473551  -0.746534     1.220085      6.373105   0.189214    1.314506   0.864843   1.573493   2.496062  -0.050165     2.546227    247.588091    1.32057    -0.42093   0.825593   0.926707   0.733783  -1.616519     2.350302     85.878605   0.729764            1.563202           0.977137           1.843474           2.965971           0.049544            339.839801           2.239548     0.65324   0.117304   0.663688    0.80957   0.490723     0.318847     44.048228    0.65324    0.294194   0.102831   0.311648   0.568848   0.196777     0.372071      9.712456   0.294194    -0.34544   0.314827   0.467381    0.31665   -0.67041      0.98706      21.84448   0.414268            0.869483           0.007174           0.869512           0.883983           0.856128             75.605164           1.361702
```

**PPG** (`ppg`, window 0) -- 7 feature columns:

```
modality channel  window_index  peak_count  heart_rate_bpm  mean_pulse_interval_s  std_pulse_interval_s  min_pulse_interval_s  max_pulse_interval_s  signal_energy
     PPG     ppg             0        14.0       85.997795               0.697692              0.123423                 0.415                 0.795       3.884296
```
