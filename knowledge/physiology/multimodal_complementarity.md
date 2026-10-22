---
chunk_id: PHYS-MULTIMODAL-01
title: Multimodal Complementarity — Why EMG, IMU, and PPG Together Are Stronger Than Any Single Sensor
category: physiology
source: Yang et al. (2021) Multimodal Physiological Fatigue Assessment Review
date: 2026-01-15
tags: [multimodal, emg, imu, ppg, sensor_fusion, complementarity, fatigue_detection]
---
# Multimodal Complementarity — Why EMG, IMU, and PPG Together Are Stronger Than Any Single Sensor

**The fundamental problem with single-sensor fatigue assessment:**
* **EMG alone:** Captures local neuromuscular fatigue but is sensitive to electrode displacement and sweat. Cannot measure cardiovascular strain or kinematic breakdown independently.
* **IMU alone:** Captures movement coordination and tremor but cannot detect early-stage localized muscle fatigue (a limb may move normally until a muscle is near failure).
* **PPG alone:** Captures systemic cardiovascular load but is confounded by motion artifacts and cannot localize which muscle is fatiguing.

**Complementary roles in this framework:**
1. **EMG** detects the earliest intramuscular biochemical signature: spectral frequency compression preceding any observable motor failure.
2. **IMU** detects the behavioral consequence that follows: coordination breakdown, increasing jerk, tremor bursts, and tempo slowing.
3. **PPG** provides the systemic context: elevated heart rate and declining HRV confirm that fatigue is genuine whole-body metabolic exertion, not local discomfort.

Cross-modal attention fusion (implemented in the proposed model) quantifies which modality is most informative for each prediction — the attention heat-map reveals whether EMG, IMU, or PPG was the dominant signal in any given window.
