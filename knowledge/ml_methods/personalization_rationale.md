---
chunk_id: ML-PERSONALIZATION-01
title: Personalised Fatigue Models — Why Population Models Underfit Individuals
category: ml_methods
source: Gjoreski et al. (2020) Personalization of Machine Learning Models; Spierer et al. (2022)
date: 2026-01-15
tags: [personalization, individual_variability, baseline_calibration, loso, fatigue_tolerance]
---
# Personalised Fatigue Models — Why Population Models Underfit Individuals

**The inter-subject variability problem:**
Fatigue tolerance differs dramatically across individuals based on:
* **Body composition:** Muscle fibre type ratio (slow-twitch vs. fast-twitch), muscle cross-sectional area, arm leverage length.
* **Baseline fitness:** Trained athletes have higher oxidative capacity and recruit type II fibres later; sedentary individuals exhaust their motor reserve earlier at the same relative workload.
* **Anthropometric factors:** Upper arm length affects the moment arm of shoulder rotations — participants with longer arms experience greater mechanical load per contraction.

**What this means for a population model:**
A global model trained on all 34 WSD4FEDSRM participants learns the *average* EMG-fatigue relationship. For any given individual, this average may be systematically biased — the model may overpredict fatigue for a strong athlete or underpredict for a deconditioned participant.

**The personalisation solution in this framework:**
The personalisation layer computes each individual's resting baseline (mean and standard deviation of each feature during their first low-fatigue windows) and adjusts the population model's output using a deviation signal: how unusual the *current* window is relative to *this specific person's* own normal range.
