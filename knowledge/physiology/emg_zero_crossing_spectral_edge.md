---
chunk_id: PHYS-EMG-04
title: Zero-Crossing Rate and Spectral Edge Frequency as Fatigue Indicators
category: physiology
source: Stulen & De Luca (1981); Oskoei & Hu (2007) Myoelectric Control Systems
date: 2026-01-15
tags: [emg, zero_crossing, spectral_edge, fatigue, frequency_domain]
---
# Zero-Crossing Rate and Spectral Edge Frequency as Fatigue Indicators

Beyond Median Frequency (MDF) and Mean Power Frequency (MPF), two additional frequency-domain features track muscle fatigue:

**Zero-Crossing Rate (ZCR):**
The number of times the EMG signal crosses zero amplitude per unit time decreases with fatigue, because the signal spectral content shifts toward lower, slower oscillations. A drop in ZCR greater than 20% below resting baseline is a secondary indicator of localized neuromuscular fatigue.

**Spectral Edge Frequency (SEF90):**
The frequency below which 90% of total EMG spectral power lies. As fatigue progresses, SEF90 decreases as power shifts from high-frequency motor unit firings to lower-frequency components. SEF90 is less sensitive to electrode placement variation than MDF, making it useful for hardware sensors.

In the WSD4FEDSRM dataset, the Anterior Deltoid shows the most pronounced spectral edge shift across all six shoulder muscles during internal rotation tasks.
