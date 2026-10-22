---
chunk_id: ML-LOSO-01
title: Leave-One-Subject-Out (LOSO) Cross-Validation and Why It Matters for Wearables
category: ml_methods
source: Hammerla et al. (2016) Deep, Convolutional, and Recurrent Models for Human Activity Recognition; Stiefelhagen (2020)
date: 2026-01-15
tags: [loso, cross_validation, generalization, subject_independence, evaluation, leakage]
---
# Leave-One-Subject-Out (LOSO) Cross-Validation and Why It Matters for Wearables

**Standard k-fold cross-validation is wrong for wearable biosignal models.**

When continuous sensor recordings are split into overlapping sliding windows and randomly shuffled into folds, adjacent windows (e.g., window at second 12.0 vs second 12.5) end up in train and test simultaneously. Because these windows share near-identical electrode placement, skin impedance, and body configuration, the model learns this specific person's physiology by heart — achieving inflated test accuracy that collapses immediately on a new user.

**What LOSO does instead:**
Leave-One-Subject-Out completely holds out all windows from one participant (e.g., subject P03) as the test set, trains on the remaining 33, and reports performance on P03. This simulates the realistic deployment scenario: "How does this model perform on a person it has never seen before?"

**Reporting convention for this framework:**
LOSO accuracy is the primary generalization metric reported in all evaluation tables. Within-subject (random split) accuracy is only reported as an upper bound for reference. A model that achieves 90% LOSO accuracy on 34 subjects is a genuinely generalizing model.
