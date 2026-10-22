---
chunk_id: ML-SHAP-01
title: Interpreting SHAP Values for Physiological Feature Attribution
category: ml_methods
source: Lundberg & Lee (2017) NIPS Unified Approach to Interpreting Model Predictions
date: 2026-01-15
tags: [shap, explainability, feature_attribution, local_explanation, global_explanation]
---
# Interpreting SHAP Values for Physiological Feature Attribution

SHAP (SHapley Additive exPlanations) values explain each model prediction as a sum of signed feature contributions grounded in cooperative game theory.

**Reading SHAP values for fatigue predictions:**
* **Positive SHAP value:** This feature pushed the predicted Borg RPE **upward** (towards more fatigue). Example: EMG_MDF_AntDeltoid with SHAP +3.2 means this muscle's frequency decline contributed an estimated +3.2 RPE points to the final prediction.
* **Negative SHAP value:** This feature pulled the predicted RPE **downward** (towards less fatigue or recovery). Example: PPG_HeartRate with SHAP -0.4 means a lower-than-expected heart rate slightly moderated the fatigue estimate.
* **Near-zero SHAP value:** Feature had negligible influence on this particular prediction.

**Key distinction for this framework:**
SHAP values explain *what the model used to make a prediction*. They do not explain *why the physiology is happening*. For physiological explanations, SHAP values are always paired with knowledge retrieval (e.g., the spectral shift chunks), not presented in isolation.
