"""Loads and caches the trained model this dashboard's live replay uses for
prediction and explanation.

Reuses, unmodified:
- The Phase 9 `random_forest` checkpoint trained on EMG/all/binary_fatigue
  (`ml/models/artifacts/classical/emg_all_binary_fatigue/random_forest.joblib`)
  -- the same checkpoint `notebooks/phase9_baseline_models_report.py` and
  `notebooks/phase11_explainability_report.py` already load for evaluation
  and explanation.
- `ml.explainability.shap_explainer.build_shap_explainer`, given the exact
  same P01-P03 background distribution Phase 11 used, so the SHAP
  explainer this dashboard streams from is identical in kind to the one
  already reviewed in that phase's report.

Everything here is built exactly once per server process (guarded by a
lock, since multiple concurrent requests could otherwise race to build it)
and reused for every subject and every replay step -- only per-subject
*data* varies at request time, not the model or explainer.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

from ml.explainability.config import ShapConfig
from ml.explainability.shap_explainer import ShapExplainerBundle, build_shap_explainer
from ml.feature_engineering.feature_dataset import build_feature_datasets
from ml.feature_engineering.feature_pipeline import FeaturePipeline
from ml.models.common.config import DatasetConfig
from ml.models.common.dataset_builder import apply_scaler, build_modeling_dataset
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "models" / "artifacts" / "classical" / "emg_all_binary_fatigue" / "random_forest.joblib"

# Identical to Phase 9/11's EMG baseline configuration -- this is the exact
# dataset the loaded checkpoint was trained on.
EMG_DATASET_CONFIG = DatasetConfig(modality="EMG", feature_subset="all", target_type="binary_fatigue")
BACKGROUND_SUBJECT_IDS = ("P01", "P02", "P03")


@dataclass
class ModelBundle:
    """Everything the replay engine needs to predict and explain one window.

    Attributes:
        model: The fitted `RandomForestClassifier` (Phase 9 checkpoint).
        scaler: The `RobustScaler` fit alongside it -- must be applied to a
            raw feature vector before calling `model.predict_proba`.
        shap_bundle: A `ShapExplainerBundle` built once against a shared
            background distribution.
        feature_names: The 12 EMG feature names, in the exact column order
            `model`/`scaler` expect.
    """

    model: Any
    scaler: RobustScaler
    shap_bundle: ShapExplainerBundle
    feature_names: tuple


_bundle: Optional[ModelBundle] = None
_lock = threading.Lock()


def _build_background_modeling_dataset():
    """Rebuild the exact EMG/all/binary_fatigue dataset for P01-P03 -- the
    same bounded, documented subject subset every prior phase's report used
    -- purely to supply `build_shap_explainer` a representative background
    distribution. No model is trained here; this only re-runs Phases 5-8.5's
    deterministic feature extraction."""
    loader = WSD4FEDSRMLoader()
    condition_catalog = loader.discover_conditions()
    subject_folders = loader.discover_subjects()

    subjects_features = []
    processed_subjects = []
    for subject_id in BACKGROUND_SUBJECT_IDS:
        subject = loader.load_subject(subject_id, subject_folders[subject_id], condition_catalog)
        processed_subject = PreprocessingPipeline().process_subject(subject)
        subject_features = FeaturePipeline().extract_subject(processed_subject)
        processed_subjects.append(processed_subject)
        subjects_features.append(subject_features)

    feature_datasets = build_feature_datasets(subjects_features, processed_subjects)
    return build_modeling_dataset(feature_datasets["EMG"], EMG_DATASET_CONFIG)


def get_model_bundle() -> ModelBundle:
    """Return the process-wide `ModelBundle`, building it on first call.

    Thread-safe: concurrent callers block on the same build rather than
    each triggering their own (the build re-runs Phases 5-8.5 for 3
    subjects, taking on the order of a minute).
    """
    global _bundle
    if _bundle is not None:
        return _bundle

    with _lock:
        if _bundle is not None:  # another thread built it while we waited
            return _bundle

        logger.info("Loading random_forest checkpoint and building the shared SHAP explainer (one-time)...")
        payload = joblib.load(CHECKPOINT_PATH)
        model, scaler = payload["model"], payload["scaler"]

        background_dataset = _build_background_modeling_dataset()
        background_scaled = apply_scaler(scaler, background_dataset.X)
        shap_bundle = build_shap_explainer(model, "random_forest", background_scaled, ShapConfig())

        _bundle = ModelBundle(
            model=model, scaler=scaler, shap_bundle=shap_bundle, feature_names=background_dataset.feature_names,
        )
        logger.info("Model bundle ready (%d background samples).", len(background_dataset.X))
        return _bundle


def predict_proba(x_raw: np.ndarray) -> np.ndarray:
    """Scale one raw feature vector and return the RF's class probabilities."""
    bundle = get_model_bundle()
    x_scaled = apply_scaler(bundle.scaler, x_raw.reshape(1, -1))
    return bundle.model.predict_proba(x_scaled)[0]
