"""Real-data replay engine: turns one WSD4FEDSRM subject's recordings into
a sequence of "live" windows -- the dashboard's stand-in for a physical
EMG/IMU/PPG sensor feed, built entirely from already-recorded data.

No synthetic signal is generated anywhere in this module. Every array,
prediction, explanation, and personalization value returned by
`compute_replay_step` is produced by re-running this project's existing
pipelines (`ml.preprocessing`, `ml.feature_engineering`,
`ml.models.common`, `ml.explainability`, `ml.personalization`) over one
subject's real recordings -- nothing here re-implements feature
extraction, modeling, explanation, or personalization logic; it only
orchestrates and slices already-computed results.

## Design note for a future live sensor

`SubjectSession` is deliberately the seam a real-time sensor feed would
replace: everything downstream of it (`compute_replay_step`) only depends
on having, for the *current* window, a raw signal slice per modality and a
feature vector -- it does not care whether that window came from replaying
a recording (this module) or from a live acquisition buffer. Swapping the
source means implementing a different session/window provider with the
same shape, not changing the prediction/explanation/personalization code.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.services import model_service
from ml.data_models.enums import EMGMuscle, IMULocation
from ml.explainability.config import FeatureRankingConfig, ShapConfig
from ml.explainability.global_explainer import GlobalExplanation, explain_global
from ml.explainability.local_explainer import explain_prediction
from ml.feature_engineering.feature_dataset import build_feature_datasets
from ml.feature_engineering.feature_pipeline import FeaturePipeline
from ml.models.borg_regression import BorgEstimateTable, build_estimate_table, condition_key
from ml.models.common.dataset_builder import ModelingDataset, apply_scaler, build_modeling_dataset, derive_target
from ml.personalization.baseline_builder import ModalityBaseline, build_modality_baseline
from ml.personalization.config import PersonalizationConfig
from ml.personalization.personalization_model import predict_personalized
from ml.personalization.subject_profiles import SubjectProfile, build_subject_profile
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

# EMG/IMU windows are 1.0s with 50% overlap (Phase 8 default), so
# consecutive window_index values are 0.5s apart -- used only to compute a
# real elapsed-time offset for slicing the (much coarser-windowed) PPG
# signal, not to re-derive any feature.
EMG_IMU_WINDOW_STEP_SECONDS = 0.5
PPG_DISPLAY_SECONDS = 4.0

DISPLAY_EMG_MUSCLES = (EMGMuscle.ANTERIOR_DELTOID, EMGMuscle.INFRASPINATUS, EMGMuscle.LATISSIMUS_DORSI)
DISPLAY_IMU_LOCATION = IMULocation.UPPER_ARM

BORG_REGRESSION_MODEL_NAME = "Ridge regression (trained on the other 33 athletes)"
# Pooled out-of-sample MAE of that model across all 34 held-out athletes in the
# research study (reproduced by tests/test_borg_regression.py). The honest
# "typical error" to show next to any single estimate.
STUDY_POOLED_LOPO_MAE = 2.31

# Features whose deviation from the athlete's own baseline is reported on every
# window: RMS (how hard the muscle is working) and median frequency (which
# drifts down as a muscle tires).
ALWAYS_REPORTED_ZSCORE_FEATURES = ("rms", "median_frequency")

PERSONALIZATION_CONFIG = PersonalizationConfig()
RANKING_CONFIG = FeatureRankingConfig(top_k=10)
SHAP_CONFIG = ShapConfig(max_explain_samples=200)


@dataclass
class SubjectSession:
    """Everything computed once for one subject, reused by every replay step."""

    subject_id: str
    subject: Any  # ml.data_models.subject.Subject (raw)
    processed: Any  # ml.preprocessing.preprocessing_results.ProcessedSubject
    features: Any  # ml.feature_engineering.feature_vector.SubjectFeatures (has window sample bounds)
    modeling_dataset: ModelingDataset
    aligned_metadata: pd.DataFrame  # row-aligned with modeling_dataset.X
    replay_row_indices: np.ndarray  # into modeling_dataset.X / aligned_metadata, in replay order
    profile: SubjectProfile
    emg_baseline: ModalityBaseline
    global_explanation: GlobalExplanation
    # Out-of-sample Borg RPE estimates (Ridge fitted on the other 33 athletes);
    # None if this athlete has no sequences in the research study dataset.
    borg_table: Optional[BorgEstimateTable] = None


_SESSION_CACHE: Dict[str, SubjectSession] = {}
_CACHE_LOCK = threading.Lock()
_loader: Optional[WSD4FEDSRMLoader] = None


def _get_loader() -> WSD4FEDSRMLoader:
    global _loader
    if _loader is None:
        _loader = WSD4FEDSRMLoader()
    return _loader


def list_available_subjects() -> List[str]:
    """Every subject_id actually discoverable under the configured dataset root."""
    return sorted(_get_loader().discover_subjects().keys())


def _build_session(subject_id: str) -> SubjectSession:
    loader = _get_loader()
    condition_catalog = loader.discover_conditions()
    subject_folders = loader.discover_subjects()
    if subject_id not in subject_folders:
        raise ValueError(f"Unknown subject_id: {subject_id!r}")

    logger.info("Building replay session for subject '%s' (Phases 5-8.5)...", subject_id)
    subject = loader.load_subject(subject_id, subject_folders[subject_id], condition_catalog)
    processed = PreprocessingPipeline().process_subject(subject)
    features = FeaturePipeline().extract_subject(processed)
    feature_datasets = build_feature_datasets([features], [processed])

    modeling_dataset = build_modeling_dataset(feature_datasets["EMG"], model_service.EMG_DATASET_CONFIG)

    y_full, valid_mask, _ = derive_target(feature_datasets["EMG"].metadata, model_service.EMG_DATASET_CONFIG)
    aligned_metadata = feature_datasets["EMG"].metadata.loc[valid_mask].reset_index(drop=True)

    replay_mask = (aligned_metadata["source_sensor"] == EMGMuscle.ANTERIOR_DELTOID.value) & (
        aligned_metadata["condition_type"] == "fatigue_task"
    )
    replay_row_indices = aligned_metadata[replay_mask].sort_values(["condition_id", "window_index"]).index.to_numpy()
    if len(replay_row_indices) == 0:
        raise ValueError(f"Subject '{subject_id}' has no replayable (anterior deltoid, fatigue-task) windows.")

    profile = build_subject_profile(subject, feature_datasets)
    emg_baseline = build_modality_baseline(feature_datasets["EMG"], subject_id)

    bundle = model_service.get_model_bundle()
    X_scaled = apply_scaler(bundle.scaler, modeling_dataset.X)
    global_explanation = explain_global(bundle.shap_bundle, X_scaled, modeling_dataset.feature_names, SHAP_CONFIG, RANKING_CONFIG)

    borg_table = build_estimate_table(subject_id)

    logger.info("Session ready for '%s': %d replay steps.", subject_id, len(replay_row_indices))
    return SubjectSession(
        subject_id=subject_id, subject=subject, processed=processed, features=features,
        modeling_dataset=modeling_dataset, aligned_metadata=aligned_metadata, replay_row_indices=replay_row_indices,
        profile=profile, emg_baseline=emg_baseline, global_explanation=global_explanation,
        borg_table=borg_table,
    )


def get_or_build_session(subject_id: str) -> SubjectSession:
    """Return the cached `SubjectSession` for `subject_id`, building it on first request."""
    if subject_id in _SESSION_CACHE:
        return _SESSION_CACHE[subject_id]
    with _CACHE_LOCK:
        if subject_id not in _SESSION_CACHE:
            _SESSION_CACHE[subject_id] = _build_session(subject_id)
    return _SESSION_CACHE[subject_id]


def get_cached_session(subject_id: str) -> Optional[SubjectSession]:
    """Return `subject_id`'s `SubjectSession` if already built, else `None`
    -- never triggers a build. Lets callers (e.g. the agent-analysis
    endpoint) opportunistically reuse an active replay session's real data
    without paying for a fresh session build on a request that isn't the
    replay stream itself."""
    return _SESSION_CACHE.get(subject_id)


def _signal_slice(processed_condition, features_condition, muscle_or_location, is_imu: bool, window_index: int):
    """Slice the real, already-preprocessed signal for one modality/window,
    using the exact sample bounds `FeaturePipeline` recorded for that window
    -- no new windowing logic, just indexing into results already computed."""
    if not is_imu:
        channel_features = features_condition.emg.get(muscle_or_location)
        processed_channel = processed_condition.emg.get(muscle_or_location)
        if channel_features is None or processed_channel is None or window_index >= len(channel_features.windows):
            return None
        windowed = channel_features.windows[window_index]
        return processed_channel.signal.values[windowed.window_start_sample : windowed.window_end_sample].tolist()

    location_features = features_condition.imu.get(muscle_or_location)
    processed_location = processed_condition.imu.get(muscle_or_location)
    if location_features is None or processed_location is None or window_index >= len(location_features.windows):
        return None
    windowed = location_features.windows[window_index]
    acc = processed_location.accelerometer.values[windowed.window_start_sample : windowed.window_end_sample]
    return {"acc_x": acc[:, 0].tolist(), "acc_y": acc[:, 1].tolist(), "acc_z": acc[:, 2].tolist()}


def _ppg_slice(processed_condition, elapsed_seconds: float) -> Optional[Dict[str, Any]]:
    if processed_condition.ppg is None:
        return None
    signal = processed_condition.ppg.signal
    rate = signal.sampling_rate_hz
    end_sample = min(int(elapsed_seconds * rate) + 1, len(signal.values))
    start_sample = max(0, end_sample - int(PPG_DISPLAY_SECONDS * rate))
    return {"sampling_rate_hz": rate, "values": signal.values[start_sample:end_sample].tolist()}


def _borg_regression_block(
    session: SubjectSession, direction: str, intensity_band: str, elapsed_seconds: float
) -> Dict[str, Any]:
    """Out-of-sample Ridge Borg estimate for this moment of the replay, or an
    explicit "not available yet" block -- never a made-up number.

    The study's sequences are 3 consecutive 5 s windows, so the first
    estimate for a condition exists only once ~30 s of that condition have
    been replayed.
    """
    block: Dict[str, Any] = {
        "available": False,
        "model": BORG_REGRESSION_MODEL_NAME,
        "typical_error_borg": STUDY_POOLED_LOPO_MAE,
        "trained_on_athletes": None,
    }
    table = session.borg_table
    if table is None:
        block["reason"] = "no_study_data"
        return block

    block["trained_on_athletes"] = table.n_training_participants
    block["alpha"] = table.alpha
    condition = condition_key(direction, intensity_band)
    hit = table.lookup(condition, elapsed_seconds)
    if hit is None:
        block["reason"] = "warming_up"
        block["available_from_s"] = table.first_estimate_available_s(condition)
        return block

    block.update({"available": True, **hit})
    return block


def _feature_baseline_zscores(session: SubjectSession, raw_by_feature: Dict[str, float], features) -> Dict[str, float]:
    """How far each listed feature sits from this athlete's own mean, in their own
    standard deviations -- from the already-built `ModalityBaseline`."""
    means, stds = session.emg_baseline.feature_means, session.emg_baseline.feature_stds
    zscores: Dict[str, float] = {}
    for name in features:
        std = stds.get(name, 0.0)
        if std > 0 and name in means and name in raw_by_feature:
            zscores[name] = float((raw_by_feature[name] - means[name]) / std)
    return zscores


def compute_replay_step(session: SubjectSession, step: int) -> Dict[str, Any]:
    """Compute one real, fully-explained replay window -- the payload streamed to the dashboard."""
    bundle = model_service.get_model_bundle()
    n_steps = len(session.replay_row_indices)
    row = int(session.replay_row_indices[step % n_steps])
    meta = session.aligned_metadata.loc[row]

    x_raw = session.modeling_dataset.X[row]
    x_scaled = apply_scaler(bundle.scaler, x_raw.reshape(1, -1))[0]
    proba = bundle.model.predict_proba(x_scaled.reshape(1, -1))[0]
    predicted_class = int(np.argmax(proba))
    probability_fatigued = float(proba[1])

    local = explain_prediction(bundle.shap_bundle, bundle.model, x_scaled, session.modeling_dataset.feature_names)
    raw_by_feature = dict(zip(session.modeling_dataset.feature_names, x_raw.tolist()))
    personalized = predict_personalized(
        session.subject_id, probability_fatigued, x_raw, session.modeling_dataset.feature_names,
        session.emg_baseline, PERSONALIZATION_CONFIG,
    )

    condition_id = meta["condition_id"]
    window_index = int(meta["window_index"])
    processed_condition = session.processed.conditions[condition_id]
    features_condition = session.features.conditions[condition_id]
    elapsed_seconds = window_index * EMG_IMU_WINDOW_STEP_SECONDS

    emg_signals = {
        muscle.value: _signal_slice(processed_condition, features_condition, muscle, False, window_index)
        for muscle in DISPLAY_EMG_MUSCLES
    }
    imu_signal = _signal_slice(processed_condition, features_condition, DISPLAY_IMU_LOCATION, True, window_index)
    ppg_signal = _ppg_slice(processed_condition, elapsed_seconds)

    demographic = session.profile.demographic
    anthropometric = session.profile.anthropometric
    mvic = session.subject.mvic_force
    fatigue_history = session.profile.fatigue_history

    rpe_nearest = meta["rpe_nearest"]

    rpe_proxy = round(6.0 + probability_fatigued * 14.0, 1)
    borg_regression = _borg_regression_block(session, meta["direction"], meta["intensity_band"], elapsed_seconds)
    estimated_rpe = (
        {"value": round(borg_regression["value"], 1), "source": "ridge_lopo"}
        if borg_regression["available"]
        else {"value": rpe_proxy, "source": "rf_probability_proxy"}
    )

    return {
        "step": step,
        "total_steps": n_steps,
        "subject_id": session.subject_id,
        "condition_id": condition_id,
        "direction": meta["direction"],
        "intensity_band": meta["intensity_band"],
        "elapsed_seconds": elapsed_seconds,
        "signals": {
            "emg": {"sampling_rate_hz": 1000, "channels": emg_signals},
            "imu": {"sampling_rate_hz": 100, "location": DISPLAY_IMU_LOCATION.value, **(imu_signal or {})},
            "ppg": ppg_signal,
        },
        "ground_truth": {"rpe_nearest": None if pd.isna(rpe_nearest) else float(rpe_nearest)},
        "prediction": {
            "predicted_class": predicted_class,
            "predicted_label": "fatigued" if predicted_class == 1 else "not_fatigued",
            "probability_fatigued": probability_fatigued,
            "confidence": float(proba[predicted_class]),
            # Linear proxy onto the Borg 6-20 scale, derived solely from the
            # real classifier's own probability -- not a separate regression
            # model. Documented in docs/replay_dashboard.md.
            "estimated_rpe_proxy": rpe_proxy,
            # The Borg RPE figure the rest of the system uses: the Ridge
            # regression's out-of-sample estimate once it exists, otherwise
            # the probability proxy above (clearly tagged by `source`).
            "estimated_rpe": estimated_rpe,
            "borg_regression": borg_regression,
        },
        "explainability": {
            "local_top_features": [
                {
                    "feature": c.feature_name, "raw_value": raw_by_feature[c.feature_name],
                    "scaled_value": c.feature_value, "shap": c.shap_value,
                }
                for c in local.top_contributions
            ],
            "global_top_features": [
                {"feature": f.feature_name, "importance": f.importance} for f in session.global_explanation.feature_ranking[:8]
            ],
        },
        "personalization": {
            "generic_probability": personalized.global_probability,
            "personalized_probability": personalized.personalized_probability,
            "baseline_deviation_score": personalized.baseline_deviation_score,
            "feature_baseline_zscores": _feature_baseline_zscores(
                session, raw_by_feature,
                # the plain-language "muscle effort" / "tiredness sign" tiles always need these two
                list(dict.fromkeys([*ALWAYS_REPORTED_ZSCORE_FEATURES, *(c.feature_name for c in local.top_contributions)])),
            ),
            "profile": {
                "demographic": None
                if demographic is None
                else {
                    "age": demographic.age, "height_cm": demographic.height_cm,
                    "sex": demographic.sex.value, "dominant_hand": demographic.dominant_hand,
                },
                "anthropometric": None
                if anthropometric is None
                else {
                    "mass_kg": anthropometric.mass_kg, "bmi": anthropometric.bmi,
                    "body_fat_percent": anthropometric.body_fat_percent, "muscle_percent": anthropometric.muscle_percent,
                },
                "mvic": None if mvic is None else {"ir_mean_n": mvic.ir_mean_n, "er_mean_n": mvic.er_mean_n},
                "fatigue_history": {
                    "n_trials": fatigue_history.n_trials, "mean_rpe": fatigue_history.mean_rpe,
                    "max_rpe": fatigue_history.max_rpe, "min_rpe": fatigue_history.min_rpe,
                },
            },
        },
    }
