"""Leave-one-participant-out (LOPO) Ridge regression of Borg RPE.

This is the regression model from the research study in `research/`
(`research/nested_lopo_ridge_comparison.py`), packaged so the backend can use
it. It predicts a Borg RPE (6-20) from 3 consecutive 5-second feature windows
(55 statistical features per window, 165 inputs in total).

Why LOPO: the estimate for athlete X always comes from a model fitted *only*
on the other 33 athletes, with the Ridge penalty chosen by an inner
participant-grouped cross-validation on those 33. The athlete being scored
never contributes to the model that scores them, so the numbers shown in the
dashboard are genuine out-of-sample estimates, not in-sample fits.

Honest accuracy: across all 34 held-out athletes this model reaches a pooled
MAE of about 2.3 Borg points with a correlation near 0.16 -- about the same
as always predicting the training median (MAE about 2.2). It is a research
estimate, not a validated measurement.

The precomputed sequence tensors in `data/` were produced by
`research/feature_extraction.py` and `research/prepare_lstm_sequences.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
SEQUENCES_FILE = DATA_DIR / "borg_sequences.npz"
METADATA_FILE = DATA_DIR / "borg_sequence_metadata.csv"

# Same grid and inner-CV layout as the research study.
ALPHAS: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
INNER_SPLITS = 4

BORG_MIN, BORG_MAX = 6.0, 20.0


@dataclass(frozen=True)
class BorgSequenceData:
    """The research study's sequence dataset, flattened for Ridge.

    Attributes:
        X: `(n_sequences, 165)` features (3 windows x 55 features, flattened).
        y: `(n_sequences,)` recorded Borg RPE at the end of each sequence.
        participants: `(n_sequences,)` participant keys, e.g. `"subject_9"`.
        conditions: `(n_sequences,)` condition keys, e.g. `"30-40_ external rotation"`.
        sequence_end_s: `(n_sequences,)` time (s from condition start) at which
            each sequence's last window ends -- the earliest moment its
            estimate could be known in a live setting.
    """

    X: np.ndarray
    y: np.ndarray
    participants: np.ndarray
    conditions: np.ndarray
    sequence_end_s: np.ndarray


@lru_cache(maxsize=1)
def load_sequence_data() -> BorgSequenceData:
    """Load and validate the packaged sequence dataset (cached)."""
    data = np.load(SEQUENCES_FILE)
    X = data["X"]
    y = data["y"].astype(float)
    metadata = pd.read_csv(METADATA_FILE)

    if X.ndim != 3 or X.shape[1:] != (3, 55):
        raise ValueError(f"Expected sequences shaped (n, 3, 55), got {X.shape}.")
    if not (len(X) == len(y) == len(metadata)):
        raise ValueError("Sequence tensor, labels, and metadata row counts differ.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Sequence data contains non-finite values.")
    if not np.allclose(y, metadata["target_borg"].to_numpy(dtype=float)):
        raise ValueError("Labels do not match the metadata file.")

    return BorgSequenceData(
        X=X.reshape(len(X), -1),
        y=y,
        participants=metadata["participant"].astype(str).to_numpy(),
        conditions=metadata["condition"].astype(str).to_numpy(),
        sequence_end_s=metadata["sequence_end_s"].to_numpy(dtype=float),
    )


def participant_key(subject_id: str) -> str:
    """Map the app's subject id (`"P09"`) to the study's key (`"subject_9"`)."""
    digits = "".join(ch for ch in subject_id if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot derive a participant number from {subject_id!r}.")
    return f"subject_{int(digits)}"


def condition_key(direction: str, intensity_band: str) -> str:
    """Map the app's condition metadata to the study's condition label.

    `("external_rotation", "30-40")` -> `"30-40_ external rotation"`.
    """
    return f"{intensity_band}_ {direction.replace('_', ' ')}"


def _inner_cv_mae(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float) -> float:
    """Participant-grouped inner CV score for one alpha (training data only)."""
    n_splits = min(INNER_SPLITS, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Not enough training participants for inner cross-validation.")

    actual, predicted = [], []
    for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        scaler = StandardScaler()
        model = Ridge(alpha=alpha).fit(scaler.fit_transform(X[train_idx]), y[train_idx])
        actual.extend(y[val_idx])
        predicted.extend(model.predict(scaler.transform(X[val_idx])))
    return float(mean_absolute_error(actual, predicted))


@dataclass(frozen=True)
class LopoRidgeModel:
    """A Ridge model fitted on every participant except `held_out`."""

    held_out: str
    alpha: float
    alpha_cv_mae: Dict[float, float]
    scaler: StandardScaler
    model: Ridge
    n_training_sequences: int
    n_training_participants: int

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X_flat))


def fit_lopo_ridge(held_out: str, data: Optional[BorgSequenceData] = None) -> LopoRidgeModel:
    """Fit the study's nested-LOPO Ridge for one held-out participant.

    The alpha is selected by inner participant-grouped CV on the *training*
    participants only, then the model is refit on all of them.
    """
    data = data or load_sequence_data()
    test_mask = data.participants == held_out
    if not test_mask.any():
        raise KeyError(f"Participant {held_out!r} has no sequences in the study dataset.")

    train_mask = ~test_mask
    X_train, y_train, groups = data.X[train_mask], data.y[train_mask], data.participants[train_mask]

    alpha_cv = {alpha: _inner_cv_mae(X_train, y_train, groups, alpha) for alpha in ALPHAS}
    best_alpha = min(alpha_cv, key=alpha_cv.get)

    scaler = StandardScaler()
    model = Ridge(alpha=best_alpha).fit(scaler.fit_transform(X_train), y_train)
    return LopoRidgeModel(
        held_out=held_out, alpha=float(best_alpha), alpha_cv_mae=alpha_cv, scaler=scaler, model=model,
        n_training_sequences=int(train_mask.sum()), n_training_participants=int(len(np.unique(groups))),
    )


@dataclass(frozen=True)
class BorgEstimateTable:
    """Out-of-sample Borg estimates for every sequence of one athlete.

    Rows are ordered by `(condition, sequence_end_s)`. `lookup` returns the
    latest estimate whose sequence has *finished* by a given elapsed time.
    """

    participant: str
    alpha: float
    n_training_participants: int
    frame: pd.DataFrame  # columns: condition, sequence_end_s, estimate_raw, estimate, recorded_borg
    athlete_mae: float

    def lookup(self, condition: str, elapsed_seconds: float) -> Optional[Dict[str, float]]:
        """Latest finished estimate (and the one before it) for `condition`, or None.

        Returns None while the first 3-window sequence has not completed yet.
        """
        rows = self.frame[self.frame["condition"] == condition]
        done = rows[rows["sequence_end_s"] <= elapsed_seconds]
        if done.empty:
            return None
        latest = done.iloc[-1]
        previous = done.iloc[-2] if len(done) > 1 else None
        return {
            "value": float(latest["estimate"]),
            "raw_value": float(latest["estimate_raw"]),
            "recorded_borg": float(latest["recorded_borg"]),
            "sequence_end_s": float(latest["sequence_end_s"]),
            "previous_value": None if previous is None else float(previous["estimate"]),
            "previous_sequence_end_s": None if previous is None else float(previous["sequence_end_s"]),
        }

    def first_estimate_available_s(self, condition: str) -> Optional[float]:
        rows = self.frame[self.frame["condition"] == condition]
        return None if rows.empty else float(rows["sequence_end_s"].min())


def build_estimate_table(subject_id: str, data: Optional[BorgSequenceData] = None) -> Optional[BorgEstimateTable]:
    """Score every sequence of `subject_id` with a model that never saw them.

    Returns None if the athlete has no sequences in the study dataset.
    """
    data = data or load_sequence_data()
    key = participant_key(subject_id)
    mask = data.participants == key
    if not mask.any():
        logger.warning("No Borg-regression sequences for %s (%s); falling back to probability proxy.", subject_id, key)
        return None

    fitted = fit_lopo_ridge(key, data)
    raw = fitted.predict(data.X[mask])
    frame = pd.DataFrame(
        {
            "condition": data.conditions[mask],
            "sequence_end_s": data.sequence_end_s[mask],
            "estimate_raw": raw,
            "estimate": np.clip(raw, BORG_MIN, BORG_MAX),
            "recorded_borg": data.y[mask],
        }
    ).sort_values(["condition", "sequence_end_s"]).reset_index(drop=True)

    return BorgEstimateTable(
        participant=key, alpha=fitted.alpha, n_training_participants=fitted.n_training_participants,
        frame=frame, athlete_mae=float(np.mean(np.abs(frame["estimate_raw"] - frame["recorded_borg"]))),
    )
