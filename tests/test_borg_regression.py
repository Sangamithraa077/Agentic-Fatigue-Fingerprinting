"""Tests for the leave-one-participant-out Ridge Borg estimator.

The central guarantee is reproduction: the packaged model must give the same
out-of-sample predictions, alpha choices, and pooled error that the research
study published in `research/results/`. If these drift, the dashboard would be
showing numbers the study never produced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.services import replay_engine
from ml.models.borg_regression import (
    BorgEstimateTable,
    build_estimate_table,
    condition_key,
    fit_lopo_ridge,
    load_sequence_data,
    participant_key,
)
from ml.models.borg_regression.ridge_lopo import DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = REPO_ROOT / "research"
RESULTS_DIR = RESEARCH_DIR / "results"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packaged_data_is_identical_to_the_research_copy():
    assert _sha(DATA_DIR / "borg_sequences.npz") == _sha(RESEARCH_DIR / "lstm_sequences.npz")
    assert _sha(DATA_DIR / "borg_sequence_metadata.csv") == _sha(RESEARCH_DIR / "lstm_sequence_metadata.csv")


def test_sequence_data_shape_and_coverage():
    data = load_sequence_data()
    assert data.X.shape == (888, 165)  # 3 windows x 55 features, flattened
    assert len(np.unique(data.participants)) == 34
    assert np.isfinite(data.X).all() and np.isfinite(data.y).all()
    assert data.y.min() >= 6 and data.y.max() <= 20


@pytest.mark.parametrize(
    "subject_id, expected",
    [("P01", "subject_1"), ("P09", "subject_9"), ("P34", "subject_34")],
)
def test_participant_key_mapping(subject_id, expected):
    assert participant_key(subject_id) == expected


def test_participant_key_rejects_ids_without_a_number():
    with pytest.raises(ValueError):
        participant_key("athlete")


def test_condition_key_matches_the_study_labels():
    data = load_sequence_data()
    assert condition_key("external_rotation", "30-40") == "30-40_ external rotation"
    assert condition_key("internal_rotation", "50-60") == "50-60_ internal rotation"
    for direction in ("internal_rotation", "external_rotation"):
        for band in ("30-40", "40-50", "50-60"):
            assert condition_key(direction, band) in set(data.conditions)


def test_reproduces_published_nested_lopo_results_for_every_athlete():
    """Predictions, alpha choices, and pooled MAE all match the study's CSVs."""
    data = load_sequence_data()
    published = pd.read_csv(RESULTS_DIR / "nested_lopo_ridge_predictions.csv")
    summary = pd.read_csv(RESULTS_DIR / "nested_lopo_ridge_summary.csv").set_index("Model")

    mine, actual = [], []
    for participant in sorted(set(data.participants)):
        mask = data.participants == participant
        fitted = fit_lopo_ridge(participant, data)
        predictions = fitted.predict(data.X[mask])

        rows = published[published["participant"] == participant]
        assert len(rows) == mask.sum()
        np.testing.assert_allclose(predictions, rows["ridge_prediction"].to_numpy(), atol=1e-3)
        assert fitted.alpha == rows["selected_alpha"].iloc[0]

        mine.append(predictions)
        actual.append(data.y[mask])

    pooled_mae = float(np.mean(np.abs(np.concatenate(mine) - np.concatenate(actual))))
    assert pooled_mae == pytest.approx(summary.loc["Nested Ridge", "MAE"], abs=1e-3)
    # The constant the API reports as the "typical error" must be that same figure.
    assert replay_engine.STUDY_POOLED_LOPO_MAE == pytest.approx(pooled_mae, abs=0.01)


def test_held_out_athlete_is_never_in_the_training_set():
    data = load_sequence_data()
    fitted = fit_lopo_ridge("subject_9", data)
    n_held_out = int((data.participants == "subject_9").sum())
    assert fitted.n_training_participants == 33
    assert fitted.n_training_sequences == len(data.y) - n_held_out


def test_unknown_participant_has_no_model():
    with pytest.raises(KeyError):
        fit_lopo_ridge("subject_999")
    assert build_estimate_table("P99") is None


@pytest.fixture(scope="module")
def table() -> BorgEstimateTable:
    result = build_estimate_table("P09")
    assert result is not None
    return result


def test_estimates_are_clipped_to_the_borg_scale(table):
    assert table.frame["estimate"].between(6.0, 20.0).all()


def test_lookup_is_none_until_the_first_sequence_has_finished(table):
    condition = table.frame["condition"].iloc[0]
    first_end = table.first_estimate_available_s(condition)
    assert table.lookup(condition, first_end - 0.5) is None
    hit = table.lookup(condition, first_end)
    assert hit is not None and hit["sequence_end_s"] == first_end
    assert hit["previous_value"] is None  # no earlier sequence yet


def test_lookup_returns_latest_finished_sequence_and_its_predecessor(table):
    condition = table.frame["condition"].value_counts().idxmax()  # a condition with several sequences
    ends = table.frame[table.frame["condition"] == condition]["sequence_end_s"].to_numpy()
    assert len(ends) >= 3
    hit = table.lookup(condition, ends[2] + 1.0)
    assert hit["sequence_end_s"] == ends[2]
    assert hit["previous_sequence_end_s"] == ends[1]


def test_replay_block_reports_warming_up_instead_of_inventing_a_value(table):
    session = SimpleNamespace(borg_table=table)
    block = replay_engine._borg_regression_block(session, "external_rotation", "30-40", elapsed_seconds=1.0)
    assert block["available"] is False and block["reason"] == "warming_up"
    assert block["available_from_s"] >= 25
    assert "value" not in block


def test_replay_block_without_study_data_is_explicitly_unavailable():
    block = replay_engine._borg_regression_block(
        SimpleNamespace(borg_table=None), "external_rotation", "30-40", elapsed_seconds=100.0
    )
    assert block == {
        "available": False,
        "model": replay_engine.BORG_REGRESSION_MODEL_NAME,
        "typical_error_borg": replay_engine.STUDY_POOLED_LOPO_MAE,
        "trained_on_athletes": None,
        "reason": "no_study_data",
    }


def test_replay_block_available_after_warmup(table):
    condition_rows = table.frame[table.frame["condition"] == "30-40_ external rotation"]
    if condition_rows.empty:
        pytest.skip("P09 has no 30-40 external-rotation sequences")
    elapsed = float(condition_rows["sequence_end_s"].iloc[0]) + 1.0
    block = replay_engine._borg_regression_block(SimpleNamespace(borg_table=table), "external_rotation", "30-40", elapsed)
    assert block["available"] is True
    assert 6.0 <= block["value"] <= 20.0
    assert block["trained_on_athletes"] == 33


def test_feature_baseline_zscores_use_the_athletes_own_statistics():
    session = SimpleNamespace(
        emg_baseline=SimpleNamespace(feature_means={"rms": 1.0, "mav": 5.0}, feature_stds={"rms": 0.5, "mav": 0.0})
    )
    z = replay_engine._feature_baseline_zscores(session, {"rms": 2.0, "mav": 9.0, "unknown": 1.0}, ["rms", "mav", "unknown"])
    assert z == {"rms": pytest.approx(2.0)}  # zero-variance and unknown features are omitted, not guessed


def test_evaluation_metrics_file_matches_the_published_results():
    report = json.loads((REPO_ROOT / "reports" / "evaluation_metrics.json").read_text(encoding="utf-8"))
    published = pd.read_csv(RESULTS_DIR / "nested_lopo_ridge_summary.csv").set_index("Model")
    assert report["loso_metrics"]["ridge_nested"]["mae"] == pytest.approx(published.loc["Nested Ridge", "MAE"], abs=1e-3)
    assert report["loso_metrics"]["median_baseline"]["mae"] == pytest.approx(published.loc["Median baseline", "MAE"], abs=1e-3)
    assert report["n_participants"] == 34 and report["n_sequences"] == 888
