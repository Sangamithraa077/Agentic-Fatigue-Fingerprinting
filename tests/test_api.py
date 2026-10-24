"""Unit tests for Backend FastAPI endpoints.

The replay stream itself (`GET /api/replay/stream/{subject_id}`) is not
exercised end-to-end here: building one subject's session replays this
project's full Phase 5-8.5 pipeline over their real recordings, taking on
the order of tens of seconds to minutes -- appropriate for a manual/browser
verification pass (see `docs/replay_dashboard.md`), not a fast unit test
run on every change. This suite covers the endpoints that are cheap to
exercise directly (subject discovery, error handling, the agent-analysis
fallback path) plus a fast, isolated test of one real replay step via
`backend.services.replay_engine` directly (bypassing the SSE transport).
"""

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["agents_active"] == 4
    assert data["active_subjects"] == 34


def test_replay_subjects_endpoint_returns_real_discoverable_subjects():
    response = client.get("/api/replay/subjects")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 34
    assert "P01" in data["subjects"]
    assert "P34" in data["subjects"]
    assert data["subjects"] == sorted(data["subjects"])


def test_replay_stream_rejects_unknown_subject():
    response = client.get("/api/replay/stream/NOT_A_REAL_SUBJECT")
    assert response.status_code == 404


def test_agent_analysis_endpoint_fallback_path():
    """Without `step`, the endpoint never touches the replay engine and
    runs the real agent orchestrator over the fallback context -- this is
    the fast path this suite exercises; the real-session path is covered
    by the manual verification pass."""
    payload = {"subject_id": "P03", "predicted_rpe": 16.4, "threshold": 15.0}
    response = client.post("/api/agents/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    results = data["agent_pipeline_results"]
    assert "data_agent" in results
    assert "personalization_agent" in results
    assert "explainability_agent" in results
    assert "decision_agent" in results


def test_agent_analysis_endpoint_uses_cached_replay_session_when_available(monkeypatch):
    """When a subject's replay session is already cached, the endpoint
    should recompute real predicted_rpe/shap_values from it rather than
    using the request body's fallback values."""
    from backend.services import replay_engine

    fake_step_payload = {
        "prediction": {"estimated_rpe_proxy": 18.2},
        "explainability": {"local_top_features": [{"feature": "mean_frequency", "shap": 0.05}]},
        "personalization": {"profile": {"demographic": {"sex": "male", "age": 30}, "mvic": {"er_mean_n": 200.0}}},
        "signals": {
            "emg": {"channels": {"anterior_deltoid": [0.1, 0.2], "infraspinatus": [0.1, 0.2], "latissimus_dorsi": [0.1, 0.2]}},
            "imu": {"acc_x": [1.0], "acc_y": [1.0], "acc_z": [1.0]},
            "ppg": {"values": [0.5]},
        },
    }

    monkeypatch.setattr(replay_engine, "get_cached_session", lambda subject_id: object())
    monkeypatch.setattr(replay_engine, "compute_replay_step", lambda session, step: fake_step_payload)

    response = client.post("/api/agents/analyze", json={"subject_id": "P99", "step": 0, "threshold": 15.0})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


# --- No invented numbers --------------------------------------------------------------------------
# The endpoints below used to fall back to hardcoded RPE/SHAP/baseline values and
# zero-filled "signals". These tests pin the replacement behaviour: real values from
# the replay session, or an explicit error / "not available" -- never a made-up number.


def _fake_step(estimated=13.7, previous=11.2):
    """A minimal replay-step payload with the fields the endpoints read."""
    return {
        "step": 3,
        "elapsed_seconds": 42.0,
        "prediction": {
            "estimated_rpe_proxy": 9.9,
            "estimated_rpe": {"value": estimated, "source": "ridge_lopo"},
            "borg_regression": {
                "available": True, "value": estimated, "previous_value": previous,
                "sequence_end_s": 40.0, "previous_sequence_end_s": 30.0,
            },
        },
        "explainability": {
            "local_top_features": [
                {"feature": "waveform_length", "shap": 0.08},
                {"feature": "mean_frequency", "shap": -0.03},
            ]
        },
        "personalization": {
            "feature_baseline_zscores": {"mean_frequency": -1.4},
            "profile": {"demographic": {"sex": "female", "age": 23}, "mvic": {"er_mean_n": 35.0}},
        },
        "signals": {
            "emg": {"channels": {"anterior_deltoid": [0.1, 0.2], "infraspinatus": [0.1, 0.2], "latissimus_dorsi": [0.1, 0.2]}},
            "imu": {"acc_x": [1.0, 1.1], "acc_y": [1.0, 1.1], "acc_z": [9.8, 9.7]},
            "ppg": {"values": [0.5, 0.6]},
        },
    }


def test_agent_analysis_without_session_or_explicit_rpe_is_rejected():
    response = client.post("/api/agents/analyze", json={"subject_id": "P01"})
    assert response.status_code == 409
    assert "No active replay session" in response.json()["detail"]


def test_agent_analysis_with_explicit_rpe_does_not_invent_signals():
    response = client.post("/api/agents/analyze", json={"subject_id": "P03", "predicted_rpe": 16.4})
    assert response.status_code == 200
    data_agent = response.json()["agent_pipeline_results"]["data_agent"]
    assert data_agent["data"]["imu_status"] == "NOT_PROVIDED"
    assert data_agent["data"]["valid_emg_channels"] == 0
    assert "certified" not in data_agent["narrative"].lower()


def test_agent_analysis_prefers_the_ridge_estimate_over_the_probability_proxy(monkeypatch):
    from backend.services import replay_engine

    monkeypatch.setattr(replay_engine, "get_cached_session", lambda subject_id: object())
    monkeypatch.setattr(replay_engine, "compute_replay_step", lambda session, step: _fake_step(estimated=16.0))

    response = client.post("/api/agents/analyze", json={"subject_id": "P99", "step": 3, "threshold": 15.0})
    decision = response.json()["agent_pipeline_results"]["personalization_agent"]["data"]
    assert decision["population_rpe"] == 16.0  # the Ridge value, not the 9.9 proxy


def test_rag_explain_without_session_or_explicit_rpe_is_rejected():
    response = client.post("/api/rag/explain", json={"subject_id": "P01", "step": 3})
    assert response.status_code == 409
    assert "No active replay session" in response.json()["detail"]


def test_rag_explain_is_grounded_in_the_replay_window(monkeypatch):
    from backend.services import replay_engine

    monkeypatch.setattr(replay_engine, "get_cached_session", lambda subject_id: object())
    monkeypatch.setattr(replay_engine, "compute_replay_step", lambda session, step: _fake_step())

    response = client.post("/api/rag/explain", json={"subject_id": "P77", "step": 3})
    assert response.status_code == 200
    body = response.json()
    summary = body["session_summary"]

    assert body["grounding"] == {"source": "replay_session", "step": 3, "rpe_source": "ridge_lopo"}
    assert summary["current_fatigue_index"] == 13.7
    assert summary["fatigue_trend"] == "RISING_SHARPLY"  # 11.2 -> 13.7 over 10 s
    assert summary["baseline_changes"] == {"mean_frequency": -1.4}  # the athlete's real z-score, not a constant
    assert summary["top_contributing_features"][0]["feature"] == "waveform_length"
    assert "Ridge" in summary["model_used"] and "CrossModalAttention" not in summary["model_used"]
    assert summary["data_quality_flags"] == ["SIGNAL_AUDIT_CERTIFIED"]  # an audit really ran on the window's signals


def test_rag_explain_flags_when_trend_history_or_regression_is_missing(monkeypatch):
    from backend.services import replay_engine

    step = _fake_step()
    step["prediction"]["estimated_rpe"] = {"value": 9.9, "source": "rf_probability_proxy"}
    step["prediction"]["borg_regression"] = {"available": False, "reason": "warming_up"}
    monkeypatch.setattr(replay_engine, "get_cached_session", lambda subject_id: object())
    monkeypatch.setattr(replay_engine, "compute_replay_step", lambda session, step_index: step)

    flags = client.post("/api/rag/explain", json={"subject_id": "P77", "step": 3}).json()["session_summary"]["data_quality_flags"]
    assert "TREND_UNAVAILABLE_INSUFFICIENT_HISTORY" in flags
    assert "RPE_IS_PROBABILITY_PROXY_NOT_REGRESSION" in flags


def test_rag_explain_with_explicit_values_has_no_invented_baseline():
    response = client.post("/api/rag/explain", json={"subject_id": "P03", "current_rpe": 15.2})
    assert response.status_code == 200
    body = response.json()
    assert body["grounding"] == {"source": "caller_supplied"}
    assert body["session_summary"]["baseline_changes"] == {}
    assert "CALLER_SUPPLIED_VALUES_NOT_VERIFIED" in body["session_summary"]["data_quality_flags"]


def test_rag_explain_reports_provenance_and_accepts_history():
    body = client.post("/api/rag/explain", json={
        "subject_id": "P03", "current_rpe": 15.2, "user_query": "What should I do now?",
        "history": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
    }).json()
    assert body["source"] in ("llm", "template")
    assert body["answer"] and "does not provide medical diagnosis" not in body["answer"]
    assert isinstance(body["follow_ups"], list) and isinstance(body["source_details"], list)
    bad = client.post("/api/rag/explain", json={"subject_id": "P03", "current_rpe": 15.2, "history": [{"role": "system", "content": "x"}]})
    assert bad.status_code == 422


def test_reports_metrics_serves_measured_values_only():
    data = client.get("/api/reports/metrics").json()
    assert data.get("source") != "reference_benchmarks"
    assert data["loso_metrics"]["ridge_nested"]["mae"] == pytest.approx(2.3139, abs=1e-3)
    assert data["loso_metrics"]["median_baseline"]["mae"] == pytest.approx(2.1791, abs=1e-3)
    # Retrieval recall is measured on the loaded knowledge base; a faithfulness figure is not
    # measurable here, so it must not be reported at all.
    assert 0.0 <= data["rag_metrics"]["retrieval_recall_top3_pct"] <= 100.0
    assert "faithfulness_rate" not in data["rag_metrics"]


def test_reports_metrics_without_a_results_file_returns_empty_not_placeholders(monkeypatch, tmp_path):
    import backend.api.app as app_module

    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "backend" / "api" / "app.py"))
    data = client.get("/api/reports/metrics").json()
    assert data["source"] == "unavailable"
    assert data["loso_metrics"] == {}


def test_hardware_simulator_stream_is_labelled_and_has_no_fabricated_heart_rate():
    from backend.api.app import FATIGUE_API_KEY

    response = client.get("/api/signals/live/P01?max_frames=2", headers={"X-API-Key": FATIGUE_API_KEY})
    assert response.status_code == 200
    assert "ppg_bpm_estimate" not in response.text
    assert '"source": "hardware_simulator"' in response.text
