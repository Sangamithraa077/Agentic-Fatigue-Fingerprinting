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
