"""Unit tests for Backend FastAPI endpoints."""

from fastapi.testclient import TestClient

from backend.api.app import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["agents_active"] == 4


def test_get_subjects_endpoint():
    response = client.get("/api/subjects")
    assert response.status_code == 200
    data = response.json()
    assert "subjects" in data
    assert len(data["subjects"]) >= 6


def test_get_subject_detail_endpoint():
    response = client.get("/api/subjects/P03")
    assert response.status_code == 200
    data = response.json()
    assert data["subject_id"] == "P03"
    assert data["sex"] == "Male"

    # Test missing subject
    invalid_resp = client.get("/api/subjects/NONEXISTENT")
    assert invalid_resp.status_code == 404


def test_signal_stream_endpoint():
    response = client.get("/api/signals/stream/P03")
    assert response.status_code == 200
    data = response.json()
    assert data["subject_id"] == "P03"
    assert "emg" in data
    assert "imu" in data
    assert "ppg" in data
    assert len(data["emg"]["values"]) == 1000


def test_agent_analysis_endpoint():
    payload = {
        "subject_id": "P03",
        "predicted_rpe": 16.4,
        "shap_values": {
            "EMG_MDF_AntDeltoid": 3.20,
            "IMU_Jerk_UpperArm": 1.80,
            "PPG_HeartRate": -0.40,
        },
    }
    response = client.post("/api/agents/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    results = data["agent_pipeline_results"]
    assert "data_agent" in results
    assert "personalization_agent" in results
    assert "explainability_agent" in results
    assert "decision_agent" in results
