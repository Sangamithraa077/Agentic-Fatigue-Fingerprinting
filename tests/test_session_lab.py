"""Modules 9-12: the Session lab service and API, on small synthetic pipeline tables (no real outputs needed)."""

import time

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import pipeline_routes
from backend.services import pipeline_service as svc
from ml.dataset_pipeline import common as c
from ml.rag.hybrid_router import HybridRouter
from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase

N = 12
TASK = "task1_35i"


@pytest.fixture()
def store(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(1, N + 1) * 10
    rpe = np.clip(np.linspace(7, 20, N).round(), 6, 20)
    key = {"sid": 1, "task": TASK, "win": np.arange(N), "t_end": t}
    feats = pd.DataFrame({**key, "rpe": rpe})
    for k in svc.KEY_TRENDS:
        feats[k] = np.linspace(0, -0.3 if "mdf" in k else 0.4, N)
        feats[k[:-4]] = 1.0
    feats["emg_infraspinatus_mdf"] = 85.0
    feats["hr_bpm"] = 90.0
    live = pd.DataFrame({**key, "rpe": rpe, "generalised": rpe + rng.normal(0, 0.6, N)})
    for k in (3, 4, 5):
        live[f"personalised_K{k}"] = rpe + rng.normal(0, 0.4, N)
    windows = pd.DataFrame({**key, "valid_emg_anterior_deltoid": 1.0, "valid_ppg": 1.0})
    windows.loc[5, "valid_emg_anterior_deltoid"] = 0.4
    quality = pd.DataFrame({"sid": [1], "quality_score": [97.0]})
    inv = pd.DataFrame({"sid": [1], "task": [TASK], "borg_trial_sec": [120.0]})
    part = pd.DataFrame({"sid": [1], "age": [27], "sex": ["female"], "n_problem_flags": [0]})
    contrib = pd.DataFrame([{"sid": 1, "task": TASK, "win": w, "rank": 1, "feature": "emg_infraspinatus_mdf_rel", "impact": 1.5 + w * 0.1}
                            for w in range(N)])
    for name, df in (("m4_features", feats), ("m7_live_calibration_predictions", live), ("m3_windows", windows), ("m3_quality", quality),
                     ("m1_inventory", inv), ("m1_participant_summary", part), ("m8_driver_contributions", contrib)):
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    return svc.PipelineStore(tmp_path)


@pytest.fixture()
def client(store, monkeypatch):
    monkeypatch.setattr(svc, "get_store", lambda: store)
    monkeypatch.setattr(svc, "usual_rate", lambda sid: 10.0)       # this person's usual build-up, points per minute
    kb = LocalRAGKnowledgeBase(use_dense=False)
    app = FastAPI()
    app.include_router(pipeline_routes.router)
    pipeline_routes.set_hybrid_router(HybridRouter(knowledge_base=kb, use_ollama=False, use_anthropic=False))
    return TestClient(app)


def test_session_contents(store):
    s = store.session(1, TASK, 4)
    w = s["windows_data"]
    assert len(w) == N and w[0]["baseline_forming"] and w[3]["calibrating"] and not w[4]["calibrating"]
    assert w[5]["quality_flags"] and "anterior deltoid" in w[5]["quality_flags"][0]
    assert w[2]["drivers"][0]["label"] == "rotator cuff muscle fibre tiredness"
    assert s["summary"]["peak_self_report"] == 20 and s["summary"]["warning_at_s"] is not None
    sm = [x["smoothed_generalised"] for x in w]
    assert sm[0] == w[0]["generalised"] and sm[1] == pytest.approx(0.5 * w[1]["generalised"] + 0.5 * sm[0], abs=0.02)


def test_summary_uses_only_stored_numbers_and_flags_quality(store):
    summary, extra = store.build_summary(1, TASK, 5, 4, "personalised")
    assert summary.metadata["baseline_unit"] == "percent" and summary.metadata["elapsed_seconds"] == 60
    assert summary.calibration_status == "CALIBRATED_TO_P01"
    assert any("unreliable" in f for f in summary.data_quality_flags)
    assert extra["grounding"]["estimate"] == "personalised"
    early, _ = store.build_summary(1, TASK, 1, 4, "personalised")
    assert early.calibration_status == "POPULATION_UNADAPTED" and "BASELINE_STILL_FORMING" in early.data_quality_flags


def test_ask_your_data_is_a_fixed_set_of_lookups(store):
    assert "20 out of 20" in store.ask(1, TASK, "What was the highest effort?")["answer"]
    r = store.ask(1, TASK, "When did effort reach the warning level?")
    assert r["matched"] == "threshold_times" and "seconds" in r["answer"]
    for q in ("Is the personalised model better than the generalised one?", "How good was the data quality?",
              "What changed the most from my start?", "How long was this exercise?"):
        assert store.ask(1, TASK, q)["matched"], q
    nope = store.ask(1, TASK, "DROP TABLE sessions; --")
    assert nope["matched"] is None and nope["supported"]


def test_reports(store):
    md = svc.report_markdown(store, 1, TASK, 4, explanation="Effort is very hard.")
    assert "Session report: P01" in md and "Generated text" in md and "| 120 |" in md
    html_doc = svc.report_html(store, 1, TASK, 4)
    assert html_doc.startswith("<!doctype html>") and "<svg" in html_doc and "polyline" in html_doc


def test_api_flow_with_background_explanation(client):
    assert client.get("/api/pipeline/status").json()["llm_enabled"] is False
    assert client.get("/api/pipeline/participants").json()["participants"][0]["id"] == "P01"
    s = client.get(f"/api/pipeline/session/1/{TASK}").json()
    assert s["levels"] == {"warning": 15.0, "stop": 18.0}
    quiet = client.post("/api/pipeline/explain", json={"sid": 1, "task": TASK, "step": 1})
    assert quiet.json()["triggered"] is False and quiet.json()["status"] == "ml_only"
    loud = client.post("/api/pipeline/explain", json={"sid": 1, "task": TASK, "step": N - 1, "session_concluded": True}).json()
    assert loud["triggered"] and loud["job_id"]
    for _ in range(50):
        job = client.get(f"/api/pipeline/explain/{loud['job_id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done" and job["result"]["generated_by"] == "rule-based template"
    assert job["result"]["answer"] and job["result"]["numbers_used"]


def test_api_errors_and_reports(client):
    assert client.get("/api/pipeline/session/1/task9_99x").status_code == 404
    assert client.get(f"/api/pipeline/session/1/{TASK}?k=9").status_code == 422
    assert client.post("/api/pipeline/explain", json={"sid": 99, "task": TASK, "step": 0}).status_code == 422
    assert client.post("/api/pipeline/ask", json={"sid": 1, "task": TASK, "question": "highest effort?"}).json()["matched"] == "peak_effort"
    html_resp = client.get(f"/api/pipeline/report/1/{TASK}?format=html")
    assert html_resp.status_code == 200 and "Generated text" in html_resp.text
    assert client.get(f"/api/pipeline/report/1/{TASK}?format=pdf").status_code == 422
    assert client.get(f"/api/pipeline/report/1/{TASK}?format=md").headers["content-type"].startswith("text/markdown")


def test_missing_tables_give_a_clear_503(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "get_store", lambda: svc.PipelineStore(tmp_path))
    app = FastAPI()
    app.include_router(pipeline_routes.router)
    r = TestClient(app).get("/api/pipeline/participants")
    assert r.status_code == 503 and "pipeline" in r.json()["detail"]
