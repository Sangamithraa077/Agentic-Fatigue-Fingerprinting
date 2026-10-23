"""FastAPI REST API Server for Agentic Fatigue Fingerprinting Framework."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

# Add project root directory to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Security  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from fastapi.security import APIKeyHeader  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from backend.agents.data_agent import DataIntelligenceAgent
from backend.agents.orchestrator import AgentOrchestrator
from backend.api.replay_routes import router as replay_router
from backend.services import replay_engine
from ml.hardware.streaming_engine import StreamingWindowProcessor
from ml.rag.evaluator import LLMLayerEvaluator
from ml.rag.hybrid_router import HybridRouter
from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase
from ml.rag.session_summary import build_session_summary_from_pipeline

# ── API Key Security ──────────────────────────────────────────────────────────
# Default dev key. Override via env variable in production:
#   export FATIGUE_API_KEY="your-secret-key"
FATIGUE_API_KEY = os.environ.get("FATIGUE_API_KEY", "dev-fatigue-key-2026")
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> str:
    """Dependency that validates the X-API-Key header."""
    if api_key == FATIGUE_API_KEY:
        return api_key
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    raise HTTPException(status_code=403, detail="Invalid API key.")


# Public dependency alias — use this on protected routes
_AUTH = Depends(_get_api_key)

app = FastAPI(
    title="Agentic Fatigue Fingerprinting API",
    description="REST API serving real-time multimodal fatigue intelligence, SHAP explainability, and multi-agent workflow orchestration.",
    version="2.0.0",
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(replay_router)

# Mount Dashboard Static Files
dashboard_dir = Path(__file__).resolve().parents[2] / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

# Global instances
rag_kb = LocalRAGKnowledgeBase()
hybrid_router = HybridRouter(knowledge_base=rag_kb)
streaming_processor = StreamingWindowProcessor(router=hybrid_router)

# Name reported to the explanation layer for the models that actually produce
# the numbers it is asked to explain.
EXPLAINED_MODEL_NAME = "RandomForest (fatigue probability) + Ridge regression (Borg RPE)"


class AnalysisRequest(BaseModel):
    """Request body for `/api/agents/analyze`.

    `predicted_rpe` and `shap_values` are only used when `subject_id`/`step`
    do not resolve to an already-built replay session (see
    `run_agent_analysis`) -- whenever a session is available, the endpoint
    recomputes both from the real trained models and SHAP explainer itself,
    rather than trusting client-supplied numbers. With neither a session nor
    an explicit `predicted_rpe`, the request is rejected: nothing is invented.
    """

    subject_id: str = Field(default="P01", description="Subject identifier (e.g., P01 to P34)")
    step: Optional[int] = Field(default=None, description="Replay step index this analysis corresponds to")
    threshold: float = Field(default=15.0, description="Borg RPE threshold the Decision and Reporting Agent applies for this request")
    predicted_rpe: Optional[float] = Field(default=None, description="Caller-supplied Borg RPE, used only if no replay session is available")
    shap_values: Optional[Dict[str, float]] = Field(default=None, description="Caller-supplied SHAP attributions, used only if no replay session is available")


@app.get("/api/health")
def health_check():
    """Returns system status and module readiness."""
    return {
        "status": "healthy",
        "framework": "Agentic Fatigue Fingerprinting",
        "dataset": "WSD4FEDSRM",
        "agents_active": 4,
        "active_subjects": len(replay_engine.list_available_subjects()),
    }


@app.post("/api/agents/analyze")
def run_agent_analysis(req: AnalysisRequest):
    """Executes the full 4-agent orchestration workflow.

    When `req.step` is given and the subject's replay session is already
    cached (i.e. the dashboard has an active replay stream open for this
    subject), the analysis context -- signals, predicted fatigue estimate,
    SHAP attributions, subject profile -- is recomputed directly from that
    real session rather than taken from the request body. Without a session
    the caller must supply `predicted_rpe` explicitly (HTTP 409 otherwise),
    and no raw signals are fabricated for the signal audit.
    """
    subject_profile: Dict[str, float] = {}
    predicted_rpe = req.predicted_rpe
    shap_values = req.shap_values or {}
    raw_signals: Dict[str, Any] = {}

    session = replay_engine.get_cached_session(req.subject_id) if req.step is not None else None
    if session is None and predicted_rpe is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"No active replay session for {req.subject_id!r}. Open the athlete in the dashboard "
                "(or pass an explicit predicted_rpe) before requesting an analysis."
            ),
        )

    if session is not None:
        step_data = replay_engine.compute_replay_step(session, req.step)

        prediction = step_data["prediction"]
        predicted_rpe = prediction.get("estimated_rpe", {}).get("value", prediction["estimated_rpe_proxy"])
        shap_values = {f["feature"]: f["shap"] for f in step_data["explainability"]["local_top_features"]}
        profile = step_data["personalization"]["profile"]
        # Only what the athlete's real profile contains; missing fields stay absent
        # (the agent reports them as n/a) instead of being filled with placeholders.
        demographic, mvic = profile["demographic"] or {}, profile["mvic"] or {}
        subject_profile = {
            key: value
            for key, value in {
                "sex": demographic.get("sex"),
                "age": demographic.get("age"),
                "mvic_force_n": mvic.get("er_mean_n"),
            }.items()
            if value is not None
        }

        raw_signals = _signals_from_step(step_data)

    context = {
        "subject_id": req.subject_id,
        "subject_profile": subject_profile,
        "population_rpe": predicted_rpe,
        "predicted_rpe": predicted_rpe,
        "shap_values": shap_values,
        "raw_signals": raw_signals,
    }

    agent_results = AgentOrchestrator(rpe_warning_threshold=req.threshold).analyze_fatigue(context)

    formatted_results = {}
    for agent_name, res in agent_results.items():
        formatted_results[agent_name] = {
            "agent_name": res.agent_name,
            "success": res.success,
            "narrative": res.narrative,
            "data": res.data,
            "warnings": res.warnings,
        }

    return {
        "status": "success",
        "subject_id": req.subject_id,
        "agent_pipeline_results": formatted_results,
    }


class RAGExplainRequest(BaseModel):
    """Request body for `/api/rag/explain`.

    With `step` and an active replay session, every number in the
    explanation (Borg estimate, trend, SHAP drivers, baseline deviations,
    signal-quality flags) is recomputed from that real window. The explicit
    `current_rpe`/`previous_rpe`/`shap_values` fields are only honoured when
    there is no session; with neither, the request is rejected.
    """

    subject_id: str = Field(default="P01")
    step: Optional[int] = Field(default=None, description="Replay step index to ground the explanation in")
    user_query: Optional[str] = Field(default=None)
    current_rpe: Optional[float] = Field(default=None, description="Caller-supplied Borg RPE (no-session use only)")
    previous_rpe: Optional[float] = Field(default=None)
    time_span_seconds: Optional[float] = Field(default=None)
    shap_values: Optional[Dict[str, float]] = Field(default=None)


def _signals_from_step(step_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Rebuild the raw-signal arrays the Data Intelligence Agent audits from a replay step."""
    signals: Dict[str, np.ndarray] = {}
    emg_arrays = [v for v in step_data["signals"]["emg"]["channels"].values() if v]
    if emg_arrays:
        signals["emg"] = np.array(emg_arrays).T
    imu = step_data["signals"]["imu"]
    if "acc_x" in imu:
        signals["imu"] = np.array([imu["acc_x"], imu["acc_y"], imu["acc_z"]]).T
    ppg = step_data["signals"]["ppg"]
    if ppg and ppg.get("values"):
        signals["ppg"] = np.array(ppg["values"]).reshape(-1, 1)
    return signals


def _grounded_rag_inputs(req: RAGExplainRequest) -> Dict[str, Any]:
    """Resolve the real numbers the explanation may cite, or raise HTTP 409."""
    session = replay_engine.get_cached_session(req.subject_id) if req.step is not None else None

    if session is None:
        if req.current_rpe is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"No active replay session for {req.subject_id!r}. Open the athlete in the dashboard "
                    "(or pass an explicit current_rpe) before requesting an explanation."
                ),
            )
        return {
            "current_rpe": req.current_rpe,
            "previous_rpe": req.previous_rpe,
            "time_span_seconds": req.time_span_seconds if req.time_span_seconds is not None else 0.0,
            "shap_values": req.shap_values or {},
            "baseline_deviations": {},
            "quality_warnings": ["CALLER_SUPPLIED_VALUES_NOT_VERIFIED"],
            "grounding": {"source": "caller_supplied"},
        }

    step_data = replay_engine.compute_replay_step(session, req.step)
    prediction = step_data["prediction"]
    borg = prediction.get("borg_regression", {})
    rpe_source = prediction.get("estimated_rpe", {}).get("source", "rf_probability_proxy")
    current_rpe = prediction.get("estimated_rpe", {}).get("value", prediction["estimated_rpe_proxy"])

    flags: List[str] = []
    previous_rpe: Optional[float] = None
    time_span = float(step_data["elapsed_seconds"])
    if borg.get("available") and borg.get("previous_value") is not None:
        previous_rpe = borg["previous_value"]
        time_span = borg["sequence_end_s"] - borg["previous_sequence_end_s"]
    else:
        flags.append("TREND_UNAVAILABLE_INSUFFICIENT_HISTORY")
    if rpe_source != "ridge_lopo":
        flags.append("RPE_IS_PROBABILITY_PROXY_NOT_REGRESSION")

    audit = DataIntelligenceAgent().process({"raw_signals": _signals_from_step(step_data)})
    flags.extend(audit.warnings or ([] if audit.success else ["SIGNAL_AUDIT_FAILED"]))

    return {
        "current_rpe": current_rpe,
        "previous_rpe": previous_rpe,
        "time_span_seconds": time_span,
        "shap_values": {f["feature"]: f["shap"] for f in step_data["explainability"]["local_top_features"]},
        "baseline_deviations": step_data["personalization"].get("feature_baseline_zscores", {}),
        "quality_warnings": flags or None,  # None -> summary reports the (now real) clean audit
        "grounding": {"source": "replay_session", "step": step_data["step"], "rpe_source": rpe_source},
    }


@app.get("/api/rag/sources")
def get_rag_sources():
    """Returns all available source-tagged knowledge chunks for transparency (Module 8 & 12.8)."""
    return {
        "sources": [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "category": c.category,
                "source": c.source,
                "date": c.date,
                "tags": c.tags,
            }
            for c in rag_kb.chunks
        ]
    }


@app.post("/api/rag/explain")
def get_rag_explanation(req: RAGExplainRequest):
    """Executes Selective RAG-grounded explanation with the Number Verifier guardrail (Modules 9 & 10)."""
    inputs = _grounded_rag_inputs(req)
    summary = build_session_summary_from_pipeline(
        person_id=req.subject_id,
        current_rpe=inputs["current_rpe"],
        previous_rpe=inputs["previous_rpe"],
        time_span_seconds=inputs["time_span_seconds"],
        shap_values=inputs["shap_values"],
        baseline_deviations=inputs["baseline_deviations"],
        quality_warnings=inputs["quality_warnings"],
        model_name=EXPLAINED_MODEL_NAME,
    )

    resp = hybrid_router.explain(summary, user_query=req.user_query, force_trigger=True)
    return {
        "explanation": resp.text,
        "is_llm_generated": resp.is_llm_generated,
        "triggered_by": resp.triggered_by,
        "retrieved_sources": resp.retrieved_sources,
        "verification_passed": resp.verification_passed,
        "execution_time_ms": round(resp.execution_time_ms, 2),
        "disclaimer": resp.disclaimer,
        "session_summary": summary.to_dict(),
        "grounding": inputs["grounding"],
        "llm_backend": hybrid_router._llm_backend,
        "kb_backend": rag_kb.retrieval_backend,
    }


# ── SSE Live-Streaming Endpoint (Improvement #6) ─────────────────────────────

async def _sse_event_generator(subject_id: str, max_frames: int = 120) -> AsyncGenerator[str, None]:
    """Yields real-time fatigue metrics as Server-Sent Events from the hardware simulator."""
    from ml.hardware.simulator import HardwareSimulator

    sim = HardwareSimulator(subject_id=subject_id)
    frames_sent = 0

    for frame in sim.stream_session(total_duration_s=60.0, real_time_delay=False):
        if frames_sent >= max_frames:
            break

        # Feed packets for all modalities
        for packet in frame.values():
            streaming_processor.ingest_packet(packet)

        # Update streaming processor window if ready
        streaming_processor.process_window()

        emg_pkt = frame.get("EMG")
        imu_pkt = frame.get("IMU")

        emg_rms = float(np.sqrt(np.mean(emg_pkt.data ** 2))) if emg_pkt is not None else 0.0
        imu_acc = float(np.mean(np.linalg.norm(imu_pkt.data[:, :3], axis=1))) if imu_pkt is not None else 9.81
        ppg_pkt = frame.get("PPG")
        ppg_mean = float(np.mean(ppg_pkt.data)) if ppg_pkt is not None else None

        payload = {
            "subject_id": subject_id,
            # Hardware-simulator stream used to exercise the ESP32 packet contract
            # (docs/hardware_contract.md); not recorded athlete data.
            "source": "hardware_simulator",
            "timestamp_s": round(time.time(), 3),
            "emg_rms_mean": round(emg_rms, 4),
            "imu_acc_magnitude": round(imu_acc, 4),
            "ppg_mean": None if ppg_mean is None else round(ppg_mean, 4),
            "smoothed_fatigue_index": round(streaming_processor.get_smoothed_fatigue_index(), 2),
            "window_count": streaming_processor.window_count,
            "fault_flags": emg_pkt.quality_flag if emg_pkt else "OK",
            "frame": frames_sent,
        }
        yield f"data: {json.dumps(payload)}\n\n"
        frames_sent += 1
        await asyncio.sleep(0.01)

    yield "data: {\"event\": \"stream_end\"}\n\n"


@app.get("/api/signals/live/{subject_id}", dependencies=[_AUTH])
async def live_stream_signals(subject_id: str, max_frames: int = 120):
    """Server-Sent Events (SSE) endpoint for real-time hardware signal streaming.

    Streams EMA-smoothed fatigue indices and raw sensor statistics for the dashboard.
    Requires X-API-Key header.
    """
    return StreamingResponse(
        _sse_event_generator(subject_id, max_frames=max_frames),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Reports & Metrics Endpoint (Improvement #8) ───────────────────────────────

@app.get("/api/reports/metrics")
def get_model_metrics():
    """Returns evaluation metrics measured from this project's own runs.

    `loso_metrics` come from `reports/evaluation_metrics.json`, which is
    generated from the research study's leave-one-participant-out results
    (`research/build_evaluation_metrics.py`). If that file is absent the
    section is returned empty with `source: "unavailable"` -- no placeholder
    benchmark numbers are ever substituted. `rag_metrics` are measured live
    against the loaded knowledge base.
    """
    metrics_file = Path(__file__).resolve().parents[2] / "reports" / "evaluation_metrics.json"
    if metrics_file.exists():
        report = json.loads(metrics_file.read_text(encoding="utf-8"))
    else:
        report = {"source": "unavailable", "loso_metrics": {}}

    report["rag_metrics"] = _measure_rag_metrics()
    report["llm_backend"] = hybrid_router._llm_backend
    return report


_RAG_METRICS_CACHE: Optional[Dict[str, Any]] = None


def _measure_rag_metrics() -> Dict[str, Any]:
    """Retrieval recall measured now, on the loaded knowledge base (cached per process)."""
    global _RAG_METRICS_CACHE
    if _RAG_METRICS_CACHE is None:
        recall_top1, recall_top3 = LLMLayerEvaluator(rag_kb).evaluate_retrieval_quality()
        _RAG_METRICS_CACHE = {
            "retrieval_recall_top1_pct": round(recall_top1, 1),
            "retrieval_recall_top3_pct": round(recall_top3, 1),
            "retrieval_queries_evaluated": LLMLayerEvaluator.RETRIEVAL_TEST_QUERY_COUNT,
            "knowledge_chunk_count": len(rag_kb.chunks),
            "retrieval_backend": rag_kb.retrieval_backend,
        }
    return _RAG_METRICS_CACHE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
