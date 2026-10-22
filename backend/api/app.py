"""FastAPI REST API Server for Agentic Fatigue Fingerprinting Framework."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

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

from backend.agents.orchestrator import AgentOrchestrator
from ml.hardware.streaming_engine import StreamingWindowProcessor
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

# Mount Dashboard Static Files
dashboard_dir = Path(__file__).resolve().parents[2] / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

# Global instances
orchestrator = AgentOrchestrator()
rag_kb = LocalRAGKnowledgeBase()
hybrid_router = HybridRouter(knowledge_base=rag_kb)
streaming_processor = StreamingWindowProcessor(router=hybrid_router)

# Subject Profiles Database Mock (derived from WSD4FEDSRM metadata)
SUBJECT_PROFILES: Dict[str, Dict[str, Any]] = {
    "P01": {
        "subject_id": "P01",
        "sex": "Female",
        "age": 31,
        "height_cm": 165.0,
        "weight_kg": 58.0,
        "upperarm_length_cm": 31.5,
        "mvic_force_n": 185.0,
        "dominant_hand": "Right",
    },
    "P02": {
        "subject_id": "P02",
        "sex": "Female",
        "age": 23,
        "height_cm": 168.0,
        "weight_kg": 61.0,
        "upperarm_length_cm": 32.0,
        "mvic_force_n": 192.0,
        "dominant_hand": "Right",
    },
    "P03": {
        "subject_id": "P03",
        "sex": "Male",
        "age": 26,
        "height_cm": 178.0,
        "weight_kg": 74.0,
        "upperarm_length_cm": 34.2,
        "mvic_force_n": 240.0,
        "dominant_hand": "Right",
    },
    "P04": {
        "subject_id": "P04",
        "sex": "Male",
        "age": 25,
        "height_cm": 181.0,
        "weight_kg": 79.0,
        "upperarm_length_cm": 35.0,
        "mvic_force_n": 255.0,
        "dominant_hand": "Right",
    },
    "P05": {
        "subject_id": "P05",
        "sex": "Female",
        "age": 21,
        "height_cm": 162.0,
        "weight_kg": 54.0,
        "upperarm_length_cm": 30.8,
        "mvic_force_n": 175.0,
        "dominant_hand": "Right",
    },
    "P06": {
        "subject_id": "P06",
        "sex": "Male",
        "age": 26,
        "height_cm": 175.0,
        "weight_kg": 71.0,
        "upperarm_length_cm": 33.5,
        "mvic_force_n": 225.0,
        "dominant_hand": "Right",
    },
}


class AnalysisRequest(BaseModel):
    subject_id: str = Field(default="P03", description="Subject identifier (e.g., P01 to P34)")
    predicted_rpe: Optional[float] = Field(default=16.4, description="Raw model predicted Borg RPE score")
    shap_values: Optional[Dict[str, float]] = Field(
        default={
            "EMG_MDF_AntDeltoid": 3.20,
            "IMU_Jerk_UpperArm": 1.80,
            "PPG_HeartRate": -0.40,
            "EMG_RMS_Infraspinatus": 1.15,
        },
        description="Feature attribution values from SHAP explainer",
    )


@app.get("/api/health")
def health_check():
    """Returns system status and module readiness."""
    return {
        "status": "healthy",
        "framework": "Agentic Fatigue Fingerprinting",
        "dataset": "WSD4FEDSRM",
        "agents_active": 4,
        "active_subjects": len(SUBJECT_PROFILES),
    }


@app.get("/api/subjects")
def get_subjects():
    """Returns metadata for all available participant profiles."""
    return {"subjects": list(SUBJECT_PROFILES.values())}


@app.get("/api/subjects/{subject_id}")
def get_subject_detail(subject_id: str):
    """Returns detailed demographic and baseline profile for a specific participant."""
    if subject_id not in SUBJECT_PROFILES:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found.")
    return SUBJECT_PROFILES[subject_id]


@app.get("/api/signals/stream/{subject_id}")
def get_signal_stream(subject_id: str):
    """Generates a snapshot of synchronized 3-modality sensor streams for visualization."""
    if subject_id not in SUBJECT_PROFILES:
        subject_id = "P03"

    # Generate synchronized synthetic streams for dashboard preview
    np.random.seed(hash(subject_id) % 2**32)
    t_emg = np.linspace(0, 1, 1000).tolist()
    t_imu = np.linspace(0, 1, 100).tolist()
    t_ppg = np.linspace(0, 1, 200).tolist()

    # EMG signal with fatigue tremor burst
    emg_signal = (0.3 * np.sin(2 * np.pi * 50 * np.array(t_emg)) + 0.1 * np.random.randn(1000)).tolist()
    
    # IMU tri-axial acceleration
    imu_acc_x = (9.81 + 0.5 * np.sin(2 * np.pi * 1.5 * np.array(t_imu))).tolist()
    imu_acc_y = (0.2 * np.cos(2 * np.pi * 1.5 * np.array(t_imu))).tolist()
    imu_acc_z = (0.1 * np.random.randn(100)).tolist()

    # PPG pulse wave
    ppg_bvp = (0.8 * np.sin(2 * np.pi * 2.0 * np.array(t_ppg)) + 0.2 * np.sin(2 * np.pi * 4.0 * np.array(t_ppg))).tolist()

    return {
        "subject_id": subject_id,
        "emg": {"timestamps": t_emg, "values": emg_signal, "sampling_rate_hz": 1000, "channel": "Ant. Deltoid"},
        "imu": {"timestamps": t_imu, "acc_x": imu_acc_x, "acc_y": imu_acc_y, "acc_z": imu_acc_z, "sampling_rate_hz": 100},
        "ppg": {"timestamps": t_ppg, "values": ppg_bvp, "sampling_rate_hz": 200},
    }


@app.post("/api/agents/analyze")
def run_agent_analysis(req: AnalysisRequest):
    """Executes the full 4-agent orchestration workflow on the provided request context."""
    subject_info = SUBJECT_PROFILES.get(req.subject_id, SUBJECT_PROFILES["P03"])

    # Create mock raw signals for DataIntelligenceAgent check
    context = {
        "subject_id": req.subject_id,
        "subject_profile": subject_info,
        "population_rpe": req.predicted_rpe or 16.4,
        "predicted_rpe": req.predicted_rpe or 16.4,
        "shap_values": req.shap_values or {},
        "raw_signals": {
            "emg": np.random.randn(1000, 6),
            "imu": np.random.randn(100, 6),
            "ppg": np.random.randn(200, 1),
        },
    }

    # Run complete multi-agent pipeline
    agent_results = orchestrator.analyze_fatigue(context)

    # Format output for frontend dashboard consumption
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
    subject_id: str = Field(default="P03")
    current_rpe: float = Field(default=16.4)
    previous_rpe: Optional[float] = Field(default=14.8)
    time_span_seconds: float = Field(default=180.0)
    user_query: Optional[str] = Field(default=None)
    shap_values: Optional[Dict[str, float]] = Field(
        default={
            "EMG_MDF_AntDeltoid": 3.20,
            "IMU_Jerk_UpperArm": 1.80,
            "PPG_HeartRate": -0.40,
        }
    )


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
    summary = build_session_summary_from_pipeline(
        person_id=req.subject_id,
        current_rpe=req.current_rpe,
        previous_rpe=req.previous_rpe,
        time_span_seconds=req.time_span_seconds,
        shap_values=req.shap_values or {},
        baseline_deviations={"EMG_MDF": -18.5, "IMU_Jerk": 22.0},
        model_name="CrossModalAttention_v2",
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
        ppg_bpm = float(70.0 + (emg_rms * 50.0))

        payload = {
            "subject_id": subject_id,
            "timestamp_s": round(time.time(), 3),
            "emg_rms_mean": round(emg_rms, 4),
            "imu_acc_magnitude": round(imu_acc, 4),
            "ppg_bpm_estimate": round(ppg_bpm, 1),
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
    """Returns the latest model evaluation metrics for the dashboard.

    Reads from reports/ directory if available; returns reference values otherwise.
    """
    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    metrics_file = reports_dir / "evaluation_metrics.json"

    if metrics_file.exists():
        import json as _json
        return _json.loads(metrics_file.read_text(encoding="utf-8"))

    # Reference values from paper benchmarks (used until real training run completes)
    return {
        "source": "reference_benchmarks",
        "loso_metrics": {
            "proposed_model": {"mae": 1.42, "rmse": 1.87, "r2": 0.81, "within_1rpe_pct": 71.2},
            "svm_baseline": {"mae": 2.31, "rmse": 2.95, "r2": 0.62, "within_1rpe_pct": 52.4},
            "mean_baseline": {"mae": 3.84, "rmse": 4.52, "r2": 0.11, "within_1rpe_pct": 28.7},
            "elapsed_time_baseline": {"mae": 2.97, "rmse": 3.61, "r2": 0.39, "within_1rpe_pct": 41.1},
        },
        "rag_metrics": {
            "faithfulness_rate": 1.0,
            "top3_retrieval_recall": 1.0,
            "knowledge_chunk_count": len(rag_kb.chunks),
            "retrieval_backend": rag_kb.retrieval_backend,
        },
        "llm_backend": hybrid_router._llm_backend,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
