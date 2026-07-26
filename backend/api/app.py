"""FastAPI REST API Server for Agentic Fatigue Fingerprinting Framework."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root directory to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from backend.agents.orchestrator import AgentOrchestrator
from backend.api.replay_routes import router as replay_router
from backend.services import replay_engine

app = FastAPI(
    title="Agentic Fatigue Fingerprinting API",
    description="REST API serving real-time multimodal fatigue intelligence, SHAP explainability, and multi-agent workflow orchestration.",
    version="1.0.0",
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


class AnalysisRequest(BaseModel):
    """Request body for `/api/agents/analyze`.

    `predicted_rpe` and `shap_values` are only used as a fallback when
    `subject_id`/`step` do not resolve to an already-built replay session
    (see `run_agent_analysis`) -- whenever a session is available, the
    endpoint recomputes both from the real trained model and SHAP
    explainer itself, rather than trusting client-supplied numbers.
    """

    subject_id: str = Field(default="P01", description="Subject identifier (e.g., P01 to P34)")
    step: Optional[int] = Field(default=None, description="Replay step index this analysis corresponds to")
    threshold: float = Field(default=15.0, description="Borg RPE proxy threshold the Decision and Reporting Agent applies for this request")
    predicted_rpe: Optional[float] = Field(default=None, description="Fallback Borg RPE proxy, used only if no replay session is available")
    shap_values: Optional[Dict[str, float]] = Field(default=None, description="Fallback SHAP attributions, used only if no replay session is available")


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
    real session rather than taken from the request body.
    """
    subject_profile: Dict[str, float] = {}
    predicted_rpe = req.predicted_rpe or 16.4
    shap_values = req.shap_values or {}
    raw_signals = {
        "emg": np.zeros((1000, 3)),
        "imu": np.zeros((100, 3)),
        "ppg": np.zeros((200, 1)),
    }

    session = replay_engine.get_cached_session(req.subject_id) if req.step is not None else None
    if session is not None:
        step_data = replay_engine.compute_replay_step(session, req.step)

        predicted_rpe = step_data["prediction"]["estimated_rpe_proxy"]
        shap_values = {f["feature"]: f["shap"] for f in step_data["explainability"]["local_top_features"]}
        profile = step_data["personalization"]["profile"]
        subject_profile = {
            "sex": (profile["demographic"] or {}).get("sex", "unknown"),
            "age": (profile["demographic"] or {}).get("age", 0),
            "upperarm_length_cm": 0.0,
            "mvic_force_n": (profile["mvic"] or {}).get("er_mean_n", 0.0),
        }

        emg = step_data["signals"]["emg"]["channels"]
        emg_arrays = [v for v in emg.values() if v]
        if emg_arrays:
            raw_signals["emg"] = np.array(emg_arrays).T
        imu = step_data["signals"]["imu"]
        if "acc_x" in imu:
            raw_signals["imu"] = np.array([imu["acc_x"], imu["acc_y"], imu["acc_z"]]).T
        ppg = step_data["signals"]["ppg"]
        if ppg and ppg.get("values"):
            raw_signals["ppg"] = np.array(ppg["values"]).reshape(-1, 1)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
