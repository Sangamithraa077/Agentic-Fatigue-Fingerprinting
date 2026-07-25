"""FastAPI REST API Server for Agentic Fatigue Fingerprinting Framework."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root directory to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from backend.agents.orchestrator import AgentOrchestrator

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

# Global Agent Orchestrator instance
orchestrator = AgentOrchestrator()

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
