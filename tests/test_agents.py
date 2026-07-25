import sys
from pathlib import Path

# Add project root directory to sys.path for robust package imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pytest

from backend.agents.base_agent import AgentResult
from backend.agents.data_agent import DataIntelligenceAgent
from backend.agents.decision_agent import DecisionAndReportingAgent
from backend.agents.explainability_agent import ExplainabilityAgent
from backend.agents.orchestrator import AgentOrchestrator
from backend.agents.personalization_agent import PersonalizationAgent


def test_data_intelligence_agent_success():
    agent = DataIntelligenceAgent()
    context = {
        "raw_signals": {
            "emg": np.random.randn(1000, 6),
            "imu": np.random.randn(100, 6),
            "ppg": np.random.randn(200, 1),
        }
    }
    result = agent.process(context)
    assert result.success is True
    assert result.data["valid_emg_channels"] == 6
    assert result.data["imu_status"] == "OK"
    assert "Certified" in result.narrative


def test_data_intelligence_agent_nan_flagging():
    agent = DataIntelligenceAgent()
    bad_emg = np.random.randn(1000, 6)
    bad_emg[10, 2] = np.nan

    context = {
        "raw_signals": {
            "emg": bad_emg,
            "imu": np.random.randn(100, 6),
        }
    }
    result = agent.process(context)
    assert result.success is False
    assert len(result.warnings) > 0
    assert "NaN" in result.warnings[0]


def test_explainability_agent():
    agent = ExplainabilityAgent()
    context = {
        "predicted_rpe": 16.4,
        "shap_values": {
            "EMG_MDF_AntDeltoid": 3.20,
            "IMU_Jerk_UpperArm": 1.80,
            "PPG_HeartRate": -0.40,
        },
    }
    result = agent.process(context)
    assert result.success is True
    assert result.data["predicted_rpe"] == 16.4
    assert len(result.data["top_drivers"]) == 3
    assert result.data["top_drivers"][0][0] == "EMG_MDF_AntDeltoid"
    assert "Primary Driver" in result.narrative


def test_personalization_agent():
    agent = PersonalizationAgent()
    context = {
        "subject_id": "P03",
        "population_rpe": 15.0,
        "subject_profile": {
            "sex": "Male",
            "age": 26,
            "upperarm_length_cm": 34.2,
            "mvic_force_n": 240.0,
        },
    }
    result = agent.process(context)
    assert result.success is True
    assert result.data["subject_id"] == "P03"
    # Stronger MVIC force (240N > 200N baseline) gives negative offset (higher endurance)
    assert result.data["personalized_offset"] < 0
    assert result.data["personalized_rpe"] < 15.0


def test_decision_and_reporting_agent_warning():
    agent = DecisionAndReportingAgent(rpe_warning_threshold=15.0)

    # Mock agent outputs
    explain_res = AgentResult(
        agent_name="ExplainabilityAgent",
        success=True,
        narrative="Primary Driver: EMG_MDF_AntDeltoid (+3.2 SHAP impact).",
    )
    personal_res = AgentResult(
        agent_name="PersonalizationAgent",
        success=True,
        data={"subject_id": "P03", "personalized_rpe": 16.4},
    )

    context = {
        "explainability_agent_result": explain_res,
        "personalization_agent_result": personal_res,
    }

    result = agent.process(context)
    assert result.success is True
    assert result.data["safety_alert"] is True
    assert result.data["action_required"] == "REST_RECOMMENDED"
    assert "HEAVY FATIGUE WARNING" in result.data["recommendation"]
    assert "📋 AI Agentic Fatigue Assessment Report" in result.narrative


def test_agent_orchestrator_pipeline():
    orchestrator = AgentOrchestrator(rpe_warning_threshold=15.0)

    context = {
        "subject_id": "P03",
        "raw_signals": {
            "emg": np.random.randn(1000, 6),
            "imu": np.random.randn(100, 6),
            "ppg": np.random.randn(200, 1),
        },
        "subject_profile": {
            "sex": "Male",
            "age": 26,
            "upperarm_length_cm": 34.2,
            "mvic_force_n": 240.0,
        },
        "population_rpe": 16.0,
        "shap_values": {
            "EMG_MDF_AntDeltoid": 3.20,
            "IMU_Jerk_UpperArm": 1.80,
        },
    }

    pipeline_results = orchestrator.analyze_fatigue(context)

    assert "data_agent" in pipeline_results
    assert "personalization_agent" in pipeline_results
    assert "explainability_agent" in pipeline_results
    assert "decision_agent" in pipeline_results

    assert pipeline_results["data_agent"].success is True
    assert pipeline_results["decision_agent"].data["final_rpe"] > 0
    assert "formatted_report" in pipeline_results["decision_agent"].data
