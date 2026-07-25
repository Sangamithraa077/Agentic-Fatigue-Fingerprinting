"""Decision and Reporting Agent for synthesis, safety policy enforcement, and report generation."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from backend.agents.base_agent import AgentResult, BaseAgent


class DecisionAndReportingAgent(BaseAgent):
    """Agent responsible for safety decision policy enforcement and report synthesis.

    Combines output from Data, Explainability, and Personalization agents, evaluates safety
    threshold limits (e.g. Borg RPE >= 15.0), and formulates final recommendations.
    """

    def __init__(self, rpe_warning_threshold: float = 15.0):
        super().__init__(agent_name="DecisionAndReportingAgent")
        self.rpe_warning_threshold = rpe_warning_threshold

    def process(self, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        data_res = context.get("data_agent_result")
        explain_res = context.get("explainability_agent_result")
        personal_res = context.get("personalization_agent_result")

        # Determine effective RPE score
        final_rpe = 6.0
        if personal_res and personal_res.data.get("personalized_rpe"):
            final_rpe = personal_res.data["personalized_rpe"]
        elif explain_res and explain_res.data.get("predicted_rpe"):
            final_rpe = explain_res.data["predicted_rpe"]

        # Safety evaluation
        safety_alert = final_rpe >= self.rpe_warning_threshold
        recommendation = ""
        action_required = "CONTINUE_MONITORING"

        if final_rpe >= 18.0:
            action_required = "IMMEDIATE_TERMINATION"
            recommendation = (
                f"CRITICAL EXERTION (RPE {final_rpe:.1f} >= 18.0): High risk of muscle injury. "
                "Halt exercise task immediately and initiate active recovery rest period (min 180s)."
            )
        elif final_rpe >= 15.0:
            action_required = "REST_RECOMMENDED"
            recommendation = (
                f"HEAVY FATIGUE WARNING (RPE {final_rpe:.1f} >= 15.0): Significant neuromuscular degradation. "
                "Recommend reducing task load by 20% or inserting a 60s rest window."
            )
        else:
            action_required = "CONTINUE_MONITORING"
            recommendation = (
                f"NORMAL EXERTION (RPE {final_rpe:.1f} < 15.0): Muscle fatigue levels within safe physiological limits. "
                "Continue task protocol as planned."
            )

        # Synthesize Markdown Report
        report_lines = [
            "### 📋 AI Agentic Fatigue Assessment Report",
            f"**Subject ID**: {personal_res.data.get('subject_id', 'N/A') if personal_res else 'N/A'}",
            f"**Final Borg RPE Estimate**: `{final_rpe:.2f} / 20.0`",
            f"**Exertion Status**: `{action_required}`",
            "",
            "#### 🔍 Key Findings & Explanations",
        ]

        if explain_res and explain_res.narrative:
            report_lines.append(explain_res.narrative)
        else:
            report_lines.append("No feature attribution narrative available.")

        report_lines.extend([
            "",
            "#### 🛡️ Clinical Recommendation",
            f"> {recommendation}"
        ])

        formatted_report = "\n".join(report_lines)
        exec_time = (time.time() - start_time) * 1000.0

        return self._create_result(
            success=True,
            data={
                "final_rpe": final_rpe,
                "safety_alert": safety_alert,
                "action_required": action_required,
                "recommendation": recommendation,
                "formatted_report": formatted_report,
            },
            narrative=formatted_report,
            execution_time_ms=exec_time,
        )
