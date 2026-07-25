"""Multi-Agent Pipeline Orchestrator coordinating all 4 AI agents."""

from __future__ import annotations

from typing import Any, Dict

from backend.agents.base_agent import AgentResult
from backend.agents.data_agent import DataIntelligenceAgent
from backend.agents.decision_agent import DecisionAndReportingAgent
from backend.agents.explainability_agent import ExplainabilityAgent
from backend.agents.personalization_agent import PersonalizationAgent


class AgentOrchestrator:
    """Orchestrator managing the multi-agent pipeline workflow.

    Sequential Execution Flow:
    1. DataIntelligenceAgent (Verifies signal viability & quality)
    2. PersonalizationAgent (Calculates subject-specific baseline offset)
    3. ExplainabilityAgent (Translates SHAP attributions into narrative)
    4. DecisionAndReportingAgent (Enforces safety policy & generates Markdown report)
    """

    def __init__(self, rpe_warning_threshold: float = 15.0):
        self.data_agent = DataIntelligenceAgent()
        self.personalization_agent = PersonalizationAgent()
        self.explainability_agent = ExplainabilityAgent()
        self.decision_agent = DecisionAndReportingAgent(rpe_warning_threshold=rpe_warning_threshold)

    def analyze_fatigue(self, context: Dict[str, Any]) -> Dict[str, AgentResult]:
        """Runs the complete 4-agent fatigue intelligence pipeline on the provided context.

        Args:
            context: Input payload containing raw_signals, subject_id, subject_profile,
                predicted_rpe, shap_values, etc.

        Returns:
            Dict[str, AgentResult]: Mapping of agent name to its execution result.
        """
        results: Dict[str, AgentResult] = {}

        # Step 1: Data Intelligence Audit
        data_res = self.data_agent.process(context)
        results["data_agent"] = data_res

        if not data_res.success:
            # If critical signal defect detected, abort early with warning
            context["data_agent_result"] = data_res
            decision_res = self.decision_agent.process(context)
            results["decision_agent"] = decision_res
            return results

        # Step 2: Personalization & LOSO Adjustment
        context["data_agent_result"] = data_res
        personal_res = self.personalization_agent.process(context)
        results["personalization_agent"] = personal_res

        # Step 3: Explainability Analysis
        context["personalization_agent_result"] = personal_res
        explain_res = self.explainability_agent.process(context)
        results["explainability_agent"] = explain_res

        # Step 4: Decision & Reporting Synthesis
        context["explainability_agent_result"] = explain_res
        decision_res = self.decision_agent.process(context)
        results["decision_agent"] = decision_res

        return results
