"""Explainability Agent for interpreting SHAP feature attributions and attention weights."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from backend.agents.base_agent import AgentResult, BaseAgent


class ExplainabilityAgent(BaseAgent):
    """Agent responsible for translating mathematical SHAP attributions into clinical narratives.

    Analyzes feature contribution vectors, identifies primary muscle/kinematic fatigue drivers,
    and formats explanations for clinicians and ergonomists.
    """

    def __init__(self):
        super().__init__(agent_name="ExplainabilityAgent")

    def process(self, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        # Extract SHAP data or feature importances from context
        shap_values = context.get("shap_values", {})
        feature_names = context.get("feature_names", [])
        predicted_rpe = context.get("predicted_rpe", 6.0)

        top_drivers: List[Tuple[str, float]] = []

        if isinstance(shap_values, dict):
            # Dict mapping feature name -> SHAP impact value
            sorted_feats = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            top_drivers = [(name, float(val)) for name, val in sorted_feats[:3]]
        elif isinstance(shap_values, list) and feature_names and len(shap_values) == len(feature_names):
            pairs = list(zip(feature_names, shap_values))
            sorted_feats = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
            top_drivers = [(name, float(val)) for name, val in sorted_feats[:3]]

        # Construct natural language explanation narrative
        if top_drivers:
            primary_feat, primary_val = top_drivers[0]
            direction = "increasing" if primary_val > 0 else "reducing"
            
            narrative_lines = [
                f"Prediction Explanation (Predicted Borg RPE: {predicted_rpe:.1f}):",
                f"1. Primary Driver: '{primary_feat}' contributing +{primary_val:.2f} to fatigue score ({direction} exertion).",
            ]
            if len(top_drivers) > 1:
                sec_feat, sec_val = top_drivers[1]
                narrative_lines.append(f"2. Secondary Driver: '{sec_feat}' (+{sec_val:.2f} SHAP impact).")
            if len(top_drivers) > 2:
                tri_feat, tri_val = top_drivers[2]
                narrative_lines.append(f"3. Tertiary Modulator: '{tri_feat}' ({tri_val:+.2f} SHAP impact).")
            
            narrative = "\n".join(narrative_lines)
        else:
            narrative = f"Prediction Explanation: Predicted Borg RPE is {predicted_rpe:.1f}. No specific SHAP attribution vector provided."

        exec_time = (time.time() - start_time) * 1000.0

        return self._create_result(
            success=True,
            data={
                "predicted_rpe": predicted_rpe,
                "top_drivers": top_drivers,
                "driver_count": len(top_drivers),
            },
            narrative=narrative,
            execution_time_ms=exec_time,
        )
