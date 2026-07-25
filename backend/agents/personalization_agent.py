"""Personalization Agent for subject profile adaptation and baseline offset calculation."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from backend.agents.base_agent import AgentResult, BaseAgent


class PersonalizationAgent(BaseAgent):
    """Agent responsible for subject profile adaptation and individual baseline alignment.

    Queries subject anthropometric measurements (upper arm length, MVIC force baselines),
    applies Leave-One-Subject-Out (LOSO) domain offsets, and contextualizes predictions.
    """

    def __init__(self):
        super().__init__(agent_name="PersonalizationAgent")

    def process(self, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        subject_id = context.get("subject_id", "P01")
        subject_profile = context.get("subject_profile", {})
        population_rpe = context.get("population_rpe", 6.0)

        # Extract anthropometric / demographic parameters
        sex = subject_profile.get("sex", "Unknown")
        age = subject_profile.get("age", 25)
        arm_length_cm = subject_profile.get("upperarm_length_cm", 34.0)
        mvic_force_n = subject_profile.get("mvic_force_n", 200.0)

        # Calculate personalized adjustment offset based on MVIC baseline strength
        # Higher baseline strength -> slightly higher endurance capability
        strength_factor = (mvic_force_n - 200.0) / 200.0
        personalized_offset = -0.4 * strength_factor

        personalized_rpe = max(6.0, min(20.0, population_rpe + personalized_offset))

        narrative = (
            f"Personalization Applied for Subject {subject_id} ({sex}, {age}y, Arm: {arm_length_cm}cm, "
            f"MVIC: {mvic_force_n}N):\n"
            f"Population Baseline RPE ({population_rpe:.2f}) adjusted by {personalized_offset:+.2f} "
            f"offset to Personalized RPE: {personalized_rpe:.2f}."
        )

        exec_time = (time.time() - start_time) * 1000.0

        return self._create_result(
            success=True,
            data={
                "subject_id": subject_id,
                "population_rpe": population_rpe,
                "personalized_offset": personalized_offset,
                "personalized_rpe": personalized_rpe,
                "mvic_force_n": mvic_force_n,
                "arm_length_cm": arm_length_cm,
            },
            narrative=narrative,
            execution_time_ms=exec_time,
        )
