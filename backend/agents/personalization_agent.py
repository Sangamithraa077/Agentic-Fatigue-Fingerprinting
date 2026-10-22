"""Personalization Agent (Improved): uses the data-driven PersonalizationModel
from ml.personalization instead of the previous heuristic formula.

The agent:
1. Queries the subject's feature baseline from context (mean/std per feature).
2. Calls PersonalizationLayer.predict() with a z-score deviation approach.
3. Falls back gracefully to the MVIC-formula if no feature baseline is available.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from backend.agents.base_agent import AgentResult, BaseAgent

try:
    from ml.personalization.personalization_model import PersonalizationLayer
    from ml.personalization.baseline_builder import ModalityBaseline
    from ml.personalization.config import PersonalizationConfig
    _PERSONALIZATION_AVAILABLE = True
except ImportError:
    _PERSONALIZATION_AVAILABLE = False


class PersonalizationAgent(BaseAgent):
    """Agent responsible for subject profile adaptation and individual baseline alignment.

    Uses the data-driven PersonalizationLayer (z-score deviation blend) when a feature
    baseline is available in context. Falls back to the MVIC strength-factor formula
    for zero-baseline (new-subject) situations.
    """

    def __init__(self, personalization_alpha: float = 0.6):
        super().__init__(agent_name="PersonalizationAgent")
        self._alpha = personalization_alpha
        if _PERSONALIZATION_AVAILABLE:
            cfg = PersonalizationConfig(alpha=personalization_alpha)
            self._layer = PersonalizationLayer(config=cfg)
        else:
            self._layer = None

    def process(self, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        subject_id = context.get("subject_id", "P01")
        subject_profile = context.get("subject_profile", {})
        population_rpe = context.get("population_rpe", 6.0)

        sex = subject_profile.get("sex", "Unknown")
        age = subject_profile.get("age", 25)
        arm_length_cm = subject_profile.get("upperarm_length_cm", 34.0)
        mvic_force_n = subject_profile.get("mvic_force_n", 200.0)

        # Feature baseline (mean/std arrays) may be passed in context for data-driven adjustment
        baseline_means: Optional[np.ndarray] = context.get("baseline_feature_means")
        baseline_stds: Optional[np.ndarray] = context.get("baseline_feature_stds")
        current_features: Optional[np.ndarray] = context.get("current_features")

        personalization_method = "mvic_heuristic"
        deviation_score: Optional[float] = None

        if (
            _PERSONALIZATION_AVAILABLE
            and self._layer is not None
            and baseline_means is not None
            and baseline_stds is not None
            and current_features is not None
        ):
            # ── DATA-DRIVEN PATH: z-score deviation blend ──
            try:
                baseline = ModalityBaseline(
                    subject_id=subject_id,
                    means=np.asarray(baseline_means, dtype=float),
                    stds=np.asarray(baseline_stds, dtype=float),
                )
                # Convert population RPE → population probability (6–20 → 0–1)
                population_prob = (population_rpe - 6.0) / 14.0

                pred = self._layer.predict(
                    subject_id=subject_id,
                    global_probability=float(population_prob),
                    features=np.asarray(current_features, dtype=float),
                    baseline=baseline,
                )
                # Convert personalized probability back to Borg RPE scale
                personalized_rpe = max(6.0, min(20.0, pred.personalized_probability * 14.0 + 6.0))
                personalized_offset = personalized_rpe - population_rpe
                deviation_score = pred.baseline_deviation_score
                personalization_method = "data_driven_z_score"

                narrative = (
                    f"Personalisation Applied for Subject {subject_id} ({sex}, {age}y, "
                    f"Arm: {arm_length_cm}cm, MVIC: {mvic_force_n}N):\n"
                    f"Population RPE: {population_rpe:.2f} → Data-Driven Personalised RPE: {personalized_rpe:.2f} "
                    f"(offset {personalized_offset:+.2f}). "
                    f"Baseline deviation z-score: {deviation_score:.2f} "
                    f"(alpha blend = {self._alpha:.2f})."
                )
            except Exception:
                # Graceful fallback
                personalization_method = "mvic_heuristic_fallback"
                personalized_rpe, personalized_offset = self._mvic_heuristic(
                    population_rpe, mvic_force_n
                )
                narrative = self._mvic_narrative(
                    subject_id, sex, age, arm_length_cm, mvic_force_n,
                    population_rpe, personalized_rpe, personalized_offset
                )
        else:
            # ── HEURISTIC PATH: MVIC strength formula ──
            personalized_rpe, personalized_offset = self._mvic_heuristic(population_rpe, mvic_force_n)
            narrative = self._mvic_narrative(
                subject_id, sex, age, arm_length_cm, mvic_force_n,
                population_rpe, personalized_rpe, personalized_offset
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
                "personalization_method": personalization_method,
                "baseline_deviation_score": deviation_score,
            },
            narrative=narrative,
            execution_time_ms=exec_time,
        )

    def _mvic_heuristic(
        self, population_rpe: float, mvic_force_n: float
    ):
        """Original MVIC strength-factor offset formula (zero-baseline fallback)."""
        strength_factor = (mvic_force_n - 200.0) / 200.0
        personalized_offset = -0.4 * strength_factor
        personalized_rpe = max(6.0, min(20.0, population_rpe + personalized_offset))
        return personalized_rpe, personalized_offset

    def _mvic_narrative(
        self, subject_id, sex, age, arm_length_cm, mvic_force_n,
        population_rpe, personalized_rpe, personalized_offset
    ) -> str:
        return (
            f"Personalisation Applied for Subject {subject_id} ({sex}, {age}y, "
            f"Arm: {arm_length_cm}cm, MVIC: {mvic_force_n}N):\n"
            f"Population Baseline RPE ({population_rpe:.2f}) adjusted by {personalized_offset:+.2f} "
            f"offset to Personalized RPE: {personalized_rpe:.2f}."
        )
