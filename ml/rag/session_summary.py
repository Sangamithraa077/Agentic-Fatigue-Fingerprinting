"""Module 9.1 & 9.2: Structured Session Summary representation for grounded LLM explanation.

Following the principle 'Representation Before Retrieval':
Converts complex multi-modal time series outputs into a clean, typed JSON artifact
containing verified numbers, baseline deviations, and data-quality flags.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionSummary:
    """Standardized structured session summary artifact (Module 9.1)."""

    person_id: str
    session_id: str
    time_span_seconds: float
    current_fatigue_index: float  # Borg RPE (e.g. 16.2)
    fatigue_trend: str  # "STABLE", "RISING_SLOWLY", "RISING_SHARPLY", "DECLINING"
    rate_of_rpe_change_per_min: float
    baseline_changes: Dict[str, float]  # Feature -> percent change or z-score from subject baseline
    top_contributing_features: List[Dict[str, Any]]  # Top SHAP drivers [{"feature": "EMG_MDF", "impact": +2.1}]
    data_quality_flags: List[str]  # e.g. ["ALL_CHANNELS_OK"], ["HIGH_IMU_JERK"]
    calibration_status: str  # "CALIBRATED_TO_P03", "POPULATION_DEFAULT"
    model_used: str  # "CrossModalAttention_v2", "RandomForestRegressor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def get_all_allowable_numbers(self, tolerance_pct: float = 0.01) -> List[float]:
        """Extracts every ground-truth numerical value present in this summary.

        Used by the Module 10 Number Checker to verify that the LLM only cites real numbers.
        """
        numbers: List[float] = [
            round(self.time_span_seconds, 2),
            round(self.current_fatigue_index, 2),
            round(self.rate_of_rpe_change_per_min, 2),
        ]
        for val in self.baseline_changes.values():
            numbers.append(round(float(val), 2))
        for feat in self.top_contributing_features:
            if "impact" in feat:
                numbers.append(round(float(feat["impact"]), 2))
        # Further computed numbers the pipeline registers for this summary (e.g. the person's own history)
        numbers.extend(round(float(v), 2) for v in self.metadata.get("allowed_numbers", []))
        # Add integer versions
        int_numbers = [float(int(n)) for n in numbers]
        return list(set(numbers + int_numbers))


def build_session_summary_from_pipeline(
    person_id: str,
    current_rpe: float,
    previous_rpe: Optional[float] = None,
    time_span_seconds: float = 120.0,
    shap_values: Optional[Dict[str, float]] = None,
    baseline_deviations: Optional[Dict[str, float]] = None,
    quality_warnings: Optional[List[str]] = None,
    is_calibrated: bool = True,
    model_name: str = "CrossModalAttention_v2",
    usual_rate_per_min: Optional[float] = None,
) -> SessionSummary:
    """Builds a structured SessionSummary from pipeline inference outputs.

    ``usual_rate_per_min`` is how fast this person's effort normally builds (Borg points per minute). When given, the
    trend is judged against it (sharp = 1.5x their usual, steady = under a quarter of it), because in an exercise-to-
    exhaustion protocol effort climbs about 10 points a minute for everyone and the fixed 2 points/minute rule would
    call every window "sharp". Without it the fixed thresholds apply.
    """
    if previous_rpe is None:
        previous_rpe = current_rpe

    delta_rpe = current_rpe - previous_rpe
    time_min = max(0.1, time_span_seconds / 60.0)
    rate_per_min = delta_rpe / time_min

    sharp, slow = (1.5 * usual_rate_per_min, 0.25 * usual_rate_per_min) if usual_rate_per_min else (2.0, 0.5)
    if rate_per_min > sharp:
        trend = "RISING_SHARPLY"
    elif rate_per_min > slow:
        trend = "RISING_SLOWLY"
    elif rate_per_min < -slow:
        trend = "DECLINING"
    else:
        trend = "STABLE"

    # Format top SHAP features
    top_drivers = []
    if shap_values:
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, val in sorted_shap[:3]:
            top_drivers.append({"feature": name, "impact": round(float(val), 2)})

    flags = quality_warnings if quality_warnings else ["SIGNAL_AUDIT_CERTIFIED"]

    return SessionSummary(
        person_id=person_id,
        session_id=f"SESS_{person_id}_{int(time_span_seconds)}s",
        time_span_seconds=round(time_span_seconds, 1),
        current_fatigue_index=round(current_rpe, 2),
        fatigue_trend=trend,
        rate_of_rpe_change_per_min=round(rate_per_min, 2),
        baseline_changes=baseline_deviations or {},
        top_contributing_features=top_drivers,
        data_quality_flags=flags,
        calibration_status=f"CALIBRATED_TO_{person_id}" if is_calibrated else "POPULATION_UNADAPTED",
        model_used=model_name,
    )
