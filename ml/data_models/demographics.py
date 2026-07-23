"""Static, subject-level demographic profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ml.data_models.enums import Sex


@dataclass(frozen=True)
class DemographicData:
    """One subject's demographic profile, sourced from demographic.csv.

    Attributes:
        age: Age in years.
        height_cm: Height in centimeters.
        sex: Self-reported sex.
        dominant_hand: Self-reported dominant hand ("left" or "right").
        execution_order_group: Counterbalancing group (1, 2, or 3) that
            determined the chronological order in which this subject
            performed the 6 fatigue tasks. Does not affect which task is
            which (task codes are fixed regardless of group).
        exercise_type: Free-text self-reported habitual exercise type.
        exercise_frequency_per_week: Self-reported weekly exercise frequency.
        assessment_date: Raw session date string, kept unparsed since the
            source field is inconsistently formatted; date parsing/cleaning
            is a preprocessing concern, not a loading concern.
        assessment_time: Raw session time string, kept unparsed for the
            same reason as assessment_date.
    """

    age: int
    height_cm: float
    sex: Sex
    dominant_hand: str
    execution_order_group: int
    exercise_type: str
    exercise_frequency_per_week: int
    assessment_date: Optional[str] = None
    assessment_time: Optional[str] = None
