"""One participant: demographic/anthropometric profile, subject-level
session outcomes, and every trial (Condition) they performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ml.data_models.anthropometry import AnthropometricData
from ml.data_models.condition import Condition
from ml.data_models.demographics import DemographicData


@dataclass(frozen=True)
class MVICForceProfile:
    """One subject's maximum voluntary isometric contraction (MVIC)
    calibration force, sourced from MVIC_force_data.csv.

    This holds the *derived calibration values* (single numbers per trial),
    distinct from the raw EMG/IMU sensor recordings captured during the
    MVIC trials themselves, which live inside the corresponding
    MVIC_CALIBRATION Condition objects.

    Attributes:
        ir_trial_1_n: Internal rotation MVIC force, 1st attempt (Newtons).
        ir_trial_2_n: Internal rotation MVIC force, 2nd attempt (Newtons).
        ir_mean_n: Mean of the two internal rotation attempts.
        er_trial_1_n: External rotation MVIC force, 1st attempt (Newtons).
        er_trial_2_n: External rotation MVIC force, 2nd attempt (Newtons).
        er_mean_n: Mean of the two external rotation attempts.
    """

    ir_trial_1_n: float
    ir_trial_2_n: float
    ir_mean_n: float
    er_trial_1_n: float
    er_trial_2_n: float
    er_mean_n: float


@dataclass(frozen=True)
class Subject:
    """One study participant.

    Attributes:
        subject_id: Canonical identifier, e.g. "P01" (normalizes the raw
            "Subject 1" folder name / "subject_1" CSV value).
        demographic: This subject's demographic profile, or None if it
            could not be loaded/parsed.
        anthropometric: This subject's body measurements, or None if it
            could not be loaded/parsed.
        mvic_force: This subject's MVIC calibration force values, or None
            if it could not be loaded/parsed.
        kss_before: Karolinska Sleepiness Scale score before the session.
        kss_after: Karolinska Sleepiness Scale score after the session.
        conditions: All trials this subject performed, keyed by
            condition_id (up to 10: 6 fatigue tasks + 4 MVIC trials).
    """

    subject_id: str
    demographic: Optional[DemographicData]
    anthropometric: Optional[AnthropometricData]
    mvic_force: Optional[MVICForceProfile]
    kss_before: Optional[int] = None
    kss_after: Optional[int] = None
    conditions: Dict[str, Condition] = field(default_factory=dict)
