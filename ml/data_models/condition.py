"""One experimental trial: the unit bundling synchronized multimodal sensor
recordings and (if applicable) fatigue labels for one subject performing one
task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ml.data_models.enums import ConditionType, Direction, EMGMuscle, IMULocation, IntensityBand
from ml.data_models.labels import Labels
from ml.data_models.sensors import EMGData, IMUData, PPGData


@dataclass(frozen=True)
class Condition:
    """One subject's single trial (either a fatigue task or an MVIC
    calibration trial).

    Attributes:
        condition_id: Canonical identifier, e.g. "P01_external_rotation_30-40"
            or "P01_mvic_internal_rotation_trial1".
        subject_id: Back-reference to the owning subject.
        condition_type: FATIGUE_TASK or MVIC_CALIBRATION.
        direction: Internal or external rotation.
        intensity_band: Target MVIC load band; set only for FATIGUE_TASK.
        trial_number: 1 or 2; set only for MVIC_CALIBRATION.
        task_order_code: Raw Borg task code (e.g. "task4_35e"); set only for
            FATIGUE_TASK, used to cross-reference borg_data.csv.
        emg: EMG recordings keyed by muscle (up to 6 entries, present for
            all condition types).
        imu: IMU recordings keyed by location (up to 6 entries, present for
            all condition types).
        ppg: PPG recording, or None (expected for MVIC_CALIBRATION trials).
        labels: Borg RPE labels, or None (expected for MVIC_CALIBRATION
            trials, which were not scored on the Borg scale).
    """

    condition_id: str
    subject_id: str
    condition_type: ConditionType
    direction: Direction
    intensity_band: Optional[IntensityBand]
    trial_number: Optional[int]
    task_order_code: Optional[str]
    emg: Dict[EMGMuscle, EMGData] = field(default_factory=dict)
    imu: Dict[IMULocation, IMUData] = field(default_factory=dict)
    ppg: Optional[PPGData] = None
    labels: Optional[Labels] = None
