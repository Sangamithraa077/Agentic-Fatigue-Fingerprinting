"""Fatigue ground-truth labels for one fatigue-task condition trial."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class Labels:
    """Borg RPE fatigue trajectory for one fatigue-task trial.

    Only populated for FATIGUE_TASK conditions; MVIC_CALIBRATION conditions
    have no Borg RPE data collected (verified against the source dataset:
    borg_data.csv contains exactly 34 x 6 = 204 rows, one per subject per
    fatigue task, no MVIC entries).

    Attributes:
        task_order_code: Raw source code (e.g. "task4_35e") identifying this
            trial in borg_data.csv, kept for traceability.
        rpe_before: Borg RPE (6-20) reported immediately before the trial.
        rpe_timeseries: Sparse mapping of elapsed seconds -> RPE value,
            recorded every 10 seconds. Sparse because a trial legitimately
            ends (reaches exhaustion) before the full 250s window.
        rpe_at_end_of_trial: Terminal RPE value (protocol target is 20).
        trial_duration_seconds: Total trial duration until exhaustion.
    """

    task_order_code: str
    rpe_before: Optional[int]
    rpe_timeseries: Dict[int, int] = field(default_factory=dict)
    rpe_at_end_of_trial: Optional[int] = None
    trial_duration_seconds: Optional[float] = None
