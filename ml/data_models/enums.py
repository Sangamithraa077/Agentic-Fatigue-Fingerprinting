"""Closed-vocabulary enumerations used across the WSD4FEDSRM data model.

Every value is defined to match the *normalized* form of the real dataset's
folder/file/column names (lowercase, spaces replaced with underscores), so
the loader can convert a discovered on-disk name into an enum member with a
single, simple normalization step instead of a hardcoded name-to-name table.
"""

from __future__ import annotations

from enum import Enum


class Direction(str, Enum):
    """Shoulder rotation direction performed during a trial."""

    INTERNAL_ROTATION = "internal_rotation"
    EXTERNAL_ROTATION = "external_rotation"


class IntensityBand(str, Enum):
    """Target load band, expressed as a percentage range of MVIC force.

    Only applicable to fatigue-task conditions; MVIC calibration conditions
    have no intensity band (see ConditionType).
    """

    BAND_30_40 = "30-40"
    BAND_40_50 = "40-50"
    BAND_50_60 = "50-60"


class ConditionType(str, Enum):
    """The two kinds of experimental trial present in WSD4FEDSRM."""

    FATIGUE_TASK = "fatigue_task"
    MVIC_CALIBRATION = "mvic_calibration"


class EMGMuscle(str, Enum):
    """The 6 muscles instrumented with surface EMG (dominant side)."""

    ANTERIOR_DELTOID = "anterior_deltoid"
    POSTERIOR_DELTOID = "posterior_deltoid"
    INFRASPINATUS = "infraspinatus"
    UPPER_TRAPEZIUS = "upper_trapezius"
    LATISSIMUS_DORSI = "latissimus_dorsi"
    PECTORALIS_MAJOR = "pectoralis_major"


class IMULocation(str, Enum):
    """The 6 body locations instrumented with an IMU (dominant side)."""

    HAND = "hand"
    FOREARM = "forearm"
    UPPER_ARM = "upper_arm"
    SHOULDER = "shoulder"
    STERNUM = "sternum"
    PELVIS = "pelvis"


class Sex(str, Enum):
    """Participant sex, as self-reported in demographic.csv."""

    MALE = "male"
    FEMALE = "female"
