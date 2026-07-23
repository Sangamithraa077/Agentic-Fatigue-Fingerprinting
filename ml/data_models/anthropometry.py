"""Static, subject-level anthropometric (body) measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class AnthropometricData:
    """One subject's body measurements, merged from 5 source files:
    body_composition.csv, breadth.csv, circumference.csv, depth.csv,
    length.csv.

    Attributes:
        mass_kg: Body mass in kilograms.
        body_fat_percent: Body fat percentage.
        muscle_percent: Skeletal muscle mass percentage.
        visceral_fat: Visceral fat rating.
        bmi: Body mass index.
        breadth_cm: Limb breadth measurements, keyed by the *original*
            source column name (e.g. "upper_arm_mid_b_(cm)"). Keys are kept
            verbatim rather than renamed, since the source column names
            contain typos and inconsistencies (e.g. "ditsal" for "distal")
            that should not be silently reinterpreted at the loading layer.
        circumference_cm: Limb circumference measurements, same key
            convention as breadth_cm.
        depth_cm: Limb depth measurements, same key convention as breadth_cm.
        length_cm: Limb length measurements, same key convention as
            breadth_cm.
    """

    mass_kg: float
    body_fat_percent: float
    muscle_percent: float
    visceral_fat: float
    bmi: float
    breadth_cm: Dict[str, float] = field(default_factory=dict)
    circumference_cm: Dict[str, float] = field(default_factory=dict)
    depth_cm: Dict[str, float] = field(default_factory=dict)
    length_cm: Dict[str, float] = field(default_factory=dict)
