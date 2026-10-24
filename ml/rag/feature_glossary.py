"""Plain-language names for the signal features and internal tags the model reports.

The explanation layer talks to athletes and coaches, so raw feature keys such
as ``slope_sign_changes`` or tags such as ``CALIBRATED_TO_P01`` must never reach
the reader. The dashboard keeps a mirror of the feature table in
``dashboard/src/main.js``; keep the two in step.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# feature key -> (short label, one-line meaning)
FEATURE_GLOSSARY: Dict[str, Tuple[str, str]] = {
    "mav": ("average muscle effort", "how hard the muscle worked on average"),
    "rms": ("muscle activity level", "the overall strength of the muscle signal"),
    "variance": ("effort variability", "how much the effort fluctuated"),
    "waveform_length": ("signal complexity", "how busy the muscle signal is"),
    "zero_crossings": ("signal fluctuation rate", "how quickly the muscle signal changes direction"),
    "slope_sign_changes": ("muscle firing pattern", "how the muscle fibres are firing"),
    "willison_amplitude": ("contraction bursts", "how many strong contractions occurred"),
    "integrated_emg": ("total muscle engagement", "the total muscle activity in this window"),
    "mean_frequency": ("muscle signal speed", "a rhythm that slows down as muscles tire"),
    "median_frequency": ("muscle fibre tiredness", "a rhythm that slows down as muscles tire"),
    "peak_frequency": ("dominant muscle rhythm", "the strongest rhythm in the muscle signal"),
    "spectral_entropy": ("signal spread", "how spread out the muscle signal is"),
}

_TAG_PATTERNS = (
    (re.compile(r"\s*\(?CALIBRATED_TO_[A-Za-z0-9_]+\)?"), ""),
    (re.compile(r"POPULATION_UNADAPTED"), "not adjusted for this athlete"),
)


def feature_label(key: str) -> str:
    """Everyday name for a feature key; unknown keys are humanised, never dropped."""
    entry = FEATURE_GLOSSARY.get(key)
    if entry:
        return entry[0]
    return key.replace("_", " ").lower()


def feature_meaning(key: str) -> Optional[str]:
    entry = FEATURE_GLOSSARY.get(key)
    return entry[1] if entry else None


def humanise_text(text: str) -> str:
    """Swap feature keys and internal tags inside free text for everyday words."""
    out = text
    for pattern, replacement in _TAG_PATTERNS:
        out = pattern.sub(replacement, out)
    for key in sorted(FEATURE_GLOSSARY, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(key)}\b", FEATURE_GLOSSARY[key][0], out)
    return out


# Standard Borg 6-20 verbal anchors; a reading counts as an anchor once it has
# reached it (14.6 is still "somewhat hard", 15 is "hard"), in step with the
# 15 / 18 intervention thresholds used by the Decision Agent.
BORG_ANCHORS = (
    (6, "no effort"), (7, "extremely light"), (9, "very light"), (11, "light"),
    (13, "somewhat hard"), (15, "hard"), (17, "very hard"), (19, "extremely hard"), (20, "maximal"),
)


def effort_word(value: float) -> str:
    word = BORG_ANCHORS[0][1]
    for threshold, label in BORG_ANCHORS:
        if threshold <= value:
            word = label
    return word


# Everyday phrases an athlete types -> the technical vocabulary the knowledge base uses.
QUERY_SYNONYMS = (
    ("tired", "fatigue"), ("tiredness", "fatigue"), ("exhausted", "fatigue exhaustion"),
    ("effort", "exertion RPE"), ("borg", "Borg RPE scale"),
    ("muscle signal", "surface EMG"), ("muscle fibre", "muscle fiber"),
    ("pulse", "PPG heart rate"), ("heart", "PPG autonomic"), ("movement", "IMU kinematic"),
    ("shaky", "tremor jerk IMU"), ("shaking", "tremor jerk IMU"),
    ("usual", "baseline personalization"), ("my own", "baseline personalization"),
    ("reliable", "uncertainty error validation"), ("accurate", "uncertainty error validation"),
)


def expand_query(text: str) -> str:
    """Append knowledge-base vocabulary for every everyday phrase or feature label in ``text``."""
    low = text.lower()
    extras = []
    for phrase, technical in QUERY_SYNONYMS:
        if phrase in low:
            extras.append(technical)
    for key, (label, _) in FEATURE_GLOSSARY.items():
        if label in low or key in low:
            extras.append(key.replace("_", " ") + " EMG")
            if key in ("mean_frequency", "median_frequency"):
                extras.append("spectral shift compression MDF MPF")
    return f"{text} {' '.join(extras)}".strip()
