"""Module 10.3: Strict Number Verifier & Grounding Guardrail.

Extracts all numerical values from LLM-generated narrative text and verifies that each
number is explicitly grounded in the pipeline's structured SessionSummary or known scale
domain constants (such as Borg 6-20 anchors). 

Rejects or triggers fallback if any ungrounded number is detected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set, Tuple

from ml.rag.session_summary import SessionSummary

# Whitelist of domain scale constants (e.g. Borg boundaries and counting numbers)
ALLOWED_DOMAIN_CONSTANTS: Set[float] = {
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    50.0, 100.0,
}


@dataclass
class NumberCheckResult:
    """Verification outcome of the number grounding audit."""

    is_valid: bool
    extracted_numbers: List[float]
    grounded_numbers: List[float]
    ungrounded_numbers: List[float]
    message: str


class NumberChecker:
    """Audits generated explanations for ungrounded numbers (Module 10.3)."""

    def __init__(self, tolerance: float = 0.15):
        self.tolerance = tolerance
        # Regex capturing floats, integers, percentages, signed numbers
        self.num_pattern = re.compile(r"(?<![A-Za-z_])[-+]?\d*\.?\d+(?:%|s)?(?![A-Za-z_])")

    def extract_numbers_from_text(self, text: str) -> List[float]:
        """Extracts all numerical literals from a text string, excluding citation metadata."""
        # 1. Strip bracketed citations like [PHYS-EMG-01: ... (Source: ...)]
        clean_text = re.sub(r"\[[^\]]*\]", " ", text)
        # 2. Strip bibliographic publication years like (1997) or (2026)
        clean_text = re.sub(r"\b(?:19|20)\d{2}\b", " ", clean_text)
        # 3. Strip markdown header hashes or bullets
        clean_text = re.sub(r"^#+\s*", " ", clean_text)

        raw_matches = self.num_pattern.findall(clean_text)
        cleaned_numbers = []
        for m in raw_matches:
            s = m.rstrip("%s").strip()
            if not s or s in ("-", "+", "."):
                continue
            try:
                cleaned_numbers.append(float(s))
            except ValueError:
                continue
        return cleaned_numbers

    def verify(self, text: str, summary: SessionSummary) -> NumberCheckResult:
        """Verifies that all numbers in `text` exist in `summary` or domain constants."""
        extracted = self.extract_numbers_from_text(text)
        allowable = summary.get_all_allowable_numbers()
        allowable_set = set(allowable).union(ALLOWED_DOMAIN_CONSTANTS)

        grounded: List[float] = []
        ungrounded: List[float] = []

        for num in extracted:
            # Check if num is close to any allowable number
            matched = False
            for target in allowable_set:
                if abs(num - target) <= self.tolerance:
                    matched = True
                    break
            if matched:
                grounded.append(num)
            else:
                ungrounded.append(num)

        is_valid = len(ungrounded) == 0
        if is_valid:
            msg = f"PASSED: All {len(extracted)} numbers in the text are grounded in pipeline outputs."
        else:
            msg = (
                f"FAILED: Found {len(ungrounded)} ungrounded number(s) in explanation: {ungrounded}. "
                "Triggering verifier rejection."
            )

        return NumberCheckResult(
            is_valid=is_valid,
            extracted_numbers=extracted,
            grounded_numbers=grounded,
            ungrounded_numbers=ungrounded,
            message=msg,
        )
