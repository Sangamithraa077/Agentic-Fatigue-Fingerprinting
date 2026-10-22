"""Shared preprocessing abstraction used by every modality-specific
preprocessor (EMG, IMU, PPG).

Defines `ProcessedSignal`, the common result type carrying the processed
values alongside full provenance (which steps were applied, and quality
reports computed both before and after processing), and `SignalPreprocessor`,
the abstract base class each modality preprocessor implements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ml.preprocessing.quality_checks import SignalQualityReport, validate_signal


@dataclass(frozen=True)
class ProcessedSignal:
    """Result of preprocessing a single signal stream.

    Attributes:
        values: The processed signal array. The raw input array is never
            mutated -- this is always a newly allocated array.
        sampling_rate_hz: Sampling rate of `values` (equal to the input
            rate unless the preprocessor's config requested resampling).
        steps_applied: Ordered names of every transform actually applied,
            for provenance/auditability (e.g. `("remove_dc_offset",
            "bandpass_filter(20-450Hz)", "rectify")`).
        input_quality: Quality report computed on the raw input, before any
            processing was applied.
        output_quality: Quality report computed on `values`, after
            processing.
    """

    values: np.ndarray
    sampling_rate_hz: float
    steps_applied: Tuple[str, ...]
    input_quality: SignalQualityReport
    output_quality: SignalQualityReport


class SignalPreprocessor(ABC):
    """Base contract for a single-modality signal preprocessor.

    Subclasses apply a configurable, ordered sequence of transforms to one
    raw signal array and return a `ProcessedSignal` that preserves full
    provenance. Subclasses declare `expected_sampling_rate_hz` so the base
    class can validate the input against the modality's known fixed rate.
    """

    expected_sampling_rate_hz: float = 0.0

    @abstractmethod
    def process(self, signal_values: np.ndarray, sampling_rate_hz: float) -> ProcessedSignal:
        """Apply this preprocessor's configured transforms to one signal.

        Args:
            signal_values: 1D (single-channel) or 2D (n_samples, n_channels)
                raw signal array. Never mutated.
            sampling_rate_hz: The signal's sampling rate in Hz.

        Returns:
            A `ProcessedSignal` with the transformed values and full
            before/after provenance.
        """

    def _validate(self, signal_values: np.ndarray, sampling_rate_hz: float) -> SignalQualityReport:
        """Run the standard quality-check battery against this
        preprocessor's expected sampling rate."""
        return validate_signal(
            signal_values,
            sampling_rate_hz=sampling_rate_hz,
            expected_sampling_rate_hz=self.expected_sampling_rate_hz or None,
        )
