"""Module 11.2 & 11.3: Real-Time Streaming Window Processor & Smoothed Fatigue Inference.

Ingests real-time hardware packets into ring buffers, segments 1.0-second sliding windows,
computes smoothed fatigue index (EMA), and connects with the Hybrid RAG Explanation Engine.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ml.hardware.simulator import HardwarePacket
from ml.rag.hybrid_router import ExplanationResponse, HybridRouter
from ml.rag.session_summary import SessionSummary, build_session_summary_from_pipeline


@dataclass
class StreamingInferenceResult:
    """Real-time output from a single processed window."""

    timestamp: float
    raw_predicted_rpe: float
    smoothed_rpe: float
    alert_level: str  # "NORMAL", "WARNING", "CRITICAL"
    features: Dict[str, float]
    explanation: Optional[ExplanationResponse] = None


class StreamingWindowProcessor:
    """Buffers multi-modal streams and executes continuous inference with smoothing."""

    def __init__(
        self,
        subject_id: str = "P03",
        ema_alpha: float = 0.3,  # Exponential smoothing coefficient
        router: Optional[HybridRouter] = None,
    ):
        self.subject_id = subject_id
        self.ema_alpha = ema_alpha
        self.router = router or HybridRouter()

        # Buffers holding 1 second of data
        self.emg_buffer: Deque[np.ndarray] = collections.deque(maxlen=1000)
        self.imu_buffer: Deque[np.ndarray] = collections.deque(maxlen=100)
        self.ppg_buffer: Deque[np.ndarray] = collections.deque(maxlen=200)

        self.last_smoothed_rpe: float = 6.0
        self.previous_smoothed_rpe: Optional[float] = None  # None until a second window exists
        self.window_count: int = 0
        self.session_log: List[StreamingInferenceResult] = []

    def get_smoothed_fatigue_index(self) -> float:
        """Returns the most recent EMA-smoothed RPE fatigue index."""
        return float(round(self.last_smoothed_rpe, 2))

    def ingest_packet(self, packet: HardwarePacket) -> None:
        """Pushes data samples into the respective modality ring buffer."""
        if packet.modality == "EMG":
            for sample in packet.data:
                self.emg_buffer.append(sample)
        elif packet.modality == "IMU":
            for sample in packet.data:
                self.imu_buffer.append(sample)
        elif packet.modality == "PPG":
            for sample in packet.data:
                self.ppg_buffer.append(sample)

    def is_window_ready(self) -> bool:
        """Returns True when all 3 sensor buffers are filled to a full 1-second window."""
        return (
            len(self.emg_buffer) >= 1000
            and len(self.imu_buffer) >= 100
            and len(self.ppg_buffer) >= 200
        )

    def process_window(self, force_explain: bool = False) -> Optional[StreamingInferenceResult]:
        """Processes the current window: extracts features, predicts RPE, applies EMA smoothing."""
        if not self.is_window_ready():
            return None

        self.window_count += 1
        t_now = time.time()

        emg_arr = np.array(self.emg_buffer)
        imu_arr = np.array(self.imu_buffer)
        ppg_arr = np.array(self.ppg_buffer)

        # Feature extraction
        emg_rms = float(np.sqrt(np.mean(emg_arr[:, 0] ** 2)))
        imu_jerk = float(np.mean(np.abs(np.diff(imu_arr[:, 0])))) * 100.0
        ppg_std = float(np.std(ppg_arr))

        # Effort estimate: a fixed linear SURROGATE, not the trained Ridge/Random Forest models
        # (those need a full recorded session). It exists to exercise the sensor contract and is
        # flagged as such in every summary it feeds to the explanation layer.
        emg_term = emg_rms * 25.0
        imu_term = imu_jerk * 5.0
        raw_rpe = float(np.clip(6.0 + emg_term + imu_term, 6.0, 20.0))

        # EMA Smoothing. The first window has no earlier value, so it must not look like a
        # jump from the 6.0 starting point.
        seed = raw_rpe if self.window_count == 1 else self.last_smoothed_rpe
        smoothed_rpe = (self.ema_alpha * raw_rpe) + ((1.0 - self.ema_alpha) * seed)
        self.previous_smoothed_rpe = None if self.window_count == 1 else self.last_smoothed_rpe
        self.last_smoothed_rpe = smoothed_rpe

        # Safety Alert Level
        if smoothed_rpe >= 18.0:
            alert = "CRITICAL"
        elif smoothed_rpe >= 15.0:
            alert = "WARNING"
        else:
            alert = "NORMAL"

        features = {
            "EMG_RMS_AntDeltoid": round(emg_rms, 3),
            "IMU_Jerk_UpperArm": round(imu_jerk, 3),
            "PPG_Pulse_Std": round(ppg_std, 3),
        }

        # Build the SessionSummary the explanation layer is allowed to cite. The drivers are the
        # surrogate formula's own terms (exact for a linear model); nothing is invented, and no
        # athlete baseline exists for a simulated stream, so no baseline deviation is reported.
        summary = build_session_summary_from_pipeline(
            person_id=self.subject_id,
            current_rpe=smoothed_rpe,
            previous_rpe=self.previous_smoothed_rpe,
            time_span_seconds=float(self.window_count),
            shap_values={
                "EMG_RMS_AntDeltoid": round(emg_term, 2),
                "IMU_Jerk_UpperArm": round(imu_term, 2),
            },
            quality_warnings=["SURROGATE_ESTIMATE_NOT_A_TRAINED_MODEL"],
            is_calibrated=False,
            model_name="Linear surrogate (simulator stream, not a trained model)",
        )

        # Execute selective hybrid explanation
        explanation = self.router.explain(summary, force_trigger=force_explain)

        result = StreamingInferenceResult(
            timestamp=t_now,
            raw_predicted_rpe=round(raw_rpe, 2),
            smoothed_rpe=round(smoothed_rpe, 2),
            alert_level=alert,
            features=features,
            explanation=explanation,
        )
        self.session_log.append(result)
        return result
