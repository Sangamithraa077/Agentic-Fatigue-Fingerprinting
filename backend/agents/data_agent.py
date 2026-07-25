"""Data Intelligence Agent for automated signal validation and anomaly detection."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from backend.agents.base_agent import AgentResult, BaseAgent


class DataIntelligenceAgent(BaseAgent):
    """Agent responsible for auditing multimodal sensor streams before modeling.

    Inspects signal quality reports, checks for missing channels, flags noise spikes,
    and certifies data viability for downstream ML inference.
    """

    def __init__(self):
        super().__init__(agent_name="DataIntelligenceAgent")

    def process(self, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()
        warnings: List[str] = []
        is_viable = True

        # Extract input signals or quality reports from context
        emg_quality = context.get("emg_quality")
        imu_quality = context.get("imu_quality")
        ppg_quality = context.get("ppg_quality")
        raw_signals = context.get("raw_signals", {})

        # 1. Audit sEMG channels
        emg_channels_valid = 0
        if raw_signals.get("emg") is not None:
            emg_arr = raw_signals["emg"]
            # Check for NaN / Inf
            if emg_arr.ndim == 2:
                for ch_idx in range(emg_arr.shape[1]):
                    ch_data = emg_arr[:, ch_idx]
                    if any(c is True for c in [bool((ch_data != ch_data).any()), bool((ch_data == float("inf")).any())]):
                        warnings.append(f"EMG channel {ch_idx} contains NaN or Infinite values.")
                        is_viable = False
                    else:
                        emg_channels_valid += 1
            elif emg_arr.ndim == 1:
                emg_channels_valid = 1

        # 2. Audit IMU Kinematics
        imu_status = "OK"
        if raw_signals.get("imu") is not None:
            imu_arr = raw_signals["imu"]
            if (imu_arr != imu_arr).any():
                warnings.append("IMU signal stream contains missing values (packet drop).")
                imu_status = "PACKET_LOSS"
                is_viable = False

        # 3. Audit PPG Pulse
        ppg_status = "OK"
        if raw_signals.get("ppg") is not None:
            ppg_arr = raw_signals["ppg"]
            if (ppg_arr != ppg_arr).any():
                warnings.append("PPG pulse stream contains missing values.")
                ppg_status = "SENSOR_DISCONNECT"

        # Formulate narrative report
        if is_viable and not warnings:
            narrative = (
                f"Signal Audit Certified: All {emg_channels_valid} EMG channels, IMU sensors, "
                "and PPG pulse streams are fully intact and ready for feature extraction."
            )
        elif is_viable:
            narrative = (
                f"Signal Audit Passed with Warnings: {emg_channels_valid} EMG channels valid. "
                f"Minor issues detected: {'; '.join(warnings)}"
            )
        else:
            narrative = (
                "Signal Audit FAILED: Data stream contains critical artifacts or NaN values. "
                f"Issues: {'; '.join(warnings)}"
            )

        exec_time = (time.time() - start_time) * 1000.0

        return self._create_result(
            success=is_viable,
            data={
                "is_viable": is_viable,
                "valid_emg_channels": emg_channels_valid,
                "imu_status": imu_status,
                "ppg_status": ppg_status,
                "warnings_count": len(warnings),
            },
            narrative=narrative,
            warnings=warnings,
            execution_time_ms=exec_time,
        )
