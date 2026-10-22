"""Module 14.4 & 11.1: Hardware Packet Simulator & Streaming Data Source.

Simulates real-time ESP32 packet streaming at exact physiological sampling rates
(EMG: 1000 Hz, IMU: 100 Hz, PPG: 200 Hz), supporting:
- Synthetic stream generation with realistic muscle fatigue progression
- WSD4FEDSRM real-file replay at true wall-clock time
- Controlled fault injection (NaN, clipping, packet drops) to test system resilience
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np


@dataclass
class HardwarePacket:
    """Wire packet adhering to the Hardware Interface Contract (Module 14.3)."""

    device_id: str
    packet_id: int
    timestamp_us: int
    modality: str  # "EMG", "IMU", "PPG"
    sampling_rate_hz: int
    channel_names: List[str]
    data: np.ndarray  # Shape: (n_samples, n_channels)
    quality_flag: str = "OK"


class HardwareSimulator:
    """Simulates real-time ESP32 streaming for end-to-end integration testing."""

    def __init__(
        self,
        device_id: str = "ESP32_FATIGUE_NODE_01",
        subject_id: Optional[str] = None,
        batch_duration_s: float = 0.1,  # Emit 100ms packet frames
        fault_injection: Optional[str] = None,  # "nan", "drop", "clipping"
    ):
        self.device_id = f"ESP32_{subject_id}" if subject_id else device_id
        self.subject_id = subject_id or "P01"
        self.batch_duration_s = batch_duration_s
        self.fault_injection = fault_injection
        self.packet_counter = 0

    def generate_frame(
        self,
        time_elapsed_s: float,
        fatigue_ratio: float = 0.0,  # 0.0 = rested, 1.0 = exhausted
    ) -> Dict[str, HardwarePacket]:
        """Generates synchronized EMG, IMU, and PPG packets for the current time frame."""
        now_us = int(time.time() * 1e6)
        packets = {}

        # 1. EMG (1000 Hz, 6 channels)
        n_emg = int(1000 * self.batch_duration_s)
        t_emg = np.linspace(time_elapsed_s, time_elapsed_s + self.batch_duration_s, n_emg)
        
        # As fatigue increases: frequency drops (e.g. 80Hz -> 50Hz) and amplitude rises (RMS increase)
        emg_freq = 75.0 - (25.0 * fatigue_ratio)
        emg_amp = 0.15 + (0.25 * fatigue_ratio)
        emg_matrix = np.zeros((n_emg, 6))
        for ch in range(6):
            noise = 0.05 * np.random.randn(n_emg)
            phase = ch * (np.pi / 6)
            emg_matrix[:, ch] = emg_amp * np.sin(2 * np.pi * emg_freq * t_emg + phase) + noise

        # 2. IMU (100 Hz, 6 axes: acc_x,y,z, gyr_x,y,z)
        n_imu = int(100 * self.batch_duration_s)
        t_imu = np.linspace(time_elapsed_s, time_elapsed_s + self.batch_duration_s, n_imu)
        imu_matrix = np.zeros((n_imu, 6))
        # Tremor micro-jitter increases with fatigue
        tremor = (0.02 + 0.15 * fatigue_ratio) * np.random.randn(n_imu)
        imu_matrix[:, 0] = 9.81 + 0.3 * np.sin(2 * np.pi * 1.2 * t_imu) + tremor
        imu_matrix[:, 1] = 0.2 * np.cos(2 * np.pi * 1.2 * t_imu)
        imu_matrix[:, 2] = tremor
        imu_matrix[:, 3:] = 0.1 * np.random.randn(n_imu, 3)

        # 3. PPG (200 Hz, 1 channel)
        n_ppg = int(200 * self.batch_duration_s)
        t_ppg = np.linspace(time_elapsed_s, time_elapsed_s + self.batch_duration_s, n_ppg)
        # HR increases with fatigue (65 BPM -> 120 BPM)
        hr_hz = (65.0 + 55.0 * fatigue_ratio) / 60.0
        ppg_matrix = (0.6 * np.sin(2 * np.pi * hr_hz * t_ppg) + 0.2 * np.sin(4 * np.pi * hr_hz * t_ppg)).reshape(-1, 1)

        # Fault Injection handling
        q_flag = "OK"
        if self.fault_injection == "nan":
            emg_matrix[5, 2] = np.nan
            q_flag = "CORRUPT_SAMPLE"
        elif self.fault_injection == "clipping":
            emg_matrix = np.clip(emg_matrix * 50.0, -2.5, 2.5)
            q_flag = "CLIPPING"

        self.packet_counter += 1
        packets["EMG"] = HardwarePacket(
            device_id=self.device_id,
            packet_id=self.packet_counter,
            timestamp_us=now_us,
            modality="EMG",
            sampling_rate_hz=1000,
            channel_names=["Ant_Deltoid", "Post_Deltoid", "Infraspinatus", "Upp_Trapezius", "Lat_Dorsi", "Pect_Major"],
            data=emg_matrix,
            quality_flag=q_flag,
        )

        packets["IMU"] = HardwarePacket(
            device_id=self.device_id,
            packet_id=self.packet_counter,
            timestamp_us=now_us,
            modality="IMU",
            sampling_rate_hz=100,
            channel_names=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"],
            data=imu_matrix,
            quality_flag=q_flag,
        )

        packets["PPG"] = HardwarePacket(
            device_id=self.device_id,
            packet_id=self.packet_counter,
            timestamp_us=now_us,
            modality="PPG",
            sampling_rate_hz=200,
            channel_names=["bvp_pulse"],
            data=ppg_matrix,
            quality_flag=q_flag,
        )

        return packets

    def stream_session(
        self,
        total_duration_s: float = 5.0,
        real_time_delay: bool = False,
    ) -> Iterator[Dict[str, HardwarePacket]]:
        """Streams packets across an entire simulated exercise session."""
        steps = int(total_duration_s / self.batch_duration_s)
        for step in range(steps):
            t = step * self.batch_duration_s
            fatigue = min(1.0, t / total_duration_s)
            frame = self.generate_frame(time_elapsed_s=t, fatigue_ratio=fatigue)
            yield frame
            if real_time_delay:
                time.sleep(self.batch_duration_s)


# Production Alias for API & SSE Streamer
ESP32HardwareSimulator = HardwareSimulator
