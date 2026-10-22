"""Unit tests for Modules 11 & 14: Hardware Simulator, Ingestion Contract, and Streaming Processor."""

import numpy as np
import pytest

from ml.hardware.simulator import HardwarePacket, HardwareSimulator
from ml.hardware.streaming_engine import StreamingWindowProcessor


def test_hardware_simulator_frame_generation():
    sim = HardwareSimulator(device_id="ESP32_TEST_01", batch_duration_s=0.1)
    frame = sim.generate_frame(time_elapsed_s=1.0, fatigue_ratio=0.5)

    assert "EMG" in frame
    assert "IMU" in frame
    assert "PPG" in frame

    emg_pkt = frame["EMG"]
    assert emg_pkt.sampling_rate_hz == 1000
    assert emg_pkt.data.shape == (100, 6)  # 100ms at 1000Hz = 100 samples
    assert emg_pkt.quality_flag == "OK"

    imu_pkt = frame["IMU"]
    assert imu_pkt.sampling_rate_hz == 100
    assert imu_pkt.data.shape == (10, 6)  # 100ms at 100Hz = 10 samples


def test_hardware_simulator_fault_injection():
    sim_nan = HardwareSimulator(fault_injection="nan")
    frame = sim_nan.generate_frame(time_elapsed_s=0.0)
    assert np.isnan(frame["EMG"].data).any()
    assert frame["EMG"].quality_flag != "OK"


def test_streaming_window_processor_pipeline():
    processor = StreamingWindowProcessor(subject_id="P03", ema_alpha=0.4)
    sim = HardwareSimulator(batch_duration_s=0.1)

    # Ingest 9 packets (0.9s) -> not ready
    for i in range(9):
        packets = sim.generate_frame(time_elapsed_s=i * 0.1)
        for pkt in packets.values():
            processor.ingest_packet(pkt)
    assert not processor.is_window_ready()

    # Ingest 10th packet (1.0s) -> ready to process
    packets = sim.generate_frame(time_elapsed_s=0.9)
    for pkt in packets.values():
        processor.ingest_packet(pkt)
    assert processor.is_window_ready()

    result = processor.process_window(force_explain=True)
    assert result is not None
    assert 6.0 <= result.smoothed_rpe <= 20.0
    assert result.explanation is not None
    assert "EMG_RMS_AntDeltoid" in result.features
