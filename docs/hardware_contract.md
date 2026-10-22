# Hardware Interface Contract Specification

**Document Version:** 1.0.0  
**Compliance:** Module Split v2 (Module 14.3 — Hardware Readiness)  
**Target Microcontroller:** ESP32 / ESP32-S3 (BLE / USB-UART Serial)

---

## 1. Overview & Data Ingestion Schema

To support seamless integration between public dataset experiments (WSD4FEDSRM) and physical wearable hardware, this contract establishes the exact internal schema that any hardware reader or simulator must provide.

The software core processes all incoming telemetry through a unified `SensorPacket` interface, guaranteeing that downstream signal filtering, windowing, and ML inference are completely decoupled from the physical transmission medium.

---

## 2. Sensor Channel Specifications

| Modality | Physical Sensor | Channels Required | Sampling Rate | Physical Units | Valid Dynamic Range |
|---|---|---|---|---|---|
| **sEMG** | Differential surface electrodes | 6 channels:<br>1. Anterior Deltoid<br>2. Posterior Deltoid<br>3. Infraspinatus<br>4. Upper Trapezius<br>5. Latissimus Dorsi<br>6. Pectoralis Major | **1,000 Hz** (±1.0%) | Millivolts ($\text{mV}$) | $-2.5\text{ mV}$ to $+2.5\text{ mV}$ |
| **IMU** | 6-DOF / 9-DOF MEMS (e.g. MPU6050/ICM-42688) | 6 placements (Hand, Forearm, Upper Arm, Shoulder, Sternum, Pelvis):<br>- Accelerometer: $a_x, a_y, a_z$<br>- Gyroscope: $\omega_x, \omega_y, \omega_z$ | **100 Hz** (±1.0%) | Accel: $\text{m/s}^2$<br>Gyro: $\text{rad/s}$ | Accel: $\pm 39.2\text{ m/s}^2$ ($\pm 4g$)<br>Gyro: $\pm 17.45\text{ rad/s}$ ($\pm 1000^\circ/\text{s}$) |
| **PPG** | Optical reflective pulse (e.g. MAX30102) | 1 channel (Red / IR raw photoplethysmogram) | **200 Hz** (±1.0%) | Raw ADC count / normalized arbitrary units | $0$ to $2^{18}-1$ (18-bit ADC) |

---

## 3. Packet Transmission Wire Format (JSON / Serial Frame)

Every packet received from the ESP32 hardware reader or simulator must adhere to the following JSON specification:

```json
{
  "device_id": "ESP32_FATIGUE_NODE_01",
  "packet_id": 41209,
  "timestamp_us": 1729448920150230,
  "modality": "EMG",
  "sampling_rate_hz": 1000,
  "channels": [
    "Ant_Deltoid",
    "Post_Deltoid",
    "Infraspinatus",
    "Upp_Trapezius",
    "Lat_Dorsi",
    "Pect_Major"
  ],
  "values": [
    [0.124, -0.045, 0.312, 0.088, -0.012, 0.045],
    [0.138, -0.042, 0.320, 0.091, -0.010, 0.048]
  ],
  "quality_flag": "OK"
}
```

### Quality Flag Enumeration:
* `OK`: Normal transmission, valid physiological readings.
* `PACKET_LOSS`: Missing sequence frames detected.
* `CLIPPING`: Signal exceeding ADC rail voltage.
* `ELECTRODE_DETACHED`: High-impedance open circuit detected.
* `BATTERY_LOW`: Hardware voltage dropping below stable reference.
