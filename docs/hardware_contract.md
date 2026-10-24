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

---

## 4. What the live software actually requires (verified against the dataset and the code)

The wire format above is the transport. Inside the software the live pipeline (`ml/dataset_pipeline/live.py`) consumes
**one-second chunks in the dataset's own units**; `validate_chunk()` checks them and `CONTRACT` holds these values.

| Stream | Shape per 1 s chunk | Units | Dataset column / file | Hard limit used to flag samples |
|---|---|---|---|---|
| EMG | 6 x 1000 (rows in this order: anterior_deltoid, infraspinatus, latissimus_dorsi, pectoralis_major, posterior_deltoid, upper_trapezius) | mV | `EMG(mV)` in `EMG data/<muscle>.csv` | abs value over 10 mV |
| IMU | 36 x 100 (per site forearm, hand, pelvis, shoulder, sternum, upper_arm: acc x,y,z then gyr x,y,z) | m/s^2 and rad/s | `acc_x(m/s^2)` ... `gyr_z(rad/s)` | acc over 160, gyro over 35 |
| PPG | 200 | volts (0 to 3.3) | `PPG(Volt)` in `PPG data/ppg.csv` | outside 0 to 3.3 V |

Corrections to section 2, found by checking the real recordings:

* **EMG range.** Recorded values reach 3.9 mV, so the "+-2.5 mV" range above would flag real data. Use +-5 mV or wider at the ADC.
* **IMU range.** The dataset description does not state the sensors' ranges. In the recordings the accelerometer stays within about 12 m/s^2 (99.9th percentile) and the gyro within 9.5 rad/s, apart from one corrupt spike. The pipeline treats acceleration over 160 m/s^2 and angular rate over 35 rad/s as invalid; these are generous limits chosen by us, not sensor specifications. +-4 g / +-1000 deg/s hardware covers the study movements but not faster ones.
* **PPG units.** The dataset gives PPG in volts; the pipeline filters and counts beats on the signal shape, so raw ADC counts must be scaled to 0 to 3.3 V (or the limit changed) before they enter the pipeline.
* **No mains hum.** No 50 or 60 Hz peak was found in any of the 204 recordings, so the pipeline notches only if one is measured (peak over three times its neighbours).
* **Magnetometers** are not used by any model and need not be streamed.
* Streams must be aligned to the same start time; the pipeline cuts all of them to the shortest.

### Software interfaces

| Piece | Where | Status |
|---|---|---|
| `DataSource` interface | `ml/dataset_pipeline/live.py` | done |
| `FileReplaySource` (recorded, cleaned signals) | same | done, replay equals offline (checked by `verify_replay`) |
| `SyntheticSource` (hardware simulator) | same | done, contract-conforming, can inject dropouts and spikes |
| `SerialSource` (ESP32 over USB-serial / BLE) | same | **placeholder**: implement `read_chunk()` to return a `Chunk` |
| Causal front-end filtering for raw hardware | not written | the replay path uses the offline zero-phase filters; a causal filter changes the features slightly and must be validated on a real device |
