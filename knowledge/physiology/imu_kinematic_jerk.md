---
chunk_id: PHYS-IMU-01
title: Kinematic Smoothness, Jerk, and Physiological Tremor from IMU
category: physiology
source: Hogan & Flash (1985) / Balasubramanian et al.
date: 2026-01-15
tags: [imu, jerk, tremor, kinematics, coordination]
---
# Kinematic Smoothness, Jerk, and Physiological Tremor from IMU

In multi-joint movements, neuromuscular fatigue impairs fine motor control and inter-muscular coordination, producing detectable signatures in inertial measurement unit (IMU) data:

1. **Jerk Metric Elevation:** Jerk (the time derivative of linear acceleration, $d\mathbf{a}/dt$) reflects movement smoothness. As prime movers fatigue and stabilizer muscles falter, compensatory jerks and abrupt trajectory corrections elevate the RMS jerk of the limbs.
2. **Physiological Micro-Tremor:** Severe motor unit exhaustion introduces tremor bursts typically centered in the 8–12 Hz band on 3-axis arm gyroscopes and accelerometers.
3. **Tempo & Cycle Variability:** Repetition tempo becomes erratic, cycle duration increases, and cycle-to-cycle kinematic variance widens as volitional exhaustion approaches.
