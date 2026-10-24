# Module 1 – Data dictionary and participant problems (WSD4FEDSRM)

Source: `description.txt` / `README.txt`, checked against the files themselves by `src/m1_audit.py`.
Load any recording with one command: `py src/common.py <participant 1-34> <task>` (task = `task1_35i` … `task6_55e`).

## Design
34 participants x 6 tasks = 204 recordings, **one continuous set per participant and condition**.
There are no repeat sessions on other days, so personalisation can be tested *within* a session, not across days.
Tasks: IR/ER (internal/external shoulder rotation) at 30-40 %, 40-50 %, 50-60 % of MVIC, each performed until Borg RPE 20.
Task order differs by `group` (1, 2, 3) in `demographic.csv`, so fatigue carried over from earlier tasks differs by person.

## Signals (per recording)
| Stream | Files | Rate | Unit | Notes |
|---|---|---|---|---|
| EMG | 6 files `EMG data/<muscle>.csv` (anterior_deltoid, infraspinatus, latissimus_dorsi, pectoralis_major, posterior_deltoid, upper_trapezius) | 1000 Hz | mV | 1 column `EMG(mV)`; resolution 0.001 mV |
| IMU | 6 sites (Forearm, Hand, Pelvis, Shoulder, Sternum, Upper arm) x acc / gyr / mag, 3 axes each = 54 channels | 100 Hz | m/s², rad/s, a.u. | `acc_x(m/s^2),acc_y,acc_z`; magnetometer not used (a.u., environment dependent) |
| PPG | `PPG data/ppg.csv` | 200 Hz | V | not recorded in MVIC tests |
There is no time column: time = row index / sampling rate, t = 0 at the first row. Raw folder names are `Subject N` (not `subject_N`).

## Labels and tables
| File | Content | Quirks |
|---|---|---|
| `Borg data/borg_data.csv` | per participant and task: `before_task` (t = 0), a rating every 10 s (`10_sec` … `250_sec`), `end_of_trial` (always 20), `length_of_trial_(sec)` | subject name only on the first row of each block (forward-fill); header names contain stray spaces (`120_sec `); the rating grid stops at the last full 10 s before failure |
| `KSS data/KSS_data.csv` | sleepiness before / after | one value per participant |
| `MVIC force data/MVIC_force_data.csv` | IR and ER MVIC (N), two trials and mean | used for target load in N |
| `Demographic and antropometric data/*.csv` | age, sex, hand, group, body composition, segment sizes | folder name is misspelled in the data |
Where the ratings sit: only in `borg_data.csv`; the sensor files contain no markers.

## Participants / recordings with problems (`m1_problems.csv`, `m3_cleaning_log.csv`)
| Where | Problem | Handling |
|---|---|---|
| P3, 45 % IR | EMG anterior deltoid and infraspinatus contain corrupt values (up to 4e34, 86-94 % constant) | flagged invalid; those two channels' features are NaN for that recording (11 windows); other channels kept |
| P27, 30-40 % IR | a sternum accelerometer spike up to 143,956 m/s² (the pipeline treats over 160 as invalid; the normal maximum is about 12) | 6 samples flagged, replaced by interpolation |
| P27, 50-60 % IR | Borg `length_of_trial` = 122 s but sensors last 80.9 s; last grid rating is at 70 s. Probably a typo of ~82 s (unverified) | windows built from the rating grid only (ratings up to 70 s) |
| P3, 30-40 % IR | IMU channels differ in length by 11 s (16,994 vs 18,099 rows) | all streams cut to the shortest (aligned 169.9 s); rating at 180 s rejected |
| P21, 50-60 % IR | Borg duration 65 s vs sensors 59.3 s | rating beyond sensor end rejected |
| P1, 30-40 % ER | Borg duration 5.8 s shorter than sensors | no impact (windows follow ratings) |
| P34, 50-60 % ER | only one mid-trial rating (trial 18 s) | kept, 1 window |
| 13 recordings | IMU channels differ by 1 sample | harmless, cut to shortest |
Sensor streams agree on duration to 0.01 s in all 204 recordings. No NaN values anywhere in the raw files.
