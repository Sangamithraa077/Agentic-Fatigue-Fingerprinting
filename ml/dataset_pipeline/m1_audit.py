"""Module 1: inventory table, problem list, data dictionary.

Outputs (outputs/): m1_inventory.csv, m1_problems.csv, m1_participant_summary.csv,
m1_data_dictionary.md
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from ml.dataset_pipeline import common as c


def nrows(path):
    """data rows in a CSV (lines minus header), counted on bytes: fast and parse-free."""
    with open(path, "rb") as f:
        n = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 22), b""))
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            n += 1
    return n - 1


def main():
    borg_wide, borg_long = c.load_borg()
    rows = []
    for task in c.TASKS:
        load, direction = c.task_meta(task)
        for sid in range(1, c.N_SUBJECTS + 1):
            rec = dict(sid=sid, task=task, load_pct=load, direction=direction)
            try:
                d = c.subject_dir(task, sid)
            except FileNotFoundError:
                rec["folder_ok"] = False
                rows.append(rec)
                continue
            rec["folder_ok"] = True
            emg = [d / "EMG data" / f"{m}.csv" for m in c.MUSCLES]
            imu = [d / "IMU data" / f / f"{k}_{t}.csv"
                   for f, t in c.IMU_SITES.items() for k in c.IMU_KINDS]
            ppg = d / "PPG data" / "ppg.csv"
            rec["n_emg_files"] = sum(p.exists() for p in emg)
            rec["n_imu_files"] = sum(p.exists() for p in imu)
            rec["ppg_file"] = ppg.exists()
            en = [nrows(p) for p in emg if p.exists()]
            im = [nrows(p) for p in imu if p.exists()]
            rec["emg_rows_min"], rec["emg_rows_max"] = min(en), max(en)
            rec["imu_rows_min"], rec["imu_rows_max"] = min(im), max(im)
            rec["ppg_rows"] = nrows(ppg)
            rec["emg_sec"] = np.median(en) / c.FS["emg"]
            rec["imu_sec"] = np.median(im) / c.FS["imu"]
            rec["ppg_sec"] = rec["ppg_rows"] / c.FS["ppg"]
            rows.append(rec)
    inv = pd.DataFrame(rows)

    b = borg_wide[["sid", "task_order", "length_of_trial_(sec)", "end_of_trial"]].rename(
        columns={"task_order": "task", "length_of_trial_(sec)": "borg_trial_sec",
                 "end_of_trial": "borg_end_rpe"})
    n_rate = borg_long[(borg_long.t_sec > 0) & (borg_long.t_sec < borg_long.trial_sec)] \
        .groupby(["sid", "task"]).size().rename("n_borg_ratings_in_trial").reset_index()
    inv = inv.merge(b, on=["sid", "task"], how="left").merge(n_rate, on=["sid", "task"], how="left")
    inv["n_sets"] = 1                              # one continuous set per participant/condition
    inv["sensor_sec"] = inv[["emg_sec", "imu_sec", "ppg_sec"]].median(axis=1)
    inv["sensor_spread_sec"] = inv[["emg_sec", "imu_sec", "ppg_sec"]].max(axis=1) - \
        inv[["emg_sec", "imu_sec", "ppg_sec"]].min(axis=1)
    inv["borg_minus_sensor_sec"] = inv["borg_trial_sec"] - inv["sensor_sec"]
    inv["borg_ratings_location"] = "Borg data/borg_data.csv (10 s grid, 'before_task' = t0)"
    inv.to_csv(c.OUT / "m1_inventory.csv", index=False)

    # ---- problem list ----------------------------------------------------------
    prob = []
    for _, r in inv.iterrows():
        k = (r.sid, r.task)
        if not r.folder_ok:
            prob.append((*k, "missing_folder", "")); continue
        if r.n_emg_files != 6: prob.append((*k, "missing_emg_files", r.n_emg_files))
        if r.n_imu_files != 18: prob.append((*k, "missing_imu_files", r.n_imu_files))
        if not r.ppg_file: prob.append((*k, "missing_ppg", ""))
        if r.emg_rows_max != r.emg_rows_min: prob.append((*k, "emg_length_mismatch_between_muscles", f"{r.emg_rows_min}-{r.emg_rows_max}"))
        if r.imu_rows_max != r.imu_rows_min: prob.append((*k, "imu_length_mismatch_between_channels", f"{r.imu_rows_min}-{r.imu_rows_max}"))
        if r.sensor_spread_sec > 1: prob.append((*k, "sensor_duration_spread_gt_1s", round(r.sensor_spread_sec, 2)))
        if abs(r.borg_minus_sensor_sec) > 5: prob.append((*k, "borg_duration_vs_sensor_gt_5s", round(r.borg_minus_sensor_sec, 1)))
        if r.borg_end_rpe != 20: prob.append((*k, "trial_did_not_end_at_rpe20", r.borg_end_rpe))
        if r.n_borg_ratings_in_trial < 2: prob.append((*k, "fewer_than_2_mid_trial_ratings", r.n_borg_ratings_in_trial))
    prob = pd.DataFrame(prob, columns=["sid", "task", "problem", "detail"])
    prob.to_csv(c.OUT / "m1_problems.csv", index=False)

    # ---- participant summary: demographics, MVIC, KSS ---------------------------
    demo = c.load_demographics()
    mvic = pd.read_csv(c.sub("MVIC*") / "MVIC_force_data.csv")
    mvic["sid"] = mvic["subject"].str.extract(r"(\d+)").astype(int)
    kss = pd.read_csv(c.sub("KSS*") / "KSS_data.csv")
    kss["sid"] = kss["subject"].str.extract(r"(\d+)").astype(int)
    part = demo[["sid", "group", "age", "sex", "dominant_hand"]] \
        .merge(mvic.drop(columns="subject"), on="sid").merge(kss.drop(columns="subject"), on="sid")
    part["n_conditions_complete"] = inv[inv.folder_ok & (inv.n_emg_files == 6) & (inv.n_imu_files == 18)
                                         & inv.ppg_file].groupby("sid").size().reindex(part.sid).fillna(0).to_numpy()
    part["n_problem_flags"] = prob.groupby("sid").size().reindex(part.sid).fillna(0).astype(int).to_numpy()
    part.to_csv(c.OUT / "m1_participant_summary.csv", index=False)

    print("records:", len(inv), " folders ok:", int(inv.folder_ok.sum()))
    print("sensor spread max (s):", round(inv.sensor_spread_sec.max(), 3))
    print("sampling check: EMG/IMU/PPG durations median diff (s):", round(inv.sensor_spread_sec.median(), 4))
    print(prob.groupby("problem").size().to_string() if len(prob) else "no problems")
    print(prob.to_string())
    print("trial sec: min/median/max", inv.borg_trial_sec.min(), inv.borg_trial_sec.median(), inv.borg_trial_sec.max())


if __name__ == "__main__":
    main()
