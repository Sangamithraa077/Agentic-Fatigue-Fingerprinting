"""Module 3: align, flag, filter, window, label, score.

Per recording (participant x task):
  3.1 align EMG/IMU/PPG on one time axis (t=0 = start of recording; cut to the shortest stream)
  3.2 invalid samples -> NaN + flag array (nothing deleted); short gaps (<=0.2 s) interpolated
  3.3 filter each modality separately; EMG power-line check is measured, notch only if present
  3.4 10 s windows ending on each Borg rating time
  3.5 each window takes the rating given at its end (the nearest rating on the 10 s grid)
  3.7 participant quality score
Outputs: data/clean/<task>/s<sid>.npz, outputs/m3_windows.csv, m3_cleaning_log.csv, m3_quality.csv
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, welch

from ml.dataset_pipeline import common as c

CLEAN = c.CLEAN
WIN = 10.0                       # window length = Borg sampling interval (s)
MAX_GAP = {"emg": 200, "imu": 20, "ppg": 40}     # 0.2 s of interpolation allowed
FLAT_RUN = {"emg": 500, "imu": 50, "ppg": 100}   # 0.5 s of identical values = sensor stuck
LIMITS = {"emg": 10.0,           # mV: observed real max 3.9 mV
          "acc": 160.0,          # m/s^2: generous limit (about 16 g); the dataset gives no sensor spec
          "gyr": 35.0,           # rad/s: generous limit (about 2000 deg/s); observed max is 9.5
          "ppg": (0.0, 3.3)}     # V
MIN_VALID_FRAC = 0.8             # window channel usable if >=80% samples valid

SOS = {
    "emg": butter(4, [20, 450], btype="band", fs=c.FS["emg"], output="sos"),
    "acc": butter(4, 20, btype="low", fs=c.FS["imu"], output="sos"),
    "gyr": butter(4, 20, btype="low", fs=c.FS["imu"], output="sos"),
    "ppg": butter(2, [0.5, 8], btype="band", fs=c.FS["ppg"], output="sos"),
}


def flat_runs(x, min_len):
    """True where x is part of a run of >= min_len identical consecutive values."""
    same = np.r_[False, np.diff(x) == 0]
    out = np.zeros(len(x), bool)
    if not same.any():
        return out
    d = np.diff(np.r_[0, same.astype(np.int8), 0])
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s + 1 >= min_len:           # run covers samples s-1 .. e-1
            out[max(s - 1, 0):e] = True
    return out


def repair(x, bad, max_gap):
    """Set bad samples to NaN, linearly interpolate gaps <= max_gap, return (signal, still_missing)."""
    x = x.astype(float).copy()
    x[bad] = np.nan
    s = pd.Series(x)
    interp = s.interpolate(limit=max_gap, limit_area="inside")
    # interpolate() fills the first max_gap NaNs of a longer gap; undo those
    isn = s.isna().to_numpy()
    gap_id = np.cumsum(np.r_[False, isn[1:] != isn[:-1]])
    glen = pd.Series(isn).groupby(gap_id).transform("sum").to_numpy()
    long_gap = isn & (glen > max_gap)
    out = interp.to_numpy()
    out[long_gap] = np.nan
    return out, np.isnan(out)


def clean_channel(x, kind, fs_key):
    if kind == "ppg":
        lo, hi = LIMITS["ppg"]
        bad = ~np.isfinite(x) | (x < lo) | (x > hi)
    else:
        bad = ~np.isfinite(x) | (np.abs(x) > LIMITS[kind])
    bad |= flat_runs(np.where(np.isfinite(x), x, np.nan), FLAT_RUN[fs_key])
    n_bad_raw = int(bad.sum())
    y, missing = repair(x, bad, MAX_GAP[fs_key])
    flag = bad.copy()                                  # flag = originally invalid (even if interpolated)
    if missing.all():
        return np.zeros_like(y), np.ones(len(y), bool), n_bad_raw
    y_f = np.where(np.isnan(y), np.nanmean(y), y)      # filter needs a gap-free input; NaN stays flagged
    if kind == "emg":
        y_f = y_f - np.mean(y_f)
    out = sosfiltfilt(SOS[kind], y_f)
    out[missing] = np.nan
    return out, flag, n_bad_raw


def mains_ratio(x, fs):
    """Power at 50 Hz and 60 Hz relative to neighbouring bands (1.0 = no peak)."""
    x = x[np.isfinite(x)]
    f, P = welch(x, fs=fs, nperseg=2048)
    b = lambda a, z: P[(f >= a) & (f <= z)].mean()
    return (b(49, 51) / ((b(44, 47) + b(53, 56)) / 2), b(59, 61) / ((b(54, 57) + b(63, 66)) / 2))


def process(task, sid, borg_row):
    raw = c.load_participant(task, sid)
    log = dict(sid=sid, task=task)
    # --- 3.1 align ---------------------------------------------------------------
    emg_n = min(len(v) for v in raw["emg"].values())
    imu_n = min(len(v) for v in raw["imu"].values())
    ppg_n = len(raw["ppg"])
    t_end = min(emg_n / c.FS["emg"], imu_n / c.FS["imu"], ppg_n / c.FS["ppg"])
    t_end = np.floor(t_end * 10) / 10                   # common end on a 0.1 s grid
    n = {k: int(round(t_end * c.FS[k])) for k in c.FS}
    log["aligned_sec"] = t_end
    log["lost_sec_vs_longest_stream"] = max(emg_n / c.FS["emg"], imu_n / c.FS["imu"], ppg_n / c.FS["ppg"]) - t_end

    emg, emg_flag, imu, imu_flag, imu_names = [], [], [], [], []
    ratios = []
    nb = 0
    for m in c.MUSCLES:
        x = raw["emg"][m][:n["emg"]]
        ratios.append(mains_ratio(x, c.FS["emg"]))
        y, fl, b = clean_channel(x, "emg", "emg")
        emg.append(y); emg_flag.append(fl); nb += b
    log["emg_flagged_samples"] = nb
    r = np.array(ratios)
    log["mains_50hz_ratio_max"], log["mains_60hz_ratio_max"] = r[:, 0].max(), r[:, 1].max()
    if r[:, 0].max() > 3:                               # measured, not assumed
        b_, a_ = iirnotch(50.0, 30.0, c.FS["emg"])
        sos = tf2sos(b_, a_)
        emg = [np.where(np.isnan(y), np.nan, sosfiltfilt(sos, np.nan_to_num(y))) for y in emg]
        log["notch_applied_hz"] = 50
    nb = 0
    for site, tag in c.IMU_SITES.items():
        for k in ("acc", "gyr"):
            arr = raw["imu"][f"{k}_{tag}"][:n["imu"]]
            for ax, a in zip("xyz", arr.T):
                y, fl, b = clean_channel(a, k, "imu")
                imu.append(y); imu_flag.append(fl); imu_names.append(f"{k}_{tag}_{ax}"); nb += b
    log["imu_flagged_samples"] = nb
    ppg, ppg_flag, nb = clean_channel(raw["ppg"][:n["ppg"]], "ppg", "ppg")
    log["ppg_flagged_samples"] = nb

    emg, emg_flag = np.array(emg, np.float32), np.array(emg_flag)
    imu, imu_flag = np.array(imu, np.float32), np.array(imu_flag)
    d = CLEAN / task
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"s{sid}.npz", emg=emg, emg_flag=emg_flag, imu=imu, imu_flag=imu_flag,
                        ppg=ppg.astype(np.float32), ppg_flag=ppg_flag,
                        imu_names=np.array(imu_names), muscles=np.array(c.MUSCLES), t_end=t_end)

    # --- 3.4 / 3.5 windows + labels ------------------------------------------------
    wins, rej = [], []
    trial = float(borg_row["length_of_trial_(sec)"])
    grid_cols = [col for col in borg_row.index if col.endswith("_sec") and col != "length_of_trial_(sec)"]
    ratings = {int(col.split("_")[0]): borg_row[col] for col in grid_cols if pd.notna(borg_row[col])}
    for i, (t, rpe) in enumerate(sorted(ratings.items())):
        rec = dict(sid=sid, task=task, win=i, t_start=t - WIN, t_end=t, rpe=float(rpe),
                   rpe_before=float(borg_row["before_task"]), trial_sec=trial)
        if t > t_end + 1e-6:
            rec["reason"] = "rating_after_end_of_aligned_sensor_data"
            rej.append(rec); continue
        a, z = int((t - WIN) * c.FS["emg"]), int(t * c.FS["emg"])
        for j, m in enumerate(c.MUSCLES):
            rec[f"valid_emg_{m}"] = 1 - np.mean(emg_flag[j, a:z] | np.isnan(emg[j, a:z]))
        a, z = int((t - WIN) * c.FS["imu"]), int(t * c.FS["imu"])
        for site, tag in c.IMU_SITES.items():
            idx = [q for q, nm in enumerate(imu_names) if nm.split("_", 1)[1].rsplit("_", 1)[0] == tag]
            rec[f"valid_imu_{tag}"] = 1 - np.mean(imu_flag[idx, a:z] | np.isnan(imu[idx, a:z]))
        a, z = int((t - WIN) * c.FS["ppg"]), int(t * c.FS["ppg"])
        rec["valid_ppg"] = 1 - np.mean(ppg_flag[a:z] | np.isnan(ppg[a:z]))
        wins.append(rec)
    return log, wins, rej


def main():
    wide, _ = c.load_borg()
    wide = wide.set_index(["sid", "task_order"])
    logs, wins, rejs = [], [], []
    t0 = time.time()
    for task in c.TASKS:
        for sid in range(1, c.N_SUBJECTS + 1):
            lg, w, rj = process(task, sid, wide.loc[(sid, task)])
            logs.append(lg); wins += w; rejs += rj
        print(f"{task} done ({time.time() - t0:.0f}s)", flush=True)
    log = pd.DataFrame(logs)
    W = pd.DataFrame(wins)
    R = pd.DataFrame(rejs)
    log.to_csv(c.OUT / "m3_cleaning_log.csv", index=False)
    W.to_csv(c.OUT / "m3_windows.csv", index=False)
    R.to_csv(c.OUT / "m3_rejected_windows.csv", index=False)

    # --- 3.7 participant quality score ----------------------------------------------
    valid_cols = [x for x in W.columns if x.startswith("valid_")]
    W["win_ok"] = (W[valid_cols] >= MIN_VALID_FRAC).all(axis=1)
    inv = pd.read_csv(c.OUT / "m1_inventory.csv")
    q = []
    for sid, g in W.groupby("sid"):
        n_all = len(g) + (R.sid == sid).sum()
        flagged = log[log.sid == sid][["emg_flagged_samples", "imu_flagged_samples", "ppg_flagged_samples"]].sum().sum()
        total = log[log.sid == sid].aligned_sec.sum() * (6 * c.FS["emg"] + 36 * c.FS["imu"] + c.FS["ppg"])
        i = inv[inv.sid == sid]
        s = dict(sid=sid,
                 windows=len(g), windows_rejected_no_sensor=int((R.sid == sid).sum()),
                 frac_windows_all_channels_ok=g.win_ok.mean(),
                 frac_samples_flagged=flagged / total,
                 frac_windows_labelled=len(g) / n_all,
                 duration_consistent=float((i.borg_minus_sensor_sec.abs() <= 5).mean()),
                 sensor_lengths_consistent=float(((i.imu_rows_max - i.imu_rows_min) / c.FS["imu"] <= 1).mean()))
        s["quality_score"] = round(100 * np.mean([s["frac_windows_all_channels_ok"], 1 - 10 * s["frac_samples_flagged"] if 10 * s["frac_samples_flagged"] < 1 else 0,
                                                  s["frac_windows_labelled"], s["duration_consistent"],
                                                  s["sensor_lengths_consistent"]]), 1)
        q.append(s)
    Q = pd.DataFrame(q)
    Q.to_csv(c.OUT / "m3_quality.csv", index=False)
    W.to_csv(c.OUT / "m3_windows.csv", index=False)
    print("windows:", len(W), "rejected:", len(R), "windows with all channels ok:", int(W.win_ok.sum()))
    print("mains 50Hz ratio max over recordings:", round(log.mains_50hz_ratio_max.max(), 2),
          " 60Hz:", round(log.mains_60hz_ratio_max.max(), 2),
          " notch applied:", int(log.get("notch_applied_hz", pd.Series(dtype=float)).notna().sum()))
    print(Q.sort_values("quality_score").head(8).to_string())
    print("quality score range:", Q.quality_score.min(), Q.quality_score.max())


if __name__ == "__main__":
    main()
