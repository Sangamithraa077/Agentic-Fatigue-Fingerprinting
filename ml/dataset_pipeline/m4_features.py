"""Module 4: feature extraction from the cleaned recordings -> one table.

One row = one 10 s window ending on a Borg rating.  Columns:
  keys:    sid, task, load_pct, direction, win, t_start, t_end, rpe, rpe_before
  EMG:     emg_<muscle>_{rms,mdf,mnf}                (6 muscles)
  IMU:     imu_<site>_{gyr_mean,gyr_sd,jerk}, tempo_<site>, trunk_sternum_acc_sd
  PPG:     hr_bpm, hrv_rmssd_ms, hrv_sdnn_ms, ppg_beats
  force:   mvic_dir_N, target_load_N (constant per recording; context, not a fatigue signal)
  *_rel :  change from the same recording's early baseline (first 2 windows)  (Module 3.6)
Features of a channel are NaN when <80% of its samples in the window are valid.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.m3_clean import CLEAN, MIN_VALID_FRAC

N_BASE = 2                                # windows that define each recording's early baseline
SITES = {"forearm": "forearm", "hand": "hand", "upper_arm": "upper_arm", "shoulder": "shoulder"}


def emg_feats(x):
    x = np.nan_to_num(x)
    f, P = welch(x, fs=c.FS["emg"], nperseg=1024, noverlap=512)
    m = (f >= 20) & (f <= 450)
    f, P = f[m], P[m]
    cum = np.cumsum(P)
    mdf = f[np.searchsorted(cum, cum[-1] / 2)]
    mnf = (f * P).sum() / P.sum()
    return np.sqrt(np.mean(x ** 2)), mdf, mnf


def tempo_hz(sig, fs):
    sig = np.nan_to_num(sig - np.nanmean(sig))
    f, P = welch(sig, fs=fs, nperseg=min(len(sig), int(10 * fs)))
    m = (f >= 0.2) & (f <= 3.0)
    return f[m][np.argmax(P[m])]


def ppg_feats(x):
    fs = c.FS["ppg"]
    x = np.nan_to_num(x)
    pk, _ = find_peaks(x, distance=int(0.33 * fs), prominence=0.5 * np.std(x))
    if len(pk) < 5:
        return np.nan, np.nan, np.nan, len(pk)
    ibi = np.diff(pk) / fs * 1000
    ibi = ibi[(ibi > 330) & (ibi < 1500)]               # 40-180 bpm
    if len(ibi) < 4:
        return np.nan, np.nan, np.nan, len(pk)
    return 60000 / ibi.mean(), np.sqrt(np.mean(np.diff(ibi) ** 2)), ibi.std(ddof=1), len(pk)


def window_feats(z, names, w):
    """Signal features of one window. z: dict-like with emg (6,N), imu (36,M), ppg (P,) arrays;
    w: has t_start, t_end and the valid_* fractions (Module 3). Also used by Module 6 on augmented signals."""
    r = {}
    a, b = int(round(w["t_start"] * c.FS["emg"])), int(round(w["t_end"] * c.FS["emg"]))
    for j, m in enumerate(c.MUSCLES):
        ok = w[f"valid_emg_{m}"] >= MIN_VALID_FRAC
        vals = emg_feats(z["emg"][j, a:b]) if ok else (np.nan,) * 3
        for k, v in zip(("rms", "mdf", "mnf"), vals):
            r[f"emg_{m}_{k}"] = v
    a, b = int(round(w["t_start"] * c.FS["imu"])), int(round(w["t_end"] * c.FS["imu"]))
    for site, tag in SITES.items():
        ok = w[f"valid_imu_{tag}"] >= MIN_VALID_FRAC
        if ok:
            gi = [names.index(f"gyr_{tag}_{ax}") for ax in "xyz"]
            ai = [names.index(f"acc_{tag}_{ax}") for ax in "xyz"]
            gm = np.linalg.norm(np.nan_to_num(z["imu"][gi, a:b]), axis=0)
            am = np.linalg.norm(np.nan_to_num(z["imu"][ai, a:b]), axis=0)
            r[f"imu_{site}_gyr_mean"], r[f"imu_{site}_gyr_sd"] = gm.mean(), gm.std()
            r[f"imu_{site}_jerk"] = np.sqrt(np.mean((np.diff(am) * c.FS["imu"]) ** 2))
            if site in ("forearm", "upper_arm"):
                r[f"tempo_{site}_hz"] = tempo_hz(gm, c.FS["imu"])
        else:
            for k in ("gyr_mean", "gyr_sd", "jerk"):
                r[f"imu_{site}_{k}"] = np.nan
            if site in ("forearm", "upper_arm"):
                r[f"tempo_{site}_hz"] = np.nan
    if w["valid_imu_sternum"] >= MIN_VALID_FRAC:
        ai = [names.index(f"acc_sternum_{ax}") for ax in "xyz"]
        r["trunk_sternum_acc_sd"] = np.linalg.norm(np.nan_to_num(z["imu"][ai, a:b]), axis=0).std()
    else:
        r["trunk_sternum_acc_sd"] = np.nan
    a, b = int(round(w["t_start"] * c.FS["ppg"])), int(round(w["t_end"] * c.FS["ppg"]))
    vals = ppg_feats(z["ppg"][a:b]) if w["valid_ppg"] >= MIN_VALID_FRAC else (np.nan,) * 3 + (0,)
    r["hr_bpm"], r["hrv_rmssd_ms"], r["hrv_sdnn_ms"], r["ppg_beats"] = vals
    return r


def add_relative(F, sig_cols, base=None):
    """*_rel = (value - early baseline) / |early baseline|, baseline = mean of windows with win < N_BASE per recording."""
    base = F[F.win < N_BASE].groupby(["sid", "task"])[sig_cols].mean() if base is None else base
    B = F[["sid", "task"]].merge(base, left_on=["sid", "task"], right_index=True, how="left")[sig_cols]
    rel = (F[sig_cols].to_numpy() - B.to_numpy()) / (np.abs(B.to_numpy()) + 1e-9)
    return pd.concat([F.reset_index(drop=True), pd.DataFrame(rel, columns=[x + "_rel" for x in sig_cols])], axis=1)


def main():
    W = pd.read_csv(c.OUT / "m3_windows.csv")
    mvic = pd.read_csv(c.sub("MVIC*") / "MVIC_force_data.csv")
    mvic["sid"] = mvic["subject"].str.extract(r"(\d+)").astype(int)
    mvic = mvic.set_index("sid")
    rows = []
    for (task, sid), g in W.groupby(["task", "sid"], sort=False):
        z = np.load(CLEAN / task / f"s{sid}.npz")
        names = list(z["imu_names"])
        load, direction = c.task_meta(task)
        mv = mvic.loc[sid, "IR_MVIC_mean_(N)" if direction == "internal" else "ER_MVIC_mean_(N)"]
        for _, w in g.iterrows():
            r = dict(sid=sid, task=task, load_pct=load, direction=direction, win=int(w.win),
                     t_start=w.t_start, t_end=w.t_end, rpe=w.rpe, rpe_before=w.rpe_before,
                     mvic_dir_N=mv, target_load_N=mv * load / 100)
            r.update(window_feats(z, names, w))
            rows.append(r)
    F = pd.DataFrame(rows)

    # --- 3.6 / 4.5 baseline-relative features -------------------------------------------
    sig_cols = [x for x in F.columns if x.startswith(("emg_", "imu_", "tempo_", "trunk_", "hr", "hrv_"))]
    F = F.sort_values(["sid", "task", "win"]).reset_index(drop=True)
    F = add_relative(F, sig_cols)
    F.to_csv(c.OUT / "m4_features.csv", index=False)

    # --- 4.6 checks -------------------------------------------------------------------------
    raw_cols = sig_cols
    miss = F[raw_cols].isna().sum()
    cor = F[raw_cols].corr(method="spearman").abs()
    pairs = [(a, b, cor.loc[a, b]) for i, a in enumerate(raw_cols) for b in raw_cols[i + 1:] if cor.loc[a, b] > 0.9]
    with open(c.OUT / "m4_feature_check.txt", "w") as f:
        f.write(f"rows {len(F)}, participants {F.sid.nunique()}, tasks {F.task.nunique()}, raw feature columns {len(raw_cols)}\n")
        f.write(f"duplicate (sid,task,win) keys: {int(F.duplicated(['sid', 'task', 'win']).sum())}\n")
        f.write(f"rpe outside 6-20: {int(((F.rpe < 6) | (F.rpe > 20)).sum())}\n")
        f.write(f"non-finite values (inf): {int(np.isinf(F[raw_cols]).sum().sum())}\n")
        f.write("\nMissing values per feature (only those > 0):\n" + miss[miss > 0].to_string() + "\n")
        f.write("\nWindows with any missing raw feature: %d of %d\n" % (F[raw_cols].isna().any(axis=1).sum(), len(F)))
        f.write("\nRecordings with missing EMG features:\n")
        e = F[[x for x in raw_cols if x.startswith("emg_")]].isna().any(axis=1)
        f.write(F[e].groupby(["sid", "task"]).size().to_string() + "\n")
        f.write("\nHeavily overlapping features (|Spearman| > 0.9):\n")
        for a, b, v in sorted(pairs, key=lambda t: -t[2]):
            f.write(f"  {a:34s} {b:34s} {v:.3f}\n")
    print(open(c.OUT / "m4_feature_check.txt").read())


if __name__ == "__main__":
    main()
