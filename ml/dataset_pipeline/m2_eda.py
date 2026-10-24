"""Module 2: exploratory analysis (figures + tables).  Run after m3_clean.py and m4_features.py.

2.1 raw signals for one set      -> figs/m2_1_raw_signals.png
2.2 RPE over time per participant -> figs/m2_2_rpe_over_time.png
2.3 EMG RMS / MDF vs RPE         -> figs/m2_3_*.png, m2_marker_correlations.csv
2.4 between-person differences   -> m2_between_person.csv (note written separately)
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.m3_clean import CLEAN

FIG = c.OUT / "figs"
FIG.mkdir(exist_ok=True)


def fig_raw(sid=1, task="task1_35i"):
    raw = c.load_participant(task, sid)
    z = np.load(CLEAN / task / f"s{sid}.npz")
    _, bl = c.load_borg()
    b = bl[(bl.sid == sid) & (bl.task == task)]
    fig, ax = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    t = np.arange(len(raw["emg"]["anterior_deltoid"])) / 1000
    for m, col in (("anterior_deltoid", "C0"), ("upper_trapezius", "C1")):
        ax[0].plot(t, raw["emg"][m], lw=.3, color=col, label=m)
    ax[0].set_ylabel("raw EMG (mV)"); ax[0].legend(loc="upper right")
    j = c.MUSCLES.index("anterior_deltoid")
    ax[1].plot(np.arange(z["emg"].shape[1]) / 1000, z["emg"][j], lw=.3, color="C0")
    ax[1].set_ylabel("filtered EMG\nant. deltoid (mV)")
    names = list(z["imu_names"])
    for s in ("forearm", "upper_arm"):
        g = np.linalg.norm(np.nan_to_num(z["imu"][[names.index(f"gyr_{s}_{a}") for a in "xyz"]]), axis=0)
        ax[2].plot(np.arange(len(g)) / 100, g, lw=.6, label=f"{s} |gyro|")
    ax[2].set_ylabel("gyro magnitude (rad/s)"); ax[2].legend(loc="upper right")
    ax[3].plot(np.arange(len(z["ppg"])) / 200, z["ppg"], lw=.5, color="C3")
    ax[3].set_ylabel("PPG filtered (V)")
    ax[4].step(b.t_sec, b.rpe, where="post", color="k"); ax[4].plot(b.t_sec, b.rpe, "ko", ms=3)
    ax[4].set_ylabel("Borg RPE"); ax[4].set_xlabel("time (s)"); ax[4].set_ylim(5, 21)
    fig.suptitle(f"Participant {sid}, {task}: raw and cleaned signals with Borg ratings")
    fig.tight_layout(); fig.savefig(FIG / "m2_1_raw_signals.png", dpi=110); plt.close(fig)


def fig_rpe(bl):
    fig, axs = plt.subplots(6, 6, figsize=(18, 14), sharex=True, sharey=True)
    for sid, ax in zip(range(1, 35), axs.ravel()):
        for k, task in enumerate(c.TASKS):
            d = bl[(bl.sid == sid) & (bl.task == task)]
            ax.plot(d.t_sec, d.rpe, color=f"C{k}", lw=1, label=task if sid == 1 else None)
        ax.set_title(f"P{sid}", fontsize=8)
    for ax in axs.ravel()[34:]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=6)
    fig.supxlabel("time since task start (s)"); fig.supylabel("Borg RPE")
    fig.tight_layout(); fig.savefig(FIG / "m2_2_rpe_over_time.png", dpi=100); plt.close(fig)


def partial_spearman(x, y, z):
    """Spearman(x, y) after removing the linear (rank) effect of z from both."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1]


def main():
    F = pd.read_csv(c.OUT / "m4_features.csv")
    wide, bl = c.load_borg()
    fig_raw(); fig_rpe(bl)

    # ---- 2.3 within-recording association of each feature with RPE (and with RPE beyond elapsed time)
    feats = [x for x in F.columns if x.endswith(("_rms", "_mdf", "_mnf", "_gyr_mean", "_gyr_sd", "_jerk", "_hz")) or
             x in ("trunk_sternum_acc_sd", "hr_bpm", "hrv_rmssd_ms", "hrv_sdnn_ms")]
    rows = []
    for f in feats:
        rho, prho = [], []
        for _, g in F.groupby(["sid", "task"]):
            g = g.dropna(subset=[f])
            if len(g) >= 5 and g[f].nunique() > 2 and g.rpe.nunique() > 2:
                rho.append(spearmanr(g[f], g.rpe)[0])
                prho.append(partial_spearman(g[f], g.rpe, g.t_end))
        rho, prho = np.array(rho), np.array(prho)
        kind = f.rsplit("_", 1)[-1]
        expected = -1 if kind in ("mdf", "mnf") else 1       # EMG spectrum shifts down with fatigue; amplitude/HR up
        rows.append(dict(feature=f, n_recordings=len(rho), median_rho=np.median(rho), mean_rho=rho.mean(),
                         share_expected_sign=np.mean(np.sign(rho) == expected) if f.startswith("emg_") else np.nan,
                         share_abs_rho_gt_0_5=np.mean(np.abs(rho) > .5),
                         median_partial_rho_given_time=np.median(prho),
                         share_partial_same_sign=np.mean(np.sign(prho) == np.sign(np.median(rho)))))
    M = pd.DataFrame(rows).sort_values("median_rho", key=np.abs, ascending=False)
    M.to_csv(c.OUT / "m2_marker_correlations.csv", index=False)

    # per participant (average across his/her tasks) for the headline markers
    emg_rms = [f"emg_{m}_rms" for m in c.MUSCLES]; emg_mdf = [f"emg_{m}_mdf" for m in c.MUSCLES]
    pp = []
    for sid, g0 in F.groupby("sid"):
        r = {"sid": sid}
        for f in emg_rms + emg_mdf:
            v = []
            for _, g in g0.groupby("task"):
                g = g.dropna(subset=[f])
                if len(g) >= 5 and g[f].nunique() > 2:
                    v.append(spearmanr(g[f], g.rpe)[0])
            r[f] = np.mean(v) if v else np.nan
        pp.append(r)
    PP = pd.DataFrame(pp)
    PP.to_csv(c.OUT / "m2_per_person_marker_rho.csv", index=False)

    # figures: markers vs RPE
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, cols, ttl, sign in ((axs[0], emg_rms, "EMG RMS", 1), (axs[1], emg_mdf, "EMG median frequency", -1)):
        data = PP[cols].to_numpy()
        ax.boxplot([data[~np.isnan(data[:, i]), i] for i in range(6)], tick_labels=[m[:10] for m in c.MUSCLES])
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(f"{ttl}: within-person Spearman with RPE\n(expected {'positive' if sign > 0 else 'negative'}; one point = one participant)")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(FIG / "m2_3_marker_rho_by_muscle.png", dpi=110); plt.close(fig)

    best_rms = M[M.feature.isin(emg_rms)].iloc[0].feature; best_mdf = M[M.feature.isin(emg_mdf)].iloc[0].feature
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, f in zip(axs, (best_rms + "_rel", best_mdf + "_rel")):
        g = F.dropna(subset=[f])
        ax.scatter(g.rpe + np.random.default_rng(0).normal(0, .12, len(g)), g[f], s=5, alpha=.35)
        mean = g.groupby("rpe")[f].mean()
        ax.plot(mean.index, mean.values, "r-o", ms=3)
        ax.set_xlabel("Borg RPE"); ax.set_ylabel(f + " (change vs own early baseline)")
        ax.set_ylim(np.nanpercentile(g[f], 1), np.nanpercentile(g[f], 99))
    fig.tight_layout(); fig.savefig(FIG / "m2_3_marker_vs_rpe.png", dpi=110); plt.close(fig)

    # ---- 2.4 between-person differences ---------------------------------------------------------
    w = wide.copy()
    w["slope"] = (w["end_of_trial"] - w["before_task"]) / w["length_of_trial_(sec)"]
    bp = w.groupby("sid").agg(mean_trial_sec=("length_of_trial_(sec)", "mean"), sd_trial_sec=("length_of_trial_(sec)", "std"),
                              mean_rpe_slope=("slope", "mean"), start_rpe=("before_task", "mean"))
    F["rep"] = 1
    # share of variance in each (baseline-relative) feature that is between people (ICC-style, one-way)
    icc = {}
    for f in [x for x in F.columns if x.endswith("_rel") and F[x].notna().sum() > 100]:
        g = F[["sid", f]].dropna()
        if g[f].var() > 0:
            icc[f] = g.groupby("sid")[f].mean().var() / g[f].var()
    bp.to_csv(c.OUT / "m2_between_person.csv")
    pd.Series(icc, name="between_person_variance_share").sort_values(ascending=False).to_csv(c.OUT / "m2_feature_between_person_share.csv")
    dur = w.groupby("task_order")["length_of_trial_(sec)"].describe()
    dur.to_csv(c.OUT / "m2_trial_duration_by_task.csv")

    pd.set_option("display.width", 220)
    print(M.head(18).to_string(index=False))
    print("\nper-person median rho: RMS", PP[emg_rms].median().round(2).to_dict())
    print("per-person median rho: MDF", PP[emg_mdf].median().round(2).to_dict())
    print("\ntrial duration by task:\n", dur[["mean", "std", "min", "max"]])
    print("\ntrial duration across people: CV of mean trial sec =", round(bp.mean_trial_sec.std() / bp.mean_trial_sec.mean(), 2))
    print("RPE slope per participant: min/median/max", bp.mean_rpe_slope.min().round(3), bp.mean_rpe_slope.median().round(3), bp.mean_rpe_slope.max().round(3))


if __name__ == "__main__":
    main()
