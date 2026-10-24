"""Module 6: augmentation (training data only).

Augmented rows are produced from each recording's own cleaned signals and carry that person's id.
In every leave-one-person-out fold the held-out person contributes NO augmented rows to training and
is tested on real, untouched windows only.  Order is never shuffled (time-ordered windows, labels kept).

Methods (each tested alone against the Module 5 baseline model, then the helpful ones combined):
  noise    additive Gaussian noise, sd = 5% of each channel's sd
  scale    per-sensor gain U(0.9, 1.1) with slow linear drift of up to +-5% across the recording
  warp     small smooth time warp: t' = t + D sin(2 pi t / P + phi), D 0.2-0.5 s, P 30-60 s
  overlap  extra windows ending half-way between Borg ratings (5 s hop), label = linear interpolation
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.m3_clean import CLEAN
from ml.dataset_pipeline.m4_features import window_feats, add_relative
from ml.dataset_pipeline.m5_baselines import ridge, forest, feature_sets, metrics

COPIES = 2
NOISE_FRAC = 0.05
MODELS = {"RandomForest": forest, "Ridge": ridge}


def aug_signals(z, method, rng):
    emg, imu, ppg = z["emg"].astype(float), z["imu"].astype(float), z["ppg"].astype(float)
    names = list(z["imu_names"])
    if method == "noise":
        for a in (emg, imu):
            a += rng.normal(size=a.shape) * NOISE_FRAC * np.nanstd(a, axis=1, keepdims=True)
        ppg += rng.normal(size=ppg.shape) * NOISE_FRAC * np.nanstd(ppg)
    elif method == "scale":
        def gain(n):
            return rng.uniform(.9, 1.1) * (1 + rng.uniform(-.05, .05) * np.linspace(-1, 1, n))
        for j in range(emg.shape[0]):
            emg[j] *= gain(emg.shape[1])
        for site in c.IMU_SITES.values():                       # same gain on x, y, z of a sensor
            for k in ("acc", "gyr"):
                g = gain(imu.shape[1])
                for ax in "xyz":
                    imu[names.index(f"{k}_{site}_{ax}")] *= g
        ppg *= gain(len(ppg))
    elif method == "warp":
        D, P, ph = rng.uniform(.2, .5), rng.uniform(30, 60), rng.uniform(0, 2 * np.pi)
        def warp(x, fs):
            t = np.arange(x.shape[-1]) / fs
            tau = np.clip(t + D * np.sin(2 * np.pi * t / P + ph), 0, t[-1])
            return np.array([np.interp(tau, t, r) for r in np.atleast_2d(x)]).reshape(x.shape)
        emg, imu, ppg = warp(emg, 1000), warp(imu, 100), warp(ppg, 200)
    return {"emg": emg, "imu": imu, "ppg": ppg}, names


def build(method, W, F, seed):
    """Augmented feature rows for one method (all participants)."""
    rows = []
    sigcols = [x for x in F.columns if x.startswith(("emg_", "imu_", "tempo_", "trunk_", "hr", "hrv_")) and not x.endswith("_rel")]
    valid_cols = [x for x in W.columns if x.startswith("valid_")]
    for copy in range(1 if method == "overlap" else COPIES):
        out = []
        for (task, sid), g in W.groupby(["task", "sid"], sort=False):
            z = np.load(CLEAN / task / f"s{sid}.npz")
            rng = np.random.default_rng([seed, copy, sid, c.TASKS.index(task)])
            s, names = ({"emg": z["emg"], "imu": z["imu"], "ppg": z["ppg"]}, list(z["imu_names"])) if method == "overlap" \
                else aug_signals(z, method, rng)
            g = g.sort_values("win").reset_index(drop=True)
            for i, w in g.iterrows():
                if method != "overlap":
                    r = dict(sid=sid, task=task, win=int(w.win), t_end=w.t_end, rpe=w.rpe)
                    r.update(window_feats(s, names, w)); out.append(r)
                elif i >= 1:                                   # half-step window between rating i-1 and i
                    p = g.loc[i - 1]
                    ww = w.copy(); ww["t_end"] = w.t_end - 5; ww["t_start"] = w.t_start - 5
                    for vc in valid_cols:
                        ww[vc] = min(w[vc], p[vc])
                    r = dict(sid=sid, task=task, win=-1, t_end=ww.t_end, rpe=(w.rpe + p.rpe) / 2)
                    r.update(window_feats(s, names, ww)); out.append(r)
        A = pd.DataFrame(out)
        if method == "overlap":
            A = A.merge(F[["sid", "task", "load_pct", "direction"]].drop_duplicates(), on=["sid", "task"])
            base = F[F.win < 2].groupby(["sid", "task"])[sigcols].mean()     # baseline of the real recording
            A["win"] = 99                                         # never part of the baseline
            A = add_relative(A, sigcols, base=base)
        else:
            A = add_relative(A, sigcols)                          # baseline from the augmented recording's own start
        A["method"], A["copy"] = method, copy
        rows.append(A)
    return pd.concat(rows, ignore_index=True)


def lopo(F, cols, mk, aug=None):
    y = F.rpe.to_numpy(float)
    X = F[cols].to_numpy(float)
    p = np.full(len(F), np.nan)
    for sid in sorted(F.sid.unique()):
        tr, te = np.where(F.sid != sid)[0], np.where(F.sid == sid)[0]
        Xtr, ytr = X[tr], y[tr]
        if aug is not None:
            a = aug[aug.sid != sid]                                # never the held-out person
            Xtr = np.vstack([Xtr, a[cols].to_numpy(float)])
            ytr = np.r_[ytr, a.rpe.to_numpy(float)]
        p[te] = mk().fit(Xtr, ytr).predict(X[te])
    return np.clip(p, 6, 20)


def per_person(F, p):
    return pd.DataFrame({"sid": F.sid, "ae": np.abs(F.rpe.to_numpy() - p)}).groupby("sid").ae.mean()


def main():
    F = pd.read_csv(c.OUT / "m4_features.csv")
    W = pd.read_csv(c.OUT / "m3_windows.csv")
    cols = feature_sets(F, True)["FUSED"]
    methods = ["noise", "scale", "warp", "overlap"]
    A = {}
    for m in methods:
        A[m] = build(m, W, F, seed=1)
        print(f"{m}: {len(A[m])} augmented rows (real rows: {len(F)})", flush=True)
        A[m].to_csv(c.OUT / f"m6_aug_features_{m}.csv", index=False)

    rows, det = [], {}
    for mname, mk in MODELS.items():
        pb = lopo(F, cols, mk)
        det[(mname, "none")] = pb
        base_pp = per_person(F, pb)
        r0 = metrics(F.rpe.to_numpy(), pb, F.sid.astype(str) + F.task)
        rows.append(dict(model=mname, augmentation="none (baseline)", train_rows_added=0, **{k: r0[k] for k in ("MAE", "pearson_r", "within_recording_spearman", "within1")},
                         delta_MAE=0.0, people_improved=np.nan, wilcoxon_p=np.nan, helped=np.nan))
        print(mname, "baseline MAE", round(r0["MAE"], 4), flush=True)
        for m in methods:
            p = lopo(F, cols, mk, A[m])
            det[(mname, m)] = p
            pp = per_person(F, p)
            r = metrics(F.rpe.to_numpy(), p, F.sid.astype(str) + F.task)
            d = r["MAE"] - r0["MAE"]
            pv = wilcoxon(pp, base_pp).pvalue
            rows.append(dict(model=mname, augmentation=m, train_rows_added=len(A[m]), **{k: r[k] for k in ("MAE", "pearson_r", "within_recording_spearman", "within1")},
                             delta_MAE=d, people_improved=int((pp < base_pp).sum()), wilcoxon_p=pv,
                             helped=bool(d <= -0.02 and pv < 0.05)))
            print(f"  {m:8s} MAE {r['MAE']:.4f} (delta {d:+.4f}), people improved {(pp < base_pp).sum()}/34, p={pv:.3f}", flush=True)
        helped = [x["augmentation"] for x in rows if x["model"] == mname and x["helped"] is True]
        if len(helped) > 1:
            comb = pd.concat([A[m] for m in helped], ignore_index=True)
            p = lopo(F, cols, mk, comb)
            pp = per_person(F, p); r = metrics(F.rpe.to_numpy(), p, F.sid.astype(str) + F.task)
            rows.append(dict(model=mname, augmentation="combined: " + "+".join(helped), train_rows_added=len(comb),
                             **{k: r[k] for k in ("MAE", "pearson_r", "within_recording_spearman", "within1")},
                             delta_MAE=r["MAE"] - r0["MAE"], people_improved=int((pp < base_pp).sum()),
                             wilcoxon_p=wilcoxon(pp, base_pp).pvalue, helped=np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(c.OUT / "m6_augmentation_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
