"""Module 5: evaluation setup and baselines.

5.1 splits    leave-one-person-out (LOPO) for generalised models; time-ordered split inside a
              recording (first half = calibration/train, second half = test) for personalised models
5.2 metrics   MAE, pooled Pearson r, mean within-recording Spearman r, share within +-1 RPE point
5.3 dummies   train-mean; elapsed-time-only
5.4 single-sensor models (EMG / IMU / PPG), 5.5 fused model
5.6 saves results table every later model must beat
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.dataset_pipeline import common as c

warnings.filterwarnings("ignore")
SEED = 0


# ---------------- 5.1 splits -------------------------------------------------------------------
def lopo_splits(df):
    for sid in sorted(df.sid.unique()):
        yield sid, np.where(df.sid != sid)[0], np.where(df.sid == sid)[0]


def time_split(df, frac=0.5):
    """Within each recording: first `frac` of its windows -> train/calibration, rest -> test. Never shuffled."""
    n = df.groupby(["sid", "task"]).win.transform("count")
    cal = df.win < np.ceil(n * frac)
    return np.where(cal)[0], np.where(~cal)[0]


# ---------------- 5.2 metrics ------------------------------------------------------------------
def metrics(y, p, groups=None):
    out = dict(n=len(y), MAE=np.mean(np.abs(y - p)), within1=np.mean(np.abs(y - p) <= 1))
    out["pearson_r"] = pearsonr(y, p)[0] if np.std(p) > 0 else np.nan
    if groups is not None:
        rs = []
        for _, idx in pd.Series(range(len(y))).groupby(groups.to_numpy()):
            i = idx.to_numpy()
            if len(i) >= 4 and np.std(p[i]) > 0 and np.std(y[i]) > 0:
                rs.append(spearmanr(y[i], p[i])[0])
        out["within_recording_spearman"] = np.mean(rs) if rs else np.nan
    return out


# ---------------- models -----------------------------------------------------------------------
def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-1, 3, 9)))


def forest():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(300, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=SEED))


class MeanModel:
    def fit(self, X, y): self.m = float(np.mean(y)); return self
    def predict(self, X): return np.full(len(X), self.m)


class TimeOnly:
    """Monotone (isotonic) map from elapsed time to RPE, fitted on training people only."""
    def fit(self, X, y): self.iso = IsotonicRegression(out_of_bounds="clip").fit(X[:, 0], y); return self
    def predict(self, X): return self.iso.predict(X[:, 0])


def feature_sets(F, rel):
    sfx = "_rel" if rel else ""
    emg = [f"emg_{m}_{k}{sfx}" for m in c.MUSCLES for k in ("rms", "mdf", "mnf")]
    imu = [x for x in F.columns if (x.startswith("imu_") or x.startswith("tempo_") or x.startswith("trunk_")) and x.endswith("_rel") == rel]
    ppg = [f"{k}{sfx}" for k in ("hr_bpm", "hrv_rmssd_ms", "hrv_sdnn_ms")]
    return {"EMG": emg, "IMU": imu, "PPG": ppg, "FUSED": emg + imu + ppg}


def main():
    F = pd.read_csv(c.OUT / "m4_features.csv")
    F = F[F.win >= 0].reset_index(drop=True)
    y = F.rpe.to_numpy(float)
    rows, preds = [], {}
    specs = []                                     # (name, family, features, model factory, feature variant)
    specs.append(("Dummy: train-mean", "dummy", ["t_end"], MeanModel, "-"))
    specs.append(("Dummy: elapsed time only", "dummy", ["t_end"], TimeOnly, "-"))
    for rel in (False, True):
        fs = feature_sets(F, rel)
        v = "baseline-relative" if rel else "raw"
        for sensor in ("EMG", "IMU", "PPG", "FUSED"):
            for mname, mk in (("Ridge", ridge), ("RandomForest", forest)):
                nm = f"{'Fused' if sensor == 'FUSED' else sensor + '-only'} {mname} ({v})"
                specs.append((nm, "sensor", fs[sensor], mk, v))
    # elapsed time added to the best-practice fused features: reference for how much sensors add beyond time
    fs = feature_sets(F, True)
    specs.append(("Fused + elapsed time RandomForest (baseline-relative)", "reference", fs["FUSED"] + ["t_end"], forest, "baseline-relative"))

    for name, fam, cols, mk, var in specs:
        X = F[cols].to_numpy(float)
        p = np.full(len(F), np.nan)
        for sid, tr, te in lopo_splits(F):
            m = mk().fit(X[tr], y[tr])
            p[te] = m.predict(X[te])
        p = np.clip(p, 6, 20)
        preds[name] = p
        r = metrics(y, p, F.sid.astype(str) + F.task)
        per = pd.DataFrame({"sid": F.sid, "ae": np.abs(y - p)}).groupby("sid").ae.mean()
        r.update(model=name, family=fam, features=var, n_features=len(cols),
                 per_person_MAE_median=per.median(), per_person_MAE_min=per.min(), per_person_MAE_max=per.max(),
                 per_person_MAE_mean=per.mean())
        rows.append(r)
        print(f"{name:62s} MAE {r['MAE']:.3f}  r {r['pearson_r']:.3f}  within-rec rho {r['within_recording_spearman']:.3f}  within1 {r['within1']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    # paired comparison with the train-mean dummy, participant level
    base = pd.DataFrame({"sid": F.sid, "ae": np.abs(y - preds["Dummy: train-mean"])}).groupby("sid").ae.mean()
    tbase = pd.DataFrame({"sid": F.sid, "ae": np.abs(y - preds["Dummy: elapsed time only"])}).groupby("sid").ae.mean()
    for i, r in res.iterrows():
        per = pd.DataFrame({"sid": F.sid, "ae": np.abs(y - preds[r.model])}).groupby("sid").ae.mean()
        res.loc[i, "people_better_than_mean_dummy"] = int((per < base).sum())
        res.loc[i, "wilcoxon_p_vs_mean_dummy"] = wilcoxon(per, base).pvalue if (per != base).any() else np.nan
        res.loc[i, "people_better_than_time_only"] = int((per < tbase).sum())
        res.loc[i, "wilcoxon_p_vs_time_only"] = wilcoxon(per, tbase).pvalue if (per != tbase).any() else np.nan
    cols = ["model", "family", "features", "n_features", "n", "MAE", "within1", "pearson_r", "within_recording_spearman",
            "per_person_MAE_mean", "per_person_MAE_median", "per_person_MAE_min", "per_person_MAE_max",
            "people_better_than_mean_dummy", "wilcoxon_p_vs_mean_dummy", "people_better_than_time_only", "wilcoxon_p_vs_time_only"]
    res[cols].to_csv(c.OUT / "m5_baseline_results.csv", index=False)

    out = F[["sid", "task", "win", "t_end", "rpe"]].copy()
    for k, v in preds.items():
        out[k] = v
    out.to_csv(c.OUT / "m5_baseline_predictions.csv", index=False)
    per_person = []
    for k in preds:
        d = pd.DataFrame({"sid": F.sid, "ae": np.abs(y - preds[k])}).groupby("sid").ae.mean().rename(k)
        per_person.append(d)
    pd.concat(per_person, axis=1).to_csv(c.OUT / "m5_baseline_by_person.csv")

    # time-ordered within-recording split (for Module 7), same dummies for reference
    tr, te = time_split(F)
    rows2 = []
    for name, cols_, mk in (("Dummy: train-mean (within-recording)", ["t_end"], MeanModel),):
        pass
    # per-recording train-mean / time-only: fitted on the person's own first half
    pm, pt = np.full(len(F), np.nan), np.full(len(F), np.nan)
    for (sid, task), g in F.groupby(["sid", "task"]):
        a = g.index[g.index.isin(tr)]; b = g.index[g.index.isin(te)]
        if len(a) == 0 or len(b) == 0:
            continue                                   # recordings with 1 window have no test half
        pm[b] = y[a].mean()
        pt[b] = IsotonicRegression(out_of_bounds="clip").fit(F.t_end[a], y[a]).predict(F.t_end[b])
    for nm, p in (("Own first half: mean RPE", pm), ("Own first half: isotonic(time)", pt)):
        ok = ~np.isnan(p[te])
        r = metrics(y[te][ok], np.clip(p[te][ok], 6, 20)); r["model"] = nm; rows2.append(r)
    pd.DataFrame(rows2).to_csv(c.OUT / "m5_time_split_dummies.csv", index=False)
    sp = F[["sid", "task", "win"]].copy(); sp["time_split"] = np.where(F.index.isin(tr), "calibration", "test")
    sp["lopo_fold"] = F.sid
    sp.to_csv(c.OUT / "m5_splits.csv", index=False)
    print(pd.DataFrame(rows2).to_string())


if __name__ == "__main__":
    main()
