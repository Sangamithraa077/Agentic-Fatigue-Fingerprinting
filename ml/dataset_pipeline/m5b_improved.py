"""Module 5b: stronger baselines and improved generalised models (still leave-one-person-out).

Adds to Module 5:
  - time + condition baseline (load %, direction, target load in N): known before the task starts
  - causal history features: rolling mean of the last 3 windows and the change from the previous window
    (uses only past windows of the same recording, so it can run live)
  - sensors + time + condition, with and without history
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.m5_baselines import feature_sets, metrics, ridge, forest

COND = ["t_end", "load_pct", "dir_int", "target_load_N", "mvic_dir_N"]


def hgb():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=20, random_state=0)


def main():
    F = pd.read_csv(c.OUT / "m4_features.csv").sort_values(["sid", "task", "win"]).reset_index(drop=True)
    F["dir_int"] = (F.direction == "internal").astype(int)
    fs = feature_sets(F, True)
    base = fs["FUSED"]
    g = F.groupby(["sid", "task"])
    hist = {}
    for col in base:
        hist[col + "_m3"] = g[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        hist[col + "_d1"] = g[col].transform(lambda s: s.diff())
    F = pd.concat([F, pd.DataFrame(hist)], axis=1)
    H = list(hist)
    emg_core = [x for x in base if x.startswith("emg_") and x.endswith(("mdf_rel", "mnf_rel", "rms_rel"))]

    specs = [
        ("Time only (isotonic) [Module 5]", None, None),
        ("Time + condition, RF", ["t_end", "load_pct", "dir_int", "target_load_N", "mvic_dir_N"], forest),
        ("Time + condition, boosting", COND, lambda: make_pipeline(SimpleImputer(strategy="median"), hgb())),
        ("Fused sensors, RF [Module 5]", base, forest),
        ("Fused sensors + time + condition, RF", base + COND, forest),
        ("Fused sensors + time + condition, boosting", base + COND, lambda: make_pipeline(SimpleImputer(strategy="median"), hgb())),
        ("Fused + history + time + condition, RF", base + H + COND, forest),
        ("Fused + history + time + condition, boosting", base + H + COND, lambda: make_pipeline(SimpleImputer(strategy="median"), hgb())),
        ("Fused + history + time + condition, Ridge", base + H + COND, ridge),
        ("EMG core + history + time + condition, RF", emg_core + [x + s for x in emg_core for s in ("_m3", "_d1")] + COND, forest),
    ]
    y = F.rpe.to_numpy(float)
    from sklearn.isotonic import IsotonicRegression
    preds, rows = {}, []
    for name, cols, mk in specs:
        p = np.full(len(F), np.nan)
        for sid in sorted(F.sid.unique()):
            tr, te = np.where(F.sid != sid)[0], np.where(F.sid == sid)[0]
            if cols is None:
                m = IsotonicRegression(out_of_bounds="clip").fit(F.t_end.iloc[tr], y[tr]); p[te] = m.predict(F.t_end.iloc[te])
            else:
                X = F[cols].to_numpy(float)
                p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        p = np.clip(p, 6, 20); preds[name] = p
        r = metrics(y, p, F.sid.astype(str) + F.task)
        pp = pd.DataFrame({"sid": F.sid, "ae": np.abs(y - p)}).groupby("sid").ae.mean()
        rows.append(dict(model=name, MAE=r["MAE"], pearson_r=r["pearson_r"], within_recording_spearman=r["within_recording_spearman"],
                         within1=r["within1"], per_person_MAE_median=pp.median(), per_person_MAE_max=pp.max()))
        print(f"{name:50s} MAE {r['MAE']:.3f} r {r['pearson_r']:.3f} rho {r['within_recording_spearman']:.3f} within1 {r['within1']:.3f}", flush=True)
    R = pd.DataFrame(rows)
    t0 = pd.DataFrame({"sid": F.sid, "ae": np.abs(y - preds[specs[0][0]])}).groupby("sid").ae.mean()
    bc = pd.DataFrame({"sid": F.sid, "ae": np.abs(y - preds[specs[1][0]])}).groupby("sid").ae.mean()
    imp, pv, impc, pvc = [], [], [], []
    for n in R.model:
        pp = pd.DataFrame({"sid": F.sid, "ae": np.abs(y - preds[n])}).groupby("sid").ae.mean()
        imp.append(int((pp < t0).sum())); pv.append(wilcoxon(pp, t0).pvalue if (pp != t0).any() else np.nan)
        impc.append(int((pp < bc).sum())); pvc.append(wilcoxon(pp, bc).pvalue if (pp != bc).any() else np.nan)
    R["people_better_than_time_only"], R["p_vs_time_only"] = imp, pv
    R["people_better_than_time+condition_RF"], R["p_vs_time+condition_RF"] = impc, pvc
    R.to_csv(c.OUT / "m5b_improved_results.csv", index=False)
    out = F[["sid", "task", "win", "t_end", "rpe"]].copy()
    for k, v in preds.items():
        out[k] = v
    out.to_csv(c.OUT / "m5b_improved_predictions.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
