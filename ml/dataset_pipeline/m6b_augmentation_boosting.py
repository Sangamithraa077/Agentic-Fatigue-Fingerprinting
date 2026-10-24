"""Module 6b: re-test the four augmentations on the best Module 5b model
(fused baseline-relative sensors + elapsed time + condition, gradient boosting), leave-one-person-out,
augmented rows added to training only (never for the held-out person). Reuses the augmented feature
tables built by m6_augmentation.py (same seed)."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.m5_baselines import feature_sets, metrics
from ml.dataset_pipeline.m5b_improved import COND, hgb
from ml.dataset_pipeline.m6_augmentation import per_person

METHODS = ["noise", "scale", "warp", "overlap"]
mk = lambda: make_pipeline(SimpleImputer(strategy="median"), hgb())


def lopo(F, cols, aug=None):
    X, y = F[cols].to_numpy(float), F.rpe.to_numpy(float)
    p = np.full(len(F), np.nan)
    for sid in sorted(F.sid.unique()):
        tr, te = np.where(F.sid != sid)[0], np.where(F.sid == sid)[0]
        Xtr, ytr = X[tr], y[tr]
        if aug is not None:
            a = aug[aug.sid != sid]
            Xtr, ytr = np.vstack([Xtr, a[cols].to_numpy(float)]), np.r_[ytr, a.rpe.to_numpy(float)]
        p[te] = mk().fit(Xtr, ytr).predict(X[te])
    return np.clip(p, 6, 20)


def main():
    F = pd.read_csv(c.OUT / "m4_features.csv")
    F["dir_int"] = (F.direction == "internal").astype(int)
    cols = feature_sets(F, True)["FUSED"] + COND
    ctx = F[["sid", "task", "load_pct", "dir_int", "target_load_N", "mvic_dir_N"]].drop_duplicates()
    A = {}
    for m in METHODS:
        a = pd.read_csv(c.OUT / f"m6_aug_features_{m}.csv").drop(columns=["load_pct", "direction"], errors="ignore")
        A[m] = a.merge(ctx, on=["sid", "task"])
    g = F.rpe.to_numpy(float); grp = F.sid.astype(str) + F.task
    p0 = lopo(F, cols); pp0 = per_person(F, p0); r0 = metrics(g, p0, grp)
    rows = [dict(augmentation="none (baseline)", train_rows_added=0, MAE=r0["MAE"], pearson_r=r0["pearson_r"],
                 within_recording_spearman=r0["within_recording_spearman"], within1=r0["within1"], delta_MAE=0.0)]
    print("baseline MAE", round(r0["MAE"], 4), flush=True)

    def run(name, aug):
        p = lopo(F, cols, aug); pp = per_person(F, p); r = metrics(g, p, grp); d = r["MAE"] - r0["MAE"]
        pv = wilcoxon(pp, pp0).pvalue
        rows.append(dict(augmentation=name, train_rows_added=len(aug), MAE=r["MAE"], pearson_r=r["pearson_r"],
                         within_recording_spearman=r["within_recording_spearman"], within1=r["within1"], delta_MAE=d,
                         people_improved=int((pp < pp0).sum()), wilcoxon_p=pv, helped=bool(d <= -0.02 and pv < 0.05)))
        print(f"{name:28s} MAE {r['MAE']:.4f} ({d:+.4f}) improved {(pp < pp0).sum()}/34 p={pv:.3f}", flush=True)
        return rows[-1]["helped"]

    helped = [m for m in METHODS if run(m, A[m])]
    if len(helped) > 1:
        run("combined: " + "+".join(helped), pd.concat([A[m] for m in helped], ignore_index=True))
    R = pd.DataFrame(rows)
    R.to_csv(c.OUT / "m6b_augmentation_boosting_results.csv", index=False)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
