"""Module 7: personalised models, evaluated leave-one-person-out + time-ordered split inside each recording.

For each held-out person: population model trained on the other 33 (all their windows).
The person's first half of every recording = calibration, second half = test (never shuffled).
Everything is scored on the SAME test windows, so the generalised model is the like-for-like model to beat.

CAUTION: "first half of the recording" is an oracle: a live system does not know where half-way is, and normalising
with those statistics probably leaks how long the exercise lasts. The 7.1 gain it shows (1.37 to 1.18) is therefore NOT
deployable. ``live_calibration`` below repeats 7.1 with a fixed calibration period (first K windows, K = 3, 4, 5), which
a device can do: there the gain disappears (e.g. 1.37 to 1.35 at 40 s, p = 0.81). Rely on the fixed-K results.
7.1 per-person baseline normalisation   z-score features with the person's own calibration half
7.2 population model fine-tuned on calibration windows  (bias / linear / weighted retrain)
7.3 small per-person models              fitted only on that person's calibration windows
7.4 per-person comparison table
7.5 final models + calibration function saved to models/
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.m5_baselines import time_split, metrics
from ml.dataset_pipeline.personalise import Population, ZScoreModel, feature_columns, fit_population, calibrate, new_boost, COND

MODELS = c.MODELS


def zscore_by_calibration(F, raw_cols, cal_mask):
    """Per recording: (x - mean) / sd using only that recording's calibration windows."""
    Z = F.copy()
    g = F[cal_mask].groupby(["sid", "task"])[raw_cols]
    mu, sd = g.transform("mean"), g.transform("std")
    stats = F[["sid", "task"]].merge(F[cal_mask][["sid", "task"]].assign(i=1).drop_duplicates(), on=["sid", "task"], how="left")
    m = g.mean().add_suffix("_mu"); s = g.std().add_suffix("_sd")
    J = F[["sid", "task"]].merge(m, left_on=["sid", "task"], right_index=True, how="left").merge(s, left_on=["sid", "task"], right_index=True, how="left")
    for col in raw_cols:
        Z[col + "_z"] = (F[col] - J[col + "_mu"]) / (J[col + "_sd"].fillna(0) + 1e-6 * (J[col + "_mu"].abs() + 1e-9))
        Z[col + "_z"] = Z[col + "_z"].clip(-10, 10)
    return Z


def small_person_model(calib, cols):
    X = calib[cols].to_numpy(float)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 9))).fit(X, calib.rpe.to_numpy(float))


def live_calibration(F, ks=(3, 4, 5)):
    """Deployable variant of 7.1: calibrate on the first K windows (30-50 s) of a recording instead of its first
    half, which a live system cannot know in advance. Windows before K use the generalised model.
    Writes m7_live_calibration_predictions.csv and m7_live_calibration_results.csv."""
    y = F.rpe.to_numpy(float)
    cols = feature_columns(F)
    raw_sensor = [x[:-4] for x in cols if x.endswith("_rel")]
    zcols = [x + "_z" for x in raw_sensor] + COND
    out = F[["sid", "task", "win", "t_end", "rpe"]].copy()
    out["generalised"] = np.nan
    rows = []
    for K in ks:
        cal = (F.win < K).to_numpy()
        Z = zscore_by_calibration(F, raw_sensor, cal)
        pers = np.full(len(F), np.nan)
        for sid in sorted(F.sid.unique()):
            oth, me = (F.sid != sid).to_numpy(), (F.sid == sid).to_numpy()
            if K == ks[0]:
                out.loc[me, "generalised"] = fit_population(F[oth]).predict(F[me])
            mz = new_boost().fit(Z.loc[oth, zcols].to_numpy(float), y[oth])
            pers[me] = np.clip(mz.predict(Z.loc[me, zcols].to_numpy(float)), 6, 20)
        out[f"personalised_K{K}"] = np.where(cal, out["generalised"], pers)
        # deployable model for a new person: fitted on everyone, calibrated with their first K windows
        zm = ZScoreModel(raw_sensor, new_boost().fit(Z[zcols].to_numpy(float), y))
        with open(MODELS / f"zscore_live_K{K}.pkl", "wb") as f:
            pickle.dump(zm, f)
        test = ~cal
        g = pd.DataFrame({"sid": F.sid[test], "g": np.abs(y[test] - out["generalised"][test]),
                          "p": np.abs(y[test] - pers[test])}).groupby("sid").mean()
        rows.append(dict(calibration_windows=K, calibration_seconds=K * 10, n_test_windows=int(test.sum()),
                         MAE_generalised=float(np.mean(np.abs(y[test] - out["generalised"][test]))),
                         MAE_personalised=float(np.mean(np.abs(y[test] - pers[test]))),
                         people_improved=int((g.p < g.g).sum()), n_people=int(len(g)), wilcoxon_p=wilcoxon(g.p, g.g).pvalue))
        print(f"live calibration K={K}: generalised {rows[-1]['MAE_generalised']:.3f} -> personalised "
              f"{rows[-1]['MAE_personalised']:.3f}, improved {rows[-1]['people_improved']}/{rows[-1]['n_people']}", flush=True)
    out.to_csv(c.OUT / "m7_live_calibration_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(c.OUT / "m7_live_calibration_results.csv", index=False)


def main():
    MODELS.mkdir(exist_ok=True)
    F = pd.read_csv(c.OUT / "m4_features.csv").sort_values(["sid", "task", "win"]).reset_index(drop=True)
    F["dir_int"] = (F.direction == "internal").astype(int)
    tr_i, te_i = time_split(F)
    cal = F.index.isin(tr_i)
    test = F.index.isin(te_i)
    # recordings with a single window have no test half; they stay in calibration only
    cols = feature_columns(F)
    raw_sensor = [x[:-4] for x in cols if x.endswith("_rel")]
    Zdf = zscore_by_calibration(F, raw_sensor, cal)
    zcols = [x + "_z" for x in raw_sensor] + COND
    small_cols = ["t_end"] + [x for x in cols if x.startswith("emg_") and x.endswith(("mdf_rel", "mnf_rel"))] + ["hr_bpm_rel"]

    y = F.rpe.to_numpy(float)
    names = ["Generalised (model to beat)", "7.1 Per-person z-score normalisation (population model)",
             "7.2a Fine-tune: bias correction", "7.2b Fine-tune: linear recalibration",
             "7.2c Fine-tune: weighted retrain (w=10)", "7.2d Fine-tune: weighted retrain (w=30)",
             "7.3a Small per-person: isotonic(time)", "7.3b Small per-person: ridge on EMG freq + HR + time"]
    P = {n: np.full(len(F), np.nan) for n in names}
    for sid in sorted(F.sid.unique()):
        oth, me = (F.sid != sid).to_numpy(), (F.sid == sid).to_numpy()
        pop = fit_population(F[oth])
        Fm = F[me]; Cm = Fm[cal[me]]
        P[names[0]][me] = pop.predict(Fm)
        P[names[2]][me] = calibrate(pop, Cm, "bias")(Fm)
        P[names[3]][me] = calibrate(pop, Cm, "linear")(Fm)
        P[names[4]][me] = calibrate(pop, Cm, "weighted", weight=10)(Fm)
        P[names[5]][me] = calibrate(pop, Cm, "weighted", weight=30)(Fm)
        # 7.1: population model trained on z-scored features of the other people
        mz = new_boost().fit(Zdf.loc[oth, zcols].to_numpy(float), y[oth])
        P[names[1]][me] = np.clip(mz.predict(Zdf.loc[me, zcols].to_numpy(float)), 6, 20)
        # 7.3: small per-person models, fitted on calibration windows of that person only
        iso = IsotonicRegression(out_of_bounds="clip").fit(Cm.t_end, Cm.rpe)
        P[names[6]][me] = iso.predict(Fm.t_end)
        P[names[7]][me] = np.clip(small_person_model(Cm, small_cols).predict(Fm[small_cols].to_numpy(float)), 6, 20)
        print(f"person {sid} done", flush=True)

    grp = (F.sid.astype(str) + F.task)[test]
    base_pp = pd.DataFrame({"sid": F.sid[test], "ae": np.abs(y[test] - P[names[0]][test])}).groupby("sid").ae.mean()
    rows, byp = [], {}
    for n in names:
        p = P[n]
        r = metrics(y[test], p[test], grp)
        pp = pd.DataFrame({"sid": F.sid[test], "ae": np.abs(y[test] - p[test])}).groupby("sid").ae.mean()
        byp[n] = pp
        d = pp - base_pp
        rows.append(dict(model=n, n_test_windows=int(test.sum()), MAE=r["MAE"], pearson_r=r["pearson_r"],
                         within_recording_spearman=r["within_recording_spearman"], within1=r["within1"],
                         per_person_MAE_median=pp.median(), per_person_MAE_min=pp.min(), per_person_MAE_max=pp.max(),
                         mean_gain_vs_generalised=-d.mean(), people_improved=int((d < 0).sum()),
                         wilcoxon_p=wilcoxon(pp, base_pp).pvalue if n != names[0] else np.nan))
        print(f"{n:62s} MAE {r['MAE']:.3f}  r {r['pearson_r']:.3f}  improved {(d < 0).sum()}/34", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(c.OUT / "m7_results.csv", index=False)
    B = pd.DataFrame(byp)
    B["n_test_windows"] = F[test].groupby("sid").size()
    B.to_csv(c.OUT / "m7_by_person.csv")
    out = F[["sid", "task", "win", "t_end", "rpe"]].copy(); out["split"] = np.where(test, "test", "calibration")
    for n in names:
        out[n] = P[n]
    out.to_csv(c.OUT / "m7_predictions.csv", index=False)

    # ---- 7.5 save final models ---------------------------------------------------------------
    pop = fit_population(F)
    with open(MODELS / "population_boosting.pkl", "wb") as f:
        pickle.dump(pop, f)
    zm = ZScoreModel(raw_sensor, new_boost().fit(Zdf[zcols].to_numpy(float), y))
    with open(MODELS / "zscore_personalised_boosting.pkl", "wb") as f:
        pickle.dump(zm, f)
    # reference: elapsed-time-only (LOPO) scored on the same test windows
    T = pd.read_csv(c.OUT / "m5b_improved_predictions.csv")
    key = ["sid", "task", "win"]
    T = F[key].merge(T, on=key, how="left")
    tm = np.abs(T["rpe"].to_numpy() - T["Time only (isotonic) [Module 5]"].to_numpy())[test]
    print(f"reference: elapsed-time-only MAE on the same test windows = {tm.mean():.3f}")
    pd.DataFrame({"model": ["Elapsed time only (LOPO)"], "MAE": [tm.mean()]}).to_csv(c.OUT / "m7_time_only_reference.csv", index=False)
    pd.Series(pop.cols).to_csv(MODELS / "population_feature_columns.csv", index=False, header=False)
    print(R.round(3).to_string(index=False))
    live_calibration(F)


if __name__ == "__main__":
    main()
