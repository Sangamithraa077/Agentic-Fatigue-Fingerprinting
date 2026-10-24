"""Module 11.7: replaying a recording through the live pipeline must give the offline results.

For each chosen recording the cleaned signals are fed to ``LivePipeline`` one second at a time, and its generalised
and personalised estimates are compared window by window with the offline table path
(m4_features.csv -> population model / calibrated z-score model).  Windows 0 and 1 are skipped: the baseline is
the mean of the first two windows, so the first estimate exists after window 1 (the dashboard shows "Calibrating").

Output: reports/dataset_modules_1_7/m11_replay_equality.csv
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.live import FileReplaySource, LivePipeline, load_models
from ml.dataset_pipeline.personalise import ZScoreModel

K = 4
RECORDINGS = [(1, "task4_35e"), (5, "task2_45i"), (12, "task6_55e"), (20, "task1_35i"), (27, "task3_55i"), (33, "task1_35i")]


def offline_estimates(F: pd.DataFrame, pop, zm) -> pd.DataFrame:
    F = F.copy()
    F["dir_int"] = (F.direction == "internal").astype(int)
    out = pd.DataFrame({"win": F.win.to_numpy(), "generalised": pop.predict(F)})
    calib = F[F.win < K]
    zf = ZScoreModel.z(F, calib, zm.raw_cols)
    pers = np.clip(zm.model.predict(zf[zm.zcols].to_numpy(float)), 6, 20)
    out["personalised"] = np.where(F.win.to_numpy() >= K, pers, out.generalised)
    return out


def main() -> None:
    pop, zm = load_models(K)
    feats = pd.read_csv(c.OUT / "m4_features.csv")
    rows = []
    for sid, task in RECORDINGS:
        F = feats[(feats.sid == sid) & (feats.task == task)].sort_values("win").reset_index(drop=True)
        cond = {"load_pct": float(F.load_pct[0]), "direction": F.direction[0], "mvic_dir_N": float(F.mvic_dir_N[0])}
        pipe = LivePipeline(pop, zm, condition=cond, calibration_windows=K)
        for ch in FileReplaySource(sid, task).chunks():
            pipe.push(ch)
        live = pd.DataFrame([{"win": r["win"], "generalised": r["generalised"], "personalised": r["personalised"]} for r in pipe.results])
        off = offline_estimates(F, pop, zm)
        m = live.merge(off, on="win", suffixes=("_live", "_offline"))
        m = m[m.win >= 2]
        row = {"participant": sid, "task": task, "windows_compared": len(m),
               "max_abs_diff_generalised": float(np.abs(m.generalised_live - m.generalised_offline).max()),
               "max_abs_diff_personalised": float(np.abs(m.personalised_live - m.personalised_offline).max()),
               "live_windows": len(live), "offline_windows": len(F)}
        rows.append(row)
        print(row, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(c.OUT / "m11_replay_equality.csv", index=False)
    assert (R.windows_compared > 0).all(), "a recording had no window to compare"
    worst = max(R.max_abs_diff_generalised.max(), R.max_abs_diff_personalised.max())
    print(f"worst difference between replay and offline estimates: {worst:.2e} Borg points")
    assert worst < 1e-6, "live replay does not reproduce the offline results"


if __name__ == "__main__":
    main()
