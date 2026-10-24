"""Like-for-like comparison of the Modules 1-7 models with the earlier study in research/.

The study scores 888 sequences, each ending at 30 s or later (label = Borg rating at the sequence end).
The Module 5b/7 tables score every 10 s window from 10 s on, including the low-RPE start of each trial, so
their headline MAEs are not comparable. This script scores both on exactly the same windows:
those whose (participant, task, end time) match a research sequence.

Output: reports/dataset_modules_1_7/comparison_with_research.csv
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from ml.dataset_pipeline import common as c

RESEARCH = c.REPO / "research"


def task_code(condition):
    """'30-40_ external rotation' -> 'task4_35e'."""
    load = int(condition[:2]) + 5
    d = "i" if "internal" in condition else "e"
    return next(t for t in c.TASKS if t.endswith(f"{load}{d}"))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def main():
    meta = pd.read_csv(RESEARCH / "lstm_sequence_metadata.csv")
    rp = pd.read_csv(RESEARCH / "results" / "nested_lopo_ridge_predictions.csv")
    assert (meta.participant.to_numpy() == rp.participant.to_numpy()).all() and \
        np.allclose(meta.target_borg, rp.target_borg), "research metadata and predictions are not row-aligned"
    meta["sid"] = meta.participant.str.extract(r"(\d+)").astype(int)
    meta["task"] = meta.condition.map(task_code)
    meta["t_end"] = meta.sequence_end_s
    R = pd.concat([meta[["sid", "task", "t_end", "target_borg"]], rp[["median_prediction", "ridge_prediction", "lstm_prediction"]]], axis=1)

    P = pd.read_csv(c.OUT / "m5b_improved_predictions.csv")
    mine = [x for x in P.columns if x not in ("sid", "task", "win", "t_end", "rpe")]
    J = R.merge(P, on=["sid", "task", "t_end"], how="inner")
    J["label_gap"] = (J.target_borg - J.rpe).abs()
    n_match = len(J)
    ok = J.label_gap < 1e-6
    J = J[ok]

    rows = [dict(model=f"Research: {name}", source="research/", MAE=mae(J.target_borg, J[col]),
                 within1=float((J.target_borg - J[col]).abs().le(1).mean()))
            for name, col in (("training median", "median_prediction"), ("Ridge (deployed)", "ridge_prediction"),
                              ("LSTM + attention", "lstm_prediction"))]
    for name in mine:
        rows.append(dict(model=f"Modules 5b: {name}", source="ml/dataset_pipeline", MAE=mae(J.rpe, J[name]),
                         within1=float((J.rpe - J[name]).abs().le(1).mean())))
    T = pd.DataFrame(rows)
    T.insert(1, "n_windows", len(J))
    T.to_csv(c.OUT / "comparison_with_research.csv", index=False)
    print(f"research sequences {len(R)}; matched to a Module 5b window {n_match}; with identical label {len(J)}")
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
