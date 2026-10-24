"""Module 12.7: which features sit behind each window's estimate.

Occlusion attribution against the saved population model: for each feature, set it to "no change from this
person's own start" (rel = 0, elapsed time = 20 s) and record how far the estimate moves. impact > 0 means the
feature pushes the estimate up. It explains the model's output; it is not a causal statement about the muscle.

Output: reports/dataset_modules_1_7/m8_driver_contributions.csv (top 5 drivers per window).
"""

import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from ml.dataset_pipeline import common as c

TOP = 5


def main():
    with open(c.MODELS / "population_boosting.pkl", "rb") as f:
        pop = pickle.load(f)
    F = pd.read_csv(c.OUT / "m4_features.csv").sort_values(["sid", "task", "win"]).reset_index(drop=True)
    F["dir_int"] = (F.direction == "internal").astype(int)
    cols = pop.cols
    X = F[cols].to_numpy(float)
    base = pop.model.predict(X)
    impacts = {}
    for j, name in enumerate(cols):
        if name.endswith("_rel"):
            neutral = 0.0
        elif name == "t_end":
            neutral = 20.0                      # the first two windows define the personal baseline
        else:
            continue                            # load, direction, strength are context, not evidence
        Xo = X.copy()
        Xo[:, j] = neutral
        impacts[name] = base - pop.model.predict(Xo)
    names = list(impacts)
    M = np.column_stack([impacts[n] for n in names])
    rows = []
    for i in range(len(F)):
        for rank, j in enumerate(np.argsort(-np.abs(M[i]))[:TOP], 1):
            rows.append((F.sid[i], F.task[i], int(F.win[i]), rank, names[j], float(M[i, j])))
    out = pd.DataFrame(rows, columns=["sid", "task", "win", "rank", "feature", "impact"])
    out.to_csv(c.OUT / "m8_driver_contributions.csv", index=False)
    top = out[out["rank"] == 1].feature.value_counts().head(8)
    print(f"{len(F)} windows; most frequent top driver:\n{top.to_string()}")


if __name__ == "__main__":
    main()
