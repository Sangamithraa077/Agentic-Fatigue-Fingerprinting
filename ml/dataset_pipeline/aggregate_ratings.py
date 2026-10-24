"""Module 13.4 / 13.6: aggregate the usefulness ratings once 3-5 people have filled the sheet.

Give reports/llm_eval/usefulness_rating_sheet.csv to each rater (one copy each, fill rater_id and the 1-5 columns),
save the returned files as reports/llm_eval/ratings_*.csv, then run:  python -m ml.dataset_pipeline.aggregate_ratings
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from ml.dataset_pipeline import common as c

DIR = c.REPO / "reports" / "llm_eval"
COLS = ["A_clarity_1to5", "A_helpfulness_1to5", "B_clarity_1to5", "B_helpfulness_1to5"]


def main() -> None:
    files = sorted(DIR.glob("ratings_*.csv"))
    if not files:
        print("No ratings_*.csv files yet: usefulness (13.4) and the human ablation (13.6) are still to be collected.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df[COLS] = df[COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=COLS, how="all")
    raters = df.rater_id.nunique()
    summary = df[COLS].agg(["mean", "std", "count"]).T
    pref = df.prefers_A_or_B.str.upper().value_counts(normalize=True).to_dict()
    summary.to_csv(DIR / "usefulness_results.csv")
    print(f"{raters} raters, {len(df)} ratings\n{summary.round(2)}\npreference: {pref}")
    if raters < 3:
        print("fewer than 3 raters: treat as anecdotal")


if __name__ == "__main__":
    main()
