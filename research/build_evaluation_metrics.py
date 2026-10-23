"""Build `reports/evaluation_metrics.json` from the study's published results.

Run from anywhere:  python research/build_evaluation_metrics.py

Nothing is computed or estimated here -- every number is read from a result
file this study produced (`research/results/`), so `GET /api/reports/metrics`
serves measured values only.

* `loso_metrics` come from `nested_lopo_ridge_summary.csv`, the output of
  `nested_lopo_ridge_comparison.py`: the exact procedure the deployed Ridge
  Borg estimator (`ml/models/borg_regression`) implements, over all 888
  sequences. `tests/test_borg_regression.py` re-derives these figures.
* `model_family_comparison` is Stage 3 of `final_consolidated_metrics.csv`
  (identical LOPO folds, a slightly different sequence set -- see
  `results/final_synthesis_report.md`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = RESEARCH_DIR / "results"
OUTPUT_FILE = RESEARCH_DIR.parent / "reports" / "evaluation_metrics.json"

_MODEL_KEYS = {
    "Median baseline": "median_baseline",
    "Nested Ridge": "ridge_nested",
    "Existing LSTM": "lstm_attention",
}


def _number(value):
    return None if pd.isna(value) else round(float(value), 4)


def build() -> dict:
    summary = pd.read_csv(RESULTS_DIR / "nested_lopo_ridge_summary.csv")
    loso_metrics = {
        _MODEL_KEYS[row["Model"]]: {
            "mae": _number(row["MAE"]),
            "correlation": _number(row["Correlation"]),
            "within_1_borg_pct": _number(row["Within_1_Borg_Percent"]),
            "mean_participant_mae": _number(row["Mean_Participant_MAE"]),
        }
        for _, row in summary.iterrows()
    }

    consolidated = pd.read_csv(RESULTS_DIR / "final_consolidated_metrics.csv")
    stage3 = consolidated[consolidated["stage"] == "Stage 3"]
    model_family = [
        {
            "model": row["model"],
            "mae": _number(row["pooled_mae"]),
            "rmse": _number(row["rmse"]),
            "correlation": _number(row["correlation"]),
            "within_1_borg_pct": _number(row["within_1_borg_pct"]),
            "fold_mae_std": _number(row["std_fold_mae"]),
        }
        for _, row in stage3.iterrows()
    ]

    return {
        "source": "research/results (34-fold leave-one-participant-out study)",
        "protocol": "Leave-one-participant-out; every held-out athlete scored by a model that never saw them",
        "target": "Borg RPE (6-20 scale), regression",
        "n_participants": 34,
        "n_sequences": 888,
        "loso_metrics": loso_metrics,
        "model_family_comparison": model_family,
        "interpretation": (
            "Errors are in Borg points (lower is better). The Ridge estimate (MAE about 2.3, correlation about "
            "0.16) is not meaningfully better than always predicting the training median (MAE about 2.2), so "
            "treat individual estimates as a research indication, not a measurement."
        ),
    }


if __name__ == "__main__":
    report = build()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE}")
    for name, metrics in report["loso_metrics"].items():
        print(f"  {name:16s} MAE {metrics['mae']}  r {metrics['correlation']}")
