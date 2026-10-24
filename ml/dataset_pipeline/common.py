"""Shared loader and constants for the WSD4FEDSRM dataset pipeline (Modules 1-7).

Where things live (override the dataset location with the WSD4FEDSRM_DATA environment variable):
    DATA    datasets/raw/WSD4FEDSRM/                  raw dataset, read-only
    CLEAN   datasets/processed/clean_signals/         cleaned signals + flags (git-ignored, ~0.5 GB)
    OUT     reports/dataset_modules_1_7/              tables, figures, notes
    MODELS  ml/models/dataset_pipeline/               saved population / personalised models

One command loads any participant and task:
    python -m ml.dataset_pipeline.common 1 task4_35e
"""

import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("WSD4FEDSRM_DATA", REPO / "datasets" / "raw" / "WSD4FEDSRM"))
CLEAN = REPO / "datasets" / "processed" / "clean_signals"
OUT = REPO / "reports" / "dataset_modules_1_7"
MODELS = REPO / "ml" / "models" / "dataset_pipeline"
OUT.mkdir(parents=True, exist_ok=True)

FS = {"emg": 1000.0, "imu": 100.0, "ppg": 200.0}
N_SUBJECTS = 34

# task code (as in borg_data.csv) -> raw condition folder, found by name pattern because the
# release spells folders differently ("30-40_ external rotation", "30–40% external rotation")
TASKS = ["task1_35i", "task2_45i", "task3_55i", "task4_35e", "task5_45e", "task6_55e"]


def sub(pattern):
    """The one file or folder under DATA matching a glob (e.g. 'MVIC*')."""
    hits = sorted(DATA.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"nothing matching {pattern!r} under {DATA}; extract the dataset there "
                                f"or set WSD4FEDSRM_DATA")
    return hits[0]


def raw_root():
    return sub("EMG*")


def task_folder(task):
    load, direction = task_meta(task)
    for d in raw_root().iterdir():
        m = re.match(r"(\d\d)\D{1,3}(\d\d)\D*?(internal|external)", d.name)
        if d.is_dir() and m and int(m.group(1)) + 5 == load and m.group(3) == direction:
            return d
    raise FileNotFoundError(f"condition folder for {task}")
MUSCLES = ["anterior_deltoid", "infraspinatus", "latissimus_dorsi",
           "pectoralis_major", "posterior_deltoid", "upper_trapezius"]
IMU_SITES = {"Forearm": "forearm", "Hand": "hand", "Pelvis": "pelvis",
             "Shoulder": "shoulder", "Sternum": "sternum", "Upper arm": "upper_arm"}
IMU_KINDS = ["acc", "gyr", "mag"]


def task_meta(task):
    """load % (35/45/55) and direction (internal/external) from a task code."""
    m = re.match(r"task\d_(\d\d)([ie])", task)
    return int(m.group(1)), ("internal" if m.group(2) == "i" else "external")


def subject_dir(task, subject):
    d = task_folder(task)
    for name in (f"Subject {subject}", f"subject_{subject}"):
        if (d / name).is_dir():
            return d / name
    raise FileNotFoundError(f"{task} subject {subject}")


def load_emg(task, subject):
    d = subject_dir(task, subject) / "EMG data"
    return {m: pd.read_csv(d / f"{m}.csv").iloc[:, 0].to_numpy(float) for m in MUSCLES}


def load_imu(task, subject):
    d = subject_dir(task, subject) / "IMU data"
    out = {}
    for folder, tag in IMU_SITES.items():
        for k in IMU_KINDS:
            df = pd.read_csv(d / folder / f"{k}_{tag}.csv")
            out[f"{k}_{tag}"] = df.to_numpy(float)       # (n, 3)
    return out


def load_ppg(task, subject):
    return pd.read_csv(subject_dir(task, subject) / "PPG data" / "ppg.csv").iloc[:, 0].to_numpy(float)


def load_participant(task, subject):
    """EMG + IMU + PPG for one participant and one task."""
    return {"emg": load_emg(task, subject), "imu": load_imu(task, subject),
            "ppg": load_ppg(task, subject)}


def load_borg():
    """Long Borg table: subject, task, t_sec, rpe, plus per-trial duration.

    borg_data.csv only names the subject on the first row of each block, so it is forward-filled.
    Column names carry stray spaces ('120_sec '); they are stripped.
    """
    df = pd.read_csv(sub("Borg*") / "borg_data.csv")
    df.columns = [c.strip() for c in df.columns]
    df["subject"] = df["subject"].ffill()
    df["sid"] = df["subject"].str.extract(r"(\d+)").astype(int)
    rows = []
    t_cols = [c for c in df.columns if re.fullmatch(r"\d+_sec", c)]
    for _, r in df.iterrows():
        dur = r["length_of_trial_(sec)"]
        pts = [(0, r["before_task"])] + [(int(c.split("_")[0]), r[c]) for c in t_cols]
        for t, v in pts:
            if pd.notna(v):
                rows.append((r["sid"], r["task_order"], t, float(v), dur))
        rows.append((r["sid"], r["task_order"], dur, float(r["end_of_trial"]), dur))
    long = pd.DataFrame(rows, columns=["sid", "task", "t_sec", "rpe", "trial_sec"])
    return df, long


def load_demographics():
    demo = pd.read_csv(sub("Demographic*") / "demographic.csv")      # folder is misspelled in some releases
    demo["sid"] = demo["subject"].str.extract(r"(\d+)").astype(int)
    return demo


if __name__ == "__main__":
    s, t = int(sys.argv[1]), sys.argv[2]
    p = load_participant(t, s)
    print(f"subject {s} {t}: EMG {len(next(iter(p['emg'].values())))} samples, "
          f"IMU {len(next(iter(p['imu'].values())))} samples, PPG {len(p['ppg'])} samples")
