"""Module 8.1(d): per-person facts from past sessions, as a source-tagged knowledge chunk.

Built from the dataset pipeline's own tables (reports/dataset_modules_1_7): the person's usual
trial lengths and effort build-up, their early-session baseline values and their strength. Every number
is registered with the session summary (``allowed_numbers``), so the number checker can verify it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.dataset_pipeline import common as c
from ml.rag.knowledge_retriever import KnowledgeChunk

TASK_NAMES = {
    "task1_35i": "internal rotation 30-40% of max force", "task2_45i": "internal rotation 40-50%",
    "task3_55i": "internal rotation 50-60%", "task4_35e": "external rotation 30-40%",
    "task5_45e": "external rotation 40-50%", "task6_55e": "external rotation 50-60%",
}


@lru_cache(maxsize=1)
def _tables() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    try:
        inv = pd.read_csv(c.OUT / "m1_inventory.csv")
        part = pd.read_csv(c.OUT / "m1_participant_summary.csv")
        feats = pd.read_csv(c.OUT / "m4_features.csv", usecols=["sid", "task", "win", "t_end", "rpe",
                                                                  "emg_infraspinatus_mdf", "hr_bpm"])
    except FileNotFoundError:
        return None
    return inv, part, feats


def usual_rate(sid: int) -> Optional[float]:
    """This person's usual effort build-up in Borg points per minute (median over their exercises), or None."""
    tables = _tables()
    if tables is None:
        return None
    f = tables[2][tables[2].sid == sid]
    slopes = [np.polyfit(g.t_end, g.rpe, 1)[0] * 60 for _, g in f.groupby("task") if len(g) >= 3]
    return float(np.median(slopes)) if slopes else None


def person_chunk(sid: int) -> Tuple[Optional[KnowledgeChunk], List[float]]:
    """(chunk, numbers it contains) for participant ``sid``; (None, []) if the pipeline tables are missing."""
    tables = _tables()
    if tables is None:
        return None, []
    inv, part, feats = tables
    mine = inv[inv.sid == sid]
    if mine.empty:
        return None, []
    p = part[part.sid == sid].iloc[0]
    nums: List[float] = []
    lines = [f"Facts from this athlete's earlier recorded sessions (participant {sid}):"]
    lens = mine.borg_trial_sec.dropna()
    lines.append(f"- Usual time to reach maximal effort across the six exercises: median {lens.median():.0f} seconds "
                 f"(shortest {lens.min():.0f}, longest {lens.max():.0f}).")
    nums += [lens.median(), lens.min(), lens.max()]
    f = feats[feats.sid == sid]
    slope = (f.groupby("task").apply(lambda g: np.polyfit(g.t_end, g.rpe, 1)[0] * 60 if len(g) >= 3 else np.nan,
                                     include_groups=False)).dropna()
    if len(slope):
        lines.append(f"- Usual effort build-up: about {slope.median():.1f} Borg points per minute (median over exercises).")
        nums.append(slope.median())
    base = f[f.win < 2]
    if base["emg_infraspinatus_mdf"].notna().any():
        v = float(base["emg_infraspinatus_mdf"].mean())
        lines.append(f"- Starting infraspinatus muscle median frequency in the first 20 seconds: {v:.0f} Hz.")
        nums.append(v)
    if base["hr_bpm"].notna().any():
        v = float(base["hr_bpm"].mean())
        lines.append(f"- Starting pulse in the first 20 seconds: {v:.0f} beats per minute.")
        nums.append(v)
    for col, label in (("IR_MVIC_mean_(N)", "internal"), ("ER_MVIC_mean_(N)", "external")):
        if col in p and pd.notna(p[col]):
            lines.append(f"- Maximal {label} rotation strength: {float(p[col]):.0f} newtons.")
            nums.append(float(p[col]))
    chunk = KnowledgeChunk(
        chunk_id=f"PERSON-P{sid:02d}", title=f"Personal history of participant {sid}", category="person",
        source="earlier recorded sessions (WSD4FEDSRM pipeline tables)", date="session history",
        body="\n".join(lines), tags=["person", "baseline", "history"],
    )
    return chunk, [round(float(n), 2) for n in nums]
