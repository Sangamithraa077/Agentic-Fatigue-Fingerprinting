"""Session lab data service (Modules 9-12) over the dataset pipeline's own outputs.

Reads the tables written by ``ml.dataset_pipeline`` (reports/dataset_modules_1_7/*.csv), so it works without the
raw dataset and loads instantly. Every number it returns is a stored pipeline output or a deterministic
function of one (smoothing, thresholds, error against the self-reported rating); nothing is generated.

* sessions(): one session = one participant x one exercise, 10 s windows, with
  - self-reported Borg RPE (ground truth, shown beside the estimate, never used to make it),
  - generalised estimate (trained on the other 33 people, out-of-sample),
  - personalised estimate (z-scored on the person's first K windows, out-of-sample),
  - EMA-smoothed index and the 15 / 18 alerts, feature trends against the person's own start,
    top drivers, data-quality flags.
* build_summary(): the structured SessionSummary the explanation layer is allowed to talk about (Module 9.1/9.2).
* ask(): "ask your data" - a fixed set of read-only lookups, never free-form queries (Module 12.10).
* report_markdown() / report_html(): the exportable session report (Modules 12.9 / 12.11).
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.dataset_pipeline import common as c
from ml.rag.feature_glossary import feature_label
from ml.rag.person_facts import person_chunk, usual_rate
from ml.rag.session_summary import SessionSummary, build_session_summary_from_pipeline

TASK_LABELS = {
    "task1_35i": "Internal rotation, 30-40% of max force", "task2_45i": "Internal rotation, 40-50% of max force",
    "task3_55i": "Internal rotation, 50-60% of max force", "task4_35e": "External rotation, 30-40% of max force",
    "task5_45e": "External rotation, 40-50% of max force", "task6_55e": "External rotation, 50-60% of max force",
}
WARN_LEVEL, STOP_LEVEL = 15.0, 18.0
EMA_ALPHA = 0.5
DEFAULT_K = 4                       # calibration windows (40 s) for the personalised estimate
VALID_MIN = 0.8
# Feature trends shown against the person's own start (all are *_rel columns: fractional change from baseline).
KEY_TRENDS = ["emg_infraspinatus_mdf_rel", "emg_upper_trapezius_mdf_rel", "emg_anterior_deltoid_rms_rel",
              "emg_pectoralis_major_rms_rel", "hr_bpm_rel", "imu_forearm_gyr_mean_rel", "imu_forearm_jerk_rel",
              "tempo_forearm_hz_rel"]
MODEL_NAMES = {
    "generalised": "Gradient boosting on muscle, movement and pulse features plus elapsed time and exercise "
                   "(trained on the other 33 people)",
    "personalised": "The same model, with each feature scaled to this person's own first {k} windows",
}


def _f(x: Any) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def ema(values: List[float], alpha: float = EMA_ALPHA) -> List[float]:
    out: List[float] = []
    for v in values:
        out.append(v if not out else alpha * v + (1 - alpha) * out[-1])
    return out


class PipelineStore:
    """Lazy, read-only access to the pipeline tables."""

    def __init__(self, directory: Optional[Path] = None):
        self.dir = Path(directory) if directory else c.OUT
        self._loaded = False
        self._lock = None

    # ------------------------------------------------------------------ loading
    def _read(self, name: str, **kw) -> Optional[pd.DataFrame]:
        p = self.dir / name
        return pd.read_csv(p, **kw) if p.exists() else None

    def _load(self) -> None:
        if self._loaded:
            return
        self.feats = self._read("m4_features.csv")
        self.live = self._read("m7_live_calibration_predictions.csv")
        self.windows = self._read("m3_windows.csv")
        self.quality = self._read("m3_quality.csv")
        self.inv = self._read("m1_inventory.csv")
        self.part = self._read("m1_participant_summary.csv")
        self.contrib = self._read("m8_driver_contributions.csv")
        self.live_results = self._read("m7_live_calibration_results.csv")
        self.ready = all(t is not None for t in (self.feats, self.live, self.windows, self.quality, self.inv, self.part))
        if self.ready:
            key = ["sid", "task", "win"]
            valid_cols = [x for x in self.windows.columns if x.startswith("valid_")]
            self.table = (self.feats.merge(self.live[[x for x in self.live.columns if x in key or x not in self.feats.columns]],
                                           on=key, how="inner")
                          .merge(self.windows[key + valid_cols], on=key, how="left")
                          .sort_values(key).reset_index(drop=True))
        self._loaded = True

    @property
    def available(self) -> bool:
        self._load()
        return bool(self.ready)

    def _require(self) -> None:
        if not self.available:
            raise FileNotFoundError(f"pipeline tables not found in {self.dir}; run the dataset pipeline "
                                    "(python -m ml.dataset_pipeline.m3_clean ... m7_personalised)")

    # ------------------------------------------------------------------ catalogue
    def participants(self) -> List[Dict[str, Any]]:
        self._require()
        q = self.quality.set_index("sid")
        out = []
        for _, p in self.part.iterrows():
            sid = int(p.sid)
            n = int((self.table.sid == sid).sum())
            out.append({"sid": sid, "id": f"P{sid:02d}", "age": _f(p.age), "sex": p.sex, "windows": n,
                        "quality_score": _f(q.quality_score.get(sid)), "problem_flags": int(p.n_problem_flags)})
        return out

    def sessions(self, sid: int) -> List[Dict[str, Any]]:
        self._require()
        t = self.table[self.table.sid == sid]
        out = []
        for task in c.TASKS:
            g = t[t.task == task]
            if g.empty:
                continue
            out.append({"task": task, "label": TASK_LABELS[task], "windows": int(len(g)),
                        "duration_s": float(g.t_end.max()), "peak_rpe": float(g.rpe.max())})
        return out

    # ------------------------------------------------------------------ one session
    def _rows(self, sid: int, task: str) -> pd.DataFrame:
        self._require()
        g = self.table[(self.table.sid == sid) & (self.table.task == task)].reset_index(drop=True)
        if g.empty:
            raise KeyError(f"no session for participant {sid}, {task}")
        return g

    def _valid_flags(self, row: pd.Series) -> List[str]:
        flags = []
        for col, v in row.items():
            if str(col).startswith("valid_") and pd.notna(v) and v < VALID_MIN:
                flags.append(f"{str(col)[6:].replace('_', ' ')} signal unreliable ({v * 100:.0f}% valid)")
        return flags

    def session(self, sid: int, task: str, k: int = DEFAULT_K) -> Dict[str, Any]:
        g = self._rows(sid, task)
        pers_col = f"personalised_K{k}"
        if pers_col not in g.columns:
            raise ValueError(f"calibration windows must be one of {[int(x[14:]) for x in g.columns if x.startswith('personalised_K')]}")
        gen = g.generalised.astype(float).tolist()
        per = g[pers_col].astype(float).tolist()
        sm_g, sm_p = ema(gen), ema(per)
        contrib: Dict[int, List[Dict[str, Any]]] = {}
        if self.contrib is not None:
            cc = self.contrib[(self.contrib.sid == sid) & (self.contrib.task == task)]
            for w, grp in cc.groupby("win"):
                contrib[int(w)] = [{"feature": r.feature, "label": feature_label(r.feature), "impact": round(float(r.impact), 2)}
                                   for r in grp.sort_values("rank").itertuples()][:3]
        wins = []
        for i, r in g.iterrows():
            wins.append({
                "win": int(r.win), "t": float(r.t_end), "self_report": float(r.rpe),
                "generalised": round(gen[i], 2), "personalised": round(per[i], 2),
                "smoothed_generalised": round(sm_g[i], 2), "smoothed_personalised": round(sm_p[i], 2),
                "baseline_forming": bool(r.win < 2), "calibrating": bool(r.win < k),
                "trends": {key: _f(r.get(key)) for key in KEY_TRENDS},
                "drivers": contrib.get(int(r.win), []),
                "quality_flags": self._valid_flags(r),
            })
        base = g[g.win < 2]
        raw_keys = [x[:-4] for x in KEY_TRENDS]
        post = g.win >= k
        s = {
            "windows": len(g), "duration_s": float(g.t_end.max()), "peak_self_report": float(g.rpe.max()),
            "mae_generalised": float(np.mean(np.abs(g.rpe - g.generalised))),
            "mae_generalised_after_calibration": float(np.mean(np.abs(g.rpe[post] - g.generalised[post]))) if post.any() else None,
            "mae_personalised_after_calibration": float(np.mean(np.abs(g.rpe[post] - g[pers_col][post]))) if post.any() else None,
            "warning_at_s": next((w["t"] for w in wins if w["smoothed_generalised"] >= WARN_LEVEL), None),
            "stop_at_s": next((w["t"] for w in wins if w["smoothed_generalised"] >= STOP_LEVEL), None),
        }
        # What testing on all 34 people showed for this calibration length (leave-one-person-out), so the page can
        # say honestly whether the personalised estimate is worth trusting over the generalised one.
        test = None
        if self.live_results is not None:
            row = self.live_results[self.live_results.calibration_windows == k]
            if len(row):
                r = row.iloc[0]
                test = {"calibration_seconds": int(r.calibration_seconds), "mae_generalised": float(r.MAE_generalised),
                        "mae_personalised": float(r.MAE_personalised), "people_improved": int(r.people_improved),
                        "n_people": int(r.n_people), "p_value": float(r.wilcoxon_p),
                        "significant_gain": bool(r.wilcoxon_p < 0.05 and r.MAE_personalised < r.MAE_generalised)}
        return {
            "personalisation_test": test,
            "sid": sid, "id": f"P{sid:02d}", "task": task, "label": TASK_LABELS[task], "calibration_windows": k,
            "window_seconds": 10, "levels": {"warning": WARN_LEVEL, "stop": STOP_LEVEL},
            "quality_score": _f(self.quality.set_index("sid").quality_score.get(sid)),
            "baseline": {key: _f(base[key].mean()) for key in raw_keys if key in base},
            "trend_labels": {key: feature_label(key) for key in KEY_TRENDS},
            "windows_data": wins, "summary": s, "model_names": {m: n.format(k=k) for m, n in MODEL_NAMES.items()},
        }

    # ------------------------------------------------------------------ Module 9.1 / 9.2
    def build_summary(self, sid: int, task: str, step: int, k: int = DEFAULT_K,
                      model: str = "generalised") -> Tuple[SessionSummary, Dict[str, Any]]:
        """Structured summary of the session up to window ``step``: the only source of numbers for the coach."""
        sess = self.session(sid, task, k)
        wins = sess["windows_data"]
        step = int(np.clip(step, 0, len(wins) - 1))
        use_p = model == "personalised" and step >= k
        key = "smoothed_personalised" if use_p else "smoothed_generalised"
        cur = wins[step]
        back = max(0, step - 3)
        prev = wins[back][key]
        span = max(10.0, (step - back) * 10.0)
        drivers = {d["feature"]: d["impact"] for d in cur["drivers"]}
        g = self._rows(sid, task).iloc[step]
        dev = {}
        for feat in list(drivers) + KEY_TRENDS:
            if feat.endswith("_rel") and feat not in dev and pd.notna(g.get(feat)):
                dev[feat] = round(float(g[feat]) * 100, 1)
            if len(dev) >= 4:
                break
        flags = list(cur["quality_flags"])
        if step < 2:
            flags.append("BASELINE_STILL_FORMING")
        pc, person_nums = person_chunk(sid)
        summary = build_session_summary_from_pipeline(
            person_id=f"P{sid:02d}", current_rpe=cur[key], previous_rpe=prev, time_span_seconds=span,
            shap_values=drivers, baseline_deviations=dev, quality_warnings=flags or None, is_calibrated=use_p,
            model_name=MODEL_NAMES["personalised" if use_p else "generalised"].format(k=k),
            usual_rate_per_min=usual_rate(sid),
        )
        summary.metadata.update({
            "elapsed_seconds": cur["t"], "baseline_unit": "percent", "exercise": sess["label"],
            "allowed_numbers": [cur["self_report"], cur["t"], WARN_LEVEL, STOP_LEVEL, *person_nums],
        })
        grounding = {"source": "pipeline_session", "participant": sess["id"], "task": task, "step": step,
                     "estimate": "personalised" if use_p else "generalised", "elapsed_s": cur["t"],
                     "self_report": cur["self_report"]}
        return summary, {"grounding": grounding, "person_chunk": pc}

    # ------------------------------------------------------------------ Module 12.10
    def ask(self, sid: int, task: str, question: str, k: int = DEFAULT_K) -> Dict[str, Any]:
        sess = self.session(sid, task, k)
        for qid, pattern, fn, _ in LOOKUPS:
            if pattern.search(question):
                text, data = fn(self, sess)
                return {"matched": qid, "answer": text, "data": data, "source": "read-only lookup on this session's stored results"}
        return {"matched": None, "answer": "I can only answer from a fixed set of questions about this session's results.",
                "data": None, "supported": [{"id": q, "example": ex} for q, _, _, ex in LOOKUPS]}

    def task_comparison(self, sid: int) -> List[Dict[str, Any]]:
        t = self.table[self.table.sid == sid]
        rows = []
        for task, g in t.groupby("task"):
            rows.append({"task": task, "label": TASK_LABELS[task], "duration_s": float(g.t_end.max()),
                         "peak": float(g.rpe.max()), "mae": float(np.mean(np.abs(g.rpe - g.generalised)))})
        return sorted(rows, key=lambda r: r["duration_s"])

    # ------------------------------------------------------------------ published results
    def results(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, name in (("generalised_models", "m5b_improved_results.csv"), ("personalisation", "m7_results.csv"),
                          ("live_calibration", "m7_live_calibration_results.csv"),
                          ("augmentation", "m6b_augmentation_boosting_results.csv"),
                          ("research_comparison", "comparison_with_research.csv")):
            df = self._read(name)
            out[key] = None if df is None else json_safe(df)
        return out


def json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
            for r in df.round(4).to_dict("records")]


# ---------------------------------------------------------------------------- ask-your-data lookups
def _lk_peak(store, s):
    w = max(s["windows_data"], key=lambda w: w["self_report"])
    return (f"The highest self-reported effort was {w['self_report']:.0f} out of 20, at {w['t']:.0f} seconds.",
            {"peak_self_report": w["self_report"], "at_s": w["t"]})


def _lk_threshold(store, s):
    sm = s["summary"]
    parts = []
    for name, v in (("warning level 15", sm["warning_at_s"]), ("stop level 18", sm["stop_at_s"])):
        parts.append(f"The smoothed estimate reached the {name} at {v:.0f} seconds." if v is not None
                     else f"The smoothed estimate never reached the {name}.")
    return " ".join(parts), {"warning_at_s": sm["warning_at_s"], "stop_at_s": sm["stop_at_s"]}


def _lk_accuracy(store, s):
    sm = s["summary"]
    return (f"Across this exercise the estimate was off from the self-reported rating by {sm['mae_generalised']:.1f} "
            f"Borg points on average (all {sm['windows']} windows).", {"mean_abs_error": sm["mae_generalised"]})


def _lk_models(store, s):
    sm = s["summary"]
    if sm["mae_personalised_after_calibration"] is None:
        return "This exercise is too short to compare the models after calibration.", None
    a, b = sm["mae_generalised_after_calibration"], sm["mae_personalised_after_calibration"]
    better = "personalised" if b < a else "generalised"
    return (f"After the {s['calibration_windows'] * 10}-second calibration, the generalised estimate was off by {a:.2f} points "
            f"and the personalised estimate by {b:.2f}, so the {better} model was closer in this session.",
            {"generalised": a, "personalised": b})


def _lk_change(store, s):
    last = s["windows_data"][-1]
    items = [(k, v) for k, v in last["trends"].items() if v is not None]
    if not items:
        return "No feature trends are available for this session.", None
    k, v = max(items, key=lambda kv: abs(kv[1]))
    return (f"By the end, the biggest change from this person's own start was in {s['trend_labels'][k]}: {v * 100:+.0f}%.",
            {"feature": k, "change_percent": v * 100})


def _lk_quality(store, s):
    flagged = [w for w in s["windows_data"] if w["quality_flags"]]
    q = s["quality_score"]
    base = f"The data-quality score for this participant is {q:.0f} out of 100. " if q is not None else ""
    return (base + (f"{len(flagged)} of {s['summary']['windows']} windows have an unreliable channel." if flagged
                    else f"All {s['summary']['windows']} windows have clean channels."), {"flagged_windows": len(flagged)})


def _lk_duration(store, s):
    return (f"This exercise lasted {s['summary']['duration_s']:.0f} seconds of recorded signal "
            f"({s['summary']['windows']} windows of 10 seconds).", {"duration_s": s["summary"]["duration_s"]})


def _lk_tasks(store, s):
    rows = store.task_comparison(s["sid"])
    longest = max(rows, key=lambda r: r["duration_s"])
    shortest = min(rows, key=lambda r: r["duration_s"])
    return (f"Longest before reaching maximal effort: {longest['label']} ({longest['duration_s']:.0f} s). "
            f"Shortest: {shortest['label']} ({shortest['duration_s']:.0f} s).", rows)


LOOKUPS: List[Tuple[str, "re.Pattern", Callable, str]] = [
    ("peak_effort", re.compile(r"\b(highest|peak|max(?:imum)?|top)\b.*\b(effort|rpe|rating|score)\b|\bhardest\b", re.I), _lk_peak,
     "What was the highest effort?"),
    ("threshold_times", re.compile(r"\b(when|how long|what time)\b.*\b(15|18|warning|stop|threshold|reach)", re.I), _lk_threshold,
     "When did effort reach the warning level?"),
    ("model_comparison", re.compile(r"personali[sz]ed.*general|general.*personali[sz]ed|which model|compare.*model", re.I), _lk_models,
     "Is the personalised model better than the generalised one?"),
    ("accuracy", re.compile(r"\b(accura|error|how close|off by|predicted vs|estimate.*(?:report|rating))", re.I), _lk_accuracy,
     "How close was the estimate to my own rating?"),
    ("biggest_change", re.compile(r"\b(which|what)\b.*\b(changed|change)\b|\bbiggest\b.*\b(change|shift)", re.I), _lk_change,
     "What changed the most from my start?"),
    ("data_quality", re.compile(r"\b(quality|clean|dropout|problem|reliable signal|artifact)", re.I), _lk_quality,
     "How good was the data quality?"),
    ("exercise_comparison", re.compile(r"\b(compare|which)\b.*\b(exercise|task|rotation)s?\b|\blongest\b|\bshortest\b", re.I), _lk_tasks,
     "Which exercise lasted longest?"),
    ("duration", re.compile(r"\b(how long|duration|lasted|length)\b", re.I), _lk_duration, "How long was this exercise?"),
]


# ---------------------------------------------------------------------------- report export (Modules 12.9 / 12.11)
def _svg_chart(sess: Dict[str, Any], width: int = 720, height: int = 260) -> str:
    wins = sess["windows_data"]
    ts = [w["t"] for w in wins]
    x0, x1 = 44, width - 14
    y0, y1 = height - 30, 12
    tmax = max(ts) or 1

    def X(t): return x0 + (x1 - x0) * t / tmax
    def Y(v): return y0 - (y0 - y1) * (np.clip(v, 6, 20) - 6) / 14

    def line(key, color, dash=""):
        pts = " ".join(f"{X(w['t']):.1f},{Y(w[key]):.1f}" for w in wins)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" {dash} points="{pts}"/>'
    grid = "".join(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#d9dde3"/>'
                   f'<text x="{x0 - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#555">{v}</text>'
                   for v in (6, 10, 14, 18, 20))
    lev = "".join(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{col}" stroke-dasharray="5 4"/>'
                  for v, col in ((WARN_LEVEL, "#d99a00"), (STOP_LEVEL, "#c0392b")))
    xl = "".join(f'<text x="{X(t):.1f}" y="{height - 10}" font-size="11" text-anchor="middle" fill="#555">{t:.0f}s</text>'
                 for t in np.linspace(0, tmax, 6))
    legend = ('<text x="52" y="26" font-size="12" fill="#111">self-reported</text>'
              '<text x="140" y="26" font-size="12" fill="#2563eb">generalised</text>'
              '<text x="222" y="26" font-size="12" fill="#0d9488">personalised</text>')
    dash = 'stroke-dasharray="6 3"'
    curves = line("self_report", "#111") + line("smoothed_generalised", "#2563eb") + line("smoothed_personalised", "#0d9488", dash)
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Estimated versus self-reported effort">'
            f'{grid}{lev}{curves}{xl}{legend}</svg>')


def report_markdown(store: PipelineStore, sid: int, task: str, k: int = DEFAULT_K, explanation: Optional[str] = None) -> str:
    s = store.session(sid, task, k)
    sm = s["summary"]
    lines = [f"# Session report: {s['id']}, {s['label']}", "",
             f"- Duration: {sm['duration_s']:.0f} s ({sm['windows']} windows of 10 s)",
             f"- Peak self-reported effort: {sm['peak_self_report']:.0f} out of 20",
             f"- Warning level (15) reached at: {'never' if sm['warning_at_s'] is None else str(int(sm['warning_at_s'])) + ' s'}",
             f"- Stop level (18) reached at: {'never' if sm['stop_at_s'] is None else str(int(sm['stop_at_s'])) + ' s'}",
             f"- Estimate error versus the self-report, generalised: {sm['mae_generalised']:.2f} Borg points on average"]
    if sm["mae_personalised_after_calibration"] is not None:
        lines.append(f"- After the {k * 10} s calibration: generalised {sm['mae_generalised_after_calibration']:.2f}, "
                     f"personalised {sm['mae_personalised_after_calibration']:.2f} Borg points")
    if s["quality_score"] is not None:
        lines.append(f"- Data-quality score: {s['quality_score']:.0f} out of 100")
    lines += ["", "## Effort over time", "", "| Time (s) | Self-reported | Generalised | Personalised |", "|---|---|---|---|"]
    lines += [f"| {w['t']:.0f} | {w['self_report']:.0f} | {w['smoothed_generalised']:.1f} | {w['smoothed_personalised']:.1f} |"
              for w in s["windows_data"]]
    last = s["windows_data"][-1]
    if last["drivers"]:
        lines += ["", "## What drove the final estimate", ""]
        lines += [f"- {d['label']}: {'raises' if d['impact'] >= 0 else 'lowers'} the estimate by {abs(d['impact']):.2f} Borg points"
                  for d in last["drivers"]]
    if explanation:
        lines += ["", "## Plain-language explanation", "", "*Generated text. Every number in it is checked against the figures above.*", "",
                  explanation]
    lines += ["", "*Research estimate, not a medical measurement. Typical error is about 1.7 Borg points "
              "(leave-one-person-out). For pain or health concerns, see a professional.*"]
    return "\n".join(lines) + "\n"


def report_html(store: PipelineStore, sid: int, task: str, k: int = DEFAULT_K, explanation: Optional[str] = None) -> str:
    s = store.session(sid, task, k)
    md = report_markdown(store, sid, task, k, explanation)
    body = []
    in_table = False
    for line in md.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_table: body.append("</table>"); in_table = False
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
            if line[3:].startswith("Effort over time"):
                body.append(_svg_chart(s))
        elif line.startswith("|---"):
            continue
        elif line.startswith("|"):
            cells = [html.escape(x.strip()) for x in line.strip("|").split("|")]
            tag = "th" if not in_table else "td"
            if not in_table: body.append("<table>"); in_table = True
            body.append("<tr>" + "".join(f"<{tag}>{x}</{tag}>" for x in cells) + "</tr>")
        elif line.startswith("- "):
            body.append(f"<p class='li'>{html.escape(line[2:])}</p>")
        elif line.strip():
            if in_table: body.append("</table>"); in_table = False
            body.append(f"<p>{html.escape(line.strip('*'))}</p>")
    if in_table: body.append("</table>")
    style = ("body{font:15px/1.5 system-ui,sans-serif;max-width:780px;margin:24px auto;padding:0 16px;color:#111}"
             "h1{font-size:22px}h2{font-size:17px;margin-top:22px}table{border-collapse:collapse;width:100%;font-size:13px}"
             "th,td{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left}.li{margin:2px 0}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Session report {s['id']}</title><style>{style}</style></head><body>{''.join(body)}</body></html>")


@lru_cache(maxsize=1)
def get_store() -> PipelineStore:
    return PipelineStore()
