"""Module 14.2 and 14.5: end-to-end run driven by the hardware simulator, with and without the language model.

    python -m ml.dataset_pipeline.e2e_run            # runs three modes and writes the report

Modes
  off    explanation layer without any language model: rule-based text only
  mock   a scripted stand-in for a language model (plumbing check: proves the verifier, the background thread and
         the log work; it says nothing about the quality of a real model). A real model is used instead if
         ANTHROPIC_API_KEY / Ollama is configured and ``--real`` is passed.
  lying  a stand-in that invents numbers: the number checker must catch every one

The simulator is synthetic: this run validates the software path (rates, units, faults, latency, alerts,
explanations), not accuracy on people.

Output: reports/dataset_modules_1_7/m14_e2e_report.json and .md
"""

import json
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.live import ExplanationService, LivePipeline, SyntheticSource, load_models, validate_chunk
from ml.rag.hybrid_router import HybridRouter
from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase

SECONDS = 160
FAULTS = [{"t": (60, 80), "stream": "emg", "channel": 0, "kind": "dropout"}]
LOG_DIR = c.REPO / "reports" / "sessions"


def mock_llm(prompt: str) -> str:
    """Writes a short answer from the FACTS block only, after a delay like a real call would have."""
    time.sleep(0.3)
    m = re.search(r"Effort estimate \(Borg 6-20 scale\): ([\d.]+) \((.+?)\)", prompt)
    trend = re.search(r"Trend: (.+?) \(", prompt)
    return (f"Effort is {m.group(2)} at {m.group(1)} out of 20 and {trend.group(1) if trend else 'steady'}. "
            "Keep watching it over the next minute.")


def lying_llm(prompt: str) -> str:
    return "Your pulse reached 171 beats per minute and your muscle signal fell 63 percent."


def run_mode(mode: str, kb, real: bool = False) -> dict:
    pop, zm = load_models(4)
    log = LOG_DIR / f"e2e_{mode}.jsonl"
    call_log = LOG_DIR / f"e2e_{mode}_llm_calls.jsonl"
    for p in (log, call_log):
        p.unlink(missing_ok=True)
    kw = {"off": dict(use_ollama=False, use_anthropic=False),
          "mock": dict(custom_llm_call=mock_llm), "lying": dict(custom_llm_call=lying_llm)}[mode]
    if mode == "mock" and real:
        kw = {}
    router = HybridRouter(knowledge_base=kb, call_log_path=call_log, **kw)
    explainer = ExplanationService(router, usual_rate_per_min=10.0)     # the simulated person builds effort at ~10 points/min
    pipe = LivePipeline(pop, zm, condition={"load_pct": 35, "direction": "internal", "mvic_dir_N": 60.0},
                        calibration_windows=4, session_log=log, explainer=explainer, person_id="SIM")
    src = SyntheticSource(seconds=SECONDS, seed=3, faults=FAULTS)
    truth, issues, push_ms = [], 0, []
    t_loop = time.perf_counter()
    for ch in src.chunks():
        issues += len(validate_chunk(ch))
        t0 = time.perf_counter()
        res = pipe.push(ch)
        if res is not None:
            push_ms.append((time.perf_counter() - t0) * 1000)
            truth.append(ch.meta["true_effort"])
    loop_s = time.perf_counter() - t_loop
    explainer.wait()
    explainer.shutdown()
    est = [r for r in pipe.results if r["smoothed_generalised"] is not None]
    t = np.array([r["t"] for r in est])
    e_pers = np.array([r["smoothed_personalised"] for r in est])
    tr = np.array(truth)[-len(est):]
    late = t >= 40
    fault_flags = [r["win"] for r in pipe.results if any("emg_anterior_deltoid" in f for f in r["quality_flags"])]
    calls = [json.loads(x) for x in call_log.read_text().splitlines()] if call_log.exists() else []
    out = {
        "mode": mode, "seconds": SECONDS, "windows": len(pipe.results), "contract_issues_in_chunks": issues,
        "first_estimate_s": float(t[0]) if len(t) else None,
        "estimate_vs_simulated_effort": {"pearson_r": float(np.corrcoef(e_pers[late], tr[late])[0, 1]),
                                          "mean_abs_diff_borg_points_after_40s": float(np.mean(np.abs(e_pers[late] - tr[late])))},
        "alerts": pipe.alerts, "dropout_windows_flagged": fault_flags,
        "window_processing_ms": {"mean": float(np.mean(pipe.latencies_ms)), "p95": float(np.percentile(pipe.latencies_ms, 95)),
                                 "max": float(np.max(pipe.latencies_ms)), "budget_ms": 10000},
        "loop_never_waited_for_explanations": bool(max(push_ms) < 250),
        "max_push_ms_including_explainer_submit": float(max(push_ms)),
        "explanations": {"triggered": len(explainer.history), "ml_only_windows": explainer.skipped,
                         "from_language_model": sum(h["is_llm"] for h in explainer.history),
                         "fell_back": sum(1 for h in explainer.history if not h["is_llm"]),
                         "fallback_reasons": sorted({h["fallback_reason"] for h in explainer.history if h["fallback_reason"]}),
                         "latest": explainer.latest},
        "logged_calls": len(calls), "calls_with_all_numbers_grounded": sum(1 for x in calls if x["number_check"]["passed"]),
        "calibration_final": pipe.calibration_state, "session_log": str(log.relative_to(c.REPO)),
        "wall_seconds": round(loop_s, 2),
    }
    return out


def main() -> None:
    real = "--real" in sys.argv
    kb = LocalRAGKnowledgeBase()
    pop, zm = load_models(4)          # warm-up: load and first-call costs of the models are not per-window latency
    warm = LivePipeline(pop, zm, condition={"load_pct": 35, "direction": "internal", "mvic_dir_N": 60.0})
    for ch in SyntheticSource(seconds=30, seed=0).chunks():
        warm.push(ch)
    runs = [run_mode(m, kb, real) for m in ("off", "mock", "lying")]
    rep = {"simulator": "synthetic (not recorded data)", "runs": runs}
    (c.OUT / "m14_e2e_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    md = ["# End-to-end run with the hardware simulator (Modules 14.2 and 14.5)", "",
          "Synthetic signals in the dataset's units and rates feed the live pipeline; this validates the software path, "
          "not accuracy on people.", "",
          "| | LLM off | mock LLM | lying LLM |", "|---|---|---|---|"]
    def row(label, f): md.append(f"| {label} | " + " | ".join(str(f(r)) for r in runs) + " |")
    row("windows processed", lambda r: r["windows"])
    row("first estimate at (s)", lambda r: r["first_estimate_s"])
    row("estimate vs simulated effort, r", lambda r: round(r["estimate_vs_simulated_effort"]["pearson_r"], 3))
    row("window processing, mean / p95 / max (ms; budget 10000)", lambda r: "%.0f / %.0f / %.0f" % (
        r["window_processing_ms"]["mean"], r["window_processing_ms"]["p95"], r["window_processing_ms"]["max"]))
    row("alerts", lambda r: ", ".join(f"{a['level']}@{a['t']:.0f}s" for a in r["alerts"]) or "none")
    row("EMG dropout flagged in windows", lambda r: r["dropout_windows_flagged"])
    row("loop waited for explanations?", lambda r: "no" if r["loop_never_waited_for_explanations"] else "YES")
    row("explanations (model / fell back)", lambda r: f"{r['explanations']['from_language_model']} / {r['explanations']['fell_back']}")
    row("fallback reasons", lambda r: ", ".join(r["explanations"]["fallback_reasons"]) or "-")
    row("logged calls, all numbers grounded", lambda r: f"{r['calls_with_all_numbers_grounded']}/{r['logged_calls']}")
    (c.OUT / "m14_e2e_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
