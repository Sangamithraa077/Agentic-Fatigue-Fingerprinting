"""Module 13: evaluation of the explanation layer on real pipeline sessions.

13.1 faithfulness        50 explanations from real windows: share whose numbers all match the summary
     guardrail           same 50 windows, a language model that invents numbers: share caught and replaced
13.2 hallucination       questions the data cannot answer: share declined with no numbers or sources
13.3 retrieval quality   20 fixed questions: top-1 / top-3 hit rate, and share of answers citing a retrieved note
13.4 usefulness          rating sheet for 3-5 people (NOT filled in here; see usefulness_rating_sheet.csv)
13.5 cost and speed      always-call vs selective-call over all recorded sessions; delay per call
13.6 ablation            ML-only panel text vs ML + explanation: readability proxy (human ratings pending)

Output: reports/llm_eval/llm_layer_evaluation.json, llm_layer_evaluation.md, usefulness_rating_sheet.csv
"""

import json
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from backend.services import pipeline_service as svc
from ml.dataset_pipeline import common as c
from ml.rag.evaluator import LLMLayerEvaluator
from ml.rag.hybrid_router import HybridRouter
from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase
from ml.rag.number_checker import NumberChecker
from ml.rag.person_facts import usual_rate
from ml.rag.session_summary import build_session_summary_from_pipeline

OUT = c.REPO / "reports" / "llm_eval"
N_SAMPLES = 50
SEED = 7


def syllables(word: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", word.lower())))


def flesch(text: str) -> float:
    """Flesch reading ease (higher = easier). A readability proxy only, not a usefulness rating."""
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return float("nan")
    return 206.835 - 1.015 * len(words) / sents - 84.6 * sum(syllables(w) for w in words) / len(words)


def numeric_panel_text(summary) -> str:
    """What the ML-only dashboard shows as text: the numbers, no explanation."""
    drivers = "; ".join(f"{d['feature']} {d['impact']:+.2f}" for d in summary.top_contributing_features)
    return (f"Effort estimate {summary.current_fatigue_index:.1f} / 20. Trend {summary.fatigue_trend.lower().replace('_', ' ')} "
            f"({summary.rate_of_rpe_change_per_min:+.1f} per min). Drivers: {drivers or 'none'}.")


def sample_windows(store: svc.PipelineStore, n: int, seed: int):
    rng = np.random.default_rng(seed)
    keys = store.table[["sid", "task", "win"]].to_numpy()
    keys = keys[keys[:, 2] >= 2]
    return [keys[i] for i in rng.choice(len(keys), size=n, replace=False)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    store = svc.get_store()
    if not store.available:
        raise SystemExit("pipeline tables missing; run the dataset pipeline first")
    kb = LocalRAGKnowledgeBase()
    off = HybridRouter(knowledge_base=kb, use_ollama=False, use_anthropic=False)          # LLM layer switched off
    checker = NumberChecker()
    report = {"llm_backend_available": HybridRouter(knowledge_base=kb)._llm_backend}
    windows = sample_windows(store, N_SAMPLES, SEED)

    # ---- 13.1 faithfulness (template path always; language-model path when a backend is configured)
    live = HybridRouter(knowledge_base=kb)
    results = {"template": [], "llm": []}
    texts_ml_only, texts_ml_llm = [], []
    for sid, task, win in windows:
        summary, extra = store.build_summary(int(sid), str(task), int(win), model="personalised")
        chunks = [extra["person_chunk"]] if extra["person_chunk"] is not None else None
        r = off.explain(summary, force_trigger=True, extra_chunks=chunks)
        results["template"].append(checker.verify(r.answer, summary).is_valid)
        texts_ml_only.append(numeric_panel_text(summary))
        texts_ml_llm.append(numeric_panel_text(summary) + " " + r.answer)
        if live.custom_llm_call is not None:
            r2 = live.explain(summary, force_trigger=True, extra_chunks=chunks)
            results["llm"].append((checker.verify(r2.answer, summary).is_valid, r2.is_llm_generated, r2.execution_time_ms))
    report["faithfulness"] = {
        "n": N_SAMPLES, "windows": "random real windows (seed 7) from the pipeline sessions",
        "template_numbers_all_grounded_pct": 100 * float(np.mean(results["template"])),
        "llm": None if not results["llm"] else {
            "numbers_all_grounded_pct": 100 * float(np.mean([x[0] for x in results["llm"]])),
            "answers_written_by_model_pct": 100 * float(np.mean([x[1] for x in results["llm"]]))},
    }
    if not results["llm"]:
        report["faithfulness"]["llm_note"] = "no language model configured here (set ANTHROPIC_API_KEY or run Ollama); not measured"

    # ---- guardrail: a model that invents numbers must be caught and replaced by the verified template
    def lying_llm(prompt: str) -> str:
        return "Your pulse is 142 beats per minute and your muscle signal dropped 37 percent, so effort is 17.9 out of 20."
    adversary = HybridRouter(knowledge_base=kb, custom_llm_call=lying_llm)
    caught = 0
    for sid, task, win in windows:
        summary, _ = store.build_summary(int(sid), str(task), int(win))
        r = adversary.explain(summary, force_trigger=True)
        caught += int((not r.is_llm_generated) and r.fallback_reason == "ungrounded_numbers" and checker.verify(r.answer, summary).is_valid)
    report["guardrail"] = {"n": N_SAMPLES, "invented_numbers_caught_and_replaced_pct": 100 * caught / N_SAMPLES,
                           "method": "mock model that always states a pulse of 142 and a drop of 37%, which are in no summary"}

    # ---- 13.2 hallucination (questions the data cannot answer)
    ev = LLMLayerEvaluator(kb)
    pct, passed, total = ev.evaluate_hallucination_robustness()
    report["hallucination"] = {"declined_pct": pct, "passed": passed, "total": total}

    # ---- 13.3 retrieval quality
    r1, r3 = ev.evaluate_retrieval_quality()
    cited = 0
    for sid, task, win in windows[:20]:
        summary, _ = store.build_summary(int(sid), str(task), int(win))
        cited += int(bool(off.explain(summary, force_trigger=True).retrieved_sources))
    report["retrieval"] = {"top1_pct": r1, "top3_pct": r3, "queries": ev.RETRIEVAL_TEST_QUERY_COUNT, "backend": kb.retrieval_backend,
                           "answers_citing_a_note_pct": 100 * cited / 20,
                           "caveat": "query expansion terms were tuned after seeing misses on these questions: optimistic, not held-out"}

    # ---- 13.5 cost and speed: always-call vs selective-call over every recorded session
    total_w, fired, by_trigger = 0, 0, {}
    for sid in range(1, 35):
        ur = usual_rate(sid)                    # trend is judged against this person's own usual build-up
        for sess in store.sessions(sid):
            s = store.session(sid, sess["task"])
            sm = [w["smoothed_personalised"] if w["win"] >= s["calibration_windows"] else w["smoothed_generalised"]
                  for w in s["windows_data"]]
            for i in range(len(sm)):
                prev = sm[max(0, i - 3)]
                summ = build_session_summary_from_pipeline("P", sm[i], prev, max(10.0, (i - max(0, i - 3)) * 10.0),
                                                           usual_rate_per_min=ur)
                ok, why = off.should_trigger_explanation(summ, session_concluded=(i == len(sm) - 1))
                total_w += 1
                fired += int(ok)
                by_trigger[why] = by_trigger.get(why, 0) + int(ok)
    lat = []
    summary, extra = store.build_summary(5, "task1_35i", 6)
    for _ in range(30):
        t0 = time.perf_counter()
        off.explain(summary, force_trigger=True)
        lat.append((time.perf_counter() - t0) * 1000)
    prompt_len = len(off._build_user_prompt(summary, kb.search("fatigue", top_k=3) and [x for x, _ in kb.search("fatigue", top_k=3)], None, "why"))
    report["cost_and_speed"] = {
        "windows_in_all_recorded_sessions": total_w, "windows_where_a_call_fires": fired,
        "selective_call_share_pct": 100 * fired / total_w, "calls_saved_vs_always_call_pct": 100 * (1 - fired / total_w),
        "fires_by_trigger": by_trigger,
        "template_delay_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))},
        "language_model_delay_ms": None if not results["llm"] else {
            "median": float(np.median([x[2] for x in results["llm"]])), "p95": float(np.percentile([x[2] for x in results["llm"]], 95))},
        "language_model_delay_note": None if results["llm"] else "not measured: no language model configured in this run",
        "prompt_characters_per_call": prompt_len, "approx_prompt_tokens_per_call": prompt_len // 4,
        "note": "cost per call = tokens x the chosen model's price; prices are not hard-coded here",
    }

    # ---- 13.6 ablation: readability proxy, ML-only vs ML + explanation
    report["ablation"] = {
        "ml_only": {"flesch_reading_ease_mean": float(np.nanmean([flesch(t) for t in texts_ml_only])),
                    "words_mean": float(np.mean([len(t.split()) for t in texts_ml_only]))},
        "ml_plus_explanation": {"flesch_reading_ease_mean": float(np.nanmean([flesch(t) for t in texts_ml_llm])),
                                "words_mean": float(np.mean([len(t.split()) for t in texts_ml_llm]))},
        "note": "readability and length are proxies. Clarity and helpfulness ratings from people are still to be collected (13.4).",
    }

    # ---- 9.6 explanations for 10 recorded sessions, written out to be read against their numbers
    doc = ["# Ten recorded sessions: numbers beside the explanation (Module 9.6)", "",
           "Each block is the end of a real exercise. Read the explanation against the numbers above it.", ""]
    picks = [(3, "task2_45i"), (7, "task1_35i"), (11, "task4_35e"), (14, "task6_55e"), (19, "task3_55i"),
             (22, "task5_45e"), (26, "task1_35i"), (29, "task2_45i"), (31, "task4_35e"), (34, "task3_55i")]
    for sid, task in picks:
        n = len(store.session(sid, task)["windows_data"])
        for step in (max(1, n // 2), n - 1):
            summary, extra = store.build_summary(sid, task, step, model="personalised")
            chunks = [extra["person_chunk"]] if extra["person_chunk"] is not None else None
            resp = off.explain(summary, force_trigger=True, extra_chunks=chunks)
            ok = checker.verify(resp.answer, summary)
            s = summary
            doc += [f"## P{sid:02d}, {summary.metadata['exercise']}, {s.metadata['elapsed_seconds']:.0f} s "
                    f"({'end' if step == n - 1 else 'middle'} of the exercise)", "",
                    f"- Estimate {s.current_fatigue_index:.1f} out of 20, trend {s.fatigue_trend.lower().replace('_', ' ')} "
                    f"({s.rate_of_rpe_change_per_min:+.1f} per minute); athlete's own rating {extra['grounding']['self_report']:.0f}",
                    f"- Drivers: " + (", ".join(f"{d['feature']} {d['impact']:+.2f}" for d in s.top_contributing_features) or "none yet"),
                    f"- Data quality: {', '.join(s.data_quality_flags)}", "", f"> {resp.answer}", "",
                    f"Number check: {'all numbers grounded' if ok.is_valid else 'UNGROUNDED ' + str(ok.ungrounded_numbers)}; "
                    f"sources offered: {', '.join(x.split(':')[0].lstrip('[') for x in resp.retrieved_sources) or 'none'}", ""]
    (OUT / "ten_session_explanations.md").write_text("\n".join(doc) + "\n", encoding="utf-8")

    # ---- 13.4 rating sheet for human raters (empty on purpose)
    rows = []
    for i, (sid, task, win) in enumerate(windows[:12], 1):
        summary, extra = store.build_summary(int(sid), str(task), int(win), model="personalised")
        chunks = [extra["person_chunk"]] if extra["person_chunk"] is not None else None
        rows.append({"item": i, "participant": f"P{int(sid):02d}", "exercise": str(task), "window": int(win),
                     "A_numbers_only": numeric_panel_text(summary),
                     "B_numbers_plus_explanation": off.explain(summary, force_trigger=True, extra_chunks=chunks).answer,
                     "rater_id": "", "A_clarity_1to5": "", "A_helpfulness_1to5": "", "B_clarity_1to5": "", "B_helpfulness_1to5": "",
                     "prefers_A_or_B": "", "comments": ""})
    pd.DataFrame(rows).to_csv(OUT / "usefulness_rating_sheet.csv", index=False)
    report["usefulness"] = {"status": "not collected", "how": "give usefulness_rating_sheet.csv to 3-5 people, then run "
                                                              "python -m ml.dataset_pipeline.aggregate_ratings"}

    (OUT / "llm_layer_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    f, g, h, rt, cs, ab = (report[k] for k in ("faithfulness", "guardrail", "hallucination", "retrieval", "cost_and_speed", "ablation"))
    md = [
        "# Explanation layer evaluation (Module 13)", "",
        f"Language model backend in this run: **{report['llm_backend_available']}** (the layer works with it switched off).", "",
        "| Test | Result |", "|---|---|",
        f"| 13.1 Faithfulness, rule-based text, {f['n']} real windows | {f['template_numbers_all_grounded_pct']:.0f}% have every number grounded |",
        f"| 13.1 Faithfulness, language model | {'not measured (no model configured)' if f['llm'] is None else str(round(f['llm']['numbers_all_grounded_pct'])) + '% grounded'} |",
        f"| Guardrail, a model that invents numbers | {g['invented_numbers_caught_and_replaced_pct']:.0f}% caught and replaced |",
        f"| 13.2 Questions the data cannot answer | {h['passed']}/{h['total']} declined with no figures |",
        f"| 13.3 Retrieval, top-1 / top-3 | {rt['top1_pct']:.0f}% / {rt['top3_pct']:.0f}% on {rt['queries']} questions (optimistic: terms tuned on these) |",
        f"| 13.3 Answers citing a retrieved note | {rt['answers_citing_a_note_pct']:.0f}% |",
        f"| 13.5 Selective calls | {cs['windows_where_a_call_fires']} of {cs['windows_in_all_recorded_sessions']} windows ({cs['selective_call_share_pct']:.0f}%); "
        f"{cs['calls_saved_vs_always_call_pct']:.0f}% fewer calls than always-call |",
        f"| 13.5 Delay, rule-based answer | median {cs['template_delay_ms']['median']:.0f} ms, p95 {cs['template_delay_ms']['p95']:.0f} ms |",
        f"| 13.5 Delay, language model | {cs['language_model_delay_note'] or 'median %.0f ms' % cs['language_model_delay_ms']['median']} |",
        f"| 13.6 Readability (Flesch), ML only vs ML + explanation | {ab['ml_only']['flesch_reading_ease_mean']:.0f} vs {ab['ml_plus_explanation']['flesch_reading_ease_mean']:.0f} (proxy) |",
        "| 13.4 Usefulness ratings from people | **not collected yet** |", "",
        "## Where it fails or is unmeasured",
        "- Usefulness (13.4) needs 3 to 5 raters; the sheet is generated but empty. The ablation (13.6) is therefore a readability proxy only.",
        "- The number checker also accepts any whole number 0 to 20, 50 and 100, so an invented count in that range would pass.",
        "- Retrieval questions were used to tune query expansion, so the retrieval score is optimistic.",
        "- Language-model delay and quality are unmeasured here without an API key or a local model.", "",
        f"Fires by trigger: {cs['fires_by_trigger']}",
    ]
    (OUT / "llm_layer_evaluation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
