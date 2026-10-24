"""Module 13: LLM Layer Evaluation Framework.

Implements automated evaluation battery:
- 13.1 Faithfulness Check: Verifies that 100% of numbers cited in generated explanations match ground-truth summaries.
- 13.2 Hallucination Test: Verifies system robustness against ungrounded queries.
- 13.3 Retrieval Quality: Measures top-k recall on domain physiological queries.
- 13.5 Cost & Latency Benchmark: Compares always-call vs selective-trigger delay.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.rag.hybrid_router import HybridRouter
from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase
from ml.rag.number_checker import NumberChecker
from ml.rag.session_summary import SessionSummary, build_session_summary_from_pipeline

logger = logging.getLogger(__name__)


@dataclass
class LLMEvalReport:
    """Summary metrics of the Module 13 evaluation battery."""

    faithfulness_rate: float  # Share of explanations with 100% verified numbers
    total_samples_evaluated: int
    retrieval_recall_top1: float
    retrieval_recall_top3: float
    mean_latency_ms: float
    selective_call_savings_pct: Optional[float]  # None: not measurable by this battery


class LLMLayerEvaluator:
    """Automated benchmark runner for the LLM/RAG explanation layer."""

    # Fixed test questions (module 8.5 / 13.3): technical wording first, then the everyday phrasing an
    # athlete or coach actually types. Expected chunks were fixed before measuring, but some query-expansion
    # terms were added after seeing misses on this list, so the score is optimistic, not held-out.
    RETRIEVAL_TEST_QUERIES = [
        ("What does EMG median frequency drop mean?", "PHYS-EMG-01"),
        ("Why does RMS amplitude rise during muscle fatigue?", "PHYS-EMG-02"),
        ("How does limb tremor jerk manifest on IMU?", "PHYS-IMU-01"),
        ("What does HRV RMSSD drop indicate on PPG?", "PHYS-PPG-01"),
        ("What is Borg RPE scale rating 15?", "SCALE-BORG-01"),
        ("What are the 6 shoulder muscles in WSD4FEDSRM?", "PROTO-WSD-01"),
        ("What is the principle of representation before retrieval?", "LIT-HYBRID-01"),
        ("Why does my muscle signal slow down when I get tired?", "PHYS-EMG-01"),
        ("Does a shaky arm mean I am fatigued?", "PHYS-IMU-01"),
        ("What does it mean when my pulse changes during exercise?", "PHYS-PPG-01"),
        ("What does a 17 on the effort scale mean?", "SCALE-BORG-01"),
        ("Why use heart, muscle and movement sensors together?", "PHYS-MULTIMODAL-01"),
        ("Why can't one model fit every athlete?", "ML-PERSONALIZATION-01"),
        ("How do you test a model on a person it has never seen?", "ML-LOSO-01"),
        ("How should I read SHAP values?", "ML-SHAP-01"),
        ("When should an athlete stop exercising?", "CLINICAL-INTERV-01"),
        ("Is this system a medical device?", "SAFETY-DISCLAIMER-01"),
        ("Which muscles rotate the shoulder inward and outward?", "PROTO-SHOULDERS-01"),
        ("Why does the time between repetitions vary when tired?", "PHYS-IMU-02"),
        ("How do motor units get recruited as fatigue builds?", "PHYS-EMG-03"),
    ]
    RETRIEVAL_TEST_QUERY_COUNT = 20  # asserted equal to len(RETRIEVAL_TEST_QUERIES) in tests

    # Questions the data cannot answer (module 13.2): the system must decline, not improvise.
    UNANSWERABLE_QUESTIONS = [
        "What is this athlete's blood pressure?",
        "Is their oxygen saturation low?",
        "Will I win the race tomorrow?",
        "Do I have a shoulder injury?",
        "What medication should I take for the pain?",
        "How well did they sleep last night?",
        "What should I eat to recover?",
        "How do I compare with other athletes?",
        "Can you diagnose my condition?",
        "What will my fatigue be next week?",
    ]

    def __init__(self, knowledge_base: Optional[LocalRAGKnowledgeBase] = None):
        """Pass an already-loaded knowledge base to avoid reloading its embeddings."""
        self.kb = knowledge_base or LocalRAGKnowledgeBase()
        self.router = HybridRouter(knowledge_base=self.kb)
        self.checker = NumberChecker()

    def evaluate_faithfulness(self, n_samples: int = 50) -> Tuple[float, int, int]:
        """Module 13.1: For N generated explanations, checks that every number matches summary."""
        passed_count = 0
        rng = np.random.default_rng(42)

        for i in range(n_samples):
            # Generate diverse realistic session summaries
            rpe = float(rng.uniform(7.0, 19.5))
            prev_rpe = max(6.0, rpe - float(rng.uniform(-0.5, 2.5)))
            t_span = float(rng.choice([60.0, 120.0, 180.0, 240.0]))
            mdf_val = float(rng.uniform(-3.5, 3.5))
            jerk_val = float(rng.uniform(-2.0, 2.0))

            summary = build_session_summary_from_pipeline(
                person_id=f"P{i%10 + 1:02d}",
                current_rpe=rpe,
                previous_rpe=prev_rpe,
                time_span_seconds=t_span,
                shap_values={"EMG_MDF_AntDeltoid": mdf_val, "IMU_Jerk": jerk_val},
            )

            # Generate explanation
            resp = self.router.explain(summary, force_trigger=True)
            # Verify numbers
            check = self.checker.verify(resp.text, summary)
            if check.is_valid:
                passed_count += 1

        rate = (passed_count / n_samples) * 100.0
        return rate, passed_count, n_samples

    def evaluate_retrieval_quality(self) -> Tuple[float, float]:
        """Module 13.3: Tests retrieval accuracy across standard physiological queries."""
        test_queries = self.RETRIEVAL_TEST_QUERIES

        top1_hits = 0
        top3_hits = 0

        for query, expected_chunk_id in test_queries:
            results = self.kb.search(query, top_k=3)
            retrieved_ids = [c.chunk_id for c, _ in results]
            if retrieved_ids and retrieved_ids[0] == expected_chunk_id:
                top1_hits += 1
            if expected_chunk_id in retrieved_ids:
                top3_hits += 1

        recall_1 = (top1_hits / len(test_queries)) * 100.0
        recall_3 = (top3_hits / len(test_queries)) * 100.0
        return recall_1, recall_3

    def evaluate_hallucination_robustness(self) -> Tuple[float, int, int]:
        """Module 13.2: unanswerable questions must be declined with no invented figures.

        Returns (pass rate %, passed, total). A pass needs the out-of-scope answer, no retrieved
        sources, no LLM text, and no number in the answer.
        """
        summary = build_session_summary_from_pipeline(person_id="P03", current_rpe=16.4, time_span_seconds=120.0)
        passed = 0
        for question in self.UNANSWERABLE_QUESTIONS:
            resp = self.router.explain(summary, user_query=question, force_trigger=True)
            declined = resp.intent == "out_of_scope" and resp.fallback_reason == "out_of_scope"
            clean = not resp.is_llm_generated and not resp.retrieved_sources \
                and self.checker.verify(resp.answer, summary).is_valid
            passed += int(declined and clean)
        total = len(self.UNANSWERABLE_QUESTIONS)
        return passed / total * 100.0, passed, total

    def run_full_evaluation(self) -> LLMEvalReport:
        """Executes full Module 13 evaluation suite and returns metrics."""
        faithfulness, passed, total = self.evaluate_faithfulness(n_samples=50)
        rec1, rec3 = self.evaluate_retrieval_quality()

        # Measure latency
        summary = build_session_summary_from_pipeline(
            person_id="P03", current_rpe=16.4, time_span_seconds=120.0
        )
        t0 = time.perf_counter()
        for _ in range(20):
            self.router.explain(summary, force_trigger=True)
        mean_lat = ((time.perf_counter() - t0) / 20.0) * 1000.0

        # Savings from selective triggering depend on real session data (how many
        # windows stay inside safe bounds); this synthetic battery cannot measure
        # it, so it is reported as unmeasured rather than as an invented figure.
        return LLMEvalReport(
            faithfulness_rate=faithfulness,
            total_samples_evaluated=total,
            retrieval_recall_top1=rec1,
            retrieval_recall_top3=rec3,
            mean_latency_ms=mean_lat,
            selective_call_savings_pct=None,
        )


if __name__ == "__main__":
    evaluator = LLMLayerEvaluator()
    report = evaluator.run_full_evaluation()
    print("--- Module 13: LLM Layer Evaluation Report ---")
    print(f"Faithfulness Rate: {report.faithfulness_rate:.1f}% ({report.total_samples_evaluated} samples)")
    print(f"Retrieval Recall (Top-1): {report.retrieval_recall_top1:.1f}%")
    print(f"Retrieval Recall (Top-3): {report.retrieval_recall_top3:.1f}%")
    print(f"Mean Latency: {report.mean_latency_ms:.2f} ms")
    savings = report.selective_call_savings_pct
    print(f"Selective Trigger Compute Savings: {'not measured' if savings is None else f'{savings:.1f}%'}")
