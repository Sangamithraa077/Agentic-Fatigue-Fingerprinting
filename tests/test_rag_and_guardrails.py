"""Unit tests for Modules 8, 9, 10: RAG Knowledge Base, Session Summary, and Number Verifier."""

import pytest

from ml.rag.hybrid_router import SAFETY_DISCLAIMER, HybridRouter
from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase
from ml.rag.number_checker import NumberChecker
from ml.rag.session_summary import SessionSummary, build_session_summary_from_pipeline


def test_local_rag_knowledge_base_loading_and_search():
    kb = LocalRAGKnowledgeBase()
    assert len(kb.chunks) >= 4

    # Search for EMG spectral shift
    results = kb.search("EMG median frequency spectral shift", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert score > 0.1
    assert "EMG" in top_chunk.title or "EMG" in top_chunk.body


def test_session_summary_allowable_numbers():
    summary = build_session_summary_from_pipeline(
        person_id="P03",
        current_rpe=16.4,
        previous_rpe=14.2,
        time_span_seconds=120.0,
        shap_values={"EMG_MDF": 2.50, "IMU_Jerk": -1.20},
        baseline_deviations={"EMG_RMS": 15.0},
    )
    numbers = summary.get_all_allowable_numbers()
    assert 16.4 in numbers
    assert 120.0 in numbers
    assert 2.50 in numbers


def test_number_checker_passes_grounded_text():
    summary = build_session_summary_from_pipeline(
        person_id="P03",
        current_rpe=16.4,
        time_span_seconds=120.0,
        shap_values={"EMG_MDF": 2.50},
    )
    checker = NumberChecker()
    valid_text = (
        "Participant P03 reached Borg RPE 16.4 after 120.0 seconds. "
        "The primary driver was EMG MDF contributing +2.50 to the score."
    )
    res = checker.verify(valid_text, summary)
    assert res.is_valid is True
    assert len(res.ungrounded_numbers) == 0


def test_number_checker_rejects_hallucinated_numbers():
    summary = build_session_summary_from_pipeline(
        person_id="P03",
        current_rpe=16.4,
        time_span_seconds=120.0,
        shap_values={"EMG_MDF": 2.50},
    )
    checker = NumberChecker()
    # 73.85 and 99.4 are completely fabricated/unsupported numbers
    hallucinated_text = "The fatigue was 16.4 with a heart rate of 73.85 and blood oxygen at 99.4."
    res = checker.verify(hallucinated_text, summary)
    assert res.is_valid is False
    assert any(n in res.ungrounded_numbers for n in [73.85, 99.4])


def test_hybrid_router_selective_triggers_and_fallback():
    router = HybridRouter()

    # 1. Normal exertion below threshold -> telemetry only
    normal_summary = build_session_summary_from_pipeline(
        person_id="P03", current_rpe=11.0, previous_rpe=11.0
    )
    resp = router.explain(normal_summary)
    assert resp.triggered_by == "ML_TELEMETRY_ONLY"

    # 2. Heavy exertion crossing 15.0 threshold -> triggers RAG explanation
    heavy_summary = build_session_summary_from_pipeline(
        person_id="P03", current_rpe=16.4, previous_rpe=14.0
    )
    resp_heavy = router.explain(heavy_summary)
    assert "HEAVY_FATIGUE" in resp_heavy.triggered_by or "THRESHOLD" in resp_heavy.triggered_by
    assert resp_heavy.verification_passed is True
    assert len(resp_heavy.retrieved_sources) > 0
    # Must append medical disclaimer
    assert "does not provide medical diagnosis" in resp_heavy.text


def test_llm_layer_evaluator():
    from ml.rag.evaluator import LLMLayerEvaluator

    evaluator = LLMLayerEvaluator()
    rate, passed, total = evaluator.evaluate_faithfulness(n_samples=10)
    assert rate == 100.0
    assert passed == 10

    rec1, rec3 = evaluator.evaluate_retrieval_quality()
    assert rec3 >= 70.0  # Top-3 recall >= 70%

