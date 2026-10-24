"""Tests for the upgraded explanation pipeline: verify-then-repair, intents, plain language, retrieval."""

from unittest.mock import MagicMock

from ml.rag.feature_glossary import expand_query, feature_label, humanise_text
from ml.rag.hybrid_router import HybridRouter, classify_intent
from ml.rag.session_summary import build_session_summary_from_pipeline


def _summary(rpe: float = 16.2):
    return build_session_summary_from_pipeline(
        person_id="P01", current_rpe=rpe, previous_rpe=14.0, time_span_seconds=60.0,
        shap_values={"median_frequency": -0.31, "rms": 0.22, "slope_sign_changes": 0.1},
        baseline_deviations={"median_frequency": -1.2}, model_name="X",
    )


def _router(llm=None):
    return HybridRouter(custom_llm_call=llm, use_ollama=False, use_anthropic=False)


def test_repair_loop_recovers_from_ungrounded_number():
    answers = iter([
        "Effort is hard at 16.2 with a heart rate of 142.",
        "Effort is hard at 16.2 out of 20 and rising [PHYS-EMG-01].",
    ])
    llm = MagicMock(side_effect=lambda prompt: next(answers))
    resp = _router(llm).explain(_summary(), "why?", force_trigger=True)
    assert resp.is_llm_generated and resp.attempts == 2 and resp.fallback_reason is None
    assert "CORRECTION NEEDED" in llm.call_args_list[1].args[0]
    assert "142" in llm.call_args_list[1].args[0]


def test_falls_back_after_two_bad_attempts_and_reports_reason():
    llm = MagicMock(return_value="Your heart rate is 142 bpm.")
    resp = _router(llm).explain(_summary(), "why?", force_trigger=True)
    assert not resp.is_llm_generated
    assert resp.fallback_reason == "ungrounded_numbers"
    assert resp.attempts == 2 and llm.call_count == 2
    assert "142" not in resp.answer


def test_no_llm_reports_reason_and_answer_has_no_disclaimer():
    resp = _router().explain(_summary(), "What should I do now?", force_trigger=True)
    assert resp.fallback_reason == "no_llm_configured"
    assert "does not provide medical diagnosis" not in resp.answer
    assert "does not provide medical diagnosis" in resp.text


def test_prohibited_term_matches_whole_words_only():
    # "secure" contains "cure" but is not a prohibited word.
    llm = MagicMock(return_value="Effort is hard at 16.2 out of 20; keep a secure grip.")
    assert _router(llm).explain(_summary(), "why?", force_trigger=True).is_llm_generated


def test_answers_use_plain_language_not_feature_keys():
    resp = _router().explain(_summary(), None, force_trigger=True)
    assert "median_frequency" not in resp.answer and "slope_sign_changes" not in resp.answer
    assert "CALIBRATED_TO" not in resp.answer


def test_intents_and_template_shape():
    assert classify_intent("What should I do now?") == "advice"
    assert classify_intent("How reliable is this?") == "reliability"
    assert classify_intent("What does RMS mean?") == "meaning"
    assert classify_intent("Give me a quick session summary") == "progress"
    advice = _router().explain(_summary(), "What should I do now?", force_trigger=True)
    assert "Lower the load" in advice.answer
    assert "typically off by about two points" in _router().explain(_summary(), "How reliable is this?", force_trigger=True).answer


def test_history_reaches_prompt_and_follow_ups_returned():
    llm = MagicMock(return_value="Effort is hard at 16.2 out of 20.")
    history = [{"role": "user", "content": "Is my arm shaking a sign?"}, {"role": "assistant", "content": "Possibly."}]
    resp = _router(llm).explain(_summary(), "and now?", force_trigger=True, history=history)
    assert "EARLIER IN THIS CONVERSATION" in llm.call_args.args[0] and "arm shaking" in llm.call_args.args[0]
    assert 1 <= len(resp.follow_ups) <= 3


def test_retrieval_uses_plain_phrases_and_returns_sources():
    resp = _router().explain(_summary(), "What does muscle fibre tiredness mean?", force_trigger=True)
    assert resp.source_details[0]["chunk_id"] == "PHYS-EMG-01"
    assert resp.source_details[0]["snippet"]
    assert "PHYS-EMG-01" in resp.answer


def test_forced_explanation_below_threshold_reports_user_request():
    resp = _router().explain(_summary(rpe=10.0), "why?", force_trigger=True)
    assert resp.triggered_by in ("USER_EXPLANATION_QUERY", "USER_REQUEST")


def test_glossary_helpers():
    assert feature_label("median_frequency") == "muscle fibre tiredness"
    assert feature_label("brand_new_feature") == "brand new feature"
    assert "CALIBRATED_TO_P01" not in humanise_text("mean (CALIBRATED_TO_P01) slope_sign_changes")
    assert "median frequency" in expand_query("muscle fibre tiredness")
