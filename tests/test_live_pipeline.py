"""Modules 10, 11 and 14: live pipeline, hardware contract, simulator, explanation service, call log."""

import json
import time

import numpy as np
import pytest

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.live import (
    CONTRACT, IMU_NAMES, Chunk, ExplanationService, LivePipeline, SerialSource, SyntheticSource, validate_chunk,
)
from ml.rag.hybrid_router import HybridRouter
from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase
from ml.rag.session_summary import build_session_summary_from_pipeline


class StubPopulation:
    """Stands in for the trained model: effort = 6 + 14 * (window index / 15), clipped. Plumbing tests only."""

    def predict(self, F):
        return np.clip(6 + 14 * F.win.to_numpy(float) / 15, 6, 20)


@pytest.fixture(scope="module")
def kb():
    return LocalRAGKnowledgeBase(use_dense=False)


def run(source, pipe):
    return [r for r in (pipe.push(ch) for ch in source.chunks()) if r is not None]


def test_channel_order_matches_cleaned_recordings():
    assert len(IMU_NAMES) == 36 and IMU_NAMES[0] == "acc_forearm_x" and IMU_NAMES[3] == "gyr_forearm_x"
    assert CONTRACT["emg"]["channels"] == c.MUSCLES


def test_simulator_conforms_to_the_contract():
    for ch in SyntheticSource(seconds=12, seed=1).chunks():
        assert validate_chunk(ch) == []


def test_contract_violations_are_reported():
    ch = next(SyntheticSource(seconds=1, seed=1).chunks())
    ch.emg[0, 5] = 50.0
    assert "emg outside +-10 mV" in validate_chunk(ch)
    assert any("shape" in i for i in validate_chunk(Chunk(ch.emg[:5], ch.imu, ch.ppg)))


def test_serial_source_is_an_explicit_placeholder():
    with pytest.raises(NotImplementedError):
        next(SerialSource().chunks())


def test_windows_calibration_and_alerts():
    pipe = LivePipeline(StubPopulation(), condition={"load_pct": 35, "direction": "internal", "mvic_dir_N": 60.0},
                        calibration_windows=4)
    res = run(SyntheticSource(seconds=160, seed=2), pipe)
    assert len(res) == 16 and res[0]["t"] == 10 and res[-1]["t"] == 160
    assert res[0]["generalised"] is None                       # baseline needs the first two windows
    assert res[1]["generalised"] is not None
    assert "forming baseline" in res[0]["calibration"] and pipe.calibration_state == "calibrated"
    levels = [a["level"] for a in pipe.alerts]
    assert levels == ["warning", "stop"] and pipe.alerts[0]["t"] < pipe.alerts[1]["t"]
    s = [r["smoothed_generalised"] for r in res if r["smoothed_generalised"] is not None]
    assert all(b >= a for a, b in zip(s, s[1:]))               # smoothing keeps a rising input rising


def test_dropout_is_flagged_and_features_become_nan():
    pipe = LivePipeline(StubPopulation(), condition={"load_pct": 35, "direction": "internal", "mvic_dir_N": 60.0})
    faults = [{"t": (30, 50), "stream": "emg", "channel": 0, "kind": "dropout"}]
    res = run(SyntheticSource(seconds=80, seed=2, faults=faults), pipe)
    assert any("emg_anterior_deltoid" in f for f in res[3]["quality_flags"])
    assert np.isnan(res[3]["features"]["emg_anterior_deltoid_rms"]) and not np.isnan(res[3]["features"]["emg_infraspinatus_rms"])
    assert not res[0]["quality_flags"]


def test_session_log_is_written(tmp_path):
    log = tmp_path / "s.jsonl"
    pipe = LivePipeline(StubPopulation(), condition={"load_pct": 35, "direction": "internal", "mvic_dir_N": 60.0}, session_log=log)
    run(SyntheticSource(seconds=30, seed=2), pipe)
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["win"] for r in rows] == [0, 1, 2] and "features" not in rows[0]


def test_the_live_loop_never_waits_for_the_language_model(kb, tmp_path):
    def slow_llm(prompt):
        time.sleep(0.4)
        return "Effort is hard at 16.0 out of 20."
    router = HybridRouter(knowledge_base=kb, custom_llm_call=slow_llm, call_log_path=tmp_path / "calls.jsonl")
    ex = ExplanationService(router)
    pipe = LivePipeline(StubPopulation(), condition={"load_pct": 35, "direction": "internal", "mvic_dir_N": 60.0}, explainer=ex)
    t0 = time.perf_counter()
    res = run(SyntheticSource(seconds=160, seed=2), pipe)
    loop_s = time.perf_counter() - t0
    assert max(r["latency_ms"] for r in res) < 400                  # no window waited for a 400 ms model call
    ex.wait()
    ex.shutdown()
    assert len(ex.history) >= 2 and ex.skipped >= 1                # selective: some windows stay machine-only
    assert loop_s < 0.4 * len(ex.history) + 30


def test_invented_numbers_are_caught_and_logged(kb, tmp_path):
    log = tmp_path / "calls.jsonl"
    router = HybridRouter(knowledge_base=kb, custom_llm_call=lambda p: "Your pulse is 171 beats per minute.", call_log_path=log)
    summary = build_session_summary_from_pipeline("P01", 16.4, 15.0, 30.0, shap_values={"emg_infraspinatus_mdf_rel": -1.2})
    resp = router.explain(summary, force_trigger=True)
    assert not resp.is_llm_generated and resp.fallback_reason == "ungrounded_numbers"
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["number_check"]["passed"] is True                    # the text that was shown is grounded
    assert rec["attempts"] == 2 and rec["input_summary"]["person_id"] == "P01" and rec["retrieved_chunk_ids"]
    assert {"trigger", "intent", "output", "delay_ms", "llm_backend"} <= set(rec)


def test_template_says_when_the_reading_is_less_reliable(kb):
    router = HybridRouter(knowledge_base=kb, use_ollama=False, use_anthropic=False)
    bad = build_session_summary_from_pipeline("P01", 16.0, 15.0, 30.0,
                                              quality_warnings=["emg anterior deltoid signal unreliable (12% valid)"])
    assert "unreliable" in router.explain(bad, force_trigger=True).answer
    early = build_session_summary_from_pipeline("P01", 8.0, 7.5, 10.0, quality_warnings=["BASELINE_STILL_FORMING"])
    assert "first 20 seconds" in router.explain(early, force_trigger=True).answer
    clean = build_session_summary_from_pipeline("P01", 16.0, 15.0, 30.0)
    assert "unreliable" not in router.explain(clean, force_trigger=True).answer


def test_person_facts_numbers_are_registered_with_the_summary():
    summary = build_session_summary_from_pipeline("P01", 12.0)
    summary.metadata["allowed_numbers"] = [76.0, 12.19]
    allowed = summary.get_all_allowable_numbers()
    assert 76.0 in allowed and 12.19 in allowed and 12.0 in allowed


def test_pipeline_feature_names_are_plain_language():
    from ml.rag.feature_glossary import feature_label
    assert feature_label("emg_infraspinatus_mdf_rel") == "rotator cuff muscle fibre tiredness"
    assert feature_label("hr_bpm_rel") == "pulse rate"
    assert feature_label("imu_forearm_jerk") == "forearm jerkiness"
    assert feature_label("median_frequency") == "muscle fibre tiredness"        # existing keys unchanged
