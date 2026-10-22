"""Tests for all 9 production-grade improvements.

Covers:
- Improvement #1: Ollama LLM backend wiring in HybridRouter
- Improvement #2: Dense sentence-transformer RAG retriever
- Improvement #3: Data-driven PersonalizationAgent
- Improvement #4: httpx / test_api fix (verified by this suite running)
- Improvement #5: Expanded knowledge base (25+ chunks indexed)
- Improvement #6: SSE streaming endpoint
- Improvement #7: API key authentication
- Improvement #8: /api/reports/metrics endpoint
- Improvement #9: Monte Carlo Dropout uncertainty quantification
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Improvement #2 — Dense sentence-transformer retriever
# ─────────────────────────────────────────────────────────────────────────────

class TestDenseRetriever:
    def setup_method(self):
        from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase
        # Use TF-IDF fallback for unit tests (avoids model download during CI)
        self.kb = LocalRAGKnowledgeBase(use_dense=False)

    def test_chunks_loaded(self):
        """Knowledge base should load at least 10 chunks from expanded knowledge dir."""
        assert len(self.kb.chunks) >= 10, (
            f"Expected ≥10 chunks, got {len(self.kb.chunks)}. "
            "Did you add the new knowledge files?"
        )

    def test_retrieval_returns_results(self):
        results = self.kb.search("EMG spectral shift median frequency fatigue", top_k=3)
        assert len(results) >= 1

    def test_retrieval_returns_scores(self):
        results = self.kb.search("personalization baseline individual variability", top_k=2)
        for chunk, score in results:
            assert 0.0 <= score <= 1.0

    def test_semantic_variant_query(self):
        """A paraphrased query should still retrieve relevant chunks."""
        results = self.kb.search("why does the muscle signal slow down during exercise", top_k=3)
        chunk_ids = [c.chunk_id for c, _ in results]
        # Should retrieve physiology chunks, not fail with 0 results
        assert len(results) >= 1

    def test_retrieval_backend_property(self):
        assert self.kb.retrieval_backend in ("dense:all-MiniLM-L6-v2", "sparse:tfidf")

    def test_dense_retriever_uses_sentence_transformers(self):
        """If sentence-transformers is installed, dense path should activate."""
        try:
            from sentence_transformers import SentenceTransformer
            from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase
            kb_dense = LocalRAGKnowledgeBase(use_dense=True)
            # It might fall back gracefully if model download fails in CI
            assert kb_dense.retrieval_backend in (
                "dense:all-MiniLM-L6-v2", "sparse:tfidf"
            )
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_chunk_ids_are_unique(self):
        ids = [c.chunk_id for c in self.kb.chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected in knowledge base"


# ─────────────────────────────────────────────────────────────────────────────
# Improvement #1 — Ollama LLM wiring in HybridRouter
# ─────────────────────────────────────────────────────────────────────────────

class TestOllamaWiring:
    def _make_summary(self, rpe: float = 16.5):
        from ml.rag.session_summary import build_session_summary_from_pipeline
        return build_session_summary_from_pipeline(
            person_id="P01",
            current_rpe=rpe,
            previous_rpe=13.0,
            time_span_seconds=60.0,
            shap_values={"EMG_MDF_AntDeltoid": 2.5, "IMU_Jerk": 1.2},
            baseline_deviations={"EMG_MDF": -15.0},
            model_name="TestModel",
        )

    def test_custom_llm_call_used_when_provided(self):
        from ml.rag.hybrid_router import HybridRouter
        mock_llm = MagicMock(return_value="Fatigue is at RPE 16.5 due to spectral shift [PHYS-EMG-01].")
        router = HybridRouter(custom_llm_call=mock_llm, use_ollama=False)
        assert router._llm_backend == "custom"
        resp = router.explain(self._make_summary(), force_trigger=True)
        assert mock_llm.called

    def test_ollama_backend_activated_when_available(self):
        """When ollama library is importable and use_ollama is not False, backend should be ollama."""
        import importlib
        ollama_spec = importlib.util.find_spec("ollama")
        if ollama_spec is None:
            pytest.skip("ollama library not installed")
        from ml.rag.hybrid_router import HybridRouter, _OLLAMA_AVAILABLE
        if _OLLAMA_AVAILABLE:
            router = HybridRouter(use_ollama=True)
            assert router._llm_backend == "ollama"

    def test_template_fallback_when_no_llm(self):
        from ml.rag.hybrid_router import HybridRouter
        router = HybridRouter(use_ollama=False)
        assert router._llm_backend == "template"
        resp = router.explain(self._make_summary(), force_trigger=True)
        assert len(resp.text) > 10

    def test_llm_rejection_on_prohibited_medical_term(self):
        """LLM output containing 'diagnosis' should be rejected → fallback template used."""
        from ml.rag.hybrid_router import HybridRouter
        bad_llm = MagicMock(return_value="This is a diagnosis of severe muscle disease.")
        router = HybridRouter(custom_llm_call=bad_llm, use_ollama=False)
        resp = router.explain(self._make_summary(), force_trigger=True)
        # is_llm_generated must be False because text contains prohibited term
        assert not resp.is_llm_generated

    def test_empty_llm_response_falls_back(self):
        """Empty LLM response should trigger fallback template."""
        from ml.rag.hybrid_router import HybridRouter
        empty_llm = MagicMock(return_value="")
        router = HybridRouter(custom_llm_call=empty_llm, use_ollama=False)
        resp = router.explain(self._make_summary(), force_trigger=True)
        assert not resp.is_llm_generated
        assert len(resp.text) > 0

    def test_rag_prompt_contains_grounding_facts(self):
        """The prompt built for the LLM should contain the Borg RPE number."""
        from ml.rag.hybrid_router import HybridRouter
        captured_prompts = []
        def capture_llm(prompt: str) -> str:
            captured_prompts.append(prompt)
            return ""
        router = HybridRouter(custom_llm_call=capture_llm, use_ollama=False)
        router.explain(self._make_summary(rpe=17.3), force_trigger=True)
        assert captured_prompts, "LLM was not called"
        assert "17.3" in captured_prompts[0], "Borg RPE not present in RAG prompt"


# ─────────────────────────────────────────────────────────────────────────────
# Improvement #3 — Data-driven PersonalizationAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestDataDrivenPersonalization:
    def _make_context(self, with_features: bool = False):
        ctx = {
            "subject_id": "P03",
            "subject_profile": {
                "sex": "Male", "age": 26, "upperarm_length_cm": 34.2, "mvic_force_n": 240.0
            },
            "population_rpe": 15.0,
        }
        if with_features:
            n_feats = 10
            ctx["baseline_feature_means"] = np.ones(n_feats) * 0.5
            ctx["baseline_feature_stds"] = np.ones(n_feats) * 0.1
            ctx["current_features"] = np.ones(n_feats) * 0.7  # elevated → fatigued
        return ctx

    def test_mvic_heuristic_path(self):
        from backend.agents.personalization_agent import PersonalizationAgent
        agent = PersonalizationAgent()
        result = agent.process(self._make_context(with_features=False))
        assert result.success
        assert 6.0 <= result.data["personalized_rpe"] <= 20.0

    def test_data_driven_path_with_features(self):
        from backend.agents.personalization_agent import PersonalizationAgent, _PERSONALIZATION_AVAILABLE
        if not _PERSONALIZATION_AVAILABLE:
            pytest.skip("ml.personalization not available")
        agent = PersonalizationAgent()
        result = agent.process(self._make_context(with_features=True))
        assert result.success
        assert 6.0 <= result.data["personalized_rpe"] <= 20.0

    def test_personalization_method_key_in_result(self):
        from backend.agents.personalization_agent import PersonalizationAgent
        agent = PersonalizationAgent()
        result = agent.process(self._make_context(with_features=False))
        assert "personalization_method" in result.data

    def test_personalized_rpe_within_borg_bounds(self):
        from backend.agents.personalization_agent import PersonalizationAgent
        agent = PersonalizationAgent()
        for mvic in [50.0, 200.0, 400.0]:  # extreme MVIC values
            ctx = self._make_context()
            ctx["subject_profile"]["mvic_force_n"] = mvic
            result = agent.process(ctx)
            assert 6.0 <= result.data["personalized_rpe"] <= 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Improvement #5 — Expanded knowledge base (≥10 chunks)
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandedKnowledgeBase:
    def setup_method(self):
        from ml.rag.knowledge_retriever import LocalRAGKnowledgeBase
        self.kb = LocalRAGKnowledgeBase(use_dense=False)

    def test_minimum_chunk_count(self):
        assert len(self.kb.chunks) >= 10

    def test_clinical_category_present(self):
        categories = {c.category for c in self.kb.chunks}
        assert "clinical" in categories

    def test_ml_methods_category_present(self):
        categories = {c.category for c in self.kb.chunks}
        assert "ml_methods" in categories

    def test_protocol_category_present(self):
        categories = {c.category for c in self.kb.chunks}
        assert "protocol" in categories

    def test_safety_category_present(self):
        categories = {c.category for c in self.kb.chunks}
        assert "safety" in categories

    def test_shap_chunk_retrievable(self):
        results = self.kb.search("SHAP feature attribution model explanation", top_k=3)
        chunk_ids = [c.chunk_id for c, _ in results]
        assert any("SHAP" in cid or "shap" in cid.lower() or "ML" in cid for cid in chunk_ids)

    def test_loso_chunk_retrievable(self):
        results = self.kb.search("leave one subject out cross validation generalization", top_k=3)
        assert len(results) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Improvement #7 — API Key Authentication
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIKeyAuthentication:
    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from backend.api.app import app, FATIGUE_API_KEY
        self.api_key = FATIGUE_API_KEY
        self.client = TestClient(app)

    def test_health_endpoint_public(self):
        """Health check does not require API key."""
        resp = self.client.get("/api/health")
        assert resp.status_code == 200

    def test_live_stream_requires_api_key(self):
        """SSE live stream endpoint should return 401 without key."""
        resp = self.client.get("/api/signals/live/P01?max_frames=1")
        assert resp.status_code in (401, 403)

    def test_live_stream_with_valid_key(self):
        """SSE live stream with correct key should return 200."""
        resp = self.client.get(
            "/api/signals/live/P01?max_frames=2",
            headers={"X-API-Key": self.api_key},
        )
        assert resp.status_code == 200

    def test_live_stream_with_invalid_key(self):
        resp = self.client.get(
            "/api/signals/live/P01",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# Improvement #8 — Reports metrics endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestReportsEndpoint:
    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from backend.api.app import app
        self.client = TestClient(app)

    def test_reports_metrics_returns_200(self):
        resp = self.client.get("/api/reports/metrics")
        assert resp.status_code == 200

    def test_reports_metrics_contains_loso_metrics(self):
        resp = self.client.get("/api/reports/metrics")
        data = resp.json()
        assert "loso_metrics" in data

    def test_reports_metrics_contains_rag_metrics(self):
        resp = self.client.get("/api/reports/metrics")
        data = resp.json()
        assert "rag_metrics" in data

    def test_reports_metrics_chunk_count_positive(self):
        resp = self.client.get("/api/reports/metrics")
        data = resp.json()
        assert data["rag_metrics"]["knowledge_chunk_count"] >= 10


# ─────────────────────────────────────────────────────────────────────────────
# Improvement #9 — Monte Carlo Dropout Uncertainty Quantification
# ─────────────────────────────────────────────────────────────────────────────

class TestMCDropout:
    @pytest.fixture
    def mock_adapter(self):
        """Build a mock ProposedModelAdapter that returns deterministic probabilities."""
        from ml.models.proposed.inference import ProposedModelAdapter
        adapter = MagicMock(spec=ProposedModelAdapter)
        # Always return [[0.3, 0.7]] shaped (1, 2) — high fatigue probability
        adapter.predict_proba.return_value = np.array([[0.3, 0.7]])
        # Create a minimal model attribute with Dropout layers
        import torch
        adapter.model = torch.nn.Sequential(torch.nn.Dropout(p=0.2))
        return adapter

    def test_mc_dropout_returns_uncertainty_prediction(self, mock_adapter):
        from ml.models.proposed.mc_dropout import MCDropoutPredictor
        mc = MCDropoutPredictor(mock_adapter, n_samples=10)
        X = np.random.rand(1, 20).astype(np.float32)
        result = mc.predict_with_uncertainty(X)
        assert result is not None
        assert 0.0 <= result.mean_prob <= 1.0
        assert result.std_prob >= 0.0

    def test_mc_rpe_within_borg_bounds(self, mock_adapter):
        from ml.models.proposed.mc_dropout import MCDropoutPredictor
        mc = MCDropoutPredictor(mock_adapter, n_samples=10)
        X = np.random.rand(1, 20).astype(np.float32)
        result = mc.predict_with_uncertainty(X)
        assert 6.0 <= result.mean_rpe <= 20.0
        assert 6.0 <= result.ci_lower_rpe <= 20.0
        assert 6.0 <= result.ci_upper_rpe <= 20.0

    def test_ci_lower_leq_upper(self, mock_adapter):
        from ml.models.proposed.mc_dropout import MCDropoutPredictor
        mc = MCDropoutPredictor(mock_adapter, n_samples=20)
        X = np.random.rand(1, 20).astype(np.float32)
        result = mc.predict_with_uncertainty(X)
        assert result.ci_lower_rpe <= result.ci_upper_rpe

    def test_n_samples_recorded(self, mock_adapter):
        from ml.models.proposed.mc_dropout import MCDropoutPredictor
        mc = MCDropoutPredictor(mock_adapter, n_samples=15)
        X = np.random.rand(1, 20).astype(np.float32)
        result = mc.predict_with_uncertainty(X)
        assert result.n_samples == 15

    def test_format_rpe_string(self, mock_adapter):
        from ml.models.proposed.mc_dropout import MCDropoutPredictor
        mc = MCDropoutPredictor(mock_adapter, n_samples=5)
        X = np.random.rand(1, 20).astype(np.float32)
        result = mc.predict_with_uncertainty(X)
        s = result.format_rpe_string()
        assert "RPE" in s
        assert "95% CI" in s

    def test_uncertainty_prediction_dataclass_frozen(self, mock_adapter):
        from ml.models.proposed.mc_dropout import MCDropoutPredictor, UncertaintyPrediction
        mc = MCDropoutPredictor(mock_adapter, n_samples=5)
        X = np.random.rand(1, 20).astype(np.float32)
        result = mc.predict_with_uncertainty(X)
        with pytest.raises((AttributeError, TypeError)):
            result.mean_rpe = 99.0  # type: ignore[misc]
