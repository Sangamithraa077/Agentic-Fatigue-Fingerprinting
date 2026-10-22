"""Modules 9 & 10: Hybrid ML-LLM Router, Prompt Engine, and Guardrails.

Combines:
- Selective LLM Routing Triggers (Thresholds, Trend shifts, 'Why' queries)
- RAG Context Retrieval with Source Grounding
- Strict Safety Rules & Medical Disclaimer Policy
- Automated Fallback to Deterministic Template if Verification Fails
- Execution Logging
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ml.rag.knowledge_retriever import KnowledgeChunk, LocalRAGKnowledgeBase
from ml.rag.number_checker import NumberChecker
from ml.rag.session_summary import SessionSummary

logger = logging.getLogger(__name__)

# Try to import Ollama for local LLM generation
try:
    import ollama as _ollama_lib
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

OLLAMA_DEFAULT_MODEL = "llama3.2:3b"

# Mandatory Medical Safety Disclaimer (Module 10.4)
SAFETY_DISCLAIMER = (
    "\n\n*Notice: This system provides physiological fatigue assessment for research and "
    "ergonomic evaluation only. It does not provide medical diagnosis or treatment. "
    "Consult a qualified healthcare professional for pain, injury, or persistent symptoms.*"
)

PROHIBITED_MEDICAL_TERMS = {
    "diagnose", "diagnosis", "pathology", "disease", "prescribe", "prescription",
    "medication", "cure", "treatment plan",
}


@dataclass
class ExplanationResponse:
    """Standardized response from the explanation layer."""

    text: str
    triggered_by: str
    is_llm_generated: bool
    retrieved_sources: List[str]
    verification_passed: bool
    execution_time_ms: float
    disclaimer: str = SAFETY_DISCLAIMER


class HybridRouter:
    """Orchestrates selective RAG retrieval, prompt formatting, number checking, and fallback."""

    def __init__(
        self,
        knowledge_base: Optional[LocalRAGKnowledgeBase] = None,
        rpe_warning_threshold: float = 15.0,
        rpe_critical_threshold: float = 18.0,
        custom_llm_call: Optional[Callable[[str], str]] = None,
        ollama_model: str = OLLAMA_DEFAULT_MODEL,
        use_ollama: Optional[bool] = None,
    ):
        self.kb = knowledge_base or LocalRAGKnowledgeBase()
        self.checker = NumberChecker()
        self.rpe_warning_threshold = rpe_warning_threshold
        self.rpe_critical_threshold = rpe_critical_threshold
        self.ollama_model = ollama_model
        self.call_logs: List[Dict[str, Any]] = []

        # LLM priority: custom_llm_call > Ollama auto-detect > deterministic template
        if custom_llm_call is not None:
            self.custom_llm_call: Optional[Callable[[str], str]] = custom_llm_call
            self._llm_backend = "custom"
        elif _OLLAMA_AVAILABLE and (use_ollama is not False):
            self.custom_llm_call = self._call_ollama
            self._llm_backend = "ollama"
        else:
            self.custom_llm_call = None
            self._llm_backend = "template"

        logger.info("HybridRouter LLM backend: %s | KB backend: %s",
                    self._llm_backend, self.kb.retrieval_backend)

    def _call_ollama(self, prompt: str) -> str:
        """Calls local Ollama LLM and returns the generated text string."""
        try:
            response = _ollama_lib.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            # ollama library returns a dict or Pydantic model depending on version
            if hasattr(response, "message"):
                return response.message.content.strip()  # type: ignore[union-attr]
            if isinstance(response, dict):
                return response["message"]["content"].strip()
            return str(response)
        except Exception as exc:
            logger.warning("Ollama call failed (%s). Falling back to template.", exc)
            return ""

    def should_trigger_explanation(
        self,
        summary: SessionSummary,
        user_query: Optional[str] = None,
        session_concluded: bool = False,
    ) -> Tuple[bool, str]:
        """Determines if an explanation should be generated or remain ML-only (Module 10.1)."""
        if user_query and any(w in user_query.lower() for w in ["why", "driver", "reason", "explain", "detail"]):
            return True, "USER_EXPLANATION_QUERY"

        if summary.current_fatigue_index >= self.rpe_critical_threshold:
            return True, "CRITICAL_EXHAUSTION_THRESHOLD_EXCEEDED"

        if summary.current_fatigue_index >= self.rpe_warning_threshold:
            return True, "HEAVY_FATIGUE_WARNING_THRESHOLD"

        if summary.fatigue_trend == "RISING_SHARPLY":
            return True, "SHARP_FATIGUE_TREND_ACCELERATION"

        if session_concluded:
            return True, "SESSION_CONCLUDED_SUMMARY"

        return False, "ML_TELEMETRY_ONLY"

    def _generate_template_fallback(
        self, summary: SessionSummary, retrieved_chunks: List[KnowledgeChunk]
    ) -> str:
        """Deterministic, hallucination-free explanation built directly from numbers (Module 10.5)."""
        lines = [
            f"**Fatigue Status**: Borg RPE {summary.current_fatigue_index:.1f}/20.0 ({summary.fatigue_trend.replace('_', ' ')}).",
            f"Evaluated with model '{summary.model_used}' ({summary.calibration_status}).",
        ]

        if summary.top_contributing_features:
            driver_texts = []
            for feat in summary.top_contributing_features:
                driver_texts.append(f"{feat['feature']} (impact: {feat['impact']:+.2f})")
            lines.append(f"**Key Drivers**: {', '.join(driver_texts)}.")

        if retrieved_chunks:
            sources = ", ".join([c.format_citation() for c in retrieved_chunks[:2]])
            lines.append(f"**Physiological Grounding**: Corroborated by {sources}.")

        return "\n".join(lines)

    def explain(
        self,
        summary: SessionSummary,
        user_query: Optional[str] = None,
        session_concluded: bool = False,
        force_trigger: bool = False,
    ) -> ExplanationResponse:
        """Main explanation pipeline with guardrails and verifier."""
        t0 = time.perf_counter()

        should_run, trigger_reason = self.should_trigger_explanation(
            summary, user_query, session_concluded
        )

        if not should_run and not force_trigger:
            return ExplanationResponse(
                text=f"Telemetry Monitoring: Borg RPE {summary.current_fatigue_index:.1f}. Normal working parameters.",
                triggered_by="ML_TELEMETRY_ONLY",
                is_llm_generated=False,
                retrieved_sources=[],
                verification_passed=True,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 1. Retrieve relevant knowledge chunks
        search_query = f"{summary.person_id} fatigue {summary.fatigue_trend} "
        if summary.top_contributing_features:
            search_query += " ".join([f["feature"] for f in summary.top_contributing_features])
        if user_query:
            search_query += f" {user_query}"

        retrieved_results = self.kb.search(search_query, top_k=2)
        retrieved_chunks = [c for c, _ in retrieved_results]
        source_tags = [c.format_citation() for c in retrieved_chunks]

        # 2. Try LLM Generation (Ollama > custom > fallback template)
        generated_text = None
        is_llm = False
        if self.custom_llm_call is not None:
            # Build structured RAG prompt from retrieved knowledge
            knowledge_context = "\n\n".join(
                [f"[{c.chunk_id}] {c.title}:\n{c.body}" for c in retrieved_chunks]
            ) if retrieved_chunks else "No specific knowledge retrieved."

            prompt = (
                "You are an expert physiological fatigue assessment assistant.\n"
                "Your job is to explain fatigue monitoring results in clear, grounded language.\n\n"
                "=== GROUNDING FACTS (use ONLY these numbers) ===\n"
                f"Borg RPE: {summary.current_fatigue_index:.1f}/20.0\n"
                f"Fatigue Trend: {summary.fatigue_trend.replace('_', ' ')}\n"
                f"Person ID: {summary.person_id} | Model: {summary.model_used}\n"
                f"Calibration Status: {summary.calibration_status}\n"
            )
            if summary.top_contributing_features:
                feat_str = ", ".join(
                    [f"{f['feature']} (SHAP: {f['impact']:+.2f})" for f in summary.top_contributing_features]
                )
                prompt += f"Top Signal Drivers: {feat_str}\n"
            prompt += (
                "\n=== RETRIEVED SCIENTIFIC KNOWLEDGE ===\n"
                f"{knowledge_context}\n\n"
                "=== STRICT RULES ===\n"
                "1. ONLY use numbers that appear in GROUNDING FACTS above.\n"
                "2. Cite knowledge chunks by their ID (e.g., [PHYS-EMG-01]).\n"
                "3. NEVER diagnose, prescribe, or claim clinical certainty.\n"
                "4. Write 3-5 clear sentences. Be specific, not generic.\n\n"
                f"=== USER QUESTION ===\n{user_query or 'Explain the current fatigue status and what is driving it.'}\n\n"
                "=== YOUR RESPONSE ==="
            )
            try:
                candidate = self.custom_llm_call(prompt)
                if candidate:
                    lower_cand = candidate.lower()
                    if not any(term in lower_cand for term in PROHIBITED_MEDICAL_TERMS):
                        check_res = self.checker.verify(candidate, summary)
                        if check_res.is_valid:
                            generated_text = candidate
                            is_llm = True
                        else:
                            logger.info("LLM output failed number check — using fallback template.")
                    else:
                        logger.warning("LLM output contained prohibited medical term — using fallback template.")
            except Exception as e:
                logger.warning("LLM call failed: %s; invoking fallback template.", e)

        # 3. Fallback to deterministic template if LLM absent or rejected
        if generated_text is None:
            generated_text = self._generate_template_fallback(summary, retrieved_chunks)

        total_time_ms = (time.perf_counter() - t0) * 1000.0

        # Log call (Module 10.6)
        log_entry = {
            "timestamp": time.time(),
            "person_id": summary.person_id,
            "rpe": summary.current_fatigue_index,
            "trigger": trigger_reason,
            "is_llm": is_llm,
            "sources": source_tags,
            "delay_ms": total_time_ms,
        }
        self.call_logs.append(log_entry)

        return ExplanationResponse(
            text=generated_text + SAFETY_DISCLAIMER,
            triggered_by=trigger_reason,
            is_llm_generated=is_llm,
            retrieved_sources=source_tags,
            verification_passed=True,
            execution_time_ms=total_time_ms,
        )
