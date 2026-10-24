"""Modules 9 & 10: Hybrid ML-LLM Router, Prompt Engine, and Guardrails.

Combines:
- Selective LLM routing triggers (thresholds, trend shifts, 'why' queries)
- RAG context retrieval with source grounding
- Strict safety rules and a medical disclaimer policy
- A pluggable LLM backend (custom callable > Anthropic Claude > local Ollama > template)
- A verify-then-repair loop: output that cites an ungrounded number or a
  prohibited medical term is sent back once with corrective feedback before the
  deterministic template takes over
- Execution logging

Everything the model may say is grounded in the structured ``SessionSummary``;
the number checker enforces it, so a chat-style answer can never invent a figure.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ml.rag.feature_glossary import effort_word, feature_label, humanise_text
from ml.rag.knowledge_retriever import KnowledgeChunk, LocalRAGKnowledgeBase
from ml.rag.number_checker import NumberChecker
from ml.rag.session_summary import SessionSummary

logger = logging.getLogger(__name__)

# Optional LLM backends, probed once at import.
try:
    import anthropic as _anthropic_lib
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic_lib = None
    _ANTHROPIC_AVAILABLE = False

try:
    import ollama as _ollama_lib
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

OLLAMA_DEFAULT_MODEL = "llama3.2:3b"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-5-5"
LLM_TIMEOUT_S = 25.0
MAX_CALL_LOGS = 500
MAX_HISTORY_TURNS = 6

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
# Whole-word match so "secure" or "procure" never trips the "cure" rule.
_PROHIBITED_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(PROHIBITED_MEDICAL_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are the AI Coach inside Athlete Intelligence, a wearable fatigue-monitoring product. "
    "You talk to athletes and coaches, not data scientists.\n\n"
    "Style:\n"
    "- Plain, calm, professional English. No jargon, no feature names, no markdown headings.\n"
    "- Lead with the answer in one sentence, then give the reason, then (when useful) one concrete next step.\n"
    "- 2 to 4 short sentences in total. Never pad or repeat the question.\n\n"
    "Rules you must not break:\n"
    "1. Use ONLY the numbers listed under FACTS. Never invent a heart rate, percentage, duration or count.\n"
    "2. The effort figure is a research-grade estimate with a typical error of about two points; say so "
    "if the athlete asks how reliable it is.\n"
    "3. Never diagnose, prescribe, or imply a medical condition. For pain or injury, advise seeing a professional.\n"
    "4. Cite supporting knowledge by its ID in square brackets, e.g. [PHYS-EMG-01], only when you used it.\n"
    "5. If the facts do not answer the question, say what you can and cannot tell from the current reading."
)

# Question intents, in priority order. Used to shape the prompt, the template
# answer and the suggested follow-up questions.
_INTENT_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    # Things the sensors cannot tell us. Checked first so the coach declines instead of guessing.
    ("out_of_scope", re.compile(
        r"\b(blood pressure|oxygen|spo2|body temperature|temperature|sleep|diet|nutrition|calorie|hydrat\w*|"
        r"eat|food|drink|supplements?|protein|caffeine|"
        r"weather|injur\w*|pain|medicat\w*|diagnos\w*|surgery|win|race|competition|will i|"
        r"future|tomorrow|next week|other athletes?|compare me|salary|age of|who is)\b", re.I)),
    ("reliability", re.compile(r"\b(reliable|accurate|accuracy|trust|confiden|error|sure)\b", re.I)),
    ("advice", re.compile(r"\b(should|stop|rest|continue|keep going|slow down|safe|what (?:do|now|next)|recover)\b", re.I)),
    ("meaning", re.compile(r"\b(what (?:does|is|are)|mean|means|define|explain what)\b", re.I)),
    ("why", re.compile(r"\b(why|driver|reason|cause|because|behind)\b", re.I)),
    ("progress", re.compile(r"\b(summary|summari[sz]e|overview|brief|how (?:am|is|are|has)|so far|progress|trend)\b", re.I)),
)


OUT_OF_SCOPE_ANSWER = (
    "I can't tell that from this athlete's readings. The sensors measure muscle activity, arm movement and "
    "pulse, and the coach estimates effort on the 6 to 20 Borg scale from them. For pain, injury or any "
    "health concern, please speak to a qualified professional. I can explain the current effort reading, "
    "what is driving it, or what to do next."
)


def classify_intent(query: Optional[str]) -> str:
    if not query:
        return "why"
    for name, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return name
    return "general"


@dataclass
class ExplanationResponse:
    """Standardized response from the explanation layer.

    ``text`` keeps the legacy contract (answer + disclaimer). ``answer`` is the
    answer alone, for UIs that show the disclaimer as a footnote.
    """

    text: str
    triggered_by: str
    is_llm_generated: bool
    retrieved_sources: List[str]
    verification_passed: bool
    execution_time_ms: float
    disclaimer: str = SAFETY_DISCLAIMER
    answer: str = ""
    headline: str = ""
    intent: str = "why"
    fallback_reason: Optional[str] = None
    llm_backend: str = "template"
    llm_model: Optional[str] = None
    attempts: int = 0
    source_details: List[Dict[str, Any]] = field(default_factory=list)
    follow_ups: List[str] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)


def _first_sentence(text: str, limit: int = 200) -> str:
    plain = re.sub(r"\s+", " ", re.sub(r"[*_`#>]", "", text)).strip()
    match = re.match(r"(.+?[.!?])(?:\s|$)", plain)
    sentence = match.group(1) if match else plain
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


_STOPWORDS = frozenset(
    "a an the and or of to in on is are was be it its this that what why how does do my me i you "
    "for with as at by about from can should am".split()
)


def _sentences(chunk: KnowledgeChunk) -> List[str]:
    """Prose sentences of a chunk body, stripped of markdown, headings and list scaffolding."""
    out: List[str] = []
    for line in chunk.body.splitlines():
        line = re.sub(r"^\s*(?:[*\-]|\d+\.|>)\s*", "", line.strip())
        line = re.sub(r"\*\*|__|`", "", line).strip("*_ ")
        if not line or line.startswith(("#", "|")) or line.endswith(":"):
            continue
        out.extend(p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if len(p.strip()) >= 40)
    return out


def _knowledge_snippet(chunk: KnowledgeChunk, query: str = "", limit: int = 240) -> str:
    """The sentence of ``chunk`` that best answers ``query`` (first prose sentence if none match)."""
    sentences = _sentences(chunk)
    if not sentences:
        return chunk.title
    terms = {w for w in re.findall(r"[a-z]+", query.lower()) if w not in _STOPWORDS and len(w) > 2}
    best = max(
        sentences,
        key=lambda sent: len(terms & set(re.findall(r"[a-z]+", sent.lower()))) if terms else 0,
    )
    return best if len(best) <= limit else best[: limit - 1].rstrip() + "…"


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
        anthropic_model: Optional[str] = None,
        use_anthropic: Optional[bool] = None,
    ):
        self.kb = knowledge_base or LocalRAGKnowledgeBase()
        self.checker = NumberChecker()
        self.rpe_warning_threshold = rpe_warning_threshold
        self.rpe_critical_threshold = rpe_critical_threshold
        self.ollama_model = ollama_model
        self.anthropic_model = anthropic_model or os.environ.get("AI_COACH_MODEL", ANTHROPIC_DEFAULT_MODEL)
        self.call_logs: List[Dict[str, Any]] = []
        self._anthropic_client: Any = None
        self._last_error: Optional[str] = None

        # LLM priority: custom_llm_call > Anthropic (needs ANTHROPIC_API_KEY) > Ollama > template
        anthropic_wanted = use_anthropic is not False and _ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))
        if custom_llm_call is not None:
            self.custom_llm_call: Optional[Callable[[str], str]] = custom_llm_call
            self._llm_backend = "custom"
        elif anthropic_wanted:
            self.custom_llm_call = self._call_anthropic
            self._llm_backend = "anthropic"
        elif _OLLAMA_AVAILABLE and (use_ollama is not False):
            self.custom_llm_call = self._call_ollama
            self._llm_backend = "ollama"
        else:
            self.custom_llm_call = None
            self._llm_backend = "template"

        logger.info("HybridRouter LLM backend: %s | KB backend: %s", self._llm_backend, self.kb.retrieval_backend)

    # ------------------------------------------------------------------ LLM backends

    @property
    def llm_model(self) -> Optional[str]:
        if self._llm_backend == "anthropic":
            return self.anthropic_model
        if self._llm_backend == "ollama":
            return self.ollama_model
        return None

    def _call_anthropic(self, prompt: str) -> str:
        """Calls Claude. The system prompt is passed separately when the prompt carries it."""
        try:
            if self._anthropic_client is None:
                self._anthropic_client = _anthropic_lib.Anthropic(timeout=LLM_TIMEOUT_S, max_retries=1)
            system, _, user = prompt.partition("\n\n=====\n\n")
            if not user:
                system, user = SYSTEM_PROMPT, prompt
            response = self._anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=450,
                temperature=0.2,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
        except Exception as exc:
            logger.warning("Anthropic call failed (%s). Falling back to template.", exc)
            self._last_error = f"{type(exc).__name__}"
            return ""

    def _call_ollama(self, prompt: str) -> str:
        """Calls local Ollama LLM and returns the generated text string."""
        try:
            system, _, user = prompt.partition("\n\n=====\n\n")
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}] if user \
                else [{"role": "user", "content": prompt}]
            response = _ollama_lib.chat(model=self.ollama_model, messages=messages)
            # ollama library returns a dict or Pydantic model depending on version
            if hasattr(response, "message"):
                return response.message.content.strip()  # type: ignore[union-attr]
            if isinstance(response, dict):
                return response["message"]["content"].strip()
            return str(response)
        except Exception as exc:
            logger.warning("Ollama call failed (%s). Falling back to template.", exc)
            self._last_error = f"{type(exc).__name__}"
            return ""

    # ------------------------------------------------------------------ routing

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

    # ------------------------------------------------------------------ plain-language helpers

    @staticmethod
    def _trend_phrase(summary: SessionSummary) -> str:
        return {
            "RISING_SHARPLY": "climbing quickly",
            "RISING_SLOWLY": "creeping up",
            "DECLINING": "easing off",
            "STABLE": "holding steady",
        }.get(summary.fatigue_trend, "holding steady")

    def _advice_line(self, summary: SessionSummary) -> str:
        rpe = summary.current_fatigue_index
        if rpe >= self.rpe_critical_threshold:
            return "Stop the exercise now and rest for at least 3 minutes."
        if rpe >= self.rpe_warning_threshold:
            return "Lower the load by about 20% or take a rest of around a minute."
        if summary.fatigue_trend == "RISING_SHARPLY":
            return "Effort is rising fast, so watch it closely over the next minute."
        return "You are in a safe range, so keep going and keep monitoring."

    def _driver_sentences(self, summary: SessionSummary) -> List[str]:
        out = []
        for feat in summary.top_contributing_features[:3]:
            up = feat["impact"] >= 0
            out.append(f"{feature_label(feat['feature'])} is {'pushing the reading up' if up else 'pulling the reading down'}")
        return out

    def _suggest_follow_ups(self, summary: SessionSummary, intent: str) -> List[str]:
        pool = {
            "why": ["What should I do now?", "How reliable is this estimate?", "What does muscle fibre tiredness mean?"],
            "advice": ["Why is the effort at this level?", "How reliable is this estimate?", "Give me a quick session summary"],
            "meaning": ["Why is the effort at this level?", "What should I do now?", "How reliable is this estimate?"],
            "reliability": ["Why is the effort at this level?", "What should I do now?", "What does the borg scale mean?"],
            "progress": ["What should I do now?", "Why is the effort at this level?", "How reliable is this estimate?"],
            "general": ["Why is the effort at this level?", "What should I do now?", "Give me a quick session summary"],
            "out_of_scope": ["Why is the effort at this level?", "What should I do now?", "How reliable is this estimate?"],
        }
        picks = pool.get(intent, pool["general"])
        if summary.current_fatigue_index >= self.rpe_warning_threshold and "What should I do now?" not in picks:
            picks = ["What should I do now?"] + picks[:2]
        return picks[:3]

    # ------------------------------------------------------------------ deterministic template

    def _generate_template_fallback(
        self,
        summary: SessionSummary,
        retrieved_chunks: List[KnowledgeChunk],
        intent: str = "why",
        query: str = "",
    ) -> str:
        """Deterministic, hallucination-free answer built directly from the numbers (Module 10.5).

        Written in the same plain voice as the LLM prompt asks for, and shaped
        by the question's intent, so a coach sees a sensible answer even with
        no LLM configured.
        """
        rpe = summary.current_fatigue_index
        status = f"Effort is {effort_word(rpe)}, {rpe:.1f} out of 20, and {self._trend_phrase(summary)}."
        drivers = self._driver_sentences(summary)
        driver_text = ""
        if drivers:
            driver_text = "The biggest factors: " + "; ".join(drivers) + "."
        grounding = ""
        if retrieved_chunks:
            c = retrieved_chunks[0]
            grounding = f"{_knowledge_snippet(c, query)} [{c.chunk_id}]"
        flags = [f for f in summary.data_quality_flags if f not in ("SIGNAL_AUDIT_CERTIFIED", "ALL_CHANNELS_OK")]
        caution = ""
        if "RPE_IS_PROBABILITY_PROXY_NOT_REGRESSION" in flags:
            caution = "This is an early, rough figure from the fatigue detector, so treat it with extra care."
        elif "SURROGATE_ESTIMATE_NOT_A_TRAINED_MODEL" in flags:
            caution = "This figure comes from a simple test formula on simulated sensor data, not from the trained model."

        if intent == "advice":
            parts = [self._advice_line(summary), status]
        elif intent == "meaning":
            parts = [grounding or status, status]
        elif intent == "reliability":
            parts = [
                "This estimate is a research-grade guide: it is typically off by about two points on the 6 to 20 scale, "
                "so use it to spot trends rather than as an exact measurement.",
                caution or status,
            ]
        elif intent == "progress":
            parts = [status, driver_text, self._advice_line(summary)]
        else:
            parts = [status, driver_text, grounding, caution]
        return " ".join(p for p in parts if p).strip()

    # ------------------------------------------------------------------ prompt

    def _build_user_prompt(
        self,
        summary: SessionSummary,
        retrieved_chunks: List[KnowledgeChunk],
        user_query: Optional[str],
        intent: str,
        history: Optional[List[Dict[str, str]]] = None,
        repair_note: Optional[str] = None,
    ) -> str:
        facts = [
            f"Effort estimate (Borg 6-20 scale): {summary.current_fatigue_index:.1f} ({effort_word(summary.current_fatigue_index)})",
            f"Trend: {self._trend_phrase(summary)} ({summary.rate_of_rpe_change_per_min:+.1f} points per minute)",
            f"Time into this exercise: {summary.time_span_seconds:.0f} seconds",
            f"Athlete: {summary.person_id}",
            f"Adjustment: {humanise_text(summary.calibration_status)}",
            f"Warning threshold: {self.rpe_warning_threshold:.0f}; stop threshold: {self.rpe_critical_threshold:.0f}",
        ]
        if summary.top_contributing_features:
            drivers = ", ".join(
                f"{feature_label(f['feature'])} ({'raises' if f['impact'] >= 0 else 'lowers'} the reading, strength {abs(f['impact']):.2f})"
                for f in summary.top_contributing_features
            )
            facts.append(f"Main drivers: {drivers}")
        if summary.baseline_changes:
            baseline = ", ".join(f"{feature_label(k)} {v:+.1f} standard deviations from this athlete's usual" for k, v in list(summary.baseline_changes.items())[:4])
            facts.append(f"Versus the athlete's own usual: {baseline}")
        quality = [f for f in summary.data_quality_flags if f not in ("SIGNAL_AUDIT_CERTIFIED", "ALL_CHANNELS_OK")]
        facts.append("Data quality: " + (", ".join(q.replace("_", " ").lower() for q in quality) if quality else "clean"))

        knowledge = "\n\n".join(f"[{c.chunk_id}] {c.title}:\n{c.body[:900]}" for c in retrieved_chunks) \
            or "No specific knowledge retrieved."

        convo = ""
        if history:
            lines = []
            for turn in history[-MAX_HISTORY_TURNS:]:
                role = "Athlete/coach" if turn.get("role") == "user" else "AI Coach"
                lines.append(f"{role}: {turn.get('content', '')[:400]}")
            convo = "EARLIER IN THIS CONVERSATION\n" + "\n".join(lines) + "\n\n"

        guide = {
            "advice": "The question is about what to do. Give a clear recommendation tied to the effort level.",
            "meaning": "The question asks what something means. Define it simply, then link it to this athlete's reading.",
            "reliability": "The question is about reliability. Be honest about the typical error and what the figure can and cannot show.",
            "progress": "The question asks for a summary. Cover where effort is now, the trend, the main driver and one next step.",
            "why": "The question asks why. Name the main drivers in everyday words and say which direction they push.",
            "general": "Answer the question directly using only the facts available.",
        }[intent]

        prompt = (
            f"{SYSTEM_PROMPT}\n\n=====\n\n"
            f"FACTS (the only numbers you may use)\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"
            f"RETRIEVED KNOWLEDGE\n{knowledge}\n\n"
            f"{convo}"
            f"QUESTION TYPE: {guide}\n"
            f"QUESTION: {user_query or 'Explain the current fatigue status and what is driving it.'}\n"
        )
        if repair_note:
            prompt += f"\nCORRECTION NEEDED: {repair_note}\n"
        return prompt + "\nANSWER:"

    # ------------------------------------------------------------------ main entry

    def _try_llm(self, prompt_builder: Callable[[Optional[str]], str], summary: SessionSummary) -> Dict[str, Any]:
        """Runs the verify-then-repair loop. Returns text/attempts/reason/verification."""
        result: Dict[str, Any] = {"text": None, "attempts": 0, "reason": None, "verification": {}}
        repair: Optional[str] = None
        for attempt in (1, 2):
            result["attempts"] = attempt
            self._last_error = None
            try:
                candidate = self.custom_llm_call(prompt_builder(repair))  # type: ignore[misc]
            except Exception as exc:
                logger.warning("LLM call failed: %s", exc)
                result["reason"] = "llm_error"
                return result
            candidate = (candidate or "").strip()
            if not candidate:
                result["reason"] = "llm_error" if self._last_error else "llm_empty"
                return result

            banned = _PROHIBITED_RE.search(candidate)
            if banned:
                logger.warning("LLM output used prohibited term %r (attempt %d).", banned.group(0), attempt)
                result["reason"] = "prohibited_term"
                repair = f"Your answer used the word '{banned.group(0)}'. Rewrite without any diagnostic or treatment language."
                continue

            check = self.checker.verify(candidate, summary)
            result["verification"] = {
                "numbers_checked": len(check.extracted_numbers),
                "ungrounded": check.ungrounded_numbers,
            }
            if check.is_valid:
                result.update(text=candidate, reason=None)
                return result
            logger.info("LLM output failed number check (attempt %d): %s", attempt, check.ungrounded_numbers)
            result["reason"] = "ungrounded_numbers"
            repair = (
                f"Your answer contained numbers that are not in the FACTS: {check.ungrounded_numbers}. "
                "Rewrite it using only numbers from the FACTS, or no numbers at all."
            )
        return result

    def explain(
        self,
        summary: SessionSummary,
        user_query: Optional[str] = None,
        session_concluded: bool = False,
        force_trigger: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ExplanationResponse:
        """Main explanation pipeline with guardrails and verifier."""
        t0 = time.perf_counter()
        intent = classify_intent(user_query)

        should_run, trigger_reason = self.should_trigger_explanation(summary, user_query, session_concluded)

        if not should_run and force_trigger:
            trigger_reason = "USER_REQUEST"

        if not should_run and not force_trigger:
            rpe = summary.current_fatigue_index
            answer = f"Effort is {effort_word(rpe)}, {rpe:.1f} out of 20. Nothing unusual to flag right now."
            return ExplanationResponse(
                text=answer,
                triggered_by="ML_TELEMETRY_ONLY",
                is_llm_generated=False,
                retrieved_sources=[],
                verification_passed=True,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                answer=answer,
                headline=answer,
                intent=intent,
                llm_backend=self._llm_backend,
                follow_ups=self._suggest_follow_ups(summary, intent),
            )

        if intent == "out_of_scope":
            # Decline deterministically: no retrieval, no language model, nothing to hallucinate.
            total_time_ms = (time.perf_counter() - t0) * 1000.0
            self.call_logs.append({
                "timestamp": time.time(), "person_id": summary.person_id, "rpe": summary.current_fatigue_index,
                "trigger": trigger_reason, "intent": intent, "is_llm": False, "attempts": 0,
                "fallback_reason": "out_of_scope", "sources": [], "delay_ms": total_time_ms,
            })
            del self.call_logs[:-MAX_CALL_LOGS]
            return ExplanationResponse(
                text=OUT_OF_SCOPE_ANSWER + SAFETY_DISCLAIMER,
                triggered_by=trigger_reason,
                is_llm_generated=False,
                retrieved_sources=[],
                verification_passed=True,
                execution_time_ms=total_time_ms,
                answer=OUT_OF_SCOPE_ANSWER,
                headline=_first_sentence(OUT_OF_SCOPE_ANSWER),
                intent=intent,
                fallback_reason="out_of_scope",
                llm_backend=self._llm_backend,
                follow_ups=self._suggest_follow_ups(summary, intent),
            )

        # 1. Retrieve knowledge. The question leads the query (everyday phrases are expanded to the
        #    knowledge base's vocabulary); athlete IDs are noise for retrieval and are left out.
        driver_terms = " ".join(feature_label(f["feature"]) for f in summary.top_contributing_features)
        if user_query:
            search_query = f"{user_query} {user_query}"
            if intent in ("why", "progress", "general"):
                search_query += f" fatigue {driver_terms}"
        else:
            search_query = f"fatigue {summary.fatigue_trend.replace('_', ' ').lower()} {driver_terms}"
        if intent == "reliability":
            search_query += " estimate error uncertainty validation"
        retrieved_results = self.kb.search(search_query, top_k=3)
        retrieved_chunks = [c for c, _ in retrieved_results]
        source_tags = [c.format_citation() for c in retrieved_chunks]
        source_details = [
            {
                "chunk_id": c.chunk_id, "title": c.title, "category": c.category,
                "source": c.source, "snippet": _knowledge_snippet(c, user_query or ""), "score": round(score, 3),
            }
            for c, score in retrieved_results
        ]

        # 2. LLM generation with verify-then-repair; 3. template fallback otherwise.
        generated_text: Optional[str] = None
        fallback_reason: Optional[str] = "no_llm_configured"
        attempts = 0
        verification: Dict[str, Any] = {}
        if self.custom_llm_call is not None:
            outcome = self._try_llm(
                lambda repair: self._build_user_prompt(summary, retrieved_chunks, user_query, intent, history, repair),
                summary,
            )
            attempts, verification = outcome["attempts"], outcome["verification"]
            generated_text, fallback_reason = outcome["text"], outcome["reason"]

        is_llm = generated_text is not None
        if generated_text is None:
            generated_text = self._generate_template_fallback(summary, retrieved_chunks, intent, user_query or "")

        answer = humanise_text(generated_text)
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        # Log call (Module 10.6)
        self.call_logs.append({
            "timestamp": time.time(),
            "person_id": summary.person_id,
            "rpe": summary.current_fatigue_index,
            "trigger": trigger_reason,
            "intent": intent,
            "is_llm": is_llm,
            "attempts": attempts,
            "fallback_reason": fallback_reason,
            "sources": source_tags,
            "delay_ms": total_time_ms,
        })
        del self.call_logs[:-MAX_CALL_LOGS]

        return ExplanationResponse(
            text=answer + SAFETY_DISCLAIMER,
            triggered_by=trigger_reason,
            is_llm_generated=is_llm,
            retrieved_sources=source_tags,
            verification_passed=True,
            execution_time_ms=total_time_ms,
            answer=answer,
            headline=_first_sentence(answer),
            intent=intent,
            fallback_reason=None if is_llm else fallback_reason,
            llm_backend=self._llm_backend,
            llm_model=self.llm_model,
            attempts=attempts,
            source_details=source_details,
            follow_ups=self._suggest_follow_ups(summary, intent),
            verification=verification,
        )
