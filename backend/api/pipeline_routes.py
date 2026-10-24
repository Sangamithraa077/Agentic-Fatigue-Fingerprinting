"""Session lab API (Modules 9-12): sessions from the dataset pipeline, selective background explanations,
ask-your-data lookups, report export and the published evaluation tables.

Explanations never block the caller (Module 10.2): ``POST /explain`` answers immediately, either "not triggered"
(the estimate stays ML-only, Module 10.1) or with a job id the page polls. With the language model switched
off (or failing) the same endpoint still answers, from the deterministic template.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException  # type: ignore
from fastapi.responses import HTMLResponse, PlainTextResponse  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from backend.services import pipeline_service as svc
from ml.rag.hybrid_router import HybridRouter

router = APIRouter(prefix="/api/pipeline", tags=["session lab"])

_router_ref: Dict[str, Optional[HybridRouter]] = {"router": None}
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="explain")
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
MAX_JOBS = 300


def set_hybrid_router(r: HybridRouter) -> None:
    """Called once by the application so the lab and the coach share one router (and one call log)."""
    _router_ref["router"] = r


def _router() -> HybridRouter:
    r = _router_ref["router"]
    if r is None:
        raise HTTPException(status_code=503, detail="explanation layer not initialised")
    return r


def _store() -> svc.PipelineStore:
    store = svc.get_store()
    if not store.available:
        raise HTTPException(status_code=503, detail="pipeline tables are missing; run the dataset pipeline first "
                                                    "(see docs/dataset_pipeline_modules_1_7.md)")
    return store


def _session_or_404(store: svc.PipelineStore, sid: int, task: str, k: int) -> None:
    try:
        store.session(sid, task, k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class ExplainRequest(BaseModel):
    sid: int = Field(ge=1, le=34)
    task: str
    step: int = Field(ge=0)
    model: str = Field(default="generalised", pattern="^(generalised|personalised)$")
    calibration_windows: int = Field(default=svc.DEFAULT_K, ge=3, le=5)
    user_query: Optional[str] = Field(default=None, max_length=500)
    session_concluded: bool = False
    force: bool = Field(default=False, description="Explain even if no trigger fired (the person asked)")


class AskRequest(BaseModel):
    sid: int = Field(ge=1, le=34)
    task: str
    question: str = Field(max_length=300)
    calibration_windows: int = Field(default=svc.DEFAULT_K, ge=3, le=5)


@router.get("/status")
def status() -> Dict[str, Any]:
    store = svc.get_store()
    r = _router_ref["router"]
    return {"pipeline_tables": store.available, "llm_backend": None if r is None else r._llm_backend,
            "llm_enabled": r is not None and r.custom_llm_call is not None,
            "calibration_windows": svc.DEFAULT_K, "levels": {"warning": svc.WARN_LEVEL, "stop": svc.STOP_LEVEL}}


@router.get("/participants")
def participants() -> Dict[str, Any]:
    return {"participants": _store().participants()}


@router.get("/participants/{sid}/sessions")
def sessions(sid: int) -> Dict[str, Any]:
    out = _store().sessions(sid)
    if not out:
        raise HTTPException(status_code=404, detail=f"unknown participant {sid}")
    return {"sessions": out}


@router.get("/session/{sid}/{task}")
def session(sid: int, task: str, k: int = svc.DEFAULT_K) -> Dict[str, Any]:
    store = _store()
    _session_or_404(store, sid, task, k)
    return store.session(sid, task, k)


@router.post("/explain")
def explain(req: ExplainRequest) -> Dict[str, Any]:
    """Selective, non-blocking explanation (Modules 9, 10.1, 10.2)."""
    store = _store()
    _session_or_404(store, req.sid, req.task, req.calibration_windows)
    summary, extra = store.build_summary(req.sid, req.task, req.step, req.calibration_windows, req.model)
    r = _router()
    fire, trigger = r.should_trigger_explanation(summary, req.user_query, req.session_concluded)
    base = {"trigger": trigger, "summary": summary.to_dict(), "grounding": extra["grounding"]}
    if not fire and not req.force:
        return {**base, "triggered": False, "status": "ml_only"}

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "submitted": time.time()}
        for old in list(_jobs)[:-MAX_JOBS]:
            _jobs.pop(old, None)
    extra_chunks = [extra["person_chunk"]] if extra["person_chunk"] is not None else None

    def work() -> None:
        try:
            resp = r.explain(summary, user_query=req.user_query, session_concluded=req.session_concluded,
                             force_trigger=True, extra_chunks=extra_chunks)
            result = {
                "answer": resp.answer, "headline": resp.headline, "intent": resp.intent,
                "generated_by": "language model" if resp.is_llm_generated else "rule-based template",
                "is_llm_generated": resp.is_llm_generated, "fallback_reason": resp.fallback_reason,
                "verification": resp.verification, "sources": resp.source_details, "follow_ups": resp.follow_ups,
                "delay_ms": round(resp.execution_time_ms, 1), "disclaimer": resp.disclaimer,
                "numbers_used": [round(x, 2) for x in sorted(summary.get_all_allowable_numbers())][:60],
            }
            with _jobs_lock:
                _jobs[job_id].update(status="done", result=result)
        except Exception as exc:                                 # never leave the page waiting
            with _jobs_lock:
                _jobs[job_id].update(status="error", error=str(exc)[:200])

    _pool.submit(work)
    return {**base, "triggered": True, "status": "running", "job_id": job_id}


@router.get("/explain/{job_id}")
def explain_result(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown or expired job")
        return dict(job)


@router.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    store = _store()
    _session_or_404(store, req.sid, req.task, req.calibration_windows)
    return store.ask(req.sid, req.task, req.question, req.calibration_windows)


@router.get("/report/{sid}/{task}")
def report(sid: int, task: str, format: str = "html", k: int = svc.DEFAULT_K, explain_text: bool = True):
    """Exportable session report. The plain-language paragraph is generated text and labelled as such."""
    store = _store()
    _session_or_404(store, sid, task, k)
    text = None
    if explain_text:
        n = len(store.session(sid, task, k)["windows_data"])
        summary, extra = store.build_summary(sid, task, n - 1, k, "personalised")
        chunks = [extra["person_chunk"]] if extra["person_chunk"] is not None else None
        text = _router().explain(summary, session_concluded=True, force_trigger=True, extra_chunks=chunks).answer
    if format == "md":
        return PlainTextResponse(svc.report_markdown(store, sid, task, k, text), media_type="text/markdown")
    if format != "html":
        raise HTTPException(status_code=422, detail="format must be html or md")
    return HTMLResponse(svc.report_html(store, sid, task, k, text))


@router.get("/results")
def results() -> Dict[str, Any]:
    """The published evaluation tables behind the numbers (Modules 5-7)."""
    return _store().results()


@router.get("/llm-evaluation")
def llm_evaluation() -> Dict[str, Any]:
    import json
    from ml.dataset_pipeline import common as c
    p = c.REPO / "reports" / "llm_eval" / "llm_layer_evaluation.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="run python -m ml.rag.evaluate_layer to create it")
    return json.loads(p.read_text(encoding="utf-8"))
