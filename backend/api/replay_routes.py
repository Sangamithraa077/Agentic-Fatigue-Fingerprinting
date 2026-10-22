"""Real-data replay endpoints: the dashboard's live-signal stand-in.

Every response here is computed by `backend.services.replay_engine`, which
in turn only calls this project's existing preprocessing, feature
engineering, modeling, explainability, and personalization modules over
real WSD4FEDSRM recordings -- nothing in this router generates or
approximates a signal.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Query  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore

from backend.services import replay_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/replay", tags=["replay"])

MIN_SPEED_MS = 150
MAX_SPEED_MS = 3000


@router.get("/subjects")
def list_subjects():
    """Every subject_id actually discoverable in the configured WSD4FEDSRM dataset root."""
    subjects = replay_engine.list_available_subjects()
    return {"subjects": subjects, "count": len(subjects)}


@router.get("/stream/{subject_id}")
async def stream_subject(subject_id: str, speed_ms: int = Query(default=600, ge=MIN_SPEED_MS, le=MAX_SPEED_MS)):
    """Server-Sent Events stream of real, fully-explained replay windows for one subject.

    Building a subject's session (preprocessing + feature extraction over
    their full recording) takes on the order of tens of seconds the first
    time it is requested; it is cached afterward. That build runs in a
    worker thread so it does not block the server's event loop.
    """
    available = replay_engine.list_available_subjects()
    if subject_id not in available:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found in the dataset.")

    try:
        session = await asyncio.to_thread(replay_engine.get_or_build_session, subject_id)
    except Exception as exc:  # pragma: no cover - defensive, surfaces a real pipeline failure to the client
        logger.exception("Failed to build replay session for '%s'", subject_id)
        raise HTTPException(status_code=500, detail=f"Failed to build replay session: {exc}") from exc

    async def event_generator():
        step = 0
        try:
            while True:
                payload = await asyncio.to_thread(replay_engine.compute_replay_step, session, step)
                yield f"data: {json.dumps(payload)}\n\n"
                step += 1
                await asyncio.sleep(speed_ms / 1000.0)
        except asyncio.CancelledError:
            logger.info("Replay stream for '%s' closed by client.", subject_id)
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
