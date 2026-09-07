"""POST /chat - one conversation turn, streamed as NDJSON."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from chatbot_engine.api.dependencies import ChatServiceDep, SettingsDep
from chatbot_engine.api.streaming import MEDIA_TYPE, to_ndjson
from chatbot_engine.models.chat import ChatRequest

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=None,
    responses={
        200: {
            "content": {MEDIA_TYPE: {}},
            "description": (
                "A stream of events, one JSON object per line: retrieval, token, "
                "usage, error, done."
            ),
        },
        422: {"description": "The assistant named an agent this engine does not have."},
        501: {"description": "No model provider key is configured."},
    },
)
async def chat(
    request: ChatRequest, service: ChatServiceDep, settings: SettingsDep
) -> StreamingResponse:
    """Run one turn and stream its events as NDJSON.

    Both preconditions are checked before the response starts, so an unknown
    agent name (422) or a missing provider key (501) arrives as a status code.
    Left to surface lazily, the key would first be needed during retrieval,
    inside a stream that has already committed to 200.
    """
    settings.require_openrouter_key()
    events = service.stream(request)

    return StreamingResponse(
        to_ndjson(events),
        media_type=MEDIA_TYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
