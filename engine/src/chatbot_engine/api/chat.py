"""POST /chat - one conversation turn, streamed as NDJSON."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from chatbot_engine.api.dependencies import ChatServiceDep
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
        501: {"description": "No Agent implementation is registered yet."},
    },
)
async def chat(request: ChatRequest, service: ChatServiceDep) -> StreamingResponse:
    """Receive a chat request from the backend and stream engine events as NDJSON.
    The request is passed to the chat service, which runs the agent. Streaming
    starts only after setup succeeds, so errors can still return the correct
    HTTP status before any response body is sent.
    """
    events = service.stream(request)

    return StreamingResponse(
        to_ndjson(events),
        media_type=MEDIA_TYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
