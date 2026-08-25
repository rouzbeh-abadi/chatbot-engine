"""Chat endpoints: POST /chat (streaming) and POST /chat/sync."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from support_agent.api.options import CHAT_MODELS
from support_agent.api.schemas import ChatRequest, ChatResult
from support_agent.api.streaming import collect, to_sse
from support_agent.assistant import ProjectNotFoundError, load_project
from support_agent.engine import EngineDep
from support_agent.engine_client.models import EngineChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])


def _build_request(body: ChatRequest, user_id: str) -> EngineChatRequest:
    """Turn the frontend request into the engine request.

    Loads the assistant config server-side and applies the model override, so the
    engine always receives a complete, validated definition the browser never saw.
    """
    try:
        project = load_project(body.project)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if body.model is not None:
        if body.model not in CHAT_MODELS:
            raise HTTPException(
                status_code=422, detail=f"unknown model: {body.model!r}"
            )
        # A copy, not a mutation: `load_project` is cached and its result shared.
        project = project.model_copy(update={"model": body.model})

    # This backend is a showcase for working with the engine, not a production
    # service: the X-User-Id header is a placeholder, and the admin routes have
    # no auth at all. A real deployment would replace the header with real
    # authentication and check this user may use this project; the engine only
    # ever sees an opaque id.
    return EngineChatRequest(
        project=project,
        message=body.message,
        session_id=body.session_id,
        user_id=user_id,
        history=body.history,
    )


@router.post("", response_model=None)
async def chat(
    body: ChatRequest,
    engine: EngineDep,
    x_user_id: Annotated[str, Header()] = "demo-user",
) -> StreamingResponse:
    """Receive a chat request from the client and stream the engine response back.

    The client request is converted to an engine request, sent to the chatbot engine,
    and the returned events are streamed back to the client using SSE.
    """
   
    request = _build_request(body, x_user_id)

    # Awaited, so an unreachable engine or a 501 becomes a proper status code
    # here rather than an empty 200 with the error buried in the stream.
    events = await engine.start_chat(request)

    return StreamingResponse(
        to_sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sync")
async def chat_sync(
    body: ChatRequest,
    engine: EngineDep,
    x_user_id: Annotated[str, Header()] = "demo-user",
) -> ChatResult:
    """Non-streaming variant, for smoke tests and simple clients."""
    request = _build_request(body, x_user_id)
    return await collect(await engine.start_chat(request))
