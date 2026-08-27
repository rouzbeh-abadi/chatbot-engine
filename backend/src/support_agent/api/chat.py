"""Chat endpoints: POST /chat (streaming) and POST /chat/sync."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from support_agent.api.identity import UserIdDep
from support_agent.api.options import CHAT_MODELS
from support_agent.api.rate_limit import limit_chat
from support_agent.api.schemas import ChatRequest, ChatResult
from support_agent.api.streaming import collect, to_sse
from support_agent.assistant import ProjectNotFoundError, load_project
from support_agent.engine import EngineDep
from support_agent.engine_client.models import EngineChatRequest

# Every turn here is a model call on the provider key, so both routes are
# metered. On the router, not the endpoints, so a third one cannot forget.
router = APIRouter(
    prefix="/chat", tags=["chat"], dependencies=[Depends(limit_chat)]
)


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

    # `user_id` has already been decided by `api/identity.py` -- it is either a
    # proxy-authenticated id or `anonymous`, never whatever the browser typed.
    # What is still missing for a multi-tenant product is authorisation: nothing
    # checks that *this* user may use *this* project. The engine only ever sees
    # an opaque id, so that check belongs here.
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
    user_id: UserIdDep,
) -> StreamingResponse:
    """Receive a chat request from the client and stream the engine response back.

    The client request is converted to an engine request, sent to the chatbot engine,
    and the returned events are streamed back to the client using SSE.
    """
    request = _build_request(body, user_id)

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
    user_id: UserIdDep,
) -> ChatResult:
    """Non-streaming variant, for smoke tests and simple clients."""
    request = _build_request(body, user_id)
    return await collect(await engine.start_chat(request))
