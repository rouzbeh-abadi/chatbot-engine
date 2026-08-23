"""Chat.

This backend's whole job for a turn: identify the user, choose the project, load
its configuration, call the engine, and translate the engine's event stream into
SSE for the browser. No prompts, no retrieval, no model calls.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from support_agent.api.options import CHAT_MODELS
from support_agent.api.streaming import ChatResult, collect, to_sse
from support_agent.assistant import ProjectNotFoundError, load_project
from support_agent.engine import EngineDep
from support_agent.engine_client.models import EngineChatRequest, Message

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """What our own frontend sends.

    Deliberately smaller than the engine's request: the browser has no business
    supplying a system prompt or an MCP server list. Those come from
    `projects/*.yaml`, server-side.

    The two things it may choose, it chooses *by name* from a list this backend
    published — a project id, and a model id checked against `CHAT_MODELS`.
    Neither is free text that reaches the engine unexamined.
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8_000)
    session_id: str | None = None
    project: str | None = None
    #: Overrides the assistant's configured model. Checked against an allowlist:
    #: an unchecked model name from a browser is someone else's bill.
    model: str | None = None
    history: list[Message] = Field(default_factory=list)


def _build_request(body: ChatRequest, user_id: str) -> EngineChatRequest:
    try:
        project = load_project(body.project)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if body.model is not None:
        if body.model not in CHAT_MODELS:
            raise HTTPException(status_code=422, detail=f"unknown model: {body.model!r}")
        # A copy, not a mutation: `load_project` is cached and its result shared.
        project = project.model_copy(update={"model": body.model})

    # TODO: replace the header with real authentication, and check that this user
    # is allowed to use this project. The engine only ever sees an opaque id.
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
    """Stream a turn to the browser as server-sent events."""
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
