"""Chat endpoints: POST /chat (streaming) and POST /chat/sync."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from support_agent.api.identity import MemoryOwnerDep
from support_agent.api.memory import recall_for_prompt
from support_agent.api.options import CHAT_MODELS
from support_agent.api.rate_limit import limit_chat
from support_agent.api.schemas import ChatRequest, ChatResult
from support_agent.api.streaming import collect, to_sse
from support_agent.assistant import ProjectNotFoundError, load_project
from support_agent.engine import EngineDep
from support_agent.engine_client.models import EngineChatRequest

# Every turn here is a model call on the provider key, so both routes are
# metered. On the router, not the endpoints, so a third one cannot forget.
router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(limit_chat)])


def _project(body: ChatRequest):
    """The assistant config the request names, or 404.

    Separate from `_build_request` because the memory lookup needs the project
    id too, and an unknown project must answer 404 before anything else runs.
    `load_project` is cached, so asking twice costs nothing.
    """
    try:
        return load_project(body.project)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _build_request(
    body: ChatRequest, user_id: str, notes: str = ""
) -> EngineChatRequest:
    """Turn the frontend request into the engine request.

    Loads the assistant config server-side and applies the model override, so the
    engine always receives a complete, validated definition the browser never saw.
    """
    project = _project(body)

    if body.model is not None:
        if body.model not in CHAT_MODELS:
            raise HTTPException(
                status_code=422, detail=f"unknown model: {body.model!r}"
            )
        # A copy, not a mutation: `load_project` is cached and its result shared.
        project = project.model_copy(update={"model": body.model})

    if body.agent is not None:
        # Not checked here: which agents exist depends on what is installed in
        # the engine, so it validates and answers 422 with the installed list,
        # which passes straight back through.
        project = project.model_copy(update={"agent": body.agent})

    if notes:
        # Read deterministically rather than through a tool the model may not
        # think to call. A preference it has to *decide* to look up is a
        # preference it will sometimes ignore, which reads as the assistant
        # having forgotten. Writing stays a tool: the model chooses what is
        # worth keeping, but it never chooses whether to remember at all.
        project = project.model_copy(
            update={"system_prompt": f"{project.system_prompt}\n\n{notes}"}
        )

    # `user_id` has already been decided by `api/identity.py`. Behind a proxy it
    # is the authenticated user; otherwise it is the browser's `X-Client-Id`,
    # which is what lets memory outlive a conversation without authentication.
    # The engine forwards it to the tool server as `X-User-Id` and attaches no
    # meaning to it.
    #
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
    user_id: MemoryOwnerDep,
) -> StreamingResponse:
    """Receive a chat request from the client and stream the engine response back.

    The client request is converted to an engine request, sent to the chatbot engine,
    and the returned events are streamed back to the client using SSE.
    """
    notes = await recall_for_prompt(user_id, _project(body).project_id)
    request = _build_request(body, user_id, notes)

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
    user_id: MemoryOwnerDep,
) -> ChatResult:
    """Non-streaming variant, for smoke tests and simple clients."""
    notes = await recall_for_prompt(user_id, _project(body).project_id)
    request = _build_request(body, user_id, notes)
    return await collect(await engine.start_chat(request))
