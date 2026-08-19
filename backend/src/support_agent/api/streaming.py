"""Server-sent events, for the frontend.

This is where the browser-facing transport is decided. The engine streams NDJSON
because its client is a service; the frontend wants SSE, so the translation
happens here — one hop from the browser, in the service that actually knows it is
talking to one.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from support_agent.engine_client.models import (
    ChatEvent,
    DoneEvent,
    ErrorEvent,
    RetrievalEvent,
    SourceRef,
    TokenEvent,
    ToolCallFinishedEvent,
    UsageEvent,
)


def sse_frame(event: ChatEvent) -> str:
    """Render one event as an SSE frame.

    The trailing blank line matters -- without it a browser buffers forever.
    """
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


async def to_sse(events: AsyncIterable[ChatEvent]) -> AsyncIterator[str]:
    """Frame a turn for the browser.

    A failure part-way through becomes a terminal `error` + `done` pair rather
    than an exception: the 200 status has already gone out, so raising here would
    reach the browser as a truncated response with no explanation.
    """
    try:
        async for event in events:
            yield sse_frame(event)
    except Exception as exc:  # noqa: BLE001 - last line of defence for the stream
        yield sse_frame(ErrorEvent(code="engine_error", message=str(exc)))
        yield sse_frame(DoneEvent(finish_reason="error"))


class ChatResult(BaseModel):
    """A whole turn, collected -- what the non-streaming endpoint returns."""

    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    sources: list[SourceRef] = Field(default_factory=list)
    #: Which tools ran, so a non-streaming client can show them too. The
    #: finished events carry everything worth reporting.
    tool_calls: list[ToolCallFinishedEvent] = Field(default_factory=list)
    usage: UsageEvent | None = None
    finish_reason: str = "stop"
    error: ErrorEvent | None = None


async def collect(events: AsyncIterable[ChatEvent]) -> ChatResult:
    """Fold a turn into one result, for clients that do not stream."""
    result = ChatResult()
    parts: list[str] = []

    async for event in events:
        match event:
            case TokenEvent():
                parts.append(event.text)
            case RetrievalEvent():
                result.sources = event.sources
            case ToolCallFinishedEvent():
                result.tool_calls.append(event)
            case UsageEvent():
                result.usage = event
            case DoneEvent():
                result.finish_reason = event.finish_reason
            case ErrorEvent():
                result.error = event
                result.finish_reason = "error"

    result.answer = "".join(parts)
    return result
