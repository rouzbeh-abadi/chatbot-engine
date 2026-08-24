"""Adapt the engine's event stream for the browser.

The engine streams NDJSON (one JSON object per line); browsers consume SSE. This
module translates one to the other for the streaming endpoint, and folds the
whole stream into a single result for the non-streaming one.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from support_agent.api.schemas import ChatResult
from support_agent.engine_client.models import (
    ChatEvent,
    DoneEvent,
    ErrorEvent,
    RetrievalEvent,
    TokenEvent,
    ToolCallFinishedEvent,
    UsageEvent,
)


def sse_frame(event: ChatEvent) -> str:
    """Format one event as an SSE frame.

    The trailing blank line terminates the frame; without it the browser buffers
    the event instead of dispatching it.
    """
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


async def to_sse(events: AsyncIterable[ChatEvent]) -> AsyncIterator[str]:
    """Convert engine events into SSE messages for the browser.
    If streaming fails after the response has started, send `error` and `done`
    events so the browser knows the stream ended because of a failure.

    """
    try:
        async for event in events:
            yield sse_frame(event)
    except Exception as exc:  # noqa: BLE001 - last line of defence for the stream
        yield sse_frame(ErrorEvent(code="engine_error", message=str(exc)))
        yield sse_frame(DoneEvent(finish_reason="error"))


async def collect(events: AsyncIterable[ChatEvent]) -> ChatResult:
    """Consume the full event stream and fold it into one `ChatResult`."""
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
