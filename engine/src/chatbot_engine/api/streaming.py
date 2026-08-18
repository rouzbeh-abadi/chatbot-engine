"""NDJSON framing for the chat stream.

One JSON object per line, `application/x-ndjson`. Not SSE: the engine's client is
the application backend, not a browser, and SSE framing is the backend's job when
it forwards to the frontend. NDJSON keeps this API usable from any HTTP client.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from chatbot_engine.models.events import DoneEvent, ErrorEvent, Event

MEDIA_TYPE = "application/x-ndjson"


async def to_ndjson(events: AsyncIterable[Event]) -> AsyncIterator[str]:
    """Serialise a run, one event per line.

    A failure part-way through becomes a terminal `error` + `done` pair rather
    than an exception: the 200 status has already been sent, so raising here would
    reach the caller as a truncated body with no explanation.
    """
    try:
        async for event in events:
            yield event.model_dump_json() + "\n"
    except Exception as exc:  # noqa: BLE001 - last line of defence for the stream
        yield ErrorEvent(code="engine_error", message=str(exc)).model_dump_json() + "\n"
        yield DoneEvent(finish_reason="error").model_dump_json() + "\n"
