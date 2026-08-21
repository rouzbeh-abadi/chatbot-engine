"""NDJSON framing for the chat stream.

One JSON object per line, `application/x-ndjson`. Not SSE: the engine's client is
the application backend, not a browser, and SSE framing is the backend's job when
it forwards to the frontend. NDJSON keeps this API usable from any HTTP client.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from chatbot_engine.models.events import DoneEvent, ErrorEvent, Event

MEDIA_TYPE = "application/x-ndjson"


def describe(exc: BaseException) -> str:
    """A readable message, including the causes hidden inside an ExceptionGroup.

    `str(ExceptionGroup)` is only "unhandled errors in a TaskGroup", which says
    nothing. Anything built on anyio task groups -- the MCP client, for one --
    fails that way, so the real cause has to be dug out.
    """
    if isinstance(exc, BaseExceptionGroup):
        inner = "; ".join(describe(sub) for sub in exc.exceptions)

        return f"{exc.message}: {inner}" if inner else exc.message

    described = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__

    if exc.__cause__ is not None:
        return f"{described} (caused by {describe(exc.__cause__)})"

    return described


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
        yield ErrorEvent(
            code="engine_error", message=describe(exc)
        ).model_dump_json() + "\n"
        yield DoneEvent(finish_reason="error").model_dump_json() + "\n"
