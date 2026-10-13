"""NDJSON framing for the chat stream.

One JSON object per line, `application/x-ndjson`. Not SSE: the engine's client is
the application backend, not a browser, and SSE framing is the backend's job when
it forwards to the frontend. NDJSON keeps this API usable from any HTTP client.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator

from chatbot_engine.agent.client import (
    error_text,
    price_usage,
    start_meter,
    usage_event,
)
from chatbot_engine.models.events import DoneEvent, ErrorEvent, Event, UsageEvent

logger = logging.getLogger(__name__)

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


async def to_ndjson(
    events: AsyncIterable[Event], *, model: str | None = None
) -> AsyncIterator[str]:
    """Serialise a run, one event per line.

    A failure part-way through becomes a terminal `error` + `done` pair rather
    than an exception: the 200 status has already been sent, so raising here would
    reach the caller as a truncated body with no explanation. When the turn
    had already spent tokens and not yet said so, a `usage` event with what
    it spent comes first, from the turn's meter (`start_meter`), so the caller
    can count a failed turn as it counts any other; `model` names the model it
    is priced at.

    `done` is the last line. What comes after it (an event, or a failure in
    the work an agent does once its turn has ended) is logged, not sent: a
    caller stops reading at `done`, and a second one would end the turn twice.
    """
    meter = start_meter()
    reported = ended = False
    try:
        async for event in events:
            if ended:
                logger.warning(
                    "a %s event after the turn's done was not sent", event.type
                )
                continue
            if isinstance(event, UsageEvent):
                reported = True
            elif isinstance(event, DoneEvent):
                ended = True
            yield event.model_dump_json() + "\n"
    except Exception as exc:
        if ended:
            logger.warning(
                "the turn failed after its done: %s", error_text(describe(exc))
            )
            return
        if not reported and meter["total_tokens"] > 0:
            yield usage_event(price_usage(meter, model)).model_dump_json() + "\n"
        message = error_text(describe(exc))
        yield ErrorEvent(code="engine_error", message=message).model_dump_json() + "\n"
        yield DoneEvent(finish_reason="error").model_dump_json() + "\n"
