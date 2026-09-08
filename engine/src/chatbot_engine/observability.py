"""What the engine reports about itself: a request id, one log line per turn,
and metrics.

Three things an operator needs the moment something goes wrong, none of which
a service gets by default:

- **A request id on every log line**, taken from `X-Request-Id` when the
  caller sends one and generated otherwise, returned in the response, and
  forwarded to the tool server. One id follows a turn from the browser through
  the backend, the engine, and the tools, which is what lets a failure in one
  be found in the logs of the others.
- **One structured line per chat turn**: who asked, which agent and model,
  tokens and cost, tool calls, outcome, duration. Enough to answer "what
  happened to that request" without reading the stream.
- **Metrics** at `GET /metrics`, in Prometheus format: turns by outcome, turn
  latency, tokens and cost by model, tool calls by result.

Logs are text by default and JSON under `ENGINE_LOG_FORMAT=json`, for a
collector that indexes fields.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from contextvars import ContextVar
from typing import Any, Literal

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from chatbot_engine.models.events import (
    DoneEvent,
    ErrorEvent,
    Event,
    ToolCallFinishedEvent,
    UsageEvent,
)

logger = logging.getLogger("chatbot_engine.turn")

REQUEST_ID_HEADER = "X-Request-Id"

#: What a caller-supplied id may look like. Anything else is replaced rather
#: than trusted: the id lands in logs and in headers to other services.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def request_id() -> str | None:
    """The id of the request being handled, or None outside one."""
    return _request_id.get()


def new_request_id() -> str:
    return uuid.uuid4().hex


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach an id to every request, and return it.

    A caller's own id is kept when it is well-formed, so a backend that
    already tags its requests sees the same id in the engine's logs. Otherwise
    one is generated. Either way the response carries it, which is how a
    caller learns which id to quote.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        offered = request.headers.get(REQUEST_ID_HEADER, "")
        rid = offered if _REQUEST_ID.match(offered) else new_request_id()

        token = _request_id.set(rid)
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)

        response.headers[REQUEST_ID_HEADER] = rid
        return response


# --- logging -------------------------------------------------------------------


class RequestIdFilter(logging.Filter):
    """Puts the current request id on every record, as `request_id`."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with whatever fields the record carries."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        # Fields passed as `extra=`, which is how the turn line carries its data.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and key not in line:
                line[key] = value
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


_STANDARD_RECORD_FIELDS = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "request_id"}

_TEXT_FORMAT = "%(levelname)s %(name)s [%(request_id)s] %(message)s"
_HANDLER_NAME = "chatbot_engine"


def configure_logging(level: str, log_format: Literal["text", "json"]) -> None:
    """Root logging for the process: a level, a format, and the request id.

    Replaces the handler this function installed before, and only that one:
    a test harness or a host application may have handlers of its own on the
    root logger, and removing those would silence what they capture.
    """
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        JsonFormatter() if log_format == "json" else logging.Formatter(_TEXT_FORMAT)
    )
    handler.set_name(_HANDLER_NAME)
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


# --- metrics ---------------------------------------------------------------------

TURNS = Counter(
    "chatbot_engine_turns_total",
    "Chat turns, by caller, agent and how they ended.",
    ["caller", "agent", "outcome"],
)
TURN_SECONDS = Histogram(
    "chatbot_engine_turn_seconds",
    "Wall-clock seconds from the request to the last event.",
    ["agent"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
TOKENS = Counter(
    "chatbot_engine_tokens_total",
    "Tokens spent, by caller, model and direction.",
    ["caller", "model", "direction"],
)
COST_USD = Counter(
    "chatbot_engine_cost_usd_total",
    "Priced cost of turns, by caller and model. Zero for unpriced models.",
    ["caller", "model"],
)
TOOL_CALLS = Counter(
    "chatbot_engine_tool_calls_total",
    "Tool calls, by tool and whether the tool reported success.",
    ["tool", "ok"],
)


async def record_turn(
    events: AsyncIterable[Event], *, caller: str, agent: str
) -> AsyncIterator[Event]:
    """Pass a turn's events through, then log and count what happened.

    Wraps the stream rather than the route, because the stream is where the
    facts are: the usage event carries the tokens, the tool events carry the
    calls, and the last event says how the turn ended. Runs in `finally` so a
    turn the client abandoned is still recorded, as `cancelled`.
    """
    started = time.monotonic()
    usage: UsageEvent | None = None
    tools = {"ok": 0, "failed": 0}
    outcome = "cancelled"
    error: str | None = None

    try:
        async for event in events:
            if isinstance(event, UsageEvent):
                usage = event
            elif isinstance(event, ToolCallFinishedEvent):
                tools["ok" if event.ok else "failed"] += 1
                TOOL_CALLS.labels(tool=event.tool, ok=str(event.ok).lower()).inc()
            elif isinstance(event, ErrorEvent):
                error = event.message
            elif isinstance(event, DoneEvent):
                outcome = event.finish_reason
            yield event
    except Exception as exc:
        outcome, error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        seconds = time.monotonic() - started
        model = (usage.model if usage else None) or "unknown"
        TURNS.labels(caller=caller, agent=agent, outcome=outcome).inc()
        TURN_SECONDS.labels(agent=agent).observe(seconds)
        if usage:
            TOKENS.labels(caller=caller, model=model, direction="input").inc(
                usage.input_tokens
            )
            TOKENS.labels(caller=caller, model=model, direction="output").inc(
                usage.output_tokens
            )
            COST_USD.labels(caller=caller, model=model).inc(usage.cost_usd or 0.0)

        logger.info(
            "turn %s in %.2fs: %s tokens, %d tool calls",
            outcome,
            seconds,
            usage.total_tokens if usage else 0,
            tools["ok"] + tools["failed"],
            extra={
                "caller": caller,
                "agent": agent,
                "model": model,
                "outcome": outcome,
                "seconds": round(seconds, 3),
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "cost_usd": usage.cost_usd if usage else None,
                "tool_calls_ok": tools["ok"],
                "tool_calls_failed": tools["failed"],
                "error": error,
            },
        )
