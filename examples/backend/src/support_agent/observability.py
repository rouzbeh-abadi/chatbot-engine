"""A request id on every log line, and forwarded to the engine.

The backend is where a request enters, so it is where the id is minted unless
the caller sent one. It goes out again on every call to the engine, which
keeps it and forwards it to the tool server, so one id follows a turn through
all three processes.
"""

from __future__ import annotations

import logging
import re
import uuid
from contextvars import ContextVar
from typing import Literal

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-Id"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def request_id() -> str | None:
    return _request_id.get()


def request_id_header() -> dict[str, str]:
    """The header to put on an outgoing call, or nothing outside a request."""
    rid = request_id()
    return {REQUEST_ID_HEADER: rid} if rid else {}


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        offered = request.headers.get(REQUEST_ID_HEADER, "")
        rid = offered if _REQUEST_ID.match(offered) else uuid.uuid4().hex
        token = _request_id.set(rid)
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)
        response.headers[REQUEST_ID_HEADER] = rid
        return response


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id() or "-"
        return True


def configure_logging(level: str, log_format: Literal["text", "json"]) -> None:
    import json

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return json.dumps(
                {
                    "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "request_id": getattr(record, "request_id", "-"),
                },
                default=str,
            )

    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        JsonFormatter()
        if log_format == "json"
        else logging.Formatter("%(levelname)s %(name)s [%(request_id)s] %(message)s")
    )
    # Replace only the handler this function installed before: a test harness
    # or a host application may have its own on the root logger.
    handler.set_name("support_agent")
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == "support_agent":
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
