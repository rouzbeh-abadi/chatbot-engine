"""A cap on a request's body, applied before the body is read.

FastAPI reads a JSON body in full before it checks a single field, so the
limits in `models/` (a message of 32,000 characters, 200 turns of history)
only apply once the whole body is in memory, however large it is. This
refuses a body over `ENGINE_MAX_BODY_BYTES` first: at once when its
`Content-Length` says it is larger, and as soon as it grows past the cap
when it arrives in chunks without one.

The uploads, `PUT /documents` and `POST /extract`, take a file of up to
25 MB and have a cap of their own: that and room for the form around it.
Their key is checked here as well, before any of the body is read: FastAPI
parses a form before it runs a route's dependencies, the key's included, so
a caller without the key could otherwise make the engine take in, and spool
to disk, any amount before it heard 401 (docs/review-2026-10.md, API-1).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from fastapi import HTTPException, status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from chatbot_engine.api.auth import _match
from chatbot_engine.settings import get_settings

logger = logging.getLogger(__name__)

#: The routes that take a file, by method and path.
UPLOADS = frozenset({("PUT", "/documents"), ("POST", "/extract")})

#: Room in an upload's body for its form beside the file: the fields, the
#: boundaries and the part headers.
FORM_ROOM = 64 * 1024


def _too_large(max_bytes: int) -> str:
    return f"request body exceeds {max_bytes} bytes"


class BodyLimitMiddleware:
    """Refuse a request body larger than `max_bytes` with 413, unread.

    A plain ASGI middleware rather than a `BaseHTTPMiddleware`, so the body
    passes through as it arrives and is counted on its way to the route.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int,
        own_limits: Mapping[tuple[str, str], Callable[[], int]] | None = None,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.own_limits = own_limits or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        route = (scope["method"], _route(scope))
        own = self.own_limits.get(route)
        # Read for each request, so it is always the route's own limit.
        cap = own() if own is not None else self.max_bytes
        if route in UPLOADS and not _key_offered(scope):
            # As the route's own check logs it: a refused key is the one sign
            # of someone guessing at one. Never the key itself.
            logger.warning(
                "rejected an upload to %s: missing or invalid X-API-Key", route[1]
            )
            response = JSONResponse(
                {"detail": "missing or invalid X-API-Key"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
            await response(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > cap:
            response = JSONResponse(
                {"detail": _too_large(cap)},
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
            await response(scope, receive, send)
            return

        received = 0

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > cap:
                    # Raised where the route reads its body: FastAPI passes an
                    # HTTPException from there on, and it answers 413.
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=_too_large(cap),
                    )
            return message

        await self.app(scope, counted, send)


def _key_offered(scope: Scope) -> bool:
    """Whether the request carries a key the engine knows, or the engine has
    none (an open engine). The route checks it again, and names the caller."""
    credentials = get_settings().credentials()
    if not credentials:
        return True
    for name, value in scope.get("headers", []):
        if name == b"x-api-key":
            return _match(value.decode("latin-1"), credentials) is not None
    return False


def _route(scope: Scope) -> str:
    """The path the routes match, without the prefix a proxy mounts the engine under."""
    path, root = scope["path"], scope.get("root_path", "")
    return path[len(root) :] if root and path.startswith(root) else path


def _content_length(scope: Scope) -> int | None:
    """The body's declared length, or None when it has none, or none that reads as one."""
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None
