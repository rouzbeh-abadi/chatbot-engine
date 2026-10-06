"""A cap on a request's body, applied before the body is read.

FastAPI reads a JSON body in full before it checks a single field, so the
limits in `models/` (a message of 32,000 characters, 200 turns of history)
only apply once the whole body is in memory, however large it is. This
refuses a body over `ENGINE_MAX_BODY_BYTES` first: at once when its
`Content-Length` says it is larger, and as soon as it grows past the cap
when it arrives in chunks without one.

The uploads are left to their own limit: `PUT /documents` and
`POST /extract` take a file of up to 25 MB, spooled to disk rather than
held, and check its size themselves.
"""

from __future__ import annotations

from collections.abc import Collection

from fastapi import HTTPException, status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: The routes that take a file, by method and path, with a limit of their own.
UPLOADS = frozenset({("PUT", "/documents"), ("POST", "/extract")})


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
        exempt: Collection[tuple[str, str]] = UPLOADS,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.exempt = exempt

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or (scope["method"], _route(scope)) in self.exempt:
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            response = JSONResponse(
                {"detail": _too_large(self.max_bytes)},
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
                if received > self.max_bytes:
                    # Raised where the route reads its body: FastAPI passes an
                    # HTTPException from there on, and it answers 413.
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=_too_large(self.max_bytes),
                    )
            return message

        await self.app(scope, counted, send)


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
