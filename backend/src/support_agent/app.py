"""FastAPI application: routes, and engine-error to HTTP-status mapping."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from support_agent.api.chat import router as chat_router
from support_agent.api.documents import router as documents_router
from support_agent.api.admin import router as admin_router
from support_agent.api.options import router as options_router
from support_agent.engine_client import (
    EngineError,
    EngineNotImplemented,
    EngineRejected,
    EngineUnavailable,
)

app = FastAPI(
    title="Support Agent API",
    version="0.1.0",
    description=(
        "Travel support assistant. Calls the chatbot-engine service over HTTP and "
        "exposes this project's domain tools to it over MCP."
    ),
)


@app.exception_handler(EngineNotImplemented)
async def engine_not_implemented(_: Request, exc: EngineNotImplemented) -> JSONResponse:
    """The engine is up but that capability is unwritten. Pass the 501 through."""
    return JSONResponse(status_code=501, content={"detail": str(exc)})


@app.exception_handler(EngineUnavailable)
async def engine_unavailable(_: Request, exc: EngineUnavailable) -> JSONResponse:
    """The engine is a separate process; it being down is an expected outcome."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


#: Engine 4xx codes about the caller's own payload, handed back unchanged. Any
#: other 4xx - a bad shared secret, a route we called wrongly - is our fault,
#: and must not look like a problem with their request.
PASS_THROUGH = frozenset({400, 413, 415, 422, 429})


@app.exception_handler(EngineRejected)
async def engine_rejected(_: Request, exc: EngineRejected) -> JSONResponse:
    """A 4xx from the engine. Whose fault it was decides what we send."""
    if exc.status_code in PASS_THROUGH:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(EngineError)
async def engine_failed(_: Request, exc: EngineError) -> JSONResponse:
    """Anything else from the engine is a bad gateway, not our client's fault."""
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Liveness for this backend only. It does not probe the engine."""
    return {"status": "ok"}


app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(options_router)
app.include_router(admin_router)
