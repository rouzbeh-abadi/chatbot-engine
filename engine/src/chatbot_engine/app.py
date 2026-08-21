"""The engine service.

    uvicorn chatbot_engine.app:app --port 8100

Thin on purpose: routing, authentication, error mapping. The AI logic sits behind
`services/` and `ports/`, so none of it needs to know it is behind HTTP.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from chatbot_engine import __version__
from chatbot_engine.api import chat, documents, health, judge
from chatbot_engine.api.deps import require_api_key
from chatbot_engine.documents.extractor import UnsupportedDocumentTypeError
from chatbot_engine.errors import (
    DocumentRejectedError,
    EngineError,
    NotConfiguredError,
)
from chatbot_engine.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="Chatbot Engine",
        version=__version__,
        description=(
            "RAG and MCP tool-calling service. Configuration arrives with every "
            "request, so the engine stores none of it."
        ),
    )

    # Most specific first: Starlette resolves handlers by walking the exception's
    # MRO, so `NotConfiguredError` needs its own entry to beat `EngineError`.
    @app.exception_handler(NotConfiguredError)
    async def not_configured(_: Request, exc: NotConfiguredError) -> JSONResponse:
        """A capability has no implementation yet -- say so, do not 500."""
        return JSONResponse(status_code=501, content={"detail": str(exc)})

    @app.exception_handler(NotImplementedError)
    async def not_implemented(_: Request, exc: NotImplementedError) -> JSONResponse:
        """A bare `raise NotImplementedError` from half-written code, too."""
        return JSONResponse(status_code=501, content={"detail": str(exc)})

    @app.exception_handler(DocumentRejectedError)
    async def document_rejected(
        _: Request, exc: DocumentRejectedError
    ) -> JSONResponse:
        """Readable, and still nothing to index."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(UnsupportedDocumentTypeError)
    async def unsupported_document_type(
        _: Request, exc: UnsupportedDocumentTypeError
    ) -> JSONResponse:
        """No extractor handles this MIME type."""
        return JSONResponse(status_code=415, content={"detail": str(exc)})

    @app.exception_handler(EngineError)
    async def engine_error(_: Request, exc: EngineError) -> JSONResponse:
        """Anything the engine raises deliberately that is not "unwritten"."""
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # Unauthenticated: probes and `docker compose` healthchecks use it.
    app.include_router(health.router)

    # Everything else sits behind the optional shared secret.
    protected = APIRouter(dependencies=[Depends(require_api_key)])
    protected.include_router(chat.router)
    protected.include_router(documents.router)
    protected.include_router(judge.router)
    app.include_router(protected)

    return app


app = create_app()
