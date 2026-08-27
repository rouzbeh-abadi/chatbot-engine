"""The engine's FastAPI application: routing, auth, and error mapping.

The AI logic sits behind `services/` and `ports/`, so it does
not depend on being served over HTTP.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from chatbot_engine import __version__
from chatbot_engine.api import chat, documents, eval_rag, health, judge
from chatbot_engine.api.auth import require_api_key
from chatbot_engine.documents.extractor import UnsupportedDocumentTypeError
from chatbot_engine.errors import (
    DocumentRejectedError,
    EngineError,
    NotConfiguredError,
)
from chatbot_engine.settings import get_settings


def create_app() -> FastAPI:
    """Build the FastAPI app: exception handlers, then the routers."""
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

    # Register most-specific first: Starlette matches handlers by walking the
    # exception's MRO, so NotConfiguredError needs its own entry to beat the
    # EngineError handler below it.
    @app.exception_handler(NotConfiguredError)
    async def not_configured(_: Request, exc: NotConfiguredError) -> JSONResponse:
        """501: a capability has no implementation yet."""
        return JSONResponse(status_code=501, content={"detail": str(exc)})

    @app.exception_handler(NotImplementedError)
    async def not_implemented(_: Request, exc: NotImplementedError) -> JSONResponse:
        """501: a bare `raise NotImplementedError` from half-written code."""
        return JSONResponse(status_code=501, content={"detail": str(exc)})

    @app.exception_handler(DocumentRejectedError)
    async def document_rejected(
        _: Request, exc: DocumentRejectedError
    ) -> JSONResponse:
        """422: the document was readable but had nothing to index."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(UnsupportedDocumentTypeError)
    async def unsupported_document_type(
        _: Request, exc: UnsupportedDocumentTypeError
    ) -> JSONResponse:
        """415: no extractor handles this MIME type."""
        return JSONResponse(status_code=415, content={"detail": str(exc)})

    @app.exception_handler(EngineError)
    async def engine_error(_: Request, exc: EngineError) -> JSONResponse:
        """500: any deliberate engine error that is not one of the above."""
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # Health is unauthenticated, so probes and Docker healthchecks can reach it.
    app.include_router(health.router)

    # Everything else requires the shared secret (see api/auth.py).
    protected = APIRouter(dependencies=[Depends(require_api_key)])
    protected.include_router(chat.router)
    protected.include_router(documents.router)
    protected.include_router(judge.router)
    protected.include_router(eval_rag.router)
    app.include_router(protected)

    return app


app = create_app()
