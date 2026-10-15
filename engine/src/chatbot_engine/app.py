"""The engine's FastAPI application: routing, auth, and error mapping.

The AI logic sits behind `services/` and `ports/`, so it does not depend on
being served over HTTP.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import openai
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from chatbot_engine import __version__, tracing
from chatbot_engine.agent.registry import UnknownAgentError
from chatbot_engine.api import (
    agents,
    chat,
    documents,
    eval_rag,
    extract,
    health,
    judge,
    metrics,
    projects,
)
from chatbot_engine.api.auth import require_api_key
from chatbot_engine.api.body_limit import FORM_ROOM, BodyLimitMiddleware
from chatbot_engine.api.dependencies import close_agents
from chatbot_engine.api.rate_limit import limit_chat, limit_eval
from chatbot_engine.documents.extractor import UnsupportedDocumentTypeError
from chatbot_engine.errors import (
    DocumentDeletedError,
    DocumentRejectedError,
    EngineError,
    NotConfiguredError,
    StorageFullError,
    provider_reason,
)
from chatbot_engine.models.common import whole
from chatbot_engine.observability import (
    REQUEST_ID_HEADER,
    RequestIdMiddleware,
    configure_logging,
    request_id,
)
from chatbot_engine.rag.vector_store import (
    claim_chroma_dir,
    move_everything,
    release_chroma_dir,
)
from chatbot_engine.settings import get_settings

logger = logging.getLogger(__name__)


def _failed(detail: str, exc: BaseException) -> dict[str, object]:
    """A refusal's body, with `embedded_chunks` when an index failed after
    it had embedded that many chunks: billed by the provider, though none of
    them is kept (docs/review-2026-10.md, INGEST-19)."""
    embedded = getattr(exc, "embedded_chunks", 0)
    return {"detail": detail, **({"embedded_chunks": embedded} if embedded else {})}


class InsecureConfiguration(RuntimeError):
    """`ENGINE_ENV=production` with a default that is only safe on a laptop."""


#: Where a request keeps its id for the catch-all 500 (`unexpected`).
_REQUEST_ID_STATE = "request_id"


class _KeepRequestId:
    """Puts the request's id in its state, for the catch-all 500.

    That handler runs outside every middleware, the request id's included,
    once the request has left them, so the id is no longer the request's to
    ask for; its state, which the handler is given, still holds it. A plain
    ASGI middleware, inside the request id's.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})[_REQUEST_ID_STATE] = request_id()
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Refuse to serve a production deployment with development defaults.

    Checked at startup, not per request: an engine that will hand out provider
    credits to anonymous callers should fail its health check and never enter
    the load balancer, rather than serve and log about it.
    """
    settings = get_settings()
    problems = settings.unsafe_for_production()

    if problems and settings.env == "production":
        raise InsecureConfiguration(
            "refusing to start with ENGINE_ENV=production:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )

    for problem in problems:
        logger.warning("insecure for deployment: %s", problem)

    tracing.configure(settings)
    # One engine process to an embedded Chroma directory: a second is refused
    # here, before it serves, rather than found out once it has written over
    # the first one's indexes.
    claim_chroma_dir()
    # Every project's chunks into a collection of its own (0.1.30), in the
    # background: a project used before its turn is moved then.
    mover = asyncio.create_task(move_everything())
    yield
    mover.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await mover
    await close_agents()
    tracing.flush()
    release_chroma_dir()


#: How long a thread busy in Python keeps the interpreter before another
#: that is waiting gets it. Indexing reads, cuts and scans documents on
#: threads beside the event loop, and the loop gives the interpreter up on
#: every turn: at Python's 5 ms, a request that takes the loop a few dozen
#: turns waited 0.2 s while a 2 MB document was cut, and so did every answer
#: being streamed. At 1 ms it waits a fifth as long, and the threads lose
#: next to nothing.
_SWITCH_INTERVAL_S = 0.001


def create_app() -> FastAPI:
    """Build the FastAPI app: exception handlers, then the routers."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    sys.setswitchinterval(_SWITCH_INTERVAL_S)
    production = settings.env == "production"

    app = FastAPI(
        title="Chatbot Engine",
        version=__version__,
        description=(
            "RAG and MCP tool-calling service. Configuration arrives with every "
            "request, so the engine stores none of it."
        ),
        lifespan=lifespan,
        # The interactive docs and the schema are for building against the
        # engine on a laptop, not a map for whoever reaches it in production.
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )

    # Register most-specific first: Starlette matches handlers by walking the
    # exception's MRO, so NotConfiguredError needs its own entry to beat the
    # EngineError handler below it.
    @app.exception_handler(NotConfiguredError)
    async def not_configured(_: Request, exc: NotConfiguredError) -> JSONResponse:
        """501: the engine is missing configuration it needs for this route."""
        return JSONResponse(status_code=501, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        """422 naming what is wrong and where, never what was sent.

        FastAPI's own answer repeats each error's `input`: a server's whole
        headers, or for a missing field the whole project, its keys included
        (docs/review-2026-10.md, API-5); and one it cannot encode is a 500
        (API-4). Here each error is its type, its place and its message.
        """
        errors = [
            {
                "type": error.get("type"),
                "loc": [whole(part) for part in error.get("loc", ())],
                "msg": whole(str(error.get("msg", ""))),
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(UnknownAgentError)
    async def unknown_agent(_: Request, exc: UnknownAgentError) -> JSONResponse:
        """422: the assistant named an agent this engine does not have."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(DocumentRejectedError)
    async def document_rejected(_: Request, exc: DocumentRejectedError) -> JSONResponse:
        """422: the document was readable but had nothing to index."""
        return JSONResponse(status_code=422, content=_failed(str(exc), exc))

    @app.exception_handler(UnsupportedDocumentTypeError)
    async def unsupported_document_type(
        _: Request, exc: UnsupportedDocumentTypeError
    ) -> JSONResponse:
        """415: no extractor handles this MIME type."""
        return JSONResponse(status_code=415, content={"detail": str(exc)})

    @app.exception_handler(DocumentDeletedError)
    async def document_deleted(_: Request, exc: DocumentDeletedError) -> JSONResponse:
        """409: deleted while it was being indexed; nothing of it is kept."""
        return JSONResponse(status_code=409, content=_failed(str(exc), exc))

    @app.exception_handler(StorageFullError)
    async def storage_full(_: Request, exc: StorageFullError) -> JSONResponse:
        """503: the volume is below its free-space floor; retry after room is made."""
        return JSONResponse(status_code=503, content=_failed(str(exc), exc))

    @app.exception_handler(EngineError)
    async def engine_error(_: Request, exc: EngineError) -> JSONResponse:
        """500: any deliberate engine error that is not one of the above."""
        return JSONResponse(status_code=500, content=_failed(str(exc), exc))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> Response:
        """500, as Starlette answers it; as JSON only for an index that failed
        after it had embedded, so its caller hears what that cost. With the
        request's id, as every other answer has it: the one answer a caller
        most needs to quote (`_KeepRequestId`)."""
        response: Response
        if getattr(exc, "embedded_chunks", 0):
            response = JSONResponse(
                status_code=500,
                content=_failed("the document could not be indexed", exc),
            )
        else:
            response = PlainTextResponse("Internal Server Error", status_code=500)
        rid = request.scope.get("state", {}).get(_REQUEST_ID_STATE)
        if rid:
            response.headers[REQUEST_ID_HEADER] = rid
        return response

    @app.exception_handler(openai.APIError)
    async def provider_error(_: Request, exc: openai.APIError) -> JSONResponse:
        """502: the model provider refused the call or could not be reached.

        Its own reason goes back to the caller, since that is what the caller
        can act on: a model blocked by an account's guardrail, a key with no
        credit, a model that does not exist. A chat turn is not affected; a
        failure after its stream has started arrives as an `error` event.
        """
        logger.warning("provider call failed: %s", provider_reason(exc))
        return JSONResponse(status_code=502, content=_failed(provider_reason(exc), exc))

    # A JSON body over the cap is refused before it is read. Added first, so
    # it runs inside the request id's middleware and a 413 carries the id.
    # An upload's cap is its route's own file limit and room for the form.
    app.add_middleware(
        BodyLimitMiddleware,
        max_bytes=settings.max_body_bytes,
        own_limits={
            ("PUT", "/documents"): lambda: documents.MAX_UPLOAD_BYTES + FORM_ROOM,
            ("POST", "/extract"): lambda: extract.MAX_UPLOAD_BYTES + FORM_ROOM,
        },
    )

    # Inside the request id's middleware, where the id is known, for the
    # catch-all 500 outside it.
    app.add_middleware(_KeepRequestId)

    # Every request gets an id, kept from the caller when it sent one.
    app.add_middleware(RequestIdMiddleware)

    # Health is unauthenticated, so probes can reach it. Metrics are too
    # locally; in production they name callers and models, and need the key
    # unless `ENGINE_METRICS_PUBLIC` opens them for a scraper.
    app.include_router(health.router)
    if settings.metrics_enabled:
        public = (
            settings.metrics_public
            if settings.metrics_public is not None
            else not production
        )
        app.include_router(
            metrics.router,
            dependencies=[] if public else [Depends(require_api_key)],
        )

    # Everything else requires the shared secret (see api/auth.py), and the
    # routes that spend provider credits are metered on top of it. On the
    # routers where every route costs something, so one added later inherits the
    # limit instead of quietly arriving unmetered. Documents are the exception:
    # only the upload embeds, so that limit sits on the route itself -- listing
    # and deleting are free and must not be throttled with it.
    protected = APIRouter(dependencies=[Depends(require_api_key)])
    protected.include_router(agents.router)
    protected.include_router(chat.router, dependencies=[Depends(limit_chat)])
    protected.include_router(documents.router)
    protected.include_router(projects.router)
    # Reads a file and keeps nothing. A document calls no provider, so the router
    # is not metered; an image is read by a model and metered in the route.
    protected.include_router(extract.router)
    protected.include_router(judge.router, dependencies=[Depends(limit_eval)])
    protected.include_router(eval_rag.router, dependencies=[Depends(limit_eval)])
    app.include_router(protected)

    return app


app = create_app()
