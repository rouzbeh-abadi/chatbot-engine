"""Liveness and readiness. Unauthenticated, so probes and compose can use it."""

from __future__ import annotations

from fastapi import APIRouter

from chatbot_engine import __version__
from chatbot_engine.api.deps import ChatServiceDep, DocumentServiceDep
from chatbot_engine.models.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> HealthResponse:
    """Is the process up. Always ok if it can answer at all."""
    return HealthResponse(version=__version__)


@router.get("/health/ready")
async def ready(chat: ChatServiceDep, documents: DocumentServiceDep) -> dict[str, bool]:
    """Which capabilities have an implementation registered.

    Useful while building: it answers "why am I getting a 501" without reading
    the source.
    """
    return {"chat": chat.is_ready, "documents": documents.is_ready}
