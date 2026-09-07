"""Liveness and readiness. Unauthenticated, so probes and compose can use it."""

from __future__ import annotations

from fastapi import APIRouter

from chatbot_engine import __version__
from chatbot_engine.agent.registry import available_agents
from chatbot_engine.api.dependencies import SettingsDep
from chatbot_engine.models.common import HealthResponse, ReadinessResponse
from chatbot_engine.rag.vector_store import vector_store_reachable

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> HealthResponse:
    """Is the process up. Always ok if it can answer at all."""
    return HealthResponse(version=__version__)


@router.get("/health/ready")
async def ready(settings: SettingsDep) -> ReadinessResponse:
    """Can this engine serve a chat turn, and with which agents.

    `ready` is false without a model provider key, or when the vector store
    does not answer. A readiness probe should gate on that field rather than on
    the status code, which stays 200 so the reason is readable.
    """
    has_provider = settings.openrouter_api_key is not None
    has_vectors = vector_store_reachable()

    return ReadinessResponse(
        ready=has_provider and has_vectors,
        model_provider=has_provider,
        vector_store=has_vectors,
        agents=sorted(available_agents()),
    )
