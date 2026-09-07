"""Small shared response shapes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    service: str = "chatbot-engine"
    version: str


class ReadinessResponse(BaseModel):
    """What `/health/ready` reports."""

    model_config = ConfigDict(extra="forbid")

    #: Whether a chat turn can be served: a provider key is set and the vector
    #: store answers.
    ready: bool
    #: Whether a model provider key is configured.
    model_provider: bool
    #: Whether the vector store answers. Always true embedded; against a Chroma
    #: server it is the check that matters.
    vector_store: bool
    #: The agents this engine can run: the built-in, plus installed plugins.
    agents: list[str]
