"""Request and response models for the backend's own HTTP API.

These are the shapes the frontend sends and receives. They are distinct from
`engine_client.models`, which mirrors the engine's wire contract; keep the two
separate so a change to our public API is not mistaken for an engine change.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from support_agent.engine_client.models import (
    ErrorEvent,
    Message,
    SourceRef,
    ToolCallFinishedEvent,
    UsageEvent,
)


class ChatRequest(BaseModel):
    """Request body for the chat endpoints.

    Only these fields are accepted from the client. The system prompt, tools, and
    other engine settings are loaded server-side from the project config, so the
    client cannot override them (see `chat._build_request`).
    """

    model_config = ConfigDict(extra="forbid")

    # The user's message. Required.
    message: str = Field(min_length=1, max_length=8_000)
    # Conversation id, echoed back by the engine. Optional.
    session_id: str | None = None
    # Assistant to use, by config name. Defaults to the default project.
    project: str | None = None
    # Model override, by name. Must be one of CHAT_MODELS; defaults to the
    # project's configured model.
    model: str | None = None
    # Prior turns, oldest first.
    history: list[Message] = Field(default_factory=list)


class ChatResult(BaseModel):
    """A complete turn collected into one object, returned by `/chat/sync`."""

    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    sources: list[SourceRef] = Field(default_factory=list)
    # Only finished tool calls; they carry the tool name, result, and timing.
    tool_calls: list[ToolCallFinishedEvent] = Field(default_factory=list)
    usage: UsageEvent | None = None
    finish_reason: str = "stop"
    error: ErrorEvent | None = None
