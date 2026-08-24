"""Models for data exchanged between this backend and the chatbot engine.

The backend sends requests such as `EngineChatRequest` and receives streamed
events such as tokens, retrieval results, tool-call updates, usage, and errors.

These models are copied here instead of imported from the engine so the backend
and engine stay independent services. A contract test checks that both sides
still use the same request and response shapes.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- what we send -----------------------------------------------------------


class Message(BaseModel):
    """One turn of conversation history."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class McpServerConfig(BaseModel):
    """An MCP tool server the engine may call, and the tools it is allowed to use.

    `allowed_tools` must be non-empty. A tool's name and description are placed
    into the engine's prompt, so the backend pins an explicit allowlist rather
    than trusting whatever the server advertises.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    allowed_tools: list[str] = Field(min_length=1)


class AssistantConfig(BaseModel):
    """The full assistant definition, loaded from `projects/*.yaml`.

    The backend is the source of truth: the engine stores none of this and
    receives the whole config on every request.
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str
    name: str
    system_prompt: str
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int = Field(default=5, ge=1, le=100)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    max_tool_iterations: int = Field(default=6, ge=1, le=50)


class EngineChatRequest(BaseModel):
    """The body of the engine's `POST /chat`."""

    model_config = ConfigDict(extra="forbid")

    project: AssistantConfig
    message: str = Field(min_length=1)
    session_id: str | None = None
    # Opaque to the engine: it forwards this to MCP tool servers for their own
    # authorization. User identity remains the backend's responsibility.
    user_id: str | None = None
    history: list[Message] = Field(default_factory=list)


# --- what we read back ------------------------------------------------------


class SourceRef(BaseModel):
    """One retrieved document chunk, cited as a source for the answer."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    score: float
    heading: str | None = None
    excerpt: str | None = None


class RetrievalEvent(BaseModel):
    """The sources retrieved for the turn, sent before the answer begins."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["retrieval"] = "retrieval"
    query: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class TokenEvent(BaseModel):
    """One piece of the answer text, streamed as the model generates it."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["token"] = "token"
    text: str


class UsageEvent(BaseModel):
    """Token counts and cost for the turn, sent once near the end."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["usage"] = "usage"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    model: str | None = None


class ToolCallStartedEvent(BaseModel):
    """Emitted before a tool runs, so the UI can show it in progress.

    `call_id` pairs this with its `ToolCallFinishedEvent`. A turn may run several
    tools, and they need not finish in the order they started.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    tool: str
    server: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolCallFinishedEvent(BaseModel):
    """Emitted when a tool returns or fails.

    `result_preview` is a short excerpt for display only. Full tool output is
    untrusted data and goes into the model's context, not into the UI.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call_finished"] = "tool_call_finished"
    call_id: str
    tool: str
    ok: bool
    duration_ms: int | None = None
    result_preview: str | None = None
    error: str | None = None


class ErrorEvent(BaseModel):
    """A failure reported mid-stream, after the 200 response has already begun."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    code: str
    message: str


class DoneEvent(BaseModel):
    """The final event of a turn; `finish_reason` says how it ended."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["done"] = "done"
    finish_reason: Literal["stop", "length", "tool_limit", "error", "cancelled"] = "stop"


# One chat turn arrives as a sequence of these events. Pydantic uses the `type`
# field to decide which model to parse each one into (the discriminator).
ChatEvent = Annotated[
    RetrievalEvent
    | TokenEvent
    | ToolCallStartedEvent
    | ToolCallFinishedEvent
    | UsageEvent
    | ErrorEvent
    | DoneEvent,
    Field(discriminator="type"),
]


class IngestStatus(StrEnum):
    """Where an uploaded document is in the indexing lifecycle."""

    RECEIVED = "received"
    INDEXED = "indexed"
    FAILED = "failed"
    UNCHANGED = "unchanged"


class DocumentRecord(BaseModel):
    """One document in the knowledge base, as the engine reports it."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    external_id: str
    project_id: str
    filename: str
    mimetype: str
    size_bytes: int
    content_hash: str
    status: IngestStatus
    chunk_count: int = 0
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DeleteResult(BaseModel):
    """The outcome of deleting a document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    deleted: bool
