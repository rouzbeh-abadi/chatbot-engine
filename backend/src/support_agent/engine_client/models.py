"""The engine's wire contract, as this backend sees it.

These models are a deliberate **copy** of `chatbot_engine.models`, not an import.
The engine is a separate service; depending on its Python package would put the
two back in one deployable and defeat the split.

The duplication is the honest cost of a service boundary, and it is guarded:
`tests/test_contract_parity.py` (at the repository root, where neither service
owns it) compares the two schemas and fails when they drift.

Only what this backend actually sends or reads lives here. If a field is not in
this file, the backend does not use it.
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
    """An MCP server the engine should connect to, and the tools it may use.

    `allowed_tools` must be non-empty: names and descriptions come back from the
    server and end up in the engine's prompt, so this backend pins the list
    rather than letting a server offer whatever it likes.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    allowed_tools: list[str] = Field(min_length=1)


class AssistantConfig(BaseModel):
    """The assistant definition, loaded from `projects/*.yaml`.

    This backend is the source of truth: the engine stores none of it and
    receives the whole thing with every request.
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
    #: Opaque to the engine. It forwards this to MCP servers so *they* can
    #: authorise; identity itself stays this backend's responsibility.
    user_id: str | None = None
    history: list[Message] = Field(default_factory=list)


# --- what we read back ------------------------------------------------------


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    score: float
    heading: str | None = None
    excerpt: str | None = None


class RetrievalEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["retrieval"] = "retrieval"
    query: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class TokenEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["token"] = "token"
    text: str


class UsageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["usage"] = "usage"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    model: str | None = None


class ToolCallStartedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """Emitted before a tool runs, so a UI can show what is happening.

    `call_id` pairs this with the matching finished event; a turn may run several
    tools, and they are not guaranteed to finish in the order they started.
    """

    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    tool: str
    server: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolCallFinishedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """Emitted when a tool returns or fails.

    `result_preview` is for display only -- a short excerpt. Tool output is
    untrusted data and belongs in the model's context, not spliced into a UI.
    """

    type: Literal["tool_call_finished"] = "tool_call_finished"
    call_id: str
    tool: str
    ok: bool
    duration_ms: int | None = None
    result_preview: str | None = None
    error: str | None = None


class ErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    code: str
    message: str


class DoneEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["done"] = "done"
    finish_reason: Literal["stop", "length", "tool_limit", "error", "cancelled"] = "stop"


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
    RECEIVED = "received"
    INDEXED = "indexed"
    FAILED = "failed"
    UNCHANGED = "unchanged"


class DocumentRecord(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    deleted: bool
