"""What the backend sends when it asks for a chat turn.

The engine stores no configuration. The caller is the source of truth and sends
the whole assistant definition with every request, which is why the engine can
be restarted, scaled out, or shared by several applications without migrating
anything.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """One turn of conversation history, supplied by the caller."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=32_000)


class McpServerConfig(BaseModel):
    """An MCP server the engine should connect to as a client.

    `allowed_tools` is required and must be non-empty. Tool names and
    descriptions come from the server and end up inside the prompt, so exposing
    whatever a server happens to offer is a prompt-injection vector. Pin the
    list on the caller's side.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    allowed_tools: list[str] = Field(min_length=1)


class AssistantConfig(BaseModel):
    """Everything that makes one assistant different from another."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    name: str
    system_prompt: str
    model: str | None = None
    #: Which agent runs the turn: `loop`, the engine's built-in tool loop, or
    #: the name of an installed plugin. None uses the engine default. An
    #: unknown name is a 422 listing the installed set.
    agent: str | None = None
    #: The embedding model for this project's knowledge base. Like `model`, the
    #: backend supplies it and the engine falls back to its own default. It keys
    #: the vector collection, so a query only ever meets vectors produced by
    #: the same model.
    embedding_model: str | None = None
    #: How the knowledge base is cut into chunks, and the size cap every
    #: strategy ends with. None uses the engine's defaults. Applied when a
    #: document is ingested, so changing them means re-indexing.
    chunking_strategy: Literal["size", "headings", "page"] | None = None
    chunk_size: int | None = Field(default=None, ge=100, le=8000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=2000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    #: How many chunks reach the model per turn.
    top_k: int = Field(default=5, ge=1, le=100)
    #: How chunks are found. `vector` is similarity search alone; `hybrid`
    #: fuses it with a keyword (BM25) search, which catches exact terms such
    #: as fare names and codes that embeddings blur. None uses the engine
    #: default.
    retrieval: Literal["vector", "hybrid"] | None = None
    #: Whether the assistant's model re-orders the candidates by relevance
    #: before the top `top_k` are kept. One extra model call per turn.
    rerank: bool | None = None
    #: How many candidates each search returns before fusion and reranking.
    #: More improves recall at the cost of a longer rerank prompt.
    retrieval_candidates: int | None = Field(default=None, ge=1, le=200)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    #: Bounds the tool-calling loop, so a misbehaving model cannot spin.
    max_tool_iterations: int = Field(default=6, ge=1, le=50)


class ChatRequest(BaseModel):
    """The body of `POST /chat`.

    `user_id` is opaque: the engine forwards it to MCP servers so *they* can
    authorise the call, and never interprets it itself. Authentication and
    authorisation belong to the application backend.
    """

    model_config = ConfigDict(extra="forbid")

    project: AssistantConfig
    #: Bounded here as well as by the caller: the engine cannot assume its
    #: caller validated anything, and an unbounded prompt is an unbounded bill.
    message: str = Field(min_length=1, max_length=32_000)
    session_id: str | None = Field(default=None, max_length=256)
    user_id: str | None = Field(default=None, max_length=256)
    history: list[Message] = Field(default_factory=list, max_length=200)
