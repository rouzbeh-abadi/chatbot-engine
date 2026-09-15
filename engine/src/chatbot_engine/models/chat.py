"""What the backend sends when it asks for a chat turn.

The engine stores no configuration. The caller is the source of truth and sends
the whole assistant definition with every request, which is why the engine can
be restarted, scaled out, or shared by several applications without migrating
anything.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chatbot_engine.models.workflow import WorkflowSpec


class Message(BaseModel):
    """One turn of conversation history, supplied by the caller."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=32_000)


class TracingConfig(BaseModel):
    """Where this assistant's turns are traced, when it is not the engine's default.

    Lets a caller send each assistant's traces to its own Langfuse, self-hosted
    or a cloud project, rather than the one the engine is configured with.
    The keys travel with the request, so the engine must only be reachable
    over TLS from the backend, as the deployment guide requires anyway.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["langfuse"] = "langfuse"
    public_key: str = Field(min_length=1)
    secret_key: str = Field(min_length=1)
    host: str = "https://cloud.langfuse.com"


#: An HTTP header name (RFC 9110 token characters).
_HEADER_NAME = re.compile(r"[A-Za-z0-9!#$%&'*+.^_`|~-]{1,64}")
#: Headers a server entry may not replace: the engine's own request id, which
#: lines the tool server's logs up with the engine's.
_ENGINE_HEADERS = frozenset({"x-request-id"})
_MAX_SERVER_HEADERS = 10
_MAX_HEADER_VALUE = 4096


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
    #: HTTP headers sent to this server only, with discovery and with every
    #: call: a credential, or an identity only this server should see. They
    #: take precedence over `X-User-Id` and `X-Session-Id`, so one server can
    #: receive its own user identity (a token the embedding site signed, say)
    #: while every other server, the traces and the logs keep the request's
    #: `user_id`. Never logged, traced or shown in a repr.
    headers: dict[str, str] = Field(default_factory=dict, repr=False)

    @field_validator("headers")
    @classmethod
    def _safe_headers(cls, headers: dict[str, str]) -> dict[str, str]:
        # The messages name the header, never its value: a value is usually a
        # secret, and a validation error travels back in the response body.
        if len(headers) > _MAX_SERVER_HEADERS:
            raise ValueError(f"at most {_MAX_SERVER_HEADERS} headers per server")
        for name, value in headers.items():
            if not _HEADER_NAME.fullmatch(name):
                raise ValueError(f"{name!r} is not a valid header name")
            if name.lower() in _ENGINE_HEADERS:
                raise ValueError(f"{name} is set by the engine")
            if len(value) > _MAX_HEADER_VALUE or any(c in value for c in "\r\n\x00"):
                raise ValueError(
                    f"the value of {name} is too long or not a single line"
                )
        return headers


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
    #: The most tokens one model reply may contain. The provider stops the
    #: reply there and the turn's `done` event says `length`. None leaves it
    #: to the provider. Small calls around the answer (rewrite, rerank,
    #: condition) ignore it.
    max_output_tokens: int | None = Field(default=None, ge=1, le=128_000)
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
    #: The least vector similarity (0 to 1, about the cosine of the two
    #: embeddings) a chunk needs to reach the model. A message nothing in the
    #: knowledge base is close to ("hi", "thanks") then gets no chunks at all,
    #: so the answer's prompt carries no context and no rerank call is made.
    #: With `hybrid`, the best keyword match is kept alongside, but only when
    #: at least one chunk clears the bar. What a good value is
    #: depends on the embedding model; around 0.4 suits
    #: text-embedding-3-small. None keeps every chunk, as before.
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    #: What the visitor is told when a tool the turn needs is unavailable or
    #: fails: a workflow's tool step says it and ends the turn, and a model is
    #: told to say it. None uses the engine's default sentence.
    unavailable_message: str | None = Field(default=None, min_length=1, max_length=500)
    #: This assistant's own trace destination. Unset, the engine's
    #: `ENGINE_TRACING` setting applies.
    tracing: TracingConfig | None = None
    #: The turn as a graph of steps, run by the `workflow` agent. Unset, the
    #: named agent's built-in shape applies. See models/workflow.py.
    workflow: WorkflowSpec | None = None
    #: Bounds the tool-calling loop, so a misbehaving model cannot spin.
    max_tool_iterations: int = Field(default=6, ge=1, le=50)
    #: The caller's own OpenRouter key for this request. Every model call the
    #: request makes -- the answer, the rewrite, the rerank, a judge -- is
    #: billed to it instead of the engine's key, so an application can let
    #: each of its customers pay for their own evaluations or traffic. Set,
    #: it also satisfies the engine's own key requirement: an engine with no
    #: key of its own still serves the request. Embedding the question at
    #: query time stays on the engine's key, since the collection is keyed by
    #: the embedding model, not by who paid. Never logged or traced.
    provider_api_key: str | None = Field(default=None, repr=False, min_length=1)


class ResumeInput(BaseModel):
    """The answer to a question a paused turn asked (`input_required`)."""

    model_config = ConfigDict(extra="forbid")

    #: From the `input_required` event.
    thread_id: str = Field(min_length=1, max_length=200)
    #: The answer: the text typed, or the chosen option's `value`.
    value: str | None = Field(default=None, max_length=2000)
    #: The visitor skipped an optional question.
    skipped: bool = False


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
    #: Continue a turn that paused on a question, with the answer. `message`
    #: is still required: it is how the answer reads in the conversation (the
    #: typed text, or the chosen option's label). Agents that never pause
    #: ignore it.
    resume: ResumeInput | None = None
