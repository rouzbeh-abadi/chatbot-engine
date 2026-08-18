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
    content: str


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
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    #: How many chunks to retrieve per turn.
    top_k: int = Field(default=5, ge=1, le=100)
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
    message: str = Field(min_length=1)
    session_id: str | None = None
    user_id: str | None = None
    history: list[Message] = Field(default_factory=list)
