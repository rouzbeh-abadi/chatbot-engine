"""Request and response models for the backend's own HTTP API.

These are the shapes the frontend sends and receives. They are distinct from
`engine_client.models`, which mirrors the engine's wire contract; keep the two
separate so a change to our public API is not mistaken for an engine change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from support_agent.engine_client.models import (
    ErrorEvent,
    SourceRef,
    ToolCallFinishedEvent,
    UsageEvent,
)

#: The most characters one message may have.
MESSAGE_CHARS = 8_000

# The engine's own bounds (`EngineChatRequest`), mirrored so a request it would
# refuse is a 422 here rather than a failure inside the route.
SESSION_ID_CHARS = 256
HISTORY_TURNS = 200
TURN_CHARS = 32_000

#: How much history, in characters, goes with one message: eight times the
#: message cap. The browser holds the history, so without this it decides how
#: large every model call is, up to the engine's whole prompt budget.
HISTORY_CHARS = 8 * MESSAGE_CHARS


class HistoryTurn(BaseModel):
    """One earlier turn of the conversation, as the browser sends it back.

    Not the engine's `Message`, whose role may also be `system`: the engine
    reads a system turn as the operator's own words, so one taken from the
    browser would let a visitor speak with the operator's authority. A
    conversation has two roles, and only those are accepted.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=TURN_CHARS)


class ChatRequest(BaseModel):
    """Request body for the chat endpoints.

    Only these fields are accepted from the client. The system prompt, tools, and
    other engine settings are loaded server-side from the project config, so the
    client cannot override them (see `chat._build_request`).
    """

    model_config = ConfigDict(extra="forbid")

    # The user's message. Required.
    message: str = Field(min_length=1, max_length=MESSAGE_CHARS)
    # Conversation id, echoed back by the engine. Optional.
    session_id: str | None = Field(default=None, max_length=SESSION_ID_CHARS)
    # Assistant to use, by config name. Defaults to the default project.
    project: str | None = None
    # Model override, by name. Must be one of CHAT_MODELS; defaults to the
    # project's configured model.
    model: str | None = None
    # Agent override, by name. Validated by the engine, which is the only place
    # that knows which agents are installed; defaults to the project's `agent`.
    agent: str | None = None
    # Prior turns, oldest first. Only the newest that fit in HISTORY_CHARS are
    # kept (see `_newest_that_fit`).
    history: list[HistoryTurn] = Field(default_factory=list, max_length=HISTORY_TURNS)

    @field_validator("history")
    @classmethod
    def _newest_that_fit(cls, history: list[HistoryTurn]) -> list[HistoryTurn]:
        """The newest turns whose text fits in `HISTORY_CHARS`.

        Cut rather than refused: the example UI sends the whole conversation
        back with every message, so a long one would otherwise stop working
        part way through. The oldest turns go first, and a turn that does not
        fit ends the history there, as the engine does with its own budget.
        """
        room = HISTORY_CHARS
        kept: list[HistoryTurn] = []
        for turn in reversed(history):
            room -= len(turn.content)
            if room < 0:
                break
            kept.append(turn)
        kept.reverse()
        return kept


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
