"""What a chat turn emits, one event at a time.

A turn is a sequence of events rather than a single answer, because a UI needs
retrieved sources, tool progress and token cost while the answer is still being
written. `POST /chat` streams these as NDJSON: one JSON object per line.

Add `tool_call_started` / `tool_call_finished` here when the tool-calling loop
lands -- same shape, a `type` discriminator plus a payload.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRef(BaseModel):
    """One retrieved chunk, reduced to what a UI needs in order to cite it."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    score: float
    heading: str | None = None
    excerpt: str | None = None


class RetrievalEvent(_Event):
    """Emitted when retrieval finishes, so sources can render before the answer."""

    type: Literal["retrieval"] = "retrieval"
    query: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class TokenEvent(_Event):
    """One incremental piece of the answer."""

    type: Literal["token"] = "token"
    text: str


class UsageEvent(_Event):
    """Token counts and cost for the turn."""

    type: Literal["usage"] = "usage"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    model: str | None = None


class ErrorEvent(_Event):
    """May arrive mid-stream, once the 200 status has already been sent."""

    type: Literal["error"] = "error"
    code: str
    message: str


class DoneEvent(_Event):
    """Always last. `finish_reason` says why the turn ended."""

    type: Literal["done"] = "done"
    finish_reason: Literal["stop", "length", "tool_limit", "error", "cancelled"] = "stop"


Event = Annotated[
    RetrievalEvent | TokenEvent | UsageEvent | ErrorEvent | DoneEvent,
    Field(discriminator="type"),
]
