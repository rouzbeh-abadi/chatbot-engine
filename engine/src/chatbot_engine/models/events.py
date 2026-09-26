"""What a chat turn emits, one event at a time.

A turn is a sequence of events rather than a single answer, because a UI needs
retrieved sources, tool progress and token cost while the answer is still being
written. `POST /chat` streams these as NDJSON: one JSON object per line.

Every event carries a `type` discriminator, so a caller can add handling for a
new one without breaking on the others.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRef(BaseModel):
    """One retrieved chunk, reduced to what a UI needs in order to cite it.

    `heading` and `page` are present when the chunking strategy recorded them:
    the heading trail under `headings`, the page number under `page`. Either is
    what lets a citation name a section or a page rather than only a file.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    score: float
    heading: str | None = None
    page: int | None = None
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
    """Token counts and cost for the turn.

    The counts are the whole turn's. The `utility_` counts are the part of
    them spent on the utility model (the query rewrite, the rerank, a
    workflow's condition step), named in `utility_model`, so a caller can
    price that part at its own rate; `utility_model` is null when those calls
    ran on the answer model.
    """

    type: Literal["usage"] = "usage"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    model: str | None = None
    utility_input_tokens: int = 0
    utility_output_tokens: int = 0
    utility_model: str | None = None


class ToolCallStartedEvent(_Event):
    """Emitted before a tool runs, so a UI can show what is happening.

    `call_id` pairs this with the matching finished event; a turn may run several
    tools, and they are not guaranteed to finish in the order they started.
    """

    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    tool: str
    server: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolCallFinishedEvent(_Event):
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


class AskOption(BaseModel):
    """One option of a question the turn paused on."""

    model_config = ConfigDict(extra="forbid")

    value: str
    label: str


class InputRequiredEvent(_Event):
    """The turn paused to ask the visitor something, and is waiting for the answer.

    Comes just before `done` (`input_required`). The question's words have
    already streamed as tokens; this says how to answer. Send the answer as a
    new request with `resume: {thread_id, value}` (or `skipped: true` when
    `optional`), and the turn continues from the step that asked. `error` is
    set when the previous answer was refused and the question is asked again.
    """

    type: Literal["input_required"] = "input_required"
    thread_id: str
    node: str
    prompt: str
    input: Literal["text", "phone", "email", "url", "choice"]
    options: list[AskOption] = Field(default_factory=list)
    optional: bool = False
    skip_label: str | None = None
    placeholder: str | None = None
    error: str | None = None
    #: Whether the step reads a typed reply before keeping it (a workflow's
    #: `understand`): a client may then send any words, and a no or a
    #: question back is understood. Off, only a fitting answer is taken.
    understand: bool = True


class ErrorEvent(_Event):
    """May arrive mid-stream, once the 200 status has already been sent."""

    type: Literal["error"] = "error"
    code: str
    message: str


class DoneEvent(_Event):
    """Always last. `finish_reason` says why the turn ended."""

    type: Literal["done"] = "done"
    #: `stop`: the model finished. `length`: cut at `max_output_tokens`.
    #: `tool_limit`: the model was still asking for tools after
    #: `max_tool_iterations` rounds; the answer so far was streamed. `error`:
    #: the turn failed after the response started (an `error` event precedes).
    #: `input_required`: the turn paused on a question (an `input_required`
    #: event precedes) and continues when the answer is sent with `resume`.
    finish_reason: Literal[
        "stop", "length", "tool_limit", "error", "input_required"
    ] = "stop"


Event = Annotated[
    RetrievalEvent
    | TokenEvent
    | ToolCallStartedEvent
    | ToolCallFinishedEvent
    | UsageEvent
    | InputRequiredEvent
    | ErrorEvent
    | DoneEvent,
    Field(discriminator="type"),
]
