"""Small shared shapes: the base of every request body, and some responses."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

#: Half of a character written as two UTF-16 units. JSON carries one as
#: `\ud83d`, which a parser takes, and nothing after can encode it.
_HALF = re.compile("[\ud800-\udfff]")


def whole(value: Any) -> Any:
    """`value` with every half of a character, in every string it holds, as
    the replacement character U+FFFD."""
    if isinstance(value, str):
        return _HALF.sub("\ufffd", value)
    if isinstance(value, dict):
        return {whole(key): whole(item) for key, item in value.items()}
    if isinstance(value, list):
        return [whole(item) for item in value]
    return value


class Incoming(BaseModel):
    """A request body. Every string in it arrives whole: a caller that cut a
    string through an emoji (JavaScript's `slice` does) sends the half as
    `\\ud83d`, which would fail the turn where it is encoded, the provider's
    client or the error answer itself, and every later turn that carries it
    (docs/review-2026-10.md, API-4). The half becomes U+FFFD."""

    @model_validator(mode="before")
    @classmethod
    def _whole(cls, data: Any) -> Any:
        return whole(data)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    service: str = "chatbot-engine"
    version: str
    #: `low` once a volume the engine writes to has less than twice
    #: `ENGINE_MIN_FREE_MB` free: time to make room, before uploads are
    #: refused at the floor itself (since 0.1.32).
    disk: Literal["ok", "low"] = "ok"


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
