"""The wire contract must stay identical on both sides of the HTTP boundary.

`support_agent.engine_client.models` is a deliberate copy of
`chatbot_engine.models`, because the two services must not share a Python
package. Copies drift, and the symptom of drift is a confusing 422 at runtime
rather than a failure at build time.

This is the one test that imports both packages. It lives at the repository root
because neither service owns it, and it is a development-time guard only -- in
production the two are deployed separately and never meet in one interpreter.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from chatbot_engine import models as engine_models
from chatbot_engine.models.chat import ChatRequest as EngineChatRequestModel
from support_agent.engine_client import models as backend_models
from support_agent.engine_client.models import EngineChatRequest

#: Same name on both sides, so they can be paired automatically.
SHARED_MODELS = [
    "AssistantConfig",
    "McpServerConfig",
    "Message",
    "SourceRef",
    "RetrievalEvent",
    "TokenEvent",
    "ToolCallStartedEvent",
    "ToolCallFinishedEvent",
    "UsageEvent",
    "ErrorEvent",
    "DoneEvent",
    "DocumentRecord",
    "DeleteResult",
]


def _shape(model: type[BaseModel]) -> object:
    """A model's schema with the human-facing parts removed.

    Titles and descriptions are expected to differ -- each side documents itself
    from its own point of view. Field names, types, defaults and required-ness are
    the contract.
    """
    return _strip(model.model_json_schema(mode="serialization"))


def _strip(node: object) -> object:
    if isinstance(node, dict):
        return {
            key: _strip(value)
            for key, value in sorted(node.items())
            if key not in {"title", "description"}
        }
    if isinstance(node, list):
        return [_strip(item) for item in node]
    return node


@pytest.mark.parametrize("name", SHARED_MODELS)
def test_shared_models_have_identical_schemas(name: str) -> None:
    engine = getattr(engine_models, name)
    backend = getattr(backend_models, name)

    assert _shape(engine) == _shape(backend), (
        f"{name} has drifted between chatbot_engine.models and "
        "support_agent.engine_client.models -- update both"
    )


def test_the_chat_request_bodies_match() -> None:
    """Named differently on purpose: one is *the* request, one is the engine's."""
    assert _shape(EngineChatRequestModel) == _shape(EngineChatRequest)


def test_both_sides_know_the_same_event_types() -> None:
    """A new event the backend cannot parse would fail the whole stream."""
    engine_types = {
        getattr(engine_models, name).model_fields["type"].default
        for name in SHARED_MODELS
        if name.endswith("Event")
    }
    backend_types = {
        getattr(backend_models, name).model_fields["type"].default
        for name in SHARED_MODELS
        if name.endswith("Event")
    }

    assert engine_types == backend_types
    assert engine_types == {
        "retrieval",
        "token",
        "tool_call_started",
        "tool_call_finished",
        "usage",
        "error",
        "done",
    }


def test_ingest_status_values_match() -> None:
    engine_values = {member.value for member in engine_models.IngestStatus}
    backend_values = {member.value for member in backend_models.IngestStatus}

    assert engine_values == backend_values
