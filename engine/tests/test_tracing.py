"""Tracing: off costs nothing, ids travel with every run, misconfiguration stops startup."""

from __future__ import annotations

import os
import time

import pytest

from chatbot_engine import tracing
from chatbot_engine.errors import EngineError
from chatbot_engine.models.chat import AssistantConfig, ChatRequest, TracingConfig
from chatbot_engine.observability import _request_id
from chatbot_engine.settings import Settings


def _request() -> ChatRequest:
    return ChatRequest(
        project=AssistantConfig(
            project_id="shop", name="Shop", system_prompt="Answer."
        ),
        message="hi",
        session_id="sess-1",
        user_id="user-1",
    )


@pytest.fixture(autouse=True)
def _off():
    tracing.configure(Settings(_env_file=None, tracing="off"))
    yield
    tracing.configure(Settings(_env_file=None, tracing="off"))


def test_off_adds_the_ids_but_no_callbacks():
    token = _request_id.set("req-42")
    try:
        config = tracing.run_config(_request(), name="answer")
    finally:
        _request_id.reset(token)
    assert config["run_name"] == "answer"
    assert config["metadata"]["request_id"] == "req-42"
    assert config["metadata"]["project_id"] == "shop"
    assert config["metadata"]["session_id"] == "sess-1"
    assert config["metadata"]["user_id"] == tracing.traced_user("user-1")
    assert "project:shop" in config["tags"]
    assert "callbacks" not in config


def test_a_trace_names_the_person_by_a_keyed_pseudonym(monkeypatch):
    """A chat app's person is a phone number or an email address: the trace
    groups by person without holding it, and the pseudonym cannot be undone
    without the engine's keys."""
    from chatbot_engine.settings import get_settings

    monkeypatch.setenv("ENGINE_API_KEY", "first-key")
    get_settings.cache_clear()
    phone = tracing.traced_user("whatsapp:447700900123")
    assert phone == tracing.traced_user("whatsapp:447700900123")
    assert phone is not None and phone.startswith("user-")
    assert "447700900123" not in phone
    assert phone != tracing.traced_user("whatsapp:447700900124")
    metadata = tracing.run_config(
        _request().model_copy(update={"user_id": "whatsapp:447700900123"}), name="x"
    )["metadata"]
    assert metadata["user_id"] == phone

    monkeypatch.setenv("ENGINE_API_KEY", "another-key")
    get_settings.cache_clear()
    assert tracing.traced_user("whatsapp:447700900123") != phone
    assert tracing.traced_user(None) is None


def test_missing_ids_are_left_out_rather_than_sent_as_null():
    request = ChatRequest(
        project=AssistantConfig(project_id="p", name="P", system_prompt="."),
        message="hi",
    )
    metadata = tracing.run_config(request, name="x")["metadata"]
    assert "session_id" not in metadata
    assert "user_id" not in metadata


@pytest.fixture
def langchain_env(monkeypatch):
    """LangChain's tracing variables, unset, and unset again after the test:
    left behind, every later test's runs would be sent to LangSmith."""
    for key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
        # Set first, so the test's end removes what `configure` writes.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)


def test_langsmith_maps_the_settings_onto_langchains_environment(langchain_env):
    tracing.configure(
        Settings(
            _env_file=None,
            tracing="langsmith",
            langsmith_api_key="ls-key",
            langsmith_project="demo",
        )
    )
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-key"
    assert os.environ["LANGCHAIN_PROJECT"] == "demo"


def test_langsmith_without_a_key_refuses_to_start_and_turns_nothing_on(langchain_env):
    with pytest.raises(EngineError, match="LANGSMITH_API_KEY"):
        tracing.configure(Settings(_env_file=None, tracing="langsmith"))
    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_langfuse_without_keys_refuses_to_start():
    with pytest.raises(EngineError, match="LANGFUSE_PUBLIC_KEY"):
        tracing.configure(Settings(_env_file=None, tracing="langfuse"))


def test_langfuse_attaches_its_handler_and_its_grouping_keys():
    pytest.importorskip("langfuse")
    tracing.configure(
        Settings(
            _env_file=None,
            tracing="langfuse",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            langfuse_host="http://localhost:1",
        )
    )
    config = tracing.run_config(_request(), name="answer")
    assert len(config["callbacks"]) == 1
    assert config["metadata"]["langfuse_session_id"] == "sess-1"
    assert config["metadata"]["langfuse_user_id"] == tracing.traced_user("user-1")
    assert config["metadata"]["langfuse_tags"] == ["shop"]


def _own(public_key: str) -> ChatRequest:
    return ChatRequest(
        project=AssistantConfig(
            project_id="tenant",
            name="Tenant",
            system_prompt=".",
            tracing=TracingConfig(
                public_key=public_key, secret_key="sk", host="http://localhost:1"
            ),
        ),
        message="hi",
        session_id="s",
        user_id="u",
    )


def test_an_assistant_with_its_own_langfuse_traces_there_even_when_the_engine_is_off():
    pytest.importorskip("langfuse")
    config = tracing.run_config(_own("pk-tenant"), name="answer")
    assert len(config["callbacks"]) == 1
    assert config["metadata"]["langfuse_session_id"] == "s"
    assert config["metadata"]["langfuse_tags"] == ["tenant"]


def test_the_handler_for_a_key_is_created_once():
    pytest.importorskip("langfuse")
    first = tracing.run_config(_own("pk-once"), name="a")["callbacks"][0]
    second = tracing.run_config(_own("pk-once"), name="b")["callbacks"][0]
    assert first is second


def _destination(public_key: str, host: str, secret: str):
    request = ChatRequest(
        project=AssistantConfig(
            project_id="tenant",
            name="Tenant",
            system_prompt=".",
            tracing=TracingConfig(public_key=public_key, secret_key=secret, host=host),
        ),
        message="hi",
    )
    return tracing.run_config(request, name="answer")["callbacks"][0]


def test_one_public_key_named_with_two_hosts_makes_two_destinations_that_never_share_a_span(
    monkeypatch,
):
    """Langfuse routes a span to every exporter of its public key on a shared
    tracer provider; an assistant's own destination has a provider of its
    own, so a span made for one host is never sent to another that named the
    same key (EVALTRACE-1)."""
    pytest.importorskip("langfuse")
    from langfuse._client.span_processor import LangfuseSpanProcessor

    ended: list[tuple[int, str]] = []
    monkeypatch.setattr(
        LangfuseSpanProcessor,
        "on_end",
        lambda self, span: ended.append((id(self), span.name)),
    )
    first = _destination("pk-shared", "http://localhost:1", "sk-first")
    second = _destination("pk-shared", "http://localhost:2", "sk-second")
    assert first is not second
    assert first._langfuse_client._resources.base_url == "http://localhost:1"
    assert second._langfuse_client._resources.base_url == "http://localhost:2"

    owner = {}
    for name, handler in (("first", first), ("second", second)):
        provider = handler._langfuse_client._resources.tracer_provider
        for processor in provider._active_span_processor._span_processors:
            owner[id(processor)] = name
    first._langfuse_client._otel_tracer.start_span("made for first").end()
    second._langfuse_client._otel_tracer.start_span("made for second").end()

    assert [(owner.get(i, "elsewhere"), span) for i, span in ended] == [
        ("first", "made for first"),
        ("second", "made for second"),
    ]


def test_a_destination_that_cannot_be_made_leaves_the_turn_untraced_not_failed(
    monkeypatch,
):
    """Whatever Langfuse does with an assistant's keys, the turn still runs:
    untraced, with nothing of the attempt left in the SDK's registry."""
    pytest.importorskip("langfuse")
    import langfuse
    from langfuse._client.resource_manager import LangfuseResourceManager

    def refuse(**_kwargs):
        raise ValueError("no")

    monkeypatch.setattr(langfuse, "Langfuse", refuse)
    request = ChatRequest(
        project=AssistantConfig(
            project_id="tenant",
            name="Tenant",
            system_prompt=".",
            tracing=TracingConfig(
                public_key="pk-refused", secret_key="sk", host="http://localhost:1"
            ),
        ),
        message="hi",
    )
    assert "callbacks" not in tracing.run_config(request, name="answer")
    assert "pk-refused" not in LangfuseResourceManager._instances
    assert not tracing._per_project


def test_a_destination_pushed_out_is_shut_down(monkeypatch):
    """Each destination holds threads; past the cap, the one used longest ago
    goes, and its threads with it (EVALTRACE-2)."""
    pytest.importorskip("langfuse")
    monkeypatch.setattr(tracing, "_PER_PROJECT_MAX", 2)
    oldest = _destination("pk-1", "http://localhost:1", "sk")
    _destination("pk-2", "http://localhost:1", "sk")
    _destination("pk-1", "http://localhost:1", "sk")  # used again, so kept
    (dropped,) = [d for k, d in tracing._per_project.items() if k[0] == "pk-2"]
    _destination("pk-3", "http://localhost:1", "sk")

    kept = {key[0] for key in tracing._per_project}
    assert kept == {"pk-1", "pk-3"}
    assert _destination("pk-1", "http://localhost:1", "sk") is oldest
    # Shut down off the request, a moment later.
    deadline = time.monotonic() + 5
    while (
        not getattr(dropped.resources, "_shutdown", False)
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert dropped.resources._shutdown


def test_the_tracing_block_rejects_unknown_fields_and_empty_keys():
    with pytest.raises(ValueError):
        TracingConfig(public_key="", secret_key="x")
    with pytest.raises(ValueError):
        TracingConfig.model_validate({"public_key": "a", "secret_key": "b", "extra": 1})
