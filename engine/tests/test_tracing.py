"""Tracing: off costs nothing, ids travel with every run, misconfiguration stops startup."""

from __future__ import annotations

import os

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
    assert config["metadata"]["user_id"] == "user-1"
    assert "project:shop" in config["tags"]
    assert "callbacks" not in config


def test_missing_ids_are_left_out_rather_than_sent_as_null():
    request = ChatRequest(
        project=AssistantConfig(project_id="p", name="P", system_prompt="."),
        message="hi",
    )
    metadata = tracing.run_config(request, name="x")["metadata"]
    assert "session_id" not in metadata
    assert "user_id" not in metadata


def test_langsmith_maps_the_settings_onto_langchains_environment(monkeypatch):
    for key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
        monkeypatch.delenv(key, raising=False)
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


def test_langsmith_without_a_key_refuses_to_start(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    with pytest.raises(EngineError, match="LANGSMITH_API_KEY"):
        tracing.configure(Settings(_env_file=None, tracing="langsmith"))


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
    assert config["metadata"]["langfuse_user_id"] == "user-1"
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


def test_the_tracing_block_rejects_unknown_fields_and_empty_keys():
    with pytest.raises(ValueError):
        TracingConfig(public_key="", secret_key="x")
    with pytest.raises(ValueError):
        TracingConfig.model_validate({"public_key": "a", "secret_key": "b", "extra": 1})
