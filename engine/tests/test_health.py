"""Health is unauthenticated; readiness says whether a turn can be served."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.dependencies import reset_dependency_cache


def test_health_needs_no_api_key(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["service"] == "chatbot-engine"
    assert body["version"]


def test_readiness_is_true_with_a_provider_key(client: TestClient) -> None:
    body = client.get("/health/ready").json()

    assert body["ready"] is True
    assert body["model_provider"] is True
    assert body["vector_store"] is True
    assert "loop" in body["agents"]


def test_readiness_is_false_without_a_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documents can still be uploaded, but nothing can be answered, and a probe
    gating on `ready` should say so. The status stays 200 so the body is
    readable."""
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "")
    reset_dependency_cache()

    from chatbot_engine.app import create_app

    with TestClient(create_app()) as client:
        response = client.get("/health/ready")

    reset_dependency_cache()

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["model_provider"] is False


def test_a_blank_api_key_env_var_leaves_the_engine_open(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object]
) -> None:
    """`ENGINE_API_KEY=` in .env arrives as "" -- it must mean "unset", not
    "the key is the empty string", or copying .env.example breaks everything."""
    monkeypatch.setenv("ENGINE_API_KEY", "")
    reset_dependency_cache()

    from chatbot_engine.app import create_app

    with TestClient(create_app()) as client:
        # A route that needs no model, so this tests authentication and nothing
        # else: 200 means the request got past it.
        response = client.get("/documents", params={"project_id": "support"})
        assert response.status_code == 200

    reset_dependency_cache()




def test_readiness_is_false_when_the_vector_store_does_not_answer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine that can neither index nor retrieve must not take traffic,
    whatever its provider key says."""
    from chatbot_engine.api import health

    monkeypatch.setattr(health, "vector_store_reachable", lambda: False)

    body = client.get("/health/ready").json()

    assert body["vector_store"] is False
    assert body["ready"] is False
    assert body["model_provider"] is True
