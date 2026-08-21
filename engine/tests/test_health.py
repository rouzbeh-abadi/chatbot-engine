"""Health is unauthenticated and reports which capabilities are wired."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.deps import reset_dependency_cache


def test_health_needs_no_api_key(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["service"] == "chatbot-engine"
    assert body["version"]


def test_health_is_reachable_even_when_a_key_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe must not need the shared secret, or compose cannot start us."""
    monkeypatch.setenv("ENGINE_API_KEY", "s3cret")
    reset_dependency_cache()

    from chatbot_engine.app import create_app

    with TestClient(create_app()) as unauthenticated:
        assert unauthenticated.get("/health").status_code == 200

    reset_dependency_cache()


def test_readiness_reports_which_capabilities_are_wired(client: TestClient) -> None:
    assert client.get("/health/ready").json() == {"chat": True, "documents": True}


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


def test_a_configured_api_key_is_enforced(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object]
) -> None:
    monkeypatch.setenv("ENGINE_API_KEY", "s3cret")
    reset_dependency_cache()

    from chatbot_engine.app import create_app

    with TestClient(create_app()) as client:
        url, params = "/documents", {"project_id": "support"}
        assert client.get(url, params=params).status_code == 401
        assert client.get(
            url, params=params, headers={"X-API-Key": "wrong"}
        ).status_code == 401
        assert client.get(
            url, params=params, headers={"X-API-Key": "s3cret"}
        ).status_code == 200

    reset_dependency_cache()
