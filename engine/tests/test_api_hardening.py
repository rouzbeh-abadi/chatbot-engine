"""What the engine's HTTP surface shows, and reads, before any route runs.

A JSON body is refused over `ENGINE_MAX_BODY_BYTES` before it is read, by its
`Content-Length` or as it arrives in chunks; the uploads keep their own
limit. Under `ENGINE_ENV=production` the interactive docs and the schema are
not served, and `/metrics` needs the API key unless `ENGINE_METRICS_PUBLIC`
opens it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api import dependencies
from chatbot_engine.models.events import DoneEvent, TokenEvent
from chatbot_engine.services.chat import ChatService


class _Recording:
    """An agent that says it ran."""

    def __init__(self) -> None:
        self.ran = 0

    async def run(self, request):
        self.ran += 1
        yield TokenEvent(text="ok")
        yield DoneEvent()


@pytest.fixture
def capped(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, _Recording]]:
    """An open engine whose JSON bodies may be at most 2,048 bytes."""
    monkeypatch.delenv("ENGINE_API_KEY", raising=False)
    monkeypatch.setenv("ENGINE_MAX_BODY_BYTES", "2048")
    dependencies.reset_dependency_cache()
    from chatbot_engine.app import create_app

    agent = _Recording()
    app = create_app()
    app.dependency_overrides[dependencies.get_chat_service] = lambda: ChatService(
        agent=agent
    )
    with TestClient(app) as client:
        yield client, agent
    dependencies.reset_dependency_cache()


def _chat(project: dict[str, object], size: int) -> dict[str, object]:
    return {"project": project, "message": "x" * size}


def test_a_body_within_the_cap_is_served(capped, project) -> None:
    client, agent = capped

    response = client.post("/chat", json=_chat(project, 100))

    assert response.status_code == 200
    assert agent.ran == 1


def test_a_body_whose_length_is_over_the_cap_is_refused_unread(capped, project):
    client, agent = capped

    response = client.post("/chat", json=_chat(project, 5_000))

    assert response.status_code == 413
    assert response.json()["detail"] == "request body exceeds 2048 bytes"
    assert response.headers["x-request-id"], "the refusal carries the request id"
    assert agent.ran == 0


def test_a_body_sent_in_chunks_is_refused_once_it_grows_past_the_cap(
    capped, project
) -> None:
    """No `Content-Length` to read: the body is counted as it arrives."""
    import json

    client, agent = capped
    body = json.dumps(_chat(project, 5_000)).encode()

    def chunks():
        for start in range(0, len(body), 512):
            yield body[start : start + 512]

    response = client.post(
        "/chat", content=chunks(), headers={"content-type": "application/json"}
    )

    assert response.status_code == 413
    assert agent.ran == 0


def test_an_upload_keeps_its_own_limit(capped) -> None:
    client, _ = capped

    response = client.put(
        "/documents",
        data={"project_id": "support", "external_id": "faq"},
        files={
            "file": (
                "faq.md",
                b"# Returns\n\n" + b"Thirty days. " * 400,
                "text/markdown",
            )
        },
    )

    assert response.status_code == 201, response.text


# --- production -------------------------------------------------------------


@pytest.fixture
def production(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    monkeypatch.setenv("ENGINE_ENV", "production")
    monkeypatch.setenv("ENGINE_API_KEY", "s3cret")
    monkeypatch.delenv("ENGINE_METRICS_PUBLIC", raising=False)
    dependencies.reset_dependency_cache()
    yield monkeypatch
    dependencies.reset_dependency_cache()


def _app_client() -> TestClient:
    from chatbot_engine.app import create_app

    return TestClient(create_app())


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_production_serves_no_docs_or_schema(production, path: str) -> None:
    with _app_client() as client:
        assert client.get(path, headers={"X-API-Key": "s3cret"}).status_code == 404


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_local_serves_the_docs_and_the_schema(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200


def test_production_metrics_need_the_key(production) -> None:
    with _app_client() as client:
        assert client.get("/metrics").status_code == 401
        keyed = client.get("/metrics", headers={"X-API-Key": "s3cret"})
        assert keyed.status_code == 200
        assert keyed.headers["content-type"].startswith("text/plain")


def test_production_metrics_can_be_opened_for_a_scraper(production) -> None:
    production.setenv("ENGINE_METRICS_PUBLIC", "true")
    dependencies.reset_dependency_cache()

    with _app_client() as client:
        assert client.get("/metrics").status_code == 200


def test_production_metrics_can_still_be_switched_off(production) -> None:
    production.setenv("ENGINE_METRICS_ENABLED", "false")
    dependencies.reset_dependency_cache()

    with _app_client() as client:
        assert (
            client.get("/metrics", headers={"X-API-Key": "s3cret"}).status_code == 404
        )
