"""The document contract: upload validation happens before the engine's logic."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _upload(client: TestClient, content: bytes = b"# Baggage\n\nOne bag.\n"):
    return client.put(
        "/documents",
        data={"project_id": "support", "external_id": "baggage.md"},
        files={"file": ("baggage.md", content, "text/markdown")},
    )


def test_upload_reports_no_pipeline_is_registered(client: TestClient) -> None:
    response = _upload(client)

    assert response.status_code == 501
    assert "get_ingest_pipeline" in response.json()["detail"]


def test_empty_upload_is_rejected_before_the_pipeline(client: TestClient) -> None:
    """Validation the engine owns, since it cannot assume its caller validated."""
    response = _upload(client, content=b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_upload_requires_project_and_external_id(client: TestClient) -> None:
    response = client.put(
        "/documents", files={"file": ("a.md", b"data", "text/markdown")}
    )

    assert response.status_code == 422


def test_list_reports_no_registry_is_registered(client: TestClient) -> None:
    response = client.get("/documents", params={"project_id": "support"})

    assert response.status_code == 501
    assert "get_registry" in response.json()["detail"]


def test_list_requires_a_project_id(client: TestClient) -> None:
    assert client.get("/documents").status_code == 422


def test_delete_reports_no_registry_is_registered(client: TestClient) -> None:
    response = client.delete("/documents/abc", params={"project_id": "support"})

    assert response.status_code == 501
