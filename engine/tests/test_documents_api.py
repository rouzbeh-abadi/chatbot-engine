"""The document contract, over HTTP.

What the route rejects on its own, then the round trip through a real pipeline.
The 501 path is still tested, against a deliberately empty service, because it
is what every unwired capability does.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.deps import get_document_service
from chatbot_engine.services.documents import DocumentService

MARKDOWN = b"# Baggage\n\nOne cabin bag up to 8 kg, plus one personal item.\n"


def _upload(
    client: TestClient,
    content: bytes = MARKDOWN,
    external_id: str = "baggage.md",
    mimetype: str = "text/markdown",
):
    return client.put(
        "/documents",
        data={"project_id": "support", "external_id": external_id},
        files={"file": ("baggage.md", content, mimetype)},
    )


@pytest.fixture
def unwired(client: TestClient) -> Iterator[TestClient]:
    """A client whose document service has neither a pipeline nor a registry."""
    # A lambda, not the class: FastAPI introspects an override's signature and
    # would try to turn `DocumentService.__init__`'s parameters into query fields.
    client.app.dependency_overrides[get_document_service] = lambda: DocumentService()

    yield client

    client.app.dependency_overrides.clear()


# --- validation the route owns ----------------------------------------------


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


def test_list_requires_a_project_id(client: TestClient) -> None:
    assert client.get("/documents").status_code == 422


# --- the unwired path, which is what every 501 in this engine looks like -----


def test_upload_reports_no_pipeline_is_registered(unwired: TestClient) -> None:
    response = _upload(unwired)

    assert response.status_code == 501
    assert "get_ingest_pipeline" in response.json()["detail"]


def test_list_reports_no_registry_is_registered(unwired: TestClient) -> None:
    response = unwired.get("/documents", params={"project_id": "support"})

    assert response.status_code == 501
    assert "get_registry" in response.json()["detail"]


def test_delete_reports_no_registry_is_registered(unwired: TestClient) -> None:
    response = unwired.delete("/documents/abc", params={"project_id": "support"})

    assert response.status_code == 501


# --- the wired round trip ----------------------------------------------------


def test_upload_chunks_the_document(client: TestClient) -> None:
    response = _upload(client)

    assert response.status_code == 201
    body = response.json()
    assert body["chunk_count"] >= 1
    assert body["size_bytes"] == len(MARKDOWN)
    assert body["external_id"] == "baggage.md"


def test_upload_is_indexed(client: TestClient) -> None:
    """`indexed` is earned by the vectors landing, not claimed on arrival."""
    assert _upload(client).json()["status"] == "indexed"


def test_reupload_of_identical_bytes_does_no_work(client: TestClient) -> None:
    first = _upload(client).json()
    second = _upload(client).json()

    assert second["status"] == "unchanged"
    assert second["doc_id"] == first["doc_id"]


def test_reupload_of_changed_bytes_replaces_the_document(client: TestClient) -> None:
    first = _upload(client).json()
    second = _upload(client, content=MARKDOWN + b"\nTwo bags on Flexible fares.\n")

    body = second.json()
    assert body["status"] == "indexed"
    assert body["doc_id"] == first["doc_id"], "same external_id, same document"
    assert body["content_hash"] != first["content_hash"]

    listed = client.get("/documents", params={"project_id": "support"}).json()
    assert len(listed) == 1, "replaced, not duplicated"


def test_uploaded_document_appears_in_the_list(client: TestClient) -> None:
    uploaded = _upload(client).json()

    listed = client.get("/documents", params={"project_id": "support"}).json()

    assert [record["doc_id"] for record in listed] == [uploaded["doc_id"]]


def test_documents_are_scoped_to_their_project(client: TestClient) -> None:
    _upload(client)

    other = client.get("/documents", params={"project_id": "other"}).json()

    assert other == []


def test_delete_removes_the_document(client: TestClient) -> None:
    doc_id = _upload(client).json()["doc_id"]

    deleted = client.delete(f"/documents/{doc_id}", params={"project_id": "support"})

    assert deleted.json() == {"doc_id": doc_id, "deleted": True}
    assert client.get("/documents", params={"project_id": "support"}).json() == []


def test_deleting_an_unknown_document_is_not_an_error(client: TestClient) -> None:
    response = client.delete("/documents/nope", params={"project_id": "support"})

    assert response.status_code == 200
    assert response.json()["deleted"] is False


def test_an_unreadable_type_is_415(client: TestClient) -> None:
    response = _upload(client, external_id="notes.docx", mimetype="application/msword")

    assert response.status_code == 415
    assert "application/msword" in response.json()["detail"]


def test_an_unreadable_type_leaves_no_record(client: TestClient) -> None:
    """Nothing was ingested, so nothing should show up as having been tried."""
    _upload(client, external_id="notes.docx", mimetype="application/msword")

    assert client.get("/documents", params={"project_id": "support"}).json() == []


def test_a_document_with_no_text_is_422(client: TestClient) -> None:
    """The scanned-PDF case: valid file, supported type, nothing to index."""
    response = _upload(client, content=b"   \n\n  \n")

    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]


def test_a_rejected_document_is_recorded_as_failed(client: TestClient) -> None:
    """Visible in the list, so the failure is discoverable without reading logs."""
    _upload(client, content=b"   \n\n  \n")

    listed = client.get("/documents", params={"project_id": "support"}).json()

    assert len(listed) == 1
    assert listed[0]["status"] == "failed"
    assert "OCR" in listed[0]["error"]
