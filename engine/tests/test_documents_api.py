"""The document contract, over HTTP.

What the route rejects on its own, then the round trip through a real pipeline.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

MARKDOWN = b"# Baggage\n\nOne cabin bag up to 8 kg, plus one personal item.\n"


def _upload(
    client: TestClient,
    content: bytes = MARKDOWN,
    external_id: str = "baggage.md",
    mimetype: str = "text/markdown",
    project_id: str = "support",
):
    return client.put(
        "/documents",
        data={"project_id": project_id, "external_id": external_id},
        files={"file": ("baggage.md", content, mimetype)},
    )


# --- validation the route owns ----------------------------------------------


def test_empty_upload_is_rejected_before_the_pipeline(client: TestClient) -> None:
    """Validation the engine owns, since it cannot assume its caller validated."""
    response = _upload(client, content=b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


# --- the wired round trip ----------------------------------------------------


def test_upload_chunks_the_document(client: TestClient) -> None:
    response = _upload(client)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "indexed", "earned by the vectors landing, not claimed"
    assert body["chunk_count"] >= 1
    assert body["size_bytes"] == len(MARKDOWN)
    assert body["external_id"] == "baggage.md"


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


# --- one project cannot reach another's documents ----------------------------


def test_another_project_cannot_delete_a_document(client: TestClient) -> None:
    """A `doc_id` is derived from the project and the file's name, so it can
    be worked out. Asked for by another project, it is a document that
    project does not have: nothing is deleted, not the chunks, not the file,
    not the record."""
    from chatbot_engine.rag.vector_store import count_chunks
    from chatbot_engine.settings import get_settings

    record = _upload(client).json()

    response = client.delete(
        f"/documents/{record['doc_id']}", params={"project_id": "other"}
    )

    assert response.json() == {"doc_id": record["doc_id"], "deleted": False}
    listed = client.get("/documents", params={"project_id": "support"}).json()
    assert [d["doc_id"] for d in listed] == [record["doc_id"]]
    assert count_chunks() == record["chunk_count"]
    assert (get_settings().blob_dir / record["doc_id"]).exists()


async def test_another_projects_delete_leaves_retrieval_alone(
    client: TestClient,
) -> None:
    from chatbot_engine.agent.retriever import retrieve
    from chatbot_engine.models.chat import AssistantConfig, ChatRequest

    record = _upload(client).json()
    client.delete(f"/documents/{record['doc_id']}", params={"project_id": "other"})

    hits = await retrieve(
        ChatRequest(
            project=AssistantConfig(
                project_id="support", name="S", system_prompt="s", retrieval="hybrid"
            ),
            message="cabin bag",
        )
    )

    assert [document.metadata["doc_id"] for document, _ in hits] == [record["doc_id"]]


def test_another_project_cannot_reindex_a_document(client: TestClient) -> None:
    record = _upload(client).json()

    response = client.post(
        f"/documents/{record['doc_id']}/reindex?project_id=other", json={}
    )

    assert response.status_code == 404
    listed = client.get("/documents", params={"project_id": "support"}).json()
    assert listed[0]["updated_at"] == record["updated_at"]


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


def test_text_that_is_not_utf8_is_422_with_the_reason(client: TestClient) -> None:
    """A clean refusal and a `failed` record that says why, not a 500."""
    response = _upload(client, content="Préavis".encode("latin-1"))

    assert response.status_code == 422
    assert "not UTF-8" in response.json()["detail"]
    listed = client.get("/documents", params={"project_id": "support"}).json()
    assert listed[0]["status"] == "failed"
    assert listed[0]["error"] == response.json()["detail"]


def test_a_damaged_pdf_is_422_with_the_readers_words(client: TestClient) -> None:
    response = client.put(
        "/documents",
        data={"project_id": "support", "external_id": "manual.pdf"},
        files={"file": ("manual.pdf", b"%PDF-1.4\nnot a pdf", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["detail"].startswith("'manual.pdf' could not be read: ")
    listed = client.get("/documents", params={"project_id": "support"}).json()
    assert listed[0]["status"] == "failed"


def test_a_rejected_document_is_recorded_as_failed(client: TestClient) -> None:
    """Visible in the list, so the failure is discoverable without reading logs."""
    _upload(client, content=b"   \n\n  \n")

    listed = client.get("/documents", params={"project_id": "support"}).json()

    assert len(listed) == 1
    assert listed[0]["status"] == "failed"
    assert "OCR" in listed[0]["error"]


# --- chunking settings, and rebuilding with new ones -------------------------

LONG_MARKDOWN = (
    b"# Baggage\n\n"
    + b"One cabin bag up to 8 kg, plus one personal item. " * 60
    + b"\n"
)


def test_an_upload_can_choose_how_it_is_cut(client: TestClient) -> None:
    response = client.put(
        "/documents",
        data={
            "project_id": "support",
            "external_id": "long.md",
            "chunking_strategy": "size",
            "chunk_size": "400",
            "chunk_overlap": "40",
        },
        files={"file": ("long.md", LONG_MARKDOWN, "text/markdown")},
    )

    assert response.status_code == 201
    body = response.json()
    assert (body["chunking_strategy"], body["chunk_size"], body["chunk_overlap"]) == (
        "size",
        400,
        40,
    )


def test_chunk_settings_out_of_range_are_422(client: TestClient) -> None:
    def upload(**fields: str):
        return client.put(
            "/documents",
            data={"project_id": "support", "external_id": "long.md", **fields},
            files={"file": ("long.md", LONG_MARKDOWN, "text/markdown")},
        )

    assert upload(chunk_size="50").status_code == 422
    assert upload(chunk_overlap="5000").status_code == 422
    too_wide = upload(chunk_size="300", chunk_overlap="300")
    assert too_wide.status_code == 422
    assert "chunk_overlap" in too_wide.json()["detail"]


def test_reindex_rebuilds_a_document_cut_differently(client: TestClient) -> None:
    first = client.put(
        "/documents",
        data={
            "project_id": "support",
            "external_id": "long.md",
            "chunk_size": "2000",
            "chunk_overlap": "0",
        },
        files={"file": ("long.md", LONG_MARKDOWN, "text/markdown")},
    ).json()

    rebuilt = client.post(
        f"/documents/{first['doc_id']}/reindex?project_id=support",
        json={"chunk_size": 300, "chunk_overlap": 30},
    )

    assert rebuilt.status_code == 200
    body = rebuilt.json()
    assert body["status"] == "indexed"
    assert (body["chunk_size"], body["chunk_overlap"]) == (300, 30)
    assert body["chunk_count"] > first["chunk_count"], "smaller chunks, more of them"
    listed = client.get("/documents?project_id=support").json()
    assert [(d["doc_id"], d["chunk_size"]) for d in listed] == [(first["doc_id"], 300)]


def test_reindex_fills_settings_it_is_not_given_from_the_defaults(
    client: TestClient,
) -> None:
    """As on upload, so one rule covers both routes."""
    from chatbot_engine.settings import get_settings

    first = client.put(
        "/documents",
        data={
            "project_id": "support",
            "external_id": "long.md",
            "chunking_strategy": "headings",
            "chunk_size": "700",
            "chunk_overlap": "70",
        },
        files={"file": ("long.md", LONG_MARKDOWN, "text/markdown")},
    ).json()

    body = client.post(
        f"/documents/{first['doc_id']}/reindex?project_id=support",
        json={"chunk_overlap": 0},
    ).json()

    defaults = get_settings()
    assert (body["chunking_strategy"], body["chunk_size"], body["chunk_overlap"]) == (
        defaults.chunk_strategy,
        defaults.chunk_size,
        0,
    )


def test_reindexing_an_unknown_document_is_404(client: TestClient) -> None:
    response = client.post("/documents/nope/reindex?project_id=support", json={})

    assert response.status_code == 404


# --- forgetting a project (X-9, INGEST-4) --------------------------------------


def test_a_purge_removes_every_document_of_the_project_and_refuses_new_ones(
    client: TestClient,
) -> None:
    from chatbot_engine.rag.vector_store import open_vector_store

    _upload(client, project_id="gone", external_id="a.md")
    _upload(client, project_id="gone", external_id="b.md")
    kept = _upload(client, project_id="kept", external_id="a.md").json()

    response = client.delete("/projects/gone")

    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "gone"
    assert body["knowledge"]["documents"] == 2
    assert client.get("/documents", params={"project_id": "gone"}).json() == []
    chunks = open_vector_store().get(where={"project_id": "gone"}, include=[])
    assert chunks["ids"] == []
    # Another project is untouched.
    assert [
        d["doc_id"]
        for d in client.get("/documents", params={"project_id": "kept"}).json()
    ] == [kept["doc_id"]]
    # A crawl still running for the deleted chatbot cannot put pages back.
    again = _upload(client, project_id="gone", external_id="c.md")
    assert again.status_code == 409
    assert "deleted a moment ago" in again.json()["detail"]


def test_a_purge_removes_chunks_no_record_names(client: TestClient) -> None:
    """Left by a write the volume cut short, or an engine from before the
    record was written first."""
    from langchain_core.documents import Document

    from chatbot_engine.rag.vector_store import open_vector_store

    open_vector_store().add_documents(
        [
            Document(
                page_content="orphan", metadata={"project_id": "gone", "doc_id": "x"}
            )
        ],
        ids=["x:0"],
    )

    body = client.delete("/projects/gone").json()

    assert body["knowledge"]["orphaned_chunks"] == 1
    assert (
        open_vector_store().get(where={"project_id": "gone"}, include=[])["ids"] == []
    )


def test_forgetting_a_session_answers_how_many_paused_turns_went(
    client: TestClient,
) -> None:
    response = client.delete("/projects/shop/sessions/s-1")
    assert response.status_code == 200
    assert response.json() == {
        "project_id": "shop",
        "session_id": "s-1",
        "paused_turns": 0,
    }
