"""The document contract, over HTTP.

What the route rejects on its own, then the round trip through a real pipeline.
"""

from __future__ import annotations

import pytest
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
    from chatbot_engine.rag.vector_store import stored_chunks

    _upload(client, project_id="gone", external_id="a.md")
    _upload(client, project_id="gone", external_id="b.md")
    kept = _upload(client, project_id="kept", external_id="a.md").json()

    response = client.delete("/projects/gone")

    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "gone"
    assert body["knowledge"]["documents"] == 2
    assert client.get("/documents", params={"project_id": "gone"}).json() == []
    chunks = stored_chunks(where={"project_id": "gone"})
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

    from chatbot_engine.rag.vector_store import open_vector_store, stored_chunks

    open_vector_store(None, "gone").add_documents(
        [
            Document(
                page_content="orphan", metadata={"project_id": "gone", "doc_id": "x"}
            )
        ],
        ids=["x:0"],
    )

    body = client.delete("/projects/gone").json()

    assert body["knowledge"]["orphaned_chunks"] == 1
    assert stored_chunks(where={"project_id": "gone"})["ids"] == []


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


# --- a delete that stops part-way -----------------------------------------------


def test_a_delete_that_stops_part_way_leaves_a_record_that_says_so(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunks are gone, then removing the stored file fails (an object
    store that does not answer). The record must not go on saying
    `indexed`: an upload of the same file would answer `unchanged` and the
    document would stay unsearchable. It says `failed`, an upload indexes
    the document again, and a second delete finishes the first."""
    from chatbot_engine.documents.blobs import DocumentBlobs
    from chatbot_engine.rag.vector_store import count_chunks, find_vector_store

    def held() -> int:
        store = find_vector_store(None, "support")
        return count_chunks(store) if store is not None else 0

    record = _upload(client).json()
    real_delete = DocumentBlobs.delete

    async def unanswered(self, *, doc_id):
        raise OSError("the object store did not answer")

    monkeypatch.setattr(DocumentBlobs, "delete", unanswered)
    with pytest.raises(OSError):
        client.delete(
            f"/documents/{record['doc_id']}", params={"project_id": "support"}
        )
    monkeypatch.setattr(DocumentBlobs, "delete", real_delete)

    (listed,) = client.get("/documents", params={"project_id": "support"}).json()
    assert listed["status"] == "failed" and "did not finish" in listed["error"]
    assert held() == 0

    again = _upload(client).json()
    assert again["status"] == "indexed" and held() == again["chunk_count"] > 0

    deleted = client.delete(
        f"/documents/{record['doc_id']}", params={"project_id": "support"}
    )
    assert deleted.json()["deleted"] is True
    assert client.get("/documents", params={"project_id": "support"}).json() == []


# --- what goes wrong on the way, answered as such ------------------------------


def test_an_unexpected_failure_still_carries_the_request_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one answer a caller most needs to quote: the catch-all 500 runs
    outside the request id's middleware, and must still carry the id."""
    from chatbot_engine.documents.blobs import DocumentBlobs

    record = _upload(client).json()

    async def unanswered(self, *, doc_id):
        raise OSError("the object store did not answer")

    monkeypatch.setattr(DocumentBlobs, "delete", unanswered)
    quiet = TestClient(client.app, raise_server_exceptions=False)

    sent = quiet.delete(
        f"/documents/{record['doc_id']}",
        params={"project_id": "support"},
        headers={"X-Request-Id": "a" * 32},
    )
    made = quiet.delete(
        f"/documents/{record['doc_id']}", params={"project_id": "support"}
    )

    assert (sent.status_code, sent.headers.get("x-request-id")) == (500, "a" * 32)
    assert made.status_code == 500 and len(made.headers.get("x-request-id", "")) == 32


def test_a_lookup_error_from_inside_a_store_is_a_500_not_a_missing_document(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the engine's own word for a missing document is a 404: a
    `KeyError` from an object store's client is a failure, not an answer."""
    from chatbot_engine.documents.blobs import DocumentBlobs

    record = _upload(client).json()

    async def no_body(self, *, doc_id):
        raise KeyError("Body")  # an S3 answer without its body

    monkeypatch.setattr(DocumentBlobs, "read", no_body)
    quiet = TestClient(client.app, raise_server_exceptions=False)
    answer = quiet.post(
        f"/documents/{record['doc_id']}/reindex", params={"project_id": "support"}
    )

    assert answer.status_code == 500
    missing = client.post("/documents/nope/reindex", params={"project_id": "support"})
    assert missing.status_code == 404


def test_a_query_parameter_a_route_does_not_take_is_refused(client: TestClient) -> None:
    """Dropped quietly, `?chunk_size=200` re-chunked at the defaults with a
    200; every documents and projects route names the one it does not take."""
    record = _upload(client).json()
    doc = record["doc_id"]

    answers = {
        "reindex": client.post(
            f"/documents/{doc}/reindex?project_id=support&chunk_size=200&chunk_overlap=0"
        ),
        "list": client.get("/documents?project_id=support&limit=5"),
        "delete": client.delete(f"/documents/{doc}?project_id=support&force=1"),
        "upload": client.put(
            "/documents?chunk_size=200",
            data={"project_id": "support", "external_id": "other.md"},
            files={"file": ("other.md", MARKDOWN, "text/markdown")},
        ),
        "purge": client.delete("/projects/support?keep_documents=1"),
        "forget": client.delete("/projects/support/sessions/s-1?all=1"),
    }

    assert {name: a.status_code for name, a in answers.items()} == dict.fromkeys(
        answers, 422
    )
    assert [e["loc"] for e in answers["reindex"].json()["detail"]] == [
        ["query", "chunk_overlap"],
        ["query", "chunk_size"],
    ]
    listed = client.get("/documents", params={"project_id": "support"}).json()
    assert [d["chunk_size"] for d in listed] == [record["chunk_size"]], "untouched"


# --- a purge ------------------------------------------------------------------


def test_a_purge_rebuilds_no_collection_it_is_about_to_drop(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chatbot_engine.rag import vector_store as vs

    monkeypatch.setattr(vs, "REBUILD_AFTER", 1)
    for n in range(6):
        body = MARKDOWN + b"More about bags. " * 40 + str(n).encode()
        assert _upload(client, body, external_id=f"doc{n}.md").status_code == 201
    rebuilds = 0
    real = vs._rebuild

    async def counting(project_id: str, name: str) -> None:
        nonlocal rebuilds
        rebuilds += 1
        await real(project_id, name)

    monkeypatch.setattr(vs, "_rebuild", counting)
    body = client.delete("/projects/support").json()

    assert body["knowledge"]["documents"] == 6
    assert rebuilds == 0
    assert vs.find_vector_store(None, "support") is None


def test_a_purge_removes_stored_originals_no_record_names(client: TestClient) -> None:
    """Left by engines before 0.1.28, which stored a first upload's original
    before its record and could stop in between (INGEST-16). Which project
    such a file was for cannot be told, so a purge takes every one; an
    original a record names stays, and a file that is not an original is
    not the engine's to touch."""
    from chatbot_engine.settings import get_settings

    blobs = get_settings().blob_dir
    kept = _upload(client, project_id="kept").json()
    orphan = blobs / ("ab" * 16)
    orphan.write_bytes(b"a file a pre-0.1.28 engine stored and never recorded")
    foreign = blobs / "README.txt"
    foreign.write_text("not an original")

    assert client.delete("/projects/gone").status_code == 200

    assert not orphan.exists()
    assert foreign.exists()
    assert (blobs / kept["doc_id"]).exists()
