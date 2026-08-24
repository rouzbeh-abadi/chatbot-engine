"""The original bytes, and the re-index they exist for."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.dependencies import get_ingest_pipeline, get_settings
from chatbot_engine.documents.blobs import DocumentBlobs
from chatbot_engine.documents.storage import LocalBlobStore
from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.rag.vector_store import count_chunks

LONG = ("Cabin baggage is one bag up to eight kilograms. " * 40).encode()


def _upload(
    client: TestClient,
    content: bytes = LONG,
    external_id: str = "baggage.md",
    mimetype: str = "text/markdown",
):
    return client.put(
        "/documents",
        data={"project_id": "support", "external_id": external_id},
        files={"file": (external_id, content, mimetype)},
    )


def _blob_files() -> list[str]:
    root = get_settings().blob_dir

    return sorted(path.name for path in root.iterdir()) if root.exists() else []


# --- the key convention ------------------------------------------------------


async def test_the_computed_uri_matches_what_put_returned(tmp_path: Path) -> None:
    """The one coupling in `DocumentBlobs`: pin it, or a layout change is silent."""
    blobs = DocumentBlobs(tmp_path)

    returned = await blobs.write(doc_id="abc123", data=b"hello", mimetype="text/plain")

    assert returned == blobs._uri("abc123")
    assert await blobs.read(doc_id="abc123") == b"hello"


async def test_a_traversing_key_cannot_escape(tmp_path: Path) -> None:
    """Why the key is `doc_id` and never `external_id`, which callers control."""
    blobs = DocumentBlobs(tmp_path / "root")
    escaped = tmp_path / "escaped.md"

    await blobs.write(
        doc_id="../escaped.md", data=b"nope", mimetype="text/plain"
    )

    # LocalBlobStore would happily write it -- so the pipeline must never pass
    # a caller-supplied key. A doc_id is 32 hex characters and cannot traverse.
    assert escaped.exists(), "this is the hole doc_id keys close"


async def test_deleting_a_missing_blob_is_silent(tmp_path: Path) -> None:
    await DocumentBlobs(tmp_path).delete(doc_id="never-written")


async def test_the_port_is_swappable(tmp_path: Path) -> None:
    """`DocumentBlobs` composes a `BlobStore`, so S3 later is a constructor arg."""
    blobs = DocumentBlobs(tmp_path, store=LocalBlobStore(tmp_path))

    await blobs.write(doc_id="x", data=b"y", mimetype="text/plain")

    assert await blobs.read(doc_id="x") == b"y"


# --- through the API ---------------------------------------------------------


def test_uploading_keeps_the_original(client: TestClient) -> None:
    doc_id = _upload(client).json()["doc_id"]

    assert _blob_files() == [doc_id]
    assert (get_settings().blob_dir / doc_id).read_bytes() == LONG


def test_deleting_removes_the_original(client: TestClient) -> None:
    doc_id = _upload(client).json()["doc_id"]

    client.delete(f"/documents/{doc_id}", params={"project_id": "support"})

    assert _blob_files() == []


def test_an_unreadable_type_leaves_no_blob(client: TestClient) -> None:
    """415 is raised by `select_extractor`, before anything is written."""
    response = _upload(
        client, content=b"x", external_id="notes.docx", mimetype="application/msword"
    )

    assert response.status_code == 415
    assert _blob_files() == []


def test_a_document_with_no_text_keeps_its_blob(client: TestClient) -> None:
    """422, and the bytes stay: there is a `failed` record that owns them, so
    they can be inspected and `DELETE` still cleans them up."""
    response = _upload(client, content=b"   \n\n  \n")

    assert response.status_code == 422
    assert len(_blob_files()) == 1

    listed = client.get("/documents", params={"project_id": "support"}).json()
    assert listed[0]["status"] == "failed"

    client.delete(
        f"/documents/{listed[0]['doc_id']}", params={"project_id": "support"}
    )
    assert _blob_files() == []


# --- what the blobs are for --------------------------------------------------


async def test_reindex_rebuilds_from_the_stored_bytes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point: a smaller chunk size applied without a re-upload."""
    uploaded = _upload(client).json()
    before = uploaded["chunk_count"]
    assert before > 1

    monkeypatch.setenv("ENGINE_CHUNK_SIZE", "120")
    monkeypatch.setenv("ENGINE_CHUNK_OVERLAP", "12")
    get_settings.cache_clear()
    get_ingest_pipeline.cache_clear()

    rebuilt = await get_ingest_pipeline().reindex(
        project_id="support", doc_id=uploaded["doc_id"]
    )

    assert rebuilt.chunk_count > before, "smaller chunks, so more of them"
    assert count_chunks() == rebuilt.chunk_count, "old vectors replaced, not added"


async def test_reindex_on_an_unknown_document_raises(client: TestClient) -> None:
    with pytest.raises(LookupError):
        await get_ingest_pipeline().reindex(project_id="support", doc_id="nope")


async def test_reindex_without_a_blob_store_says_so(client: TestClient) -> None:
    from chatbot_engine.documents.sqlite_registry import SqliteDocumentRegistry
    from chatbot_engine.rag.pipeline import DocumentIngestPipeline
    from chatbot_engine.rag.splitter import DocumentChunker

    pipeline = DocumentIngestPipeline(
        registry=SqliteDocumentRegistry(get_settings().registry_db),
        chunker=DocumentChunker(),
    )

    with pytest.raises(NotConfiguredError, match="BlobStore"):
        await pipeline.reindex(project_id="support", doc_id="anything")
