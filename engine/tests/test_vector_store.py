"""The chunk lifecycle in Chroma.

A real store in a temporary directory (see conftest), with fake vectors. What
matters here is bookkeeping, not similarity: chunks belong to a document, and a
re-upload or a delete must take all of them and nothing else.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.rag.vector_store import (
    ChromaChunkStore,
    EmptyVectorStoreError,
    count_chunks,
    load_vector_store,
    open_vector_store,
)

LONG = ("Cabin baggage is one bag up to eight kilograms. " * 40).encode()
SHORT = b"# Baggage\n\nOne bag.\n"


def _upload(client: TestClient, content: bytes, external_id: str = "baggage.md"):
    return client.put(
        "/documents",
        data={"project_id": "support", "external_id": external_id},
        files={"file": (external_id, content, "text/markdown")},
    )


def _stored_ids() -> list[str]:
    return open_vector_store().get(include=[])["ids"]


# --- through the API ---------------------------------------------------------


def test_uploading_writes_one_vector_per_chunk(client: TestClient) -> None:
    record = _upload(client, LONG).json()

    assert record["chunk_count"] > 1, "the fixture should span several chunks"
    assert count_chunks() == record["chunk_count"]


def test_chunk_ids_are_derived_from_the_document(client: TestClient) -> None:
    doc_id = _upload(client, LONG).json()["doc_id"]

    assert all(chunk_id.startswith(f"{doc_id}:") for chunk_id in _stored_ids())


def test_a_shorter_second_version_leaves_no_orphans(client: TestClient) -> None:
    """The bug this guards: v1's tail surviving v2 and still being retrievable."""
    first = _upload(client, LONG).json()
    second = _upload(client, SHORT).json()

    assert second["chunk_count"] < first["chunk_count"]
    assert count_chunks() == second["chunk_count"]


def test_re_uploading_identical_bytes_does_not_duplicate(client: TestClient) -> None:
    _upload(client, LONG)
    before = count_chunks()

    assert _upload(client, LONG).json()["status"] == "unchanged"
    assert count_chunks() == before


def test_deleting_a_document_takes_its_chunks(client: TestClient) -> None:
    doc_id = _upload(client, LONG).json()["doc_id"]

    client.delete(f"/documents/{doc_id}", params={"project_id": "support"})

    assert count_chunks() == 0


def test_deleting_one_document_leaves_the_others(client: TestClient) -> None:
    doc_id = _upload(client, LONG, "baggage.md").json()["doc_id"]
    kept = _upload(client, LONG, "refunds.md").json()

    client.delete(f"/documents/{doc_id}", params={"project_id": "support"})

    assert count_chunks() == kept["chunk_count"]


def test_a_rejected_document_writes_no_vectors(client: TestClient) -> None:
    _upload(client, b"   \n\n  \n")

    assert count_chunks() == 0


def test_chunks_carry_the_project_so_a_query_can_be_scoped(
    client: TestClient,
) -> None:
    """A missing filter would let one project answer another's questions."""
    _upload(client, LONG)

    stored = open_vector_store().get(include=["metadatas"])

    assert {meta["project_id"] for meta in stored["metadatas"]} == {"support"}
    assert all("source" in meta for meta in stored["metadatas"])


# --- the empty-store guard ---------------------------------------------------


def test_loading_an_empty_store_fails_loudly(client: TestClient) -> None:
    """Better than retrieving zero chunks and letting the model invent an answer."""
    with pytest.raises(EmptyVectorStoreError, match="make seed"):
        load_vector_store()


def test_loading_a_populated_store_succeeds(client: TestClient) -> None:
    _upload(client, LONG)

    assert load_vector_store() is open_vector_store()


# --- the store on its own ----------------------------------------------------


async def test_writing_no_chunks_is_a_delete(client: TestClient) -> None:
    doc_id = _upload(client, LONG).json()["doc_id"]
    store = ChromaChunkStore()

    await store.write(doc_id=doc_id, chunks=[])

    assert count_chunks() == 0


async def test_deleting_an_unknown_document_reports_zero(client: TestClient) -> None:
    assert await ChromaChunkStore().delete(doc_id="nope") == 0
