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
    collection_name,
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


# --- through the API ---------------------------------------------------------


def test_uploading_writes_one_vector_per_chunk(client: TestClient) -> None:
    record = _upload(client, LONG).json()

    assert record["chunk_count"] > 1, "the fixture should span several chunks"
    assert count_chunks() == record["chunk_count"]


def test_a_shorter_second_version_leaves_no_orphans(client: TestClient) -> None:
    """The bug this guards: v1's tail surviving v2 and still being retrievable."""
    first = _upload(client, LONG).json()
    second = _upload(client, SHORT).json()

    assert second["chunk_count"] < first["chunk_count"]
    assert count_chunks() == second["chunk_count"]


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


# --- the store on its own ----------------------------------------------------


async def test_writing_no_chunks_is_a_delete(client: TestClient) -> None:
    doc_id = _upload(client, LONG).json()["doc_id"]
    store = ChromaChunkStore()

    await store.write(doc_id=doc_id, chunks=[])

    assert count_chunks() == 0


# --- one collection per embedding model ---------------------------------------


async def test_a_query_only_meets_vectors_from_its_own_model(
    client: TestClient,
) -> None:
    """Indexed under one model, invisible to a search under another."""
    client.put(
        "/documents",
        data={
            "project_id": "support",
            "external_id": "baggage.md",
            "embedding_model": "openai/text-embedding-3-small",
        },
        files={"file": ("baggage.md", LONG, "text/markdown")},
    )

    assert count_chunks(open_vector_store("openai/text-embedding-3-small")) > 0
    assert count_chunks(open_vector_store("openai/text-embedding-3-large")) == 0


async def test_deleting_reaches_a_document_indexed_under_another_model(
    client: TestClient,
) -> None:
    """The default store must still find, and remove, chunks another model wrote."""
    doc_id = client.put(
        "/documents",
        data={
            "project_id": "support",
            "external_id": "baggage.md",
            "embedding_model": "openai/text-embedding-3-large",
        },
        files={"file": ("baggage.md", LONG, "text/markdown")},
    ).json()["doc_id"]

    removed = await ChromaChunkStore().delete(doc_id=doc_id)

    assert removed > 0
    assert count_chunks(open_vector_store("openai/text-embedding-3-large")) == 0


def test_a_legacy_collection_is_adopted_by_the_default_model() -> None:
    """Before collections were keyed by model there was one, named after
    `ENGINE_CHROMA_COLLECTION`, and all of it was embedded with the default
    model. Opening the default store must find it rather than start empty."""
    import chromadb

    from chatbot_engine.rag.vector_store import _chroma_client, reset_vector_store
    from chatbot_engine.settings import get_settings

    settings = get_settings()
    legacy = chromadb.PersistentClient(path=str(settings.chroma_dir))
    legacy.get_or_create_collection(settings.chroma_collection).add(
        ids=["old:0"], embeddings=[[0.1] * 64], documents=["old chunk"]
    )
    reset_vector_store()

    assert count_chunks(open_vector_store()) == 1
    names = {c.name for c in _chroma_client().list_collections()}
    assert settings.chroma_collection not in names, "renamed, not copied"
    assert collection_name() in names


def test_a_legacy_collection_is_not_adopted_by_another_model() -> None:
    """Its vectors were made with the default model, so another model's
    collection must not claim them."""
    import chromadb

    from chatbot_engine.rag.vector_store import reset_vector_store
    from chatbot_engine.settings import get_settings

    settings = get_settings()
    legacy = chromadb.PersistentClient(path=str(settings.chroma_dir))
    legacy.get_or_create_collection(settings.chroma_collection).add(
        ids=["old:0"], embeddings=[[0.1] * 64], documents=["old chunk"]
    )
    reset_vector_store()

    assert count_chunks(open_vector_store("openai/text-embedding-3-large")) == 0


def test_an_empty_new_collection_does_not_block_adoption() -> None:
    """A query before any upload creates the model's collection, empty. That
    must not leave the legacy vectors stranded behind it."""
    import chromadb

    from chatbot_engine.rag.vector_store import reset_vector_store
    from chatbot_engine.settings import get_settings

    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    client.get_or_create_collection(settings.chroma_collection).add(
        ids=["old:0"], embeddings=[[0.1] * 64], documents=["old chunk"]
    )
    client.get_or_create_collection(collection_name())
    reset_vector_store()

    assert count_chunks(open_vector_store()) == 1


# --- embedded or server ---------------------------------------------------------


def test_a_chroma_url_selects_the_server_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedded Chroma belongs to one process; a URL is how replicas share one."""
    import chromadb

    from chatbot_engine.rag.vector_store import _chroma_client, reset_vector_store

    seen: dict = {}

    def http_client(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(chromadb, "HttpClient", http_client)
    monkeypatch.setenv("ENGINE_CHROMA_URL", "https://vectors.internal:8443")
    reset_vector_store()
    from chatbot_engine.api.dependencies import reset_dependency_cache

    reset_dependency_cache()

    _chroma_client()

    assert seen == {
        "host": "vectors.internal",
        "port": 8443,
        "ssl": True,
        "headers": None,
    }
    reset_vector_store()


def test_a_chroma_token_is_sent_in_the_configured_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that requires a credential gets one on every request, in the
    form its configuration expects: a bearer token by default, or a plain
    value in a named header."""
    import chromadb

    from chatbot_engine.api.dependencies import reset_dependency_cache
    from chatbot_engine.rag.vector_store import _chroma_client, reset_vector_store

    seen: dict = {}
    monkeypatch.setattr(
        chromadb, "HttpClient", lambda **kw: seen.update(kw) or object()
    )
    monkeypatch.setenv("ENGINE_CHROMA_URL", "http://vectors:8000")
    monkeypatch.setenv("ENGINE_CHROMA_TOKEN", "s3cret")

    reset_dependency_cache()
    reset_vector_store()
    _chroma_client()
    assert seen["headers"] == {"Authorization": "Bearer s3cret"}

    monkeypatch.setenv("ENGINE_CHROMA_TOKEN_HEADER", "X-Chroma-Token")
    reset_dependency_cache()
    reset_vector_store()
    _chroma_client()
    assert seen["headers"] == {"X-Chroma-Token": "s3cret"}
    reset_vector_store()
