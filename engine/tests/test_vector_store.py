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

    await store.write(doc_id=doc_id, project_id="support", chunks=[])

    assert count_chunks() == 0


async def test_a_delete_reaches_only_its_own_projects_chunks(
    client: TestClient,
) -> None:
    """A `doc_id` can be worked out from a project and a file name, so the
    project is part of every delete: another project's id removes nothing."""
    record = _upload(client, LONG).json()

    removed = await ChromaChunkStore().delete(
        doc_id=record["doc_id"], project_id="other"
    )

    assert removed == 0
    assert count_chunks() == record["chunk_count"]


async def test_a_failed_embedding_leaves_the_version_already_indexed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this guards: a provider outage during a re-index used to leave
    the document with no chunks at all, since the old ones went first."""
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding

    first = _upload(client, LONG).json()
    before = open_vector_store().get(include=["documents"])

    def outage(self, texts):
        raise RuntimeError("provider down")

    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", outage)
    with pytest.raises(RuntimeError, match="provider down"):
        await ChromaChunkStore().write(
            doc_id=first["doc_id"],
            project_id="support",
            chunks=[Document(page_content="new", metadata={"project_id": "support"})],
        )

    after = open_vector_store().get(include=["documents"])
    assert sorted(after["ids"]) == sorted(before["ids"])
    assert sorted(after["documents"]) == sorted(before["documents"])


async def test_a_new_version_replaces_the_old_one_in_place(
    client: TestClient,
) -> None:
    """Written under ids of its own, then the old version removed: what is
    left is exactly the new version."""
    from langchain_core.documents import Document

    doc_id = _upload(client, LONG).json()["doc_id"]
    chunks = [
        Document(
            page_content=f"part {n}",
            metadata={"doc_id": doc_id, "project_id": "support"},
        )
        for n in range(2)
    ]

    await ChromaChunkStore().write(doc_id=doc_id, project_id="support", chunks=chunks)

    stored = open_vector_store().get(include=["documents"])
    assert len(stored["ids"]) == 2
    assert all(chunk_id.startswith(f"{doc_id}:") for chunk_id in stored["ids"])
    assert sorted(stored["documents"]) == ["part 0", "part 1"]


def _set(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    from chatbot_engine.settings import get_settings

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


async def test_a_failure_part_way_leaves_the_old_version_whole(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written a slice at a time, a new version can fail after some of it
    landed. What landed goes again, and the old version answers as before."""
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding

    first = _upload(client, LONG).json()
    before = open_vector_store().get(include=["documents"])
    _set(monkeypatch, ENGINE_INDEX_BATCH_SIZE="1")
    calls = []
    original = DeterministicFakeEmbedding.embed_documents

    def second_call_fails(self, texts):
        calls.append(len(texts))
        if len(calls) == 2:
            raise RuntimeError("provider down")
        return original(self, texts)

    monkeypatch.setattr(
        DeterministicFakeEmbedding, "embed_documents", second_call_fails
    )
    with pytest.raises(RuntimeError, match="provider down"):
        await ChromaChunkStore().write(
            doc_id=first["doc_id"],
            project_id="support",
            chunks=[
                Document(page_content=f"new {n}", metadata={"project_id": "support"})
                for n in range(3)
            ],
        )

    after = open_vector_store().get(include=["documents"])
    assert calls == [1, 1], "one slice landed before the failure"
    assert sorted(after["ids"]) == sorted(before["ids"])
    assert sorted(after["documents"]) == sorted(before["documents"])


async def test_two_writes_of_one_document_at_once_leave_one_version_whole(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each version goes in under ids of its own and removes the rest, so two
    at once must not each remove the other's."""
    import asyncio

    from langchain_core.documents import Document

    doc_id = _upload(client, LONG).json()["doc_id"]
    _set(monkeypatch, ENGINE_INDEX_BATCH_SIZE="1")

    def version(name: str, count: int) -> list[Document]:
        return [
            Document(
                page_content=f"{name} {n}",
                metadata={"doc_id": doc_id, "project_id": "support"},
            )
            for n in range(count)
        ]

    store = ChromaChunkStore()
    await asyncio.gather(
        store.write(doc_id=doc_id, project_id="support", chunks=version("a", 3)),
        store.write(doc_id=doc_id, project_id="support", chunks=version("b", 2)),
    )

    left = sorted(open_vector_store().get(include=["documents"])["documents"])
    assert left in (["a 0", "a 1", "a 2"], ["b 0", "b 1"])


def test_a_document_of_too_many_chunks_is_refused_before_it_is_embedded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    _set(monkeypatch, ENGINE_INDEX_MAX_CHUNKS="2")
    embedded = []
    original = DeterministicFakeEmbedding.embed_documents

    def counting(self, texts):
        embedded.extend(texts)
        return original(self, texts)

    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", counting)
    response = _upload(client, LONG)

    assert response.status_code == 422
    assert (
        "more than the 2 one document may give the index" in response.json()["detail"]
    )
    assert embedded == []
    assert count_chunks() == 0


def test_an_upload_below_the_free_space_floor_is_refused_and_writes_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A full volume can hang embedded Chroma for every tenant, so indexing
    stops well before it: a 503, and nothing written, the original included."""
    _set(monkeypatch, ENGINE_MIN_FREE_MB=str(10**12))
    response = _upload(client, LONG)

    assert response.status_code == 503
    assert "ENGINE_MIN_FREE_MB" in response.json()["detail"]
    assert count_chunks() == 0
    assert not any((tmp_path / "blobs").rglob("*.*"))
    assert client.get("/documents", params={"project_id": "support"}).json() == []


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

    removed = await ChromaChunkStore().delete(doc_id=doc_id, project_id="support")

    assert removed > 0
    assert count_chunks(open_vector_store("openai/text-embedding-3-large")) == 0


async def test_identical_bytes_under_another_model_move_to_its_collection(
    client: TestClient,
) -> None:
    """Answered `unchanged`, the document would stay where the new model's
    queries never look."""

    def upload(model: str):
        return client.put(
            "/documents",
            data={
                "project_id": "support",
                "external_id": "baggage.md",
                "embedding_model": model,
            },
            files={"file": ("baggage.md", LONG, "text/markdown")},
        ).json()

    small = upload("openai/text-embedding-3-small")
    large = upload("openai/text-embedding-3-large")

    assert large["status"] == "indexed"
    assert (small["embedding_model"], large["embedding_model"]) == (
        "openai/text-embedding-3-small",
        "openai/text-embedding-3-large",
    )
    assert count_chunks(open_vector_store("openai/text-embedding-3-small")) == 0
    assert (
        count_chunks(open_vector_store("openai/text-embedding-3-large"))
        == (large["chunk_count"])
    )
    assert upload("openai/text-embedding-3-large")["status"] == "unchanged"


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
