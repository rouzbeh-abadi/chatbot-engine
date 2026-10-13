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
    find_vector_store,
    load_vector_store,
    open_vector_store,
    stored_chunks,
)

LONG = ("Cabin baggage is one bag up to eight kilograms. " * 40).encode()
SHORT = b"# Baggage\n\nOne bag.\n"


def _held(model: str, project_id: str = "support") -> int:
    """How many chunks a project holds under one embedding model."""
    store = find_vector_store(model, project_id)
    return count_chunks(store) if store is not None else 0


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

    stored = stored_chunks()

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
    before = stored_chunks()

    def outage(self, texts):
        raise RuntimeError("provider down")

    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", outage)
    with pytest.raises(RuntimeError, match="provider down"):
        await ChromaChunkStore().write(
            doc_id=first["doc_id"],
            project_id="support",
            chunks=[Document(page_content="new", metadata={"project_id": "support"})],
        )

    after = stored_chunks()
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

    stored = stored_chunks()
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
    before = stored_chunks()
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

    after = stored_chunks()
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

    left = sorted(stored_chunks()["documents"])
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

    assert _held("openai/text-embedding-3-small") > 0
    assert _held("openai/text-embedding-3-large") == 0


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
    assert _held("openai/text-embedding-3-large") == 0


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
    assert _held("openai/text-embedding-3-small") == 0
    assert _held("openai/text-embedding-3-large") == (large["chunk_count"])
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

    assert _held("openai/text-embedding-3-large") == 0


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


# --- one collection per project (0.1.30, DISK-1) ---------------------------------


def _segments_on_disk() -> set[str]:
    """The segment folders under the Chroma directory."""
    from chatbot_engine.rag.vector_store import _SEGMENT
    from chatbot_engine.settings import get_settings

    root = get_settings().chroma_dir
    return {p.name for p in root.iterdir() if p.is_dir() and _SEGMENT.match(p.name)}


def _segments_named() -> set[str]:
    """The segments Chroma's own catalogue names."""
    import sqlite3

    from chatbot_engine.settings import get_settings

    with sqlite3.connect(get_settings().chroma_dir / "chroma.sqlite3") as db:
        return {row[0] for row in db.execute("SELECT id FROM segments")}


def _docs(project_id: str, doc_id: str, n: int) -> list:
    from langchain_core.documents import Document

    return [
        Document(
            page_content=f"{doc_id} part {i}",
            metadata={"doc_id": doc_id, "project_id": project_id, "source": doc_id},
        )
        for i in range(n)
    ]


async def test_each_project_has_a_collection_of_its_own(client: TestClient) -> None:
    """A search reads only its project's vectors, and a project's knowledge
    can be dropped whole."""
    await ChromaChunkStore().write(
        doc_id="a", project_id="alpha", chunks=_docs("alpha", "a", 3)
    )
    await ChromaChunkStore().write(
        doc_id="b", project_id="beta", chunks=_docs("beta", "b", 2)
    )

    alpha, beta = find_vector_store(None, "alpha"), find_vector_store(None, "beta")
    assert alpha is not None and beta is not None
    assert alpha._collection.name != beta._collection.name
    assert (count_chunks(alpha), count_chunks(beta)) == (3, 2)
    assert find_vector_store(None, "gamma") is None, "a search makes no collection"


async def test_a_project_whose_last_document_goes_gives_its_files_back(
    client: TestClient,
) -> None:
    """Chroma keeps a deleted vector's room in its collection, and a dropped
    collection's files: the collection is dropped, and its files removed."""
    await ChromaChunkStore().write(
        doc_id="k", project_id="keep", chunks=_docs("keep", "k", 1200)
    )
    await ChromaChunkStore().write(
        doc_id="g", project_id="gone", chunks=_docs("gone", "g", 1200)
    )
    assert len(_segments_on_disk()) >= 2, "precondition: both indexes on disk"

    await ChromaChunkStore().delete(doc_id="g", project_id="gone")

    assert find_vector_store(None, "gone") is None
    assert _segments_on_disk() <= _segments_named(), "a dropped collection left files"
    assert count_chunks(find_vector_store(None, "keep")) == 1200


async def test_a_collection_is_rebuilt_once_its_deleted_chunks_outnumber_its_live_ones(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chatbot_engine.rag import vector_store as vector_store_module

    monkeypatch.setattr(vector_store_module, "REBUILD_AFTER", 5)
    store = ChromaChunkStore()
    await store.write(doc_id="kept", project_id="p", chunks=_docs("p", "kept", 4))
    await store.write(doc_id="churn", project_id="p", chunks=_docs("p", "churn", 3))
    before = find_vector_store(None, "p")
    assert before is not None
    old_id = before._collection.id

    # Replaced twice: six chunks deleted, four live; then the churned one goes.
    await store.write(doc_id="churn", project_id="p", chunks=_docs("p", "churn", 3))
    await store.write(doc_id="churn", project_id="p", chunks=_docs("p", "churn", 3))
    await store.delete(doc_id="churn", project_id="p")

    after = find_vector_store(None, "p")
    assert after is not None
    assert after._collection.id != old_id, "rebuilt into a new collection"
    assert after._collection.name == before._collection.name
    assert sorted(stored_chunks(where={"project_id": "p"})["documents"]) == [
        f"kept part {i}" for i in range(4)
    ]
    assert not (after._collection.metadata or {}).get("deleted_chunks")
    assert _segments_on_disk() <= _segments_named()


async def test_a_rebuild_cut_short_is_finished_when_the_collection_is_next_used(
    client: TestClient,
) -> None:
    """The engine stopped after the old collection was dropped and before the
    new one took its name: it takes it the next time the project is used."""
    from chatbot_engine.rag.vector_store import _chroma_client, reset_vector_store

    await ChromaChunkStore().write(
        doc_id="d", project_id="p", chunks=_docs("p", "d", 2)
    )
    name = collection_name(None, "p")
    _chroma_client().get_collection(name).modify(name=f"{name}--new")
    reset_vector_store()

    store = find_vector_store(None, "p")
    assert store is not None and store._collection.name == name
    assert count_chunks(store) == 2


async def test_a_purge_drops_every_collection_of_the_project(
    client: TestClient,
) -> None:
    await ChromaChunkStore().write(
        doc_id="d", project_id="p", chunks=_docs("p", "d", 2)
    )
    await ChromaChunkStore("openai/text-embedding-3-large").write(
        doc_id="e", project_id="p", chunks=_docs("p", "e", 2)
    )

    assert await ChromaChunkStore().delete_project(project_id="p") == 4
    assert find_vector_store(None, "p") is None
    assert find_vector_store("openai/text-embedding-3-large", "p") is None
    assert _segments_on_disk() <= _segments_named()


def _shared_with(chunks: dict[str, int]) -> None:
    """Chunks in the collection every project shared until 0.1.30."""
    from chatbot_engine.rag.vector_store import reset_vector_store

    shared = open_vector_store()
    for project_id, n in chunks.items():
        shared._collection.add(
            ids=[f"{project_id}:{i}" for i in range(n)],
            embeddings=[[0.1] * 64] * n,
            documents=[f"{project_id} {i}" for i in range(n)],
            metadatas=[{"project_id": project_id, "doc_id": "d"}] * n,
        )
    reset_vector_store()  # the engine restarts on 0.1.30


async def test_a_projects_chunks_move_into_its_own_collection_before_it_is_used() -> (
    None
):
    from chatbot_engine.rag.vector_store import ensure_moved

    _shared_with({"alpha": 3, "beta": 2})

    await ensure_moved("alpha")

    alpha = find_vector_store(None, "alpha")
    assert alpha is not None and count_chunks(alpha) == 3
    assert count_chunks(open_vector_store()) == 2, "the other project waits its turn"


async def test_every_project_is_moved_in_the_background_and_the_shared_collection_dropped() -> (
    None
):
    from chatbot_engine.rag.vector_store import _chroma_client, move_everything

    _shared_with({"alpha": 3, "beta": 2})
    shared = open_vector_store()
    # A chunk with no project, which no search could ever have reached.
    shared._collection.add(ids=["stray"], embeddings=[[0.1] * 64], documents=["stray"])

    await move_everything()

    assert count_chunks(find_vector_store(None, "alpha")) == 3
    assert count_chunks(find_vector_store(None, "beta")) == 2
    names = {c.name for c in _chroma_client().list_collections()}
    assert collection_name() not in names, "the emptied shared collection is dropped"
    assert _segments_on_disk() <= _segments_named()


async def test_a_move_the_volume_has_no_room_for_leaves_the_project_searchable_where_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chatbot_engine.errors import StorageFullError
    from chatbot_engine.rag import vector_store as vector_store_module
    from chatbot_engine.rag.vector_store import store_to_search, try_to_move

    _shared_with({"alpha": 3})

    def full(*_: object) -> None:
        raise StorageFullError("below the free-space floor")

    monkeypatch.setattr(vector_store_module, "ensure_room", full)

    assert await try_to_move("alpha") is False
    store = store_to_search(None, "alpha", moved=False)
    assert store is not None
    found = store.similarity_search("alpha", k=5, filter={"project_id": "alpha"})
    assert len(found) == 3, "every chunk still where the search reads"
    assert count_chunks(open_vector_store()) == 3, "nothing deleted from the shared one"


async def test_a_rebuild_that_cannot_finish_leaves_the_collection_and_no_part_copy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chatbot_engine.errors import StorageFullError
    from chatbot_engine.rag import vector_store as vector_store_module
    from chatbot_engine.rag.vector_store import _chroma_client

    monkeypatch.setattr(vector_store_module, "REBUILD_AFTER", 1)
    store = ChromaChunkStore()
    await store.write(doc_id="kept", project_id="p", chunks=_docs("p", "kept", 2))
    await store.write(doc_id="churn", project_id="p", chunks=_docs("p", "churn", 3))

    def full(*_: object) -> None:
        raise StorageFullError("below the free-space floor")

    monkeypatch.setattr(vector_store_module, "ensure_room", full)
    await store.delete(doc_id="churn", project_id="p")  # the rebuild fails, logged

    name = collection_name(None, "p")
    names = {c.name for c in _chroma_client().list_collections()}
    assert name in names and f"{name}--new" not in names
    assert sorted(stored_chunks(where={"project_id": "p"})["documents"]) == [
        "kept part 0",
        "kept part 1",
    ]


async def test_a_dropped_collection_takes_a_leftover_copy_with_it(
    client: TestClient,
) -> None:
    """Kept, a copy left by a rebuild the engine stopped mid-way would be
    taken for one to finish, and bring back deleted chunks."""
    from chatbot_engine.rag.vector_store import _chroma_client

    await ChromaChunkStore().write(
        doc_id="d", project_id="p", chunks=_docs("p", "d", 2)
    )
    name = collection_name(None, "p")
    _chroma_client().create_collection(f"{name}--new", embedding_function=None).add(
        ids=["stale"], embeddings=[[0.1] * 64], documents=["deleted long ago"]
    )

    await ChromaChunkStore().delete(doc_id="d", project_id="p")

    names = {c.name for c in _chroma_client().list_collections()}
    assert name not in names and f"{name}--new" not in names
    assert find_vector_store(None, "p") is None


async def test_a_shared_collection_dropped_by_another_move_meanwhile_is_passed_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chatbot_engine.rag import vector_store as vector_store_module

    _shared_with({"alpha": 2})
    real = vector_store_module._shared_collections
    monkeypatch.setattr(
        vector_store_module,
        "_shared_collections",
        lambda client: [*real(client), "documents--already-gone"],
    )

    await vector_store_module.ensure_moved("alpha")

    assert count_chunks(find_vector_store(None, "alpha")) == 2


# --- a project's chats and its tidying --------------------------------------------


async def test_a_projects_search_does_not_wait_for_another_projects_indexing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project p's write waits for indexing's one thread, which another
    tenant's document holds (a PDF read can take a minute); p's update or
    delete then tidies. p's chats must not wait for the other tenant: the
    write holds no gate while it waits for a thread, and a tidy with
    nothing to do never has the gate alone, which would stop every search
    that arrives after it."""
    import asyncio
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from chatbot_engine.rag import vector_store as vs
    from chatbot_engine.rag import workers

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index")
    monkeypatch.setattr(workers, "_pool", pool)
    store = ChromaChunkStore()
    await store.write(doc_id="a", project_id="p", chunks=_docs("p", "a", 1))

    release = threading.Event()
    real_embed = vs._embed

    def embed(model, texts):
        vectors = real_embed(model, texts)
        # The other tenant's document reaches the pool as p's is embedded.
        pool.submit(release.wait, 10)
        return vectors

    monkeypatch.setattr(vs, "_embed", embed)
    writing = asyncio.create_task(
        store.write(doc_id="b", project_id="p", chunks=_docs("p", "b", 1))
    )
    await asyncio.sleep(0.3)
    tidying = asyncio.create_task(vs.tidy("p"))
    await asyncio.sleep(0.1)

    start = time.monotonic()
    try:
        async with asyncio.timeout(2):
            async with vs.gate("p").shared():  # agent/retriever.py's search
                waited = time.monotonic() - start
            await tidying
    finally:
        release.set()
        await writing
        pool.shutdown(wait=True)

    assert waited < 0.5, f"p's search waited {waited:.1f}s for another tenant"
    assert sorted(stored_chunks(where={"project_id": "p"})["documents"]) == [
        "a part 0",
        "b part 0",
    ]


async def test_an_update_or_delete_with_nothing_to_tidy_never_has_the_gate_alone(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chatbot_engine.rag import workers

    store = ChromaChunkStore()
    await store.write(doc_id="a", project_id="p", chunks=_docs("p", "a", 3))
    await store.write(doc_id="b", project_id="p", chunks=_docs("p", "b", 3))
    alone = 0
    real = workers.Gate.alone

    def counting(self):
        nonlocal alone
        alone += 1
        return real(self)

    monkeypatch.setattr(workers.Gate, "alone", counting)
    await store.write(doc_id="a", project_id="p", chunks=_docs("p", "a", 2))
    await store.delete(doc_id="b", project_id="p")

    assert alone == 0
    assert count_chunks(find_vector_store(None, "p")) == 2


async def test_a_rebuild_copies_while_the_project_is_searched_written_and_deleted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy is made sharing the gate, so the project's searches go on
    while it runs; only the swap at the end has the gate alone. What was
    written and deleted meanwhile is caught up before the copy takes the
    collection's place: nothing written is lost, and nothing deleted is
    brought back."""
    import asyncio
    import threading

    from chatbot_engine.rag import vector_store as vs

    monkeypatch.setattr(vs, "REBUILD_AFTER", 1)
    monkeypatch.setattr(vs, "_COPY_BATCH", 2)
    store = ChromaChunkStore()
    await store.write(doc_id="kept", project_id="p", chunks=_docs("p", "kept", 4))
    await store.write(doc_id="x", project_id="p", chunks=_docs("p", "x", 2))
    await store.write(doc_id="churn", project_id="p", chunks=_docs("p", "churn", 7))
    before = find_vector_store(None, "p")
    assert before is not None
    old_id = before._collection.id

    loop = asyncio.get_running_loop()
    copying, go = asyncio.Event(), threading.Event()
    real_copy = vs._copy_slice
    slices = 0

    def copy_slice(name: str, offset: int) -> int:
        nonlocal slices
        slices += 1
        if slices == 2:  # one slice copied: the project goes on meanwhile
            loop.call_soon_threadsafe(copying.set)
            go.wait(10)
        return real_copy(name, offset)

    monkeypatch.setattr(vs, "_copy_slice", copy_slice)
    rebuilding = asyncio.create_task(store.delete(doc_id="churn", project_id="p"))
    try:
        async with asyncio.timeout(5):
            await copying.wait()
            async with vs.gate("p").shared():  # a search, while it copies
                pass
            assert await store.delete(doc_id="x", project_id="p") == 2
            await store.write(doc_id="new", project_id="p", chunks=_docs("p", "new", 2))
    finally:
        go.set()
    await rebuilding

    after = find_vector_store(None, "p")
    assert after is not None and after._collection.id != old_id, "rebuilt"
    assert sorted(stored_chunks(where={"project_id": "p"})["documents"]) == [
        *(f"kept part {i}" for i in range(4)),
        "new part 0",
        "new part 1",
    ]
    names = {c.name for c in vs._chroma_client().list_collections()}
    assert f"{collection_name(None, 'p')}--new" not in names


async def test_a_rebuild_racing_a_delete_from_outside_its_gate_keeps_every_live_chunk(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pages read by offset shift under a delete made meanwhile, here one
    no gate of this process saw: the copy skips live chunks and keeps
    deleted ones, until the catch-up puts both right."""
    from chatbot_engine.rag import vector_store as vs

    monkeypatch.setattr(vs, "REBUILD_AFTER", 1)
    monkeypatch.setattr(vs, "_COPY_BATCH", 2)
    store = ChromaChunkStore()
    await store.write(doc_id="x", project_id="p", chunks=_docs("p", "x", 2))
    await store.write(doc_id="kept", project_id="p", chunks=_docs("p", "kept", 4))
    await store.write(doc_id="churn", project_id="p", chunks=_docs("p", "churn", 7))
    name = collection_name(None, "p")
    calls = 0
    real_room = vs.ensure_room

    def room(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:  # one slice copied: x goes, behind the rebuild's back
            collection = vs._chroma_client().get_collection(name)
            gone = collection.get(where={"doc_id": "x"}, include=[])["ids"]
            collection.delete(ids=gone)
        real_room(*args)

    monkeypatch.setattr(vs, "ensure_room", room)
    await store.delete(doc_id="churn", project_id="p")

    assert sorted(stored_chunks(where={"project_id": "p"})["documents"]) == [
        f"kept part {i}" for i in range(4)
    ]


async def test_against_a_chroma_server_nothing_is_rebuilt_or_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The gate is one process's: another replica's writes and deletes would
    race a rebuild or a drop, so neither runs against a server. Nor is a
    delete counted, which only a rebuild would read."""
    import chromadb

    from chatbot_engine.rag import vector_store as vs

    _set(monkeypatch, ENGINE_CHROMA_URL="http://chroma.internal:8000")
    vs.reset_vector_store()
    monkeypatch.setattr(vs, "_client", chromadb.PersistentClient(path=str(tmp_path)))
    monkeypatch.setattr(vs, "REBUILD_AFTER", 1)
    store = ChromaChunkStore()
    await store.write(doc_id="kept", project_id="p", chunks=_docs("p", "kept", 2))
    await store.write(doc_id="churn", project_id="p", chunks=_docs("p", "churn", 5))
    before = find_vector_store(None, "p")
    assert before is not None

    await store.delete(doc_id="churn", project_id="p")
    after = find_vector_store(None, "p")
    assert after is not None and after._collection.id == before._collection.id
    assert not (after._collection.metadata or {}).get("deleted_chunks")

    await store.delete(doc_id="kept", project_id="p")
    emptied = find_vector_store(None, "p")
    assert emptied is not None and count_chunks(emptied) == 0, "not dropped"
    vs.reset_vector_store()


# --- a project the 0.1.30 move has not reached yet --------------------------------


async def test_a_project_not_moved_yet_is_deleted_and_purged_where_it_is_on_a_full_volume(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A move needs room for a copy of the project's chunks, and a delete
    is how room is made: a delete or a purge of a project still in the
    shared collection removes its chunks there, below the floor too."""
    from chatbot_engine.errors import StorageFullError
    from chatbot_engine.rag import vector_store as vs

    _shared_with({"alpha": 3, "beta": 2})

    def full(*_: object) -> None:
        raise StorageFullError("below the free-space floor")

    monkeypatch.setattr(vs, "ensure_room", full)

    assert await ChromaChunkStore().delete(doc_id="d", project_id="alpha") == 3
    assert await ChromaChunkStore().delete_project(project_id="beta") == 2

    names = {c.name for c in vs._chroma_client().list_collections()}
    assert collection_name() not in names, "the emptied shared collection is dropped"
    assert not any("--p-" in n for n in names), "nothing was moved, nor needed room"
    assert stored_chunks()["ids"] == []


def test_the_free_space_refusal_says_what_it_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not always "the upload was not indexed": a move or a rebuild is
    refused too, and its message should send its reader to the right thing."""
    from chatbot_engine.disk import ensure_room
    from chatbot_engine.errors import StorageFullError

    _set(monkeypatch, ENGINE_MIN_FREE_MB=str(10**12))

    with pytest.raises(StorageFullError, match="so the document was not indexed"):
        ensure_room()
    with pytest.raises(
        StorageFullError, match="so the project's chunks were not moved"
    ):
        ensure_room(
            "the project's chunks were not moved into a collection of their own"
        )


async def test_a_move_refused_for_room_says_it_was_the_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chatbot_engine.errors import StorageFullError
    from chatbot_engine.rag.vector_store import ensure_moved

    _shared_with({"alpha": 3})
    _set(monkeypatch, ENGINE_MIN_FREE_MB=str(10**12))

    with pytest.raises(StorageFullError, match="chunks were not moved"):
        await ensure_moved("alpha")


# --- a rebuild cut short ----------------------------------------------------------


async def test_a_delete_and_a_purge_reach_a_rebuild_copy_whose_collection_is_gone(
    client: TestClient,
) -> None:
    """The engine stopped between dropping a collection and giving its
    rebuilt copy the name. A delete or a purge must reach the copy, or the
    next search takes it back with what they removed."""
    from chatbot_engine.rag.vector_store import _chroma_client, reset_vector_store

    def cut_short() -> None:
        name = collection_name(None, "p")
        _chroma_client().get_collection(name).modify(name=f"{name}--new")
        reset_vector_store()

    store = ChromaChunkStore()
    await store.write(doc_id="d", project_id="p", chunks=_docs("p", "d", 2))
    await store.write(doc_id="e", project_id="p", chunks=_docs("p", "e", 2))
    cut_short()

    assert await ChromaChunkStore().delete(doc_id="d", project_id="p") == 2
    cut_short()
    assert await ChromaChunkStore().delete_project(project_id="p") == 2

    assert find_vector_store(None, "p") is None, "the next search finds nothing"
    assert stored_chunks()["ids"] == []


# --- a write that fails part-way ----------------------------------------------------


async def test_a_failed_write_counts_what_it_took_back_and_leaves_no_empty_collection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What landed of a failed write is removed like a delete removes it: its
    room is counted towards a rebuild, and a project's first document that
    failed leaves no empty collection behind."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from chatbot_engine.rag.vector_store import _chroma_client

    _set(monkeypatch, ENGINE_INDEX_BATCH_SIZE="1")
    store = ChromaChunkStore()
    await store.write(doc_id="kept", project_id="p", chunks=_docs("p", "kept", 2))
    original = DeterministicFakeEmbedding.embed_documents
    calls = 0

    def third_call_fails(self, texts):
        nonlocal calls
        calls += 1
        if calls % 3 == 0:
            raise RuntimeError("provider down")
        return original(self, texts)

    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", third_call_fails)
    with pytest.raises(RuntimeError):
        await store.write(doc_id="new", project_id="p", chunks=_docs("p", "new", 4))
    kept = find_vector_store(None, "p")
    assert kept is not None and count_chunks(kept) == 2
    collection = _chroma_client().get_collection(collection_name(None, "p"))
    assert (collection.metadata or {}).get("deleted_chunks") == 2

    calls = 0
    with pytest.raises(RuntimeError):
        await store.write(doc_id="first", project_id="q", chunks=_docs("q", "first", 4))
    assert find_vector_store(None, "q") is None, "no empty collection is left"


async def test_a_cancelled_write_embeds_no_more_slices(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delete during an upload stops it between slices: nothing more is
    embedded, or paid for, and what landed goes again."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from chatbot_engine.errors import DocumentDeletedError
    from chatbot_engine.rag.vector_store import EmbeddedChunks

    _set(monkeypatch, ENGINE_INDEX_BATCH_SIZE="1")
    original = DeterministicFakeEmbedding.embed_documents
    embedded: list[str] = []
    deleted = False

    def embed(self, texts):
        nonlocal deleted
        embedded.extend(texts)
        deleted = True  # the delete arrives while the first slice embeds
        return original(self, texts)

    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", embed)
    counted = EmbeddedChunks()
    with pytest.raises(DocumentDeletedError):
        await ChromaChunkStore().write(
            doc_id="d",
            project_id="p",
            chunks=_docs("p", "d", 5),
            embedded=counted,
            cancelled=lambda: deleted,
        )

    assert embedded == ["d part 0"] and counted.chunks == 1
    assert find_vector_store(None, "p") is None, "what landed went again"
