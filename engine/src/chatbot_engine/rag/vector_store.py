"""Chroma, keyed so re-uploading a document replaces its chunks.

Embedded Chroma: no server, just files under `ENGINE_CHROMA_DIR`. One client per
collection per process, because several clients on one directory contend for
the same SQLite file.

One collection per embedding model. Vectors from two models are not comparable,
so chunks embedded with one must never be searched with another. Keying the
collection by model makes that structural: a project that embeds with a
different model reads and writes its own collection, and a query can only ever
meet vectors produced the same way it was.
"""

from __future__ import annotations

import re
import threading
from urllib.parse import urlparse

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from chatbot_engine.rag.embeddings import get_embeddings, resolve_embedding_model
from chatbot_engine.settings import get_settings


class EmptyVectorStoreError(RuntimeError):
    """The store holds no documents, so retrieval cannot return anything."""


# One chromadb client per process, and one `Chroma` wrapper per collection.
# Construction is guarded by a lock rather than `@lru_cache`: FastAPI runs the
# sync document dependencies in a threadpool, and `lru_cache` does not
# serialise concurrent first calls -- two threads would both build a client and
# race on chromadb's process-global state (a KeyError deep in
# `PersistentClient`). The lock makes exactly one thread construct each.
_client: chromadb.ClientAPI | None = None
_stores: dict[str, Chroma] = {}
_store_lock = threading.Lock()


def _chroma_client() -> chromadb.ClientAPI:
    """The process's one Chroma client: a server when `ENGINE_CHROMA_URL` is
    set, otherwise the files under `ENGINE_CHROMA_DIR`.

    The two are interchangeable above this line. Embedded Chroma is owned by
    one process, so it is the right default for one engine and the wrong one
    for several; a server is what lets replicas share a knowledge base.
    """
    global _client

    if _client is None:
        settings = get_settings()
        if settings.chroma_url:
            url = urlparse(settings.chroma_url)
            _client = chromadb.HttpClient(
                host=url.hostname or "localhost",
                port=url.port or (443 if url.scheme == "https" else 8000),
                ssl=url.scheme == "https",
                headers=_auth_headers(
                    settings.chroma_token, settings.chroma_token_header
                ),
            )
        else:
            _client = chromadb.PersistentClient(path=str(settings.chroma_dir))

    return _client


def _auth_headers(token: str | None, header: str) -> dict[str, str] | None:
    """The credential a Chroma server expects, in the header it expects it in."""
    if not token:
        return None
    if header.lower() == "authorization":
        return {"Authorization": f"Bearer {token}"}
    return {header: token}


def vector_store_reachable() -> bool:
    """Whether the vector store answers. For the readiness probe.

    Always true embedded, short of a broken disk. Against a server it is the
    check that matters: an engine whose Chroma is down can neither index nor
    retrieve, and should say so before it takes traffic.
    """
    try:
        _chroma_client().heartbeat()
    except Exception:
        return False

    return True


def collection_name(embedding_model: str | None = None) -> str:
    """The collection that holds vectors from one embedding model.

    `ENGINE_CHROMA_COLLECTION` is the base name; the model is appended so that
    `openai/text-embedding-3-small` and `-large` land in different collections.
    Chroma allows only `[a-zA-Z0-9._-]`, so the slash is folded away.
    """
    model = resolve_embedding_model(embedding_model)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-")

    return f"{get_settings().chroma_collection}--{slug}"


def open_vector_store(embedding_model: str | None = None) -> Chroma:
    """Open the collection for `embedding_model`, creating it once if absent.

    `embedding_model` comes from the assistant config; None uses the engine
    default. Each model has its own collection, so a project embedding with a
    different model neither sees nor disturbs another's vectors.
    """
    name = collection_name(embedding_model)

    if name not in _stores:
        with _store_lock:
            # Re-check inside the lock: another thread may have built it while
            # this one waited.
            if name not in _stores:
                client = _chroma_client()
                _adopt_legacy_collection(client, name)
                _stores[name] = Chroma(
                    client=client,
                    collection_name=name,
                    embedding_function=get_embeddings(embedding_model),
                )

    return _stores[name]


def _adopt_legacy_collection(client: chromadb.ClientAPI, name: str) -> None:
    """Rename a pre-model-keyed collection into the default model's place.

    Before collections were keyed by model there was one, named
    `ENGINE_CHROMA_COLLECTION`, and everything in it was embedded with the
    engine's single default model. So when the default model's collection is
    first opened and does not exist yet, an existing legacy collection is that
    collection under its old name, and is renamed rather than left orphaned.
    A non-default model is never adopted: its vectors could not be in there.
    """
    settings = get_settings()
    legacy = settings.chroma_collection

    if name != collection_name(None):
        return

    existing = {collection.name for collection in client.list_collections()}
    if legacy not in existing:
        return

    if name in existing:
        # Created by a query that ran before anything was indexed. Empty, so
        # it holds nothing worth keeping and only blocks the rename.
        if client.get_collection(name).count() > 0:
            return
        client.delete_collection(name)

    client.get_collection(legacy).modify(name=name)


def reset_vector_store() -> None:
    """Drop the cached clients. For tests, and after changing ENGINE_CHROMA_DIR."""
    global _client

    with _store_lock:
        _stores.clear()
        _client = None


def _every_store() -> list[Chroma]:
    """Every collection this engine has, opened or not.

    Collections are found on disk rather than in `_stores`, so a delete after a
    restart still reaches a collection this process has not touched yet. The
    prefix filter keeps to this engine's own collections.
    """
    settings = get_settings()
    default = open_vector_store()
    client = _chroma_client()
    prefix = f"{settings.chroma_collection}--"

    stores = {collection_name(): default}
    for collection in client.list_collections():
        name = collection.name
        if not name.startswith(prefix) or name in stores:
            continue
        with _store_lock:
            if name not in _stores:
                _stores[name] = Chroma(
                    client=client,
                    collection_name=name,
                    embedding_function=default.embeddings,
                )
        stores[name] = _stores[name]

    return list(stores.values())


def load_vector_store(embedding_model: str | None = None) -> Chroma:
    """Open an already populated collection, for querying.

    Raises if nothing has been ingested with this model, so an empty collection
    fails loudly instead of silently retrieving zero chunks and letting the
    model invent an answer from nothing. The message names the model, because
    "nothing indexed" after a successful seed usually means the query and the
    documents disagree about which model they use.
    """
    store = open_vector_store(embedding_model)

    if count_chunks(store) == 0:
        raise EmptyVectorStoreError(
            f"no documents indexed for embedding model "
            f"{resolve_embedding_model(embedding_model)!r} in "
            f"{get_settings().chroma_dir} -- upload some through PUT /documents "
            "with that model, or run `make seed` to load the example knowledge "
            "base"
        )

    return store


def count_chunks(store: Chroma | None = None) -> int:
    """How many chunks are stored in one collection, across every project."""
    store = store or open_vector_store()

    return len(store.get(include=[])["ids"])


class ChromaChunkStore:
    """The write side: one document's chunks, replaced as a unit."""

    def __init__(
        self,
        embedding_model: str | None = None,
        store: Chroma | None = None,
    ) -> None:
        # Opened lazily, so constructing this at wiring time does not fix the
        # embedding model before a request has said which one to use.
        self._explicit = store
        self._embedding_model = embedding_model

    @property
    def _store(self) -> Chroma:
        return self._explicit or open_vector_store(self._embedding_model)

    async def write(self, *, doc_id: str, chunks: list[Document]) -> None:
        """Replace everything stored for `doc_id` with `chunks`.

        Delete first, rather than overwriting ids one by one: a shorter second
        version would otherwise leave the tail of the first one behind, still
        answering queries.
        """
        await self.delete(doc_id=doc_id)

        if not chunks:
            return

        # Deterministic ids, so this is a replace even if the delete missed.
        await self._store.aadd_documents(
            documents=chunks,
            ids=[f"{doc_id}:{index}" for index, _ in enumerate(chunks)],
        )

    async def delete(self, *, doc_id: str) -> int:
        """Remove every chunk belonging to one document, and say how many.

        Swept across every collection, not only this store's own: a document
        may have been indexed under a different embedding model than the one
        deleting it, and a delete that missed it would leave chunks that still
        answer queries with nothing left to name them.
        """
        removed = 0

        for store in _every_store():
            existing = store.get(where={"doc_id": doc_id}, include=[])["ids"]
            if existing:
                await store.adelete(ids=existing)
                removed += len(existing)

        return removed
