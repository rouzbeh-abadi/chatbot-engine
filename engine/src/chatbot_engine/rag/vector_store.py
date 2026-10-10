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

import asyncio
import contextlib
import re
import threading
import uuid
from collections.abc import Collection
from typing import Any
from urllib.parse import urlparse

import chromadb
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

from chatbot_engine.disk import ensure_room
from chatbot_engine.rag.embeddings import get_embeddings, resolve_embedding_model
from chatbot_engine.rag.workers import KeyedLocks, run_indexing
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


def _owned_by(doc_id: str, project_id: str) -> dict[str, Any]:
    """The filter for one document's chunks: its id and its project together.

    Never the id alone. A `doc_id` is derived from the project and the caller's
    own name for the file, so anyone who can guess a file name can work one
    out; the project is what keeps one project's request from reaching
    another's chunks.
    """
    return {"$and": [{"doc_id": doc_id}, {"project_id": project_id}]}


#: One write of a document at a time in this process.
_writing = KeyedLocks()


def _embed_and_write(store: Chroma, chunks: list[Document], ids: list[str]) -> None:
    """Embed `chunks` and upsert them under `ids`. Blocks; run off the loop.

    The room is checked first, every slice: a full volume can hang embedded
    Chroma for every tenant and leave its index unreadable (DISK-3, DISK-4).
    """
    ensure_room()
    embedder = store.embeddings
    if embedder is None:  # every store here is opened with one
        raise RuntimeError(f"the collection {store._collection.name!r} has no embedder")
    texts = [chunk.page_content for chunk in chunks]
    vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
    store._collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
    )


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

    async def write(
        self, *, doc_id: str, project_id: str, chunks: list[Document]
    ) -> None:
        """Replace everything stored for one document with `chunks`.

        The new version goes in under ids of its own, `ENGINE_INDEX_BATCH_SIZE`
        chunks at a time: each slice is embedded and written before the next
        is embedded. So a document's vectors are never all in memory at once,
        however many chunks it makes (INGEST-1), they reach Chroma as float32
        arrays, which it reads without holding the interpreter the way it
        holds it for lists (DISK-2), and no write is larger than Chroma takes
        in one (INGEST-13).

        Only once every slice has landed is the old version removed: all of
        it, and its chunks under another embedding model. Embedding is the
        step that fails (a rate limit, a provider outage, a bad key); a
        failure part-way removes what of the new version landed, so the
        version already indexed goes on answering, whole.
        """
        store = self._store
        version = uuid.uuid4().hex[:12]
        ids = [f"{doc_id}:{version}:{index}" for index in range(len(chunks))]
        size = get_settings().index_batch_size
        tried = 0

        # One write of a document at a time in this process: two at once
        # would each remove the other's version as the old one, and leave
        # the document with part of one or none.
        async with _writing((project_id, doc_id)):
            try:
                for start in range(0, len(chunks), size):
                    tried = start + size
                    await self._add(store, chunks[start:tried], ids[start:tried])
            except BaseException:
                if tried:
                    with contextlib.suppress(Exception):
                        await run_indexing(store._collection.delete, ids[:tried])
                raise

            await self._remove(doc_id=doc_id, project_id=project_id, keeping=ids)

    async def _add(self, store: Chroma, chunks: list[Document], ids: list[str]) -> None:
        """Embed one slice of chunks and write it, on indexing's own threads."""
        await run_indexing(_embed_and_write, store, chunks, ids)

    async def delete(self, *, doc_id: str, project_id: str) -> int:
        """Remove every chunk of one project's document, and say how many."""
        return await self._remove(doc_id=doc_id, project_id=project_id)

    async def delete_project(self, *, project_id: str) -> int:
        """Remove every chunk of a project, under every embedding model, and
        say how many: what a purge leaves no record to name."""
        removed = 0
        for store in await asyncio.to_thread(_every_store):
            found = await asyncio.to_thread(
                store.get, where={"project_id": project_id}, include=[]
            )
            if found["ids"]:
                await store.adelete(ids=found["ids"])
                removed += len(found["ids"])
        return removed

    async def _remove(
        self, *, doc_id: str, project_id: str, keeping: Collection[str] = ()
    ) -> int:
        """Remove one document's chunks, except `keeping` in this store's own
        collection, and say how many went.

        Swept across every collection, not only this store's own: a document
        may have been indexed under a different embedding model than the one
        now writing or deleting it, and a sweep that missed it would leave
        chunks that still answer queries with nothing left to name them.
        """
        own = self._store._collection.name
        kept = set(keeping)
        removed = 0

        # In a thread: listing the collections is a call into Chroma, and on a
        # full disk one can hang holding its lock (DISK-3).
        for store in await asyncio.to_thread(_every_store):
            found = await asyncio.to_thread(
                store.get, where=_owned_by(doc_id, project_id), include=[]
            )
            stale = [
                chunk_id
                for chunk_id in found["ids"]
                if store._collection.name != own or chunk_id not in kept
            ]
            if stale:
                await store.adelete(ids=stale)
                removed += len(stale)

        return removed
