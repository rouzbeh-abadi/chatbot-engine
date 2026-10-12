"""Chroma, keyed so re-uploading a document replaces its chunks.

Embedded Chroma: no server, just files under `ENGINE_CHROMA_DIR`. One client per
collection per process, because several clients on one directory contend for
the same SQLite file.

One collection per embedding model and project. Vectors from two models are
not comparable, so chunks embedded with one must never be searched with
another. Keying the collection by model makes that structural: a project
that embeds with a different model reads and writes its own collection, and
a query can only ever meet vectors produced the same way it was.

Keying it by project too means a search reads only its project's vectors, and
a project's knowledge can be dropped whole, with the room it took. Chroma 1.5
keeps a deleted vector's room in its collection's index for as long as the
collection lives, and keeps a dropped collection's files on disk
(docs/review-2026-10.md, DISK-1). So a project's collection is dropped when
its last chunk goes, rebuilt from its live chunks once its deleted ones
outnumber them (`tidy`), and the files a dropped collection leaves behind are
removed (`_sweep_orphans`). Until 0.1.30 every project of a model shared one
collection; each project's chunks are moved into a collection of their own
the first time the project is used, and the rest in the background
(`ensure_moved`, `move_everything`).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import re
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Collection
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chromadb
import numpy as np
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document

from chatbot_engine.disk import ensure_room
from chatbot_engine.rag import sparse
from chatbot_engine.rag.embeddings import get_embeddings, resolve_embedding_model
from chatbot_engine.rag.workers import Gate, KeyedGates, KeyedLocks, run_indexing
from chatbot_engine.settings import get_settings

logger = logging.getLogger(__name__)


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


#: What a project's collection name has after its model's: the project, as a
#: hash, since a collection name may hold only `[a-zA-Z0-9._-]`.
_PROJECT = "--p-"

#: What a collection being rebuilt is called until it takes the old one's place.
_REBUILDING = "--new"

#: How many chunks are read and written at a time when a collection is moved
#: or rebuilt.
_COPY_BATCH = 1_000

#: Deleted chunks a project's collection may hold past its live ones before
#: it is rebuilt from them: the room it takes stays within about twice what
#: it holds, and a small project is not rebuilt for every delete.
REBUILD_AFTER = 2_000

#: Where a collection keeps how many chunks have been deleted from it since
#: it was made, in its own metadata.
_DELETED = "deleted_chunks"


def _project_key(project_id: str) -> str:
    """A project id as a collection name may hold it."""
    return hashlib.sha256(project_id.encode()).hexdigest()[:32]


def collection_name(
    embedding_model: str | None = None, project_id: str | None = None
) -> str:
    """The collection that holds one project's vectors from one embedding model.

    `ENGINE_CHROMA_COLLECTION` is the base name; the model is appended so that
    `openai/text-embedding-3-small` and `-large` land in different collections,
    and the project after it. Chroma allows only `[a-zA-Z0-9._-]`, so the
    model's slash is folded away and the project is hashed. Without a project,
    the collection every project of the model shared until 0.1.30.
    """
    model = resolve_embedding_model(embedding_model)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-")
    shared = f"{get_settings().chroma_collection}--{slug}"

    return (
        shared
        if project_id is None
        else f"{shared}{_PROJECT}{_project_key(project_id)}"
    )


def _store_for(
    name: str, embedding_model: str | None, *, create: bool
) -> Chroma | None:
    """The `Chroma` wrapper of one collection, made once and kept; None when
    the collection does not exist and `create` is false.

    The collection is looked up every time, so a wrapper of one that was
    dropped or rebuilt since (by this process or, against a Chroma server,
    another) is never used.
    """
    client = _chroma_client()
    try:
        found = client.get_collection(name, embedding_function=None)
    except NotFoundError:
        if _recover(client, name):
            found = client.get_collection(name, embedding_function=None)
        elif create:
            found = client.get_or_create_collection(name, embedding_function=None)
        else:
            _forget(name)
            return None

    with _store_lock:
        store = _stores.get(name)
        if store is None or store._collection.id != found.id:
            store = _stores[name] = Chroma(
                client=client,
                collection_name=name,
                embedding_function=get_embeddings(embedding_model),
            )
    return store


def _forget(name: str) -> None:
    """Drop the kept wrapper of a collection that is gone or replaced."""
    with _store_lock:
        _stores.pop(name, None)


def open_vector_store(
    embedding_model: str | None = None, project_id: str | None = None
) -> Chroma:
    """Open a project's collection for `embedding_model`, creating it if absent.

    `embedding_model` comes from the assistant config; None uses the engine
    default. Without a project, the collection every project of the model
    shared until 0.1.30, which is only read, and emptied, by the move.
    """
    name = collection_name(embedding_model, project_id)
    if project_id is None:
        with _store_lock:
            _adopt_legacy_collection(_chroma_client(), name)
    store = _store_for(name, embedding_model, create=True)
    assert store is not None  # created when absent
    return store


def find_vector_store(embedding_model: str | None, project_id: str) -> Chroma | None:
    """A project's collection for `embedding_model`, or None when the
    project has nothing indexed with it. For searching: a search never makes
    a collection."""
    return _store_for(
        collection_name(embedding_model, project_id), embedding_model, create=False
    )


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
    global _client, _shared_left

    with _store_lock:
        _stores.clear()
        _client = None
    _moved.clear()
    _shared_left = None


def _engine_collections(client: chromadb.ClientAPI) -> list[str]:
    """The names of this engine's collections, by the base name's prefix."""
    prefix = f"{get_settings().chroma_collection}--"
    return [c.name for c in client.list_collections() if c.name.startswith(prefix)]


def _shared_collections(client: chromadb.ClientAPI) -> list[str]:
    """The collections every project of a model shared until 0.1.30."""
    with _store_lock:
        _adopt_legacy_collection(client, collection_name(None))
    return [
        name
        for name in _engine_collections(client)
        if _PROJECT not in name and not name.endswith(_REBUILDING)
    ]


def _project_collections(client: chromadb.ClientAPI, project_id: str) -> list[str]:
    """A project's collections, one per embedding model it was indexed with."""
    suffix = f"{_PROJECT}{_project_key(project_id)}"
    return [name for name in _engine_collections(client) if name.endswith(suffix)]


def _recover(client: chromadb.ClientAPI, name: str) -> bool:
    """Finish a rebuild the engine stopped in the middle of, for a collection
    that is not there: the old one dropped, the new one not yet under its
    name. Whether there was one to finish."""
    if name.endswith(_REBUILDING):
        return False
    try:
        rebuilt = client.get_collection(f"{name}{_REBUILDING}", embedding_function=None)
    except NotFoundError:
        return False
    rebuilt.modify(name=name)
    logger.warning("finished rebuilding %s, cut short before", name)
    return True


def count_chunks(store: Chroma | None = None) -> int:
    """How many chunks one collection holds; without one, every collection of
    this engine together."""
    if store is not None:
        return store._collection.count()
    client = _chroma_client()
    return sum(
        client.get_collection(name).count() for name in _engine_collections(client)
    )


def stored_chunks(where: dict[str, Any] | None = None) -> dict[str, list[Any]]:
    """The ids, documents and metadatas of the chunks `where` names, in every
    collection of this engine. For diagnostics and tests: a search reads one
    project's collection."""
    client = _chroma_client()
    out: dict[str, list[Any]] = {"ids": [], "documents": [], "metadatas": []}
    for name in _engine_collections(client):
        got = client.get_collection(name).get(
            where=where, include=["documents", "metadatas"]
        )
        out["ids"].extend(got["ids"])
        out["documents"].extend(got["documents"] or [])
        out["metadatas"].extend(got["metadatas"] or [])
    return out


def load_vector_store(
    embedding_model: str | None = None, project_id: str = "support"
) -> Chroma:
    """Open a project's populated collection, for the CLI's search.

    Raises if nothing has been ingested for the project with this model, so
    an empty collection fails loudly instead of silently retrieving zero
    chunks. The message names the model, because "nothing indexed" after a
    successful seed usually means the query and the documents disagree about
    which model they use.
    """
    store = find_vector_store(embedding_model, project_id)

    if store is None or count_chunks(store) == 0:
        raise EmptyVectorStoreError(
            f"no documents indexed for project {project_id!r} with embedding "
            f"model {resolve_embedding_model(embedding_model)!r} in "
            f"{get_settings().chroma_dir} -- upload some through PUT /documents "
            "with that model, or run `make seed` to load the example knowledge "
            "base"
        )

    return store


# --- one project's collections: moving, tidying, dropping ---------------------

#: Per project: searches and writes share it; a move, a rebuild or a drop of
#: its collections has it alone.
_gates = KeyedGates()

#: Projects whose chunks this process knows are in collections of their own.
_moved: set[str] = set()

#: Whether any collection shared by every project is left; checked once.
_shared_left: bool | None = None

#: One update of a collection's deleted count at a time.
_counting = threading.Lock()


def gate(project_id: str) -> Gate:
    """The gate a project's vectors are read, written and replaced behind."""
    return _gates(project_id)


async def ensure_moved(project_id: str) -> None:
    """Move a project's chunks out of the collections every project shared
    until 0.1.30, into its own, before anything reads or writes them. Once
    per project and process; nothing to do once no shared collection is left.
    """
    global _shared_left
    if project_id in _moved or _shared_left is False:
        return
    async with gate(project_id).alone():
        if project_id in _moved:
            return
        left = await asyncio.to_thread(_move_project, project_id)
        if not left:
            _shared_left = False
        _moved.add(project_id)


def _move_project(project_id: str) -> bool:
    """Move one project's chunks from every shared collection into its own;
    whether any shared collection is left after. Blocks."""
    client = _chroma_client()
    left = False
    for shared_name in _shared_collections(client):
        try:
            left = _move_out_of(client, shared_name, project_id) or left
        except NotFoundError:
            # Emptied and dropped meanwhile by the move of another project.
            continue
    return left


def _move_out_of(client: chromadb.ClientAPI, shared_name: str, project_id: str) -> bool:
    """Move one project's chunks out of one shared collection; whether the
    collection is still there after, holding other projects' chunks."""
    shared = client.get_collection(shared_name)
    target = f"{shared_name}{_PROJECT}{_project_key(project_id)}"
    # Every chunk copied first, and only then deleted from the shared
    # collection: a move cut short (a full volume) leaves the project
    # whole where it was, and searched there (`store_to_search`).
    copied: list[str] = []
    while True:
        got = shared.get(
            where={"project_id": project_id},
            include=["embeddings", "documents", "metadatas"],
            limit=_COPY_BATCH,
            offset=len(copied),
        )
        if not got["ids"]:
            break
        ensure_room()
        own = client.get_or_create_collection(target, embedding_function=None)
        own.upsert(
            ids=got["ids"],
            embeddings=got["embeddings"],
            documents=got["documents"],
            metadatas=got["metadatas"],
        )
        copied.extend(got["ids"])
    for start in range(0, len(copied), _COPY_BATCH):
        shared.delete(ids=copied[start : start + _COPY_BATCH])
    if copied:
        logger.info(
            "moved %d chunks of a project into a collection of its own",
            len(copied),
        )
        sparse.invalidate(project_id)
    if shared.count() > 0:
        return True
    with contextlib.suppress(NotFoundError):
        client.delete_collection(shared_name)
        logger.info("dropped %s, emptied by the move", shared_name)
    _forget(shared_name)
    _sweep_orphans()
    return False


async def try_to_move(project_id: str) -> bool:
    """`ensure_moved`, for a search: whether the project's chunks are in its
    own collections. A move that could not run (a full volume) is logged, and
    the search reads the shared collection, where they all still are."""
    try:
        await ensure_moved(project_id)
    except Exception:
        logger.exception("moving a project's chunks failed; searching where they are")
        return False
    return True


def store_to_search(
    embedding_model: str | None, project_id: str, *, moved: bool = True
) -> Chroma | None:
    """The collection a project's search reads: its own, or, when its chunks
    could not be moved out of the shared one yet, that one."""
    if moved:
        return find_vector_store(embedding_model, project_id)
    return _store_for(collection_name(embedding_model), embedding_model, create=False)


async def move_everything() -> None:
    """Move every project still in a shared collection, one at a time, then
    drop the emptied collections. Run in the background at startup: a project
    used before its turn is moved then (`ensure_moved`)."""
    global _shared_left
    settings = get_settings()
    if (
        not settings.chroma_url
        and not (Path(settings.chroma_dir) / "chroma.sqlite3").exists()
    ):
        # No knowledge base on this engine yet: nothing to move, and none made.
        _shared_left = False
        return
    try:
        while True:
            project_id = await asyncio.to_thread(_next_unmoved)
            if project_id is None:
                _shared_left = False
                return
            async with gate(project_id).alone():
                await asyncio.to_thread(_move_project, project_id)
                _moved.add(project_id)
    except Exception:
        # A full volume, say: what is left moves when its project is next used.
        logger.exception("moving chunks into collections of their own stopped")


def _next_unmoved() -> str | None:
    """A project with chunks still in a shared collection; None when none is
    left. A chunk with no project, which no search could ever reach, is
    deleted."""
    client = _chroma_client()
    for shared_name in _shared_collections(client):
        try:
            shared = client.get_collection(shared_name)
        except NotFoundError:
            continue
        while True:
            got = shared.get(include=["metadatas"], limit=1)
            if not got["ids"]:
                break
            metadata = (got["metadatas"] or [None])[0] or {}
            project_id = metadata.get("project_id")
            if isinstance(project_id, str) and project_id:
                return project_id
            shared.delete(ids=got["ids"])
        with contextlib.suppress(NotFoundError):
            client.delete_collection(shared_name)
        _forget(shared_name)
        _sweep_orphans()
    return None


async def tidy(project_id: str) -> None:
    """Give a project's room back: drop each of its collections that holds
    nothing, and rebuild one whose deleted chunks have come to outnumber its
    live ones (`REBUILD_AFTER`). Best effort: a failure is logged, and the
    next delete tries again."""
    try:
        async with gate(project_id).alone():
            await asyncio.to_thread(_tidy, project_id)
    except Exception:
        logger.exception("tidying a project's collections failed")


def _tidy(project_id: str) -> None:
    client = _chroma_client()
    dropped = False
    for name in _project_collections(client, project_id):
        collection = client.get_collection(name)
        live = collection.count()
        if live == 0:
            _drop(client, name)
            dropped = True
            continue
        deleted = int((collection.metadata or {}).get(_DELETED, 0))
        if deleted > max(live, REBUILD_AFTER):
            _rebuild(client, name)
            dropped = True
    if dropped:
        sparse.invalidate(project_id)
        _sweep_orphans()


def _rebuild(client: chromadb.ClientAPI, name: str) -> None:
    """Copy a collection's live chunks into a new one, which then takes its
    place and name; the old one, and the room its deleted chunks held, go."""
    with contextlib.suppress(NotFoundError):
        client.delete_collection(f"{name}{_REBUILDING}")  # a rebuild cut short
    old = client.get_collection(name)
    new = client.create_collection(f"{name}{_REBUILDING}", embedding_function=None)
    offset = 0
    try:
        while True:
            ensure_room()
            got = old.get(
                include=["embeddings", "documents", "metadatas"],
                limit=_COPY_BATCH,
                offset=offset,
            )
            if not got["ids"]:
                break
            new.add(
                ids=got["ids"],
                embeddings=got["embeddings"],
                documents=got["documents"],
                metadatas=got["metadatas"],
            )
            offset += len(got["ids"])
    except BaseException:
        # Not finished: the old one stays, and the part-copy goes, or it
        # would be taken later for a rebuild to finish.
        with contextlib.suppress(Exception):
            client.delete_collection(f"{name}{_REBUILDING}")
        raise
    client.delete_collection(name)
    new.modify(name=name)
    _forget(name)
    logger.info("rebuilt %s from its %d live chunks", name, offset)


def _count_deleted(name: str, removed: int) -> None:
    """Add to the deleted chunks a collection counts in its metadata, read
    afresh, so two deletes at once both count."""
    with _counting:
        collection = _chroma_client().get_collection(name, embedding_function=None)
        metadata = dict(collection.metadata or {})
        metadata[_DELETED] = int(metadata.get(_DELETED, 0)) + removed
        collection.modify(metadata=metadata)


#: A folder Chroma names after a segment of a collection.
_SEGMENT = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _sweep_orphans() -> int:
    """Remove the files of collections embedded Chroma has dropped; how many
    folders went.

    Chroma 1.5 keeps a dropped collection's index files, so dropping one would
    give nothing back (DISK-1). Its catalogue (`segments` in chroma.sqlite3)
    names every segment that still exists; a folder named like a segment
    that it does not name is one a dropped collection left. Read-only, and
    nothing is removed when the catalogue cannot be read. A Chroma server
    keeps its own files.
    """
    settings = get_settings()
    if settings.chroma_url:
        return 0
    root = Path(settings.chroma_dir)
    catalogue = root / "chroma.sqlite3"
    if not catalogue.exists():
        return 0
    # The folders first, the catalogue after: Chroma names a segment before it
    # makes its folder, so a folder seen here is named by the time the
    # catalogue is read, and one made meanwhile is not looked at at all.
    found = [e for e in root.iterdir() if e.is_dir() and _SEGMENT.match(e.name)]
    try:
        with contextlib.closing(
            sqlite3.connect(f"file:{catalogue}?mode=ro", uri=True, timeout=10)
        ) as db:
            named = {row[0] for row in db.execute("SELECT id FROM segments")}
    except sqlite3.Error as exc:
        logger.warning("could not read Chroma's segments, nothing removed: %s", exc)
        return 0
    removed = 0
    for entry in found:
        if entry.name not in named:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    if removed:
        logger.info("removed the files of %d dropped collections", removed)
    return removed


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


def _embed(embedding_model: str | None, texts: list[str]) -> np.ndarray:
    """Embed one slice of chunks. Blocks; run off the loop.

    The room is checked first, every slice: a full volume can hang embedded
    Chroma for every tenant and leave its index unreadable (DISK-3, DISK-4).
    """
    ensure_room()
    embedder = get_embeddings(embedding_model)
    return np.asarray(embedder.embed_documents(texts), dtype=np.float32)


def _write(
    store: Chroma, chunks: list[Document], ids: list[str], vectors: np.ndarray
) -> None:
    """Upsert one embedded slice of chunks under `ids`. Blocks; run off the loop."""
    ensure_room()
    store._collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )


class EmbeddedChunks:
    """What a write has done so far: how many chunks it had embedded, paid
    for whether or not the write then lands (docs/review-2026-10.md,
    INGEST-19), and whether every one of them landed, so the version written
    is the one that answers whatever fails after."""

    __slots__ = ("chunks", "landed")

    def __init__(self) -> None:
        self.chunks = 0
        self.landed = False


#: The write under way in this task, for `_add` to count what it embeds.
_EMBEDDING: ContextVar[EmbeddedChunks | None] = ContextVar("embedding", default=None)


class ChromaChunkStore:
    """The write side: one document's chunks, replaced as a unit, in its
    project's own collection."""

    def __init__(self, embedding_model: str | None = None) -> None:
        # Opened per call, so constructing this at wiring time does not fix
        # the embedding model before a request has said which one to use.
        self._embedding_model = embedding_model

    async def write(
        self,
        *,
        doc_id: str,
        project_id: str,
        chunks: list[Document],
        embedded: EmbeddedChunks | None = None,
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
        version already indexed goes on answering, whole. The room the old
        version took is given back after (`tidy`).

        `embedded`, when given, counts the chunks as each slice is embedded,
        which the provider bills whether or not the slice then lands, and
        says when every slice has landed.
        """
        await ensure_moved(project_id)
        version = uuid.uuid4().hex[:12]
        ids = [f"{doc_id}:{version}:{index}" for index in range(len(chunks))]
        size = get_settings().index_batch_size
        tried = 0

        # One write of a document at a time in this process: two at once
        # would each remove the other's version as the old one, and leave
        # the document with part of one or none.
        async with _writing((project_id, doc_id)):
            counting = _EMBEDDING.set(embedded)
            try:
                for start in range(0, len(chunks), size):
                    tried = start + size
                    await self._add(project_id, chunks[start:tried], ids[start:tried])
            except BaseException:
                if tried:
                    with contextlib.suppress(Exception):
                        await self._delete_ids(project_id, ids[:tried])
                raise
            finally:
                _EMBEDDING.reset(counting)
            if embedded is not None:
                embedded.landed = True

            async with gate(project_id).shared():
                replaced = await self._remove(
                    doc_id=doc_id,
                    project_id=project_id,
                    own=collection_name(self._embedding_model, project_id),
                    keeping=ids,
                )
        if replaced or not chunks:
            await tidy(project_id)

    async def _add(
        self, project_id: str, chunks: list[Document], ids: list[str]
    ) -> None:
        """Embed one slice of chunks, then write it, on indexing's own threads.

        Only the write is behind the project's gate: embedding takes as long as
        the provider does, and a rebuild or drop of the collection, and every
        search waiting behind one, must not wait for it.
        """
        texts = [chunk.page_content for chunk in chunks]
        vectors = await run_indexing(_embed, self._embedding_model, texts)
        # Billed now, whether or not the write below lands.
        embedding = _EMBEDDING.get()
        if embedding is not None:
            embedding.chunks += len(texts)
        async with gate(project_id).shared():
            store = await asyncio.to_thread(
                open_vector_store, self._embedding_model, project_id
            )
            await run_indexing(_write, store, chunks, ids, vectors)

    async def _delete_ids(self, project_id: str, ids: list[str]) -> None:
        """Remove chunks by id from the project's collection for this model."""
        async with gate(project_id).shared():
            store = await asyncio.to_thread(
                find_vector_store, self._embedding_model, project_id
            )
            if store is not None:
                await run_indexing(store._collection.delete, ids)

    async def delete(self, *, doc_id: str, project_id: str) -> int:
        """Remove every chunk of one project's document, and say how many;
        the room they took is given back (`tidy`)."""
        await ensure_moved(project_id)
        async with gate(project_id).shared():
            removed = await self._remove(doc_id=doc_id, project_id=project_id)
        if removed:
            await tidy(project_id)
        return removed

    async def delete_project(self, *, project_id: str) -> int:
        """Remove every chunk of a project, under every embedding model, and
        say how many: its collections are dropped, with their files."""
        await ensure_moved(project_id)
        async with gate(project_id).alone():
            return await asyncio.to_thread(_drop_project, project_id)

    async def _remove(
        self,
        *,
        doc_id: str,
        project_id: str,
        own: str | None = None,
        keeping: Collection[str] = (),
    ) -> int:
        """Remove one document's chunks, except `keeping` in the collection
        named `own`, and say how many went.

        Swept across every collection of the project, not only the one being
        written: a document may have been indexed under a different embedding
        model than the one now writing or deleting it, and a sweep that
        missed it would leave chunks that still answer queries with nothing
        left to name them.
        """
        kept = set(keeping)
        removed = 0

        # In a thread: listing the collections is a call into Chroma, and on a
        # full disk one can hang holding its lock (DISK-3).
        client = _chroma_client()
        for name in await asyncio.to_thread(_project_collections, client, project_id):
            collection = await asyncio.to_thread(
                client.get_collection, name, embedding_function=None
            )
            found = await asyncio.to_thread(
                collection.get, where=_owned_by(doc_id, project_id), include=[]
            )
            stale = [
                chunk_id
                for chunk_id in found["ids"]
                if name != own or chunk_id not in kept
            ]
            if stale:
                await run_indexing(collection.delete, stale)
                await asyncio.to_thread(_count_deleted, name, len(stale))
                removed += len(stale)

        return removed


def _drop_project(project_id: str) -> int:
    """Drop every collection of a project, and the files they leave; how many
    chunks they held. Blocks."""
    client = _chroma_client()
    removed = 0
    for name in _project_collections(client, project_id):
        removed += client.get_collection(name).count()
        _drop(client, name)
    sparse.invalidate(project_id)
    _sweep_orphans()
    return removed


def _drop(client: chromadb.ClientAPI, name: str) -> None:
    """Drop a collection, and any copy a rebuild of it left: kept, that copy
    would be taken for a rebuild to finish and bring back what was deleted.
    The copy first: stopped in between, what is left is the collection."""
    for gone in (f"{name}{_REBUILDING}", name):
        with contextlib.suppress(NotFoundError):
            client.delete_collection(gone)
    _forget(name)
