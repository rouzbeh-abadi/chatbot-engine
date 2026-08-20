"""Chroma, keyed so re-uploading a document replaces its chunks.

Embedded Chroma: no server, just files under `ENGINE_CHROMA_DIR`. One client per
process, because several clients on one directory contend for the same SQLite file.
"""

from __future__ import annotations

from functools import lru_cache

from chatbot_engine.rag.embeddings import get_embeddings
from chatbot_engine.settings import get_settings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class EmptyVectorStoreError(RuntimeError):
    """The store holds no documents, so retrieval cannot return anything."""


@lru_cache
def open_vector_store() -> Chroma:
    """Open the store for reading or writing, creating it if absent."""
    settings = get_settings()

    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def load_vector_store() -> Chroma:
    """Open an already populated store, for querying.

    Raises if nothing has been ingested, so an empty database fails loudly
    instead of silently retrieving zero chunks and letting the model invent
    an answer from nothing.
    """
    store = open_vector_store()

    if count_chunks(store) == 0:
        raise EmptyVectorStoreError(
            f"no documents indexed in {get_settings().chroma_dir} -- upload some "
            "through PUT /documents, or run `make seed` to load the example "
            "knowledge base"
        )

    return store


def count_chunks(store: Chroma | None = None) -> int:
    """How many chunks are stored, across every project."""
    store = store or open_vector_store()

    return len(store.get(include=[])["ids"])


def reset_collection() -> None:
    """Delete every chunk. For tests and for a forced re-index."""
    open_vector_store().reset_collection()


class ChromaChunkStore:
    """The write side: one document's chunks, replaced as a unit."""

    def __init__(self, store: Chroma | None = None) -> None:
        self._store = store or open_vector_store()

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
        """Remove every chunk belonging to one document, and say how many."""
        existing = self._store.get(where={"doc_id": doc_id}, include=[])["ids"]

        if existing:
            await self._store.adelete(ids=existing)

        return len(existing)
