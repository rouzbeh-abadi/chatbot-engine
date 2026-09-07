"""Keyword search over a project's chunks, for the half of hybrid retrieval
that embeddings cannot do.

A vector search finds what is *about* the question. It is poor at exact terms:
a fare name, a booking code, a product number, a word that appears once. BM25
ranks by term match and is good at exactly those, which is why the two are
fused rather than either used alone.

The index is built from the chunks already in the vector store, so there is
one source of truth and nothing extra to persist. It is rebuilt per project
when the collection's chunk count changes or a short interval has passed, and
dropped at once when this process ingests or deletes. Another replica's
changes reach this one at the next rebuild, which is the cost of not
persisting a second index.

Tokenisation is `\\w+` on lowercased text: adequate for languages that separate
words with spaces, and not for ones that do not, where the vector half of the
search carries the query alone.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

#: How long an index is trusted before the collection is consulted again.
#: Long enough that a burst of questions rebuilds nothing; short enough that a
#: document ingested by another replica is searchable within a minute.
INDEX_TTL_S = 60.0

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class SparseIndex:
    """BM25 over one project's chunks, as they were when it was built."""

    documents: list[Document]
    bm25: BM25Okapi
    chunk_count: int
    built_at: float

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """The `k` best-matching chunks with their BM25 scores, best first.

        Chunks that match no query term score zero and are left out: an
        absent result is more useful to fusion than a tie at nothing.
        """
        terms = tokenize(query)
        if not terms or not self.documents:
            return []

        scores = self.bm25.get_scores(terms)
        ranked = sorted(
            ((self.documents[i], float(score)) for i, score in enumerate(scores)),
            key=lambda hit: hit[1],
            reverse=True,
        )
        return [(document, score) for document, score in ranked[:k] if score > 0]


_indexes: dict[tuple[str, str], SparseIndex] = {}
_lock = threading.Lock()


def sparse_index(store: Chroma, project_id: str) -> SparseIndex:
    """The project's index, rebuilt when the collection has visibly changed."""
    key = (store._collection.name, project_id)
    count = _chunk_count(store, project_id)
    now = time.monotonic()

    with _lock:
        cached = _indexes.get(key)
        fresh = (
            cached is not None
            and cached.chunk_count == count
            and now - cached.built_at < INDEX_TTL_S
        )
        if fresh:
            return cached

        index = _build(store, project_id, count, now)
        _indexes[key] = index
        return index


def invalidate(project_id: str | None = None) -> None:
    """Drop cached indexes, for one project or all. Called after an ingest or
    delete in this process, so the next search sees the change at once."""
    with _lock:
        for key in list(_indexes):
            if project_id is None or key[1] == project_id:
                del _indexes[key]


def _chunk_count(store: Chroma, project_id: str) -> int:
    return len(store.get(where={"project_id": project_id}, include=[])["ids"])


def _build(store: Chroma, project_id: str, count: int, now: float) -> SparseIndex:
    got = store.get(
        where={"project_id": project_id}, include=["documents", "metadatas"]
    )
    documents = [
        Document(page_content=text, metadata=dict(metadata or {}))
        for text, metadata in zip(got["documents"], got["metadatas"], strict=True)
    ]
    corpus = [tokenize(document.page_content) for document in documents] or [[""]]

    return SparseIndex(
        documents=documents,
        bm25=BM25Okapi(corpus),
        chunk_count=count,
        built_at=now,
    )
