"""Keyword search over a project's chunks, for the half of hybrid retrieval
that embeddings cannot do.

A vector search finds what is *about* the question. It is poor at exact terms:
a fare name, a booking code, a product number, a word that appears once. BM25
ranks by term match and is good at exactly those, which is why the two are
fused rather than either used alone.

The index is built from the chunks already in the vector store, so there is
one source of truth and nothing extra to persist. It is kept per project for
`INDEX_TTL_S` and rebuilt after that, and dropped at once when this process
ingests or deletes. Until then a search consults nothing but the index, not
the collection. Another replica's changes reach this one at the next rebuild,
which is the cost of not persisting a second index. At most `MAX_INDEXES` are
kept, the one searched longest ago dropped first, so an engine serving many
projects holds only the ones in use.

Building an index reads and tokenises every chunk of a project, and a search
scores every one of them: both are blocking work, which a caller on the event
loop runs in a thread (agent/retriever.py does).

Tokenisation is `\\w+` on lowercased text: adequate for languages that separate
words with spaces, and not for ones that do not, where the vector half of the
search carries the query alone.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

#: How long an index is trusted before it is built again from the collection.
#: Long enough that a burst of questions rebuilds nothing; short enough that a
#: document ingested by another replica is searchable within a minute.
INDEX_TTL_S = 60.0

#: How many projects' indexes are kept at once. Each holds its project's
#: chunk texts and their term counts; an engine serving more projects than
#: this rebuilds the index of one that comes back after the others pushed it
#: out, which costs that search one read of the project's chunks.
MAX_INDEXES = 32

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class SparseIndex:
    """BM25 over one project's chunks, as they were when it was built."""

    documents: list[Document]
    bm25: BM25Okapi
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


#: The indexes kept, by collection and project, the one searched longest ago
#: first. A collection is known by its id rather than its name: a collection
#: made again under the same name (another `ENGINE_CHROMA_DIR`, a collection
#: dropped and re-created) holds other chunks, and must not meet an index
#: built from the old one.
_indexes: OrderedDict[tuple[str, str], SparseIndex] = OrderedDict()
_lock = threading.Lock()
#: Counts the calls to `invalidate`. A build that began before one may hold
#: the chunks from before the change, so it answers the search that asked for
#: it and is not kept.
_generation = 0


def sparse_index(store: Chroma, project_id: str) -> SparseIndex:
    """The project's index: the one built within `INDEX_TTL_S`, or a new one.

    Blocking: a new index reads every chunk of the project. Run it off the
    event loop.
    """
    key = (str(store._collection.id), project_id)
    now = time.monotonic()

    with _lock:
        cached = _indexes.get(key)
        if cached is not None and now - cached.built_at < INDEX_TTL_S:
            _indexes.move_to_end(key)
            return cached
        generation = _generation

    # Built outside the lock, so a large project's build holds up no other
    # project's search.
    index = _build(store, project_id, now)

    with _lock:
        if generation == _generation:
            _indexes[key] = index
            _indexes.move_to_end(key)
            while len(_indexes) > MAX_INDEXES:
                _indexes.popitem(last=False)

    return index


def invalidate(project_id: str | None = None) -> None:
    """Drop cached indexes, for one project or all. Called after an ingest or
    delete in this process, so the next search sees the change at once."""
    global _generation

    with _lock:
        _generation += 1
        for key in list(_indexes):
            if project_id is None or key[1] == project_id:
                del _indexes[key]


def _build(store: Chroma, project_id: str, now: float) -> SparseIndex:
    got = store.get(
        where={"project_id": project_id}, include=["documents", "metadatas"]
    )
    documents = [
        Document(page_content=text, metadata=dict(metadata or {}))
        for text, metadata in zip(got["documents"], got["metadatas"], strict=True)
    ]
    corpus = [tokenize(document.page_content) for document in documents] or [[""]]

    return SparseIndex(documents=documents, bm25=BM25Okapi(corpus), built_at=now)
