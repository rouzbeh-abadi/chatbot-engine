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
kept, and no more than `ENGINE_KEYWORD_INDEX_MB` of them, the one searched
longest ago dropped first, so an engine serving many projects holds only the
ones in use. Searches that find a project's index cold while it is being
built wait for that build rather than start their own, and a project of more
than `MAX_INDEX_CHUNKS` chunks is searched by vector alone
(docs/review-2026-10.md, RETRIEVAL-1).

An index is BM25 as `rank_bm25.BM25Okapi` scores it, kept as arrays: for
each term (known by its hash), the chunks it occurs in and how often. A
little over twice its project's text in memory, where a dict of term counts
per chunk took eleven times.

Building an index reads and tokenises every chunk of a project, and a search
scores every one of them: both are blocking work, which a caller on the event
loop runs in a thread (agent/retriever.py does).

Tokenisation is `\\w+` on lowercased text: adequate for languages that separate
words with spaces, and not for ones that do not, where the vector half of the
search carries the query alone.
"""

from __future__ import annotations

import logging
import math
import re
import sys
import threading
import time
from array import array
from collections import Counter, OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

from chatbot_engine.settings import get_settings

logger = logging.getLogger(__name__)

#: How long an index is trusted before it is built again from the collection.
#: Long enough that a burst of questions rebuilds nothing; short enough that a
#: document ingested by another replica is searchable within a minute.
INDEX_TTL_S = 60.0

#: How many projects' indexes are kept at once, within the bytes
#: `ENGINE_KEYWORD_INDEX_MB` allows them all. An engine serving more projects
#: than this rebuilds the index of one that comes back after the others
#: pushed it out, which costs that search one read of the project's chunks.
MAX_INDEXES = 32

#: The most chunks a project may have and still get a keyword index: about a
#: hundred million characters at the default chunk size. A larger project is
#: searched by vector alone, as if its retrieval were `vector`.
MAX_INDEX_CHUNKS = 100_000

#: BM25's constants, as `rank_bm25.BM25Okapi` sets them.
_K1, _B, _EPSILON = 1.5, 0.75, 0.25

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass(eq=False)
class SparseIndex:
    """BM25 over one project's chunks, as they were when it was built.

    `terms` holds the hash of every term, sorted; the chunks a term occurs in
    are `chunks[starts[t]:starts[t + 1]]`, with its counts in them at the
    same places of `counts`.
    """

    texts: list[str]
    metadatas: list[dict[str, Any]]
    terms: np.ndarray
    starts: np.ndarray
    chunks: np.ndarray
    counts: np.ndarray
    idf: np.ndarray
    #: `k1 * (1 - b + b * length / average length)`, per chunk.
    norms: np.ndarray
    built_at: float
    #: False for a project over `MAX_INDEX_CHUNKS`: nothing indexed, and its
    #: searches go by vector alone.
    searchable: bool = True
    nbytes: int = field(init=False)

    def __post_init__(self) -> None:
        arrays = (
            self.terms,
            self.starts,
            self.chunks,
            self.counts,
            self.idf,
            self.norms,
        )
        # The texts as Python holds them (two or four bytes a character for
        # most scripts past Latin), a metadata dict and its pointers per
        # chunk, and the arrays.
        self.nbytes = (
            sum(sys.getsizeof(text) for text in self.texts)
            + 240 * len(self.texts)
            + sum(a.nbytes for a in arrays)
        )

    @property
    def documents(self) -> list[Document]:
        """Every chunk, as a Document, made when asked for."""
        return [self._document(i) for i in range(len(self.texts))]

    def _document(self, i: int) -> Document:
        return Document(page_content=self.texts[i], metadata=dict(self.metadatas[i]))

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """The `k` best-matching chunks with their BM25 scores, best first.

        Chunks that match no query term score zero and are left out: an
        absent result is more useful to fusion than a tie at nothing.
        """
        terms = tokenize(query)
        if not terms or not self.texts:
            return []

        scores = np.zeros(len(self.texts))
        for term in terms:
            key = hash(term)
            at = int(np.searchsorted(self.terms, key))
            if at == len(self.terms) or self.terms[at] != key:
                continue
            start, end = self.starts[at], self.starts[at + 1]
            chunks, counts = self.chunks[start:end], self.counts[start:end]
            scores[chunks] += (
                self.idf[at] * counts * (_K1 + 1) / (counts + self.norms[chunks])
            )

        ranked = np.argsort(-scores, kind="stable")[:k]
        return [(self._document(i), float(scores[i])) for i in ranked if scores[i] > 0]


#: The indexes kept, by collection and project, the one searched longest ago
#: first. A collection is known by its id rather than its name: a collection
#: made again under the same name (another `ENGINE_CHROMA_DIR`, a collection
#: dropped and re-created) holds other chunks, and must not meet an index
#: built from the old one.
_indexes: OrderedDict[tuple[str, str], SparseIndex] = OrderedDict()
#: The builds under way, so a search that finds an index cold while it is
#: being built waits for it instead of building a copy of its own.
_building: dict[tuple[str, str], Future[SparseIndex]] = {}
_lock = threading.Lock()
#: Counts the calls to `invalidate`. A build that began before one may hold
#: the chunks from before the change, so it answers the searches that asked
#: for it and is not kept.
_generation = 0


def sparse_index(store: Chroma, project_id: str) -> SparseIndex | None:
    """The project's index: the one built within `INDEX_TTL_S`, the one being
    built, or a new one. None for a project searched by vector alone.

    Blocking: a new index reads every chunk of the project. Run it off the
    event loop.
    """
    key = (str(store._collection.id), project_id)
    now = time.monotonic()

    with _lock:
        cached = _indexes.get(key)
        if cached is not None and now - cached.built_at < INDEX_TTL_S:
            _indexes.move_to_end(key)
            return cached if cached.searchable else None
        pending = _building.get(key)
        building = pending is None
        if pending is None:
            pending = _building[key] = Future()
        generation = _generation

    if not building:
        index = pending.result()
        return index if index.searchable else None

    # Built outside the lock, so a large project's build holds up no other
    # project's search.
    try:
        index = _build(store, project_id, now)
    except BaseException as exc:
        with _lock:
            if _building.get(key) is pending:
                del _building[key]
        pending.set_exception(exc)
        raise

    try:
        with _lock:
            if _building.get(key) is pending:
                del _building[key]
            if generation == _generation:
                _keep(key, index, now)
    finally:
        # Whatever keeping it did, the searches waiting for it get it.
        pending.set_result(index)

    return index if index.searchable else None


def _keep(key: tuple[str, str], index: SparseIndex, now: float) -> None:
    """Cache `index`, within `MAX_INDEXES` and the byte budget. Under `_lock`."""
    budget = get_settings().keyword_index_mb * 1024 * 1024
    if index.nbytes > budget:
        # Larger than every index may be together: it answers the searches
        # waiting for it, and the next one builds it again.
        _indexes.pop(key, None)
        return
    _indexes[key] = index
    _indexes.move_to_end(key)
    for old in [
        k for k, kept in _indexes.items() if now - kept.built_at >= INDEX_TTL_S
    ]:
        del _indexes[old]
    held = sum(kept.nbytes for kept in _indexes.values())
    while len(_indexes) > MAX_INDEXES or held > budget:
        _, dropped = _indexes.popitem(last=False)
        held -= dropped.nbytes


def invalidate(project_id: str | None = None) -> None:
    """Drop cached indexes, for one project or all. Called after an ingest or
    delete in this process, so the next search sees the change at once. A
    build under way is let finish for the searches waiting on it; the next
    search starts another."""
    global _generation

    with _lock:
        _generation += 1
        for cache in (_indexes, _building):
            for key in list(cache):
                if project_id is None or key[1] == project_id:
                    del cache[key]


def _build(store: Chroma, project_id: str, now: float) -> SparseIndex:
    where = {"project_id": project_id}
    count = len(store.get(where=where, include=[])["ids"])
    if count > MAX_INDEX_CHUNKS:
        logger.warning(
            "project %s has %d chunks, more than the %d a keyword index takes; "
            "it is searched by vector alone",
            project_id,
            count,
            MAX_INDEX_CHUNKS,
        )
        return _index([], [], array("q"), array("i"), array("f"), [], now, False)

    got = store.get(where=where, include=["documents", "metadatas"])
    texts: list[str] = list(got["documents"])
    # Chunks of one document share most of their metadata; one copy of each
    # value, not one per chunk.
    shared: dict[Any, Any] = {}

    def once(value: Any) -> Any:
        return shared.setdefault(value, value) if isinstance(value, str) else value

    metadatas = [
        {once(name): once(value) for name, value in (metadata or {}).items()}
        for metadata in got["metadatas"]
    ]

    hashes, chunks, counts = array("q"), array("i"), array("f")
    lengths: list[int] = []
    for number, text in enumerate(texts):
        terms = Counter(tokenize(text))
        lengths.append(sum(terms.values()))
        for term, times in terms.items():
            hashes.append(hash(term))
            chunks.append(number)
            counts.append(times)

    return _index(texts, metadatas, hashes, chunks, counts, lengths, now, True)


def _index(
    texts: list[str],
    metadatas: list[dict[str, Any]],
    hashes: array,
    chunks: array,
    counts: array,
    lengths: list[int],
    now: float,
    searchable: bool,
) -> SparseIndex:
    """The arrays of an index, from one (term hash, chunk, count) per term of
    each chunk; idf and lengths as `rank_bm25.BM25Okapi` takes them."""
    order = np.argsort(np.frombuffer(hashes, dtype=np.int64), kind="stable")
    by_term = np.frombuffer(hashes, dtype=np.int64)[order]
    terms, starts = np.unique(by_term, return_index=True)
    starts = np.append(starts, len(by_term)).astype(np.int64)

    total = len(texts)
    holding = np.diff(starts)
    idf = np.log(total - holding + 0.5) - np.log(holding + 0.5)
    if len(idf):
        # A term in more than half the chunks would score below zero; it
        # gets a small share of the average instead.
        average = math.fsum(idf.tolist()) / len(idf)
        idf = np.where(idf < 0, _EPSILON * average, idf)

    length = np.asarray(lengths, dtype=np.float64)
    average_length = length.sum() / total if total else 1.0
    norms = _K1 * (1 - _B + _B * length / (average_length or 1.0))

    return SparseIndex(
        texts=texts,
        metadatas=metadatas,
        terms=terms,
        starts=starts,
        chunks=np.frombuffer(chunks, dtype=np.int32)[order],
        counts=np.frombuffer(counts, dtype=np.float32)[order],
        idf=idf,
        norms=norms,
        built_at=now,
        searchable=searchable,
    )
