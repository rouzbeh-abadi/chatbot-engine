"""Document ingestion pipeline.

Uploaded files are extracted, split into chunks, optionally embedded into the
vector store, and recorded in the document registry.

When a `ChromaChunkStore` is configured, chunks are persisted for retrieval and
the document is marked `indexed`. Without one, which is the case when no model
provider key is set, the document is still validated, chunked and counted, but
is not searchable and is marked `received`; `GET /documents` and the UI show
that status.

`DocumentBlobs` preserves the original upload bytes so `reindex` can rebuild the
document after changes to chunking or embedding configuration without requiring
another upload.

A document is read within bounds, as a file sent in a chat is: a PDF in a
process of its own that is killed at a deadline (documents/bounded.py), and no
document past the most text the index takes from one file. A file that cannot
be read within them is a `failed` record that says why.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TypedDict

from langchain_core.documents import Document

from chatbot_engine.disk import ensure_room
from chatbot_engine.documents.blobs import DocumentBlobs
from chatbot_engine.documents.bounded import (
    INFLATE_MAX_BYTES,
    ReadFailed,
    ReadTimeoutError,
    read_bounded,
)
from chatbot_engine.documents.extractor import (
    DocumentExtractor,
    TextDocumentExtractor,
    select_extractor,
)
from chatbot_engine.documents.models import ExtractedDocument
from chatbot_engine.errors import DocumentRejectedError, NotConfiguredError
from chatbot_engine.models.documents import DocumentRecord, IngestStatus
from chatbot_engine.ports.documents import DocumentRegistry
from chatbot_engine.rag import sparse
from chatbot_engine.rag.embeddings import resolve_embedding_model
from chatbot_engine.rag.splitter import ChunkStrategy, DocumentChunker
from chatbot_engine.rag.vector_store import ChromaChunkStore
from chatbot_engine.rag.workers import KeyedLocks, run_indexing
from chatbot_engine.settings import get_settings
from chatbot_engine.untrusted import instruction_warnings

#: The most of a reader's own words a failed record repeats.
_REASON_CHARS = 200


def doc_id_for(project_id: str, external_id: str) -> str:
    """A stable id, so re-uploading a file overwrites rather than duplicates."""
    # The NUL separator stops ("a", "bc") from colliding with ("ab", "c").
    seed = f"{project_id}\x00{external_id}".encode()

    return hashlib.sha256(seed).hexdigest()[:32]


class _Chunking(TypedDict):
    chunking_strategy: ChunkStrategy
    chunk_size: int
    chunk_overlap: int


def _settings_of(chunker: DocumentChunker) -> _Chunking:
    """The chunking a record stores: what the chunker actually uses."""
    return {
        "chunking_strategy": chunker.strategy,
        "chunk_size": chunker.size,
        "chunk_overlap": chunker.overlap,
    }


def _cut_the_same(
    record: DocumentRecord, chunker: DocumentChunker, *, requested: bool
) -> bool:
    """Whether `record` was chunked the way `chunker` would chunk it.

    A record from before chunking was recorded says nothing either way; it
    counts as current unless the caller asks for chunking explicitly, so an
    engine upgrade alone never re-embeds a knowledge base.
    """
    if record.chunking_strategy is None:
        return not requested
    return (record.chunking_strategy, record.chunk_size, record.chunk_overlap) == (
        chunker.strategy,
        chunker.size,
        chunker.overlap,
    )


def _embedded_the_same(
    record: DocumentRecord, model: str, *, requested: bool, embeds: bool
) -> bool:
    """Whether `record`'s vectors are the ones `model` would make.

    Vectors from two models are not comparable, so identical bytes sent with
    another model must be embedded again: answered `unchanged`, they would
    stay in the old model's collection, where the new model's queries never
    look. An engine that cannot embed has nothing to compare; a document
    recorded while it could not (`received`) is out of date once it can. A
    record from before the model was recorded says nothing either way, and
    counts as current unless the caller names a model, as with chunking.
    """
    if not embeds:
        return True
    if record.status is IngestStatus.RECEIVED:
        return False
    if record.embedding_model is None:
        return not requested
    return record.embedding_model == model


def _read(
    extractor: DocumentExtractor, data: bytes, record: DocumentRecord
) -> ExtractedDocument:
    """The document's text, read within the index's bounds.

    A PDF is read in a process of its own that is killed at
    `ENGINE_INDEX_READ_TIMEOUT_S`, with pypdf's inflation caps lowered, as a file
    sent in a chat is: pypdf is pure Python, and a PDF made to take minutes or
    to inflate into gigabytes would otherwise hold a worker thread and the
    engine's memory for as long as it liked. Plain text and Markdown are only
    decoded, which no file can make slow or large past the upload limit, so
    they are read where they are. Either way, no more than
    `ENGINE_INDEX_MAX_CHARS` characters.

    Raises:
        DocumentRejectedError: The file cannot be read within those bounds;
            the message says why, in words its owner can act on.
    """
    name = repr(record.filename)
    settings = get_settings()
    max_chars, timeout_s = settings.index_max_chars, settings.index_read_timeout_s
    try:
        if isinstance(extractor, TextDocumentExtractor):
            extracted = extractor.extract_text(
                data=data, mimetype=record.mimetype, max_chars=max_chars
            )
        else:
            extracted = read_bounded(
                data,
                record.mimetype,
                max_chars=max_chars,
                timeout_s=timeout_s,
            )
    except UnicodeDecodeError as exc:
        raise DocumentRejectedError(
            f"{name} is not UTF-8 text -- save it as UTF-8 and upload it again"
        ) from exc
    except ReadTimeoutError as exc:
        raise DocumentRejectedError(
            f"{name} took longer than {timeout_s:g} seconds to read "
            "-- split it into smaller files"
        ) from exc
    except ReadFailed as exc:
        raise DocumentRejectedError(f"{name} could not be read: {_why(exc)}") from exc
    except DocumentRejectedError as exc:
        # The reading process ended without a word: the system killed it,
        # usually for the memory it took.
        raise DocumentRejectedError(
            f"{name} could not be read: the reading stopped without a reason, "
            "usually for want of memory"
        ) from exc

    if extracted.truncated:
        raise DocumentRejectedError(
            f"{name} has more text than one document may give the index "
            f"({max_chars:,} characters) -- split it into smaller files"
        )
    return extracted


def _why(failure: ReadFailed) -> str:
    """Why the reading process gave up, in words; the reader's own for a
    damaged or encrypted PDF ("EOF marker not found")."""
    if failure.kind == "LimitReachedError":
        megabytes = INFLATE_MAX_BYTES // (1024 * 1024)
        return f"a compressed part of it inflates past {megabytes} MB"
    if failure.kind == "MemoryError":
        return "reading it takes more memory than the engine allows"
    said = " ".join(failure.message.split()) or failure.kind
    return said[:_REASON_CHARS]


#: One ingest or re-index of a document at a time in this process.
_indexing = KeyedLocks()


class DocumentIngestPipeline:
    """Turns uploaded bytes into chunks, and remembers what happened."""

    def __init__(
        self,
        *,
        registry: DocumentRegistry,
        chunker: DocumentChunker,
        vectors: ChromaChunkStore | None = None,
        blobs: DocumentBlobs | None = None,
    ) -> None:
        self._registry = registry
        self._chunker = chunker
        #: Without one, documents are chunked but not searchable, and come out
        #: `received` rather than `indexed`.
        self._vectors = vectors
        #: Without one, `reindex` is impossible: the original bytes are not kept.
        self._blobs = blobs

    async def ingest(
        self,
        *,
        project_id: str,
        external_id: str,
        filename: str,
        mimetype: str,
        data: bytes,
        embedding_model: str | None = None,
        chunking_strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> DocumentRecord:
        """Ingest one document and report what happened to it.

        Identical bytes answer `unchanged` and do no work, which is what makes
        `make seed` cheap to re-run once embedding costs money per chunk; not
        when they are to be cut or embedded differently from last time.

        Raises:
            UnsupportedDocumentTypeError: No extractor handles `mimetype`.
            DocumentRejectedError: The document yields no text, or cannot be
                read within the index's bounds.
        """
        # First, so a file the engine cannot read leaves no trace in the registry.
        extractor = select_extractor(mimetype)

        doc_id = doc_id_for(project_id, external_id)
        content_hash = hashlib.sha256(data).hexdigest()
        current = await self._registry.get(project_id=project_id, doc_id=doc_id)

        # Only when the request actually specifies chunking. Otherwise the
        # chunker this pipeline was wired with stands, which is what keeps it
        # injectable. Built before the unchanged check: settings that cannot
        # work together are refused even for bytes already indexed.
        requested = (chunking_strategy, chunk_size, chunk_overlap) != (None, None, None)
        chunker = (
            DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=chunking_strategy,
            )
            if requested
            else None
        )
        effective = chunker or self._chunker
        model = resolve_embedding_model(embedding_model)

        if (
            current is not None
            and current.content_hash == content_hash
            and current.status is not IngestStatus.FAILED
            and _cut_the_same(current, effective, requested=requested)
            and _embedded_the_same(
                current,
                model,
                requested=bool(embedding_model),
                embeds=self._vectors is not None,
            )
        ):
            # `unchanged` describes this call, not the document, so the stored
            # record keeps the status it earned last time.
            return current.model_copy(update={"status": IngestStatus.UNCHANGED})

        now = datetime.now(UTC)
        record = DocumentRecord(
            doc_id=doc_id,
            external_id=external_id,
            project_id=project_id,
            filename=filename,
            mimetype=mimetype,
            size_bytes=len(data),
            content_hash=content_hash,
            status=IngestStatus.RECEIVED,
            created_at=current.created_at if current is not None else now,
            updated_at=now,
            **_settings_of(effective),
        )

        return await self._index(
            record,
            extractor,
            data,
            keep_original=True,
            embedding_model=model,
            chunker=chunker,
        )

    async def reindex(
        self,
        *,
        project_id: str,
        doc_id: str,
        embedding_model: str | None = None,
        chunking_strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> DocumentRecord:
        """Re-chunk and re-embed one document from the bytes already stored.

        What the blob store is for: after changing `chunk_size` or the embedding
        model, every document has to be rebuilt, and asking the backend to upload
        them all again is the wrong way to do it.

        Settings work as on upload: none given, the chunker this pipeline was
        wired with (the engine's current defaults) applies; any given, the
        ones left out take the engine's defaults.

        Raises:
            NotConfiguredError: No blob store, so the original was never kept.
            LookupError: No such document in the registry.
            ChunkingError: The settings cannot work together.
        """
        if self._blobs is None:
            raise NotConfiguredError(
                "no BlobStore is registered -- the original bytes were never "
                "kept, so re-index the document by uploading it again"
            )

        # First, and scoped by project: the stored file is kept by `doc_id`
        # alone, so only the project's own record may lead to it.
        record = await self._registry.get(project_id=project_id, doc_id=doc_id)
        if record is None:
            raise LookupError(f"no document {doc_id!r} in project {project_id!r}")

        requested = (chunking_strategy, chunk_size, chunk_overlap) != (None, None, None)
        chunker = (
            DocumentChunker(
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if requested
            else self._chunker
        )
        data = await self._blobs.read(doc_id=doc_id)
        extractor = select_extractor(record.mimetype)
        rebuilt = record.model_copy(
            update={
                "status": IngestStatus.RECEIVED,
                "updated_at": datetime.now(UTC),
                **_settings_of(chunker),
            }
        )

        return await self._index(
            rebuilt,
            extractor,
            data,
            keep_original=False,
            embedding_model=resolve_embedding_model(embedding_model),
            chunker=chunker,
        )

    async def _index(
        self,
        record: DocumentRecord,
        extractor: DocumentExtractor,
        data: bytes,
        *,
        keep_original: bool,
        embedding_model: str,
        chunker: DocumentChunker | None = None,
    ) -> DocumentRecord:
        """Store, split, embed, record. Shared by `ingest` and `reindex`.

        A failure anywhere leaves a `failed` record that says why, and the
        version indexed before it, if any, still answering: the store embeds a
        new version before it removes the old one.

        One document at a time in this process: two uploads of it at once
        would otherwise leave its stored original from one and its chunks
        and record from the other (docs/review-2026-10.md, INGEST-3).
        """
        async with _indexing((record.project_id, record.doc_id)):
            return await self._index_one(
                record,
                extractor,
                data,
                keep_original=keep_original,
                embedding_model=embedding_model,
                chunker=chunker,
            )

    async def _index_one(
        self,
        record: DocumentRecord,
        extractor: DocumentExtractor,
        data: bytes,
        *,
        keep_original: bool,
        embedding_model: str,
        chunker: DocumentChunker | None,
    ) -> DocumentRecord:
        # Before anything is written, the record included: a full volume can
        # hang embedded Chroma for every tenant (DISK-3), and a version
        # already indexed goes on answering, so its record stays as it is.
        ensure_room()
        try:
            # The original first: if chunking or embedding fails, the bytes are
            # still there to retry from.
            if keep_original and self._blobs is not None:
                await self._blobs.write(
                    doc_id=record.doc_id, data=data, mimetype=record.mimetype
                )

            chunks, warnings = await run_indexing(
                self._split, extractor, data, record, chunker or self._chunker
            )

            if self._vectors is not None:
                # `self._vectors` is only the "embeddings are available" signal;
                # the store to write to depends on the embedding model, which
                # arrives with the request. Inside the try on purpose: a rate
                # limit or a bad key must leave a `failed` record, not a document
                # that vanishes.
                store = ChromaChunkStore(embedding_model)
                await store.write(
                    doc_id=record.doc_id, project_id=record.project_id, chunks=chunks
                )
                # The keyword index is built from the collection; this process
                # must not keep searching the version from before this write.
                sparse.invalidate(record.project_id)
        except Exception as exc:
            # Keep the failure in GET /documents, not only in a log line.
            await self._registry.upsert(
                record.model_copy(
                    update={"status": IngestStatus.FAILED, "error": str(exc)}
                )
            )
            raise

        embedded = self._vectors is not None
        return await self._registry.upsert(
            record.model_copy(
                update={
                    "chunk_count": len(chunks),
                    "error": None,
                    "warnings": warnings,
                    # `indexed` is earned by the vectors landing, not claimed.
                    "status": IngestStatus.INDEXED
                    if embedded
                    else IngestStatus.RECEIVED,
                    # What made the vectors, so a change of model is noticed.
                    "embedding_model": embedding_model if embedded else None,
                }
            )
        )

    def _split(
        self,
        extractor: DocumentExtractor,
        data: bytes,
        record: DocumentRecord,
        chunker: DocumentChunker | None = None,
    ) -> tuple[list[Document], list[str]]:
        """Read the text and split it, carrying the document's identity along,
        and say what in the text reads like orders to an AI (`warnings`).

        Both steps block, a PDF's reading on a process of its own, so `_index`
        runs this off the event loop: a slow file would otherwise stall every
        request in flight.
        """
        extracted = _read(extractor, data, record)

        if not extracted.text.strip():
            # Usually a scanned PDF. Recording zero chunks would let it look
            # ingested while being invisible to every search.
            raise DocumentRejectedError(
                f"no text could be extracted from {record.filename!r} -- "
                "a scanned document needs OCR before it can be indexed"
            )

        # This metadata is copied onto every chunk, and is what a citation is
        # built from later. The chunker adds `start_index`, and -- depending on
        # the strategy -- the page number or the heading trail.
        chunks = (chunker or self._chunker).chunk(
            extracted,
            {
                "doc_id": record.doc_id,
                "project_id": record.project_id,
                "source": record.external_id,
                "filename": record.filename,
            },
        )
        # Before anything is embedded, so a refused document costs nothing.
        max_chunks = get_settings().index_max_chunks
        if len(chunks) > max_chunks:
            raise DocumentRejectedError(
                f"{record.filename!r} makes {len(chunks):,} chunks, more than "
                f"the {max_chunks:,} one document may give the index -- split "
                "it into smaller files, or use larger chunks"
            )
        return chunks, instruction_warnings(extracted.text)
