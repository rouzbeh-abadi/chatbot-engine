"""Extract, chunk, embed, record.

With a `ChromaChunkStore` the chunks are written and the document comes out
`indexed`. Without one it is chunked and counted but not searchable, so it comes
out `received` -- which is what `/health/ready` and the UI then report.

`DocumentBlobs` keeps the uploaded bytes, which is what lets `reindex` rebuild a
document after a change to chunk size or embedding model, with no re-upload.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime

from chatbot_engine.documents.blobs import DocumentBlobs
from chatbot_engine.documents.extractor import DocumentExtractor, select_extractor
from chatbot_engine.errors import DocumentRejectedError, NotConfiguredError
from chatbot_engine.models.documents import DocumentRecord, IngestStatus
from chatbot_engine.ports.documents import DocumentRegistry
from chatbot_engine.rag.splitter import DocumentChunker
from chatbot_engine.rag.vector_store import ChromaChunkStore
from langchain_core.documents import Document


def doc_id_for(project_id: str, external_id: str) -> str:
    """A stable id, so re-uploading a file overwrites rather than duplicates."""
    # The NUL separator stops ("a", "bc") from colliding with ("ab", "c").
    seed = f"{project_id}\x00{external_id}".encode()

    return hashlib.sha256(seed).hexdigest()[:32]


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
    ) -> DocumentRecord:
        """Ingest one document and report what happened to it.

        Identical bytes answer `unchanged` and do no work, which is what makes
        `make seed` cheap to re-run once embedding costs money per chunk.

        Raises:
            UnsupportedDocumentTypeError: No extractor handles `mimetype`.
            DocumentRejectedError: The document yields no text.
        """
        # First, so a file the engine cannot read leaves no trace in the registry.
        extractor = select_extractor(mimetype)

        doc_id = doc_id_for(project_id, external_id)
        content_hash = hashlib.sha256(data).hexdigest()
        current = await self._registry.get(project_id=project_id, doc_id=doc_id)

        if (
            current is not None
            and current.content_hash == content_hash
            and current.status is not IngestStatus.FAILED
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
        )

        return await self._index(record, extractor, data, keep_original=True)

    async def reindex(self, *, project_id: str, doc_id: str) -> DocumentRecord:
        """Re-chunk and re-embed one document from the bytes already stored.

        What the blob store is for: after changing `chunk_size` or the embedding
        model, every document has to be rebuilt, and asking the backend to upload
        them all again is the wrong way to do it.

        Raises:
            NotConfiguredError: No blob store, so the original was never kept.
            LookupError: No such document in the registry.
        """
        if self._blobs is None:
            raise NotConfiguredError(
                "no BlobStore is registered -- the original bytes were never "
                "kept, so re-index the document by uploading it again"
            )

        record = await self._registry.get(project_id=project_id, doc_id=doc_id)
        if record is None:
            raise LookupError(f"no document {doc_id!r} in project {project_id!r}")

        data = await self._blobs.read(doc_id=doc_id)
        extractor = select_extractor(record.mimetype)

        return await self._index(record, extractor, data, keep_original=False)

    async def _index(
        self,
        record: DocumentRecord,
        extractor: DocumentExtractor,
        data: bytes,
        *,
        keep_original: bool,
    ) -> DocumentRecord:
        """Store, split, embed, record. Shared by `ingest` and `reindex`."""
        try:
            # The original first: if chunking or embedding fails, the bytes are
            # still there to retry from.
            if keep_original and self._blobs is not None:
                await self._blobs.write(
                    doc_id=record.doc_id, data=data, mimetype=record.mimetype
                )

            chunks = await asyncio.to_thread(self._split, extractor, data, record)

            if self._vectors is not None:
                # Inside the try on purpose: a rate limit or a bad key here must
                # leave a `failed` record, not a document that vanishes.
                await self._vectors.write(doc_id=record.doc_id, chunks=chunks)
        except Exception as exc:
            # Keep the failure in GET /documents, not only in a log line.
            await self._registry.upsert(
                record.model_copy(
                    update={"status": IngestStatus.FAILED, "error": str(exc)}
                )
            )
            raise

        return await self._registry.upsert(
            record.model_copy(
                update={
                    "chunk_count": len(chunks),
                    "error": None,
                    # `indexed` is earned by the vectors landing, not claimed.
                    "status": (
                        IngestStatus.INDEXED
                        if self._vectors is not None
                        else IngestStatus.RECEIVED
                    ),
                }
            )
        )

    def _split(
        self,
        extractor: DocumentExtractor,
        data: bytes,
        record: DocumentRecord,
    ) -> list[Document]:
        """Extract the text and split it, carrying the document's identity along.

        Both steps are synchronous and CPU-bound, so `ingest` runs this off the
        event loop: a slow PDF would otherwise stall every request in flight.
        """
        extracted = extractor.extract_text(data=data, mimetype=record.mimetype)

        if not extracted.text.strip():
            # Usually a scanned PDF. Recording zero chunks would let it look
            # ingested while being invisible to every search.
            raise DocumentRejectedError(
                f"no text could be extracted from {record.filename!r} -- "
                "a scanned document needs OCR before it can be indexed"
            )

        # This metadata is copied onto every chunk, and is what a citation is
        # built from later. The chunker adds `start_index` on top.
        return self._chunker.chunk(
            [
                Document(
                    page_content=extracted.text,
                    metadata={
                        "doc_id": record.doc_id,
                        "project_id": record.project_id,
                        "source": record.external_id,
                        "filename": record.filename,
                    },
                )
            ]
        )
