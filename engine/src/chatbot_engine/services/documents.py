"""Document service: the boundary between the HTTP layer and the RAG pipeline.

Holds no logic of its own. Ingestion is the pipeline's job; listing and deletion
touch the registry, the vector store and the blob store together, which is the
one piece of coordination that lives here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from chatbot_engine.documents.blobs import DocumentBlobs
from chatbot_engine.models.documents import (
    ChunkStrategy,
    DocumentRecord,
    IngestStatus,
    PurgedDocuments,
)
from chatbot_engine.ports.documents import DocumentRegistry, IngestPipeline
from chatbot_engine.rag import sparse
from chatbot_engine.rag.pipeline import cancel_uploads, committing, mark_purged
from chatbot_engine.rag.vector_store import ChromaChunkStore
from chatbot_engine.settings import get_settings

logger = logging.getLogger(__name__)

#: What the record of a document being deleted says until the delete is done:
#: all it ever says is when the delete stopped part-way.
_DELETING = (
    "a delete of this document did not finish -- delete it again, or upload "
    "it again to index it"
)

#: How many stored originals are looked up in the registry at a time.
_SWEEP_BATCH = 500


class DocumentService:
    def __init__(
        self,
        *,
        pipeline: IngestPipeline,
        registry: DocumentRegistry,
        vectors: ChromaChunkStore | None = None,
        blobs: DocumentBlobs | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._registry = registry
        #: None when there is no provider key: documents are then recorded and
        #: chunked but not embedded, and come out `received` rather than
        #: `indexed`.
        self._vectors = vectors
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
        return await self._pipeline.ingest(
            project_id=project_id,
            external_id=external_id,
            filename=filename,
            mimetype=mimetype,
            data=data,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
        return await self._pipeline.reindex(
            project_id=project_id,
            doc_id=doc_id,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def list(self, *, project_id: str) -> Sequence[DocumentRecord]:
        return await self._registry.list(project_id=project_id)

    async def delete(self, *, project_id: str, doc_id: str) -> bool:
        """Remove a document: chunks, then the file, then the record.

        Only a document the project has. The record is looked up first, and
        without one nothing is touched and the answer is False: a `doc_id` is
        derived from the project and the caller's name for the file, so it can
        be worked out by anyone who knows the name, and the record is what
        says the document belongs to this project. The chunks are then removed
        by project and document together, and the stored file, which is kept
        by `doc_id` alone, only once the record has vouched for it.

        The record last: it is the only thing that knows the document existed, so
        losing it first would leave vectors and a file nothing can find. It
        is marked `failed` before anything goes, though: a delete that stops
        part-way (an object store that does not answer) leaves a record that
        says so, and an upload of the same file indexes it again, rather than
        an `indexed` record over chunks already gone, which an upload would
        answer `unchanged` and leave unsearchable.

        An upload of the document under way is cancelled, and removes what it
        wrote once it gets to its end, instead of writing its record: a
        delete is never undone by an upload that was already past it
        (docs/review-2026-10.md, INGEST-4). The delete answers at once.
        """
        return await self._delete(project_id=project_id, doc_id=doc_id, tidy=True)

    async def _delete(self, *, project_id: str, doc_id: str, tidy: bool) -> bool:
        """`delete`; `tidy` false for a purge, which drops the project's
        collections whole after, so none is rebuilt on the way."""
        async with committing(project_id, doc_id):
            cancelled = cancel_uploads(project_id, doc_id)
            record = await self._registry.get(project_id=project_id, doc_id=doc_id)
            if record is None:
                return cancelled > 0

            if self._vectors is not None or self._blobs is not None:
                await self._registry.upsert(
                    record.model_copy(
                        update={
                            "status": IngestStatus.FAILED,
                            "error": _DELETING,
                            # A failed record's count is what was embedded
                            # before it failed: nothing, for a delete.
                            "chunk_count": 0,
                        }
                    )
                )
            if self._vectors is not None:
                await self._vectors.delete(
                    doc_id=doc_id, project_id=project_id, tidy_after=tidy
                )
                sparse.invalidate(project_id)
            if self._blobs is not None:
                await self._blobs.delete(doc_id=doc_id)

            return await self._registry.delete(project_id=project_id, doc_id=doc_id)

    async def purge(self, *, project_id: str) -> PurgedDocuments:
        """Remove everything a project has: every document, with its chunks
        and stored original, any chunk of it no record names, and the uploads
        under way, which keep nothing. Uploads to the project are refused for
        a while after (`PURGED_FOR_S`), so a crawl still running when its
        chatbot was deleted cannot put pages back (X-9).

        No document's delete tidies the project's collections: they are
        dropped whole at the end, and a rebuild of one on the way would copy
        what is about to go. Stored originals no record names are removed
        too, of any project (`_sweep_originals`)."""
        mark_purged(project_id)
        cancelled = cancel_uploads(project_id)
        documents = 0
        for record in await self._registry.list(project_id=project_id):
            if await self._delete(
                project_id=project_id, doc_id=record.doc_id, tidy=False
            ):
                documents += 1
        chunks = 0
        if self._vectors is not None:
            # Chunks no record names: left by a write the volume cut short, or
            # by an engine older than the record written first.
            chunks = await self._vectors.delete_project(project_id=project_id)
            sparse.invalidate(project_id)
        if self._blobs is not None:
            await self._sweep_originals()
        return PurgedDocuments(
            documents=documents, orphaned_chunks=chunks, uploads_cancelled=cancelled
        )

    async def _sweep_originals(self) -> int:
        """Delete the stored originals no record names, of any project; how
        many went.

        Left by engines before 0.1.28, which stored a first upload's original
        before writing its record: one stopped in between kept a file that no
        delete could reach (docs/review-2026-10.md, INGEST-16). It cannot be
        told which project such a file was for, its key being the document's
        id alone, so a purge takes every one. Since 0.1.28 a record is written
        before its original, so an original no record names is never an
        upload's under way: the originals are listed first and the records
        looked up after, and every original listed had its record by then.

        Not with originals in a bucket and records in this engine's own file
        (`ENGINE_BLOB_S3_BUCKET` without `ENGINE_REGISTRY_URL`): the bucket
        may be shared by replicas whose records this one cannot see. Nor with
        a store or a registry of the caller's own that cannot list or look up.
        """
        settings = get_settings()
        if self._blobs is None or (
            settings.blob_s3_bucket and not settings.registry_url
        ):
            return 0
        named = getattr(self._registry, "named", None)
        if named is None:
            return 0
        stored = await self._blobs.stored()
        if not stored:
            return 0
        orphans: list[str] = []
        for start in range(0, len(stored), _SWEEP_BATCH):
            batch = stored[start : start + _SWEEP_BATCH]
            known = await named(batch)
            orphans.extend(doc_id for doc_id in batch if doc_id not in known)
        for doc_id in orphans:
            await self._blobs.delete(doc_id=doc_id)
        if orphans:
            logger.info("removed %d stored originals no record named", len(orphans))
        return len(orphans)
