"""Document ingestion boundary.

Like `ChatService`, this holds no logic of its own: it checks that the pieces are
registered and delegates. Extraction, chunking, embedding and storage live behind
`IngestPipeline`, `BlobStore` and `DocumentRegistry`.
"""

from __future__ import annotations

from collections.abc import Sequence

from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.models.documents import DocumentRecord
from chatbot_engine.ports.documents import DocumentRegistry, IngestPipeline
from chatbot_engine.documents.blobs import DocumentBlobs
from chatbot_engine.rag.vector_store import ChromaChunkStore

_MISSING = (
    "no {component} is registered -- implement one under chatbot_engine/rag/ "
    "and return it from chatbot_engine.api.dependencies.{factory}()"
)


class DocumentService:
    def __init__(
        self,
        *,
        pipeline: IngestPipeline | None = None,
        registry: DocumentRegistry | None = None,
        vectors: ChromaChunkStore | None = None,
        blobs: DocumentBlobs | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._registry = registry
        self._vectors = vectors
        self._blobs = blobs

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None and self._registry is not None

    async def ingest(
        self,
        *,
        project_id: str,
        external_id: str,
        filename: str,
        mimetype: str,
        data: bytes,
    ) -> DocumentRecord:
        if self._pipeline is None:
            raise NotConfiguredError(
                _MISSING.format(component="IngestPipeline", factory="get_ingest_pipeline")
            )
        return await self._pipeline.ingest(
            project_id=project_id,
            external_id=external_id,
            filename=filename,
            mimetype=mimetype,
            data=data,
        )

    async def list(self, *, project_id: str) -> Sequence[DocumentRecord]:
        return await self._require_registry().list(project_id=project_id)

    async def delete(self, *, project_id: str, doc_id: str) -> bool:
        """Remove a document: chunks, then the file, then the record.

        The record last: it is the only thing that knows the document existed, so
        losing it first would leave vectors and a file nothing can find.
        """
        registry = self._require_registry()

        if self._vectors is not None:
            await self._vectors.delete(doc_id=doc_id)
        if self._blobs is not None:
            await self._blobs.delete(doc_id=doc_id)

        return await registry.delete(project_id=project_id, doc_id=doc_id)

    def _require_registry(self) -> DocumentRegistry:
        if self._registry is None:
            raise NotConfiguredError(
                _MISSING.format(component="DocumentRegistry", factory="get_registry")
            )
        return self._registry
