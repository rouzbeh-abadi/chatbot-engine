"""Document service: the boundary between the HTTP layer and the RAG pipeline.

Holds no logic of its own. Ingestion is the pipeline's job; listing and deletion
touch the registry, the vector store and the blob store together, which is the
one piece of coordination that lives here.
"""

from __future__ import annotations

from collections.abc import Sequence

from chatbot_engine.documents.blobs import DocumentBlobs
from chatbot_engine.models.documents import DocumentRecord
from chatbot_engine.ports.documents import DocumentRegistry, IngestPipeline
from chatbot_engine.rag.vector_store import ChromaChunkStore


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
        chunking_strategy: str | None = None,
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

    async def list(self, *, project_id: str) -> Sequence[DocumentRecord]:
        return await self._registry.list(project_id=project_id)

    async def delete(self, *, project_id: str, doc_id: str) -> bool:
        """Remove a document: chunks, then the file, then the record.

        The record last: it is the only thing that knows the document existed, so
        losing it first would leave vectors and a file nothing can find.
        """
        if self._vectors is not None:
            await self._vectors.delete(doc_id=doc_id)
        if self._blobs is not None:
            await self._blobs.delete(doc_id=doc_id)

        return await self._registry.delete(project_id=project_id, doc_id=doc_id)
