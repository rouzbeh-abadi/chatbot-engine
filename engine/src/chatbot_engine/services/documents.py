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

_MISSING = (
    "no {component} is registered -- implement one under chatbot_engine/rag/ "
    "and return it from chatbot_engine.api.deps.{factory}()"
)


class DocumentService:
    def __init__(
        self,
        *,
        pipeline: IngestPipeline | None = None,
        registry: DocumentRegistry | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._registry = registry

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
        return await self._require_registry().delete(
            project_id=project_id, doc_id=doc_id
        )

    def _require_registry(self) -> DocumentRegistry:
        if self._registry is None:
            raise NotConfiguredError(
                _MISSING.format(component="DocumentRegistry", factory="get_registry")
            )
        return self._registry
