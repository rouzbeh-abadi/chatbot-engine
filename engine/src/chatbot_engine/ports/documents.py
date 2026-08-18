"""The document side of the engine: ingestion and the two stores.

Three separate concerns on purpose:

* the pipeline turns bytes into something searchable,
* the blob store keeps the original file, so a change of chunk size or embedding
  model is an internal re-index rather than a re-upload for every caller,
* the registry answers "what is indexed, is it current, delete it" -- which a
  vector store does badly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from chatbot_engine.models.documents import DocumentRecord, IngestStatus


class IngestPipeline(Protocol):
    """Raw bytes in, an indexed document out.

    An implementation is expected to hash the bytes and skip everything when the
    hash is unchanged, derive a stable `doc_id` from `project_id` +
    `external_id` so a re-upload overwrites rather than duplicates, then extract,
    chunk, embed and store.
    """

    async def ingest(
        self,
        *,
        project_id: str,
        external_id: str,
        filename: str,
        mimetype: str,
        data: bytes,
    ) -> DocumentRecord: ...


class BlobStore(Protocol):
    """The original uploaded files."""

    async def put(self, *, key: str, data: bytes, mimetype: str) -> str: ...

    async def get(self, *, uri: str) -> bytes: ...

    async def delete(self, *, uri: str) -> None: ...


class DocumentRegistry(Protocol):
    """Document-level bookkeeping, separate from the vectors."""

    async def upsert(self, record: DocumentRecord) -> DocumentRecord: ...

    async def get(self, *, project_id: str, doc_id: str) -> DocumentRecord | None: ...

    async def list(self, *, project_id: str) -> Sequence[DocumentRecord]: ...

    async def set_status(self, *, doc_id: str, status: IngestStatus) -> None: ...

    async def delete(self, *, project_id: str, doc_id: str) -> bool: ...
