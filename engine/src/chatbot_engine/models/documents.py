"""Document contracts.

The caller uploads raw bytes over multipart and gets a record back. Extraction,
chunking and embedding happen inside the engine, because the original bytes
carry page numbers and layout that extracted text has already discarded.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class IngestStatus(StrEnum):
    RECEIVED = "received"
    INDEXED = "indexed"
    FAILED = "failed"
    UNCHANGED = "unchanged"


class DocumentRecord(BaseModel):
    """What the caller gets back: is it indexed, is it current, how many chunks."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    external_id: str
    project_id: str
    filename: str
    mimetype: str
    size_bytes: int
    content_hash: str
    status: IngestStatus
    chunk_count: int = 0
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DeleteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    deleted: bool
