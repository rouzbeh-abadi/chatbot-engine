"""Document contracts.

The caller uploads raw bytes over multipart and gets a record back. Extraction,
chunking and embedding happen inside the engine, because the original bytes
carry page numbers and layout that extracted text has already discarded.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: How a document is cut into chunks. Part of the contract: it is sent with an
#: upload and named in the assistant configuration.
ChunkStrategy = Literal["size", "headings", "page"]


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
    #: How the document was cut when it was last indexed: the strategy and the
    #: size and overlap in characters, after the engine's defaults filled in
    #: what the caller left out. A caller compares them with its current
    #: settings to find documents that need re-indexing. Null for a document
    #: indexed by an engine before 0.1.10, which did not record them.
    chunking_strategy: ChunkStrategy | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class ReindexRequest(BaseModel):
    """The body of `POST /documents/{doc_id}/reindex`; every field optional.

    Settings work as on upload: an empty body rebuilds with the engine's
    current defaults (after changing `ENGINE_CHUNK_SIZE`, say), and a field
    left out of a non-empty body takes the engine's default.
    """

    model_config = ConfigDict(extra="forbid")

    chunking_strategy: ChunkStrategy | None = None
    chunk_size: int | None = Field(default=None, ge=100, le=8000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=2000)
    embedding_model: str | None = None


class DeleteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    deleted: bool
