"""A stand-in for the engine service.

Speaks the same methods as `EngineClient`, so the routes, the request validation
and the SSE framing under test are the real ones -- only the network is fake.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from support_agent.engine_client.models import (
    ChatEvent,
    DeleteResult,
    DocumentRecord,
    DoneEvent,
    EngineChatRequest,
    IngestStatus,
    RetrievalEvent,
    SourceRef,
    TokenEvent,
    UsageEvent,
)


class FakeEngine:
    """Stands in for the engine service. Records what it was asked for."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.raises = raises
        self.chat_requests: list[EngineChatRequest] = []
        self.ingested: list[dict[str, object]] = []
        self.deleted: list[tuple[str, str]] = []

    async def start_chat(self, request: EngineChatRequest) -> AsyncIterator[ChatEvent]:
        if self.raises is not None:
            raise self.raises
        self.chat_requests.append(request)

        async def events() -> AsyncIterator[ChatEvent]:
            yield RetrievalEvent(
                query=request.message,
                sources=[
                    SourceRef(
                        doc_id="d1",
                        source="baggage.md",
                        score=0.82,
                        heading="Baggage Policy > Cabin Baggage",
                    )
                ],
            )
            yield TokenEvent(text="One ")
            yield TokenEvent(text="cabin bag.")
            yield UsageEvent(total_tokens=42, model="test-model")
            yield DoneEvent()

        return events()

    async def ingest_document(self, **kwargs: object) -> DocumentRecord:
        if self.raises is not None:
            raise self.raises
        self.ingested.append(kwargs)
        data = kwargs["data"]
        assert isinstance(data, bytes)
        return DocumentRecord(
            doc_id="doc-1",
            external_id=str(kwargs["external_id"]),
            project_id=str(kwargs["project_id"]),
            filename=str(kwargs["filename"]),
            mimetype=str(kwargs["mimetype"]),
            size_bytes=len(data),
            content_hash="deadbeef",
            status=IngestStatus.INDEXED,
            chunk_count=3,
        )

    async def list_documents(self, *, project_id: str) -> list[DocumentRecord]:
        if self.raises is not None:
            raise self.raises
        return []

    async def delete_document(self, *, project_id: str, doc_id: str) -> DeleteResult:
        if self.raises is not None:
            raise self.raises
        self.deleted.append((project_id, doc_id))
        return DeleteResult(doc_id=doc_id, deleted=True)
