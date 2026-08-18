from collections.abc import Sequence

from chatbot_engine.models.documents import DocumentRecord, IngestStatus
from chatbot_engine.ports.documents import DocumentRegistry


class InMemoryDocumentRegistry(DocumentRegistry):
    """Store document metadata in memory for the lifetime of the engine process."""

    def __init__(self) -> None:
        """Initialize an empty document registry."""
        self._records: dict[tuple[str, str], DocumentRecord] = {}

    async def upsert(
        self,
        record: DocumentRecord,
    ) -> DocumentRecord:
        """Create or replace one document record.

        Args:
            record: Complete document metadata to store.
        """
        key = (record.project_id, record.doc_id)
        self._records[key] = record

        return record

    async def get(
        self,
        *,
        project_id: str,
        doc_id: str,
    ) -> DocumentRecord | None:
        """Return one document record if it exists.

        Args:
            project_id: Identifier of the project owning the document.
            doc_id: Identifier of the document.
        """
        return self._records.get((project_id, doc_id))

    async def list(
        self,
        *,
        project_id: str,
    ) -> Sequence[DocumentRecord]:
        """Return all document records belonging to one project.

        Args:
            project_id: Identifier of the project whose documents should be listed.
        """
        return [
            record
            for record in self._records.values()
            if record.project_id == project_id
        ]

    async def set_status(
        self,
        *,
        doc_id: str,
        status: IngestStatus,
    ) -> None:
        """Update the ingestion status of one document.

        Args:
            doc_id: Identifier of the document.
            status: New ingestion status.
        """
        for key, record in self._records.items():
            if record.doc_id == doc_id:
                self._records[key] = record.model_copy(
                    update={"status": status}
                )
                return

    async def delete(
        self,
        *,
        project_id: str,
        doc_id: str,
    ) -> bool:
        """Delete one document record and report whether it existed.

        Args:
            project_id: Identifier of the project owning the document.
            doc_id: Identifier of the document.
        """
        key = (project_id, doc_id)

        return self._records.pop(key, None) is not None