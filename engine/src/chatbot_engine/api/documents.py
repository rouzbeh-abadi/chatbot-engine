"""Document ingestion: upload, list, delete.

The caller sends raw bytes over multipart. Extraction and chunking are the
engine's job, so a caller that pre-extracted the text would be throwing away page
numbers and layout before the engine ever saw them.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from chatbot_engine.api.deps import DocumentServiceDep
from chatbot_engine.models.documents import DeleteResult, DocumentRecord

router = APIRouter(prefix="/documents", tags=["documents"])

#: A second guard behind the caller's own limit -- the engine is a service and
#: cannot assume its caller validated anything.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.put(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={501: {"description": "No IngestPipeline is registered yet."}},
)
async def upsert_document(
    service: DocumentServiceDep,
    project_id: str = Form(...),
    external_id: str = Form(...),
    file: UploadFile = File(...),
) -> DocumentRecord:
    """Upsert one document, keyed by the caller's `external_id`.

    Idempotent by contract: the same `external_id` replaces the previous version,
    and an implementation should skip the work entirely when the content hash is
    unchanged.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes",
        )

    return await service.ingest(
        project_id=project_id,
        external_id=external_id,
        filename=file.filename or external_id,
        mimetype=file.content_type or "application/octet-stream",
        data=data,
    )


@router.get("", responses={501: {"description": "No DocumentRegistry registered yet."}})
async def list_documents(
    service: DocumentServiceDep,
    project_id: str = Query(...),
) -> list[DocumentRecord]:
    """What is indexed for one project."""
    return list(await service.list(project_id=project_id))


@router.delete(
    "/{doc_id}", responses={501: {"description": "No DocumentRegistry registered yet."}}
)
async def delete_document(
    doc_id: str,
    service: DocumentServiceDep,
    project_id: str = Query(...),
) -> DeleteResult:
    """Remove a document.

    An implementation must remove the blob, the chunks and the registry row
    together -- a partial delete leaves orphaned vectors that still surface in
    retrieval.
    """
    deleted = await service.delete(project_id=project_id, doc_id=doc_id)
    return DeleteResult(doc_id=doc_id, deleted=deleted)
