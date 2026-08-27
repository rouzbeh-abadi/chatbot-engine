"""Knowledge base endpoints: upload, list, and delete documents.

This backend validates the request and resolves the project; the engine handles
extraction, chunking, and indexing. Uploads are forwarded as raw bytes.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from support_agent.api.auth import AdminOnly
from support_agent.assistant import ProjectNotFoundError, load_project
from support_agent.engine import EngineDep
from support_agent.engine_client.models import DeleteResult, DocumentRecord

router = APIRouter(prefix="/documents", tags=["documents"])

# Writing to the knowledge base is an operator action, so the write routes carry
# the same `BACKEND_ADMIN_KEY` guard as /admin. Whoever can replace a document
# can change what the assistant tells every user -- a slower, quieter version of
# editing the system prompt, and the reason these are not merely "authenticated".
#
# Listing stays open: it is what the UI's Knowledge panel shows, and it reveals
# only which documents are indexed, which the answers already cite.

# Matches the engine's own limit. Enforced here too, so an oversized file is
# rejected before it crosses the network.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _project_id(name: str | None) -> str:
    try:
        return load_project(name).project_id
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("", status_code=status.HTTP_201_CREATED, dependencies=[AdminOnly])
async def upsert_document(
    engine: EngineDep,
    external_id: str = Form(...),
    file: UploadFile = File(...),
    project: str | None = Form(default=None),
) -> DocumentRecord:
    """Create or replace one document, keyed by `external_id`.

    Idempotent: re-uploading the same `external_id` replaces the previous
    version, and the engine skips re-indexing when the file content is unchanged.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes"
        )

    # Resolve the project first, so an unknown name is a 404 from us rather than
    # an error from the engine.
    project_id = _project_id(project)

    return await engine.ingest_document(
        project_id=project_id,
        external_id=external_id,
        filename=file.filename or external_id,
        mimetype=file.content_type or "application/octet-stream",
        data=data,
    )


@router.get("")
async def list_documents(
    engine: EngineDep, project: str | None = None
) -> list[DocumentRecord]:
    """List the documents indexed for a project."""
    return await engine.list_documents(project_id=_project_id(project))


@router.delete("/{doc_id}", dependencies=[AdminOnly])
async def delete_document(
    doc_id: str, engine: EngineDep, project: str | None = None
) -> DeleteResult:
    """Remove a document."""
    return await engine.delete_document(
        project_id=_project_id(project), doc_id=doc_id
    )
