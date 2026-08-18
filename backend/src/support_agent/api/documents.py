"""The knowledge base.

Validation and project resolution happen here; indexing happens in the engine.
Files are forwarded as raw bytes -- parsing a PDF in this backend would take over
a responsibility that belongs to the engine.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from support_agent.assistant import ProjectNotFoundError, load_project
from support_agent.engine import EngineDep
from support_agent.engine_client.models import DeleteResult, DocumentRecord

router = APIRouter(prefix="/documents", tags=["documents"])

# TODO: these routes carry no identity yet. Uploading and deleting from a
# knowledge base are privileged actions, so real authentication belongs here as
# well as on /chat -- see api/chat.py for the placeholder header.

#: Matches the engine's own limit. Checked here so an oversized file never
#: crosses the network, and there too because the engine cannot assume its caller
#: validated anything.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _project_id(name: str | None) -> str:
    try:
        return load_project(name).project_id
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("", status_code=status.HTTP_201_CREATED)
async def upsert_document(
    engine: EngineDep,
    external_id: str = Form(...),
    file: UploadFile = File(...),
    project: str | None = Form(default=None),
) -> DocumentRecord:
    """Upsert one document by `external_id`.

    Idempotent: the same `external_id` replaces the previous version, and the
    engine skips the work when the content hash is unchanged.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes"
        )

    # Resolve the project before calling out, so a bad name is our 404.
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
    """What the assistant currently knows, so the UI can show its sources."""
    return await engine.list_documents(project_id=_project_id(project))


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str, engine: EngineDep, project: str | None = None
) -> DeleteResult:
    """Remove a document."""
    return await engine.delete_document(
        project_id=_project_id(project), doc_id=doc_id
    )
