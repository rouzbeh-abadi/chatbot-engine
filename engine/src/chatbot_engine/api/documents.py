"""Document ingestion: upload, list, re-index, delete.

The caller sends raw bytes over multipart. Extraction and chunking are the
engine's job, so a caller that pre-extracted the text would be throwing away page
numbers and layout before the engine ever saw them.

Every route refuses a query parameter it does not take, as the upload refuses
a form field (`only_params`, `only_fields`): a misspelled or misplaced setting
is a 422 that names it, never one quietly left out.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from chatbot_engine.api.dependencies import DocumentServiceDep
from chatbot_engine.api.forms import only_fields, only_params
from chatbot_engine.api.rate_limit import limit_ingest
from chatbot_engine.errors import (
    DocumentNotFoundError,
    NotConfiguredError,
    OriginalChangedError,
)
from chatbot_engine.models.documents import DeleteResult, DocumentRecord, ReindexRequest
from chatbot_engine.rag.splitter import ChunkingError, ChunkStrategy

router = APIRouter(prefix="/documents", tags=["documents"])

#: A second guard behind the caller's own limit -- the engine is a service and
#: cannot assume its caller validated anything.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.put(
    "",
    status_code=status.HTTP_201_CREATED,
    # Indexing embeds every chunk, which is billed per token. Listing and
    # deleting call no provider, so neither is metered.
    dependencies=[
        Depends(limit_ingest),
        Depends(
            only_fields(
                "project_id",
                "external_id",
                "file",
                "embedding_model",
                "chunking_strategy",
                "chunk_size",
                "chunk_overlap",
            )
        ),
        Depends(only_params()),
    ],
)
async def upsert_document(
    service: DocumentServiceDep,
    project_id: str = Form(...),
    external_id: str = Form(...),
    file: UploadFile = File(...),
    embedding_model: str | None = Form(default=None),
    chunking_strategy: ChunkStrategy | None = Form(default=None),
    chunk_size: int | None = Form(default=None, ge=100, le=8000),
    chunk_overlap: int | None = Form(default=None, ge=0, le=2000),
) -> DocumentRecord:
    """Upsert one document, keyed by the caller's `external_id`.

    Idempotent: the same `external_id` replaces the previous version, and
    identical bytes skip the work entirely and answer `unchanged`.
    """
    # One byte past the limit at most: enough to know it is too large.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes",
        )

    try:
        return await service.ingest(
            project_id=project_id,
            external_id=external_id,
            filename=file.filename or external_id,
            mimetype=file.content_type or "application/octet-stream",
            data=data,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ChunkingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/{doc_id}/reindex",
    responses={
        404: {"description": "No such document in the project."},
        409: {
            "description": "The stored original is not the version the record "
            "describes; upload the document again."
        },
        501: {"description": "No blob store, so the original was never kept."},
    },
    # Re-embeds every chunk, billed like an upload. The settings go in the
    # body; one sent in the query instead is refused, not left out.
    dependencies=[Depends(limit_ingest), Depends(only_params("project_id"))],
)
async def reindex_document(
    doc_id: str,
    service: DocumentServiceDep,
    project_id: str = Query(...),
    request: ReindexRequest | None = None,
) -> DocumentRecord:
    """Rebuild one document from its stored original, optionally cut differently.

    What a caller uses after changing chunking settings: the document is
    re-chunked and re-embedded without being uploaded again. Settings work as
    on upload: the ones left out take the engine's defaults.
    """
    body = request or ReindexRequest()
    try:
        return await service.reindex(
            project_id=project_id,
            doc_id=doc_id,
            embedding_model=body.embedding_model,
            chunking_strategy=body.chunking_strategy,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ChunkingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OriginalChangedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NotConfiguredError:
        raise


@router.get("", dependencies=[Depends(only_params("project_id"))])
async def list_documents(
    service: DocumentServiceDep,
    project_id: str = Query(...),
) -> list[DocumentRecord]:
    """What is indexed for one project."""
    return list(await service.list(project_id=project_id))


@router.delete("/{doc_id}", dependencies=[Depends(only_params("project_id"))])
async def delete_document(
    doc_id: str,
    service: DocumentServiceDep,
    project_id: str = Query(...),
) -> DeleteResult:
    """Remove a document.

    Removes the chunks, the stored file and the registry row together: a partial
    delete would leave orphaned vectors that still surface in retrieval. One
    that stops part-way (a store that does not answer) is a 500 that leaves
    the record `failed`: still listed, to be deleted again, and indexed again
    by an upload of the same file. Only a document `project_id` has: another
    project's, whatever its id, is not found, and the answer is
    `deleted: false` with nothing touched.
    """
    deleted = await service.delete(project_id=project_id, doc_id=doc_id)
    return DeleteResult(doc_id=doc_id, deleted=deleted)
