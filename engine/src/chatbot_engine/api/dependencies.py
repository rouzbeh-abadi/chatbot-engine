"""Create and connect the dependencies used by the chatbot engine.

This module builds the engine's main components, such as the chat agent,
document pipeline, vector store, registry, and MCP tool provider. FastAPI uses
these dependencies when handling API requests.

Dependencies are cached and reused instead of being created for every request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from chatbot_engine.agent.chat_agent import ChatAgent
from chatbot_engine.documents.blobs import DocumentBlobs
from chatbot_engine.documents.sqlite_registry import SqliteDocumentRegistry
from chatbot_engine.Eval.prompt_evaluation import evaluate_dataset
from chatbot_engine.mcp.client import McpToolProvider
from chatbot_engine.models.evals import JudgeReport, JudgeRequest
from chatbot_engine.ports.agent import Agent, Judge, ToolProvider
from chatbot_engine.ports.documents import DocumentRegistry, IngestPipeline
from chatbot_engine.rag.embeddings import get_embeddings
from chatbot_engine.rag.pipeline import DocumentIngestPipeline
from chatbot_engine.rag.splitter import DocumentChunker
from chatbot_engine.rag.vector_store import (
    ChromaChunkStore,
    open_vector_store,
)
from chatbot_engine.services.chat import ChatService
from chatbot_engine.services.documents import DocumentService
from chatbot_engine.settings import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def get_agent() -> Agent | None:
    """Create the agent responsible for processing chat turns.

    The agent retrieves relevant context, calls the model with access to MCP tools,

    and streams the resulting chat events.
    """
    return ChatAgent(tools=get_tool_provider())


@lru_cache
def get_judge() -> Judge | None:
    """Scores an eval run against a rubric the caller supplies.

    Needs the agent, since it answers every case before grading it.
    """
    agent = get_agent()
    if agent is None:
        return None

    async def judge(request: JudgeRequest) -> JudgeReport:
        return await evaluate_dataset(request, agent=agent)

    return judge


@lru_cache
def get_ingest_pipeline() -> IngestPipeline | None:
    """Document ingestion: extract, chunk, embed, store. Fully wired."""
    return DocumentIngestPipeline(
        registry=get_registry(),
        chunker=DocumentChunker(),
        vectors=get_chunk_store(),
        blobs=get_blob_store(),
    )


@lru_cache
def get_registry() -> DocumentRegistry:
    """Document bookkeeping: what is indexed, is it current, delete it.

    On disk, because the vectors are: an in-memory registry plus a persistent
    Chroma leaves chunks after a restart that nothing lists and nothing can
    delete. `InMemoryDocumentRegistry` is still there for tests.
    """
    return SqliteDocumentRegistry(get_settings().registry_db)


@lru_cache
def get_blob_store() -> DocumentBlobs:
    """The original uploaded files, addressed by `doc_id`.

    `DocumentBlobs` computes the URI from the root and the id, so nothing has to
    be stored on `DocumentRecord` and no filesystem path goes on the wire.
    """
    return DocumentBlobs(get_settings().blob_dir)


@lru_cache
def get_chunk_store() -> ChromaChunkStore | None:
    """The vectors, when there is a key to embed with.

    Without a provider key there is nothing to embed with, so documents are
    chunked and recorded but not searchable: `received` rather than `indexed`,
    which is what `GET /documents` and the UI then show. Better than taking the
    whole document surface down over a key that only search needs.
    """
    if get_settings().openrouter_api_key is None:
        return None

    return ChromaChunkStore()


@lru_cache
def get_tool_provider() -> ToolProvider:
    """MCP connectivity. Constructed, but its calls are not implemented yet."""
    return McpToolProvider(timeout_s=get_settings().mcp_timeout_s)


# services


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(agent=get_agent())


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(
        pipeline=get_ingest_pipeline(),
        registry=get_registry(),
        vectors=get_chunk_store(),
        blobs=get_blob_store(),
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
JudgeDep = Annotated[Judge | None, Depends(get_judge)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


# caller authentication


async def require_api_key(
    settings: SettingsDep,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Optional shared secret between the backend and the engine.

    Unset `ENGINE_API_KEY` leaves the engine open, which is fine on localhost.
    Set it in any deployment: the engine holds provider credentials and has no notion of end-user permissions, so it must not be reachable by anyone but the application backend.
    """
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )


def reset_dependency_cache() -> None:
    """Drop cached services. For tests, and after re-wiring an implementation.

    The two caches in `rag/` belong here too: a Chroma client holds an open
    directory, so a test that changes `ENGINE_CHROMA_DIR` would otherwise keep
    writing to the previous one.
    """
    for cached in (
        get_settings,
        get_embeddings,
        open_vector_store,
        get_agent,
        get_ingest_pipeline,
        get_registry,
        get_blob_store,
        get_chunk_store,
        get_tool_provider,
        get_chat_service,
        get_document_service,
        get_judge,
    ):
        cached.cache_clear()
