"""Where implementations get plugged in.

This is the engine's single wiring point. Each factory below returns `None`
today, which makes the service layer raise `NotConfiguredError` (an
`NotImplementedError`) and the API answer 501 with a pointer.

To bring a capability online, write it under `chatbot_engine/agent/` or
`chatbot_engine/rag/` and return an instance from the matching factory. Nothing
in `api/` or `services/` changes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from chatbot_engine.mcp.client import McpToolProvider
from chatbot_engine.ports.agent import Agent, ToolProvider
from chatbot_engine.ports.documents import (
    BlobStore,
    DocumentRegistry,
    IngestPipeline,
)
from chatbot_engine.services.chat import ChatService
from chatbot_engine.services.documents import DocumentService
from chatbot_engine.settings import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]


# --- your implementations go here -------------------------------------------


@lru_cache
def get_agent() -> Agent | None:
    """The chat run loop: retrieval, prompt, model call, tool loop, events.

    Hand it `get_tool_provider()` when you build it -- the agent is what runs the
    tool loop, so it is the thing that needs tool access.
    """
    return None


@lru_cache
def get_ingest_pipeline() -> IngestPipeline | None:
    """Document ingestion: extract, chunk, embed, store."""
    return None


@lru_cache
def get_registry() -> DocumentRegistry | None:
    """Document bookkeeping: what is indexed, is it current, delete it."""
    return None


@lru_cache
def get_blob_store() -> BlobStore | None:
    """The original uploaded files.

    Nothing here consumes it directly -- hand it to your `IngestPipeline` below,
    which is what needs to keep and re-read the originals.
    """
    return None


@lru_cache
def get_tool_provider() -> ToolProvider:
    """MCP connectivity. Constructed, but its calls are not implemented yet."""
    return McpToolProvider(timeout_s=get_settings().mcp_timeout_s)


# --- services ---------------------------------------------------------------


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(agent=get_agent())


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(pipeline=get_ingest_pipeline(), registry=get_registry())


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


# --- caller authentication --------------------------------------------------


async def require_api_key(
    settings: SettingsDep,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Optional shared secret between the backend and the engine.

    Unset `ENGINE_API_KEY` leaves the engine open, which is fine on localhost.
    Set it in any deployment: the engine holds provider credentials and has no
    notion of end-user permissions, so it must not be reachable by anyone but
    the application backend.
    """
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )


def reset_dependency_cache() -> None:
    """Drop cached services. For tests, and after re-wiring an implementation."""
    for cached in (
        get_settings,
        get_agent,
        get_ingest_pipeline,
        get_registry,
        get_blob_store,
        get_tool_provider,
        get_chat_service,
        get_document_service,
    ):
        cached.cache_clear()
