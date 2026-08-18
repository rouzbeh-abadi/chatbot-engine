"""Ports: the interfaces the engine's future AI logic will satisfy.

Framework-free by design -- a port that imported FastAPI or a vendor SDK would
drag the whole transport into every implementation.

    Agent             one chat turn, as a stream of events
    ToolProvider      tool discovery and invocation over MCP
    IngestPipeline    raw bytes in, an indexed document out
    BlobStore         the original uploaded files
    DocumentRegistry  what is indexed and whether it is current
"""

from chatbot_engine.ports.agent import Agent, ToolProvider
from chatbot_engine.ports.documents import (
    BlobStore,
    DocumentRegistry,
    IngestPipeline,
)

__all__ = [
    "Agent",
    "BlobStore",
    "DocumentRegistry",
    "IngestPipeline",
    "ToolProvider",
]
