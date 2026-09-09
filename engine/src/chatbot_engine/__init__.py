"""chatbot-engine: a standalone RAG and MCP tool-calling service.

Runs as its own HTTP service. The application backend calls it over HTTP; it
calls the backend's domain tools back over MCP.

    api/        the HTTP surface: /chat, /agents, /documents, /judge, /health
    models/     request and response contracts (pure pydantic)
    ports/      the interfaces the rest of the engine depends on
    services/   the boundary between HTTP and the ports
    agent/      the chat turn: registry, router, the built-in loop, retrieval
    rag/        chunking, embedding, the vector store, the ingest pipeline
    mcp/        the MCP client that reaches the application's tools
    documents/  document bookkeeping and stored originals
    eval/       the system-prompt judge and the RAGAS retrieval evaluation

Kept deliberately import-light so `from chatbot_engine import __version__` costs
nothing: the package root pulls in no framework.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    #: The installed package's version, so /health and the OpenAPI document
    #: cannot drift from pyproject.toml the way a hand-written string did.
    __version__ = version("chatbot-engine")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0+unknown"
