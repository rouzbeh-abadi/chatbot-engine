"""chatbot-engine: a standalone RAG and MCP tool-calling service.

Runs as its own HTTP service. The application backend calls it over HTTP; it
calls the backend's domain tools back over MCP.

    api/       the HTTP surface -- /chat, /documents, /health
    models/    request and response contracts (pure pydantic)
    services/  the boundary between HTTP and AI logic
    ports/     the interfaces future AI logic must satisfy
    mcp/       MCP client connectivity: configuration and transport
    agent/     YOURS -- the chat run loop
    rag/       YOURS -- chunking, embedding, the vector store, the retriever
    llm/       YOURS -- the chat-model client and the embedder

Kept deliberately import-light so `from chatbot_engine import __version__` costs
nothing: the package root pulls in no framework.
"""

__version__ = "0.1.0"
