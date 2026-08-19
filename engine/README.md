# chatbot-engine

A standalone RAG and MCP tool-calling service.

The engine answers questions from a knowledge base, calls tools when it needs
live data, and streams the result back token by token. It knows how to retrieve,
how to prompt a model, and how to run a tool loop — and nothing at all about your
users, your product, or your domain.

That split is deliberate. Everything specific to an application stays in the
application: who the user is, what the assistant is called, which prompt it uses,
which tools it may reach. All of it arrives with each request, so the engine keeps
no state of yours and one engine can serve several applications.

It runs as its own HTTP service. Your backend calls it over HTTP; it calls your
tools back over MCP, because a tool that reads your data has to run where your
data and your permissions are.

```text
Your backend        →  HTTP    →  chatbot engine
chatbot engine      →  MCP     →  your tool server
```

## Running it

```bash
uv run uvicorn chatbot_engine.app:app --port 8100 --reload
```

Interactive API docs: http://localhost:8100/docs

## Connecting a backend to it

Everything a backend needs — the endpoints, the request shape, how to read the
streamed answer, how to upload documents, how to handle the engine being down, and
how to expose your own tools over MCP — is written up here:

**→ [Connecting a backend to the chatbot engine](../docs/backend-integration.md)**

Start there. It includes a complete worked example and points at a working
reference backend.
