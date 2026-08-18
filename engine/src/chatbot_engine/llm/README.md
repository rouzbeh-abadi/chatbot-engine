# llm/ — yours to write

The chat-model client, and the embedder.

Keep the provider behind an extra in `pyproject.toml` and import it only from
this folder, so the rest of the engine stays provider-agnostic.

Two things worth getting right:

- **Read the API key when you build the client, not at import time.** A service
  that dies on startup because a provider it was not asked to use lacks a key is
  painful to deploy.
- **Keep the raw response.** Token counts live on it, and OpenRouter reports spend
  under `response_metadata.token_usage.cost`, outside the standard usage block.
  Discard the raw message and `UsageEvent` silently reports zero.
