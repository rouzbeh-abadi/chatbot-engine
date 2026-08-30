"""Turn chunks into vectors.

The embedding model is not fixed by the engine: it arrives as part of the
assistant's config, the same way the chat model does, so one engine can serve
projects that embed with different models. `resolve_embedding_model` supplies the
engine's configured default when a caller passes none.

Embedding calls are asynchronous to keep network-bound batches from blocking the
event loop and delaying unrelated requests in the same process.
"""

from __future__ import annotations

from functools import lru_cache

from chatbot_engine.settings import get_settings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def resolve_embedding_model(model: str | None = None) -> str:
    """The caller's model, or the engine default when they gave none.

    Mirrors how the chat model falls back to `settings.chat_model`: the backend
    owns the choice, the engine only fills in a default so it still runs with no
    per-request config at all.
    """
    return model or get_settings().embedding_model


@lru_cache
def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    """An embedder for `model`, one per distinct model in the process.

    Cached by model, so a project embedding with a different model gets its own
    client rather than sharing one bound to the wrong model. A missing provider
    key is a 501, not a crash.
    """
    settings = get_settings()

    return OpenAIEmbeddings(
        model=resolve_embedding_model(model),
        api_key=settings.require_openrouter_key(),
        base_url=settings.openrouter_base_url,
    )


async def embed_documents(
    documents: list[Document], model: str | None = None
) -> list[list[float]]:
    """Embed each chunk, in the order it was given."""
    texts = [document.page_content for document in documents]

    return await get_embeddings(model).aembed_documents(texts)


async def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed one question, for searching with.

    Must use the same model the documents were embedded with, or the distances
    mean nothing -- which the vector store guarantees by keying each collection
    to its embedding model.
    """
    return await get_embeddings(model).aembed_query(text)
