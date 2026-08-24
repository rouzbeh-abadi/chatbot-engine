"""Turn chunks into vectors.

Embedding calls are asynchronous to keep network-bound batches from blocking the
event loop and delaying unrelated requests in the same process.
"""

from __future__ import annotations

from functools import lru_cache

from chatbot_engine.settings import get_settings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """One embedder for the process. A missing key is a 501, not a crash."""
    settings = get_settings()

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.require_openrouter_key(),
        base_url=settings.openrouter_base_url,
    )


async def embed_documents(documents: list[Document]) -> list[list[float]]:
    """Embed each chunk, in the order it was given."""
    texts = [document.page_content for document in documents]

    return await get_embeddings().aembed_documents(texts)


async def embed_query(text: str) -> list[float]:
    """Embed one question, for searching with."""
    return await get_embeddings().aembed_query(text)
