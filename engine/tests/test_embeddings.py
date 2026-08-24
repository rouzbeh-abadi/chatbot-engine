"""The embedder. No network: the provider call itself is faked."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.rag import embeddings as module
from chatbot_engine.rag.embeddings import embed_documents, embed_query, get_embeddings


class FakeEmbeddings:
    """Records what it was asked to embed."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.seen = texts
        return [[float(index), 0.5] for index, _ in enumerate(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        self.seen = [text]
        return [0.3, 0.4]


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    get_embeddings.cache_clear()
    reset_dependency_cache()

    yield

    get_embeddings.cache_clear()
    reset_dependency_cache()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeEmbeddings:
    stub = FakeEmbeddings()
    monkeypatch.setattr(module, "get_embeddings", lambda: stub)

    return stub


# --- the credential ----------------------------------------------------------


def test_a_missing_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "")
    reset_dependency_cache()

    with pytest.raises(NotConfiguredError, match="ENGINE_OPENROUTER_API_KEY"):
        get_embeddings()


def test_the_model_and_base_url_come_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ENGINE_EMBEDDING_MODEL", "openai/text-embedding-3-large")
    monkeypatch.setenv("ENGINE_OPENROUTER_BASE_URL", "http://localhost:11434/v1")
    reset_dependency_cache()

    embedder = get_embeddings()

    assert embedder.model == "openai/text-embedding-3-large"
    assert embedder.openai_api_base == "http://localhost:11434/v1"


def test_the_embedder_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "sk-or-test")
    reset_dependency_cache()

    assert get_embeddings() is get_embeddings()


# --- what gets sent ----------------------------------------------------------


async def test_only_the_chunk_text_is_sent(fake: FakeEmbeddings) -> None:
    """Metadata stays behind -- it is not part of what gets embedded."""
    await embed_documents(
        [
            Document(page_content="one bag", metadata={"source": "baggage.md"}),
            Document(page_content="non-refundable", metadata={"source": "refunds.md"}),
        ]
    )

    assert fake.seen == ["one bag", "non-refundable"]


async def test_vectors_come_back_in_chunk_order(fake: FakeEmbeddings) -> None:
    """The pipeline pairs vector[i] with chunk[i], so order is the contract."""
    vectors = await embed_documents(
        [Document(page_content=text) for text in ("a", "b", "c")]
    )

    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0]


async def test_no_chunks_means_no_call(fake: FakeEmbeddings) -> None:
    assert await embed_documents([]) == []
    assert fake.seen == []


async def test_a_query_is_embedded_on_its_own(fake: FakeEmbeddings) -> None:
    """Same model as the documents, or the distances mean nothing."""
    assert await embed_query("how much luggage?") == [0.3, 0.4]
    assert fake.seen == ["how much luggage?"]
