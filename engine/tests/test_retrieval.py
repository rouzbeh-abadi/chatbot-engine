"""Hybrid retrieval: what keyword search adds, how the two are fused, and what
the reranker may and may not do.

The embedder is a deterministic fake that hashes text, so vector similarity
here is meaningless on purpose: a chunk that shares the question's exact term
is found by the keyword half or not at all. That is the case hybrid retrieval
exists for, and the fake makes it observable without a provider.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from chatbot_engine.agent import retriever
from chatbot_engine.agent.retriever import _fuse, retrieve
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.rag import rerank as rerank_module
from chatbot_engine.rag import sparse

CHUNKS = {
    "fares.md": "The Basic fare is non-refundable. Flexible fares can be refunded in full.",
    "baggage.md": "Cabin baggage is one bag up to eight kilograms.",
    "checkin.md": "Online check-in opens 24 hours before departure.",
    "seats.md": "Seat selection is available in My Trips after booking.",
}


def _seed(client: TestClient) -> None:
    for name, text in CHUNKS.items():
        response = client.put(
            "/documents",
            data={
                "project_id": "support",
                "external_id": name,
                "chunking_strategy": "size",
            },
            files={"file": (name, text.encode(), "text/markdown")},
        )
        assert response.status_code == 201, response.text


def _request(**config: object) -> ChatRequest:
    project = AssistantConfig(
        project_id="support", name="S", system_prompt="s", top_k=2, **config
    )
    return ChatRequest(project=project, message="is the Basic fare refundable?")


def _sources(hits) -> list[str]:
    return [document.metadata["source"] for document, _ in hits]


# --- what keyword search adds -------------------------------------------------


async def test_hybrid_finds_the_chunk_that_names_the_term(client: TestClient) -> None:
    """With meaningless vectors, only the keyword half can find "Basic fare"."""
    _seed(client)

    hits = await retrieve(_request(retrieval="hybrid"))

    assert _sources(hits)[0] == "fares.md"
    assert all(0.0 <= score <= 1.0 for _, score in hits)
    assert hits[0][1] == 1.0, "the best chunk scores 1.0 whatever produced it"


async def test_vector_mode_is_untouched_by_keywords(client: TestClient) -> None:
    """Chosen explicitly, `vector` must not consult the keyword index at all."""
    _seed(client)

    with patch.object(sparse, "sparse_index", side_effect=AssertionError("consulted")):
        hits = await retrieve(_request(retrieval="vector"))

    assert len(hits) == 2


async def test_the_engine_default_applies_when_the_config_says_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chatbot_engine.api.dependencies import reset_dependency_cache

    monkeypatch.setenv("ENGINE_RETRIEVAL", "vector")
    reset_dependency_cache()
    _seed(client)

    with patch.object(sparse, "sparse_index", side_effect=AssertionError("consulted")):
        await retrieve(_request())

    reset_dependency_cache()


# --- fusion ---------------------------------------------------------------------


def test_fusion_ranks_a_chunk_both_searches_found_above_one_only_one_found() -> None:
    """The property reciprocal rank fusion has and score-averaging lacks: being
    first in the vector search alone is worth less than appearing in both."""
    a, b, c = (Document(page_content=text) for text in ("a", "b", "c"))
    dense = [(a, 0.9), (b, 0.8), (c, 0.7)]
    keyword = [(c, 5.0), (b, 4.0)]

    fused = _fuse(dense, keyword)

    order = [d.page_content for d, _ in fused]
    assert order[-1] == "a", "first in one search loses to any chunk in both"
    assert fused[0][1] == 1.0


def test_fusion_of_nothing_is_nothing() -> None:
    assert _fuse([], []) == []


# --- the keyword index's lifecycle ------------------------------------------


async def test_the_keyword_index_sees_a_document_ingested_after_it_was_built(
    client: TestClient,
) -> None:
    """The bug a cached index has: answering from before the last upload."""
    _seed(client)
    first = await retrieve(_request(retrieval="hybrid"))
    assert "fares.md" in _sources(first)

    client.put(
        "/documents",
        data={
            "project_id": "support",
            "external_id": "promo.md",
            "chunking_strategy": "size",
        },
        files={
            "file": (
                "promo.md",
                b"Basic fare holders get a free Basic fare upgrade voucher.",
                "text/markdown",
            )
        },
    )
    # Terms only the new document has, so the keyword half alone decides.
    request = _request(retrieval="hybrid").model_copy(
        update={"message": "upgrade voucher"}
    )

    later = await retrieve(request)

    assert _sources(later)[0] == "promo.md"


async def test_the_keyword_index_forgets_a_deleted_document(client: TestClient) -> None:
    _seed(client)
    fares = next(
        r
        for r in client.get("/documents", params={"project_id": "support"}).json()
        if r["external_id"] == "fares.md"
    )
    client.delete(f"/documents/{fares['doc_id']}", params={"project_id": "support"})

    hits = await retrieve(_request(retrieval="hybrid"))

    assert "fares.md" not in _sources(hits)


# --- reranking -----------------------------------------------------------------


class _Reranker:
    """A model that answers a listwise prompt with whatever it is told to."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self.reply)


async def test_the_reranker_reorders_the_candidates(client: TestClient) -> None:
    _seed(client)
    model = _Reranker('Sure. {"ranking": [4, 1, 2, 3]}')

    with patch.object(rerank_module, "build_chat_model", return_value=model):
        hits = await retrieve(
            _request(retrieval="hybrid", rerank=True, retrieval_candidates=4)
        )

    # The order the reranker was shown is the fused order; it put #4 first.
    fused = await retrieve(
        _request(retrieval="hybrid", retrieval_candidates=4).model_copy(
            update={
                "project": _request(
                    retrieval="hybrid", retrieval_candidates=4
                ).project.model_copy(update={"top_k": 4})
            }
        )
    )
    assert model.prompts, "the model was asked"
    assert _sources(hits)[0] == _sources(fused)[3]


async def test_a_garbled_rerank_reply_keeps_the_fused_order(client: TestClient) -> None:
    """A rerank may degrade; it must never lose a candidate or the turn."""
    _seed(client)
    without = await retrieve(_request(retrieval="hybrid"))

    with patch.object(
        rerank_module,
        "build_chat_model",
        return_value=_Reranker("I cannot rank these."),
    ):
        with_rerank = await retrieve(_request(retrieval="hybrid", rerank=True))

    assert _sources(with_rerank) == _sources(without)


async def test_a_failing_rerank_call_keeps_the_fused_order(client: TestClient) -> None:
    _seed(client)

    class Exploding:
        async def ainvoke(self, messages):
            raise RuntimeError("provider down")

    without = await retrieve(_request(retrieval="hybrid"))
    with patch.object(rerank_module, "build_chat_model", return_value=Exploding()):
        with_rerank = await retrieve(_request(retrieval="hybrid", rerank=True))

    assert _sources(with_rerank) == _sources(without)


def test_the_ranking_parser_tolerates_prose_and_repairs_omissions() -> None:
    parse = rerank_module._parse

    assert parse('Here you go: {"ranking": [2, 2, 9, 1]}', 3) == [2, 1, 3]
    assert parse("no json here", 3) is None
    assert parse('{"ranking": "2,1"}', 3) is None


# --- what retrieval costs, and on which model ----------------------------------


class _Utility:
    """A model that answers with `reply` and reports fixed token usage."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def ainvoke(self, messages):
        return AIMessage(
            content=self.reply,
            usage_metadata={"input_tokens": 40, "output_tokens": 5, "total_tokens": 45},
        )


async def test_the_rewrite_and_rerank_are_counted_in_the_turns_usage(
    client: TestClient,
) -> None:
    """A turn's cost is every model call it made, not only the answer's."""
    from chatbot_engine.agent.retriever import retrieve_with_usage

    _seed(client)
    from chatbot_engine.models.chat import Message

    request = _request(retrieval="hybrid", rerank=True).model_copy(
        update={"history": [Message(role="user", content="hello")]}
    )

    with (
        patch.object(
            retriever,
            "build_chat_model",
            return_value=_Utility("Basic fare refundable"),
        ),
        patch.object(
            rerank_module, "build_chat_model", return_value=_Utility('{"ranking": [1]}')
        ),
    ):
        _, spent = await retrieve_with_usage(request)

    assert spent == {"input_tokens": 80, "output_tokens": 10, "total_tokens": 90}


async def test_retrieval_usage_reaches_the_usage_event() -> None:
    """Counted by retrieval is worthless unless the agent reports it."""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    from chatbot_engine.agent.chat_agent import ChatAgent
    from chatbot_engine.models.events import UsageEvent

    class Answering(BaseChatModel):
        model_name: str = "openai/gpt-5-mini"

        @property
        def _llm_type(self) -> str:
            return "answering"

        def _generate(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="Hi.",
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                    },
                )
            )

    async def retrieval(request):
        return [], {"input_tokens": 80, "output_tokens": 10, "total_tokens": 90}

    class NoTools:
        async def list_tools(self, config):
            return []

    with (
        patch("chatbot_engine.agent.client.build_chat_model", return_value=Answering()),
        patch("chatbot_engine.agent.chat_agent.retrieve_with_usage", new=retrieval),
    ):
        events = [e async for e in ChatAgent(tools=NoTools()).run(_request())]

    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
        90,
        12,
        102,
    )


def test_the_utility_model_replaces_the_assistants_for_the_small_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chatbot_engine.agent.retriever import utility_config
    from chatbot_engine.api.dependencies import reset_dependency_cache

    project = _request(model="openai/gpt-5", temperature=0.7).project

    monkeypatch.delenv("ENGINE_UTILITY_MODEL", raising=False)
    reset_dependency_cache()
    assert utility_config(project).model == "openai/gpt-5"

    monkeypatch.setenv("ENGINE_UTILITY_MODEL", "openai/gpt-5-nano")
    reset_dependency_cache()
    assert utility_config(project).model == "openai/gpt-5-nano"
    assert utility_config(project).temperature == 0.0, (
        "the small calls are deterministic"
    )
    reset_dependency_cache()


# --- the input cap -------------------------------------------------------------


def test_an_unbounded_message_is_refused_by_the_engine(
    client: TestClient, project
) -> None:
    """The backend caps its own input; the engine cannot assume its caller did."""
    response = client.post("/chat", json={"project": project, "message": "x" * 32_001})

    assert response.status_code == 422
