"""Hybrid retrieval: what keyword search adds, how the two are fused, and what
the reranker may and may not do.

The embedder is a deterministic fake that hashes text, so vector similarity
here is meaningless on purpose: a chunk that shares the question's exact term
is found by the keyword half or not at all. That is the case hybrid retrieval
exists for, and the fake makes it observable without a provider.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from chatbot_engine.agent import retriever
from chatbot_engine.agent.retriever import (
    KEYWORD_FUSED,
    MAX_QUERIES,
    REWRITE_MAX_TOKENS,
    _fuse,
    _fuse_hybrid,
    retrieve,
    retrieve_with_usage,
    rewrite_queries,
)
from chatbot_engine.models.chat import AssistantConfig, ChatRequest, Message
from chatbot_engine.rag import rerank as rerank_module
from chatbot_engine.rag import sparse
from chatbot_engine.rag.vector_store import open_vector_store

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
        **{
            "project_id": "support",
            "name": "S",
            "system_prompt": "s",
            "top_k": 2,
            **config,
        }
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


async def test_a_project_without_documents_gets_no_hits_rather_than_an_error(
    client: TestClient,
) -> None:
    """The engine serves many projects; one that has uploaded nothing yet is a
    normal state. Its turn answers without context instead of failing."""
    assert await retrieve(_request(retrieval="hybrid")) == []
    assert await retrieve(_request(retrieval="vector")) == []

    # Another project's documents do not leak in, and do not change the answer.
    for name, text in CHUNKS.items():
        client.put(
            "/documents",
            data={"project_id": "other", "external_id": name},
            files={"file": (name, text.encode(), "text/markdown")},
        )
    assert await retrieve(_request(retrieval="hybrid")) == []


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


def test_common_words_do_not_push_out_the_chunk_closest_in_meaning() -> None:
    """The chunk that answers "How long do I have to return a lamp?" says
    "accepts returns within 30 days": first on meaning, and no word in common
    with the question. Every other chunk shares a common word ("to", "a") and
    so is in both rankings. Only the keyword search's best few take part in
    fusion, so the answer still reaches the top five."""
    answer = Document(page_content="Lumen Lamps accepts returns within 30 days.")
    others = [Document(page_content=f"chunk {n} with to and a") for n in range(6)]
    dense = [(answer, 0.65), *((d, 0.5 - n / 100) for n, d in enumerate(others))]
    keyword = [(d, 2.0 - n / 10) for n, d in enumerate(others)]

    top5 = [d.page_content for d, _ in _fuse_hybrid(dense, keyword)[:5]]
    everything = [d.page_content for d, _ in _fuse(dense, keyword)[:5]]

    assert answer.page_content in top5
    assert answer.page_content not in everything, "fusing every keyword hit loses it"
    # The keyword search's best still ranks first: exact terms keep their weight.
    assert top5[0] == others[0].page_content
    assert KEYWORD_FUSED == 3


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


async def test_the_keyword_index_is_built_and_searched_off_the_event_loop(
    client: TestClient,
) -> None:
    """Both read or score every chunk of the project: on the event loop they
    would hold up every other request in flight."""
    _seed(client)
    threads: list[threading.Thread] = []
    build = sparse.sparse_index
    search = sparse.SparseIndex.search

    def building(store, project_id):
        threads.append(threading.current_thread())
        return build(store, project_id)

    def searching(self, query, k):
        threads.append(threading.current_thread())
        return search(self, query, k)

    with (
        patch.object(sparse, "sparse_index", building),
        patch.object(sparse.SparseIndex, "search", searching),
    ):
        await retrieve(_request(retrieval="hybrid"))

    assert len(threads) == 2
    assert threading.main_thread() not in threads


async def test_a_search_within_the_interval_reads_nothing_from_the_collection(
    client: TestClient,
) -> None:
    """The index used to list every chunk id of the project on every query to
    see whether it had changed. Within its interval it is now trusted, and
    an ingest or a delete in this process drops it at once."""
    _seed(client)
    await retrieve(_request(retrieval="hybrid"))

    with patch.object(Chroma, "get", side_effect=AssertionError("read the chunks")):
        hits = await retrieve(_request(retrieval="hybrid"))

    assert _sources(hits)[0] == "fares.md"


async def test_the_index_is_built_again_once_its_interval_has_passed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """How another replica's upload reaches this one."""
    _seed(client)
    builds: list[str] = []
    build = sparse._build

    def counting(store, project_id, now):
        builds.append(project_id)
        return build(store, project_id, now)

    monkeypatch.setattr(sparse, "_build", counting)
    await retrieve(_request(retrieval="hybrid"))
    await retrieve(_request(retrieval="hybrid"))
    assert builds == ["support"]

    monkeypatch.setattr(sparse, "INDEX_TTL_S", 0.0)
    await retrieve(_request(retrieval="hybrid"))
    assert builds == ["support", "support"]


def test_only_so_many_indexes_are_kept_the_least_recently_searched_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparse, "MAX_INDEXES", 2)
    sparse.invalidate()
    store = open_vector_store()

    for project in ("a", "b", "a", "c"):
        sparse.sparse_index(store, project)

    assert [project for _, project in sparse._indexes] == ["a", "c"]


def test_the_keyword_index_scores_as_bm25okapi_does() -> None:
    """Kept as arrays instead of a dict of counts per chunk, the index must
    still rank and score as `rank_bm25.BM25Okapi` did."""
    import random
    from array import array
    from collections import Counter

    from rank_bm25 import BM25Okapi

    rng = random.Random(7)
    for _ in range(100):
        vocab = [f"w{i}" for i in range(rng.randint(3, 60))]
        texts = [
            " ".join(rng.choices(vocab, k=rng.randint(1, 30)))
            for _ in range(rng.randint(1, 40))
        ]
        hashes, chunks, counts, lengths = array("q"), array("i"), array("f"), []
        for number, text in enumerate(texts):
            terms = Counter(sparse.tokenize(text))
            lengths.append(sum(terms.values()))
            for term, times in terms.items():
                hashes.append(hash(term))
                chunks.append(number)
                counts.append(times)
        index = sparse._index(
            texts, [{}] * len(texts), hashes, chunks, counts, lengths, 0.0, True
        )
        reference = BM25Okapi([sparse.tokenize(text) for text in texts])

        for _ in range(5):
            query = " ".join(rng.choices([*vocab, "absent"], k=rng.randint(1, 4)))
            scores = reference.get_scores(sparse.tokenize(query))
            expected = [
                (texts[i], score)
                for i, score in sorted(
                    enumerate(scores.tolist()), key=lambda hit: hit[1], reverse=True
                )
                if score > 0
            ]
            got = [(d.page_content, s) for d, s in index.search(query, len(texts))]
            assert [text for text, _ in got] == [text for text, _ in expected]
            assert [s for _, s in got] == pytest.approx([s for _, s in expected])


def test_the_indexes_kept_stay_within_their_bytes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counted by bytes as well as by number: past the budget, the one
    searched longest ago goes, and one larger than all of it is not kept."""
    _seed(client)
    sparse.invalidate()
    store = open_vector_store()
    index = sparse.sparse_index(store, "support")
    assert index is not None
    monkeypatch.setattr(
        sparse, "get_settings", lambda: SimpleNamespace(keyword_index_mb=1)
    )
    sparse.invalidate()

    with patch.object(sparse.SparseIndex, "__post_init__", _sized(600_000)):
        for project in ("support", "other", "third"):
            sparse.sparse_index(store, project)
        assert [project for _, project in sparse._indexes] == ["third"]

    with patch.object(sparse.SparseIndex, "__post_init__", _sized(2_000_000)):
        sparse.invalidate()
        assert sparse.sparse_index(store, "support") is not None
        assert not sparse._indexes


def _sized(nbytes: int):
    def post_init(self) -> None:
        self.nbytes = nbytes

    return post_init


def test_a_project_of_too_many_chunks_is_searched_by_vector_alone(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(client)
    sparse.invalidate()
    monkeypatch.setattr(sparse, "MAX_INDEX_CHUNKS", 1)

    assert sparse.sparse_index(open_vector_store(), "support") is None


def test_an_index_built_across_an_invalidation_is_not_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Built from the chunks before an ingest that finished meanwhile, it
    would answer from the old version for the rest of its interval."""
    sparse.invalidate()
    build = sparse._build

    def racing(store, project_id, now):
        index = build(store, project_id, now)
        sparse.invalidate(project_id)
        return index

    monkeypatch.setattr(sparse, "_build", racing)
    sparse.sparse_index(open_vector_store(), "support")

    assert not sparse._indexes


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

    async def ainvoke(self, messages, config=None):
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
        async def ainvoke(self, messages, config=None):
            raise RuntimeError("provider down")

    without = await retrieve(_request(retrieval="hybrid"))
    with patch.object(rerank_module, "build_chat_model", return_value=Exploding()):
        with_rerank = await retrieve(_request(retrieval="hybrid", rerank=True))

    assert _sources(with_rerank) == _sources(without)


async def test_the_rerank_judges_against_every_query(client: TestClient) -> None:
    """Not the first query alone: a message that asks two things is two
    queries, and a passage that answers the second is as relevant."""
    _seed(client)
    model = _Reranker('{"ranking": [1]}')

    with patch.object(rerank_module, "build_chat_model", return_value=model):
        await retrieve_with_usage(
            _request(retrieval="hybrid", rerank=True),
            queries=["is the Basic fare refundable", "how heavy can a cabin bag be"],
        )

    assert model.prompts[0].startswith(
        "Question: is the Basic fare refundable\nhow heavy can a cabin bag be\n"
    )


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

    async def ainvoke(self, messages, config=None):
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

    # All of it ran on the utility model, and is marked so, for the caller to price at that model's rate.
    assert spent == {
        "input_tokens": 80,
        "output_tokens": 10,
        "total_tokens": 90,
        "utility_input_tokens": 80,
        "utility_output_tokens": 10,
        # Two calls, neither of which said what it was billed (a stub, not OpenRouter).
        "calls": 2,
        "billed_calls": 0,
        "billed_nano_usd": 0,
    }


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


def _follow_up() -> ChatRequest:
    return _request().model_copy(
        update={"history": [Message(role="user", content="hello")]}
    )


async def test_a_rewrite_becomes_at_most_a_few_distinct_queries() -> None:
    """A model that does not stop would otherwise turn one message into a
    search, and an embedding call, per line."""
    reply = _Utility("basic fare\ncabin bag\nbasic fare\nseats\ncheck-in\npets\nwifi")

    with patch.object(retriever, "build_chat_model", return_value=reply) as built:
        queries = await rewrite_queries(_follow_up())

    assert queries == ["basic fare", "cabin bag", "seats", "check-in"]
    assert len(queries) == MAX_QUERIES
    assert built.call_args.args[0].max_output_tokens == REWRITE_MAX_TOKENS


async def test_a_rewrite_cut_off_by_its_cap_drops_the_unfinished_line() -> None:
    class Cut:
        async def ainvoke(self, messages, config=None):
            return AIMessage(
                content="basic fare\ncabin bag weig",
                response_metadata={"finish_reason": "length"},
            )

    with patch.object(retriever, "build_chat_model", return_value=Cut()):
        assert await rewrite_queries(_follow_up()) == ["basic fare"]


async def test_the_queries_are_searched_at_once(client: TestClient) -> None:
    """Each is an embedding call and a search; the turn waits for the
    slowest, not for their sum."""
    _seed(client)
    running = 0
    most = 0
    dense = retriever._dense

    async def slow(store, project_id, query, k):
        nonlocal running, most
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.05)
        try:
            return await dense(store, project_id, query, k)
        finally:
            running -= 1

    with patch.object(retriever, "_dense", slow):
        await retrieve_with_usage(
            _request(retrieval="hybrid"),
            queries=["basic fare", "cabin bag", "check-in"],
        )

    assert most == 3


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


# --- the minimum score --------------------------------------------------------


def _scored(**similarities: float):
    """A `_dense` that gives each seeded chunk the similarity named for its file."""

    async def dense(store, project_id, query, k):
        found = await original(store, project_id, query, k)
        return sorted(
            (
                (document, similarities.get(document.metadata["source"], 0.1))
                for document, _ in found
            ),
            key=lambda hit: hit[1],
            reverse=True,
        )

    original = retriever._dense
    return patch.object(retriever, "_dense", dense)


async def test_nothing_reaches_the_model_when_no_chunk_is_close(
    client: TestClient,
) -> None:
    """Small talk: even a keyword hit is dropped when no vector is near."""
    _seed(client)

    with (
        _scored(),
        patch.object(
            rerank_module, "rerank", side_effect=AssertionError("reranked nothing")
        ),
    ):
        hits = await retrieve(_request(retrieval="hybrid", min_score=0.4, rerank=True))

    assert hits == []


async def test_only_chunks_that_clear_the_bar_are_kept(client: TestClient) -> None:
    _seed(client)

    with _scored(**{"baggage.md": 0.7, "checkin.md": 0.35}):
        hits = await retrieve(_request(retrieval="vector", min_score=0.4, top_k=4))

    assert _sources(hits) == ["baggage.md"]


async def test_with_hybrid_a_keyword_hit_joins_a_question_that_is_on_topic(
    client: TestClient,
) -> None:
    """ "Basic fare" is matched by keyword alone, and kept once something is close."""
    _seed(client)

    with _scored(**{"baggage.md": 0.7, "fares.md": 0.2}):
        hits = await retrieve(_request(retrieval="hybrid", min_score=0.4, top_k=4))

    assert set(_sources(hits)) == {"baggage.md", "fares.md"}


async def test_without_a_minimum_every_chunk_is_kept_as_before(
    client: TestClient,
) -> None:
    _seed(client)

    with _scored():
        hits = await retrieve(_request(retrieval="vector", top_k=4))

    assert len(hits) == 4


# --- the score a hit is cited with ----------------------------------------------


async def test_a_hit_is_cited_with_its_similarity_not_its_place(
    client: TestClient,
) -> None:
    """Fusion orders by rank, and its best chunk always scored 1.0, which read
    as full confidence in whatever came first. The order stays the fused one;
    the score is how close the chunk is to the question."""
    _seed(client)

    with _scored(**{"baggage.md": 0.62, "fares.md": 0.31}):
        hits = await retrieve(_request(retrieval="hybrid", top_k=4))

    cited = dict(zip(_sources(hits), (score for _, score in hits), strict=True))
    assert cited == {
        "baggage.md": 0.62,
        "fares.md": 0.31,
        "checkin.md": 0.1,
        "seats.md": 0.1,
    }


async def test_a_chunk_only_the_keyword_search_found_is_cited_with_zero(
    client: TestClient,
) -> None:
    """It has no vector score to cite."""
    _seed(client)

    async def baggage_only(store, project_id, query, k):
        found = Document(
            page_content=CHUNKS["baggage.md"], metadata={"source": "baggage.md"}
        )
        return [(found, 0.6)]

    with patch.object(retriever, "_dense", baggage_only):
        hits = await retrieve(_request(retrieval="hybrid", top_k=4))

    cited = dict(zip(_sources(hits), (score for _, score in hits), strict=True))
    assert cited.pop("baggage.md") == 0.6
    assert "fares.md" in cited, "the keyword search's best match"
    assert set(cited.values()) == {0.0}


def test_the_similarity_is_the_cosine_for_vectors_of_unit_length() -> None:
    """Chroma's default space is squared L2: 2 - 2cos for unit vectors."""
    from chatbot_engine.agent.retriever import _similarity

    assert _similarity(0.0) == 1.0
    assert _similarity(1.0) == 0.5, "cos 0.5 is a squared distance of 1"
    assert _similarity(2.0) == 0.0, "orthogonal"
    assert _similarity(4.0) == 0.0, "opposite, floored at zero"
