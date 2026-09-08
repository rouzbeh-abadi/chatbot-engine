"""RAG evaluation workflow, backed by RAGAS.

Where the prompt evaluation grades the assistant's *behaviour*, this grades the
*retrieval*: for each question it runs the same retrieve-then-answer path a chat
turn uses, then scores four things with RAGAS.

- **faithfulness** - is the answer supported by the retrieved context, or made up.
- **answer relevancy** - does the answer address the question that was asked.
- **context precision** - are the retrieved chunks relevant, best ones first.
- **context recall** - did retrieval find the context the reference answer needs.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI

from chatbot_engine.agent.client import stream_completion
from chatbot_engine.agent.retriever import (
    retrieve_with_usage,
    rewrite_queries,
    to_context,
)
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.evals import (
    RagCaseResult,
    RagCategorySummary,
    RagEvalCase,
    RagEvalRequest,
    RagMetricAverages,
    RagReport,
)
from chatbot_engine.settings import get_settings

logger = logging.getLogger(__name__)


class _NoTools:
    """A tool provider that offers nothing.

    Retrieval cases never call a tool, so binding none keeps the answer grounded
    in the retrieved context and skips the round-trip to the tool server.
    """

    async def list_tools(self, config: AssistantConfig) -> list[Mapping[str, Any]]:
        return []

    async def call_tool(
        self,
        *,
        config: AssistantConfig,
        server: str,
        name: str,
        arguments: Mapping[str, Any],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        return ""


_NO_TOOLS = _NoTools()

_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _averages(results: list[RagCaseResult]) -> RagMetricAverages:
    """Mean of each metric across `results`, skipping the ones scored null."""
    means: dict[str, float | None] = {}
    for metric in _METRICS:
        scores = [
            value
            for result in results
            if (value := getattr(result, metric)) is not None
        ]
        means[metric] = round(sum(scores) / len(scores), 3) if scores else None
    return RagMetricAverages(**means)


def _by_category(results: list[RagCaseResult]) -> list[RagCategorySummary]:
    """One averaged summary per category, in name order."""
    summaries: list[RagCategorySummary] = []
    for category in sorted({result.category for result in results}):
        in_category = [r for r in results if r.category == category]
        summaries.append(
            RagCategorySummary(
                category=category,
                count=len(in_category),
                averages=_averages(in_category),
            )
        )
    return summaries


def _build_metrics() -> tuple:
    """The four RAGAS metrics, wired to the judge model over OpenRouter.

    Collections metrics take a ragas-native LLM built from an OpenAI client, so
    we point one at OpenRouter rather than reusing the LangChain chat model.
    """
    # The shim must load before anything from ragas. A call rather than an
    # import statement, because an import sorter once moved it below the ragas
    # imports and broke the evaluation in a way nothing here exercised.
    importlib.import_module("chatbot_engine.eval._ragas_compat")

    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    settings = get_settings()
    # RAGAS fires many calls per case, so a rate-limited account (429s) will
    # otherwise exhaust the client's default two retries and leave metric cells
    # blank. More retries let the SDK back off and wait it out.
    client = AsyncOpenAI(
        api_key=settings.require_openrouter_key(),
        base_url=settings.openrouter_base_url,
        max_retries=6,
    )
    llm = llm_factory(settings.rag_judge_model, provider="openai", client=client)
    embeddings = embedding_factory(
        provider="openai",
        model=settings.embedding_model,
        client=client,
        interface="modern",
    )

    return (
        Faithfulness(llm=llm),
        # strictness=1: OpenRouter's Claude returns a single generation anyway,
        # so asking for the default three only adds cost.
        AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        ContextPrecisionWithReference(llm=llm),
        ContextRecall(llm=llm),
    )


async def _answer_and_contexts(
    project: AssistantConfig, case: RagEvalCase
) -> tuple[str, str, list[str]]:
    """Retrieve once, then answer from exactly those chunks.

    Using the same retrieval for the answer and for the scored contexts is both
    cheaper (one search, no tool discovery) and more correct: faithfulness then
    grades the answer against the context it was actually generated from.

    Returns the question the metrics should judge against as well. For a
    follow-up that is the rewritten, self-contained form ("how long must the
    passport stay valid", not "how long does it need to stay valid?"): the
    metrics see one message, not the conversation, and would otherwise mark a
    correct answer as off-topic for answering a question they cannot see. A
    first-turn question is returned as it was asked.
    """
    request = ChatRequest(project=project, message=case.question, history=case.history)
    queries = await rewrite_queries(request)
    hits, spent = await retrieve_with_usage(request, queries=queries)
    contexts = [document.page_content for document, _ in hits]

    answer = "".join(
        [
            item
            async for item in stream_completion(
                request, _NO_TOOLS, to_context(hits), prior=spent
            )
            if isinstance(item, str)
        ]
    )
    return answer, queries[0], contexts


async def _score(metric, /, **kwargs) -> float | None:
    """One metric's value for one case, or None if it could not be computed.

    A single metric failing (an empty retrieval, a model hiccup) should leave a
    gap in the row, not abort the whole run.
    """
    try:
        result = await metric.ascore(**kwargs)
    except Exception:
        return None

    value = result.value
    if isinstance(value, (int, float)) and not math.isnan(value):
        return float(value)
    return None


async def evaluate_rag_dataset(request: RagEvalRequest) -> RagReport:
    """Answer every case, then score its retrieval with RAGAS."""
    faithfulness, answer_relevancy, precision, recall = _build_metrics()

    results: list[RagCaseResult] = []
    total = len(request.cases)
    for index, case in enumerate(request.cases, start=1):
        answer, question, contexts = await _answer_and_contexts(request.project, case)
        # Scoring a case is twenty-odd judge calls. Say when each finishes, so
        # an operator can see an evaluation move rather than guess at it.
        logger.info("rag eval: scoring case %d/%d %s", index, total, case.id)
        results.append(
            RagCaseResult(
                id=case.id,
                category=case.category,
                question=case.question,
                answer=answer,
                contexts=contexts,
                faithfulness=await _score(
                    faithfulness,
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                ),
                answer_relevancy=await _score(
                    answer_relevancy, user_input=question, response=answer
                ),
                context_precision=await _score(
                    precision,
                    user_input=question,
                    reference=case.reference,
                    retrieved_contexts=contexts,
                ),
                context_recall=await _score(
                    recall,
                    user_input=question,
                    retrieved_contexts=contexts,
                    reference=case.reference,
                ),
            )
        )

    return RagReport(
        results=results,
        overall=_averages(results),
        by_category=_by_category(results),
        model=get_settings().rag_judge_model,
    )
