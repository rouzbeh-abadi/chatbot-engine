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

import math

from chatbot_engine.agent.retriever import retrieve
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.evals import (
    RagCaseResult,
    RagCategorySummary,
    RagEvalCase,
    RagEvalRequest,
    RagMetricAverages,
    RagReport,
)
from chatbot_engine.models.events import TokenEvent
from chatbot_engine.ports.agent import Agent
from chatbot_engine.settings import get_settings
from openai import AsyncOpenAI

#: A different family than the assistant's `openai/gpt-5-mini`, so the metrics
#: are not self-graded. Must be a model the OpenRouter account can reach.
RAG_JUDGE_MODEL = "anthropic/claude-haiku-4.5"

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
    from chatbot_engine.Eval import _ragas_compat  # noqa: F401  patch before ragas
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.require_openrouter_key(),
        base_url=settings.openrouter_base_url,
    )
    llm = llm_factory(RAG_JUDGE_MODEL, provider="openai", client=client)
    embeddings = embedding_factory(
        provider="openai",
        model=settings.embedding_model,
        client=client,
        interface="modern",
    )

    return (
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecisionWithReference(llm=llm),
        ContextRecall(llm=llm),
    )


async def _answer_and_contexts(
    agent: Agent, project: AssistantConfig, case: RagEvalCase
) -> tuple[str, list[str]]:
    """Answer one case through the agent, and capture the chunks it retrieved.

    Retrieval runs a second time here to get the chunks at full length: the
    agent only emits them truncated, for the UI to cite.
    """
    request = ChatRequest(
        project=project, message=case.question, history=case.history
    )
    hits = await retrieve(request)
    contexts = [document.page_content for document, _ in hits]

    answer = "".join(
        [
            event.text
            async for event in agent.run(request)
            if isinstance(event, TokenEvent)
        ]
    )
    return answer, contexts


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


async def evaluate_rag_dataset(
    request: RagEvalRequest, agent: Agent
) -> RagReport:
    """Answer every case, then score its retrieval with RAGAS."""
    faithfulness, answer_relevancy, precision, recall = _build_metrics()

    results: list[RagCaseResult] = []
    for case in request.cases:
        answer, contexts = await _answer_and_contexts(
            agent, request.project, case
        )
        results.append(
            RagCaseResult(
                id=case.id,
                category=case.category,
                question=case.question,
                answer=answer,
                contexts=contexts,
                faithfulness=await _score(
                    faithfulness,
                    user_input=case.question,
                    response=answer,
                    retrieved_contexts=contexts,
                ),
                answer_relevancy=await _score(
                    answer_relevancy, user_input=case.question, response=answer
                ),
                context_precision=await _score(
                    precision,
                    user_input=case.question,
                    reference=case.reference,
                    retrieved_contexts=contexts,
                ),
                context_recall=await _score(
                    recall,
                    user_input=case.question,
                    retrieved_contexts=contexts,
                    reference=case.reference,
                ),
            )
        )

    return RagReport(
        results=results,
        overall=_averages(results),
        by_category=_by_category(results),
        model=RAG_JUDGE_MODEL,
    )
