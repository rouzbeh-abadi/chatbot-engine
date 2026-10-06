"""The RAGAS evaluation's own bookkeeping, with no judge and no provider.

The metrics and the answering model are stand-ins: what is pinned here is what
the evaluation does around them. The assistant it evaluates is not told its
tools are missing, a metric that cannot be scored is logged and counted
rather than lost, and the cases the knowledge base does not answer are
averaged apart from the rest.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from chatbot_engine.agent.client import empty_totals
from chatbot_engine.eval import rag_evaluation
from chatbot_engine.models.chat import AssistantConfig
from chatbot_engine.models.evals import RagEvalCase, RagEvalRequest


def _project(**config: object) -> AssistantConfig:
    return AssistantConfig(
        project_id="support", name="S", system_prompt="You help.", **config
    )


def _case(case_id: str, category: str = "single_turn") -> RagEvalCase:
    return RagEvalCase(
        id=case_id,
        category=category,
        question="Is the Basic fare refundable?",
        reference="No.",
    )


# --- the evaluated assistant ----------------------------------------------------


class _Answering(BaseChatModel):
    """Answers "No." and keeps the messages it was sent."""

    model_name: str = "openai/gpt-5-mini"
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "answering"

    def _generate(self, *args, **kwargs):  # pragma: no cover - streamed only
        raise NotImplementedError

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        yield ChatGenerationChunk(message=AIMessageChunk(content="No."))


async def test_the_assistant_is_evaluated_without_its_tools_and_is_not_told() -> None:
    """Offered no tools while allowing some, it would read in its prompt that
    every allowed tool is unavailable right now, a line no real turn has."""
    project = _project(
        mcp_servers=[
            {
                "name": "support-tools",
                "url": "http://tools.invalid/mcp",
                "allowed_tools": ["get_booking_status"],
            }
        ]
    )
    model = _Answering()

    async def retrieval(request, queries=None):
        assert request.project.mcp_servers == []
        return [], empty_totals()

    with (
        patch.object(rag_evaluation, "retrieve_with_usage", new=retrieval),
        patch("chatbot_engine.agent.client.build_chat_model", return_value=model),
    ):
        answer, question, contexts = await rag_evaluation._answer_and_contexts(
            project, _case("basic_fare")
        )

    assert (answer, question, contexts) == ("No.", "Is the Basic fare refundable?", [])
    system = model.seen[0][0].content
    assert "unavailable" not in system
    assert "get_booking_status" not in system


# --- scoring ----------------------------------------------------------------------


@dataclass
class _Result:
    value: object


class _Metric:
    """A stand-in for a RAGAS metric: a fixed value, or a failure."""

    def __init__(self, value: object = 1.0, *, fails: bool = False) -> None:
        self.value = value
        self.fails = fails

    async def ascore(self, **kwargs) -> _Result:
        if self.fails:
            raise RuntimeError("judge reply did not parse")
        return _Result(self.value)


async def _answered(project, case):
    return "No.", case.question, ["Basic fares are non-refundable."]


async def _evaluate(cases: list[RagEvalCase], metrics: tuple):
    with (
        patch.object(rag_evaluation, "_build_metrics", return_value=metrics),
        patch.object(rag_evaluation, "_answer_and_contexts", new=_answered),
    ):
        return await rag_evaluation._evaluate(
            RagEvalRequest(project=_project(), cases=cases)
        )


async def test_a_metric_that_cannot_be_scored_is_logged_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """It used to leave an empty cell and no trace of why."""
    metrics = (
        _Metric(fails=True),
        _Metric(math.nan),
        _Metric(0.8),
        _Metric(0.9),
    )

    with caplog.at_level(logging.WARNING, logger=rag_evaluation.__name__):
        report, unscored = await _evaluate([_case("a"), _case("b")], metrics)

    assert unscored == {
        "faithfulness": 2,
        "answer_relevancy": 2,
        "context_precision": 0,
        "context_recall": 0,
    }
    assert [r.faithfulness for r in report.results] == [None, None]
    assert report.overall.context_precision == 0.8
    messages = [record.getMessage() for record in caplog.records]
    assert "rag eval: no faithfulness for case a: judge reply did not parse" in messages
    assert "rag eval: no answer_relevancy for case b: the score was nan" in messages


async def test_the_report_carries_how_many_cases_each_metric_could_not_score() -> None:
    """On the wire, so a caller can say an average stands on fewer cases."""
    metrics = (_Metric(fails=True), _Metric(0.7), _Metric(0.8), _Metric(0.9))

    with (
        patch.object(rag_evaluation, "_build_metrics", return_value=metrics),
        patch.object(
            rag_evaluation,
            "_answer_and_contexts",
            new=_answered,
        ),
    ):
        report = await rag_evaluation.evaluate_rag_dataset(
            RagEvalRequest(project=_project(), cases=[_case("a")])
        )

    assert report.unscored == {
        "faithfulness": 1,
        "answer_relevancy": 0,
        "context_precision": 0,
        "context_recall": 0,
    }


async def test_negative_cases_are_averaged_apart_from_the_rest() -> None:
    """The right answer to one is a refusal, which these metrics cannot reward:
    averaged in, they would pull every overall number down by construction."""

    class ByCase(_Metric):
        async def ascore(self, **kwargs) -> _Result:
            negative = kwargs["user_input"] == "What is the weather in Paris?"
            return _Result(0.0 if negative else 1.0)

    negative = _case("weather", "negative").model_copy(
        update={"question": "What is the weather in Paris?"}
    )

    report, _ = await _evaluate(
        [_case("a"), _case("b", "follow_up"), negative], (ByCase(),) * 4
    )

    assert report.overall.faithfulness == 1.0
    assert report.overall.context_recall == 1.0
    by_category = {summary.category: summary for summary in report.by_category}
    assert by_category["negative"].count == 1
    assert by_category["negative"].averages.faithfulness == 0.0
