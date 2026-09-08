"""Batched evaluation reports fold into one that reads as if run at once."""

from __future__ import annotations

from support_agent.evals.merge import merge_rag_reports
from support_agent.evals.models import (
    RagCaseResult,
    RagCategorySummary,
    RagMetricAverages,
    RagReport,
)


def _case(id: str, category: str, precision: float | None) -> RagCaseResult:
    return RagCaseResult(
        id=id,
        category=category,
        question="q",
        answer="a",
        contexts=[],
        faithfulness=1.0,
        answer_relevancy=0.5,
        context_precision=precision,
        context_recall=1.0,
    )


def _report(*cases: RagCaseResult) -> RagReport:
    return RagReport(
        results=list(cases),
        overall=RagMetricAverages(),
        by_category=[
            RagCategorySummary(category="x", count=0, averages=RagMetricAverages())
        ],
        model="fake/judge",
    )


def test_cases_concatenate_and_averages_are_recomputed_over_all_of_them() -> None:
    merged = merge_rag_reports(
        [
            _report(_case("a", "single_turn", 1.0), _case("b", "follow_up", 0.5)),
            _report(_case("c", "single_turn", 0.0)),
        ]
    )

    assert [c.id for c in merged.results] == ["a", "b", "c"]
    assert merged.overall.context_precision == 0.5
    assert merged.overall.faithfulness == 1.0
    assert {c.category: c.count for c in merged.by_category} == {
        "follow_up": 1,
        "single_turn": 2,
    }
    single = next(c for c in merged.by_category if c.category == "single_turn")
    assert single.averages.context_precision == 0.5
    assert merged.model == "fake/judge"


def test_an_unscored_metric_is_left_out_of_the_mean_not_counted_as_zero() -> None:
    merged = merge_rag_reports([_report(_case("a", "x", 1.0), _case("b", "x", None))])

    assert merged.overall.context_precision == 1.0


def test_no_scores_at_all_is_none_not_a_division_error() -> None:
    merged = merge_rag_reports([_report(_case("a", "x", None))])

    assert merged.overall.context_precision is None
