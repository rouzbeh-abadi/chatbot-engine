"""Fold several RAG reports into one.

An evaluation is sent to the engine a few cases at a time, so that no single
request lasts long enough to hit a timeout, and a failure part-way keeps every
batch already paid for. The per-case results simply concatenate; the averages
are recomputed here over all of them, the same way the engine computes them.
"""

from __future__ import annotations

from collections import defaultdict

from support_agent.evals.models import (
    RagCaseResult,
    RagCategorySummary,
    RagMetricAverages,
    RagReport,
)

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _averages(results: list[RagCaseResult]) -> RagMetricAverages:
    """Mean of each metric over the cases that have it; None when none do."""
    values: dict[str, list[float]] = {}
    for metric in METRICS:
        values[metric] = [
            v for v in (getattr(r, metric) for r in results) if v is not None
        ]
    return RagMetricAverages(
        **{m: (sum(v) / len(v) if v else None) for m, v in values.items()}
    )


def merge_rag_reports(reports: list[RagReport]) -> RagReport:
    results = [case for report in reports for case in report.results]
    by_category: dict[str, list[RagCaseResult]] = defaultdict(list)
    for case in results:
        by_category[case.category].append(case)

    return RagReport(
        results=results,
        overall=_averages(results),
        by_category=[
            RagCategorySummary(category=c, count=len(cs), averages=_averages(cs))
            for c, cs in sorted(by_category.items())
        ],
        model=next((r.model for r in reports if r.model), None),
    )
