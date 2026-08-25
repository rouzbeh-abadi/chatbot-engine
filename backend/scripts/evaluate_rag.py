"""Score retrieval with RAGAS from the terminal (same run as the dashboard tab).

The backend only picks the cases and forwards them; the engine answers, scores,
and summarises. This just prints the finished report.

    make dev            # engine and backend
    make tools          # the MCP tool server
    make seed           # so retrieval has something to find
    make eval-rag       # this
    make eval-rag ARGS="--only follow_up"

Needs the engine's `eval` extra installed (RAGAS): `uv pip install -e "engine[eval]"`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from support_agent.assistant import load_project
from support_agent.engine import get_engine_client
from support_agent.engine_client import EngineError
from support_agent.evals import RagReport, load_rag_cases

METRICS = [
    ("faithfulness", "faith"),
    ("answer_relevancy", "answ_rel"),
    ("context_precision", "ctx_prec"),
    ("context_recall", "ctx_rec"),
]


def _cell(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "  -"


def render(report: RagReport) -> None:
    """Print the engine's report: per case, per category, then overall."""
    header = f"{'case':<26} {'category':<12} " + " ".join(
        f"{short:>8}" for _, short in METRICS
    )
    print("\n" + header)
    print("-" * len(header))

    for result in report.results:
        cells = " ".join(f"{_cell(getattr(result, m)):>8}" for m, _ in METRICS)
        print(f"{result.id:<26} {result.category:<12} {cells}")

    print("\nby category")
    for summary in report.by_category:
        cells = " ".join(
            f"{_cell(getattr(summary.averages, m)):>8}" for m, _ in METRICS
        )
        print(f"  {summary.category:<24} ({summary.count:>2}) {cells}")

    overall = " ".join(f"{_cell(getattr(report.overall, m)):>8}" for m, _ in METRICS)
    print(f"\n  {'overall':<24} ({len(report.results):>2}) {overall}")
    if report.model:
        print(f"\nmetrics by {report.model}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run one category, or one case id")
    args = parser.parse_args()

    cases = [
        case
        for case in load_rag_cases()
        if args.only in (None, case["category"], case["id"])
    ]
    if not cases:
        print(f"nothing matches {args.only!r}", file=sys.stderr)
        return 1

    print(f"{len(cases)} cases -- answering and scoring, several model calls each\n")
    try:
        report = await get_engine_client().evaluate_rag(
            project=load_project(), cases=cases
        )
    except EngineError as exc:
        print(f"\nRAG eval failed: {exc}", file=sys.stderr)
        print(
            "Needs a seeded engine with the `eval` extra installed.",
            file=sys.stderr,
        )
        return 1

    render(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
