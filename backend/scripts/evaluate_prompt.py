"""Score the assistant's system prompt against a dataset of cases.

The dataset and the rubric live here; the engine does the work. It answers every
question with the same agent `/chat` uses -- retrieval, tools and all -- then
grades the finished run against the rubric and sends back a score per case.

    make dev            # engine and backend
    make tools          # the MCP tool server
    make seed           # so retrieval has something to find
    make eval           # this

Every finished run is saved to `backend/evals/last_run.json`, so `--show` can
print the grades again without asking the model anything.

`--dry-run` prints the cases it would send instead of running them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

from support_agent.assistant import load_project
from support_agent.engine import get_engine_client
from support_agent.engine_client import EngineError
from support_agent.evals import load_dataset, load_judge_prompt
from support_agent.evals.dataset import EVALS_DIR
from support_agent.evals.models import EvalCase, JudgeReport

BASE_URL = "http://localhost:8000"
PASS_MARK = 8
LAST_RUN = EVALS_DIR / "last_run.json"


def to_rows(cases: list[EvalCase], judged: JudgeReport) -> list[dict[str, object]]:
    """One flat row per case, which is all the report and the file both need."""
    by_id = {verdict.id: verdict for verdict in judged.verdicts}
    rows: list[dict[str, object]] = []

    for case in cases:
        verdict = by_id.get(case.id)
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "question": case.question,
                "score": None if verdict is None else verdict.score,
                "reason": "not judged" if verdict is None else verdict.reason,
                "answer": "" if verdict is None else verdict.answer,
            }
        )

    return rows


def save(rows: list[dict[str, object]], model: str | None) -> None:
    LAST_RUN.write_text(
        json.dumps({"model": model, "rows": rows}, indent=2) + "\n"
    )


def render(rows: list[dict[str, object]], model: str | None) -> int:
    """Print a score per case, then the totals. Returns the exit code."""
    categories: dict[str, list[int]] = {}
    failures = 0

    print(f"\n{'case':<20} {'category':<15} score  reason")
    print("-" * 100)

    for row in rows:
        score = row["score"]

        if score is None:
            print(f"{row['id']:<20} {row['category']:<15}     ?  {row['reason']}")
            failures += 1
            continue

        assert isinstance(score, int)
        categories.setdefault(str(row["category"]), []).append(score)
        if score < PASS_MARK:
            failures += 1

        mark = " " if score >= PASS_MARK else "!"
        print(
            f"{row['id']:<20} {row['category']:<15} {score:>2}/10{mark} "
            f"{str(row['reason'])[:56]}"
        )
        if score < PASS_MARK and row["answer"]:
            # A score with no answer beside it cannot be acted on.
            print(f"{'':<37}   said: {str(row['answer'])[:118]}")

    print("\nby category")
    for category, scores in sorted(categories.items()):
        print(f"  {category:<15} {sum(scores) / len(scores):.1f}/10   "
              f"({len(scores)} cases)")

    scored = [s for scores in categories.values() for s in scores]
    overall = sum(scored) / len(scored) if scored else 0.0
    print(f"\noverall {overall:.2f}/10 -- {failures} of {len(rows)} below "
          f"{PASS_MARK}/10")
    if model:
        print(f"judged by {model}")

    return 1 if failures else 0


def show_last_run() -> int:
    """Print the saved run. No questions asked, no model called."""
    if not LAST_RUN.is_file():
        print(f"no saved run at {LAST_RUN} -- run `make eval` first", file=sys.stderr)
        return 1

    saved = json.loads(LAST_RUN.read_text())
    print(f"saved run from {LAST_RUN}")

    return render(saved["rows"], saved.get("model"))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="system_prompt_cases.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the transcript instead of judging it",
    )
    parser.add_argument("--only", help="run one category, or one case id")
    parser.add_argument(
        "--show",
        action="store_true",
        help="print the last saved run instead of asking anything",
    )
    args = parser.parse_args()

    if args.show:
        return show_last_run()

    dataset = load_dataset(args.dataset)
    cases = [
        case
        for case in dataset.cases
        if args.only in (None, case.category, case.id)
    ]
    if not cases:
        print(f"nothing matches {args.only!r}", file=sys.stderr)
        return 1

    print(f"{dataset.name}: {len(cases)} cases")

    if args.dry_run:
        for case in cases:
            print(f"\n  {case.id} ({case.category})\n    Q: {case.question}"
                  f"\n    expected: {case.expected}")
        return 0

    print("asking and grading -- this makes one model call per case\n")
    try:
        judged = await get_engine_client().judge(
            project=load_project(),
            judge_prompt=load_judge_prompt(),
            cases=cases,
        )
    except EngineError as exc:
        print(f"\njudging failed: {exc}", file=sys.stderr)
        print(
            "\nIf the engine has no POST /judge yet, run with --dry-run to see "
            "the transcript it would receive.",
            file=sys.stderr,
        )
        return 1

    rows = to_rows(cases, judged)
    save(rows, judged.model)
    print(f"\nsaved to {LAST_RUN} -- re-read it with `make eval ARGS=--show`")

    return render(rows, judged.model)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
