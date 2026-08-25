"""Admin dashboard endpoints: inspect the database and run evaluations.

Read-only views of the application data, plus a trigger for the system-prompt
evaluation. These are operator tools, not part of the chat product.

Note: these routes carry no authentication. This backend is a showcase for
working with the engine, not a production service - a real admin dashboard
would sit behind a login. See the note in `api/chat.py`.
"""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from support_agent.assistant import load_project
from support_agent.database.connection import get_session_factory
from support_agent.database.models import Booking, SupportTicket
from support_agent.engine import EngineDep
from support_agent.engine_client.models import AssistantConfig
from support_agent.evals import load_dataset, load_judge_prompt

router = APIRouter(prefix="/admin", tags=["admin"])

#: A case scores well at or above this. Matches `scripts/evaluate_prompt.py`.
PASS_MARK = 8

#: Tools that change data. An evaluation grades behaviour, so it must not run
#: these -- otherwise every graded run leaves rows behind in the database.
WRITE_TOOLS = frozenset({"create_support_ticket"})


def _read_only(project: AssistantConfig) -> AssistantConfig:
    """A copy of the config with data-changing tools removed.

    The assistant can still *offer* to raise a ticket in its answer -- which is
    what the cases expect -- it just cannot actually create one during grading.
    """
    servers = []
    for server in project.mcp_servers:
        allowed = [t for t in server.allowed_tools if t not in WRITE_TOOLS]
        if allowed:
            servers.append(server.model_copy(update={"allowed_tools": allowed}))
    return project.model_copy(update={"mcp_servers": servers})


# --- database views ---------------------------------------------------------


class BookingRow(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    booking_reference: str
    passenger_name: str
    origin: str
    destination: str
    travel_date: date
    flight_number: str
    fare_type: str
    status: str
    checked_baggage: str | None = None


class TicketRow(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: int
    booking_reference: str
    category: str
    status: str
    summary: str
    created_at: datetime


@router.get("/bookings")
async def list_bookings() -> list[BookingRow]:
    """Every booking in the database."""
    async with get_session_factory()() as session:
        rows = (await session.scalars(select(Booking).order_by(Booking.id))).all()
    return [BookingRow.model_validate(row) for row in rows]


@router.get("/tickets")
async def list_tickets() -> list[TicketRow]:
    """Every support ticket, newest first."""
    async with get_session_factory()() as session:
        rows = (
            await session.scalars(
                select(SupportTicket).order_by(SupportTicket.created_at.desc())
            )
        ).all()
    return [TicketRow.model_validate(row) for row in rows]


# --- system-prompt evaluation -----------------------------------------------


class EvalCaseInfo(BaseModel):
    """One case, for the dashboard's run selector."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str


@router.get("/eval/system-prompt/cases")
async def list_eval_cases() -> list[EvalCaseInfo]:
    """The dataset's cases, so the UI can offer 'all', a category, or one case."""
    return [
        EvalCaseInfo(id=case.id, category=case.category, question=case.question)
        for case in load_dataset().cases
    ]


class EvalRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    score: int | None
    reason: str
    answer: str


class EvalRunResult(BaseModel):
    """The graded run plus a summary, for the dashboard to render."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "system_prompt"
    model: str | None = None
    overall: float
    passed: int
    total: int
    pass_mark: int = PASS_MARK
    rows: list[EvalRow]


@router.post("/eval/system-prompt")
async def run_system_prompt_eval(
    engine: EngineDep,
    only: str | None = Query(
        default=None,
        description="Run just one category or one case id. Omit to run them all.",
    ),
) -> EvalRunResult:
    """Answer and grade the system-prompt dataset, then summarise the scores.

    This makes one model call per case, so a full run takes a while. `only` runs
    a single category (e.g. `refuse_advice`) for a quick check.
    """
    project = _read_only(load_project())
    dataset = load_dataset()
    cases = [
        case for case in dataset.cases if only in (None, case.category, case.id)
    ]

    report = await engine.judge(
        project=project,
        judge_prompt=load_judge_prompt(),
        cases=cases,
    )

    by_id = {verdict.id: verdict for verdict in report.verdicts}
    rows: list[EvalRow] = []
    scored: list[int] = []

    for case in cases:
        verdict = by_id.get(case.id)
        score = verdict.score if verdict is not None else None
        if score is not None:
            scored.append(score)
        rows.append(
            EvalRow(
                id=case.id,
                category=case.category,
                question=case.question,
                score=score,
                reason="not judged" if verdict is None else verdict.reason,
                answer="" if verdict is None else verdict.answer,
            )
        )

    return EvalRunResult(
        model=report.model,
        overall=round(sum(scored) / len(scored), 2) if scored else 0.0,
        passed=sum(1 for s in scored if s >= PASS_MARK),
        total=len(rows),
        rows=rows,
    )
