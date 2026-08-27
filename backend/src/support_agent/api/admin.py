"""Admin dashboard endpoints: inspect the database and run evaluations.

Read-only views of the application data, plus a trigger for the system-prompt
evaluation. These are operator tools, not part of the chat product.

These routes are guarded by `BACKEND_ADMIN_KEY`: a shared operator secret, not
a login. This backend is a showcase for working with the engine, not a
production service - a real admin dashboard would sit behind real accounts and
per-user permissions. See the note in `api/chat.py`.
"""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from support_agent.api.auth import AdminOnly
from support_agent.api.rate_limit import limit_eval
from support_agent.assistant import load_project
from support_agent.database.connection import get_session_factory
from support_agent.database.models import Booking, SupportTicket
from support_agent.engine import EngineDep
from support_agent.engine_client.models import AssistantConfig
from support_agent.evals import (
    RagReport,
    Verdict,
    load_judge_cases,
    load_judge_prompt,
    load_rag_cases,
)

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[AdminOnly])

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
        EvalCaseInfo(
            id=case["id"], category=case["category"], question=case["question"]
        )
        for case in load_judge_cases()
    ]


class EvalRunResult(BaseModel):
    """The engine's graded run plus this app's pass mark, for the dashboard.

    The engine answers, grades, and averages; the only thing added here is the
    pass mark -- the product's own bar for what counts as a good enough answer.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = "system_prompt"
    model: str | None = None
    overall: float | None = None
    passed: int
    total: int
    pass_mark: int = PASS_MARK
    rows: list[Verdict]


@router.post("/eval/system-prompt", dependencies=[Depends(limit_eval)])
async def run_system_prompt_eval(
    engine: EngineDep,
    only: str | None = Query(
        default=None,
        description="Run just one category or one case id. Omit to run them all.",
    ),
) -> EvalRunResult:
    """Answer and grade the system-prompt dataset, then apply the pass mark.

    The backend only picks the cases and forwards them; the engine answers,
    grades, and averages. One model call per case, so a full run takes a while.
    `only` runs a single category (e.g. `refuse_advice`) for a quick check.
    """
    project = _read_only(load_project())
    cases = [
        case
        for case in load_judge_cases()
        if only in (None, case["category"], case["id"])
    ]

    report = await engine.judge(
        project=project,
        judge_prompt=load_judge_prompt(),
        cases=cases,
    )

    return EvalRunResult(
        model=report.model,
        overall=report.overall,
        passed=sum(
            1
            for verdict in report.verdicts
            if verdict.score is not None and verdict.score >= PASS_MARK
        ),
        total=len(report.verdicts),
        rows=report.verdicts,
    )


# --- RAG evaluation (RAGAS) --------------------------------------------------


@router.get("/eval/rag/cases")
async def list_rag_cases() -> list[EvalCaseInfo]:
    """The retrieval cases, so the UI can offer 'all', a category, or one case."""
    return [
        EvalCaseInfo(
            id=case["id"], category=case["category"], question=case["question"]
        )
        for case in load_rag_cases()
    ]


@router.post("/eval/rag", dependencies=[Depends(limit_eval)])
async def run_rag_eval(
    engine: EngineDep,
    only: str | None = Query(
        default=None,
        description="Run just one category or one case id. Omit to run them all.",
    ),
) -> RagReport:
    """Answer and score the retrieval dataset with RAGAS.

    The backend only picks the cases and forwards them as-is; the engine owns
    the case shape and validates it, then answers, scores, and summarises. Each
    case is answered then graded by several metric model calls, so a full run
    takes a while. `only` runs a single category or case for a quick check.
    """
    project = _read_only(load_project())
    cases = [
        case
        for case in load_rag_cases()
        if only in (None, case["category"], case["id"])
    ]

    return await engine.evaluate_rag(project=project, cases=cases)
