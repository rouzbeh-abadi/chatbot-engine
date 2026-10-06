"""Prompt evaluation workflow.

Evaluation cases are answered by the same agents `/chat` runs, serialized
into a judge transcript, graded by the configured judge chain, and returned
as a report that preserves the assistant's original answers.

Answering a case must not act on the world: the agent that answers is given
`EvalToolProvider`, which offers the chatbot's tools as they are and runs
none of them, so a case about a booking books nothing. The calls the model
asks for are recorded and shown to the judge instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from chatbot_engine.agent.client import build_chat_model
from chatbot_engine.agent.judge_chain import create_judge_chain
from chatbot_engine.agent.router import DEADLINE_PASSED
from chatbot_engine.api.streaming import describe
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.evals import (
    EvalCase,
    JudgeReport,
    JudgeRequest,
    JudgeVerdicts,
    Verdict,
)
from chatbot_engine.models.events import ErrorEvent, TokenEvent
from chatbot_engine.ports.agent import Agent, ToolProvider

#: What the model reads as a tool's result while an evaluation answers a
#: case: neither a success nor a failure, so it answers with what it has.
NOT_RUN_IN_EVALUATION = "(not run in an evaluation)"


@dataclass(frozen=True)
class RecordedCall:
    """A tool call the model asked for while answering a case."""

    tool: str
    arguments: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.tool}({json.dumps(self.arguments, ensure_ascii=False)})"


#: Where the calls of the case being answered are recorded. One list per
#: case, set around its turn, so cases answered at the same time by one
#: provider never share a list; the graph agents' tasks inherit it.
_RECORDED: ContextVar[list[RecordedCall] | None] = ContextVar(
    "eval_recorded_calls", default=None
)


class EvalToolProvider:
    """The chatbot's tools as an evaluation sees them: offered, never run.

    `list_tools` asks the real provider, so the model is offered exactly what
    it would be offered in a chat. `call_tool` runs nothing: it records the
    call for the case being answered and returns `NOT_RUN_IN_EVALUATION`,
    since an evaluation of a support assistant must not raise tickets, send
    emails or change bookings, however many times it is run.
    """

    def __init__(self, tools: ToolProvider) -> None:
        self._tools = tools

    async def list_tools(self, config: AssistantConfig) -> Sequence[Mapping[str, Any]]:
        return await self._tools.list_tools(config)

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
        recorded = _RECORDED.get()
        if recorded is not None:
            recorded.append(RecordedCall(tool=name, arguments=dict(arguments)))
        return NOT_RUN_IN_EVALUATION


@dataclass(frozen=True)
class Answer:
    """What the assistant did with one case.

    `text` is what it said; `calls` the tools it asked for, none of which
    ran; `error` why its turn failed, when it did, in which case `text` is
    whatever it said before that and is not graded.
    """

    text: str
    calls: tuple[RecordedCall, ...] = ()
    error: str | None = None


def serialize_questions_for_judge(
    cases: list[EvalCase],
    answers: list[Answer],
) -> str:
    """Lay the run out as one block of text for the judge.

    A case whose turn failed is left out: there is no answer to grade, and
    the report marks it as an error instead.
    """
    blocks: list[str] = []

    for index, (case, answer) in enumerate(zip(cases, answers, strict=True), start=1):
        if answer.error is not None:
            continue
        calls = (
            "Tools the assistant called (not run in an evaluation, so it got no "
            f"result): {'; '.join(str(call) for call in answer.calls)}\n"
            if answer.calls
            else ""
        )
        blocks.append(
            f"### Case {index} (id: {case.id}, category: {case.category})\n"
            f"Question: {case.question}\n"
            f"Expected behaviour: {case.expected}\n"
            f"{calls}"
            f"Assistant answered: {answer.text.strip() or '(nothing)'}"
        )

    return "\n\n".join(blocks)


async def generate_answer(
    agent: Agent,
    project: AssistantConfig,
    case: EvalCase,
) -> Answer:
    """Ask one question through an agent `/chat` runs.

    The tools it calls are recorded (`EvalToolProvider`). A turn that ends
    with an `error` event, raises, or is stopped at its deadline
    (`ENGINE_TURN_DEADLINE_S`) is an error, not a bad answer: what went
    wrong is the provider's or the engine's, and grading it would lower the
    score for something the prompt did not do.
    """
    request = ChatRequest(project=project, message=case.question)
    recorded: list[RecordedCall] = []
    token = _RECORDED.set(recorded)
    passed: list[float] = []
    deadline = DEADLINE_PASSED.set(passed)
    said: list[str] = []
    error: str | None = None
    try:
        async for event in agent.run(request):
            if isinstance(event, TokenEvent):
                said.append(event.text)
            elif isinstance(event, ErrorEvent):
                error = f"{event.code}: {event.message}"
    except Exception as exc:
        error = describe(exc)
    finally:
        _RECORDED.reset(token)
        DEADLINE_PASSED.reset(deadline)
    if passed and error is None:
        error = f"the turn passed its deadline of {passed[0]:.0f}s"

    return Answer(text="".join(said), calls=tuple(recorded), error=error)


async def generate_answers(
    agent: Agent,
    project: AssistantConfig,
    cases: list[EvalCase],
) -> list[Answer]:
    """Answer every case, in order; a case that brings its own answer is not asked."""
    return [
        Answer(text=case.answer)
        if case.answer is not None
        else await generate_answer(agent, project, case)
        for case in cases
    ]


def judge_config(
    project: AssistantConfig, judge_model: str | None = None
) -> AssistantConfig:
    """The settings the grading call runs with.

    The project's model, or `judge_model` when one is given, at temperature
    0, so one run grades like the next, and without the project's answer
    cap, which is sized for answers, not for grading a whole run; on the
    same key. Every case is still graded in the one call.
    """
    return project.model_copy(
        update={
            "model": judge_model or project.model,
            "temperature": 0.0,
            "max_output_tokens": None,
        }
    )


async def judge_answers(
    project: AssistantConfig,
    judge_prompt: str,
    transcript: str,
    judge_model: str | None = None,
) -> tuple[JudgeVerdicts, str]:
    """Grade the run in one call, and report which model did it."""
    model = build_chat_model(judge_config(project, judge_model))
    chain = create_judge_chain(model, judge_prompt)
    graded: JudgeVerdicts = await chain.ainvoke({"transcript": transcript})

    return graded, model.model_name


def build_judge_report(
    cases: list[EvalCase],
    answers: list[Answer],
    graded: JudgeVerdicts,
    model: str | None,
) -> JudgeReport:
    """Join each case with its score and answer into a self-contained report.

    One verdict per case, in dataset order, so a case the judge skipped shows as
    `not judged` rather than vanishing, and a case whose turn failed shows as
    `error: ...` with no score. `overall` is the mean of the scores that came
    back.
    """
    said = dict(zip([case.id for case in cases], answers, strict=True))
    by_id = {verdict.id: verdict for verdict in graded.verdicts}

    verdicts: list[Verdict] = []
    for case in cases:
        answer = said.get(case.id, Answer(text=""))
        graded_verdict = None if answer.error is not None else by_id.get(case.id)
        if answer.error is not None:
            reason = f"error: {answer.error}"
        elif graded_verdict is not None:
            reason = graded_verdict.reason
        else:
            reason = "not judged"
        verdicts.append(
            Verdict(
                id=case.id,
                category=case.category,
                question=case.question,
                score=graded_verdict.score if graded_verdict else None,
                reason=reason,
                answer=answer.text,
            )
        )

    scored = [v.score for v in verdicts if v.score is not None]
    overall = round(sum(scored) / len(scored), 2) if scored else None

    return JudgeReport(verdicts=verdicts, overall=overall, model=model)


async def evaluate_dataset(
    request: JudgeRequest,
    agent: Agent,
) -> JudgeReport:
    """Answer every case that brought no answer, then grade the run.

    When every case failed there is nothing to grade, and no grading call
    is made.
    """
    answers = await generate_answers(agent, request.project, request.cases)
    transcript = serialize_questions_for_judge(request.cases, answers)
    if not transcript:
        return build_judge_report(request.cases, answers, JudgeVerdicts(), None)
    graded, model = await judge_answers(
        request.project, request.judge_prompt, transcript, request.judge_model
    )

    return build_judge_report(request.cases, answers, graded, model)
