"""Prompt evaluation workflow.

Evaluation cases are answered through the same agent used by `/chat`, serialized
into a judge transcript, graded by the configured judge chain, and returned as a
report that preserves the assistant's original answers.
"""

from __future__ import annotations

from chatbot_engine.agent.Chains.judge_chain import create_judge_chain
from chatbot_engine.agent.client import build_chat_model
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.evals import (
    EvalCase,
    JudgeReport,
    JudgeRequest,
    JudgeVerdicts,
    Verdict,
)
from chatbot_engine.models.events import TokenEvent
from chatbot_engine.ports.agent import Agent


def serialize_questions_for_judge(
    cases: list[EvalCase],
    answers: list[str],
) -> str:
    """Lay the whole run out as one block of text for the judge."""
    blocks: list[str] = []

    for index, (case, answer) in enumerate(zip(cases, answers), start=1):
        blocks.append(
            f"### Case {index} (id: {case.id}, category: {case.category})\n"
            f"Question: {case.question}\n"
            f"Expected behaviour: {case.expected}\n"
            f"Assistant answered: {answer.strip() or '(nothing)'}"
        )

    return "\n\n".join(blocks)


async def generate_answer(
    agent: Agent,
    project: AssistantConfig,
    case: EvalCase,
) -> str:
    """Ask one question through the agent that serves /chat."""
    request = ChatRequest(project=project, message=case.question)

    return "".join(
        [
            event.text
            async for event in agent.run(request)
            if isinstance(event, TokenEvent)
        ]
    )


async def generate_answers(
    agent: Agent,
    project: AssistantConfig,
    cases: list[EvalCase],
) -> list[str]:
    """Answer every case, in order."""
    return [await generate_answer(agent, project, case) for case in cases]


async def judge_answers(
    project: AssistantConfig,
    judge_prompt: str,
    transcript: str,
) -> tuple[JudgeVerdicts, str]:
    """Grade the run in one call, and report which model did it."""
    model = build_chat_model(project)
    chain = create_judge_chain(model, judge_prompt)
    graded: JudgeVerdicts = await chain.ainvoke({"transcript": transcript})

    return graded, model.model_name


def build_judge_report(
    cases: list[EvalCase],
    answers: list[str],
    graded: JudgeVerdicts,
    model: str,
) -> JudgeReport:
    """Join each case with its score and answer into a self-contained report.

    One verdict per case, in dataset order, so a case the judge skipped shows as
    `not judged` rather than vanishing. `overall` is the mean of the scores that
    came back.
    """
    said = dict(zip([case.id for case in cases], answers))
    by_id = {verdict.id: verdict for verdict in graded.verdicts}

    verdicts: list[Verdict] = []
    for case in cases:
        graded_verdict = by_id.get(case.id)
        verdicts.append(
            Verdict(
                id=case.id,
                category=case.category,
                question=case.question,
                score=graded_verdict.score if graded_verdict else None,
                reason=graded_verdict.reason if graded_verdict else "not judged",
                answer=said.get(case.id, ""),
            )
        )

    scored = [v.score for v in verdicts if v.score is not None]
    overall = round(sum(scored) / len(scored), 2) if scored else None

    return JudgeReport(verdicts=verdicts, overall=overall, model=model)


async def evaluate_dataset(
    request: JudgeRequest,
    agent: Agent,
) -> JudgeReport:
    """Answer every case, then grade the run."""
    answers = await generate_answers(agent, request.project, request.cases)
    transcript = serialize_questions_for_judge(request.cases, answers)
    graded, model = await judge_answers(
        request.project, request.judge_prompt, transcript
    )

    return build_judge_report(request.cases, answers, graded, model)
