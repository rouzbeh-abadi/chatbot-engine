"""An evaluation answers its cases without acting on the world.

`/judge` used to answer every case through the live agent and its real tool
provider, so an evaluation of a support assistant could raise tickets and
change bookings. Now the cases are answered by agents whose tools are listed
as in a chat and never run; the calls are recorded and shown to the judge. A
case whose turn failed is an error, not a bad answer, and the grading runs at
temperature 0 whichever model grades.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessageChunk, ToolMessage
from test_agent_parity import FakeTools, ScriptedModel, _no_retrieval

from chatbot_engine.api import dependencies
from chatbot_engine.eval import prompt_evaluation
from chatbot_engine.eval.prompt_evaluation import (
    NOT_RUN_IN_EVALUATION,
    EvalToolProvider,
    evaluate_dataset,
)
from chatbot_engine.models.chat import AssistantConfig
from chatbot_engine.models.evals import GradedVerdict, JudgeRequest, JudgeVerdicts
from chatbot_engine.models.events import DoneEvent, ErrorEvent, TokenEvent

PROJECT = AssistantConfig(
    project_id="support",
    name="S",
    system_prompt="You are helpful.",
    model="openai/gpt-5-mini",
)


def _case(case_id: str, question: str = "Is AB12 delayed?") -> dict:
    return {
        "id": case_id,
        "category": "booking",
        "question": question,
        "expected": "Looks the booking up.",
    }


class _Grading:
    """Stands in for the grading model: records what it was sent, gives a 7."""

    def __init__(self, monkeypatch) -> None:
        self.transcripts: list[str] = []
        self.configs: list[AssistantConfig] = []
        grading = self

        class _Model:
            def __init__(self, config) -> None:
                grading.configs.append(config)
                self.model_name = config.model or "project/default"

        class _Chain:
            async def ainvoke(self, inputs: dict[str, str]) -> JudgeVerdicts:
                grading.transcripts.append(inputs["transcript"])
                ids = [
                    line.split("id: ")[1].split(",")[0]
                    for line in inputs["transcript"].splitlines()
                    if line.startswith("### Case")
                ]
                return JudgeVerdicts(
                    verdicts=[GradedVerdict(id=i, score=7, reason="ok") for i in ids]
                )

        monkeypatch.setattr(prompt_evaluation, "build_chat_model", _Model)
        monkeypatch.setattr(
            prompt_evaluation, "create_judge_chain", lambda model, prompt: _Chain()
        )


class NeverRun(FakeTools):
    """Lists the tool; fails the test if anything tries to run it."""

    async def call_tool(self, *, name, **kwargs):
        raise AssertionError(f"an evaluation ran {name}")


async def test_the_judge_answers_with_tools_listed_and_never_run(monkeypatch):
    """Through the judge the engine wires up: the real loop agent, the model
    asking for a tool, the tool provider never asked to run it."""
    grading = _Grading(monkeypatch)
    dependencies.get_judge.cache_clear()
    asks = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "get_booking_status",
                "args": '{"booking_reference": "AB12"}',
                "id": "c1",
                "index": 0,
            }
        ],
    )
    model = ScriptedModel(
        rounds=[[asks], [AIMessageChunk(content="I could not check that now.")]],
        seen=[],
    )

    with (
        # Patched for this block only: the cache reset after the test needs
        # the real, cached factory back.
        patch.object(dependencies, "get_tool_provider", lambda: NeverRun()),
        patch("chatbot_engine.agent.client.build_chat_model", return_value=model),
        patch("chatbot_engine.agent.chat_agent.retrieve_with_usage", new=_no_retrieval),
    ):
        report = await dependencies.get_judge()(
            JudgeRequest(project=PROJECT, judge_prompt="Grade it.", cases=[_case("a")])
        )

    # The model read a neutral result, neither the tool's answer nor a failure.
    result = model.seen[1][-1]
    assert isinstance(result, ToolMessage) and result.content == NOT_RUN_IN_EVALUATION
    # The judge is shown the call, and that it did not run.
    assert (
        "Tools the assistant called (not run in an evaluation, so it got no result): "
        'get_booking_status({"booking_reference": "AB12"})'
    ) in grading.transcripts[0]
    assert report.verdicts[0].answer == "I could not check that now."
    assert report.verdicts[0].score == 7


class _ScriptedAgent:
    """Answers each question as scripted: words, a failure event, or a raise."""

    def __init__(self, script: dict[str, object]) -> None:
        self.script = script

    async def run(self, request):
        outcome = self.script[request.message]
        if isinstance(outcome, Exception):
            yield TokenEvent(text="Part")
            raise outcome
        if isinstance(outcome, ErrorEvent):
            yield TokenEvent(text="Half an answer")
            yield outcome
            yield DoneEvent(finish_reason="error")
            return
        yield TokenEvent(text=str(outcome))
        yield DoneEvent()


async def test_a_case_whose_turn_failed_is_an_error_not_a_bad_answer(monkeypatch):
    grading = _Grading(monkeypatch)
    agent = _ScriptedAgent(
        {
            "ok?": "Yes.",
            "event?": ErrorEvent(code="engine_error", message="provider timeout"),
            "raise?": RuntimeError("tool server reset"),
        }
    )
    request = JudgeRequest(
        project=PROJECT,
        judge_prompt="Grade it.",
        cases=[_case("a", "ok?"), _case("b", "event?"), _case("c", "raise?")],
    )

    report = await evaluate_dataset(request, agent=agent)

    by_id = {v.id: v for v in report.verdicts}
    assert (by_id["a"].score, by_id["a"].reason) == (7, "ok")
    assert by_id["b"].score is None
    assert by_id["b"].reason == "error: engine_error: provider timeout"
    assert by_id["b"].answer == "Half an answer"
    assert by_id["c"].score is None
    assert by_id["c"].reason == "error: RuntimeError: tool server reset"
    # Only the answer is graded, and only it counts towards the mean.
    assert "event?" not in grading.transcripts[0]
    assert "raise?" not in grading.transcripts[0]
    assert report.overall == 7.0


async def test_a_run_where_every_case_failed_makes_no_grading_call(monkeypatch):
    grading = _Grading(monkeypatch)
    agent = _ScriptedAgent({"raise?": RuntimeError("down")})

    report = await evaluate_dataset(
        JudgeRequest(
            project=PROJECT, judge_prompt="Grade it.", cases=[_case("c", "raise?")]
        ),
        agent=agent,
    )

    assert grading.transcripts == []
    assert report.overall is None and report.model is None
    assert report.verdicts[0].reason.startswith("error: ")


async def test_the_calls_are_recorded_per_case():
    """Each case's turn has its own record, so one case's calls never show
    against another's."""

    class Calling:
        def __init__(self, tools) -> None:
            self.tools = tools

        async def run(self, request):
            if request.message == "book?":
                await self.tools.call_tool(
                    config=request.project,
                    server="s",
                    name="book",
                    arguments={"slot": "Wed"},
                )
            yield TokenEvent(text="Done.")

    agent = Calling(EvalToolProvider(NeverRun()))
    answers = await prompt_evaluation.generate_answers(
        agent,
        PROJECT,
        [
            JudgeRequest.model_validate(
                {"project": PROJECT, "judge_prompt": "x", "cases": [_case("a", q)]}
            ).cases[0]
            for q in ("book?", "hello?")
        ],
    )

    assert [str(c) for c in answers[0].calls] == ['book({"slot": "Wed"})']
    assert answers[1].calls == ()


async def test_a_call_outside_any_evaluation_is_not_recorded_anywhere():
    tools = EvalToolProvider(NeverRun())

    result = await tools.call_tool(
        config=PROJECT, server="s", name="book", arguments={}
    )

    assert result == NOT_RUN_IN_EVALUATION
    assert await tools.list_tools(PROJECT) == await NeverRun().list_tools(PROJECT)
