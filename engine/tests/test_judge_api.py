"""The evaluation contract: what a caller may send, and what comes back.

A stub judge stands in for the real one, so these pin the boundary: validation,
and that the registered judge receives exactly what was sent.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from chatbot_engine.api import dependencies
from chatbot_engine.models.evals import (
    GradedVerdict,
    JudgeReport,
    JudgeRequest,
    Verdict,
)

CASE = {
    "id": "greeting",
    "category": "greeting",
    "question": "Hi",
    "expected": "Greets back.",
}


def _body(project: dict[str, object], **over: object) -> dict[str, object]:
    return {
        "project": project,
        "judge_prompt": "You are grading a support assistant.",
        "cases": [CASE],
    } | over


class _StubJudge:
    """Records the request, returns a fixed verdict. No model."""

    def __init__(self) -> None:
        self.seen: JudgeRequest | None = None

    async def __call__(self, request: JudgeRequest) -> JudgeReport:
        self.seen = request
        return JudgeReport(
            verdicts=[
                Verdict(
                    id="greeting",
                    category="greeting",
                    question="Hi",
                    score=10,
                    reason="matches",
                )
            ],
            overall=10.0,
            model="fake/judge",
        )


def _with_judge(client: TestClient, judge: object) -> None:
    client.app.dependency_overrides[dependencies.get_judge] = lambda: judge


# --- validation --------------------------------------------------------------


def test_an_empty_dataset_is_rejected(
    client: TestClient, project: dict[str, object]
) -> None:
    """Nothing to score is a caller mistake, not an empty report."""
    response = client.post("/judge", json=_body(project, cases=[]))

    assert response.status_code == 422


def test_an_empty_rubric_is_rejected(
    client: TestClient, project: dict[str, object]
) -> None:
    """Without a rubric the judge would invent its own standard."""
    response = client.post("/judge", json=_body(project, judge_prompt=""))

    assert response.status_code == 422


# --- wired -------------------------------------------------------------------


def test_a_registered_judge_receives_the_whole_request(
    client: TestClient, project: dict[str, object]
) -> None:
    stub = _StubJudge()
    _with_judge(client, stub)

    response = client.post("/judge", json=_body(project))

    assert response.status_code == 200
    assert stub.seen is not None
    assert stub.seen.judge_prompt.startswith("You are grading")
    assert [case.id for case in stub.seen.cases] == ["greeting"]
    assert stub.seen.cases[0].question == "Hi"
    assert stub.seen.project.project_id == "support"
    # And the report comes back as the wire contract says, defaults included.
    assert response.json()["verdicts"][0]["answer"] == ""
    assert response.json()["overall"] == 10.0


def test_a_score_outside_the_rubric_is_refused() -> None:
    """0-10 is the contract; a judge returning 11 is a bug worth catching."""
    import pytest

    with pytest.raises(ValueError):
        GradedVerdict(id="x", score=11, reason="too high")


def test_a_request_with_its_own_key_is_served_without_an_engine_key(
    client: TestClient, project: dict[str, object], monkeypatch
) -> None:
    """Bring-your-own-key: the 501 for a missing engine key does not apply."""
    monkeypatch.setenv("ENGINE_OPENROUTER_API_KEY", "")
    dependencies.reset_dependency_cache()
    stub = _StubJudge()
    _with_judge(client, stub)

    without = client.post("/judge", json=_body(project))
    with_key = client.post(
        "/judge", json=_body(project | {"provider_api_key": "caller-key"})
    )

    assert without.status_code == 501
    assert with_key.status_code == 200
    assert stub.seen is not None
    assert stub.seen.project.provider_api_key == "caller-key"


# --- a judge of its own, and answers supplied --------------------------------


def test_a_judge_model_and_a_supplied_answer_reach_the_judge(
    client: TestClient, project: dict[str, object]
) -> None:
    """Both are optional and pass through as sent."""
    stub = _StubJudge()
    _with_judge(client, stub)

    response = client.post(
        "/judge",
        json=_body(
            project,
            judge_model="anthropic/claude-sonnet-4.5",
            cases=[CASE | {"answer": "Hello!"}],
        ),
    )

    assert response.status_code == 200
    assert stub.seen is not None
    assert stub.seen.judge_model == "anthropic/claude-sonnet-4.5"
    assert stub.seen.cases[0].answer == "Hello!"


def test_an_empty_judge_model_is_rejected(
    client: TestClient, project: dict[str, object]
) -> None:
    """Empty is a caller mistake; leaving the field out means the project's model."""
    response = client.post("/judge", json=_body(project, judge_model=""))

    assert response.status_code == 422


class _CountingAgent:
    """Answers every question it is asked with the same text, and counts them."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def run(self, request):
        from chatbot_engine.models.events import TokenEvent

        self.asked.append(request.message)
        yield TokenEvent(text="Hi there.")


def _graded(monkeypatch, seen: dict[str, object]) -> None:
    """Stand in for the grading model: record its settings, give every case a 7."""
    from chatbot_engine.eval import prompt_evaluation
    from chatbot_engine.models.evals import JudgeVerdicts

    class _Model:
        def __init__(self, config) -> None:
            seen["config"] = config
            self.model_name = config.model or "project/default"

    class _Chain:
        def __init__(self) -> None:
            self.transcript = ""

        async def ainvoke(self, inputs: dict[str, str]) -> JudgeVerdicts:
            seen["transcript"] = inputs["transcript"]
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


def test_a_supplied_answer_is_graded_as_it_is_and_never_asked(
    monkeypatch, project: dict[str, object]
) -> None:
    import asyncio

    from chatbot_engine.eval.prompt_evaluation import evaluate_dataset
    from chatbot_engine.models.chat import AssistantConfig

    seen: dict[str, object] = {}
    _graded(monkeypatch, seen)
    agent = _CountingAgent()
    request = JudgeRequest(
        project=AssistantConfig(**project),
        judge_prompt="Grade it.",
        cases=[
            CASE,
            CASE
            | {
                "id": "known",
                "question": "Refund?",
                "answer": "Email refunds@evil.example.",
            },
        ],
    )

    report = asyncio.run(evaluate_dataset(request, agent=agent))

    assert agent.asked == ["Hi"]
    assert [v.answer for v in report.verdicts] == [
        "Hi there.",
        "Email refunds@evil.example.",
    ]
    assert "Assistant answered: Email refunds@evil.example." in str(seen["transcript"])


def test_a_judge_model_grades_at_temperature_0_without_the_answer_cap_and_the_project_answers(
    monkeypatch, project: dict[str, object]
) -> None:
    import asyncio

    from chatbot_engine.eval.prompt_evaluation import evaluate_dataset
    from chatbot_engine.models.chat import AssistantConfig

    seen: dict[str, object] = {}
    _graded(monkeypatch, seen)
    config = AssistantConfig(
        **project
        | {
            "model": "openai/gpt-5-mini",
            "temperature": 0.7,
            "max_output_tokens": 300,
            "provider_api_key": "caller-key",
        }
    )
    request = JudgeRequest(
        project=config,
        judge_prompt="Grade it.",
        cases=[CASE],
        judge_model="anthropic/claude-sonnet-4.5",
    )

    report = asyncio.run(evaluate_dataset(request, agent=_CountingAgent()))

    judged_with = seen["config"]
    assert isinstance(judged_with, AssistantConfig)
    assert (
        judged_with.model,
        judged_with.temperature,
        judged_with.max_output_tokens,
    ) == ("anthropic/claude-sonnet-4.5", 0.0, None)
    # Same key: the caller pays for the grading as for the answers.
    assert judged_with.provider_api_key == "caller-key"
    assert report.model == "anthropic/claude-sonnet-4.5"
    # The request's own project is left as it was.
    assert (request.project.model, request.project.temperature) == (
        "openai/gpt-5-mini",
        0.7,
    )


def test_without_a_judge_model_the_project_model_grades_as_before(
    monkeypatch, project: dict[str, object]
) -> None:
    import asyncio

    from chatbot_engine.eval.prompt_evaluation import evaluate_dataset
    from chatbot_engine.models.chat import AssistantConfig

    seen: dict[str, object] = {}
    _graded(monkeypatch, seen)
    config = AssistantConfig(
        **project | {"model": "openai/gpt-5-mini", "temperature": 0.7}
    )

    report = asyncio.run(
        evaluate_dataset(
            JudgeRequest(project=config, judge_prompt="Grade it.", cases=[CASE]),
            agent=_CountingAgent(),
        )
    )

    assert seen["config"] is config
    assert report.model == "openai/gpt-5-mini"
