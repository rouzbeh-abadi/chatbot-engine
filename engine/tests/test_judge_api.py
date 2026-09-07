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
