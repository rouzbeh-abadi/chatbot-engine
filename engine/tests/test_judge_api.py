"""The evaluation contract: what a caller may send, and what comes back.

The judging itself is unwritten, so these pin the boundary -- validation, the
501 that names what to implement, and that a registered judge is reached.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from chatbot_engine.api import dependencies
from chatbot_engine.models.evals import JudgeReport, JudgeRequest, Verdict

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
            verdicts=[Verdict(id="greeting", score=10, reason="matches")],
            model="fake/judge",
        )


def _with_judge(client: TestClient, judge: object) -> None:
    client.app.dependency_overrides[dependencies.get_judge] = lambda: judge


def _without_judge(client: TestClient) -> None:
    client.app.dependency_overrides[dependencies.get_judge] = lambda: None


# --- unwired -----------------------------------------------------------------


def test_it_reports_no_judge_is_registered(
    client: TestClient, project: dict[str, object]
) -> None:
    _without_judge(client)

    response = client.post("/judge", json=_body(project))

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert "Judge" in detail
    assert "get_judge" in detail, "the 501 should name where to register it"


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


def test_a_malformed_case_is_rejected(
    client: TestClient, project: dict[str, object]
) -> None:
    response = client.post("/judge", json=_body(project, cases=[{"id": "x"}]))

    assert response.status_code == 422


def test_an_unknown_field_is_rejected(
    client: TestClient, project: dict[str, object]
) -> None:
    """`extra="forbid"`: a typo fails loudly instead of being dropped."""
    response = client.post("/judge", json=_body(project, judge_prmopt="oops"))

    assert response.status_code == 422


def test_the_project_is_required(client: TestClient) -> None:
    """It carries the model, so there is nothing to judge with without it."""
    response = client.post(
        "/judge",
        json={"judge_prompt": "r", "cases": [CASE]},
    )

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


def test_the_verdicts_come_back_as_the_caller_expects(
    client: TestClient, project: dict[str, object]
) -> None:
    _with_judge(client, _StubJudge())

    body = client.post("/judge", json=_body(project)).json()

    assert body == {
        "verdicts": [
            {"id": "greeting", "score": 10, "reason": "matches", "answer": ""}
        ],
        "model": "fake/judge",
    }


def test_a_score_outside_the_rubric_is_refused() -> None:
    """0-10 is the contract; a judge returning 11 is a bug worth catching."""
    import pytest

    with pytest.raises(ValueError):
        Verdict(id="x", score=11, reason="too high")
