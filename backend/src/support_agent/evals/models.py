"""The eval contract, shared with the engine's judge endpoint.

Deliberately separate from `engine_client.models`: that is the product's wire
contract, guarded by the parity test. Evaluation is a development tool, and
mixing the two would make every eval tweak a contract change.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    """One question, and what a good answer to it does."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    #: Prose, not an exact string: there are many good phrasings of one correct
    #: answer, and the judge compares behaviour rather than wording.
    expected: str


class EvalDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    cases: list[EvalCase] = Field(min_length=1)


class Verdict(BaseModel):
    """The judge's opinion of one answer."""

    model_config = ConfigDict(extra="forbid")

    id: str
    #: 0 = unsafe, 10 = does everything `expected` asks.
    score: int = Field(ge=0, le=10)
    #: One sentence. Long explanations make a failing run unreadable.
    reason: str
    #: What the assistant said, so a low score can be read without re-running it.
    answer: str = ""


class JudgeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict] = Field(default_factory=list)
    #: The judge's own model, so a run can be compared against another.
    model: str | None = None
