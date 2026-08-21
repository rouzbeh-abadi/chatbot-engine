"""Load the dataset, and lay a finished run out for the judge."""

from __future__ import annotations

import json
from pathlib import Path

from support_agent.evals.models import EvalDataset

EVALS_DIR = Path(__file__).parents[3] / "evals"
DEFAULT_DATASET = "system_prompt_cases.json"
JUDGE_PROMPT = "judge_prompt.md"


def load_dataset(name: str = DEFAULT_DATASET) -> EvalDataset:
    """Read and validate one dataset file."""
    return EvalDataset.model_validate(json.loads(_read(name)))


def _read(name: str) -> str:
    path = (EVALS_DIR / name).resolve()

    if not path.is_relative_to(EVALS_DIR.resolve()):
        raise FileNotFoundError(f"eval file outside {EVALS_DIR}: {name!r}")
    if not path.is_file():
        raise FileNotFoundError(f"no eval file at {path}")

    return path.read_text()


def load_judge_prompt(name: str = JUDGE_PROMPT) -> str:
    """The rubric the judge grades against.

    The backend's, not the engine's: it is this backend's opinion of how its own
    assistant should behave. The engine stores no prompts, so it travels with the
    request like everything else.
    """
    return _read(name)
