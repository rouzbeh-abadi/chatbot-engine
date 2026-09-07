"""Load the eval datasets and the judge rubric.

The datasets are read as raw case dicts and forwarded to the engine unchanged:
the engine owns the case shapes and validates on receipt, so the backend only
reads its files and passes them on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The datasets and rubric are packaged with the code (they travel with the
# assistant they describe), so they resolve the same in a source checkout and in
# an installed image -- unlike a path outside the package, which is missing once
# the wheel is built.
EVALS_DIR = Path(__file__).parent / "data"
DEFAULT_DATASET = "system_prompt_cases.json"
DEFAULT_RAG_DATASET = "retrieval_cases.json"
JUDGE_PROMPT = "judge_prompt.md"


def _cases(name: str) -> list[dict[str, Any]]:
    """A dataset's cases as raw dicts, to forward to the engine unchanged."""
    return json.loads(_read(name))["cases"]


def load_judge_cases(name: str = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """The system-prompt cases."""
    return _cases(name)


def load_rag_cases(name: str = DEFAULT_RAG_DATASET) -> list[dict[str, Any]]:
    """The retrieval cases."""
    return _cases(name)


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
