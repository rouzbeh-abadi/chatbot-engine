"""The assistant: its configuration, loaded from YAML.

The backend owns this configuration; the engine stores none of it. Swapping YAML
for a database table changes only this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from support_agent.engine_client.models import AssistantConfig

PROJECTS_DIR = Path(__file__).parent / "projects"
KNOWLEDGE_DIR = Path(__file__).parents[2] / "knowledge"
DEFAULT_PROJECT = "support"


class ProjectNotFoundError(LookupError):
    pass


@lru_cache
def load_project(name: str | None = None) -> AssistantConfig:
    """Read and validate one project's config.

    Validating here means a malformed prompt or MCP declaration fails with our
    error message, not somewhere deep in the engine.

    Cached, so restart the app after editing the YAML.
    """
    # Names arrive from HTTP, so keep them inside the config directory.
    candidate = (PROJECTS_DIR / f"{name or DEFAULT_PROJECT}.yaml").resolve()
    if not candidate.is_relative_to(PROJECTS_DIR.resolve()):
        raise ProjectNotFoundError(f"invalid project name: {name!r}")
    if not candidate.is_file():
        raise ProjectNotFoundError(f"no project config at {candidate}")

    return AssistantConfig.model_validate(yaml.safe_load(candidate.read_text()) or {})


def available_projects() -> list[str]:
    return sorted(path.stem for path in PROJECTS_DIR.glob("*.yaml"))
