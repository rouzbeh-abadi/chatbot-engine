"""The assistant: its configuration, loaded from YAML.

The backend owns this configuration; the engine stores none of it. Swapping YAML
for a database table changes only this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from support_agent.engine_client.models import AssistantConfig
from support_agent.settings import get_settings

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

    config = AssistantConfig.model_validate(
        yaml.safe_load(candidate.read_text()) or {}
    )
    return _apply_deployment_overrides(config)


def _apply_deployment_overrides(config: AssistantConfig) -> AssistantConfig:
    """Re-point the tool server for the environment we are actually running in.

    Only the address is environment-specific: under Docker Compose the engine
    dials `mcp-tools`, not localhost. Which tools are allowed is a security
    decision and stays in the YAML, where it is reviewable.
    """
    override = get_settings().mcp_tools_url
    if override is None or not config.mcp_servers:
        return config

    return config.model_copy(
        update={
            "mcp_servers": [
                server.model_copy(update={"url": override})
                for server in config.mcp_servers
            ]
        }
    )


def available_projects() -> list[str]:
    return sorted(path.stem for path in PROJECTS_DIR.glob("*.yaml"))
