"""Load and validate assistant configuration from `projects/*.yaml`.

The backend owns this config; the engine stores none of it. Loading it is
isolated here, so moving it from YAML to a database would change only this file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from support_agent.engine_client.models import AssistantConfig
from support_agent.settings import get_settings

PROJECTS_DIR = Path(__file__).parent / "projects"
PROMPTS_DIR = PROJECTS_DIR / "prompts"
KNOWLEDGE_DIR = Path(__file__).parents[2] / "knowledge"
DEFAULT_PROJECT = "support"


class ProjectNotFoundError(LookupError):
    """Raised when a project name does not resolve to a valid config file."""


@lru_cache
def load_project(name: str | None = None) -> AssistantConfig:
    """Load and validate one project's config; defaults to the `support` project.

    Validating here means a bad config fails with our own error, not a 422 from
    the engine. Cached, so restart the app after editing a YAML file.
    """
    # `name` may come from an HTTP request, so confine the path to PROJECTS_DIR
    # to prevent traversal (e.g. name="../../etc/passwd").
    candidate = (PROJECTS_DIR / f"{name or DEFAULT_PROJECT}.yaml").resolve()
    if not candidate.is_relative_to(PROJECTS_DIR.resolve()):
        raise ProjectNotFoundError(f"invalid project name: {name!r}")
    if not candidate.is_file():
        raise ProjectNotFoundError(f"no project config at {candidate}")

    raw = yaml.safe_load(candidate.read_text()) or {}
    config = AssistantConfig.model_validate(_read_prompt_file(raw))

    return _apply_deployment_overrides(config)


def _read_prompt_file(raw: dict[str, object]) -> dict[str, object]:
    """Resolve `system_prompt_file` into an inline `system_prompt`.

    Lets a project keep its (often long) prompt in a separate Markdown file.
    Inline `system_prompt` still works; setting both is an error.
    """
    filename = raw.pop("system_prompt_file", None)
    if filename is None:
        return raw

    if raw.get("system_prompt"):
        raise ProjectNotFoundError(
            "set system_prompt or system_prompt_file, not both"
        )

    path = (PROMPTS_DIR / str(filename)).resolve()
    if not path.is_relative_to(PROMPTS_DIR.resolve()):
        raise ProjectNotFoundError(f"prompt file outside {PROMPTS_DIR}: {filename!r}")
    if not path.is_file():
        raise ProjectNotFoundError(f"no prompt file at {path}")

    raw["system_prompt"] = path.read_text().strip()

    return raw


def _apply_deployment_overrides(config: AssistantConfig) -> AssistantConfig:
    """Override the MCP server URL for the current deployment, if configured.

    Only the address changes per environment (e.g. `mcp-tools` under Docker
    Compose instead of localhost). The tool allowlist stays in the YAML, where
    it is reviewable.
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
    """List the names of all configured projects."""
    return sorted(path.stem for path in PROJECTS_DIR.glob("*.yaml"))
