"""The assistant's YAML configuration.

This backend owns the configuration and validates it on load, so a typo fails
here with our error message rather than as a 422 from the engine.
"""

from __future__ import annotations

import pytest

from support_agent.assistant import (
    ProjectNotFoundError,
    available_projects,
    load_project,
)
from support_agent.engine_client.models import AssistantConfig


def test_support_project_loads() -> None:
    config = load_project("support")

    assert isinstance(config, AssistantConfig)
    assert config.project_id == "support"
    assert config.system_prompt.strip()


def test_support_project_is_listed() -> None:
    assert "support" in available_projects()


def test_unknown_project_raises() -> None:
    with pytest.raises(ProjectNotFoundError):
        load_project("does-not-exist")


def test_project_names_cannot_escape_the_config_directory() -> None:
    """Project names arrive over HTTP, so path traversal has to be impossible."""
    with pytest.raises(ProjectNotFoundError):
        load_project("../../../etc/passwd")


def test_declared_tools_are_allowlisted_not_open() -> None:
    """The engine dials these servers; an empty allowlist would let the tool
    server put arbitrary descriptions into the prompt."""
    config = load_project("support")

    assert config.mcp_servers, "the demo assistant declares its tool server"
    for server in config.mcp_servers:
        assert server.allowed_tools, f"{server.name} must pin its tool names"
