"""The assistant's YAML configuration.

This backend owns the configuration and validates it on load, so a typo fails
here with our error message rather than as a 422 from the engine.
"""

from __future__ import annotations

import pytest

from support_agent import assistant
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


def test_the_tool_server_url_can_be_overridden_for_a_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under Compose the engine dials `mcp-tools`, not localhost. Only the address
    is environment-specific -- the allowlist stays in the YAML, reviewable."""
    from support_agent import assistant, settings

    monkeypatch.setenv("BACKEND_MCP_TOOLS_URL", "http://mcp-tools:8200/mcp")
    settings.get_settings.cache_clear()
    assistant.load_project.cache_clear()
    try:
        server = assistant.load_project("support").mcp_servers[0]
        assert server.url == "http://mcp-tools:8200/mcp"
        assert "get_booking_status" in server.allowed_tools
    finally:
        settings.get_settings.cache_clear()
        assistant.load_project.cache_clear()


# --- the prompt in its own file ----------------------------------------------


def test_the_prompt_is_read_from_its_file() -> None:
    """A long prompt with examples belongs in Markdown, not a YAML block."""
    config = load_project("support")

    assert "SkyDesk" in config.system_prompt
    assert "## Examples" in config.system_prompt


def test_an_inline_prompt_still_works(tmp_path, monkeypatch) -> None:
    project = tmp_path / "inline.yaml"
    project.write_text(
        "project_id: p\nname: N\nsystem_prompt: straight from the yaml\n"
    )
    monkeypatch.setattr(assistant, "PROJECTS_DIR", tmp_path)
    load_project.cache_clear()

    assert load_project("inline").system_prompt == "straight from the yaml"


def test_setting_both_is_refused(tmp_path, monkeypatch) -> None:
    project = tmp_path / "both.yaml"
    project.write_text(
        "project_id: p\nname: N\nsystem_prompt: here\nsystem_prompt_file: support.md\n"
    )
    monkeypatch.setattr(assistant, "PROJECTS_DIR", tmp_path)
    load_project.cache_clear()

    with pytest.raises(ProjectNotFoundError, match="not both"):
        load_project("both")


def test_a_prompt_file_cannot_escape_the_prompts_directory(
    tmp_path, monkeypatch
) -> None:
    """The filename is ours, but a traversal guard costs one line."""
    project = tmp_path / "escape.yaml"
    project.write_text(
        "project_id: p\nname: N\nsystem_prompt_file: ../../../../etc/passwd\n"
    )
    monkeypatch.setattr(assistant, "PROJECTS_DIR", tmp_path)
    load_project.cache_clear()

    with pytest.raises(ProjectNotFoundError, match="outside"):
        load_project("escape")


def test_a_missing_prompt_file_says_so(tmp_path, monkeypatch) -> None:
    project = tmp_path / "gone.yaml"
    project.write_text("project_id: p\nname: N\nsystem_prompt_file: nope.md\n")
    monkeypatch.setattr(assistant, "PROJECTS_DIR", tmp_path)
    load_project.cache_clear()

    with pytest.raises(ProjectNotFoundError, match="no prompt file"):
        load_project("gone")
