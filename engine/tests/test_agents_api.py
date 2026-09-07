"""GET /agents reports what is installed, not a fixed list."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from chatbot_engine.agent import registry


def test_an_installed_plugin_appears(client: TestClient) -> None:
    """Anything offering the choice to a user should see a new agent at once."""

    class EP:
        name = "my-graph"

        def load(self):
            return lambda tools: object()

    # Patching discovery replaces the real plugins, leaving the engine's own
    # built-in and this fake.
    with patch.object(registry, "entry_points", lambda group: [EP()]):
        assert client.get("/agents").json() == ["loop", "my-graph"]
