"""The engine's startup contract.

The engine holds the provider credentials and has no notion of end users, so an
unauthenticated engine on a reachable port is an open provider account. That is
the one thing it must refuse to do.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.app import InsecureConfiguration, app
from chatbot_engine.settings import get_settings


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Set explicitly rather than trusted from this machine's .env."""
    get_settings.cache_clear()
    monkeypatch.setenv("ENGINE_API_KEY", "")
    yield monkeypatch
    get_settings.cache_clear()


def test_production_refuses_to_start_without_a_key(env: pytest.MonkeyPatch) -> None:
    env.setenv("ENGINE_ENV", "production")

    with pytest.raises(InsecureConfiguration, match="ENGINE_API_KEY"), TestClient(app):
        pass


def test_local_still_starts_without_any_configuration(
    env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    env.setenv("ENGINE_ENV", "local")

    with caplog.at_level("WARNING"), TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert "ENGINE_API_KEY" in caplog.text


def test_production_refuses_to_start_with_the_free_space_floor_off(
    env: pytest.MonkeyPatch,
) -> None:
    """With no floor an upload can fill the volume, and a full volume
    freezes embedded Chroma for every tenant."""
    env.setenv("ENGINE_ENV", "production")
    env.setenv("ENGINE_API_KEY", "s3cret")
    env.setenv("ENGINE_MIN_FREE_MB", "0")

    with (
        pytest.raises(InsecureConfiguration, match="ENGINE_MIN_FREE_MB=0"),
        TestClient(app),
    ):
        pass

    env.setenv("ENGINE_MIN_FREE_MB", "64")
    get_settings.cache_clear()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_a_second_engine_on_the_same_chroma_directory_is_refused(
    env: pytest.MonkeyPatch, tmp_path
) -> None:
    """Embedded Chroma keeps its indexes in the memory of the process that
    opened them: a second engine on the directory (or a second uvicorn
    worker) would write over the first one's. It refuses to start instead,
    and starts once the first has gone."""
    import os
    import subprocess
    import sys

    env.setenv("ENGINE_ENV", "local")
    env.setenv("ENGINE_CHROMA_DIR", str(tmp_path / "chroma"))
    second = [
        sys.executable,
        "-c",
        "from chatbot_engine.rag.vector_store import claim_chroma_dir; "
        "claim_chroma_dir()",
    ]
    with TestClient(app):
        refused = subprocess.run(
            second, env=dict(os.environ), capture_output=True, text=True, timeout=120
        )
    # Stopped, the first lets the directory go.
    started = subprocess.run(
        second, env=dict(os.environ), capture_output=True, text=True, timeout=120
    )

    assert refused.returncode != 0
    assert "another engine process holds" in refused.stderr
    assert started.returncode == 0, started.stderr[-2000:]
