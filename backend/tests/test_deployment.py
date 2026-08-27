"""The startup contract: a production deployment must not boot on demo defaults.

These assertions are the difference between "we documented the risk" and "the
risk cannot ship". Each one corresponds to a way a real deployment has been
compromised: an open admin surface, an open engine, a database with the password
from the README.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from support_agent.app import InsecureConfiguration, app
from support_agent.settings import Settings, get_settings


def _settings(**overrides: object) -> Settings:
    """Settings built from explicit values only, ignoring any .env on this machine."""
    safe: dict[str, object] = {
        "admin_key": "an-admin-key",
        "engine_api_key": "an-engine-key",
        "database_url": "postgresql+psycopg://app:a-real-password@db:5432/app",
    }
    return Settings.model_construct(**{**safe, **overrides})


def test_a_fully_configured_backend_has_nothing_to_report() -> None:
    assert _settings().unsafe_for_production() == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"admin_key": None}, "BACKEND_ADMIN_KEY"),
        ({"engine_api_key": None}, "BACKEND_ENGINE_API_KEY"),
        (
            {
                "database_url": (
                    "postgresql+psycopg://support_agent:support_agent@db:5432/app"
                )
            },
            "BACKEND_DATABASE_URL",
        ),
    ],
)
def test_each_unsafe_default_is_reported_by_name(
    override: dict[str, object], expected: str
) -> None:
    """The message must name the variable to set -- it is all a stuck operator reads."""
    (problem,) = _settings(**override).unsafe_for_production()

    assert expected in problem


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Drive the real startup path through the environment, not an override.

    `lifespan` reads the cached settings directly -- there is no request, so
    there is no dependency to override -- so the cache is cleared around the
    test. Set here rather than trusted from this machine's .env, or the result
    depends on whose laptop runs it.
    """
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    monkeypatch.setenv("BACKEND_ADMIN_KEY", "")
    monkeypatch.setenv("BACKEND_ENGINE_API_KEY", "")
    yield monkeypatch
    get_settings.cache_clear()


def test_production_refuses_to_start_on_an_unsafe_default(
    env: pytest.MonkeyPatch,
) -> None:
    """A container that would leak on its first request must fail its health check."""
    env.setenv("BACKEND_ENV", "production")

    with pytest.raises(InsecureConfiguration, match="BACKEND_ADMIN_KEY"):
        with TestClient(app):
            pass


def test_local_serves_the_same_configuration_with_a_warning(
    env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`make dev` must still need no configuration at all -- but not quietly."""
    env.setenv("BACKEND_ENV", "local")

    with caplog.at_level("WARNING"):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

    assert "BACKEND_ADMIN_KEY" in caplog.text
