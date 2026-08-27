"""Backend tests run without a live engine.

The engine is a remote service, so it is replaced through FastAPI's dependency
override rather than by patching internals.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fakes import FakeEngine
from fastapi.testclient import TestClient

from support_agent.app import app
from support_agent.engine import get_engine_client
from support_agent.api.rate_limit import reset_rate_limits
from support_agent.settings import get_settings


@pytest.fixture(autouse=True)
def _fresh_rate_limits() -> Iterator[None]:
    """Buckets are process-wide, so one test's spending must not reach the next."""
    reset_rate_limits()
    yield
    reset_rate_limits()


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(engine: FakeEngine) -> Iterator[TestClient]:
    """The default client: no admin key, so `/admin` is open as it is on localhost.

    Pinned rather than inherited, or a developer with `BACKEND_ADMIN_KEY` in
    their own .env would see every admin test fail with a 401.
    """
    settings = get_settings().model_copy(update={"admin_key": None})
    app.dependency_overrides[get_engine_client] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def proxied_client(engine: FakeEngine) -> Iterator[TestClient]:
    """A backend that believes `X-User-Id`, as it may only behind an authenticating proxy."""
    settings = get_settings().model_copy(
        update={"admin_key": None, "trust_user_header": True}
    )
    app.dependency_overrides[get_engine_client] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def admin_key() -> str:
    return "s3cret-operator-key"


@pytest.fixture
def guarded_client(engine: FakeEngine, admin_key: str) -> Iterator[TestClient]:
    """A client for a backend that requires `X-Admin-Key` on the admin routes.

    Settings are overridden rather than set through the environment: they are
    cached, and a `BACKEND_ADMIN_KEY` left behind in the process would leak into
    every later test.
    """
    settings = get_settings().model_copy(update={"admin_key": admin_key})
    app.dependency_overrides[get_engine_client] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def limited_client(engine: FakeEngine) -> Iterator[TestClient]:
    """A backend with the limits turned down to two calls, so a test can reach them."""
    settings = get_settings().model_copy(
        update={
            "admin_key": None,
            "chat_rate_limit_per_minute": 2,
            "eval_rate_limit_per_hour": 2,
        }
    )
    app.dependency_overrides[get_engine_client] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    reset_rate_limits()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
