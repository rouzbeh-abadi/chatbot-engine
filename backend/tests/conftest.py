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


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(engine: FakeEngine) -> Iterator[TestClient]:
    app.dependency_overrides[get_engine_client] = lambda: engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
