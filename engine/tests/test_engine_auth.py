"""Who may call the engine, and what the answer is counted against.

The engine holds the provider credentials, so this boundary is the one that
matters most in the package. Each test below is a property the implementation
must keep, not an implementation detail: constant-time matching, rotation
without downtime, and a caller name that rate limits and logs can use.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chatbot_engine.api.auth import OPEN_CALLER, Caller, _match
from chatbot_engine.api.dependencies import reset_dependency_cache
from chatbot_engine.api.rate_limit import RateLimiter, reset_rate_limits
from chatbot_engine.settings import Settings

# --- matching ----------------------------------------------------------------


def test_the_matching_key_is_identified_by_name() -> None:
    assert _match("b", {"web": "a", "batch": "b"}) == "batch"


def test_an_unknown_secret_matches_nothing() -> None:
    assert _match("nope", {"web": "a"}) is None


def test_a_non_ascii_key_is_refused_rather_than_crashing() -> None:
    """Headers are latin-1 decoded, and `compare_digest` raises on such a str.

    Without the encode in `_match` this is a 500 -- an unauthenticated caller
    crashing the request handler.
    """
    assert _match("clé-secrète", {"web": "a"}) is None


def test_a_prefix_of_the_real_key_does_not_match() -> None:
    """The check is whole-value equality, not a prefix or a startswith."""
    assert _match("s3cr", {"web": "s3cret"}) is None


# --- key configuration -------------------------------------------------------


def test_named_keys_are_parsed_into_callers() -> None:
    settings = Settings.model_construct(api_key=None, api_keys="web:a,batch:b")

    assert settings.credentials() == {"web": "a", "batch": "b"}


def test_the_single_key_shorthand_is_named_default() -> None:
    settings = Settings.model_construct(api_key="k", api_keys=None)

    assert settings.credentials() == {"default": "k"}


def test_rotation_accepts_the_old_and_new_key_at_once() -> None:
    """The point of naming keys: issue the new one, move callers, withdraw the old.

    With a single key every rotation is a synchronised restart of everything
    that calls the engine, which is why rotations do not happen.
    """
    credentials = Settings.model_construct(
        api_key=None, api_keys="web:old,web-next:new"
    ).credentials()

    assert _match("old", credentials) == "web"
    assert _match("new", credentials) == "web-next"


@pytest.mark.parametrize("entry", ["oops", "name:", ":secret"])
def test_a_malformed_entry_is_refused_rather_than_skipped(entry: str) -> None:
    """Skipping it would drop a key silently -- or drop the only key, opening the engine."""
    with pytest.raises(ValueError, match="ENGINE_API_KEYS"):
        Settings.model_construct(api_key=None, api_keys=entry).credentials()


def test_a_repeated_name_is_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        Settings.model_construct(api_key=None, api_keys="web:a,web:b").credentials()


# --- through the app ---------------------------------------------------------


@pytest.fixture
def keyed_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """An engine that accepts two named keys."""
    monkeypatch.setenv("ENGINE_API_KEY", "")
    monkeypatch.setenv("ENGINE_API_KEYS", "web:web-secret,batch:batch-secret")
    reset_dependency_cache()
    reset_rate_limits()

    from chatbot_engine.app import create_app

    return TestClient(create_app())


DOCUMENTS = ("/documents", {"project_id": "support"})


def test_every_configured_key_is_accepted(keyed_client: TestClient) -> None:
    url, params = DOCUMENTS

    for secret in ("web-secret", "batch-secret"):
        response = keyed_client.get(url, params=params, headers={"X-API-Key": secret})
        assert response.status_code == 200, secret


def test_an_unknown_key_is_refused(keyed_client: TestClient) -> None:
    url, params = DOCUMENTS

    assert keyed_client.get(url, params=params).status_code == 401
    assert (
        keyed_client.get(url, params=params, headers={"X-API-Key": "guess"}).status_code
        == 401
    )


def test_a_rejected_key_is_logged_without_logging_the_key(
    keyed_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed key is the only visible sign of someone guessing at one."""
    url, params = DOCUMENTS

    with caplog.at_level("WARNING"):
        keyed_client.get(url, params=params, headers={"X-API-Key": "hunter2"})

    assert "invalid X-API-Key" in caplog.text
    assert "hunter2" not in caplog.text


def test_health_stays_reachable_without_a_key(keyed_client: TestClient) -> None:
    """A probe must not need the secret, or the container never becomes healthy."""
    assert keyed_client.get("/health").status_code == 200


# --- attribution -------------------------------------------------------------


def test_rate_limits_are_counted_per_caller_not_globally() -> None:
    """One noisy client must not spend everyone else's allowance.

    This is the whole practical payoff of naming keys: without it every caller
    shares one bucket and the only available response to abuse is to turn the
    engine off for everybody.
    """
    limiter = RateLimiter(name="chat", capacity=1, window_s=60.0)

    limiter.check(Caller(name="web").name)
    limiter.check(Caller(name="batch").name)

    with pytest.raises(Exception) as caught:
        limiter.check(Caller(name="web").name)
    assert getattr(caught.value, "status_code", None) == 429


def test_an_open_engine_still_names_its_caller() -> None:
    """Logs and buckets need something to group by even with no keys set."""
    assert not Caller(name=OPEN_CALLER).is_authenticated
    assert Caller(name="web").is_authenticated
