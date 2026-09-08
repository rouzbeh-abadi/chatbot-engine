"""Cost is computed in exactly one place, from a table the operator owns."""

from __future__ import annotations

import pytest

from chatbot_engine.agent.client import price_usage
from chatbot_engine.settings import Settings

TOTALS = {"input_tokens": 80, "output_tokens": 30, "total_tokens": 110}
TABLE = {"openai/gpt-5-mini": (0.25, 2.0)}


def test_a_listed_model_is_priced_at_its_per_token_rate() -> None:
    # 80 in @ $0.25/1M + 30 out @ $2.00/1M.
    assert price_usage(TOTALS, "openai/gpt-5-mini", TABLE).cost_usd == pytest.approx(
        0.00008
    )


def test_an_unlisted_model_reports_no_cost_rather_than_a_wrong_one() -> None:
    assert price_usage(TOTALS, "mystery/model", TABLE).cost_usd is None


def test_the_engine_ships_no_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prices belong to the provider and change without notice, so they are
    configuration: an engine with none set shows no cost for any model."""
    monkeypatch.delenv("ENGINE_PRICING", raising=False)

    # `_env_file=None`: the developer's own .env may well list prices, and
    # this is about the default, not about that file.
    assert Settings(_env_file=None).pricing == {}
    assert price_usage(TOTALS, "openai/gpt-5-mini", {}).cost_usd is None


def test_the_table_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENGINE_PRICING", '{"x/y": [1.0, 2.0]}')

    assert Settings().pricing == {"x/y": (1.0, 2.0)}


def test_a_negative_price_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        Settings(pricing={"x/y": (-1.0, 2.0)})


def test_a_blank_env_var_means_no_prices_rather_than_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ENGINE_PRICING=` in a .env file, or a compose default of `${VAR:-}`,
    arrives as an empty string. It must mean "none", not fail at startup."""
    monkeypatch.setenv("ENGINE_PRICING", "")

    assert Settings().pricing == {}

    monkeypatch.setenv("ENGINE_PRICING", "   ")
    assert Settings().pricing == {}
