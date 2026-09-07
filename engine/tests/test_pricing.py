"""Cost is computed in exactly one place, for every agent."""

from __future__ import annotations

import pytest

from chatbot_engine.agent.client import price_usage

TOTALS = {"input_tokens": 80, "output_tokens": 30, "total_tokens": 110}


def test_a_listed_model_is_priced_at_its_per_token_rate() -> None:
    # 80 in @ $0.25/1M + 30 out @ $2.00/1M.
    assert price_usage(TOTALS, "openai/gpt-5-mini").cost_usd == pytest.approx(0.00008)


def test_an_unlisted_model_reports_no_cost_rather_than_a_wrong_one() -> None:
    assert price_usage(TOTALS, "mystery/model").cost_usd is None
