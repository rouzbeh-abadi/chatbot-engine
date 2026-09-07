"""The domain tools, against a real database.

Skipped when no database is reachable, so `make test` stays green without Docker.
Run `make up && make migrate && make seed-db` first to exercise them.

What is asserted here is the tool *contract* the model depends on: that a missing
record is data rather than an exception, that a booking hands over the flight
number needed to chain the next call, and that bad arguments come back with an
explanation the model can act on.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from support_agent.database.connection import dispose_engine, get_session_factory
from support_agent.mcp_tools import (
    create_support_ticket,
    get_booking_status,
    get_flight_status,
)


def _database_is_reachable() -> bool:
    async def check() -> bool:
        try:
            async with get_session_factory()() as session:
                await session.execute(text("select 1 from bookings limit 1"))
            return True
        except Exception:
            return False
        finally:
            await dispose_engine()

    try:
        return asyncio.run(check())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _database_is_reachable(),
    reason="needs a seeded database: make up && make migrate && make seed-db",
)


# --- get_booking_status ------------------------------------------------------


async def test_a_booking_hands_over_the_flight_number_for_the_next_call() -> None:
    """This is what makes tool chaining possible: the model gets a booking, then
    uses its flight number to ask about the flight."""
    booking = await get_booking_status("AB12CD")

    assert booking["flight_number"] == "SD204"
    flight = await get_flight_status(
        booking["flight_number"], booking["travel_date"]
    )
    assert flight["status"] == "on_time"


async def test_booking_lookup_is_case_insensitive() -> None:
    assert (await get_booking_status("ab12cd"))["booking_reference"] == "AB12CD"


async def test_an_unknown_booking_is_data_not_an_exception() -> None:
    """A tool that raises for "not found" teaches the model the tool is broken."""
    result = await get_booking_status("NOPE99")

    assert result["status"] == "not_found"
    assert "message" in result


async def test_a_cabin_only_fare_says_so_in_words() -> None:
    """`None` would invite the model to invent an explanation."""
    assert "cabin baggage only" in (await get_booking_status("BG88QP"))["checked_baggage"]


async def test_a_connecting_itinerary_reports_both_legs() -> None:
    booking = await get_booking_status("MS55TR")

    assert booking["connecting_flight_number"] == "SD416"
    assert "segment" in booking["itinerary"]


# --- get_flight_status ------------------------------------------------------


async def test_a_delayed_flight_reports_an_expected_departure() -> None:
    booking = await get_booking_status("XY34ZT")
    flight = await get_flight_status("SD311", booking["travel_date"])

    assert flight["status"] == "delayed"
    assert "expected_departure" in flight


async def test_a_cancelled_flight_is_reported_as_cancelled() -> None:
    booking = await get_booking_status("RF77KL")
    flight = await get_flight_status("SD522", booking["travel_date"])

    assert flight["status"] == "cancelled"


async def test_a_non_iso_date_explains_the_expected_format() -> None:
    """The model can retry from this; a bare "invalid" leaves it guessing."""
    result = await get_flight_status("SD204", "24/08/2026")

    assert result["status"] == "invalid_request"
    assert "YYYY-MM-DD" in result["message"]


async def test_an_unknown_flight_is_data_not_an_exception() -> None:
    result = await get_flight_status("ZZ999", "2026-08-24")

    assert result["status"] == "not_found"


# --- create_support_ticket --------------------------------------------------


async def test_creating_a_ticket_returns_its_reference() -> None:
    result = await create_support_ticket(
        booking_reference="rf77kl",
        summary="Airline cancelled the flight; customer wants a refund.",
        category="REFUND",
    )

    assert result["status"] == "open"
    assert result["booking_reference"] == "RF77KL", "reference is normalised"
    assert result["category"] == "refund", "category is normalised"
    assert int(result["ticket_id"]) > 0


async def test_a_ticket_needs_a_real_booking() -> None:
    """A ticket against a reference that does not exist is unactionable."""
    result = await create_support_ticket(
        booking_reference="NOPE99", summary="help", category="refund"
    )

    assert result["status"] == "not_found"


async def test_an_unknown_category_lists_the_valid_ones() -> None:
    result = await create_support_ticket(
        booking_reference="AB12CD", summary="help", category="nonsense"
    )

    assert result["status"] == "invalid_request"
    assert "refund" in result["message"]


async def test_a_blank_summary_is_refused() -> None:
    result = await create_support_ticket(
        booking_reference="AB12CD", summary="   ", category="refund"
    )

    assert result["status"] == "invalid_request"
