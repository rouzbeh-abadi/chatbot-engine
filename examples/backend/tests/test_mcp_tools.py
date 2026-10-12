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
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from support_agent.database.connection import dispose_engine, get_session_factory
from support_agent.mcp_tools import (
    create_support_ticket,
    get_booking_status,
    get_flight_status,
)

#: The engine forwards the caller's identifiers as headers; the tools read
#: nothing else from the context.
_CTX = SimpleNamespace(headers={"x-request-id": "test-run"})


def _database_is_reachable() -> bool:
    """A reachable, migrated *and seeded* database.

    The tests read the demo bookings, so an empty table is as useless to them
    as no table: both skip, with the same instruction.
    """

    async def check() -> bool:
        try:
            async with get_session_factory()() as session:
                row = await session.execute(text("select 1 from bookings limit 1"))
                return row.first() is not None
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
    booking = await get_booking_status(_CTX, "AB12CD", "Miller")

    assert booking["flight_number"] == "SD204"
    flight = await get_flight_status(
        _CTX, booking["flight_number"], booking["travel_date"]
    )
    assert flight["status"] == "on_time"


async def test_booking_lookup_is_case_insensitive() -> None:
    booking = await get_booking_status(_CTX, "ab12cd", "MILLER")

    assert booking["booking_reference"] == "AB12CD"


async def test_a_reference_alone_asks_for_the_last_name() -> None:
    """Six characters that travel on every confirmation are not enough to hand
    over a passenger's name, route and date."""
    result = await get_booking_status(_CTX, "AB12CD")

    assert result["status"] == "last_name_required"
    assert "passenger_name" not in result


async def test_a_wrong_last_name_reads_as_an_unknown_reference() -> None:
    """The same answer as a reference that does not exist, so the tool cannot be
    used to learn which references do."""
    wrong = await get_booking_status(_CTX, "AB12CD", "Wilson")
    unknown = await get_booking_status(_CTX, "NOPE99", "Wilson")

    assert {k: v for k, v in wrong.items() if k != "booking_reference"} == {
        k: v for k, v in unknown.items() if k != "booking_reference"
    }
    assert "passenger_name" not in wrong


async def test_an_unknown_booking_is_data_not_an_exception() -> None:
    """A tool that raises for "not found" teaches the model the tool is broken."""
    result = await get_booking_status(_CTX, "NOPE99", "Miller")

    assert result["status"] == "not_found"
    assert "message" in result


async def test_a_cabin_only_fare_says_so_in_words() -> None:
    """`None` would invite the model to invent an explanation."""
    assert (
        "cabin baggage only"
        in (await get_booking_status(_CTX, "BG88QP", "Lindqvist"))["checked_baggage"]
    )


async def test_a_connecting_itinerary_reports_both_legs() -> None:
    booking = await get_booking_status(_CTX, "MS55TR", "Rossi")

    assert booking["connecting_flight_number"] == "SD416"
    assert "segment" in booking["itinerary"]


# --- get_flight_status ------------------------------------------------------


async def test_a_delayed_flight_reports_an_expected_departure() -> None:
    booking = await get_booking_status(_CTX, "XY34ZT", "Wilson")
    flight = await get_flight_status(_CTX, "SD311", booking["travel_date"])

    assert flight["status"] == "delayed"
    assert "expected_departure" in flight


async def test_a_cancelled_flight_is_reported_as_cancelled() -> None:
    booking = await get_booking_status(_CTX, "RF77KL", "Fischer")
    flight = await get_flight_status(_CTX, "SD522", booking["travel_date"])

    assert flight["status"] == "cancelled"


async def test_a_non_iso_date_explains_the_expected_format() -> None:
    """The model can retry from this; a bare "invalid" leaves it guessing."""
    result = await get_flight_status(_CTX, "SD204", "24/08/2026")

    assert result["status"] == "invalid_request"
    assert "YYYY-MM-DD" in result["message"]


async def test_an_unknown_flight_is_data_not_an_exception() -> None:
    result = await get_flight_status(_CTX, "ZZ999", "2026-08-24")

    assert result["status"] == "not_found"


# --- create_support_ticket --------------------------------------------------


async def test_creating_a_ticket_returns_its_reference() -> None:
    result = await create_support_ticket(
        _CTX,
        booking_reference="rf77kl",
        summary="Airline cancelled the flight; customer wants a refund.",
        category="REFUND",
        last_name="Fischer",
    )

    assert result["status"] == "open"
    assert result["booking_reference"] == "RF77KL", "reference is normalised"
    assert result["category"] == "refund", "category is normalised"
    assert int(result["ticket_id"]) > 0


async def test_a_ticket_needs_a_real_booking() -> None:
    """A ticket against a reference that does not exist is unactionable."""
    result = await create_support_ticket(
        _CTX,
        booking_reference="NOPE99",
        summary="help",
        category="refund",
        last_name="Fischer",
    )

    assert result["status"] == "not_found"


async def test_a_ticket_needs_the_passengers_last_name_and_a_wrong_one_finds_nothing() -> (
    None
):
    """As for `get_booking_status`: a reference alone files nothing, and a
    wrong name reads exactly as an unknown reference."""
    without = await create_support_ticket(
        _CTX, booking_reference="RF77KL", summary="help", category="refund"
    )
    wrong = await create_support_ticket(
        _CTX,
        booking_reference="RF77KL",
        summary="help",
        category="refund",
        last_name="Miller",
    )
    unknown = await create_support_ticket(
        _CTX,
        booking_reference="NOPE99",
        summary="help",
        category="refund",
        last_name="Miller",
    )

    assert without["status"] == "last_name_required"
    assert wrong["status"] == "not_found"
    assert wrong["message"] == unknown["message"]


async def test_an_unknown_category_lists_the_valid_ones() -> None:
    result = await create_support_ticket(
        _CTX, booking_reference="AB12CD", summary="help", category="nonsense"
    )

    assert result["status"] == "invalid_request"
    assert "refund" in result["message"]


async def test_a_blank_summary_is_refused() -> None:
    result = await create_support_ticket(
        _CTX, booking_reference="AB12CD", summary="   ", category="refund"
    )

    assert result["status"] == "invalid_request"
