"""Load the demo business data the AI engine exercises through the MCP tools.

Idempotent: re-running updates rows in place rather than duplicating them, so it
is safe after every migration.

    make migrate
    make seed-db

## Why the data looks like this

The backend exists to give the engine something realistic to work against. Every
booking below covers a case the knowledge base actually discusses, so a retrieved
policy can be applied to a concrete booking -- which is the interesting half of
RAG plus tool calling:

| Booking | Case it exercises | Knowledge base document |
| ------- | ----------------- | ----------------------- |
| AB12CD  | Flexible fare, refundable, inside the check-in window | refunds, check_in |
| XY34ZT  | Basic fare, non-refundable, already cancelled | refunds, cancellations |
| RF77KL  | Airline cancelled the flight -- involuntary refund | cancellations, refunds |
| MS55TR  | Two-leg itinerary, partial refund scope | refunds, booking_changes |
| BG88QP  | Cabin baggage only, no checked allowance | baggage |
| PS22WD  | Travel already completed -- refund window closed | refunds |

Dates are relative to today, not fixed. A hardcoded date drifts out of the
check-in window within days and the demo quietly stops being interesting.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from support_agent.database.connection import dispose_engine, get_session_factory
from support_agent.database.models import Booking, Flight

TODAY = date.today()


def _at(day: date, hour: int, minute: int) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=UTC)


#: Relative offsets, named so the intent survives someone changing the numbers.
TOMORROW = TODAY + timedelta(days=1)  # inside the 24-48h check-in window
SOON = TODAY + timedelta(days=3)
LATER = TODAY + timedelta(days=10)
CONNECTION_DAY = TODAY + timedelta(days=14)
FAR = TODAY + timedelta(days=21)
PAST = TODAY - timedelta(days=2)

FLIGHTS = [
    {
        "flight_number": "SD204",
        "departure_date": TOMORROW,
        "origin": "BER",
        "destination": "AMS",
        "status": "on_time",
        "scheduled_departure": _at(TOMORROW, 9, 35),
        "expected_departure": None,
        "gate": "B12",
    },
    {
        "flight_number": "SD630",
        "departure_date": SOON,
        "origin": "ARN",
        "destination": "CPH",
        "status": "on_time",
        "scheduled_departure": _at(SOON, 7, 20),
        "expected_departure": None,
        "gate": None,
    },
    {
        "flight_number": "SD311",
        "departure_date": LATER,
        "origin": "CDG",
        "destination": "LHR",
        "status": "delayed",
        "scheduled_departure": _at(LATER, 17, 10),
        "expected_departure": _at(LATER, 18, 45),
        "gate": None,
    },
    # The two legs of MS55TR.
    {
        "flight_number": "SD415",
        "departure_date": CONNECTION_DAY,
        "origin": "FCO",
        "destination": "LIS",
        "status": "on_time",
        "scheduled_departure": _at(CONNECTION_DAY, 11, 5),
        "expected_departure": None,
        "gate": "C4",
    },
    {
        "flight_number": "SD416",
        "departure_date": CONNECTION_DAY,
        "origin": "LIS",
        "destination": "MAD",
        "status": "on_time",
        "scheduled_departure": _at(CONNECTION_DAY, 15, 40),
        "expected_departure": None,
        "gate": None,
    },
    # Cancelled by the airline: an involuntary refund, which the policy treats
    # differently from a customer cancelling.
    {
        "flight_number": "SD522",
        "departure_date": FAR,
        "origin": "VIE",
        "destination": "ZRH",
        "status": "cancelled",
        "scheduled_departure": _at(FAR, 6, 55),
        "expected_departure": None,
        "gate": None,
    },
    {
        "flight_number": "SD741",
        "departure_date": PAST,
        "origin": "DUB",
        "destination": "MAN",
        "status": "departed",
        "scheduled_departure": _at(PAST, 13, 15),
        "expected_departure": None,
        "gate": "A7",
    },
]

BOOKINGS = [
    {
        "booking_reference": "AB12CD",
        "passenger_name": "Daniel Miller",
        "origin": "Berlin",
        "destination": "Amsterdam",
        "travel_date": TOMORROW,
        "fare_type": "Flexible",
        "status": "confirmed",
        "flight_number": "SD204",
        "connecting_flight_number": None,
        "checked_baggage": "1 x 23 kg",
    },
    {
        "booking_reference": "XY34ZT",
        "passenger_name": "Sarah Wilson",
        "origin": "Paris",
        "destination": "London",
        "travel_date": LATER,
        "fare_type": "Basic",
        "status": "cancelled",
        "flight_number": "SD311",
        "connecting_flight_number": None,
        "checked_baggage": None,
    },
    {
        "booking_reference": "RF77KL",
        "passenger_name": "Elena Fischer",
        "origin": "Vienna",
        "destination": "Zurich",
        "travel_date": FAR,
        "fare_type": "Standard",
        "status": "confirmed",
        "flight_number": "SD522",
        "connecting_flight_number": None,
        "checked_baggage": "1 x 23 kg",
    },
    {
        "booking_reference": "MS55TR",
        "passenger_name": "Marco Rossi",
        "origin": "Rome",
        "destination": "Madrid",
        "travel_date": CONNECTION_DAY,
        "fare_type": "Standard",
        "status": "confirmed",
        "flight_number": "SD415",
        "connecting_flight_number": "SD416",
        "checked_baggage": "1 x 23 kg",
    },
    {
        "booking_reference": "BG88QP",
        "passenger_name": "Anna Lindqvist",
        "origin": "Stockholm",
        "destination": "Copenhagen",
        "travel_date": SOON,
        "fare_type": "Basic",
        "status": "confirmed",
        "flight_number": "SD630",
        "connecting_flight_number": None,
        "checked_baggage": None,
    },
    {
        "booking_reference": "PS22WD",
        "passenger_name": "Tom Byrne",
        "origin": "Dublin",
        "destination": "Manchester",
        "travel_date": PAST,
        "fare_type": "Flexible",
        "status": "completed",
        "flight_number": "SD741",
        "connecting_flight_number": None,
        "checked_baggage": "1 x 23 kg",
    },
]


async def _upsert(
    session: AsyncSession, model: type, rows: list[dict], *match: str
) -> tuple[int, int]:
    """Insert rows that are absent, update the rest. `match` is the natural key."""
    created = updated = 0
    for row in rows:
        existing = await session.scalar(
            select(model).where(*(getattr(model, key) == row[key] for key in match))
        )
        if existing is None:
            session.add(model(**row))
            created += 1
        else:
            for field, value in row.items():
                setattr(existing, field, value)
            updated += 1
    return created, updated


async def main() -> int:
    try:
        async with get_session_factory()() as session:
            f_new, f_old = await _upsert(
                session, Flight, FLIGHTS, "flight_number", "departure_date"
            )
            b_new, b_old = await _upsert(
                session, Booking, BOOKINGS, "booking_reference"
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - a CLI should explain, not traceback
        print(f"seeding failed: {type(exc).__name__}: {exc}")
        print("is the database up and migrated?  'make up' then 'make migrate'")
        return 1
    finally:
        await dispose_engine()

    print(f"flights:  {f_new} created, {f_old} updated")
    print(f"bookings: {b_new} created, {b_old} updated")
    print(f"\ncheck-in window demo: {BOOKINGS[0]['booking_reference']} departs {TOMORROW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
