"""Application-owned domain tools exposed to the AI engine through MCP.

These run in the application backend, not in the engine, because they read this
application's business data and must eventually enforce the calling user's
permissions -- something the engine cannot evaluate. The engine discovers them
over MCP and invokes them; it never owns their implementation.

    make tools     # streamable HTTP on http://localhost:8200/mcp

Two conventions the tools follow, both aimed at the model that reads them:

* every value comes back as a string, in the words a customer would recognise
  ("cabin baggage only" rather than `null`), because the model relays these to a
  person and a bare null invites it to invent an explanation;
* a missing booking or flight is a normal answer, returned as data. Only an
  unusable *request* raises. A tool that raises for "not found" teaches the model
  that the tool is broken, and it stops trying.
"""

from __future__ import annotations

from datetime import date

from mcp.server.mcpserver import MCPServer
from sqlalchemy import select

from support_agent.database.connection import get_session_factory
from support_agent.database.models import Booking, Flight, SupportTicket

HOST = "0.0.0.0"
PORT = 8200

TICKET_CATEGORIES = ("refund", "baggage", "schedule_change", "complaint", "other")

mcp = MCPServer(
    name="support-tools",
    instructions=(
        "Domain tools for a travel support assistant. Use them to look up real "
        "booking and flight data instead of guessing, and to escalate to a human "
        "when the knowledge base cannot resolve the customer's problem."
    ),
)


@mcp.tool()
async def get_booking_status(
    booking_reference: str,
) -> dict[str, str]:
    """Look up one booking by its reference.

    Returns the passenger, route, travel date, fare type, checked baggage
    allowance, flight number, and current status. Use this whenever a customer
    mentions a booking reference, and use the flight number it returns to check
    that flight's status.

    Args:
        booking_reference: Six-character booking reference, e.g. "AB12CD".

    Returns:
        Booking details and its current status, or a not-found result.
    """
    reference = booking_reference.strip().upper()

    async with get_session_factory()() as session:
        booking = await session.scalar(
            select(Booking).where(Booking.booking_reference == reference)
        )

    if booking is None:
        return {
            "booking_reference": reference,
            "status": "not_found",
            "message": "No booking was found with this reference.",
        }

    result = {
        "booking_reference": booking.booking_reference,
        "passenger_name": booking.passenger_name,
        "route": f"{booking.origin} → {booking.destination}",
        "travel_date": booking.travel_date.isoformat(),
        "fare_type": booking.fare_type,
        "status": booking.status,
        "flight_number": booking.flight_number,
        "checked_baggage": booking.checked_baggage or "none (cabin baggage only)",
    }

    if booking.connecting_flight_number:
        result["connecting_flight_number"] = booking.connecting_flight_number
        result["itinerary"] = "two flights; refund rules may differ per segment"

    return result


@mcp.tool()
async def get_flight_status(
    flight_number: str,
    departure_date: str,
) -> dict[str, str]:
    """Check whether a flight is on time, delayed, or cancelled.

    Returns the scheduled departure, the expected departure when a flight is
    delayed, the gate once assigned, and a status. Use this for a specific flight
    on a specific date; it is not a schedule search. The flight number for a
    customer's booking comes from get_booking_status.

    Args:
        flight_number: Airline code and flight number, e.g. "SD204".
        departure_date: Departure date in ISO format, e.g. "2026-08-24".

    Returns:
        Current flight status and available timing information.
    """
    number = flight_number.strip().upper()

    try:
        parsed_date = date.fromisoformat(departure_date.strip())
    except ValueError:
        # An unusable argument, not a missing record: say what was wrong so the
        # model can retry with a corrected date rather than giving up.
        return {
            "flight_number": number,
            "status": "invalid_request",
            "message": (
                f"{departure_date!r} is not an ISO date. Use YYYY-MM-DD, "
                "e.g. 2026-08-24."
            ),
        }

    async with get_session_factory()() as session:
        flight = await session.scalar(
            select(Flight).where(
                Flight.flight_number == number,
                Flight.departure_date == parsed_date,
            )
        )

    if flight is None:
        return {
            "flight_number": number,
            "departure_date": parsed_date.isoformat(),
            "status": "not_found",
            "message": "No flight was found with this number on this date.",
        }

    result = {
        "flight_number": flight.flight_number,
        "departure_date": flight.departure_date.isoformat(),
        "route": f"{flight.origin} → {flight.destination}",
        "status": flight.status,
        "scheduled_departure": flight.scheduled_departure.isoformat(),
        "gate": flight.gate or "not assigned yet",
    }

    if flight.expected_departure is not None:
        result["expected_departure"] = flight.expected_departure.isoformat()

    return result


@mcp.tool()
async def create_support_ticket(
    booking_reference: str,
    summary: str,
    category: str,
) -> dict[str, str]:
    """Escalate to a human agent and return the new ticket reference.

    Use this only when the customer's problem cannot be resolved from the
    knowledge base or the other tools -- a refund dispute, a medical request, or a
    complaint. Summarise the issue in your own words; do not paste the whole
    conversation.

    Args:
        booking_reference: The booking the ticket relates to.
        summary: One or two sentences describing what the customer needs.
        category: One of "refund", "baggage", "schedule_change", "complaint",
            "other".

    Returns:
        The created ticket, or a rejection explaining what to fix.
    """
    reference = booking_reference.strip().upper()
    chosen = category.strip().lower()

    if chosen not in TICKET_CATEGORIES:
        return {
            "status": "invalid_request",
            "message": (
                f"{category!r} is not a valid category. Choose one of: "
                + ", ".join(TICKET_CATEGORIES)
            ),
        }

    if not summary.strip():
        return {
            "status": "invalid_request",
            "message": "A summary is required so the agent knows what to act on.",
        }

    async with get_session_factory()() as session:
        # Tickets hang off a real booking. Creating one for a reference that does
        # not exist would hand a human agent an unactionable ticket.
        booking = await session.scalar(
            select(Booking.booking_reference).where(
                Booking.booking_reference == reference
            )
        )
        if booking is None:
            return {
                "booking_reference": reference,
                "status": "not_found",
                "message": (
                    "No booking was found with this reference, so no ticket was "
                    "created. Confirm the reference with the customer."
                ),
            }

        ticket = SupportTicket(
            booking_reference=reference,
            summary=summary.strip(),
            category=chosen,
            status="open",
        )
        session.add(ticket)
        await session.commit()
        await session.refresh(ticket)

        return {
            "ticket_id": str(ticket.id),
            "booking_reference": ticket.booking_reference,
            "category": ticket.category,
            "status": ticket.status,
            "created_at": ticket.created_at.isoformat(),
            "message": "A support agent will follow up on this ticket.",
        }


def main() -> None:
    """Start the backend MCP tool server over streamable HTTP."""
    mcp.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
    )


if __name__ == "__main__":
    main()
