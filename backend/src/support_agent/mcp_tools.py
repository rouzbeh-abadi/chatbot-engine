"""Application-owned domain tools exposed to the AI engine through MCP.

These tools run in the application backend because they operate on business
data and, eventually, must enforce the calling user's permissions. The AI engine
discovers and invokes them through MCP but does not own their implementation.

Run locally with:

    make tools

The MCP server is exposed over streamable HTTP on port 8200.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

HOST = "0.0.0.0"
PORT = 8200

mcp = MCPServer(
    name="support-tools",
    instructions="Domain tools for the customer support assistant.",
)


@mcp.tool()
def get_booking_status(
    booking_reference: str,
) -> dict[str, str]:
    """Look up one booking by its reference.

    Args:
        booking_reference: Six-character booking reference, e.g. "AB12CD".

    Returns:
        Booking details and its current status, or a not-found result.
    """
    bookings = {
        "AB12CD": {
            "booking_reference": "AB12CD",
            "passenger_name": "Daniel Miller",
            "route": "Berlin → Amsterdam",
            "travel_date": "2026-08-24",
            "fare_type": "Flex",
            "status": "confirmed",
        },
        "XY34ZT": {
            "booking_reference": "XY34ZT",
            "passenger_name": "Sarah Wilson",
            "route": "Paris → London",
            "travel_date": "2026-08-29",
            "fare_type": "Basic",
            "status": "cancelled",
        },
    }

    reference = booking_reference.strip().upper()
    booking = bookings.get(reference)

    if booking is None:
        return {
            "booking_reference": reference,
            "status": "not_found",
            "message": "No booking was found with this reference.",
        }

    return booking


@mcp.tool()
def get_flight_status(
    flight_number: str,
    departure_date: str,
) -> dict[str, str]:
    """Check whether a flight is on time, delayed, or cancelled.

    Args:
        flight_number: Airline code and flight number, e.g. "SD204".
        departure_date: Departure date in ISO format, e.g. "2026-08-24".

    Returns:
        Current flight status and available timing information.
    """
    raise NotImplementedError(
        "get_flight_status is not implemented yet."
    )


@mcp.tool()
def create_support_ticket(
    booking_reference: str,
    summary: str,
    category: str,
) -> dict[str, str]:
    """Create a support ticket for escalation to a human agent.

    Args:
        booking_reference: Booking reference associated with the issue.
        summary: Short description of the customer's problem.
        category: Support category such as refund, baggage, or complaint.

    Returns:
        Information about the newly created support ticket.
    """
    raise NotImplementedError(
        "create_support_ticket is not implemented yet."
    )


def main() -> None:
    """Start the backend MCP tool server over streamable HTTP."""
    mcp.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
    )


if __name__ == "__main__":
    main()