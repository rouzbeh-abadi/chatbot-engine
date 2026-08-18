"""This project's domain tools, served over MCP.

Tools run *here*, in the application backend, not in the engine:

* they read this application's business data, which the engine has no access to;
* they must execute with the calling user's permissions, which only this service
  can evaluate;
* they are domain logic, and the engine is meant to be domain-agnostic.

The engine connects as an MCP client using the `mcp_servers` block in
`projects/support.yaml`, and may only call the names listed in `allowed_tools`
there.

    make tools     # http://localhost:8200/mcp

Each signature below is complete and each docstring is written for the model --
the name, the type hints and the docstring *are* the schema the model reads when
deciding whether to call a tool. The bodies are yours to write.
"""

from __future__ import annotations

# MCP SDK 2.x. In 1.x this class was `FastMCP` in `mcp.server.fastmcp`.
from mcp.server.mcpserver import MCPServer

HOST = "0.0.0.0"
PORT = 8200

mcp = MCPServer(
    name="support-tools",
    instructions=(
        "Tools for a travel support assistant. Use them to look up real booking "
        "and flight data instead of guessing, and to escalate to a human."
    ),
)

_TODO = "{tool} is not implemented yet -- add the lookup in support_agent/mcp_tools.py"


@mcp.tool()
def get_booking_status(booking_reference: str) -> dict[str, str]:
    """Look up one booking by its reference.

    Returns the passenger name, route, travel date, fare type and current status
    (confirmed, cancelled, or awaiting payment). Use this whenever a customer
    mentions a booking reference, rather than asking them to repeat details.

    Args:
        booking_reference: The six-character booking reference, e.g. "AB12CD".
    """
    raise NotImplementedError(_TODO.format(tool="get_booking_status"))


@mcp.tool()
def get_flight_status(flight_number: str, departure_date: str) -> dict[str, str]:
    """Check whether a flight is on time, delayed, or cancelled.

    Returns the scheduled and expected times, the gate if one is assigned, and a
    status. Use this for questions about a specific flight today or in the near
    future; it is not a schedule search.

    Args:
        flight_number: Airline code and number, e.g. "SD204".
        departure_date: Departure date in ISO format, e.g. "2026-08-24".
    """
    raise NotImplementedError(_TODO.format(tool="get_flight_status"))


@mcp.tool()
def create_support_ticket(
    booking_reference: str, summary: str, category: str
) -> dict[str, str]:
    """Escalate to a human agent and return the new ticket id.

    Use this only when the customer's problem cannot be resolved from the
    knowledge base or the other tools -- for example a refund dispute, a medical
    request, or a complaint. Summarise the issue in your own words.

    Args:
        booking_reference: The booking the ticket relates to.
        summary: One or two sentences describing what the customer needs.
        category: One of "refund", "baggage", "schedule_change", "complaint",
            "other".
    """
    raise NotImplementedError(_TODO.format(tool="create_support_ticket"))


def main() -> None:
    mcp.run(transport="streamable-http", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
