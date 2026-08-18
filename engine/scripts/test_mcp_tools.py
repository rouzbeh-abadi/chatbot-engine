import asyncio

from chatbot_engine.mcp.client import McpToolProvider
from chatbot_engine.models.chat import AssistantConfig, McpServerConfig


async def main() -> None:
    """Discover and invoke tools from the backend MCP server."""

    config = AssistantConfig(
        project_id="support",
        name="Customer Support Assistant",
        system_prompt="You are a customer support assistant.",
        mcp_servers=[
            McpServerConfig(
                name="support-tools",
                url="http://localhost:8200/mcp",
                allowed_tools=[
                    "get_booking_status",
                    "get_flight_status",
                    "create_support_ticket",
                ],
            )
        ],
    )

    provider = McpToolProvider(timeout_s=30.0)

    result = await provider.call_tool(
        config=config,
        server="support-tools",
        name="get_booking_status",
        arguments={
            "booking_reference": "AB12CD",
        },
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())