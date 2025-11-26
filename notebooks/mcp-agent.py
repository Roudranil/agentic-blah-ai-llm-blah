import asyncio

from mcp_agent.app import MCPApp
from mcp_agent.mcp.gen_client import gen_client
from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager

app = MCPApp(name="mcp_hello_world")


async def example_usage():
    async with app.run() as hello_world_app:
        context = hello_world_app.context
        logger = hello_world_app.logger

        logger.info("Hello, world!")

        # Use an async context manager to connect to the fetch server
        # At the end of the block, the connection will be closed automatically
        async with gen_client(
            "fetch", server_registry=context.server_registry
        ) as fetch_client:
            logger.info("fetch: Connected to server, calling list_tools...")
            result = await fetch_client.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            # Connect to the filesystem server using a persistent connection via connect/disconnect
            # This is useful when you need to make multiple requests to the same server

            connection_manager = MCPConnectionManager(context.server_registry)
            await connection_manager.__aenter__()
            server_names = ["filesystem", "fetch", "python-lsp", "github"]
            try:
                # Connect to all servers
                clients = {}
                for name in server_names:
                    client = await connection_manager.get_server(
                        server_name=name,
                        client_session_factory=MCPAgentClientSession,
                    )
                    logger.info(
                        f"{name}: Connected to server with persistent connection."
                    )
                    clients[name] = client

                # List tools from each
                for name, client in clients.items():
                    result = await client.session.list_tools()
                    logger.info(f"{name}: Tools available: {result.model_dump()}")

            finally:
                # Disconnect all
                for name in server_names:
                    await connection_manager.disconnect_server(server_name=name)
                    logger.info(f"{name}: Disconnected from server.")
                await connection_manager.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(example_usage())
