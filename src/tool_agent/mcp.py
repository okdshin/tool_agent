import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import mcp

logger = logging.getLogger(__name__)


@dataclass
class McpServerConfig:
    command: str
    arguments: list[str] = field(default_factory=list)


class McpClient:
    def __init__(self, config: McpServerConfig):
        self.server_params = mcp.StdioServerParameters(
            command=config.command,
            args=config.arguments,
        )

    async def __aenter__(self):
        self.stdio_client = mcp.client.stdio.stdio_client(self.server_params)
        read, write = await self.stdio_client.__aenter__()

        # Create client session
        self.session = await mcp.ClientSession(read, write).__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        # Close in reverse order of creation
        await self.session.__aexit__(exc_type, exc_value, traceback)
        await self.stdio_client.__aexit__(exc_type, exc_value, traceback)

    async def list_tools(self) -> list[dict]:
        raw_tools = (await self.session.list_tools()).tools
        tools = []
        for raw_tool in raw_tools:
            tool = dict(
                type="function",
                function=dict(
                    name=raw_tool.name,
                    description=raw_tool.description,
                    parameters=raw_tool.inputSchema,
                ),
            )
            tools.append(tool)
        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        print(f"!!!{tool_name}, {arguments}")
        return await self.session.call_tool(tool_name, arguments=arguments)


@dataclass
class McpManagerConfig:
    server_configs: dict[str, McpServerConfig] = field(default_factory=dict)

    @staticmethod
    def load_from_json_file(json_path):
        server_configs = {}
        with open(json_path) as f:
            config = json.load(f)
            for name, command_args in config.items():
                server_configs[name] = McpServerConfig(
                    command=command_args.get("command"),
                    arguments=command_args.get("args"),
                )
        return McpManagerConfig(server_configs=server_configs)


class McpManager:
    def __init__(self, config: McpManagerConfig):
        self.clients: dict[str, McpClient] = {}
        self.failed_servers: set[str] = set()
        for name, server_config in config.server_configs.items():
            self.clients[name] = McpClient(config=server_config)

    async def __aenter__(self):
        # gather causes error
        # await asyncio.gather(*[client.__aenter__() for client in self.clients.values()])
        for name, client in self.clients.items():
            try:
                await client.__aenter__()
                logger.info(f"Successfully initialized MCP server: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize MCP server '{name}': {e}")
                self.failed_servers.add(name)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        for name, client in reversed(
            self.clients.items()
        ):  # reverse order is required!
            if name not in self.failed_servers:
                try:
                    await client.__aexit__(exc_type, exc_value, traceback)
                except Exception as e:
                    logger.warning(f"Error closing MCP server '{name}': {e}")

    async def list_tools(self) -> list[dict]:
        gathered_tools = []
        for server_name, client in self.clients.items():
            if server_name not in self.failed_servers:
                try:
                    tools = await client.list_tools()
                    for tool in tools:
                        tool["function"][
                            "name"
                        ] = f"{server_name}__{tool['function']['name']}"
                    gathered_tools.extend(tools)
                except Exception as e:
                    logger.warning(
                        f"Error listing tools from server '{server_name}': {e}"
                    )
        return gathered_tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        server_name_and_tool_name = tool_name.split("__")
        assert len(server_name_and_tool_name) == 2
        server_name, tool_name = tuple(server_name_and_tool_name)

        if server_name in self.failed_servers:
            raise RuntimeError(f"Cannot call tool from failed server '{server_name}'")

        client = self.clients.get(server_name)
        if client is None:
            raise RuntimeError(f"MCP server '{server_name}' not found")

        return await client.call_tool(tool_name, arguments)


async def async_main():
    config = McpServerConfig(
        command="npx",
        arguments=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/home/okada/claude_workspace",
        ],
    )
    async with McpClient(config) as mcp_client:
        pass


if __name__ == "__main__":
    import asyncio

    asyncio.run(async_main())
