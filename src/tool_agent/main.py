#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import openai

from .mcp import McpManager, McpManagerConfig


def create_parser():
    parser = argparse.ArgumentParser(
        description="Interactive CLI Agent with MCP support", prog="agent"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="mcp_config.json",
        help="MCP server configuration file (default: mcp_config.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="OpenAI model name (default: gpt-4)",
    )
    return parser


class Agent:
    def __init__(self, model: str, mcp_manager: McpManager | None, verbose: bool):
        self.mcp_manager = mcp_manager
        self.verbose = verbose
        self.model = model
        self.tools = []
        self.client = openai.AsyncOpenAI()

    async def __aenter__(self):
        if self.mcp_manager:
            await self.mcp_manager.__aenter__()
            self.tools = await self.mcp_manager.list_tools()
            if self.verbose:
                print(f"Loaded {len(self.tools)} tools from MCP servers")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.mcp_manager:
            await self.mcp_manager.__aexit__(exc_type, exc_value, traceback)

    async def process_messages(
        self, user_message: str, messages: List[Dict[str, Any]]
    ) -> str:
        messages.append({"role": "user", "content": user_message})

        while True:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools if self.tools else None,
            )

            message = response.choices[0].message
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls if message.tool_calls else None,
                }
            )

            if not message.tool_calls:
                return message.content

            for tool_call in message.tool_calls:
                try:
                    result = await self.mcp_manager.call_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )
                except Exception as e:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {str(e)}",
                        }
                    )

    async def handle_tool_command(self, tool_command: str) -> str:
        parts = tool_command.split(" ", 1)
        tool_name = parts[0]
        args_str = parts[1] if len(parts) > 1 else "{}"

        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON arguments: {args_str}"

        if not self.mcp_manager:
            return "Error: No MCP manager available"

        try:
            result = await self.mcp_manager.call_tool(tool_name, arguments)
            return f"Tool result: {result}"
        except Exception as e:
            return f"Error calling tool {tool_name}: {e}"

    def list_available_tools(self) -> str:
        if not self.tools:
            return "No tools available"

        tool_list = []
        for tool in self.tools:
            func_info = tool["function"]
            tool_list.append(f"- {func_info['name']}: {func_info['description']}")

        return "Available tools:\n" + "\n".join(tool_list)


async def run_chat_mode(agent: Agent, verbose: bool, messages: List[Dict[str, Any]]):
    print("Starting interactive chat mode...")
    print("Commands:")
    print("  help - Show this help")
    print("  tools - List available tools")
    print("  exit/quit - End session")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "help":
                print("Commands:")
                print("  help - Show this help")
                print("  tools - List available tools")
                print("  exit/quit - End session")
            elif user_input.lower() == "tools":
                print(agent.list_available_tools())
            elif user_input:
                result = await agent.process_messages(user_input, messages)
                print(result)
            else:
                continue

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


async def async_main():
    parser = create_parser()
    args = parser.parse_args()

    mcp_manager = None
    config_path = Path(args.config)

    if config_path.exists():
        try:
            config = McpManagerConfig.load_from_json_file(args.config)
            mcp_manager = McpManager(config)
            if args.verbose:
                print(f"Loaded MCP configuration from {args.config}")
        except Exception as e:
            print(f"Warning: Failed to load MCP config: {e}", file=sys.stderr)
    elif args.verbose:
        print(f"No MCP config file found at {args.config}")

    agent = Agent(model=args.model, mcp_manager=mcp_manager, verbose=args.verbose)
    messages = []

    try:
        async with agent:
            await run_chat_mode(agent, args.verbose, messages)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
