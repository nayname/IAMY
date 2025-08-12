import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
import asyncio
import argparse

server_params = StdioServerParameters(
    command="node",
    args=["/root/andromeda/docs/mcp/mcp-server.js"],
    env=None,
)


async def main():
    async with stdio_client(server_params) as (read, write):
# async def main():
#     async with streamablehttp_client(math_server_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # # List available prompts
            # response = await session.list_prompts()
            # print("\n/////////////////prompts//////////////////")
            # for prompt in response.prompts:
            #     print(prompt)
            #
            # # List available resources
            # response = await session.list_resources()
            # print("\n/////////////////resources//////////////////")
            # for resource in response.resources:
            #     print(resource)
            #
            # # List available resource templates
            # response = await session.list_resource_templates()
            # print("\n/////////////////resource_templates//////////////////")
            # for resource_template in response.resourceTemplates:
            #     print(resource_template)
            #
            # # List available tools
            # response = await session.list_tools()
            # print("\n/////////////////tools//////////////////")
            # for tool in response.tools:
            #     print(tool)
            #
            # # Get a prompt
            # prompt = await session.get_prompt("example_prompt", arguments={"question": "what is 2+2"})
            # print("\n/////////////////prompt//////////////////")
            # print(prompt.messages[0].content.text)
            #
            # # Read a resource
            # content, mime_type = await session.read_resource("greeting://Alice")
            # print("\n/////////////////content//////////////////")
            # print(mime_type[1][0].text)
            #
            # Call a tool




            result = await session.call_tool("search", arguments={"query": args.intent})
            print("\n/////////////////result//////////////////")
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())