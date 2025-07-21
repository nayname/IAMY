import argparse
import asyncio
import json

from typing import List
from typing_extensions import TypedDict
from typing import Annotated

from create import generate_code
from lib.classifier import classifier_condition, classify
# from lib.json_gen import generate_json
# from iamy.docsrs_client import call_docsrs

from langgraph.graph import END, START, StateGraph

from lib.docsrs import call_docsrs
from lib.pp import Pipeline
from lib.states import OverallState, InputState, OutputState
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.prompts import load_mcp_prompt
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="node",
    args=["/root/neutron/docs/mcp/mcp-server.js"],
    env=None,
)


def tools_list():
    #list of all generated scripts (for the frontend)
    f = open('tools.json')
    map = json.load(f)
    res = ""

    for k, v in map.items():
        res += ' '+k+': Description: '+v['description']+': Scaffold: '+v['scaffold']+'; '

    return res


async def create_graph(session):
    llm = ChatOpenAI(model="o3", reasoning_effort="high")

    tools = await load_mcp_tools(session)
    llm_with_tool = llm.bind_tools(tools)

    chat_query = 'Lets suppose we have an application (MCP server) that executes different user intents using MCP tools, each of them implements functions' \
                 ' based on Neutron docs (you will be given a list of tools). I will provide you intent and tools, return proposed tools and  variable' \
                 ' \'workflow\' to describe how this intent will be proceeded with that MCP tools  which functionality based on the \'scaffold\'. Response JSON in format: "{{intent}}":"{{text}}", "{{tools}}":[list], "{{workflow}}":"{{text}}" ' \
                 ' Tools:'+tools_list()
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                chat_query,
            ),
            (
                "human",
                ("{messages}"),
            ),
        ]
    )
    chat_llm = chat_prompt | llm_with_tool

    # State Management
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]

    # Nodes
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Building the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", ToolNode(tools=tools))
    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", "__end__": END})
    graph_builder.add_edge("tool_node", "chat_node")
    graph = graph_builder.compile(checkpointer=MemorySaver())
    return graph

async def main():
    config = {"configurable": {"thread_id": 1234}}
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            parser = argparse.ArgumentParser(description="Client")
            parser.add_argument("intent", type=str, help="User intent")
            args = parser.parse_args()

            agent = await create_graph(session)
            response = await agent.ainvoke({"messages": args.intent}, config=config)
            print("Response: " + json.dumps(json.loads(response["messages"][-1].content), indent=4))

if __name__ == "__main__":
    asyncio.run(main())
# Agent(queries['crowdfund'][0], 'crowdfund')
# Agent(queries['cw20_exchange'][0], 'cw20_exchange')
