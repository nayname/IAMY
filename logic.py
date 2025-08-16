import argparse
import asyncio
import json
import os
import random

from typing import List, Literal
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

def tools_list():
    #list of all generated scripts (for the frontend)
    f = open('tools.json')
    map = json.load(f)
    res = ""

    for k, v in map.items():
        res += ' '+k+': Description: '+v['description']+': Scaffold: '+v['scaffold']+'; '

    return res

def get_one_liners():
    f = open('data/raw/all.json', 'r')
    all = json.load(f)
    list = []

    for f in all:
        for k, v in all[f].items():
            # for kj, vj in all[f][k].items():
            for o in all[f][k]['labels']:
                list.append({"intent":o['text'], "cli":o['cmd']})

    return json.dumps(random.sample(list, 50))

list_tools_query = 'Lets suppose we have an application (MCP server) that executes different user intents using MCP tools, each of them implements functions' \
             ' based on Neutron docs (you will be given a list of tools). I will provide you intent and tools, return proposed tools and  variable' \
             ' \'workflow\' to describe how this intent will be proceeded with that MCP tools  which functionality based on the \'scaffold\'. Response JSON in format: "{{intent}}":"{{text}}", "{{tools}}":[list], "{{workflow}}":"{{text}}" ' \
             ' Tools:' + tools_list()

synthesize_query = (' Lets suppose we have an application that executes different user intents using backend server tools.'
                    ' Based on a topic you will be given synthesize me an'
                    ' appropriate number of plausible user intents (not more than 150) than '
                    ' reflects the corresponding content of Neutrom Doicumentation. Sort the list by popularity, give he response in JSON in format:'
                    ' "{{intent1}}":"{{popularity1}}", "{{intent2}}":"{{popularity2}}"" ')

synthesize_query_oneliner_old = (' Lets suppose we have an application that executes different user intents using backend server tools.'
                    ' Based on a topic you will be given synthesize me an'
                    ' appropriate number of plausible user intents (not more than 150) than '
                    ' reflects the corresponding content of Neutrom Doicumentation. Sort the list by popularity, give he response in JSON in format:'
                    ' "{{intent1}}":"{{popularity1}}", "{{intent2}}":"{{popularity2}}"". Intents should follow the format, displayed in the document,'
                    ' namely being implemented through one-liner CLI, using console commands. Document with example NL->one-liner examples:' + get_one_liners())

# 'Lets suppose we have an application that executes user intents using backend server through CLI.'
#                              ' Based on a topic you will be given synthesize me an appropriate number of plausible pairs NL user'
#                              ' intents->CLI one-liner (not more than 150 pairs). Proposed user intents and CLI command must be '
#                              'based content of Neutron Documentation. You will be given an input argument "command", build your '
#                              'proposed CLI command around it. Sort the list by popularity, give he response in JSON in format: '
#                              '"{{intent1}}":{"{{popularity}}":{{num}},"{{command}}":{{str}}"}, "{{intent2}}":{"{{popularity}}":{{num}},'
#                              '"{{command}}":{{str}}"}". Intents should follow the format, displayed in the document, namely being '
#                              'implemented through one-liner CLI, using console commands. Document with example NL->one-liner examples:'
synthesize_query_oneliner = """
You are an expert AI assistant specializing in 'neutrond' command-line operations.
Your task is to generate a list of plausible pairs of natural language (NL) user intents and their corresponding CLI one-liner commands based
 on Neutron Documentation.

**Context:**
The generated pairs will be used in an application that executes user intents on a backend server. All commands must be executable in a standard shell.

**Core Command and Base Topic:**
All CLI commands you generate must be based on and built around the following core command: `**command**`. All NL intents should be based on the topic,
 given to you by user as an input argument.

**Requirements:**
1.  **Synthesize Pairs:** Create a list of NL intent -> CLI command pairs.
2.  **Content:** The intents and commands must be realistic and reflect common operations found in the Neutron Documentation for the given core command.
3.  **Quantity:** Generate an appropriate number of pairs, but do not exceed 150.
4.  **Popularity:** Assign a "popularity" score from 1 to 100 to each pair, where 100 is the most common and 1 is the most obscure.
5.  **Sorting:** The final JSON output must be sorted by the popularity score in descending order (most popular first).

**Output Format:**
The entire output must be a single, valid JSON object.
- The **keys** of the main JSON object are the natural language intent strings.
- The **values** are nested JSON objects, each containing two key-value pairs:
    - `"popularity"`: An integer score.
    - `"command"`: The corresponding CLI command string.

**Example of the required JSON structure:**
{{
  "intent1": {{
    "popularity": int,
    "command": str
  }},
  "intent2": {{
    "popularity": int,
    "command": str
  }}
}}
"""


intents = [("Query Specific Balance","neutrond query bank balance <address> --node <grpc_node>"),
           ("Query All User Balances", "neutrond query bank balances <address> --node <grpc_node>"),
           ("Send Tokens", "neutrond tx bank send <from> <to> <amount> --node <grpc_node> <signing info> <gas>"),
           # "Build Smart Contract", "Test Contract",
           ("Upload Contract", "neutrond tx wasm store <wasm-file> --node <grpc_node> <signing info> <gas>"),
           ("Instantiate Contract", "neutrond tx wasm instantiate <code_id> <init-msg> --node <grpc_node> <signing info> <gas>"),
           ("Execute Contract", "neutrond tx wasm execute <contract-address> <exec-msg> --node <grpc_node> <signing info> <gas>"),
           ("Migrate Contract", "neutrond tx wasm execute <contract-address> <exec-msg> --node <grpc_node> <signing info> <gas>"),
           ("Query Contract State", "neutrond query wasm contract-state smart <contract-address> <query-msg> --node <grpc_node>")]



server_params = StdioServerParameters(
    command="node",
    args=["/root/neutron/docs/mcp/mcp-server.js"],
    env=None,
)


def a_wrapper_for_tools_condition(state) -> Literal["tools", "__end__"]:
    """
    This wrapper function will log a message and then
    execute the original tools_condition.
    """
    print("---CHECKING: My wrapper was called, about to execute the real tools_condition.---")

    # 2. Call the original, pre-built function and return its result
    result = tools_condition(state)

    with open("tracing.txt", "r") as file:
        prev_states = json.load(file)

    serializable_state = {}
    for key, value in state.items():
        if key == 'messages':
            serializable_state[key] = [msg.model_dump() for msg in value]
        else:
            serializable_state[key] = value
    prev_states.append(serializable_state)

    with open('tracing.txt', 'w') as f:
        json.dump(prev_states, f, indent=4)

    print(f"---DECISION: The pre-built tools_condition decided: '{result}'---")
    return result


async def create_graph(session, command):
    llm = ChatOpenAI(model="o3", reasoning_effort="high")

    tools = await load_mcp_tools(session)
    llm_with_tool = llm.bind_tools(tools)


    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                synthesize_query_oneliner.replace("**command**", command)
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

    with open('tracing.txt', 'w') as f:
        f.write("[]")

    # Building the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", ToolNode(tools=tools))
    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_conditional_edges("chat_node", a_wrapper_for_tools_condition, {"tools": "tool_node", "__end__": END})
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

            for intent in intents:
                if not os.path.exists('data/intents_oneliner/'+intent[0]):
                    agent = await create_graph(session, intent[1])
                    response = await agent.ainvoke({"messages": intent[0]}, config=config)
                    print(response["messages"][-1].content)

                    serializable_state = {}
                    for key, value in response.items():
                        if key == 'messages':
                            serializable_state[key] = [msg.model_dump() for msg in value]
                        else:
                            serializable_state[key] = value

                    with open('data/intents_oneliner/'+intent[0]+'result.txt', 'w') as f:
                        json.dump(serializable_state, f, indent=4)
                    with open('data/intents_oneliner/'+intent[0], 'w') as f:
                        json.dump({intent[0]: json.loads(response["messages"][-1].content)}, f, indent=4)
                    # print("Response: " + json.dumps(json.loads(response["messages"][-1].content), indent=4))

if __name__ == "__main__":
    asyncio.run(main())

# Agent(queries['crowdfund'][0], 'crowdfund')
# Agent(queries['cw20_exchange'][0], 'cw20_exchange')
