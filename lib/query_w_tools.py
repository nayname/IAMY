import json
from typing import Literal, TypedDict, List

from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from typing_extensions import Annotated


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

async def create_graph(session, query, provider="OpenAI"):

    if provider == "OpenAI":
        llm = ChatOpenAI(model="o3", reasoning_effort="high")
    else:
        # Use Claude Sonnet 4.5 - the smartest and most efficient model
        llm = ChatAnthropic(
            model="claude-opus-4-1-20250805",
            temperature=1,
            max_tokens=8192
        )
    print(llm)

    tools = await load_mcp_tools(session)
    llm_with_tool = llm.bind_tools(tools)


    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                query
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