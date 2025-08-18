import asyncio
import json
import os
import random

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..lib.query_w_tools import create_graph


def tools_list():
    #list of all generated scripts (for the frontend)
    f = open('../tools.json')
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

# 'Lets suppose we have an application that executes user intents using backend server through CLI.'
#                              ' Based on a topic you will be given synthesize me an appropriate number of plausible pairs NL user'
#                              ' intents->CLI one-liner (not more than 150 pairs). Proposed user intents and CLI command must be '
#                              'based content of Neutron Documentation. You will be given an input argument "command", build your '
#                              'proposed CLI command around it. Sort the list by popularity, give he response in JSON in format: '
#                              '"{{intent1}}":{"{{popularity}}":{{num}},"{{command}}":{{str}}"}, "{{intent2}}":{"{{popularity}}":{{num}},'
#                              '"{{command}}":{{str}}"}". Intents should follow the format, displayed in the document, namely being '
#                              'implemented through one-liner CLI, using console commands. Document with example NL->one-liner examples:'

synthesize_ners = """
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

CLI_DATA_PATHS = {
    "Query All User Balances": "intents_oneliner/Query All User Balances",
    "Execute Contract": "intents_oneliner/Execute Contract",
    "Upload Contract": "intents_oneliner/Upload Contract",
    "Migrate Contract": "intents_oneliner/Migrate Contract",
    "Query Specific Balance": "intents_oneliner/Query Specific Balance",
    "Query Contract State": "intents_oneliner/Query Contract State",
    "Send Tokens": "intents_oneliner/Send Tokens",
}

server_params = StdioServerParameters(
    command="node",
    args=["/root/neutron/docs/mcp/mcp-server.js"],
    env=None,
)

def create_ner_dataset(filepaths):
    """Loads data from all provided CLI files and creates a unified NER dataset."""
    processed_data = []

    for key, filepath in filepaths.items():
        batches = []
        try:
            with open(filepath, 'r') as f:
                content = json.load(f)
                cli_data = content[key]
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not read or parse {filepath}. Skipping. Error: {e}")
            continue

        for intent, details in cli_data.items():
            text = intent
            command = details.get("command", "")

            batches.append({"label":key, "query":text, "command":command})

            if len(batches) > 9:
                processed_data.append(batches)
                batches.clear()

        if batches:
            processed_data.append(batches)

    with open("processed_data.json", 'w') as f:
        f.write(json.dumps(processed_data))
    die
    return Dataset.from_list(processed_data)

async def main():
    config = {"configurable": {"thread_id": 1234}}
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            create_ner_dataset(CLI_DATA_PATHS)

            for intent in intents:
                if not os.path.exists('data/intents_oneliner/'+intent[0]):
                    agent = await create_graph(session, synthesize_query_oneliner.replace("**command**", intent[1]))
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
