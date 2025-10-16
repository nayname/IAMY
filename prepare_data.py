import argparse
import asyncio
import json
import os
import random
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from lib.query_w_tools import create_graph
from langchain.globals import set_verbose

intents = [("Query Specific Balance", "neutrond query bank balance <address> --node <grpc_node>"),
           ("Query All User Balances", "neutrond query bank balances <address> --node <grpc_node>"),
           ("Send Tokens", "neutrond tx bank send <from> <to> <amount> --node <grpc_node> <signing info> <gas>"),
           # "Build Smart Contract", "Test Contract",
           ("Upload Contract", "neutrond tx wasm store <wasm-file> --node <grpc_node> <signing info> <gas>"),
           ("Instantiate Contract",
            "neutrond tx wasm instantiate <code_id> <init-msg> --node <grpc_node> <signing info> <gas>"),
           ("Execute Contract",
            "neutrond tx wasm execute <contract-address> <exec-msg> --node <grpc_node> <signing info> <gas>"),
           ("Migrate Contract",
            "neutrond tx wasm execute <contract-address> <exec-msg> --node <grpc_node> <signing info> <gas>"),
           ("Query Contract State",
            "neutrond query wasm contract-state smart <contract-address> <query-msg> --node <grpc_node>")]

onliners = {
    "Query All User Balances": "data/intents_oneliner/Query All User Balances",
    "Execute Contract": "data/intents_oneliner/Execute Contract",
    "Upload Contract": "data/intents_oneliner/Upload Contract",
    "Migrate Contract": "data/intents_oneliner/Migrate Contract",
    "Query Specific Balance": "data/intents_oneliner/Query Specific Balance",
    "Query Contract State": "data/intents_oneliner/Query Contract State",
    "Send Tokens": "data/intents_oneliner/Send Tokens",
}

pages = {
    # "NeutronTemplate":"data/pages/NeutronTemplate",
    # "Cron": "data/pages/Cron",
    #***
    # COSMOS
    #***
    "UserGuides": "data/pages/UserGuides",
    # "ToolingResources": "data/pages/ToolingResources",
    # "EthereumJSON_RPC": "data/pages/EthereumJSON_RPC",
    # "Learn": "data/pages/Learn",
}

synthesize_ners = open("prompts/synthesize_ners").read()
synthesize_query_intents = open("prompts/synthesize_query_intents").read()
synthesize_query_oneliner = open("prompts/synthesize_query_oneliner").read()
synthesize_recipes = open("prompts/synthesize_recipes").read()
synthesize_cosmwasm = open("prompts/synthesize_cosmwasm").read()
synthesize_picking_functions = open("prompts/synthesize_picking_functions").read()

server_params = StdioServerParameters(
    command="node",
    args=["/root/neutron/cosmos-docs/docs/mcp/mcp-server.js"],
    env=None,
)

MCP_ENDPOINT = "https://evm.cosmos.network/mcp"

def create_ner_dataset(filepaths, num):
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

            batches.append({"label": key, "query": text, "command": command})

            if len(batches) > num - 1:
                processed_data.append(batches.copy())
                batches.clear()

        if batches:
            processed_data.append(batches)

    return processed_data


def create_batch(filepaths, num):
    """Loads data from all provided CLI files and creates a unified NER dataset."""
    processed_data = []

    for key, filepath in filepaths.items():
        batches = []
        # try:
        with open(filepath.replace("/pages/", "/intents/"), 'r') as f:
            content = json.load(f)
            data = content[key]
        # except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        #     print(f"Warning: Could not read or parse {filepath}. Skipping. Error: {e}")
        #     continue

        for intent in data.keys():
            batches.append({"label": key, "query": intent})

            if len(batches) > num - 1:
                processed_data.append(batches.copy())
                batches.clear()

        if batches:
            processed_data.append(batches)

    return processed_data


async def create_intents(session, config):
    for intent, filepath in pages.items():
        if not os.path.exists('data/intents/' + intent):
            file_object = open(filepath, "r")
            content = file_object.read()

            agent = await create_graph(session, synthesize_query_intents)
            response = await agent.ainvoke({"messages": content}, config=config)
            print(response["messages"][-1].content)

            serializable_state = {}
            for key, value in response.items():
                if key == 'messages':
                    serializable_state[key] = [msg.model_dump() for msg in value]
                else:
                    serializable_state[key] = value

            with open('data/intents/' + intent + 'result.txt', 'w') as f:
                json.dump(serializable_state, f, indent=4)
            with open('data/intents/' + intent, 'w') as f:
                json.dump({intent: json.loads(response["messages"][-1].content)}, f, indent=4)


async def create_oneliner(session, config):
    for intent in intents:
        if not os.path.exists('data/intents_oneliner/' + intent[0]):
            agent = await create_graph(session, synthesize_query_oneliner.replace("**command**", intent[1]))
            response = await agent.ainvoke({"messages": intent[0]}, config=config)
            print(response["messages"][-1].content)

            serializable_state = {}
            for key, value in response.items():
                if key == 'messages':
                    serializable_state[key] = [msg.model_dump() for msg in value]
                else:
                    serializable_state[key] = value

            with open('intents_oneliner/' + intent[0] + 'result.txt', 'w') as f:
                json.dump(serializable_state, f, indent=4)
            with open('intents_oneliner/' + intent[0], 'w') as f:
                json.dump({intent[0]: json.loads(response["messages"][-1].content)}, f, indent=4)
            # print("Response: " + json.dumps(json.loads(response["messages"][-1].content), indent=4))


async def create_pre_recipe(session, config):
    count = 1

    for batch in create_batch(pages, 5):
        if not os.path.exists('data/pre_recipes/' + batch[0]['label'] + str(count)):
            agent = await create_graph(session, synthesize_recipes)
            set_verbose(True)
            response = await agent.ainvoke({"messages": json.dumps(batch)}, config=config)
            print(response["messages"][-1].content)

            serializable_state = {}
            for key, value in response.items():
                if key == 'messages':
                    serializable_state[key] = [msg.model_dump() for msg in value]
                else:
                    serializable_state[key] = value

            with open('data/pre_recipes/' + batch[0]['label'] + str(count) + 'result.txt', 'w') as f:
                json.dump(serializable_state, f, indent=4)
            with open('data/pre_recipes/' + batch[0]['label'] + str(count), 'w') as f:
                json.dump(json.loads(response["messages"][-1].content), f, indent=4)
            print("SLEEPING")
            time.sleep(30)

        count += 1


async def create_ner(session, config):
    count = 1

    for batch in create_ner_dataset(onliners, 5):
        if not os.path.exists('data/rephrases/' + str(count) + 'result.txt'):
            agent = await create_graph(session, synthesize_ners)
            set_verbose(True)
            response = await agent.ainvoke({"messages": json.dumps(batch)}, config=config)
            print(response["messages"][-1].content)

            serializable_state = {}
            for key, value in response.items():
                if key == 'messages':
                    serializable_state[key] = [msg.model_dump() for msg in value]
                else:
                    serializable_state[key] = value

            with open('data/rephrases/' + str(count) + 'result.txt', 'w') as f:
                json.dump(serializable_state, f, indent=4)
            with open('data/rephrases/' + str(count), 'w') as f:
                json.dump(json.loads(response["messages"][-1].content), f, indent=4)
                # print("Response: " + json.dumps(json.loads(response["messages"][-1].content), indent=4))
        count += 1


def escape(param):
    return param.lower().replace(" ", "_").replace("'", "_").replace('"', '_').replace("\n", "_")\
            .replace("\t", "_").replace("\r", "_").replace("/", "_").replace("\\", "_")


async def create_actions(session, config):
    count = 0
    with os.scandir("data/pre_recipes") as entries:
        for entry in entries:
            if entry.is_file():  # Check if it's a file
                if not entry.path.endswith("result.txt"):
                    found = False
                    title = ""

                    for key in pages.keys():
                        if key in entry.path:
                            found = True
                            title = key
                            break

                    if found:
                        print(entry.path)
                        with open(entry.path, 'r') as f:
                            items = json.load(f)

                        for key in items:
                            count += 1
                            # if action == key['intent']:
                            workflow = key['workflow']

                            if workflow and not os.path.exists('recipes/actions/' + escape(key['intent'])):
                                agent = await create_graph(session, synthesize_cosmwasm)
                                response = await agent.ainvoke({"messages": json.dumps(workflow)}, config=config)
                                print(response["messages"][-1].content)

                                with open('recipes/actions/' + escape(key['intent']), 'w') as f:
                                    json.dump(json.loads(response["messages"][-1].content), f, indent=4)

                                print("Page:"+title+", count: "+str(count))
                                time.sleep(30)


async def pick_tools(session, config):
    count = 0
    with (os.scandir("data/pre_recipes") as entries):
        for entry in entries:
            if entry.is_file():  # Check if it's a file
                if not entry.path.endswith("result.txt"):
                    found = False
                    title = ""

                    for key in pages.keys():
                        if key in entry.path:
                            found = True
                            title = key
                            break

                    if found:
                        with open(entry.path, 'r') as f:
                            items = json.load(f)

                        for key in items:
                            count += 1
                            if not os.path.exists('recipes/tools/' + escape(key['intent'])) \
                                and os.path.exists('recipes/actions/' + escape(key['intent'])):  #action == key['intent'] and
                                with open("/root/neutron/IAMY/recipes/frontend.jsx", "r") as f:
                                    frontend_tools = f.read()

                                with open("/root/neutron/IAMY/recipes/backend.py", "r") as f:
                                    backend_tools = f.read()

                                agent = await create_graph(session, synthesize_picking_functions
                                       .replace("*#*INPUT_JSON*#*", json.dumps(key['workflow'])).replace("{", "{{")
                                       .replace("}", "}}").replace("*#*INTENT*#*", key['intent']))
                                response = await agent.ainvoke({
                                    "messages": "FILE: frontend.jsx Content: " + frontend_tools + "\n\n\n FILE: backend.py Content: " + backend_tools},
                                    config=config)

                                frontend = []
                                backend = []
                                for l in json.loads(response["messages"][-1].content)['workflow']:
                                    for t in key['workflow']:
                                        if t['step'] == l['step']:
                                            d = t.copy()
                                    if l['label'] == 'frontend':
                                        frontend.append(l['usage'] + "//step: " + str(d['step']) + " Tool: " + d[
                                            'tool'] + " Desciption: " + d['description'])
                                    elif l['label'] == 'backend':
                                        backend.append(l['usage'] + "#step: " + str(l['step']) + " Tool: " + d[
                                            'tool'] + " Desciption: " + d['description'])

                                res = {"tools": json.loads(response["messages"][-1].content)['workflow'],
                                       "frontend": frontend, "backend": backend,
                                       "intent": key['intent'], "workflow": key['workflow'],
                                       "outcome_checks": key['outcome_checks']}

                                with open('recipes/tools/' + escape(key['intent']), 'w') as f:
                                    json.dump(res, f, indent=4)

                                print("Page:" + title + ", count: " + str(count))
                                time.sleep(30)


async def main(flag):
    config = {"configurable": {"thread_id": 1234}}
    async with stdio_client(server_params) as (read, write):
    # async with streamablehttp_client(MCP_ENDPOINT) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Handshake
            await session.initialize()

            # parser = argparse.ArgumentParser(description="Client")
            # parser.add_argument("intent", type=str, help="User intent")
            if flag == "oneliner":
                await create_oneliner(session, config)
            elif flag == "ner":
                await create_ner(session, config)
            ##-------------------------------------------##
            # I
            elif flag == "intent":
                await create_intents(session, config)
            # II
            elif flag == "pre_recipe":
                await create_pre_recipe(session, config)
            # III
            elif flag == "recipe":
                await create_actions(session, config)
            # IV
            elif flag == "pick_tools":
                await pick_tools(session, config)


if __name__ == "__main__":
    # try:
        asyncio.run(main(sys.argv[1]))
    # except Exception as e:
    #     print("ERROR SLEEEPING")
    #     time.sleep(100)
    #     asyncio.run(main(sys.argv[1]))


# Agent(queries['crowdfund'][0], 'crowdfund')
# Agent(queries['cw20_exchange'][0], 'cw20_exchange')
