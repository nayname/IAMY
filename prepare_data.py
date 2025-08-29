import argparse
import asyncio
import json
import os
import random
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lib.query_w_tools import create_graph
from langchain.globals import set_verbose

intents = [("Query Specific Balance","neutrond query bank balance <address> --node <grpc_node>"),
           ("Query All User Balances", "neutrond query bank balances <address> --node <grpc_node>"),
           ("Send Tokens", "neutrond tx bank send <from> <to> <amount> --node <grpc_node> <signing info> <gas>"),
           # "Build Smart Contract", "Test Contract",
           ("Upload Contract", "neutrond tx wasm store <wasm-file> --node <grpc_node> <signing info> <gas>"),
           ("Instantiate Contract", "neutrond tx wasm instantiate <code_id> <init-msg> --node <grpc_node> <signing info> <gas>"),
           ("Execute Contract", "neutrond tx wasm execute <contract-address> <exec-msg> --node <grpc_node> <signing info> <gas>"),
           ("Migrate Contract", "neutrond tx wasm execute <contract-address> <exec-msg> --node <grpc_node> <signing info> <gas>"),
           ("Query Contract State", "neutrond query wasm contract-state smart <contract-address> <query-msg> --node <grpc_node>")]

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
    "NeutronTemplate":"data/pages/NeutronTemplate"
}

synthesize_ners = open("prompts/synthesize_ners").read()
synthesize_query_intents = open("prompts/synthesize_query_intents").read()
synthesize_query_oneliner = open("prompts/synthesize_query_oneliner").read()
synthesize_recipes = open("prompts/synthesize_recipes").read()


server_params = StdioServerParameters(
    command="node",
    args=["/root/neutron/docs/mcp/mcp-server.js"],
    env=None,
)

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

            batches.append({"label":key, "query":text, "command":command})

            if len(batches) > num-1:
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
            batches.append({"label":key, "query":intent})

            if len(batches) > num-1:
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


async def create_recipe(session, config):
    count = 1

    for batch in create_batch(pages, 5):
        if not os.path.exists('data/recipes/' + batch[0]['label'] + str(count)):
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

            with open('data/recipes/' + batch[0]['label'] + str(count) + 'result.txt', 'w') as f:
                json.dump(serializable_state, f, indent=4)
            with open('data/recipes/' + batch[0]['label'] + str(count), 'w') as f:
                json.dump(json.loads(response["messages"][-1].content), f, indent=4)

        count += 1



async def create_ner(session, config):
    count = 1

    for batch in create_ner_dataset(onliners, 5):
        if not os.path.exists('data/rephrases/'+str(count)+'result.txt'):
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

            with open('data/rephrases/'+str(count)+'result.txt', 'w') as f:
                json.dump(serializable_state, f, indent=4)
            with open('data/rephrases/'+str(count), 'w') as f:
                json.dump(json.loads(response["messages"][-1].content), f, indent=4)
                # print("Response: " + json.dumps(json.loads(response["messages"][-1].content), indent=4))
        count += 1


async def main(flag):
    config = {"configurable": {"thread_id": 1234}}
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # parser = argparse.ArgumentParser(description="Client")
            # parser.add_argument("intent", type=str, help="User intent")
            if flag == "oneliner":
                await create_oneliner(session, config)
            elif flag == "recipe":
                await create_recipe(session, config)
            elif flag == "ner":
                await create_ner(session, config)
            else:
                await create_intents(session, config)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))

# Agent(queries['crowdfund'][0], 'crowdfund')
# Agent(queries['cw20_exchange'][0], 'cw20_exchange')
