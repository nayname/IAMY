import asyncio
import json
import os
import random

from langchain.globals import set_verbose
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lib.query_w_tools import create_graph


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

synthesize_ners = """
You are an expert AI assistant specializing in the neutrond command-line tool and Natural Language Understanding (NLU). Your primary 
function is to generate high-quality synthetic data for training an NLU model that maps natural language queries to neutrond CLI commands. 
The data generation will focus on creating varied user intents and their corresponding BIO-formatted Named Entity Recognition (NER) tags. 

**Core Task:**
You will be given a JSON list of intents. For each intent object in the list, you must:

1.  Analyze the original `intent` text and the associated `command`.
2.  Identify the entities in the `command` that need to be extracted from the text (e.g., addresses, amounts, wallet names).
3.  Synthesize **10 unique rephrasings** of the original `intent`. These rephrasings should be diverse, using different sentence structures, synonyms, and levels of formality.
4.  For **each** of the 10 rephrasings, create a corresponding list of **BIO NER tags**. The tags must correctly identify the entities required for the original CLI `command`.

**Key Concepts:**

  * **BIO NER Schema:**
      * **B-`entity_name`**: The beginning of a multi-token entity.
      * **I-`entity_name`**: The inside of a multi-token entity.
      * **O**: Outside of any entity (not a parameter).
  * **Entity-Command Mapping:** The entities you tag (e.g., `B-amount`, `B-to_address`) must directly correspond to the arguments in the `command` field. For example, the command `neutrond tx bank send wallet1 neutron1... 1000stake` requires a `from_key` (`wallet1`), a `to_address` (`neutron1...`), and an `amount` (`1000stake`). Your NER tags must be designed to extract these specific pieces of information.

BIO LABELS LIST = [
    "O",
    "B-WALLET", "I-WALLET",
    "B-ADDRESS", "I-ADDRESS",
    "B-AMOUNT", "I-AMOUNT",
    "B-MEMO", "I-MEMO",
    "B-NODE", "I-NODE",
    "B-CHAIN_ID", "I-CHAIN_ID",
    "B-DENOM", "I-DENOM",
    "B-CONTRACT", "I-CONTRACT",
    "B-GAS", "I-GAS",
    "B-FEES", "I-FEES",
    "B-WASM_MSG", "I-WASM_MSG",
    "B-FILE_PATH", "I-FILE_PATH",
    "B-CODE_ID", "I-CODE_ID",
]

**Requirements & Constraints:**

1.  **Output Format:** The output must be a single JSON list, mirroring the input structure. For each intent object from the input, add a new key named `generated_examples` which contains a list of your 10 generated objects.
2.  **Generated Object Structure:** Each object within the `generated_examples` list must contain two keys: `rephrase` (the new sentence) and `bio_tags` (the list of NER tags).
3.  **Token-Tag Alignment:** The number of words (tokens) in each `rephrase` must exactly match the number of items in its corresponding `bio_tags` list.
4.  **Consistency:** All 10 generated examples for a given intent must map back to the *original* CLI command and its specific parameters.

-----

### **Example of Input and Expected Output**

**Input JSON:**
[
  {{
    "topic": "Bank Transactions",
    "intent": "send 1000 untrn from my_wallet to neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g",
    "command": "neutrond tx bank send my_wallet neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g 1000untrn"
  }}
]

**Expected Output JSON:**
[
  {{
    "topic": "Bank Transactions",
    "intent": "send 1000 untrn from my_wallet to neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g",
    "command": "neutrond tx bank send my_wallet neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g 1000untrn",
    "generated_examples": [
      {{
        "rephrase": "Please transfer 1000 untrn to the address neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g using my_wallet",
        "bio_tags": ["O", "O", "B-amount", "I-amount", "O", "O", "O", "B-to_address", "O", "B-from_key"]
      }},
      {{
        "rephrase": "From my_wallet, I want to send 1000 untrn to neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g",
        "bio_tags": ["O", "B-from_key", "O", "O", "O", "O", "B-amount", "I-amount", "O", "B-to_address"]
      }},
      {{
        "rephrase": "Execute a transaction of 1000 untrn from my_wallet to neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g",
        "bio_tags": ["O", "O", "O", "O", "B-amount", "I-amount", "O", "B-from_key", "O", "B-to_address"]
      }},
      {{
        "rephrase": "I need my_wallet to dispatch 1000 untrn to the neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g account",
        "bio_tags": ["O", "O", "B-from_key", "O", "O", "B-amount", "I-amount", "O", "O", "B-to_address", "O"]
      }},
      {{
        "rephrase": "Can you move 1000 untrn from account my_wallet over to neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g",
        "bio_tags": ["O", "O", "O", "B-amount", "I-amount", "O", "O", "B-from_key", "O", "O", "B-to_address"]
      }},
      {{
        "rephrase": "Make a payment of 1000 untrn with my_wallet to neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g",
        "bio_tags": ["O", "O", "O", "O", "B-amount", "I-amount", "O", "B-from_key", "O", "B-to_address"]
      }},
      {{
        "rephrase": "Initiate a transfer from my_wallet for 1000 untrn to the recipient neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g",
        "bio_tags": ["O", "O", "O", "O", "B-from_key", "O", "B-amount", "I-amount", "O", "O", "O", "B-to_address"]
      }},
      {{
        "rephrase": "Send funds, specifically 1000 untrn, from my_wallet to neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g",
        "bio_tags": ["O", "O", "O", "B-amount", "I-amount", "O", "B-from_key", "O", "B-to_address"]
      }},
      {{
        "rephrase": "Wire 1000 untrn from my wallet, which is my_wallet, to the address neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g",
        "bio_tags": ["O", "B-amount", "I-amount", "O", "O", "O", "O", "O", "B-from_key", "O", "O", "O", "B-to_address"]
      }},
      {{
        "rephrase": "The destination is neutron1pqc82p64s4f2jkcnlq3fddax98j5epax6s2f0g, the source is my_wallet, and the amount is 1000 untrn",
        "bio_tags": ["O", "O", "O", "B-to_address", "O", "O", "O", "O", "B-from_key", "O", "O", "O", "O", "O", "B-amount", "I-amount"]
      }}
    ]
  }}
]
"""

CLI_DATA_PATHS = {
    "Query All User Balances": "data/intents_oneliner/Query All User Balances",
    "Execute Contract": "data/intents_oneliner/Execute Contract",
    "Upload Contract": "data/intents_oneliner/Upload Contract",
    "Migrate Contract": "data/intents_oneliner/Migrate Contract",
    "Query Specific Balance": "data/intents_oneliner/Query Specific Balance",
    "Query Contract State": "data/intents_oneliner/Query Contract State",
    "Send Tokens": "data/intents_oneliner/Send Tokens",
}

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

    # with open("processed_data.json", 'w') as f:
    #     f.write(json.dumps(processed_data))
    # die
    return processed_data

async def main():
    config = {"configurable": {"thread_id": 1234}}
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            count = 1

            for batch in create_ner_dataset(CLI_DATA_PATHS, 5):
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

if __name__ == "__main__":
    asyncio.run(main())

# Agent(queries['crowdfund'][0], 'crowdfund')
# Agent(queries['cw20_exchange'][0], 'cw20_exchange')
