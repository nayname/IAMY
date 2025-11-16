import json
import os
import markdown

from starlette.responses import JSONResponse

# from api import tamples_recipes
from prepare_data import escape, pick_intent
from rag.retrieve import retrieve, explain_workflow

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def get_answer(input_text, quadrant_client, embedding_model, query):
    if query["recipe"] == "none" or query["recipe"] == "None":
        return retrieve(input_text, quadrant_client, embedding_model)
    else:
        return explain_workflow(query)


def formate_response(query):
    # print("TEST:", query)
    if "workflow" in query.keys():
        print({
            "mode": "mixed",
            "answer": markdown.markdown(query['answer']),
            "recipe": query['recipe'],
            "workflow": query['workflow']
        })
        return JSONResponse(content={
            "mode": "mixed",
            "answer": markdown.markdown(query['answer']),
            "recipe": query['recipe'],
            "workflow": query['workflow']
        })
    else:
        print({
            "mode": "mixed",
            "answer": markdown.markdown(query['answer'])
        })
        return JSONResponse(content={
            "mode": "mixed",
            "answer": markdown.markdown(query['answer'])
        })

# Read-only (no state change)
# query_contract_state — smart-contract query entrypoints.
# query_balance — bank balances (native + cw20).
# query_tx — tx by hash / events.
# query_chain_info — height, node status, params.
# estimate_fees — dry-run/Simulate to estimate gas/fees.
# Write (state-changing)
# deploy_contract — upload wasm.
# instantiate_contract — new instance from code_id.
# execute_contract_msg — call execute with msg JSON.
# transfer_tokens — native token transfer.
# stake_tokens — delegate to validator.
# unstake_tokens — undelegate.
# withdraw_rewards — distribution withdraw.
# ibc_transfer — ICS-20 transfer (optional for v1).
# Utility / wallet
# ensure_wallet — connect, pick account, chain switch.
# fund_testnet — request faucet (testnet only).

def get_intent_type(query):
    return "unstake_tokens"


async def to_recipe(query):
    tools_list = []
    with os.scandir("recipes/tools") as entries:
        for entry in entries:
            if entry.is_file():  # Check if it's a file
                with open(entry.path, 'r') as f:
                    found_undef = False
                    items = json.load(f)
                    i = {"intent":items["intent"], "functions":[]}

                    for t in items["tools"]:
                        i["functions"].append({"descr":t["introduction"],
                                               "function_name":t["function"]})
                        if t["function"] == "undef":
                            found_undef = True
                    if not found_undef:
                        tools_list.append(i)

    # f = open("generated/test", "a")
    # with open("generated/test", "w") as f:
    #     json.dump(tools_list, f, indent=4)
    # die

    query["recipe"] = await pick_intent(query["query"], tools_list)
    # query["balance"] = "0"


def get_code(tool, functions):
    for key, value in functions.items():
        if (tool["function"] in key or key in tool["function"]) and (
                "def " + tool["function"][:tool["function"].find("(")] + "(" in value
                or "export const " + tool["function"][:tool["function"].find("(")] in value):
            return value
    return None


def get_function(step, actions):
    for action in actions:
        if step == action["step"]:
            return action
    return None


def make_workflow(query):
    # response = await tamples_recipes(query["query"], query)
    query["workflow"] = []

    if os.path.exists('recipes/tools/' + escape(query["recipe"])):
        with open('recipes/tools/' + escape(query["recipe"]), 'r') as f:
            steps = json.load(f)
        # with open('recipes/actions/' + escape(query["recipe"]), 'r') as f:
        #     actions = json.load(f)

    with open("recipes/functions.json", 'r') as f:
        functions = json.load(f)

    for s in steps["workflow"]:
        function = get_function(s["step"], steps["tools"])
        query["workflow"].append({"tool":s["tool"], "code":get_code(function, functions),
                          "function":function["function"], "type":function["label"], "description":s["description"]})


async def response(request_data, quadrant_client, embedding_model):
    req = await request_data.json()
    input_text = req["query"]
    print("QUERY: " + input_text)

    query = {"query":input_text,
             "intent_type": get_intent_type(input_text)}

    await to_recipe(query)
    if query["recipe"] != "none" and query["recipe"] != "None":
        make_workflow(query)
        
    query["answer"] = get_answer(input_text, quadrant_client, embedding_model, query)
    return formate_response(query)
