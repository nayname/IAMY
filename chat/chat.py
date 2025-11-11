import json
import os
import markdown

from starlette.responses import JSONResponse

# from api import tamples_recipes
from prepare_data import escape
from rag.retrieve import retrieve


def get_answer(input_text, quadrant_client, embedding_model):
    return retrieve(input_text, quadrant_client, embedding_model)


def formate_response(query):
    # print("TEST:", query)
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


def to_recipe(query):
    query["recipe"] = "Query a wallet’s bank balances via the REST API"
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


async def make_workflow(query):
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
             "answer": get_answer(input_text, quadrant_client, embedding_model),
             "intent_type": get_intent_type(input_text)}
    to_recipe(query)
    await make_workflow(query)


    return formate_response(query)