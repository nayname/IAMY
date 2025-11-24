import json
import os
import markdown

from starlette.responses import JSONResponse

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
            if entry.is_file() and os.path.exists('recipes/actions/' + entry.name):  # Check if it's a file
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


def get_code(function, functions):
    print("!!!", function)
    for key, value in functions.items():
        if (function in key or key in function) and (
                "def " + function + "(" in value
                or "export const " + function in value):
            return value
    return None


def get_function(key):
    if key['label'] == 'backend':
        if "await" in key['function']:
            function = key['function'][key['function'].find("await "):]
            return function[6:function.find("(")].strip()
        else:
            return key['function'][0:key['function'].find("(")].strip()
    else:
        function = key['code'][key['code'].find("export const "):]
        return function[13:function.find("=")].strip()


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

    for s in steps["tools"]:
        function = get_function(s)
        query["workflow"].append({"tool":function, "code":get_code(function, functions),
                          "function":s["function"], "type":s["label"], "description":s["introduction"]})


    # print("REQ: ", req)
    #
    # if "workflows" in req.keys() and len(req["workflows"]) > 0:
    #     recipe = req["workflows"][0]
    # else:
    #     query = {"query":input_text,
    #              "intent_type": get_intent_type(input_text)}
    #
    #     await to_recipe(query)
    #     if query["recipe"] != "none" and query["recipe"] != "None":
    #         make_workflow(query)
    #
    #     query["answer"] = get_answer(input_text, quadrant_client, embedding_model, query)
    # return formate_response(query)


def get_recipe(req, query):
    if "workflows" in req.keys() and len(req["workflows"]) > 0:
        query['recipe'] = req["workflows"][0]


def no_recipe(query):
    return "recipe" not in query.keys() or not query["recipe"] \
        or query["recipe"] == "none" or query["recipe"] == "None"


async def response(req, input_text, quadrant_client, embedding_model):
    print("REQ: ", req)

    query = {"query":input_text,
             "intent_type": get_intent_type(input_text)}

    get_recipe(req, query)

    if no_recipe(query):
        await to_recipe(query)
    # else:
    #     parse_params_for_recipe(req, query)

    if not no_recipe(query):
        make_workflow(query)

    query["answer"] = get_answer(input_text, quadrant_client, embedding_model, query)
    return formate_response(query)
