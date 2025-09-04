import os
import re
import shutil
import subprocess
import sys

import grpc
import requests
import uvicorn
import json

from bech32 import bech32_decode, bech32_encode
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from cosmpy.protos.cosmwasm.wasm.v1.query_pb2_grpc import QueryStub
from cosmpy.protos.cosmwasm.wasm.v1.query_pb2 import QuerySmartContractStateRequest

from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.exceptions import NotFoundError
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgStoreCode, MsgInstantiateContract
from cosmpy.aerial.client import LedgerClient
from sympy.strategies.core import switch

from create import generate_code, glue
from ner.inference import NERExtractor
from recipes.backend import extract_code_id_from_tx, select_data_provider, build_history_query, execute_query_request, \
    normalize_tx_results, open_celatone_explorer, search_contract_address, navigate_to_metadata_tab, \
    download_metadata_json, query_contract_info, query_code_info, extract_code_hash, query_bank_balance, \
    connect_rpc_endpoint, neutrond_status, extract_block_height, build_msg_delete_schedule, query_cron_schedule, \
    query_all_cron_schedules, query_cron_params, build_msg_add_schedule, build_dao_proposal, construct_update_admin_tx, \
    get_admin_wallet, get_contract_address, construct_tx_execute_contract, query_contracts_by_creator, \
    fetch_all_contracts_by_creator


# --- Placeholder for your generation logic ---
# This function simulates the work of your generate_code() and NER model.
# In your real application, you would replace this with your actual model inference.
def generate_mock_response(text: str):
    """
    Takes text and returns a mock dictionary with label, params, and command.
    """
    # Simple logic to choose a response based on keywords
    if "send" in text.lower() and "token" in text.lower():
        return {
            "label": "Send Tokens",
            "params": {"amount": "10 NTRN", "recipient": "ntrn1...", "wallet": "mywallet"},
            "command": "neutrond tx bank send mywallet ntrn1bobaddress... 100000000000000000000000000000000000000000000000000000000000000000000000000000untrn --gas auto"
        }
    elif "balance" in text.lower():
        return {
            "label": "Query Balance",
            "params": {"wallet": "$WALLET_ADDR"},
            "command": "neutrond query bank balances $WALLET_ADDR --node https://grpc-kaiyo-1.neutron.org:443"
        }
    else:
        return {
            "label": "Unknown Intent",
            "params": {},
            "command": "Could not generate a command for the given text."
        }


# --- FastAPI App Setup ---
app = FastAPI(
    title="Command Generator"
)

origins = [
   "https://thousandmonkeystypewriter.org", # The domain for Neutron docs
]

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"], # Allows all methods
   allow_headers=["*"], # Allows all headers
)

# Mount a directory for static files (like CSS, JS) if you have them
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic model for the request body to ensure type safety
class GenerateRequest(BaseModel):
    text: str
    address: str

cfg = NetworkConfig(
    chain_id="neutron-1",
    url="grpc+https://grpc-kralum.neutron-1.neutron.org",
    fee_minimum_gas_price=0.01,
    fee_denomination="untrn",
    staking_denomination="untrn",
)
client = LedgerClient(cfg)

def format_amount(micro_amount: int, decimals: int = 6) -> str:
    """
    Step 4: Converts a micro-denom amount to a human-readable value.
    """
    if not isinstance(micro_amount, int):
        raise TypeError("micro_amount must be an integer.")

    # Using floating point division for simplicity in Python
    amount = micro_amount / (10 ** decimals)
    # Format to remove unnecessary trailing zeros
    return f'{amount:f}'.rstrip('0').rstrip('.')


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main index.html page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


def compile_wasm_contract(contract_dir: str) -> str:
    """Compile a CosmWasm contract and return the path to the optimised .wasm file."""
    try:
        # 1. Compile to Wasm (un-optimised)
        subprocess.run(['cargo', 'wasm'], cwd=contract_dir, check=True)
        # 2. Run the optimiser (expects `cargo run-script optimize` set up by rust-optimizer)
        subprocess.run(['cargo', 'run-script', 'optimize'], cwd=contract_dir, check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f'Contract compilation failed: {err}') from err

    # Locate the optimised file (typically placed in <contract>/artifacts)
    artifacts_dir = os.path.join(contract_dir, 'artifacts')
    wasm_files = [f for f in os.listdir(artifacts_dir) if f.endswith('.wasm')]
    if not wasm_files:
        raise FileNotFoundError('Optimised wasm not found in artifacts directory.')
    return os.path.join(artifacts_dir, wasm_files[0])


def get_local_chain_account(key_name: str = 'cosmopark', faucet_url: str = 'http://localhost:4500/credit') -> dict:
    """Load or create a key and optionally request faucet funds."""
    try:
        key_info_raw = subprocess.check_output([
            'neutrond', 'keys', 'show', key_name,
            '--output', 'json', '--keyring-backend', 'test'
        ])
    except subprocess.CalledProcessError:
        # Key does not exist \u2013 create it
        subprocess.run([
            'neutrond', 'keys', 'add', key_name,
            '--output', 'json', '--keyring-backend', 'test'
        ], check=True)
        key_info_raw = subprocess.check_output([
            'neutrond', 'keys', 'show', key_name,
            '--output', 'json', '--keyring-backend', 'test'
        ])

    key_info = json.loads(key_info_raw)
    address = key_info['address']

    # Ask faucet to top-up (optional / local-net only)
    if faucet_url:
        try:
            requests.post(faucet_url, json={'address': address, 'denom': 'untrn'})
        except Exception as faucet_err:
            print(f'Faucet funding skipped/failed: {faucet_err}')

    return {'name': key_name, 'address': address}


def construct_tx_wasm_store(wasm_path: str, wallet, chain_id: str, gas: int = 2_000_000, fee: int = 300_000):
    """Return an unsigned `Transaction` containing `MsgStoreCode`."""
    with open(wasm_path, 'rb') as wasm_file:
        wasm_bytes = wasm_file.read()

    msg = MsgStoreCode(
        sender=wallet.address(),
        wasm_byte_code=wasm_bytes
    )

    tx = (
        Transaction()
        .with_messages(msg)
        .with_chain_id(chain_id)
        .with_gas(gas)
        .with_fee(fee)
    )
    return tx


def sign_and_broadcast_tx(tx, wallet, client: LedgerClient):
    """Generic helper to sign & broadcast any prepared transaction."""
    # Populate account meta-data
    account = client.query_account(wallet.address())
    tx = tx.with_sequence(account.sequence).with_account_number(account.account_number)

    # Sign
    wallet.sign(tx)

    # Broadcast and wait for finality
    try:
        response = client.broadcast_tx_block(tx)
    except Exception as err:
        raise RuntimeError(f'Broadcast failed: {err}') from err

    if response.code != 0:
        raise RuntimeError(f'Transaction failed (code {response.code}): {response.raw_log}')
    return response


def parse_code_id_from_receipt(tx_response) -> int:
    """Search TxResponse logs for the `store_code` event and return its `code_id`."""
    logs = tx_response.logs if hasattr(tx_response, 'logs') else tx_response['logs']
    for event in logs[0]['events']:
        if event['type'] == 'store_code':
            for attr in event['attributes']:
                if attr['key'] in ('code_id', 'codeID'):
                    return int(attr['value'])
    raise ValueError('code_id not found in transaction logs.')


def construct_tx_wasm_instantiate(code_id: int, init_msg: dict, label: str, wallet, chain_id: str, admin: str, gas: int = 500_000, fee: int = 150_000):
    """Return an unsigned instantiate transaction."""
    msg = MsgInstantiateContract(
        sender=wallet.address(),
        admin=admin or '',
        code_id=code_id,
        label=label,
        msg=json.dumps(init_msg).encode('utf-8'),
        funds=[]  # Provide coins if the contract expects them
    )

    tx = (
        Transaction()
        .with_messages(msg)
        .with_chain_id(chain_id)
        .with_gas(gas)
        .with_fee(fee)
    )
    return tx


def broadcast_instantiate_tx(instantiate_tx, wallet, client):
    return sign_and_broadcast_tx(instantiate_tx, wallet, client)


def parse_contract_address_from_receipt(tx_response) -> str:
    logs = tx_response.logs if hasattr(tx_response, 'logs') else tx_response['logs']
    for event in logs[0]['events']:
        if event['type'] == 'instantiate':
            for attr in event['attributes']:
                if attr['key'] == '_contract_address':
                    return attr['value']
    raise ValueError('Contract address not found in instantiate logs.')


def query_contract_state(client: LedgerClient, contract_address: str, query_msg: dict):
    try:
        return client.wasm_query(contract_address, query_msg)
    except Exception as err:
        raise RuntimeError(f'Contract query failed: {err}') from err


def ensure_cosmopark_installed() -> None:
    """Ensure that CosmoPark CLI and its Docker images are available."""
    # 1. Check CosmoPark binary
    if shutil.which("cosmopark") is None:
        print("CosmoPark CLI not found. Attempting installation via pip")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cosmopark-cli"])
        except subprocess.CalledProcessError as err:
            raise RuntimeError("Automatic installation of CosmoPark CLI failed.") from err
    else:
        print("CosmoPark CLI detected")

    # 2. Verify Docker is installed \u2013 required by CosmoPark
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required but not installed or not in PATH.")

    # 3. Pull (or update) all CosmoPark Docker images
    try:
        subprocess.check_call(["cosmopark", "pull", "--all"])
        print("CosmoPark Docker images pulled")
    except subprocess.CalledProcessError as err:
        raise RuntimeError("Failed to pull CosmoPark Docker images.") from err


@app.post("/test")
async def handle_generate(request_data: GenerateRequest):
    wasm_path = compile_wasm_contract('/root/neutron/NeutronTemplate/contract')
    wallet = get_local_chain_account()  # \u2192 {'name': 'cosmopark', 'address': 'neutron1\u2026'}
    store_tx = construct_tx_wasm_store(wasm_path, wallet, 'neutron-1')
    store_receipt = sign_and_broadcast_tx(store_tx, wallet, client)
    code_id = parse_code_id_from_receipt(store_receipt)
    instantiate_tx = construct_tx_wasm_instantiate(code_id, {"count": 0}, "counter-instance", wallet, "neutron-1")
    instantiate_receipt = broadcast_instantiate_tx(instantiate_tx, wallet, client)
    contract_address = parse_contract_address_from_receipt(instantiate_receipt)
    state = query_contract_state(client, contract_address, {"get_count": {}})

    ensure_cosmopark_installed()

    return JSONResponse(content=response_data)
    # return templates.TemplateResponse("table.html", {"request": request, "table": map})


@app.post("/generate_")
async def handle_generate(request_data: GenerateRequest):
    """
    Receives text from the frontend, generates a response,
    and returns it as JSON.
    """
    # Get the text from the request body
    input_text = request_data.text

    # Call your generation logic
    response_data = generate_code(input_text)
    print(response_data)

    return JSONResponse(content=response_data)
    # return templates.TemplateResponse("table.html", {"request": request, "table": map})


def fill_queries():
    res = {}

    with open("recipes/functions.json", 'r') as f:
        functions = json.load(f)

    with (os.scandir("recipes/tools") as entries):
        for entry in entries:
            if entry.is_file():  # Check if it's a file
                # if entry.name == "Withdraw 50 NTRN from the smart contract":
                    res[entry.name] = {}
                    with open(entry, "r") as f:
                        data = json.load(f)
                    res[entry.name]["workflow"] = []
                    for tool in data['tools']:
                        code = ""
                        # print("def "+tool["function"][:tool["function"].find("(")]+"(", "export const "+tool["function"][:tool["function"].find("(")])
                        for key, value in functions.items():
                            if (tool["function"] in key or key in tool["function"]) and ("def "+tool["function"][:tool["function"].find("(")]+"(" in value
                                                            or "export const "+tool["function"][:tool["function"].find("(")] in value):
                                    code = value
                        res[entry.name]["workflow"].append({"type": tool["label"].title(), "tool": tool["function"][:tool["function"].find("(")],
                                                            "description": tool["introduction"], "code": code})

                    with os.scandir("recipes/actions") as entries1:
                        for entry1 in entries1:
                            if entry1.name == entry.name:
                                with open(entry1, "r") as f:
                                    data1 = json.load(f)
                                res[entry.name]["label"] = data1["label"].title()




    return res


@app.post("/generate_reponse")
async def generate_reponse(request_data: Request):
    req = await request_data.json();
    input_text = req["text"]

    ner_ = NERExtractor().extract_entities(input_text)
    queries = fill_queries()

    if input_text in queries:
        print( {
            "label": queries[input_text]["label"],
            "params": glue(ner_),
            "workflow": queries[input_text]["workflow"]
        })
        return {
            "label": queries[input_text]["label"],
            "params": glue(ner_),
            "workflow": queries[input_text]["workflow"]
        }
    else:
        return {"label": "Others", "params": glue(ner_), "workflow": "undef"}

@app.post("/generate")
async def handle_generate(request_data: Request):
    """
    Receives text from the frontend, generates a response,
    and returns it as JSON.
    """
    # Get the text from the request body
    req = await request_data.json();
    input_text = req["text"]
    print("QUERY: "+input_text)

    match input_text:
        case "Query transaction history for my address":
            try:
                provider = select_data_provider()#step: 1 Tool: select_data_provider Desciption: Choose a data source (Celatone API, SubQuery, or LCD /txs endpoint) based on latency and pagination needs.",
                query, vars = build_history_query(provider, req["address"])#step: 2 Tool: build_history_query Desciption: Construct a REST or GraphQL query filtering by `message.sender={address}` and order by timestamp descending.",
                raw_results, next_token = execute_query_request(provider, query, vars)#step: 3 Tool: execute_query_request Desciption: Send the HTTP request (fetch/axios) and handle pagination via `offset` or `pageInfo.endCursor`.",
                table_rows = normalize_tx_results(provider, raw_results)#step: 4 Tool: normalize_tx_results Desciption: Map raw results into a uniform schema (hash, blockHeight, action, fee, success) for the frontend table."
                return JSONResponse(content=table_rows)
            except Exception as e:
                return JSONResponse(content="No data provider is reachable at the moment.")
        case "Show details for the cron schedule \"dasset-updator\"":
            metadata = query_cron_schedule("dasset-updator")#step: 1 Tool: query_cron_show_schedule Desciption: Run `neutrond query cron show-schedule protocol_update` to fetch the schedule's full metadata.
            return JSONResponse(content=metadata)
        case "List all existing cron schedules":
            metadata = query_all_cron_schedules()  # step: 1 Tool: query_cron_show_schedule Desciption: Run `neutrond query cron show-schedule protocol_update` to fetch the schedule's full metadata.
            return JSONResponse(content=metadata)
        case "Remove the existing schedule named protocol_update":
            res = []
            authority_addr = query_cron_params()["security_address"]#step: 1 Tool: get_dao_authority_address Desciption: Fetch the Main DAO authority address (required to delete schedules).",
            res.append("Main DAO authority address: " + authority_addr)

            delete_msg = build_msg_delete_schedule(authority_addr, 'protocol_update')#step: 2 Tool: build_msg_delete_schedule Desciption: Create a MsgDeleteSchedule with: name=\"protocol_update\".",
            res.append("MsgDeleteSchedule: " + json.dumps(delete_msg))

            proposal = build_dao_proposal(delete_msg, 'Remove protocol_update schedule', 'This proposal removes the obsolete protocol_update cron schedule.')#step: 3 Tool: package_into_gov_proposal Desciption: Embed the MsgDeleteSchedule in a DAO governance proposal explaining why the schedule should be removed.",
            res.append("Proposal: " + json.dumps(proposal))

            return JSONResponse(content=res)
        case "Query contract metadata on Celatone":
            driver = open_celatone_explorer('neutron-1', '/tmp')#step: 1 Tool: open_celatone_explorer Desciption: Launch the Celatone explorer in a web browser and select the correct Neutron network (mainnet: neutron-1 or testnet: pion-1).",
            search_contract_address(driver, req["address"])#step: 2 Tool: search_contract_address Desciption: Paste the target contract address into the Celatone search bar and press Enter.",
            navigate_to_metadata_tab(driver)#step: 3 Tool: navigate_to_metadata_tab Desciption: Click the \u201cMetadata\u201d (or equivalent) tab in Celatone\u2019s contract view to load stored contract information.",
            metadata_path = download_metadata_json(driver, '/tmp')#step: 4 Tool: download_metadata_json Desciption: Use Celatone\u2019s \u201cDownload\u201d (</>) button to fetch the raw metadata JSON for local inspection or downstream processing."
            return JSONResponse(content=metadata_path)
        case "List all smart contracts deployed by my account":
            page_data = query_contracts_by_creator(req["address"])#step: 3 Tool: query_contracts_by_creator Desciption: Execute `neutrond query wasm list-contract-by-creator <creator-address> --limit 1000` to retrieve contracts deployed by the user.",
            # all_contracts = await fetch_all_contracts_by_creator(req["address"]);#step: 4 Tool: handle_pagination Desciption: If the response includes a `pagination.next_key`, repeat the query with `--page-key` until all contracts are collected."
            return JSONResponse(content=page_data)
        case "Query the code hash of a specific smart contract":
            # UNDEF
            contract_info = query_contract_info('neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')#step: 2 Tool: query_contract_info Desciption: Execute `neutrond query wasm contract <contract-address>` to obtain the contract\u2019s metadata, including its `code_id`.",
            # undef#step: 3 Tool: extract_code_id Desciption: Read the `code_id` field from the contract info response.",
            # code_info = query_code_info(code_id)#step: 4 Tool: query_code_info Desciption: Run `neutrond query wasm code-info <code_id>` to fetch the code information that contains the `code_hash`.",
            # code_hash = extract_code_hash(code_info)#step: 5 Tool: extract_code_hash Desciption: Parse the `code_hash` value from the code-info JSON response."
        case "Withdraw 50 NTRN from the smart contract":
            new_balance = query_bank_balance(req["address"])#step: 6 Tool: query_bank_balance Desciption: After confirmation, re-query the user\u2019s bank balance to reflect the incoming 50 NTRN."
            return JSONResponse(content=new_balance)
        case "Show the current block height of the Neutron chain":
            res = []
            rpc = connect_rpc_endpoint('https://rpc-kralum.neutron-1.neutron.org')#step: 1 Tool: connect_rpc_endpoint Desciption: Connect to a reachable Neutron RPC endpoint (e.g., https://rpc-kralum.neutron.org) to ensure the CLI can query chain data.",
            res.append("RPC endpoint: "+rpc)

            status_json = neutrond_status(rpc)#step: 2 Tool: neutrond_status Desciption: Run `neutrond status --node <rpc-endpoint>` to fetch the node\u2019s sync information.",
            res.append("Status: " + json.dumps(status_json))

            latest_height = extract_block_height(status_json)#step: 3 Tool: extract_block_height Desciption: Parse the JSON response and read `result.sync_info.latest_block_height`."
            res.append("latest_height: " + str(latest_height))

            return JSONResponse(content=res)

    # case _:
        #     # Default case (optional), executed if no other pattern matches
        # Call your generation logic
    # response_data = generate_code(input_text)
    # print(response_data)

    # return templates.TemplateResponse("table.html", {"request": request, "table": map})


@app.get("/queryBankBalance")
async def queryBankBalance(address: str):
    """
    An API endpoint that implements steps 3 and 4 of the workflow.
    It queries the NTRN balance of any given Neutron address.
    """
    try:
        # Create a client to connect to the network
        client = LedgerClient(cfg)

        # Step 3: Query the balance for the specified address and "untrn"
        untrn_balance = client.query_bank_balance(address, denom="untrn")

    # except NotFoundError:
    #     # Handle cases where the address has no balance or does not exist on-chain
    #     return {
    #         "address": address,
    #         "balance": "0 NTRRRRN"
    #     }
        return JSONResponse(content={"raw_balance":untrn_balance})
    except Exception as e:
        # If the address is invalid or the network call fails, return an error.
        raise HTTPException(status_code=500, detail=f"Failed to query balance: {str(e)}")

@app.get("/formatAmount")
async def formatAmount(untrn_balance: int, address: str):
    ntrn_balance_str = format_amount(untrn_balance)

    return {
        "address": address,
        "balance": f"{ntrn_balance_str} NTRN"
    }


@app.get("/test")
async def formatAmount(untrn_balance: int, address: str):
    pass
    # extract_code_id_from_tx("")
    #
    # return {
    #     "address": address,
    #     "balance": f"{ntrn_balance_str} NTRN"
    # }


# ===================================================================================
# == API Endpoints
# ===================================================================================


def _channel_from_cosmpy_url(url: str):
    """
    Convert cosmpy-style gRPC URL (e.g. 'grpc+http://host:80') into a gRPC channel.
    """
    m = re.match(r"grpc\+(http|https)://([^:/]+)(?::(\d+))?$", url)
    if not m:
        raise ValueError(f"Unrecognized gRPC URL: {url}")
    scheme = m.group(1)        # <-- define scheme properly
    host = m.group(2)
    port = m.group(3) or ("443" if scheme == "https" else "80")
    target = f"{host}:{port}"
    if scheme == "https":
        return grpc.secure_channel(target, grpc.ssl_channel_credentials())
    else:
        return grpc.insecure_channel(target)

@app.post("/api/query-contract")
async def handle_query_contract(data: Request):
    """
    Handles a smart contract query request from the frontend.
    This endpoint keeps the RPC URL and query logic secure on the backend.

    Expected JSON body:
    {
        "contractAddress": "neutron1...",
        "query": { "get_count": {} }
    }
    """
    req = await data.json();
    print(req)

    client = LedgerClient(NetworkConfig(
        chain_id="pion-1",
        url="grpc+http://neutron-testnet-grpc.polkachu.com:19190",  # <-- note grpc+http (not https)
        fee_minimum_gas_price=0.025,
        fee_denomination="untrn",
        staking_denomination="untrn",
    ))

    if not client:
        return JSONResponse(content={"message": "Backend client is not connected to the blockchain."}, status_code=503)

    # if not data or "contractAddress" not in data or "query" not in data:
    #     return JSONResponse(content={"message": "Missing 'contractAddress' or 'query' in request body."}, status_code=400)

    contract_address = req["contractAddress"]
    query_msg = req["query"]

    # try:
        # Use the cosmpy client to perform the smart query
    ch = _channel_from_cosmpy_url(client.network_config.url)
    stub = QueryStub(ch)
    req = QuerySmartContractStateRequest(
        address=contract_address,
        query_data=json.dumps(query_msg).encode("utf-8"),
    )
    res = stub.SmartContractState(req)
    return json.loads(res.data)
    # except NotFoundError:
    #     return JSONResponse(content={"message": f"Contract not found at address: {contract_address}"}, status_code=404)
    # except Exception as e:
    #     print(f"[ERROR] /api/query-contract: {e}")
    #     return JSONResponse(content={"message": "An unexpected error occurred on the server."}, status_code=500)


@app.post("/api/validate-address")
async def handle_validate_address(data: Request):
    """
    Validates a bech32 address. This moves the validation logic and bech32
    library dependency from the frontend to the backend.

    Expected JSON body:
    {
        "address": "neutron1..."
    }
    """
    req = await data.json();
    input_text = req["text"]

    if not input_text:
        return JSONResponse(content={"isValid": False, "message": "Missing 'address' in request body."}, status_code=400)

    address = input_text
    expected_prefix = "neutron"

    try:
        hrp, _ = bech32_decode(address)
        if hrp != expected_prefix:
            return JSONResponse(content={"isValid": False, "message": f"Invalid prefix: expected '{expected_prefix}', got '{hrp}'."})

        # Re-encoding serves as a checksum validation
        bech32_encode(hrp, _)
        return JSONResponse(content={"isValid": True})
    except (ValueError, TypeError):
        return JSONResponse(content={"isValid": False, "message": "Address is not a valid bech32 string."})
    except Exception as e:
        print(f"[ERROR] /api/validate-address: {e}")
        return JSONResponse(content={"isValid": False, "message": "An unexpected server error occurred."}, status_code=500)


@app.post("/api/broadcast-tx")
async def handle_broadcast_tx(data: Request):
    """
    Receives raw, signed transaction bytes from the client and broadcasts them
    to the blockchain. This is a "relayer" or "gas station" pattern.

    Expected JSON body:
    {
        "signedTxBytes": "<base64_encoded_signed_tx_bytes>"
    }
    """
    req = await data.json()
    input_text = req["text"]

    if not client:
        return JSONResponse(content={"message": "Backend client is not connected to the blockchain."}, status_code=503)

    if not data:
        return JSONResponse(content={"message": "Missing 'signedTxBytes' in request body."}, status_code=400)

    # In a real application, you would decode the base64 bytes here.
    # For this example, we assume the frontend sends the bytes in a format
    # that cosmpy's `broadcast_tx` can handle directly.
    signed_tx_bytes = input_text  # Placeholder

    try:
        # The frontend signs, the backend just broadcasts
        # Note: A real implementation would need to properly deserialize the bytes first.
        # This is a conceptual example.
        # response = client.broadcast_tx(signed_tx_bytes)

        # MOCK RESPONSE for demonstration since we can't get real signed bytes here
        mock_tx_hash = "A1B2C3D4E5F6..."
        print(f"Mock Broadcasting Tx... would have broadcasted bytes: {signed_tx_bytes[:30]}...")

        # Real logic would check the response code
        # if response.tx_response.code != 0:
        #    return jsonify({"message": f"Transaction failed on-chain: {response.tx_response.raw_log}"}), 400

        return JSONResponse(content={"transactionHash": mock_tx_hash})
    except Exception as e:
        print(f"[ERROR] /api/broadcast-tx: {e}")
        return JSONResponse(content={"message": "Failed to broadcast transaction."}, status_code=500)



if __name__ == "__main__":
    uvicorn.run(app, host='88.198.17.207', port=1958, reload=True)
