import os
import re
import shutil
import subprocess
import sys

import os
import json
from decimal import Decimal
from typing import Dict, Any, List

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
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from starlette.middleware.cors import CORSMiddleware
from cosmpy.protos.cosmwasm.wasm.v1.query_pb2_grpc import QueryStub
from cosmpy.protos.cosmwasm.wasm.v1.query_pb2 import QuerySmartContractStateRequest

from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.exceptions import NotFoundError
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgStoreCode, MsgInstantiateContract
from cosmpy.aerial.client import LedgerClient
from sympy.strategies.core import switch

from chat.chat import response
from create import generate_code, glue
from ner.inference import NERExtractor
from prepare_data import escape
from rag.retrieve import retrieve
# from recipes.backend import extract_code_id_from_tx, select_data_provider, build_history_query, execute_query_request, \
#     normalize_tx_results, open_celatone_explorer, search_contract_address, navigate_to_metadata_tab, \
#     download_metadata_json, query_contract_info, query_code_info, extract_code_hash, query_bank_balance, \
#     connect_rpc_endpoint, neutrond_status, extract_block_height, build_msg_delete_schedule, query_cron_schedule, \
#     query_all_cron_schedules, query_cron_params, build_msg_add_schedule, build_dao_proposal, construct_update_admin_tx, \
#     get_admin_wallet, get_contract_address, query_contracts_by_creator, \
#     extract_last_execution_height, amber_positions, construct_supervault_deposit_tx, \
#     sign_and_broadcast_tx_, check_balance, get_supervault_details, build_deposit, get_controller_address, \
#     _query_wasm_smart, supervault_address, validate_balances, construct_and_sign, broadcast_signed_tx, \
#     parse_balance_response, build_send_tx, sign_tx, broadcast_tx, get_chain_home, locate_genesis, backup_genesis, \
#     update_inflation, save_genesis, validate_genesis, get_ufw_status, allow_ssh_via_ufw, reload_ufw, \
#     list_ufw_rules_numbered, gather_gentx_files, collect_gentxs, open_config_file, update_mempool_max_txs, \
#     save_config_file, restart_node_service

from recipes.backend import (
        verify_docker_installed,
        verify_contract_builds_locally,
        run_rust_optimizer_arm64,
        collect_optimized_wasm_artifacts,
        compile_clock_example,
        load_wasm_artifact_bytes,
        fetch_minimum_gas_price,
        fetch_sender_account_state,
        construct_msg_store_code_tx_for_simulation,
        simulate_store_code_tx,
        sign_store_code_tx,
        broadcast_store_code_tx,
        extract_code_id_from_logs,
        verify_code_uploaded_on_chain,
        get_sender_wallet,
        validate_contract_address,
        check_sender_balance,
        construct_msg_send,
        build_unsigned_tx,
        simulate_tx_for_gas,
        apply_min_gas_price_fee,
        fetch_account_number_and_sequence,
        sign_tx_with_sender_key,
        encode_tx_to_bytes,
        broadcast_tx_via_lcd,
        verify_broadcast_result,
        prepare_backend_signer,
        init_signing_cosmwasm_client,
        resolve_contract_address,
        execute_increment_msg_over_rpc,
        confirm_tx_via_lcd,
        verify_incremented_count,
        lcd_get_tx_by_hash,
        parse_tx_logs_for_contract_address,
        validate_contract_via_lcd,
        _build_error_payload,
        load_cargo_toml,
        add_cw_orch_optional_dependency,
        configure_cw_orch_feature,
        verify_cargo_with_cw_orch,
        build_get_count_query_payload,
        lcd_query_wasm_smart,
        decode_and_extract_count,
        validate_contract_and_token,
        build_nft_info_query_json,
        encode_query_to_base64,
        lcd_smart_query_nft_info,
        decode_response_data,
        return_nft_metadata,
        get_sender_address,
        get_account_info,
        check_spendable_balance,
        construct_msg_execute_reset,
        simulate_tx_and_update_fee,
        sign_tx,
        broadcast_tx,
        verify_reset_effect,
        lcd_verify_contract_exists,
        bff_prepare_execute_msg_increment,
        bff_get_chain_and_account_info,
        lcd_check_spendable_balance_for_ujuno,
        bff_construct_execute_contract_tx,
        lcd_simulate_tx,
        bff_sign_execute_tx,
        lcd_broadcast_tx,
        lcd_verify_execute_effect,
        collect_query_parameters,
        encode_smart_query_for_lcd,
        http_get_contract_smart_state,
        decode_lcd_smart_query_response,
        execute_junod_cli_smart_query,
        compare_cli_and_lcd_results,
        detect_contract_project_root,
        read_contract_name_from_cargo,
        run_rust_optimizer,
        verify_wasm_artifact,
        parse_and_validate_amounts,
        check_spendable_balances_for_sender,
        build_execute_contract_msg_with_funds,
        construct_execute_tx,
        simulate_execute_tx,
        sign_and_broadcast_execute_tx,
        verify_funds_transferred_to_contract,
        check_junod_installed,
        set_junod_node,
        set_junod_chain_id,
        set_junod_output_json,
        test_junod_connectivity,
        validate_tx_hash,
        fetch_tx_by_hash,
        check_tx_found_and_success,
        extract_code_id_from_events,
        fallback_parse_raw_log_for_code_id,
        return_code_id,
        detect_workspace_root,
        list_workspace_members,
        run_workspace_optimizer,
        collect_and_verify_wasm_outputs,
        validate_contract_address_format,
        build_query_json_string,
        http_get_lcd_smart_query,
        parse_lcd_smart_query_response,
        decode_contract_response_data,
        read_and_validate_wasm_artifact,
        compute_wasm_checksum,
        get_chain_and_account_info,
        construct_store_code_tx,
        simulate_and_update_fee,
        sign_store_code_tx as sign_store_code_tx_upload,
        broadcast_signed_tx,
        fetch_tx_and_extract_code_id,
        verify_uploaded_code_hash,
        load_cargo_toml as load_cargo_toml_orch,
        ensure_cw_orch_dependency,
        ensure_features_table_exists,
        add_interface_feature_flag,
        ensure_default_includes_interface,
        write_cargo_toml,
        cargo_check_with_interface_feature,
        initialize_pagination_state,
        build_all_tokens_query_json,
        lcd_smart_query_all_tokens,
        decode_all_tokens_response,
        fetch_all_cw721_token_ids,
        validate_juno_address_format as validate_juno_address_faucet,
        query_junox_balance_before_faucet,
        compare_junox_balance_after_faucet,
        retrieve_last_tx_output,
        parse_code_id_from_output,
        export_shell_variable_CODE_ID,
        select_juno_chain_key_params,
        generate_bip39_mnemonic,
        derive_hd_key_from_mnemonic,
        derive_public_key_and_address,
        persist_wallet_metadata,
        optional_verify_address_on_chain,
        build_txflags_string,
        set_txflags_env,
        test_junod_tx_with_txflags,
        get_minimum_gas_price,
        get_chain_id,
        validate_juno_address,
        fetch_all_balances as fetch_all_balances_balance,
        fetch_spendable_balances as fetch_spendable_balances_balance,
        fetch_denoms_metadata,
        format_balances,
    )

# --- FastAPI App Setup ---
app = FastAPI(
    title="Command Generator"
)

origins = [
   "https://thousandmonkeystypewriter.org", # The domain for Neutron docs
   "https://cosmos.thousandmonkeystypewriter.org", # The domain for Neutron docs
   "https://neutron.thousandmonkeystypewriter.org",
   "http://88.198.17.207",
   "http://88.198.17.207:4000",
]

SUPER_VAULT_CONTRACT_ADDRESS = os.getenv("SUPER_VAULT_CONTRACT_ADDRESS", "neutron1vaultxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
WBTC_DENOM = os.getenv("WBTC_DENOM", "ibc/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
USDC_DENOM = os.getenv("USDC_DENOM", "uusdc")

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

embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
quadrant_client = QdrantClient("localhost", port=6333)

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

    return JSONResponse(content=state)


@app.post("/chat")
async def handle_chat(request_data: Request):
    return await response(request_data, quadrant_client, embedding_model)

@app.post("/mock_execute")
async def handle_mock(request_data: Request):
    # return await response(request_data, quadrant_client, embedding_model)
    return JSONResponse(content={
  "step_results": [
    { "tool": "collect_target_address", "output": "mock address 123..." },
    { "tool": "rest_get_request", "output": "{ balances: [...] }" },
    { "tool": "parse_json_response", "output": "{ amount: 12.3 }" }
  ]
})

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
                    with open(entry, "r") as f:
                        data = json.load(f)
                    res[data['intent']] = {"workflow":[]}

                    for tool in data['tools']:
                        code = ""
                        # print("def "+tool["function"][:tool["function"].find("(")]+"(", "export const "+tool["function"][:tool["function"].find("(")])
                        for key, value in functions.items():
                            if (tool["function"] in key or key in tool["function"]) and ("def "+tool["function"][:tool["function"].find("(")]+"(" in value
                                                            or "export const "+tool["function"][:tool["function"].find("(")] in value):
                                    code = value
                        res[data['intent']]["workflow"].append({"type": tool["label"].title(), "tool": tool["function"][:tool["function"].find("(")],
                                                            "description": tool["introduction"], "code": code})

                    with os.scandir("recipes/actions") as entries1:
                        for entry1 in entries1:
                            if entry1.name == entry.name:
                                with open(entry1, "r") as f:
                                    data1 = json.load(f)
                                res[data['intent']]["label"] = data1["label"].title()




    return res


@app.post("/generate_reponse")
async def generate_reponse(request_data: Request):
    req = await request_data.json();
    input_text = req["text"]

    ner_ = NERExtractor().extract_entities(input_text)
    queries = fill_queries()

    if input_text in queries:
        return {
            "label": queries[input_text]["label"],
            # "params": glue(ner_),
            "workflow": queries[input_text]["workflow"]
        }
    elif escape(input_text) in queries:
        return {
            "label": queries[escape(input_text)]["label"],
            "workflow": queries[escape(input_text)]["workflow"]
        }
    else:
        return {"label": "Others", "params": glue(ner_), "workflow": "undef"}

async def query_supervault_details():
    return {
        "contract_address": SUPER_VAULT_CONTRACT_ADDRESS,
        "tokens": [
            {"denom": WBTC_DENOM, "symbol": "WBTC"},
            {"denom": USDC_DENOM, "symbol": "USDC"}
        ]
    }

async def tamples_recipes(input_text, req):
    # Extract common required parameters from the request dictionary
    project_root = req.get("project_root", ".")
    contract_address = req.get("contract_address", "")
    address = req.get("address", "")
    txhash = req.get("txhash", "")
    private_key_hex = req.get("private_key_hex", "")
    code_id = req.get("code_id")
    token_id = req.get("token_id", "8")
    coin_str = req.get("coin_str", "1000000ujuno")
    execute_msg = req.get("execute_msg", {"increment": {}})
    wallet_name = req.get("wallet_name", "MyWalletName")
    node_url = req.get("node_url", "http://localhost:26657")
    chain_id = req.get("chain_id", "uni-6")
    page_limit = req.get("page_limit", 100)
    raw_output = req.get("raw_output", "")
    rpc_endpoint = req.get("rpc_endpoint", "https://rpc.juno.strange.love")

    # Defaults for simplicity in the implementation, but note that production
    # code should retrieve these securely, potentially outside of 'req'.
    JUNO_SENDER_ADDRESS = os.getenv("JUNO_SENDER_ADDRESS", "juno1...")
    JUNO_PRIV_KEY_HEX = os.getenv("JUNO_PRIVATE_KEY_HEX", private_key_hex)
    JUNO_MNEMONIC = os.getenv("JUNO_MNEMONIC", "")

    # --- Match statement to handle intents ---

    match input_text:
        # case "Query transaction history for my address":
        #     try:
        #         provider = select_data_provider()  # step: 1 Tool: select_data_provider Desciption: Choose a data source (Celatone API, SubQuery, or LCD /txs endpoint) based on latency and pagination needs.",
        #         query, vars = build_history_query(provider, req[
        #             "address"])  # step: 2 Tool: build_history_query Desciption: Construct a REST or GraphQL query filtering by `message.sender={address}` and order by timestamp descending.",
        #         raw_results, next_token = execute_query_request(provider, query,
        #                                                         vars)  # step: 3 Tool: execute_query_request Desciption: Send the HTTP request (fetch/axios) and handle pagination via `offset` or `pageInfo.endCursor`.",
        #         table_rows = normalize_tx_results(provider,
        #                                           raw_results)  # step: 4 Tool: normalize_tx_results Desciption: Map raw results into a uniform schema (hash, blockHeight, action, fee, success) for the frontend table."
        #         return table_rows
        #     except Exception as e:
        #         return "No data provider is reachable at the moment."
        # case "Show details for the cron schedule \"dasset-updator\"":
        #     metadata = query_cron_schedule(
        #         "dasset-updator")  # step: 1 Tool: query_cron_show_schedule Desciption: Run `neutrond query cron show-schedule protocol_update` to fetch the schedule's full metadata.
        #     return metadata
        # case "List all existing cron schedules":
        #     metadata = query_all_cron_schedules()  # step: 1 Tool: query_cron_show_schedule Desciption: Run `neutrond query cron show-schedule protocol_update` to fetch the schedule's full metadata.
        #     return metadata
        # case "Query the cron schedule named \"dasset-updator\"":
        #     res = []
        #     schedule = query_cron_schedule(
        #         "dasset-updator")  # step: 1 Tool: query_cron_schedule Desciption: Invoke `neutrond query cron schedule daily_maintenance` (or REST `/neutron/cron/schedule/daily_maintenance`).",
        #     res.append("Schedule: " + json.dumps(schedule))
        #
        #     last_height = extract_last_execution_height(
        #         schedule)  # step: 2 Tool: parse_json_response Desciption: Parse the returned schedule details: `name`, `period`, `msgs`, `last_execution_height`."
        #     res.append("Last execution height: " + str(last_height))
        #
        #     return res
        # case "Remove the existing schedule named protocol_update":
        #     res = []
        #     authority_addr = query_cron_params()[
        #         "security_address"]  # step: 1 Tool: get_dao_authority_address Desciption: Fetch the Main DAO authority address (required to delete schedules).",
        #     res.append("Main DAO authority address: " + authority_addr)
        #
        #     delete_msg = build_msg_delete_schedule(authority_addr,
        #                                            'protocol_update')  # step: 2 Tool: build_msg_delete_schedule Desciption: Create a MsgDeleteSchedule with: name=\"protocol_update\".",
        #     res.append("MsgDeleteSchedule: " + json.dumps(delete_msg))
        #
        #     proposal = build_dao_proposal(delete_msg, 'Remove protocol_update schedule',
        #                                   'This proposal removes the obsolete protocol_update cron schedule.')  # step: 3 Tool: package_into_gov_proposal Desciption: Embed the MsgDeleteSchedule in a DAO governance proposal explaining why the schedule should be removed.",
        #     res.append("Proposal: " + json.dumps(proposal))
        #
        #     return res
        # case "Query contract metadata on Celatone":
        #     driver = open_celatone_explorer('neutron-1',
        #                                     '/tmp')  # step: 1 Tool: open_celatone_explorer Desciption: Launch the Celatone explorer in a web browser and select the correct Neutron network (mainnet: neutron-1 or testnet: pion-1).",
        #     search_contract_address(driver, req[
        #         "address"])  # step: 2 Tool: search_contract_address Desciption: Paste the target contract address into the Celatone search bar and press Enter.",
        #     navigate_to_metadata_tab(
        #         driver)  # step: 3 Tool: navigate_to_metadata_tab Desciption: Click the \u201cMetadata\u201d (or equivalent) tab in Celatone\u2019s contract view to load stored contract information.",
        #     metadata_path = download_metadata_json(driver,
        #                                            '/tmp')  # step: 4 Tool: download_metadata_json Desciption: Use Celatone\u2019s \u201cDownload\u201d (</>) button to fetch the raw metadata JSON for local inspection or downstream processing."
        #     return metadata_path
        # case "List all smart contracts deployed by my account":
        #     page_data = query_contracts_by_creator(req[
        #                                                "address"])  # step: 3 Tool: query_contracts_by_creator Desciption: Execute `neutrond query wasm list-contract-by-creator <creator-address> --limit 1000` to retrieve contracts deployed by the user.",
        #     # all_contracts = await fetch_all_contracts_by_creator(req["address"]);#step: 4 Tool: handle_pagination Desciption: If the response includes a `pagination.next_key`, repeat the query with `--page-key` until all contracts are collected."
        #     return page_data
        # case "Query the code hash of a specific smart contract":
        #     # UNDEF
        #     contract_info = query_contract_info(
        #         'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')  # step: 2 Tool: query_contract_info Desciption: Execute `neutrond query wasm contract <contract-address>` to obtain the contract\u2019s metadata, including its `code_id`.",
        #     # undef#step: 3 Tool: extract_code_id Desciption: Read the `code_id` field from the contract info response.",
        #     # code_info = query_code_info(code_id)#step: 4 Tool: query_code_info Desciption: Run `neutrond query wasm code-info <code_id>` to fetch the code information that contains the `code_hash`.",
        #     # code_hash = extract_code_hash(code_info)#step: 5 Tool: extract_code_hash Desciption: Parse the `code_hash` value from the code-info JSON response."
        # case "Withdraw 50 NTRN from the smart contract":
        #     new_balance = query_bank_balance(req[
        #                                          "address"])  # step: 6 Tool: query_bank_balance Desciption: After confirmation, re-query the user\u2019s bank balance to reflect the incoming 50 NTRN."
        #     return new_balance
        # case "Show the current block height of the Neutron chain":
        #     res = []
        #     rpc = connect_rpc_endpoint(
        #         'https://rpc-kralum.neutron-1.neutron.org')  # step: 1 Tool: connect_rpc_endpoint Desciption: Connect to a reachable Neutron RPC endpoint (e.g., https://rpc-kralum.neutron.org) to ensure the CLI can query chain data.",
        #     res.append("RPC endpoint: " + rpc)
        #
        #     status_json = neutrond_status(
        #         rpc)  # step: 2 Tool: neutrond_status Desciption: Run `neutrond status --node <rpc-endpoint>` to fetch the node\u2019s sync information.",
        #     res.append("Status: " + json.dumps(status_json))
        #
        #     latest_height = extract_block_height(
        #         status_json)  # step: 3 Tool: extract_block_height Desciption: Parse the JSON response and read `result.sync_info.latest_block_height`."
        #     res.append("latest_height: " + str(latest_height))
        #
        #     return res
        # case "Check my health factor on Amber Finance":
        #     positions = await amber_positions(req[
        #                                           "address"])  # step: 6 Tool: query_bank_balance Desciption: After confirmation, re-query the user\u2019s bank balance to reflect the incoming 50 NTRN."
        #     return positions
        # case "Deposit 3 eBTC into the maxBTC/eBTC Supervault":
        #     res = []
        #     details = await query_supervault_details()  # step: 3 Tool: query_supervault_details Desciption: Look up the maxBTC/eBTC Supervault contract address and confirm single-sided deposits with eBTC are permissible.",
        #     res.append("Supervault details: " + json.dumps(details))
        #
        #     unsigned_tx = construct_supervault_deposit_tx({'address': req['address'], 'wbtc_amount': 3000000,
        #                                                    'usdc_amount': 0})  # step: 4 Tool: construct_supervault_deposit_tx Desciption: Build a deposit message specifying 3 eBTC as the amount and the vault address from step 3.",
        #     res.append("Unsigned_tx: " + json.dumps(unsigned_tx))
        #
        #     tx_hash = await sign_and_broadcast_tx_(
        #         unsigned_tx)  # step: 5 Tool: sign_and_broadcast_tx Desciption: Sign and broadcast the deposit transaction."
        #     res.append("Tx_hash: " + json.dumps(tx_hash))
        #
        #     return res
        # case "Execute an emergency withdrawal for the user's Amber trading position":
        #     positions = await amber_positions(req[
        #                                           "address"])  # step: 6 Tool: query_bank_balance Desciption: After confirmation, re-query the user\u2019s bank balance to reflect the incoming 50 NTRN."
        #     return positions
        # case "Increase the user's deposit in the WBTC/USDC Supervault by 0.2 WBTC and 12 000 USDC":
        #     res = []
        #
        #     # await validate_balances(userAddress)#step: 2 Tool: validate_token_balances Desciption: Ensure the wallet has at least 0.2 WBTC and 12 000 USDC available.",
        #     vault_addr = supervault_address()[
        #         "address"]  # step: 3 Tool: get_supervault_contract_address Desciption: Look up the contract address for the WBTC/USDC Supervault.",
        #     res.append("Vault details: " + json.dumps(vault_addr))
        #
        #     tx_pkg = construct_supervault_deposit_tx({"address": vault_addr, "wbtc_amount": 20000000,
        #                                               "usdc_amount": 12000000000})  # step: 4 Tool: construct_tx_supervault_deposit Desciption: Create the deposit message specifying 0.2 WBTC and 12 000 USDC as the amounts.",
        #     res.append("Deposit message: " + json.dumps(tx_pkg))
        #
        #     tx_hash = await sign_and_broadcast_tx_({"tx_base64": tx_pkg[
        #         "tx_base64"]})  # step: 5 Tool: sign_and_broadcast_tx Desciption: Sign and broadcast the deposit transaction."
        #     res.append("Tx_hash: " + json.dumps(tx_hash))
        #
        #     return res
        # case "Enable USDC gas payments for my next transaction":
        #     res = []
        #     signed_tx = await construct_and_sign(
        #         "")  # step: 4 Tool: construct_and_sign_next_tx Desciption: When building the user\u2019s next transaction, set `--fees <amount>uusdc` where `<amount>` \u2265 Step 2\u2019s minimum threshold, then sign.",
        #     res.append("Signed_tx: " + json.dumps(signed_tx))
        #
        #     tx_hash = await broadcast_signed_tx()  # step: 5 Tool: broadcast_tx Desciption: Broadcast the signed transaction to Neutron and await inclusion in a block."
        #     res.append("Tx_hash: " + json.dumps(tx_hash))
        #
        #     return res
        # case "Query a wallet’s bank balances via the REST API":
        #     parsed = parse_balance_response(req["balance"],
        #                                     denom='untrn')  # step: 3 Tool: parse_json_response Desciption: Decode the JSON payload to view the \"balances\" array containing denomination/amount pairs.
        #     return parsed
        # case "Broadcast a pre-signed transaction over gRPC":
        #     res = []
        #     unsigned_file = build_send_tx('cosmos1from...', 'cosmos1to...', '100stake', '50stake', 'cosmoshub-4',
        #                                   'unsigned_tx.json')  # step: 1 Tool: construct_unsigned_tx_cli Desciption: Generate an unsigned tx JSON with \"<appd> tx <module> <msg> --generate-only\".",
        #     res.append("Unsigned tx JSON: " + json.dumps(unsigned_file))
        #
        #     signed_file = sign_tx(unsigned_file, 'mykey', 'test', 'cosmoshub-4',
        #                           'signed_tx.json')  # step: 2 Tool: sign_tx_cli Desciption: Sign the tx with \"<appd> tx sign\" to obtain a base64 \"tx_bytes\" field.",
        #     res.append("Sign the tx with: " + json.dumps(signed_file))
        #
        #     tx_hash = broadcast_tx(signed_file, 'cosmoshub-4',
        #                            'http://localhost:26657')  # step: 3 Tool: grpcurl_broadcast_tx Desciption: Invoke the gRPC method with: \n  grpcurl -plaintext -d '{\"tx_bytes\":\"<base64>\",\"mode\":\"BROADCAST_MODE_SYNC\"}' \\\n          localhost:9090 cosmos.tx.v1beta1.Service/BroadcastTx"
        #     res.append("Invoke the gRPC method: " + json.dumps(tx_hash))
        #
        #     return res
        #
        # case "Update the on-chain voting period in genesis to 600 s":
        #     res = []
        #     home_dir = get_chain_home()[
        #         "chain_home"]  # step: 1 Tool: get_chain_home Desciption: Resolve the node\u2019s home directory.",
        #     res.append("Node's home directory: " + json.dumps(home_dir))
        #
        #     genesis_meta = locate_genesis(home_dir)  # step: 2 Tool: locate_file Desciption: Open config/genesis.json.",
        #     res.append("Open config/genesis.json: " + json.dumps(genesis_meta))
        #
        #     backup_genesis(
        #         genesis_meta["genesis_path"])  # step: 3 Tool: backup_file Desciption: Create genesis.json.bak.",
        #     res.append("Create genesis.json.bak")
        #
        #     updated = await update_inflation(genesis_meta["genesis_data"], "600s")[
        #         "updated_genesis"]  # step: 4 Tool: update_json_value Desciption: Navigate to gov.params.voting_period (or gov.params.voting_params.voting_period depending on SDK version) and set the value to \"600s\".",
        #     res.append("Set value: " + json.dumps(updated))
        #
        #     await save_genesis(genesis_meta["genesis_path"],
        #                        updated)  # step: 5 Tool: save_file Desciption: Persist the modified genesis.json.",
        #     res.append("Persist the modified genesis.json")
        #
        #     validate_genesis(chain_binary="simd",
        #                      home=home_dir)  # step: 6 Tool: validate_genesis Desciption: Run \"simd validate-genesis\" to confirm structural correctness."
        #     res.append("Structurally correct")
        #
        #     return res
        #
        # case "Add a UFW rule to allow SSH on port 22":
        #     res = []
        #     ufw_status = get_ufw_status()  # step: 1 Tool: run_shell_command Desciption: Confirm UFW is active: `sudo ufw status`.  If inactive, prompt user to enable it first.",
        #     res.append("UFW is active")
        #
        #     add_rule_result = allow_ssh_via_ufw()  # step: 2 Tool: run_shell_command Desciption: Allow SSH traffic: `sudo ufw allow 22/tcp comment 'Allow SSH'`.",
        #     res.append("SSH traffic allowed")
        #
        #     reload_result = reload_ufw()  # step: 3 Tool: run_shell_command Desciption: Reload UFW to apply rule set changes (optional on most systems): `sudo ufw reload`.",
        #     res.append("UFW reload")
        #
        #     rules_list = list_ufw_rules_numbered()  # step: 4 Tool: run_shell_command Desciption: List numbered rules to confirm insertion: `sudo ufw status numbered`."
        #     res.append("List UFW rules: " + json.dumps(rules_list))
        #
        #     return res
        # case "Collect all gentxs into the genesis file":
        #     res = []
        #     gather_gentx_files(["/path/to/val1/gentx",
        #                         "/path/to/val2/gentx"])  # step: 1 Tool: gather_gentx_files Desciption: Ensure all validator gentx JSON files are placed inside the `config/gentx/` directory.",
        #     res.append("Validator gentx JSON files")
        #
        #     genesis_file = collect_gentxs(
        #         chain_binary="neutrond")  # step: 2 Tool: collect_gentxs Desciption: Run `<chain_binary> collect-gentxs` to merge every gentx into the genesis file.",
        #     res.append("Gentx merged")
        #
        #     is_valid = validate_genesis(
        #         chain_binary="neutrond")  # step: 3 Tool: validate_genesis Desciption: Re-run `<chain_binary> validate-genesis` to verify the final genesis is consistent and ready for launch."
        #     res.append("Genesis validated")
        #
        #     return res
        # case "Set mempool max-txs to -1 in app.toml":
        #     res = []
        #     lines = open_config_file(
        #         config_path='~/.neutrond/config/app.toml')  # step: 1 Tool: open_config_file Desciption: Open `$HOME/.<daemon>/config/app.toml` for editing.",
        #     cfg = update_mempool_max_txs(cfg,
        #                                  new_value=-1)  # step: 2 Tool: update_toml_value Desciption: Under the `[mempool]` section change `max_txs = 5000` (or whatever it is) to `max_txs = -1` to disable the cap.",
        #     save_config_file('~/.neutrond/config/app.toml',
        #                      cfg)  # step: 3 Tool: save_and_close_file Desciption: Write the file and exit the editor.",
        #     restart_node_service(
        #         'neutrond')  # step: 4 Tool: restart_node_service Desciption: Restart the node service, e.g., `sudo systemctl restart <daemon>`. Wait until the node is fully synced again."
        #
        #     return res
        # case "Allow p2p port 26656 through ufw":
        #     res = []
        #     result = allow_ssh_via_ufw()  # step: 1 Tool: ufw_command Desciption: Run `sudo ufw allow 26656/tcp comment 'Cosmos P2P'`.",
        #     result = reload_ufw()  # step: 2 Tool: ufw_reload Desciption: Execute `sudo ufw reload` to apply the new rule."
        #
        #     return res
        case "Compile a CosmWasm contract for ARM64 using the rust-optimizer-arm64 Docker image":
            res = []
            output1 = verify_docker_installed()
            res.append(f"Docker verified: {output1['message']}")
            output2 = verify_contract_builds_locally(project_root)
            res.append(f"Contract local build check completed successfully.")
            output3 = run_rust_optimizer_arm64(project_root)
            res.append(f"Optimizer ran successfully with message: {output3['message']}")
            output4 = collect_optimized_wasm_artifacts(project_root)
            res.append(f"Optimized WASM artifacts collected: {output4}")
            return {"status": "success", "message": "CosmWasm contract compiled successfully for ARM64.",
                    "data": {"artifacts": output4}}

        case "Compile the clock_example CosmWasm contract and upload clock_example.wasm to the Juno testnet (chain-id uni-6)":
            res = []
            output1 = compile_clock_example(project_root)
            res.append(f"Clock example compiled to: {output1}")
            output2 = load_wasm_artifact_bytes(output1)
            res.append(f"WASM artifact loaded, size: {len(output2)} bytes.")
            output3 = fetch_minimum_gas_price()
            res.append(f"Minimum gas price fetched: {output3[0]} {output3[1]}.")
            output4 = fetch_sender_account_state(address, fee_denom=output3[1], required_balance=Decimal(
                '10000000'))  # Assume required balance for fees
            res.append(
                f"Sender account state fetched: number={output4['account_number']}, sequence={output4['sequence']}")
            output5 = construct_msg_store_code_tx_for_simulation(address, output2, output3[0], output3[1],
                                                                 chain_id='uni-6')
            res.append(f"MsgStoreCode Tx constructed with initial gas limit {output5['initial_gas_limit']}.")
            output6 = simulate_store_code_tx(output5["tx_bytes"])
            res.append(
                f"Tx simulated, gas used: {output6['gas_used']}, adjusted limit: {output6['adjusted_gas_limit']}")
            output7 = sign_store_code_tx(address, output2, output3[0], output3[1], output6["adjusted_gas_limit"],
                                         output4["account_number"], output4["sequence"], 'uni-6', JUNO_PRIV_KEY_HEX)
            res.append(f"Tx signed with computed fee amount: {output7['fee_amount']} {output3[1]}.")
            output8 = broadcast_store_code_tx(output7["tx_bytes"])
            res.append(f"Tx broadcasted successfully, txhash: {output8['txhash']}")
            output9 = extract_code_id_from_logs(output8)
            res.append(f"Code ID extracted from logs: {output9}")
            output10 = verify_code_uploaded_on_chain(output9, address)
            res.append(f"Code ID {output9} verified on-chain. Creator matches sender.")
            return {"status": "success", "message": f"Contract uploaded with Code ID {output9}.",
                    "data": {"code_id": output9, "tx_response": output8}}

        case "Send 1,000,000 ujuno to a specified contract address on Juno by constructing, simulating, signing, and broadcasting a bank send transaction via the LCD broadcast endpoint.":
            res = []
            output1 = await get_sender_wallet(env_var='JUNO_SENDER_PRIVATE_KEY')
            res.append(f"Sender wallet loaded, address: {output1['address']}")
            output2 = await validate_contract_address(contract_address)
            res.append(f"Contract address {contract_address} validated on-chain.")
            output3 = await check_sender_balance(output1['address'])
            res.append(f"Sender balance checked, total available: {output3['balance']}")
            output4 = construct_msg_send(output1['address'], contract_address)
            res.append("MsgSend constructed for 1,000,000 ujuno.")
            output5 = build_unsigned_tx(output4)
            res.append(f"Unsigned Tx built with initial gas limit {output5.auth_info.fee.gas_limit}.")
            output6 = await simulate_tx_for_gas(output5)
            res.append(f"Tx simulated, adjusted gas limit: {output6}.")
            output7 = await apply_min_gas_price_fee(output5)
            res.append(f"Minimum gas price applied, final fee amount: {output7} ujuno.")
            output8 = await fetch_account_number_and_sequence(output1['address'])
            res.append(f"Account info fetched: number={output8['account_number']}, sequence={output8['sequence']}")
            output9 = sign_tx_with_sender_key(output5, output1['private_key_hex'], output8['account_number'],
                                              output8['sequence'])
            res.append(f"Tx signed, signature attached.")
            output10 = encode_tx_to_bytes(output9)
            res.append("Signed Tx encoded to base64 bytes.")
            output11 = await broadcast_tx_via_lcd(output10)
            res.append(f"Tx broadcasted, txhash: {output11['txhash']}")
            output12 = verify_broadcast_result(output11)
            res.append(f"Broadcast verified, final result: {output12['txhash']}")
            return {"status": "success",
                    "message": f"Bank send transaction successful. TxHash: {output12['txhash']}", "data": output12}

        case "Connect a SigningCosmWasmClient to RPC https://rpc.juno.strange.love and execute increment on the contract":
            res = []
            output1 = prepare_backend_signer()
            res.append(f"Backend signer wallet prepared.")
            output2 = init_signing_cosmwasm_client(output1)
            res.append("Signing CosmWasm client initialized.")
            output3 = resolve_contract_address()
            res.append(f"Contract address resolved: {output3}")
            output4 = verify_incremented_count(output3, before=None)  # Query before
            res.append(f"Current counter value before: {output4['after']}")
            output5 = execute_increment_msg_over_rpc(output2, output1, output3)
            res.append(f"Increment message executed, txhash: {output5['txhash']}")
            output6 = confirm_tx_via_lcd(output5['txhash'])
            res.append(f"Transaction confirmed via LCD at height {output6['height']}.")
            output7 = verify_incremented_count(output3, before=output4['after'])  # Query after and compare
            res.append(f"Counter value after: {output7['after']}. Verified increment: {output7['verified']}.")
            return {"status": "success",
                    "message": f"Increment executed successfully. New count: {output7['after']}",
                    "data": {"txhash": output5["txhash"], "after_count": output7["after"]}}

        case "Automatically extract the contract address from instantiate txhash":
            res = []
            try:
                output1 = await lcd_get_tx_by_hash(txhash)
                res.append(f"Transaction fetched successfully for hash {txhash}.")
                output2 = parse_tx_logs_for_contract_address(output1)
                res.append(f"Contract address extracted from logs: {output2}")
                output3 = await validate_contract_via_lcd(output2)
                res.append(
                    f"Contract address {output2} validated on-chain with code ID {output3['contract_info']['code_id']}")
                return {"status": "success", "message": f"Contract address extracted and validated: {output2}",
                        "data": {"contract_address": output2}}
            except Exception as e:
                error_payload = _build_error_payload("UNEXPECTED_ERROR", details=str(e))
                return {"status": "error", "message": error_payload["error"]["message"], "data": error_payload}

        case "Add cw-orch as an optional dependency in a CosmWasm contract's Cargo.toml":
            res = []
            output1 = load_cargo_toml(project_root)
            res.append(f"Cargo.toml loaded from {project_root}.")
            add_cw_orch_optional_dependency(project_root)
            res.append("cw-orch added as an optional dependency.")
            configure_cw_orch_feature(project_root)
            res.append("cw-orch feature configured.")
            output4 = verify_cargo_with_cw_orch(project_root)
            res.append(f"Cargo check with cw-orch feature succeeded with return code {output4['returncode']}.")
            return {"status": "success",
                    "message": "cw-orch successfully added as optional dependency and verified to compile.",
                    "data": {"res": res}}

        case "Get current count from contract CONTRACT_ADDRESS":
            res = []
            output1 = resolve_contract_address(contract_address)
            res.append(f"Contract address resolved: {output1}")
            output2 = build_get_count_query_payload()
            res.append(f"Query payload constructed and base64-encoded: {output2[:20]}...")
            output3 = lcd_query_wasm_smart(output1, output2)
            res.append("LCD smart query executed successfully.")
            output4 = decode_and_extract_count(output3)
            res.append(f"Counter value decoded and extracted: {output4}")
            return {"status": "success", "message": f"Current count is {output4}.", "data": {"count": output4}}

        case "Query detailed CW721 nft_info for token ID 8 from a given NFT contract on Juno using a CosmWasm smart query.":
            res = []
            output1 = validate_contract_and_token(contract_address, token_id)
            res.append(f"Contract {output1[0]} and token ID {output1[1]} validated.")
            output2 = build_nft_info_query_json(output1[1])
            res.append(f"Query JSON built: {json.dumps(output2)}")
            output3 = encode_query_to_base64(output2)
            res.append(f"Query encoded to base64: {output3[:20]}...")
            output4 = lcd_smart_query_nft_info(output1[0], output3)
            res.append("LCD smart query executed successfully.")
            output5 = decode_response_data(output4)
            res.append(f"Response data decoded: {output5.keys()}")
            output6 = return_nft_metadata(contract_address, token_id)
            res.append("Full validation and query flow successful.")
            return {"status": "success", "message": f"NFT metadata fetched for token {token_id}.",
                    "data": {"metadata": output6}}

        case "Execute reset on contract CONTRACT_ADDRESS setting count to 0":
            res = []
            sender_address = JUNO_SENDER_ADDRESS  # Use a predefined sender address for the backend functions
            required_amount_for_fees = 1000000  # Placeholder for sufficient balance

            output1 = get_sender_address()
            res.append(f"Sender address resolved: {output1}")
            output2 = get_account_info(output1)
            res.append(f"Account info fetched: number={output2['account_number']}, sequence={output2['sequence']}")
            output3 = check_spendable_balance(output1, required_amount_for_fees)
            res.append(f"Spendable balance checked: {output3} ujuno available.")
            output4 = construct_msg_execute_reset(output1, contract_address)
            res.append("MsgExecuteContract (reset) constructed.")
            output5 = build_unsigned_tx(output4, output2["sequence"], output2["pub_key"])
            res.append("Unsigned Tx skeleton built.")
            output6 = simulate_tx_and_update_fee(output5)
            res.append(f"Tx simulated, new gas limit: {output6['gas_limit']}, fee: {output6['fee_amount']} ujuno.")
            output7 = sign_tx(output5, output2["account_number"], private_key_hex=JUNO_PRIV_KEY_HEX)
            res.append("Tx signed with sender's private key.")
            output8 = broadcast_tx(output7['tx_bytes_b64'])
            res.append(f"Tx broadcasted, txhash: {output8['txhash']}")
            output9 = verify_reset_effect(contract_address)
            res.append(f"Contract count verified to be {output9}.")
            return {"status": "success",
                    "message": f"Contract reset to count 0 successfully. TxHash: {output8['txhash']}",
                    "data": {"final_count": output9}}

        case "Execute increment on contract address CONTRACT_ADDRESS with 10ujuno":
            res = []
            sender_address = JUNO_SENDER_ADDRESS
            funds_ujuno = 10

            output1 = lcd_verify_contract_exists(contract_address)
            res.append(f"Contract {contract_address} verified to exist.")
            output2 = bff_prepare_execute_msg_increment()
            res.append("Execute message ('increment') prepared.")
            output3 = bff_get_chain_and_account_info(sender_address)
            res.append(f"Chain ({output3['chain_id']}) and account info fetched: seq={output3['sequence']}")
            output4 = lcd_check_spendable_balance_for_ujuno(sender_address,
                                                            funds_ujuno + 50000)  # Assume 50k ujuno fee buffer
            res.append(f"Spendable balance checked: {output4['available_ujuno']} ujuno available.")
            output5 = bff_construct_execute_contract_tx(sender_address, contract_address, output2[1], funds_ujuno)
            res.append("Unsigned MsgExecuteContract Tx constructed with 10 ujuno attached.")
            output6 = lcd_simulate_tx(output5)
            res.append(f"Tx simulated, recommended gas limit: {output6['recommended_gas_limit']}")
            output7 = bff_sign_execute_tx(output5, output3['chain_id'], output3['account_number'],
                                          output3['sequence'], output6['recommended_fee_ujuno'],
                                          output6['recommended_gas_limit'], JUNO_PRIV_KEY_HEX)
            res.append("Tx signed successfully.")
            output8 = lcd_broadcast_tx(output7)
            res.append(f"Tx broadcasted, txhash: {output8['txhash']}")
            output9 = lcd_verify_execute_effect(contract_address, output8)
            res.append(f"Execution verified via events. Success: {output9['success']}")
            return {"status": "success",
                    "message": f"Increment with 10ujuno executed successfully. TxHash: {output8['txhash']}",
                    "data": output9}

        case "Query the smart (contract-state smart) view of a CosmWasm contract on Juno, using junod CLI and/or the LCD endpoint.":
            res = []
            payload = req.get("payload",
                              {"contract_address": contract_address, "query": {"config": {}}})  # Example payload

            output1 = collect_query_parameters(payload)
            res.append(f"Query parameters collected. Target: {output1.contract_address}")
            output2 = encode_smart_query_for_lcd(output1.query)
            res.append(f"Query encoded for LCD: {output2[:20]}...")
            output3 = http_get_contract_smart_state(output1.contract_address, output2, output1.node.lcd_url)
            res.append("LCD smart query executed successfully.")
            output4 = decode_lcd_smart_query_response(output3)
            res.append(f"LCD response decoded: {type(output4).__name__}.")

            # Assume CLI parameters are available if junod execution is desired
            cli_result = None
            if output1.node.rpc_url and output1.node.chain_id:
                output5 = execute_junod_cli_smart_query(output1.contract_address, output1.query,
                                                        output1.node.rpc_url, output1.node.chain_id)
                res.append("junod CLI query executed successfully.")
                output6 = compare_cli_and_lcd_results(output4, output3, output5)
                res.append("LCD and CLI results compared and verified to match.")
                cli_result = output6

            return {"status": "success", "message": "Smart query executed and results returned.",
                    "data": {"lcd_result": output4, "cli_comparison": cli_result, "steps": res}}

        case "Compile the current CosmWasm smart contract using rust-optimizer":
            res = []
            output1 = detect_contract_project_root(project_root)
            res.append(f"Contract project root detected at: {output1}")
            output2 = read_contract_name_from_cargo(str(output1))
            res.append(f"Contract name: {output2[0]}, Expected artifact path: {output2[1]}")
            output3 = run_rust_optimizer(str(output1))
            res.append(f"rust-optimizer ran successfully. Return code: {output3['returncode']}")
            output4 = verify_wasm_artifact(str(output2[1]))
            res.append(f"WASM artifact verified: size {output4['size_bytes']} bytes.")
            return {"status": "success", "message": "CosmWasm contract compiled and verified.", "data": output4}

        case "Execute a CosmWasm contract while attaching tokens, equivalent to using the `--amount` flag in `junod tx wasm execute`.":
            res = []
            sender_address = JUNO_SENDER_ADDRESS
            private_key_hex_val = JUNO_PRIV_KEY_HEX

            output1 = parse_and_validate_amounts(coin_str)
            res.append(f"Attached funds parsed: {output1}")
            output2_account = await fetch_account_number_and_sequence(sender_address)
            output2 = await check_spendable_balances_for_sender(sender_address, output1,
                                                                {"ujuno": 500000})  # Assume ujuno fee
            res.append("Spendable balances checked and sufficient.")
            output3 = build_execute_contract_msg_with_funds(sender_address, contract_address, execute_msg, output1)
            res.append("MsgExecuteContract constructed with attached funds.")

            # Extract public key (cosmpy's PrivateKey helper is easier to use here than explicit protobuf)
            from cosmpy.crypto.keypairs import PrivateKey
            priv_key = PrivateKey(bytes.fromhex(private_key_hex_val))
            pubkey_bytes = priv_key.public_key.bytes

            output4 = await construct_execute_tx(output3, pubkey_bytes, output2_account['sequence'])
            res.append(f"Unsigned Tx constructed with provisional gas limit {output4[1]}.")
            output5 = await simulate_execute_tx(output4[0])
            res.append(f"Tx simulated, gas used: {output5[0]}, adjusted limit: {output5[1]}.")

            # Recalculate fee and update Tx (simulate_execute_tx only updates gas_limit)
            from decimal import Decimal, ROUND_UP
            # Assumes Juno mainnet gas price 0.025ujuno
            gas_price_amount = Decimal("0.025")
            fee_amount_int = int((Decimal(output5[1]) * gas_price_amount).to_integral_value(rounding=ROUND_UP))
            output4[0].auth_info.fee.gas_limit = output5[1]
            output4[0].auth_info.fee.amount[0].amount = str(fee_amount_int)

            output6 = await sign_and_broadcast_execute_tx(output4[0], private_key_hex_val,
                                                          output2_account['account_number'],
                                                          output2_account['sequence'], 'juno-1')
            res.append(f"Tx signed and broadcasted: {output6[0]}")
            output7 = await verify_funds_transferred_to_contract(contract_address, sender_address, output1)
            res.append("Funds transfer verified on-chain.")
            return {"status": "success", "message": f"Execution with funds successful. TxHash: {output6[0]}",
                    "data": output7}

        case "Configure the junod CLI to use node NODE_URL and chain-id uni-6":
            res = []
            output1 = check_junod_installed()
            res.append(f"junod binary verified to be installed.")
            output2 = set_junod_node(node_url)
            res.append(f"junod config node set to {node_url}.")
            output3 = set_junod_chain_id(chain_id)
            res.append(f"junod config chain-id set to {chain_id}.")
            output4 = set_junod_output_json()
            res.append("junod config output set to json.")
            output5 = test_junod_connectivity(node_url, expected_chain_id=chain_id)
            res.append(f"Connectivity check passed. Reported chain-id: {output5['reported_chain_id']}")
            return {"status": "success", "message": "junod CLI configured and connectivity verified.",
                    "data": {"test_result": output5}}

        case "Look up a Juno transaction by hash and extract the CosmWasm code_id from its events.":
            res = []
            output1 = validate_tx_hash(txhash)
            res.append(f"Transaction hash validated: {output1}")
            output2 = fetch_tx_by_hash(output1)
            res.append(f"Transaction data fetched from LCD, HTTP status: {output2['status_code']}")
            output3 = check_tx_found_and_success(output2['status_code'], output2['data'])
            res.append(f"Transaction verified found and successful.")
            output4 = extract_code_id_from_events(output3)
            res.append(f"Found {len(output4)} code_id candidates in events.")
            output5 = fallback_parse_raw_log_for_code_id(output3)
            res.append(f"Fallback parse found: {output5}")
            output6 = return_code_id(output4, output5)
            res.append(f"Final resolved code ID: {output6['code_id']}")
            return {"status": "success", "message": f"Code ID extracted: {output6['code_id']}", "data": output6}

        case "Compile all workspace contracts with workspace-optimizer":
            res = []
            output1 = detect_workspace_root()
            res.append(f"Workspace root detected at: {output1['workspace_root']}")
            output2 = list_workspace_members(output1['workspace_root'])
            res.append(f"Found {len(output2['valid_members'])} valid contracts in the workspace.")
            output3 = run_workspace_optimizer(output1['workspace_root'])
            res.append(f"Workspace optimizer ran successfully. Return code: {output3['return_code']}")
            output4 = collect_and_verify_wasm_outputs(output1['workspace_root'], members=output2['valid_members'])
            res.append(f"Verified {len(output4['verified_contracts'])} WASM outputs in artifacts directory.")
            return {"status": "success", "message": "All workspace contracts compiled and verified.",
                    "data": output4}

        case "Query a CosmWasm smart contract on Juno via REST using a base64-encoded smart query.":
            res = []
            query_input = req.get("query_input", {"get_count": {}})  # Example query

            output1 = validate_contract_address_format(contract_address)
            res.append(f"Contract address validated: {output1}")
            output2 = build_query_json_string(query_input)
            res.append(f"Query JSON string created: {output2}")
            output3 = encode_query_to_base64(output2)
            res.append(f"Query base64-encoded and URL-escaped: {output3[:20]}...")
            output4 = await http_get_lcd_smart_query(output1, output3)
            res.append("LCD HTTP GET successful.")
            output5 = parse_lcd_smart_query_response(output4)
            res.append(f"Base64 data field extracted.")
            output6 = decode_contract_response_data(output5)
            res.append(f"Contract response decoded as JSON: {json.dumps(output6)}")
            return {"status": "success", "message": "Smart query successful.", "data": {"response": output6}}

        case "Upload the compiled CosmWasm wasm file artifacts/CONTRACT_NAME.wasm to the Juno chain":
            res = []
            contract_name = req.get("contract_name", "counter")
            sender_address = JUNO_SENDER_ADDRESS

            output1 = read_and_validate_wasm_artifact(contract_name)
            res.append(f"WASM artifact loaded, size: {len(output1)} bytes.")
            output2 = compute_wasm_checksum(output1)
            res.append(f"WASM checksum computed: {output2.hex[:10]}...")
            output3 = await get_chain_and_account_info(sender_address)
            res.append(f"Chain ({output3.chain_id}) and account info fetched: seq={output3.sequence}")
            output4 = construct_store_code_tx(sender_address, output1)
            res.append("MsgStoreCode Tx constructed with provisional fees.")
            output5 = await simulate_and_update_fee(output4[0])
            res.append(
                f"Tx simulated, gas limit updated to {output5[2]}, fee to {output5[0].auth_info.fee.amount[0].amount} ujuno.")
            output6 = sign_store_code_tx_upload(output5[0], output3.chain_id, output3.account_number,
                                                output3.sequence, JUNO_PRIV_KEY_HEX)
            res.append("Tx signed successfully.")
            output7 = await broadcast_signed_tx(output6)
            res.append(f"Tx broadcasted, txhash: {output7['txhash']}")
            output8 = await fetch_tx_and_extract_code_id(output7['txhash'])
            res.append(f"Code ID extracted from logs: {output8}")
            output9 = await verify_uploaded_code_hash(output8, output2.digest)
            res.append(f"Code hash verified on-chain: {output9}")
            return {"status": "success", "message": f"Contract uploaded with Code ID {output8}.",
                    "data": {"code_id": output8, "txhash": output7['txhash']}}

        case "Enable cw-orch integration by adding an 'interface' feature flag in Cargo.toml for a CosmWasm contract.":
            res = []
            output1 = load_cargo_toml_orch(path=f"{project_root}/Cargo.toml")
            res.append("Cargo.toml loaded using tomlkit.")
            output2 = ensure_cw_orch_dependency(output1, version='0.18.0')
            res.append("cw-orch dependency ensured.")
            output3 = ensure_features_table_exists(output2)
            res.append("Features table ensured.")
            output4 = add_interface_feature_flag(output3)
            res.append("Interface feature flag added/updated.")
            output5 = ensure_default_includes_interface(output4)
            res.append("Default features updated to include 'interface'.")
            output6 = write_cargo_toml(output5, path=f"{project_root}/Cargo.toml")
            res.append(f"Modified Cargo.toml written to {output6}.")
            output7 = cargo_check_with_interface_feature(project_root)
            res.append(f"Verification check passed. Return code: {output7['returncode']}")
            return {"status": "success", "message": "cw-orch interface feature enabled and verified.",
                    "data": {"res": res}}

        case "Query the CW721 all_tokens list from a given NFT contract on Juno using CosmWasm smart queries.":
            res = []
            output1 = validate_contract_address(contract_address)
            res.append(f"Contract address validated: {output1}")
            output2 = initialize_pagination_state(page_limit)
            res.append(f"Pagination state initialized with limit {output2.page_limit}.")
            output3 = await fetch_all_cw721_token_ids(contract_address, page_limit)
            res.append(f"Fetched {len(output3)} total token IDs across pages.")
            return {"status": "success", "message": f"Fetched {len(output3)} token IDs.",
                    "data": {"token_ids": output3}}

        case "Claim JUNOX test tokens from the Juno faucet for a given address and verify receipt on-chain":
            res = []
            output1 = validate_juno_address_faucet(address)
            res.append(f"Juno address validated: {output1['is_valid']}")
            output2 = await query_junox_balance_before_faucet(address)
            res.append(f"Pre-faucet JUNOX balance: {output2['amount']}")

            # NOTE: Skipped the user interaction steps 3 and 4 (open/instruct) for non-interactive execution.
            res.append("User instructed to request faucet tokens (skipped actual UI interaction).")

            # NOTE: For non-interactive execution, we can't get a txhash from the faucet directly.
            # Assuming a hypothetical faucet function or manually inputting a txhash from req
            faucet_txhash = req.get("faucet_txhash", "dummy_faucet_tx_hash")
            output5 = await poll_faucet_tx_until_final(faucet_txhash)
            res.append(f"Faucet transaction polling finished with status: {output5['status']}")
            output6 = await compare_junox_balance_after_faucet(address, output2['amount'])
            res.append(f"Post-faucet balance: {output6['post_amount']}, delta: {output6['delta']}")
            return {"status": "success", "message": f"Faucet request processed. JUNOX delta: {output6['delta']}.",
                    "data": output6}

        case "Store the returned code id in shell variable CODE_ID":
            res = []
            output1 = retrieve_last_tx_output(raw_output)
            res.append("Last transaction output retrieved and parsed.")
            output2 = parse_code_id_from_output(output1)
            res.append(f"Code ID extracted: {output2}")
            output3 = export_shell_variable_CODE_ID(output2)
            res.append("Shell export command generated.")
            return {"status": "success", "message": f"Code ID stored in shell variable. Run: {output3}",
                    "data": {"shell_command": output3}}

        case "Create a new Juno wallet named \"MyWalletName\" by generating a mnemonic and deriving a Juno-compatible address.":
            res = []
            output1 = select_juno_chain_key_params()
            res.append(f"Juno key parameters selected: coin type {output1['bip44_coin_type']}")
            output2 = generate_bip39_mnemonic(words=24)
            res.append("24-word BIP-39 mnemonic generated.")
            output3 = derive_hd_key_from_mnemonic(output2, output1['derivation_path'])
            res.append("HD private key derived from mnemonic.")
            output4 = derive_public_key_and_address(output3, output1['bech32_prefix'])
            res.append(f"Juno address derived: {output4['address']}")
            output5 = persist_wallet_metadata(wallet_name, output4['address'], output4['public_key_hex'], output3)
            res.append(f"Wallet metadata persisted for {output5['address']}.")
            # The original intent has an optional verification step; we skip the function as it is not strictly required.
            network_info = optional_verify_address_on_chain(expected_network=output1['chain_id'])
            res.append(f"Network verification: {network_info['network']}")
            return {"status": "success",
                    "message": f"Wallet '{wallet_name}' created. Address: {output4['address']}",
                    "data": {"address": output4['address'], "mnemonic": output2,
                             "warning": "Store the mnemonic securely!"}}

        case "Set TXFLAGS environment variable with gas settings":
            res = []
            rpc_endpoint = req.get("rpc_endpoint", "https://rpc.junonetwork.io")  # Default RPC for Juno mainnet

            output1 = get_minimum_gas_price()
            res.append(f"Minimum gas price fetched: {output1['raw']}")
            output2 = get_chain_id()
            res.append(f"Chain ID fetched: {output2}")
            output3 = build_txflags_string(output1['raw'], output2, rpc_endpoint=rpc_endpoint)
            res.append("TXFLAGS string constructed.")
            set_txflags_env(output3)
            res.append("TXFLAGS environment variable set.")
            output5 = test_junod_tx_with_txflags(JUNO_SENDER_ADDRESS)
            res.append(f"junod TXFLAGS test (dry-run) succeeded. Return code: {output5['returncode']}")
            return {"status": "success", "message": f"TXFLAGS set and verified: {output3}",
                    "data": {"txflags": output3}}

        case "Check the balance of a given wallet address on Juno":
            res = []
            output1 = validate_juno_address(address)
            res.append(f"Juno address validated: {output1}")
            output2 = await fetch_all_balances_balance(output1)
            res.append(f"Fetched {len(output2)} total balance entries.")
            output3 = await fetch_spendable_balances_balance(output1)
            res.append(f"Fetched {len(output3)} spendable balance entries.")
            output4 = await fetch_denoms_metadata()
            res.append(f"Fetched {len(output4)} denom metadata records.")
            metadata_index = {meta.get('base'): meta for meta in output4}  # Create index for format_balances
            output5 = format_balances(output2, output3, metadata_index)
            res.append(f"Balances formatted into {len(output5)} human-readable entries.")
            return {"status": "success", "message": f"Balance check completed for address {output1}.",
                    "data": {"balances": output5}}

        case _:
            return {"status": "error", "message": "Unknown intent."}


def propose_recipe(query):
    print("Predict for "+query)


@app.post("/generate")
async def handle_generate(request_data: Request):
    """
    Receives text from the frontend, generates a response,
    and returns it as JSON.
    """
    # Get the text from the request body
    req = await request_data.json()
    input_text = req["text"]
    print("QUERY: "+input_text)

    response = await tamples_recipes(input_text, req)

    if response:
        return JSONResponse(content=response)
    else:
        return propose_recipe(input_text)


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
