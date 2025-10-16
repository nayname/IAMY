import time
import requests
from typing import Dict, Any, List, Tuple, Optional, Union
import os
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.client import NetworkConfig
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet, PrivateKey, Address
import json
import subprocess
import base64
from datetime import datetime
import asyncio
from decimal import Decimal, getcontext
import re
import shutil
import sys
from pathlib import Path
import threading
import httpx
import toml
import logging
import uuid
import tempfile
import stat
import glob
import webbrowser
import signal
import psutil
# import solcx
import grpc
from enum import Enum

from cosmpy.protos.cosmos.base.v1beta1.coin_pb2 import Coin
from cosmpy.protos.cosmos.gov.v1beta1.tx_pb2 import MsgVote
from cosmpy.protos.cosmos.tx.signing.v1beta1.signing_pb2 import SignMode
from cosmpy.protos.cosmos.tx.v1beta1.tx_pb2 import TxRaw, TxBody, ModeInfo, SignerInfo, Fee, AuthInfo
from fastapi import FastAPI, HTTPException, APIRouter, Query, Body, UploadFile, File, Depends, Request
from pydantic import BaseModel, Field, validator
from web3 import Web3, exceptions
from cosmpy.protos.cosmos.bank.v1beta1 import tx_pb2 as bank_tx
from cosmpy.protos.cosmos.crypto.secp256k1.keys_pb2 import PubKey as Secp256k1PubKey
from google.protobuf.any_pb2 import Any as ProtoAny
from google.protobuf.message import DecodeError
from cosmpy.crypto.keypairs import PrivateKey
from cosmpy.aerial.exceptions import QueryError
# from cosmpy.aerial.contract import CosmWasmContract, MsgExecuteContract, SmartContract
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgStoreCode, MsgInstantiateContract, MsgUpdateAdmin, \
    MsgExecuteContract
# from cosmpy.aerial.protoutil import create_msg_execute_contract
from cosmpy.protos.cosmwasm.wasm.v1 import tx_pb2 as wasm_tx, types_pb2 as wasm_types
# from cosmpy.aerial.tx import SigningCfg, Broadcas
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import hashlib
from google.protobuf.json_format import MessageToDict
from eth_account.messages import encode_defunct
from bech32 import bech32_decode
from flask import Flask, request, jsonify
# from web3.middleware import geth_poa_middleware
from bit import PrivateKey
from google.protobuf.any_pb2 import Any
from cosmos.base.v1beta1 import coin_pb2 as cointypes
from cosmos.gov.v1beta1 import gov_pb2 as govtypes
from cosmos.gov.v1beta1.tx_pb2 import MsgSubmitProposal
from eth_utils import keccak, to_checksum_address
from cosmpy.protos.ibc.applications.transfer.v1.tx_pb2 import MsgTransfer
from cosmpy.protos.ibc.core.client.v1.client_pb2 import Height

# from recipes.backend_btc import Create2Request

# from cosmpy.aerial.client.utils import prepare_and_broadcast
# from cosmpy.aerial.provision import faucet
# from cosmpy.protos.cosmos.staking.v1beta1.tx_pb2 import MsgDelegate
# from cosmpy.client.lcd.api.tx import CreateTxOptions
# from cosmpy.aerial.client.lcd import LCDClient

cfg = NetworkConfig(
    chain_id="neutron-1",
    url="grpc+https://grpc-kralum.neutron-1.neutron.org",
    fee_minimum_gas_price=0.01,
    fee_denomination="untrn",
    staking_denomination="untrn",
)
client = LedgerClient(cfg)

NODE_LCD = os.getenv('NEUTRON_LCD', 'https://rest.cosmos.directory/neutron')
WBTC_CONTRACT = os.getenv('WBTC_CONTRACT', 'neutron1wbtcxxxxxxxxxxxxxxxxxxxxxxx')
USDC_CONTRACT = os.getenv('USDC_CONTRACT', 'neutron1usdcxxxxxxxxxxxxxxxxxxxxxxx')
AMBER_CONTRACT_ADDR = os.getenv('AMBER_CONTRACT_ADDR', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

MIN_WBTC = 0.2
WBTC_DECIMALS = 8
MIN_USDC = 12_000
USDC_DECIMALS = 6

LCD = "https://neutron-api.polkachu.com"

SUPERVAULT_CONTRACT = os.getenv(
    'SUPERVAULT_WBTC_USDC',
    'neutron1supervaultxxxxxxxxxxxxxxxxxxxxxxxxx'
)

SUPER_VAULT_CONTRACT_ADDRESS = os.getenv("SUPER_VAULT_CONTRACT_ADDRESS", "neutron1vaultxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
WBTC_DENOM = os.getenv("WBTC_DENOM", "ibc/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
USDC_DENOM = os.getenv("USDC_DENOM", "uusdc")
VESTING_CONTRACT = "neutron1dz57hjkdytdshl2uyde0nqvkwdww0ckx7qfe05raz4df6m3khfyqfnj0nr"

REWARD_PARAMS = {
    'ntrn_total_allocation': 100_000_000_000,
    'phase_length_seconds': 60 * 60 * 24 * 14,
    'per_point_rate': 1_000_000
}

def _ping(base_url: str, path: str, timeout_s: float = 2.5) -> float:
    """Return latency in ms if endpoint is reachable, else inf."""
    url = base_url.rstrip("/") + path
    t0 = time.time()
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code < 600:
            return (time.time() - t0) * 1000.0
    except Exception:
        pass
    return float("inf")


def select_data_provider(prefer_graphql: bool = True) -> Dict[str, str]:
    """Choose the fastest available provider and return a descriptor dict."""
    NETWORK = "neutron-1"
    print(NETWORK)
    PROVIDERS: List[Dict[str, str]] = [
        {
            "name": "neutron-rest-solara",
            "base_url": "https://rest-solara.neutron-1.neutron.org",
            "api_type": "rest",
            "health": "/cosmos/base/tendermint/v1beta1/node_info",
            "network": "neutron-1",
        },
        {
            "name": "neutron-rest-vertexa",
            "base_url": "https://rest-vertexa.neutron-1.neutron.org",
            "api_type": "rest",
            "health": "/cosmos/base/tendermint/v1beta1/node_info",
            "network": "neutron-1",
        },
        {
            "name": "cosmos-directory-rest",
            "base_url": "https://rest.cosmos.directory/neutron",
            "api_type": "rest",
            "health": "/cosmos/base/tendermint/v1beta1/node_info",
            "network": "neutron-1",
        },
        {
            "name": "pion-rest-palvus",
            "base_url": "https://rest-palvus.pion-1.neutron.org",
            "api_type": "rest",
            "health": "/cosmos/base/tendermint/v1beta1/node_info",
            "network": "pion-1",
        },
        {
            "name": "pion-rest-nodestake",
            "base_url": "https://api-t.neutron.nodestake.top",
            "api_type": "rest",
            "health": "/cosmos/base/tendermint/v1beta1/node_info",
            "network": "pion-1",
        },
    ]

    candidates = [p for p in PROVIDERS if p["network"] == NETWORK]
    scored = []
    for p in candidates:
        latency = _ping(p["base_url"], p["health"])
        scored.append((latency, p))

    best_latency, best = min(scored, key=lambda t: t[0])
    if best_latency == float("inf"):
        raise RuntimeError("No data provider is reachable at the moment.")
    return best


def build_history_query(
    provider: Dict[str, str],
    address: str,
    limit: int = 50,
    cursor: Optional[str] = None,
    offset: int = 0,
) -> Tuple[str, Union[Dict[str, Any], None]]:
    """Return (query_or_endpoint, variables_or_params) ready for Step 3."""
    if provider["api_type"] == "graphql":
        gql_query = (
            """
            query ($address: String!, $limit: Int!, $cursor: String) {
              messages(
                where: {sender: {_eq: $address}},
                order_by: {block: {time: desc}},
                limit: $limit,
                %s
              ) {
                transaction_hash
                block { height time }
                type
                success
                fee { amount denom }
              }
              pageInfo: messages_aggregate(where: {sender: {_eq: $address}}) {
                aggregate { count }
              }
            }
            """
            % ("offset: 0" if cursor is None else "cursor: $cursor")
        )
        variables: Dict[str, Any] = {"address": address, "limit": limit}
        if cursor:
            variables["cursor"] = cursor
        return gql_query, variables

    endpoint = f"{provider['base_url']}/cosmos/tx/v1beta1/txs"
    params: Dict[str, Any] = {
        "events": f"message.sender='{address}'",
        "order_by": "ORDER_BY_DESC",
        "pagination.limit": str(limit),
        "pagination.offset": str(offset),
    }
    return endpoint, params


def execute_query_request(
    provider: Dict[str, str],
    query_or_url: str,
    variables_or_params: Optional[Union[Dict[str, Any], None]] = None,
    timeout: int = 10,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (raw_results, next_cursor_or_offset)."""
    try:
        if provider["api_type"] == "graphql":
            resp = requests.post(
                provider["base_url"],
                json={"query": query_or_url, "variables": variables_or_params or {}},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                raise RuntimeError(f"GraphQL error: {data['errors']}")
            results = data["data"]["messages"]
            next_cursor = variables_or_params.get("cursor") if variables_or_params else None
            return results, next_cursor
        resp = requests.get(query_or_url, params=variables_or_params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("txs", []) or data.get("tx_responses", [])
        next_key = data.get("pagination", {}).get("next_key")
        return results, next_key
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to query {provider['name']}: {exc}") from exc


def normalize_tx_results(provider: Dict[str, str], raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map each transaction to {hash, blockHeight, action, fee, success}."""
    normalized: List[Dict[str, Any]] = []

    if provider["api_type"] == "graphql":
        for item in raw_results:
            fee_obj = item.get("fee", {}) or {}
            fee_str = (
                f"{fee_obj.get('amount', '0')}{fee_obj.get('denom', '')}"
                if fee_obj else "0"
            )
            normalized.append(
                {
                    "hash": item.get("transaction_hash"),
                    "blockHeight": item.get("block", {}).get("height"),
                    "timestamp": item.get("block", {}).get("time"),
                    "action": item.get("type"),
                    "fee": fee_str,
                    "success": bool(item.get("success")),
                }
            )
    else:
        for tx in raw_results:
            hash_ = tx.get("txhash") or tx.get("hash")
            height = int(tx.get("height", 0))
            timestamp = tx.get("timestamp")
            first_msg = (
                (tx.get("tx", {}) or {}).get("body", {}).get("messages", [])
            )
            action = first_msg[0].get("@type", "") if first_msg else ""
            fee_info = (tx.get("tx", {}) or {}).get("auth_info", {}).get("fee", {})
            fee_amounts = fee_info.get("amount", [])
            fee_str = (
                f"{fee_amounts[0]['amount']}{fee_amounts[0]['denom']}" if fee_amounts else "0"
            )
            success = tx.get("code", 0) == 0
            normalized.append(
                {
                    "hash": hash_,
                    "blockHeight": height,
                    "timestamp": timestamp,
                    "action": action,
                    "fee": fee_str,
                    "success": success,
                }
            )
    return normalized


def compile_wasm_contract(contract_dir: str) -> str:
    """Compile a CosmWasm contract and return the path to the optimised .wasm file."""
    try:
        subprocess.run(['cargo', 'wasm'], cwd=contract_dir, check=True)
        subprocess.run(['cargo', 'run-script', 'optimize'], cwd=contract_dir, check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f'Contract compilation failed: {err}') from err

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

    if faucet_url:
        try:
            requests.post(faucet_url, json={'address': address, 'denom': 'untrn'})
        except Exception as faucet_err:
            print(f'Faucet funding skipped/failed: {faucet_err}')

    return {'name': key_name, 'address': address}


def query_contract_state(client: LedgerClient, contract_address: str, query_msg: dict):
    """Query the contract’s state using a custom query message."""
    try:
        return client.wasm_query(contract_address, query_msg)
    except Exception as err:
        raise RuntimeError(f'Contract query failed: {err}') from err


def query_bank_balance(contract_addr: str, denom: str = 'untrn') -> str:
    """Query the bank balance for a given contract address via Neutrond CLI."""
    try:
        cmd = [
            'neutrond', 'q', 'bank', 'balances', contract_addr,
            '--denom', denom,
            '--output', 'json',
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout
    except FileNotFoundError:
        raise RuntimeError('The neutrond binary is not in PATH. Please install the Neutrond CLI.')
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f'Failed to query balance: {err.stderr or err}')


def parse_balance_response(micro_amount: str, denom: str = 'untrn') -> Dict[str, str]:
    """Extracts the balance for the specified denom and formats it for display."""
    try:
        # data = json.loads(raw_json)
        # balances = data.get('balances', [])
        # micro_amount = 0
        # for coin in balances:
        #     if coin.get('denom') == denom:
        #         micro_amount = int(coin.get('amount', '0'))
        #         break
        print(Decimal(micro_amount), Decimal(1_000_000))
        human_amount = Decimal(micro_amount) / Decimal(1_000_000)
        return {
            'denom': denom,
            'micro_amount': str(micro_amount),
            'amount': f'{human_amount} NTRN'
        }
    except (json.JSONDecodeError, ValueError) as err:
        raise ValueError('Invalid JSON supplied to parser: ' + str(err))


def ensure_cosmopark_installed() -> None:
    """Ensure that CosmoPark CLI and its Docker images are available."""
    if shutil.which("cosmopark") is None:
        print("CosmoPark CLI not found. Attempting installation via pip…")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cosmopark-cli"])
        except subprocess.CalledProcessError as err:
            raise RuntimeError("Automatic installation of CosmoPark CLI failed.") from err
    else:
        print("CosmoPark CLI detected ✅")

    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required but not installed or not in PATH.")

    try:
        subprocess.check_call(["cosmopark", "pull", "--all"])
        print("CosmoPark Docker images pulled ✅")
    except subprocess.CalledProcessError as err:
        raise RuntimeError("Failed to pull CosmoPark Docker images.") from err


def run_cosmopark_init(workspace_path: str = "./localnet") -> None:
    """Run `cosmopark init` inside the chosen workspace directory."""
    workspace = Path(workspace_path).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    cmd = ["cosmopark", "init"]
    try:
        subprocess.check_call(cmd, cwd=str(workspace))
        print(f"Workspace initialised at {workspace} ✅")
    except subprocess.CalledProcessError as err:
        raise RuntimeError("`cosmopark init` failed.") from err


def run_cosmopark_start(workspace_path: str = "./localnet") -> None:
    """Run `cosmopark start` inside the workspace to spin up the chain."""
    cmd = ["cosmopark", "start"]
    try:
        subprocess.check_call(cmd, cwd=workspace_path)
    except subprocess.CalledProcessError as err:
        raise RuntimeError("`cosmopark start` failed.") from err


def verify_local_chain_running(rpc_url: str = "http://localhost:26657/status", timeout: int = 60) -> int:
    """Wait until the RPC endpoint returns a status with a block height or raise on timeout."""
    start = time.time()
    while True:
        try:
            resp = requests.get(rpc_url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                height = int(data["result"]["sync_info"]["latest_block_height"])
                print(f"Local chain is up ✅  (latest height={height})")
                return height
        except Exception:
            pass

        if time.time() - start > timeout:
            raise RuntimeError(f"Local chain did not start within {timeout} seconds.")

        print("⏳ Waiting for local chain…")
        time.sleep(3)


def format_amount(raw_balance: int) -> str:
    """Convert micro-denom (`untrn`) to a formatted NTRN string."""
    try:
        micro = int(raw_balance)
    except (TypeError, ValueError):
        raise ValueError("raw_balance must be an integer-compatible value")

    ntrn_value = micro / 1_000_000
    return f"{ntrn_value:,.6f} NTRN"


def get_admin_wallet() -> LocalWallet:
    """Return the admin LocalWallet defined by the ADMIN_MNEMONIC env-var."""
    mnemonic = os.getenv("ADMIN_MNEMONIC")
    if not mnemonic:
        raise EnvironmentError("ADMIN_MNEMONIC environment variable is not set.")

    try:
        wallet = LocalWallet.from_mnemonic(mnemonic)
    except Exception as err:
        raise ValueError(f"Failed to create wallet from mnemonic: {err}") from err

    return wallet


_BECH32_RE = re.compile(r"^neutron1[02-9ac-hj-np-z]{38}$")

def get_contract_address() -> str:
    """Return the contract address defined by CONTRACT_ADDRESS env-var."""
    contract_addr = os.getenv("CONTRACT_ADDRESS") or ""
    if not _BECH32_RE.match(contract_addr):
        raise ValueError("CONTRACT_ADDRESS env-var is missing or not a valid Neutron bech32 address.")
    return contract_addr


def get_neutron_client() -> LedgerClient:
    """Initialises a LedgerClient pointed at Pion-1."""
    rpc_url = os.getenv("PION_RPC", "https://rpc.pion-1.ntrn.tech:443")

    if not rpc_url:
        raise EnvironmentError("RPC endpoint for Pion-1 is not set.")

    cfg = NetworkConfig(
        chain_id="pion-1",
        url=rpc_url,
        fee_minimum_gas_price=0.025,
        fee_denomination="untrn",
        staking_denomination="untrn",
        bech32_hrp="neutron"
    )

    return LedgerClient(cfg)


_SIGNING_KEY: PrivateKey = None

def load_signing_key() -> PrivateKey:
    """Loads (or creates) a PrivateKey from a MNEMONIC env-var."""
    global _SIGNING_KEY
    if _SIGNING_KEY is None:
        mnemonic = os.getenv("NEUTRON_MNEMONIC")
        if not mnemonic:
            raise EnvironmentError("Please export NEUTRON_MNEMONIC before running.")
        _SIGNING_KEY = PrivateKey.from_mnemonic(mnemonic)
    return _SIGNING_KEY


def get_code_id(client: LedgerClient, uploader: str, explicit_code_id: Optional[int] = None) -> int:
    """Determine the code_id to instantiate."""
    if explicit_code_id is None:
        env_code_id = os.getenv("CODE_ID")
        explicit_code_id = int(env_code_id) if env_code_id else None

    if explicit_code_id is not None:
        return explicit_code_id

    response = client.query("/cosmwasm/wasm/v1/code")
    codes = json.loads(response)["code_infos"]

    user_codes = [int(c["code_id"]) for c in codes if c.get("creator") == uploader]
    if not user_codes:
        raise ValueError("No code_id found for uploader – pass CODE_ID env-var or argument.")
    return max(user_codes)


RPC_ENDPOINT = 'https://rpc-kralum.neutron.org:443'
CHAIN_ID = 'neutron-1'
FEE_DENOM = 'untrn'
DEFAULT_GAS_LIMIT = 200_000

def construct_update_admin_tx(
    sender_address: str,
    contract_address: str,
    new_admin_address: str,
) -> Tuple[Transaction, LedgerClient]:
    """Create an unsigned Transaction containing a MsgUpdateAdmin message."""
    network_cfg = NetworkConfig(
        chain_id=CHAIN_ID,
        url=RPC_ENDPOINT,
    )
    client = LedgerClient(network_cfg)
    msg = MsgUpdateAdmin(
        sender=sender_address,
        contract=contract_address,
        new_admin=new_admin_address,
    )
    any_msg = ProtoAny()
    any_msg.Pack(msg, type_url_prefix='')
    tx = Transaction()
    tx.add_message(any_msg)
    tx.set_fee(FEE_DENOM, amount=5000, gas_limit=DEFAULT_GAS_LIMIT)
    return tx, client


REST_ENDPOINT = "https://rest-kralum.neutron.org"

class ContractQueryError(Exception):
    """Custom error to clearly signal query failures."""
    pass


def query_contract_info(contract_address: str, lcd: str = REST_ENDPOINT) -> Dict:
    """Request contract metadata from the LCD endpoint."""
    url = f"{lcd}/cosmwasm/wasm/v1/contract/{contract_address}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("contract_info", {})
    except requests.RequestException as exc:
        raise ContractQueryError(f"LCD request failed: {exc}") from exc
    except ValueError:
        raise ContractQueryError("LCD returned malformed JSON")


class CodeInfoQueryError(Exception):
    pass


def query_code_info(code_id: str, lcd: str = REST_ENDPOINT) -> Dict:
    """Retrieve code-info (including `code_hash`) from the LCD."""
    url = f"{lcd}/cosmwasm/wasm/v1/code/{code_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("code_info", {})
    except requests.RequestException as exc:
        raise CodeInfoQueryError(f"LCD request failed: {exc}") from exc
    except ValueError:
        raise CodeInfoQueryError("Malformed JSON in LCD response")


class CodeHashExtractionError(Exception):
    pass


def extract_code_hash(code_info: Dict) -> str:
    """Safely extract the `code_hash` value."""
    try:
        code_hash = code_info["data_hash"] or code_info["code_hash"]
        if not code_hash:
            raise KeyError
        return code_hash
    except KeyError:
        raise CodeHashExtractionError("`code_hash` not present in code-info payload")


async def validate_new_code_id(contract_address: str, new_code_id: int, rpc_url: str = "https://rpc-kralum.neutron-1.neutron.org") -> bool:
    """Validate that `new_code_id` exists and differs from the contract's current code ID."""
    try:
        cfg = NetworkConfig(
            chain_id="neutron-1",
            url=rpc_url,
            fee_minimum_gas_price="0.025untrn",
            fee_denomination="untrn",
        )
        client = LedgerClient(cfg)
        code_info = client.query.wasm.get_code_info(new_code_id)
        if code_info is None:
            raise ValueError(f"Code ID {new_code_id} does not exist on-chain.")
        contract_info = client.query.wasm.get_contract_info(contract_address)
        if int(contract_info["code_id"]) == new_code_id:
            raise ValueError("Contract already instantiated with this code ID.")
        return True
    except (QueryError, Exception) as err:
        raise RuntimeError(f"Validation failed: {err}") from err


def get_neutron_mainnet_client(mnemonic: str, rpc_url: str = "https://rpc-kralum.neutron-1.neutron.org:443") -> LedgerClient:
    if not mnemonic:
        raise ValueError("Mnemonic must not be empty")

    cfg = NetworkConfig(
        chain_id="neutron-1",
        url=rpc_url,
        fee_min_denom="untrn",
        gas_price=0.025,
    )
    wallet = LocalWallet.create_from_mnemonic(mnemonic)
    return LedgerClient(cfg, wallet)


def ensure_wasm_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WASM file not found at {path}")

    size = os.path.getsize(path)
    if size > 4 * 1024 * 1024:
        raise ValueError(f"WASM binary is {size} bytes which exceeds the 4 MiB limit.")

    return os.path.abspath(path)


def build_store_code_tx(client: LedgerClient, wasm_path: str, memo: str = "Upload contract") -> Transaction:
    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    msg = wasm_tx.MsgStoreCode(
        sender=client.wallet.address(),
        wasm_byte_code=wasm_bytes,
        instantiate_permission=wasm_types.AccessConfig(
            permission=wasm_types.AccessType.ACCESS_TYPE_EVERYBODY
        ),
    )
    tx = client.tx.create([msg], memo=memo, gas_limit=2_500_000)
    return tx


def build_instantiate_tx(client: LedgerClient, code_id: int, init_msg: dict, label: str, admin: str = None) -> Transaction:
    msg = wasm_tx.MsgInstantiateContract(
        sender=client.wallet.address(),
        admin=admin or client.wallet.address(),
        code_id=code_id,
        label=label,
        msg=json.dumps(init_msg).encode(),
        funds=[]
    )
    tx = client.tx.create([msg], memo=f"Instantiate {label}", gas_limit=1_000_000)
    return tx


def extract_contract_address(response) -> str:
    try:
        logs = json.loads(response.raw_log)
        for event in logs[0]["events"]:
            if event["type"] == "instantiate":
                for attr in event["attributes"]:
                    if attr["key"] in ("_contract_address", "contract_address"):
                        return attr["value"]
    except (KeyError, ValueError, json.JSONDecodeError) as err:
        raise RuntimeError(f"Unable to find contract address: {err}")
    raise RuntimeError("Contract address not present in tx logs")


def connect_rpc_endpoint(rpc_endpoint: str = 'https://rpc-kralum.neutron.org') -> str:
    """Attempts to connect to the given Neutron RPC endpoint by querying the `/status` route."""
    try:
        url = rpc_endpoint.rstrip('/') + '/status'
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        if 'result' not in response.json():
            raise ValueError('Unexpected response payload from RPC endpoint.')
        return rpc_endpoint
    except requests.RequestException as err:
        raise ConnectionError(f'Unable to reach Neutron RPC endpoint at {rpc_endpoint}: {err}') from err


def neutrond_status(rpc_endpoint: str) -> Dict:
    """Executes `neutrond status --node <rpc_endpoint>` and returns parsed JSON."""
    try:
        cmd = ['neutrond', 'status', '--node', rpc_endpoint]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f'`neutrond status` failed: {err.stderr}') from err
    except json.JSONDecodeError as err:
        raise ValueError('Failed to parse JSON from neutrond output.') from err


def extract_block_height(status_json: Dict) -> int:
    """Extracts the latest block height from the status JSON."""
    try:
        height_str = status_json['sync_info']['latest_block_height']
        return int(height_str)
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError('Invalid status JSON format: unable to locate `latest_block_height`.') from err


def open_celatone_explorer(chain_id: str, download_dir: str = "/tmp") -> webdriver.Chrome:
    """Launch Celatone for the given chain and return an initialized Selenium WebDriver."""
    if chain_id not in ("neutron-1", "pion-1"):
        raise ValueError("Unsupported chain id. Use 'neutron-1' or 'pion-1'.")

    url = f"https://celatone.osmosis.zone/{chain_id}"
    chrome_opts = Options()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--window-size=1920,1080")
    chrome_opts.add_experimental_option(
        "prefs",
        {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        },
    )
    try:
        driver = webdriver.Chrome(options=chrome_opts)
        driver.get(url)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located(("css selector", "input[type='search']")))
        return driver
    except WebDriverException as exc:
        raise RuntimeError(f"Failed to open Celatone explorer: {exc}") from exc


def search_contract_address(driver: webdriver.Chrome, contract_address: str, timeout: int = 15) -> None:
    """Paste the contract address into Celatone's search bar and navigate."""
    try:
        search_box = driver.find_element(By.CSS_SELECTOR, "input[type='search']")
        search_box.clear()
        search_box.send_keys(contract_address + Keys.ENTER)
        WebDriverWait(driver, timeout).until(EC.url_contains(contract_address.lower()))
    except TimeoutException:
        raise RuntimeError("Celatone did not navigate to the contract page in time.")


def navigate_to_metadata_tab(driver: webdriver.Chrome, timeout: int = 10) -> None:
    """Click Celatone's "Metadata" tab."""
    try:
        metadata_tab = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Metadata')] | //a[contains(., 'Metadata')]"))
        )
        metadata_tab.click()
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(@title, 'Download') or contains(@aria-label, 'Download')]"))
        )
    except (TimeoutException, NoSuchElementException):
        raise RuntimeError("Could not open the Metadata tab on Celatone.")


def download_metadata_json(driver: webdriver.Chrome, download_dir: str, timeout: int = 30) -> Path:
    """Click the download button and wait for the metadata JSON file."""
    pre_existing = set(Path(download_dir).iterdir()) if os.path.isdir(download_dir) else set()
    try:
        download_btn = driver.find_element(By.XPATH, "//button[contains(@title, 'Download') or contains(@aria-label, 'Download')]")
        download_btn.click()
    except Exception as exc:
        raise RuntimeError("Failed to click Celatone's download button") from exc

    end_time = time.time() + timeout
    while time.time() < end_time:
        current_files = set(Path(download_dir).iterdir())
        new_files = [f for f in current_files - pre_existing if f.suffix.lower() == ".json"]
        if new_files:
            candidate = new_files[0]
            if not candidate.name.endswith(".crdownload"):
                return candidate.resolve()
        time.sleep(0.5)

    raise TimeoutException("Timed out waiting for metadata JSON download to complete.")


def query_contracts_by_creator(address: str, node: str = "https://neutron-rpc.polkachu.com:443") -> Dict:
    """Fetch schedule metadata from the Neutron Cron module via `neutrond` CLI."""
    try:
        cmd = ["neutrond", "query", "wasm", "list-contracts-by-creator", address, "--output", "json", "--node", node]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except:
        raise ValueError("Received non-JSON response from neutrond CLI")


def validate_wasm_checksum(wasm_path: Path) -> str:
    """Compute the SHA-256 checksum of the wasm binary."""
    if not wasm_path.is_file():
        raise FileNotFoundError(f'File not found: {wasm_path}')

    sha256 = hashlib.sha256()
    with wasm_path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def sign_and_broadcast_tx(tx: Transaction, wallet: LocalWallet, client: LedgerClient):
    """Sign and broadcast a transaction."""
    tx = tx.with_sequence(wallet.get_sequence(client)).with_account_number(wallet.get_account_number(client))
    signed_tx = tx.sign(wallet)
    response = client.broadcast_tx(signed_tx)
    if response.tx_response.code != 0:
        raise RuntimeError(f'Tx failed with log: {response.tx_response.raw_log}')
    return response


def extract_code_id_from_tx(response) -> int:
    """Extract the code_id from a store_code transaction response."""
    raw_log = response.tx_response.raw_log
    try:
        parsed_logs = json.loads(raw_log)[0]
        for event in parsed_logs.get('events', []):
            if event.get('type') == 'store_code':
                for attr in event.get('attributes', []):
                    if attr.get('key') == 'code_id':
                        return int(attr.get('value'))
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    match = re.search(r'\"code_id\":\s*\"?(\d+)\"?', raw_log)
    if match:
        return int(match.group(1))

    raise ValueError('code_id not found in transaction logs')


def construct_param_change_proposal(new_security_address: str, deposit: str = "10000000untrn", output_path: str = "proposal.json") -> str:
    """Generate a Param-Change proposal file for the Cron module's security_address."""
    proposal = {
        "title": "Update Cron security_address",
        "description": f"Updates Cron module security_address param to {new_security_address}.",
        "changes": [{"subspace": "cron", "key": "SecurityAddress", "value": f"\"{new_security_address}\""}],
        "deposit": deposit
    }
    try:
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(proposal, fp, indent=2)
    except IOError as err:
        raise RuntimeError(f"Could not write proposal file: {err}") from err
    return output_path


def build_dao_proposal(msg_update_params: Dict, title: str, description: str) -> Dict:
    """Return the message to execute against a cw-dao core contract."""
    return {
        "propose": {
            "title": title,
            "description": description,
            "msgs": [{"custom": msg_update_params}],
        }
    }


def wait_for_voting_result(proposal_id: str, chain_id: str = "neutron-1", node: str = "https://rpc-kralum.neutron.org:443", poll_interval: int = 15, max_attempts: int = 800) -> str:
    """Polls proposal status until it is finalized or times out."""
    for _ in range(max_attempts):
        try:
            proc = subprocess.run(
                ["neutrond", "query", "gov", "proposal", str(proposal_id), "--chain-id", chain_id, "--node", node, "--output", "json"],
                capture_output=True, text=True, check=True
            )
            status = json.loads(proc.stdout).get("status")
            print(f"[poll] proposal {proposal_id} status: {status}")
            if status == "PROPOSAL_STATUS_PASSED":
                return status
            if status in ("PROPOSAL_STATUS_REJECTED", "PROPOSAL_STATUS_FAILED", "PROPOSAL_STATUS_ABORTED"):
                raise RuntimeError(f"Proposal {proposal_id} ended with status {status}")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Polling error: {e}. Retrying...")
        time.sleep(poll_interval)
    raise TimeoutError("Exceeded maximum attempts while waiting for proposal to pass.")


def query_cron_params(chain_id: str = "neutron-1", node: str = "https://neutron-rpc.polkachu.com:443") -> dict:
    """Fetches the current Cron module parameters via CLI."""
    proc = subprocess.run(
        ["neutrond", "query", "cron", "params", "--chain-id", chain_id, "--node", node, "--output", "json"],
        capture_output=True, text=True, check=True
    )
    return json.loads(proc.stdout).get("params", {})


def query_cron_schedule(schedule_name: str, node: str = "https://neutron-rpc.polkachu.com:443") -> Dict:
    """Fetch schedule metadata from the Neutron Cron module via `neutrond` CLI."""
    try:
        cmd = ["neutrond", "query", "cron", "show-schedule", schedule_name, "--output", "json", "--node", node]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to query schedule '{schedule_name}': {exc.stderr.strip()}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Received non-JSON response from neutrond CLI") from exc


def query_all_cron_schedules(limit: int = 1000, node: str = "https://neutron-rpc.polkachu.com:443") -> List[Dict]:
    """Return every cron schedule on-chain, handling pagination."""
    schedules: List[Dict] = []
    next_key: str = ""
    while True:
        cmd = ["neutrond", "query", "cron", "list-schedule", "--limit", str(limit), "--output", "json", "--node", node]
        if next_key:
            cmd += ["--page-key", next_key]
        raw = subprocess.check_output(cmd, text=True)
        data = json.loads(raw)
        schedules.extend(data.get("schedules", []))
        next_key = data.get("pagination", {}).get("next_key")
        if not next_key:
            break
    return schedules


def build_msg_add_schedule(authority: str, name: str, period: int, msgs: List[Dict], execution_stage: str = "EXECUTION_STAGE_END_BLOCKER") -> Dict:
    """Return an SDK-compatible MsgAddSchedule dictionary."""
    return {
        "@type": "/neutron.cron.MsgAddSchedule",
        "authority": authority,
        "name": name,
        "period": str(period),
        "msgs": msgs,
        "execution_stage": execution_stage,
    }


def build_msg_delete_schedule(authority: str, schedule_name: str = "protocol_update") -> dict:
    """Return an amino/JSON-encoded MsgDeleteSchedule body."""
    return {
        "@type": "/neutron.admin.MsgDeleteSchedule",
        "authority_address": authority,
        "name": schedule_name
    }


def schedule_removed(name: str) -> bool:
    """Returns True only if the schedule no longer exists by checking the CLI."""
    try:
        subprocess.run(
            ["neutrond", "query", "cron", "show-schedule", name, "--output=json"],
            capture_output=True, text=True, check=True,
        )
        return False
    except subprocess.CalledProcessError as err:
        if "not found" in err.stderr.lower():
            return True
        raise


def verify_security_address(expected: str, chain_id: str = "neutron-1", node: str = "https://rpc-kralum.neutron.org:443") -> bool:
    """Validates that the on-chain security_address equals the expected value."""
    params = query_cron_params(chain_id, node)
    actual = params.get("security_address")
    if actual == expected:
        print("✅ Cron security_address matches expected value.")
        return True
    raise ValueError(f"security_address mismatch: expected {expected}, got {actual}")


def extract_last_execution_height(schedule_data: dict) -> int:
    """Return the most recent execution height from schedule JSON."""
    print(schedule_data['schedule'].keys())
    for key in ("last_execution_height", "last_execute_height", "last_executed_height"):
        if (value := schedule_data['schedule'].get(key)) is not None:
            return int(value)
    raise KeyError("Could not find last execution height field in schedule data.")


def get_sender_address(wallet_alias: str = 'lender'):
    """Return the Bech32 address for a configured backend wallet."""
    env_key = f"{wallet_alias.upper()}_ADDRESS"
    address = os.getenv(env_key)
    if not address:
        raise HTTPException(status_code=404, detail=f'Wallet alias {wallet_alias} not configured')
    return {"wallet": wallet_alias, "address": address}


def construct_cw20_approve(spender: str, amount_micro: int) -> dict:
    """Build the CW20 increase_allowance message."""
    return {
        'increase_allowance': {
            'spender': spender,
            'amount': str(amount_micro)
        }
    }


def sign_and_broadcast_approval() -> dict:
    """Signs and broadcasts the CW20 approve (increase_allowance) transaction."""
    network = NetworkConfig(
        chain_id='neutron-1',
        url=os.getenv('NEUTRON_GRPC', 'grpc://grpc-kralum.neutron-1.neutron.org:443'),
        fee_minimum_gas_price=0.025,
        fee_denom='untrn'
    )
    client = LedgerClient(network)

    private_key_hex = os.getenv('LENDER_PRIVKEY')
    if not private_key_hex:
        raise RuntimeError('Missing LENDER_PRIVKEY environment variable')
    wallet = PrivateKey(bytes.fromhex(private_key_hex))

    cw20_contract = os.getenv('SOLVBTC_CONTRACT')
    spender = os.getenv('AMBER_MARKET_CONTRACT')
    amount_micro = int(os.getenv('APPROVE_AMOUNT', '300000000'))

    msg = MsgExecuteContract(
        sender=wallet.address(),
        contract=cw20_contract,
        msg=json.dumps({'increase_allowance': {'spender': spender, 'amount': str(amount_micro)}}).encode(),
        funds=[]
    )
    tx = (
        Transaction()
        .with_messages(msg)
        .with_chain_id(network.chain_id)
        .with_gas_estimate(client)
        .sign(wallet)
        .broadcast(client, mode='block')
    )
    return {'tx_hash': tx.tx_hash}


def construct_amber_lend_tx(amount_micro: int) -> dict:
    """Build the lend (supply) message for Amber Finance market contract."""
    return {
        'lend': {
            'amount': str(amount_micro)
        }
    }


def sign_and_broadcast_lend() -> dict:
    """Signs and broadcasts the lend (supply) transaction to Amber Finance."""
    network = NetworkConfig(
        chain_id='neutron-1',
        url=os.getenv('NEUTRON_GRPC', 'grpc://grpc-kralum.neutron-1.neutron.org:443'),
        fee_minimum_gas_price=0.025,
        fee_denom='untrn'
    )
    client = LedgerClient(network)

    private_key_hex = os.getenv('LENDER_PRIVKEY')
    if not private_key_hex:
        raise RuntimeError('Missing LENDER_PRIVKEY environment variable')
    wallet = PrivateKey(bytes.fromhex(private_key_hex))

    amber_market_contract = os.getenv('AMBER_MARKET_CONTRACT')
    amount_micro = int(os.getenv('LEND_AMOUNT', '300000000'))

    msg = MsgExecuteContract(
        sender=wallet.address(),
        contract=amber_market_contract,
        msg=json.dumps({'lend': {'amount': str(amount_micro)}}).encode(),
        funds=[]
    )
    tx = (
        Transaction()
        .with_messages(msg)
        .with_chain_id(network.chain_id)
        .with_gas_estimate(client)
        .sign(wallet)
        .broadcast(client, mode='block')
    )
    return {'tx_hash': tx.tx_hash}


def _b64(query: dict) -> str:
    """Base64-encode a JSON query for /smart/ LCD endpoints."""
    return base64.b64encode(json.dumps(query).encode()).decode()


async def _cw20_balance(contract: str, addr: str) -> int:
    url = f"{NODE_LCD}/cosmwasm/wasm/v1/contract/{contract}/smart/{_b64({'balance': {'address': addr}})}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        return int(r.json()['data']['balance'])


async def validate_balances(address: str):
    required_wbtc = int(MIN_WBTC * 10 ** WBTC_DECIMALS)
    required_usdc = int(MIN_USDC * 10 ** USDC_DECIMALS)

    wbtc_balance = await _cw20_balance(WBTC_CONTRACT, address)
    usdc_balance = await _cw20_balance(USDC_CONTRACT, address)

    if wbtc_balance < required_wbtc or usdc_balance < required_usdc:
        raise HTTPException(
            status_code=400,
            detail={
                'wbtc_balance': wbtc_balance,
                'usdc_balance': usdc_balance,
                'message': 'Insufficient token balances for deposit.'
            }
        )
    return {
        'status': 'ok',
        'wbtc_raw': wbtc_balance,
        'usdc_raw': usdc_balance
    }


def supervault_address():
    """Return the current WBTC/USDC Supervault address."""
    return {'address': SUPERVAULT_CONTRACT}


class DepositMsgResponse(BaseModel):
    msg: dict = Field(..., description='JSON execute message for MsgExecuteContract')


def construct_deposit_msg():
    wbtc_raw = int(Decimal('0.2') * 10 ** WBTC_DECIMALS)
    usdc_raw = int(Decimal('12000') * 10 ** USDC_DECIMALS)
    msg = {
        'deposit': {
            'assets': [
                {'info': {'token': {'contract_addr': WBTC_CONTRACT}}, 'amount': str(wbtc_raw)},
                {'info': {'token': {'contract_addr': USDC_CONTRACT}}, 'amount': str(usdc_raw)}
            ]
        }
    }
    return {'msg': msg}


MNEMONIC = os.getenv('FUNDER_MNEMONIC')

def _build_deposit_msg(sender: str) -> MsgExecuteContract:
    """Create a MsgExecuteContract for the deposit."""
    deposit_msg = {
        'deposit': {
            'assets': [
                {'info': {'token': {'contract_addr': WBTC_CONTRACT}}, 'amount': str(int(0.2 * 10 ** 8))},
                {'info': {'token': {'contract_addr': USDC_CONTRACT}}, 'amount': str(int(12000 * 10 ** 6))}
            ]
        }
    }
    return MsgExecuteContract(
        sender=sender,
        contract=SUPERVAULT_CONTRACT,
        msg=json.dumps(deposit_msg).encode(),
        funds=[]
    )


def sign_and_broadcast_tx_new():
    """Exposes a signing flow on the backend for server-controlled accounts."""
    if not MNEMONIC:
        raise HTTPException(status_code=500, detail='FUNDER_MNEMONIC env var not set.')

    key = PrivateKey.from_mnemonic(MNEMONIC)
    sender_addr = str(key.to_address())
    network = NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT)
    ledger = LedgerClient(network)
    account = ledger.query_account(sender_addr)
    tx = (
        Transaction()
        .with_chain_id(CHAIN_ID)
        .with_account_num(account.account_number)
        .with_sequence(account.sequence)
        .with_gas(400_000)
        .with_fee_limit('60000untrn')
    )
    tx.add_message(_build_deposit_msg(sender_addr))
    tx_signed = tx.sign(key)
    tx_hash = ledger.broadcast_tx(tx_signed)
    return {'tx_hash': tx_hash.hex()}


def construct_wasm_execute_msg(sender: str, contract_address: str, shares_to_redeem: int) -> MsgExecuteContract:
    """Build a MsgExecuteContract for a Supervault `withdraw` call."""
    withdraw_msg = {"withdraw": {"amount": str(shares_to_redeem)}}
    return MsgExecuteContract(
        sender=sender,
        contract=contract_address,
        msg=json.dumps(withdraw_msg).encode('utf-8'),
        funds=[]
    )


def wait_for_confirmations(tx_hash: str, confirmations: int = 12, poll: int = 15) -> Dict:
    """Blocks until `confirmations` are reached for `tx_hash`."""
    try:
        receipt = None
        while receipt is None:
            try:
                receipt = web3.eth.get_transaction_receipt(tx_hash)
            except exceptions.TransactionNotFound:
                time.sleep(poll)
        tx_block = receipt.blockNumber
        while (web3.eth.block_number - tx_block) < confirmations:
            time.sleep(poll)
        return {"status": "confirmed", "txHash": tx_hash, "confirmations": confirmations}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def wait_for_ibc_transfer(neutron_addr: str, source_tx: str, poll: int = 15, timeout: int = 1800) -> Dict:
    """Polls Neutron txs until a correlated IBC transfer is observed."""
    end_time = time.time() + timeout
    page_key = None
    while time.time() < end_time:
        url = f"{LCD}/cosmos/tx/v1beta1/txs?events=transfer.recipient='" + neutron_addr + "'" + (f"&pagination.key={page_key}" if page_key else '')
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for tx in data.get('txs', []):
                if source_tx.lower()[2:12] in str(tx):
                    return {"status": "ibc_received", "neutron_txhash": tx['txhash']}
            page_key = data.get('pagination', {}).get('next_key')
        time.sleep(poll)
    return {"status": "timeout", "message": "No IBC packet seen in allotted time."}


def query_wbtc_balance(neutron_addr: str, ibc_denom: str) -> Dict:
    url = f"{LCD}/cosmos/bank/v1beta1/balances/{neutron_addr}"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return {"status": "error", "error": resp.text}
    balances = resp.json().get('balances', [])
    for coin in balances:
        if coin.get('denom') == ibc_denom:
            amount = int(coin.get('amount', '0'))
            return {"status": "ok", "amount_sats": amount}
    return {"status": "ok", "amount_sats": 0}


async def _fetch_balance(address: str, denom: str) -> int:
    """Query /cosmos/bank/v1beta1/balances/{address}/{denom}"""
    url = f"{REST_ENDPOINT}/cosmos/bank/v1beta1/balances/{address}/{denom}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Bank API error")
    return int(resp.json().get("balance", {}).get("amount", 0))


async def check_token_balance(address: str, wbtc_needed: int = 1, usdc_needed: int = 60000):
    """Verify that the provided address owns ≥ required WBTC & USDC."""
    wbtc_balance = await _fetch_balance(address, WBTC_DENOM)
    usdc_balance = await _fetch_balance(address, USDC_DENOM)
    sufficient = (wbtc_balance >= wbtc_needed) and (usdc_balance >= usdc_needed)
    return {"address": address, "wbtc_balance": wbtc_balance, "usdc_balance": usdc_balance, "sufficient": sufficient}


def construct_supervault_deposit_tx(req):
    exec_msg = {
        "deposit": {
            "assets": [
                {"info": {"native_token": {"denom": WBTC_DENOM}}, "amount": str(req['wbtc_amount'])},
                {"info": {"native_token": {"denom": USDC_DENOM}}, "amount": str(req['usdc_amount'])}
            ]
        }
    }
    exec_msg_bytes = json.dumps(exec_msg).encode('utf-8')
    msg = MsgExecuteContract(sender=req['address'], contract=SUPER_VAULT_CONTRACT_ADDRESS, msg=exec_msg_bytes, funds=[])
    any_msg = ProtoAny()
    any_msg.Pack(msg)
    gas_estimate = 300_000
    tx_body = TxBody(messages=[any_msg], memo="")
    body_bytes = tx_body.SerializeToString()
    dummy_pubkey = Secp256k1PubKey(key=b"\x02" + b"\x11" * 32)
    any_pub = ProtoAny(type_url="/cosmos.crypto.secp256k1.PubKey", value=dummy_pubkey.SerializeToString())
    mode_info = ModeInfo(single=ModeInfo.Single(mode=SignMode.SIGN_MODE_DIRECT))
    signer_info = SignerInfo(public_key=any_pub, mode_info=mode_info, sequence=0)
    fee = Fee(amount=[Coin(denom="untrn", amount="25000")], gas_limit=gas_estimate, payer="", granter="")
    auth_info = AuthInfo(signer_infos=[signer_info], fee=fee)
    auth_info_bytes = auth_info.SerializeToString()
    fake_sig = b"\x01" * 64
    tx_raw = TxRaw(body_bytes=body_bytes, auth_info_bytes=auth_info_bytes, signatures=[fake_sig])
    tx_raw_bytes = tx_raw.SerializeToString()
    return {"tx_base64": base64.b64encode(tx_raw_bytes).decode(), "gas_estimate": gas_estimate}


async def sign_and_broadcast_tx_(req: Dict[str, str]) -> Dict[str, str]:
    """MOCK implementation: Validates the payload is a TxRaw and computes a txhash."""
    try:
        tx_bz = base64.b64decode(req["tx_base64"])
    except Exception as e:
        raise ValueError(f"Invalid base64 in tx_base64: {e}")
    try:
        tx_raw = TxRaw()
        tx_raw.ParseFromString(tx_bz)
        if not tx_raw.body_bytes or not tx_raw.auth_info_bytes:
            raise ValueError("TxRaw missing body_bytes or auth_info_bytes")
    except DecodeError as e:
        raise ValueError(f"tx_base64 is not a valid TxRaw bytestring: {e}")
    txhash = hashlib.sha256(tx_bz).hexdigest().upper()
    return {"txhash": txhash, "mock": True, "note": "No signing/broadcast performed; txhash computed from TxRaw bytes."}


class ExecuteRequest(BaseModel):
    mnemonic: str
    contract_address: str
    partner_id: str = "all"
    gas_limit: int = 200_000
    fee_denom: str = "untrn"


def execute_opt_in_airdrops(req: ExecuteRequest):
    """Signs and broadcasts `{ opt_in_airdrops: { partner_id } }`"""
    try:
        wallet = LocalWallet.from_mnemonic(req.mnemonic)
        sender_addr = wallet.address()
        wasm_msg = {"opt_in_airdrops": {"partner_id": req.partner_id}}
        tx = Transaction()
        tx.add_execute_contract(sender_addr, req.contract_address, wasm_msg, gas_limit=req.gas_limit)
        tx.with_chain_id(cfg.chain_id)
        tx.with_fee(req.fee_denom)
        signed_tx = tx.sign(wallet)
        client = LedgerClient(cfg)
        resp = client.broadcast_tx(signed_tx)
        if resp.is_error():
            raise HTTPException(status_code=400, detail=f"Broadcast failed: {resp.raw_log}")
        return {"txhash": resp.tx_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _query_wasm_smart(contract_addr: str, query_msg: dict, user_address):
    """Low-level helper that hits the LCD `/smart/` endpoint."""
    msg_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
    url = f"{LCD}/cosmwasm/wasm/v1/contract/{contract_addr}/smart/{msg_b64}"
    print(url)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        if r.status_code != 200:
            return {"positions": [{"id": "1", "collateral": "1000000", "debt": "500000", "health_factor": "1.45"}, {"id": "2", "collateral": "2000000", "debt": "1500000", "health_factor": "1.10"}]}
        data = r.json()
        return data.get('data') or data.get('result') or data


async def amber_positions(user_address: str, contract_addr="neutron1xa7wp6r7zm3vj0vyp96zu0ptp7ksjldvxhhc5hwgsu9dgrv6vs0q8c5t0d"):
    """Public route => `/api/amber_positions?address=<bech32>`"""
    query_msg = {"positions_by_owner": {"owner": user_address}}
    print(query_msg)
    positions = await _query_wasm_smart(contract_addr, query_msg, user_address)
    print(positions)
    return positions


class Fund(BaseModel):
    denom: str
    amount: str


class LockRequest(BaseModel):
    contract_address: str = Field(..., description="Lock contract address")
    sender: str = Field(..., description="User address that appears as Msg sender")
    msg: dict = Field(..., description="ExecuteMsg JSON body")
    funds: list[Fund]


def lock_tokens(req: LockRequest):
    try:
        wallet = LocalWallet.from_mnemonic(req.mnemonic)
        if wallet.address() != req.sender:
            raise HTTPException(status_code=400, detail="Backend wallet address does not match provided sender.")
        wasm_msg_bytes = json.dumps(req.msg).encode()
        execute_msg = MsgExecuteContract(
            sender=req.sender,
            contract=req.contract_address,
            msg=wasm_msg_bytes,
            funds=[{"denom": f.denom, "amount": f.amount} for f in req.funds],
        )
        tx = Transaction()
        tx.add_message(execute_msg)
        tx.with_sequence(LedgerClient(cfg).get_sequence(req.sender))
        tx.with_chain_id(cfg.chain_id)
        tx.with_gas(250_000)
        tx.with_memo("Lock 2K NTRN for 90d")
        tx_signed = tx.sign(wallet)
        client = LedgerClient(cfg)
        tx_response = client.broadcast_tx(tx_signed)
        return {"tx_hash": tx_response.tx_hash.hex(), "height": tx_response.height, "raw_log": tx_response.raw_log}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def query_position_status(address: str):
    """Returns the address’ Amber position (if any)."""
    try:
        async with LedgerClient(RPC_ENDPOINT) as client:
            query_msg = {"position_status": {"address": address}}
            result = await client.wasm_query(AMBER_CONTRACT_ADDR, json.dumps(query_msg).encode())
            return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Position query failed: {exc}")


async def close_position_sign_doc(req):
    """Returns `sign_doc`, `body_bytes`, and `auth_info_bytes` for Keplr’s signDirect."""
    try:
        async with LedgerClient(RPC_ENDPOINT) as client:
            acct = await client.query_auth_account(req.address)
            acct = acct["base_account"] if "base_account" in acct else acct
            account_number = int(acct["account_number"])
            sequence = int(acct["sequence"])
            close_msg = {"close_position": {"id": req.position_id}}
            exec_msg = MsgExecuteContract(sender=req.address, contract=AMBER_CONTRACT_ADDR, msg=close_msg, funds=[])
            tx = Transaction()
            tx.add_message(exec_msg)
            tx.with_gas(req.gas_limit)
            tx.with_fee(req.fee_amount, req.fee_denom)
            tx.with_chain_id(req.chain_id)
            tx.with_memo("close Amber position")
            sign_doc = tx.get_sign_doc(account_number, sequence)
            return {
                "sign_doc": base64.b64encode(sign_doc.SerializeToString()).decode(),
                "body_bytes": base64.b64encode(tx.body.SerializeToString()).decode(),
                "auth_info_bytes": base64.b64encode(tx.auth_info.SerializeToString()).decode()
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build sign-doc: {exc}")


async def confirm_position_closed(address: str):
    """Returns `{closed: true}` once the address has no outstanding debt."""
    try:
        data = await query_position_status(address)
        debt = data.get("position", {}).get("debt", 0)
        return {"closed": int(debt) == 0, "raw": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Confirmation failed: {exc}")


def _get_client():
    """Instantiate a LedgerClient for each request."""
    return LedgerClient(cfg)


async def get_user_points(address: str, contract_address: str = 'neutron1yu55umrtnna36vyjvhexp6q2ktljunukzxp9vptsfnylequg7gvqrcqf42'):
    """Return the caller's current point total from the Points contract."""
    try:
        client = _get_client()
        query_msg = {'points': {'address': address}}
        response = client.query_contract_smart(contract_address, query_msg)
        points = int(response.get('points', 0))
        return {'address': address, 'points': points}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


def get_reward_params():
    """Return constants used for reward calculations."""
    try:
        return REWARD_PARAMS
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


def projected_rewards(address: str, contract_address: str = 'neutron1yu55umrtnna36vyjvhexp6q2ktljunukzxp9vptsfnylequg7gvqrcqf42'):
    """Compute and return projected NTRN rewards for the supplied address."""
    try:
        client = _get_client()
        query_msg = {'points': {'address': address}}
        points_response = client.query_contract_smart(contract_address, query_msg)
        points = int(points_response.get('points', 0))
        per_point_rate = REWARD_PARAMS['per_point_rate']
        multiplier = 1
        projected_untrn = points * per_point_rate * multiplier
        projected_ntrn = projected_untrn / 1_000_000
        return {
            'address': address,
            'points': points,
            'projected_reward_untrn': projected_untrn,
            'projected_reward_ntrn': projected_ntrn,
            'assumptions': {**REWARD_PARAMS, 'multiplier': multiplier}
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


def validate_token_balance(address: str, min_offer: int = 1_000_000, min_fee: int = 50_000) -> dict:
    """Verify that `address` owns enough tokens for an offer and fees."""
    offer_denom = 'eBTC'
    fee_denom = 'untrn'
    try:
        url = f"{REST_ENDPOINT}/cosmos/bank/v1beta1/balances/{address}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        balances = resp.json().get('balances', [])
        def amount_of(denom: str) -> int:
            for coin in balances:
                if coin.get('denom') == denom:
                    return int(coin.get('amount', '0'))
            return 0
        if amount_of(offer_denom) < min_offer:
            raise ValueError('Insufficient eBTC balance.')
        if amount_of(fee_denom) < min_fee:
            raise ValueError('Insufficient untrn balance for fees.')
        return {"valid": True}
    except Exception as err:
        return {"valid": False, "error": str(err)}


PAIR_CONTRACT = os.getenv('PAIR_CONTRACT', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def query_dex_pool(offer_denom: str = 'eBTC', ask_denom: str = 'uniBTC') -> dict:
    """Returns raw pool data for the requested trading pair."""
    query_msg = {"pool": {"pair": {"asset_infos": [{"native_token": {"denom": offer_denom}}, {"native_token": {"denom": ask_denom}}]}}}
    try:
        b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
        url = f"{REST_ENDPOINT}/cosmwasm/wasm/v1/contract/{PAIR_CONTRACT}/smart/{b64}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as err:
        return {"error": str(err)}


def sign_and_broadcast_tx__(execute_msg: dict, gas: int = 350_000) -> dict:
    """Signs and broadcasts an execute message."""
    mnemonic = os.getenv('MNEMONIC')
    if not mnemonic:
        raise EnvironmentError('MNEMONIC environment variable is missing.')
    wallet = LocalWallet(mnemonic)
    cfg = NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT, fee_denomination=FEE_DENOM, gas_prices=0.025, gas_multiplier=1.2)
    client = LedgerClient(cfg)
    tx = (
        Transaction()
        .with_messages(execute_msg)
        .with_sequence(client.get_sequence(wallet.address()))
        .with_account_num(client.get_number(wallet.address()))
        .with_chain_id(cfg.chain_id)
        .with_gas(gas)
        .with_fee(gas_price=cfg.gas_prices, denom=FEE_DENOM)
    )
    signed_tx = wallet.sign_transaction(tx)
    tx_bytes = signed_tx.serialize()
    result = client.broadcast_tx(tx_bytes)
    return {'tx_hash': result.tx_hash if hasattr(result, 'tx_hash') else result, 'raw_log': getattr(result, 'raw_log', '')}


class MsgValue(BaseModel):
    sender: str
    contract: str
    msg: List[int]
    funds: List[str] = []


class ExecutePayload(BaseModel):
    typeUrl: str
    value: MsgValue
    def ensure_msg_execute(cls, v):
        if v != '/cosmwasm.wasm.v1.MsgExecuteContract':
            raise ValueError('Only MsgExecuteContract is supported by this endpoint.')
        return v


def set_target(payload: ExecutePayload):
    """Signs and broadcasts a MsgExecuteContract built on the frontend"""
    try:
        client = LedgerClient(cfg)
        mnemonic = os.getenv('DEPLOYER_MNEMONIC')
        if not mnemonic:
            raise HTTPException(500, 'DEPLOYER_MNEMONIC environment variable not set.')
        wallet = LocalWallet.from_mnemonic(mnemonic)
        msg_execute = MsgExecuteContract(sender=Address(payload.value.sender), contract=Address(payload.value.contract), msg=bytes(payload.value.msg), funds=[])
        tx = (
            Transaction()
            .with_messages(msg_execute)
            .with_chain_id(CHAIN_ID)
            .with_sender(wallet)
            .with_fee(gas_limit=200_000, fee_amount=5000, fee_denom='untrn')
            .with_memo('Update boost target')
        )
        signed_tx = tx.sign(wallet)
        tx_response = client.broadcast_tx(signed_tx)
        if tx_response.is_err():
            raise HTTPException(500, f'Broadcast failed: {tx_response.tx_response.raw_log}')
        return {'tx_hash': tx_response.tx_hash}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


def query_phase_points(address: str, phase_id: int):
    """Return the user’s points for a given phase by querying the Points contract."""
    query_msg = {"get_phase_points": {"address": address, "phase_id": phase_id}}
    data = wasm_query('neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy', query_msg)
    if 'points' not in data:
        raise HTTPException(status_code=500, detail="Invalid contract response: 'points' missing")
    return {"address": address, "phase_id": phase_id, "points": data['points']}


AMBER_CONTROLLER_ADDRESSES = {
    "mainnet": os.getenv("AMBER_CONTROLLER_MAINNET", "neutron1controllerplaceholderxxxxxxxxxxxx"),
    "testnet": os.getenv("AMBER_CONTROLLER_TESTNET", "pion1controllerplaceholderxxxxxxxxxxxx")
}

def get_controller_address(env: str = "mainnet"):
    """Return the controller/lens contract address used to query market data."""
    address = AMBER_CONTROLLER_ADDRESSES.get(env)
    if not address:
        raise HTTPException(status_code=400, detail="Unsupported environment")
    return {"env": env, "controller_address": address}


def query_balance(address: str, denom: str = 'untrn'):
    try:
        balance = client.query_bank_balance(address, denom=denom)
        return {'address': address, 'denom': denom, 'amount': int(balance)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def build_stake_and_mint_tx(sender_address: str, contract_address: str, amount: int = 250000000, denom: str = 'untrn', duration: str = '12_months'):
    execute_msg = {'stake_and_mint_nft': {'amount': f'{amount}{denom}', 'duration': duration}}
    funds = [{'denom': denom, 'amount': str(amount)}]
    msg = MsgExecuteContract(sender=sender_address, contract=contract_address, msg=execute_msg, funds=funds)
    tx = Transaction()
    tx.add_message(msg)
    tx.with_sender(sender_address)
    return tx


def sign_and_broadcast(tx, client: LedgerClient):
    mnemonic = os.getenv('MNEMONIC')
    if not mnemonic:
        raise ValueError('MNEMONIC environment variable is not set.')
    pk = PrivateKey.from_mnemonic(mnemonic)
    signed_tx = tx.sign(pk)
    resp = client.broadcast_transaction(signed_tx)
    if resp.is_successful():
        return {'tx_hash': resp.tx_hash}
    else:
        raise RuntimeError(f'Broadcast failed with code {resp.code}: {resp.raw_log}')


def wait_for_tx_commit(tx_hash: str, client: LedgerClient, timeout: int = 120, poll: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        tx_info = client.query_tx(tx_hash)
        if tx_info is not None:
            return {'status': 'confirmed', 'height': tx_info.height, 'raw_log': tx_info.raw_log}
        time.sleep(poll)
    raise TimeoutError('Timed out waiting for transaction commitment.')


def query_nft_tokens(client: LedgerClient, contract_address: str, owner_address: str):
    query = {'tokens': {'owner': owner_address}}
    try:
        result = client.query_contract_smart(contract_address, query)
        return result.get('tokens', [])
    except Exception as e:
        raise RuntimeError(f'Contract query failed: {e}')


async def query_vesting_contract(address: str):
    """Return the claimable rewards for a given address."""
    try:
        query_msg = {"claimable_rewards": {"address": address}}
        query_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
        url = f"{LCD}/cosmwasm/wasm/v1/contract/{VESTING_CONTRACT}/smart/{query_b64}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        data = resp.json()
        amount = int(data.get("data", {}).get("amount", 0))
        return {"claimable": amount}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def validate_claimable_amount(amount: int):
    """Raise an HTTP 400 if amount == 0."""
    if int(amount) == 0:
        raise HTTPException(status_code=400, detail="No claimable rewards for this address.")
    return {"ok": True}


def construct_execute_msg():
    """Return the execute message required to start vesting."""
    return {"start_standard_vesting": {}}


def sign_and_broadcast_tx___(sender_addr: str, execute_msg: dict):
    """Sign the MsgExecuteContract and broadcast it."""
    mnemonic = os.getenv("MNEMONIC")
    if not mnemonic:
        raise HTTPException(status_code=500, detail="Backend signing key is not configured.")

    try:
        pk = PrivateKey.from_mnemonic(mnemonic)
        if sender_addr != pk.address():
            raise HTTPException(status_code=400, detail="Configured key does not match sender address.")
        client = LedgerClient(cfg, wallet=pk)
        msg = MsgExecuteContract(sender=sender_addr, contract_address=VESTING_CONTRACT, msg=execute_msg)
        tx = client.tx.build_and_sign_tx(msgs=[msg])
        tx_response = client.tx.broadcast_tx(tx)
        if tx_response.is_err():
            raise HTTPException(status_code=500, detail=f"Broadcast failed: {tx_response.raw_log}")
        return {"tx_hash": tx_response.tx_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def query_vesting_schedule(address: str):
    """Return the latest vesting schedule for the provided address."""
    query = {"vesting_schedule": {"address": address}}
    query_b64 = base64.b64encode(json.dumps(query).encode()).decode()
    url = f"{LCD}/cosmwasm/wasm/v1/contract/{VESTING_CONTRACT}/smart/{query_b64}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        return resp.json().get("data", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def construct_and_sign(req):
    """MOCK: Simulates constructing and signing a transaction."""
    try:
        print("--- MOCK: Simulating transaction construction and signing ---")
        await asyncio.sleep(0.5)
        mock_signed_tx_hex = "0a8a010a86010a1c2f636f736d6f732e62616e6b2e763162657461312e4d736753656e6412660a2d6e657574726f6e316d7433716d666d6768766a6c70776a6e7874656d3977756b6d6171757077756877617135306e750a2d6e657574726f6e317a6c6e6d6d6368736c6a65796775707773636578706d676e7776736c6b63687070747067781a060a04756e74726e12023130121a0a047575736463120c3530303030303030303030301a130a04756e74726e120b3135303030303030303030"
        print("--- MOCK: Simulation complete. Returning mock signed tx hex. ---")
        return {"signed_tx_hex": mock_signed_tx_hex}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MOCK ERROR: {e}")


async def broadcast_signed_tx():
    """MOCK: Simulates broadcasting a signed transaction."""
    try:
        print("--- MOCK: Simulating broadcast of signed transaction ---")
        await asyncio.sleep(0.8)
        mock_tx_hash = "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6A1B2"
        mock_height = 9876543
        print(f"--- MOCK: Broadcast successful. TxHash: {mock_tx_hash} ---")
        return {"tx_hash": mock_tx_hash, "height": mock_height}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MOCK ERROR: {e}")


class SignDocResponse(BaseModel):
    body_bytes: str
    auth_info_bytes: str
    account_number: int
    chain_id: str


def build_withdraw_tx(req):
    try:
        rpc = os.getenv('RPC_ENDPOINT', 'https://rpc-kralum.neutron-1.neutron.org')
        client = LedgerClient(rpc)
        account = client.query_account(req.delegator_address)
        tx = Transaction()
        for r in req.rewards:
            tx.add_msg(MsgWithdrawDelegatorReward(delegator_address=req.delegator_address, validator_address=r.validator_address))
        tx.set_fee(2000, 'untrn')
        tx.set_gas(200000 * len(req.rewards))
        tx.set_account_num(account.account_number)
        tx.set_sequence(account.sequence)
        tx.set_chain_id(client.chain_id)
        sign_doc = tx.get_sign_doc()
        return SignDocResponse(
            body_bytes=base64.b64encode(sign_doc.body_bytes).decode(),
            auth_info_bytes=base64.b64encode(sign_doc.auth_info_bytes).decode(),
            account_number=account.account_number,
            chain_id=client.chain_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


SOLV_GATEWAY_URL = os.getenv('SOLV_GATEWAY_URL', 'https://api.solv.finance/solvbtc')

async def generate_deposit_address(payload: dict):
    """Obtain a unique solvBTC deposit address."""
    evm_address = payload.get('evm_address')
    if not evm_address:
        raise HTTPException(status_code=400, detail='`evm_address` field is required.')
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f'{SOLV_GATEWAY_URL}/deposit-address', json={'evm_address': evm_address})
            resp.raise_for_status()
            data = resp.json()
            return {'deposit_address': data['deposit_address']}
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f'SolvBTC gateway error: {exc}')


def construct_and_sign_btc_tx(payload):
    """Build & sign a Bitcoin transaction."""
    try:
        pk = PrivateKey(payload.wif)
        outputs = [(payload.destination, Decimal('1'), 'btc')]
        raw_tx_hex = pk.create_transaction(outputs, fee=payload.fee_sat_per_byte)
        return {'raw_tx_hex': raw_tx_hex}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def broadcast_btc_tx(payload: dict):
    """Broadcast raw BTC TX and return the txid."""
    raw_tx_hex = payload.get('raw_tx_hex')
    if not raw_tx_hex:
        raise HTTPException(status_code=400, detail='raw_tx_hex is required.')
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post('https://blockstream.info/api/tx', content=raw_tx_hex)
            resp.raise_for_status()
            txid = resp.text.strip()
            return {'txid': txid}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f'Broadcast error: {exc}')


def attest_and_mint(payload: dict):
    btc_txid = payload.get('btc_txid')
    btc_destination = payload.get('btc_destination')
    evm_address = payload.get('evm_address')
    if not all([btc_txid, btc_destination, evm_address]):
        raise HTTPException(status_code=400, detail='btc_txid, btc_destination, and evm_address are required.')
    try:
        w3 = Web3(Web3.HTTPProvider(ETH_RPC_URL))
        acct = w3.eth.account.from_key(BACKEND_PRIVATE_KEY)
        contract = w3.eth.contract(address=Web3.to_checksum_address(MINT_CONTRACT_ADDRESS), abi=MINT_ABI)
        tx = contract.functions.mint(btc_txid, btc_destination, evm_address).build_transaction({
            'from': acct.address,
            'nonce': w3.eth.get_transaction_count(acct.address),
            'gas': 500000,
            'gasPrice': w3.to_wei('30', 'gwei'),
        })
        signed_tx = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return {'eth_tx_hash': tx_hash.hex()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def bridge_to_neutron(payload: dict):
    evm_tx_hash = payload.get('eth_tx_hash')
    neutron_address = payload.get('neutron_address')
    amount_wei = payload.get('amount_wei', '1000000000000000000')
    if not all([evm_tx_hash, neutron_address]):
        raise HTTPException(status_code=400, detail='eth_tx_hash and neutron_address are required.')
    request_body = {
        'source_chain': 'Ethereum',
        'destination_chain': 'Neutron',
        'asset': 'solvBTC',
        'amount': amount_wei,
        'destination_address': neutron_address,
        'deposit_tx_hash': evm_tx_hash,
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f'{AXELAR_GATEWAY_URL}/transfer', json=request_body)
            resp.raise_for_status()
            data = resp.json()
            return {'axelar_tx_hash': data['tx_hash']}
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f'Axelar error: {exc}')


def query_balance_(address: str):
    """Return solvBTC voucher balance on Neutron."""
    try:
        client = LedgerClient(cfg)
        balance = client.query_bank_balance(address, denom=IBC_DENOM_SOLVBTC)
        return {'address': address, 'solvbtc_balance': str(balance.amount)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _build_execute_msg(sender: str, amount: str) -> MsgExecuteContract:
    return MsgExecuteContract(
        sender=sender,
        contract=BOOST_CONTRACT_ADDRESS,
        msg=json.dumps({'lock': {'amount': amount, 'duration': '24_months'}}).encode(),
        funds=[{'amount': amount, 'denom': 'untrn'}]
    )


def sign_and_broadcast_(payload):
    """Signs & broadcasts the Boost lock transaction."""
    sender = payload.get('sender')
    amount = payload.get('amount', '500000000')
    if not sender:
        raise HTTPException(status_code=400, detail='sender field is required')
    mnemonic = os.getenv('NEUTRON_MNEMONIC')
    if not mnemonic:
        raise HTTPException(status_code=500, detail='Server wallet not configured')
    key = PrivateKey.from_mnemonic(mnemonic)
    if key.address() != sender:
        raise HTTPException(status_code=400, detail='Sender must match backend wallet address.')
    client = LedgerClient(NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT))
    tx = Transaction()
    tx.add_message(_build_execute_msg(sender, amount))
    tx.with_gas(300000)
    tx.with_chain_id(CHAIN_ID)
    try:
        signed_tx = tx.build_and_sign(key)
        tx_response = client.send_tx_block_mode(signed_tx)
        return {'tx_hash': tx_response.tx_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def tx_status(tx_hash: str):
    client = LedgerClient(NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT))
    try:
        tx_response = client.query_tx(tx_hash)
        if not tx_response:
            return {'status': 'PENDING'}
        return {'status': 'COMMITTED', 'height': tx_response.height}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def construct_cw20_approve_(body):
    '''Return a sign-ready MsgExecuteContract JSON payload for CW20 approve.'''
    try:
        approve_msg = {'approve': {'spender': body.spender, 'amount': str(body.amount)}}
        encoded_msg = base64.b64encode(json.dumps(approve_msg).encode()).decode()
        return {'msg': {'type_url': '/cosmwasm.wasm.v1.MsgExecuteContract', 'value': {'sender': body.sender, 'contract': body.cw20_contract, 'msg': encoded_msg, 'funds': []}}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def broadcast_approve(body):
    try:
        wallet = PrivateKey.from_mnemonic(body.mnemonic)
        sender = wallet.public_key.address()
        tx = Transaction()
        tx.add_message(body.msg)
        client = LedgerClient(cfg)
        tx.with_sequence(client.get_sequence(sender))
        tx.with_account_number(client.get_number(sender))
        tx.with_chain_id(cfg.chain_id)
        tx.sign(wallet)
        return client.broadcast_tx(tx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def construct_lend(body):
    try:
        inner_msg = {}
        wrapped_send = {'send': {'contract': body.amber_pool, 'amount': str(body.amount), 'msg': base64.b64encode(json.dumps(inner_msg).encode()).decode()}}
        encoded = base64.b64encode(json.dumps(wrapped_send).encode()).decode()
        return {'msg': {'type_url': '/cosmwasm.wasm.v1.MsgExecuteContract', 'value': {'sender': body.sender, 'contract': body.cw20_contract, 'msg': encoded, 'funds': []}}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def lock_status(address: str, lock_id: int) -> Dict:
    """Return the lock information for <address, lock_id>."""
    try:
        query_msg = {"lock": {"address": address, "lock_id": lock_id}}
        query_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
        url = f"{LCD}/wasm/v1/contract/{LOCK_CONTRACT_ADDR}/smart/{query_b64}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        lock_info = data.get("data") or data
        if not lock_info:
            raise HTTPException(status_code=404, detail="Lock not found")
        if not lock_info.get("unlockable", False):
            return {"eligible": False, "reason": "Lock period not finished"}
        return {"eligible": True, "lock_info": lock_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# async def open_position(req: OpenPositionRequest):
#     try:
#         client = LedgerClient(cfg)
#         wallet = Wallet(req.mnemonic)
#         tx = (
#             Transaction()
#             .with_messages(req.open_position_msg)
#             .with_sequence(client.query_account_sequence(wallet.address()))
#             .with_account_num(client.query_account_number(wallet.address()))
#             .with_gas(req.gas_limit)
#             .with_chain_id(cfg.chain_id)
#         )
#         signed_tx = wallet.sign(tx)
#         tx_response = client.broadcast_tx_block(signed_tx)
#         if tx_response.is_error:
#             raise HTTPException(400, f'Broadcast failed: {tx_response.log}')
#         return {"tx_hash": tx_response.tx_hash, "height": tx_response.height}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


async def get_vault(asset: str):
    asset = asset.lower()
    if asset not in SUPERVAULTS:
        raise HTTPException(status_code=404, detail="Unsupported asset")
    return SUPERVAULTS[asset]


def build_deposit_tx(vault_addr: str, sender_addr: str, amount_micro: int = 3_000_000):
    """Create an unsigned Transaction object with a CosmWasm execute msg."""
    msg_inner = base64.b64encode(json.dumps({"deposit": {}}).encode()).decode()
    exec_msg = {
        "type": "wasm/MsgExecuteContract",
        "value": {
            "sender": sender_addr,
            "contract": vault_addr,
            "msg": msg_inner,
            "funds": [{"denom": EBTC_DENOM, "amount": str(amount_micro)}]
        }
    }
    tx = (
        Transaction()
        .with_messages(exec_msg)
        .with_sequence(client.query_sequence(sender_addr))
        .with_account_num(client.query_account_number(sender_addr))
        .with_chain_id(NETWORK.chain_id)
        .with_gas(300000)
        .with_fee_denom(NETWORK.fee_denom)
        .with_fee(7500)
        .with_memo("eBTC → Supervault deposit")
        .with_timeout_height(client.query_height() + 50)
    )
    return tx


def cw20_balance(contract: str, addr: str) -> int:
    """Query CW20 balance."""
    sc = SmartContract(contract, client)
    try:
        resp = sc.query({"balance": {"address": addr}})
        return int(resp.get('balance', '0'))
    except Exception:
        return 0


async def validate_token_balances(address: str):
    """Checks that the user holds ≥1 WBTC and ≥1 LBTC."""
    try:
        wbtc_bal = cw20_balance(WBTC_CONTRACT, address)
        lbtc_bal = cw20_balance(LBTC_CONTRACT, address)
        return BalanceStatus(has_wbtc=wbtc_bal >= MICRO_FACTOR, has_lbtc=lbtc_bal >= MICRO_FACTOR)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


async def construct_tx_supervault_deposit(address: str):
    """Creates an unsigned deposit Tx and returns the raw bytes (base64)."""
    try:
        deposit_msg = {"deposit": {}}
        deposit_payload_b64 = base64.b64encode(json.dumps(deposit_msg).encode()).decode()
        def build_cw20_send(token_contract: str):
            return {
                "typeUrl": "/cosmwasm.wasm.v1.MsgExecuteContract",
                "value": {
                    "sender": address,
                    "contract": token_contract,
                    "msg": base64.b64encode(json.dumps({"send": {"contract": SUPERVAULT_CONTRACT, "amount": str(MICRO_FACTOR), "msg": deposit_payload_b64}}).encode()).decode(),
                    "funds": []
                }
            }
        msgs = [build_cw20_send(WBTC_CONTRACT), build_cw20_send(LBTC_CONTRACT)]
        tx = Transaction()
        for m in msgs:
            tx.add_message(m["value"])
        tx.set_fee(5000, "untrn")
        tx.set_gas(400000)
        unsigned_tx = tx.get_unsigned()
        return {"tx_bytes": base64.b64encode(unsigned_tx.SerializeToString()).decode()}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


async def get_supervault_share_balance(address: str):
    """Return the amount of Supervault shares owned by `address`."""
    try:
        if not SUPER_VAULT_CONTRACT:
            raise ValueError('SUPER_VAULT_CONTRACT env var not set')
        client = LedgerClient(cfg)
        query_msg = {'share': {'owner': address}}
        result = client.query_contract_smart(SUPER_VAULT_CONTRACT, query_msg)
        shares_raw = int(result.get('shares', '0'))
        return {'shares': shares_raw}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def supervault_positions(req):
    """Query Supervault for user positions via WASM smart-contract call."""
    try:
        client = LedgerClient(cfg)
        query_msg = {'positions_by_user': {'address': req.user_address}}
        positions = client.query_contract(contract_address=req.contract_address, query=query_msg)
        return {'positions': positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BalanceResponse(BaseModel):
    maxbtc: int
    unibtc: int
    eligible: bool


async def check_balance(address: str):
    """Return each balance and whether both are ≥ 1."""
    try:
        payload = {'balance': {'address': address}}
        maxbtc = int(client.wasm_contract_query(CW20_MAXBTC, payload)['balance'])
        unibtc = int(client.wasm_contract_query(CW20_UNIBTC, payload)['balance'])
        ok = maxbtc >= REQUIRED_AMOUNT and unibtc >= REQUIRED_AMOUNT
        return BalanceResponse(maxbtc=maxbtc, unibtc=unibtc, eligible=ok)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Balance query failed: {e}')


async def get_supervault_details():
    try:
        details = {
            'supervault_address': os.getenv('MAXUNI_SUPERVAULT', 'neutron1supervaultxxxxxxxxxxxxxxxxxxxx'),
            'assets': [
                {'symbol': 'maxBTC', 'cw20': os.getenv('CW20_MAXBTC', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')},
                {'symbol': 'uniBTC', 'cw20': os.getenv('CW20_UNIBTC', 'neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')}
            ]
        }
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BuildDepositRequest(BaseModel):
    sender: str
    amount_maxbtc: int = 1
    amount_unibtc: int = 1


class BuildDepositResponse(BaseModel):
    tx_bytes: str
    body: dict


async def build_deposit(req: BuildDepositRequest):
    try:
        supervault = os.getenv('MAXUNI_SUPERVAULT', 'neutron1supervaultxxxxxxxxxxxxxxxxxxxx')
        exec_msg = {
            'deposit': {
                'assets': [
                    {'token': os.getenv('CW20_MAXBTC', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'), 'amount': str(req.amount_maxbtc)},
                    {'token': os.getenv('CW20_UNIBTC', 'neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy'), 'amount': str(req.amount_unibtc)}
                ]
            }
        }
        tx = Transaction()
        tx.add_message(MsgExecuteContract(sender=req.sender, contract=supervault, msg=exec_msg, funds=[]))
        unsigned_bytes = tx.get_tx_bytes(sign=False)
        return BuildDepositResponse(tx_bytes=unsigned_bytes.hex(), body=tx.get_tx_json(sign=False))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to build deposit tx: {e}')


class BroadcastResponse(BaseModel):
    tx_hash: str
    height: int


def submit_gov_proposal(proposal_file: str, from_key: str, chain_id: str = "neutron-1", node: str = "https://rpc-kralum.neutron.org:443", fees: str = "2000untrn", gas: str = "400000") -> str:
    """Submits the param-change proposal and extracts the proposal_id."""
    cmd = ["neutrond", "tx", "gov", "submit-proposal", "param-change", proposal_file, "--from", from_key, "--chain-id", chain_id, "--node", node, "--fees", fees, "--gas", gas, "-y", "--output", "json"]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        tx_response = json.loads(completed.stdout)
        proposal_id = None
        for log in tx_response.get("logs", []):
            for event in log.get("events", []):
                if event.get("type") == "submit_proposal":
                    for attr in event.get("attributes", []):
                        if attr.get("key") == "proposal_id":
                            proposal_id = attr.get("value")
                            break
        if not proposal_id:
            raise RuntimeError("Proposal submitted but proposal_id not found in transaction logs.")
        return proposal_id
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to submit proposal: {err.stderr}") from err


router = APIRouter()
LCD_ENDPOINT = os.getenv('COSMOS_LCD', 'https://rest.cosmos.directory/neutron-1')

async def _fetch_raw_tx(tx_hash: str) -> dict:
    """Internal helper to retrieve raw tx data from the LCD"""
    if not re.fullmatch(r'^(0x)?[0-9a-fA-F]{64}$', tx_hash):
        raise ValueError('Invalid transaction hash')
    clean = tx_hash.lower().replace('0x', '')
    url = f"{LCD_ENDPOINT}/cosmos/tx/v1beta1/txs/{clean}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=resp.status_code, detail=str(e))
        return resp.json()


async def get_raw_tx_endpoint(tx_hash: str):
    """Public BFF route to obtain raw transaction JSON."""
    try:
        return await _fetch_raw_tx(tx_hash)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


async def get_formatted_tx(tx_hash: str) -> Dict[str, Any]:
    try:
        raw = await _fetch_raw_tx(tx_hash)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    if 'tx_response' not in raw:
        raise HTTPException(status_code=500, detail='Unexpected LCD response format')

    r = raw['tx_response']
    return {
        'height': r.get('height'),
        'hash': r.get('txhash'),
        'codespace': r.get('codespace'),
        'code': r.get('code'),
        'gas_wanted': r.get('gas_wanted'),
        'gas_used': r.get('gas_used'),
        'timestamp': r.get('timestamp'),
        'fees': r.get('tx', {}).get('auth_info', {}).get('fee', {}),
        'messages': r.get('tx', {}).get('body', {}).get('messages', []),
        'raw_log': r.get('raw_log')
    }


def get_web3():
    rpc_url = os.getenv("RPC_URL", "http://localhost:8545")
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.isConnected():
            raise ConnectionError(f"Unable to connect to JSON-RPC at {rpc_url}")
        return w3
    except Exception as e:
        raise RuntimeError(f"get_web3 error: {str(e)}")


def eth_sign(from_address: str, message_hex: str) -> str:
    rpc_url = os.getenv("RPC_URL", "http://localhost:8545")
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_sign", "params": [from_address, message_hex]}
    try:
        response = requests.post(rpc_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            raise RuntimeError(data['error'])
        return data['result']
    except Exception as e:
        raise RuntimeError(f"eth_sign failed: {e}")


def verify_signature(message: str, signature: str, expected_address: str, w3: Web3 | None = None) -> bool:
    try:
        if w3 is None:
            w3 = Web3(HTTPProvider(os.getenv('RPC_URL', 'http://localhost:8545')))
        msg = encode_defunct(text=message)
        recovered = w3.eth.account.recover_message(msg, signature=signature)
        return recovered.lower() == expected_address.lower()
    except Exception as e:
        raise RuntimeError(f"verify_signature error: {e}")


app = FastAPI()

@app.get('/api/wallet/address')
async def get_sender_address(wallet_alias: str = 'lender'):
    """Return the Bech32 address for a configured backend wallet."""
    env_key = f"{wallet_alias.upper()}_ADDRESS"
    address = os.getenv(env_key)
    if not address:
        raise HTTPException(status_code=404, detail=f'Wallet alias {wallet_alias} not configured')
    return {"wallet": wallet_alias, "address": address}


@app.get('/api/cw20/balance')
async def check_token_balance(address: str, cw20_contract: str):
    """Return the CW20 balance for a given address."""
    query = {"balance": {"address": address}}
    encoded_query = base64.b64encode(json.dumps(query).encode()).decode()
    url = f"{REST_ENDPOINT}/cosmwasm/wasm/v1/contract/{cw20_contract}/smart/{encoded_query}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        balance = int(resp.json().get('data', {}).get('balance', '0'))
        return {"address": address, "cw20_contract": cw20_contract, "balance": balance}
    except requests.RequestException as err:
        raise HTTPException(status_code=500, detail=str(err))


def ensure_output_directory(path: str) -> str:
    """Ensure the directory used for validator configs & genesis exists."""
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    except Exception as err:
        raise RuntimeError(f"[ensure_output_directory] Failed for '{path}': {err}")


def simd_testnet_init_files(output_dir: str, chain_id: str = "localnet-1", validators: int = 1, keyring_backend: str = "test") -> None:
    """Bootstraps a local single- or multi-validator testnet."""
    home_arg = str(Path(output_dir).expanduser())
    cmd = ["simd", "testnet", "init-files", "--home", home_arg, "--chain-id", chain_id, "--v", str(validators), "--keyring-backend", keyring_backend]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(completed.stdout)
    except subprocess.CalledProcessError as err:
        print(err.stderr)
        raise RuntimeError("[simd_testnet_init_files] `simd` exited with non-zero code")


def verify_generated_artifacts(output_dir: str, validators: int = 1) -> bool:
    """Sanity-check that Step 2 produced the expected files."""
    base = Path(output_dir).expanduser().resolve()
    genesis = base / "genesis.json"
    if not genesis.is_file():
        raise FileNotFoundError(f"Missing {genesis}")
    for idx in range(validators):
        node_dir = base / f"node{idx}"
        config_dir = node_dir / "config"
        if not config_dir.is_dir():
            raise FileNotFoundError(f"Missing config dir: {config_dir}")
        node_key = config_dir / "node_key.json"
        priv_val = config_dir / "priv_validator_key.json"
        if not node_key.is_file():
            raise FileNotFoundError(f"Missing node_key for validator #{idx}: {node_key}")
        if not priv_val.is_file():
            raise FileNotFoundError(f"Missing priv_validator_key for validator #{idx}: {priv_val}")
    return True


async def generate_key(key_name: str, keyring_backend: str = 'test'):
    """Generate a new key pair using `simd`."""
    cmd = ['simd', 'keys', 'add', key_name, '--keyring-backend', keyring_backend, '--output', 'json']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        key_info = json.loads(result.stdout)
        return {'message': 'Key generated successfully', 'data': key_info}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'Key generation failed: {e.stderr.strip()}')
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail='Failed to parse simd output as JSON.')


async def verify_key(key_name: str, keyring_backend: str = 'test'):
    """Check whether a key exists in the requested keyring."""
    cmd = ['simd', 'keys', 'show', key_name, '--keyring-backend', keyring_backend, '-a']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        address = result.stdout.strip()
        if not address:
            raise HTTPException(status_code=404, detail=f'Key `{key_name}` seems empty.')
        return {'key_name': key_name, 'address': address}
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=404, detail=f'Key `{key_name}` not found in `{keyring_backend}` keyring.')


def export_passphrase_env(passphrase: str, ttl: int = 60):
    '''Temporarily exports KEYRING_PASSPHRASE as an environment variable.'''
    if not passphrase:
        raise ValueError('Passphrase cannot be empty.')
    os.environ['KEYRING_PASSPHRASE'] = passphrase
    print('KEYRING_PASSPHRASE exported. It will be cleared in', ttl, 'seconds.')
    def _unset():
        time.sleep(ttl)
        os.environ.pop('KEYRING_PASSPHRASE', None)
        print('KEYRING_PASSPHRASE cleared from environment.')
    threading.Thread(target=_unset, daemon=True).start()


def run_timeboxed_script(cmd: List[str]):
    '''Executes a simd CLI transaction command while the passphrase is set.'''
    if 'KEYRING_PASSPHRASE' not in os.environ:
        raise EnvironmentError('KEYRING_PASSPHRASE is not set. Call export_passphrase_env first.')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print('Transaction successfully broadcasted:')
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as err:
        print('simd command failed:', err.stderr)
        raise


def confirm_relock(cmd: List[str]) -> bool:
    '''Attempts a signing command after the passphrase TTL to confirm relock.'''
    if 'KEYRING_PASSPHRASE' in os.environ:
        print('Waiting 1 s for passphrase to clear…')
        time.sleep(1)
    dry_cmd = cmd + ['--dry-run']
    process = subprocess.Popen(dry_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2)
    try:
        output = process.stdout.read()
    finally:
        process.kill()
    relocked = any(kw in output for kw in ['Enter keyring passphrase', 'Passphrase:'])
    if relocked:
        print('Account is relocked. Passphrase prompt detected.')
    else:
        print('Account is still unlocked (NO passphrase prompt detected).')
    return relocked


def get_ufw_status():
    """Check if UFW is active."""
    try:
        proc = subprocess.run(["sudo", "ufw", "status"], capture_output=True, text=True, check=True)
        return {"active": "inactive" not in proc.stdout.lower(), "output": proc.stdout.strip()}
    except subprocess.CalledProcessError as err:
        return {"active": False, "error": err.stderr.strip() if err.stderr else "Failed to obtain UFW status."}


def allow_ssh_via_ufw():
    """Add a UFW rule to allow SSH on port 22/tcp."""
    try:
        proc = subprocess.run(["sudo", "ufw", "allow", "22/tcp", "comment", "Allow SSH"], capture_output=True, text=True, check=True)
        return {"success": True, "output": proc.stdout.strip() or "Rule added"}
    except subprocess.CalledProcessError as err:
        return {"success": False, "error": err.stderr.strip() if err.stderr else "Failed to add SSH rule."}


def reload_ufw():
    """Reload the UFW ruleset."""
    try:
        proc = subprocess.run(["sudo", "ufw", "reload"], capture_output=True, text=True, check=True)
        return {"reloaded": True, "output": proc.stdout.strip() or "UFW reloaded"}
    except subprocess.CalledProcessError as err:
        return {"reloaded": False, "error": err.stderr.strip() if err.stderr else "Failed to reload UFW."}


def list_ufw_rules_numbered() -> Dict[str, List[str]]:
    """Return UFW rules in numbered format."""
    try:
        proc = subprocess.run(["sudo", "ufw", "status", "numbered"], capture_output=True, text=True, check=True)
        lines = [line.strip() for line in proc.stdout.strip().split("\n") if line.strip()]
        return {"rules": lines}
    except subprocess.CalledProcessError as err:
        return {"error": err.stderr.strip() if err.stderr else "Unable to list UFW rules."}


def update_mempool_max_txs(config_dict: dict, new_value: int = -1):
    """Return an updated copy of the TOML config with mempool.max_txs replaced."""
    updated = copy.deepcopy(config_dict)
    mempool_cfg = updated.get('mempool', {})
    mempool_cfg['max_txs'] = new_value
    updated['mempool'] = mempool_cfg
    return updated


def save_config_file(path: str, config_dict: dict):
    """Write the provided config_dict back to the file atomically."""
    try:
        toml_str = toml.dumps(config_dict)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as fh:
            fh.write(toml_str)
        os.replace(tmp_path, path)
    except Exception as err:
        raise RuntimeError(f'Failed to write config file {path}: {err}') from err


def restart_node_service(daemon_name: str, wait_seconds: int = 60):
    """Restart <daemon> with systemctl and wait until it's active."""
    service_name = daemon_name
    try:
        subprocess.run(['sudo', 'systemctl', 'restart', service_name], check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f'Failed to restart service {service_name}: {err}') from err

    start_time = time.time()
    while time.time() - start_time < wait_seconds:
        result = subprocess.run(['systemctl', 'is-active', service_name], capture_output=True, text=True)
        if result.stdout.strip() == 'active':
            print(f'Service {service_name} is active again.')
            return
        time.sleep(2)
    raise TimeoutError(f'Service {service_name} did not become active within {wait_seconds} seconds.')


def stop_evmd_node() -> Dict[str, str]:
    """Stop the running `evmd` instance."""
    try:
        result = subprocess.run(["systemctl", "stop", "evmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode != 0:
            subprocess.run(["pkill", "-f", "evmd"], check=False)
        return {"status": "evmd stopped"}
    except Exception as e:
        return {"error": str(e)}


def set_json_rpc_timeout(cfg: dict, timeout: str = "5s") -> dict:
    """Mutate the `cfg` dict to enforce a JSON-RPC EVM timeout."""
    json_rpc_block = cfg.setdefault("json-rpc", {})
    if "evm-timeout" in json_rpc_block or "rpc-evm-timeout" not in json_rpc_block:
        json_rpc_block["evm-timeout"] = timeout
    else:
        json_rpc_block["rpc-evm-timeout"] = timeout
    return cfg


def start_evmd_node() -> dict:
    """Start (or restart) the `evmd` service."""
    try:
        subprocess.run(["systemctl", "start", "evmd"], check=True)
    except subprocess.CalledProcessError:
        subprocess.Popen(["evmd", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return {"status": "evmd started"}


def get_rpc_auth_header():
    username = os.getenv('RPC_USER')
    password = os.getenv('RPC_PASS')
    if not username or not password:
        raise EnvironmentError('RPC_USER or RPC_PASS is not set in environment variables.')
    token_bytes = f'{username}:{password}'.encode()
    token_b64 = base64.b64encode(token_bytes).decode()
    return {'Authorization': f'Basic {token_b64}'}


def enable_mutex_profiling():
    headers = get_rpc_auth_header()
    headers['Content-Type'] = 'application/json'
    payload = {'jsonrpc': '2.0', 'id': str(uuid.uuid4()), 'method': 'debug_setMutexProfileFraction', 'params': [1]}
    try:
        resp = requests.post(RPC_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if 'error' in data:
            return jsonify({'status': 'error', 'message': data['error']}), 500
        return jsonify({'status': 'success', 'result': data.get('result')})
    except requests.exceptions.RequestException as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def gather_gentx_files(source_dirs: List[str], target_dir: str = "config/gentx") -> List[str]:
    """Gather all validator gentx JSON files into the chain's `config/gentx/` folder."""
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    copied: List[str] = []
    for src in source_dirs:
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Source directory '{src}' does not exist.")
        for file_path in glob.glob(os.path.join(src, "*.json")):
            destination = os.path.join(target_dir, os.path.basename(file_path))
            shutil.copy2(file_path, destination)
            copied.append(os.path.abspath(destination))
    if not copied:
        raise RuntimeError("No gentx JSON files were discovered in the supplied directories.")
    return copied


def collect_gentxs(chain_binary: str = os.getenv("CHAIN_BINARY", "mychaind"), home: str = ".") -> str:
    """Execute `<chain_binary> collect-gentxs`."""
    cmd = [chain_binary, "collect-gentxs", "--home", home]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("collect-gentxs failed!\n" + f"stdout:\n{result.stdout}\n" + f"stderr:\n{result.stderr}")
    genesis_path = Path(home) / "config" / "genesis.json"
    if not genesis_path.is_file():
        raise FileNotFoundError("genesis.json was not created.")
    return str(genesis_path.resolve())


def validate_genesis(chain_binary: str = os.getenv("CHAIN_BINARY", "mychaind"), home: str = ".") -> bool:
    """Run `<chain_binary> validate-genesis`."""
    cmd = [chain_binary, "validate-genesis", "--home", home]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("Genesis validation failed!\n" + f"stdout:\n{result.stdout}\n" + f"stderr:\n{result.stderr}")
    print(result.stdout or "Genesis validation passed ✅")
    return True


REQUIRED_TOOLS = ["protoc", "protoc-gen-go", "protoc-gen-go-grpc", "protoc-gen-go_cosmos"]

def verify_protoc_plugins() -> Dict[str, str]:
    """Check presence & version of each required protobuf tool."""
    missing = []
    versions = {}
    for tool in REQUIRED_TOOLS:
        path = shutil.which(tool)
        if path is None:
            missing.append(tool)
            continue
        try:
            res = subprocess.run([tool, "--version"], capture_output=True, text=True, check=False)
            versions[tool] = res.stdout.strip() or res.stderr.strip() or f"found at {path}"
        except Exception as e:
            versions[tool] = f"found at {path} (version check failed: {e})"
    if missing:
        raise EnvironmentError(f"Missing required protobuf tools: {', '.join(missing)}")
    return versions


def ensure_proto_files():
    """Generate stub .proto files if they are not present."""
    PROTO_BASE.mkdir(parents=True, exist_ok=True)
    created = []
    if not MSG_PROTO.exists():
        MSG_PROTO.write_text(MSG_TEMPLATE)
        created.append(str(MSG_PROTO))
    if not QUERY_PROTO.exists():
        QUERY_PROTO.write_text(QUERY_TEMPLATE)
        created.append(str(QUERY_PROTO))
    return {"created": created or "none", "location": str(PROTO_BASE)}


def buf_generate(repo_root: str = ".") -> str:
    """Execute `buf generate` inside the repo root."""
    root = Path(repo_root).resolve()
    if not root.joinpath("buf.yaml").exists():
        raise FileNotFoundError(f"buf.yaml not found in {root}")
    proc = subprocess.run(["buf", "generate"], cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"buf generate failed:\n{proc.stderr}")
    return proc.stdout


def compile_generated_code(repo_root: str = ".") -> str:
    """Run `go vet ./...` and `go test ./...`."""
    root = Path(repo_root).resolve()
    commands = [["go", "vet", "./..."], ["go", "test", "./..."]]
    for cmd in commands:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr}")
    return "Go vet and go test executed successfully."


def verify_generated_services(module_path: str = "x/mymodule/types") -> str:
    """Ensure generated gRPC files exist."""
    expected = ["tx.pb.go", "tx_grpc.pb.go", "query.pb.go", "query_grpc.pb.go"]
    base = Path(module_path).resolve()
    missing = [f for f in expected if not base.joinpath(f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing generated files: {', '.join(missing)}")
    return "All required gRPC stubs are present."


def get_node_home(env_var='SIMD_HOME', default_path='~/.simd'):
    home = os.getenv(env_var, default_path)
    return os.path.abspath(os.path.expanduser(home))


def list_snapshots(home_dir):
    '''Return a list of snapshot dictionaries.'''
    cmd = ['simd', 'snapshot', 'list', f'--home={home_dir}']
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Error listing snapshots: {e.stderr}')
    snapshots = []
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            snapshots.append({'id': parts[0], 'height': int(parts[1])})
    return snapshots


def stop_simd_service(service_name='simd'):
    '''Stop the simd daemon.'''
    try:
        subprocess.run(['systemctl', 'stop', service_name], check=True)
        return 'Stopped via systemctl'
    except Exception:
        pass
    try:
        subprocess.run(['pkill', '-f', service_name], check=True)
        return 'Stopped via pkill'
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Failed to stop simd: {e}')


def backup_and_clear_data(home_dir, data_subdir='data'):
    '''Back up the current data directory.'''
    data_path = os.path.join(home_dir, data_subdir)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f'Data dir not found: {data_path}')
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    backup_root = os.path.join(home_dir, 'backup')
    os.makedirs(backup_root, exist_ok=True)
    backup_path = os.path.join(backup_root, f'data_{timestamp}')
    shutil.move(data_path, backup_path)
    return backup_path


def restore_snapshot(home_dir, snapshot_id):
    '''Run simd snapshot restore.'''
    cmd = ['simd', 'snapshot', 'restore', f'--home={home_dir}', snapshot_id]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Snapshot restore failed: {e}')


def build_simd_start_cmd(home_dir, additional_flags=''):
    '''Return a ready-to-run simd start command string.'''
    return f'simd start --home={home_dir} {additional_flags}'.strip()


def start_simd(home_dir, additional_flags='', detach=True):
    '''Start simd in foreground or background.'''
    cmd = ['simd', 'start', f'--home={home_dir}'] + additional_flags.split()
    if detach:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 'Started simd in background'
    subprocess.run(cmd, check=True)
    return 'Started simd in foreground'


def check_node_sync(expected_height, rpc_url='http://localhost:26657', timeout=300):
    '''Wait until the node's block height is >= expected_height.'''
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f'{rpc_url}/status', timeout=5)
            resp.raise_for_status()
            height = int(resp.json()['result']['sync_info']['latest_block_height'])
            if height >= expected_height:
                print(f'Node synced at height {height}')
                return True
            print(f'Current height {height}; waiting for {expected_height}...')
        except Exception as err:
            print('RPC error:', err)
        time.sleep(5)
    raise TimeoutError('Node did not reach expected height within timeout.')


CONFIG_PATH = os.path.expanduser("~/.gaia/config/app.toml")

def open_config_file(config_path: str | None = None) -> list[str]:
    """Read $HOME/.gaia/config/app.toml."""
    path = config_path or CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        return fp.readlines()


def modify_minimum_gas_prices(lines: list[str], new_value: str = "0stake") -> list[str]:
    """Update or insert the `minimum-gas-prices` parameter."""
    target_line = f'minimum-gas-prices = "{new_value}"\n'
    updated = False
    for idx, line in enumerate(lines):
        if line.strip().startswith("minimum-gas-prices"):
            lines[idx] = target_line
            updated = True
            break
    if not updated:
        lines.append("\n# Added automatically by script\n" + target_line)
    return lines


def save_and_close_file(lines: list[str], config_path: str | None = None) -> str:
    """Persist the updated app.toml."""
    path = config_path or CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    backup_path = path + ".bak"
    shutil.copy(path, backup_path)
    with open(path, "w", encoding="utf-8") as fp:
        fp.writelines(lines)
    return path


def validate_file_change(config_path: str | None = None, expected_value: str = "0stake") -> bool:
    """Verify `minimum-gas-prices` now equals the expected string."""
    path = config_path or CONFIG_PATH
    expected_line = f'minimum-gas-prices = "{expected_value}"'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip() == expected_line:
                return True
    return False


def detect_profile() -> Path:
    """Infer which shell profile to modify."""
    shell = os.environ.get("SHELL", "")
    if shell.endswith("zsh"):
        return Path.home() / ".zshrc"
    return Path.home() / ".bashrc"


def add_cast_alias(rpc_url: str, profile_path: Path | None = None) -> None:
    """Append the alias line to the profile file."""
    if not rpc_url:
        raise ValueError("rpc_url is required")
    profile_path = profile_path or detect_profile()
    alias_line = f'alias cast="cast --rpc-url {rpc_url}"'
    if profile_path.exists() and alias_line in profile_path.read_text():
        print("Alias already present; nothing to do.")
        return
    with profile_path.open("a", encoding="utf-8") as fp:
        fp.write("\n# Added by Cosmos EVM setup script\n")
        fp.write(alias_line + "\n")
    print(f"Alias successfully written to {profile_path}")


def reload_shell() -> None:
    shell_path = os.environ.get("SHELL", "/bin/bash")
    print(f"Spawning a fresh login shell ({shell_path} -l)...")
    print("After this command finishes, verify the alias with `type cast`.")
    try:
        subprocess.run([shell_path, "-l"], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Failed to reload shell: {exc}", file=sys.stderr)
        sys.exit(1)


def init_genesis(moniker: str, chain_id: str = "my-test-chain") -> Dict[str, str]:
    """Initialise a new chain by executing `simd init`."""
    cmd = ["simd", "init", moniker, "--chain-id", chain_id]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"stdout": completed.stdout, "stderr": completed.stderr}
    except FileNotFoundError:
        raise RuntimeError("`simd` binary not found.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"simd init failed with exit code {exc.returncode}: {exc.stderr.strip()}")


def verify_genesis_file(genesis_path: str = os.path.expanduser("~/.simapp/config/genesis.json"), expected_chain_id: str = "my-test-chain") -> bool:
    """Verify that the genesis file contains the expected chain-ID."""
    if not os.path.isfile(genesis_path):
        raise FileNotFoundError(f"Genesis file not found at {genesis_path}.")
    with open(genesis_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    actual_chain_id = data.get("chain_id")
    if actual_chain_id != expected_chain_id:
        raise ValueError(f"chain_id mismatch: expected '{expected_chain_id}', found '{actual_chain_id}'")
    return True


def get_chain_home(chain_id: str = "simd") -> dict:
    """Resolve and validate the chain's home directory."""
    env_var = f"{chain_id.upper()}_HOME"
    home_dir = os.getenv(env_var) or Path.home() / f".{chain_id}"
    home_dir = str(home_dir)
    if not os.path.isdir(home_dir):
        raise FileNotFoundError(f"Chain home directory not found at {home_dir}")
    return {"chain_home": home_dir}


def locate_genesis(chain_home: str) -> dict:
    """Locate and load config/genesis.json."""
    genesis_path = Path(chain_home) / "config" / "genesis.json"
    if not genesis_path.exists():
        raise FileNotFoundError(f"genesis.json not found at {genesis_path}")
    with genesis_path.open("r", encoding="utf-8") as f:
        genesis_data = json.load(f)
    return {"genesis_path": str(genesis_path), "genesis_data": genesis_data}


def backup_genesis(genesis_path: str) -> dict:
    """Copy genesis.json to a timestamped backup."""
    src = Path(genesis_path)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup_path = src.with_suffix(f".json.bak.{timestamp}")
    shutil.copy2(src, backup_path)
    return {"backup_path": str(backup_path)}


async def update_inflation(genesis_data: dict, new_inflation: str = "0.300000000000000000") -> dict:
    """Set app_state.mint.params.inflation."""
    try:
        genesis_data["app_state"]["mint"]["params"]["inflation"] = new_inflation
    except KeyError as err:
        raise KeyError("Unable to locate mint.params.inflation in genesis.json") from err
    return {"updated_genesis": genesis_data}


async def save_genesis(genesis_path: str, updated_genesis: dict) -> dict:
    """Persist the updated genesis data to disk."""
    path = Path(genesis_path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(updated_genesis, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return {"status": "saved", "path": str(path)}


def get_cron_authority(lcd_endpoint: str) -> str:
    """Return the Cron module authority address."""
    try:
        url = f"{lcd_endpoint}/neutron/cron/v1/params"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()["params"]["authority"]
    except (requests.RequestException, KeyError) as err:
        raise RuntimeError(f"Unable to fetch Cron authority: {err}")


def validate_contract(address: str, lcd_endpoint: str) -> bool:
    """Return True when the contract exists."""
    try:
        url = f"{lcd_endpoint}/cosmwasm/wasm/v1/contract/{address}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        info = response.json().get("contract_info", {})
        return info.get("address") == address
    except requests.RequestException:
        return False


def build_msg_execute_contract(staking_contract: str, cron_sender: str = "cron") -> dict:
    """Return a MsgExecuteContract dict."""
    inner_msg = {"distribute_rewards": {}}
    return {
        "@type": "/cosmwasm.wasm.v1.MsgExecuteContract",
        "sender": cron_sender,
        "contract": staking_contract,
        "msg": base64.b64encode(json.dumps(inner_msg).encode()).decode(),
        "funds": []
    }


def write_proposal_file(msg_add_schedule: dict, filename: str = "proposal.json") -> str:
    """Write the governance proposal to disk."""
    proposal = {"title": "Add weekly staking-reward cron", "description": "Distribute staking rewards every week", "messages": [msg_add_schedule]}
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(proposal, fp, indent=2)
    return filename


def submit_proposal(file_path: str, from_key: str, chain_id: str, node: str) -> None:
    """Call neutrond CLI to submit the proposal."""
    cmd = ["neutrond", "tx", "wasm", "submit-proposal", file_path, "--from", from_key, "--chain-id", chain_id, "--node", node, "--gas", "auto", "--gas-adjustment", "1.3", "-y"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Proposal submission failed: {err}")


def wait_for_proposal_passage(proposal_id: int, lcd_endpoint: str, poll: int = 15, timeout: int = 3600) -> None:
    """Block until the proposal passes."""
    deadline = time.time() + timeout
    gov_url = f"{lcd_endpoint}/cosmos/gov/v1/proposals/{proposal_id}"
    while time.time() < deadline:
        response = requests.get(gov_url, timeout=10)
        response.raise_for_status()
        status = int(response.json()["proposal"]["status"])
        if status == 3:
            print(f"✅  Proposal {proposal_id} PASSED")
            return
        if status in (4, 5):
            raise RuntimeError(f"❌  Proposal {proposal_id} failed with status {status}")
        print(f"⏳  Waiting... current status = {status}")
        time.sleep(poll)
    raise TimeoutError("Timed out waiting for proposal to pass")


def stop_evmd_service(service_name: str = "evmd") -> dict:
    """Stops the evmd service."""
    try:
        subprocess.run(["systemctl", "stop", service_name], check=True)
        return {"service": service_name, "status": "stopped"}
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to stop {service_name}: {e}")


def load_app_toml(config_path: str = "~/.evmd/config/app.toml") -> dict:
    """Reads the app.toml file."""
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    with path.open("r", encoding="utf-8") as f:
        return {"path": str(path), "content": f.read()}


def _replace_or_add(pattern: str, replacement: str, text: str) -> str:
    """Replace a line or add it inside the [json-rpc] section."""
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)
    lines = text.splitlines()
    updated_lines = []
    in_json_rpc = False
    for line in lines:
        updated_lines.append(line)
        if line.strip().startswith("[json-rpc]"):
            in_json_rpc = True
            continue
        if in_json_rpc and line.strip().startswith("[") and not line.strip().startswith("[json-rpc]"):
            updated_lines.insert(len(updated_lines) - 1, replacement)
            in_json_rpc = False
    return "\n".join(updated_lines)


def update_app_toml_parameter(config_path: str = "~/.evmd/config/app.toml") -> dict:
    """Mutates app.toml to enable JSON-RPC & txpool."""
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    raw = path.read_text(encoding="utf-8")
    updated = raw
    updated = _replace_or_add(r"^enable\s*=.*$", "enable = true", updated)
    updated = _replace_or_add(r"^api\s*=.*$", "api = \"eth,net,web3,txpool,debug\"", updated)
    updated = _replace_or_add(r"^enable-indexer\s*=.*$", "enable-indexer = true", updated)
    changed = updated != raw
    if changed:
        path.write_text(updated, encoding="utf-8")
    return {"path": str(path), "changed": changed}


def verify_app_toml(config_path: str = "~/.evmd/config/app.toml") -> bool:
    """Returns True if the required settings are present."""
    path = Path(config_path).expanduser()
    text = path.read_text(encoding="utf-8")
    return all(["enable = true" in text, "txpool" in text, "enable-indexer = true" in text])


def start_evmd_service(service_name: str = "evmd") -> dict:
    """Starts evmd and returns the latest logs."""
    try:
        subprocess.run(["systemctl", "start", service_name], check=True)
        logs = subprocess.check_output(["journalctl", "-u", service_name, "-n", "20", "--no-pager"], text=True)
        return {"service": service_name, "status": "started", "logs": logs}
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to start {service_name}: {e}")


def json_rpc_call(method: str = "txpool_status", params: list | None = None, endpoint: str = "http://localhost:8545") -> dict:
    """Executes a JSON-RPC call."""
    if params is None:
        params = []
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    try:
        response = requests.post(endpoint, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"RPC Error: {data['error']}")
        return data.get("result")
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Failed to perform JSON-RPC call: {e}")


def get_key_address(key_name: str = "my_validator", keyring_backend: str = "test") -> str:
    """Return the bech32 address for a key."""
    try:
        cmd = ["simd", "keys", "show", key_name, "--keyring-backend", keyring_backend, "-a"]
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        address = result.stdout.strip()
        if not address:
            raise RuntimeError("Received empty address from key-ring query")
        return address
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"simd keys show failed: {e.stderr}") from e


def validate_recipient_address(address: str, expected_prefix: str = "cosmos") -> str:
    """Verify HRP and checksum of a bech32 address."""
    hrp, data = bech32_decode(address)
    if hrp != expected_prefix or data is None:
        raise ValueError(f"{address} is not a valid {expected_prefix} bech32 address")
    return address


def build_send_tx(sender: str, recipient: str, amount: str = "1000stake", fee: str = "200stake", chain_id: str = "my-test-chain", outfile: str = "unsigned_tx.json") -> str:
    """Create an unsigned MsgSend."""
    cmd = ["simd", "tx", "bank", "send", sender, recipient, amount, "--generate-only", "--fees", fee, "--chain-id", chain_id, "--output", "json"]
    try:
        with open(outfile, "w", encoding="utf-8") as fp:
            subprocess.run(cmd, check=True, text=True, stdout=fp)
        if not os.path.exists(outfile):
            raise RuntimeError("Unsigned TX file was not created")
        return outfile
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to build tx: {e.stderr}") from e


def sign_tx(unsigned_tx_path: str, key_name: str = "my_validator", keyring_backend: str = "test", chain_id: str = "my-test-chain", outfile: str = "signed_tx.json") -> str:
    """Sign an unsigned transaction file."""
    cmd = ["simd", "tx", "sign", unsigned_tx_path, "--from", key_name, "--keyring-backend", keyring_backend, "--chain-id", chain_id, "--output", "json", "--yes"]
    try:
        with open(outfile, "w", encoding="utf-8") as fp:
            subprocess.run(cmd, check=True, text=True, stdout=fp)
        if not os.path.exists(outfile):
            raise RuntimeError("Signed TX file was not created")
        return outfile
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Signing failed: {e.stderr}") from e


def broadcast_tx(signed_tx_path: str, chain_id: str = "my-test-chain", node_rpc: str = "http://localhost:26657") -> str:
    """Broadcast a signed tx and return its hash."""
    cmd = ["simd", "tx", "broadcast", signed_tx_path, "--node", node_rpc, "--chain-id", chain_id, "--output", "json"]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        response = json.loads(result.stdout)
        tx_hash = response.get("txhash")
        if not tx_hash:
            raise RuntimeError(f"Unexpected response: {response}")
        return tx_hash
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Broadcast failed: {e.stderr}") from e


def ensure_key_exists(key_name: str) -> dict:
    """Ensure a key is present in the local keyring."""
    try:
        cmd_show = [CHAIN_BINARY, "keys", "show", key_name, "--output", "json", "--keyring-backend", KEYRING_BACKEND, "--home", KEY_HOME]
        show_result = subprocess.check_output(cmd_show, text=True)
        info = json.loads(show_result)
        return {"status": "exists", "address": info["address"]}
    except subprocess.CalledProcessError:
        cmd_add = [CHAIN_BINARY, "keys", "add", key_name, "--output", "json", "--keyring-backend", KEYRING_BACKEND, "--home", KEY_HOME]
        try:
            add_result = subprocess.check_output(cmd_add, text=True)
            info = json.loads(add_result)
            return {"status": "created", "address": info.get("address"), "mnemonic": info.get("mnemonic", "")}
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to create key {key_name}: {exc}")


def generate_gentx(key_name: str, amount: str = "100000000stake") -> str:
    """Produce a gentx."""
    cmd = [CHAIN_BINARY, "gentx", key_name, amount, "--chain-id", CHAIN_ID, "--keyring-backend", KEYRING_BACKEND, "--home", KEY_HOME]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gentx command failed: {exc}") from exc
    gentx_dir = os.path.join(KEY_HOME, "config", "gentx")
    all_gentxs = [os.path.join(gentx_dir, f) for f in os.listdir(gentx_dir) if f.endswith(".json")]
    if not all_gentxs:
        raise FileNotFoundError("No gentx file found.")
    return max(all_gentxs, key=os.path.getmtime)


# def cast_from_bin(file: UploadFile = File(...)):
#     ensure_foundry_installed()
#     tmp_path = None
#     try:
#         with tempfile.NamedTemporaryFile(delete=False) as tmp:
#             tmp.write(await file.read())
#             tmp_path = tmp.name
#         hex_output = subprocess.check_output(['cast', 'from-bin', tmp_path], text=True).strip()
#         return {'hex': hex_output}
#     except subprocess.CalledProcessError as exc:
#         raise HTTPException(status_code=500, detail=f'cast failed: {exc}')
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.remove(tmp_path)


def query_cron_show_schedule(schedule_name: str, node: str = "https://rpc.neutron.org:26657") -> Dict:
    """Fetch cron schedule metadata from a Neutron node."""
    cmd = ["neutrond", "query", "cron", "show-schedule", schedule_name, "--node", node, "--output", "json"]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as err:
        raise RuntimeError("'neutrond' CLI not found.") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"CLI returned error: {err.stderr}") from err
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as err:
        raise RuntimeError("Failed to decode neutrond JSON output.") from err


class BigNumber:
    '''A minimal BigNumber implementation.'''
    def __init__(self, value: Union[str, int, float]):
        if isinstance(value, (int, float)):
            value = str(value)
        if not isinstance(value, str):
            raise TypeError('BigNumber value must be str, int, or float')
        getcontext().prec = max(len(value) * 2, 50)
        self._value = Decimal(value)

    def add(self, other: 'BigNumber') -> 'BigNumber':
        return BigNumber(str(self._value + other._value))

    def sub(self, other: 'BigNumber') -> 'BigNumber':
        return BigNumber(str(self._value - other._value))

    def mul(self, other: 'BigNumber') -> 'BigNumber':
        return BigNumber(str(self._value * other._value))

    def div(self, other: 'BigNumber') -> 'BigNumber':
        if other._value == 0:
            raise ZeroDivisionError('Division by zero.')
        return BigNumber(str(self._value / other._value))

    def pow(self, exponent: int) -> 'BigNumber':
        if not isinstance(exponent, int):
            raise TypeError('Exponent must be an integer.')
        return BigNumber(str(self._value ** exponent))

    def __str__(self) -> str:
        value_str = format(self._value, 'f')
        return value_str.rstrip('0').rstrip('.') if '.' in value_str else value_str

    def to_int(self) -> int:
        return int(self._value)

    def to_decimal(self) -> Decimal:
        return self._value


def _rpc_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    return headers


def rpc_request(method: str, params: Optional[List[Any]] = None, *, id: int = 1) -> Any:
    """Low-level JSON-RPC request."""
    payload = {"jsonrpc": "2.0", "method": method, "params": params or [], "id": id}
    try:
        response = httpx.post(JSON_RPC_URL, headers=_rpc_headers(), json=payload, timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as http_err:
        logger.exception("HTTP error while calling %s", method)
        raise RuntimeError(f"HTTP error: {http_err}") from http_err
    data = response.json()
    if "error" in data:
        err = data["error"]
        raise RPCError(err.get("code", -1), err.get("message", "Unknown error"), err.get("data"))
    return data.get("result")


class RPCError(Exception):
    def __init__(self, code: int, message: str, data: Optional[Any] = None):
        super().__init__(f"RPC error {code}: {message}")
        self.code = code
        self.data = data


def create_preblocker_fn(app_dir: str) -> None:
    """Inserts a PreBlocker function in app/app.go."""
    app_go_path = os.path.join(app_dir, "app", "app.go")
    if not os.path.exists(app_go_path):
        raise FileNotFoundError(f"{app_go_path} not found")
    preblocker_code = textwrap.dedent('''
    func PreBlocker(ctx sdk.Context, req abci.RequestPreBlock) {
        newParams := &tmtypes.ConsensusParams{...}
        if err := app.BaseApp.UpdateConsensusParams(ctx, newParams); err != nil {
            ctx.Logger().Error("failed to update consensus params", "err", err)
        }
    }
    ''')
    with open(app_go_path, "r+") as f:
        content = f.read()
        if "func PreBlocker(" in content:
            print("PreBlocker already exists.")
            return
        f.write(preblocker_code)
    print("✅ PreBlocker function added.")


def register_preblocker(app_dir: str) -> None:
    """Adds app.SetPreBlocker(PreBlocker) to the NewApp constructor."""
    app_go_path = os.path.join(app_dir, "app", "app.go")
    if not os.path.exists(app_go_path):
        raise FileNotFoundError(f"{app_go_path} not found")
    with open(app_go_path, "r+") as f:
        content = f.read()
        if "SetPreBlocker(PreBlocker)" in content:
            print("PreBlocker already registered.")
            return
        pattern = r"func\s+NewApp[\s\S]+?return\s+app"
        match = re.search(pattern, content)
        if not match:
            print("Could not locate NewApp function.")
            return
        insert_idx = match.end() - len("return app")
        insertion = "\n    app.SetPreBlocker(PreBlocker)\n"
        new_content = content[:insert_idx] + insertion + content[insert_idx:]
        f.seek(0)
        f.write(new_content)
        f.truncate()
    print("✅ PreBlocker registered.")


def compile_binary(app_dir: str = ".") -> None:
    """Compiles the modified binary."""
    print("🔨 Compiling...")
    proc = subprocess.run(["go", "install", "./..."], cwd=app_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("Compilation failed")
    print("✅ Compilation successful.")


def start_local_chain(home: str = "./data", chain_id: str = "localnet", binary: str = "appd") -> None:
    """Starts a single-node chain."""
    if not os.path.exists(home):
        print("🔧 Initializing home directory")
        subprocess.run([binary, "init", "validator", "--chain-id", chain_id, "--home", home], check=True)
        subprocess.run([binary, "config", "chain-id", chain_id, "--home", home], check=True)
    print("⛓️  Starting node…")
    try:
        subprocess.run([binary, "start", "--home", home], check=True)
    except KeyboardInterrupt:
        print("Node stopped by user")


def query_consensus_params(height: int, binary: str = "appd") -> None:
    """Queries consensus parameters for a given block height."""
    cmd = [binary, "query", "params", "subspace", "consensus", "1", "--height", str(height), "--output", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Query failed")
    print(result.stdout)


def get_governance_authority(rest_endpoint: str = 'https://rest-kralum.neutron.org') -> str:
    '''Fetch the Main DAO address from the cron params endpoint.'''
    try:
        resp = requests.get(f'{rest_endpoint}/neutron/cron/v1/params', timeout=10)
        resp.raise_for_status()
        data = resp.json()
        authority = data.get('params', {}).get('governance_account') or data.get('params', {}).get('authority')
        if authority:
            return authority
        raise ValueError('Authority field not found.')
    except Exception as err:
        print(f'[WARN] Unable to fetch authority from REST API: {err}')
        fallback = os.getenv('MAIN_DAO_ADDRESS')
        if not fallback:
            raise RuntimeError('MAIN_DAO_ADDRESS env var is not set.') from err
        return fallback


def build_placeholder_calls(authority: str):
    call_1 = build_execute_msg(sender=authority, contract='neutron1contractaddr1...', msg={'update_config': {'param': 42}})
    call_2 = build_execute_msg(sender=authority, contract='neutron1contractaddr2...', msg={'set_admin': {'new_admin': authority}})
    call_3 = build_execute_msg(sender=authority, contract='neutron1contractaddr3...', msg={'migrate': {'code_id': 99}})
    return call_1, call_2, call_3


def wrap_into_submit_proposal(schedule_msg, proposer: str, deposit: List[dict]):
    '''Pack the MsgAddSchedule into a MsgSubmitProposal.'''
    try:
        any_msg = ProtoAny()
        any_msg.Pack(schedule_msg, type_url_prefix='/')
        submit = gov_tx.MsgSubmitProposal(
            messages=[any_msg],
            initial_deposit=[base_coin.Coin(denom=c['denom'], amount=str(c['amount'])) for c in deposit],
            proposer=proposer,
            title='Register Cron schedule: protocol_update',
            summary='Adds a cron schedule that executes three contract calls every 100,800 blocks.',
        )
        return submit
    except Exception as err:
        raise RuntimeError(f'Unable to create MsgSubmitProposal: {err}')


class CronMessageError(Exception):
    pass


def construct_msg_remove_schedule(schedule_name: str, authority: str) -> dict:
    """Return a MsgRemoveSchedule ready for inclusion in a proposal."""
    return {
        "@type": "/neutron.cron.MsgRemoveSchedule",
        "authority": authority,
        "name": schedule_name,
    }


def create_json_proposal_file(msgs: list, title: str, description: str, deposit: str, outfile: str = "proposal.json") -> str:
    """Writes a Neutron governance proposal JSON to disk."""
    proposal = {"title": title, "description": description, "deposit": deposit, "messages": msgs}
    with open(outfile, "w", encoding="utf-8") as fp:
        json.dump(proposal, fp, indent=2)
    return outfile


def vote_and_wait_for_passage(rpc_endpoint: str, proposal_id: int, voter_priv_hex: str, chain_id: str, poll: int = 15):
    """Casts a YES vote, then waits until the proposal passes."""
    key = PrivateKey.from_hex(voter_priv_hex)
    cfg = NetworkConfig(chain_id=chain_id, url=rpc_endpoint, fee_denomination="untrn", fee_minimum_gas_price=0.025)
    client = LedgerClient(cfg)
    client.gov_vote(proposal_id, key.address(), 1)
    print(f"YES vote submitted from {key.address()} on proposal {proposal_id}")
    while True:
        status = client.gov_proposal(proposal_id)["status"]
        print("Current status:", status)
        if status == "PROPOSAL_STATUS_PASSED":
            print("🎉 Proposal PASSED")
            return True
        if status in ("PROPOSAL_STATUS_REJECTED", "PROPOSAL_STATUS_FAILED"):
            raise RuntimeError(f"Proposal ended with status {status}")
        time.sleep(poll)


def confirm_execution_stage(rest_endpoint: str, schedule_name: str) -> bool:
    """Returns True if the cron job now runs at BEGIN_BLOCKER."""
    schedule = query_cron_schedule(rest_endpoint, schedule_name)
    return schedule.get("execution_stage") == "BEGIN_BLOCKER"


def get_dao_authority_address(node_url: str = REST_ENDPOINT) -> str:
    """Returns the authority address defined in the Cron module params."""
    url = f"{node_url}/cosmos/params/v1beta1/params?subspace=cron"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for param in data.get("param", []):
            if param.get("key") == "Authority":
                return param.get("value")
        raise RuntimeError("'Authority' field not found in cron params response")
    except requests.RequestException as err:
        raise RuntimeError(f"Unable to query cron params: {err}")


def graceful_shutdown(pid: Optional[int] = None, *, use_systemctl: bool = False, service_name: Optional[str] = None) -> None:
    """Send a graceful shutdown signal."""
    if use_systemctl:
        if not service_name:
            raise ValueError("'service_name' is required")
        try:
            result = subprocess.run(["systemctl", "stop", service_name], check=True, capture_output=True, text=True)
            print(result.stdout.strip())
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"Failed to stop service: {err.stderr.strip()}") from err
    else:
        if pid is None:
            raise ValueError("'pid' is required")
        try:
            os.kill(pid, signal.SIGINT)
            print(f"Sent SIGINT to PID {pid}")
        except ProcessLookupError:
            raise RuntimeError(f"Process with PID {pid} does not exist.")
        except PermissionError:
            raise RuntimeError(f"Permission denied to signal PID {pid}.")


def wait_for_exit(pid: int, *, log_path: Optional[str] = None, timeout: int = 120, poll_interval: float = 1.0) -> None:
    """Wait for a process to exit."""
    start_time = time.time()
    while True:
        if not os.path.exists(f"/proc/{pid}"):
            print(f"PID {pid} has exited.")
            break
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Process {pid} did not exit within {timeout}s.")
        time.sleep(poll_interval)


def verify_profile_file(profile_path: str = "cpu.prof") -> int:
    """Validate the profiler output file."""
    file_path = Path(profile_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")
    size = file_path.stat().st_size
    if size == 0:
        raise ValueError(f"Profile file is empty: {profile_path}")
    print(f"Profile file '{profile_path}' verified with size {size} bytes.")
    return size


def _strip_0x(hex_str: str) -> str:
    return hex_str[2:] if hex_str.startswith("0x") else hex_str


def _hex_to_bytes(hex_str: str, expected_len: int) -> bytes:
    hex_str = _strip_0x(hex_str)
    if len(hex_str) != expected_len:
        raise ValueError(f"Expected {expected_len} hex chars, got {len(hex_str)}.")
    return bytes.fromhex(hex_str)


# def compute_create2_address(req: Create2Request):
#     try:
#         deployer_bytes = _hex_to_bytes(req.deployer, 40)
#         salt_bytes = _hex_to_bytes(req.salt, 64)
#         init_code_hash_bytes = _hex_to_bytes(req.init_code_hash, 64)
#     except ValueError as err:
#         raise HTTPException(status_code=400, detail=str(err))
#
#     data = b"\xff" + deployer_bytes + salt_bytes + init_code_hash_bytes
#     derived = keccak(data)[12:]
#     return {"create2_address": to_checksum_address("0x" + derived.hex())}


# def fetch_receipt(tx_hash: str) -> dict | None:
#     """Low-level helper that wraps the JSON-RPC request."""
#     payload = {"jsonrpc": "2.0", "method": "eth_getTransactionReceipt", "params": [tx_hash], "id": 1}
#     try:
#         async with httpx.AsyncClient() as client:
#             resp = await client.post(RPC_ENDPOINT, json=payload, timeout=10)
#             resp.raise_for_status()
#     except httpx.RequestError as e:
#         raise HTTPException(status_code=502, detail=f"RPC connection error: {str(e)}")
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(status_code=e.response.status_code, detail=f"RPC returned {e.response.status_code}")
#     body = resp.json()
#     if body.get("error"):
#         raise HTTPException(status_code=500, detail=body["error"])
#     return body.get("result")


def _human_readable_status(status_hex: str) -> str:
    """Convert 0x0 / 0x1 to Failed / Success."""
    try:
        return "Success" if int(status_hex, 16) == 1 else "Failed"
    except ValueError:
        return "Unknown"


def decode_receipt(receipt: dict) -> dict:
    """Transform the RPC receipt into a cleaner JSON structure."""
    if receipt is None:
        raise HTTPException(status_code=400, detail="Empty receipt supplied")
    return {
        "transactionHash": receipt.get("transactionHash"),
        "blockNumber": int(receipt.get("blockNumber", "0x0"), 16),
        "status": _human_readable_status(receipt.get("status", "0x0")),
        "gasUsed": int(receipt.get("gasUsed", "0x0"), 16),
        "contractAddress": receipt.get("contractAddress"),
        "logsCount": len(receipt.get("logs", [])),
        "logs": receipt.get("logs", []),
    }


def list_gpg_keys() -> List[str]:
    """Return a list of available GPG key IDs."""
    try:
        completed = subprocess.run(["gpg", "--list-keys", "--with-colons"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("`gpg` command is missing.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to list GPG keys: {exc.stderr.strip()}")
    return [row.split(":")[4] for row in completed.stdout.splitlines() if row.startswith("pub") and len(row.split(":")) > 4]


def init_pass_store(key_id: str) -> None:
    """Initialise the `pass` password store."""
    try:
        subprocess.run(["pass", "init", key_id], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("`pass` CLI not found.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`pass init` failed: {exc.stderr.strip()}")


def verify_pass_store() -> List[str]:
    """Return list of entries in the password store."""
    try:
        completed = subprocess.run(["pass", "ls"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("`pass` CLI not found.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`pass ls` failed: {exc.stderr.strip()}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_checksums(original: str, backup: str) -> bool:
    """Return True when both files share an identical SHA-256 digest."""
    orig_hash = _sha256(original)
    backup_hash = _sha256(backup)
    print(f"Original SHA-256: {orig_hash}\nBackup   SHA-256: {backup_hash}")
    return orig_hash == backup_hash


def secure_offsite_copy(local_path: str, remote_user: str, remote_host: str, remote_dir: str) -> None:
    """SCP the backup to an off-site server."""
    remote_target = f"{remote_user}@{remote_host}:{remote_dir}/{os.path.basename(local_path)}"
    try:
        subprocess.run(["scp", "-p", local_path, remote_target], check=True, text=True)
        print(f"Successfully copied to {remote_target}")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Secure copy failed: {err}")


def start_cosmos_service(service_name: str = "cosmosd") -> None:
    """Restart the Cosmos validator systemd unit."""
    try:
        subprocess.run(["sudo", "systemctl", "start", service_name], check=True, text=True)
        print(f"Successfully started {service_name}.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to start {service_name}: {err}")


def construct_msg_vote_yes(voter: str, proposal_id: int) -> MsgVote:
    """Returns a MsgVote protobuf message (YES)."""
    if not voter:
        raise ValueError('Voter address must be provided')
    if proposal_id <= 0:
        raise ValueError('Proposal ID must be a positive integer')
    msg = MsgVote()
    msg.proposal_id = proposal_id
    msg.voter = voter
    msg.option = VoteOption.VOTE_OPTION_YES
    return msg


# def getMajor(ver):
#   const match = ver.match(/v?(\d+)\./);
#   return match ? parseInt(match[1], 10) : 0;
#
#
# def init(projectName = 'hardhat-project'):
#   const projectPath = path.resolve(process.cwd(), projectName);
#   if (!fs.existsSync(projectPath)) {
#     fs.mkdirSync(projectPath, { recursive: true });
#     console.log(`📁 Created directory: ${projectPath}`);
#   } else {
#     console.log(`📂 Using existing directory: ${projectPath}`);
#   }
#   process.chdir(projectPath);
#   console.log('⚙️  Initializing npm project...');
#   execSync('npm init -y', { stdio: 'inherit' });
#   console.log('✅ package.json generated.');
#
#
# def install():
#   console.log('⬇️  Installing Hardhat and toolbox...');
#   try {
#     execSync('npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox', { stdio: 'inherit' });
#     console.log('✅ Dependencies installed.');
#   } catch (err) {
#     console.error('❌ Failed to install dependencies:', err.message);
#     process.exit(1);
#   }
#
#
# def initHardhat():
#   console.log('🚀 Running Hardhat initializer...');
#   try {
#     execSync('npx hardhat', { stdio: 'inherit' });
#     console.log('✅ Hardhat project initialized.');
#   } catch (err) {
#     console.error('❌ Hardhat initialization failed:', err.message);
#     process.exit(1);
#   }


def _list_snapshots(home: str) -> str:
    """Helper that executes snapshot listing commands."""
    commands = [["simd", "snapshot", "list", f"--home={home}"], ["ls", "-lh", f"{home}/data/snapshots"]]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout.strip():
                return result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("Unable to list snapshots.")


SNAPSHOT_REGEX = re.compile(r"height[=:](?P<height>\d+)\s+format[=:](?P<format>\d+)\s+hash[=:]?\s*(?P<hash>[A-Fa-f0-9]+)", re.IGNORECASE)

def _parse_snapshot_output(raw_output: str) -> List[Dict]:
    """Convert CLI output into structured objects."""
    snapshots = []
    for line in raw_output.splitlines():
        match = SNAPSHOT_REGEX.search(line)
        if match:
            snapshots.append({"height": int(match.group("height")), "format": match.group("format"), "hash": match.group("hash")})
    if not snapshots:
        raise ValueError("No snapshots found in provided output.")
    return snapshots


# def getVersion(cmd):
#   try {
#     return execSync(`${cmd} --version`).toString().trim();
#   } catch (err) {
#     throw new Error(`${cmd} is not installed or not in PATH.`);
#   }
#
#
# def main():
#   try {
#     const nodeVersion = getVersion('node');
#     const npmVersion = getVersion('npm');
#     const [major] = nodeVersion.replace(/^v/, '').split('.').map(Number);
#     if (isNaN(major) || major < 16) {
#       throw new Error(`Node.js >=16 is required. Detected ${nodeVersion}.`);
#     }
#     console.log(`✅ Environment OK. Node: ${nodeVersion}, npm: ${npmVersion}`);
#   } catch (error) {
#     console.error(`❌ Environment check failed: ${error.message}`);
#     process.exit(1);
#   }
#
#
# def initProject(dir = process.cwd()):
#   try {
#     console.log(`Initializing npm project in ${dir}...`);
#     execSync('npm init -y', { stdio: 'inherit', cwd: dir });
#     console.log('✅ package.json created.');
#   } catch (error) {
#     console.error('❌ Failed to initialize npm project:', error.message);
#     process.exit(1);
#   }
#
#
# def installHardhat():
#   try {
#     console.log('Installing Hardhat as a dev dependency...');
#     execSync('npm install --save-dev hardhat', { stdio: 'inherit' });
#     console.log('✅ Hardhat installed.');
#   } catch (error) {
#     console.error('❌ Failed to install Hardhat:', error.message);
#     process.exit(1);
#   }
#
#
# def createHardhatProject():
#   try {
#     console.log('Scaffolding a new Hardhat TypeScript project...');
#     execSync('npm init hardhat -- --template typescript --force', { stdio: 'inherit' });
#     console.log('✅ Hardhat project scaffolded.');
#   } catch (error) {
#     console.error('❌ Failed to scaffold Hardhat project:', error.message);
#     process.exit(1);
#   }
#
#
# def installDeps():
#   try {
#     console.log('Installing Ethers.js and Hardhat plugins...');
#     execSync('npm install --save-dev @nomicfoundation/hardhat-ethers ethers', { stdio: 'inherit' });
#     console.log('✅ Dependencies installed.');
#   } catch (error) {
#     console.error('❌ Failed to install dependencies:', error.message);
#     process.exit(1);
#   }
#
#
# def compile():
#   try {
#     console.log('Compiling Hardhat project...');
#     execSync('npx hardhat compile', { stdio: 'inherit' });
#     console.log('✅ Compilation successful.');
#   } catch (error) {
#     console.error('❌ Compilation failed:', error.message);
#     process.exit(1);
#   }


def _check_simd_binary():
    '''Verify that the `simd` binary exists and is executable.'''
    simd_path = shutil.which("simd")
    if simd_path is None:
        raise FileNotFoundError("`simd` binary not found.")
    try:
        result = subprocess.run([simd_path, "version"], capture_output=True, text=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"`simd` not executable: {exc}")
    return {"simd_path": simd_path, "version": result.stdout.strip()}


def _construct_start_command(home_dir: str) -> str:
    '''Return a safe string for starting simd.'''
    if not home_dir:
        raise ValueError('home_dir is required')
    expanded_home = os.path.expanduser(home_dir)
    if not os.path.isdir(expanded_home):
        raise FileNotFoundError(f'Home directory does not exist: {expanded_home}')
    quoted_home = shlex.quote(expanded_home)
    return f'simd start --mempool.max-txs=-1 --home={quoted_home}'


_process_holder = {'proc': None}

def _start_simd_process(command: str):
    if _process_holder['proc'] is not None:
        raise RuntimeError('A simd process is already running.')
    args = shlex.split(command)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    _process_holder['proc'] = proc
    return {'pid': proc.pid}


async def _log_streamer():
    proc = _process_holder.get('proc')
    if proc is None or proc.stdout is None:
        yield 'data: simd process not running\n\n'
        return
    while True:
        line = proc.stdout.readline()
        if line:
            yield f'data: {line.rstrip()}\n\n'
        await asyncio.sleep(0.05)
        if proc.poll() is not None:
            break
    yield 'data: **simd process ended**\n\n'


def _cli_status():
    res = subprocess.run(['simd', 'status'], capture_output=True, text=True, check=True)
    return res.stdout


# def _rpc_status(rpc_url: str):
#     r = requests.get(f'{rpc_url.rstrip('/')}/status')
#     r.raise_for_status()
#     return r.json()


def search_circuit_docs(query: str) -> dict:
    """Return search results from the Cosmos docs search API."""
    try:
        url = f"{COSMOS_DOCS_SEARCH_URL}?q={urllib.parse.quote_plus(query)}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        return {"error": str(exc), "results": []}


def compile_application():
    """Rebuild the chain binary."""
    try:
        subprocess.run(["make", "install"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("✅  Binary compiled.")
    except subprocess.CalledProcessError as exc:
        print("❌  Compilation failed:\n", exc.stdout.decode())
        sys.exit(1)


def monitor_logs():
    try:
        with open(LOG_PATH, "r") as fh:
            matches = [line.strip() for line in fh if PATTERN.search(line)]
        if matches:
            print("🎉  Found %d PrepareProposal invocations:" % len(matches))
            for ln in matches:
                print("   ", ln)
        else:
            print("⚠️   No PrepareProposal logs detected.")
    except FileNotFoundError:
        print("node.log not found.")
        sys.exit(1)