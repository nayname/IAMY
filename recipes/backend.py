import time
import requests
from typing import Dict
import os
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.client import NetworkConfig
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction

import json
import subprocess
import time
import base64
from datetime import datetime
from typing import Dict, Any, List

import requests
from cosmpy.aerial.wallet import LocalWallet
# Note: In a real project, these proto imports would point to your generated files.
# from neutron_proto.cron import MsgAddSchedule
# from cosmos_proto.cosmos.gov.v1 import MsgSubmitProposal, TextProposal
# from google.protobuf.any_pb2 import Any as Any_pb

from typing import List, Dict, Any, Optional
import asyncio

cfg = NetworkConfig(
    chain_id="neutron-1",
    url="grpc+https://grpc-kralum.neutron-1.neutron.org",
    fee_minimum_gas_price=0.01,
    fee_denomination="untrn",
    staking_denomination="untrn",
)
client = LedgerClient(cfg)

def _ping(base_url: str, path: str, timeout_s: float = 2.5) -> float:
    """Return latency in ms if endpoint is reachable, else inf."""
    url = base_url.rstrip("/") + path
    t0 = time.time()
    try:
        # For REST health we can GET the node_info. Any HTTP status < 600 counts as reachable.
        r = requests.get(url, timeout=timeout_s)
        if r.status_code < 600:
            return (time.time() - t0) * 1000.0
    except Exception:
        pass
    return float("inf")


def select_data_provider(prefer_graphql: bool = True) -> Dict[str, str]:
    """Choose the fastest available provider and return a descriptor dict."""
    # Pick network: "neutron-1" (mainnet) or "pion-1" (testnet)
    NETWORK = "neutron-1"
    print(NETWORK)
    PROVIDERS: List[Dict[str, str]] = [
        # ---- MAINNET ----
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
        # Cosmos Directory proxy (routes to live nodes)
        {
            "name": "cosmos-directory-rest",
            "base_url": "https://rest.cosmos.directory/neutron",
            "api_type": "rest",
            "health": "/cosmos/base/tendermint/v1beta1/node_info",
            "network": "neutron-1",
        },

        # ---- TESTNET (pion-1) ----
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

    # (Optional) If you later add a *real* GraphQL indexer (e.g., SubQuery),
    # you can prioritize it here. For now, Celatone GraphQL is not a public endpoint.
    # See: Neutron docs recommend REST/RPC/GRPC endpoints. :contentReference[oaicite:1]{index=1}

    # Measure latency once per provider using its proper health path
    scored = []
    for p in candidates:
        latency = _ping(p["base_url"], p["health"])
        scored.append((latency, p))

    # Pick the best reachable one
    best_latency, best = min(scored, key=lambda t: t[0])
    if best_latency == float("inf"):
        raise RuntimeError("No data provider is reachable at the moment.")
    return best


# step:2 file: Query transaction history for my address
from typing import Tuple, Dict, Any, Optional, Union


def build_history_query(
    provider: Dict[str, str],
    address: str,
    limit: int = 50,
    cursor: Optional[str] = None,
    offset: int = 0,
) -> Tuple[str, Union[Dict[str, Any], None]]:
    """Return (query_or_endpoint, variables_or_params) ready for Step 3."""
    if provider["api_type"] == "graphql":
        # Celatone GraphQL query string with optional cursor for pagination.
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

    # ---------- REST / LCD ----------
    endpoint = f"{provider['base_url']}/cosmos/tx/v1beta1/txs"
    params: Dict[str, Any] = {
        "events": f"message.sender='{address}'",
        "order_by": "ORDER_BY_DESC",
        "pagination.limit": str(limit),
        "pagination.offset": str(offset),
    }
    return endpoint, params


# step:3 file: Query transaction history for my address
import requests
from typing import List, Tuple, Dict, Any, Optional, Union


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
            # Cursor-based pagination (Celatone may not expose pageInfo directly − adjust if needed)
            next_cursor = variables_or_params.get("cursor") if variables_or_params else None
            return results, next_cursor
        # ---------------- REST / LCD ----------------
        resp = requests.get(query_or_url, params=variables_or_params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("txs", []) or data.get("tx_responses", [])
        next_key = data.get("pagination", {}).get("next_key")
        return results, next_key
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to query {provider['name']}: {exc}") from exc


# step:4 file: Query transaction history for my address
from typing import List, Dict, Any


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
    else:  # REST / LCD
        for tx in raw_results:
            # Transaction hash and height
            hash_ = tx.get("txhash") or tx.get("hash")
            height = int(tx.get("height", 0))
            timestamp = tx.get("timestamp")

            # First message type as action indicator
            first_msg = (
                (tx.get("tx", {}) or {}).get("body", {}).get("messages", [])
            )
            action = first_msg[0].get("@type", "") if first_msg else ""

            # Fee formatting
            fee_info = (tx.get("tx", {}) or {}).get("auth_info", {}).get("fee", {})
            fee_amounts = fee_info.get("amount", [])
            fee_str = (
                f"{fee_amounts[0]['amount']}{fee_amounts[0]['denom']}" if fee_amounts else "0"
            )

            success = tx.get("code", 0) == 0  # code == 0 indicates success

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


# step:1 file: Compile and deploy the Neutron example contract to the local CosmoPark testnet
import subprocess, os

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


# step:2 file: Compile and deploy the Neutron example contract to the local CosmoPark testnet
import subprocess, json, requests

def get_local_chain_account(key_name: str = 'cosmopark', faucet_url: str | None = 'http://localhost:4500/credit') -> dict:
    """Load or create a key and optionally request faucet funds."""
    try:
        key_info_raw = subprocess.check_output([
            'neutrond', 'keys', 'show', key_name,
            '--output', 'json', '--keyring-backend', 'test'
        ])
    except subprocess.CalledProcessError:
        # Key does not exist – create it
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


def query_contract_state(client: LedgerClient, contract_address: str, query_msg: dict):
    """Query the contract’s state using a custom query message."""
    try:
        return client.wasm_query(contract_address, query_msg)
    except Exception as err:
        raise RuntimeError(f'Contract query failed: {err}') from err


# step:3 file: Query a contract’s NTRN balance
import subprocess


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



# step:4 file: Query a contract’s NTRN balance
import json
from decimal import Decimal
from typing import Dict


def parse_balance_response(raw_json: str, denom: str = 'untrn') -> Dict[str, str]:
    """Extracts the balance for the specified denom and formats it for display."""
    try:
        data = json.loads(raw_json)
        balances = data.get('balances', [])
        micro_amount = 0
        for coin in balances:
            if coin.get('denom') == denom:
                micro_amount = int(coin.get('amount', '0'))
                break
        human_amount = Decimal(micro_amount) / Decimal(1_000_000)  # 1e6 micro = 1 NTRN
        return {
            'denom': denom,
            'micro_amount': str(micro_amount),
            'amount': f'{human_amount.normalize()} NTRN'
        }
    except (json.JSONDecodeError, ValueError) as err:
        raise ValueError('Invalid JSON supplied to parser: ' + str(err))



# step:1 file: Launch a local Neutron CosmoPark testnet
import shutil
import subprocess
import sys


def ensure_cosmopark_installed() -> None:
    """Ensure that CosmoPark CLI and its Docker images are available."""
    # 1. Check CosmoPark binary
    if shutil.which("cosmopark") is None:
        print("CosmoPark CLI not found. Attempting installation via pip…")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cosmopark-cli"])
        except subprocess.CalledProcessError as err:
            raise RuntimeError("Automatic installation of CosmoPark CLI failed.") from err
    else:
        print("CosmoPark CLI detected ✅")

    # 2. Verify Docker is installed – required by CosmoPark
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required but not installed or not in PATH.")

    # 3. Pull (or update) all CosmoPark Docker images
    try:
        subprocess.check_call(["cosmopark", "pull", "--all"])
        print("CosmoPark Docker images pulled ✅")
    except subprocess.CalledProcessError as err:
        raise RuntimeError("Failed to pull CosmoPark Docker images.") from err


if __name__ == "__main__":
    try:
        ensure_cosmopark_installed()
        print("CosmoPark environment is ready 🟢")
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)


# step:2 file: Launch a local Neutron CosmoPark testnet
import subprocess
from pathlib import Path
import sys


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


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./localnet"
    try:
        run_cosmopark_init(path)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)


# step:3 file: Launch a local Neutron CosmoPark testnet
import subprocess
import sys


def run_cosmopark_start(workspace_path: str = "./localnet") -> None:
    """Run `cosmopark start` inside the workspace to spin up the chain."""
    cmd = ["cosmopark", "start"]
    try:
        subprocess.check_call(cmd, cwd=workspace_path)
    except subprocess.CalledProcessError as err:
        raise RuntimeError("`cosmopark start` failed.") from err


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./localnet"
    try:
        run_cosmopark_start(path)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)


# step:4 file: Launch a local Neutron CosmoPark testnet
import time
import requests
import sys


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
            # Ignore and retry until timeout
            pass

        if time.time() - start > timeout:
            raise RuntimeError(f"Local chain did not start within {timeout} seconds.")

        print("⏳ Waiting for local chain…")
        time.sleep(3)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:26657/status"
    try:
        verify_local_chain_running(url)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)


def query_bank_balance(address: str, denom: str = "untrn") -> int:
    """Return current balance for `address` in the given `denom`."""
    try:
        balance = client.query_bank_balance(address, denom=denom)
        return int(balance.amount)
    except Exception as exc:
        raise RuntimeError(f"Failed to query balance: {exc}") from exc


# step:4 file: Query the connected wallet’s NTRN balance
def format_amount(raw_balance: int) -> str:
    """Convert micro-denom (`untrn`) to a formatted NTRN string."""
    try:
        micro = int(raw_balance)
    except (TypeError, ValueError):
        raise ValueError("raw_balance must be an integer-compatible value")

    ntrn_value = micro / 1_000_000  # 1 NTRN = 1,000,000 untrn
    return f"{ntrn_value:,.6f} NTRN"


# step:1 file: Reset the global counter in an example contract
import os
from cosmpy.aerial.wallet import LocalWallet


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


# step:2 file: Reset the global counter in an example contract
import os
import re

_BECH32_RE = re.compile(r"^neutron1[02-9ac-hj-np-z]{38}$")  # very shallow check


def get_contract_address() -> str:
    """Return the contract address defined by CONTRACT_ADDRESS env-var."""
    contract_addr = os.getenv("CONTRACT_ADDRESS") or ""
    if not _BECH32_RE.match(contract_addr):
        raise ValueError("CONTRACT_ADDRESS env-var is missing or not a valid Neutron bech32 address.")
    return contract_addr


def construct_tx_execute_contract(contract_addr: str, wallet, gas: int = 200000) -> Transaction:
    """Create an unsigned Transaction carrying the reset execute message."""
    execute_msg = {"reset": {}}

    # Build protobuf MsgExecuteContract using the helper (encodes & sets funds = [])
    msg = create_msg_execute_contract(
        sender=str(wallet.address()),
        contract=contract_addr,
        msg=json.dumps(execute_msg).encode(),
        funds=[],
    )

    tx = Transaction()
    tx.add_message(msg)
    tx.with_chain_id(NETWORK_CFG.chain_id)
    tx.with_sender(wallet.address())
    tx.with_gas(gas)
    # Fee is automatically derived from gas*gas_price if not specified explicitly
    return tx


# step:1 file: Instantiate the example contract on Pion-1
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.crypto.keypairs import PrivateKey
import os

# ---------------------------
# Step 1  •  Chain Context
# ---------------------------

def get_neutron_client() -> LedgerClient:
    """Initialises a LedgerClient pointed at Pion-1.

    Raises:
        EnvironmentError: If an RPC endpoint is missing.
    """
    rpc_url = os.getenv("PION_RPC", "https://rpc.pion-1.ntrn.tech:443")

    if not rpc_url:
        raise EnvironmentError("RPC endpoint for Pion-1 is not set.")

    cfg = NetworkConfig(
        chain_id="pion-1",
        url=rpc_url,
        fee_minimum_gas_price=0.025,  # 0.025 NTRN / gas
        fee_denomination="untrn",
        staking_denomination="untrn",
        bech32_hrp="neutron"
    )

    return LedgerClient(cfg)

# Optional: load a signing key once so future steps can re-use it
# NOTE: store your mnemonic securely – this is *just* for local testing!
_SIGNING_KEY: PrivateKey | None = None

def load_signing_key() -> PrivateKey:
    """Loads (or creates) a PrivateKey from a MNEMONIC env-var."""
    global _SIGNING_KEY
    if _SIGNING_KEY is None:
        mnemonic = os.getenv("NEUTRON_MNEMONIC")
        if not mnemonic:
            raise EnvironmentError("Please export NEUTRON_MNEMONIC before running.")
        _SIGNING_KEY = PrivateKey.from_mnemonic(mnemonic)
    return _SIGNING_KEY


# step:2 file: Instantiate the example contract on Pion-1
import json
from typing import Optional

# ---------------------------
# Step 2  •  Resolve code_id
# ---------------------------

def get_code_id(client: LedgerClient, uploader: str, explicit_code_id: Optional[int] = None) -> int:
    """Determine the code_id to instantiate.

    Args:
        client:   The LedgerClient from Step 1.
        uploader: Address that stored the code (usually our wallet).
        explicit_code_id: Optional override (e.g. via CLI flag).

    Returns:
        int: The wasm `code_id`.

    Raises:
        ValueError: If we cannot discover a code_id.
    """
    # Highest priority: explicit argument / env-var
    if explicit_code_id is None:
        env_code_id = os.getenv("CODE_ID")
        explicit_code_id = int(env_code_id) if env_code_id else None

    if explicit_code_id is not None:
        return explicit_code_id

    # Fallback: query the chain for all codes uploaded by `uploader`
    response = client.query("/cosmwasm/wasm/v1/code")  # REST path for all codes
    codes = json.loads(response)["code_infos"]

    # Filter codes by creator and pick the latest
    user_codes = [int(c["code_id"]) for c in codes if c.get("creator") == uploader]
    if not user_codes:
        raise ValueError("No code_id found for uploader – pass CODE_ID env-var or argument.")
    return max(user_codes)


# step:3 file: Set the user's own address as the admin of a smart contract
from typing import Tuple
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgUpdateAdmin
from google.protobuf.any_pb2 import Any

# ---- CONFIGURE YOUR NETWORK ENDPOINT ----
RPC_ENDPOINT = 'https://rpc-kralum.neutron.org:443'  # Public RPC or your own node
CHAIN_ID = 'neutron-1'
FEE_DENOM = 'untrn'
DEFAULT_GAS_LIMIT = 200_000  # adjust as needed


def construct_update_admin_tx(
    sender_address: str,
    contract_address: str,
    new_admin_address: str,
) -> Tuple[Transaction, LedgerClient]:
    """Create an unsigned Transaction containing a MsgUpdateAdmin message.

    Args:
        sender_address: Current admin / governance address signing the tx.
        contract_address: Address of the CosmWasm contract.
        new_admin_address: Address that will become the new admin.

    Returns:
        A tuple of (tx, ledger_client) ready for signing & broadcasting.
    """
    # Initialize client
    network_cfg = NetworkConfig(
        chain_id=CHAIN_ID,
        url=RPC_ENDPOINT,
    )
    client = LedgerClient(network_cfg)

    # Build the MsgUpdateAdmin protobuf message
    msg = MsgUpdateAdmin(
        sender=sender_address,
        contract=contract_address,
        new_admin=new_admin_address,
    )

    # Pack into Any type
    any_msg = Any()
    any_msg.Pack(msg, type_url_prefix='')  # cosmpy handles type_url internally

    # Create transaction and add message
    tx = Transaction()
    tx.add_message(any_msg)

    # Set a placeholder fee & gas (will be adjusted when we sign)
    tx.set_fee(FEE_DENOM, amount=5000, gas_limit=DEFAULT_GAS_LIMIT)

    return tx, client


# step:2 file: Query the code hash of a specific smart contract
# query_contract_info.py
# Fetch contract info from Neutron LCD and return JSON.

import requests
from typing import Dict

REST_ENDPOINT = "https://rest-kralum.neutron.org"  # Change to your preferred LCD

class ContractQueryError(Exception):
    """Custom error to clearly signal query failures."""


def query_contract_info(contract_address: str, lcd: str = REST_ENDPOINT) -> Dict:
    """Request contract metadata from the LCD endpoint.

    Args:
        contract_address (str): Bech32 contract address.
        lcd (str): Base URL for the LCD server.

    Returns:
        Dict: Parsed JSON with contract metadata.
    """
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


# step:4 file: Query the code hash of a specific smart contract
# query_code_info.py

import requests
from typing import Dict

REST_ENDPOINT = "https://rest-kralum.neutron.org"  # Same endpoint as Step 2

class CodeInfoQueryError(Exception):
    pass

def query_code_info(code_id: str, lcd: str = REST_ENDPOINT) -> Dict:
    """Retrieve code-info (including `code_hash`) from the LCD.

    Args:
        code_id (str): The code ID extracted in Step 3.
        lcd (str): Base URL for the LCD server.

    Returns:
        Dict: JSON payload containing code-info.
    """
    url = f"{lcd}/cosmwasm/wasm/v1/code/{code_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("code_info", {})
    except requests.RequestException as exc:
        raise CodeInfoQueryError(f"LCD request failed: {exc}") from exc
    except ValueError:
        raise CodeInfoQueryError("Malformed JSON in LCD response")


# step:5 file: Query the code hash of a specific smart contract
# extract_code_hash.py

from typing import Dict

class CodeHashExtractionError(Exception):
    pass

def extract_code_hash(code_info: Dict) -> str:
    """Safely extract the `code_hash` value.

    Args:
        code_info (Dict): Output from `query_code_info`.

    Returns:
        str: The hexadecimal code hash.
    """
    try:
        code_hash = code_info["data_hash"] or code_info["code_hash"]  # field name may differ
        if not code_hash:
            raise KeyError
        return code_hash
    except KeyError:
        raise CodeHashExtractionError("`code_hash` not present in code-info payload")


# step:2 file: Migrate an existing smart contract to a new code ID
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.exceptions import QueryError

async def validate_new_code_id(contract_address: str, new_code_id: int, rpc_url: str = "https://rpc-kralum.neutron-1.neutron.org") -> bool:
    """Validate that `new_code_id` exists and differs from the contract's current code ID.

    Args:
        contract_address (str): Address of the contract to migrate.
        new_code_id (int): The code ID to migrate to.
        rpc_url (str): RPC endpoint for Neutron.
    Returns:
        bool: True if the validation succeeds, otherwise an exception is raised.
    """
    try:
        cfg = NetworkConfig(
            chain_id="neutron-1",
            url=rpc_url,
            fee_minimum_gas_price="0.025untrn",
            fee_denomination="untrn",
        )
        client = LedgerClient(cfg)

        # Ensure the new code ID exists
        code_info = client.query.wasm.get_code_info(new_code_id)
        if code_info is None:
            raise ValueError(f"Code ID {new_code_id} does not exist on-chain.")

        # Fetch current contract info
        contract_info = client.query.wasm.get_contract_info(contract_address)
        if int(contract_info["code_id"]) == new_code_id:
            raise ValueError("Contract already instantiated with this code ID.")

        return True
    except (QueryError, Exception) as err:
        raise RuntimeError(f"Validation failed: {err}") from err


# step:1 file: Deploy the example contract to Neutron mainnet
# Step 1 – Initialise a Neutron mainnet client using cosmpy
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet


def get_neutron_mainnet_client(mnemonic: str, rpc_url: str = "https://rpc-kralum.neutron-1.neutron.org:443") -> LedgerClient:
    # Returns a cosmpy LedgerClient configured for Neutron mainnet
    if not mnemonic:
        raise ValueError("Mnemonic must not be empty")

    cfg = NetworkConfig(
        chain_id="neutron-1",
        url=rpc_url,
        fee_min_denom="untrn",
        gas_price=0.025,  # 0.025 NTRN/gas is a reasonable default for mainnet
    )

    wallet = LocalWallet.create_from_mnemonic(mnemonic)
    return LedgerClient(cfg, wallet)



# step:2 file: Deploy the example contract to Neutron mainnet
# Step 2 – Ensure that the .wasm file is present and valid
import os


def ensure_wasm_file(path: str) -> str:
    # Validates the existence and size (< 4 MiB) of the compiled .wasm file
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WASM file not found at {path}")

    size = os.path.getsize(path)
    if size > 4 * 1024 * 1024:
        raise ValueError(f"WASM binary is {size} bytes which exceeds the 4 MiB limit.")

    return os.path.abspath(path)



# step:3 file: Deploy the example contract to Neutron mainnet
# Step 3 – Build a MsgStoreCode tx
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1 import tx_pb2 as wasm_tx
from cosmpy.protos.cosmwasm.wasm.v1 import types_pb2 as wasm_types


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


# step:5 file: Deploy the example contract to Neutron mainnet
# Step 5 – Construct instantiate tx
from cosmpy.protos.cosmwasm.wasm.v1 import tx_pb2 as wasm_tx
import json as jsonlib


def build_instantiate_tx(client: LedgerClient, code_id: int, init_msg: dict, label: str, admin: str | None = None) -> Transaction:
    msg = wasm_tx.MsgInstantiateContract(
        sender=client.wallet.address(),
        admin=admin or client.wallet.address(),
        code_id=code_id,
        label=label,
        msg=jsonlib.dumps(init_msg).encode(),
        funds=[]  # add coins here if your contract expects an initial deposit
    )
    tx = client.tx.create([msg], memo=f"Instantiate {label}", gas_limit=1_000_000)
    return tx


# step:7 file: Deploy the example contract to Neutron mainnet
# Step 7 – Retrieve contract address
import json


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



# step:1 file: Show the current block height of the Neutron chain
import requests
from typing import Optional

def connect_rpc_endpoint(rpc_endpoint: str = 'https://rpc-kralum.neutron.org') -> str:
    """
    Attempts to connect to the given Neutron RPC endpoint by querying the `/status`
    route. Returns the endpoint string if successful; raises an exception otherwise.
    """
    try:
        # Hit `/status` to confirm the node is alive
        url = rpc_endpoint.rstrip('/') + '/status'
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        # Basic sanity check on the payload
        if 'result' not in response.json():
            raise ValueError('Unexpected response payload from RPC endpoint.')

        return rpc_endpoint
    except requests.RequestException as err:
        raise ConnectionError(
            f'Unable to reach Neutron RPC endpoint at {rpc_endpoint}: {err}'
        ) from err


# step:2 file: Show the current block height of the Neutron chain
import json
import subprocess
from typing import Dict

def neutrond_status(rpc_endpoint: str) -> Dict:
    """
    Executes `neutrond status --node <rpc_endpoint>` via subprocess and returns
    the parsed JSON dictionary containing the node's sync information.
    """
    try:
        cmd = [
            'neutrond',
            'status',
            '--node',
            rpc_endpoint,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f'`neutrond status` failed: {err.stderr}') from err
    except json.JSONDecodeError as err:
        raise ValueError('Failed to parse JSON from neutrond output.') from err


# step:3 file: Show the current block height of the Neutron chain
from typing import Dict

def extract_block_height(status_json: Dict) -> int:
    """
    Extracts the latest block height from the status JSON returned by `neutrond status`.
    """
    try:
        height_str = status_json['sync_info']['latest_block_height']
        return int(height_str)
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(
            'Invalid status JSON format: unable to locate `latest_block_height`.'
        ) from err


# step:1 file: Query contract metadata on Celatone
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException


def open_celatone_explorer(chain_id: str, download_dir: str = "/tmp") -> webdriver.Chrome:
    """Launch Celatone (https://celatone.osmosis.zone) for the given chain and
    return an initialized Selenium WebDriver.

    Args:
        chain_id (str): Either "neutron-1" (mainnet) or "pion-1" (testnet).
        download_dir (str): Directory where Celatone will drop the metadata JSON.

    Returns:
        webdriver.Chrome: A configured Chrome WebDriver pointing at Celatone.
    """

    if chain_id not in ("neutron-1", "pion-1"):
        raise ValueError("Unsupported chain id. Use 'neutron-1' or 'pion-1'.")

    url = f"https://celatone.osmosis.zone/{chain_id}"

    # Configure Chrome for head-less use and automatic downloads
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

        # Wait until the search bar is rendered so we know the page finished loading
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(("css selector", "input[type='search']"))
        )
        return driver
    except WebDriverException as exc:
        raise RuntimeError(f"Failed to open Celatone explorer: {exc}") from exc


# step:2 file: Query contract metadata on Celatone
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def search_contract_address(driver: webdriver.Chrome, contract_address: str, timeout: int = 15) -> None:
    """Paste the contract address into Celatone's search bar and navigate to the
    contract page.

    Args:
        driver (webdriver.Chrome): Active Celatone WebDriver.
        contract_address (str): Bech32 address of the target contract.
        timeout (int): Max seconds to wait for the contract page to load.
    """

    try:
        # Locate the search bar element and submit the address
        search_box = driver.find_element(By.CSS_SELECTOR, "input[type='search']")
        search_box.clear()
        search_box.send_keys(contract_address + Keys.ENTER)

        # Wait until URL contains the contract address, indicating navigation
        WebDriverWait(driver, timeout).until(
            EC.url_contains(contract_address.lower())
        )
    except TimeoutException:
        raise RuntimeError("Celatone did not navigate to the contract page in time.")


# step:3 file: Query contract metadata on Celatone
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


def navigate_to_metadata_tab(driver: webdriver.Chrome, timeout: int = 10) -> None:
    """Click Celatone's "Metadata" tab for the currently opened contract page."""
    try:
        # The tab usually appears as a button or anchor containing the visible text "Metadata"
        metadata_tab = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Metadata')] | //a[contains(., 'Metadata')]"))
        )
        metadata_tab.click()

        # Wait until the JSON download (</>) icon is visible in the Metadata view
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(@title, 'Download') or contains(@aria-label, 'Download')]"))
        )
    except (TimeoutException, NoSuchElementException):
        raise RuntimeError("Could not open the Metadata tab on Celatone.")


# step:4 file: Query contract metadata on Celatone
import os
import time
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def download_metadata_json(driver: webdriver.Chrome, download_dir: str, timeout: int = 30) -> Path:
    """Click the download button and wait until the metadata JSON is fully
    written to disk.

    Args:
        driver (webdriver.Chrome): Active WebDriver on the Metadata tab.
        download_dir (str): Directory configured in open_celatone_explorer().
        timeout (int): Max seconds to wait for the file to finish downloading.

    Returns:
        Path: Absolute path to the downloaded metadata JSON file.
    """

    # Grab a snapshot of existing files so we can detect the new one later
    pre_existing = set(Path(download_dir).iterdir()) if os.path.isdir(download_dir) else set()

    # Click the download (code / </>) button
    try:
        download_btn = driver.find_element(By.XPATH, "//button[contains(@title, 'Download') or contains(@aria-label, 'Download')]")
        download_btn.click()
    except Exception as exc:
        raise RuntimeError("Failed to click Celatone's download button") from exc

    # Poll for a new .json file that was not present earlier
    end_time = time.time() + timeout
    while time.time() < end_time:
        current_files = set(Path(download_dir).iterdir())
        new_files = [f for f in current_files - pre_existing if f.suffix.lower() == ".json"]
        if new_files:
            # Celatone sometimes writes a *.crdownload first; wait until file stabilises.
            candidate = new_files[0]
            if not candidate.name.endswith(".crdownload"):
                return candidate.resolve()
        time.sleep(0.5)

    raise TimeoutException("Timed out waiting for metadata JSON download to complete.")


# step:3 file: List all smart contracts deployed by my account
"""query_contracts.py
Python utility for querying contracts by creator via Neutron's LCD (REST) endpoint.
Requires the `httpx` dependency:  pip install httpx[http2]
"""
from typing import Optional, Dict, Any
import httpx

LCD_URL = "https://lcd.neutron.org"  # Change to your preferred public or self-hosted LCD

def query_contracts_by_creator(address: str, node: str = "https://neutron-rpc.polkachu.com:443") -> Dict:
    """Fetch schedule metadata from the Neutron Cron module via `neutrond` CLI."""
    try:
        cmd = ["neutrond", "query", "wasm", "list-contracts-by-creator", address, "--output", "json", "--node", node]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except:
        raise ValueError("Received non-JSON response from neutrond CLI") from exc


async def fetch_all_contracts_by_creator(creator_address: str, page_limit: int = 1000) -> List[str]:
    """Return a complete list of contract addresses deployed by `creator_address`."""
    contracts: List[str] = []
    next_key: Optional[str] = None

    while True:
        page = await query_contracts_by_creator(
            creator_address=creator_address,
            limit=page_limit,
            pagination_key=next_key,
        )

        # Extract contracts list from page data and extend our accumulator
        contracts.extend(page.get("contracts", []))

        # Determine if more pages exist
        next_key = page.get("pagination", {}).get("next_key")
        if not next_key:
            break  # No more pages

    return contracts

# Example standalone execution for quick testing
if __name__ == "__main__":
    address = "neutron1..."  # Replace with a real creator address
    all_contracts = asyncio.run(fetch_all_contracts_by_creator(address))
    print(f"Total contracts found: {len(all_contracts)}")
    for idx, c in enumerate(all_contracts, start=1):
        print(f"{idx}. {c}")


# step:2 file: Upload the example contract WASM code
import hashlib
from pathlib import Path

# Compute the SHA-256 checksum of the wasm binary and return it as a hex string.
# This checksum can be cross-checked against the chain for provenance.
def validate_wasm_checksum(wasm_path: Path) -> str:
    if not wasm_path.is_file():
        raise FileNotFoundError(f'File not found: {wasm_path}')

    sha256 = hashlib.sha256()
    with wasm_path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    checksum = sha256.hexdigest()
    return checksum


# Sign the Transaction with the provided wallet and broadcast it.
# Raises an exception on error and returns the successful TxResponse.
def sign_and_broadcast_tx(tx: Transaction, wallet: LocalWallet, client: LedgerClient):
    # Fill in sequence & account number from chain state
    tx = tx.with_sequence(wallet.get_sequence(client)).with_account_number(wallet.get_account_number(client))

    signed_tx = tx.sign(wallet)
    response = client.broadcast_tx(signed_tx)

    if response.tx_response.code != 0:
        raise RuntimeError(f'Tx failed with log: {response.tx_response.raw_log}')

    return response


# step:5 file: Upload the example contract WASM code
import json
import re

# Extract the code_id emitted by the store_code event in the TxResponse.
# Tries JSON parsing first, then falls back to regex scanning.
def extract_code_id_from_tx(response) -> int:
    raw_log = response.tx_response.raw_log

    # Attempt JSON parsing (preferred because it is deterministic)
    try:
        parsed_logs = json.loads(raw_log)[0]
        for event in parsed_logs.get('events', []):
            if event.get('type') == 'store_code':
                for attr in event.get('attributes', []):
                    if attr.get('key') == 'code_id':
                        return int(attr.get('value'))
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    # Fallback: regex scanning for robustness
    match = re.search(r'\"code_id\":\s*\"?(\d+)\"?', raw_log)
    if match:
        return int(match.group(1))

    raise ValueError('code_id not found in transaction logs')


# ===================================================================================
# == CRON FUNCTIONS!!!!
# == Governance and Proposal Functions
# ===================================================================================

def construct_param_change_proposal(new_security_address: str, deposit: str = "10000000untrn", output_path: str = "proposal.json") -> str:
    """Generate a Param-Change proposal file that updates the Cron module's security_address."""
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
    proposal = {
        "propose": {
            "title": title,
            "description": description,
            "msgs": [{"custom": msg_update_params}],
        }
    }
    return proposal

def wait_for_voting_result(proposal_id: str, chain_id: str = "neutron-1", node: str = "https://rpc-kralum.neutron.org:443", poll_interval: int = 15, max_attempts: int = 800) -> str:
    """Polls proposal status via CLI until it is finalized or times out."""
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

# ===================================================================================
# == Cron Module Functions
# ===================================================================================

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

# ===================================================================================
# == Utility and Verification Functions
# ===================================================================================

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
    for key in ("last_execution_height", "last_executed_height"):
        if (value := schedule_data.get(key)) is not None:
            return int(value)
    raise KeyError("Could not find last execution height field in schedule data.")

