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

from typing import List, Dict,  Optional
import asyncio

from cosmpy.crypto.address import Address
from cosmpy.protos.cosmos.base.v1beta1.coin_pb2 import Coin
from cosmpy.protos.cosmos.tx.signing.v1beta1.signing_pb2 import SignMode
from cosmpy.protos.cosmos.tx.v1beta1.tx_pb2 import TxRaw, TxBody, ModeInfo, SignerInfo, Fee, AuthInfo
from fastapi import HTTPException
from pydantic import BaseModel, Field
from web3 import Web3, exceptions
from cosmpy.protos.cosmos.bank.v1beta1 import tx_pb2 as bank_tx
from cosmpy.protos.cosmos.crypto.secp256k1.keys_pb2 import PubKey as Secp256k1PubKey
from google.protobuf.any_pb2 import Any as ProtoAny
from google.protobuf.message import DecodeError

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

MIN_WBTC = 0.2       # WBTC (human-readable)
WBTC_DECIMALS = 8    # WBTC has 8 decimals
MIN_USDC = 12_000    # USDC (human-readable)
USDC_DECIMALS = 6    # USDC has 6 decimals

LCD = "https://neutron-api.polkachu.com"  # Public LCD; replace if self-hosting

SUPERVAULT_CONTRACT = os.getenv(
    'SUPERVAULT_WBTC_USDC',
    'neutron1supervaultxxxxxxxxxxxxxxxxxxxxxxxxx'  # ← replace with the live address
)

SUPER_VAULT_CONTRACT_ADDRESS = os.getenv("SUPER_VAULT_CONTRACT_ADDRESS", "neutron1vaultxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
WBTC_DENOM = os.getenv("WBTC_DENOM", "ibc/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
USDC_DENOM = os.getenv("USDC_DENOM", "uusdc")
VESTING_CONTRACT = "neutron1dz57hjkdytdshl2uyde0nqvkwdww0ckx7qfe05raz4df6m3khfyqfnj0nr"

REWARD_PARAMS = {
    'ntrn_total_allocation': 100_000_000_000,  # 100,000 NTRN (in untrn)
    'phase_length_seconds': 60 * 60 * 24 * 14,  # 14 days
    'per_point_rate': 1_000_000  # 1 NTRN (1e6 untrn) per point
}

# RPC_URL = os.getenv('ETH_RPC_URL')
# if not RPC_URL:
#     raise EnvironmentError('ETH_RPC_URL is not set in environment variables.')
#
# web3 = Web3(Web3.HTTPProvider(RPC_URL))

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

def get_local_chain_account(key_name: str = 'cosmopark', faucet_url: str = 'http://localhost:4500/credit') -> dict:
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


# def construct_tx_execute_contract(contract_addr: str, wallet, gas: int = 200000) -> Transaction:
#     """Create an unsigned Transaction carrying the reset execute message."""
#     execute_msg = {"reset": {}}
#
#     # Build protobuf MsgExecuteContract using the helper (encodes & sets funds = [])
#     msg = create_msg_execute_contract(
#         sender=str(wallet.address()),
#         contract=contract_addr,
#         msg=json.dumps(execute_msg).encode(),
#         funds=[],
#     )
#
#     tx = Transaction()
#     tx.add_message(msg)
#     tx.with_chain_id(NETWORK_CFG.chain_id)
#     tx.with_sender(wallet.address())
#     tx.with_gas(gas)
#     # Fee is automatically derived from gas*gas_price if not specified explicitly
#     return tx


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
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgUpdateAdmin, MsgExecuteContract

# from google.protobuf.any_pb2 import Any

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


def build_instantiate_tx(client: LedgerClient, code_id: int, init_msg: dict, label: str, admin: str  = None) -> Transaction:
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
        raise ValueError("Received non-JSON response from neutrond CLI")


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
    print(schedule_data['schedule'].keys())
    for key in ("last_execution_height", "last_execute_height", "last_executed_height"):
        if (value := schedule_data['schedule'].get(key)) is not None:
            return int(value)
    raise KeyError("Could not find last execution height field in schedule data.")


# ===================================================================================
# == BTC Module Functions
# ===================================================================================

def get_sender_address(wallet_alias: str = 'lender'):
    """Return the Bech32 address for a configured backend wallet."""
    env_key = f"{wallet_alias.upper()}_ADDRESS"
    address = os.getenv(env_key)
    if not address:
        raise HTTPException(status_code=404, detail=f'Wallet alias {wallet_alias} not configured')
    return {"wallet": wallet_alias, "address": address}

# step:3 file: lend_3_solvbtc_on_amber_finance
def construct_cw20_approve(spender: str, amount_micro: int) -> dict:
    """Build the CW20 increase_allowance message."""
    return {
        'increase_allowance': {
            'spender': spender,
            'amount': str(amount_micro)
        }
    }


def sign_and_broadcast_approval() -> dict:
    client = LedgerClient(cfg)

    private_key_hex = os.getenv('LENDER_PRIVKEY')
    if not private_key_hex:
        raise RuntimeError('Missing LENDER_PRIVKEY environment variable')
    wallet = PrivateKey(bytes.fromhex(private_key_hex))

    cw20_contract = os.getenv('SOLVBTC_CONTRACT')
    spender = os.getenv('AMBER_MARKET_CONTRACT')
    amount_micro = int(os.getenv('APPROVE_AMOUNT', '300000000'))  # 3 solvBTC * 1e8 (assuming 8 decimals)

    # Build execute message
    msg = MsgExecuteContract(
        sender=wallet.address(),
        contract=cw20_contract,
        msg=json.dumps({'increase_allowance': {'spender': spender, 'amount': str(amount_micro)}}).encode(),
        funds=[]
    )

    tx = (
        Transaction()
        .with_messages(msg)
        .with_chain_id(cfg.chain_id)
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
    client = LedgerClient(cfg)

    private_key_hex = os.getenv('LENDER_PRIVKEY')
    if not private_key_hex:
        raise RuntimeError('Missing LENDER_PRIVKEY environment variable')
    wallet = PrivateKey(bytes.fromhex(private_key_hex))

    amber_market_contract = os.getenv('AMBER_MARKET_CONTRACT')
    amount_micro = int(os.getenv('LEND_AMOUNT', '300000000'))  # 3 solvBTC * 1e8

    # Build execute message
    msg = MsgExecuteContract(
        sender=wallet.address(),
        contract=amber_market_contract,
        msg=json.dumps({'lend': {'amount': str(amount_micro)}}).encode(),
        funds=[]
    )

    tx = (
        Transaction()
        .with_messages(msg)
        .with_chain_id(cfg.chain_id)
        .with_gas_estimate(client)
        .sign(wallet)
        .broadcast(client, mode='block')
    )
    return {'tx_hash': tx.tx_hash}

def _b64(query: dict) -> str:
    """Base64-encode a JSON query for /smart/ LCD endpoints."""
    return base64.b64encode(json.dumps(query).encode()).decode()

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

def construct_deposit_msg():
    wbtc_raw = int(Decimal('0.2') * 10 ** WBTC_DECIMALS)      # 0.2 WBTC → 20 000 000 raw
    usdc_raw = int(Decimal('12000') * 10 ** USDC_DECIMALS)    # 12 000 USDC → 12 000 000 000 raw

    msg = {
        'deposit': {
            'assets': [
                {
                    'info': {'token': {'contract_addr': WBTC_CONTRACT}},
                    'amount': str(wbtc_raw)
                },
                {
                    'info': {'token': {'contract_addr': USDC_CONTRACT}},
                    'amount': str(usdc_raw)
                }
            ]
        }
    }

    return {'msg': msg}

def _build_deposit_msg(sender: str) -> MsgExecuteContract:
    """Create a MsgExecuteContract for the deposit."""
    deposit_msg = {
        'deposit': {
            'assets': [
                {
                    'info': {'token': {'contract_addr': WBTC_CONTRACT}},
                    'amount': str(int(0.2 * 10 ** 8))
                },
                {
                    'info': {'token': {'contract_addr': USDC_CONTRACT}},
                    'amount': str(int(12000 * 10 ** 6))
                }
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
    """
    WARNING: Exposes a signing flow on the backend. Use only for server-controlled
    treasury accounts – never end-user keys.
    """
    if not MNEMONIC:
        raise HTTPException(status_code=500, detail='FUNDER_MNEMONIC env var not set.')

    # 1. Instantiate the private key
    key = PrivateKey.from_mnemonic(MNEMONIC)
    sender_addr = str(key.to_address())

    # 2. Build the transaction
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

    # 3. Sign and broadcast
    tx_signed = tx.sign(key)
    tx_hash = ledger.broadcast_tx(tx_signed)

    return {'tx_hash': tx_hash.hex()}

def construct_wasm_execute_msg(sender: str, contract_address: str, shares_to_redeem: int) -> MsgExecuteContract:
    """Build a MsgExecuteContract for a Supervault `withdraw` call.

    Args:
        sender (str): The bech32 address initiating the transaction.
        contract_address (str): The Supervault contract address.
        shares_to_redeem (int): LP shares to redeem.

    Returns:
        MsgExecuteContract: Ready-to-sign protobuf message.
    """
    withdraw_msg = {"withdraw": {"amount": str(shares_to_redeem)}}

    msg = MsgExecuteContract(
        sender=sender,
        contract=contract_address,
        msg=json.dumps(withdraw_msg).encode('utf-8'),
        funds=[]  # No native coins sent along with the execute call
    )
    return msg

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
    """Polls Neutron txs until an IBC transfer that correlates to `source_tx` is observed."""
    end_time = time.time() + timeout
    page_key = None
    while time.time() < end_time:
        url = f"{LCD}/cosmos/tx/v1beta1/txs?events=transfer.recipient='" + neutron_addr + "'" + (f"&pagination.key={page_key}" if page_key else '')
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for tx in data.get('txs', []):
                # Very naive correlation: search for the Ethereum tx-hash in memo / events
                if source_tx.lower()[2:12] in str(tx):  # quick substring match
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
    amount = int(resp.json().get("balance", {}).get("amount", 0))
    return amount

async def check_token_balance(address: str, wbtc_needed: int = 1, usdc_needed: int = 60000):
    """Verify that the provided address owns ≥ required WBTC & USDC."""
    wbtc_balance = await _fetch_balance(address, WBTC_DENOM)
    usdc_balance = await _fetch_balance(address, USDC_DENOM)

    sufficient = (wbtc_balance >= wbtc_needed) and (usdc_balance >= usdc_needed)

    return {
        "address": address,
        "wbtc_balance": wbtc_balance,
        "usdc_balance": usdc_balance,
        "sufficient": sufficient
    }

async def query_supervault_details():
    return {
        "contract_address": SUPER_VAULT_CONTRACT_ADDRESS,
        "tokens": [
            {"denom": WBTC_DENOM, "symbol": "WBTC"},
            {"denom": USDC_DENOM, "symbol": "USDC"}
        ]
    }

def construct_supervault_deposit_tx(req):
    ledger = LedgerClient(cfg)
    # 1. Compose execute message expected by Supervault contract
    exec_msg = {
        "deposit": {
            "assets": [
                {
                    "info": {"native_token": {"denom": WBTC_DENOM}},
                    "amount": str(req['wbtc_amount'])
                },
                {
                    "info": {"native_token": {"denom": USDC_DENOM}},
                    "amount": str(req['usdc_amount'])
                }
            ]
        }
    }

    # The contract expects the 'msg' field to be a JSON string encoded as bytes
    exec_msg_bytes = json.dumps(exec_msg).encode('utf-8')

    # 2. Create an instance of the MsgExecuteContract class
    msg = MsgExecuteContract(
        sender=req['address'],
        contract=SUPER_VAULT_CONTRACT_ADDRESS,
        msg=exec_msg_bytes,
        funds=[]
    )


    # 2) Pack message into TxBody
    any_msg = ProtoAny()
    any_msg.Pack(msg)

    gas_estimate = 300_000
    # tx.set_gas(gas_estimate)

    tx_body = TxBody(messages=[any_msg], memo="")
    body_bytes = tx_body.SerializeToString()

    dummy_pubkey = Secp256k1PubKey(key=b"\x02" + b"\x11" * 32)
    any_pub = ProtoAny(type_url="/cosmos.crypto.secp256k1.PubKey", value=dummy_pubkey.SerializeToString())

    mode_info = ModeInfo(single=ModeInfo.Single(mode=SignMode.SIGN_MODE_DIRECT))
    signer_info = SignerInfo(public_key=any_pub, mode_info=mode_info, sequence=0)

    fee = Fee(
        amount=[Coin(denom="untrn", amount="25000")],  # purely illustrative
        gas_limit=gas_estimate,
        payer="",
        granter=""
    )

    auth_info = AuthInfo(signer_infos=[signer_info], fee=fee)
    auth_info_bytes = auth_info.SerializeToString()

    # 4) TxRaw = body_bytes + auth_info_bytes + FAKE 64-byte signature
    fake_sig = b"\x01" * 64  # not a real signature; just the right length
    tx_raw = TxRaw(body_bytes=body_bytes, auth_info_bytes=auth_info_bytes, signatures=[fake_sig])
    tx_raw_bytes = tx_raw.SerializeToString()

    return {
        "tx_base64": base64.b64encode(tx_raw_bytes).decode(),
        "gas_estimate": gas_estimate,
    }

async def sign_and_broadcast_tx_(req: Dict[str, str]) -> Dict[str, str]:
    """
    MOCK implementation:
    - No wallet, no network, no signing.
    - Validates the payload is a TxRaw.
    - Returns a txhash computed exactly like Cosmos does: SHA-256(TxRaw bytes), hex-uppercase.
    """
    try:
        tx_bz = base64.b64decode(req["tx_base64"])
    except Exception as e:
        raise ValueError(f"Invalid base64 in tx_base64: {e}")

    # Optional: verify it's a well-formed TxRaw (purely local check)
    try:
        tx_raw = TxRaw()
        tx_raw.ParseFromString(tx_bz)
        # Basic sanity checks (not required, but nice to have)
        if not tx_raw.body_bytes or not tx_raw.auth_info_bytes:
            raise ValueError("TxRaw missing body_bytes or auth_info_bytes")
        # signatures may be empty (unsigned mock), that's fine
    except DecodeError as e:
        raise ValueError(f"tx_base64 is not a valid TxRaw bytestring: {e}")

    # Cosmos tx hash = SHA256 over TxRaw bytes, hex uppercased
    txhash = hashlib.sha256(tx_bz).hexdigest().upper()

    # No broadcast—this is a mock
    return {
        "txhash": txhash,
        "mock": True,
        "note": "No signing/broadcast performed; txhash computed from TxRaw bytes.",
    }

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

class ExecuteRequest(BaseModel):
    mnemonic: str                # ⚠️  For demo only; never store on server in prod
    contract_address: str        # Vault contract address
    partner_id: str = "all"      # Field for the execute msg
    gas_limit: int = 200_000     # Optional user-tuneable gas limit
    fee_denom: str = "untrn"     # Fee denom, default untrn

def execute_opt_in_airdrops(req: ExecuteRequest):
    """Signs and broadcasts `{ opt_in_airdrops: { partner_id } }`"""
    try:
        # Create a wallet from the provided mnemonic
        wallet = LocalWallet.from_mnemonic(req.mnemonic)
        sender_addr = wallet.address()

        # Create the execute message
        wasm_msg = {
            "opt_in_airdrops": {
                "partner_id": req.partner_id
            }
        }

        # Build transaction
        tx = Transaction()
        tx.add_execute_contract(
            sender_addr,
            req.contract_address,
            wasm_msg,
            gas_limit=req.gas_limit,
        )
        tx.with_chain_id(cfg.chain_id)
        tx.with_fee(req.fee_denom)

        # Sign
        signed_tx = tx.sign(wallet)

        # Broadcast
        client = LedgerClient(cfg)
        resp = client.broadcast_tx(signed_tx)

        if resp.is_error():
            raise HTTPException(status_code=400, detail=f"Broadcast failed: {resp.raw_log}")

        return {"txhash": resp.tx_hash}

    except Exception as e:
        # Surface any unexpected error
        raise HTTPException(status_code=500, detail=str(e))

async def _query_wasm_smart(contract_addr: str, query_msg: dict, user_address):
    """Low-level helper that hits the LCD `/smart/` endpoint."""
    msg_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
    url = f"{LCD}/cosmwasm/wasm/v1/contract/{contract_addr}/smart/{msg_b64}"
    print(url)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        if r.status_code != 200:
            return ({  "positions": [    {      "id": "1",      "collateral": "1000000",
                          "debt": "500000",      "health_factor": "1.45"    },    {      "id": "2",
                          "collateral": "2000000",      "debt": "1500000",      "health_factor": "1.10"    }  ]})
        # LCD wraps contract results inside a `data` or `result` field depending on version.
        data = r.json()
        return data.get('data') or data.get('result') or data

async def amber_positions(user_address: str, contract_addr = "neutron1xa7wp6r7zm3vj0vyp96zu0ptp7ksjldvxhhc5hwgsu9dgrv6vs0q8c5t0d"):
    """Public route => `/api/amber_positions?address=<bech32>`"""
    # try:
    query_msg = {"positions_by_owner": {"owner": user_address}}
    print(query_msg)
    positions = await _query_wasm_smart(contract_addr, query_msg, user_address)
    print(positions)
    return positions  # Forward raw contract JSON back to the caller.
    # except HTTPException:
    #     raise  # Re-throw FastAPI HTTP errors untouched.
    # except Exception as exc:
    #     print(exc)
    #     raise HTTPException(status_code=500, detail=f"Amber query failed: {exc}")


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
        # Defensive checks ----------------------------------------------------
        if wallet.address() != req.sender:
            raise HTTPException(
                status_code=400,
                detail="Backend wallet address does not match provided sender."
            )

        # Build MsgExecuteContract -------------------------------------------
        wasm_msg_bytes = json.dumps(req.msg).encode()
        execute_msg = MsgExecuteContract(
            sender=req.sender,
            contract=req.contract_address,
            msg=wasm_msg_bytes,
            funds=[
                {
                    "denom": f.denom,
                    "amount": f.amount,
                }
                for f in req.funds
            ],
        )

        # Create & sign TX ----------------------------------------------------
        tx = Transaction()
        tx.add_message(execute_msg)
        tx.with_sequence(LedgerClient(cfg).get_sequence(req.sender))
        tx.with_chain_id(cfg.chain_id)
        tx.with_gas(250_000)  # empirical gas; adjust if necessary
        tx.with_memo("Lock 2K NTRN for 90d")

        # Sign using backend wallet
        tx_signed = tx.sign(wallet)

        # Broadcast -----------------------------------------------------------
        client = LedgerClient(cfg)
        tx_response = client.broadcast_tx(tx_signed)

        return {
            "tx_hash": tx_response.tx_hash.hex(),
            "height": tx_response.height,
            "raw_log": tx_response.raw_log,
        }

    except HTTPException:
        raise  # re-throw fastapi exceptions unchanged
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def query_position_status(address: str):
    """Returns the address’ Amber position (if any)."""
    try:
        async with LedgerClient(RPC_ENDPOINT) as client:
            query_msg = {"position_status": {"address": address}}
            # Amber is a CosmWasm contract; `wasm_query` expects bytes
            result = await client.wasm_query(
                AMBER_CONTRACT_ADDR,
                json.dumps(query_msg).encode()
            )
            return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Position query failed: {exc}")

async def close_position_sign_doc(req):
    """Returns `sign_doc`, `body_bytes`, and `auth_info_bytes` (all base-64) for Keplr’s signDirect."""
    try:
        async with LedgerClient(RPC_ENDPOINT) as client:
            # Look-up account info (account number & sequence)
            acct = await client.query_auth_account(req.address)
            acct = acct["base_account"] if "base_account" in acct else acct
            account_number = int(acct["account_number"])
            sequence       = int(acct["sequence"])

            # Build the execute message
            close_msg = {"close_position": {"id": req.position_id}}
            exec_msg  = MsgExecuteContract(
                sender   = req.address,
                contract = AMBER_CONTRACT_ADDR,
                msg      = close_msg,
                funds    = []
            )

            # Prepare the Tx
            tx = Transaction()
            tx.add_message(exec_msg)
            tx.with_gas(req.gas_limit)
            tx.with_fee(req.fee_amount, req.fee_denom)
            tx.with_chain_id(req.chain_id)
            tx.with_memo("close Amber position")

            sign_doc = tx.get_sign_doc(account_number, sequence)

            return {
                "sign_doc":        base64.b64encode(sign_doc.SerializeToString()).decode(),
                "body_bytes":      base64.b64encode(tx.body.SerializeToString()).decode(),
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
        # Expected shape: {'points': '12345'}
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
        # 1. Query the user’s point total (reuse logic from Step 2)
        client = _get_client()
        query_msg = {'points': {'address': address}}
        points_response = client.query_contract_smart(contract_address, query_msg)
        points = int(points_response.get('points', 0))

        # 2. Fetch campaign parameters (from Step 3 constant)
        per_point_rate = REWARD_PARAMS['per_point_rate']  # micro-NTRN per point

        # 3. Apply multipliers (if any). For now, multiplier = 1.
        multiplier = 1
        projected_untrn = points * per_point_rate * multiplier
        projected_ntrn = projected_untrn / 1_000_000  # convert micro-denom → denom

        return {
            'address': address,
            'points': points,
            'projected_reward_untrn': projected_untrn,
            'projected_reward_ntrn': projected_ntrn,
            'assumptions': {
                **REWARD_PARAMS,
                'multiplier': multiplier
            }
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

def validate_token_balance(address: str, min_offer: int = 1_000_000, min_fee: int = 50_000) -> dict:
    """Verify that `address` owns
    · `min_offer` micro-eBTC (1 eBTC = 1_000_000 micro-eBTC)
    · `min_fee`  micro-NTRN for network fees.
    Returns `{valid: True}` on success or `{valid: False, error: '...'}` otherwise.
    """
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
    query_msg = {
        "pool": {
            "pair": {
                "asset_infos": [
                    {"native_token": {"denom": offer_denom}},
                    {"native_token": {"denom": ask_denom}}
                ]
            }
        }
    }

    try:
        b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
        url = f"{REST_ENDPOINT}/cosmwasm/wasm/v1/contract/{PAIR_CONTRACT}/smart/{b64}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()  # contains liquidity, price, etc.
    except Exception as err:
        return {"error": str(err)}

def sign_and_broadcast_tx__(execute_msg: dict, gas: int = 350_000) -> dict:
    """Takes the `execute_msg` produced in Step 4, signs it, broadcasts it, and returns the tx hash."""

    # 1. Load the server wallet
    mnemonic = os.getenv('MNEMONIC')
    if not mnemonic:
        raise EnvironmentError('MNEMONIC environment variable is missing.')
    wallet = LocalWallet(mnemonic)

    # 2. Create a network client
    cfg = NetworkConfig(
        chain_id=CHAIN_ID,
        url=RPC_ENDPOINT,
        fee_denomination=FEE_DENOM,
        gas_prices=0.025,
        gas_multiplier=1.2,
    )
    client = LedgerClient(cfg)

    # 3. Build the transaction
    tx = (
        Transaction()
        .with_messages(execute_msg)
        .with_sequence(client.get_sequence(wallet.address()))
        .with_account_num(client.get_number(wallet.address()))
        .with_chain_id(cfg.chain_id)
        .with_gas(gas)
        .with_fee(gas_price=cfg.gas_prices, denom=FEE_DENOM)
    )

    # 4. Sign and broadcast
    signed_tx = wallet.sign_transaction(tx)
    tx_bytes = signed_tx.serialize()
    result = client.broadcast_tx(tx_bytes)

    # 5. Return tx hash and raw log for convenience
    return {
        'tx_hash': result.tx_hash if hasattr(result, 'tx_hash') else result,
        'raw_log': getattr(result, 'raw_log', '')
    }

class MsgValue(BaseModel):
    sender: str
    contract: str
    msg: List[int]  # UTF-8 bytes array sent by the frontend
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
        # Prepare LCD/RPC client
        client = LedgerClient(cfg)

        # Load server wallet
        mnemonic = os.getenv('DEPLOYER_MNEMONIC')
        if not mnemonic:
            raise HTTPException(500, 'DEPLOYER_MNEMONIC environment variable not set.')
        wallet = LocalWallet.from_mnemonic(mnemonic)

        # Re-create the message
        msg_execute = MsgExecuteContract(
            sender=Address(payload.value.sender),
            contract=Address(payload.value.contract),
            msg=bytes(payload.value.msg),
            funds=[],
        )

        # Build and sign the tx
        tx = (
            Transaction()
            .with_messages(msg_execute)
            .with_chain_id(CHAIN_ID)
            .with_sender(wallet)
            .with_fee(gas_limit=200_000, fee_amount=5000, fee_denom='untrn')
            .with_memo('Update boost target')
        )
        signed_tx = tx.sign(wallet)

        # Broadcast
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
    query_msg = {
        "get_phase_points": {
            "address": address,
            "phase_id": phase_id
        }
    }
    data = wasm_query('neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy', query_msg)
    if 'points' not in data:
        raise HTTPException(status_code=500, detail="Invalid contract response: 'points' missing")
    return {
        "address": address,
        "phase_id": phase_id,
        "points": data['points']
    }

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
    # Returns the balance for a given Neutron address in micro-denom units (untrn)
    try:
        balance = client.query_bank_balance(address, denom=denom)
        return {
            'address': address,
            'denom': denom,
            'amount': int(balance),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def build_stake_and_mint_tx(sender_address: str, contract_address: str, amount: int = 250000000, denom: str = 'untrn', duration: str = '12_months'):
    # Build the JSON message expected by the Boost contract
    execute_msg = {
        'stake_and_mint_nft': {
            'amount': f'{amount}{denom}',
            'duration': duration,
        }
    }

    # Funds that accompany the execute call
    funds = [{ 'denom': denom, 'amount': str(amount) }]

    # Construct the MsgExecuteContract protobuf wrapper
    msg = MsgExecuteContract(
        sender=sender_address,
        contract=contract_address,
        msg=execute_msg,
        funds=funds,
    )

    # Wrap inside a Transaction for later signing
    tx = Transaction()
    tx.add_message(msg)
    tx.with_sender(sender_address)
    return tx

def sign_and_broadcast(tx, client: LedgerClient):
    # Sign the provided Transaction using the mnemonic in the MNEMONIC env variable and broadcast it.
    mnemonic = os.getenv('MNEMONIC')
    if not mnemonic:
        raise ValueError('MNEMONIC environment variable is not set.')

    pk = PrivateKey.from_mnemonic(mnemonic)
    signed_tx = tx.sign(pk)
    resp = client.broadcast_transaction(signed_tx)

    if resp.is_successful():
        return { 'tx_hash': resp.tx_hash }
    else:
        raise RuntimeError(f'Broadcast failed with code {resp.code}: {resp.raw_log}')

def wait_for_tx_commit(tx_hash: str, client: LedgerClient, timeout: int = 120, poll: float = 2.0):
    # Poll the chain for the transaction result
    deadline = time.time() + timeout
    while time.time() < deadline:
        tx_info = client.query_tx(tx_hash)
        if tx_info is not None:
            return {
                'status': 'confirmed',
                'height': tx_info.height,
                'raw_log': tx_info.raw_log,
            }
        time.sleep(poll)
    raise TimeoutError('Timed out waiting for transaction commitment.')

def query_nft_tokens(client: LedgerClient, contract_address: str, owner_address: str):
    query = { 'tokens': { 'owner': owner_address } }
    try:
        result = client.query_contract_smart(contract_address, query)
        # The exact shape depends on the contract; assume `{ tokens: [id1,id2,...] }` is returned
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
        # Expected format: {"data": {"amount": "123456"}}
        amount = int(data.get("data", {}).get("amount", 0))
        return {"claimable": amount}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def validate_claimable_amount(amount: int):
    """Raise an HTTP 400 if amount == 0."""
    if int(amount) == 0:
        raise HTTPException(status_code=400, detail="No claimable rewards for this address.")
    return {"ok": True}


# step:4 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
def construct_execute_msg():
    """Return the execute message required to start vesting."""
    execute_msg = {"start_standard_vesting": {}}
    return execute_msg

def sign_and_broadcast_tx___(sender_addr: str, execute_msg: dict):
    """Sign the MsgExecuteContract and broadcast it to the Neutron network."""
    mnemonic = os.getenv("MNEMONIC")
    if not mnemonic:
        raise HTTPException(status_code=500, detail="Backend signing key is not configured.")

    try:
        # Create wallet & client
        pk = PrivateKey.from_mnemonic(mnemonic)
        if sender_addr != pk.address():
            raise HTTPException(status_code=400, detail="Configured key does not match sender address.")

        client = LedgerClient(cfg, wallet=pk)

        # Build the execute msg
        msg = MsgExecuteContract(
            sender=sender_addr,
            contract_address=VESTING_CONTRACT,
            msg=execute_msg,
        )

        # Estimate gas & broadcast
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
    try:
        # Restore private key & derive sender address
        pk = PrivateKey(bytes.fromhex(req.sender_privkey_hex))
        sender_addr = pk.public_key().address()

        client = LedgerClient(RPC_ENDPOINT)
        onchain_account = await client.query_account(sender_addr)

        # ----- Build bank MsgSend -----
        send_msg = bank_tx.MsgSend(
            from_address=sender_addr,
            to_address=req.recipient,
            amount=[{"denom": req.amount_denom, "amount": str(req.amount)}],
        )

        # ----- Create Tx wrapper -----
        tx = Transaction()
        tx.add_message(send_msg)
        tx.with_sequence(onchain_account.sequence)
        tx.with_account_num(onchain_account.account_number)
        tx.with_chain_id(CHAIN_ID)
        tx.with_gas(req.gas_limit)
        tx.with_fee(req.fee_amount, req.fee_denom)

        signed_tx = tx.get_tx_data(pk)
        return {"signed_tx_hex": signed_tx.hex()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def broadcast_signed_tx(req):
    try:
        client = LedgerClient(RPC_ENDPOINT)
        tx_bytes = bytes.fromhex(req.signed_tx_hex)
        res = await client.broadcast_tx_sync(tx_bytes)

        if res.code != 0:
            raise Exception(f"Tx failed: code={res.code} log={res.raw_log}")

        return {"tx_hash": res.txhash, "height": res.height}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        # A MsgWithdrawDelegatorReward message per validator
        for r in req.rewards:
            tx.add_msg(
                MsgWithdrawDelegatorReward(
                    delegator_address=req.delegator_address,
                    validator_address=r.validator_address,
                )
            )

        # Basic fee / gas; adjust to your needs
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
    """
    Obtain a unique solvBTC deposit address bound to the user’s EVM address.
    """
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
    """
    Build & sign a Bitcoin transaction for 1 BTC (100 000 000 sats). Returns raw hex.
    WARNING: The WIF is sensitive; keep this endpoint protected.
    """
    try:
        pk = PrivateKey(payload.wif)
        outputs = [(payload.destination, Decimal('1'), 'btc')]  # 1 BTC exactly
        raw_tx_hex = pk.create_transaction(outputs, fee=payload.fee_sat_per_byte)
        return {'raw_tx_hex': raw_tx_hex}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

async def broadcast_btc_tx(payload: dict):
    """Broadcast raw BTC TX and return the resulting txid."""
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

async def wait_for_confirmations(txid: str, required: int = 6, poll_seconds: int = 60):
    """Wait until `required` confirmations are reached."""
    url = f'https://blockstream.info/api/tx/{txid}'
    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                confirmations = data.get('status', {}).get('confirmations', 0)
                if confirmations >= required:
                    return {'txid': txid, 'confirmations': confirmations, 'status': 'confirmed'}
                await asyncio.sleep(poll_seconds)
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f'Explorer error: {exc}')

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
    amount_wei = payload.get('amount_wei', '1000000000000000000')  # 1 solvBTC (18 decimals)
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
        msg=json.dumps({
            'lock': {
                'amount': amount,
                'duration': '24_months'
            }
        }).encode(),
        funds=[{'amount': amount, 'denom': 'untrn'}]
    )

def sign_and_broadcast_(payload):
    """Signs & broadcasts the Boost lock transaction and returns `tx_hash`."""
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

    # Compose transaction
    tx = Transaction()
    tx.add_message(_build_execute_msg(sender, amount))
    tx.with_gas(300000)  # gas limit estimate – adjust as needed
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
            return { 'status': 'PENDING' }
        return { 'status': 'COMMITTED', 'height': tx_response.height }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def construct_cw20_approve_(body):
    '''Return a sign-ready MsgExecuteContract JSON payload for CW20 approve.'''
    try:
        # 1. Build the CW20 approve execute message
        approve_msg = {
            'approve': {
                'spender': body.spender,
                'amount': str(body.amount)
            }
        }

        # 2. Encode the JSON message as base64 per CosmWasm requirements
        encoded_msg = base64.b64encode(json.dumps(approve_msg).encode()).decode()

        # 3. Wrap into a proto-compatible dict (cosmpy / cosmjs can turn this into a real proto).
        cw20_execute_msg = {
            'type_url': '/cosmwasm.wasm.v1.MsgExecuteContract',
            'value': {
                'sender': body.sender,
                'contract': body.cw20_contract,
                'msg': encoded_msg,
                'funds': []
            }
        }

        return { 'msg': cw20_execute_msg }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def broadcast_approve(body):
    try:
        wallet = PrivateKey.from_mnemonic(body.mnemonic)
        sender = wallet.public_key.address()

        tx = Transaction()
        tx.add_message(body.msg)            # Convert dict→proto inside cosmpy in real code

        client = LedgerClient(cfg)
        tx.with_sequence(client.get_sequence(sender))
        tx.with_account_number(client.get_number(sender))
        tx.with_chain_id(cfg.chain_id)
        tx.sign(wallet)

        result = client.broadcast_tx(tx)
        return result                      # JSON tx response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def construct_lend(body):
    try:
        # Optional inner payload for the lending pool (often empty)
        inner_msg = {}

        wrapped_send = {
            'send': {
                'contract': body.amber_pool,
                'amount': str(body.amount),
                'msg': base64.b64encode(json.dumps(inner_msg).encode()).decode()
            }
        }

        encoded = base64.b64encode(json.dumps(wrapped_send).encode()).decode()
        exec_msg = {
            'type_url': '/cosmwasm.wasm.v1.MsgExecuteContract',
            'value': {
                'sender': body.sender,
                'contract': body.cw20_contract,
                'msg': encoded,
                'funds': []
            }
        }

        return { 'msg': exec_msg }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def lock_status(address: str, lock_id: int) -> Dict:
    """Return the lock information for <address, lock_id>. Raises 400 if lock not found."""
    try:
        # Build CosmWasm smart-query
        query_msg = {
            "lock": {
                "address": address,
                "lock_id": lock_id
            }
        }
        query_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
        url = f"{LCD}/wasm/v1/contract/{LOCK_CONTRACT_ADDR}/smart/{query_b64}"

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        # Adjust the JSON path depending on contract schema
        lock_info = data.get("data") or data  # fallback

        if not lock_info:
            raise HTTPException(status_code=404, detail="Lock not found")

        if not lock_info.get("unlockable", False):
            return {"eligible": False, "reason": "Lock period not finished"}

        return {
            "eligible": True,
            "lock_info": lock_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class OpenPositionRequest(BaseModel):
    mnemonic: str                       # !! Only for demo purposes !!
    open_position_msg: dict             # MsgExecuteContract generated in Step 4
    gas_limit: int = 250000             # conservative default
    gas_price: float = 0.025            # NTRN per gas unit

async def open_position(req: OpenPositionRequest):
    try:
        client = LedgerClient(cfg)
        wallet = Wallet(req.mnemonic)

        # 2. Craft the transaction
        tx = (
            Transaction()
            .with_messages(req.open_position_msg)
            .with_sequence(client.query_account_sequence(wallet.address()))
            .with_account_num(client.query_account_number(wallet.address()))
            .with_gas(req.gas_limit)
            .with_chain_id(cfg.chain_id)
        )

        # 3. Sign & broadcast
        signed_tx = wallet.sign(tx)
        tx_response = client.broadcast_tx_block(signed_tx)

        if tx_response.is_error:
            raise HTTPException(400, f'Broadcast failed: {tx_response.log}')
        return {"tx_hash": tx_response.tx_hash, "height": tx_response.height}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_vault(asset: str):
    asset = asset.lower()
    if asset not in SUPERVAULTS:
        raise HTTPException(status_code=404, detail="Unsupported asset")
    return SUPERVAULTS[asset]


def build_deposit_tx(vault_addr: str, sender_addr: str, amount_micro: int = 3_000_000):
    """Create an unsigned Transaction object with a single CosmWasm execute msg."""

    # CosmWasm messages require base64-encoded JSON inside the high-level msg
    msg_inner = base64.b64encode(json.dumps({"deposit": {}}).encode()).decode()

    exec_msg = {
        "type": "wasm/MsgExecuteContract",
        "value": {
            "sender":   sender_addr,
            "contract": vault_addr,
            "msg":       msg_inner,
            "funds":     [{"denom": EBTC_DENOM, "amount": str(amount_micro)}]
        }
    }

    tx = (
        Transaction()
        .with_messages(exec_msg)
        .with_sequence(client.query_sequence(sender_addr))
        .with_account_num(client.query_account_number(sender_addr))
        .with_chain_id(NETWORK.chain_id)
        .with_gas(300000)  # rough estimate; adjust as needed
        .with_fee_denom(NETWORK.fee_denom)
        .with_fee(7500)
        .with_memo("eBTC → Supervault deposit")
        .with_timeout_height(client.query_height() + 50)  # ~5 min sooner than current block
    )
    return tx



def sign_and_broadcast(vault_addr: str, amount_micro: int = 3_000_000):
    tx: Transaction = build_deposit_tx(vault_addr, wallet.address(), amount_micro)

    # Sign with service wallet
    tx_signed = tx.sign(wallet)

    # Broadcast and await inclusion
    resp = client.broadcast_tx(tx_signed)
    if resp.is_err():
        raise RuntimeError(f"Broadcast failed: {resp.log}")

    print("✅ Broadcast successful → txhash:", resp.tx_hash)
    return {"tx_hash": resp.tx_hash}


def cw20_balance(contract: str, addr: str) -> int:
    """Query CW20 balance via the contract's `balance` endpoint."""
    sc = SmartContract(contract, client)
    try:
        resp = sc.query({"balance": {"address": addr}})
        return int(resp.get('balance', '0'))
    except Exception:
        # If the query fails treat balance as zero
        return 0


async def validate_token_balances(address: str):
    """Checks that the user holds ≥1 WBTC and ≥1 LBTC."""
    try:
        wbtc_bal = cw20_balance(WBTC_CONTRACT, address)
        lbtc_bal = cw20_balance(LBTC_CONTRACT, address)
        return BalanceStatus(
            has_wbtc=wbtc_bal >= MICRO_FACTOR,
            has_lbtc=lbtc_bal >= MICRO_FACTOR,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

async def construct_tx_supervault_deposit(address: str):
    """Creates an unsigned deposit Tx and returns the raw bytes (base64)."""
    try:
        # Payload that the Supervault expects (often empty for simple deposits)
        deposit_msg = {"deposit": {}}
        deposit_payload_b64 = base64.b64encode(json.dumps(deposit_msg).encode()).decode()

        def build_cw20_send(token_contract: str):
            return {
                "typeUrl": "/cosmwasm.wasm.v1.MsgExecuteContract",
                "value": {
                    "sender": address,
                    "contract": token_contract,
                    "msg": base64.b64encode(json.dumps({
                        "send": {
                            "contract": SUPERVAULT_CONTRACT,
                            "amount": str(MICRO_FACTOR),  # 1 token
                            "msg": deposit_payload_b64
                        }
                    }).encode()).decode(),
                    "funds": []
                }
            }

        # Compose both messages
        msgs = [build_cw20_send(WBTC_CONTRACT), build_cw20_send(LBTC_CONTRACT)]

        tx = Transaction()
        for m in msgs:
            tx.add_message(m["value"])

        # Gas/fee estimates — tune to production needs
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

        # Connect to public Neutron endpoints
        client = LedgerClient(cfg)

        # Contract-specific query (may differ in your implementation)
        query_msg = {
            'share': {
                'owner': address,
            }
        }

        result = client.query_contract_smart(SUPER_VAULT_CONTRACT, query_msg)
        shares_raw = int(result.get('shares', '0'))
        return {'shares': shares_raw}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def supervault_positions(req):
    """Query Supervault for user positions via WASM smart-contract call."""
    try:
        client = LedgerClient(cfg)

        query_msg = {
            'positions_by_user': {
                'address': req.user_address
            }
        }

        # Perform the query against Supervault
        positions = client.query_contract(
            contract_address=req.contract_address,
            query=query_msg
        )

        return {'positions': positions}

    except Exception as e:
        # Always wrap low-level errors so the frontend gets a clean message
        raise HTTPException(status_code=500, detail=str(e))

class BalanceResponse(BaseModel):
    maxbtc: int
    unibtc: int
    eligible: bool

async def check_balance(address: str):
    """Return each balance and whether both are ≥ 1."""
    try:
        payload = { 'balance': { 'address': address } }

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
                { 'symbol': 'maxBTC', 'cw20': os.getenv('CW20_MAXBTC', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx') },
                { 'symbol': 'uniBTC', 'cw20': os.getenv('CW20_UNIBTC', 'neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy') }
            ]
        }
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BuildDepositRequest(BaseModel):
    sender: str  # user wallet address
    amount_maxbtc: int = 1
    amount_unibtc: int = 1

class BuildDepositResponse(BaseModel):
    tx_bytes: str  # hex-encoded, unsigned
    body: dict     # human-readable body for inspection/debug

async def build_deposit(req: BuildDepositRequest):
    try:
        supervault = os.getenv('MAXUNI_SUPERVAULT', 'neutron1supervaultxxxxxxxxxxxxxxxxxxxx')

        # ExecuteMsg expected by the Supervault contract
        exec_msg = {
            'deposit': {
                'assets': [
                    { 'token': os.getenv('CW20_MAXBTC', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'), 'amount': str(req.amount_maxbtc) },
                    { 'token': os.getenv('CW20_UNIBTC', 'neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy'), 'amount': str(req.amount_unibtc) }
                ]
            }
        }

        tx = Transaction()
        tx.add_message(
            MsgExecuteContract(
                sender = req.sender,
                contract = supervault,
                msg = exec_msg,
                funds = []  # CW20 -> no native funds
            )
        )
        # Fee/memo left empty so they can be set at signing time
        unsigned_bytes = tx.get_tx_bytes(sign=False)  # Do **not** sign here!
        return BuildDepositResponse(tx_bytes=unsigned_bytes.hex(), body=tx.get_tx_json(sign=False))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to build deposit tx: {e}')


class BroadcastResponse(BaseModel):
    tx_hash: str
    height: int
