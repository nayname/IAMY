# step:1 file: Set the Cron module security address to `neutron1guard...`
import json


def construct_param_change_proposal(new_security_address: str, deposit: str = "10000000untrn", output_path: str = "proposal.json") -> str:
    """Generate a Param-Change proposal file that updates the Cron module's security_address parameter."""

    proposal = {
        "title": "Update Cron security_address",
        "description": f"Updates Cron module security_address param to {new_security_address}.",
        "changes": [
            {
                "subspace": "cron",
                "key": "SecurityAddress",
                "value": f"\"{new_security_address}\""
            }
        ],
        "deposit": deposit
    }

    try:
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(proposal, fp, indent=2)
    except IOError as err:
        raise RuntimeError(f"Could not write proposal file: {err}") from err

    return output_path


# step:2 file: Set the Cron module security address to `neutron1guard...`
import json, subprocess


def submit_gov_proposal(proposal_file: str, from_key: str, chain_id: str = "neutron-1", node: str = "https://rpc-kralum.neutron.org:443", fees: str = "2000untrn", gas: str = "400000") -> str:
    """Submits the param-change proposal and extracts the proposal_id from the tx response."""

    cmd = [
        "neutrond", "tx", "gov", "submit-proposal", "param-change", proposal_file,
        "--from", from_key,
        "--chain-id", chain_id,
        "--node", node,
        "--fees", fees,
        "--gas", gas,
        "-y",
        "--output", "json"
    ]

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


# step:3 file: Set the Cron module security address to `neutron1guard...`
import json, subprocess, time


def wait_for_voting_result(proposal_id: str, chain_id: str = "neutron-1", node: str = "https://rpc-kralum.neutron.org:443", poll_interval: int = 15, max_attempts: int = 800) -> str:
    """Polls proposal status until finalised (PASSED/REJECTED) or timeout."""

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        proc = subprocess.run([
            "neutrond", "query", "gov", "proposal", str(proposal_id),
            "--chain-id", chain_id,
            "--node", node,
            "--output", "json"
        ], capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)

        status = json.loads(proc.stdout).get("status")
        print(f"[poll] proposal {proposal_id} status: {status}")

        if status == "PROPOSAL_STATUS_PASSED":
            return status
        if status in ("PROPOSAL_STATUS_REJECTED", "PROPOSAL_STATUS_FAILED", "PROPOSAL_STATUS_ABORTED"):
            raise RuntimeError(f"Proposal {proposal_id} ended with status {status}")

        time.sleep(poll_interval)

    raise TimeoutError("Exceeded maximum attempts while waiting for proposal to pass.")


# step:4 file: Set the Cron module security address to `neutron1guard...`
import json, subprocess


def query_cron_params(chain_id: str = "neutron-1", node: str = "https://rpc-kralum.neutron.org:443") -> dict:
    """Fetches the current Cron module parameters."""

    proc = subprocess.run([
        "neutrond", "query", "cron", "params",
        "--chain-id", chain_id,
        "--node", node,
        "--output", "json"
    ], capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    return json.loads(proc.stdout).get("params", {})


def verify_security_address(expected: str, chain_id: str = "neutron-1", node: str = "https://rpc-kralum.neutron.org:443") -> bool:
    """Validates that security_address equals the expected value."""

    params = query_cron_params(chain_id, node)
    actual = params.get("security_address")
    if actual == expected:
        print("✅ Cron security_address updated successfully.")
        return True
    raise ValueError(f"security_address mismatch: expected {expected}, got {actual}")


# step:2 file: query_transaction_details_with_cast_tx
import os, re, httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()
LCD_ENDPOINT = os.getenv('COSMOS_LCD', 'https://rest.cosmos.directory/neutron-1')

async def _fetch_raw_tx(tx_hash: str) -> dict:
    """Internal helper to retrieve raw tx data from the LCD"""
    if not re.fullmatch(r'^(0x)?[0-9a-fA-F]{64}$', tx_hash):
        raise ValueError('Invalid transaction hash')
    # LCD expects plain hex (no 0x)
    clean = tx_hash.lower().replace('0x', '')
    url = f"{LCD_ENDPOINT}/cosmos/tx/v1beta1/txs/{clean}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=resp.status_code, detail=str(e))
        return resp.json()

@router.get('/api/tx/{tx_hash}')
async def get_raw_tx_endpoint(tx_hash: str):
    """Public BFF route to obtain raw transaction JSON."""
    try:
        return await _fetch_raw_tx(tx_hash)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


# step:3 file: query_transaction_details_with_cast_tx
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get('/api/tx/{tx_hash}/formatted')
async def get_formatted_tx(tx_hash: str) -> Dict[str, Any]:
    try:
        raw = await _fetch_raw_tx(tx_hash)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    if 'tx_response' not in raw:
        raise HTTPException(status_code=500, detail='Unexpected LCD response format')

    r = raw['tx_response']
    formatted = {
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
    return formatted


# step:1 file: sign_an_arbitrary_message_using_eth_sign_on_a_cosmos-evm_(ethermint_evmos)_json-rpc_endpoint
from web3 import Web3
import os

# Function: get_web3
# Purpose : Establish a connection to an Ethereum JSON-RPC node.
# Notes   : Reads the RPC URL from the environment variable `RPC_URL` or defaults to
#           `http://localhost:8545`. Raises an error if the node is unreachable.

def get_web3():
    rpc_url = os.getenv("RPC_URL", "http://localhost:8545")
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.isConnected():
            raise ConnectionError(f"Unable to connect to JSON-RPC at {rpc_url}")
        return w3
    except Exception as e:
        raise RuntimeError(f"get_web3 error: {str(e)}")


# step:3 file: sign_an_arbitrary_message_using_eth_sign_on_a_cosmos-evm_(ethermint_evmos)_json-rpc_endpoint
import os
import requests

# Function: eth_sign
# Params  : from_address (str)  — the address doing the signing (must be unlocked on the node)
#           message_hex (str)   — 0x-prefixed hex string produced in Step 2
# Returns : signature (str)     — 0x-prefixed ECDSA signature

def eth_sign(from_address: str, message_hex: str) -> str:
    rpc_url = os.getenv("RPC_URL", "http://localhost:8545")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_sign",
        "params": [from_address, message_hex]
    }

    try:
        response = requests.post(rpc_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            raise RuntimeError(data['error'])
        return data['result']  # 0x-signature
    except Exception as e:
        raise RuntimeError(f"eth_sign failed: {e}")


# step:4 file: sign_an_arbitrary_message_using_eth_sign_on_a_cosmos-evm_(ethermint_evmos)_json-rpc_endpoint
from eth_account.messages import encode_defunct
from web3 import Web3, HTTPProvider
import os

# Function: verify_signature
# Purpose : Ensure that the provided signature was produced by the expected address.
# Returns : True if valid, False otherwise.

def verify_signature(message: str, signature: str, expected_address: str, w3: Web3 | None = None) -> bool:
    try:
        # Use existing Web3 instance or create a new one
        if w3 is None:
            w3 = Web3(HTTPProvider(os.getenv('RPC_URL', 'http://localhost:8545')))
        msg = encode_defunct(text=message)
        recovered = w3.eth.account.recover_message(msg, signature=signature)
        return recovered.lower() == expected_address.lower()
    except Exception as e:
        raise RuntimeError(f"verify_signature error: {e}")


# step:1 file: query_the_all_tokens_list_from_an_nft_contract
from typing import Dict
import re
import requests
from fastapi import FastAPI, HTTPException

# Regular expression for basic Bech32 address validation (prefix + 38 data chars)
BECH32_REGEX = re.compile(r'^[a-z0-9]{3,15}1[0-9a-z]{38}$')

# Public LCD endpoint for the target chain (edit to match your network)
LCD_ENDPOINT = 'https://lcd.osmosis.zone'

class ContractValidationError(Exception):
    '''Raised when the given address fails validation checks.'''


def _validate_contract_address(address: str, lcd_endpoint: str = LCD_ENDPOINT) -> Dict:
    '''Return contract_info if address is a CW-721 contract, else raise ContractValidationError.'''

    # 1. Syntax check (Bech32)
    if not BECH32_REGEX.match(address):
        raise ContractValidationError('Address is not valid Bech32 format')

    # 2. Query LCD to confirm the address is a contract account
    url = f'{lcd_endpoint}/cosmwasm/wasm/v1/contract/{address}'
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
    except requests.exceptions.RequestException as err:
        raise ContractValidationError(f'Could not reach LCD: {err}') from err

    payload = resp.json()
    info = payload.get('contract_info')
    if info is None:
        raise ContractValidationError('No contract_info returned; address is not a contract')

    # 3. Naïve CW-721 detection based on the label field
    if 'cw721' not in info.get('label', '').lower():
        raise ContractValidationError('Contract does not appear to be a CW721 instance')

    return {'is_valid': True, 'address': address, 'contract_info': info}


# ─── FastAPI wrapper so the frontend can call this as REST ───
app = FastAPI()


@app.get('/api/validate_contract')
async def api_validate_contract(address: str):
    '''HTTP GET endpoint → /api/validate_contract?address=<ADDR>'''
    try:
        return _validate_contract_address(address)
    except ContractValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


# step:1 file: lend_3_solvbtc_on_amber_finance
from fastapi import FastAPI, HTTPException
import os

app = FastAPI()

@app.get('/api/wallet/address')
async def get_sender_address(wallet_alias: str = 'lender'):
    """Return the Bech32 address for a configured backend wallet."""
    env_key = f"{wallet_alias.upper()}_ADDRESS"
    address = os.getenv(env_key)
    if not address:
        raise HTTPException(status_code=404, detail=f'Wallet alias {wallet_alias} not configured')
    return {"wallet": wallet_alias, "address": address}


# step:2 file: lend_3_solvbtc_on_amber_finance
import base64, json, os, requests
from fastapi import FastAPI, HTTPException

REST_ENDPOINT = os.getenv('NEUTRON_REST', 'https://rest-kralum.neutron-1.neutron.org')
app = FastAPI()

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


# step:3 file: lend_3_solvbtc_on_amber_finance
def construct_cw20_approve(spender: str, amount_micro: int) -> dict:
    """Build the CW20 increase_allowance message."""
    return {
        'increase_allowance': {
            'spender': spender,
            'amount': str(amount_micro)
        }
    }


# step:4 file: lend_3_solvbtc_on_amber_finance
import os, json
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgExecuteContract


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
        .with_chain_id(network.chain_id)
        .with_gas_estimate(client)
        .sign(wallet)
        .broadcast(client, mode='block')
    )
    return {'tx_hash': tx.tx_hash}


if __name__ == '__main__':
    print(sign_and_broadcast_approval())


# step:5 file: lend_3_solvbtc_on_amber_finance
def construct_amber_lend_tx(amount_micro: int) -> dict:
    """Build the lend (supply) message for Amber Finance market contract."""
    return {
        'lend': {
            'amount': str(amount_micro)
        }
    }


# step:6 file: lend_3_solvbtc_on_amber_finance
import os, json
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgExecuteContract


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
        .with_chain_id(network.chain_id)
        .with_gas_estimate(client)
        .sign(wallet)
        .broadcast(client, mode='block')
    )
    return {'tx_hash': tx.tx_hash}


if __name__ == '__main__':
    print(sign_and_broadcast_lend())


# step:2 file: increase_the_user’s_deposit_in_the_wbtc_usdc_supervault_by_0.2_wbtc_and_12_000_usdc
# backend/validate_balances.py
# FastAPI route that ensures the user owns enough WBTC & USDC for the deposit.

import os
import base64
import json
from fastapi import APIRouter, HTTPException
import httpx

router = APIRouter()

NODE_LCD = os.getenv('NEUTRON_LCD', 'https://rest.cosmos.directory/neutron')
WBTC_CONTRACT = os.getenv('WBTC_CONTRACT', 'neutron1wbtcxxxxxxxxxxxxxxxxxxxxxxx')
USDC_CONTRACT = os.getenv('USDC_CONTRACT', 'neutron1usdcxxxxxxxxxxxxxxxxxxxxxxx')

MIN_WBTC = 0.2       # WBTC (human-readable)
WBTC_DECIMALS = 8    # WBTC has 8 decimals
MIN_USDC = 12_000    # USDC (human-readable)
USDC_DECIMALS = 6    # USDC has 6 decimals

def _b64(query: dict) -> str:
    """Base64-encode a JSON query for /smart/ LCD endpoints."""
    return base64.b64encode(json.dumps(query).encode()).decode()

async def _cw20_balance(contract: str, addr: str) -> int:
    url = f"{NODE_LCD}/cosmwasm/wasm/v1/contract/{contract}/smart/{_b64({'balance': {'address': addr}})}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        return int(r.json()['data']['balance'])

@router.get('/api/validate_balances')
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


# step:3 file: increase_the_user’s_deposit_in_the_wbtc_usdc_supervault_by_0.2_wbtc_and_12_000_usdc
# backend/supervault_address.py
import os
from fastapi import APIRouter

router = APIRouter()

SUPERVAULT_CONTRACT = os.getenv(
    'SUPERVAULT_WBTC_USDC',
    'neutron1supervaultxxxxxxxxxxxxxxxxxxxxxxxxx'  # ← replace with the live address
)

@router.get('/api/supervault_address')
async def supervault_address():
    """Return the current WBTC/USDC Supervault address."""
    return {'address': SUPERVAULT_CONTRACT}


# step:4 file: increase_the_user’s_deposit_in_the_wbtc_usdc_supervault_by_0.2_wbtc_and_12_000_usdc
# backend/construct_deposit_msg.py
import os
import json
from decimal import Decimal
from fastapi import APIRouter
from pydantic import BaseModel, Field

WBTC_CONTRACT = os.getenv('WBTC_CONTRACT', 'neutron1wbtcxxxxxxxxxxxxxxxxxxxxxxx')
USDC_CONTRACT = os.getenv('USDC_CONTRACT', 'neutron1usdcxxxxxxxxxxxxxxxxxxxxxxx')
SUPERVAULT_CONTRACT = os.getenv('SUPERVAULT_WBTC_USDC', 'neutron1supervaultxxxxxxxxxxxxxxxxxxxxxxxxx')

WBTC_DECIMALS = 8
USDC_DECIMALS = 6

class DepositMsgResponse(BaseModel):
    msg: dict = Field(..., description='JSON execute message for MsgExecuteContract')

router = APIRouter()

@router.get('/api/construct_deposit_msg', response_model=DepositMsgResponse)
async def construct_deposit_msg():
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


# step:5 file: increase_the_user’s_deposit_in_the_wbtc_usdc_supervault_by_0.2_wbtc_and_12_000_usdc
# backend/sign_and_broadcast.py
import os
import json
from fastapi import APIRouter, HTTPException
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.contract import MsgExecuteContract

WBTC_CONTRACT = os.getenv('WBTC_CONTRACT', 'neutron1wbtcxxxxxxxxxxxxxxxxxxxxxxx')
USDC_CONTRACT = os.getenv('USDC_CONTRACT', 'neutron1usdcxxxxxxxxxxxxxxxxxxxxxxx')
SUPERVAULT_CONTRACT = os.getenv('SUPERVAULT_WBTC_USDC', 'neutron1supervaultxxxxxxxxxxxxxxxxxxxxxxxxx')

CHAIN_ID = os.getenv('NEUTRON_CHAIN_ID', 'neutron-1')
RPC_ENDPOINT = os.getenv('NEUTRON_RPC', 'https://rpc.cosmos.directory/neutron')
MNEMONIC = os.getenv('FUNDER_MNEMONIC')  # Never commit real mnemonics to Git!

router = APIRouter()

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

@router.post('/api/sign_and_broadcast')
async def sign_and_broadcast_tx():
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


# step:1 file: set_the_txflags_environment_variable_with_recommended_gas_settings
import os
from pathlib import Path


def set_txflags(gas_flags: str = "--gas=auto --gas-adjustment 1.3 --gas-prices 0.025uatom") -> None:
    """Set TXFLAGS in the running process and persist it to the user's shell profile.

    Args:
        gas_flags (str): The value to assign to TXFLAGS.
    """
    try:
        # 1. Set for the current Python process (effective immediately for subprocess calls)
        os.environ["TXFLAGS"] = gas_flags

        # 2. Attempt to persist in the user's shell profile so new terminals inherit the var
        shell = os.environ.get("SHELL", "")
        # Decide which RC file(s) to modify based on the active shell
        candidate_files = []
        if "zsh" in shell:
            candidate_files = [Path.home() / ".zshrc"]
        else:
            candidate_files = [Path.home() / ".bashrc", Path.home() / ".bash_profile"]

        export_line = f'export TXFLAGS="{gas_flags}"'

        for rc_file in candidate_files:
            try:
                if rc_file.exists():
                    content = rc_file.read_text()
                    if export_line in content:
                        # Already present, skip to the next file
                        continue
                # Append the export line
                with rc_file.open("a", encoding="utf-8") as fp:
                    fp.write(f"\n# Added by Cosmos helper\n{export_line}\n")
            except Exception as rc_err:
                # Log the error but continue execution
                print(f"[WARN] Could not update {rc_file}: {rc_err}")

        print("TXFLAGS environment variable set successfully.")
    except Exception as err:
        # Surface a clean error message for unexpected issues
        raise RuntimeError(f"Failed to set TXFLAGS: {err}")


if __name__ == "__main__":
    # When executed as a script, perform the side effect immediately
    set_txflags()


# step:4 file: redeem_lp_shares_from_the_maxbtc_ebtc_supervault
import json
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgExecuteContract


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


# step:5 file: redeem_lp_shares_from_the_maxbtc_ebtc_supervault
import os
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import MnemonicWallet

# IMPORTANT: never hard-code sensitive keys. Use environment variables or a secure vault.
MNEMONIC_ENV = "USER_MNEMONIC"  # The environment variable expected to store the user mnemonic.

NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url="https://rpc.kralum.neutron-1.neutron.org",  # Replace with a trusted RPC endpoint
    fee_denom="untrn",
    fee_minimum_gas_price=0.025,
)

def sign_and_broadcast_tx(execute_msg):
    """Signs and broadcasts the given execute message.

    Args:
        execute_msg (MsgExecuteContract): Message produced by `construct_wasm_execute_msg`.

    Returns:
        dict: `{ "tx_hash": "..." }` when successful.
    """
    mnemonic = os.getenv(MNEMONIC_ENV)
    if not mnemonic:
        raise EnvironmentError(f"Mnemonic not provided in env var {MNEMONIC_ENV}.")

    wallet = MnemonicWallet(mnemonic)
    client = LedgerClient(NETWORK)

    tx = Transaction()
    tx.add_message(execute_msg)
    tx.with_sender(wallet.address())

    # Estimate gas & fees, sign, then broadcast
    tx = tx.autofill(client)
    tx = tx.sign(wallet)

    response = client.broadcast_block(tx)
    if response.is_ok():
        return {"tx_hash": response.tx_hash}
    else:
        raise Exception(f"Broadcast failed: {response.raw_log}")


# step:1 file: temporarily_unlock_an_account_for_60_seconds_so_it_can_sign_transactions_without_reprompting
import os
import threading
import time


def export_passphrase_env(passphrase: str, ttl: int = 60):
    '''
    Temporarily exports KEYRING_PASSPHRASE as an environment variable and
    clears it after `ttl` seconds in a background thread.
    '''
    if not passphrase:
        raise ValueError('Passphrase cannot be empty.')

    # Export the variable so any subprocess (e.g., simd) will inherit it.
    os.environ['KEYRING_PASSPHRASE'] = passphrase
    print('KEYRING_PASSPHRASE exported. It will be cleared in', ttl, 'seconds.')

    def _unset():
        '''Waits for ttl seconds and removes the passphrase from env.'''
        time.sleep(ttl)
        os.environ.pop('KEYRING_PASSPHRASE', None)
        print('KEYRING_PASSPHRASE cleared from environment.')

    # Fire-and-forget the cleaner thread so the main program continues.
    threading.Thread(target=_unset, daemon=True).start()



# step:2 file: temporarily_unlock_an_account_for_60_seconds_so_it_can_sign_transactions_without_reprompting
import subprocess
import os
from typing import List


def run_timeboxed_script(cmd: List[str]):
    '''
    Executes a simd CLI transaction command (sign + broadcast) while the
    KEYRING_PASSPHRASE env var is present.
    '''
    if 'KEYRING_PASSPHRASE' not in os.environ:
        raise EnvironmentError('KEYRING_PASSPHRASE is not set. Call export_passphrase_env first.')

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print('Transaction successfully broadcasted:')
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as err:
        print('simd command failed:', err.stderr)
        raise



# step:3 file: temporarily_unlock_an_account_for_60_seconds_so_it_can_sign_transactions_without_reprompting
import subprocess
import os
import time
from typing import List


def confirm_relock(cmd: List[str]) -> bool:
    '''
    Attempts to execute a signing command after the KEYRING_PASSPHRASE TTL has
    expired. Returns True if the CLI prompts for a passphrase, indicating
    that the keyring is re-locked.
    '''
    # Ensure the passphrase env var is gone
    if 'KEYRING_PASSPHRASE' in os.environ:
        print('Waiting 1 s for passphrase to clear…')
        time.sleep(1)

    # Use --dry-run so we do not broadcast real transactions during the test
    dry_cmd = cmd + ['--dry-run']

    process = subprocess.Popen(
        dry_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Give the CLI a moment to output its prompt
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



# step:1 file: add_a_ufw_rule_to_allow_ssh_on_port_22
import subprocess


def get_ufw_status():
    """Check if UFW is active and return the raw output."""
    try:
        proc = subprocess.run([
            "sudo", "ufw", "status"
        ], capture_output=True, text=True, check=True)
        return {
            "active": "inactive" not in proc.stdout.lower(),
            "output": proc.stdout.strip()
        }
    except subprocess.CalledProcessError as err:
        return {
            "active": False,
            "error": err.stderr.strip() if err.stderr else "Failed to obtain UFW status."
        }

# Example direct usage:
# result = get_ufw_status()
# print(result)


# step:2 file: add_a_ufw_rule_to_allow_ssh_on_port_22
import subprocess


def allow_ssh_via_ufw():
    """Add a UFW rule to allow SSH on port 22/tcp with the comment 'Allow SSH'."""
    try:
        proc = subprocess.run([
            "sudo", "ufw", "allow", "22/tcp", "comment", "Allow SSH"
        ], capture_output=True, text=True, check=True)
        return {
            "success": True,
            "output": proc.stdout.strip() or "Rule added"
        }
    except subprocess.CalledProcessError as err:
        return {
            "success": False,
            "error": err.stderr.strip() if err.stderr else "Failed to add SSH rule."
        }

# Example direct usage:
# response = allow_ssh_via_ufw()
# print(response)


# step:3 file: add_a_ufw_rule_to_allow_ssh_on_port_22
import subprocess


def reload_ufw():
    """Reload the UFW ruleset to apply recent changes."""
    try:
        proc = subprocess.run([
            "sudo", "ufw", "reload"
        ], capture_output=True, text=True, check=True)
        return {
            "reloaded": True,
            "output": proc.stdout.strip() or "UFW reloaded"
        }
    except subprocess.CalledProcessError as err:
        return {
            "reloaded": False,
            "error": err.stderr.strip() if err.stderr else "Failed to reload UFW."
        }

# Example direct usage:
# info = reload_ufw()
# print(info)


# step:4 file: add_a_ufw_rule_to_allow_ssh_on_port_22
import subprocess
from typing import List, Dict


def list_ufw_rules_numbered() -> Dict[str, List[str]]:
    """Return UFW rules in numbered format for easy review."""
    try:
        proc = subprocess.run([
            "sudo", "ufw", "status", "numbered"
        ], capture_output=True, text=True, check=True)
        lines = [line.strip() for line in proc.stdout.strip().split("\n") if line.strip()]
        return {"rules": lines}
    except subprocess.CalledProcessError as err:
        return {
            "error": err.stderr.strip() if err.stderr else "Unable to list UFW rules."
        }

# Example direct usage:
# numbered = list_ufw_rules_numbered()
# print("\n".join(numbered.get("rules", [])))


# step:1 file: initialize_testnet_files_with_simd_testnet_init-files
import os


def ensure_output_directory(path: str) -> str:
    """Ensure the directory used for validator configs & genesis exists.

    Args:
        path (str): Relative or absolute path where `simd` will write files.

    Returns:
        str: An absolute, verified path.

    Raises:
        RuntimeError: If the directory cannot be created or accessed.
    """
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(abs_path, exist_ok=True)  # idempotent: no error if it already exists
        return abs_path
    except Exception as err:
        raise RuntimeError(f"[ensure_output_directory] Failed for '{path}': {err}")



# step:2 file: initialize_testnet_files_with_simd_testnet_init-files
import subprocess
from pathlib import Path


def simd_testnet_init_files(output_dir: str,
                             chain_id: str = "localnet-1",
                             validators: int = 1,
                             keyring_backend: str = "test") -> None:
    """Bootstraps a local single- or multi-validator testnet.

    Args:
        output_dir (str): Directory created in Step 1.
        chain_id  (str): Custom chain-id for the test chain.
        validators (int): How many validator nodes to initialise.
        keyring_backend (str): `simd` keyring backend (e.g. "test", "os").

    Raises:
        RuntimeError: If the `simd` command returns a non-zero exit code.
    """
    home_arg = str(Path(output_dir).expanduser())
    cmd = [
        "simd", "testnet", "init-files",
        "--home", home_arg,
        "--chain-id", chain_id,
        "--v", str(validators),
        "--keyring-backend", keyring_backend,
    ]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Optional: print or log stdout for user visibility
        print(completed.stdout)
    except subprocess.CalledProcessError as err:
        print(err.stderr)
        raise RuntimeError("[simd_testnet_init_files] `simd` exited with non-zero code")



# step:3 file: initialize_testnet_files_with_simd_testnet_init-files
from pathlib import Path


def verify_generated_artifacts(output_dir: str, validators: int = 1) -> bool:
    """Sanity-check that Step 2 produced the expected files.

    Args:
        output_dir (str): Base directory used by `simd`.
        validators (int): Number of validator folders expected.

    Returns:
        bool: `True` if every file is present.

    Raises:
        FileNotFoundError: When any required artifact is missing.
    """
    base = Path(output_dir).expanduser().resolve()

    # 1. Shared genesis.json
    genesis = base / "genesis.json"
    if not genesis.is_file():
        raise FileNotFoundError(f"Missing {genesis}")

    # 2. Per-validator checks
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

    return True  # All good



# step:1 file: configure_eip-1559_gas_parameters_when_sending_a_transaction_with_ethers.js
from fastapi import FastAPI, HTTPException
from os import getenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

app = FastAPI()

# --- RPC and Key Configuration ---
RPC_URL = getenv("COSMOS_EVM_RPC", "https://rpc.evmos.org:8545")  # Override in production
PRIVATE_KEY = getenv("PRIVATE_KEY")  # NEVER commit this value; set as an env var

# --- Web3 Client ---
w3 = Web3(Web3.HTTPProvider(RPC_URL))
# Inject PoA middleware (many Cosmos EVM chains like Evmos, Canto, etc. use PoA)
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

@app.get("/api/instantiate_wallet")
async def instantiate_wallet():
    """Creates a wallet instance from PRIVATE_KEY and returns its address."""
    if not PRIVATE_KEY:
        raise HTTPException(status_code=500, detail="PRIVATE_KEY is not set in environment variables")
    acct = w3.eth.account.from_key(PRIVATE_KEY)
    return {"address": acct.address}


# step:2 file: configure_eip-1559_gas_parameters_when_sending_a_transaction_with_ethers.js
from fastapi import HTTPException

@app.get("/api/fee_data")
async def fee_data():
    """Returns baseFeePerGas, maxPriorityFeePerGas and a recommended maxFeePerGas."""
    try:
        latest_block = w3.eth.get_block('latest')
        base_fee = latest_block['baseFeePerGas']
        max_priority_fee = w3.eth.max_priority_fee  # Recommended priority tip
        # Heuristic: maxFeePerGas = baseFee + 2 * maxPriorityFee
        max_fee_per_gas = base_fee + 2 * max_priority_fee
        return {
            "baseFeePerGas": base_fee,
            "maxPriorityFeePerGas": max_priority_fee,
            "maxFeePerGas": max_fee_per_gas
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:3 file: configure_eip-1559_gas_parameters_when_sending_a_transaction_with_ethers.js
from pydantic import BaseModel
from fastapi import HTTPException

class BuildTxParams(BaseModel):
    to: str
    value: int = 0            # Amount in wei
    data: str = "0x"          # Hex-encoded calldata
    gasLimit: int = 21000     # Conservative default

@app.post("/api/build_tx")
async def build_tx(params: BuildTxParams):
    """Constructs an unsigned EIP-1559 transaction object."""
    try:
        fee_info = await fee_data()          # Dynamic fee suggestions
        wallet_info = await instantiate_wallet()
        sender = wallet_info['address']

        nonce = w3.eth.get_transaction_count(sender)
        chain_id = w3.eth.chain_id

        tx = {
            "to": params.to,
            "value": params.value,
            "data": params.data,
            "gas": params.gasLimit,
            "maxFeePerGas": fee_info['maxFeePerGas'],
            "maxPriorityFeePerGas": fee_info['maxPriorityFeePerGas'],
            "nonce": nonce,
            "chainId": chain_id,
            "type": 2  # EIP-1559
        }
        return tx
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:4 file: configure_eip-1559_gas_parameters_when_sending_a_transaction_with_ethers.js
from pydantic import BaseModel
from fastapi import HTTPException

class TxObject(BaseModel):
    to: str
    value: int
    data: str
    gas: int
    maxFeePerGas: int
    maxPriorityFeePerGas: int
    nonce: int
    chainId: int
    type: int

@app.post("/api/sign_and_send")
async def sign_and_send(tx: TxObject):
    """Signs an EIP-1559 tx using PRIVATE_KEY and broadcasts it, returning the tx hash."""
    if not PRIVATE_KEY:
        raise HTTPException(status_code=500, detail="Server missing PRIVATE_KEY")
    try:
        signed_tx = w3.eth.account.sign_transaction(tx.dict(), private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return {"txHash": tx_hash.hex()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:1 file: add_a_new_key_named_my_validator_to_the_test_keyring
from fastapi import APIRouter, HTTPException
import subprocess, json

router = APIRouter()

@router.post('/api/generate-key')
async def generate_key(key_name: str, keyring_backend: str = 'test'):
    """Generate a new key pair using `simd` and persist it to the provided keyring backend."""
    cmd = [
        'simd', 'keys', 'add', key_name,
        '--keyring-backend', keyring_backend,
        '--output', 'json'  # ensures machine-readable output
    ]

    try:
        # Execute the CLI command and capture its output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        key_info = json.loads(result.stdout)  # parse the JSON given by simd
        return {
            'message': 'Key generated successfully',
            'data': key_info
        }

    except subprocess.CalledProcessError as e:
        # When simd exits with a non-zero code we surface the error message
        raise HTTPException(
            status_code=500,
            detail=f'Key generation failed: {e.stderr.strip()}'
        )
    except json.JSONDecodeError:
        # simd did not return JSON despite --output=json flag
        raise HTTPException(
            status_code=500,
            detail='Failed to parse simd output as JSON.'
        )


# step:2 file: add_a_new_key_named_my_validator_to_the_test_keyring
from fastapi import APIRouter, HTTPException
import subprocess

router = APIRouter()

@router.get('/api/verify-key/{key_name}')
async def verify_key(key_name: str, keyring_backend: str = 'test'):
    """Check whether a key exists in the requested keyring backend."""
    cmd = [
        'simd', 'keys', 'show', key_name,
        '--keyring-backend', keyring_backend,
        '-a'  # address-only output (quieter, easier to parse)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        address = result.stdout.strip()
        if not address:
            # simd printed nothing even though command succeeded
            raise HTTPException(status_code=404, detail=f'Key `{key_name}` seems empty.')
        return {
            'key_name': key_name,
            'address': address
        }

    except subprocess.CalledProcessError:
        # simd returns non-zero when key isn't found
        raise HTTPException(
            status_code=404,
            detail=f'Key `{key_name}` not found in `{keyring_backend}` keyring.'
        )


# step:1 file: Query transaction history for my address
import time
import requests
from typing import Dict


def _ping(url: str, timeout: float = 1.5) -> float:
    """Return response-time in seconds (∞ if unreachable)."""
    start = time.time()
    try:
        requests.head(url, timeout=timeout)
        return time.time() - start
    except requests.RequestException:
        return float("inf")


def select_data_provider(prefer_graphql: bool = True) -> Dict[str, str]:
    """Choose the fastest available provider and return a descriptor dict."""
    providers = [
        {
            "name": "celatone",
            "base_url": "https://celatone-api.neutron.org/v1/graphql",
            "api_type": "graphql",
        },
        {
            "name": "lcd",
            "base_url": "https://lcd.neutron.org",
            "api_type": "rest",
        },
    ]

    # If GraphQL is preferred, try it first.
    if prefer_graphql:
        graphql_providers = [p for p in providers if p["api_type"] == "graphql"]
        if graphql_providers and _ping(graphql_providers[0]["base_url"]) != float("inf"):
            return graphql_providers[0]

    # Fallback: choose the provider with the lowest latency.
    best = min(providers, key=lambda p: _ping(p["base_url"]))
    if _ping(best["base_url"]) == float("inf"):
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


# step:1 file: generate_a_typescript_client_for_a_contract_using_ts-codegen
from fastapi import FastAPI, HTTPException
import subprocess, json

app = FastAPI()

REQUIRED_PACKAGES = ["ts-node", "@cosmwasm/ts-codegen"]


def _package_installed(pkg: str) -> bool:
    """Return True if the npm package is available locally or globally."""
    try:
        local = subprocess.run([
            "npm", "list", pkg, "--depth", "0", "--json"
        ], capture_output=True, text=True, check=False)
        local_data = json.loads(local.stdout or "{}")
        if "dependencies" in local_data and pkg in local_data["dependencies"]:
            return True
        global_ = subprocess.run([
            "npm", "list", "-g", pkg, "--depth", "0", "--json"
        ], capture_output=True, text=True, check=False)
        global_data = json.loads(global_.stdout or "{}")
        return "dependencies" in global_data and pkg in global_data["dependencies"]
    except Exception:
        return False


@app.post("/api/codegen/install")
async def install_ts_codegen():
    """Ensure cosmwasm-ts-codegen tooling is present in the project."""
    missing = [p for p in REQUIRED_PACKAGES if not _package_installed(p)]
    if not missing:
        return {"status": "ok", "message": "All required packages are already installed."}
    try:
        subprocess.run(["npm", "install", "-D", *missing], check=True)
        return {"status": "ok", "message": f"Installed missing packages: {', '.join(missing)}"}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"npm install failed: {err}")


# step:2 file: generate_a_typescript_client_for_a_contract_using_ts-codegen
from fastapi import FastAPI, HTTPException
import subprocess
from pathlib import Path

app = FastAPI()

@app.post("/api/codegen/generate")
async def run_ts_codegen(schema_dir: str = "./schema", out_dir: str = "./src/ts", client: str = "react-query"):
    """Run ts-codegen to generate typed clients from the contract JSON schema."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "npx", "ts-node", "./node_modules/@cosmwasm/ts-codegen/bin/cli.js",
        "--schema", schema_dir,
        "--out", out_dir,
        "--client", client
    ]
    try:
        subprocess.run(cmd, check=True)
        return {"status": "ok", "message": "TypeScript client generated successfully."}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"ts-codegen failed: {err}")


# step:3 file: generate_a_typescript_client_for_a_contract_using_ts-codegen
from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.post("/api/codegen/compile")
async def compile_typescript(tsconfig_path: str = "tsconfig.json"):
    """Run the TypeScript compiler in no-emit mode to validate typings."""
    cmd = ["npx", "tsc", "--project", tsconfig_path, "--noEmit"]
    try:
        subprocess.run(cmd, check=True)
        return {"status": "ok", "message": "TypeScript compilation succeeded."}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"TypeScript compilation failed: {err}")


# step:4 file: bridge_1_wbtc_from_ethereum_to_neutron
'''monitor_eth_tx.py'''
import os, time
from typing import Dict
from web3 import Web3, exceptions

RPC_URL = os.getenv('ETH_RPC_URL')
if not RPC_URL:
    raise EnvironmentError('ETH_RPC_URL is not set in environment variables.')

web3 = Web3(Web3.HTTPProvider(RPC_URL))

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


# step:5 file: bridge_1_wbtc_from_ethereum_to_neutron
'''listen_bridge_relay.py'''
import requests, time
from typing import Dict

LCD = 'https://lcd-kralum.neutron-1.neutron.org'  # Public LCD; replace if self-hosting

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


# step:6 file: bridge_1_wbtc_from_ethereum_to_neutron
'''query_neutron_bank_balance.py'''
import requests
from typing import Dict

LCD = 'https://lcd-kralum.neutron-1.neutron.org'

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


# step:1 file: Query Cron module parameters
import requests
from typing import Dict, Any

def query_cron_params(rest_endpoint: str) -> Dict[str, Any]:
    """
    Fetch the Cron module parameters from a Neutron LCD (REST) endpoint.

    Args:
        rest_endpoint (str): Base URL of the LCD endpoint (e.g., "https://lcd.neutron.org").

    Returns:
        Dict[str, Any]: The JSON payload containing Cron parameters.

    Raises:
        RuntimeError: For network problems or HTTP errors.
        ValueError: If the response does not contain the expected shape.
    """
    # Ensure there is no trailing slash to avoid double slashes when concatenating
    rest_endpoint = rest_endpoint.rstrip('/')
    url = f"{rest_endpoint}/neutron/cron/v1/params"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()               # Raise if the status code indicates an error
        data = response.json()
        if "params" not in data:
            raise ValueError("Missing 'params' field in Cron parameters response.")
        return data
    except requests.exceptions.RequestException as exc:
        # Wrap lower-level network errors into a more descriptive exception
        raise RuntimeError(f"Failed to query Cron params from {url}: {exc}") from exc


# step:2 file: Query Cron module parameters
from typing import Dict, Any

def parse_cron_params(params_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract important fields from the raw Cron parameters response.

    Args:
        params_json (Dict[str, Any]): Raw JSON structure returned by `query_cron_params`.

    Returns:
        Dict[str, Any]: A simplified dictionary containing selected Cron parameters.

    Raises:
        KeyError: If the expected keys are not found.
    """
    try:
        params = params_json["params"]
        parsed = {
            "max_msg_length": int(params.get("max_msg_length", 0)),
            "min_period": int(params.get("min_period", 0)),
            "security_address": params.get("security_address", ""),
            "fee_currency": params.get("fee_currency", ""),
            "fee_amount": params.get("fee_amount", "")
        }
        return parsed
    except KeyError as exc:
        raise KeyError(f"Expected key not found while parsing Cron params: {exc}") from exc


# step:1 file: register_upgrade_handlers
# backend/search_docs.py
import requests
from typing import List, Dict


def search_cosmos_docs(query: str, limit: int = 10) -> List[Dict[str, str]]:
    '''
    Simple wrapper around the public ReadTheDocs search endpoint used by the
    Cosmos SDK documentation site.

    Args:
        query (str): search string, e.g. "SetUpgradeHandler example".
        limit (int): maximum number of results to return.

    Returns:
        list[dict]: Each dict contains 'title' and 'link' keys.

    Raises:
        RuntimeError: if the request fails or the endpoint is unreachable.
    '''
    base_url = 'https://evm.cosmos.network/search'
    try:
        resp = requests.get(base_url, params={'q': query}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f'Failed to search Cosmos docs: {e}') from e

    # The HTML response contains <a class=\"result-link\" href=\"URL\">Title</a>
    results = []
    for line in resp.text.splitlines():
        if 'class=\"result-link\"' in line:
            try:
                href_part = line.split('href=\"')[1]
                link = href_part.split('\"')[0]
                title = href_part.split('\">')[1].split('</a>')[0]
                results.append({'title': title.strip(), 'link': link.strip()})
                if len(results) >= limit:
                    break
            except IndexError:
                continue
    return results



# step:2 file: register_upgrade_handlers
/* app/upgrades.go */
package app

import (
    "context"

    upgradetypes "cosmossdk.io/x/upgrade/types"
    sdk "github.com/cosmos/cosmos-sdk/types"
    "github.com/cosmos/cosmos-sdk/types/module"
)

// Upgrade metadata
const (
    UpgradeName   = "v1.2.0"
    UpgradeHeight = 1234567 // replace with governance-defined height
)

// RegisterUpgradeHandlers wires the upgrade handler into the binary.
func (app *App) RegisterUpgradeHandlers() {
    app.UpgradeKeeper.SetUpgradeHandler(
        UpgradeName,
        func(ctx context.Context, plan upgradetypes.Plan, fromVM module.VersionMap) (module.VersionMap, error) {
            sdkCtx := sdk.UnwrapSDKContext(ctx)
            sdkCtx.Logger().Info("Running upgrade handler", "plan", plan.Name)

            // Run in-place module migrations
            newVM, err := app.ModuleManager.RunMigrations(ctx, app.configurator, fromVM)
            if err != nil {
                return nil, err
            }

            // --- add any custom migration logic below ---

            sdkCtx.Logger().Info("Upgrade handler completed", "plan", plan.Name)
            return newVM, nil
        },
    )
}



# step:3 file: register_upgrade_handlers
/* app/app.go */

import (
    upgradetypes "cosmossdk.io/x/upgrade/types"
)

/*
inside NewApp (or your app constructor) add the store loader so the daemon
knows when to apply the migrations.
*/

if err := app.UpgradeKeeper.SetStoreLoader(
    upgradetypes.UpgradeStoreLoader(UpgradeHeight, &upgradetypes.Plan{Name: UpgradeName}),
); err != nil {
    panic(err)
}



# step:4 file: register_upgrade_handlers
# scripts/build.sh
#!/usr/bin/env bash
set -euo pipefail

echo "Tidying go.mod..."
go mod tidy

echo "Compiling application..."
go build -o build/mychaind ./cmd/mychaind

echo "✅ Binary ready at build/mychaind"



# step:5 file: register_upgrade_handlers
# scripts/simulate_upgrade.sh
#!/usr/bin/env bash
set -euo pipefail

BINARY=${BINARY:-mychaind}
CHAIN_ID=upgrade-sim
UPGRADE_HEIGHT=${UPGRADE_HEIGHT:-1234567}

# 0. Cleanup
rm -rf ~/.${BINARY}

# 1. Initialise a single-node chain
$BINARY init local --chain-id $CHAIN_ID

# 2. Create a local key
echo -e "12345678\n12345678\n" | $BINARY keys add validator --keyring-backend test

# 3. Allocate genesis funds
$BINARY add-genesis-account $($BINARY keys show validator -a --keyring-backend test) 100000000stake

# 4. Generate and collect a gentx so the node starts with a validator
$BINARY gentx validator 100000000stake --keyring-backend test --chain-id $CHAIN_ID
$BINARY collect-gentxs

echo "Starting node… (logs will be saved to node.log)"
$BINARY start --minimum-gas-prices 0stake --pruning=nothing --log_level info 2>&1 | tee node.log &
NODE_PID=$!

# 5. Wait until the chain reaches the upgrade height
echo "Waiting until block height >= $UPGRADE_HEIGHT"
while true; do
    HEIGHT=$(curl -s localhost:26657/status | jq -r .result.sync_info.latest_block_height)
    echo "Current height: $HEIGHT"
    if [ "$HEIGHT" -ge "$UPGRADE_HEIGHT" ]; then
        echo "Reached target height, shutting down…"
        break
    fi
    sleep 5
done

kill $NODE_PID
echo "Node stopped. Check node.log for \"Upgrade handler completed\" to confirm successful migration."



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


# step:3 file: Compile and deploy the Neutron example contract to the local CosmoPark testnet
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgStoreCode


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


# step:4 file: Compile and deploy the Neutron example contract to the local CosmoPark testnet
from cosmpy.aerial.client import LedgerClient


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


# step:5 file: Compile and deploy the Neutron example contract to the local CosmoPark testnet
def parse_code_id_from_receipt(tx_response) -> int:
    """Search TxResponse logs for the `store_code` event and return its `code_id`."""
    logs = tx_response.logs if hasattr(tx_response, 'logs') else tx_response['logs']
    for event in logs[0]['events']:
        if event['type'] == 'store_code':
            for attr in event['attributes']:
                if attr['key'] in ('code_id', 'codeID'):
                    return int(attr['value'])
    raise ValueError('code_id not found in transaction logs.')


# step:6 file: Compile and deploy the Neutron example contract to the local CosmoPark testnet
import json
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgInstantiateContract


def construct_tx_wasm_instantiate(code_id: int, init_msg: dict, label: str, wallet, chain_id: str, admin: str | None = None, gas: int = 500_000, fee: int = 150_000):
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


# step:7 file: Compile and deploy the Neutron example contract to the local CosmoPark testnet
def broadcast_instantiate_tx(instantiate_tx, wallet, client):
    """Helper that re-uses `sign_and_broadcast_tx` for the instantiate step."""
    return sign_and_broadcast_tx(instantiate_tx, wallet, client)


# step:8 file: Compile and deploy the Neutron example contract to the local CosmoPark testnet
def parse_contract_address_from_receipt(tx_response) -> str:
    """Fetch `_contract_address` from the instantiate event."""
    logs = tx_response.logs if hasattr(tx_response, 'logs') else tx_response['logs']
    for event in logs[0]['events']:
        if event['type'] == 'instantiate':
            for attr in event['attributes']:
                if attr['key'] == '_contract_address':
                    return attr['value']
    raise ValueError('Contract address not found in instantiate logs.')


# step:9 file: Compile and deploy the Neutron example contract to the local CosmoPark testnet
from cosmpy.aerial.client import LedgerClient


def query_contract_state(client: LedgerClient, contract_address: str, query_msg: dict):
    """Query the contract’s state using a custom query message."""
    try:
        return client.wasm_query(contract_address, query_msg)
    except Exception as err:
        raise RuntimeError(f'Contract query failed: {err}') from err


# step:2 file: create_a_new_key_(account)_secured_with_a_passphrase
import json
import subprocess
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get('/keys/validate')
def validate_key_name(name: str):
    """Return `{ available: bool }` indicating whether the key name is free."""
    try:
        cmd = [
            'simd',
            'keys',
            'list',
            '--keyring-backend', 'file',
            '--output', 'json'
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=proc.stderr.strip())

        existing = json.loads(proc.stdout) if proc.stdout else []
        existing_names = {entry.get('name') for entry in existing}
        return {"available": name not in existing_names}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:3 file: create_a_new_key_(account)_secured_with_a_passphrase
import json
import subprocess
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post('/keys/add')
def add_key(payload: dict):
    """Create a key and return `{ address, mnemonic }`. Payload must contain `name` and `passphrase`."""
    name = payload.get('name')
    passphrase = payload.get('passphrase')

    if not name or not passphrase:
        raise HTTPException(status_code=400, detail='Both `name` and `passphrase` are required.')

    try:
        cmd = [
            'simd',
            'keys', 'add', name,
            '--keyring-backend', 'file',
            '--output', 'json'
        ]

        # Feed passphrase twice via STDIN (create + confirm)
        proc = subprocess.run(
            cmd,
            input=f"{passphrase}\n{passphrase}\n",
            capture_output=True,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=proc.stderr.strip())

        key_info = json.loads(proc.stdout)
        return {
            'address': key_info.get('address'),
            'mnemonic': key_info.get('mnemonic')
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:2 file: provide_paired_liquidity_of_1_wbtc_and_60,000_usdc_to_the_wbtc_usdc_supervault
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI()

# --- Constants -------------------------------------------------------------
REST_ENDPOINT = "https://rest.neutron.org"  # Replace with a trusted REST endpoint
WBTC_DENOM   = "ibc/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # ← real IBC denom for WBTC
USDC_DENOM   = "uusdc"  # ← real denom for native USDC on Neutron

# --- Helpers ---------------------------------------------------------------
async def _fetch_balance(address: str, denom: str) -> int:
    """Query /cosmos/bank/v1beta1/balances/{address}/{denom}"""
    url = f"{REST_ENDPOINT}/cosmos/bank/v1beta1/balances/{address}/{denom}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Bank API error")
    amount = int(resp.json().get("balance", {}).get("amount", 0))
    return amount

# --- Route -----------------------------------------------------------------
@app.get("/api/check-balance")
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


# step:3 file: provide_paired_liquidity_of_1_wbtc_and_60,000_usdc_to_the_wbtc_usdc_supervault
import os
from fastapi import FastAPI

app = FastAPI()

SUPER_VAULT_CONTRACT_ADDRESS = os.getenv("SUPER_VAULT_CONTRACT_ADDRESS", "neutron1vaultxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
WBTC_DENOM = os.getenv("WBTC_DENOM", "ibc/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
USDC_DENOM = os.getenv("USDC_DENOM", "uusdc")

@app.get("/api/supervault-details")
async def query_supervault_details():
    return {
        "contract_address": SUPER_VAULT_CONTRACT_ADDRESS,
        "tokens": [
            {"denom": WBTC_DENOM, "symbol": "WBTC"},
            {"denom": USDC_DENOM, "symbol": "USDC"}
        ]
    }


# step:4 file: provide_paired_liquidity_of_1_wbtc_and_60,000_usdc_to_the_wbtc_usdc_supervault
import os, base64
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction

app = FastAPI()

# -------- Configuration ----------------------------------------------------
RPC_ENDPOINT = os.getenv("NEUTRON_RPC", "https://rpc.neutron.org")
CHAIN_ID     = os.getenv("NEUTRON_CHAIN_ID", "neutron-1")
network_cfg  = NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT)
ledger       = LedgerClient(network_cfg)
SUPER_VAULT_CONTRACT = os.getenv("SUPER_VAULT_CONTRACT_ADDRESS", "neutron1vaultxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
WBTC_DENOM = os.getenv("WBTC_DENOM", "ibc/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
USDC_DENOM = os.getenv("USDC_DENOM", "uusdc")

# -------- Request model ----------------------------------------------------
class ConstructTxRequest(BaseModel):
    address: str           # Liquidity provider address (sender)
    wbtc_amount: int       # 1 WBTC  (use the correct micro-denom units)
    usdc_amount: int       # 60 000 USDC in micro-denom units

# -------- Route ------------------------------------------------------------
@app.post("/api/construct-deposit-tx")
async def construct_supervault_deposit_tx(req: ConstructTxRequest = Body(...)):
    # 1. Compose execute message expected by Supervault contract
    exec_msg = {
        "deposit": {
            "assets": [
                {
                    "info": {"native_token": {"denom": WBTC_DENOM}},
                    "amount": str(req.wbtc_amount)
                },
                {
                    "info": {"native_token": {"denom": USDC_DENOM}},
                    "amount": str(req.usdc_amount)
                }
            ]
        }
    }

    # 2. Create Tx object
    tx = Transaction()
    tx.add_message(
        ledger.execute_contract(
            sender=req.address,
            contract_address=SUPER_VAULT_CONTRACT,
            msg=exec_msg,
            funds=[]  # Contract pulls tokens from user’s balance; no explicit Coin[] required
        )
    )

    # 3. Gas estimate (rough – add a safety buffer client-side if needed)
    gas_estimate = ledger.estimate_gas(tx)
    tx.set_gas(gas_estimate)

    # 4. Return unsigned tx bytes for the next step
    unsigned_bytes = tx.serialize()
    return {
        "tx_base64": base64.b64encode(unsigned_bytes).decode(),
        "gas_estimate": gas_estimate
    }


# step:5 file: provide_paired_liquidity_of_1_wbtc_and_60,000_usdc_to_the_wbtc_usdc_supervault
import os, base64
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction

app = FastAPI()

# --- Ledger ----------------------------------------------------------------
RPC_ENDPOINT = os.getenv("NEUTRON_RPC", "https://rpc.neutron.org")
CHAIN_ID     = os.getenv("NEUTRON_CHAIN_ID", "neutron-1")
ledger       = LedgerClient(NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT))

# --- Security warning ------------------------------------------------------
# Keeping private keys on a server is NOT recommended for production.
# Instead, sign on the client or use an HSM/KMS solution.
MNEMONIC = os.getenv("LP_MNEMONIC")
if MNEMONIC is None:
    raise RuntimeError("LP_MNEMONIC environment variable must be set for backend signing demo.")

wallet = LocalWallet.from_mnemonic(MNEMONIC)

# --- Request model ---------------------------------------------------------
class SignBroadcastRequest(BaseModel):
    tx_base64: str

# --- Route -----------------------------------------------------------------
@app.post("/api/sign-and-broadcast")
async def sign_and_broadcast_tx(req: SignBroadcastRequest = Body(...)):
    try:
        # 1. Deserialize unsigned transaction
        unsigned_bytes = base64.b64decode(req.tx_base64)
        tx = Transaction.deserialize(unsigned_bytes)

        # 2. Sign with backend wallet
        tx.sign(wallet)

        # 3. Broadcast (waitUntil="sync")
        result = ledger.broadcast_block(tx)

        if result.is_tx_error():
            raise HTTPException(status_code=400, detail=f"Tx failed: {result.raw_log}")

        return {"txhash": result.tx_hash}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:4 file: opt_in_to_partner_airdrops_for_my_vault_deposits
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction
import os

app = FastAPI()

# --- Chain / network configuration ---
NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url="https://rpc-kralum.neutron-1.neutron.org:443",  # Public RPC endpoint
    fee_minimum_gas_price=0.03,
    fee_denomination="untrn",
)

# --- Pydantic request model ---
class ExecuteRequest(BaseModel):
    mnemonic: str                # ⚠️  For demo only; never store on server in prod
    contract_address: str        # Vault contract address
    partner_id: str = "all"      # Field for the execute msg
    gas_limit: int = 200_000     # Optional user-tuneable gas limit
    fee_denom: str = "untrn"     # Fee denom, default untrn

@app.post("/api/execute/opt_in_airdrops")
async def execute_opt_in_airdrops(req: ExecuteRequest):
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
        tx.with_chain_id(NETWORK.chain_id)
        tx.with_fee(req.fee_denom)

        # Sign
        signed_tx = tx.sign(wallet)

        # Broadcast
        client = LedgerClient(NETWORK)
        resp = client.broadcast_tx(signed_tx)

        if resp.is_error():
            raise HTTPException(status_code=400, detail=f"Broadcast failed: {resp.raw_log}")

        return {"txhash": resp.tx_hash}

    except Exception as e:
        # Surface any unexpected error
        raise HTTPException(status_code=500, detail=str(e))


# step:1 file: estimate_gas_for_a_contract_call_with_cast_estimate
# backend/check_foundry.py

import subprocess
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/api/check_foundry")
async def check_foundry():
    """
    Returns `{ installed: bool, version: str }`.
    Raises HTTP 500 if the `cast` binary is missing or mis-configured.
    """
    try:
        # `cast --version` is a quick way to test availability
        result = subprocess.run(["cast", "--version"], capture_output=True, text=True, check=True)
        return {"installed": True, "version": result.stdout.strip()}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Foundry not installed (`cast` binary not found).")
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"Error running cast: {err.stderr.strip()}")


# step:3 file: estimate_gas_for_a_contract_call_with_cast_estimate
# backend/estimate_gas.py

import subprocess
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

class EstimateRequest(BaseModel):
    rpc_url: str = Field(..., example="https://rpc.my-evm-chain.org")
    contract_address: str = Field(..., example="0x1234…abcd")
    function_signature: str = Field(..., example="transfer(address,uint256)")
    args: List[str] = Field(default_factory=list, example=["0xabc…", "1000000000000000000"])

@router.post("/api/estimate_gas")
async def estimate_gas(body: EstimateRequest):
    """Executes `cast estimate` and returns `{ gas_units: int }`."""
    cmd = [
        "cast",
        "estimate",
        "--rpc-url",
        body.rpc_url,
        body.contract_address,
        body.function_signature,
    ] + body.args

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        gas_units_str = completed.stdout.strip()
        gas_units = int(gas_units_str, 0)  # auto-detect base (0x…, decimal, etc.)
        return {"gas_units": gas_units}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="`cast` binary not found on server.")
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"cast error: {err.stderr.strip()}")
    except ValueError:
        raise HTTPException(status_code=500, detail="Unexpected output from cast estimate.")


# step:2 file: retrieve_full_transaction_details_from_a_cosmos-evm_chain_using_an_ethereum_transaction_hash
# rpc_client.py
import os
import requests
from typing import Any, List

class JSONRPCClient:
    """Minimal HTTP JSON-RPC client compatible with an Ethereum node (geth, erigon, etc.)."""

    def __init__(self, endpoint: str | None = None):
        # Allow the endpoint to be configured via env-var or default to localhost
        self.endpoint = endpoint or os.getenv("ETH_RPC_ENDPOINT", "http://localhost:8545")

    def _post(self, method: str, params: List[Any] | None = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or []
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectionError(f"Unable to reach JSON-RPC endpoint {self.endpoint}: {exc}") from exc

        data = resp.json()
        if "error" in data and data["error"]:
            raise RuntimeError(f"JSON-RPC error: {data['error']}")
        return data.get("result")

    # Convenience wrappers
    def get_transaction_by_hash(self, tx_hash: str) -> Any:
        return self._post("eth_getTransactionByHash", [tx_hash])

    def get_transaction_receipt(self, tx_hash: str) -> Any:
        return self._post("eth_getTransactionReceipt", [tx_hash])


# step:3 file: retrieve_full_transaction_details_from_a_cosmos-evm_chain_using_an_ethereum_transaction_hash
# routes/tx.py
from fastapi import APIRouter, HTTPException, Query
from rpc_client import JSONRPCClient

router = APIRouter()
client = JSONRPCClient()  # Uses ENV var ETH_RPC_ENDPOINT or default

@router.get("/api/transaction")
async def get_transaction(tx_hash: str = Query(..., description="0x-prefixed transaction hash")):
    try:
        tx = client.get_transaction_by_hash(tx_hash)
        if tx is None:
            # Ethereum returns null when the hash is unknown
            raise HTTPException(status_code=404, detail="Transaction not found")
        return {"transaction": tx}
    except (ConnectionError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# step:4 file: retrieve_full_transaction_details_from_a_cosmos-evm_chain_using_an_ethereum_transaction_hash
# routes/receipt.py
from fastapi import APIRouter, HTTPException, Query
from rpc_client import JSONRPCClient

router = APIRouter()
client = JSONRPCClient()

@router.get("/api/receipt")
async def get_receipt(tx_hash: str = Query(..., description="0x-prefixed transaction hash")):
    try:
        receipt = client.get_transaction_receipt(tx_hash)
        if receipt is None:
            raise HTTPException(status_code=404, detail="Receipt unavailable – the transaction may be pending or unknown")
        return {"receipt": receipt}
    except (ConnectionError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# step:1 file: initialize_the_gnu_pass_password_store_for_gaia_with_a_gpg_key
from fastapi import FastAPI, HTTPException
import subprocess
from typing import List

# FastAPI application instance for this micro-service
app = FastAPI()

def list_gpg_keys() -> List[str]:
    """Return a list of available GPG key IDs that exist on the host.

    This helper runs `gpg --list-keys --with-colons` and parses the
    machine-readable output (colon-separated fields).  The 5th column holds
    the long key ID.
    """
    try:
        completed = subprocess.run(
            ["gpg", "--list-keys", "--with-colons"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        # The `gpg` binary is not present on the system.
        raise RuntimeError("`gpg` command is missing. Please install GnuPG and try again.")
    except subprocess.CalledProcessError as exc:
        # The command executed but returned a non-zero exit status.
        raise RuntimeError(f"Failed to list GPG keys: {exc.stderr.strip()}")

    # Parse keys from command output.
    key_ids = [
        row.split(":")[4]
        for row in completed.stdout.splitlines()
        if row.startswith("pub") and len(row.split(":")) > 4
    ]
    return key_ids

@app.get("/api/gpg/keys")
async def get_gpg_keys():
    """HTTP endpoint: GET /api/gpg/keys
    Returns a JSON object with the available GPG key IDs so the frontend can
    let the user choose which one to use for `pass init`.
    """
    try:
        keys = list_gpg_keys()
    except RuntimeError as e:
        # Convert Python errors to proper HTTP errors for the client.
        raise HTTPException(status_code=500, detail=str(e))
    return {"keys": keys}


# step:2 file: initialize_the_gnu_pass_password_store_for_gaia_with_a_gpg_key
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess

app = FastAPI()

class InitRequest(BaseModel):
    key_id: str  # The GPG key to be used for encrypting the password store

def init_pass_store(key_id: str) -> None:
    """Initialise the `pass` password store with the supplied `key_id`."""
    try:
        subprocess.run(
            ["pass", "init", key_id],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("`pass` CLI not found. Install 'pass' and ensure it is on $PATH.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`pass init` failed: {exc.stderr.strip()}")

@app.post("/api/pass/init")
async def api_init_pass_store(req: InitRequest):
    """HTTP endpoint: POST /api/pass/init {"key_id": "<GPG_KEY_ID>"}.
    It runs `pass init` on the server and returns status back to the caller.
    """
    try:
        init_pass_store(req.key_id)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "initialised", "key_id": req.key_id}


# step:3 file: initialize_the_gnu_pass_password_store_for_gaia_with_a_gpg_key
from fastapi import FastAPI, HTTPException
import subprocess
from typing import List

app = FastAPI()

def verify_pass_store() -> List[str]:
    """Return list of entries in the password store (empty list if none).

    Returns an empty list when the store is freshly initialised.
    """
    try:
        completed = subprocess.run(
            ["pass", "ls"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("`pass` CLI not found. Install 'pass' and ensure it is on $PATH.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`pass ls` failed: {exc.stderr.strip()}")

    # Clean up and flatten the textual tree output.
    entries = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return entries

@app.get("/api/pass/verify")
async def api_verify_pass_store():
    """HTTP endpoint: GET /api/pass/verify.
    Returns the current list of passwords in the store. If the list is empty
    the store is valid but has no secrets yet.
    """
    try:
        entries = verify_pass_store()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"entries": entries}


# step:1 file: compile_a_cosmwasm_contract_for_arm64_using_the_rust-optimizer-arm64_image
#!/usr/bin/env python3
"""
pull_optimizer.py
Utility script to pull CosmWasm Rust optimizer Docker image.
"""

import subprocess
import sys

def pull_optimizer_image(image="cosmwasm/rust-optimizer-arm64:latest"):
    """Pulls the CosmWasm Rust optimizer Docker image."""
    print(f"Pulling Docker image: {image}")
    try:
        result = subprocess.run(["docker", "pull", image], check=True, capture_output=True, text=True)
        print(result.stdout)
        print("✅  Docker image pulled successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print("❌  Failed to pull docker image:", e.stderr, file=sys.stderr)
        return False

if __name__ == "__main__":
    success = pull_optimizer_image()
    if not success:
        sys.exit(1)



# step:2 file: compile_a_cosmwasm_contract_for_arm64_using_the_rust-optimizer-arm64_image
#!/usr/bin/env python3
"""
run_optimizer.py
Runs CosmWasm's Rust optimizer Docker container to build optimized .wasm binaries.
"""

import subprocess
import sys
from pathlib import Path

def run_rust_optimizer(contract_path: str = "."):
    """Runs the CosmWasm Rust optimizer docker container mounting the given contract directory."""
    contract_path = Path(contract_path).resolve()

    if not contract_path.exists():
        raise FileNotFoundError(f"Contract path {contract_path} does not exist")

    docker_cmd = ["docker", "run", "--rm", "-v", f"{contract_path}:/code:Z", "cosmwasm/rust-optimizer-arm64:latest"]

    print("Running Docker command:", " ".join(docker_cmd))
    try:
        subprocess.run(docker_cmd, check=True)
        artifacts_dir = contract_path / "artifacts"
        if artifacts_dir.exists():
            print(f"✅  Artifacts generated successfully in {artifacts_dir}")
        else:
            print("⚠️  Artifacts directory not found after optimizer run.")
        return True
    except subprocess.CalledProcessError as e:
        print("❌  Error while running rust optimizer:", e, file=sys.stderr)
        return False

if __name__ == "__main__":
    run_rust_optimizer(sys.argv[1] if len(sys.argv) > 1 else ".")



# step:3 file: compile_a_cosmwasm_contract_for_arm64_using_the_rust-optimizer-arm64_image
#!/usr/bin/env python3
"""
verify_artifacts.py
Validates that the optimizer produced .wasm and checksum files in the artifacts directory.
"""

from pathlib import Path
import sys
import json

def verify_artifacts(artifacts_path: str = "./artifacts"):
    """Verifies the presence of .wasm and checksum files in `artifacts/`."""
    apath = Path(artifacts_path).resolve()

    if not apath.exists() or not apath.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found at {apath}")

    wasm_files = list(apath.glob("*.wasm"))
    checksum_files = [f for f in apath.glob("*.wasm.*")] + list(apath.glob("*.sha256")) + list(apath.glob("*.checksums"))

    result = {"wasm_files": [f.name for f in wasm_files], "checksum_files": [f.name for f in checksum_files]}

    if wasm_files:
        print("✅  Found .wasm files:", ", ".join(result["wasm_files"]))
    else:
        print("❌  No .wasm files found.", file=sys.stderr)

    if checksum_files:
        print("✅  Found checksum files:", ", ".join(result["checksum_files"]))
    else:
        print("⚠️  No checksum files found.")

    return result

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "./artifacts"
    summary = verify_artifacts(directory)
    print(json.dumps(summary, indent=2))



# step:1 file: Remove the existing schedule named protocol_update
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.exceptions import CosmPyException


def get_dao_authority_address(rpc_endpoint: str, chain_id: str, dao_contract: str) -> str:
    """Return the address that has Main-DAO authority.

    Args:
        rpc_endpoint (str): Full RPC URL, e.g. "https://rpc-kralum.neutron.org:443".
        chain_id (str): The on-chain ID, e.g. "neutron-1".
        dao_contract (str): Bech-32 address of the DAO WASM contract.

    Returns:
        str: Address with delete-schedule permissions.
    """
    try:
        cfg = NetworkConfig(chain_id=chain_id, url=rpc_endpoint, fee_minimum_gas_price=0)
        client = LedgerClient(cfg)

        # The DAO contract is expected to support `{ "authority": {} }` query.
        query_msg = {"authority": {}}
        res = client.wasm.contract_query(dao_contract, query_msg)
        authority_addr = res.get("authority")
        if not authority_addr:
            raise ValueError("DAO contract did not return an authority address.")
        return authority_addr
    except (CosmPyException, ValueError) as err:
        raise RuntimeError(f"Unable to fetch DAO authority address: {err}")



# step:2 file: Remove the existing schedule named protocol_update
import json


def build_msg_delete_schedule(authority: str, schedule_name: str = "protocol_update") -> dict:
    """Return an amino/JSON-encoded MsgDeleteSchedule body ready for packing.

    Args:
        authority (str): Address returned from Step 1.
        schedule_name (str): Name of the schedule to delete.

    Returns:
        dict: Properly-typed MsgDeleteSchedule for inclusion in gov proposal.
    """
    if not authority:
        raise ValueError("Authority address is required to build MsgDeleteSchedule.")

    msg = {
        "@type": "/neutron.admin.MsgDeleteSchedule",
        "authority_address": authority,
        "name": schedule_name
    }
    # quick validation
    json.dumps(msg)  # will raise if non-serialisable
    return msg



# step:3 file: Remove the existing schedule named protocol_update
from datetime import datetime
import json


def package_into_gov_proposal(authority: str, delete_msg: dict, deposit_amount: str = "10000000", denom: str = "untrn") -> dict:
    """Embed the delete-schedule message into a MsgSubmitProposal.

    Args:
        authority (str): DAO authority address (will be listed as proposer).
        delete_msg (dict): Message from Step 2.
        deposit_amount (str): Minimum deposit in micro-denom (10 NTRN default).
        denom (str): Denomination for deposit.

    Returns:
        dict: MsgSubmitProposal ready for signing.
    """
    title = "Remove obsolete cron schedule: protocol_update"
    description = (
        "This proposal deletes the `protocol_update` cron schedule, which is no longer "
        "needed after the successful upgrade executed on " + datetime.utcnow().strftime("%Y-%m-%d") + "."
    )

    proposal_msg = {
        "@type": "/cosmos.gov.v1beta1.MsgSubmitProposal",
        "proposer": authority,
        "initial_deposit": [{"denom": denom, "amount": deposit_amount}],
        "content": {
            "@type": "/cosmos.gov.v1beta1.TextProposal",
            "title": title,
            "description": description
        },
        "messages": [delete_msg]  # custom message list supported by Neutron-gov
    }

    # Ensure JSON validity
    json.dumps(proposal_msg)
    return proposal_msg



# step:4 file: Remove the existing schedule named protocol_update
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.exceptions import CosmPyException


def sign_and_broadcast_tx(
    rpc_endpoint: str,
    chain_id: str,
    authority_mnemonic: str,
    gov_proposal_msg: dict,
    gas: int = 400000,
    fee_denom: str = "untrn",
    gas_price: float = 0.025,
) -> str:
    """Sign and submit the governance proposal to chain.

    Args:
        rpc_endpoint (str): Full RPC endpoint.
        chain_id (str): Chain identifier.
        authority_mnemonic (str): 24-word mnemonic for the authority address.
        gov_proposal_msg (dict): Message prepared in Step 3.
        gas (int): Gas limit.
        fee_denom (str): Fee denomination.
        gas_price (float): Gas price in fee_denom.

    Returns:
        str: Transaction hash if broadcast is successful.
    """
    try:
        cfg = NetworkConfig(chain_id=chain_id, url=rpc_endpoint, fee_minimum_gas_price=gas_price)
        client = LedgerClient(cfg)

        # Build wallet
        wallet = LocalWallet.create_from_mnemonic(authority_mnemonic)
        acc_num, acc_seq = client.query_account_number_and_sequence(wallet.address())

        # Build tx
        tx = Transaction()
        tx.add_message(gov_proposal_msg)
        tx.with_gas(gas)
        tx.with_chain_id(chain_id)
        tx.with_fee(int(gas * gas_price), fee_denom)
        tx.with_memo("Delete schedule governance proposal")
        tx.with_sequence(acc_seq)
        tx.with_account_number(acc_num)
        tx.sign(wallet)

        # Broadcast
        tx_response = client.broadcast_tx_block(tx)
        if tx_response.is_err():
            raise RuntimeError(f"Tx failed: {tx_response.raw_log}")
        return tx_response.tx_hash
    except (CosmPyException, RuntimeError) as err:
        raise RuntimeError(f"Unable to broadcast governance proposal: {err}")



# step:1 file: set_mempool_max-txs_to_-1_in_app.toml
import os
import toml

# ---------------------------
# Step 1 – open_config_file
# ---------------------------

def open_config_file(daemon_name: str):
    """Return (path, config_dict) for $HOME/.<daemon>/config/app.toml.

    Parameters
    ----------
    daemon_name : str
        The binary/service name (e.g. 'gaiad', 'junod').
    """
    home_dir = os.path.expanduser('~')
    path = os.path.join(home_dir, f'.{daemon_name}', 'config', 'app.toml')

    if not os.path.isfile(path):
        raise FileNotFoundError(f'Config file not found at {path}')

    try:
        with open(path, 'r', encoding='utf-8') as fh:
            raw_toml = fh.read()
        config_dict = toml.loads(raw_toml)  # `pip install toml` on Python < 3.11
        return path, config_dict
    except Exception as err:
        # Surface additional context while preserving the original traceback
        raise RuntimeError(f'Unable to open or parse {path}: {err}') from err


# step:2 file: set_mempool_max-txs_to_-1_in_app.toml
import copy

# ---------------------------
# Step 2 – update_mempool_max_txs
# ---------------------------

def update_mempool_max_txs(config_dict: dict, new_value: int = -1):
    """Return an updated copy of the TOML config with mempool.max_txs replaced."""
    updated = copy.deepcopy(config_dict)
    mempool_cfg = updated.get('mempool', {})
    mempool_cfg['max_txs'] = new_value
    updated['mempool'] = mempool_cfg
    return updated


# step:3 file: set_mempool_max-txs_to_-1_in_app.toml
import os
import toml

# ---------------------------
# Step 3 – save_config_file
# ---------------------------

def save_config_file(path: str, config_dict: dict):
    """Write the provided config_dict back to the original file atomically."""
    try:
        toml_str = toml.dumps(config_dict)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as fh:
            fh.write(toml_str)
        os.replace(tmp_path, path)  # atomic move on POSIX
    except Exception as err:
        raise RuntimeError(f'Failed to write config file {path}: {err}') from err


# step:4 file: set_mempool_max-txs_to_-1_in_app.toml
import subprocess
import time

# ---------------------------
# Step 4 – restart_node_service
# ---------------------------

def restart_node_service(daemon_name: str, wait_seconds: int = 60):
    """Restart <daemon> with systemctl and wait until it's active.

    This function assumes the current user can invoke `sudo systemctl` without
    interactive password prompts (e.g. via sudoers configuration).
    """
    service_name = daemon_name

    # Restart the service
    try:
        subprocess.run(['sudo', 'systemctl', 'restart', service_name], check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f'Failed to restart service {service_name}: {err}') from err

    # Poll until the service is active or timeout is reached
    start_time = time.time()
    while time.time() - start_time < wait_seconds:
        result = subprocess.run(['systemctl', 'is-active', service_name], capture_output=True, text=True)
        if result.stdout.strip() == 'active':
            print(f'Service {service_name} is active again.')
            return
        time.sleep(2)

    raise TimeoutError(f'Service {service_name} did not become active within {wait_seconds} seconds.')


# step:2 file: estimate_gas_fees_for_eip-1559_transactions
from flask import Flask, request, jsonify
import subprocess, json, os

app = Flask(__name__)

@app.route("/api/estimate_gas", methods=["POST"])
def estimate_gas():
    """Estimate gas limit and dynamic fees using Foundry's cast CLI."""
    tx = request.get_json(force=True)

    # ---- Validation ----
    required = ["from", "to", "value", "data", "chainId"]
    missing = [f for f in required if f not in tx]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 422

    # ---- Build cast command ----
    cmd = [
        "cast", "estimate",
        "--from", tx["from"],
        "--to", tx["to"],
        "--value", tx["value"],
        "--data", tx["data"],
        "--max-fee-per-gas", "auto",
        "--max-priority-fee-per-gas", "auto",
        "--chain-id", str(tx["chainId"]),
        "--json"
    ]

    try:
        # ETH_RPC_URL must be set in the environment for cast to work
        result = subprocess.check_output(cmd, env=os.environ, text=True)
        payload = json.loads(result)
        return jsonify({
            "estimatedGas": int(payload["gas"]),
            "maxFeePerGas": int(payload["maxFeePerGas"]),
            "maxPriorityFeePerGas": int(payload["maxPriorityFeePerGas"])
        })
    except subprocess.CalledProcessError as err:
        return jsonify({"error": err.output}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


# step:3 file: estimate_gas_fees_for_eip-1559_transactions
from flask import Flask, jsonify
import subprocess, json

app = Flask(__name__)

@app.route("/api/gas_price", methods=["GET"])
def gas_price():
    """Return baseFee and suggested priority fee from the last 5 blocks."""
    cmd = ["cast", "gas-price", "--blocks", "5", "--json"]
    try:
        raw = subprocess.check_output(cmd, text=True)
        info = json.loads(raw)
        return jsonify({
            "baseFee": int(info["baseFeePerGas"]),
            "priorityFee": int(info["maxPriorityFeePerGas"])
        })
    except subprocess.CalledProcessError as err:
        return jsonify({"error": err.output}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


# step:4 file: estimate_gas_fees_for_eip-1559_transactions
from flask import Flask, request, jsonify

app = Flask(__name__)

def _calc_total_fee(estimated_gas: int, base_fee: int, priority_fee: int, max_fee_per_gas: int):
    """Compute effectiveGasPrice and totalFee as per EIP-1559."""
    effective_gas_price = min(base_fee + priority_fee, max_fee_per_gas)
    total_fee = estimated_gas * effective_gas_price
    return {
        "effectiveGasPrice": effective_gas_price,
        "totalFee": total_fee
    }

@app.route("/api/calculate_total_fee", methods=["POST"])
def calculate_total_fee():
    data = request.get_json(force=True)
    try:
        result = _calc_total_fee(
            estimated_gas = int(data["estimatedGas"]),
            base_fee       = int(data["baseFee"]),
            priority_fee   = int(data["priorityFee"]),
            max_fee_per_gas= int(data["maxFeePerGas"])
        )
        return jsonify(result)
    except (KeyError, ValueError) as err:
        return jsonify({"error": str(err)}), 422

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


# step:1 file: set_the_evm_call_timeout_(rpcevm_timeout)_to_5_seconds
import subprocess
from typing import Dict


def stop_evmd_node() -> Dict[str, str]:
    """Stop the running `evmd` instance."""
    try:
        # Attempt a graceful shutdown via systemd
        result = subprocess.run(
            ["systemctl", "stop", "evmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        # Fallback for non-systemd or failure cases
        if result.returncode != 0:
            subprocess.run(["pkill", "-f", "evmd"], check=False)

        return {"status": "evmd stopped"}

    except Exception as e:
        # Capture and return any error information
        return {"error": str(e)}


# step:2 file: set_the_evm_call_timeout_(rpcevm_timeout)_to_5_seconds
import pathlib
import toml

APP_TOML_PATH = pathlib.Path.home() / ".evmd" / "config" / "app.toml"


def load_app_toml() -> dict:
    """Read ~/.evmd/config/app.toml into a Python dictionary."""
    if not APP_TOML_PATH.exists():
        raise FileNotFoundError(f"{APP_TOML_PATH} does not exist.")
    return toml.load(APP_TOML_PATH)


# step:3 file: set_the_evm_call_timeout_(rpcevm_timeout)_to_5_seconds
def set_json_rpc_timeout(cfg: dict, timeout: str = "5s") -> dict:
    """Mutate the `cfg` dict to enforce a 5-second JSON-RPC EVM timeout."""
    json_rpc_block = cfg.setdefault("json-rpc", {})

    # Prefer the modern key, fall back if not present
    if "evm-timeout" in json_rpc_block or "rpc-evm-timeout" not in json_rpc_block:
        json_rpc_block["evm-timeout"] = timeout
    else:
        json_rpc_block["rpc-evm-timeout"] = timeout

    return cfg


# step:4 file: set_the_evm_call_timeout_(rpcevm_timeout)_to_5_seconds
import toml


def save_app_toml(cfg: dict) -> None:
    """Write the updated configuration back to ~/.evmd/config/app.toml."""
    with open(APP_TOML_PATH, "w", encoding="utf-8") as fp:
        toml.dump(cfg, fp)


# step:5 file: set_the_evm_call_timeout_(rpcevm_timeout)_to_5_seconds
def start_evmd_node() -> dict:
    """Start (or restart) the `evmd` service so it picks up the new configuration."""
    try:
        # Systemd path (preferred)
        subprocess.run(["systemctl", "start", "evmd"], check=True)
    except subprocess.CalledProcessError:
        # Fallback: spawn the process directly if systemd is unavailable
        subprocess.Popen(["evmd", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return {"status": "evmd started"}


# step:1 file: set_the_mutex_profile_sampling_rate_to_1_(enable_full_sampling)
import os
import base64


def get_rpc_auth_header():
    # Build Basic-Auth header from RPC_USER and RPC_PASS environment variables.
    username = os.getenv('RPC_USER')
    password = os.getenv('RPC_PASS')
    if not username or not password:
        raise EnvironmentError('RPC_USER or RPC_PASS is not set in environment variables.')

    token_bytes = f'{username}:{password}'.encode()
    token_b64 = base64.b64encode(token_bytes).decode()
    return {'Authorization': f'Basic {token_b64}'}



# step:2 file: set_the_mutex_profile_sampling_rate_to_1_(enable_full_sampling)
from flask import Blueprint, jsonify
import requests
import uuid
import os

# Helper from Step 1
from .auth import get_rpc_auth_header

debug_bp = Blueprint('debug', __name__)

# RPC endpoint (defaults to local geth but can be overridden)
RPC_URL = os.getenv('RPC_URL', 'http://localhost:8545')

@debug_bp.route('/api/debug/mutex/enable', methods=['POST'])
def enable_mutex_profiling():
    # Enable mutex contention sampling by calling debug_setMutexProfileFraction(1).
    headers = get_rpc_auth_header()
    headers['Content-Type'] = 'application/json'

    payload = {
        'jsonrpc': '2.0',
        'id': str(uuid.uuid4()),
        'method': 'debug_setMutexProfileFraction',
        'params': [1]
    }

    try:
        resp = requests.post(RPC_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if 'error' in data:
            return jsonify({'status': 'error', 'message': data['error']}), 500
        return jsonify({'status': 'success', 'result': data.get('result')})
    except requests.exceptions.RequestException as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500



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



# step:1 file: collect_all_gentxs_into_the_genesis_file
import os
import shutil
import glob
from typing import List

def gather_gentx_files(source_dirs: List[str], target_dir: str = "config/gentx") -> List[str]:
    """Gather all validator gentx JSON files into the chain's `config/gentx/` folder.

    Parameters
    ----------
    source_dirs : List[str]
        A list of directories where gentx files can be found.
    target_dir : str, optional
        Destination directory for the gentx files, by default "config/gentx".

    Returns
    -------
    List[str]
        Absolute paths of all gentx files that were copied.
    """
    # Ensure the destination exists.
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    copied: List[str] = []

    for src in source_dirs:
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Source directory '{src}' does not exist.")

        # Copy every .json file found in that directory.
        for file_path in glob.glob(os.path.join(src, "*.json")):
            destination = os.path.join(target_dir, os.path.basename(file_path))
            shutil.copy2(file_path, destination)
            copied.append(os.path.abspath(destination))

    if not copied:
        raise RuntimeError("No gentx JSON files were discovered in the supplied directories.")

    return copied

# Stand-alone usage example
if __name__ == "__main__":
    files_moved = gather_gentx_files(["./gentx_inputs"])
    print(f"Successfully gathered {len(files_moved)} gentx files → {os.path.abspath('config/gentx')}")


# step:2 file: collect_all_gentxs_into_the_genesis_file
import subprocess
import os
from pathlib import Path

def collect_gentxs(chain_binary: str = os.getenv("CHAIN_BINARY", "mychaind"), home: str = ".") -> str:
    """Execute `<chain_binary> collect-gentxs` and return the final genesis path.

    Parameters
    ----------
    chain_binary : str, optional
        The binary to execute, defaults to the `CHAIN_BINARY` env var or `mychaind`.
    home : str, optional
        The node's home directory (where the `config/` folder lives), defaults to `.`.

    Returns
    -------
    str
        Absolute path to the resulting genesis.json file.
    """
    cmd = [chain_binary, "collect-gentxs", "--home", home]
    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(
            "collect-gentxs failed!\n" +
            f"stdout:\n{result.stdout}\n" +
            f"stderr:\n{result.stderr}"
        )

    genesis_path = Path(home) / "config" / "genesis.json"
    if not genesis_path.is_file():
        raise FileNotFoundError("genesis.json was not created — please check the logs above.")

    return str(genesis_path.resolve())

# Stand-alone usage example
if __name__ == "__main__":
    genesis_file = collect_gentxs()
    print(f"Genesis generated at → {genesis_file}")


# step:3 file: collect_all_gentxs_into_the_genesis_file
import subprocess
import os

def validate_genesis(chain_binary: str = os.getenv("CHAIN_BINARY", "mychaind"), home: str = ".") -> bool:
    """Run `<chain_binary> validate-genesis` to ensure the genesis file is valid.

    Parameters
    ----------
    chain_binary : str, optional
        The chain binary to run, default comes from `CHAIN_BINARY` env var.
    home : str, optional
        Node's home directory (where the `config/` folder is), defaults to `.`.

    Returns
    -------
    bool
        True if validation passes; raises otherwise.
    """
    cmd = [chain_binary, "validate-genesis", "--home", home]
    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(
            "Genesis validation failed!\n" +
            f"stdout:\n{result.stdout}\n" +
            f"stderr:\n{result.stderr}"
        )

    # Success — echo whatever the binary output plus confirm.
    print(result.stdout or "Genesis validation passed ✅")
    return True

# Stand-alone usage example
if __name__ == "__main__":
    validate_genesis()
    print("Genesis file is valid and the chain is ready to start 🚀")


# step:1 file: create_protoc-generated_tx_and_query_grpc_services_for_a_custom_module
# file: scripts/verify_protoc_plugins.py
import shutil
import subprocess
import sys
from typing import Dict

REQUIRED_TOOLS = [
    "protoc",
    "protoc-gen-go",
    "protoc-gen-go-grpc",
    "protoc-gen-go_cosmos",
]

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

if __name__ == "__main__":
    try:
        info = verify_protoc_plugins()
        print("✅ Protobuf environment looks good:")
        for name, ver in info.items():
            print(f"  - {name}: {ver}")
    except EnvironmentError as err:
        print(f"❌ {err}")
        sys.exit(1)


# step:2 file: create_protoc-generated_tx_and_query_grpc_services_for_a_custom_module
# file: scripts/ensure_proto_files.py
from pathlib import Path

PROTO_BASE = Path("proto/myapp/mymodule")
MSG_PROTO = PROTO_BASE / "tx.proto"
QUERY_PROTO = PROTO_BASE / "query.proto"

MSG_TEMPLATE = """syntax = \"proto3\";
package myapp.mymodule.v1;

option go_package = \"github.com/myorg/myapp/x/mymodule/types\";

import \"gogoproto/gogo.proto\";
import \"cosmos/msg/v1/msg.proto\";

service Msg {
  rpc SendExampleMsg(ExampleMsgRequest) returns (ExampleMsgResponse) {
    option (cosmos.msg.v1.signer) = \"creator\";
  }
}

message ExampleMsgRequest {
  string creator = 1;
  string payload = 2;
}

message ExampleMsgResponse {}
"""

QUERY_TEMPLATE = """syntax = \"proto3\";
package myapp.mymodule.v1;

option go_package = \"github.com/myorg/myapp/x/mymodule/types\";

import \"cosmos/base/query/v1beta1/pagination.proto\";
import \"cosmos/query/v1/query.proto\";

service Query {
  rpc Example(QueryExampleRequest) returns (QueryExampleResponse) {
    option (cosmos.query.v1.query) = {
      path: \"/myapp/mymodule/v1/example/{creator}\"
    };
  }
}

message QueryExampleRequest {
  string creator = 1;
}

message QueryExampleResponse {
  string payload = 1;
}
"""

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

if __name__ == "__main__":
    print(ensure_proto_files())


# step:3 file: create_protoc-generated_tx_and_query_grpc_services_for_a_custom_module
# file: scripts/buf_generate.py
import subprocess
from pathlib import Path
import sys

def buf_generate(repo_root: str = ".") -> str:
    """Execute `buf generate` inside the repo root."""
    root = Path(repo_root).resolve()
    if not root.joinpath("buf.yaml").exists():
        raise FileNotFoundError(f"buf.yaml not found in {root}")
    proc = subprocess.run(["buf", "generate"], cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"buf generate failed:\n{proc.stderr}")
    return proc.stdout

if __name__ == "__main__":
    try:
        print(buf_generate())
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)


# step:4 file: create_protoc-generated_tx_and_query_grpc_services_for_a_custom_module
# file: scripts/compile_generated_code.py
import subprocess
from pathlib import Path
import sys

def compile_generated_code(repo_root: str = ".") -> str:
    """Run `go vet ./...` and `go test ./...` to validate compilation."""
    root = Path(repo_root).resolve()
    commands = [
        ["go", "vet", "./..."],
        ["go", "test", "./..."]
    ]
    for cmd in commands:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr}")
    return "Go vet and go test executed successfully."

if __name__ == "__main__":
    try:
        print(compile_generated_code())
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)


# step:5 file: create_protoc-generated_tx_and_query_grpc_services_for_a_custom_module
# file: scripts/verify_generated_services.py
from pathlib import Path
import sys

def verify_generated_services(module_path: str = "x/mymodule/types") -> str:
    """Ensure generated gRPC files exist after code-gen."""
    expected = [
        "tx.pb.go",
        "tx_grpc.pb.go",
        "query.pb.go",
        "query_grpc.pb.go",
    ]
    base = Path(module_path).resolve()
    missing = [f for f in expected if not base.joinpath(f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing generated files: {', '.join(missing)}")
    return "All required gRPC stubs are present."

if __name__ == "__main__":
    try:
        print(verify_generated_services())
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)


# step:1 file: store_the_returned_code_id_from_a_wasm_store_transaction_into_a_shell_variable_code_id
###############################
# backend/store_code.py       #
###############################

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.crypto.keyring import PrivateKey
from cosmpy.aerial.tx import Transaction
from cosmos_proto.cosmwasm.wasm.v1 import tx_pb2 as wasm_tx_pb2

app = FastAPI()

# -------- Configuration --------
CHAIN_ID      = os.getenv("CHAIN_ID", "juno-1")
RPC_ENDPOINT  = os.getenv("RPC_ENDPOINT", "https://rpc.juno.strange.love:443")
FEE_DENOM     = os.getenv("FEE_DENOM", "ujuno")
GAS_PRICE     = float(os.getenv("GAS_PRICE", "0.025"))
SIGNER_MNEMONIC = os.getenv("SIGNER_MNEMONIC")  # make sure to export this securely

if not SIGNER_MNEMONIC:
    raise RuntimeError("Environment variable SIGNER_MNEMONIC is required for contract upload.")

net_cfg = NetworkConfig(
    chain_id   = CHAIN_ID,
    url        = RPC_ENDPOINT,
    fee_denom  = FEE_DENOM,
    gas_price  = GAS_PRICE,
)

# -------- Endpoint --------
@app.post("/api/store_code")
async def store_code(file: UploadFile = File(...)):
    """Upload compiled wasm, broadcast MsgStoreCode, and return full tx result."""
    try:
        wasm_bytes = await file.read()
        if not wasm_bytes:
            raise ValueError("Uploaded file is empty or could not be read.")

        # Initialise signer & client
        pk     = PrivateKey.from_mnemonic(SIGNER_MNEMONIC)
        client = LedgerClient(net_cfg)

        # Build MsgStoreCode
        msg = wasm_tx_pb2.MsgStoreCode(
            sender         = str(pk.address()),
            wasm_byte_code = wasm_bytes,
        )

        # Assemble & sign the transaction
        tx = (
            Transaction()
            .with_messages(msg)
            .with_chain_id(CHAIN_ID)
            .with_gas(3_000_000)        # adjust if needed
            .with_fee_amount(1_000_000) # micro-denom
            .with_fee_denom(FEE_DENOM)
            .with_memo("Mintlify: upload contract code")
        )

        signed_tx = tx.sign(pk)

        # Broadcast & wait for inclusion (block mode)
        tx_result = client.broadcast_block(signed_tx)
        return tx_result  # will be consumed by Step 2

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:2 file: store_the_returned_code_id_from_a_wasm_store_transaction_into_a_shell_variable_code_id
################################
# backend/extract_code_id.py    #
################################

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class TxPayload(BaseModel):
    """Pydantic model matching the tx JSON from Step 1"""
    tx_response: dict


def _extract_code_id(tx_json: dict) -> str:
    """Walk the tx response to locate the code_id inside the store_code event."""
    try:
        events = tx_json["logs"][0]["events"]
        for evt in events:
            if evt["type"] == "store_code":
                for attr in evt["attributes"]:
                    if attr["key"] in ("code_id", "codeID"):
                        return attr["value"]
        raise KeyError("code_id not found in logs")
    except Exception as err:
        raise ValueError(f"Unable to extract code_id: {err}")


@app.post("/api/extract_code_id")
async def extract_code_id(payload: TxPayload):
    """HTTP endpoint: returns { "code_id": <str> }"""
    try:
        code_id = _extract_code_id(payload.tx_response)
        return {"code_id": code_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# step:1 file: restore_application_state_from_an_existing_local_snapshot
import os


def get_node_home(env_var='SIMD_HOME', default_path='~/.simd'):
    # Return absolute path to the simd home directory.
    home = os.getenv(env_var, default_path)
    return os.path.abspath(os.path.expanduser(home))



# step:2 file: restore_application_state_from_an_existing_local_snapshot
import subprocess


def list_snapshots(home_dir):
    '''Return a list of snapshot dictionaries: {'id': str, 'height': int}.'''
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



# step:3 file: restore_application_state_from_an_existing_local_snapshot
import subprocess


def stop_simd_service(service_name='simd'):
    '''Stop the simd daemon via systemctl or pkill.'''
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



# step:4 file: restore_application_state_from_an_existing_local_snapshot
import shutil
import datetime
import os


def backup_and_clear_data(home_dir, data_subdir='data'):
    '''Back up the current data directory to <home>/backup/data_<timestamp>.'''
    data_path = os.path.join(home_dir, data_subdir)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f'Data dir not found: {data_path}')
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    backup_root = os.path.join(home_dir, 'backup')
    os.makedirs(backup_root, exist_ok=True)
    backup_path = os.path.join(backup_root, f'data_{timestamp}')
    shutil.move(data_path, backup_path)
    return backup_path



# step:5 file: restore_application_state_from_an_existing_local_snapshot
import subprocess


def restore_snapshot(home_dir, snapshot_id):
    '''Run simd snapshot restore for the provided snapshot ID.'''
    cmd = ['simd', 'snapshot', 'restore', f'--home={home_dir}', snapshot_id]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Snapshot restore failed: {e}')



# step:6 file: restore_application_state_from_an_existing_local_snapshot
def build_simd_start_cmd(home_dir, additional_flags=''):
    '''Return a ready-to-run simd start command string.'''
    return f'simd start --home={home_dir} {additional_flags}'.strip()



# step:7 file: restore_application_state_from_an_existing_local_snapshot
import subprocess


def start_simd(home_dir, additional_flags='', detach=True):
    '''Start simd in foreground or background based on `detach`.'''
    cmd = ['simd', 'start', f'--home={home_dir}'] + additional_flags.split()
    if detach:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 'Started simd in background'
    subprocess.run(cmd, check=True)
    return 'Started simd in foreground'



# step:8 file: restore_application_state_from_an_existing_local_snapshot
import requests
import time


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



# step:1 file: set_minimum-gas-prices_to_0stake_in_gaia’s_app.toml
import os

CONFIG_PATH = os.path.expanduser("~/.gaia/config/app.toml")

def open_config_file(config_path: str | None = None) -> list[str]:
    """Read $HOME/.gaia/config/app.toml and return a list of lines.

    Args:
        config_path: Optional custom path to the app.toml file.

    Returns:
        List of lines contained in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = config_path or CONFIG_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    return lines


# step:2 file: set_minimum-gas-prices_to_0stake_in_gaia’s_app.toml
def modify_minimum_gas_prices(lines: list[str], new_value: str = "0stake") -> list[str]:
    """Update—or insert—the `minimum-gas-prices` parameter.

    Args:
        lines: Original list of lines from app.toml.
        new_value: The desired gas-price string, e.g. `"0stake"`.

    Returns:
        A new list of lines reflecting the change.
    """
    target_line = f'minimum-gas-prices = "{new_value}"\n'
    updated = False

    for idx, line in enumerate(lines):
        if line.strip().startswith("minimum-gas-prices"):
            lines[idx] = target_line
            updated = True
            break

    if not updated:
        # If the key wasn’t found, append it to the end of the [baseapp] section or file.
        lines.append("\n# Added automatically by script\n" + target_line)

    return lines


# step:3 file: set_minimum-gas-prices_to_0stake_in_gaia’s_app.toml
import shutil

def save_and_close_file(lines: list[str], config_path: str | None = None) -> str:
    """Persist the updated app.toml to disk and keep a backup.

    Args:
        lines: The modified list of lines to write.
        config_path: Optional custom path to the app.toml file.

    Returns:
        The path to the file that was written.
    """
    path = config_path or CONFIG_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    # Create a backup before overwriting
    backup_path = path + ".bak"
    shutil.copy(path, backup_path)

    with open(path, "w", encoding="utf-8") as fp:
        fp.writelines(lines)

    return path


# step:4 file: set_minimum-gas-prices_to_0stake_in_gaia’s_app.toml
def validate_file_change(config_path: str | None = None, expected_value: str = "0stake") -> bool:
    """Verify `minimum-gas-prices` now equals the expected string.

    Args:
        config_path: Optional custom path to the app.toml file.
        expected_value: Expected gas-price value (default: "0stake").

    Returns:
        True if the parameter is set correctly, else False.
    """
    path = config_path or CONFIG_PATH
    expected_line = f'minimum-gas-prices = "{expected_value}"'

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip() == expected_line:
                return True

    return False


# step:3 file: create_a_shell_alias_for_cast_with_a_default_rpc_endpoint
#!/usr/bin/env python3
"""
add_cast_alias.py

Appends an alias for the `cast` command that includes the supplied
RPC endpoint to the user's shell profile (either ~/.zshrc or ~/.bashrc).
"""

import os
from pathlib import Path


def detect_profile() -> Path:
    """Infer which shell profile should be modified."""
    shell = os.environ.get("SHELL", "")
    if shell.endswith("zsh"):
        return Path.home() / ".zshrc"
    return Path.home() / ".bashrc"


def add_cast_alias(rpc_url: str, profile_path: Path | None = None) -> None:
    """Append (or skip if present) the alias line to the profile file."""
    if not rpc_url:
        raise ValueError("rpc_url is required")

    profile_path = profile_path or detect_profile()
    alias_line = f'alias cast="cast --rpc-url {rpc_url}"'

    # Skip if alias already exists
    if profile_path.exists() and alias_line in profile_path.read_text():
        print("Alias already present; nothing to do.")
        return

    with profile_path.open("a", encoding="utf-8") as fp:
        fp.write("\n# Added by Cosmos EVM setup script\n")
        fp.write(alias_line + "\n")

    print(f"Alias successfully written to {profile_path}")


if __name__ == "__main__":
    rpc_url = os.environ.get("COSMOS_RPC_URL") or input("Enter the RPC URL: ").strip()
    add_cast_alias(rpc_url)


# step:4 file: create_a_shell_alias_for_cast_with_a_default_rpc_endpoint
#!/usr/bin/env python3
"""
reload_shell.py

Starts a new login shell so that any recently modified shell profile
(e.g., ~/.bashrc or ~/.zshrc) is reloaded, making the 'cast' alias
immediately available.
"""

import os
import subprocess
import sys


def reload_shell() -> None:
    shell_path = os.environ.get("SHELL", "/bin/bash")
    print(f"Spawning a fresh login shell ({shell_path} -l)...")
    print("After this command finishes, verify the alias with `type cast`.")
    try:
        subprocess.run([shell_path, "-l"], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Failed to reload shell: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    reload_shell()


# step:1 file: generate_interface.rs_for_a_cosmwasm_contract_with_the_cw-orch_#[interface]_macro
from pathlib import Path
from typing import Union


def locate_contract_crate(start_path: Union[str, Path] = '.') -> Path:
    '''
    Locate the root of a CosmWasm contract crate.

    The function checks that a `Cargo.toml` file exists in the provided directory
    and returns the absolute path to the crate root.

    Args:
        start_path: Directory where the search should begin. Defaults to the
            current working directory.

    Returns:
        pathlib.Path pointing to the crate root.

    Raises:
        FileNotFoundError: If the `Cargo.toml` file cannot be found.
    '''
    crate_path = Path(start_path).expanduser().resolve()
    cargo_file = crate_path / 'Cargo.toml'

    if not cargo_file.exists():
        raise FileNotFoundError(
            f'Could not locate Cargo.toml at {cargo_file}. Please supply the correct contract crate path.'
        )

    return crate_path



# step:2 file: generate_interface.rs_for_a_cosmwasm_contract_with_the_cw-orch_#[interface]_macro
import toml
from pathlib import Path
from typing import Union


def add_cargo_dependency(crate_path: Union[str, Path]) -> None:
    '''
    Add `cw-orch` to the `[dev-dependencies]` section of Cargo.toml if it is not already present.

    Args:
        crate_path: Path to the root of the contract crate.
    '''
    crate_path = Path(crate_path).expanduser().resolve()
    cargo_toml = crate_path / 'Cargo.toml'

    if not cargo_toml.exists():
        raise FileNotFoundError(f'{cargo_toml} does not exist.')

    cargo_data = toml.load(cargo_toml)
    dev_deps = cargo_data.get('dev-dependencies', {})

    if 'cw-orch' not in dev_deps:
        dev_deps['cw-orch'] = {
            'version': '^0.9',
            'default-features': False,
            'features': ['derive']
        }
        cargo_data['dev-dependencies'] = dev_deps
        cargo_toml.write_text(toml.dumps(cargo_data))
        print('cw-orch added to dev-dependencies.')
    else:
        print('cw-orch already present in dev-dependencies; skipping.')



# step:3 file: generate_interface.rs_for_a_cosmwasm_contract_with_the_cw-orch_#[interface]_macro
from pathlib import Path
from typing import Union


def create_interface_file(crate_path: Union[str, Path]) -> Path:
    '''
    Create (or truncate) the `src/interface.rs` file within the contract crate.

    Args:
        crate_path: Path to the root of the contract crate.

    Returns:
        Path to the created file.
    '''
    crate_path = Path(crate_path).expanduser().resolve()
    src_dir = crate_path / 'src'
    src_dir.mkdir(parents=True, exist_ok=True)
    interface_path = src_dir / 'interface.rs'
    interface_path.write_text('')
    print(f'Created empty interface file at {interface_path}')
    return interface_path



# step:4 file: generate_interface.rs_for_a_cosmwasm_contract_with_the_cw-orch_#[interface]_macro
from pathlib import Path
from typing import Union

INTERFACE_BOILERPLATE = '''// Auto-generated by tooling — feel free to extend.
use cosmwasm_std::{Addr, Response, DepsMut, Env, MessageInfo};
use cw_orch::interface;

#[interface]
pub trait CounterContract {
    type Error = anyhow::Error;
    type InstantiateMsg = crate::msg::InstantiateMsg;
    type ExecuteMsg = crate::msg::ExecuteMsg;
    type QueryMsg = crate::msg::QueryMsg;

    // (Optional) declare helper wrappers, e.g.:
    fn increment(&self) -> Result<Response, Self::Error>;
}
'''


def write_interface_boilerplate(crate_path: Union[str, Path]) -> None:
    '''
    Populate `src/interface.rs` with the cw-orch interface boilerplate.
    '''
    crate_path = Path(crate_path).expanduser().resolve()
    interface_file = crate_path / 'src' / 'interface.rs'

    if not interface_file.exists():
        raise FileNotFoundError(
            f'interface.rs does not exist at {interface_file}. Did you run create_interface_file()?'
        )

    interface_file.write_text(INTERFACE_BOILERPLATE)
    print(f'Wrote interface boilerplate to {interface_file}')



# step:5 file: generate_interface.rs_for_a_cosmwasm_contract_with_the_cw-orch_#[interface]_macro
import subprocess
from pathlib import Path
from typing import Union


def cargo_check(crate_path: Union[str, Path]) -> None:
    '''
    Run `cargo check --tests` inside the provided contract crate directory.
    '''
    crate_path = Path(crate_path).expanduser().resolve()
    cmd = ['cargo', 'check', '--tests']

    try:
        subprocess.run(cmd, cwd=crate_path, check=True, text=True)
        print('cargo check --tests completed successfully.')
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout)
        if error.stderr:
            print(error.stderr)
        raise



# step:1 file: initialize_a_new_chain_with_chain-id_my-test-chain
import subprocess
from typing import Dict


def init_genesis(moniker: str, chain_id: str = "my-test-chain") -> Dict[str, str]:
    """Initialise a new chain by executing `simd init`.

    Args:
        moniker (str): A human-readable name for your node.
        chain_id (str, optional): The desired chain-ID. Defaults to "my-test-chain".

    Returns:
        Dict[str, str]: Captured stdout/stderr for logging purposes.
    """
    cmd = ["simd", "init", moniker, "--chain-id", chain_id]

    try:
        # Run the command and capture output for debugging/logging.
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return {"stdout": completed.stdout, "stderr": completed.stderr}
    except FileNotFoundError:
        raise RuntimeError(
            "`simd` binary not found. Make sure the Cosmos SDK binary is installed and in $PATH."
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"simd init failed with exit code {exc.returncode}: {exc.stderr.strip()}"
        )



# step:2 file: initialize_a_new_chain_with_chain-id_my-test-chain
import json
import os


def verify_genesis_file(
    genesis_path: str = os.path.expanduser("~/.simapp/config/genesis.json"),
    expected_chain_id: str = "my-test-chain",
) -> bool:
    """Verify that the genesis file contains the expected chain-ID.

    Args:
        genesis_path (str, optional): Path to genesis.json. Defaults to standard simd location.
        expected_chain_id (str, optional): The chain-ID we expect. Defaults to "my-test-chain".

    Returns:
        bool: True if verification succeeds, raises otherwise.
    """
    if not os.path.isfile(genesis_path):
        raise FileNotFoundError(
            f"Genesis file not found at {genesis_path}. Did you run `simd init`?"
        )

    with open(genesis_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    actual_chain_id = data.get("chain_id")

    if actual_chain_id != expected_chain_id:
        raise ValueError(
            f"chain_id mismatch: expected '{expected_chain_id}', found '{actual_chain_id}'"
        )

    return True



# step:2 file: check_my_health_factor_on_amber_finance
from fastapi import FastAPI, HTTPException
import os, json, base64, httpx

app = FastAPI()

# Environment variables keep secrets & tunables out of source-code.
NEUTRON_LCD = os.getenv('NEUTRON_LCD', 'https://rest-kralum.neutron.org')
AMBER_CONTRACT_ADDR = os.getenv('AMBER_CONTRACT_ADDR', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

async def _query_wasm_smart(contract_addr: str, query_msg: dict):
    """Low-level helper that hits the LCD `/smart/` endpoint."""
    msg_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
    url = f"{NEUTRON_LCD}/cosmwasm/wasm/v1/contract/{contract_addr}/smart/{msg_b64}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        # LCD wraps contract results inside a `data` or `result` field depending on version.
        data = r.json()
        return data.get('data') or data.get('result') or data

@app.get('/api/amber_positions')
async def amber_positions(address: str):
    """Public route => `/api/amber_positions?address=<bech32>`"""
    try:
        query_msg = {"positions_by_owner": {"owner": address}}
        positions = await _query_wasm_smart(AMBER_CONTRACT_ADDR, query_msg)
        return positions  # Forward raw contract JSON back to the caller.
    except HTTPException:
        raise  # Re-throw FastAPI HTTP errors untouched.
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Amber query failed: {exc}")


# step:6 file: lock_2000_ntrn_for_3_months_to_obtain_a_1.2×_btc_summer_boost
###############################################################################
# backend/lock_tokens.py                                                       #
###############################################################################

import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgExecuteContract

app = FastAPI(title="NTRN Lock API")

# --- Models ------------------------------------------------------------------
class Fund(BaseModel):
    denom: str
    amount: str

class LockRequest(BaseModel):
    contract_address: str = Field(..., description="Lock contract address")
    sender: str = Field(..., description="User address that appears as Msg sender")
    msg: dict = Field(..., description="ExecuteMsg JSON body")
    funds: list[Fund]

# --- Chain Config ------------------------------------------------------------
NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url="https://rpc-kralum.neutron.org",  # Public RPC; replace if necessary
    fee_denomination="untrn",
    gas_price=0.025,            # 0.025untrn is a safe over-estimate
    staking_denomination="untrn",
)

# Wallet that will sign the transaction (use with caution!)
MNEMONIC = os.getenv("NTRN_WALLET_MNEMONIC")
if MNEMONIC is None:
    raise RuntimeError("NTRN_WALLET_MNEMONIC env-var is not set")

WALLET = LocalWallet.from_mnemonic(MNEMONIC)

# --- Endpoint ----------------------------------------------------------------
@app.post("/api/lock_tokens")
async def lock_tokens(req: LockRequest):
    try:
        # Defensive checks ----------------------------------------------------
        if WALLET.address() != req.sender:
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
        tx.with_sequence(LedgerClient(NETWORK).get_sequence(req.sender))
        tx.with_chain_id(NETWORK.chain_id)
        tx.with_gas(250_000)  # empirical gas; adjust if necessary
        tx.with_memo("Lock 2K NTRN for 90d")

        # Sign using backend wallet
        tx_signed = tx.sign(WALLET)

        # Broadcast -----------------------------------------------------------
        client = LedgerClient(NETWORK)
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


# step:1 file: write_a_block_profile_to_block.prof_for_30_seconds
import os
import subprocess
from typing import Dict

NODE_BINARY = os.getenv("COSMOS_NODE_BINARY", "gaiad")  # Override with your specific binary if different
NODE_HOME = os.getenv("COSMOS_NODE_HOME", os.path.expanduser("~/.gaia"))
LOG_FILE = "/tmp/cosmos_node.log"

def enable_block_profiling() -> Dict[str, str]:
    """Restart the node with block profiling turned on (pprof exposed on :6060)."""
    # 1. Stop any running instance of the node (ignore errors if none running)
    subprocess.run(["pkill", "-f", NODE_BINARY], check=False)

    # 2. Prepare the new environment with profiling enabled
    env = os.environ.copy()
    env["GODEBUG"] = "blockprofilerate=1"  # Instruct Go runtime to collect blocking events

    # 3. Start the node
    cmd = [NODE_BINARY, "start", "--home", NODE_HOME]
    log_handle = open(LOG_FILE, "w")
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        raise RuntimeError(f"{NODE_BINARY} not found. Check COSMOS_NODE_BINARY environment variable.")

    return {
        "pid": str(proc.pid),
        "log_file": LOG_FILE,
        "message": f"Node restarted with block profiling. PID {proc.pid}. Logs at {LOG_FILE}."
    }


# step:2 file: write_a_block_profile_to_block.prof_for_30_seconds
import asyncio
from typing import Dict

async def wait_runtime(seconds: int = 30) -> Dict[str, str]:
    """Pause execution for `seconds` to let the node gather block-profile samples."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    await asyncio.sleep(seconds)
    return {"status": "ok", "waited_seconds": seconds}


# step:3 file: write_a_block_profile_to_block.prof_for_30_seconds
import requests
from typing import Dict

PPROF_URL = "http://localhost:6060/debug/pprof/block?debug=0"
OUTPUT_FILE = "/tmp/block.prof"

def fetch_block_profile(url: str = PPROF_URL, output_path: str = OUTPUT_FILE) -> Dict[str, str]:
    """Retrieve the block profile and write it to `output_path`."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as err:
        raise RuntimeError(f"Failed to fetch block profile: {err}")

    with open(output_path, "wb") as fp:
        fp.write(response.content)

    return {
        "output_path": output_path,
        "size_bytes": len(response.content),
        "message": f"Block profile saved to {output_path} ({len(response.content)} bytes)"
    }


# step:4 file: write_a_block_profile_to_block.prof_for_30_seconds
import os
from typing import Dict

PROFILE_PATH = "/tmp/block.prof"

def verify_profile_file(profile_path: str = PROFILE_PATH) -> Dict[str, str]:
    """Check that `profile_path` exists and its size is greater than zero."""
    if not os.path.isfile(profile_path):
        return {"exists": False, "size_bytes": 0, "message": f"{profile_path} not found."}

    size = os.path.getsize(profile_path)
    if size == 0:
        return {"exists": True, "size_bytes": 0, "message": f"{profile_path} is empty."}

    return {"exists": True, "size_bytes": size, "message": f"{profile_path} verified with {size} bytes."}


# step:2 file: close_my_leveraged_loop_position_on_amber
# File: backend/routes/amber.py
import os
import json
from fastapi import APIRouter, HTTPException
from cosmpy.aio.client import LedgerClient

router = APIRouter()

RPC_ENDPOINT = os.getenv("RPC_ENDPOINT", "https://rpc-palvus.neutron-1.neutron.org")
AMBER_CONTRACT = os.getenv("AMBER_CONTRACT", "neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

@router.get("/api/amber/position_status/{address}")
async def query_position_status(address: str):
    """Returns the address’ Amber position (if any)."""
    try:
        async with LedgerClient(RPC_ENDPOINT) as client:
            query_msg = {"position_status": {"address": address}}
            # Amber is a CosmWasm contract; `wasm_query` expects bytes
            result = await client.wasm_query(
                AMBER_CONTRACT,
                json.dumps(query_msg).encode()
            )
            return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Position query failed: {exc}")


# step:3 file: close_my_leveraged_loop_position_on_amber
# File: backend/routes/amber.py (continued)
import base64
from typing import Optional
from pydantic import BaseModel
from cosmpy.aio.tx import Transaction
from cosmpy.aio.msg import MsgExecuteContract

class ClosePosRequest(BaseModel):
    address: str
    position_id: int
    chain_id: str = "neutron-1"
    gas_limit: Optional[int] = 200000
    fee_amount: Optional[str] = "200000"  # in micro-denom (untrn)
    fee_denom: Optional[str] = "untrn"

@router.post("/api/amber/close_position_sign_doc")
async def close_position_sign_doc(req: ClosePosRequest):
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
                contract = AMBER_CONTRACT,
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


# step:5 file: close_my_leveraged_loop_position_on_amber
# File: backend/routes/amber.py (continued)
@router.get("/api/amber/position_status_confirm/{address}")
async def confirm_position_closed(address: str):
    """Returns `{closed: true}` once the address has no outstanding debt."""
    try:
        data = await query_position_status(address)
        debt = data.get("position", {}).get("debt", 0)
        return {"closed": int(debt) == 0, "raw": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Confirmation failed: {exc}")


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


# step:1 file: Create a schedule to perform health checks every 300 blocks
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.contract import CosmWasmContract


def get_dao_authority_address(dao_contract: str, lcd_url: str = "https://lcd.neutron.org") -> str:
    """Query the DAO contract for its authority address.

    Parameters
    ----------
    dao_contract: str
        On-chain address of the DAO contract.
    lcd_url: str
        LCD endpoint for a Neutron full node (defaults to main-net).

    Returns
    -------
    str
        The authority address able to create governance proposals.
    """
    try:
        cfg = NetworkConfig(
            chain_id="neutron-1",
            url=lcd_url,
            fee_minimum_gas_price=0.025,
            fee_denomination="untrn",
        )
        client = LedgerClient(cfg)
        contract = CosmWasmContract(client, dao_contract)

        # NB: replace the query payload with whatever your DAO exposes.
        result = contract.query({"authority": {}})
        authority = result.get("authority")
        if not authority:
            raise ValueError("Authority address missing in query response")
        return authority
    except Exception as err:
        raise RuntimeError(f"Failed to fetch DAO authority address: {err}")


# step:2 file: Create a schedule to perform health checks every 300 blocks
import json
from google.protobuf.any_pb2 import Any as Any_pb
from google.protobuf.json_format import MessageToDict

# Protobuf imports generated from Neutron & Cosmos SDK proto definitions
from neutron_proto.cron import MsgAddSchedule
from cosmos_proto.cosmwasm.wasm.v1 import MsgExecuteContract


def build_msg_add_schedule(authority: str,
                           monitoring_contract: str,
                           gas_limit: int = 500_000) -> MsgAddSchedule:
    """Compose a MsgAddSchedule ready for inclusion in a Tx."""
    try:
        # 1. Build the inner MsgExecuteContract that will be executed by Cron
        exec_msg = MsgExecuteContract(
            sender=authority,
            contract=monitoring_contract,
            msg=json.dumps({"perform_checks": {}}).encode(),
            funds=[],
        )

        # 2. Wrap everything into MsgAddSchedule
        add_schedule_msg = MsgAddSchedule(
            authority=authority,
            name="health_check",
            period=300,
            msgs=[exec_msg.SerializeToString()],  # list[bytes]
            gas_limit=gas_limit,
        )
        return add_schedule_msg
    except Exception as err:
        raise RuntimeError(f"Unable to build MsgAddSchedule: {err}")


# step:3 file: Create a schedule to perform health checks every 300 blocks
from cosmos_proto.cosmos.gov.v1 import MsgSubmitProposal, TextProposal


def package_into_gov_proposal(authority: str,
                              schedule_msg: MsgAddSchedule,
                              deposit: int = 100_000_000) -> MsgSubmitProposal:
    """Create a MsgSubmitProposal that contains our schedule message."""
    try:
        # Pack the schedule message in "Any"
        any_schedule = Any_pb()
        any_schedule.Pack(schedule_msg)

        proposal_content = TextProposal(
            title="Add periodic on-chain health check",
            description=(
                "This proposal adds a Cron schedule named `health_check` that "
                "executes the monitoring contract’s `perform_checks` message "
                "every 300 blocks (~30 minutes). The cadence is selected to "
                "provide timely detection of potential network issues while "
                "minimising gas consumption."
            ),
        )
        any_content = Any_pb()
        any_content.Pack(proposal_content)

        msg_submit = MsgSubmitProposal(
            content=any_content,
            messages=[any_schedule],
            initial_deposit=[{"denom": "untrn", "amount": str(deposit)}],
            proposer=authority,
        )
        return msg_submit
    except Exception as err:
        raise RuntimeError(f"Could not package proposal: {err}")


# step:4 file: Create a schedule to perform health checks every 300 blocks
from cosmpy.crypto.keypairs import PrivateKey
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.client import LedgerClient, NetworkConfig


def sign_and_broadcast_tx(priv_key_hex: str,
                          msgs: list,
                          rpc_url: str = "https://rpc.neutron.org") -> str:
    """Sign an array of messages and broadcast the Tx, returning its hash."""
    try:
        cfg = NetworkConfig(
            chain_id="neutron-1",
            url=rpc_url,
            fee_minimum_gas_price=0.025,
            fee_denomination="untrn",
        )
        client = LedgerClient(cfg)

        private_key = PrivateKey(bytes.fromhex(priv_key_hex))
        sender = private_key.public_key().address()

        tx = Transaction()
        for msg in msgs:
            tx.add_message(msg)

        tx.with_sender(sender)
        tx.with_chain_id(cfg.chain_id)
        tx = tx.with_gas(700_000)
        tx = tx.with_fee(int(700_000 * 0.025), "untrn")

        signed = tx.sign(private_key, client)
        tx_hash = client.broadcast_tx(signed)
        result = client.wait_for_tx(tx_hash)
        if result.code != 0:
            raise RuntimeError(f"Tx failed: {result.raw_log}")
        return tx_hash
    except Exception as err:
        raise RuntimeError(f"Broadcast error: {err}")


# step:1 file: batch_multiple_json-rpc_calls_with_the_viem_library
// backend/utils/viemClient.js
import { createPublicClient, http } from 'viem';

/**
 * Returns a viem Public Client connected to the given RPC URL.
 * @param {string} rpcUrl – Full HTTP(s) URL of the Cosmos-EVM JSON-RPC endpoint.
 * @returns {import('viem').PublicClient}
 */
export const createViemClient = (rpcUrl) => {
  try {
    if (!rpcUrl) throw new Error('RPC URL is required to create a viem client.');

    // Because we are talking to a Cosmos-EVM chain, we can omit the chain
    // definition and rely on RPC introspection. If you know the chain Id
    // you may pass a custom Chain object here instead.
    const client = createPublicClient({
      transport: http(rpcUrl)
    });

    return client;
  } catch (error) {
    console.error('[createViemClient] Unable to create client:', error);
    throw error;
  }
};


# step:2 file: batch_multiple_json-rpc_calls_with_the_viem_library
// backend/utils/batchContext.js
/**
 * Generates a batch context around a viem client. The context exposes two
 * helper functions: `add(call)` for queuing RPC calls and `execute()` for
 * sending them in a single batched HTTP request.
 *
 * @param {import('viem').PublicClient} client – The viem client created in Step 1.
 */
export const createBatchContext = (client) => {
  if (!client || typeof client.batch !== 'function') {
    throw new Error('A valid viem PublicClient must be supplied to createBatchContext().');
  }

  // Internal queue of call Promises (NOT awaited yet)
  const queue = [];

  return {
    /**
     * Enqueue a viem call (e.g., client.getBalance(...)). The call should be
     * invoked but NOT awaited – the returned promise is what we queue up.
     * @param {Promise<any>} callPromise
     */
    add: (callPromise) => {
      queue.push(callPromise);
    },

    /**
     * Executes the queued calls in a single JSON-RPC batch. Returns the ordered
     * array of results.
     * @returns {Promise<any[]>}
     */
    execute: async () => {
      try {
        if (!queue.length) return [];
        const results = await client.batch(queue);
        return results;
      } catch (error) {
        console.error('[batchContext.execute] Batch execution failed:', error);
        throw error;
      }
    }
  };
};


# step:3 file: batch_multiple_json-rpc_calls_with_the_viem_library
// backend/batch/addCalls.js
import { erc20Abi } from 'viem';

/**
 * Demonstrates how to enqueue several read-only calls onto the batch queue.
 * @param {ReturnType<typeof import('./utils/batchContext.js').createBatchContext>} batch – Batch context from Step 2.
 * @param {import('viem').PublicClient} client – The viem client.
 * @param {string} address – Bech32 / 0x-address to query.
 */
export const addReadOnlyCallsToBatch = (batch, client, address) => {
  if (!address) throw new Error('Target address is required to queue read calls.');

  // 1. Latest block number (promise is returned immediately, NOT awaited)
  batch.add(client.getBlockNumber());

  // 2. Native token balance of the address
  batch.add(
    client.getBalance({
      address
    })
  );

  // 3. Example ERC-20 totalSupply() read – replace with a real contract address
  const exampleTokenAddress = '0x0000000000000000000000000000000000000000';
  batch.add(
    client.readContract({
      address: exampleTokenAddress,
      abi: erc20Abi,
      functionName: 'totalSupply'
    })
  );
};


# step:4 file: batch_multiple_json-rpc_calls_with_the_viem_library
// backend/batch/executeBatch.js
/**
 * Executes the queued batch and handles the results.
 * @param {ReturnType<typeof import('./utils/batchContext.js').createBatchContext>} batch – Batch context containing queued calls.
 * @returns {Promise<void>}
 */
export const executeBatch = async (batch) => {
  try {
    const results = await batch.execute();

    // The results array is ordered in the same sequence the calls were queued.
    // Example unpacking for the three calls added in Step 3.
    const [blockNumber, balance, totalSupply] = results;

    console.log('Block Number :', blockNumber);
    console.log('Account Bal. :', balance.toString());
    console.log('Token Supply :', totalSupply.toString());
  } catch (error) {
    console.error('[executeBatch] Failed to execute batch:', error);
    throw error;
  }
};


# step:1 file: start_the_local_simd_node
import os
import json

async def check_chain_home(home_path: str = os.path.expanduser("~/.simapp")) -> dict:
    """Validate that the simd home directory has a readable genesis.json and config.toml."""
    result = {
        "home": home_path,
        "genesis_exists": False,
        "genesis_valid_json": False,
        "config_exists": False,
        "valid": False,
        "errors": []
    }

    genesis_path = os.path.join(home_path, "config", "genesis.json")
    config_toml_path = os.path.join(home_path, "config", "config.toml")

    try:
        # Check for genesis.json
        if os.path.isfile(genesis_path):
            result["genesis_exists"] = True
            with open(genesis_path, "r") as f:
                json.load(f)  # Raises if the JSON is malformed
            result["genesis_valid_json"] = True
        else:
            result["errors"].append(f"Missing {genesis_path}")

        # Check for config.toml
        if os.path.isfile(config_toml_path):
            result["config_exists"] = True
        else:
            result["errors"].append(f"Missing {config_toml_path}")

        # All checks must pass
        result["valid"] = result["genesis_exists"] and result["genesis_valid_json"] and result["config_exists"]
    except Exception as e:
        result["errors"].append(str(e))
        result["valid"] = False

    return result


# step:2 file: start_the_local_simd_node
import os
import subprocess
from typing import Optional

# Keep a global reference so we can ensure only one node runs per backend instance.
NODE_PROCESS: Optional[subprocess.Popen] = None

async def start_node(home_path: str = os.path.expanduser("~/.simapp")) -> dict:
    """Start `simd start` as a background subprocess and return its PID or an error."""
    global NODE_PROCESS

    if NODE_PROCESS and NODE_PROCESS.poll() is None:
        return {"status": "already_running", "pid": NODE_PROCESS.pid}

    cmd = ["simd", "start"]

    # Supply --home if user chose a custom directory
    if home_path != os.path.expanduser("~/.simapp"):
        cmd.extend(["--home", home_path])

    try:
        NODE_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # decode bytes -> str automatically
        )
        return {"status": "started", "pid": NODE_PROCESS.pid, "cmd": " ".join(cmd)}
    except FileNotFoundError:
        return {"status": "error", "error": "`simd` binary not found in PATH."}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# step:3 file: start_the_local_simd_node
import time
import requests

async def health_check(max_attempts: int = 30, interval_sec: int = 2) -> dict:
    """Poll http://localhost:26657/status until the node reports a positive block height or times out."""
    url = "http://localhost:26657/status"
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                height = int(data["result"]["sync_info"]["latest_block_height"])
                if height > 0:
                    return {"healthy": True, "latest_block_height": height}
        except Exception:
            # Ignored: transient connection errors expected while node boots up
            pass

        time.sleep(interval_sec)

    return {"healthy": False, "error": "Node did not reach a non-zero block height within the allotted time."}


# step:1 file: set_the_initial_inflation_rate_in_genesis_to_0.300000000000000000
import os
from pathlib import Path

async def get_chain_home(chain_id: str = "simd") -> dict:
    """Resolve and validate the chain's home directory."""
    env_var = f"{chain_id.upper()}_HOME"       # e.g.  SIMD_HOME
    home_dir = os.getenv(env_var) or Path.home() / f".{chain_id}"

    # Cast Path object to str to keep JSON-serialisable output
    home_dir = str(home_dir)

    if not os.path.isdir(home_dir):
        raise FileNotFoundError(f"Chain home directory not found at {home_dir}")

    return {"chain_home": home_dir}


# step:2 file: set_the_initial_inflation_rate_in_genesis_to_0.300000000000000000
import json
from pathlib import Path

async def locate_genesis(chain_home: str) -> dict:
    """Locate and load config/genesis.json from the provided home directory."""
    genesis_path = Path(chain_home) / "config" / "genesis.json"

    if not genesis_path.exists():
        raise FileNotFoundError(f"genesis.json not found at {genesis_path}")

    with genesis_path.open("r", encoding="utf-8") as f:
        genesis_data = json.load(f)

    return {"genesis_path": str(genesis_path), "genesis_data": genesis_data}


# step:3 file: set_the_initial_inflation_rate_in_genesis_to_0.300000000000000000
import shutil
from datetime import datetime
from pathlib import Path

async def backup_genesis(genesis_path: str) -> dict:
    """Copy genesis.json to genesis.json.bak.<timestamp>."""
    src = Path(genesis_path)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup_path = src.with_suffix(f".json.bak.{timestamp}")
    shutil.copy2(src, backup_path)  # copy2 keeps file metadata
    return {"backup_path": str(backup_path)}


# step:4 file: set_the_initial_inflation_rate_in_genesis_to_0.300000000000000000
async def update_inflation(genesis_data: dict, new_inflation: str = "0.300000000000000000") -> dict:
    """Set app_state.mint.params.inflation to a new value."""
    try:
        genesis_data["app_state"]["mint"]["params"]["inflation"] = new_inflation
    except KeyError as err:
        raise KeyError("Unable to locate mint.params.inflation in genesis.json") from err

    return {"updated_genesis": genesis_data}


# step:5 file: set_the_initial_inflation_rate_in_genesis_to_0.300000000000000000
import json
from pathlib import Path

async def save_genesis(genesis_path: str, updated_genesis: dict) -> dict:
    """Persist the updated genesis data to disk."""
    path = Path(genesis_path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(updated_genesis, f, indent=2, ensure_ascii=False)
        f.write("\n")  # POSIX-friendly newline at EOF

    return {"status": "saved", "path": str(path)}


# step:6 file: set_the_initial_inflation_rate_in_genesis_to_0.300000000000000000
import subprocess

async def validate_genesis(chain_home: str) -> dict:
    """Run `simd validate-genesis` and capture its output."""
    cmd = [
        "simd",
        "validate-genesis",
        "--home",
        chain_home
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        return {"validation": "failed", "stderr": proc.stderr}

    return {"validation": "success", "stdout": proc.stdout}


# step:1 file: Create a fee_collection schedule to harvest fees every 1,200 blocks
from cosmpy.aerial.client import LedgerClient, NetworkConfig


def get_dao_authority_address(rpc_endpoint: str, dao_contract: str) -> str:
    """Return the DAO authority address or raise if it cannot be found."""
    try:
        cfg = NetworkConfig(
            chain_id="neutron-1",
            url=rpc_endpoint,
            fee_minimum_gas_price=0,
            fee_denomination="untrn",
        )
        client = LedgerClient(cfg)

        # The query shape {"authority": {}} is conventional for cw-dao based DAOs.
        response: dict = client.wasm_contract_query(dao_contract, {"authority": {}})
        authority = response.get("authority")
        if not authority:
            raise ValueError("DAO contract did not return an authority field")
        return authority
    except Exception as err:
        # Convert any lower-level error into an explicit failure that upstream code can catch.
        raise RuntimeError(f"Failed to obtain DAO authority address: {err}") from err


# step:2 file: Create a fee_collection schedule to harvest fees every 1,200 blocks
import json
from google.protobuf.any_pb2 import Any
from neutron.cron.v1.cron_pb2 import MsgAddSchedule
from cosmwasm.wasm.v1.tx_pb2 import MsgExecuteContract


def build_msg_add_schedule(authority: str, treasury_contract: str, gas_limit: int = 500000) -> MsgAddSchedule:
    """Return a fully-formed MsgAddSchedule protobuf message."""
    # 1. Craft the inner Wasm execute instruction that triggers `harvest_fees {}`.
    inner_execute = MsgExecuteContract(
        sender=authority,
        contract=treasury_contract,
        msg=json.dumps({"harvest_fees": {}}).encode("utf-8"),
        funds=[]  # no additional funds are sent
    )

    # 2. Pack the execute message into protobuf Any (Cron schedules carry generic msgs).
    any_msg = Any()
    any_msg.Pack(inner_execute)

    # 3. Assemble the MsgAddSchedule itself.
    schedule_msg = MsgAddSchedule(
        authority=authority,
        name="fee_collection",
        period=1200,      # seconds
        msgs=[any_msg],
        gas_limit=gas_limit,
    )
    return schedule_msg


# step:3 file: Create a fee_collection schedule to harvest fees every 1,200 blocks
from google.protobuf.any_pb2 import Any
from cosmos.gov.v1beta1.tx_pb2 import MsgSubmitProposal
from cosmos.base.v1beta1.coin_pb2 import Coin


def package_into_gov_proposal(schedule_msg: MsgAddSchedule,
                              proposer: str,
                              deposit_amount: int = 1000000,
                              deposit_denom: str = "untrn") -> MsgSubmitProposal:
    """Return MsgSubmitProposal containing the provided schedule message."""
    # 1. Pack the schedule into Any so the gov module can understand it.
    content_any = Any()
    content_any.Pack(schedule_msg)

    # 2. Construct the on-chain proposal message.
    proposal_msg = MsgSubmitProposal(
        content=content_any,
        initial_deposit=[Coin(amount=str(deposit_amount), denom=deposit_denom)],
        proposer=proposer,
        title="Automated DAO Fee Harvesting",
        description="Adds a cron schedule named ‘fee_collection’ that calls the treasury contract’s `harvest_fees` every 20 minutes.",
    )
    return proposal_msg


# step:4 file: Create a fee_collection schedule to harvest fees every 1,200 blocks
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import Wallet
from cosmpy.aerial.tx import Transaction


def sign_and_broadcast_tx(rpc_endpoint: str,
                          chain_id: str,
                          mnemonic: str,
                          msgs: list,
                          fee_amount: int = 4000,
                          gas_limit: int = 800000,
                          fee_denom: str = "untrn") -> str:
    """Sign the provided messages and broadcast; return the tx-hash."""
    try:
        # A. Connect to the ledger.
        cfg = NetworkConfig(
            chain_id=chain_id,
            url=rpc_endpoint,
            fee_minimum_gas_price=0,
            fee_denomination=fee_denom,
        )
        client = LedgerClient(cfg)

        # B. Load DAO wallet from mnemonic (keep mnemonic secret!).
        wallet = Wallet(mnemonic=mnemonic)

        # C. Build the transaction.
        tx = Transaction()
        for m in msgs:
            tx.add_message(m)
        tx.with_chain_id(chain_id)
        tx.with_fee(gas_limit, f"{fee_amount}{fee_denom}")
        tx.with_memo("DAO governance: add fee-collection cron schedule")
        tx.with_account_num(client.get_account_number(wallet.address()))
        tx.with_sequence(client.get_sequence_number(wallet.address()))

        # D. Sign & broadcast.
        tx.sign(wallet)
        tx_bytes = tx.get_tx_data()
        resp = client.broadcast_tx(tx_bytes)
        if resp.is_err():
            raise RuntimeError(f"Broadcast failed: {resp}")
        return resp.tx_hash
    except Exception as err:
        raise RuntimeError(f"Could not sign and broadcast governance proposal: {err}") from err


# step:1 file: connect_a_signingcosmwasmclient_to_rpc_https:__rpc.juno.strange.love_and_execute_an_increment_on_the_contract
import os
from mnemonic import Mnemonic
from cosmpy.aerial.wallet import LocalWallet


def initialize_wallet() -> LocalWallet:
    """Create (or load) a wallet for subsequent contract execution.

    If `MNEMONIC` is **not** preset, a brand-new 24-word phrase is generated
    and printed to STDOUT so the operator can back it up before any funds are
    deposited.
    """
    m = Mnemonic("english")
    mnemonic = os.getenv("MNEMONIC")
    if mnemonic is None:
        mnemonic = m.generate(24)
        print("[Wallet-Init] Generated NEW mnemonic – back it up NOW:\n", mnemonic)

    # Build the wallet object (derivation path m/44'/118'/0'/0/0 for Cosmos-SDK)
    wallet = LocalWallet.from_mnemonic(mnemonic)
    print(f"[Wallet-Init] Wallet address → {wallet.address()}")
    return wallet


# step:2 file: connect_a_signingcosmwasmclient_to_rpc_https:__rpc.juno.strange.love_and_execute_an_increment_on_the_contract
from cosmpy.aerial.client import LedgerClient, NetworkConfig


RPC_ENDPOINT = "https://rpc.juno.strange.love:443"


def get_ledger_client() -> LedgerClient:
    """Establish a ready-to-use RPC client for the Juno main-net."""
    cfg = NetworkConfig(
        chain_id="juno-1",
        url=RPC_ENDPOINT,
        fee_minimum_gas_price=0.025,  # ujuno
        fee_denomination="ujuno",
        staking_denomination="ujuno"
    )
    return LedgerClient(cfg)


# step:3 file: connect_a_signingcosmwasmclient_to_rpc_https:__rpc.juno.strange.love_and_execute_an_increment_on_the_contract
def build_increment_msg() -> dict:
    """Return the execute payload understood by the counter contract."""
    return {"increment": {}}


# step:4 file: connect_a_signingcosmwasmclient_to_rpc_https:__rpc.juno.strange.love_and_execute_an_increment_on_the_contract
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.contract import MsgExecuteContract
from cosmpy.aerial.exceptions import CosmpyException


def execute_increment(
    client: "LedgerClient",
    wallet: "LocalWallet",
    contract_address: str,
    msg: dict,
):
    """Send the increment execute message and return the tx response."""
    try:
        # Build execute message
        exec_msg = MsgExecuteContract(
            wallet.address(),         # sender
            contract_address,         # cw-contract address
            msg,                      # {"increment": {}}
            funds=[]                  # no attached funds
        )

        # Assemble transaction
        tx = Transaction()
        tx.add_message(exec_msg)
        tx.with_sender(wallet)

        # Broadcast (blocking for inclusion)
        tx_response = client.broadcast_transaction(tx)
        print(f"[Tx] Broadcasted → {tx_response.tx_hash}")
        return tx_response

    except CosmpyException as err:
        # Surface a concise, debuggable error
        raise RuntimeError(f"contract execution failed → {err}") from err


# step:5 file: connect_a_signingcosmwasmclient_to_rpc_https:__rpc.juno.strange.love_and_execute_an_increment_on_the_contract
def verify_increment_event(tx_response) -> bool:
    """Parse ABCI logs and assert that the `increment` action was fired."""
    try:
        for event in tx_response.logs[0].events:
            if event.type == "wasm":
                for attr in event.attributes:
                    if attr.key == "action" and attr.value == "increment":
                        print("[Verify] Increment event detected ✅")
                        return True
        print("[Verify] Increment event NOT found ⚠️")
        return False
    except (AttributeError, IndexError):
        # The log format was not what we expected.
        raise ValueError("Malformed transaction logs — cannot verify execution.")


# step:1 file: configure_log_level_to_state:info,p2p:info,consensus:info,*:error
import os
from pathlib import Path
import tomllib  # Python 3.11+ for TOML parsing


def update_log_level(config_dict: dict, desired_level: str | None = None):
    """Mutates and returns the provided config dict with an updated log_level.

    Args:
        config_dict: Parsed TOML dictionary returned from `open_config_file`.
        desired_level: Custom log level string (defaults to the recommended value).
    Returns:
        dict: The same dictionary instance, now with the new log level set.
    """
    desired_level = desired_level or "state:info,p2p:info,consensus:info,*:error"

    # The `log_level` key lives at the top level of config.toml for most Cosmos SDK daemons.
    # If your daemon nests the setting, update the path accordingly.
    config_dict["log_level"] = desired_level

    return config_dict


# step:3 file: configure_log_level_to_state:info,p2p:info,consensus:info,*:error
try:
    import toml  # External package needed for TOML *writing*
except ModuleNotFoundError as err:
    raise SystemExit("\n[ERROR] Missing dependency: `toml`. Install via `pip install toml` and retry.\n") from err


def save_config_file(config_path: Path, config_dict: dict):
    """Write the updated config dict back to `config_path` in TOML format.

    This will overwrite the existing file, so make sure you have backups
    or version control in place.
    """
    # Create a temporary backup before overriding (best-practice safety net)
    backup_path = config_path.with_suffix(".toml.bak")
    if not backup_path.exists():
        config_path.replace(backup_path)

    # Write the new TOML data
    with config_path.open("w", encoding="utf-8") as fp:
        toml.dump(config_dict, fp)

    print(f"[INFO] Saved updated config to {config_path}")


# step:4 file: configure_log_level_to_state:info,p2p:info,consensus:info,*:error
import subprocess


def restart_node_service(daemon_name: str):
    """Restart the node's systemd service (requires sudo privileges)."""
    service_name = daemon_name  # Usually the same as the binary, adjust if different
    cmd = ["sudo", "systemctl", "restart", service_name]

    try:
        subprocess.run(cmd, check=True)
        print(f"[INFO] Successfully restarted `{service_name}` service.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to restart service `{service_name}`: {exc}. "
            "Check journal logs (journalctl -u {service_name}) for details."
        ) from exc


# step:1 file: manipulate_large_integers_using_ethers.js_bignumber
# big_number.py
from decimal import Decimal, getcontext
from typing import Union


class BigNumber:
    '''A minimal BigNumber implementation using Python Decimal for arbitrary precision arithmetic.'''

    def __init__(self, value: Union[str, int, float]):
        if isinstance(value, (int, float)):
            value = str(value)
        if not isinstance(value, str):
            raise TypeError('BigNumber value must be str, int, or float')
        # Increase precision to handle very large numbers
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
            raise ZeroDivisionError('Division by zero is not allowed.')
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



# step:2 file: manipulate_large_integers_using_ethers.js_bignumber
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from big_number import BigNumber

app = FastAPI()

class CreateRequest(BaseModel):
    value: str

class CreateResponse(BaseModel):
    big_number: str

@app.post('/api/bignumber/create', response_model=CreateResponse)
async def create_bignumber(req: CreateRequest):
    '''Instantiate a BigNumber from a given string, int, or float.'''
    try:
        bn = BigNumber(req.value)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'big_number': str(bn)}



# step:3 file: manipulate_large_integers_using_ethers.js_bignumber
from enum import Enum
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from big_number import BigNumber

app = FastAPI()

class Operation(str, Enum):
    add = 'add'
    sub = 'sub'
    mul = 'mul'
    div = 'div'
    pow = 'pow'

class OperateRequest(BaseModel):
    a: str
    b: Optional[str] = None
    operation: Operation
    exponent: Optional[int] = None

class OperateResponse(BaseModel):
    result: str

@app.post('/api/bignumber/operate', response_model=OperateResponse)
async def operate(req: OperateRequest):
    '''Perform arithmetic operations between BigNumber values or exponentiation.'''
    try:
        a_bn = BigNumber(req.a)
        if req.operation == Operation.pow:
            if req.exponent is None:
                raise HTTPException(status_code=400, detail='Missing exponent for pow operation')
            result_bn = a_bn.pow(req.exponent)
        else:
            if req.b is None:
                raise HTTPException(status_code=400, detail='Parameter b is required for the selected operation')
            b_bn = BigNumber(req.b)
            if req.operation == Operation.add:
                result_bn = a_bn.add(b_bn)
            elif req.operation == Operation.sub:
                result_bn = a_bn.sub(b_bn)
            elif req.operation == Operation.mul:
                result_bn = a_bn.mul(b_bn)
            elif req.operation == Operation.div:
                result_bn = a_bn.div(b_bn)
            else:
                raise HTTPException(status_code=400, detail='Unsupported operation')
    except (TypeError, ValueError, ZeroDivisionError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {'result': str(result_bn)}



# step:4 file: manipulate_large_integers_using_ethers.js_bignumber
from decimal import Decimal, ROUND_DOWN
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from big_number import BigNumber

app = FastAPI()

class FormatResponse(BaseModel):
    formatted: str

@app.get('/api/bignumber/format', response_model=FormatResponse)
async def format_bignumber(value: str = Query(..., description='Raw big number value as string'),
                           decimals: int = Query(18, description='Number of token decimals')):
    '''Convert a raw integer amount into a human-readable decimal representation.'''
    try:
        raw_bn = BigNumber(value)
        divisor_bn = BigNumber(10 ** decimals)
        human_value = raw_bn.div(divisor_bn)
        # Ensure fixed decimal places without scientific notation
        quantize_str = '1.' + '0' * decimals
        formatted = Decimal(str(human_value)).quantize(Decimal(quantize_str), rounding=ROUND_DOWN).normalize()
    except (TypeError, ValueError, ZeroDivisionError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'formatted': format(formatted, 'f')}



# step:2 file: retrieve_projected_ntrn_rewards_based_on_current_point_total
from fastapi import FastAPI, HTTPException
from cosmpy.aerial.client import LedgerClient, NetworkConfig

app = FastAPI()

# Network configuration (replace the RPC URL with your preferred endpoint)
NETWORK = NetworkConfig(
    chain_id='neutron-1',
    url='https://rpc-kralum.neutron.org:443',
    fee_minimum_gas_price=0,
    fee_denomination='untrn',
    staking_denomination='untrn',
)

# Deployed Points contract address
CONTRACT_ADDRESS = 'neutron1yu55umrtnna36vyjvhexp6q2ktljunukzxp9vptsfnylequg7gvqrcqf42'

def _get_client():
    """Instantiate a LedgerClient for each request."""
    return LedgerClient(NETWORK)

@app.get('/api/points')
async def get_user_points(address: str):
    """Return the caller's current point total from the Points contract."""
    try:
        client = _get_client()
        query_msg = {'points': {'address': address}}
        response = client.query_contract_smart(CONTRACT_ADDRESS, query_msg)
        # Expected shape: {'points': '12345'}
        points = int(response.get('points', 0))
        return {'address': address, 'points': points}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:3 file: retrieve_projected_ntrn_rewards_based_on_current_point_total
# This snippet lives in the same FastAPI app defined in Step 2.
from fastapi import HTTPException

# Campaign parameters (micro-denominated values)
REWARD_PARAMS = {
    'ntrn_total_allocation': 100_000_000_000,  # 100,000 NTRN (in untrn)
    'phase_length_seconds': 60 * 60 * 24 * 14,  # 14 days
    'per_point_rate': 1_000_000  # 1 NTRN (1e6 untrn) per point
}

@app.get('/api/reward_params')
async def get_reward_params():
    """Return constants used for reward calculations."""
    try:
        return REWARD_PARAMS
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:4 file: retrieve_projected_ntrn_rewards_based_on_current_point_total
# Endpoint added to the same FastAPI application
from fastapi import HTTPException

@app.get('/api/projection')
async def projected_rewards(address: str):
    """Compute and return projected NTRN rewards for the supplied address."""
    try:
        # 1. Query the user’s point total (reuse logic from Step 2)
        client = _get_client()
        query_msg = {'points': {'address': address}}
        points_response = client.query_contract_smart(CONTRACT_ADDRESS, query_msg)
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


# step:2 file: swap_1_ebtc_for_unibtc_on_neutron_dex
import os
import requests

REST_ENDPOINT = os.getenv('NEUTRON_REST', 'https://rest.neutron.org')


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


# step:3 file: swap_1_ebtc_for_unibtc_on_neutron_dex
import os
import json
import base64
import requests

REST_ENDPOINT = os.getenv('NEUTRON_REST', 'https://rest.neutron.org')
PAIR_CONTRACT = os.getenv('PAIR_CONTRACT', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')  # <-- replace with real pair address


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


# step:5 file: swap_1_ebtc_for_unibtc_on_neutron_dex
import os
import json
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction

RPC_ENDPOINT = os.getenv('NEUTRON_RPC', 'https://rpc.neutron.org')
CHAIN_ID = os.getenv('CHAIN_ID', 'neutron-1')
FEE_DENOM = 'untrn'


def sign_and_broadcast_tx(execute_msg: dict, gas: int = 350_000) -> dict:
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


# step:1 file: Update the Cron module limit parameter to 30 schedules per block
import json

def construct_msg_update_params(authority: str, schedules_per_block: int = 30) -> dict:
    """Return a dict that represents `/neutron.cron.MsgUpdateParams`.

    Args:
        authority (str): The DAO (gov) address that is allowed to change chain params.
        schedules_per_block (int): Desired value for the `schedules_per_block` param.

    Returns:
        dict: JSON-serialisable message ready to be embedded in a proposal.
    """
    # Basic validation -------------------------------------------------------
    if not authority.startswith("neutron"):
        raise ValueError("`authority` must be a valid Neutron bech32 address")
    if schedules_per_block <= 0:
        raise ValueError("`schedules_per_block` must be > 0")

    # Build the message ------------------------------------------------------
    msg = {
        "@type": "/neutron.cron.MsgUpdateParams",
        "authority": authority,
        "params": {
            "schedules_per_block": schedules_per_block
        }
    }
    return msg

# OPTIONAL: pretty-print for audit / persistence
if __name__ == "__main__":
    DAO_ADDR = "neutron1..."  # <- replace with real address
    print(json.dumps(construct_msg_update_params(DAO_ADDR), indent=2))


# step:2 file: Update the Cron module limit parameter to 30 schedules per block
import base64
import json
from typing import Dict

# The cw-dao "propose" message expects Cosmos encoded messages (base64)
# We therefore wrap the Cron MsgUpdateParams in a Cosmos "Any" and then
# base64-encode the final bytes.

def _encode_cosmos_msg(msg: Dict) -> Dict:
    """Helper: builds a `CosmosMsg::Gov`-compatible JSON envelope.

    Because cosmpy (and most clients) accept raw JSON in place of protobuf
    for Custom messages, we simply return the dict itself. If your DAO core
    requires base64-encoded `wasm/MsgExecuteContract`, encode as shown below.
    """
    return msg  # no additional wrapping needed for most cw-dao versions


def build_dao_proposal(msg_update_params: Dict,
                       title: str = "Update Cron schedules_per_block to 30",
                       description: str = "Set cron.schedules_per_block param to 30 via governance.",
                       deposit: str = "1000000untrn",
                       proposer: str | None = None) -> Dict:
    """Return the message to execute against the DAO core contract."""

    if proposer is not None and not proposer.startswith("neutron"):
        raise ValueError("Invalid proposer address")

    proposal = {
        "propose": {
            "title": title,
            "description": description,
            "msgs": [
                {"custom": _encode_cosmos_msg(msg_update_params)}
            ],
            "deposit": deposit
        }
    }
    # Some DAO cores support an explicit `proposer` field
    if proposer:
        proposal["propose"]["proposer"] = proposer

    return proposal

if __name__ == "__main__":
    dao_contract_addr = "neutron1dao..."  # <- your DAO core address
    cron_msg = construct_msg_update_params(authority=dao_contract_addr, schedules_per_block=30)
    proposal_msg = build_dao_proposal(cron_msg)
    print(json.dumps(proposal_msg, indent=2))


# step:3 file: Update the Cron module limit parameter to 30 schedules per block
import asyncio, json, base64
from pathlib import Path
from cosmpy.aerial.client import NetworkConfig, LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction

RPC = "https://rpc.ntrn.tech:443"  # <- public or private RPC endpoint
CHAIN_ID = "neutron-1"
GAS_LIMIT = 400_000
GAS_PRICE = 0.05  # in uNTRN

async def sign_and_broadcast_tx(dao_contract: str,
                                proposal_msg: dict,
                                mnemonic: str) -> str:
    """Signs the DAO `propose` execute-contract message and broadcasts it.

    Args:
        dao_contract (str): cw-dao core contract address.
        proposal_msg (dict): Message produced in Step 2.
        mnemonic (str): Mnemonic for the proposer’s wallet key.

    Returns:
        str: On-chain tx hash.
    """
    # ---------------------------------------------------------------------
    wallet = LocalWallet.from_mnemonic(mnemonic)
    network_cfg = NetworkConfig(chain_id=CHAIN_ID, url=RPC, fee_minimum_gas_price=GAS_PRICE)
    client = LedgerClient(network_cfg)

    # Build MsgExecuteContract ------------------------------------------------
    exec_msg = {
        "sender": wallet.address(),
        "contract": dao_contract,
        "msg": base64.b64encode(json.dumps(proposal_msg).encode()).decode(),
        "funds": []
    }

    tx = (Transaction()
          .with_messages(("/cosmwasm.wasm.v1.MsgExecuteContract", exec_msg))
          .with_signer(wallet)
          .with_chain_id(CHAIN_ID)
          .with_gas(GAS_LIMIT))

    try:
        tx_response = client.broadcast_tx_block(tx)
        if tx_response.is_error:
            raise RuntimeError(f"Tx failed: {tx_response.raw_log}")
        print(f"Broadcasted ✓  txhash={tx_response.tx_hash}")
        return tx_response.tx_hash
    finally:
        client.close()

# Example direct run ---------------------------------------------------------
if __name__ == "__main__":
    dao_addr = "neutron1dao..."
    mnemonic_path = Path.home() / ".dao_keyseed"
    tx_hash = asyncio.run(sign_and_broadcast_tx(dao_addr, proposal_msg, mnemonic_path.read_text().strip()))


# step:4 file: Update the Cron module limit parameter to 30 schedules per block
import asyncio, json
from cosmpy.aerial.client import LedgerClient, NetworkConfig

async def monitor_proposal_status(dao_contract: str, proposal_id: int,
                                  rpc: str = RPC,
                                  chain_id: str = CHAIN_ID,
                                  interval: int = 15):
    """Continuously poll DAO contract for proposal status until finalised."""
    client = LedgerClient(NetworkConfig(chain_id=chain_id, url=rpc))
    try:
        while True:
            try:
                result = client.query_contract_state(dao_contract, {"proposal": {"proposal_id": proposal_id}})
                status = result["proposal"].get("status", "unknown")
                print(f"Proposal {proposal_id} ➜ {status}")
                if status.lower() in {"executed", "rejected", "failed"}:
                    return status
            except Exception as err:
                print(f"query error: {err}")
            await asyncio.sleep(interval)
    finally:
        client.close()

# Usage example --------------------------------------------------------------
# final_status = await monitor_proposal_status(DAO_CORE_ADDR, PROPOSAL_ID)


# step:5 file: Update the Cron module limit parameter to 30 schedules per block
import json, subprocess

def query_cron_params() -> int:
    """Calls `neutrond query cron params` and returns schedules_per_block value.

    Raises:
        RuntimeError: if CLI call fails.
    """
    cmd = [
        "neutrond", "query", "cron", "params",
        "--output", "json"
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())

    data = json.loads(completed.stdout)
    return int(data["params"]["schedules_per_block"])

if __name__ == "__main__":
    current_value = query_cron_params()
    print(f"schedules_per_block = {current_value}")


# step:1 file: broadcast_tx_signed.json_in_sync_mode
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI()

# Default LCD; override per-request if you run your own full node
LCD_URL = "https://lcd.cosmos.network"

class BroadcastRequest(BaseModel):
    tx_bytes: str                 # Base64-encoded signed transaction bytes (TxRaw)
    mode: str = "BROADCAST_MODE_SYNC"   # sync | async | block
    lcd_url: Optional[str] = None         # Optional custom REST endpoint

@app.post("/api/broadcast_tx")
async def broadcast_tx(req: BroadcastRequest):
    """Broadcast a signed tx and return the txhash."""
    lcd = req.lcd_url or LCD_URL
    payload = {"tx_bytes": req.tx_bytes, "mode": req.mode}

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            res = await client.post(f"{lcd}/cosmos/tx/v1beta1/txs", json=payload)
            res.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Failed to broadcast tx: {e}")

    data = res.json()
    txhash = data.get("tx_response", {}).get("txhash")
    if not txhash:
        raise HTTPException(status_code=500, detail="Broadcast response missing txhash")

    return {"txhash": txhash, "raw_response": data}


# step:2 file: broadcast_tx_signed.json_in_sync_mode
import asyncio
from typing import Optional
import httpx
from fastapi import HTTPException

# Re-use the FastAPI `app` instance from Step 1

@app.get("/api/wait_tx/{txhash}")
async def wait_tx(
    txhash: str,
    lcd_url: Optional[str] = None,
    timeout: int = 60,           # seconds before giving up
    interval: float = 2.0        # seconds between polls
):
    """Poll /cosmos/tx/v1beta1/txs/{txhash} until it appears in a block."""
    lcd = lcd_url or LCD_URL
    deadline = asyncio.get_event_loop().time() + timeout

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                res = await client.get(f"{lcd}/cosmos/tx/v1beta1/txs/{txhash}")
                if res.status_code == 200:
                    data = res.json()
                    height = int(data.get("tx_response", {}).get("height", "0"))
                    if height > 0:
                        return data  # Success!
            except httpx.HTTPError:
                # Network hiccup; ignore and retry
                pass

            if asyncio.get_event_loop().time() >= deadline:
                raise HTTPException(status_code=504, detail="Timed out waiting for transaction to be included in a block")

            await asyncio.sleep(interval)


# step:1 file: run_slither_static_analysis_on_a_solidity_project
from fastapi import APIRouter, HTTPException
import shutil
import subprocess
import sys

router = APIRouter()

@router.post("/slither/install")
async def install_slither():
    """
    Ensure that the Slither static-analysis tool is installed on the host.
    Preferred installer is `pipx`; we gracefully fall back to `pip` if pipx
    is not present.  Returns a JSON payload describing the outcome.
    """
    # Early-exit if Slither already exists in $PATH
    if shutil.which("slither"):
        return {"message": "Slither is already installed."}

    try:
        # Decide which installer to use
        if shutil.which("pipx"):
            cmd = ["pipx", "install", "slither-analyzer"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "slither-analyzer"]

        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return {
            "message": "Slither installed successfully.",
            "stdout": completed.stdout,
        }
    except subprocess.CalledProcessError as exc:
        # Surface installer stderr so the caller can debug easily
        raise HTTPException(
            status_code=500,
            detail=f"Failed to install Slither: {exc.stderr}",
        )


# step:2 file: run_slither_static_analysis_on_a_solidity_project
from fastapi import APIRouter, HTTPException
import os
import shutil
import subprocess

router = APIRouter()

@router.post("/slither/compile")
async def compile_project_for_slither():
    """
    Trigger Solidity compilation for the current project directory.  The
    function prefers Hardhat (if `hardhat.config.*` files and `npx` are
    available); otherwise, it attempts a bare-bones `solc` compilation.
    """
    try:
        if shutil.which("npx") and (
            os.path.isfile("hardhat.config.js") or os.path.isfile("hardhat.config.ts")
        ):
            cmd = ["npx", "hardhat", "compile"]
        elif shutil.which("solc"):
            # Example fallback: compile every .sol file under ./contracts
            cmd = [
                "solc",
                "--bin",
                "--abi",
                "--overwrite",
                "-o",
                "build",
                "contracts/*.sol",
            ]
        else:
            raise HTTPException(
                status_code=500,
                detail="Neither Hardhat nor solc found in PATH; cannot compile project.",
            )

        completed = subprocess.run(
            " ".join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return {"message": "Compilation completed.", "stdout": completed.stdout}
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Compilation failed: {exc.stderr}",
        )


# step:3 file: run_slither_static_analysis_on_a_solidity_project
from fastapi import APIRouter, HTTPException
import subprocess

router = APIRouter()

@router.post("/slither/run")
async def run_slither():
    """
    Execute `slither .` in the project root and stream the textual
    vulnerability report back to the caller.
    """
    try:
        completed = subprocess.run(
            ["slither", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return {
            "message": "Slither analysis completed.",
            "stdout": completed.stdout,
        }
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Slither run failed: {exc.stderr}")


# step:4 file: run_slither_static_analysis_on_a_solidity_project
from fastapi import APIRouter, HTTPException
import subprocess
import os

router = APIRouter()

@router.post("/slither/report")
async def generate_slither_report():
    """
    Run `slither . --json slither-report.json` and verify that the report
    file exists before returning.  The JSON contents are *not* inlined in the
    HTTP response to avoid large payloads; only the filepath is returned.
    """
    report_file = "slither-report.json"
    try:
        cmd = ["slither", ".", "--json", report_file]
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if not os.path.isfile(report_file):
            raise HTTPException(
                status_code=500,
                detail="Slither did not produce the expected JSON report.",
            )
        return {
            "message": "JSON report generated successfully.",
            "stdout": completed.stdout,
            "report_file": report_file,
        }
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Slither JSON report: {exc.stderr}",
        )


# step:3 file: delegate_500stake_from_recipient_to_my_validator_validator
import os
from cosmpy.aerial.client import LCDClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmos.staking.v1beta1.tx_pb2 import MsgDelegate
from google.protobuf.any_pb2 import Any

# --- chain configuration (override with env if desired) ---
LCD_ENDPOINT = os.getenv('LCD_ENDPOINT', 'https://lcd.cosmos.directory/gaia')
CHAIN_ID    = os.getenv('CHAIN_ID',    'cosmoshub-4')
DENOM       = os.getenv('DENOM',       'stake')

network_cfg = NetworkConfig(
    chain_id=CHAIN_ID,
    url=LCD_ENDPOINT,
    fee_minimum_gas_price=0.025,
    fee_denomination=DENOM,
    staking_denomination=DENOM,
)

lcd = LCDClient(network_cfg)

def construct_delegate_tx(delegator: str, validator: str, amount: int) -> Transaction:
    """Return an unsigned Transaction object carrying MsgDelegate."""
    coin = {"denom": DENOM, "amount": str(amount)}
    msg = MsgDelegate(
        delegator_address = delegator,
        validator_address = validator,
        amount            = coin,
    )

    # Pack into protobuf Any so the tx can hold heterogeneous messages
    any_msg = Any()
    any_msg.Pack(msg, type_url_prefix='/')

    tx = Transaction()
    tx.add_message(any_msg)
    tx.seal_network_info(network_cfg)  # embeds chain-id etc.
    return tx


# step:4 file: delegate_500stake_from_recipient_to_my_validator_validator
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.client.utils import prepare_and_broadcast

from tx_delegate_builder import construct_delegate_tx, lcd  # Step-3 helper

app = FastAPI()

class DelegateRequest(BaseModel):
    delegator: str
    validator: str
    amount: int  # 500 (stake) in smallest units

@app.post('/api/delegate')
async def delegate(req: DelegateRequest):
    try:
        pk_hex = os.getenv('DELEGATOR_PRIVATE_KEY_HEX')
        if not pk_hex:
            raise ValueError('Backend mis-configuration: set DELEGATOR_PRIVATE_KEY_HEX env var')

        priv_key = PrivateKey.from_hex(pk_hex)
        tx = construct_delegate_tx(req.delegator, req.validator, req.amount)

        # Sign & broadcast in one go
        result = prepare_and_broadcast(lcd, tx, priv_key)

        if result.code != 0:
            raise RuntimeError(f"Tx failed (code {result.code}): {result.raw_log}")

        return {"tx_hash": result.txhash}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# step:1 file: execute_full-app_simulation_fuzz_testing
import os
from typing import Optional, Dict


def prepare_simulation_env(random_seed: Optional[int] = None) -> Dict[str, Optional[str]]:
    """
    Sets environment variables required for Cosmos SDK simulation tests.

    Args:
        random_seed (Optional[int]): Fixed seed to reproduce a run. If None, any
            previously-set SIMAPP_RANDOM_SEED is removed so the test harness can
            choose a random seed.

    Returns:
        Dict[str, Optional[str]]: Snapshot of key environment variables after
            mutation so callers can log them.
    """
    try:
        # Ensure $GOFLAGS is completely unset – it can change go-test behaviour.
        os.environ.pop("GOFLAGS", None)

        if random_seed is not None:
            os.environ["SIMAPP_RANDOM_SEED"] = str(random_seed)
        else:
            os.environ.pop("SIMAPP_RANDOM_SEED", None)

        return {
            "SIMAPP_RANDOM_SEED": os.getenv("SIMAPP_RANDOM_SEED"),
            "GOFLAGS": os.getenv("GOFLAGS"),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to prepare simulation environment: {e}") from e


# step:2 file: execute_full-app_simulation_fuzz_testing
import subprocess
import shlex
import os
from typing import List, Tuple


def run_simulation_tests(command: str = "make test-sim-nondeterminism") -> Tuple[int, List[str]]:
    """
    Runs Cosmos SDK simulation tests and captures stdout/stderr.

    Args:
        command (str): Shell command to run. Defaults to the Make target that
            wraps `go test ./sim/... -run TestFullAppSimulation -v`.

    Returns:
        Tuple[int, List[str]]: (exit_code, list_of_output_lines)
    """
    try:
        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ,     # inherits SIMAPP_RANDOM_SEED, GOFLAGS unset
            text=True,
            bufsize=1,
        )

        output_lines: List[str] = []
        for line in process.stdout:
            print(line, end="")           # realtime feedback in CI/logs
            output_lines.append(line.rstrip("\n"))

        process.wait()
        return process.returncode, output_lines
    except FileNotFoundError as e:
        raise RuntimeError(f"Simulation command not found: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error running simulation tests: {e}") from e


# step:3 file: execute_full-app_simulation_fuzz_testing
import re
from typing import List, Dict


def parse_simulation_output(output_lines: List[str]) -> List[Dict[str, int]]:
    """
    Extracts seed and block-height information printed by the test runner.

    Looks for lines such as:
        --- RUN   TestFullAppSimulation_Seed=169314843_Blocks=500
    or separate `Seed:` / `Block height:` lines.

    Args:
        output_lines (List[str]): Captured stdout/stderr lines.

    Returns:
        List[Dict[str, int]]: One dict per simulation, e.g. [{"seed": 169314843,
            "blocks": 500}].
    """
    pattern = re.compile(r".*Seed[=:\s](\d+).*(?:Blocks?|Block height)[=:\s](\d+)")
    results: List[Dict[str, int]] = []

    for line in output_lines:
        match = pattern.match(line)
        if match:
            seed = int(match.group(1))
            blocks = int(match.group(2))
            results.append({"seed": seed, "blocks": blocks})

    return results


# step:4 file: execute_full-app_simulation_fuzz_testing
def verify_simulation_results(exit_code: int, output_lines: list) -> None:
    """
    Verifies success criteria for Cosmos SDK simulations.

    Raises RuntimeError if any check fails.
    """
    if exit_code != 0:
        raise RuntimeError(f"Simulation tests terminated with exit code {exit_code}.")

    log_blob = "\n".join(output_lines).lower()

    # Fail fast if any panic occurred.
    if "panic" in log_blob:
        raise RuntimeError("'panic' detected in simulation logs.")

    # Check invariants: allow either explicit success line or absence of failures.
    if "invariants broken" in log_blob and "invariants broken: 0" not in log_blob:
        raise RuntimeError("Invariants were broken during simulation.")

    print("✅ All simulations passed without panics and with invariants intact.")


# step:1 file: allow_p2p_port_26656_through_ufw
import subprocess


def allow_cosmos_p2p_port() -> dict:
    """Allow TCP/26656 through UFW with the comment 'Cosmos P2P'.
    NOTE: This function must be executed with root privileges (e.g. the
    backend process itself runs as root or via sudo in a privileged
    execution environment)."""

    cmd = [
        "sudo",          # ensure the command runs with elevated privileges
        "ufw",
        "allow",
        "26656/tcp",
        "comment",
        "Cosmos P2P"
    ]

    try:
        # capture_output=True lets us read stdout/stderr for debugging
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True  # raises CalledProcessError when return code != 0
        )
        return {
            "status": "success",
            "message": result.stdout.strip() or "Port 26656 opened successfully."
        }

    except subprocess.CalledProcessError as err:
        # Standardized error structure for the frontend to consume
        raise RuntimeError(
            f"Failed to add UFW rule: {err.stderr.strip()}"
        )


# step:2 file: allow_p2p_port_26656_through_ufw
import subprocess


def reload_ufw() -> dict:
    """Reload UFW to apply pending firewall rule changes.
    Must be executed with root privileges."""

    cmd = ["sudo", "ufw", "reload"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return {
            "status": "success",
            "message": result.stdout.strip() or "UFW reloaded successfully."
        }

    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Failed to reload UFW: {err.stderr.strip()}"
        )


# step:2 file: query_allbalances_for_an_address_at_block_height_123_via_grpc
# backend/bank_client.py
import os
from typing import Optional

import grpc
from google.protobuf.json_format import MessageToDict

# Cosmos-SDK protobuf stubs (install with `pip install cosmos-sdk-proto`)
from cosmos.bank.v1beta1 import query_pb2 as bank_query_pb2
from cosmos.bank.v1beta1 import query_pb2_grpc as bank_query_grpc

GRPC_ENDPOINT = os.getenv('GRPC_ENDPOINT', 'localhost:9090')

def grpc_bank_all_balances(address: str, height: Optional[int] = None) -> dict:
    """Return the gRPC QueryAllBalancesResponse as a plain Python dict."""
    channel = grpc.insecure_channel(GRPC_ENDPOINT)
    stub = bank_query_grpc.QueryStub(channel)

    request = bank_query_pb2.QueryAllBalancesRequest(address=address)

    metadata = []
    if height is not None:
        metadata.append(('x-cosmos-block-height', str(height)))

    try:
        response_proto = stub.AllBalances(request, metadata=metadata, timeout=10)
        return MessageToDict(response_proto, preserving_proto_field_name=True)
    except grpc.RpcError as rpc_err:
        # Convert low-level gRPC errors into generic exceptions for the HTTP layer
        raise RuntimeError(f'gRPC query failed: {rpc_err.details()} (code={rpc_err.code()})') from rpc_err

# -----------------------------------------------------------------------------
# FastAPI wrapper so the frontend can fetch with a simple HTTP call
# -----------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Query

app = FastAPI()

@app.get('/api/balances')
def api_all_balances(
    address: str = Query(..., description='Bech32 account address'),
    height: Optional[int] = Query(None, description='Optional block height')):
    try:
        return grpc_bank_all_balances(address, height)
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:3 file: query_allbalances_for_an_address_at_block_height_123_via_grpc
# backend/parsers.py
def parse_balances_response(response: dict):
    """Extracts a simple list of {denom, amount} from the gRPC response dict."""
    try:
        balances = response.get('balances', [])
        return [{'denom': coin['denom'], 'amount': coin['amount']} for coin in balances]
    except (AttributeError, KeyError, TypeError) as err:
        raise ValueError('Malformed balances response') from err


# step:1 file: upload,_instantiate,_and_increment_a_counter_contract_using_cw-orchestrator
from dotenv import load_dotenv
import os

# config.py

def get_chain_config():
    """Read required environment variables and return them in a dict."""
    load_dotenv()
    rpc_endpoint = os.getenv("RPC_ENDPOINT")
    chain_id = os.getenv("CHAIN_ID")
    mnemonic = os.getenv("DEPLOYER_MNEMONIC")
    gas_prices = os.getenv("GAS_PRICES", "0.025untrn")  # sensible default

    # Basic validation
    if not rpc_endpoint or not chain_id or not mnemonic:
        raise EnvironmentError(
            "RPC_ENDPOINT, CHAIN_ID and DEPLOYER_MNEMONIC must be set in .env"
        )

    return {
        "rpc_endpoint": rpc_endpoint,
        "chain_id": chain_id,
        "mnemonic": mnemonic,
        "gas_prices": gas_prices,
    }

if __name__ == "__main__":
    # Quick manual test
    print(get_chain_config())


# step:2 file: upload,_instantiate,_and_increment_a_counter_contract_using_cw-orchestrator
import subprocess
from pathlib import Path

# build.py

def build_and_optimize(contract_root: str = ".") -> dict:
    """Run `cargo wasm` followed by the rust-optimizer docker image."""
    root = Path(contract_root).resolve()

    try:
        # 1. Raw wasm build (debug, un-optimized)
        subprocess.run(["cargo", "wasm"], check=True, cwd=root)

        # 2. Run optimizer docker image – output goes to <root>/artifacts/*.wasm
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{root}:/code",
                "cosmtrek/wasm:optimizer",
                ".",
            ],
            check=True,
        )

        # Pick the most recent optimized wasm file
        artifacts_dir = root / "artifacts"
        wasm_files = sorted(artifacts_dir.glob("*.wasm"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not wasm_files:
            raise FileNotFoundError("No optimized .wasm found in artifacts/")

        return {"wasm_path": str(wasm_files[0])}

    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Contract build failed: {err}")

if __name__ == "__main__":
    print(build_and_optimize())


# step:3 file: upload,_instantiate,_and_increment_a_counter_contract_using_cw-orchestrator
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction
from cosmpy.crypto.address import Address
from config import get_chain_config

# store_code.py

def store_code(wasm_path: str) -> dict:
    cfg = get_chain_config()

    # Prepare client & wallet
    network_cfg = NetworkConfig(
        chain_id=cfg["chain_id"],
        url=cfg["rpc_endpoint"],
        fee_minimum_gas_price=cfg["gas_prices"],
    )
    client = LedgerClient(network_cfg)
    wallet = LocalWallet.create_from_mnemonic(cfg["mnemonic"])

    # Build & send StoreCode tx
    with open(wasm_path, "rb") as fp:
        wasm_bytes = fp.read()

    tx = Transaction(client=client, wallet=wallet)
    tx.add_message(tx.MsgStoreCode(wallet.address(), wasm_bytes))

    try:
        result = tx.broadcast()
    except Exception as e:
        raise RuntimeError(f"Failed to store code: {e}")

    # Extract code_id from tx logs
    try:
        code_id = int(result.logs[0].events_by_type["store_code"]["code_id"][0])
    except (KeyError, IndexError, ValueError):
        raise RuntimeError("Could not parse code_id from broadcast result")

    return {"code_id": code_id, "tx_hash": result.tx_hash}

if __name__ == "__main__":
    print(store_code("artifacts/counter.wasm"))


# step:4 file: upload,_instantiate,_and_increment_a_counter_contract_using_cw-orchestrator
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction
from config import get_chain_config

# instantiate.py

def instantiate_contract(code_id: int, label: str = "counter", init_msg: dict | None = None) -> dict:
    cfg = get_chain_config()
    init_msg = init_msg or {"start": 0}

    network_cfg = NetworkConfig(
        chain_id=cfg["chain_id"],
        url=cfg["rpc_endpoint"],
        fee_minimum_gas_price=cfg["gas_prices"],
    )
    client = LedgerClient(network_cfg)
    wallet = LocalWallet.create_from_mnemonic(cfg["mnemonic"])

    tx = Transaction(client=client, wallet=wallet)
    tx.add_message(
        tx.MsgInstantiateContract(
            sender=wallet.address(),
            admin=None,  # no admin
            code_id=code_id,
            msg=init_msg,
            funds=[],
            label=label,
        )
    )

    try:
        result = tx.broadcast()
    except Exception as e:
        raise RuntimeError(f"Instantiation failed: {e}")

    try:
        contract_addr = result.logs[0].events_by_type["instantiate"]["_contract_address"][0]
    except (KeyError, IndexError):
        raise RuntimeError("Contract address not found in tx logs")

    return {"contract_addr": contract_addr, "tx_hash": result.tx_hash}

if __name__ == "__main__":
    print(instantiate_contract(17))


# step:5 file: upload,_instantiate,_and_increment_a_counter_contract_using_cw-orchestrator
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction
from config import get_chain_config

# execute.py

def increment(contract_addr: str) -> dict:
    cfg = get_chain_config()

    network_cfg = NetworkConfig(
        chain_id=cfg["chain_id"],
        url=cfg["rpc_endpoint"],
        fee_minimum_gas_price=cfg["gas_prices"],
    )
    client = LedgerClient(network_cfg)
    wallet = LocalWallet.create_from_mnemonic(cfg["mnemonic"])

    tx = Transaction(client=client, wallet=wallet)
    tx.add_message(
        tx.MsgExecuteContract(
            sender=wallet.address(),
            contract=contract_addr,
            msg={"increment": {}},
            funds=[],
        )
    )

    try:
        result = tx.broadcast()
    except Exception as e:
        raise RuntimeError(f"Execute failed: {e}")

    return {"tx_hash": result.tx_hash}

if __name__ == "__main__":
    print(increment("neutron1..."))


# step:6 file: upload,_instantiate,_and_increment_a_counter_contract_using_cw-orchestrator
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from config import get_chain_config

# query.py

def get_count(contract_addr: str) -> dict:
    cfg = get_chain_config()

    client = LedgerClient(
        NetworkConfig(
            chain_id=cfg["chain_id"],
            url=cfg["rpc_endpoint"],
            fee_minimum_gas_price=cfg["gas_prices"],
        )
    )

    try:
        # CosmWasm smart query
        response = client.wasm_query(contract_addr, {"get_count": {}})
    except Exception as e:
        raise RuntimeError(f"Query failed: {e}")

    return {"count": response.get("count")}

if __name__ == "__main__":
    print(get_count("neutron1..."))


# step:1 file: add_a_cosmos_evm_network_to_metamask
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()


def _evmos_network_config() -> dict:
    """Return a dict that follows MetaMask’s `wallet_addEthereumChain` spec."""
    return {
        "chainId": "0x2329",  # 9001 in hex
        "chainName": "Evmos",
        "nativeCurrency": {
            "name": "Evmos",
            "symbol": "EVMOS",
            "decimals": 18
        },
        "rpcUrls": [
            "https://eth.bd.evmos.org:8545"
        ],
        "blockExplorerUrls": [
            "https://escan.live"
        ]
    }


@app.get("/api/network/evmos")
async def get_network_config():
    """GET /api/network/evmos → JSON network parameters."""
    try:
        return JSONResponse(_evmos_network_config())
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:1 file: Create a schedule to distribute staking rewards weekly
import requests

def get_cron_authority(lcd_endpoint: str) -> str:
    """Return the Cron module authority address (e.g. the Main DAO address)."""
    try:
        url = f"{lcd_endpoint}/neutron/cron/v1/params"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # The expected JSON shape is: {"params": {"authority": "neutron1..."}}
        return response.json()["params"]["authority"]
    except (requests.RequestException, KeyError) as err:
        raise RuntimeError(f"Unable to fetch Cron authority: {err}")


# step:2 file: Create a schedule to distribute staking rewards weekly
import requests

def validate_contract(address: str, lcd_endpoint: str) -> bool:
    """Return True when the contract exists and is instantiated."""
    try:
        url = f"{lcd_endpoint}/cosmwasm/wasm/v1/contract/{address}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        info = response.json().get("contract_info", {})
        # Minimal sanity-check: the endpoint echoes back the queried address
        return info.get("address") == address
    except requests.RequestException:
        return False


# step:3 file: Create a schedule to distribute staking rewards weekly
import json, base64

def build_msg_execute_contract(staking_contract: str, cron_sender: str = "cron") -> dict:
    """Return a MsgExecuteContract dict compatible with protobuf/CLI JSON."""
    inner_msg = {"distribute_rewards": {}}
    return {
        "@type": "/cosmwasm.wasm.v1.MsgExecuteContract",
        "sender": cron_sender,
        "contract": staking_contract,
        "msg": base64.b64encode(json.dumps(inner_msg).encode()).decode(),  # base64-encoded
        "funds": []
    }


# step:4 file: Create a schedule to distribute staking rewards weekly
def build_msg_add_schedule(authority: str, execute_msg: dict) -> dict:
    """Create a MsgAddSchedule that invokes the provided execute message every 100 800 blocks (~7 days)."""
    return {
        "@type": "/neutron.cron.MsgAddSchedule",
        "name": "weekly_staking_rewards",
        "period": "100800",  # 7 days at 6-second blocks
        "msgs": [execute_msg],
        "execution_stage": "EXECUTION_STAGE_END_BLOCKER",  # default
        "authority": authority
    }


# step:5 file: Create a schedule to distribute staking rewards weekly
import json

def write_proposal_file(msg_add_schedule: dict, filename: str = "proposal.json") -> str:
    """Write the governance proposal to disk and return the file name."""
    proposal = {
        "title": "Add weekly staking-reward cron",
        "description": "Distribute staking rewards every week automatically",
        "messages": [msg_add_schedule]
    }
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(proposal, fp, indent=2)
    return filename


# step:6 file: Create a schedule to distribute staking rewards weekly
import subprocess

def submit_proposal(file_path: str, from_key: str, chain_id: str, node: str) -> None:
    """Call neutrond CLI to submit the proposal for voting."""
    cmd = [
        "neutrond", "tx", "wasm", "submit-proposal", file_path,
        "--from", from_key,
        "--chain-id", chain_id,
        "--node", node,
        "--gas", "auto",
        "--gas-adjustment", "1.3",
        "-y"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Proposal submission failed: {err}")


# step:7 file: Create a schedule to distribute staking rewards weekly
import requests, time

def wait_for_proposal_passage(proposal_id: int, lcd_endpoint: str, poll: int = 15, timeout: int = 3600) -> None:
    """Block until the proposal is PASSED or raise if REJECTED/EXPIRED/timeout."""
    deadline = time.time() + timeout
    gov_url = f"{lcd_endpoint}/cosmos/gov/v1/proposals/{proposal_id}"
    while time.time() < deadline:
        response = requests.get(gov_url, timeout=10)
        response.raise_for_status()
        status = int(response.json()["proposal"]["status"])
        if status == 3:  # PASSED
            print(f"✅  Proposal {proposal_id} PASSED")
            return
        if status in (4, 5):  # REJECTED or FAILED
            raise RuntimeError(f"❌  Proposal {proposal_id} failed with status {status}")
        print(f"⏳  Waiting... current status = {status}")
        time.sleep(poll)
    raise TimeoutError("Timed out waiting for proposal to pass")


# step:8 file: Create a schedule to distribute staking rewards weekly
import requests

def query_cron_schedule(name: str, lcd_endpoint: str) -> dict:
    """Return the on-chain definition of the given Cron schedule name."""
    url = f"{lcd_endpoint}/neutron/cron/v1/schedule/{name}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


# step:1 file: enable_the_txpool_json-rpc_namespace_on_a_cosmos_evm_node
import subprocess


def stop_evmd_service(service_name: str = "evmd") -> dict:
    """Stops the evmd (systemd) service gracefully."""
    try:
        # Attempt to stop the service
        subprocess.run(["systemctl", "stop", service_name], check=True)
        return {"service": service_name, "status": "stopped"}
    except subprocess.CalledProcessError as e:
        # Convert the low-level error into a clearer Python exception
        raise RuntimeError(f"Failed to stop {service_name}: {e}")


# step:2 file: enable_the_txpool_json-rpc_namespace_on_a_cosmos_evm_node
from pathlib import Path


def load_app_toml(config_path: str = "~/.evmd/config/app.toml") -> dict:
    """Reads the app.toml file and returns its contents along with the absolute path."""
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        content = f.read()

    return {"path": str(path), "content": content}


# step:3 file: enable_the_txpool_json-rpc_namespace_on_a_cosmos_evm_node
import re
from pathlib import Path


def _replace_or_add(pattern: str, replacement: str, text: str) -> str:
    """Replace a line that matches pattern or add replacement inside [json-rpc] section."""
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)

    # If the key is missing, insert it after the [json-rpc] header
    lines = text.splitlines()
    updated_lines = []
    in_json_rpc = False

    for idx, line in enumerate(lines):
        updated_lines.append(line)
        if line.strip().startswith("[json-rpc]"):
            in_json_rpc = True
            continue
        # Insert when we exit the [json-rpc] section
        if in_json_rpc and line.strip().startswith("[") and not line.strip().startswith("[json-rpc]"):
            updated_lines.insert(len(updated_lines) - 1, replacement)
            in_json_rpc = False
    return "\n".join(updated_lines)


def update_app_toml_parameter(config_path: str = "~/.evmd/config/app.toml") -> dict:
    """Mutates app.toml in-place to enable JSON-RPC & txpool along with indexing."""
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    raw = path.read_text(encoding="utf-8")

    # Apply the required replacements or insertions
    updated = raw
    updated = _replace_or_add(r"^enable\s*=.*$", "enable = true", updated)
    updated = _replace_or_add(r"^api\s*=.*$", "api = \"eth,net,web3,txpool,debug\"", updated)
    updated = _replace_or_add(r"^enable-indexer\s*=.*$", "enable-indexer = true", updated)

    # Write only if something changed
    changed = updated != raw
    if changed:
        path.write_text(updated, encoding="utf-8")

    return {"path": str(path), "changed": changed}


# step:4 file: enable_the_txpool_json-rpc_namespace_on_a_cosmos_evm_node
from pathlib import Path


def verify_app_toml(config_path: str = "~/.evmd/config/app.toml") -> bool:
    """Returns True if enable=true, txpool exists in api, and enable-indexer=true."""
    path = Path(config_path).expanduser()
    text = path.read_text(encoding="utf-8")
    return all([
        "enable = true" in text,
        "txpool" in text,
        "enable-indexer = true" in text,
    ])


# step:5 file: enable_the_txpool_json-rpc_namespace_on_a_cosmos_evm_node
import subprocess


def start_evmd_service(service_name: str = "evmd") -> dict:
    """Starts evmd and returns the latest logs to confirm successful boot."""
    try:
        subprocess.run(["systemctl", "start", service_name], check=True)
        logs = subprocess.check_output([
            "journalctl", "-u", service_name, "-n", "20", "--no-pager"
        ], text=True)
        return {"service": service_name, "status": "started", "logs": logs}
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to start {service_name}: {e}")


# step:6 file: enable_the_txpool_json-rpc_namespace_on_a_cosmos_evm_node
import requests


def json_rpc_call(method: str = "txpool_status", params: list | None = None, endpoint: str = "http://localhost:8545") -> dict:
    """Executes a JSON-RPC call and returns the result payload."""
    if params is None:
        params = []

    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"RPC Error: {data['error']}")
        return data.get("result")
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Failed to perform JSON-RPC call: {e}")


# step:3 file: execute_a_reset_on_the_contract_at_contract_address,_setting_count_to_0
# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, json
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.contract import MsgExecuteContract
from cosmpy.aerial.tx import Transaction

app = FastAPI()

# ---------------- Chain / Wallet configuration ----------------
CHAIN_ID = os.getenv("CHAIN_ID", "neutron-1")
RPC_ENDPOINT = os.getenv("RPC_ENDPOINT", "https://rpc.neutron-1.neutron.org:443")
GAS_PRICE = os.getenv("GAS_PRICE", "0.0025untrn")
MNEMONIC = os.getenv("EXECUTOR_MNEMONIC")  # keep this secret!
if MNEMONIC is None:
    raise RuntimeError("EXECUTOR_MNEMONIC environment variable not set")

network_cfg = NetworkConfig(
    chain_id=CHAIN_ID,
    url=RPC_ENDPOINT,
    fee_minimum_gas_price=GAS_PRICE,
    fee_denomination=GAS_PRICE.lstrip("0123456789.")  # naive extraction
)
client = LedgerClient(network_cfg)
wallet = LocalWallet(MNEMONIC)

# ---------------- Pydantic models ----------------
class ExecuteResetPayload(BaseModel):
    contract_address: str
    msg: dict                      # e.g. {"reset": {"count": 0}}
    gas: str | None = "auto"        # "auto" or an explicit integer string
    fees: str | None = None         # e.g. "5000untrn"
    memo: str | None = None

# ---------------- Routes ----------------
@app.get("/api/executor_address")
def executor_address():
    """Returns the address of the server-side signer used for transactions."""
    return {"address": wallet.address()}

@app.post("/api/execute_reset")
def execute_reset(payload: ExecuteResetPayload):
    """Signs and broadcasts a MsgExecuteContract that resets the counter."""
    try:
        execute_msg = MsgExecuteContract(
            sender=wallet.address(),
            contract=payload.contract_address,
            msg=json.dumps(payload.msg).encode(),  # cosmpy expects bytes
            funds=[]
        )

        tx = (Transaction()
              .with_messages(execute_msg)
              .with_chain_id(CHAIN_ID)
              .with_sequence(client.get_sequence(wallet.address()))
              .with_account_num(client.get_number(wallet.address())))

        if payload.gas != "auto":
            tx = tx.with_gas(int(payload.gas))
        if payload.fees:
            tx = tx.with_fee(payload.fees)
        if payload.memo:
            tx = tx.with_memo(payload.memo)

        signed_tx = tx.sign(wallet)
        res = client.broadcast_tx_block(signed_tx)

        if res.is_successful():
            return {
                "tx_hash": res.tx_hash,
                "height": res.height,
                "raw_log": res.raw_log
            }
        raise HTTPException(status_code=400, detail=res.raw_log)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:4 file: execute_a_reset_on_the_contract_at_contract_address,_setting_count_to_0
# backend/main.py  (continued)

from cosmpy.aerial.client import NetworkError

@app.get("/api/tx_status/{tx_hash}")
def tx_status(tx_hash: str):
    """Returns confirmation details for the supplied transaction hash."""
    try:
        tx_info = client.get_tx(tx_hash)
        if tx_info is None:
            return {"confirmed": False, "tx_hash": tx_hash}
        return {
            "confirmed": True,
            "tx_hash": tx_hash,
            "height": tx_info.height,
            "raw_log": tx_info.raw_log
        }
    except NetworkError as ne:
        raise HTTPException(status_code=503, detail=str(ne))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:1 file: update_the_on-chain_voting_period_in_genesis_to_600_s
import os


def get_chain_home() -> str:
    """
    Resolve the simd chain’s home directory.

    Order of precedence:
      1. Environment variable `SIMD_HOME`.
      2. Default path `~/.simapp`.
    """
    chain_home = os.getenv("SIMD_HOME", os.path.expanduser("~/.simapp"))
    if not os.path.isdir(chain_home):
        raise FileNotFoundError(f"Chain home directory not found: {chain_home}")
    return chain_home



# step:2 file: update_the_on-chain_voting_period_in_genesis_to_600_s
import os


def locate_genesis_file(chain_home: str) -> str:
    """Return the absolute path to config/genesis.json and verify its existence."""
    genesis_path = os.path.join(chain_home, "config", "genesis.json")
    if not os.path.isfile(genesis_path):
        raise FileNotFoundError(f"genesis.json not found at {genesis_path}")
    return genesis_path



# step:3 file: update_the_on-chain_voting_period_in_genesis_to_600_s
import shutil
import os


def backup_genesis(genesis_path: str) -> str:
    """Copy genesis.json to genesis.json.bak in the same directory."""
    backup_path = genesis_path + ".bak"
    shutil.copy2(genesis_path, backup_path)
    if not os.path.isfile(backup_path):
        raise IOError("Failed to create backup file")
    return backup_path



# step:4 file: update_the_on-chain_voting_period_in_genesis_to_600_s
import json
from typing import Dict


def update_voting_period(genesis_path: str, new_period: str = "600s") -> Dict:
    """
    Modify `voting_period` in-place within the genesis JSON structure.
    Handles both:
      • app_state.gov.params.voting_period  (newer SDK)
      • app_state.gov.voting_params.voting_period  (older SDK)
    Returns the updated Python dict (not yet persisted).
    """
    with open(genesis_path, "r", encoding="utf-8") as fp:
        genesis = json.load(fp)

    app_state = genesis.get("app_state", {})
    gov_state = app_state.get("gov", {})
    updated = False

    # Newer SDK layout
    params = gov_state.get("params", {})
    if "voting_period" in params:
        params["voting_period"] = new_period
        updated = True

    # Older SDK layout
    voting_params = gov_state.get("voting_params", {})
    if "voting_period" in voting_params:
        voting_params["voting_period"] = new_period
        updated = True

    if not updated:
        raise KeyError("voting_period field not found in genesis.json")

    return genesis



# step:5 file: update_the_on-chain_voting_period_in_genesis_to_600_s
import json
import os
import tempfile


def save_genesis(genesis_obj: dict, genesis_path: str) -> None:
    """
    Atomically write out the updated genesis.json by first writing to a temp file
    and then replacing the original.
    """
    dir_name = os.path.dirname(genesis_path)
    with tempfile.NamedTemporaryFile(mode="w", dir=dir_name, delete=False, encoding="utf-8") as tmp:
        json.dump(genesis_obj, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, genesis_path)



# step:6 file: update_the_on-chain_voting_period_in_genesis_to_600_s
import subprocess


def validate_genesis(chain_home: str) -> str:
    """Execute `simd validate-genesis` and return its stdout on success."""
    try:
        completed = subprocess.run(
            ["simd", "validate-genesis", "--home", chain_home],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return completed.stdout
    except subprocess.CalledProcessError as err:
        # Surface compiler or JSON errors cleanly
        error_output = err.stderr or err.stdout
        raise RuntimeError(f"validate-genesis failed: {error_output.strip()}") from err



# step:1 file: set_custom_inflation_minting_function
# step1.py
import os
import shutil
import logging
from pathlib import Path


def fork_mint_module(project_root: str, new_module_name: str = 'custommint') -> str:
    # Copy Cosmos-SDK mint module into x/<new_module_name> and rename packages
    mint_src = Path(project_root) / 'x' / 'mint'
    mint_dst = Path(project_root) / 'x' / new_module_name

    if not mint_src.exists():
        raise FileNotFoundError(f'Original mint module not found at {mint_src}')

    if mint_dst.exists():
        logging.warning('Destination %s already exists. Overwriting...', mint_dst)
        shutil.rmtree(mint_dst)

    shutil.copytree(mint_src, mint_dst)

    # Update package declarations inside .go files
    for go_file in mint_dst.rglob('*.go'):
        content = go_file.read_text()
        content = content.replace('package mint', f'package {new_module_name}')
        go_file.write_text(content)

    logging.info('Forked mint module into %s', mint_dst)
    return str(mint_dst)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fork Cosmos-SDK mint module')
    parser.add_argument('--project_root', required=True, help='Path to the chain source root')
    parser.add_argument('--module_name', default='custommint')
    args = parser.parse_args()
    fork_mint_module(args.project_root, args.module_name)



# step:2 file: set_custom_inflation_minting_function
# step2.py
import re
import logging
from pathlib import Path


def implement_custom_inflation_logic(project_root: str, new_module_name: str = 'custommint') -> None:
    # Inject a new exponential-decay inflation model into x/<new_module_name>/inflation.go
    inflation_file = Path(project_root) / 'x' / new_module_name / 'inflation.go'
    if not inflation_file.exists():
        raise FileNotFoundError(inflation_file)

    content = inflation_file.read_text()
    pattern = r'func\s+CalculateInflation[\s\S]*?\}'
    custom_logic = (
        '// CalculateInflation replaces the default Cosmos-SDK logic with an exponential\n'
        '// decay model.\n\n'
        'func CalculateInflation(params types.Params, firstBlockTime time.Time, blockTime time.Time) sdk.Dec {\n'
        '    epochs := int(blockTime.Sub(firstBlockTime).Hours() / (24 * 365))\n'
        '    initialInflation := sdk.NewDecWithPrec(20, 2) // 0.20\n'
        '    decayFactor := sdk.NewDecWithPrec(98, 2)      // 0.98\n\n'
        '    pow := sdk.OneDec()\n'
        '    for i := 0; i < epochs; i++ {\n'
        '        pow = pow.Mul(decayFactor)\n'
        '    }\n'
        '    return initialInflation.Mul(pow)\n'
        '}')

    new_content = re.sub(pattern, custom_logic, content, flags=re.MULTILINE)
    inflation_file.write_text(new_content)
    logging.info('Custom inflation logic written to %s', inflation_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inject custom inflation logic')
    parser.add_argument('--project_root', required=True)
    parser.add_argument('--module_name', default='custommint')
    args = parser.parse_args()
    implement_custom_inflation_logic(args.project_root, args.module_name)



# step:3 file: set_custom_inflation_minting_function
# step3.py
import logging
from pathlib import Path


def register_custom_module(project_root: str, new_module_name: str = 'custommint', go_module_path: str = 'github.com/my/app') -> None:
    # Wire the custom mint module into app.go, replacing the default mint module
    app_file = Path(project_root) / 'app.go'
    if not app_file.exists():
        raise FileNotFoundError(app_file)

    content = app_file.read_text()

    # Replace import path for mint module
    default_import = '"github.com/cosmos/cosmos-sdk/x/mint"'
    custom_import = f'"{go_module_path}/x/{new_module_name}"'
    if default_import in content and custom_import not in content:
        content = content.replace(default_import, custom_import)

    # Replace keeper and module registration references
    content = content.replace('mint.NewKeeper(', f'{new_module_name}.NewKeeper(')
    content = content.replace('mintmodule.NewAppModule', f'{new_module_name}.NewAppModule')

    app_file.write_text(content)
    logging.info('app.go updated to use %s module', new_module_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Register custom mint module')
    parser.add_argument('--project_root', required=True)
    parser.add_argument('--module_name', default='custommint')
    parser.add_argument('--go_module_path', default='github.com/my/app')
    args = parser.parse_args()
    register_custom_module(args.project_root, args.module_name, args.go_module_path)



# step:4 file: set_custom_inflation_minting_function
# step4.py
import subprocess
import logging
from pathlib import Path


def create_inflation_test(project_root: str, new_module_name: str = 'custommint') -> None:
    test_dir = Path(project_root) / 'x' / new_module_name
    test_file = test_dir / 'inflation_test.go'
    if test_file.exists():
        return

    lines = [
        f'package {new_module_name}',
        '',
        'import (',
        '    "testing"',
        '    "time"',
        ')',
        '',
        'func TestCalculateInflation(t *testing.T) {',
        '    params := DefaultParams()',
        '    now := time.Now()',
        '',
        '    infStart := CalculateInflation(params, now, now)',
        '    infNextYear := CalculateInflation(params, now, now.AddDate(1, 0, 0))',
        '',
        '    if !infNextYear.LT(infStart) {',
        '        t.Fatalf("expected inflation to decay: got %s >= %s", infNextYear, infStart)',
        '    }',
        '}',
    ]
    test_file.write_text('\n'.join(lines))
    logging.info('Created %s', test_file)


def go_build_and_unit_test(project_root: str) -> None:
    create_inflation_test(project_root)

    logging.info('Running go test ./...')
    result = subprocess.run(['go', 'test', './...'], cwd=project_root)
    if result.returncode != 0:
        raise SystemExit('Go tests failed')

    logging.info('Building binaries')
    result2 = subprocess.run(['go', 'build', './...'], cwd=project_root)
    if result2.returncode != 0:
        raise SystemExit('Go build failed')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run build and unit tests')
    parser.add_argument('--project_root', required=True)
    args = parser.parse_args()
    go_build_and_unit_test(args.project_root)



# step:5 file: set_custom_inflation_minting_function
# step5.py
import subprocess
import time
import logging
import requests
import os
import signal


def launch_devnet_and_monitor(project_root: str, home: str = './sim_home') -> None:
    logging.basicConfig(level=logging.INFO)
    node_cmd = ['simd', 'start', '--home', home]
    env = os.environ.copy()
    env['DAEMON_HOME'] = home

    proc = subprocess.Popen(node_cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        rpc = 'http://localhost:1317'
        # Wait until REST server is up
        for _ in range(60):
            try:
                r = requests.get(f'{rpc}/node_info', timeout=2)
                if r.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('REST API did not start in time')

        logging.info('Node is running. Waiting for a few blocks...')
        time.sleep(10)

        inflation = requests.get(f'{rpc}/cosmos/mint/v1beta1/inflation').json()
        provisions = requests.get(f'{rpc}/cosmos/mint/v1beta1/annual_provisions').json()
        logging.info('Current inflation: %s', inflation)
        logging.info('Annual provisions: %s', provisions)
    finally:
        logging.info('Terminating node')
        proc.send_signal(signal.SIGINT)
        proc.wait()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run local devnet and monitor inflation values')
    parser.add_argument('--project_root', required=True)
    parser.add_argument('--home', default='./sim_home')
    args = parser.parse_args()
    launch_devnet_and_monitor(args.project_root, args.home)



# step:1 file: enable_and_expose_the_json-rpc_server_on_port_8545_for_a_cosmos-evm_node
import os
import toml


def load_app_toml(config_path: str = "~/.evmd/config/app.toml"):
    """Open and parse the app.toml file located at `config_path`."""
    full_path = os.path.expanduser(config_path)

    # Ensure the file exists before attempting to open it
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Config file not found at {full_path}")

    # Parse TOML contents into a Python dictionary
    with open(full_path, "r", encoding="utf-8") as fp:
        config_data = toml.load(fp)

    return config_data, full_path


# step:2 file: enable_and_expose_the_json-rpc_server_on_port_8545_for_a_cosmos-evm_node
def set_json_rpc_enable(config: dict, enable: bool = True) -> dict:
    """Set the `enable` flag in the `[json-rpc]` section."""
    # Create the section if it does not yet exist
    config.setdefault("json-rpc", {})

    # Update the flag
    config["json-rpc"]["enable"] = enable

    return config


# step:3 file: enable_and_expose_the_json-rpc_server_on_port_8545_for_a_cosmos-evm_node
def set_json_rpc_address(config: dict, address: str = "0.0.0.0:8545") -> dict:
    """Set the HTTP address where the JSON-RPC server will listen."""
    config.setdefault("json-rpc", {})
    config["json-rpc"]["address"] = address
    return config


# step:4 file: enable_and_expose_the_json-rpc_server_on_port_8545_for_a_cosmos-evm_node
def set_ws_address_optional(
    config: dict,
    ws_address: str = "0.0.0.0:8546",
    enable_ws: bool = True,
) -> dict:
    """Add or remove the `ws-address` key inside `[json-rpc]`."""
    config.setdefault("json-rpc", {})

    if enable_ws:
        config["json-rpc"]["ws-address"] = ws_address
    else:
        # Remove the key if present and WebSocket is disabled
        config["json-rpc"].pop("ws-address", None)

    return config


# step:5 file: enable_and_expose_the_json-rpc_server_on_port_8545_for_a_cosmos-evm_node
def save_app_toml(config: dict, file_path: str) -> None:
    """Persist the updated configuration dictionary to disk."""
    # Write atomically by first dumping to a temporary file
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        toml.dump(config, fp)

    # Replace the original file only after successful write
    os.replace(tmp_path, file_path)
    print(f"Configuration saved to {file_path}")


# step:6 file: enable_and_expose_the_json-rpc_server_on_port_8545_for_a_cosmos-evm_node
import subprocess


def restart_node(service_name: str = "evmd") -> None:
    """Restart the node process via systemctl (or fall back to a manual command)."""
    try:
        # Attempt a systemd restart
        subprocess.run(["systemctl", "restart", service_name], check=True)
        print(f"Service '{service_name}' restarted successfully.")

    except FileNotFoundError:
        # systemctl not available (e.g. macOS or containers)
        raise EnvironmentError(
            "systemctl not found. Please restart the node manually or adapt this function."
        )

    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Failed to restart '{service_name}'. Systemctl returned: {err}"
        )


# step:1 file: get_the_latest_block_number_with_cast_block-number
from fastapi import FastAPI, HTTPException, Query
import httpx

app = FastAPI()

@app.get("/block-number")
async def get_block_number(rpc_url: str = Query(..., description="Full HTTP(S) JSON-RPC endpoint")):
    """Return the latest block number from a JSON-RPC endpoint."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_blockNumber",
        "params": []
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(rpc_url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Validate JSON-RPC structure
            if "result" not in data:
                raise ValueError("Malformed JSON-RPC response: 'result' key missing")
            # Convert hex string (e.g., '0x10d4f') to int
            block_number = int(data["result"], 16)
            return {"block_number": block_number}
    except (httpx.HTTPError, ValueError) as exc:
        # Propagate a clean API error to the caller
        raise HTTPException(status_code=500, detail=str(exc))


# step:1 file: Submit a proposal to update Cron module parameters
import requests

REST_ENDPOINT = "https://rest-kralum.neutron.org"  # <-- change to your preferred REST endpoint


def get_dao_authority_address(node_url: str = REST_ENDPOINT) -> str:
    """Returns the authority address defined in the Cron module params.

    Args:
        node_url: Base REST endpoint of a Neutron full-node.

    Raises:
        RuntimeError: If the request fails or the authority parameter is missing.
    """
    url = f"{node_url}/cosmos/params/v1beta1/params?subspace=cron"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # The response schema is { "param": [ { "key": "Authority", "value": "ntrn1..."}, ...] }
        for param in data.get("param", []):
            if param.get("key") == "Authority":
                return param.get("value")
        raise RuntimeError("'Authority' field not found in cron params response")
    except requests.RequestException as err:
        raise RuntimeError(f"Unable to query cron params: {err}")


# step:2 file: Submit a proposal to update Cron module parameters
from google.protobuf.any_pb2 import Any
# Proto imports generated from Neutron's proto definitions
from neutron.cron.v1.tx_pb2 import MsgUpdateParams, Params  # make sure your PYTHONPATH includes compiled protos


def build_msg_update_params(authority: str, *, max_schedules: int | None = None, default_gas_limit: int | None = None) -> MsgUpdateParams:
    """Constructs MsgUpdateParams with only the fields that need updating.

    Args:
        authority: Address allowed to perform the update (DAO address).
        max_schedules: New maximum number of active cron schedules.
        default_gas_limit: Optional default gas limit per cron execution.
    """
    params = Params()
    if max_schedules is not None:
        params.max_schedules = max_schedules
    if default_gas_limit is not None:
        params.default_gas_limit = default_gas_limit

    return MsgUpdateParams(authority=authority, params=params)


# step:3 file: Submit a proposal to update Cron module parameters
from google.protobuf.any_pb2 import Any
from cosmos.gov.v1beta1.gov_pb2 import MsgSubmitProposal  # SDK <0.50; replace with gov.v1 for newer chains
from cosmos.base.v1beta1.coin_pb2 import Coin


def package_into_gov_proposal(msg_update_params: MsgUpdateParams, proposer: str, title: str, description: str, *, deposit_amount: str = "10000000", deposit_denom: str = "untrn") -> MsgSubmitProposal:
    """Creates MsgSubmitProposal embedding MsgUpdateParams.

    Args:
        msg_update_params: The message produced in step 2.
        proposer: Address that submits the proposal (DAO address).
        title: Plain-text proposal title.
        description: Long-form markdown/description.
        deposit_amount: String amount of initial deposit (default 10 NTRN in micro-denom).
        deposit_denom: Denomination (usually 'untrn').
    """
    content_any = Any()
    # Pack the update message; type_url_prefix="/" satisfies most signing libraries
    content_any.Pack(msg_update_params, type_url_prefix="/")

    proposal_msg = MsgSubmitProposal(
        content=content_any,
        initial_deposit=[Coin(amount=deposit_amount, denom=deposit_denom)],
        proposer=proposer,
    )
    # v1beta1 requires title/description to be part of Content if not a gov TextProposal.
    # If using gov.v1 (SDK >=0.50) you'd instead set these on "MsgSubmitProposal" directly.
    proposal_msg.title = title
    proposal_msg.description = description

    return proposal_msg


# step:4 file: Submit a proposal to update Cron module parameters
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.tx import Transaction

# Chain-specific settings (adjust for testnet / mainnet)
CHAIN_ID = "neutron-1"
GRPC_ENDPOINT = "grpc-kralum.neutron.org:9090"
FEE_DENOM = "untrn"
DEFAULT_GAS = 300000  # can be tuned after simulating
DEFAULT_FEE = 5000    # 0.005 NTRN


def sign_and_broadcast_tx(msg_submit_proposal: MsgSubmitProposal, proposer_mnemonic: str) -> str:
    """Signs + broadcasts the governance proposal. Returns the transaction hash.

    Args:
        msg_submit_proposal: Message from step 3.
        proposer_mnemonic: BIP-39 mnemonic for the DAO account (ensure it’s secured!).
    """
    try:
        # 1. Initialise network client
        net_cfg = NetworkConfig(chain_id=CHAIN_ID, grpc_endpoint=GRPC_ENDPOINT)
        client = LedgerClient(net_cfg)

        # 2. Load wallet / account details
        priv = PrivateKey.from_mnemonic(proposer_mnemonic)
        wallet = priv.to_wallet()
        account = client.query_account(wallet.address())

        # 3. Build TX
        tx = Transaction()
        tx.add_message(msg_submit_proposal)
        tx.with_sequence(account.sequence)
        tx.with_account_num(account.account_number)
        tx.with_chain_id(CHAIN_ID)
        tx.with_gas(DEFAULT_GAS)
        tx.with_fee(DEFAULT_FEE, FEE_DENOM)

        # 4. Sign & broadcast
        tx.sign(priv)
        resp = client.broadcast_tx(tx)

        if resp.tx_response.code != 0:
            raise RuntimeError(f"Broadcast failed: {resp.tx_response.raw_log}")

        return resp.tx_response.txhash

    except Exception as err:
        raise RuntimeError(f"sign_and_broadcast_tx failed: {err}")


# step:1 file: set_a_preblocker_that_updates_consensus_parameters_every_block
# create_preblocker_fn.py
import os
import sys
import textwrap


def create_preblocker_fn(app_dir: str) -> None:
    """
    Inserts a PreBlocker function in app/app.go that updates consensus params
    at the beginning of every block.
    """
    app_go_path = os.path.join(app_dir, "app", "app.go")
    if not os.path.exists(app_go_path):
        raise FileNotFoundError(f"{app_go_path} not found")

    preblocker_code = textwrap.dedent('''

// -------------------------- PreBlocker ----------------------------------
// PreBlocker updates consensus parameters before every block is processed.
//
// Required imports (add to the import block in app/app.go):
//   "github.com/cosmos/cosmos-sdk/types"
//   abci "github.com/cometbft/cometbft/abci/types"
//   tmtypes "github.com/cometbft/cometbft/proto/tendermint/types"
//   "time"
func PreBlocker(ctx sdk.Context, req abci.RequestPreBlock) {
    // Example mutation: increase MaxGas every block
    newParams := &tmtypes.ConsensusParams{
        Block: &tmtypes.BlockParams{
            MaxBytes: 22020096, // 21 MB
            MaxGas:   10000000,
        },
        Evidence: &tmtypes.EvidenceParams{
            MaxAgeDuration: 48 * time.Hour,
            MaxAgeNumBlocks: 100000,
        },
        Validator: &tmtypes.ValidatorParams{
            PubKeyTypes: []string{"ed25519"},
        },
    }

    if err := app.BaseApp.UpdateConsensusParams(ctx, newParams); err != nil {
        ctx.Logger().Error("failed to update consensus params", "err", err)
    }
}
// ------------------------------------------------------------------------
''')

    with open(app_go_path, "r+") as f:
        content = f.read()
        if "func PreBlocker(" in content:
            print("PreBlocker already exists in file, skipping.")
            return
        f.write(preblocker_code)
    print("✅ PreBlocker function added to app/app.go")


if __name__ == "__main__":
    # Usage: python create_preblocker_fn.py /absolute/path/to/your/app
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    create_preblocker_fn(os.path.abspath(target))


# step:2 file: set_a_preblocker_that_updates_consensus_parameters_every_block
# register_preblocker.py
import os
import sys
import re


def register_preblocker(app_dir: str) -> None:
    """
    Adds app.SetPreBlocker(PreBlocker) to the NewApp constructor so the PreBlocker executes every block.
    """
    app_go_path = os.path.join(app_dir, "app", "app.go")
    if not os.path.exists(app_go_path):
        raise FileNotFoundError(f"{app_go_path} not found")

    with open(app_go_path, "r+") as f:
        content = f.read()
        if "SetPreBlocker(PreBlocker)" in content:
            print("PreBlocker already registered, skipping.")
            return

        # Insert right before the final return statement inside NewApp
        pattern = r"func\s+NewApp[\s\S]+?return\s+app"
        match = re.search(pattern, content)
        if not match:
            print("Could not locate NewApp function; please register manually.")
            return

        insert_idx = match.end() - len("return app")
        insertion = "\n    // Register the PreBlocker\n    app.SetPreBlocker(PreBlocker)\n"
        new_content = content[:insert_idx] + insertion + content[insert_idx:]

        f.seek(0)
        f.write(new_content)
        f.truncate()

    print("✅ PreBlocker registered in NewApp")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    register_preblocker(os.path.abspath(target))


# step:3 file: set_a_preblocker_that_updates_consensus_parameters_every_block
# compile_binary.py
import os
import sys
import subprocess


def compile_binary(app_dir: str = ".") -> None:
    """
    Compiles the modified binary and installs it to $GOBIN.
    """
    print("🔨 Compiling the blockchain binary …")
    proc = subprocess.run(["go", "install", "./..."], cwd=app_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("Compilation failed")
    print("✅ Compilation successful. Binary available in your GOPATH/bin directory.")


if __name__ == "__main__":
    compile_binary(sys.argv[1] if len(sys.argv) > 1 else ".")


# step:4 file: set_a_preblocker_that_updates_consensus_parameters_every_block
# start_local_chain.py
import subprocess
import sys
import os


def start_local_chain(home: str = "./data", chain_id: str = "localnet", binary: str = "appd") -> None:
    """
    Starts a single-node chain using the rebuilt binary.
    """
    if not os.path.exists(home):
        print("🔧 Initializing home directory")
        subprocess.run([binary, "init", "validator", "--chain-id", chain_id, "--home", home], check=True)
        subprocess.run([binary, "config", "chain-id", chain_id, "--home", home], check=True)

    print("⛓️  Starting node … (Ctrl+C to stop)")
    try:
        subprocess.run([binary, "start", "--home", home], check=True)
    except KeyboardInterrupt:
        print("Node stopped by user")


if __name__ == "__main__":
    # Usage: python start_local_chain.py [home_dir] [chain_id] [binary]
    start_local_chain(*(sys.argv[1:]))


# step:5 file: set_a_preblocker_that_updates_consensus_parameters_every_block
# query_consensus_params.py
import subprocess
import sys


def query_consensus_params(height: int, binary: str = "appd") -> None:
    """
    Queries consensus parameters for a given block height.
    """
    cmd = [
        binary,
        "query",
        "params",
        "subspace",
        "consensus",
        "1",
        "--height",
        str(height),
        "--output",
        "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Query failed")
    print(result.stdout)


if __name__ == "__main__":
    height = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    query_consensus_params(height)


# step:1 file: Create a multi-message cron schedule "protocol_update" that executes three contract calls every 100 800 blocks
# utils/governance.py
import os
import requests


def get_governance_authority(rest_endpoint: str = 'https://rest-kralum.neutron.org') -> str:
    '''
    Fetch the current Main DAO address from the cron params endpoint.
    Fallback to the MAIN_DAO_ADDRESS environment variable if the
    endpoint is unavailable or the field is missing.
    '''
    try:
        resp = requests.get(f'{rest_endpoint}/neutron/cron/v1/params', timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Try multiple likely field names for robustness.
        authority = (
            data.get('params', {}).get('governance_account')
            or data.get('params', {}).get('authority')
        )
        if authority:
            return authority
        raise ValueError('Authority field not found in response.')
    except Exception as err:
        # Log and fall back to env var so the workflow can continue.
        print(f'[WARN] Unable to fetch authority from REST API: {err}')
        fallback = os.getenv('MAIN_DAO_ADDRESS')
        if not fallback:
            raise RuntimeError('MAIN_DAO_ADDRESS env var is not set.') from err
        return fallback



# step:2 file: Create a multi-message cron schedule "protocol_update" that executes three contract calls every 100 800 blocks
# utils/contracts.py
from typing import List
import json
from google.protobuf.any_pb2 import Any
from cosmpy.protos.cosmwasm.wasm.v1 import tx_pb2 as wasm_tx


def build_execute_msg(sender: str, contract: str, msg: dict, funds: List[dict] | None = None) -> wasm_tx.MsgExecuteContract:
    '''
    Converts a Python dict into the binary-encoded message required by
    MsgExecuteContract and optionally attaches funds.
    '''
    try:
        execute = wasm_tx.MsgExecuteContract(
            sender=sender,
            contract=contract,
            msg=json.dumps(msg).encode('utf-8'),  # CosmWasm expects binary JSON
        )
        if funds:
            for coin in funds:
                execute.funds.add(denom=coin['denom'], amount=str(coin['amount']))
        return execute
    except Exception as err:
        raise ValueError(f'Failed to build MsgExecuteContract: {err}')


# ---------------------------------------------------------------------
# Example placeholder calls
# ---------------------------------------------------------------------

def build_placeholder_calls(authority: str):
    call_1 = build_execute_msg(
        sender=authority,
        contract='neutron1contractaddr1...',
        msg={'update_config': {'param': 42}},
    )
    call_2 = build_execute_msg(
        sender=authority,
        contract='neutron1contractaddr2...',
        msg={'set_admin': {'new_admin': authority}},
    )
    call_3 = build_execute_msg(
        sender=authority,
        contract='neutron1contractaddr3...',
        msg={'migrate': {'code_id': 99}},
    )
    return call_1, call_2, call_3



# step:3 file: Create a multi-message cron schedule "protocol_update" that executes three contract calls every 100 800 blocks
# utils/cron.py
from typing import List
from neutron.protos.neutron.cron import tx_pb2 as cron_tx


def build_add_schedule(authority: str, name: str, period: int, msgs: List) -> cron_tx.MsgAddSchedule:
    '''Create a MsgAddSchedule for the Cron module.'''
    if not msgs:
        raise ValueError('Msgs list cannot be empty')
    try:
        schedule = cron_tx.MsgAddSchedule(
            authority=authority,
            name=name,
            period=period,
            msgs=msgs,
        )
        return schedule
    except Exception as err:
        raise RuntimeError(f'Unable to build MsgAddSchedule: {err}')



# step:4 file: Create a multi-message cron schedule "protocol_update" that executes three contract calls every 100 800 blocks
# utils/governance.py (continued)
from typing import List
from google.protobuf.any_pb2 import Any
from cosmos.protos.cosmos.gov.v1 import tx_pb2 as gov_tx
from cosmos.protos.cosmos.base.v1beta1 import coin_pb2 as base_coin


def wrap_into_submit_proposal(schedule_msg, proposer: str, deposit: List[dict]):
    '''Pack the MsgAddSchedule into a MsgSubmitProposal.'''    
    try:
        any_msg = Any()
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



# step:5 file: Create a multi-message cron schedule "protocol_update" that executes three contract calls every 100 800 blocks
# utils/tx.py
import asyncio
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction


async def submit_proposal(rpc_endpoint: str, chain_id: str, mnemonic: str, proposal_msg):
    '''Sign and broadcast the MsgSubmitProposal.'''
    try:
        pk = PrivateKey.from_mnemonic(mnemonic)
        address = pk.to_public_key().address()

        client = LedgerClient(NetworkConfig(chain_id=chain_id, url=rpc_endpoint))

        tx = (
            Transaction()
            .with_messages(proposal_msg)
            .with_sequence(client.get_sequence(address))
            .with_account_num(client.get_number(address))
            .with_chain_id(chain_id)
            .with_gas(400000)
            .with_fee(400000, 'untrn')
            .with_memo('Cron schedule proposal')
        )

        tx = tx.sign(pk)
        resp = client.broadcast_tx(tx)
        if resp.is_err():
            raise RuntimeError(f'Broadcast failed: {resp.tx_response.raw_log}')
        print(f'Broadcast successful. TxHash: {resp.tx_response.txhash}')
        return resp.tx_response.txhash
    except Exception as err:
        raise RuntimeError(f'Unable to submit proposal: {err}')



# step:6 file: Create a multi-message cron schedule "protocol_update" that executes three contract calls every 100 800 blocks
# utils/gov_monitor.py
import time
import requests

STATUS_PASSED = 'PROPOSAL_STATUS_PASSED'
TERMINAL_STATES = ['PROPOSAL_STATUS_REJECTED', 'PROPOSAL_STATUS_FAILED', 'PROPOSAL_STATUS_ABORTED']


def wait_for_voting_result(rest_endpoint: str, proposal_id: int, poll_interval: int = 15):
    '''Block until the governance proposal reaches a final state.'''
    url = f'{rest_endpoint}/cosmos/gov/v1/proposals/{proposal_id}'
    while True:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            status = data.get('proposal', {}).get('status')
            print(f'Proposal {proposal_id} status: {status}')
            if status == STATUS_PASSED:
                print('Proposal passed 🎉')
                return True
            if status in TERMINAL_STATES:
                raise RuntimeError(f'Proposal ended with status {status}')
            time.sleep(poll_interval)
        except Exception as err:
            print(f'[WARN] error while querying proposal: {err}')
            time.sleep(poll_interval)



# step:7 file: Create a multi-message cron schedule "protocol_update" that executes three contract calls every 100 800 blocks
# utils/cron.py (continued)
import requests


def query_cron_schedule(rest_endpoint: str, name: str):
    '''Retrieve a cron schedule by name to verify successful registration.'''
    url = f'{rest_endpoint}/neutron/cron/v1/schedules/{name}'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as err:
        raise RuntimeError(f'Failed to fetch schedule {name}: {err}')



# step:2 file: hash_arbitrary_data_using_keccak-256_(cast_keccak)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import binascii

# `pysha3` provides the Ethereum-style Keccak implementation.
# Install with:  pip install pysha3 fastapi uvicorn
try:
    from sha3 import keccak_256  # noqa: E402
except ImportError as err:
    raise ImportError("pysha3 is required: pip install pysha3") from err

app = FastAPI()

class KeccakRequest(BaseModel):
    data: str  # raw text or 0x-prefixed hex

@app.post("/api/keccak")
async def keccak_hash(req: KeccakRequest):
    """Return Keccak-256 hash for the given input."""
    data_str = req.data

    # Detect hex vs text
    try:
        if data_str.startswith(("0x", "0X")):
            hex_body = data_str[2:]
            # If hex has odd length, left-pad with a zero
            if len(hex_body) % 2:
                hex_body = "0" + hex_body
            raw_bytes = binascii.unhexlify(hex_body)
        else:
            raw_bytes = data_str.encode("utf-8")
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid hex input: {exc}") from exc

    digest = keccak_256(raw_bytes).hexdigest()
    return {"hash": "0x" + digest}


# step:2 file: claim_junox_test_tokens_from_the_uni-6_faucet_for_a_given_address
'''api/faucet.py'''
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI()
FAUCET_ENDPOINT = "https://faucet.uni.junonetwork.io/credit"

@app.post("/api/faucet/credit")
async def faucet_credit(payload: dict):
    """POST /api/faucet/credit { "address": "juno1..." } -> faucet JSON/text"""
    address: str | None = payload.get("address")

    # Minimal server-side validation so we don’t hammer the faucet with bad requests
    if not address or not address.startswith("juno1"):
        raise HTTPException(status_code=400, detail="Invalid or missing Juno address.")

    url = f"{FAUCET_ENDPOINT}?address={address}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            faucet_resp = await client.post(url)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Faucet unreachable: {exc}") from exc

    if faucet_resp.status_code != 200:
        raise HTTPException(status_code=faucet_resp.status_code, detail=faucet_resp.text)

    # Return JSON when possible, fallback to raw text (some faucets return plain text)
    try:
        return faucet_resp.json()
    except ValueError:
        return {"raw_response": faucet_resp.text}



# step:1 file: query_smart-contract_event_logs_on_a_cosmos_evm_chain_with_foundry’s_cast_logs
from fastapi import FastAPI, HTTPException
import os

app = FastAPI()

# Map chain identifiers to ENV names holding their RPC endpoints
CHAIN_RPC_ENV_MAP = {
    "evmos": "RPC_ENDPOINT_EVMOS",   # example → https://evmos-rpc.publicnode.com
    "cronos": "RPC_ENDPOINT_CRONOS", # example → https://rpc.cronos.org
    "kava": "RPC_ENDPOINT_KAVA",     # example → https://evm.kava.io
    "cudos": "RPC_ENDPOINT_CUDOS"    # example → https://rpc.cudos.org
}

@app.get("/api/rpc-endpoint")
def get_rpc_endpoint(chain_id: str):
    """Return the RPC endpoint for a given EVM-compatible Cosmos chain."""
    env_key = CHAIN_RPC_ENV_MAP.get(chain_id.lower())
    if env_key is None:
        raise HTTPException(status_code=404, detail=f"Unsupported chain_id '{chain_id}'.")

    endpoint = os.getenv(env_key)
    if not endpoint:
        raise HTTPException(status_code=500, detail=f"Environment variable '{env_key}' is not set.")

    return {"chain_id": chain_id, "rpc_endpoint": endpoint}


# step:2 file: query_smart-contract_event_logs_on_a_cosmos_evm_chain_with_foundry’s_cast_logs
from fastapi import FastAPI, HTTPException
from web3 import Web3

app = FastAPI()

@app.get("/api/validate-address")
def validate_address(address: str):
    """Verify that `address` is a valid EVM address and return its checksum version."""
    if not Web3.isAddress(address):
        raise HTTPException(status_code=400, detail="Address is not a valid hex EVM address.")

    checksum_address = Web3.toChecksumAddress(address)
    return {"checksum_address": checksum_address}


# step:3 file: query_smart-contract_event_logs_on_a_cosmos_evm_chain_with_foundry’s_cast_logs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests, os, json

app = FastAPI()

ABI_DIR = os.getenv("ABI_DIR", "abis")
os.makedirs(ABI_DIR, exist_ok=True)

class AbiRequest(BaseModel):
    source_url: str       # e.g. https://raw.githubusercontent.com/…/MyContract.json
    contract_name: str    # used as filename prefix

@app.post("/api/abi")
def download_abi(req: AbiRequest):
    """Download and persist the ABI JSON so other endpoints can reuse it."""
    try:
        res = requests.get(req.source_url)
        res.raise_for_status()
        abi_json = res.json()
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Failed to fetch/parse ABI: {err}")

    file_path = os.path.join(ABI_DIR, f"{req.contract_name}.abi.json")
    with open(file_path, "w") as fp:
        json.dump(abi_json, fp, indent=2)

    return {"saved_to": file_path}


# step:4 file: query_smart-contract_event_logs_on_a_cosmos_evm_chain_with_foundry’s_cast_logs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from web3 import Web3
import os, json

app = FastAPI()

ABI_DIR = os.getenv("ABI_DIR", "abis")
CHAIN_RPC_ENV_MAP = {
    "evmos": "RPC_ENDPOINT_EVMOS",
    "cronos": "RPC_ENDPOINT_CRONOS",
    "kava": "RPC_ENDPOINT_KAVA",
    "cudos": "RPC_ENDPOINT_CUDOS"
}

class LogRequest(BaseModel):
    chain_id: str           # evmos / cronos / …
    contract_name: str      # must match the filename saved in Step 3
    address: str            # checksummed or not
    from_block: Union[int, str]
    to_block:   Union[int, str] = "latest"
    topics:     List[str] = []

@app.post("/api/logs")
def fetch_logs(req: LogRequest):
    """Return raw logs for the requested filter; decode can be done client-side if desired."""
    env_key = CHAIN_RPC_ENV_MAP.get(req.chain_id.lower())
    if env_key is None:
        raise HTTPException(status_code=404, detail=f"Unsupported chain_id '{req.chain_id}'.")

    rpc_endpoint = os.getenv(env_key)
    if not rpc_endpoint:
        raise HTTPException(status_code=500, detail=f"Environment variable '{env_key}' is not set.")

    w3 = Web3(Web3.HTTPProvider(rpc_endpoint))
    if not w3.isConnected():
        raise HTTPException(status_code=502, detail="Could not connect to RPC endpoint.")

    if not Web3.isAddress(req.address):
        raise HTTPException(status_code=400, detail="Invalid contract address.")
    checksum_address = Web3.toChecksumAddress(req.address)

    # Ensure ABI exists (even if we don’t decode here, caller likely uploaded it)
    abi_path = os.path.join(ABI_DIR, f"{req.contract_name}.abi.json")
    if not os.path.exists(abi_path):
        raise HTTPException(status_code=400, detail=f"ABI file not found: {abi_path}")

    filter_params = {
        "fromBlock": req.from_block,
        "toBlock":   req.to_block,
        "address":   checksum_address,
        "topics":    req.topics if req.topics else None
    }

    try:
        raw_logs = w3.eth.get_logs(filter_params)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"get_logs failed: {err}")

    # Transform HexBytes → hex strings for JSON friendliness
    parsed_logs = []
    for log in raw_logs:
        parsed_logs.append({
            "blockNumber": log.blockNumber,
            "transactionHash": log.transactionHash.hex(),
            "address": log.address,
            "data": log.data,
            "topics": [t.hex() for t in log.topics]
        })

    return {"total": len(parsed_logs), "logs": parsed_logs}


# step:1 file: sign_unsigned_tx.json_using_the_key_`my_validator`_in_the_test_keyring_backend
import subprocess
import json
from typing import Dict


def ensure_key_exists(key_name: str, keyring_backend: str = "test") -> Dict[str, str]:
    """Return basic key information if the key exists, otherwise raise an error."""
    try:
        # --output json yields a machine-readable response we can parse
        result = subprocess.run(
            [
                "cosmos",
                "keys",
                "show",
                key_name,
                "--keyring-backend",
                keyring_backend,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        key_info = json.loads(result.stdout)
        return {
            "key_name": key_info.get("name", key_name),
            "address": key_info.get("address"),
        }
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Key '{key_name}' was not found in key-ring '{keyring_backend}'. CLI stderr: {err.stderr}"
        ) from err


# Example FastAPI route (optional, integrate into your BFF of choice)
# from fastapi import APIRouter, HTTPException
# router = APIRouter()
#
# @router.get("/api/keys/show")
# async def api_key_show(key_name: str):
#     try:
#         return ensure_key_exists(key_name)
#     except Exception as exc:
#         raise HTTPException(status_code=400, detail=str(exc))


# step:2 file: sign_unsigned_tx.json_using_the_key_`my_validator`_in_the_test_keyring_backend
import subprocess
from pathlib import Path
from typing import Dict


def sign_transaction(
    unsigned_tx_path: str,
    from_key: str,
    chain_id: str,
    output_tx_path: str = "tx_signed.json",
    keyring_backend: str = "test",
) -> Dict[str, str]:
    """Sign an existing transaction file via the Cosmos CLI and return the output path."""
    unsigned_tx = Path(unsigned_tx_path).expanduser()
    output_tx = Path(output_tx_path).expanduser()

    if not unsigned_tx.exists():
        raise FileNotFoundError(f"Unsigned tx file not found: {unsigned_tx}")

    cmd = [
        "cosmos",
        "tx",
        "sign",
        str(unsigned_tx),
        "--from",
        from_key,
        "--chain-id",
        chain_id,
        "--keyring-backend",
        keyring_backend,
        "--output-document",
        str(output_tx),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"signed_tx_path": str(output_tx)}
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to sign transaction. CLI stderr: {err.stderr}") from err


# Example FastAPI route (optional)
# @router.post("/api/tx/sign")
# async def api_tx_sign(body: dict):
#     return sign_transaction(**body)


# step:3 file: sign_unsigned_tx.json_using_the_key_`my_validator`_in_the_test_keyring_backend
import json
from pathlib import Path
from typing import Dict


def verify_signed_tx(signed_tx_path: str = "tx_signed.json") -> Dict[str, bool]:
    """Validate that the signed transaction file contains a signature."""
    tx_file = Path(signed_tx_path).expanduser()
    if not tx_file.exists():
        raise FileNotFoundError(f"Signed transaction file not found: {tx_file}")

    with tx_file.open() as fp:
        tx_data = json.load(fp)

    signatures = tx_data.get("signatures", [])
    if not signatures or not signatures[0]:
        raise ValueError("Signature array is empty — transaction appears unsigned.")

    return {"valid": True, "signature_count": len(signatures)}


# Example FastAPI route (optional)
# @router.get("/api/tx/verify")
# async def api_tx_verify(path: str = "tx_signed.json"):
#     return verify_signed_tx(path)


# step:1 file: decode_binary_data_to_hexadecimal_using_foundry’s_cast_from-bin
import shutil
import subprocess
import logging


def ensure_foundry_installed() -> bool:
    """Check for Foundry and install it if absent.

    Returns:
        bool: True when the tooling is confirmed present.
    Raises:
        RuntimeError: if installation fails.
    """
    logger = logging.getLogger(__name__)

    # Fast-exit when Foundry is already on PATH
    if shutil.which("cast") is not None:
        logger.info("Foundry detected—skipping installation.")
        return True

    try:
        logger.info("Foundry not detected. Bootstrapping via official script…")
        # Download & run installer
        subprocess.run(
            "curl -L https://foundry.paradigm.xyz | bash",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Pull latest binaries
        subprocess.run("foundryup", shell=True, check=True)
        logger.info("Foundry installation completed successfully.")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Foundry installation failed: %s", exc)
        raise RuntimeError("Unable to install Foundry. Consult logs for details.")


# step:3 file: decode_binary_data_to_hexadecimal_using_foundry’s_cast_from-bin
from fastapi import FastAPI, UploadFile, File, HTTPException
import subprocess
import tempfile
import os
from foundry_setup import ensure_foundry_installed

app = FastAPI()


@app.post('/api/cast/from-bin')
async def cast_from_bin(file: UploadFile = File(...)):
    # Make sure Foundry exists before proceeding
    ensure_foundry_installed()

    tmp_path = None
    try:
        # Persist request body to a temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Run the conversion using Foundry’s cast
        hex_output = subprocess.check_output(
            ['cast', 'from-bin', tmp_path],
            text=True,
        ).strip()

        return {'hex': hex_output}
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f'cast failed: {exc}')
    finally:
        # House-keeping: delete the temp file (if created)
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# step:1 file: Show details for the cron schedule "dasset-updator"
import json
import subprocess
from typing import Dict


def query_cron_show_schedule(schedule_name: str, node: str = "https://rpc.neutron.org:26657") -> Dict:
    """Fetch cron schedule metadata from a Neutron node.

    Args:
        schedule_name (str): The name of the cron schedule to query.
        node (str): Optional RPC node URL. Defaults to a public Neutron RPC.

    Returns:
        Dict: Parsed JSON data describing the schedule.

    Raises:
        RuntimeError: If the neutrond binary is missing or the command fails.
    """
    # Construct neutrond CLI command
    cmd = [
        "neutrond", "query", "cron", "show-schedule", schedule_name,
        "--node", node,
        "--output", "json"
    ]

    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as err:
        raise RuntimeError("'neutrond' CLI not found. Install it and ensure it is in your PATH.") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"CLI returned error while querying schedule '{schedule_name}': {err.stderr}") from err

    # Parse the JSON output
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as err:
        raise RuntimeError("Failed to decode neutrond JSON output.") from err


if __name__ == "__main__":
    # Example invocation
    schedule_meta = query_cron_show_schedule("protocol_update")
    print(json.dumps(schedule_meta, indent=2))


# step:2 file: retrieve_the_smart-contract_bytecode_deployed_at_a_given_evm_address_(latest_block)
import os
import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI()

# You can set EVM_NODE_URL in your deployment environment to switch RPC endpoints without code changes
EVM_NODE_URL = os.getenv("EVM_NODE_URL", "https://rpc.evmos.org:443")

@app.get("/api/evm/get_code")
async def get_code(address: str):
    """Proxy eth_getCode to the configured Cosmos-EVM JSON-RPC endpoint."""
    # Basic server-side validation (defense-in-depth)
    if not address or not address.startswith("0x") or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid EVM address supplied.")

    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getCode",
        "params": [address, "latest"],
        "id": 1,
    }

    try:
        async with httpx.AsyncClient() as client:
            rpc_resp = await client.post(EVM_NODE_URL, json=payload, timeout=10)
            rpc_resp.raise_for_status()
    except Exception as exc:
        # Bubble up networking / RPC errors to the caller
        raise HTTPException(status_code=502, detail=f"RPC request failed: {exc}")

    rpc_json = rpc_resp.json()

    # The RPC node may return its own error structure
    if "error" in rpc_json:
        raise HTTPException(status_code=502, detail=rpc_json["error"].get("message", "Unknown RPC error"))

    return {
        "address": address,
        "bytecode": rpc_json.get("result", "0x")  # Empty contracts return "0x"
    }


# step:1 file: run_mutex_profiling_for_10_seconds_and_write_output_to_mutex.prof
# backend/rpc_client.py
import os
import requests
import uuid


def _get_env_or_raise(var_name: str) -> str:
    """Return the value of an env-var or raise an explicit error."""
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{var_name}' is not set.")
    return value


def get_authenticated_session() -> requests.Session:
    """Instantiate a requests.Session configured with Basic-Auth creds."""
    session = requests.Session()

    # Credentials are loaded from environment variables so secrets never
    # leave the backend.
    rpc_user = _get_env_or_raise("RPC_ADMIN_USER")
    rpc_pass = _get_env_or_raise("RPC_ADMIN_PASSWORD")
    session.auth = (rpc_user, rpc_pass)

    # JSON-RPC always uses application/json.
    session.headers.update({"Content-Type": "application/json"})

    return session


# A module-level session & endpoint can be shared across requests.
RPC_ENDPOINT = os.getenv("RPC_ADMIN_URL", "http://127.0.0.1:8545")
SESSION = get_authenticated_session()


def rpc_call(method: str, params=None, id_: str | None = None):
    """Utility that performs a JSON-RPC request and returns `result`."""
    payload = {
        "jsonrpc": "2.0",
        "id": id_ or str(uuid.uuid4()),
        "method": method,
        "params": params or [],
    }

    # Perform HTTP POST with a 30-second timeout for safety.
    resp = SESSION.post(RPC_ENDPOINT, json=payload, timeout=30)
    resp.raise_for_status()  # Raises on non-200 HTTP codes.

    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")

    return data.get("result")


# step:3 file: run_mutex_profiling_for_10_seconds_and_write_output_to_mutex.prof
# backend/api.py
from fastapi import FastAPI, HTTPException
from rpc_client import rpc_call

app = FastAPI()


@app.post('/api/debug/mutex-profile')
async def debug_mutex_profile(payload: dict):
    """Invoke the node's `debug_mutexProfile` RPC method."""
    duration = payload.get('duration', 10)
    output_path = payload.get('outputPath', 'mutex.prof')

    try:
        # In geth, the parameter order is [fileName, durationSeconds].
        result = rpc_call('debug_mutexProfile', [output_path, duration])
        return {
            'status': 'ok',
            'file': output_path,
            'rpc_result': result  # Typically `null` if everything went fine.
        }
    except Exception as exc:
        # Convert any Python/requests errors into an HTTP 500 for the client.
        raise HTTPException(status_code=500, detail=str(exc))


# step:6 file: Withdraw 50 NTRN from the smart contract
# Step 6 – Python backend balance query
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet

NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url="https://rpc-kralum.neutron.org",  # same RPC used by frontend
    fee_denomination="untrn",
    staking_denomination="untrn",
)

client = LedgerClient(NETWORK)

def query_bank_balance(address: str, denom: str = "untrn") -> int:
    """Return current balance for `address` in the given `denom`."""
    try:
        balance = client.query_bank_balance(address, denom=denom)
        return int(balance.amount)
    except Exception as exc:
        raise RuntimeError(f"Failed to query balance: {exc}") from exc


# step:5 file: set_my_boost_target_to_my_ethereum_address
import os
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator

from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.contract import MsgExecuteContract
from cosmpy.crypto.address import Address

router = APIRouter()

RPC = os.getenv('NEUTRON_LCD_URL', 'https://rest-kralum.neutron.org')
CHAIN_ID = os.getenv('NEUTRON_CHAIN_ID', 'neutron-1')
GAS_PRICE = float(os.getenv('GAS_PRICE', '0.025'))  # in untrn


class MsgValue(BaseModel):
    sender: str
    contract: str
    msg: List[int]  # UTF-8 bytes array sent by the frontend
    funds: List[str] = []


class ExecutePayload(BaseModel):
    typeUrl: str
    value: MsgValue

    @validator('typeUrl')
    def ensure_msg_execute(cls, v):
        if v != '/cosmwasm.wasm.v1.MsgExecuteContract':
            raise ValueError('Only MsgExecuteContract is supported by this endpoint.')
        return v


@router.post('/api/set_target')
async def set_target(payload: ExecutePayload):
    """Signs and broadcasts a MsgExecuteContract built on the frontend"""
    try:
        # Prepare LCD/RPC client
        config = NetworkConfig(
            chain_id=CHAIN_ID,
            url=RPC,
            fee_minimum_gas_price=GAS_PRICE,
            fee_denom='untrn',
        )
        client = LedgerClient(config)

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


# step:1 file: set_the_default_rpc_url_in_~_.foundry_foundry.toml
from pathlib import Path


def open_or_create_foundry_config() -> Path:
    """Ensure ~/.foundry/foundry.toml exists and return its Path object."""
    home_dir = Path.home()
    foundry_dir = home_dir / ".foundry"
    foundry_dir.mkdir(parents=True, exist_ok=True)  # Create ~/.foundry if it doesn't exist

    config_path = foundry_dir / "foundry.toml"
    if not config_path.exists():
        config_path.touch()  # Create an empty file
        print(f"Created new Foundry config at {config_path}")
    else:
        print(f"Found Foundry config at {config_path}")

    return config_path


# step:2 file: set_the_default_rpc_url_in_~_.foundry_foundry.toml
from pathlib import Path


def edit_toml_key_value(key: str, value: str, config_path: Path) -> bool:
    """Insert or update a "key = \"value\"" line in a TOML file.

    Args:
        key:    The TOML key to add or update, e.g. "rpc_endpoint".
        value:  The value to associate with the key, e.g. "https://my.rpc.node:8545".
        config_path: Path to the target foundry.toml file.

    Returns:
        True when the file was modified successfully.
    """
    key_line = f"{key} = \"{value}\"\n"

    # Read current contents (if any)
    try:
        existing_lines = config_path.read_text().splitlines(keepends=True)
    except FileNotFoundError:
        existing_lines = []

    updated = False
    new_lines: list[str] = []
    for line in existing_lines:
        if line.strip().startswith(f"{key} ="):
            new_lines.append(key_line)
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        # Key not present; append it.
        new_lines.append(key_line)

    # Persist changes.
    config_path.write_text("".join(new_lines))
    print(f"{'Updated' if updated else 'Added'} '{key}' in {config_path}")
    return True


# step:3 file: set_the_default_rpc_url_in_~_.foundry_foundry.toml
from pathlib import Path


def save_and_close_file(config_path: Path, key: str, expected_value: str) -> bool:
    """Confirm that `key = \"expected_value\"` is present in the file."""
    target_line = f"{key} = \"{expected_value}\""
    if target_line in config_path.read_text():
        print("Configuration written successfully.")
        return True
    raise ValueError("Configuration write failed: expected line not found.")


# step:4 file: set_the_default_rpc_url_in_~_.foundry_foundry.toml
import json
import subprocess


def validate_configuration(key: str = 'rpc_endpoint', expected_value: str | None = None) -> bool:
    """Run `forge config --json` and confirm `key` exists (and equals `expected_value` if provided)."""
    try:
        result = subprocess.run([
            'forge', 'config', '--json'
        ], capture_output=True, text=True, check=True)
    except FileNotFoundError as err:
        raise RuntimeError("`forge` CLI not found. Is Foundry installed?") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"`forge config` failed: {err.stderr}") from err

    # Try JSON parsing first.
    try:
        cfg = json.loads(result.stdout)
        current_value = cfg.get(key)
    except json.JSONDecodeError:
        # Fallback to plain-text parsing.
        current_value = None
        for ln in result.stdout.splitlines():
            if ln.strip().startswith(key):
                current_value = ln.split('=')[-1].strip().strip('"')
                break

    if current_value is None:
        raise ValueError(f"`{key}` not found in Forge configuration output.")

    if expected_value and current_value != expected_value:
        raise ValueError(
            f"`{key}` mismatch: expected '{expected_value}', got '{current_value}'.")

    print(f"Forge is configured with {key} = {current_value}")
    return True


# step:2 file: configure_junod_cli_with_a_specific_rpc_node_and_chain-id_(uni-6)
from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.post('/api/junod/config/node')
async def config_node(payload: dict):
    """Sets the default RPC endpoint for the local `junod` CLI."""
    node_url = payload.get('node_url')
    if not node_url:
        raise HTTPException(status_code=400, detail='`node_url` field is required.')

    try:
        completed = subprocess.run(
            ['junod', 'config', 'node', node_url],
            capture_output=True,
            text=True,
            check=True,
        )
        return {'stdout': completed.stdout.strip()}
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=exc.stderr.strip())


# step:3 file: configure_junod_cli_with_a_specific_rpc_node_and_chain-id_(uni-6)
from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.post('/api/junod/config/chain-id')
async def config_chain_id(payload: dict):
    """Sets the chain-id in the local `junod` CLI configuration."""
    chain_id = payload.get('chain_id')
    if not chain_id:
        raise HTTPException(status_code=400, detail='`chain_id` field is required.')

    try:
        completed = subprocess.run(
            ['junod', 'config', 'chain-id', chain_id],
            capture_output=True,
            text=True,
            check=True,
        )
        return {'stdout': completed.stdout.strip()}
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=exc.stderr.strip())


# step:4 file: configure_junod_cli_with_a_specific_rpc_node_and_chain-id_(uni-6)
from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.get('/api/junod/config/view')
async def view_config():
    """Returns the current junod CLI configuration."""
    try:
        completed = subprocess.run(
            ['junod', 'config'],
            capture_output=True,
            text=True,
            check=True,
        )
        return {'config': completed.stdout.strip()}
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=exc.stderr.strip())


# step:2 file: show_my_total_bitcoin_summer_points_earned_in_the_current_phase
from fastapi import FastAPI, HTTPException
import requests, base64, json, os

app = FastAPI()

# Replace these with actual values or set them as environment variables
LCD_ENDPOINT      = os.getenv('LCD_ENDPOINT', 'https://rest-kralum.neutron-1.neutron.org')
CAMPAIGN_CONTRACT = os.getenv('CAMPAIGN_CONTRACT', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def wasm_query(contract_address: str, query_msg: dict):
    """Utility function that performs a CosmWasm smart-query via the public LCD."""
    try:
        msg_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
        url     = f"{LCD_ENDPOINT}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{msg_b64}"
        resp    = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get('data', {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Smart-query failed: {e}')

@app.get('/api/active_phase')
def fetch_current_campaign_phase():
    """Returns the ID of the currently active campaign phase."""
    query_msg = {"get_current_phase": {}}
    data      = wasm_query(CAMPAIGN_CONTRACT, query_msg)
    if 'phase_id' not in data:
        raise HTTPException(status_code=500, detail="Invalid contract response: 'phase_id' missing")
    return {"phase_id": data['phase_id']}


# step:3 file: show_my_total_bitcoin_summer_points_earned_in_the_current_phase
from fastapi import FastAPI, HTTPException, Query
import requests, base64, json, os

app = FastAPI()

LCD_ENDPOINT   = os.getenv('LCD_ENDPOINT',  'https://rest-kralum.neutron-1.neutron.org')
POINTS_CONTRACT = os.getenv('POINTS_CONTRACT', 'neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')

async def get_controller_address(env: str = "mainnet"):
    """Return the controller/lens contract address used to query market data."""
    address = AMBER_CONTROLLER_ADDRESSES.get(env)
    if not address:
        raise HTTPException(status_code=400, detail="Unsupported environment")
    return {"env": env, "controller_address": address}


# step:2 file: list_current_amber_lending_markets_and_apys
import base64, json, os
import httpx
from fastapi import HTTPException

LCD_ENDPOINT = os.getenv("NEUTRON_LCD_ENDPOINT", "https://rest-kralum.neutron-1.neutron.org")

async def _query_smart(contract_address: str, query_msg: dict):
    """Helper to perform a CosmWasm smart-query using the LCD REST interface."""
    encoded_msg = base64.b64encode(json.dumps(query_msg).encode()).decode()
    url = f"{LCD_ENDPOINT}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{encoded_msg}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
        return resp.json().get("data") or resp.json()

@app.get("/api/amber/markets")
async def get_markets(env: str = "mainnet"):
    from amber_api import AMBER_CONTROLLER_ADDRESSES  # reuse mapping from step 1
    controller = AMBER_CONTROLLER_ADDRESSES.get(env)
    if not controller:
        raise HTTPException(status_code=400, detail="Unsupported environment")
    markets = await _query_smart(controller, {"markets": {}})
    return markets


# step:3 file: list_current_amber_lending_markets_and_apys
from fastapi import HTTPException

@app.get("/api/amber/market-state")
async def get_market_state(market_id: str, env: str = "mainnet"):
    from amber_api import AMBER_CONTROLLER_ADDRESSES
    controller = AMBER_CONTROLLER_ADDRESSES.get(env)
    if not controller:
        raise HTTPException(status_code=400, detail="Unsupported environment")
    state = await _query_smart(controller, {"market_state": {"market_id": market_id}})
    return state


# step:4 file: send_1,000,000_ujuno_to_a_cosmwasm_contract_via_an_execute_endpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import json, base64

from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wasm import MsgExecuteContract
from cosmpy.protos.cosmos.base.v1beta1.coin_pb2 import Coin

app = FastAPI()

JUNO_MAINNET = NetworkConfig(
    chain_id="juno-1",
    url="https://rpc-juno.itastakers.com:443",
    fee_minimum_gas_price=0.025,
    fee_denomination="ujuno",
)

class ExecutePayload(BaseModel):
    sender_address: str
    contract_address: str
    execute_msg: Dict[str, Any]
    funds: int  # ujuno

@app.post("/api/tx/execute/construct")
async def construct_execute_tx(p: ExecutePayload):
    try:
        client = LedgerClient(JUNO_MAINNET)

        coin = Coin(amount=str(p.funds), denom="ujuno")
        msg = MsgExecuteContract(
            sender=p.sender_address,
            contract=p.contract_address,
            msg=json.dumps(p.execute_msg).encode(),
            funds=[coin],
        )

        tx = Transaction()
        tx.add_message(msg)
        gas = client.estimate_gas(tx)

        return {
            "unsigned_tx": base64.b64encode(tx.serialize()).decode(),
            "gas_estimate": gas,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# step:5 file: send_1,000,000_ujuno_to_a_cosmwasm_contract_via_an_execute_endpoint
from fastapi import Depends
from cosmpy.aerial.wallet import LocalWallet

class SignBroadcastPayload(ExecutePayload):
    mnemonic: str  # securely store / transmit in real apps!

@app.post("/api/tx/execute/sign_and_broadcast")
async def sign_and_broadcast(p: SignBroadcastPayload):
    try:
        wallet = LocalWallet.from_mnemonic(p.mnemonic)
        if wallet.address() != p.sender_address:
            raise HTTPException(status_code=400, detail="Mnemonic does not match sender")

        client = LedgerClient(JUNO_MAINNET)

        coin = Coin(amount=str(p.funds), denom="ujuno")
        msg = MsgExecuteContract(
            sender=p.sender_address,
            contract=p.contract_address,
            msg=json.dumps(p.execute_msg).encode(),
            funds=[coin],
        )

        tx = Transaction()
        tx.add_message(msg)
        tx.with_chain_id(JUNO_MAINNET.chain_id)
        tx.with_gas_limit(300_000)
        tx_signed = tx.sign(wallet)

        tx_hash = client.broadcast_tx(tx_signed)
        return {"tx_hash": tx_hash}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# step:2 file: lock_a_specific_account_so_it_can_no_longer_send_transactions
######################## rpc_helpers.py ########################
import os
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

JSON_RPC_URL = os.getenv("JSON_RPC_URL", "http://localhost:8545")
AUTH_TOKEN = os.getenv("RPC_AUTH_TOKEN")  # set this in your environment if required

logger = logging.getLogger(__name__)

class RPCError(Exception):
    """Custom exception for JSON-RPC errors."""

    def __init__(self, code: int, message: str, data: Optional[Any] = None):
        super().__init__(f"RPC error {code}: {message}")
        self.code = code
        self.data = data


def _rpc_headers() -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json"
    }
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    return headers


def rpc_request(method: str, params: Optional[List[Any]] = None, *, id: int = 1) -> Any:
    """Low-level JSON-RPC request with bearer-token authentication and basic error handling."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or [],
        "id": id,
    }

    try:
        response = httpx.post(JSON_RPC_URL, headers=_rpc_headers(), json=payload, timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as http_err:
        logger.exception("HTTP error while calling %s", method)
        raise RuntimeError(f"HTTP error while calling {method}: {http_err}") from http_err

    data = response.json()

    if "error" in data:
        err = data["error"]
        raise RPCError(err.get("code", -1), err.get("message", "Unknown error"), err.get("data"))

    return data.get("result")


# step:3 file: lock_a_specific_account_so_it_can_no_longer_send_transactions
######################## app.py ########################
import os
from flask import Flask, request, jsonify
from rpc_helpers import rpc_request, RPCError

app = Flask(__name__)

@app.route("/api/lock_account", methods=["POST"])
def lock_account():
    """HTTP endpoint → personal_lockAccount JSON-RPC call."""
    data = request.get_json(force=True, silent=True) or {}
    address = data.get("address")

    if not address or not isinstance(address, str):
        return jsonify({"error": "Missing or invalid 'address' field"}), 400

    try:
        result = rpc_request("personal_lockAccount", [address])
        # Geth returns 'true' if the account was successfully removed from memory
        return jsonify({"address": address, "locked": bool(result)})
    except RPCError as rpc_err:
        return jsonify({"error": str(rpc_err), "code": rpc_err.code}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


# step:1 file: run_hardhat_tests_with_npx_hardhat_test
from fastapi import FastAPI, HTTPException
import subprocess
import shlex

app = FastAPI()

@app.post("/api/run-tests")
async def run_tests():
    """Run the project's Hardhat test suite and return stdout or an error."""
    command = "npx hardhat test"  # Assumes Hardhat is installed and in PATH
    try:
        # Run the command with a 15-minute timeout to prevent hanging.
        process = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            timeout=900,
            check=False  # We'll handle non-zero exit codes ourselves.
        )

        # If tests failed, surface the stderr so the caller can debug.
        if process.returncode != 0:
            raise HTTPException(status_code=400, detail=process.stderr)

        # Tests passed; return the captured stdout.
        return {"output": process.stdout}

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Test execution timed out.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:2 file: check_the_balance_of_a_wallet_address_on_the_juno_network
'''backend_balances.py ------------------------------------------------------
FastAPI microservice that proxies a balance query to a public Juno LCD API.
This keeps CORS & key-management concerns off the frontend.
'''

import re
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI()

JUNO_LCD_DEFAULT = "https://api.juno.kingnodes.com"  # Public LCD; replace if needed
ADDR_REGEX = re.compile(r"^juno1[0-9a-z]{38}$")

@app.get("/api/raw_balances", response_model=Dict[str, Any])
async def query_raw_balances(address: str, node_url: str = JUNO_LCD_DEFAULT):
    """Fetches *unmodified* balance JSON from the LCD for the provided address."""
    if not ADDR_REGEX.match(address):
        raise HTTPException(status_code=400, detail="Invalid Juno address format.")

    endpoint = f"{node_url}/cosmos/bank/v1beta1/balances/{address}"

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(endpoint)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # Surface LCD or network failures clearly to consumers
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return response.json()


# step:3 file: check_the_balance_of_a_wallet_address_on_the_juno_network
from typing import List, Dict, Any

@app.post("/api/parse_balances", response_model=List[Dict[str, str]])
async def parse_balances(raw_json: Dict[str, Any]):
    """Extracts `[{'denom': <denom>, 'amount': <amount>}, …]` from the raw LCD payload."""
    try:
        raw_balances = raw_json.get("balances", [])
        simplified = [
            {"denom": entry.get("denom", ""), "amount": entry.get("amount", "0")}  
            for entry in raw_balances
        ]
    except (AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Malformed JSON supplied. Expecting LCD bank/balances response body.")

    return simplified


# step:1 file: back_up_priv_val_state.json_to_prevent_double_signing
def stop_cosmos_service(service_name: str = "cosmosd") -> None:
    """Gracefully stop a running Cosmos systemd service."""
    import subprocess

    try:
        # Use systemd to stop the validator
        subprocess.run(["sudo", "systemctl", "stop", service_name], check=True, text=True)
        print(f"Successfully stopped {service_name}.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to stop {service_name}: {err}")


# step:2 file: back_up_priv_val_state.json_to_prevent_double_signing
def verify_file_exists(file_path: str) -> None:
    """Raise FileNotFoundError if a required file is missing."""
    import os

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    print(f"Verified file exists: {file_path}")


# step:3 file: back_up_priv_val_state.json_to_prevent_double_signing
def backup_priv_val_state(node_home: str, backup_dir: str = "/var/backups") -> str:
    """Return the absolute path of the created backup file."""
    import os, shutil, datetime

    src_path = os.path.join(node_home, "data", "priv_val_state.json")
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup_name = f"priv_val_state_{timestamp}.json"
    dst_path = os.path.join(backup_dir, backup_name)

    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(src_path, dst_path)  # preserves metadata
    print(f"Backup created at: {dst_path}")
    return dst_path


# step:4 file: back_up_priv_val_state.json_to_prevent_double_signing
def _sha256(file_path: str) -> str:
    import hashlib
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


# step:5 file: back_up_priv_val_state.json_to_prevent_double_signing
def secure_offsite_copy(local_path: str, remote_user: str, remote_host: str, remote_dir: str) -> None:
    """SCP the backup to an off-site server."""
    import os, subprocess

    remote_target = f"{remote_user}@{remote_host}:{remote_dir}/{os.path.basename(local_path)}"
    try:
        subprocess.run(["scp", "-p", local_path, remote_target], check=True, text=True)
        print(f"Successfully copied to {remote_target}")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Secure copy failed: {err}")


# step:6 file: back_up_priv_val_state.json_to_prevent_double_signing
def start_cosmos_service(service_name: str = "cosmosd") -> None:
    """Restart the Cosmos validator systemd unit."""
    import subprocess

    try:
        subprocess.run(["sudo", "systemctl", "start", service_name], check=True, text=True)
        print(f"Successfully started {service_name}.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to start {service_name}: {err}")


# step:3 file: Query the connected wallet’s NTRN balance
from typing import Dict
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import Address

# Configure connection details for Neutron main-net RPC
NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url="https://rpc-kral.neutron.org:443",  # Public RPC endpoint
    fee_minimum_gas_price=0.025,
    fee_denom="untrn",
)

def query_bank_balance(address: str, denom: str = "untrn") -> Dict[str, int]:
    """
    Query the on-chain bank balance for a specific address and denomination.

    Parameters
    ----------
    address : str
        Bech32 Neutron address to query.
    denom : str
        Token denomination (default: "untrn").

    Returns
    -------
    Dict[str, int]
        { "raw_balance": <amount_in_micro_denom> }
    """
    try:
        with LedgerClient(NETWORK) as client:
            balance_coin = client.query_bank_balance(Address(address), denom=denom)
            return {"raw_balance": int(balance_coin.amount)}
    except Exception as err:
        # Re-throw with context so callers know exactly what went wrong
        raise RuntimeError(f"Failed to query bank balance: {err}") from err


# step:4 file: Query the connected wallet’s NTRN balance
def format_amount(raw_balance: int) -> str:
    """Convert micro-denom (`untrn`) to a formatted NTRN string."""
    try:
        micro = int(raw_balance)
    except (TypeError, ValueError):
        raise ValueError("raw_balance must be an integer-compatible value")

    ntrn_value = micro / 1_000_000  # 1 NTRN = 1,000,000 untrn
    return f"{ntrn_value:,.6f} NTRN"


# step:1 file: enable_the_rest_server_and_expose_swagger_documentation
import os

# Step 1: Utilities to locate the chain's home directory and the app.toml file

def get_chain_home(custom_home: str = None) -> str:
    """Return the absolute path to the node's home directory.

    Priority order:
      1) custom_home argument if provided
      2) $SIMD_HOME environment variable if set
      3) Default to ~/.simapp
    """
    home = custom_home or os.getenv('SIMD_HOME') or os.path.expanduser('~/.simapp')
    home = os.path.expanduser(home)
    if not os.path.isdir(home):
        raise FileNotFoundError(f"Chain home directory not found: {home}")
    return home


def get_app_toml(home_dir: str) -> str:
    """Return the absolute path to config/app.toml given a chain home directory."""
    app_toml = os.path.join(home_dir, 'config', 'app.toml')
    if not os.path.isfile(app_toml):
        raise FileNotFoundError(f"app.toml not found at {app_toml}")
    return app_toml



# step:2 file: enable_the_rest_server_and_expose_swagger_documentation
import shutil
import os

# Step 2: Backup the current configuration file

def backup_file(file_path: str) -> str:
    """Copy <file_path> to <file_path>.bak, preserving metadata."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    backup_path = file_path + '.bak'
    shutil.copy2(file_path, backup_path)
    return backup_path



# step:3 file: enable_the_rest_server_and_expose_swagger_documentation
import os
import toml

# Step 3: Perform in-place update of REST API settings inside app.toml

def enable_rest_api(app_toml_path: str) -> None:
    """Sets api.enable = true and api.swagger = true inside app.toml."""
    # Load current config
    try:
        config = toml.load(app_toml_path)
    except toml.TomlDecodeError as err:
        raise ValueError(f'Failed to parse TOML file: {err}') from err

    # Ensure the api table exists and toggle the desired flags
    api_cfg = config.get('api', {})
    api_cfg['enable'] = True
    api_cfg['swagger'] = True
    config['api'] = api_cfg

    # Persist changes atomically via temporary file replacement
    tmp_path = app_toml_path + '.tmp'
    with open(tmp_path, 'w') as tmp_file:
        toml.dump(config, tmp_file)
    os.replace(tmp_path, app_toml_path)



# step:4 file: enable_the_rest_server_and_expose_swagger_documentation
import subprocess
import time

# Step 4: Restart the local simd node

def restart_simd(home_dir: str, start_command: str = 'simd start') -> None:
    """Restarts the simd process using a best-effort cross-platform approach."""
    try:
        # Attempt to gracefully stop any running simd instance
        subprocess.run(['pkill', '-f', 'simd'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # brief pause to ensure the process has terminated

        # Start the node in a detached process so the Python script can exit
        cmd = start_command.split() + ['--home', home_dir]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError(f'Failed to restart simd: {e}') from e



# step:5 file: enable_the_rest_server_and_expose_swagger_documentation
import requests
import time

# Step 5: Verify REST & Swagger are live

def wait_for_swagger(url: str = 'http://localhost:1317/swagger', timeout: int = 30) -> bool:
    """Poll <url> until it returns HTTP 200 or raise TimeoutError after <timeout> seconds."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass  # keep trying until timeout
        time.sleep(1)
    raise TimeoutError(f'Swagger endpoint did not become available within {timeout} seconds')



# step:1 file: compile_the_clock_example.wasm_contract_and_upload_it_to_the_juno_testnet
import os
import subprocess
from pathlib import Path


def compile_contract(source_dir: str, output_dir: str | None = None) -> str:
    '''
    Compile a CosmWasm contract using the rust-optimizer Docker image.

    Args:
        source_dir: root folder of the contract (where Cargo.toml lives).
        output_dir: folder that will contain the optimized .wasm. Defaults to <source_dir>/artifacts.

    Returns:
        Absolute path to the optimized .wasm file.
    '''
    source_path = Path(source_dir).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f'Contract directory {source_path} does not exist')

    out_path = Path(output_dir) if output_dir else source_path / 'artifacts'
    out_path = out_path.resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    crate_name = source_path.name.replace('-', '_')
    artifact_file = out_path / f'{crate_name}.wasm'

    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{source_path}:/code',
        '-v', f'{out_path}:/code/artifacts',
        '-w', '/code',
        'cosmwasm/workspace-optimizer:0.14.0'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f'rust-optimizer failed (exit {e.returncode}):\nSTDOUT:\n{e.stdout.decode()}\nSTDERR:\n{e.stderr.decode()}'
        )

    if not artifact_file.exists():
        raise RuntimeError(f'Compilation finished but {artifact_file} was not produced.')

    return str(artifact_file)



# step:2 file: compile_the_clock_example.wasm_contract_and_upload_it_to_the_juno_testnet
import os
from pathlib import Path

from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.protos.cosmwasm.wasm.v1 import tx_pb2 as wasm_tx_pb

JUNO_RPC = os.getenv('JUNO_RPC', 'https://rpc.uni.juno.deuslabs.fi:443')
CHAIN_ID = os.getenv('CHAIN_ID', 'uni-6')
FEE_DENOM = 'ujunox'
GAS_PRICE = float(os.getenv('GAS_PRICE', '0.025'))  # ujunox per gas unit


def store_wasm(wasm_path: str, mnemonic: str) -> str:
    '''
    Upload a compiled CosmWasm contract to the chain and return the tx hash.
    '''
    path = Path(wasm_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Wasm file {path} does not exist')

    wallet = LocalWallet.from_mnemonic(mnemonic)

    cfg = NetworkConfig(
        chain_id=CHAIN_ID,
        url=JUNO_RPC,
        fee_minimum_gas_price=GAS_PRICE,
        fee_denomination=FEE_DENOM,
        staking_denomination=FEE_DENOM,
    )
    client = LedgerClient(cfg)

    msg = wasm_tx_pb.MsgStoreCode(
        sender=wallet.address(),
        wasm_byte_code=path.read_bytes(),
    )

    tx = (
        Transaction()
        .with_messages(msg)
        .with_chain_id(CHAIN_ID)
        .with_fee_denomination(FEE_DENOM)
        .with_gas_auto_estimate(client, wallet.address())
    )

    signed = wallet.sign(tx)
    resp = client.broadcast_tx_async(signed)
    if resp.is_ok():
        return resp.txhash
    raise RuntimeError(f'Broadcast failed: {resp.raw_log}')



# step:3 file: compile_the_clock_example.wasm_contract_and_upload_it_to_the_juno_testnet
import json
import os
import time
from typing import Optional

from cosmpy.aerial.client import LedgerClient, NetworkConfig

JUNO_RPC = os.getenv('JUNO_RPC', 'https://rpc.uni.juno.deuslabs.fi:443')
CHAIN_ID = os.getenv('CHAIN_ID', 'uni-6')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))  # seconds
MAX_POLLS = int(os.getenv('MAX_POLLS', '60'))


def _extract_code_id(raw_log: str) -> Optional[str]:
    try:
        logs = json.loads(raw_log or '[]')
    except Exception:
        return None
    for entry in logs:
        for event in entry.get('events', []):
            if event.get('type') == 'store_code':
                for attr in event.get('attributes', []):
                    if attr.get('key') == 'code_id':
                        return attr.get('value')
    return None


def poll_tx_for_code_id(tx_hash: str) -> str:
    '''Polls for tx inclusion and returns the emitted code_id.'''    
    cfg = NetworkConfig(url=JUNO_RPC, chain_id=CHAIN_ID)
    client = LedgerClient(cfg)

    for _ in range(MAX_POLLS):
        tx = client.query_tx(tx_hash)
        if tx:
            if tx.get('code', 0) != 0:
                raise RuntimeError(f'Tx failed with code {tx["code"]}: {tx.get("raw_log")}')
            code_id = _extract_code_id(tx.get('raw_log', ''))
            if code_id:
                return code_id
            raise RuntimeError('Tx succeeded but code_id not found in logs.')
        time.sleep(POLL_INTERVAL)
    raise TimeoutError('Timed out waiting for transaction to be included.')



# step:1 file: install_hardhat_with_@nomicfoundation_hardhat-toolbox_in_a_new_project
/* checkEnv.js -- Verify that Node.js >=16 and npm are installed. */
const { execSync } = require('child_process');

function getMajor(ver) {
  const match = ver.match(/v?(\d+)\./);
  return match ? parseInt(match[1], 10) : 0;
}

try {
  const nodeVersion = process.version; // e.g., v18.17.0
  const npmVersion = execSync('npm -v').toString().trim(); // e.g., 9.6.1

  if (getMajor(nodeVersion) < 16) {
    throw new Error(`Node.js version ${nodeVersion} detected. v16 or newer is required.`);
  }

  console.log(`✅ Environment OK -> Node ${nodeVersion}, npm ${npmVersion}`);
} catch (err) {
  console.error('❌ Environment check failed:', err.message);
  process.exit(1);
}


# step:2 file: install_hardhat_with_@nomicfoundation_hardhat-toolbox_in_a_new_project
/* initProject.js -- Creates a new folder and runs `npm init -y` */
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function init(projectName = 'hardhat-project') {
  const projectPath = path.resolve(process.cwd(), projectName);

  if (!fs.existsSync(projectPath)) {
    fs.mkdirSync(projectPath, { recursive: true });
    console.log(`📁 Created directory: ${projectPath}`);
  } else {
    console.log(`📂 Using existing directory: ${projectPath}`);
  }

  // Change working directory to the project folder
  process.chdir(projectPath);

  console.log('⚙️  Initializing npm project...');
  execSync('npm init -y', { stdio: 'inherit' });

  console.log('✅ package.json generated.');
}

// Allow running from CLI: node initProject.js my-project-name
if (require.main === module) {
  const [, , projectNameArg] = process.argv;
  init(projectNameArg);
}

module.exports = { init };


# step:3 file: install_hardhat_with_@nomicfoundation_hardhat-toolbox_in_a_new_project
/* installDeps.js -- Installs Hardhat and toolbox as devDependencies */
const { execSync } = require('child_process');

function install() {
  console.log('⬇️  Installing Hardhat and toolbox...');
  try {
    execSync('npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox', { stdio: 'inherit' });
    console.log('✅ Dependencies installed.');
  } catch (err) {
    console.error('❌ Failed to install dependencies:', err.message);
    process.exit(1);
  }
}

if (require.main === module) {
  install();
}

module.exports = { install };


# step:4 file: install_hardhat_with_@nomicfoundation_hardhat-toolbox_in_a_new_project
/* initHardhat.js -- Bootstraps a Hardhat project template */
const { execSync } = require('child_process');

function initHardhat() {
  console.log('🚀 Running Hardhat initializer...');
  try {
    // Execute Hardhat initialization (interactive)
    execSync('npx hardhat', { stdio: 'inherit' });
    console.log('✅ Hardhat project initialized.');
  } catch (err) {
    console.error('❌ Hardhat initialization failed:', err.message);
    process.exit(1);
  }
}

if (require.main === module) {
  initHardhat();
}

module.exports = { initHardhat };


# step:2 file: broadcast_a_raw,_rlp-encoded_signed_transaction_to_the_network
from fastapi import APIRouter, HTTPException
import os, requests

router = APIRouter()

# Comma-separated list of candidate endpoints, configurable via env.
RPC_ENDPOINTS = os.getenv('EVM_RPC_ENDPOINTS', 'https://rpc-evmos.cosmos.network,https://json-rpc.ethermint.org').split(',')


def _is_endpoint_alive(url: str) -> bool:
    """Return True if the endpoint responds successfully to eth_chainId."""
    try:
        payload = {"jsonrpc": "2.0", "method": "eth_chainId", "params": [], "id": 1}
        r = requests.post(url, json=payload, timeout=5)
        return r.status_code == 200 and 'result' in r.json()
    except Exception:
        return False


@router.get('/api/get_rpc_endpoint')
async def get_rpc_endpoint():
    """Pick the first healthy RPC endpoint from the list and return it."""
    for url in RPC_ENDPOINTS:
        if _is_endpoint_alive(url):
            return {"endpoint": url}
    raise HTTPException(status_code=503, detail='No healthy RPC endpoint found')


# step:3 file: broadcast_a_raw,_rlp-encoded_signed_transaction_to_the_network
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os, requests

router = APIRouter()

class TxPayload(BaseModel):
    raw_tx: str

def _broadcast(endpoint: str, raw_tx: str) -> str:
    """Attempt to broadcast the raw transaction."""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_sendRawTransaction",
        "params": [raw_tx],
        "id": 1
    }
    response = requests.post(endpoint, json=payload, timeout=20)

    # HTTP-level errors
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f'RPC status code {response.status_code}')

    data = response.json()

    # JSON-RPC-level errors
    if 'error' in data:
        raise HTTPException(status_code=400, detail=data['error'])

    # Successful path returns the 0x-prefixed tx hash
    return data['result']

@router.post('/api/broadcast_raw_tx')
async def broadcast_raw_tx(payload: TxPayload):
    """Try each configured endpoint until the tx is accepted, then return its hash."""
    endpoints = os.getenv('EVM_RPC_ENDPOINTS', 'https://rpc-evmos.cosmos.network,https://json-rpc.ethermint.org').split(',')
    for endpoint in endpoints:
        try:
            tx_hash = _broadcast(endpoint, payload.raw_tx)
            return {"tx_hash": tx_hash, "endpoint": endpoint}
        except HTTPException:
            # Allow the next endpoint a chance
            continue
    raise HTTPException(status_code=503, detail='Failed to broadcast transaction on all endpoints')


# step:1 file: stake_tokens_in_the_liquidstakingvault_contract_via_the_stake_function
# backend/ethereum_provider.py
from web3 import Web3
import os


def get_web3_provider() -> Web3:
    # Connects to an Ethereum JSON-RPC endpoint based on ETH_RPC_URL environment variable.
    rpc_url = os.getenv("ETH_RPC_URL", "https://mainnet.infura.io/v3/YOUR_API_KEY")
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))

    # Verify connectivity
    if not w3.isConnected():
        raise ConnectionError(f"Unable to connect to Ethereum node at {rpc_url}")

    return w3


# step:2 file: stake_tokens_in_the_liquidstakingvault_contract_via_the_stake_function
# backend/credentials.py
import os
from eth_account import Account
from web3 import Web3


def load_signer(web3: Web3):
    # Loads the private key from the PRIVATE_KEY environment variable and returns an Account object.
    private_key = os.getenv("PRIVATE_KEY")
    if not private_key:
        raise ValueError("PRIVATE_KEY environment variable is not set.")

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    account = web3.eth.account.from_key(private_key)
    return account


# step:3 file: stake_tokens_in_the_liquidstakingvault_contract_via_the_stake_function
# backend/contract.py
import json
import os
from web3 import Web3


def get_liquid_staking_vault_contract(web3: Web3):
    # Instantiates the LiquidStakingVault contract using its address and ABI.
    contract_address = os.getenv("VAULT_CONTRACT_ADDRESS")
    if not contract_address:
        raise ValueError("VAULT_CONTRACT_ADDRESS environment variable is not set.")
    contract_address = Web3.toChecksumAddress(contract_address)

    abi_path = os.getenv("VAULT_CONTRACT_ABI_PATH", "backend/abis/LiquidStakingVault.abi.json")

    try:
        with open(abi_path, "r") as abi_file:
            abi = json.load(abi_file)
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Unable to locate ABI file at {abi_path}") from err

    contract = web3.eth.contract(address=contract_address, abi=abi)
    return contract


# step:4 file: stake_tokens_in_the_liquidstakingvault_contract_via_the_stake_function
# backend/transactions.py
from decimal import Decimal
from web3 import Web3


def build_stake_tx(web3: Web3, contract, signer, amount_eth: float, recipient_address: str):
    # Builds an unsigned stake transaction for vault.stake(amount, recipient).
    try:
        amount_wei = web3.to_wei(Decimal(str(amount_eth)), "ether")
    except Exception as err:
        raise ValueError(f"Invalid staking amount: {err}") from err

    recipient_address = Web3.toChecksumAddress(recipient_address)

    nonce = web3.eth.get_transaction_count(signer.address)

    tx = contract.functions.stake(amount_wei, recipient_address).build_transaction({
        "from": signer.address,
        "nonce": nonce
    })

    return tx


# step:5 file: stake_tokens_in_the_liquidstakingvault_contract_via_the_stake_function
# backend/gas.py
from web3 import Web3


def finalize_gas_and_chain_id(web3: Web3, tx: dict) -> dict:
    # Estimates gas, sets gas price, and fills in chainId.
    try:
        tx["gas"] = web3.eth.estimate_gas(tx)
    except Exception as err:
        raise ValueError(f"Gas estimation failed: {err}") from err

    try:
        tx["gasPrice"] = web3.eth.gas_price
    except Exception:
        tx["gasPrice"] = web3.to_wei(3, "gwei")

    tx["chainId"] = web3.eth.chain_id

    return tx


# step:6 file: stake_tokens_in_the_liquidstakingvault_contract_via_the_stake_function
# backend/broadcast.py
from web3 import Web3


def sign_and_send_tx(web3: Web3, signer, tx: dict) -> str:
    # Signs the provided transaction and broadcasts it to the network.
    try:
        signed_tx = signer.sign_transaction(tx)
    except Exception as err:
        raise ValueError(f"Failed to sign transaction: {err}") from err

    try:
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    except Exception as err:
        raise ConnectionError(f"Broadcast failed: {err}") from err

    return tx_hash.hex()


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


def get_admin_address() -> str:
    """A small helper that wraps get_admin_wallet() and only returns the Bech32 address."""
    return str(get_admin_wallet().address())


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


# step:3 file: Reset the global counter in an example contract
import json
import os
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.client import NetworkConfig
from cosmpy.aerial.protoutil import create_msg_execute_contract

# A single NetworkConfig / LedgerClient can be re-used across calls
NETWORK_CFG = NetworkConfig(
    chain_id=os.getenv("CHAIN_ID", "neutron-1"),
    url=os.getenv("RPC_ENDPOINT", "https://rpc-kralum.neutron.org"),
    fee_min_denom="untrn",
)


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


# step:4 file: Reset the global counter in an example contract
from cosmpy.aerial.client import LedgerClient

# LedgerClient must share the same NetworkConfig used when building the tx
LEDGER = LedgerClient(NETWORK_CFG)


def sign_and_broadcast_tx(tx: Transaction, wallet) -> dict:
    """Sign a Transaction with admin wallet and broadcast it. Raises on failure."""
    try:
        wallet.sign_transaction(tx)
        result = LEDGER.broadcast_tx(tx)
    except Exception as err:
        raise RuntimeError(f"Broadcast failed: {err}") from err

    # `result` is a tx_response dict coming from Tendermint
    if int(result.get("code", 0)) != 0:
        raise RuntimeError(f"Tx error (code {result['code']}): {result['raw_log']}")

    print(f"TxHash {result['txhash']} included in block {result['height']}")
    return result


# step:5 file: Reset the global counter in an example contract
def query_wasm_contract_state(contract_addr: str) -> int:
    """Query `{ "get_count": {} }` from the counter contract and return the integer count."""
    query_msg = {"get_count": {}}
    try:
        response = LEDGER.query_contract_smart(contract_addr, query_msg)
    except Exception as err:
        raise RuntimeError(f"Smart-contract query failed: {err}") from err

    if "count" not in response:
        raise ValueError(f"Unexpected response shape: {response}")
    return int(response["count"])


# step:1 file: add_a_blockscout_block-explorer_url_to_a_local_or_dapp_network_configuration
# network_config_utils.py

import os
import json
from typing import Optional


def find_network_config(root_dir: str = ".") -> Optional[str]:
    """Recursively search *root_dir* for a probable network configuration file.

    A file is considered a match if it:
      • Has a JSON extension **and** contains keys like `chainId`, `rpc`, `explorer`, or `chain_name`.
      • OR (for *.js, *.ts, *.yaml, *.yml, *.env*) contains the strings `chain-registry`, `walletConnect`, or `explorer`.

    Returns the full path of the first match, or *None* if nothing is found.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in {".json", ".js", ".ts", ".yaml", ".yml", ".env"}:
                continue
            full_path = os.path.join(dirpath, filename)
            try:
                if ext == ".json":
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and any(k in data for k in ("chainId", "rpc", "explorer", "chain_name")):
                        return full_path
                else:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if any(keyword in content for keyword in ("chain-registry", "walletConnect", "explorer", "rpc")):
                        return full_path
            except Exception:
                # Ignore parse errors and keep searching
                continue
    return None


if __name__ == "__main__":
    path = find_network_config()
    if path:
        print(f"Found network config file at: {path}")
    else:
        print("No network config file found.")


# step:2 file: add_a_blockscout_block-explorer_url_to_a_local_or_dapp_network_configuration
# update_blockscout_url.py

import json
import shutil
import sys
from pathlib import Path
from typing import Union

DEFAULT_BLOCKSCOUT_URL = "https://blockscout.evmos.org"


def insert_blockscout_url(config_path: Union[str, Path], blockscout_url: str = DEFAULT_BLOCKSCOUT_URL) -> None:
    """Add or update an `explorer` key in a JSON config file.

    A *.bak* backup is written before overwriting the original file.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    # Backup original
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    shutil.copyfile(config_path, backup)
    print(f"Backup created at {backup}")

    # Load JSON and mutate
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object in config file")

    previous = data.get("explorer")
    data["explorer"] = blockscout_url

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    action = "updated" if previous else "inserted"
    print(f"`explorer` field {action} -> {blockscout_url}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_blockscout_url.py <config_path> [blockscout_url]")
        sys.exit(1)

    cfg = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BLOCKSCOUT_URL
    insert_blockscout_url(cfg, url)


# step:3 file: add_a_blockscout_block-explorer_url_to_a_local_or_dapp_network_configuration
// propagate_config.js

/**
 * Removes common cache directories and (re)starts the dev server.
 * Works for npm or yarn projects. Assumes a script named `dev` exists.
 */

const { execSync, spawn } = require("child_process");

function propagateConfig() {
  try {
    console.log("Stopping existing dev servers (if any)...");
    execSync('pkill -f "npm run dev" || true', { stdio: "inherit" });
  } catch (_) {
    /* It is fine if no process was found */
  }

  try {
    console.log("Clearing build/cache directories...");
    execSync("rm -rf .next cache dist node_modules/.cache || true", { stdio: "inherit" });
  } catch (err) {
    console.warn("Cache cleanup warning:", err.message);
  }

  console.log("Starting dev server...");
  const cmd = process.env.USE_YARN ? "yarn" : "npm";
  const child = spawn(cmd, ["run", "dev"], { stdio: "inherit" });

  child.on("close", (code) => {
    console.log(`Dev server exited with code ${code}`);
  });
}

if (require.main === module) {
  propagateConfig();
}

module.exports = propagateConfig;


# step:2 file: execute_increment_on_contract_address_contract_address_with_10ujuno
### api/execute_increment.py
import os
import json
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.gas import GasPrice, calculate_fee
from cosmpy.common.types import Coins
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgExecuteContract

router = APIRouter()

# --------- configuration (override with env vars in production) ---------
RPC_ENDPOINT = os.getenv("RPC_ENDPOINT", "https://rpc.juno.deuslabs.fi")
CHAIN_ID     = os.getenv("CHAIN_ID", "juno-1")
GAS_PRICE    = GasPrice(0.025, "ujuno")  # 0.025 ujuno / gas is typical for Juno

NETWORK = NetworkConfig(
    chain_id=CHAIN_ID,
    url=RPC_ENDPOINT,
    fee_minimum_gas_price=GAS_PRICE.amount,
    fee_denomination=GAS_PRICE.denom,
)

# ------------------------ request schema -------------------------------
class ExecuteRequest(BaseModel):
    contractAddress: str
    executeMsg: dict

# ---------------------------- endpoint ---------------------------------
@router.post("/api/execute_increment")
async def execute_increment(req: ExecuteRequest):
    """Broadcast a MsgExecuteContract that sends `{increment:{}}` and 10 ujuno funds."""
    mnemonic = os.getenv("WALLET_MNEMONIC")
    if not mnemonic:
        raise HTTPException(status_code=500, detail="Missing WALLET_MNEMONIC environment variable")

    try:
        # 1.  Initialise wallet & client
        key     = PrivateKey.from_mnemonic(mnemonic)
        address = key.address()
        client  = LedgerClient(NETWORK)

        # 2.  Build the protobuf MsgExecuteContract
        msg = MsgExecuteContract(
            sender   = address,
            contract = req.contractAddress,
            msg      = json.dumps(req.executeMsg).encode(),
            funds    = Coins.from_coins("10ujuno")  # attach 10 ujuno as payment
        )

        # 3.  Craft & sign the transaction (automatic fee & gas-adjustment)
        tx = Transaction()
        tx.add_message(msg)
        tx.with_sender(address)
        tx.with_gas(calculate_fee(tx, gas_price=GAS_PRICE, gas_adjustment=1.3))
        signed_tx = tx.sign(key)

        # 4.  Broadcast and wait for finalization ("block" mode)
        result = client.broadcast_tx_block(signed_tx)
        if result.is_err():
            #  Broadcast succeeded but blockchain returned error code
            logging.error("execute_increment error: %s", result)
            raise HTTPException(status_code=400, detail=str(result))

        return {"txHash": result.tx_hash, "height": result.height}

    except Exception as e:
        logging.exception("execute_increment failed")
        raise HTTPException(status_code=500, detail=str(e))


# step:3 file: execute_increment_on_contract_address_contract_address_with_10ujuno
### utils/tx_waiter.py
import time
from typing import Optional
from cosmpy.aerial.client import LedgerClient
from .execute_increment import NETWORK  # reuse the same network config

client = LedgerClient(NETWORK)

async def wait_for_tx_commit(tx_hash: str, timeout: int = 60) -> Optional[dict]:
    """Poll the node every 2 s until the tx is indexed or the timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        tx_info = client.query_tx(tx_hash)
        if tx_info:
            return tx_info
        time.sleep(2)
    raise TimeoutError(f"Transaction {tx_hash} not found within {timeout}s")


# step:1 file: send_a_signed_transaction_with_cast
from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.get('/api/verify_cast')
async def verify_cast():
    """Endpoint that returns the installed version of `cast` or raises an error if it is not found."""
    try:
        version = subprocess.check_output(['cast', '--version'], text=True).strip()
        return {'installed': True, 'version': version}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail='Foundry `cast` CLI is not installed or not found in PATH.')
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f'Error executing cast --version: {err}')


# step:3 file: send_a_signed_transaction_with_cast
import os
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get('/api/check_private_key')
async def check_private_key():
    """Verifies that the server has a PRIVATE_KEY env variable configured."""
    if not os.getenv('PRIVATE_KEY'):
        raise HTTPException(status_code=500, detail='PRIVATE_KEY environment variable is missing.')
    return {'loaded': True}


# step:5 file: send_a_signed_transaction_with_cast
import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class TxParams(BaseModel):
    recipient: str
    amount: str
    rpc_url: str
    gas_price: str
    gas_limit: int
    data: str | None = None

@app.post('/api/send_tx')
async def send_tx(params: TxParams):
    """Signs & broadcasts a transaction using Foundry's `cast send`."""
    private_key = os.getenv('PRIVATE_KEY')
    if not private_key:
        raise HTTPException(status_code=500, detail='PRIVATE_KEY environment variable is not set on the server.')

    # Build the command
    cmd = [
        'cast', 'send', params.recipient, params.amount,
        '--rpc-url', params.rpc_url,
        '--private-key', private_key,
        '--gas-price', str(params.gas_price),
        '--gas-limit', str(params.gas_limit)
    ]
    if params.data:
        cmd.extend(['--data', params.data])

    try:
        # `cast send` prints the transaction hash on success
        tx_hash = subprocess.check_output(cmd, text=True).strip().split()[-1]
        return {'tx_hash': tx_hash}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f'Error sending transaction: {err.output}')


# step:6 file: send_a_signed_transaction_with_cast
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from web3 import Web3

app = FastAPI()

class WaitParams(BaseModel):
    tx_hash: str
    rpc_url: str
    timeout: int | None = 120  # seconds
    poll_interval: int | None = 5

@app.post('/api/wait_for_confirmation')
async def wait_for_confirmation(params: WaitParams):
    """Waits until the transaction is mined or the timeout is hit."""
    w3 = Web3(Web3.HTTPProvider(params.rpc_url))
    if not w3.isConnected():
        raise HTTPException(status_code=500, detail='Unable to connect to the provided RPC URL.')

    elapsed = 0
    while elapsed < params.timeout:
        try:
            receipt = w3.eth.get_transaction_receipt(params.tx_hash)
            if receipt and receipt.blockNumber:
                return {
                    'confirmed': True,
                    'blockNumber': receipt.blockNumber,
                    'status': receipt.status,
                    'tx_hash': params.tx_hash
                }
        except Exception:
            # Transaction not yet mined; ignore and continue polling
            pass
        await asyncio.sleep(params.poll_interval)
        elapsed += params.poll_interval

    raise HTTPException(status_code=504, detail='Timed out waiting for transaction confirmation.')


# step:3 file: get_the_current_count_value_from_the_contract_at_contract_address
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import aiohttp
import base64
import json

app = FastAPI()

@app.get('/api/query_count')
async def query_count(contract_address: str, rpc_endpoint: str = 'https://rpc.cosmos.directory/juno'):
    """
    Perform a CosmWasm smart-query (`get_count`) against the specified contract.

    Args:
        contract_address: The bech32 address of the contract.
        rpc_endpoint:   Base URL of a publicly accessible gRPC-gateway or LCD endpoint.
    """
    try:
        # 1. Build and encode the query
        query_dict = {"get_count": {}}
        encoded_query = base64.b64encode(json.dumps(query_dict).encode()).decode()

        # 2. Compose the REST path defined by Cosmos SDK <v0.46+>/CosmWasm
        #    Format: /cosmwasm/wasm/v1/contract/{address}/smart/{b64_encoded_query}
        url = f"{rpc_endpoint}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{encoded_query}"

        # 3. Execute HTTP GET
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise HTTPException(status_code=resp.status,
                                        detail=f'Upstream error: {error_text}')
                data = await resp.json()
                return JSONResponse(content=data)

    except Exception as e:
        # Surface any unexpected errors
        raise HTTPException(status_code=500, detail=str(e))


# step:2 file: mint_a_boost-receipt_nft_by_staking_250_ntrn_for_12_months
from fastapi import FastAPI, HTTPException
from cosmpy.aerial.client import LedgerClient, NetworkConfig

app = FastAPI(title='Neutron BFF')

NETWORK = NetworkConfig(
    chain_id='neutron-1',
    url='https://rpc-kralum.neutron.org',
    fee_denom='untrn',
    gas_price=0.01,
    prefix='neutron',
)

client = LedgerClient(NETWORK)

@app.get('/balance/{address}')
async def query_balance(address: str, denom: str = 'untrn'):
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


# step:3 file: mint_a_boost-receipt_nft_by_staking_250_ntrn_for_12_months
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.contract import MsgExecuteContract

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


# step:4 file: mint_a_boost-receipt_nft_by_staking_250_ntrn_for_12_months
import os
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.client import LedgerClient


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


# step:5 file: mint_a_boost-receipt_nft_by_staking_250_ntrn_for_12_months
import time
from cosmpy.aerial.client import LedgerClient


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


# step:6 file: mint_a_boost-receipt_nft_by_staking_250_ntrn_for_12_months
from cosmpy.aerial.client import LedgerClient


def query_nft_tokens(client: LedgerClient, contract_address: str, owner_address: str):
    query = { 'tokens': { 'owner': owner_address } }
    try:
        result = client.query_contract_smart(contract_address, query)
        # The exact shape depends on the contract; assume `{ tokens: [id1,id2,...] }` is returned
        return result.get('tokens', [])
    except Exception as e:
        raise RuntimeError(f'Contract query failed: {e}')


# step:1 file: write_unit_tests_for_a_solidity_contract_in_hardhat
/* File: test/MyContract.test.ts */
import { expect } from "chai";
import { ethers } from "hardhat";
import { MyContract } from "../typechain-types";

// Top-level test suite for the contract
describe("MyContract", () => {
  // The contract instance will be initialised in the next step
  let myContract: MyContract;

  // --- beforeEach & test cases are added in later steps ---
});


# step:2 file: write_unit_tests_for_a_solidity_contract_in_hardhat
/* Insert the following inside the describe block from Step 1 */

let myContract: MyContract; // re-declare for TypeScript scope safety

beforeEach(async () => {
  try {
    // Obtain a contract factory and deploy
    const MyContractFactory = await ethers.getContractFactory("MyContract");
    // Replace constructor arguments below if your contract requires any
    myContract = (await MyContractFactory.deploy()) as MyContract;
    await myContract.deployed(); // wait for tx to be mined
  } catch (error) {
    console.error("Contract deployment failed:", error);
    throw error; // Fail the test immediately if deployment breaks
  }
});


# step:3 file: write_unit_tests_for_a_solidity_contract_in_hardhat
/* Still inside test/MyContract.test.ts, underneath the beforeEach hook */

describe("MyContract core functionality", () => {
  it("returns the correct value from someCall()", async () => {
    // Arrange: prepare any state if necessary
    const expectedValue = 42;

    // Act: call the contract function under test
    const actualValue = await myContract.someCall();

    // Assert: verify that contract logic works as expected
    expect(actualValue).to.equal(expectedValue);
  });
});


# step:4 file: write_unit_tests_for_a_solidity_contract_in_hardhat
{
  "scripts": {
    "test": "npx hardhat test --network cosmos_evm"
  }
}


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


# step:3 file: Instantiate the example contract on Pion-1
from cosmpy.aerial.contract import MsgInstantiateContract
from cosmpy.aerial.tx import Transaction

# ---------------------------
# Step 3  •  Build Tx
# ---------------------------

def build_instantiate_tx(client: LedgerClient,
                         signer: PrivateKey,
                         code_id: int,
                         init_msg: dict | None = None,
                         admin: str | None = None,
                         label: str = "counter") -> Transaction:
    """Create an unsigned instantiate transaction."""

    if init_msg is None:
        init_msg = {"count": 0}

    msg = MsgInstantiateContract(
        sender=signer.address(),
        admin=admin or "",
        code_id=code_id,
        label=label,
        init_msg=init_msg,
        funds=[]  # no initial funds
    )

    tx = (Transaction()
           .with_messages(msg)
           .with_sender(signer.address())
           .with_chain_id(client.network_config.chain_id)
           .with_gas(300_000)  # enough for small contracts
           .with_fee(client.network_config.fee_denomination, 300_000 * client.network_config.fee_minimum_gas_price))

    return tx


# step:4 file: Instantiate the example contract on Pion-1
from cosmpy.aerial.tx import SigningCfg, BroadcastMode

# ---------------------------
# Step 4  •  Sign & Broadcast
# ---------------------------

def sign_and_broadcast_tx(client: LedgerClient, signer: PrivateKey, tx: Transaction):
    """Signs and broadcasts the transaction, returning the tx response."""
    try:
        tx_signed = tx.sign(SigningCfg.direct(signer))
        response  = client.broadcast_tx(tx_signed, mode=BroadcastMode.BLOCK)
        if response.is_tx_error():
            raise RuntimeError(f"Transaction failed: {response.raw_log}")
        return response
    except Exception as e:
        # Re-raise with additional context
        raise RuntimeError(f"Failed to broadcast instantiate tx: {e}") from e


# step:5 file: Instantiate the example contract on Pion-1
import json

# ---------------------------
# Step 5  •  Extract Address
# ---------------------------

def extract_contract_address_from_tx(tx_response):
    """Returns the contract address emitted by the instantiate event."""
    try:
        logs = json.loads(tx_response.raw_log)
        for event in logs[0].get("events", []):
            if event.get("type") in ("instantiate", "wasm"):
                for attr in event.get("attributes", []):
                    if attr.get("key") == "_contract_address" or attr.get("key") == "contract_address":
                        return attr.get("value")
        raise RuntimeError("Contract address not found in tx events.")
    except (KeyError, ValueError, IndexError) as e:
        raise RuntimeError(f"Error parsing tx log: {e}") from e


# step:1 file: add_my_validator_address_as_a_genesis_account_with_100000000000stake
from fastapi import FastAPI, HTTPException, Query
from bech32 import bech32_decode

app = FastAPI()

@app.get("/api/validate_address")
async def validate_address(address: str = Query(..., description="Bech32 address"), expected_hrp: str | None = Query(None, description="Expected prefix e.g. 'cosmos'")):
    '''Return True if the address is valid Bech32 and (optionally) matches the expected HRP.'''
    try:
        hrp, data = bech32_decode(address)
        if hrp is None or data is None:
            raise ValueError("Invalid Bech32 encoding or checksum.")
        if expected_hrp and hrp != expected_hrp:
            raise ValueError(f"Unexpected HRP. Expected '{expected_hrp}', got '{hrp}'.")
        return {"address": address, "hrp": hrp, "valid": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# step:2 file: add_my_validator_address_as_a_genesis_account_with_100000000000stake
import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class AddAccountPayload(BaseModel):
    address: str
    amount: int  # amount in base denomination units (e.g. 100000000000)
    denom: str = "stake"

@app.post("/api/add_genesis_account")
async def add_genesis_account(payload: AddAccountPayload):
    '''Invoke the chain binary to add an account to the genesis file.'''
    chain_binary = os.getenv("CHAIN_BINARY", "gaiad")
    cmd = [chain_binary, "add-genesis-account", payload.address, f"{payload.amount}{payload.denom}"]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return {"success": True, "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)


# step:3 file: add_my_validator_address_as_a_genesis_account_with_100000000000stake
import os
import subprocess
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/api/validate_genesis")
async def validate_genesis():
    '''Run the chain binary's genesis validation command and return the result.'''
    chain_binary = os.getenv("CHAIN_BINARY", "gaiad")
    cmd = [chain_binary, "validate-genesis"]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return {"success": True, "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)


# step:1 file: Create a schedule to rebalance portfolio every 3,600 blocks
def get_dao_authority_address(config_path: str = "dao_config.json") -> str:
    """
    Return the DAO authority address.
    Priority:
      1. Environment variable DAO_AUTHORITY_ADDRESS
      2. JSON file (default: dao_config.json) with key `authority_address`.
    """
    import os, json

    # 1️⃣  Environment override
    env_addr = os.getenv("DAO_AUTHORITY_ADDRESS")
    if env_addr:
        return env_addr.strip()

    # 2️⃣  Fallback to config file
    try:
        with open(config_path, "r") as fp:
            data = json.load(fp)
            return data["authority_address"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as err:
        raise RuntimeError("Unable to determine DAO authority address") from err



# step:2 file: Create a schedule to rebalance portfolio every 3,600 blocks
def build_msg_add_schedule(authority: str, contract_address: str, gas_limit: int = 200000):
    """
    Compose a MsgAddSchedule that calls `portfolio_manager.rebalance()` every 3600 seconds.
    """
    from google.protobuf.any_pb2 import Any
    from neutron.cron import cron_pb2 as crontypes
    from cosmwasm.wasm.v1 import tx_pb2 as wasmtypes

    if not authority or not contract_address:
        raise ValueError("Both `authority` and `contract_address` are required")

    # 👇 Inner Wasm execute message
    wasm_execute = wasmtypes.MsgExecuteContract(
        sender=authority,
        contract=contract_address,
        msg=b"{\"rebalance\":{}}",  # JSON → bytes
        funds=[]
    )

    inner_any = Any()
    inner_any.Pack(wasm_execute, type_url_prefix="/")

    # 👇 Cron schedule message
    schedule_msg = crontypes.MsgAddSchedule(
        authority=authority,
        name="portfolio_rebalance",
        period=3600,              # seconds
        msgs=[inner_any],
        gas_limit=gas_limit
    )
    return schedule_msg



# step:3 file: Create a schedule to rebalance portfolio every 3,600 blocks
def package_into_gov_proposal(authority: str, schedule_msg, deposit: int = 10000000, denom: str = "untrn"):
    """
    Package the schedule into a gov proposal message.
    """
    from google.protobuf.any_pb2 import Any
    from cosmos.gov.v1beta1 import gov_pb2 as govtypes
    from cosmos.base.v1beta1 import coin_pb2 as cointypes

    if schedule_msg is None:
        raise ValueError("`schedule_msg` cannot be None")

    content_any = Any()
    content_any.Pack(schedule_msg, type_url_prefix="/")

    proposal_msg = govtypes.MsgSubmitProposal(
        content=content_any,
        initial_deposit=[cointypes.Coin(amount=str(deposit), denom=denom)],
        proposer=authority
    )

    # Extra user-facing metadata (displayed by wallets/explorers)
    proposal_msg.title = "Add Cron Schedule: portfolio_rebalance"
    proposal_msg.summary = (
        "Adds an hourly cron job that calls portfolio_manager.rebalance to maintain target asset weights."
    )
    proposal_msg.metadata = "https://github.com/your-dao/proposals/001"
    return proposal_msg



# step:4 file: Create a schedule to rebalance portfolio every 3,600 blocks
def sign_and_broadcast_tx(proposal_msg, client, wallet, gas: int = 350000, fee_amount: int = 15000, fee_denom: str = "untrn") -> str:
    """
    Sign with the DAO authority wallet and broadcast the proposal transaction.

    Args:
        proposal_msg:   MsgSubmitProposal generated in step 3.
        client:         cosmpy LedgerClient (already connected to Neutron).
        wallet:         cosmpy PrivateKey or Wallet instance holding DAO key.
    Returns:
        Transaction hash (str).
    """
    from cosmpy.aerial.tx import Transaction

    if proposal_msg is None:
        raise ValueError("`proposal_msg` is required")

    # ✍️ Build and sign the Tx
    tx = Transaction()
    tx.add_message(proposal_msg)
    tx.with_gas(gas)
    tx.with_fee(amount=fee_amount, denom=fee_denom)

    try:
        tx_signed = tx.sign(wallet)
        tx_response = client.broadcast_block(tx_signed)
        if tx_response.is_tx_error():
            raise RuntimeError(f"Tx failed (code={tx_response.code}): {tx_response.raw_log}")
        return tx_response.tx_hash
    except Exception as err:
        raise RuntimeError(f"Failed to broadcast transaction: {err}") from err



# step:2 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
import os
import json
import base64
import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()

NEUTRON_LCD = os.getenv("NEUTRON_LCD", "https://lcd-kralum.neutron.org")
VESTING_CONTRACT = "neutron1dz57hjkdytdshl2uyde0nqvkwdww0ckx7qfe05raz4df6m3khfyqfnj0nr"

@router.get("/claimable/{address}")
async def query_vesting_contract(address: str):
    """Return the claimable rewards for a given address."""
    try:
        query_msg = {"claimable_rewards": {"address": address}}
        query_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
        url = f"{NEUTRON_LCD}/cosmwasm/wasm/v1/contract/{VESTING_CONTRACT}/smart/{query_b64}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        data = resp.json()
        # Expected format: {"data": {"amount": "123456"}}
        amount = int(data.get("data", {}).get("amount", 0))
        return {"claimable": amount}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:3 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
from fastapi import HTTPException

async def validate_claimable_amount(amount: int):
    """Raise an HTTP 400 if amount == 0."""
    if int(amount) == 0:
        raise HTTPException(status_code=400, detail="No claimable rewards for this address.")
    return {"ok": True}


# step:4 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
def construct_execute_msg():
    """Return the execute message required to start vesting."""
    execute_msg = {"start_standard_vesting": {}}
    return execute_msg


# step:5 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
import os
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import MsgExecuteContract
from cosmpy.aerial.wallet import PrivateKey
from fastapi import HTTPException

NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url=os.getenv("NEUTRON_RPC", "https://rpc-kralum.neutron.org"),
    fee_minimum_gas_price=0.025,
    fee_denom="untrn",
)

async def sign_and_broadcast_tx(sender_addr: str, execute_msg: dict):
    """Sign the MsgExecuteContract and broadcast it to the Neutron network."""
    mnemonic = os.getenv("MNEMONIC")
    if not mnemonic:
        raise HTTPException(status_code=500, detail="Backend signing key is not configured.")

    try:
        # Create wallet & client
        pk = PrivateKey.from_mnemonic(mnemonic)
        if sender_addr != pk.address():
            raise HTTPException(status_code=400, detail="Configured key does not match sender address.")

        client = LedgerClient(NETWORK, wallet=pk)

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


# step:6 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
import base64
import json
import httpx
from fastapi import HTTPException

async def query_vesting_schedule(address: str):
    """Return the latest vesting schedule for the provided address."""
    query = {"vesting_schedule": {"address": address}}
    query_b64 = base64.b64encode(json.dumps(query).encode()).decode()
    url = f"{NEUTRON_LCD}/cosmwasm/wasm/v1/contract/{VESTING_CONTRACT}/smart/{query_b64}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        return resp.json().get("data", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# step:4 file: Set the user's own address as the admin of a smart contract
import os
from cosmpy.crypto.keypair import PrivateKey
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.client import LedgerClient


ADMIN_PRIVKEY_ENV = 'CURRENT_ADMIN_PRIVKEY'  # hex-encoded secp256k1 private key


def sign_and_broadcast_tx(tx: Transaction, client: LedgerClient) -> str:
    """Sign the given transaction with the admin key and broadcast it.

    Returns the resulting transaction hash on success. Raises RuntimeError on failure.
    """
    # Retrieve private key from environment (or secret-manager of your choice)
    priv_hex = os.getenv(ADMIN_PRIVKEY_ENV)
    if not priv_hex:
        raise RuntimeError(f"Environment variable {ADMIN_PRIVKEY_ENV} not set.")

    try:
        # Create PrivateKey object
        priv = PrivateKey(bytes.fromhex(priv_hex))

        # Sign the transaction
        tx.sign(priv)

        # Broadcast and wait for inclusion
        tx_response = client.broadcast_tx(tx)

        if tx_response.is_tx_error():
            raise RuntimeError(f"Tx broadcast failed: {tx_response.raw_log}")

        return tx_response.tx_hash

    except Exception as err:
        raise RuntimeError(f"Failed to sign and/or broadcast: {err}")


# step:1 file: query_storage_slot_0_for_a_given_ethereum_address_via_json-rpc
def construct_json_rpc_request(contract_address: str, slot: str = "0x0", request_id: int = 1) -> dict:
    """
    Construct a JSON-RPC request body for eth_getStorageAt.

    Args:
        contract_address (str): Target contract address (0x-prefixed).
        slot (str): Storage slot to read, default is "0x0" (slot 0).
        request_id (int): Arbitrary JSON-RPC id.

    Returns:
        dict: A JSON-serialisable request object.
    """
    if not contract_address.startswith("0x"):
        raise ValueError("contract_address must start with 0x")

    return {
        "jsonrpc": "2.0",
        "method": "eth_getStorageAt",
        "params": [contract_address, slot, "latest"],
        "id": request_id
    }



# step:2 file: query_storage_slot_0_for_a_given_ethereum_address_via_json-rpc
import requests


def send_http_post(request_body: dict, endpoint: str = "http://localhost:8545", timeout: int = 10) -> dict:
    """
    Send a JSON-RPC POST request to an Ethereum node.

    Args:
        request_body (dict): JSON-RPC request created in Step 1.
        endpoint (str): URL of the JSON-RPC server.
        timeout (int): Network timeout (seconds).

    Returns:
        dict: Parsed JSON response from the node.

    Raises:
        RuntimeError: For network or HTTP-status errors.
        ValueError:  For malformed JSON responses.
    """
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(endpoint, headers=headers, json=request_body, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to POST to {endpoint}: {e}") from e

    try:
        return resp.json()
    except ValueError as e:
        raise ValueError(f"Response is not valid JSON: {e}") from e



# step:3 file: query_storage_slot_0_for_a_given_ethereum_address_via_json-rpc
def parse_json_rpc_response(response_json: dict) -> str:
    """
    Parse eth_getStorageAt response and return the storage value.

    Args:
        response_json (dict): JSON returned by Step 2.

    Returns:
        str: 32-byte hex string (e.g., "0x000...42").

    Raises:
        RuntimeError: If the node returned an error.
        KeyError:     If the 'result' field is missing.
        ValueError:   If the result format looks wrong.
    """
    if "error" in response_json:
        raise RuntimeError(f"JSON-RPC error: {response_json['error']}")

    if "result" not in response_json:
        raise KeyError("'result' field missing in response")

    result = response_json["result"]
    if not (isinstance(result, str) and result.startswith("0x")):
        raise ValueError("Unexpected 'result' format")

    return result



# step:1 file: generate_a_solidity_coverage_report_using_hardhat_coverage
import subprocess


def install_hardhat_coverage():
    '''
    Installs the hardhat-coverage plugin as a development dependency.
    '''
    try:
        result = subprocess.run(
            ['npm', 'install', '--save-dev', 'hardhat-coverage'],
            check=True,
            capture_output=True,
            text=True
        )
        return {'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f'Failed to install hardhat-coverage: {err.stderr}') from err


# step:2 file: generate_a_solidity_coverage_report_using_hardhat_coverage
import pathlib


def configure_hardhat():
    '''
    Adds require('hardhat-coverage'); to hardhat.config.js if not already present.
    '''
    config_path = pathlib.Path('hardhat.config.js')
    if not config_path.exists():
        raise FileNotFoundError('hardhat.config.js not found.')

    content = config_path.read_text()
    require_line = "require('hardhat-coverage');"

    if require_line not in content:
        config_path.write_text(f'{require_line}\n{content}')
        return {'status': 'updated', 'path': str(config_path)}

    return {'status': 'already_configured', 'path': str(config_path)}


# step:3 file: generate_a_solidity_coverage_report_using_hardhat_coverage
import subprocess


def run_hardhat_coverage():
    '''
    Executes the Hardhat coverage task.
    '''
    try:
        result = subprocess.run(
            ['npx', 'hardhat', 'coverage'],
            check=True,
            capture_output=True,
            text=True
        )
        return {'stdout': result.stdout}
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f'Coverage run failed: {err.stderr}') from err


# step:4 file: generate_a_solidity_coverage_report_using_hardhat_coverage
import webbrowser
import pathlib


def open_coverage_report():
    '''
    Opens coverage/lcov-report/index.html in the default browser if it exists.
    '''
    report_path = pathlib.Path('coverage/lcov-report/index.html').resolve()
    if not report_path.exists():
        raise FileNotFoundError('Coverage report not found; please run coverage first.')

    webbrowser.open(report_path.as_uri())
    return {'status': 'opened', 'report': str(report_path)}


# step:1 file: Create a vesting schedule that unlocks tokens every 216 000 blocks
import json
from neutron_proto.cosmwasm.wasm.v1.tx_pb2 import MsgExecuteContract


def build_release_tokens_msg(sender: str, vesting_contract: str) -> MsgExecuteContract:
    """Constructs a MsgExecuteContract that triggers the `release_tokens` method of the vesting contract."""
    try:
        execute_msg = {"release_tokens": {}}
        msg = MsgExecuteContract(
            sender=sender,
            contract=vesting_contract,
            msg=json.dumps(execute_msg).encode(),
            funds=[]  # No funds sent with the call
        )
        return msg
    except Exception as e:
        raise ValueError(f"Failed to build MsgExecuteContract: {e}")


# step:2 file: Create a vesting schedule that unlocks tokens every 216 000 blocks
from neutron_proto.neutron.cron.v1.tx_pb2 import MsgAddSchedule


def build_add_schedule_msg(authority: str, release_msg, period: int = 216000, name: str = "token_unlock") -> MsgAddSchedule:
    """Constructs a MsgAddSchedule message for the Cron module."""
    try:
        schedule_msg = MsgAddSchedule(
            authority=authority,
            name=name,
            period=period,
            msgs=[release_msg]
        )
        return schedule_msg
    except Exception as e:
        raise ValueError(f"Failed to build MsgAddSchedule: {e}")


# step:3 file: Create a vesting schedule that unlocks tokens every 216 000 blocks
import json
from google.protobuf.json_format import MessageToDict


def wrap_into_gov_proposal(schedule_msg, title: str = "Add token_unlock schedule", summary: str = "Adds periodic token unlock via Cron", deposit: str = "10000000untrn") -> str:
    """Serialises the schedule message into a governance proposal JSON string."""
    try:
        schedule_dict = MessageToDict(schedule_msg, preserving_proto_field_name=True)
        schedule_dict["@type"] = "/neutron.cron.MsgAddSchedule"

        proposal = {
            "messages": [schedule_dict],
            "metadata": "",
            "deposit": deposit,
            "title": title,
            "summary": summary
        }
        return json.dumps(proposal, indent=2)
    except Exception as e:
        raise ValueError(f"Failed to wrap proposal JSON: {e}")


# step:4 file: Create a vesting schedule that unlocks tokens every 216 000 blocks
import json
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmos.gov.v1.tx_pb2 import MsgSubmitProposal


def submit_gov_proposal(rpc_endpoint: str, wallet: LocalWallet, proposal_json_path: str) -> str:
    """Reads the proposal file, builds MsgSubmitProposal, signs it with `wallet` and broadcasts to `rpc_endpoint`. Returns the tx hash."""
    try:
        with open(proposal_json_path, "r") as fp:
            proposal_dict = json.load(fp)

        # Build MsgSubmitProposal message
        msg = MsgSubmitProposal(
            proposers=[str(wallet.address())],
            messages=[json.dumps(m).encode() for m in proposal_dict["messages"]],
            initial_deposit=[{"denom": proposal_dict["deposit"][-5:], "amount": proposal_dict["deposit"][:-5]}],
            title=proposal_dict["title"],
            summary=proposal_dict["summary"],
            metadata=proposal_dict.get("metadata", "")
        )

        cfg = NetworkConfig.fetch_network_config(rpc_endpoint)
        client = LedgerClient(cfg)

        tx = Transaction()
        tx.add_message(msg)
        tx.with_gas_auto_estimate(client, wallet.address())
        tx.sign(wallet)
        tx_response = client.broadcast_tx(tx)
        if tx_response.is_err():
            raise RuntimeError(f"Tx failed: {tx_response.log}")
        return tx_response.tx_hash
    except Exception as e:
        raise RuntimeError(f"Failed to submit governance proposal: {e}")


# step:5 file: Create a vesting schedule that unlocks tokens every 216 000 blocks
import asyncio, time
from cosmpy.aerial.client import LedgerClient
from cosmpy.protos.cosmos.gov.v1.gov_pb2 import Proposal


async def wait_for_voting_result(client: LedgerClient, proposal_id: int, poll_interval: int = 15, timeout: int = 86400) -> None:
    """Waits until the given proposal_id status becomes PASSED or raises if rejected/timeout."""
    start_time = time.time()
    while True:
        try:
            gov_info = client.query_governance_proposal(proposal_id)
            status_name = Proposal.Status.Name(gov_info.status)
            print(f"Proposal {proposal_id} status: {status_name}")
            if status_name == "PROPOSAL_STATUS_PASSED":
                print("✅ Proposal passed!")
                return
            if status_name in ("PROPOSAL_STATUS_REJECTED", "PROPOSAL_STATUS_FAILED"):
                raise RuntimeError(f"Proposal {proposal_id} did not pass (status={status_name})")
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for proposal to pass.")
            await asyncio.sleep(poll_interval)
        except Exception as e:
            print(f"Error fetching proposal status: {e}. Retrying…")
            await asyncio.sleep(poll_interval)


# step:6 file: Create a vesting schedule that unlocks tokens every 216 000 blocks
import grpc
from neutron_proto.neutron.cron.v1.query_pb2 import QueryScheduleRequest
from neutron_proto.neutron.cron.v1.query_pb2_grpc import QueryStub


def query_cron_schedule(grpc_endpoint: str, name: str = "token_unlock", expected_period: int = 216000) -> dict:
    """Fetches a schedule by name and asserts its `period`."""
    try:
        with grpc.insecure_channel(grpc_endpoint) as channel:
            stub = QueryStub(channel)
            response = stub.Schedule(QueryScheduleRequest(name=name))
            schedule = response.schedule
            if schedule.period != expected_period:
                raise ValueError(f"Schedule found but period is {schedule.period}, expected {expected_period}")
            return {
                "name": schedule.name,
                "period": schedule.period,
                "msgs": len(schedule.msgs)
            }
    except Exception as e:
        raise RuntimeError(f"Failed to query schedule: {e}")


# step:1 file: compile_the_current_cosmwasm_smart_contract_using_rust-optimizer
################  backend/compile_contract.py  ################
"""Compile a CosmWasm contract from a FastAPI endpoint."""
from pathlib import Path
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DOCKER_IMAGE = "cosmwasm/rust-optimizer:0.12.11"

app = FastAPI()

class CompileRequest(BaseModel):
    project_root: str  # Absolute or relative path to the contract root


def _run_optimizer(project_root: Path) -> str:
    """Internal helper that executes the Docker optimizer and returns stdout."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{project_root}:/code",
        "--mount", "type=volume,source=registry_cache,target=/usr/local/cargo/registry",
        DOCKER_IMAGE,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("Docker is not installed or not found in PATH.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Rust optimizer failed: {err.stderr}\n{err.stdout}") from err

    return result.stdout


@app.post("/api/compile-contract")
async def compile_contract(req: CompileRequest):
    """POST /api/compile-contract with `{ "project_root": "/abs/path" }` to build the contract."""
    project_root = Path(req.project_root).expanduser().resolve()

    if not project_root.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {project_root}")

    try:
        stdout = _run_optimizer(project_root)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": "Compilation successful. Artifacts are in ./artifacts/.",
        "stdout": stdout,
        "artifacts_dir": str(project_root / "artifacts")
    }

# Optional: enable CLI usage for local development
if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="Compile CosmWasm contract via docker rust-optimizer.")
    parser.add_argument("project_root", help="Path to contract root")
    args = parser.parse_args()

    try:
        output = _run_optimizer(Path(args.project_root))
        print(json.dumps({"stdout": output}, indent=2))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# step:2 file: compile_the_current_cosmwasm_smart_contract_using_rust-optimizer
################  backend/verify_artifacts.py  ################
"""Verify cosmwasm build artifacts exist."""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class VerifyRequest(BaseModel):
    project_root: str


def _verify_artifacts(project_root: Path) -> dict:
    artifacts_dir = project_root / "artifacts"

    if not artifacts_dir.is_dir():
        raise FileNotFoundError("artifacts directory not found — did the build run?")

    wasm_files = [f.name for f in artifacts_dir.glob("*.wasm")]
    if not wasm_files:
        raise FileNotFoundError("No .wasm files found in artifacts directory.")

    contracts_txt = artifacts_dir / "contracts.txt"
    if not contracts_txt.exists():
        raise FileNotFoundError("contracts.txt not found in artifacts directory.")

    return {
        "wasm_files": wasm_files,
        "contracts_txt": contracts_txt.name,
        "status": "verified"
    }


@router.post("/api/verify-artifacts")
async def verify_artifacts(req: VerifyRequest):
    project_root = Path(req.project_root).expanduser().resolve()

    if not project_root.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {project_root}")

    try:
        payload = _verify_artifacts(project_root)
    except FileNotFoundError as nf:
        raise HTTPException(status_code=404, detail=str(nf))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return payload

# Mount router into the FastAPI instance declared in compile_contract.py
# Example (add this to main entrypoint):
#   from backend.verify_artifacts import router as verify_router
#   app.include_router(verify_router)


# step:1 file: simulate_the_signed_transaction_to_estimate_gas
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from google.protobuf import message
from cosmos_sdk_proto.cosmos.tx.v1beta1 import tx_pb2 as cosmos_tx_pb2

app = FastAPI()

class TxBytes(BaseModel):
    tx_bytes: str  # base64-encoded, signed TxRaw string

@app.post("/api/validate_tx")
async def validate_tx(payload: TxBytes):
    """Validate that the provided base64 string decodes to a *signed* TxRaw."""
    # 1. Decode base64
    try:
        raw_bytes = base64.b64decode(payload.tx_bytes)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail="tx_bytes is not valid base64") from e

    # 2. Parse protobuf TxRaw to ensure it is well-formed
    try:
        tx_raw = cosmos_tx_pb2.TxRaw()
        tx_raw.ParseFromString(raw_bytes)
    except message.DecodeError as e:
        raise HTTPException(status_code=400, detail="Decoded bytes are not a valid TxRaw message") from e

    # 3. Confirm the transaction is signed
    if len(tx_raw.signatures) == 0:
        raise HTTPException(status_code=400, detail="TxRaw contains no signatures")

    return {"is_valid": True, "signature_count": len(tx_raw.signatures)}


# step:2 file: simulate_the_signed_transaction_to_estimate_gas
import os
import base64
import grpc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.protobuf.json_format import MessageToDict
from cosmos_sdk_proto.cosmos.tx.v1beta1 import service_pb2 as tx_service_pb2
from cosmos_sdk_proto.cosmos.tx.v1beta1 import service_pb2_grpc as tx_service_grpc

GRPC_ENDPOINT = os.getenv("GRPC_ENDPOINT", "localhost:9090")  # Point to your full-node gRPC port

app = FastAPI()

class TxBytes(BaseModel):
    tx_bytes: str

def _simulate_tx(base64_tx_bytes: str) -> dict:
    """Internal helper: hits cosmos.tx.v1beta1.Service/Simulate and returns the response as a dict."""
    tx_bytes = base64.b64decode(base64_tx_bytes)
    with grpc.insecure_channel(GRPC_ENDPOINT) as channel:
        stub = tx_service_grpc.ServiceStub(channel)
        request = tx_service_pb2.SimulateRequest(tx_bytes=tx_bytes)
        response = stub.Simulate(request)
    return MessageToDict(response, preserving_proto_field_name=True)

@app.post("/api/simulate_tx")
async def simulate_tx(payload: TxBytes):
    """HTTP wrapper that forwards signed tx bytes to the gRPC Simulate endpoint."""
    try:
        result = _simulate_tx(payload.tx_bytes)
        return result
    except grpc.RpcError as e:
        raise HTTPException(status_code=400, detail=f"Simulation failed: {e.details()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:1 file: encode_the_signed_transaction_to_protobuf_bytes
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import base64
import os
from typing import Optional

app = FastAPI()

class EncodeRequest(BaseModel):
    input_json_path: str               # Path to the already-signed JSON tx (e.g. "tx_signed.json")
    output_pb_path: Optional[str] = None  # If provided, the .pb file will be written here

@app.post("/api/encode-tx")
async def encode_tx(req: EncodeRequest):
    """Encode a signed JSON tx into protobuf bytes using the Cosmos CLI."""
    if not os.path.isfile(req.input_json_path):
        raise HTTPException(status_code=404, detail="input_json_path does not exist")

    cmd = ["cosmos", "tx", "encode", req.input_json_path]
    try:
        # Run the CLI command and capture stdout (base64 string)
        result = subprocess.run(cmd, check=True, capture_output=True)
        base64_tx = result.stdout.decode().strip()

        # Optionally persist the raw protobuf bytes to disk
        if req.output_pb_path:
            with open(req.output_pb_path, "wb") as f:
                f.write(base64.b64decode(base64_tx))

        return {
            "base64_tx": base64_tx,
            "output_pb_path": req.output_pb_path
        }
    except subprocess.CalledProcessError as err:
        # Surface any CLI errors back to the caller
        err_msg = err.stderr.decode() if err.stderr else "Unknown CLI error"
        raise HTTPException(status_code=500, detail=err_msg)


# step:2 file: encode_the_signed_transaction_to_protobuf_bytes
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

class DecodeRequest(BaseModel):
    pb_file_path: str   # Path to the protobuf file produced in step 1 (e.g. "tx_signed.pb")

@app.post("/api/decode-tx")
async def decode_tx(req: DecodeRequest):
    """Decode protobuf bytes back into JSON using the Cosmos CLI."""
    if not os.path.isfile(req.pb_file_path):
        raise HTTPException(status_code=404, detail="pb_file_path does not exist")

    cmd = ["cosmos", "tx", "decode", req.pb_file_path]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
        json_tx = result.stdout.decode()
        return { "decoded_tx_json": json_tx }
    except subprocess.CalledProcessError as err:
        err_msg = err.stderr.decode() if err.stderr else "Unknown CLI error"
        raise HTTPException(status_code=500, detail=err_msg)


# step:1 file: create_a_global_foundry.toml_configuration_file
############################################
# backend/ensure_foundryup.py               #
############################################
from fastapi import APIRouter, HTTPException
import subprocess, shutil, os
from pathlib import Path

router = APIRouter()

@router.post("/api/ensure_foundryup")
async def ensure_foundryup():
    """Ensure that the Foundry tool-chain is installed and current."""
    try:
        # 1. Check whether `foundryup` is already on PATH
        foundryup_path = shutil.which("foundryup")
        if foundryup_path is None:
            # 2. Install Foundry non-interactively (adds binaries under $HOME/.foundry/bin)
            install_cmd = "curl -L https://foundry.paradigm.xyz | bash"
            subprocess.run(install_cmd, shell=True, check=True)
            # 3. Add the bin directory to PATH for the life-time of this process
            os.environ["PATH"] += f":{Path.home()}/.foundry/bin"
        # 4. Update (or finish installing) the tool-chain
        subprocess.run("foundryup", shell=True, check=True)
        return {"status": "success", "message": "Foundry is installed and up to date."}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"Foundry installation failed: {err}")


# step:2 file: create_a_global_foundry.toml_configuration_file
############################################
# backend/create_foundry_toml.py            #
############################################
from fastapi import APIRouter, HTTPException
import subprocess
from pathlib import Path

router = APIRouter()

@router.post("/api/create_foundry_toml")
async def create_foundry_toml():
    """Generate the global Foundry TOML configuration file."""
    try:
        # Preferred path: let Forge initialise the file for us
        subprocess.run(["forge", "config", "--global"], check=True)
        return {"status": "success", "message": "Global foundry.toml created/updated by Forge."}
    except FileNotFoundError:
        # Fallback: Forge not on PATH yet – create an empty file manually
        cfg_path = Path.home() / ".foundry" / "foundry.toml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.touch(exist_ok=True)
        return {"status": "success", "message": f"Created empty config at {cfg_path}."}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"`forge config --global` failed: {err}")


# step:3 file: create_a_global_foundry.toml_configuration_file
############################################
# backend/edit_foundry_toml.py              #
############################################
from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter()

@router.post("/api/edit_foundry_toml")
async def edit_foundry_toml():
    """Inject recommended defaults into the global Foundry config."""
    try:
        cfg_path = Path.home() / ".foundry" / "foundry.toml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        if not cfg_path.exists():
            cfg_path.touch()

        # Desired key/value lines
        desired = [
            'rpc_endpoints = { cosmos_evm = "https://rpc.my-chain.com" }',
            'default_rpc_endpoint = "cosmos_evm"',
            'optimizer = true',
            'optimizer_runs = 200'
        ]

        existing_lines = cfg_path.read_text().splitlines()
        with cfg_path.open("a") as fp:
            for line in desired:
                # Avoid duplicate keys on successive calls
                key = line.split("=")[0].strip()
                if not any(key == l.split("=")[0].strip() for l in existing_lines):
                    fp.write(line + "\n")
        return {"status": "success", "message": "foundry.toml updated with standard settings."}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Unable to update foundry.toml: {err}")


# step:4 file: create_a_global_foundry.toml_configuration_file
############################################
# backend/validate_foundry_config.py        #
############################################
from fastapi import APIRouter, HTTPException
import subprocess

router = APIRouter()

@router.get("/api/validate_foundry_config")
async def validate_foundry_config():
    """Return the current global Foundry configuration as plain text."""
    try:
        output = subprocess.check_output(
            ["forge", "config", "--show", "--global"],
            text=True
        )
        return {"status": "success", "config": output}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Forge binary not found on PATH. Have you run /api/ensure_foundryup?")
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"Unable to display config: {err}")


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


# step:3 file: Query the code hash of a specific smart contract
# extract_code_id.py

from typing import Dict, Union

class CodeIdExtractionError(Exception):
    pass

def extract_code_id(contract_info: Dict) -> Union[int, str]:
    """Pull `code_id` out of the contract-info payload.

    Args:
        contract_info (Dict): Output from `query_contract_info`.

    Returns:
        int | str: The numeric (or string) code ID.
    """
    try:
        code_id = contract_info["code_id"]
        if code_id in (None, ""):
            raise KeyError
        return code_id
    except KeyError:
        raise CodeIdExtractionError("`code_id` not found in contract info payload")


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


# step:4 file: enable_usdc_gas_payments_for_my_next_transaction
### backend/tx_service.py
"""FastAPI micro-service that constructs & signs a MsgSend with `uusdc` fees."""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cosmpy.aio.client import LedgerClient
from cosmpy.crypto.keypairs import PrivateKey
from cosmpy.aio.tx import Transaction
from cosmpy.protos.cosmos.bank.v1beta1 import tx_pb2 as bank_tx

RPC_ENDPOINT = os.getenv("NEUTRON_RPC", "https://rpc.neutron.org:443")
CHAIN_ID = os.getenv("NEUTRON_CHAIN_ID", "neutron-1")

app = FastAPI(title="Neutron Tx Service")

class ConstructTxRequest(BaseModel):
    sender_privkey_hex: str      # hex-encoded secp256k1 private key
    recipient: str              # Bech32 address
    amount: int                 # in micro-denom (e.g., 1_000_000 = 1 UNTRN)
    amount_denom: str = "untrn"  # asset you are sending
    fee_amount: int             # must be >= Step-2 minGasPrice * gasLimit
    fee_denom: str = "uusdc"
    gas_limit: int = 200000

class ConstructTxResponse(BaseModel):
    signed_tx_hex: str

@app.post("/tx/construct-sign", response_model=ConstructTxResponse)
async def construct_and_sign(req: ConstructTxRequest):
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


# step:5 file: enable_usdc_gas_payments_for_my_next_transaction
### backend/tx_service.py (continued)
from cosmpy.aio.client import TxCommitError

class BroadcastRequest(BaseModel):
    signed_tx_hex: str

class BroadcastResponse(BaseModel):
    tx_hash: str
    height: Optional[int] = None

@app.post("/tx/broadcast", response_model=BroadcastResponse)
async def broadcast_signed_tx(req: BroadcastRequest):
    try:
        client = LedgerClient(RPC_ENDPOINT)
        tx_bytes = bytes.fromhex(req.signed_tx_hex)
        res = await client.broadcast_tx_sync(tx_bytes)

        if res.code != 0:
            raise TxCommitError(f"Tx failed: code={res.code} log={res.raw_log}")

        return {"tx_hash": res.txhash, "height": res.height}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:2 file: query_a_wallet’s_bank_balances_via_the_rest_api
# api/balances.py
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI()

REST_ENDPOINT = "http://localhost:1317"  # Change to your full-node REST host if different

@app.get("/api/balances")
async def query_balances(address: str):
    """Fetch bank balances for a bech32 address and return only the relevant JSON payload."""
    url = f"{REST_ENDPOINT}/cosmos/bank/v1beta1/balances/{address}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.RequestError as exc:
        # Network-level failure
        raise HTTPException(status_code=502, detail=f"Cannot reach REST endpoint: {exc}")
    except httpx.HTTPStatusError as exc:
        # Non-200 response from node
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)

    # Return a trimmed response to the frontend
    return {
        "address": address,
        "balances": data.get("balances", [])
    }


# step:2 file: inspect_full_txpool_contents_for_debugging_purposes
######################## api/txpool.py ########################
from fastapi import APIRouter, HTTPException, Depends, Request
import os, httpx, json

router = APIRouter(prefix="/api")

# The RPC endpoint is read from an environment variable so it is **never** exposed to the browser.
RPC_ENDPOINT = os.getenv("RPC_ENDPOINT", "http://localhost:8545")  # fallback for local dev

@router.post("/txpool")
async def txpool_proxy(request: Request):
    """Proxy any txpool_* JSON-RPC request to the configured RPC_ENDPOINT."""
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Basic sanity-check that the user is only allowed to call txpool_* methods
    if not payload.get("method", "").startswith("txpool_"):
        raise HTTPException(status_code=400, detail="Only txpool_* methods are allowed")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            rpc_response = await client.post(RPC_ENDPOINT, json=payload)
            rpc_response.raise_for_status()
            return rpc_response.json()
    except httpx.HTTPStatusError as he:
        # Bubble up node-side HTTP errors
        raise HTTPException(status_code=he.response.status_code, detail=str(he))
    except Exception as e:
        # Catch-all to ensure the client always receives a clean error structure
        raise HTTPException(status_code=500, detail=f"txpool_proxy internal error: {e}")


# step:3 file: inspect_full_txpool_contents_for_debugging_purposes
######################## utils/txpool_parser.py ########################
from typing import Any, Dict, List, Union

class TxpoolParsingError(Exception):
    """Raised when the JSON-RPC response cannot be parsed as expected."""


def parse_txpool_response(rpc_response: Dict[str, Any]) -> Union[List[str], Dict[str, Any]]:
    """Normalise the JSON-RPC `result` field.

    Args:
        rpc_response (dict): Raw response from an EVM JSON-RPC node.

    Returns:
        Union[List[str], Dict[str, Any]]: A simplified representation.

    Raises:
        TxpoolParsingError: If the response is malformed or contains an error.
    """
    # Check for JSON-RPC errors first
    if "error" in rpc_response:
        raise TxpoolParsingError(rpc_response["error"].get("message", "Unknown RPC error"))

    result = rpc_response.get("result")
    if result is None:
        raise TxpoolParsingError("Missing 'result' field in RPC response")

    # Heuristically decide which method was used based on the payload format
    if isinstance(result, str):
        # This should be the `txpool_inspect` response (a big string). Split by lines.
        # Each line looks like: "0xAccount:nonce   tx_hash   {tx details...}"
        return [line.strip() for line in result.splitlines() if line.strip()]
    elif isinstance(result, dict):
        # Likely `txpool_content`: keep the full structure (pending / queued)
        return result
    else:
        raise TxpoolParsingError("Unrecognised txpool response format")


# step:1 file: start_an_in-process_test_network_with_simd_testnet_start
# backend/testnet_manager.py
from fastapi import FastAPI, HTTPException
from pathlib import Path
import subprocess
import os

app = FastAPI()

# In-memory registry of running processes (shared by later steps)
_processes = {}

def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    """Helper that executes an external command and raises if it fails."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc

def ensure_testnet_files(home_dir: str | None = None,
                          auto_init: bool = False,
                          validators: int = 2) -> dict:
    """Checks (and optionally creates) test-net files for simd."""
    # Resolve the directory (default: $HOME/.simapp )
    if home_dir is None:
        home_dir = str(Path.home() / ".simapp")
    home_path = Path(home_dir)

    # Very light-weight existence test – looks for any genesis.json file.
    config_exists = any(home_path.glob("**/genesis.json"))

    # If everything is in place, just return the positive status.
    if config_exists:
        return {"exists": True, "path": home_dir}

    # Files are missing – either tell the caller or create them.
    if not auto_init:
        return {
            "exists": False,
            "requires_init": True,
            "path": home_dir,
            "hint": "POST again with auto_init=true or run `simd testnet init-files` manually."
        }

    # Initialise the directory.
    cmd = [
        "simd", "testnet", "init-files",
        "--v", str(validators),            # number of validators
        "--output-dir", home_dir
    ]
    proc = _run_cmd(cmd)
    return {
        "exists": True,
        "init_performed": True,
        "path": home_dir,
        "stdout": proc.stdout
    }

@app.post("/api/ensure-testnet-files")
async def api_ensure_testnet_files(home_dir: str | None = None,
                                   auto_init: bool = False,
                                   validators: int = 2):
    """HTTP wrapper around ensure_testnet_files."""
    try:
        return ensure_testnet_files(home_dir, auto_init, validators)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:2 file: start_an_in-process_test_network_with_simd_testnet_start
# Add this below the contents of testnet_manager.py (Step 1)
import subprocess


def start_testnet(home_dir: str) -> dict:
    """Launch `simd testnet start --home <dir>` as a managed background process."""
    if not Path(home_dir).exists():
        raise FileNotFoundError(f"The directory {home_dir} does not exist; run Step 1 first.")

    # If a process is already running, avoid spawning another one.
    existing = _processes.get(home_dir)
    if existing and existing.poll() is None:
        return {"started": False, "message": "Test-net is already running.", "pid": existing.pid}

    cmd = ["simd", "testnet", "start", "--home", home_dir]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # line-buffered so we can stream logs
    )
    _processes[home_dir] = proc
    return {"started": True, "pid": proc.pid, "home_dir": home_dir}


@app.post("/api/start-testnet")
async def api_start_testnet(home_dir: str):
    """HTTP endpoint that starts the test-net process."""
    try:
        return start_testnet(home_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:3 file: start_an_in-process_test_network_with_simd_testnet_start
# Append to testnet_manager.py (after Steps 1 & 2)
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/api/testnet-logs")
async def api_testnet_logs(home_dir: str):
    """Streams stdout from the running `simd testnet start` process."""
    proc = _processes.get(home_dir)
    if not proc or proc.poll() is not None:
        raise HTTPException(status_code=400, detail="No running test-net for the given home_dir")

    async def log_generator():
        """Asynchronously yield lines from the process output."""
        # Use the underlying file-descriptor in non-blocking mode.
        while True:
            line = proc.stdout.readline()
            if line:
                yield line
            elif proc.poll() is not None:
                break  # process ended
            else:
                await asyncio.sleep(0.1)  # avoid busy-loop

    return StreamingResponse(log_generator(), media_type="text/plain")


# step:4 file: instantly_claim_50%_of_my_ntrn_staking_rewards
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os, base64

from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.tx import Transaction, MsgWithdrawDelegatorReward

app = FastAPI()

# ---------------------------
# Pydantic request / response
# ---------------------------
class ValidatorPortion(BaseModel):
    validator_address: str
    amount: str  # kept for reference; MsgWithdrawDelegatorReward ignores it
    denom: str

class BuildTxRequest(BaseModel):
    delegator_address: str
    rewards: List[ValidatorPortion]

class SignDocResponse(BaseModel):
    body_bytes: str
    auth_info_bytes: str
    account_number: int
    chain_id: str

# -------------------------------------------
# Helper to build and serialise the Sign-Doc
# -------------------------------------------
@app.post('/api/build_withdraw_tx', response_model=SignDocResponse)
async def build_withdraw_tx(req: BuildTxRequest):
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


# step:6 file: instantly_claim_50%_of_my_ntrn_staking_rewards
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64, os

from cosmpy.aerial.client import LedgerClient
from cosmpy.protos.cosmos.tx.v1beta1.tx_pb2 import TxRaw

app = FastAPI()

class BroadcastRequest(BaseModel):
    body_bytes: str
    auth_info_bytes: str
    signature: str

@app.post('/api/broadcast_tx')
def check_foundryup_installation():
    """Verify that `foundryup` is on the PATH and return its version."""
    try:
        result = subprocess.run([
            "foundryup",
            "--version"
        ], check=True, capture_output=True, text=True)
        version = result.stdout.strip()
        return {"foundryup_version": version}
    except FileNotFoundError:
        raise RuntimeError("`foundryup` binary not found. Please install Foundry or add it to your PATH.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Error while checking `foundryup --version`: {err.stderr.strip()}")


# step:2 file: update_foundry_to_the_latest_nightly_build
import subprocess

def run_foundryup_nightly():
    """Install the latest nightly version of Foundry toolchain."""
    try:
        result = subprocess.run(
            ["foundryup", "nightly"],
            check=True,
            capture_output=True,
            text=True
        )
        return {"output": result.stdout.strip()}
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"`foundryup nightly` failed: {err.stderr.strip()}")


# step:3 file: update_foundry_to_the_latest_nightly_build
import subprocess

def verify_foundry_version():
    """Confirm that `forge` is on a nightly build."""
    try:
        result = subprocess.run([
            "forge",
            "--version"
        ], check=True, capture_output=True, text=True)
        version = result.stdout.strip()
        if "nightly" not in version.lower():
            raise RuntimeError(f"Forge version does not appear to be nightly: {version}")
        return {"forge_version": version}
    except FileNotFoundError:
        raise RuntimeError("`forge` binary not found. Ensure Foundry installation succeeded.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Error while checking `forge --version`: {err.stderr.strip()}")


# step:1 file: use_ethers.js_to_send_a_transaction_on_cosmos_evm
def check_dependencies():
    """Make sure the backend has the libraries we need."""
    try:
        import web3  # noqa: F401
        import dotenv  # noqa: F401
    except ImportError as err:
        raise ImportError(
            f"Missing dependency: {err.name}. Install with `pip install web3 python-dotenv`."
        )


# step:2 file: use_ethers.js_to_send_a_transaction_on_cosmos_evm
from web3 import Web3


def create_json_rpc_provider(rpc_url: str) -> Web3:
    """Return a connected `Web3` instance or raise if the RPC is unreachable."""
    provider = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    if not provider.isConnected():
        raise ConnectionError(f"Unable to connect to RPC endpoint at {rpc_url}")
    return provider


# step:3 file: use_ethers.js_to_send_a_transaction_on_cosmos_evm
import os
from eth_account import Account
from web3 import Web3


def initialize_wallet(provider: Web3, private_key: str | None = None):
    """Return an `Account` object that will sign transactions."""
    pk = private_key or os.getenv("PRIVATE_KEY")
    if not pk:
        raise ValueError("Private key not provided. Pass it explicitly or set the PRIVATE_KEY env var.")
    account = Account.from_key(pk)
    return account


# step:4 file: use_ethers.js_to_send_a_transaction_on_cosmos_evm
from web3 import Web3


def build_transaction(account, to_address: str, value_ether: float, provider: Web3, gas_limit: int | None = None):
    """Prepare an unsigned transaction."""
    nonce = provider.eth.get_transaction_count(account.address)
    chain_id = provider.eth.chain_id
    value_wei = Web3.toWei(value_ether, "ether")

    tx = {
        "nonce": nonce,
        "to": Web3.to_checksum_address(to_address),
        "value": value_wei,
        "chainId": chain_id,
    }
    if gas_limit:
        tx["gas"] = gas_limit  # will be replaced during estimation if omitted
    return tx


# step:5 file: use_ethers.js_to_send_a_transaction_on_cosmos_evm
from web3 import Web3


def estimate_gas_with_web3(tx: dict, provider: Web3) -> dict:
    """Populate `gas`, `maxFeePerGas` & `maxPriorityFeePerGas` OR `gasPrice`."""
    gas_estimate = provider.eth.estimate_gas(tx)
    tx["gas"] = gas_estimate

    try:
        # Prefer EIP-1559 style if supported
        base_fee = provider.eth.fee_history(1, "latest")["baseFeePerGas"][-1]
        max_priority = Web3.toWei(2, "gwei")  # configurable tip
        tx["maxFeePerGas"] = base_fee + max_priority
        tx["maxPriorityFeePerGas"] = max_priority
    except Exception:
        # Fallback to legacy `gasPrice`
        tx["gasPrice"] = provider.eth.gas_price

    return tx


# step:6 file: use_ethers.js_to_send_a_transaction_on_cosmos_evm
def sign_and_send_transaction(account, tx: dict, provider: Web3) -> str:
    """Sign the transaction and return its hash."""
    signed_tx = account.sign_transaction(tx)
    tx_hash = provider.eth.send_raw_transaction(signed_tx.rawTransaction)
    return tx_hash.hex()


# step:7 file: use_ethers.js_to_send_a_transaction_on_cosmos_evm
import time


def await_transaction_receipt(provider: Web3, tx_hash: str, confirmations: int = 1, poll_interval: int = 5, timeout: int = 300):
    """Block until the txn is mined + `confirmations` blocks and return its receipt."""
    receipt = provider.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)

    if confirmations > 1:
        target_block = receipt.blockNumber + confirmations - 1
        while provider.eth.block_number < target_block:
            time.sleep(poll_interval)
    return dict(receipt)


# step:1 file: Set cron execution stage to BEGIN_BLOCKER for schedule health_check
import requests


def query_cron_schedule(rest_endpoint: str, schedule_name: str) -> dict:
    """Fetch an existing cron schedule from Neutron's REST API."""
    try:
        url = f"{rest_endpoint}/neutron/cron/schedules/{schedule_name}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # The schedule is usually wrapped under the `schedule` key, but we fall back just in case.
        return data.get("schedule", data)
    except requests.RequestException as err:
        raise RuntimeError(f"Failed to query schedule: {err}") from err


# Manual test
if __name__ == "__main__":
    schedule = query_cron_schedule("https://rest.neutron.org", "health_check")
    print(schedule)


# step:2 file: Set cron execution stage to BEGIN_BLOCKER for schedule health_check
def construct_msg_remove_schedule(schedule_name: str, authority: str) -> dict:
    """Return a MsgRemoveSchedule ready for inclusion in a proposal."""
    return {
        "@type": "/neutron.cron.MsgRemoveSchedule",
        "authority": authority,
        "name": schedule_name,
    }


# step:3 file: Set cron execution stage to BEGIN_BLOCKER for schedule health_check
def construct_msg_add_schedule(schedule_name: str, period: int, msgs: list, authority: str) -> dict:
    """Return a MsgAddSchedule that runs at BEGIN_BLOCKER."""
    return {
        "@type": "/neutron.cron.MsgAddSchedule",
        "authority": authority,
        "name": schedule_name,
        "period": str(period),              # protobuf JSON expects strings for integers
        "execution_stages": ["BEGIN_BLOCKER"],
        "msgs": msgs,
    }


# step:4 file: Set cron execution stage to BEGIN_BLOCKER for schedule health_check
import json


def create_json_proposal_file(msgs: list, title: str, description: str, deposit: str, outfile: str = "proposal.json") -> str:
    """Writes a Neutron governance proposal JSON to disk."""
    proposal = {
        "title": title,
        "description": description,
        "deposit": deposit,           # e.g. "1000000untrn"
        "messages": msgs,
    }
    with open(outfile, "w", encoding="utf-8") as fp:
        json.dump(proposal, fp, indent=2)
    return outfile


# step:6 file: Set cron execution stage to BEGIN_BLOCKER for schedule health_check
import time
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import PrivateKey


def vote_and_wait_for_passage(rpc_endpoint: str, proposal_id: int, voter_priv_hex: str, chain_id: str, poll: int = 15):
    """Casts a YES vote, then waits until the proposal status is PASSED (or fails)."""
    key = PrivateKey.from_hex(voter_priv_hex)
    cfg = NetworkConfig(
        chain_id=chain_id,
        url=rpc_endpoint,
        fee_denomination="untrn",
        fee_minimum_gas_price=0.025,
    )
    client = LedgerClient(cfg)

    # VoteOptionYes = 1
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


# step:7 file: Set cron execution stage to BEGIN_BLOCKER for schedule health_check
def confirm_execution_stage(rest_endpoint: str, schedule_name: str) -> bool:
    """Returns True if the cron job now runs at BEGIN_BLOCKER."""
    schedule = query_cron_schedule(rest_endpoint, schedule_name)
    return schedule.get("execution_stage") == "BEGIN_BLOCKER"


# step:4 file: submit_a_governance_text_proposal
# backend/routes/proposal.py
import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.protos.cosmos.gov.v1beta1.tx_pb2 import MsgSubmitProposal
from cosmpy.protos.cosmos.gov.v1beta1.gov_pb2 import TextProposal
from cosmpy.protos.cosmos.base.v1beta1.coin_pb2 import Coin
from google.protobuf.any_pb2 import Any as Any_pb2

router = APIRouter(prefix="/api/proposal")
logger = logging.getLogger("proposal")

# ---------- Pydantic Schema ---------- #
class SubmitProposalBody(BaseModel):
    proposer_address: str = Field(..., description="Bech32 address of the proposer")
    title: str
    description: str
    summary: str
    deposit_amount: str = Field(..., regex=r"^\d+$", description="Micro-denom amount as string")
    deposit_denom: str

# ---------- Helpers ---------- #

def _get_network() -> NetworkConfig:
    return NetworkConfig(
        chain_id=os.getenv("CHAIN_ID", "cosmoshub-4"),
        url=os.getenv("GRPC_ENDPOINT", "grpc+https://cosmoshub.grpc.polkachu.com:443"),
    )

def _get_wallet() -> LocalWallet:
    mnemonic = os.getenv("PROPOSER_MNEMONIC")
    if not mnemonic:
        raise RuntimeError("PROPOSER_MNEMONIC env var is missing")
    return LocalWallet.from_mnemonic(mnemonic)

# ---------- Route ---------- #
@router.post("/submit")
async def submit_text_proposal(body: SubmitProposalBody):
    try:
        # 1. Init network + wallet
        network = _get_network()
        client = LedgerClient(network)
        wallet = _get_wallet()

        if wallet.address() != body.proposer_address:
            raise HTTPException(status_code=400, detail="Server wallet address ≠ proposer address.")

        # 2. Build TextProposal content
        text_content = TextProposal(title=body.title, description=body.description)
        any_content = Any_pb2()
        any_content.Pack(text_content)
        any_content.type_url = "/cosmos.gov.v1beta1.TextProposal"

        # 3. Build MsgSubmitProposal
        msg = MsgSubmitProposal(
            content=any_content,
            proposer=wallet.address(),
            initial_deposit=[Coin(amount=body.deposit_amount, denom=body.deposit_denom)],
        )

        # 4. Create & sign tx
        tx = Transaction()
        tx.add_message(msg)
        tx.with_gas(200000)  # heuristic; adjust as needed
        tx.with_fee(3000)    # micro-denom fee
        tx.with_memo(body.summary)

        signed_tx = wallet.sign_transaction(tx)

        # 5. Broadcast and wait for inclusion
        result = client.broadcast_tx_block(signed_tx)
        if result.code != 0:
            logger.error("Broadcast failed %s", result.raw_log)
            raise HTTPException(status_code=500, detail=f"Broadcast error: {result.raw_log}")

        # Extract proposal ID from logs (gov module emits it)
        proposal_id = None
        for event in result.events:
            if event["type"] == "submit_proposal":
                for attr in event["attributes"]:
                    if attr["key"] == "proposal_id":
                        proposal_id = int(attr["value"])
                        break
        if proposal_id is None:
            raise HTTPException(status_code=500, detail="Unable to parse proposal_id from tx logs")

        return {
            "tx_hash": result.tx_hash,
            "proposal_id": proposal_id
        }
    except HTTPException:
        raise
    except Exception as err:
        logger.exception("submit_text_proposal failed")
        raise HTTPException(status_code=500, detail=str(err))


# step:5 file: submit_a_governance_text_proposal
# backend/routes/proposal_status.py
import os
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/proposal")
REST_ENDPOINT = os.getenv("REST_ENDPOINT", "https://api.cosmos.network")

@router.get("/{proposal_id}")
async def get_proposal_status(proposal_id: int):
    try:
        url = f"{REST_ENDPOINT}/cosmos/gov/v1beta1/proposals/{proposal_id}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        json_data = r.json()
        status = json_data["proposal"]["status"]
        return {
            "proposal_id": proposal_id,
            "status": status,
            "raw": json_data
        }
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:1 file: Show last execution height for schedule daily_rewards
import subprocess
import json
from typing import Dict


def query_cron_schedule(schedule_name: str, node: str = "https://rpc.neutron.org:443") -> Dict:
    """Fetch schedule metadata from the Neutron Cron module via `neutrond` CLI.

    Args:
        schedule_name (str): The unique schedule identifier (e.g., "daily_rewards").
        node (str, optional): RPC endpoint to query. Defaults to main-net RPC.

    Returns:
        Dict: Parsed JSON representing the schedule metadata.

    Raises:
        RuntimeError: If the CLI call fails.
        ValueError:  If the response cannot be decoded as JSON.
    """
    try:
        cmd = [
            "neutrond",
            "query",
            "cron",
            "schedule",
            schedule_name,
            "--output",
            "json",
            "--node",
            node,
        ]
        # Run the command and capture STDOUT/STDERR
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        # Bubble up a descriptive error when the CLI exits with non-zero status.
        raise RuntimeError(
            f"Failed to query schedule '{schedule_name}': {exc.stderr.strip()}"
        ) from exc
    except json.JSONDecodeError as exc:
        # Handle malformed JSON (e.g., non-JSON output)
        raise ValueError("Received non-JSON response from neutrond CLI") from exc


# step:2 file: Show last execution height for schedule daily_rewards
def extract_last_execution_height(schedule_data: dict) -> int:
    """Return the most recent execution height from schedule JSON.

    Supports both possible proto field names: `last_execution_height` (preferred)
    or the legacy `last_executed_height`.
    """
    for key in ("last_execution_height", "last_executed_height"):
        value = schedule_data.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                raise ValueError(f"Field '{key}' is not an integer: {value}")

    raise KeyError("Neither 'last_execution_height' nor 'last_executed_height' were found in the schedule data.")


# step:1 file: broadcast_a_pre-signed_transaction_over_grpc
import subprocess, json
from typing import List, Dict, Optional


def construct_unsigned_tx_cli(tx_args: List[str], chain_id: str, node: str = 'tcp://localhost:26657', home_path: Optional[str] = None) -> Dict:
    '''
    Generate an unsigned transaction via `<appd> tx ... --generate-only`.

    Parameters
    ----------
    tx_args : list[str]
        Portion after `appd tx`, e.g. ['bank', 'send', '{from}', '{to}', '100uatom'].
    chain_id : str
        Target chain-id.
    node : str
        RPC endpoint of the node (defaults to local node).
    home_path : str | None
        Optional `$HOME` for the CLI.

    Returns
    -------
    dict
        Parsed JSON of the unsigned transaction.
    '''
    cmd = ['appd', 'tx'] + tx_args + ['--generate-only', '--chain-id', chain_id, '--node', node, '--output', 'json']
    if home_path:
        cmd += ['--home', home_path]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(completed.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Failed to construct unsigned tx: {e.stderr}') from e


# step:2 file: broadcast_a_pre-signed_transaction_over_grpc
import subprocess, json, os
from tempfile import NamedTemporaryFile
from typing import Dict, Optional


def sign_tx_cli(unsigned_tx: Dict, signer_key: str, chain_id: str, home_path: Optional[str] = None) -> str:
    '''Sign an unsigned transaction using `<appd> tx sign` and return base64 `tx_bytes`.'''
    with NamedTemporaryFile('w', delete=False, suffix='.json') as tmp:
        json.dump(unsigned_tx, tmp)
        tmp.flush()
        tmp_name = tmp.name

    cmd = ['appd', 'tx', 'sign', tmp_name, '--from', signer_key, '--chain-id', chain_id, '--output', 'json']
    if home_path:
        cmd += ['--home', home_path]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        signed_tx = json.loads(completed.stdout)
        tx_bytes = signed_tx.get('tx_bytes') or signed_tx.get('body', {}).get('tx_bytes')
        if not tx_bytes:
            raise KeyError('tx_bytes not found in signing output')
        return tx_bytes
    finally:
        os.remove(tmp_name)


# step:3 file: broadcast_a_pre-signed_transaction_over_grpc
import requests, json, subprocess
from typing import Dict


def broadcast_tx_rest(tx_bytes_b64: str, rest_endpoint: str = 'http://localhost:1317', mode: str = 'BROADCAST_MODE_SYNC') -> Dict:
    '''Broadcast the signed transaction using the REST (gRPC-gateway) endpoint.'''    
    url = f'{rest_endpoint}/cosmos/tx/v1beta1/txs'
    payload = {'tx_bytes': tx_bytes_b64, 'mode': mode}
    resp = requests.post(url, json=payload, timeout=10)
    if not resp.ok:
        raise RuntimeError(f'REST broadcast failed: {resp.status_code} {resp.text}')
    return resp.json()


def broadcast_tx_grpcurl(tx_bytes_b64: str, grpc_host: str = 'localhost:9090', mode: str = 'BROADCAST_MODE_SYNC') -> Dict:
    '''Alternative: broadcast the signed transaction via the low-level gRPC method using `grpcurl`.'''
    data = json.dumps({'tx_bytes': tx_bytes_b64, 'mode': mode})
    cmd = ['grpcurl', '-plaintext', '-d', data, grpc_host, 'cosmos.tx.v1beta1.Service/BroadcastTx']
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


# step:1 file: delete_a_local_snapshot_stored_by_the_node
# backend/snapshots_api.py
import os
import subprocess
from typing import List
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Either export APPD_BINARY (e.g. neutrond, gaiad, etc.) or fall back to the generic name "appd".
APPD_BINARY = os.environ.get('APPD_BINARY', 'appd')


def _exec(cmd: List[str]) -> str:
    """Execute a CLI command and return stdout; raise an HTTPException on error."""
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return completed.stdout
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Binary '{cmd[0]}' not found on server.")
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=err.stderr or err.stdout)


@app.get('/api/snapshots', response_model=dict)
async def list_snapshots():
    """GET /api/snapshots — returns a JSON array of available snapshots."""
    raw_output = _exec([APPD_BINARY, 'snapshots', 'list'])
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    # Remove header row if present (e.g., "Height   Format   ...")
    if lines and lines[0].lower().startswith('height'):
        lines = lines[1:]
    return {'snapshots': lines}


# step:2 file: delete_a_local_snapshot_stored_by_the_node
# backend/snapshots_api.py (continued)
import os
import subprocess
from typing import List
from fastapi import FastAPI, HTTPException, Path

app = FastAPI()

APPD_BINARY = os.environ.get('APPD_BINARY', 'appd')


def _exec(cmd: List[str]) -> str:
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return completed.stdout
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Binary '{cmd[0]}' not found on server.")
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=err.stderr or err.stdout)


@app.delete('/api/snapshots/{snapshot_id}', response_model=dict)
async def delete_snapshot(snapshot_id: str = Path(..., description='Height or identifier of the snapshot to delete')):
    """DELETE /api/snapshots/{snapshot_id} — removes the selected snapshot and returns a confirmation message."""
    _exec([APPD_BINARY, 'snapshots', 'delete', snapshot_id])
    return {'message': f'Snapshot {snapshot_id} deleted successfully.'}


# step:1 file: write_a_cpu_profile_to_cpu.prof_for_30_seconds
def check_pprof_port(host: str = "localhost", port: int = 6060, timeout: int = 2):
    '''
    Check whether the Cosmos node was started with pprof enabled.

    Args:
        host (str): Hostname where the node is running.
        port (int): Port where pprof is expected to listen (default 6060).
        timeout (int): Timeout in seconds for the HTTP request.

    Returns:
        dict: Dictionary containing 'pprof_running' (bool) and 'message' (str).
    '''
    import http.client

    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        # '/debug/pprof/' returns 200 if pprof is enabled.
        conn.request("GET", "/debug/pprof/")
        resp = conn.getresponse()

        if resp.status == 200:
            return {
                "pprof_running": True,
                "message": f"pprof is active at http://{host}:{port}/debug/pprof/"
            }
        return {
            "pprof_running": False,
            "message": f"HTTP {resp.status} received; pprof might not be enabled."
        }
    except Exception as exc:
        return {
            "pprof_running": False,
            "message": f"Unable to connect to {host}:{port}. Error: {str(exc)}"
        }
    finally:
        conn.close()


# step:2 file: write_a_cpu_profile_to_cpu.prof_for_30_seconds
def capture_cpu_profile(seconds: int = 30, host: str = "localhost", port: int = 6060, output_file: str = "cpu.prof"):
    '''
    Capture a CPU profile from the running Cosmos node via the pprof HTTP endpoint.

    The call blocks for `seconds` while profiling is active, then writes the binary
    profile to `output_file`.
    '''
    import http.client

    path = f"/debug/pprof/profile?seconds={seconds}"
    conn = http.client.HTTPConnection(host, port, timeout=seconds + 5)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()

        if resp.status != 200:
            return {"ok": False, "error": f"pprof returned HTTP {resp.status} {resp.reason}"}

        with open(output_file, "wb") as fp:
            # Stream the response body to disk.
            while True:
                data = resp.read(8192)
                if not data:
                    break
                fp.write(data)

        return {"ok": True, "output_file": output_file, "message": f"CPU profile captured to {output_file}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        conn.close()


# step:3 file: write_a_cpu_profile_to_cpu.prof_for_30_seconds
def verify_profile_file(path: str = "cpu.prof"):
    '''
    Verify that the CPU profile file exists and is not empty.
    '''
    import os

    if not os.path.exists(path):
        return {"verified": False, "message": f"{path} does not exist."}

    size = os.path.getsize(path)
    if size <= 0:
        return {"verified": False, "message": f"{path} is empty."}

    return {"verified": True, "message": f"{path} is present and is {size} bytes."}


# step:1 file: deploy_a_smart_contract_to_a_cosmos_evm_testnet_with_hardhat
from fastapi import APIRouter, HTTPException
import subprocess, os

router = APIRouter()

@router.post('/api/hardhat/init')
async def init_hardhat_project(project_name: str):
    """Initialize a Hardhat project inside a server-side working directory."""
    try:
        project_path = os.path.abspath(project_name)
        if os.path.exists(project_path):
            raise HTTPException(status_code=400, detail='Directory already exists')
        os.makedirs(project_path, exist_ok=True)

        # 1. Generate package.json
        subprocess.run(['npm', 'init', '-y'], cwd=project_path, check=True)

        # 2. Install Hardhat locally
        subprocess.run(['npm', 'install', '--save-dev', 'hardhat'], cwd=project_path, check=True)

        # 3. Scaffold Hardhat boilerplate (–yes answers all prompts)
        subprocess.run(['npx', 'hardhat', '--yes'], cwd=project_path, check=True)

        return {"status": "success", "project_path": project_path}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'Command failed: {e}')


# step:2 file: deploy_a_smart_contract_to_a_cosmos_evm_testnet_with_hardhat
from fastapi import APIRouter, HTTPException
import subprocess, os

router = APIRouter()

@router.post('/api/hardhat/install-deps')
async def install_solidity_dependencies(project_path: str):
    """Add @nomicfoundation/hardhat-toolbox and @evmos/hardhat-evmos to the project."""
    try:
        if not os.path.isdir(project_path):
            raise HTTPException(status_code=404, detail='Project directory not found')
        packages = ['@nomicfoundation/hardhat-toolbox', '@evmos/hardhat-evmos']
        subprocess.run(['npm', 'install', '--save-dev', *packages], cwd=project_path, check=True)
        return {"status": "success", "installed": packages}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'Install failed: {e}')


# step:3 file: deploy_a_smart_contract_to_a_cosmos_evm_testnet_with_hardhat
from fastapi import APIRouter, HTTPException
import os, json

router = APIRouter()

NETWORK_TEMPLATE = """
require('@nomicfoundation/hardhat-toolbox');
require('@evmos/hardhat-evmos');

module.exports = {
  solidity: '0.8.19',
  networks: {
    {network_name}: {
      url: '{rpc_url}',
      chainId: {chain_id},
      gasPrice: '{gas_price}',
      accounts: ['{private_key}']
    }
  }
};
"""

@router.post('/api/hardhat/config-network')
async def configure_hardhat_network(project_path: str, network_name: str, rpc_url: str, chain_id: int, gas_price: str, private_key: str):
    """Write or overwrite hardhat.config.js with network values sent from the client side."""
    try:
        if not os.path.isdir(project_path):
            raise HTTPException(status_code=404, detail='Project directory not found')
        config_path = os.path.join(project_path, 'hardhat.config.js')
        with open(config_path, 'w', encoding='utf-8') as fh:
            fh.write(NETWORK_TEMPLATE.format(
                network_name=network_name,
                rpc_url=rpc_url,
                chain_id=chain_id,
                gas_price=gas_price,
                private_key=private_key
            ))
        return {"status": "success", "config_path": config_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:4 file: deploy_a_smart_contract_to_a_cosmos_evm_testnet_with_hardhat
from fastapi import APIRouter, HTTPException
import subprocess, os

router = APIRouter()

@router.post('/api/hardhat/compile')
async def hardhat_compile(project_path: str):
    """Run `npx hardhat compile` inside the project directory."""
    try:
        if not os.path.isdir(project_path):
            raise HTTPException(status_code=404, detail='Project directory not found')
        subprocess.run(['npx', 'hardhat', 'compile'], cwd=project_path, check=True)
        return {"status": "success", "message": "Compilation finished"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'Compile failed: {e}')


# step:5 file: deploy_a_smart_contract_to_a_cosmos_evm_testnet_with_hardhat
from fastapi import APIRouter, HTTPException
import os, textwrap

router = APIRouter()

DEPLOY_TEMPLATE = textwrap.dedent("""
  const hre = require('hardhat');

  async function main() {
    const ContractFactory = await hre.ethers.getContractFactory('{contract_name}');
    const contract = await ContractFactory.deploy({constructor_args});
    await contract.deployed();
    console.log('Contract deployed to:', contract.address);
  }

  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
""")

@router.post('/api/hardhat/create-deploy-script')
async def create_deployment_script(project_path: str, contract_name: str, constructor_args: str = ''):
    """Create `scripts/deploy.js` with the supplied contract name and constructor args."""
    try:
        if not os.path.isdir(project_path):
            raise HTTPException(status_code=404, detail='Project directory not found')
        scripts_dir = os.path.join(project_path, 'scripts')
        os.makedirs(scripts_dir, exist_ok=True)
        deploy_script_path = os.path.join(scripts_dir, 'deploy.js')
        with open(deploy_script_path, 'w', encoding='utf-8') as fh:
            fh.write(DEPLOY_TEMPLATE.format(contract_name=contract_name, constructor_args=constructor_args))
        return {"status": "success", "deploy_script": deploy_script_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:6 file: deploy_a_smart_contract_to_a_cosmos_evm_testnet_with_hardhat
from fastapi import APIRouter, HTTPException
import subprocess, os, re

router = APIRouter()

DEPLOY_REGEX = re.compile(r'Contract deployed to:\s*(0x[0-9a-fA-F]{40})')

@router.post('/api/hardhat/deploy')
async def hardhat_run_deployment(project_path: str, network_name: str):
    """Execute `npx hardhat run scripts/deploy.js --network <network>` and parse stdout for contract address."""
    try:
        if not os.path.isdir(project_path):
            raise HTTPException(status_code=404, detail='Project directory not found')
        completed = subprocess.run(['npx', 'hardhat', 'run', 'scripts/deploy.js', '--network', network_name], cwd=project_path, check=True, capture_output=True, text=True)
        match = DEPLOY_REGEX.search(completed.stdout)
        address = match.group(1) if match else None
        return {"status": "success", "stdout": completed.stdout, "contract_address": address}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'Deployment failed: {e}\n{e.stdout}\n{e.stderr}')


# step:7 file: deploy_a_smart_contract_to_a_cosmos_evm_testnet_with_hardhat
from fastapi import APIRouter, HTTPException
import subprocess, os

router = APIRouter()

@router.post('/api/hardhat/verify')
async def hardhat_verify_contract(project_path: str, network_name: str, contract_address: str, constructor_args: str = ''):
    """Invoke `npx hardhat verify` with the given parameters."""
    try:
        if not os.path.isdir(project_path):
            raise HTTPException(status_code=404, detail='Project directory not found')
        cmd = ['npx', 'hardhat', 'verify', '--network', network_name, contract_address]
        if constructor_args:
            cmd.extend(constructor_args.split())
        subprocess.run(cmd, cwd=project_path, check=True)
        return {"status": "success", "verified": contract_address}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'Verification failed: {e}')


# step:1 file: Query the cron schedule named "dasset-updator"
import os
import requests
from typing import Any, Dict

# Constants ------------------------------------------------------------------
LCD_ENDPOINT = os.getenv("NEUTRON_LCD", "https://lcd-kralum.neutron.org")
CRON_SCHEDULE_PATH = "/neutron/cron/schedule/{schedule_name}"  # REST path

# Exceptions -----------------------------------------------------------------
class CronQueryError(Exception):
    """Raised when the cron‐schedule query fails."""

# Core Logic -----------------------------------------------------------------

def query_cron_schedule(schedule_name: str) -> Dict[str, Any]:
    """Query cron schedule details from Neutron LCD.

    Args:
        schedule_name: Name of the cron schedule to query.

    Returns:
        Parsed JSON response as a Python dict.

    Raises:
        CronQueryError: If the request fails or returns non-200 status.
    """
    url = f"{LCD_ENDPOINT}{CRON_SCHEDULE_PATH.format(schedule_name=schedule_name)}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CronQueryError(f"Failed to query cron schedule '{schedule_name}': {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise CronQueryError("LCD returned invalid JSON response") from exc


# step:2 file: Query the cron schedule named "dasset-updator"
from typing import Any, Dict, TypedDict

class ParsedSchedule(TypedDict):
    name: str
    period: str
    msgs: Any  # List of encoded Cosmos messages
    last_execution_height: int

# ---------------------------------------------------------------------------

def parse_json_response(raw: Dict[str, Any]) -> ParsedSchedule:
    """Parse required fields from cron schedule JSON.

    Args:
        raw: The raw JSON dict returned by query_cron_schedule.

    Returns:
        A TypedDict containing the requested fields.

    Raises:
        KeyError: If any expected field is missing.
        ValueError: If a field has an unexpected type/format.
    """
    try:
        schedule = raw["schedule"]  # LCD nests data under the `schedule` key
        parsed: ParsedSchedule = {
            "name": schedule["name"],
            "period": schedule["period"],
            "msgs": schedule["msgs"],
            "last_execution_height": int(schedule["last_execution_height"]),
        }
        return parsed
    except KeyError as exc:
        raise KeyError(f"Expected key not found in response: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed field in cron schedule response: {exc}") from exc


# step:3 file: query_the_balance_of_an_evm_address_at_the_latest_block
from fastapi import FastAPI, HTTPException, Query
import httpx, re, os

app = FastAPI()

ETH_ADDRESS_REGEX = re.compile(r'^0x[0-9a-fA-F]{40}$')
DEFAULT_RPC_ENDPOINT = os.getenv('RPC_ENDPOINT', 'https://cloudflare-eth.com')

@app.get('/api/eth/balance')
async def get_eth_balance(
    address: str = Query(..., description='0x-prefixed Ethereum address'),
    rpc_endpoint: str = Query(DEFAULT_RPC_ENDPOINT, description='Optional JSON-RPC endpoint')
):
    """Returns the latest ETH balance in Wei (hex encoded)."""
    # Safety check on backend as well
    if not ETH_ADDRESS_REGEX.fullmatch(address):
        raise HTTPException(status_code=400, detail='Invalid Ethereum address')

    payload = {
        'jsonrpc': '2.0',
        'method': 'eth_getBalance',
        'params': [address, 'latest'],
        'id': 1
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(rpc_endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f'Upstream RPC error: {e}')

    if data.get('error'):
        raise HTTPException(status_code=500, detail=data['error'])

    return {
        'address': address,
        'balance_hex': data.get('result', '0x0')
    }


# step:2 file: Create a cron schedule named "daily_rewards" that distributes rewards every 7,200 blocks at END_BLOCKER
import json
from typing import List, Dict

class CronMessageError(Exception):
    """Raised when mandatory fields are missing or invalid"""


def construct_msg_add_schedule(
    authority: str,
    name: str,
    period: int,
    msgs: List[Dict],
    execution_stage: str = "EXECUTION_STAGE_END_BLOCKER",
) -> Dict:
    """Return an SDK-compatible MsgAddSchedule dictionary.

    Parameters
    ----------
    authority : str
        Address that has cron authority (DAO address).
    name : str
        Unique schedule name.
    period : int
        Number of blocks between executions.
    msgs : List[Dict]
        List of Cosmos Msgs to execute (already compiled).
    execution_stage : str
        BEGIN_BLOCKER or END_BLOCKER constant (default END_BLOCKER).
    """
    if not authority or not name or not period or not msgs:
        raise CronMessageError("All fields authority, name, period, msgs are required")

    return {
        "@type": "/neutron.cron.MsgAddSchedule",
        "authority": authority,
        "name": name,
        "period": str(period),  # JSON numbers must be strings in Cosmos messages
        "msgs": msgs,
        "execution_stage": execution_stage,
    }



# step:3 file: Create a cron schedule named "daily_rewards" that distributes rewards every 7,200 blocks at END_BLOCKER
import base64
import json
from typing import Dict

class ProposalBuildError(Exception):
    pass


def wrap_in_dao_proposal(
    dao_contract: str,
    proposer_addr: str,
    schedule_msg: Dict,
    title: str = "Add daily_rewards cron schedule",
    description: str = "Creates a cron job that distributes daily rewards at END_BLOCKER every 7200 blocks.",
) -> Dict:
    """Return a MsgExecuteContract that submits a `propose` call to a cw-dao-single contract."""
    if not dao_contract or not proposer_addr or not schedule_msg:
        raise ProposalBuildError("dao_contract, proposer_addr, and schedule_msg are mandatory")

    # cw-dao expects its internal Cosmos messages to be passed as base64-encoded binary Anys.
    # For simplicity we send the raw JSON (accepted by cosmjs), letting the chain pack it.
    proposal_msg = {
        "propose": {
            "title": title,
            "description": description,
            "msgs": [
                {
                    "stargate": {
                        "type_url": "/neutron.cron.MsgAddSchedule",
                        "value": base64.b64encode(json.dumps(schedule_msg).encode()).decode()
                    }
                }
            ],
            "latest": None
        }
    }

    return {
        "@type": "/cosmwasm.wasm.v1.MsgExecuteContract",
        "sender": proposer_addr,
        "contract": dao_contract,
        "msg": base64.b64encode(json.dumps(proposal_msg).encode()).decode(),
        "funds": []
    }



# step:5 file: Create a cron schedule named "daily_rewards" that distributes rewards every 7,200 blocks at END_BLOCKER
import asyncio
import aiohttp

API_TIMEOUT = 300  # seconds

async def monitor_proposal_status(dao_contract: str, proposal_id: int, lcd_endpoint: str) -> str:
    """Return final status ("Open", "Passed", "Executed", etc.) or raise on timeout."""
    query = {
        "proposal": {"proposal_id": proposal_id}
    }

    async def fetch_status(session):
        async with session.get(f"{lcd_endpoint}/wasm/v1/contract/{dao_contract}/smart/{base64.b64encode(json.dumps(query).encode()).decode()}") as resp:
            data = await resp.json()
            return data["data"]["status"]

    start = asyncio.get_event_loop().time()
    async with aiohttp.ClientSession() as session:
        while True:
            status = await fetch_status(session)
            if status == "Executed":
                print("Proposal executed!")
                return status
            if asyncio.get_event_loop().time() - start > API_TIMEOUT:
                raise TimeoutError("Monitoring timed out")
            await asyncio.sleep(10)



# step:6 file: Create a cron schedule named "daily_rewards" that distributes rewards every 7,200 blocks at END_BLOCKER
import requests

def query_cron_show_schedule(name: str, grpc_endpoint: str = "https://rest.ntrn.tech:443") -> dict:
    """Return the on-chain representation of the given cron schedule."""
    path = f"{grpc_endpoint}/neutron/cron/v1/schedule/{name}"
    resp = requests.get(path, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Cron query failed: {resp.text}")
    return resp.json()



# step:1 file: automatically_extract_the_contract_address_from_an_instantiate_txhash
# backend/main.py
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI(title="Juno Tx Service")

LCD_BASE = "https://lcd-juno.itastakers.com"  # Public Juno LCD endpoint

@app.get("/api/tx/{tx_hash}")
async def get_tx_json(tx_hash: str):
    """Return full transaction JSON for a given hash."""
    url = f"{LCD_BASE}/cosmos/tx/v1beta1/txs/{tx_hash}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        # LCD returned a non-200 response
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"LCD error: {e.response.text}") from e
    except httpx.RequestError as e:
        # Networking problem
        raise HTTPException(status_code=500,
                            detail=f"Network error: {str(e)}") from e


# step:2 file: automatically_extract_the_contract_address_from_an_instantiate_txhash
# backend/main.py  (add below the previous code)
from fastapi import HTTPException

@app.get("/api/contract_address/{tx_hash}")
async def get_contract_address(tx_hash: str):
    """Return the contract address created by an instantiate tx."""
    tx_json = await get_tx_json(tx_hash)  # Re-use logic from Step 1

    logs = tx_json.get("tx_response", {}).get("logs", [])
    if not logs or not isinstance(logs, list):
        raise HTTPException(status_code=400, detail="Logs not found in transaction.")

    # According to CosmWasm events, the instantiate event is in the first log index
    for event in logs[0].get("events", []):
        if event.get("type") == "instantiate":
            for attr in event.get("attributes", []):
                if attr.get("key") in ("_contract_address", "contract_address"):
                    return {"contract_address": attr.get("value")}

    raise HTTPException(status_code=404,
                        detail="Unable to locate _contract_address in instantiate event.")


# step:3 file: query_a_cosmwasm_contract’s_smart_state_with_custom_payload_{_abcde_:{}}_to_explore_confirm_schema
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx, json, base64, os

app = FastAPI()

class ContractQueryRequest(BaseModel):
    contract_address: str
    query_payload: str  # '{"abcde":{}}'
    rpc_endpoint: str | None = os.getenv('COSMOS_RPC', 'https://rpc.cosmos.directory/cosmoshub/rpc')

@app.post('/api/query_contract')
async def query_contract(req: ContractQueryRequest):
    """Queries a CosmWasm smart contract and returns the decoded JSON response."""
    # Validate JSON payload
    try:
        query_dict = json.loads(req.query_payload)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f'Invalid JSON payload: {e}')

    # Encode query to base64 as required for REST endpoint /smart/{data}
    query_b64 = base64.b64encode(json.dumps(query_dict).encode()).decode()

    url = f"{req.rpc_endpoint}/cosmwasm/wasm/v1/contract/{req.contract_address}/smart/{query_b64}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return {"data": resp.json()}
    except httpx.HTTPError as err:
        raise HTTPException(status_code=502, detail=f'Upstream RPC error: {err}')


# step:1 file: export_app_state_and_validators
from fastapi import FastAPI, HTTPException, Query
import requests

app = FastAPI()

@app.get("/api/search_docs")
def search_docs(query: str = Query(..., min_length=3)):
    """
    Searches Cosmos-related documentation. This example uses DuckDuckGo’s open API
    to keep the implementation simple while avoiding CORS issues for the frontend.
    """
    try:
        search_url = f"https://duckduckgo.com/?q={query}&format=json"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Return the first five hits to keep payloads small.
        results = data.get("RelatedTopics", [])[:5]
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# step:2 file: export_app_state_and_validators
import asyncio, os
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/api/stop_node")
async def stop_node(service_name: str = os.getenv("COSMOS_SERVICE_NAME", "cosmosd")):
    """
    Invokes `systemctl stop <service_name>` to shut down the node. The service
    name can be passed in the request body or taken from the COSMOS_SERVICE_NAME
    environment variable (default: "cosmosd").
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            f"systemctl stop {service_name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode())
        return {"service": service_name, "status": "stopped", "stdout": stdout.decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop node: {e}")


# step:3 file: export_app_state_and_validators
import asyncio, os
from pathlib import Path
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/api/export_state")
async def export_state(height: int, home: str, output_path: str = "state.json", binary: str = os.getenv("COSMOS_BINARY", "cosmosd")):
    """
    Executes `<binary> export --home <home> --height <height>` and saves the
    output to `output_path`.
    """
    try:
        cmd = f"{binary} export --home {home} --height {height}"
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode())
        Path(output_path).write_text(stdout.decode())
        return {"output_file": output_path, "height": height}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")


# step:4 file: export_app_state_and_validators
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/api/validate_state")
async def validate_state(file_path: str = "state.json"):
    """
    Confirms that `app_state` and `validators` exist at the top level of the
    exported JSON file.
    """
    try:
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"{file_path} not found")
        with open(file_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        missing = [k for k in ("app_state", "validators") if k not in data]
        return {"valid": len(missing) == 0, "missing": missing}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")


# step:2 file: execute_a_contract_and_attach_tokens_with_the_--amount_flag
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.contract import MsgExecuteContract
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.protos.cosmos.base.v1beta1.coin_pb2 import Coin

app = FastAPI()

class ExecuteContractRequest(BaseModel):
    contract_address: str
    msg: dict  # already parsed JSON from the frontend
    amount: str = "0"
    denom: str = "ujuno"

@app.post("/api/execute_contract")
async def execute_contract(req: ExecuteContractRequest):
    """
    Signs and broadcasts a CosmWasm execute transaction using cosmpy.
    A hex-encoded private key must be supplied in the JUNO_PRIVKEY_HEX environment variable.
    """
    priv_key_hex = os.getenv("JUNO_PRIVKEY_HEX")
    if not priv_key_hex:
        raise HTTPException(status_code=500, detail="Missing JUNO_PRIVKEY_HEX environment variable.")

    chain_id = os.getenv("CHAIN_ID", "juno-1")
    node_url = os.getenv("NODE_URL", "https://rpc-juno.itastakers.com:443")
    gas_price = float(os.getenv("GAS_PRICE", "0.025"))

    try:
        pk = PrivateKey(bytes.fromhex(priv_key_hex))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid private key: {e}")

    cfg = NetworkConfig(
        chain_id=chain_id,
        url=node_url,
        fee_minimum_gas_price=gas_price,
        fee_denom=req.denom,
    )
    client = LedgerClient(cfg)
    sender = pk.address()

    funds = []
    if int(req.amount) > 0:
        funds.append(Coin(amount=req.amount, denom=req.denom))

    tx = (
        Transaction()
        .with_messages(
            MsgExecuteContract(
                sender=sender,
                contract=req.contract_address,
                msg=json.dumps(req.msg).encode(),
                funds=funds,
            )
        )
        .with_gas(300000)
        .with_chain_id(chain_id)
        .with_sender(sender)
    )

    signed_tx = tx.build_and_sign(pk)

    try:
        res = client.broadcast_tx(signed_tx)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Broadcast error: {e}")

    if res.tx_response.code != 0:
        raise HTTPException(status_code=400, detail=f"Execute failed: {res.tx_response.raw_log}")

    return {"tx_hash": res.tx_response.txhash, "height": res.tx_response.height}


# step:3 file: execute_a_contract_and_attach_tokens_with_the_--amount_flag
import os
import asyncio
from cosmpy.aerial.client import LedgerClient, NetworkConfig

async def poll_tx_hash(tx_hash: str, timeout: int = 60, interval: int = 3):
    """
    Repeatedly queries the RPC endpoint for the provided tx_hash until it is included
    in a block or the timeout (in seconds) is exceeded.
    Returns a dict containing the tx result.
    """
    chain_id = os.getenv("CHAIN_ID", "juno-1")
    node_url = os.getenv("NODE_URL", "https://rpc-juno.itastakers.com:443")
    denom = os.getenv("TX_DENOM", "ujuno")

    client = LedgerClient(NetworkConfig(chain_id=chain_id, url=node_url, fee_denom=denom))

    elapsed = 0
    while elapsed < timeout:
        try:
            res = client.query_tx(tx_hash)
            if res and res.tx_response and res.tx_response.height > 0:
                return {
                    "status": "confirmed" if res.tx_response.code == 0 else "failed",
                    "height": res.tx_response.height,
                    "gas_used": res.tx_response.gas_used,
                    "raw_log": res.tx_response.raw_log,
                }
        except Exception:
            # Transaction not yet indexed
            pass

        await asyncio.sleep(interval)
        elapsed += interval

    raise TimeoutError(f"Timed out after {timeout}s waiting for tx {tx_hash}.")


# step:1 file: export_the_current_application_state_to_a_new_snapshot_file
import os


def get_node_home() -> str:
    """Return the path to the node's home directory.

    Priority order:
    1. Environment variable `SIMD_HOME`
    2. Default path `~/.simapp`
    """
    home = os.environ.get("SIMD_HOME", os.path.expanduser("~/.simapp"))
    if not os.path.isdir(home):
        raise FileNotFoundError(f"Simd home directory not found at {home}")
    return home


# step:2 file: export_the_current_application_state_to_a_new_snapshot_file
import requests


def get_latest_block_height(rpc_url: str = "http://localhost:26657") -> int:
    """Return the latest block height from the node's RPC `/status` endpoint."""
    try:
        response = requests.get(f"{rpc_url}/status", timeout=10)
        response.raise_for_status()
        data = response.json()
        height = int(data["result"]["sync_info"]["latest_block_height"])
        return height
    except Exception as err:
        raise RuntimeError(f"Unable to fetch latest block height: {err}") from err


# step:3 file: export_the_current_application_state_to_a_new_snapshot_file
import subprocess


def create_snapshot(home: str) -> None:
    """Create a snapshot via `simd snapshot create`. Raises on failure."""
    cmd = [
        "simd",
        "snapshot",
        "create",
        f"--home={home}"
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"Snapshot creation failed (exit {process.returncode}): {process.stderr.strip()}"
        )
    # Log stdout for debugging purposes
    print(process.stdout.strip())


# step:4 file: export_the_current_application_state_to_a_new_snapshot_file
import subprocess
import time


def _list_snapshots_raw(home: str) -> list[str]:
    """Internal helper to call `simd snapshot list`. Returns raw line output."""
    cmd = ["simd", "snapshot", "list", f"--home={home}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Snapshot list failed: {proc.stderr.strip()}")
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def wait_for_snapshot(home: str, expected_height: int, timeout: int = 300, poll_interval: int = 5) -> str:
    """Wait until a snapshot containing `expected_height` appears or raise TimeoutError."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for line in _list_snapshots_raw(home):
            if str(expected_height) in line:
                return line  # Return snapshot identifier / line
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Snapshot at height {expected_height} not found within {timeout} seconds."
    )


# step:5 file: export_the_current_application_state_to_a_new_snapshot_file
import subprocess


def list_snapshots(home: str) -> list[str]:
    """Return every snapshot line produced by `simd snapshot list`."""
    cmd = ["simd", "snapshot", "list", f"--home={home}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Listing snapshots failed: {proc.stderr.strip()}")
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


# step:1 file: initialize_a_new_hardhat_typescript_project_for_cosmos_evm
/* scripts/checkNodeEnv.js */
const { execSync } = require('child_process');

function getVersion(cmd) {
  try {
    return execSync(`${cmd} --version`).toString().trim();
  } catch (err) {
    throw new Error(`${cmd} is not installed or not in PATH.`);
  }
}

(function main() {
  try {
    const nodeVersion = getVersion('node');
    const npmVersion = getVersion('npm');

    // Expect something like "v18.17.1" ➜ strip leading "v" then split
    const [major] = nodeVersion.replace(/^v/, '').split('.').map(Number);
    if (isNaN(major) || major < 16) {
      throw new Error(`Node.js >=16 is required. Detected ${nodeVersion}.`);
    }

    console.log(`✅ Environment OK. Node: ${nodeVersion}, npm: ${npmVersion}`);
  } catch (error) {
    console.error(`❌ Environment check failed: ${error.message}`);
    process.exit(1);
  }
})();


# step:2 file: initialize_a_new_hardhat_typescript_project_for_cosmos_evm
/* scripts/initNpmProject.js */
const { execSync } = require('child_process');
const path = require('path');

function initProject(dir = process.cwd()) {
  try {
    console.log(`Initializing npm project in ${dir}...`);
    execSync('npm init -y', { stdio: 'inherit', cwd: dir });
    console.log('✅ package.json created.');
  } catch (error) {
    console.error('❌ Failed to initialize npm project:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  const targetDir = process.argv[2] ? path.resolve(process.argv[2]) : process.cwd();
  initProject(targetDir);
}

module.exports = initProject;


# step:3 file: initialize_a_new_hardhat_typescript_project_for_cosmos_evm
/* scripts/installHardhat.js */
const { execSync } = require('child_process');

function installHardhat() {
  try {
    console.log('Installing Hardhat as a dev dependency...');
    execSync('npm install --save-dev hardhat', { stdio: 'inherit' });
    console.log('✅ Hardhat installed.');
  } catch (error) {
    console.error('❌ Failed to install Hardhat:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  installHardhat();
}

module.exports = installHardhat;


# step:4 file: initialize_a_new_hardhat_typescript_project_for_cosmos_evm
/* scripts/createHardhatProject.js */
const { execSync } = require('child_process');

function createHardhatProject() {
  try {
    console.log('Scaffolding a new Hardhat TypeScript project...');
    // Non-interactive: --template typescript  |  --force overwrites if dir already contains files
    execSync('npm init hardhat -- --template typescript --force', { stdio: 'inherit' });
    console.log('✅ Hardhat project scaffolded.');
  } catch (error) {
    console.error('❌ Failed to scaffold Hardhat project:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  createHardhatProject();
}

module.exports = createHardhatProject;


# step:5 file: initialize_a_new_hardhat_typescript_project_for_cosmos_evm
/* scripts/installEthersAndPlugins.js */
const { execSync } = require('child_process');

function installDeps() {
  try {
    console.log('Installing Ethers.js and Hardhat plugins...');
    execSync('npm install --save-dev @nomicfoundation/hardhat-ethers ethers', { stdio: 'inherit' });
    console.log('✅ Dependencies installed.');
  } catch (error) {
    console.error('❌ Failed to install dependencies:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  installDeps();
}

module.exports = installDeps;


# step:6 file: initialize_a_new_hardhat_typescript_project_for_cosmos_evm
/* hardhat.config.ts */
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-ethers";
import * as dotenv from "dotenv";

dotenv.config();

// Store your deployer's private key in an .env file:  PRIVATE_KEY=0xabc...
const PRIVATE_KEY = process.env.PRIVATE_KEY || "";

const config: HardhatUserConfig = {
  solidity: "0.8.21",
  networks: {
    cosmosEvm: {
      url: "https://rpc.your-cosmos-evm-chain.example.com", // ⬅ replace with actual RPC URL
      chainId: 777,                                          // ⬅ replace with actual chain ID
      gasPrice: "20gwei",
      accounts: PRIVATE_KEY ? [PRIVATE_KEY] : [],
    },
  },
};

export default config;


# step:7 file: initialize_a_new_hardhat_typescript_project_for_cosmos_evm
/* scripts/compileProject.js */
const { execSync } = require('child_process');

function compile() {
  try {
    console.log('Compiling Hardhat project...');
    execSync('npx hardhat compile', { stdio: 'inherit' });
    console.log('✅ Compilation successful.');
  } catch (error) {
    console.error('❌ Compilation failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  compile();
}

module.exports = compile;


# step:1 file: broadcast_a_pre-signed_transaction_via_the_rest_endpoint
import subprocess
from typing import List


def construct_unsigned_tx_cli(appd_cmd: str,
                               module: str,
                               msg: str,
                               msg_args: List[str],
                               chain_id: str,
                               fees: str,
                               unsigned_path: str = "unsigned.json") -> str:
    """Construct an unsigned transaction using the <appd> CLI.

    Args:
        appd_cmd:  Binary name or full path to the chain CLI (e.g., `neutrond`).
        module:    Cosmos SDK module (e.g., `bank`).
        msg:       Message within that module (e.g., `send`).
        msg_args:  Additional positional CLI arguments for the message.
        chain_id:  Target chain-id.
        fees:      Fee string, e.g. `2000untrn`.
        unsigned_path: Output path for the unsigned JSON.

    Returns:
        Path to the unsigned JSON on success.
    """

    cmd = [
        appd_cmd, "tx", module, msg,
        *msg_args,
        "--generate-only",
        f"--chain-id={chain_id}",
        f"--fees={fees}",
        "-o", "json"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(unsigned_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        return unsigned_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to construct unsigned tx: {e.stderr}") from e


# step:2 file: broadcast_a_pre-signed_transaction_via_the_rest_endpoint
import subprocess


def sign_tx_cli(appd_cmd: str,
                unsigned_path: str,
                from_key: str,
                chain_id: str,
                signed_path: str = "signed.json") -> str:
    """Sign an unsigned transaction using the <appd> CLI.

    Args:
        appd_cmd:    Binary name or full path to the chain CLI.
        unsigned_path: Path to the unsigned JSON produced in Step 1.
        from_key:    Name of the key in the CLI keyring.
        chain_id:    Target chain-id.
        signed_path: Output file for the signed JSON.

    Returns:
        Path to the signed JSON on success.
    """

    cmd = [
        appd_cmd, "tx", "sign", unsigned_path,
        f"--from={from_key}",
        f"--chain-id={chain_id}",
        "-o", "json"
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(signed_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        return signed_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to sign tx: {e.stderr}") from e


# step:3 file: broadcast_a_pre-signed_transaction_via_the_rest_endpoint
import json
from pathlib import Path


def extract_tx_bytes(signed_path: str = "signed.json") -> str:
    """Extract the `tx_bytes` field from the signed transaction JSON.

    Args:
        signed_path: Path to the signed JSON produced in Step 2.

    Returns:
        Base-64 encoded transaction bytes.
    """
    try:
        data = json.loads(Path(signed_path).read_text(encoding="utf-8"))
        tx_bytes = data.get("tx_bytes")
        if not tx_bytes:
            raise ValueError("`tx_bytes` not found in signed JSON.")
        return tx_bytes
    except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Failed to extract tx_bytes: {str(e)}") from e


# step:4 file: broadcast_a_pre-signed_transaction_via_the_rest_endpoint
import requests


def broadcast_tx_rest(tx_bytes: str,
                       endpoint: str = "http://localhost:1317/cosmos/tx/v1beta1/txs",
                       mode: str = "BROADCAST_MODE_SYNC",
                       timeout: int = 30) -> dict:
    """Broadcast `tx_bytes` to a Cosmos SDK REST endpoint.

    Args:
        tx_bytes:  Base-64 encoded transaction bytes from Step 3.
        endpoint:  REST endpoint for broadcasting transactions.
        mode:      Broadcast mode (`BROADCAST_MODE_SYNC`, `BROADCAST_MODE_BLOCK`, or `BROADCAST_MODE_ASYNC`).
        timeout:   HTTP request timeout in seconds.

    Returns:
        JSON response from the node.
    """
    payload = {
        "tx_bytes": tx_bytes,
        "mode": mode
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to broadcast tx: {str(e)}") from e


# step:2 file: convert_a_wei_value_to_ether_(cast_from-wei)
''' api/convert_wei.py
FastAPI route to convert Wei → Ether.
'''
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from decimal import Decimal, getcontext
import subprocess
import shutil

app = FastAPI()

getcontext().prec = 36  # high-precision arithmetic

class WeiRequest(BaseModel):
    wei_amount: str  # keep as str to avoid Python int size limits

@app.post("/api/convert_wei")
async def convert_wei(req: WeiRequest):
    wei = req.wei_amount.strip()

    # Validate ---------------------------------------------------------------
    if not wei.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Wei amount must be a positive integer represented as a string."
        )

    # Prefer the `cast` CLI if available ------------------------------------
    cast_path = shutil.which("cast")
    if cast_path:
        try:
            result = subprocess.run(
                [cast_path, "from-wei", wei],
                capture_output=True,
                text=True,
                check=True
            )
            ether_value = result.stdout.strip()
        except subprocess.CalledProcessError as err:
            # Fall back in case cast exits with non-zero status
            ether_value = str(Decimal(wei) / Decimal(10 ** 18))
    else:
        # Fallback: pure-Python conversion ---------------------------------
        ether_value = str(Decimal(wei) / Decimal(10 ** 18))

    return JSONResponse({
        "wei": wei,
        "ether": ether_value
    })


# step:2 file: read_a_smart-contract’s_state_on_a_cosmos-sdk_ethermint_chain_using_the_viem_library
# client.py
from web3 import Web3
from web3.middleware import geth_poa_middleware


def get_web3_client(rpc_url: str, chain_id: int) -> Web3:
    '''Return a configured Web3 HTTP provider.'''
    if not rpc_url.startswith(('http://', 'https://')):
        raise ValueError('RPC URL must start with http or https.')

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
    if not w3.isConnected():
        raise ConnectionError(f'Unable to connect to RPC endpoint {rpc_url}')

    # Many Cosmos EVM chains use the same header layout as POA networks
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    w3.chain_id = chain_id
    return w3


# step:4 file: read_a_smart-contract’s_state_on_a_cosmos-sdk_ethermint_chain_using_the_viem_library
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
import os

from client import get_web3_client
from eth_typing import ChecksumAddress
from web3 import Web3
from web3.exceptions import ABIFunctionNotFound

app = FastAPI()

RPC_URL = os.getenv('COSMOS_EVM_RPC', 'https://evmos-rpc.polkachu.com')
CHAIN_ID = int(os.getenv('COSMOS_EVM_CHAIN_ID', '9001'))

# The Web3 connection is created once at startup for efficiency
w3 = get_web3_client(RPC_URL, CHAIN_ID)

class ReadContractRequest(BaseModel):
    contractAddress: ChecksumAddress
    abi: List[dict]
    functionName: str
    args: List[Any] | None = []

@app.post('/api/read-contract')
async def read_contract(req: ReadContractRequest):
    try:
        contract = w3.eth.contract(address=req.contractAddress, abi=req.abi)
        try:
            fn = getattr(contract.functions, req.functionName)
        except AttributeError:
            raise ABIFunctionNotFound(f'{req.functionName} not found in provided ABI')

        data = fn(*req.args).call()
        return {'data': data}
    except Exception as e:
        # Convert all errors to HTTP 400s so the frontend can handle them uniformly
        raise HTTPException(status_code=400, detail=str(e))


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


# step:2 file: initialize_a_new_wallet_at_a_provided_url
import os
import base64
import requests
from typing import Any

# Configuration is read from environment variables for security
RPC_ENDPOINT = os.getenv('RPC_ENDPOINT', 'http://localhost:26657')
RPC_USERNAME = os.getenv('RPC_USERNAME')
RPC_PASSWORD = os.getenv('RPC_PASSWORD')

def get_rpc_session() -> requests.Session:
    """Return a requests.Session pre-configured with basic-auth headers."""
    session = requests.Session()
    session.headers.update({'Content-Type': 'application/json'})

    # Attach Basic-Auth only when credentials exist
    if RPC_USERNAME and RPC_PASSWORD:
        auth_pair = f'{RPC_USERNAME}:{RPC_PASSWORD}'.encode()
        session.headers.update({
            'Authorization': f'Basic {base64.b64encode(auth_pair).decode()}'
        })

    # Non-standard helper attribute for convenience
    session.base_url = RPC_ENDPOINT  # type: ignore[attr-defined]
    return session

# Optional quick self-test when the file is executed directly
if __name__ == '__main__':
    try:
        s = get_rpc_session()
        resp = s.post(s.base_url, json={'jsonrpc': '2.0', 'method': 'health', 'params': [], 'id': 1}, timeout=5)
        resp.raise_for_status()
        print('RPC health check succeeded')
    except Exception as err:
        print(f'RPC health check failed: {err}')


# step:3 file: initialize_a_new_wallet_at_a_provided_url
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import requests

from rpc_client import get_rpc_session

app = FastAPI()

class InitWalletRequest(BaseModel):
    remote_url: str

@app.post('/api/initialize-wallet')
def initialize_wallet(payload: InitWalletRequest):
    """Generate a new private key and create a wallet at the given remote URL."""
    session = get_rpc_session()

    json_rpc_body = {
        'jsonrpc': '2.0',
        'method': 'personal_initializeWallet',
        'params': [payload.remote_url],
        'id': 1,
    }

    try:
        response = session.post(session.base_url, json=json_rpc_body, timeout=15)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        # JSON-RPC error handling
        if data.get('error') is not None:
            raise HTTPException(status_code=500, detail=data['error'])

        return {'wallet': data.get('result')}
    except requests.RequestException as req_err:
        raise HTTPException(status_code=502, detail=f'Network error: {req_err}')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:1 file: Show the number of active cron schedules
import json
import subprocess
from typing import List, Dict


def query_all_cron_schedules(limit: int = 1000) -> List[Dict]:
    """Return every cron schedule on-chain.

    Args:
        limit: Max items per page (must be ≤ CLI max-limit).

    Returns:
        A list with all schedule objects.
    """
    schedules: List[Dict] = []
    next_key: str | None = ""

    try:
        while True:
            # Build CLI command
            cmd = [
                "neutrond", "query", "cron", "schedules",
                "--limit", str(limit), "--output", "json"
            ]
            if next_key:
                cmd += ["--page-key", next_key]

            # Execute the command and parse stdout
            raw = subprocess.check_output(cmd, text=True)
            data = json.loads(raw)

            # Merge current page
            schedules.extend(data.get("schedules", []))

            # Prepare for the next loop
            next_key = data.get("pagination", {}).get("next_key")
            if not next_key:
                break
    except FileNotFoundError:
        raise RuntimeError("`neutrond` CLI not found – please install or add to PATH.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"CLI call failed: {err.stderr or err}")
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Unexpected CLI output – JSON decode error: {err}")

    return schedules


# step:2 file: Show the number of active cron schedules
def count_array_elements(items: list) -> int:
    """Safely count array length with a sanity check."""
    if items is None:
        raise ValueError("Input is None – expected a list.")
    if not isinstance(items, list):
        raise TypeError(f"Expected list, got {type(items)}")
    return len(items)


# step:3 file: Show the number of active cron schedules
def display_result(count: int) -> None:
    """Print the final schedule count in the required format."""
    if count < 0:
        raise ValueError("Count cannot be negative.")
    print(f"Active schedules: {count}")


# step:2 file: retrieve_the_list_of_pending_(unconfirmed)_transactions_from_the_cosmos_mempool
# api_mempool.py
from fastapi import APIRouter, HTTPException, Query
import requests

router = APIRouter()


def _build_rpc_url(rpc_endpoint: str, limit: int) -> str:
    """Helper that assembles the final RPC URL without double slashes."""
    return f"{rpc_endpoint.rstrip('/')}/unconfirmed_txs?limit={limit}"


@router.get('/api/unconfirmed_txs')
def fetch_unconfirmed_txs(
    limit: int = Query(50, ge=1, le=1000),
    rpc_endpoint: str = Query('http://localhost:26657')
):
    """Fetch raw unconfirmed transactions from a CometBFT node."""
    url = _build_rpc_url(rpc_endpoint, limit)
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        # Surface networking problems as a clean 502 to the caller
        raise HTTPException(status_code=502, detail=f'Unable to reach RPC endpoint: {exc}')


# step:3 file: retrieve_the_list_of_pending_(unconfirmed)_transactions_from_the_cosmos_mempool
# Extend api_mempool.py
from fastapi import Body

@router.post('/api/parse_unconfirmed_txs')
def parse_unconfirmed_txs(raw_response: dict = Body(...)):
    """Extract `n_txs` and `txs` from the raw RPC response."""
    try:
        result = raw_response.get('result', {})
        n_txs = int(result.get('n_txs', 0))
        txs = result.get('txs', [])
        return {
            'n_txs': n_txs,
            'txs': txs  # still base64-encoded per CometBFT spec
        }
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f'Malformed RPC response: {exc}')


# step:2 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
# backend/routes/solvbtc.py
from fastapi import APIRouter, HTTPException
import httpx
import os

router = APIRouter()
SOLV_GATEWAY_URL = os.getenv('SOLV_GATEWAY_URL', 'https://api.solv.finance/solvbtc')

@router.post('/api/solvbtc/deposit-address')
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


# step:3 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
# backend/routes/btc_tx.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from decimal import Decimal
from bit import PrivateKey

router = APIRouter()

class ConstructTxPayload(BaseModel):
    wif: str
    destination: str
    fee_sat_per_byte: int = 10

    @validator('fee_sat_per_byte')
    def fee_positive(cls, v):
        if v <= 0:
            raise ValueError('fee_sat_per_byte must be positive')
        return v

@router.post('/api/btc/construct-tx')
def construct_and_sign_btc_tx(payload: ConstructTxPayload):
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


# step:4 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
# backend/routes/btc_broadcast.py
from fastapi import APIRouter, HTTPException
import httpx

router = APIRouter()

@router.post('/api/btc/broadcast')
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


# step:5 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
# backend/routes/btc_confirm.py
import asyncio
import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get('/api/btc/confirmations/{txid}')
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


# step:6 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
# backend/routes/solvbtc_mint.py
from fastapi import APIRouter, HTTPException
from web3 import Web3
import json, os

router = APIRouter()

ETH_RPC_URL = os.getenv('ETH_RPC_URL')
MINT_CONTRACT_ADDRESS = os.getenv('SOLV_MINT_CONTRACT_ADDRESS')
BACKEND_PRIVATE_KEY = os.getenv('ETH_PRIVATE_KEY')

# Load minimal ABI containing the `mint` function
with open('SolvBTCMintABI.json') as f:
    MINT_ABI = json.load(f)

@router.post('/api/solvbtc/mint')
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


# step:7 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
# backend/routes/bridge.py
from fastapi import APIRouter, HTTPException
import httpx

router = APIRouter()
AXELAR_GATEWAY_URL = 'https://axelar-api.ping.pub'  # Example public REST endpoint

@router.post('/api/bridge/solvbtc')
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


# step:8 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
# backend/routes/neutron_balance.py
from fastapi import APIRouter, HTTPException
from cosmpy.aerial.client import LedgerClient, NetworkConfig
import os

router = APIRouter()
NEUTRON_RPC = os.getenv('NEUTRON_RPC', 'https://rpc-palvus.neutron.org')
IBC_DENOM_SOLVBTC = os.getenv('IBC_DENOM_SOLVBTC', 'ibc/xxxxxxxxxxxxxxxxxxxxxxxx')

network_cfg = NetworkConfig(
    chain_id='neutron-1',
    url=NEUTRON_RPC,
    fee_denomination='untrn',
    staking_denomination='untrn',
    fee_minimum_gas_price=0,
)

@router.get('/api/neutron/balance/{address}')
def query_balance(address: str):
    """Return solvBTC voucher balance on Neutron."""
    try:
        client = LedgerClient(network_cfg)
        balance = client.query_bank_balance(address, denom=IBC_DENOM_SOLVBTC)
        return {'address': address, 'solvbtc_balance': str(balance.amount)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:1 file: send_1000stake_from_key_my_validator_to_a_recipient_address
import subprocess


def get_key_address(key_name: str = "my_validator", keyring_backend: str = "test") -> str:
    """Return the bech32 address for a key stored in the local key-ring.

    Args:
        key_name: The name of the key to look up.
        keyring_backend: One of [os|file|test]. "test" stores keys unencrypted for dev chains.

    Raises:
        RuntimeError: If the CLI call fails or returns an empty address.
    """
    try:
        cmd = [
            "simd", "keys", "show", key_name,
            "--keyring-backend", keyring_backend,
            "-a"  # address only
        ]
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        address = result.stdout.strip()
        if not address:
            raise RuntimeError("Received empty address from key-ring query")
        return address
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"simd keys show failed: {e.stderr}") from e


# step:2 file: send_1000stake_from_key_my_validator_to_a_recipient_address
from bech32 import bech32_decode


def validate_recipient_address(address: str, expected_prefix: str = "cosmos") -> str:
    """Verify HRP and checksum of a bech32 address.

    Args:
        address: The address to validate.
        expected_prefix: Human-readable prefix (HRP) for the target chain.
    Returns:
        The original address if valid.
    Raises:
        ValueError: If the address is malformed or has an unexpected prefix.
    """
    hrp, data = bech32_decode(address)
    if hrp != expected_prefix or data is None:
        raise ValueError(f"{address} is not a valid {expected_prefix} bech32 address")
    return address


# step:3 file: send_1000stake_from_key_my_validator_to_a_recipient_address
import os
import subprocess


def build_send_tx(
    sender: str,
    recipient: str,
    amount: str = "1000stake",
    fee: str = "200stake",
    chain_id: str = "my-test-chain",
    outfile: str = "unsigned_tx.json",
) -> str:
    """Create an unsigned MsgSend and persist it to disk.

    Returns the file path of the unsigned tx JSON.
    """
    cmd = [
        "simd", "tx", "bank", "send", sender, recipient, amount,
        "--generate-only",
        "--fees", fee,
        "--chain-id", chain_id,
        "--output", "json"
    ]
    try:
        with open(outfile, "w", encoding="utf-8") as fp:
            subprocess.run(cmd, check=True, text=True, stdout=fp)
        if not os.path.exists(outfile):
            raise RuntimeError("Unsigned TX file was not created")
        return outfile
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to build tx: {e.stderr}") from e


# step:4 file: send_1000stake_from_key_my_validator_to_a_recipient_address
import os
import subprocess


def sign_tx(
    unsigned_tx_path: str,
    key_name: str = "my_validator",
    keyring_backend: str = "test",
    chain_id: str = "my-test-chain",
    outfile: str = "signed_tx.json",
) -> str:
    """Sign an unsigned transaction file with a local key-ring key."""
    cmd = [
        "simd", "tx", "sign", unsigned_tx_path,
        "--from", key_name,
        "--keyring-backend", keyring_backend,
        "--chain-id", chain_id,
        "--output", "json",
        "--yes"  # auto-confirm
    ]
    try:
        with open(outfile, "w", encoding="utf-8") as fp:
            subprocess.run(cmd, check=True, text=True, stdout=fp)
        if not os.path.exists(outfile):
            raise RuntimeError("Signed TX file was not created")
        return outfile
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Signing failed: {e.stderr}") from e


# step:5 file: send_1000stake_from_key_my_validator_to_a_recipient_address
import json
import subprocess


def broadcast_tx(
    signed_tx_path: str,
    chain_id: str = "my-test-chain",
    node_rpc: str = "http://localhost:26657",
) -> str:
    """Broadcast a signed tx and return its hash."""
    cmd = [
        "simd", "tx", "broadcast", signed_tx_path,
        "--node", node_rpc,
        "--chain-id", chain_id,
        "--output", "json"
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        response = json.loads(result.stdout)
        tx_hash = response.get("txhash")
        if not tx_hash:
            raise RuntimeError(f"Unexpected response: {response}")
        return tx_hash
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Broadcast failed: {e.stderr}") from e


# step:1 file: generate_a_gentx_for_my_validator_staking_100000000stake
import os
import subprocess
import json

# Environment-driven configuration so the same code works for any chain binary
CHAIN_BINARY = os.getenv("CHAIN_BINARY", "neutrond")  # e.g. "gaiad", "junod", etc.
KEY_HOME = os.getenv("KEY_HOME", os.path.expanduser("~/.neutrond"))
KEYRING_BACKEND = os.getenv("KEYRING_BACKEND", "test")

def ensure_key_exists(key_name: str) -> dict:
    """
    Ensure a key with `key_name` is present in the local keyring. If the key is
    absent, it is created. The function returns a status dictionary describing
    what happened.
    """
    try:
        # Attempt to show the key; success means it already exists.
        cmd_show = [
            CHAIN_BINARY,
            "keys",
            "show",
            key_name,
            "--output",
            "json",
            "--keyring-backend",
            KEYRING_BACKEND,
            "--home",
            KEY_HOME,
        ]
        show_result = subprocess.check_output(cmd_show, text=True)
        info = json.loads(show_result)
        return {"status": "exists", "address": info["address"]}
    except subprocess.CalledProcessError:
        # Key does not exist – create it now.
        cmd_add = [
            CHAIN_BINARY,
            "keys",
            "add",
            key_name,
            "--output",
            "json",
            "--keyring-backend",
            KEYRING_BACKEND,
            "--home",
            KEY_HOME,
        ]
        try:
            add_result = subprocess.check_output(cmd_add, text=True)
            info = json.loads(add_result)
            return {
                "status": "created",
                "address": info.get("address"),
                "mnemonic": info.get("mnemonic", ""),
            }
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to create key {key_name}: {exc}")


# step:2 file: generate_a_gentx_for_my_validator_staking_100000000stake
import os
import subprocess

CHAIN_BINARY = os.getenv("CHAIN_BINARY", "neutrond")
CHAIN_ID = os.getenv("CHAIN_ID", "neutron-1")
KEY_HOME = os.getenv("KEY_HOME", os.path.expanduser("~/.neutrond"))
KEYRING_BACKEND = os.getenv("KEYRING_BACKEND", "test")

def generate_gentx(key_name: str, amount: str = "100000000stake") -> str:
    """
    Produce a gentx for `key_name` staking `amount`.  Returns the full path to
    the resulting gentx JSON file inside `<home>/config/gentx/`.
    """
    # Run `<chain_binary> gentx ...` exactly as the user specified.
    cmd = [
        CHAIN_BINARY,
        "gentx",
        key_name,
        amount,
        "--chain-id",
        CHAIN_ID,
        "--keyring-backend",
        KEYRING_BACKEND,
        "--home",
        KEY_HOME,
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gentx command failed: {exc}") from exc

    # After a successful run the gentx sits in `<home>/config/gentx/`.
    gentx_dir = os.path.join(KEY_HOME, "config", "gentx")
    all_gentxs = [
        os.path.join(gentx_dir, f)
        for f in os.listdir(gentx_dir)
        if f.endswith(".json")
    ]
    if not all_gentxs:
        raise FileNotFoundError("No gentx file found after running gentx command.")

    # Return the newest file (highest mtime) which is the one we just created.
    latest_path = max(all_gentxs, key=os.path.getmtime)
    return latest_path


# step:3 file: generate_a_gentx_for_my_validator_staking_100000000stake
import os
import json
import subprocess
import re

CHAIN_BINARY = os.getenv("CHAIN_BINARY", "neutrond")
KEY_HOME = os.getenv("KEY_HOME", os.path.expanduser("~/.neutrond"))
KEYRING_BACKEND = os.getenv("KEYRING_BACKEND", "test")

def validate_gentx(
    gentx_path: str,
    key_name: str,
    expected_amount: str = "100000000",
    expected_denom: str = "stake",
) -> bool:
    """
    Perform three sanity checks on a gentx:
      1. Confirm it is valid JSON.
      2. Confirm it stakes exactly `expected_amount``expected_denom`.
      3. Confirm the validator address was derived from `key_name`.
    Returns True if the transaction passes all checks, else raises an Exception.
    """
    # 1️⃣  Parse and validate JSON structure
    try:
        with open(gentx_path, "r") as fp:
            tx = json.load(fp)
    except json.JSONDecodeError as exc:
        raise ValueError(f"gentx is not valid JSON: {exc}") from exc

    # 2️⃣  Extract the create-validator message and validate the stake amount
    try:
        # cosmos-sdk v0.47+ layout
        msg = tx["body"]["messages"][0]
        stake_obj = msg["value"]["value"] if "value" in msg["value"] else msg["value"]
        amount = stake_obj["amount"]
        denom = stake_obj["denom"]
    except (KeyError, TypeError):
        # Legacy layout fallback
        try:
            msg = tx["value"]["msg"][0]["value"]
            amount = msg["value"]["amount"]
            denom = msg["value"]["denom"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Unable to extract stake amount/denom from gentx.") from exc

    if str(amount) != str(expected_amount) or denom != expected_denom:
        raise ValueError(
            f"Gentx stakes {amount}{denom}; expected {expected_amount}{expected_denom}."
        )

    # 3️⃣  Confirm the validator address matches the operator address for key_name
    try:
        key_info_raw = subprocess.check_output(
            [
                CHAIN_BINARY,
                "keys",
                "show",
                key_name,
                "--output",
                "json",
                "--keyring-backend",
                KEYRING_BACKEND,
                "--home",
                KEY_HOME,
            ],
            text=True,
        )
        key_info = json.loads(key_info_raw)
        delegator_addr = key_info["address"]
        # Derive the valoper address; this assumes standard bech32 prefixes.
        operator_addr = re.sub(r"^([a-z]+)", r"\1valoper", delegator_addr)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Unable to fetch key {key_name}: {exc}") from exc

    validator_addr_in_tx = msg.get("validator_address") or msg.get("value", {}).get("validator_address")
    if not validator_addr_in_tx:
        raise ValueError("Validator address missing from gentx.")

    if validator_addr_in_tx != operator_addr:
        raise ValueError(
            f"Gentx signed by {validator_addr_in_tx}; expected {operator_addr}."
        )

    # All checks passed 🎉
    return True


# step:1 file: persist_the_gaia_v5_keyring_backend_environment_variable_with_value_“pass”
import os
from pathlib import Path

PROFILE_CANDIDATES = [
    Path.home() / ".bashrc",
    Path.home() / ".zshrc",
]

EXPORT_LINE = "export GAIA_V5_KEYRING_BACKEND=pass\n"

def append_to_shell_profile():
    """Append the export line to the first existing shell profile found."""
    for profile in PROFILE_CANDIDATES:
        if profile.exists():
            # Read the file once to avoid duplicates
            content = profile.read_text()
            if "GAIA_V5_KEYRING_BACKEND" in content:
                print(f"[Info] Variable already present in {profile} – skipping append.")
                return str(profile)
            try:
                with profile.open("a", encoding="utf-8") as f:
                    f.write("\n" + EXPORT_LINE)
                print(f"[Success] Added GAIA_V5_KEYRING_BACKEND to {profile}")
                return str(profile)
            except Exception as e:
                raise RuntimeError(f"Unable to append to {profile}: {e}")
    # If no profile exists, create ~/.bashrc and append the line
    default_profile = Path.home() / ".bashrc"
    try:
        with default_profile.open("a", encoding="utf-8") as f:
            f.write(EXPORT_LINE)
        print(f"[Success] Created {default_profile} and set GAIA_V5_KEYRING_BACKEND.")
        return str(default_profile)
    except Exception as e:
        raise RuntimeError(f"Failed to create {default_profile}: {e}")


# step:2 file: persist_the_gaia_v5_keyring_backend_environment_variable_with_value_“pass”
import os
import subprocess
from pathlib import Path


def reload_shell_profile():
    """Source the active shell profile in a *sub-shell* and propagate variables to the Python process."""
    # Determine which profile exists
    profile_path = next((p for p in (Path.home()/'.bashrc', Path.home()/'.zshrc') if p.exists()), None)
    if not profile_path:
        raise FileNotFoundError("No shell profile found to reload.")

    # Launch a bash subshell that sources the profile and prints the variable
    try:
        cmd = ["bash", "-c", f"source {profile_path} && echo -n $GAIA_V5_KEYRING_BACKEND"]
        result = subprocess.check_output(cmd, text=True)
        # Update current process environment so later Python code can see it
        os.environ["GAIA_V5_KEYRING_BACKEND"] = result.strip()
        print(f"[Success] Reloaded profile from {profile_path}. GAIA_V5_KEYRING_BACKEND={result.strip()}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to source {profile_path}: {e}")


# step:3 file: persist_the_gaia_v5_keyring_backend_environment_variable_with_value_“pass”
import os

def verify_env_var():
    value = os.environ.get("GAIA_V5_KEYRING_BACKEND")
    if value != "pass":
        raise EnvironmentError("GAIA_V5_KEYRING_BACKEND is not set to 'pass'. Current value: {}".format(value))
    print(f"[Verified] GAIA_V5_KEYRING_BACKEND={value}")
    return value


# step:1 file: load_application_at_block_height_1000
# backend/search_cosmos_docs.py
from fastapi import FastAPI, HTTPException
import requests
import urllib.parse

app = FastAPI()

@app.get('/api/search_cosmos_docs')
async def search_cosmos_docs(q: str):
    # Searches Cosmos documentation for the supplied query string and
    # returns raw HTML of the top results.
    try:
        encoded_q = urllib.parse.quote_plus(f'site:docs.cosmos.network {q}')
        url = f'https://duckduckgo.com/html/?q={encoded_q}'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {'html': response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:2 file: load_application_at_block_height_1000
# backend/stop_node.py
from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.post('/api/stop_node')
async def stop_node(process_name: str = 'appd'):
    # Terminates any running process matching the provided name.
    try:
        completed = subprocess.run(
            ['pkill', '-f', process_name],
            check=False,
            capture_output=True,
            text=True
        )
        return {
            'returncode': completed.returncode,
            'stdout': completed.stdout,
            'stderr': completed.stderr
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:3 file: load_application_at_block_height_1000
# backend/replay_node.py
from fastapi import FastAPI, HTTPException
import subprocess
import shlex
import tempfile

app = FastAPI()

@app.post('/api/replay_node')
async def replay_node(node_home: str, halt_height: int = 1000):
    # Runs `appd start` with --recover and --halt-height flags and waits
    # for the process to exit automatically at the requested height.
    cmd = f'appd start --home {shlex.quote(node_home)} --recover --halt-height {halt_height}'
    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            check=False
        )
        # Persist stdout so it can be inspected in the next step.
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log', mode='w')
        temp_file.write(result.stdout)
        temp_file.close()
        return {
            'returncode': result.returncode,
            'log_path': temp_file.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:4 file: load_application_at_block_height_1000
# backend/verify_halt_height.py
from fastapi import FastAPI, HTTPException
import re
from pathlib import Path

app = FastAPI()

@app.get('/api/verify_halt_height')
async def verify_halt_height(log_path: str, halt_height: int = 1000):
    # Examines the log file to confirm the node halted at the expected height.
    try:
        text = Path(log_path).read_text()
        heights = [int(h) for h in re.findall(r'height[=: ]+(\d+)', text)]
        if not heights:
            return {'success': False, 'reason': 'no_height_entries_found'}
        last_height = max(heights)
        return {
            'success': last_height == halt_height,
            'last_height': last_height,
            'expected_height': halt_height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:3 file: query_a_cosmwasm_smart_contract_through_a_rest_endpoint_using_a_base64-encoded_query
from fastapi import FastAPI, HTTPException
import httpx
import os

app = FastAPI()

# Change or override via environment variable for different chains/networks
LCD_ENDPOINT = os.getenv('LCD_ENDPOINT', 'https://rest.cosmos.directory/juno')

@app.get('/api/contract/query')
async def contract_query(contract_addr: str, base64_query: str):
    """
    Forward a base64-encoded smart-contract query to the chain's LCD REST endpoint
    and return the JSON response back to the caller.
    """
    lcd_url = f"{LCD_ENDPOINT}/cosmwasm/wasm/v1/contract/{contract_addr}/smart/{base64_query}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(lcd_url)
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f'Network error: {str(e)}')

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()


# step:1 file: enable_the_rest_api_server_via_app.toml
from pathlib import Path
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Map well-known node names to the folder that the daemon initialises.
NODE_HOME_MAPPING = {
    "gaia": ".gaia",      # gaiad
    "wasmd": ".wasmd",    # wasmd
    "osmosis": ".osmosisd" # osmosisd
}

@app.get("/node-home")
async def get_node_home(node: str = "gaia"):
    """Return the absolute path to the requested node’s home directory."""
    folder = NODE_HOME_MAPPING.get(node.lower())
    if folder is None:
        raise HTTPException(status_code=400, detail=f"Unsupported node '{node}'.")

    home_path = Path.home() / folder

    if not home_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Home directory {home_path} does not exist. Initialise the node or supply the correct name."
        )

    return {"home": str(home_path)}


# step:2 file: enable_the_rest_api_server_via_app.toml
from pathlib import Path
import shutil
import toml
from fastapi import FastAPI, HTTPException, Query

app = FastAPI()

@app.get("/config")
async def open_config_file(home: str = Query(..., description="Absolute path to the node home directory")):
    """Open <home>/config/app.toml and return its contents as a Python dict."""
    config_path = Path(home) / "config" / "app.toml"

    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"{config_path} not found.")

    # Create a backup once to preserve the original file.
    backup = config_path.with_suffix(".toml.bak")
    if not backup.exists():
        shutil.copy(config_path, backup)

    with config_path.open("r") as f:
        config_dict = toml.load(f)

    return {"path": str(config_path), "config": config_dict}


# step:3 file: enable_the_rest_api_server_via_app.toml
from pathlib import Path
import toml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ApiSettings(BaseModel):
    home: str             # node home directory
    enable: bool = True   # turn REST server on/off
    swagger: bool = True  # expose Swagger docs
    address: str = "tcp://0.0.0.0:1317"  # listen address

@app.post("/config/api")
async def update_toml_key(settings: ApiSettings):
    """Patch the [api] section inside app.toml with the supplied values."""
    cfg_path = Path(settings.home) / "config" / "app.toml"

    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail=f"{cfg_path} not found.")

    # Load current configuration.
    with cfg_path.open("r") as f:
        cfg = toml.load(f)

    # Ensure the api section exists then update keys.
    cfg.setdefault("api", {})
    cfg["api"].update({
        "enable": settings.enable,
        "swagger": settings.swagger,
        "address": settings.address
    })

    # Persist changes.
    with cfg_path.open("w") as f:
        toml.dump(cfg, f)

    return {"message": "[api] section updated successfully", "api": cfg["api"]}


# step:4 file: enable_the_rest_api_server_via_app.toml
import subprocess
from fastapi import FastAPI, HTTPException

app = FastAPI()

SERVICE_NAME = {
    "gaia": "gaiad",
    "wasmd": "wasmd",
    "osmosis": "osmosisd"
}

@app.post("/restart-node")
async def save_and_restart_node(node: str = "gaia"):
    """Restart the node’s systemd service so the REST API starts with the new config."""
    service = SERVICE_NAME.get(node.lower())

    if service is None:
        raise HTTPException(status_code=400, detail=f"Unsupported node '{node}'.")

    try:
        # Requires the backend process to have permissions (run as root or via sudoers).
        subprocess.run(["systemctl", "restart", service], check=True)
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"Failed to restart {service}: {err}")

    return {"message": f"{service} restarted successfully"}


# step:5 file: connect_remix_ide_to_cosmos_evm_via_injected_provider
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import solcx

app = FastAPI()

# Install or ensure the required Solidity compiler version is present
SOLC_VERSION = '0.8.20'
try:
    solcx.install_solc(SOLC_VERSION)
except Exception:
    # Already installed or network unavailable; compilation will raise if version missing
    pass

class CompilePayload(BaseModel):
    source: str          # Solidity source code
    contract_name: str   # Contract name to compile

@app.post('/api/compile_contract')
async def compile_contract(payload: CompilePayload):
    # Compile Solidity source
    try:
        compiled = solcx.compile_source(
            payload.source,
            output_values=['abi', 'bin'],
            solc_version=SOLC_VERSION,
        )
    except solcx.exceptions.SolcError as e:
        raise HTTPException(status_code=400, detail=f'Compilation error: {e}')

    identifier = f"<stdin>:{payload.contract_name}"
    if identifier not in compiled:
        raise HTTPException(status_code=400, detail='Contract name not found after compilation.')

    contract_interface = compiled[identifier]
    abi = contract_interface['abi']
    bytecode = '0x' + contract_interface['bin']  # Prefix bytecode with 0x for deployment

    return {'abi': abi, 'bytecode': bytecode}


# step:1 file: set_garbage_collection_percentage_to_100
import argparse
import re
import shutil
from pathlib import Path
import sys


def update_environment(service_file: str, env_key: str = "GOGC", env_value: str = "100") -> bool:
    """
    Adds or updates Environment="GOGC=100" inside the [Service] section of the
    provided systemd unit file.

    Returns True if the file was modified, False if no change was necessary.
    """
    service_path = Path(service_file).expanduser()

    if not service_path.exists():
        raise FileNotFoundError(f"The systemd file {service_path} does not exist.")

    original_text = service_path.read_text()
    updated_text = original_text

    pattern = re.compile(rf'^Environment=.*{env_key}=\d+', re.MULTILINE)
    replacement = f'Environment="{env_key}={env_value}"'

    if pattern.search(original_text):
        # Replace current setting
        updated_text = pattern.sub(replacement, original_text)
    else:
        # Insert new environment line directly under [Service] section
        updated_text = original_text.replace("[Service]", f"[Service]\n{replacement}", 1)

    if updated_text == original_text:
        print(f"{env_key} already set to {env_value}. No update performed.")
        return False

    # Backup original file
    backup_path = service_path.with_suffix(".bak")
    shutil.copy(service_path, backup_path)

    # Write new content
    service_path.write_text(updated_text)
    print(f"Updated {service_path}. Backup stored at {backup_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensure GOGC=100 is present in a systemd service file.")
    parser.add_argument("--service-file", required=True, help="Path to the systemd unit file (e.g., /etc/systemd/system/cosmosd.service)")
    parser.add_argument("--value", default="100", help="GC percentage to apply (default: 100)")
    args = parser.parse_args()

    try:
        modified = update_environment(args.service_file, "GOGC", args.value)
        sys.exit(0 if modified else 1)
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(2)



# step:2 file: set_garbage_collection_percentage_to_100
import argparse
import subprocess
import sys


def restart_node(service_name: str = "cosmosd"):
    """
    Reloads systemd and restarts the provided service.
    Requires sudo privileges.
    """
    try:
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "restart", service_name], check=True)
        subprocess.run(["systemctl", "is-active", "--quiet", service_name], check=True)
        print(f"{service_name} restarted successfully.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to restart {service_name}: {err}") from err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restart a Cosmos SDK daemon to apply new environment variables.")
    parser.add_argument("--service-name", default="cosmosd", help="systemd service name (default: cosmosd)")
    args = parser.parse_args()

    try:
        restart_node(args.service_name)
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)



# step:3 file: set_garbage_collection_percentage_to_100
import argparse
import json
import urllib.request


def validate_gc_setting(host: str = "http://localhost:6060") -> int:
    """
    Fetches Go runtime debug vars from the given host and returns the GCPercent value.
    """
    url = f"{host.rstrip('/')}/debug/vars"
    with urllib.request.urlopen(url, timeout=3) as response:
        payload = json.loads(response.read().decode())
        gc_percent = payload.get("GCPercent")
        print(f"GCPercent reported: {gc_percent}")
        if gc_percent == 100:
            print("✅ GOGC is correctly set to 100.")
        else:
            print("❌ GOGC is NOT 100. Check your configuration.")
        return gc_percent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate that GOGC=100 has been applied.")
    parser.add_argument("--host", default="http://localhost:6060", help="Host where pprof/debug endpoint is exposed")
    args = parser.parse_args()
    validate_gc_setting(args.host)



# step:3 file: vote_yes_on_proposal_5
#################################
# governance_utils.py           #
#################################

from enum import IntEnum
from cosmpy.protos.cosmos.gov.v1beta1.tx_pb2 import MsgVote

class VoteOption(IntEnum):
    """Subset of the chain's VoteOption enum for clarity."""
    VOTE_OPTION_UNSPECIFIED = 0
    VOTE_OPTION_YES = 1
    VOTE_OPTION_ABSTAIN = 2
    VOTE_OPTION_NO = 3
    VOTE_OPTION_NO_WITH_VETO = 4


def construct_msg_vote_yes(voter: str, proposal_id: int) -> MsgVote:
    """Returns a MsgVote protobuf message (YES)."""
    if not voter:
        raise ValueError('Voter address must be provided')
    if proposal_id <= 0:
        raise ValueError('Proposal ID must be a positive integer')

    msg = MsgVote()
    msg.proposal_id = proposal_id
    msg.voter = voter
    msg.option = VoteOption.VOTE_OPTION_YES  # YES
    return msg


# step:4 file: vote_yes_on_proposal_5
#################################
# main.py                       #
#################################

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction
from governance_utils import construct_msg_vote_yes

app = FastAPI(title='Governance Vote BFF')

# --- Chain configuration (override via env if needed) ---
CHAIN_ID = os.getenv('CHAIN_ID', 'cosmoshub-4')
GRPC_ENDPOINT = os.getenv('GRPC_ENDPOINT', 'grpc+https://cosmoshub.grpc.cosmostation.io:443')
MNEMONIC = os.getenv('SERVER_WALLET_MNEMONIC')  # NEVER commit this to source control!

if MNEMONIC is None:
    raise RuntimeError('SERVER_WALLET_MNEMONIC environment variable is not set')

network_cfg = NetworkConfig(
    chain_id=CHAIN_ID,
    url=GRPC_ENDPOINT,
)

client = LedgerClient(network_cfg)
wallet = LocalWallet.from_mnemonic(MNEMONIC)

class VoteRequest(BaseModel):
    proposal_id: int

@app.post('/api/gov/vote_yes')
async def sign_and_broadcast_tx(payload: VoteRequest):
    """Signs a YES vote for the configured wallet and broadcasts the TX."""
    try:
        # 1. Build the MsgVote
        msg = construct_msg_vote_yes(wallet.address(), payload.proposal_id)

        # 2. Build & sign the transaction
        tx = Transaction()
        tx.add_message(msg)
        tx.with_signer(wallet)

        # 3. Broadcast and wait for inclusion in a block
        response = client.broadcast_block(tx)
        if not response.is_successful():
            raise HTTPException(status_code=500, detail=f'Broadcast error: {response.raw_log}')

        return {"tx_hash": response.tx_hash}

    except HTTPException:
        raise  # re-raise FastAPI exceptions untouched
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:1 file: add_a_`tx_bank_send`_cli_command_to_the_chain_binary
import re
import sys
from pathlib import Path


def verify_bank_module(app_go_path: str = "app/app.go") -> bool:
    '''
    Verifies that the x/bank module is imported in `app/app.go` and included in
    the ModuleManager. Returns True when both conditions pass, else raises.
    '''
    path = Path(app_go_path)
    if not path.exists():
        raise FileNotFoundError(f"{app_go_path} not found")

    content = path.read_text()

    # Check for import statement
    if '"github.com/cosmos/cosmos-sdk/x/bank"' not in content:
        raise ValueError("x/bank module import not found in app/app.go")

    # Check for module registration
    if "bank.NewAppModule" not in content and "bank.AppModuleBasic" not in content:
        raise ValueError("x/bank module not registered in ModuleManager")

    print("✅ x/bank module is correctly imported and registered.")
    return True


if __name__ == "__main__":
    verify_bank_module(*sys.argv[1:])


# step:2 file: add_a_`tx_bank_send`_cli_command_to_the_chain_binary
import os
from pathlib import Path

GO_FILE_CONTENT = '''// Code generated by automation; DO NOT EDIT.
package cmd

import (
    "fmt"

    "github.com/spf13/cobra"
    "github.com/cosmos/cosmos-sdk/client"
    sdkcli "github.com/cosmos/cosmos-sdk/client/flags"
    "github.com/cosmos/cosmos-sdk/client/tx"
    sdk "github.com/cosmos/cosmos-sdk/types"
    banktypes "github.com/cosmos/cosmos-sdk/x/bank/types"
)

// TxBankSendCmd creates and broadcasts a MsgSend transaction.
func TxBankSendCmd() *cobra.Command {
    cmd := &cobra.Command{
        Use:   "send [to_address] [amount]",
        Short: "Send tokens to another account",
        Args:  cobra.ExactArgs(2),
        RunE: func(cmd *cobra.Command, args []string) error {
            clientCtx := client.GetClientContextFromCmd(cmd)
            if clientCtx == nil {
                return fmt.Errorf("client context is nil")
            }

            toAddr, err := sdk.AccAddressFromBech32(args[0])
            if err != nil {
                return err
            }

            amount, err := sdk.ParseCoinsNormalized(args[1])
            if err != nil {
                return err
            }

            fromAddr := clientCtx.GetFromAddress()
            msg := banktypes.NewMsgSend(fromAddr, toAddr, amount)
            if err = msg.ValidateBasic(); err != nil {
                return err
            }

            return tx.GenerateOrBroadcastTxCLI(clientCtx, cmd.Flags(), msg)
        },
    }

    sdkcli.AddTxFlagsToCmd(cmd)
    return cmd
}
'''

def create_cli_command_file(file_path: str = "cmd/bank_send.go") -> None:
    '''
    Creates `cmd/bank_send.go` with the TxBankSendCmd implementation.
    '''
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"{file_path} already exists, skipping write.")
        return

    path.write_text(GO_FILE_CONTENT)
    print(f"✅ {file_path} created with TxBankSendCmd.")


if __name__ == "__main__":
    create_cli_command_file()


# step:3 file: add_a_`tx_bank_send`_cli_command_to_the_chain_binary
import re
from pathlib import Path


def register_command_root(root_file: str = "cmd/root.go") -> None:
    '''
    Inserts TxBankSendCmd into the tx sub-command tree within cmd/root.go.
    '''
    path = Path(root_file)
    if not path.exists():
        raise FileNotFoundError(f"{root_file} not found")

    content = path.read_text()

    # Skip if already exists
    if "TxBankSendCmd()" in content:
        print("TxBankSendCmd already registered, skipping.")
        return

    pattern = r"(?s)(func\\s+NewTxCmd\\s*\\(.*?\\)\\s*\\*cobra.Command\\s*{.*?})"
    match = re.search(pattern, content)
    if not match:
        raise ValueError("NewTxCmd definition not found in root.go")

    block = match.group(1)

    if "TxBankSendCmd()" not in block:
        modified_block = block.replace("return txCmd", "    txCmd.AddCommand(TxBankSendCmd())\\n\\n    return txCmd")
    else:
        modified_block = block

    updated_content = content.replace(block, modified_block)
    path.write_text(updated_content)
    print("✅ TxBankSendCmd registered in root.go.")


if __name__ == "__main__":
    register_command_root()


# step:4 file: add_a_`tx_bank_send`_cli_command_to_the_chain_binary
import subprocess
import sys
from pathlib import Path


def compile_binary() -> None:
    '''
    Compiles the binary so the new CLI command is available.
    Tries `make install` first, then falls back to `go install ./...`.
    '''
    makefile_exists = Path("Makefile").exists()

    try:
        if makefile_exists:
            subprocess.run(["make", "install"], check=True)
        else:
            raise FileNotFoundError
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("`make install` unavailable; falling back to `go install ./...`")
        try:
            subprocess.run(["go", "install", "./..."], check=True)
        except subprocess.CalledProcessError as e:
            print("❌ Binary compilation failed:", e)
            sys.exit(1)

    print("✅ Binary compiled successfully.")


if __name__ == "__main__":
    compile_binary()


# step:5 file: add_a_`tx_bank_send`_cli_command_to_the_chain_binary
import subprocess
import sys


def validate_cli_help() -> None:
    '''
    Runs `appd tx bank send --help` to ensure command wiring is correct.
    '''
    try:
        subprocess.run(["appd", "tx", "bank", "send", "--help"], check=True)
        print("✅ Help output rendered correctly.")
    except subprocess.CalledProcessError as e:
        print("❌ Unable to render help output:", e)
        sys.exit(1)


if __name__ == "__main__":
    validate_cli_help()


# step:6 file: add_a_`tx_bank_send`_cli_command_to_the_chain_binary
import subprocess
import sys


def broadcast_test_tx(from_key: str, to_addr: str, amount: str = "1stake", chain_id: str = "localnet", node: str = "tcp://localhost:26657") -> None:
    '''
    Broadcasts a small MsgSend to verify the new CLI command works end-to-end.
    '''
    cmd = [
        "appd", "tx", "bank", "send", to_addr, amount,
        "--from", from_key,
        "--chain-id", chain_id,
        "--node", node,
        "-y"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Test transaction broadcasted successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Test transaction failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: broadcast_test_tx.py <from-key> <to-addr> [amount] [chain-id] [node]")
        sys.exit(1)

    broadcast_test_tx(*sys.argv[1:])


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



# step:4 file: Deploy the example contract to Neutron mainnet
# Step 4 – Sign, broadcast and obtain code_id
import json


def sign_and_broadcast_tx(client: LedgerClient, tx: Transaction):
    signed_tx = client.sign(tx)
    response = client.broadcast_tx_block(signed_tx)
    if response.code != 0:
        raise RuntimeError(f"Tx failed (code={response.code}): {response.raw_log}")
    return response


def extract_code_id(response) -> int:
    # Parses the code_id emitted by MsgStoreCode from the tx response
    try:
        logs = json.loads(response.raw_log)
        for event in logs[0]["events"]:
            if event["type"] == "store_code":
                for attr in event["attributes"]:
                    if attr["key"] in ("code_id", "codeID"):
                        return int(attr["value"])
    except (KeyError, ValueError, json.JSONDecodeError) as err:
        raise RuntimeError(f"Unable to extract code_id: {err}")
    raise RuntimeError("code_id not found in logs")


def upload_wasm_and_get_code_id(client: LedgerClient, tx: Transaction) -> int:
    resp = sign_and_broadcast_tx(client, tx)
    code_id = extract_code_id(resp)
    print(f"✓ Contract uploaded with code_id {code_id}")
    return code_id



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



# step:6 file: Deploy the example contract to Neutron mainnet
# Step 6 – Sign and broadcast instantiation tx
def instantiate_contract(client: LedgerClient, tx: Transaction):
    resp = sign_and_broadcast_tx(client, tx)
    print("✓ Contract instantiated")
    return resp



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



# step:2 file: execute_an_emergency_withdrawal_for_the_user’s_amber_trading_position
# backend/routes/amber.py
from fastapi import APIRouter, HTTPException
import httpx, base64, json, os

router = APIRouter()

# NOTE: replace with the real Amber contract address once known
AMBER_CONTRACT = os.getenv("AMBER_CONTRACT", "neutron1ambercontract...")
LCD_ENDPOINT   = os.getenv("LCD_ENDPOINT",   "https://rest-kralum.neutron.org")

@router.get("/api/amber/positions")
async def query_amber_contract_positions(address: str):
    """Return all Amber positions owned by the given wallet address."""
    # Build the smart-query `{ "positions": { "owner": <address> } }`
    query_object = {"positions": {"owner": address}}
    query_b64    = base64.b64encode(json.dumps(query_object).encode()).decode()

    url = f"{LCD_ENDPOINT}/cosmwasm/wasm/v1/contract/{AMBER_CONTRACT}/smart/{query_b64}"

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()  # Forward Amber’s JSON response verbatim


# step:3 file: execute_an_emergency_withdrawal_for_the_user’s_amber_trading_position
# backend/routes/tx.py
from fastapi import APIRouter, HTTPException
from google.protobuf.json_format import MessageToDict
from cosmpy.aerial.client.lcd import LCDClient
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import Address
from cosmpy.aerial.provision import faucet
from cosmpy.protogen.cosmwasm.wasm.v1 import tx_pb2 as wasm_tx
import base64, os, json

router = APIRouter()

CHAIN_ID      = os.getenv("CHAIN_ID", "neutron-1")
RPC_ENDPOINT  = os.getenv("RPC_ENDPOINT", "https://rpc-kralum.neutron.org:443")
AMBER_CONTRACT = os.getenv("AMBER_CONTRACT", "neutron1ambercontract...")
FEE_DENOM      = os.getenv("FEE_DENOM", "untrn")
FEE_GAS        = int(os.getenv("FEE_GAS", "250000"))
FEE_AMOUNT     = os.getenv("FEE_AMOUNT", "5000")

lcd = LCDClient(url=RPC_ENDPOINT, chain_id=CHAIN_ID)

@router.post("/api/tx/prepare")
async def construct_tx_amber_emergency_withdraw(payload: dict):
    """
    Body example:
    {
      "sender": "neutron1...",
      "position_id": 42
    }
    """
    sender      = payload.get("sender")
    position_id = payload.get("position_id")
    if sender is None or position_id is None:
        raise HTTPException(status_code=400, detail="'sender' and 'position_id' are required")

    # 1. Build MsgExecuteContract
    msg = wasm_tx.MsgExecuteContract(
        sender   = sender,
        contract = AMBER_CONTRACT,
        msg      = json.dumps({"emergency_withdraw": {"position_id": int(position_id)}}).encode(),
        funds    = []  # No funds required
    )

    # 2. Ask the LCD for account_number / sequence
    account_info = lcd.auth.account_info(Address(sender))
    account_number = int(account_info.base_account.account_number)
    sequence       = int(account_info.base_account.sequence)

    # 3. Build the unsigned Tx
    tx = Transaction()
    tx.add_message(msg)
    tx.seal(
        gas_limit = FEE_GAS,
        fee_denom = FEE_DENOM,
        fee_amount = FEE_AMOUNT,
        memo = "Amber emergency withdraw"
    )

    # 4. Encode SignDoc fields for Keplr
    sign_doc = tx.get_sign_doc(chain_id=CHAIN_ID, account_number=account_number, sequence=sequence)

    response = {
        "bodyBytes":      base64.b64encode(sign_doc.body_bytes).decode(),
        "authInfoBytes":  base64.b64encode(sign_doc.auth_info_bytes).decode(),
        "chainId":        CHAIN_ID,
        "accountNumber":  str(account_number)
    }

    # Return everything the frontend needs to call `keplr.signDirect`
    return response


# step:5 file: execute_an_emergency_withdrawal_for_the_user’s_amber_trading_position
# backend/routes/broadcast.py
from fastapi import APIRouter, HTTPException
from cosmpy.protogen.cosmos.tx.v1beta1 import tx_pb2
from cosmpy.aerial.client.lcd import LCDClient
import base64, os, json, httpx

router = APIRouter()

CHAIN_ID     = os.getenv("CHAIN_ID", "neutron-1")
RPC_ENDPOINT = os.getenv("RPC_ENDPOINT", "https://rpc-kralum.neutron.org:443")

lcd = LCDClient(url=RPC_ENDPOINT, chain_id=CHAIN_ID)

@router.post("/api/tx/broadcast")
async def broadcast_signed_tx(payload: dict):
    """
    Expected body:
    {
      "bodyBytes":     "...base64...",
      "authInfoBytes": "...base64...",
      "signature":     "...base64..."
    }
    """
    try:
        body_bytes     = base64.b64decode(payload["bodyBytes"])
        auth_info      = base64.b64decode(payload["authInfoBytes"])
        signature_raw  = base64.b64decode(payload["signature"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 in request: {e}")

    # Assemble TxRaw manually
    tx_raw = tx_pb2.TxRaw(
        body_bytes          = body_bytes,
        auth_info_bytes     = auth_info,
        signatures          = [signature_raw]
    )

    # Broadcast (sync mode so we get a txhash immediately)
    tx_response = lcd.tx.broadcast_tx(tx_raw.SerializeToString(), broadcast_mode="sync")

    if tx_response.tx_response.code != 0:
        raise HTTPException(status_code=500, detail=tx_response.tx_response.raw_log)

    return {"txhash": tx_response.tx_response.txhash}


# step:1 file: stop_the_current_cpu_profile
import os
import signal
import subprocess
from typing import Optional


def graceful_shutdown(pid: Optional[int] = None, *, use_systemctl: bool = False, service_name: Optional[str] = None) -> None:
    """Send a graceful shutdown signal.

    Args:
        pid (int, optional): The PID of the running process. Required if `use_systemctl` is False.
        use_systemctl (bool): Whether to stop a systemd service instead of sending SIGINT.
        service_name (str, optional): Name of the systemd service. Required if `use_systemctl` is True.

    Raises:
        ValueError: If required arguments are missing.
        RuntimeError: If the shutdown command fails.
    """
    if use_systemctl:
        if not service_name:
            raise ValueError("'service_name' is required when 'use_systemctl' is True")
        try:
            result = subprocess.run(["systemctl", "stop", service_name], check=True, capture_output=True, text=True)
            print(result.stdout.strip())
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"Failed to stop service '{service_name}': {err.stderr.strip()}") from err
    else:
        if pid is None:
            raise ValueError("'pid' is required when 'use_systemctl' is False")
        try:
            os.kill(pid, signal.SIGINT)
            print(f"Sent SIGINT to PID {pid} …")
        except ProcessLookupError:
            raise RuntimeError(f"Process with PID {pid} does not exist.")
        except PermissionError:
            raise RuntimeError(f"Permission denied to signal PID {pid}.")


# step:2 file: stop_the_current_cpu_profile
import os
import time
from pathlib import Path
from typing import Optional


def wait_for_exit(pid: int, *, log_path: Optional[str] = None, timeout: int = 120, poll_interval: float = 1.0) -> None:
    """Wait for a process to exit and confirm profiler stop message.

    Args:
        pid (int): The PID to monitor.
        log_path (str, optional): Path to the application log file.
        timeout (int): Maximum seconds to wait before giving up.
        poll_interval (float): Seconds between checks.

    Raises:
        TimeoutError: If the process is still running after the timeout period.
    """
    start_time = time.time()
    profiler_msg_found = False

    log_file: Optional[Path] = Path(log_path) if log_path else None
    log_offset = log_file.stat().st_size if log_file and log_file.exists() else 0

    while True:
        # Check if the process has exited
        if not os.path.exists(f"/proc/{pid}"):
            print(f"PID {pid} has exited.")
            break

        # Optionally scan new log output for profiler message
        if log_file and log_file.exists():
            with log_file.open("r") as lf:
                lf.seek(log_offset)
                for line in lf:
                    if "stopping CPU profiler" in line:
                        profiler_msg_found = True
                        print("Detected 'stopping CPU profiler' in logs.")
                        break
                log_offset = lf.tell()

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Process {pid} did not exit within {timeout}s.")

        time.sleep(poll_interval)

    if log_file and not profiler_msg_found:
        print("Warning: 'stopping CPU profiler' not found in logs. The node may not have flushed the profile correctly.")


# step:3 file: stop_the_current_cpu_profile
from pathlib import Path


def verify_profile_file(profile_path: str = "cpu.prof") -> int:
    """Validate the profiler output file.

    Args:
        profile_path (str): Location of the profile file.

    Returns:
        int: Size of the profile file in bytes.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the file is empty.
    """
    file_path = Path(profile_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    size = file_path.stat().st_size
    if size == 0:
        raise ValueError(f"Profile file is empty: {profile_path}")

    print(f"Profile file '{profile_path}' verified with size {size} bytes.")
    return size


# step:2 file: compute_a_create2_contract_address_(cast_compute-address)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from eth_utils import keccak, to_checksum_address

app = FastAPI()

class Create2Request(BaseModel):
    deployer: str = Field(..., description="0x-prefixed deployer (factory) address")
    salt: str = Field(..., description="32-byte hex string WITHOUT 0x prefix")
    init_code_hash: str = Field(..., description="0x-prefixed keccak256 hash of the contract's init code")

def _strip_0x(hex_str: str) -> str:
    """Remove 0x prefix, if present."""
    return hex_str[2:] if hex_str.startswith("0x") else hex_str

def _hex_to_bytes(hex_str: str, expected_len: int) -> bytes:
    """Convert hex to bytes, validating exact length."""
    hex_str = _strip_0x(hex_str)
    if len(hex_str) != expected_len:
        raise ValueError(f"Expected {expected_len} hex chars, got {len(hex_str)}.")
    return bytes.fromhex(hex_str)

@app.post("/api/compute_create2_address")
async def compute_create2_address(req: Create2Request):
    try:
        deployer_bytes = _hex_to_bytes(req.deployer, 40)   # 20 bytes
        salt_bytes = _hex_to_bytes(req.salt, 64)           # 32 bytes
        init_code_hash_bytes = _hex_to_bytes(req.init_code_hash, 64)  # 32 bytes
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    # CREATE2 formula: keccak256(0xff ++ deployer ++ salt ++ initCodeHash)[12:]
    data = b"\xff" + deployer_bytes + salt_bytes + init_code_hash_bytes
    derived = keccak(data)[12:]  # Take the right-most 20 bytes
    checksum_addr = to_checksum_address("0x" + derived.hex())

    return {"create2_address": checksum_addr}


# step:2 file: fetch_a_transaction_receipt_with_cast_receipt
"""
FastAPI backend endpoint to fetch an Ethereum-style transaction receipt from a Cosmos EVM chain.
"""

import os
import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Configure your chain RPC in an environment variable, e.g.:
#   export EVM_RPC="https://rpc.evmos.org:8545"
RPC_ENDPOINT = os.getenv("EVM_RPC", "https://rpc.evmos.org:8545")

async def fetch_receipt(tx_hash: str) -> dict | None:
    """Low-level helper that wraps the JSON-RPC request."""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getTransactionReceipt",
        "params": [tx_hash],
        "id": 1,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(RPC_ENDPOINT, json=payload, timeout=10)
            resp.raise_for_status()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"RPC connection error: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"RPC returned {e.response.status_code}")

    body = resp.json()

    # Handle RPC-level errors
    if body.get("error"):
        raise HTTPException(status_code=500, detail=body["error"])

    return body.get("result")

@app.get("/api/tx_receipt")
async def tx_receipt(tx_hash: str):
    """REST: GET /api/tx_receipt?tx_hash=0x…"""
    receipt = await fetch_receipt(tx_hash)
    if receipt is None:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return receipt


# step:3 file: fetch_a_transaction_receipt_with_cast_receipt
"""
Helpers for decoding and prettifying a raw EVM transaction receipt.
"""

from fastapi import HTTPException

# Re-use fetch_receipt from Step 2 (assumed to be in scope)

def _human_readable_status(status_hex: str) -> str:
    """Convert 0x0 / 0x1 ⇒ Failed / Success."""
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

@app.get("/api/decode_receipt")
async def decode_receipt_endpoint(tx_hash: str):
    """REST: GET /api/decode_receipt?tx_hash=0x…"""
    receipt = await fetch_receipt(tx_hash)
    if receipt is None:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return decode_receipt(receipt)


# step:1 file: list_all_grpc_services_exposed_on_localhost:9090_using_grpcurl
from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.get("/api/grpcurl/verify")
async def verify_grpcurl_installation():
    """Verify that `grpcurl` is installed on the host machine."""
    try:
        # Attempt to execute `grpcurl -help` to confirm installation
        completed_process = subprocess.run(
            ["grpcurl", "-help"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Return the first 500 characters of stdout to avoid overly large payloads
        return {
            "installed": True,
            "output": completed_process.stdout[:500],
        }
    except FileNotFoundError:
        # grpcurl binary not found in PATH
        raise HTTPException(
            status_code=404,
            detail="grpcurl is not installed or not found in PATH.",
        )
    except subprocess.CalledProcessError as exc:
        # grpcurl returned a non-zero exit code
        raise HTTPException(
            status_code=500,
            detail=f"grpcurl -help failed: {exc.stderr}",
        )


# step:2 file: list_all_grpc_services_exposed_on_localhost:9090_using_grpcurl
from typing import Optional
from fastapi import Query

@app.get("/api/grpcurl/list")
async def grpcurl_list_services(
    host: str = Query("localhost:9090", description="gRPC server host:port"),
    plaintext: bool = Query(True, description="Use plaintext (disable TLS)."),
    insecure: bool = Query(False, description="Allow insecure TLS without trusted certs."),
):
    """List gRPC services exposed by the server via reflection."""
    # Build grpcurl command
    cmd = ["grpcurl"]
    if plaintext:
        cmd.append("-plaintext")
    if insecure:
        cmd.append("-insecure")
    cmd.extend([host, "list"])

    try:
        completed_process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        services = [line.strip() for line in completed_process.stdout.splitlines() if line.strip()]
        return {"host": host, "services": services}
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"grpcurl list failed: {exc.stderr}",
        )


# step:1 file: verify_a_smart_contract_on_an_etherscan-compatible_explorer_using_hardhat
import subprocess
import sys


def install_hardhat_etherscan_plugin(project_path: str = '.'):
    """
    Installs @nomiclabs/hardhat-etherscan as a dev-dependency inside the given project directory.
    """
    try:
        result = subprocess.run(
            ['npm', 'install', '--save-dev', '@nomiclabs/hardhat-etherscan'],
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(result.stdout)
        return {'status': 'success', 'message': 'Plugin installed successfully'}
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)
        return {'status': 'error', 'message': e.stderr}


# step:2 file: verify_a_smart_contract_on_an_etherscan-compatible_explorer_using_hardhat
import os


def configure_etherscan_api_key(key: str, env_path: str = '.env'):
    """
    Persists ETHERSCAN_API_KEY to .env, replacing any existing value.
    """
    try:
        lines = []
        if os.path.isfile(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
        # Remove stale entries
        lines = [l for l in lines if not l.startswith('ETHERSCAN_API_KEY=')]
        lines.append(f'ETHERSCAN_API_KEY={key}\n')
        with open(env_path, 'w') as f:
            f.writelines(lines)
        return {'status': 'success', 'message': 'ETHERSCAN_API_KEY written to .env'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# step:3 file: verify_a_smart_contract_on_an_etherscan-compatible_explorer_using_hardhat
import subprocess
import sys


def compile_contracts(project_path: str = '.'):
    """
    Executes `npx hardhat compile` inside the project directory.
    """
    try:
        result = subprocess.run(
            ['npx', 'hardhat', 'compile'],
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return {'status': 'success', 'output': result.stdout}
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)
        return {'status': 'error', 'message': e.stderr}


# step:4 file: verify_a_smart_contract_on_an_etherscan-compatible_explorer_using_hardhat
import subprocess
import sys
from typing import List


def verify_contract(network: str, address: str, constructor_args: List[str] | None = None, project_path: str = '.'):
    """
    Launches Hardhat verification for a deployed contract.
    """
    try:
        cmd = ['npx', 'hardhat', 'verify', '--network', network, address]
        if constructor_args:
            cmd.extend(constructor_args)
        result = subprocess.run(
            cmd,
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return {'status': 'success', 'output': result.stdout}
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)
        return {'status': 'error', 'message': e.stderr}


# step:1 file: upload_the_compiled_wasm_file_artifacts_contract_name.wasm_to_the_juno_chain
import os
import hashlib


def verify_wasm_artifact(contract_name: str, artifacts_dir: str = 'artifacts', expected_sha256: str | None = None) -> dict:
    """Verify local CosmWasm binary integrity.

    Parameters
    ----------
    contract_name : str
        Name of the contract (without `.wasm`).
    artifacts_dir : str, optional
        Relative path where build artifacts reside.
    expected_sha256 : str | None, optional
        If provided, the function will compare the on-disk hash to this value.

    Returns
    -------
    dict
        { 'file_path': str, 'sha256': str }
    """
    file_path = os.path.join(artifacts_dir, f'{contract_name}.wasm')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'WASM artifact not found at {file_path}')

    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()

    if expected_sha256 and file_hash.lower() != expected_sha256.lower():
        raise ValueError('SHA-256 hash mismatch: artifact may not be trusted.')

    return {"file_path": file_path, "sha256": file_hash}


# step:2 file: upload_the_compiled_wasm_file_artifacts_contract_name.wasm_to_the_juno_chain
import json
import subprocess
import shlex


def store_wasm(contract_name: str, key_name: str, chain_id: str, gas_adjustment: float = 1.3) -> dict:
    """Call `junod tx wasm store` and return the tx hash."""
    wasm_path = f'artifacts/{contract_name}.wasm'
    cmd = (
        f'junod tx wasm store {wasm_path} '
        f'--from {key_name} --chain-id {chain_id} '
        f'--gas auto --gas-adjustment {gas_adjustment} -y -o json'
    )

    try:
        output = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
        cli_response = json.loads(output.decode())
        tx_hash = cli_response.get('txhash') or cli_response.get('tx_response', {}).get('txhash')
        if not tx_hash:
            raise RuntimeError('Could not parse tx hash from CLI output.')
        return {"tx_hash": tx_hash}
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'junod CLI error: {e.output.decode()}') from e


# step:3 file: upload_the_compiled_wasm_file_artifacts_contract_name.wasm_to_the_juno_chain
import time
import requests


def wait_for_tx_commit(tx_hash: str, rest_endpoint: str, timeout: int = 180, poll_interval: int = 6) -> dict:
    """Block until the transaction lands or raise TimeoutError."""
    url = f"{rest_endpoint}/cosmos/tx/v1beta1/txs/{tx_hash}"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('tx_response', {}).get('height', '0') != '0':
                return data
        time.sleep(poll_interval)
    raise TimeoutError(f'Transaction {tx_hash} not found within {timeout} seconds')


# step:4 file: upload_the_compiled_wasm_file_artifacts_contract_name.wasm_to_the_juno_chain
import requests


def fetch_code_list(rest_endpoint: str) -> dict:
    """Return array of all CosmWasm code infos from the chain."""
    url = f"{rest_endpoint}/cosmwasm/wasm/v1/code"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


# step:4 file: lock_an_additional_500_ntrn_for_24_months_(boost)
from fastapi import FastAPI, HTTPException, Body
import os, json
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgExecuteContract

app = FastAPI()

CHAIN_ID = 'neutron-1'
RPC_ENDPOINT = os.getenv('NEUTRON_RPC', 'https://rpc-kralum.neutron.org:443')
BOOST_CONTRACT_ADDRESS = os.getenv('BOOST_CONTRACT_ADDR', 'neutron1boostcontractaddress…')  # TODO: set real address


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


@app.post('/api/boost/lock')
async def sign_and_broadcast(payload: dict = Body(...)):
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


# step:5 file: lock_an_additional_500_ntrn_for_24_months_(boost)
from fastapi import FastAPI, HTTPException
from cosmpy.aerial.client import LedgerClient

app = FastAPI()

@app.get('/api/tx_status/{tx_hash}')
async def tx_status(tx_hash: str):
    client = LedgerClient(NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT))
    try:
        tx_response = client.query_tx(tx_hash)
        if not tx_response:
            return { 'status': 'PENDING' }
        return { 'status': 'COMMITTED', 'height': tx_response.height }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:1 file: enable_block-profiling_with_rate_1_on_a_cosmos-evm_node
import os


def select_rpc_endpoint() -> str:
    # Returns the JSON-RPC endpoint configured in the environment.
    rpc_url = os.getenv("RPC_URL")
    if not rpc_url:
        raise EnvironmentError("RPC_URL environment variable is not set.")
    return rpc_url



# step:2 file: enable_block-profiling_with_rate_1_on_a_cosmos-evm_node
import os
import base64
from typing import Dict


def authenticate_debug_namespace() -> Dict[str, str]:
    # Builds Basic-Auth headers for the private debug RPC namespace.
    user = os.getenv("RPC_USER")
    password = os.getenv("RPC_PASSWORD")
    if user is None or password is None:
        raise EnvironmentError("RPC_USER or RPC_PASSWORD environment variables are missing.")
    token_bytes = f"{user}:{password}".encode()
    token_b64 = base64.b64encode(token_bytes).decode()
    return {"Authorization": f"Basic {token_b64}"}



# step:3 file: enable_block-profiling_with_rate_1_on_a_cosmos-evm_node
import os
import base64
import requests
from typing import Any, Dict


def select_rpc_endpoint() -> str:
    # Returns the JSON-RPC endpoint configured in the environment.
    rpc_url = os.getenv("RPC_URL")
    if not rpc_url:
        raise EnvironmentError("RPC_URL environment variable is not set.")
    return rpc_url


def authenticate_debug_namespace() -> Dict[str, str]:
    # Builds Basic-Auth headers for the private debug RPC namespace.
    user = os.getenv("RPC_USER")
    password = os.getenv("RPC_PASSWORD")
    if user is None or password is None:
        raise EnvironmentError("RPC_USER or RPC_PASSWORD environment variables are missing.")
    token_bytes = f"{user}:{password}".encode()
    token_b64 = base64.b64encode(token_bytes).decode()
    return {"Authorization": f"Basic {token_b64}"}


def debug_set_block_profile_rate(rate: int = 1, request_id: int = 1) -> Any:
    """Send a JSON-RPC call to debug_setBlockProfileRate."""
    rpc_url = select_rpc_endpoint()
    headers = authenticate_debug_namespace()

    payload = {
        "jsonrpc": "2.0",
        "method": "debug_setBlockProfileRate",
        "params": [rate],
        "id": request_id
    }

    try:
        res = requests.post(rpc_url, json=payload, headers=headers, timeout=10)
        res.raise_for_status()
        res_json = res.json()
    except requests.RequestException as e:
        raise ConnectionError(f"Unable to reach {rpc_url}: {e}") from e

    if "error" in res_json:
        raise RuntimeError(f"RPC responded with error: {res_json['error']}")

    return res_json.get("result")



# step:3 file: lend_2_unibtc_on_amber_finance
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64, json

app = FastAPI()

class ApproveBody(BaseModel):
    sender: str            # User address
    cw20_contract: str     # uniBTC CW20 contract address
    spender: str           # Amber Finance lending contract address
    amount: int            # Amount in micro-units (e.g. 2_000_000 for 6-decimals)

@app.post('/api/amber/approve/construct')
async def construct_cw20_approve(body: ApproveBody):
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


# step:4 file: lend_2_unibtc_on_amber_finance
from fastapi import HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.tx import Transaction

# Network configuration (adjust RPC endpoint if necessary)
NETWORK = NetworkConfig(
    chain_id='neutron-1',
    url='https://rpc-kralum.neutron.org',
    fee_minimum_gas_price='0.025untrn'
)

class BroadcastBody(BaseModel):
    mnemonic: str          # Supplied securely by the frontend (never log!)
    msg: dict              # MsgExecuteContract produced in Step 3

@app.post('/api/amber/approve/broadcast')
async def broadcast_approve(body: BroadcastBody):
    try:
        wallet = PrivateKey.from_mnemonic(body.mnemonic)
        sender = wallet.public_key.address()

        tx = Transaction()
        tx.add_message(body.msg)            # Convert dict→proto inside cosmpy in real code

        client = LedgerClient(NETWORK)
        tx.with_sequence(client.get_sequence(sender))
        tx.with_account_number(client.get_number(sender))
        tx.with_chain_id(NETWORK.chain_id)
        tx.sign(wallet)

        result = client.broadcast_tx(tx)
        return result                      # JSON tx response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:5 file: lend_2_unibtc_on_amber_finance
from fastapi import HTTPException
from pydantic import BaseModel
from base64 import b64encode
import json

class LendBody(BaseModel):
    sender: str
    cw20_contract: str   # uniBTC contract address
    amber_pool: str      # Amber Finance pool contract address
    amount: int          # micro-unit amount (2 BTC = 2_000_000 if 6 decimals)

@app.post('/api/amber/lend/construct')
async def construct_lend(body: LendBody):
    try:
        # Optional inner payload for the lending pool (often empty)
        inner_msg = {}

        wrapped_send = {
            'send': {
                'contract': body.amber_pool,
                'amount': str(body.amount),
                'msg': b64encode(json.dumps(inner_msg).encode()).decode()
            }
        }

        encoded = b64encode(json.dumps(wrapped_send).encode()).decode()
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


# step:6 file: lend_2_unibtc_on_amber_finance
from fastapi import HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.tx import Transaction

# Re-use the NETWORK object from Step 4

class LendBroadcastBody(BaseModel):
    mnemonic: str
    msg: dict

@app.post('/api/amber/lend/broadcast')
async def broadcast_lend(body: LendBroadcastBody):
    try:
        wallet = PrivateKey.from_mnemonic(body.mnemonic)
        sender = wallet.public_key.address()
        client = LedgerClient(NETWORK)

        tx = Transaction()
        tx.add_message(body.msg)
        tx.with_sequence(client.get_sequence(sender))
        tx.with_account_number(client.get_number(sender))
        tx.with_chain_id(NETWORK.chain_id)
        tx.sign(wallet)

        return client.broadcast_tx(tx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:2 file: cancel_(unlock)_the_user’s_ntrn_stake_lock_once_the_vesting_period_has_ended
# api/lock_status.py
import os
import base64
import json
from typing import Dict
from fastapi import APIRouter, HTTPException
import httpx

router = APIRouter()

LCD_ENDPOINT = os.getenv("LCD_ENDPOINT", "https://rest-kralum.neutron-1.neutron.org")
LOCK_CONTRACT_ADDR = os.getenv("LOCK_CONTRACT_ADDR", "neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

@router.get("/lock_status")
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
        url = f"{LCD_ENDPOINT}/wasm/v1/contract/{LOCK_CONTRACT_ADDR}/smart/{query_b64}"

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


# step:1 file: configure_the_gaia_cli_to_use_the_“pass”_keyring_backend
from fastapi import APIRouter, HTTPException
import shutil
import subprocess

router = APIRouter()

@router.get("/api/check_dependency")
async def check_dependency():
    """Return installation status and version of the `pass` utility."""
    # Locate the binary first
    pass_path = shutil.which("pass")
    if pass_path is None:
        raise HTTPException(status_code=404, detail="`pass` utility is not installed on this host.")

    # Obtain version information
    try:
        version_output = subprocess.check_output(["pass", "--version"], text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to execute 'pass --version': {exc}")

    return {"installed": True, "path": pass_path, "version": version_output}


# step:2 file: configure_the_gaia_cli_to_use_the_“pass”_keyring_backend
from fastapi import APIRouter, HTTPException
import subprocess
import shlex

router = APIRouter()

@router.post("/api/config_keyring_pass")
async def config_keyring_backend():
    """Run `gaiad config keyring-backend pass` and return the CLI response."""
    cmd = shlex.split("gaiad config keyring-backend pass")
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"gaiad exited with code {exc.returncode}: {exc.output}")

    return {"success": True, "output": output.strip()}


# step:3 file: configure_the_gaia_cli_to_use_the_“pass”_keyring_backend
from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter()

@router.get("/api/validate_keyring_pass")
async def validate_keyring_setting():
    """Confirm that client.toml contains the correct keyring-backend setting."""
    client_toml_path = Path.home() / ".gaia" / "config" / "client.toml"
    if not client_toml_path.exists():
        raise HTTPException(status_code=404, detail=f"{client_toml_path} not found. Make sure Gaia is initialized.")

    try:
        content = client_toml_path.read_text()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read {client_toml_path}: {exc}")

    expected_line = 'keyring-backend = "pass"'
    valid = expected_line in content
    return {
        "file": str(client_toml_path),
        "contains_setting": valid,
        "expected_line": expected_line
    }


# step:3 file: fetch_the_transaction_receipt_for_a_given_hash
import httpx
from typing import Optional

async def get_transaction_receipt_once(tx_hash: str, rpc_endpoint: str) -> Optional[dict]:
    """Fire exactly one eth_getTransactionReceipt call."""

    payload = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'eth_getTransactionReceipt',
        'params': [tx_hash],
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(rpc_endpoint, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

    # `result` is None while the tx is pending
    return data.get('result')


# step:4 file: fetch_the_transaction_receipt_for_a_given_hash
import asyncio
import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI()

DEFAULT_RPC = os.getenv('DEFAULT_RPC_ENDPOINT', 'https://eth.bd.evmos.dev:8545/')
DEFAULT_TIMEOUT = int(os.getenv('TX_RECEIPT_TIMEOUT', 60))   # seconds
DEFAULT_INTERVAL = int(os.getenv('TX_RECEIPT_INTERVAL', 5))  # seconds


async def _get_receipt_once(tx_hash: str, rpc_endpoint: str):
    """Light wrapper around the function from Step 3 so we keep the file self-contained."""
    payload = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'eth_getTransactionReceipt',
        'params': [tx_hash],
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(rpc_endpoint, json=payload, timeout=10)
        res.raise_for_status()
        return res.json().get('result')


async def poll_for_receipt(tx_hash: str, rpc_endpoint: str, timeout: int, interval: int):
    """Repeat JSON-RPC calls until we get a non-null receipt or hit the timeout."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        receipt = await _get_receipt_once(tx_hash, rpc_endpoint)
        if receipt is not None:
            return receipt
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f'Receipt not found within {timeout}s')
        await asyncio.sleep(interval)


@app.get('/api/poll_receipt')
async def api_poll_receipt(
    tx_hash: str,
    rpc_endpoint: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    interval: int = DEFAULT_INTERVAL,
):
    """Frontend-facing endpoint: /api/poll_receipt?tx_hash=<hash>&rpc_endpoint=<opt>"""

    if not (tx_hash.startswith('0x') and len(tx_hash) == 66):
        raise HTTPException(status_code=400, detail='Invalid transaction hash')

    endpoint = rpc_endpoint or DEFAULT_RPC

    try:
        receipt = await poll_for_receipt(tx_hash, endpoint, timeout, interval)
        return {'receipt': receipt}
    except TimeoutError as err:
        raise HTTPException(status_code=504, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:2 file: broadcast_a_msgsend_transaction_(bank_module)_via_cli
from fastapi import FastAPI, HTTPException
from bech32 import bech32_decode

app = FastAPI()


def is_valid_bech32(address: str, expected_prefix: str) -> bool:
    """Return True if `address` is a valid bech32 string with the given prefix."""
    hrp, data = bech32_decode(address)
    return hrp == expected_prefix and data is not None


@app.get('/api/validate_recipient')
async def validate_recipient(address: str, prefix: str = 'cosmos'):
    if not is_valid_bech32(address, prefix):
        raise HTTPException(status_code=400, detail='Invalid bech32 address')
    return {'valid': True}


# step:3 file: broadcast_a_msgsend_transaction_(bank_module)_via_cli
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.protos.cosmos.bank.v1beta1 import tx_pb2 as bank_tx_pb2

app = FastAPI()

class MsgSendRequest(BaseModel):
    sender: str
    recipient: str
    amount: int          # integer amount in base denom (e.g. uatom)
    denom: str = 'uatom'


@app.post('/api/construct_msg_send')
async def construct_msg_send(req: MsgSendRequest):
    try:
        msg = bank_tx_pb2.MsgSend(
            from_address=req.sender,
            to_address=req.recipient,
            amount=[{'amount': str(req.amount), 'denom': req.denom}]
        )
        return {'proto_hex': msg.SerializeToString().hex()}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:4 file: broadcast_a_msgsend_transaction_(bank_module)_via_cli
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import PrivateKey

app = FastAPI()

NETWORK = NetworkConfig(
    chain_id=os.getenv('CHAIN_ID', 'cosmoshub-4'),
    url=os.getenv('RPC_ENDPOINT', 'https://rpc.cosmos.network:443')
)
client = LedgerClient(NETWORK)

MNEMONIC = os.getenv('SENDER_MNEMONIC')  # NEVER commit this to source control in production!
if not MNEMONIC:
    raise RuntimeError('SENDER_MNEMONIC environment variable is required.')

class BroadcastRequest(BaseModel):
    proto_hex: str
    gas: int = 200000
    fee_amount: int = 1000
    fee_denom: str = 'uatom'


@app.post('/api/sign_and_broadcast')
async def sign_and_broadcast(req: BroadcastRequest):
    try:
        wallet = PrivateKey.from_mnemonic(MNEMONIC)

        tx = Transaction()
        tx.add_raw_message(bytes.fromhex(req.proto_hex))
        tx.set_fee(f"{req.fee_amount}{req.fee_denom}")
        tx.set_gas(req.gas)
        tx.seal(client, wallet)
        tx.sign(wallet)
        tx.complete()

        result = client.broadcast_tx(tx)
        if result.is_err():
            raise ValueError(result.raw_log)
        return { 'tx_hash': result.tx_hash }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:5 file: broadcast_a_msgsend_transaction_(bank_module)_via_cli
from fastapi import FastAPI, HTTPException
from cosmpy.aerial.client import LedgerClient, NetworkConfig
import os, asyncio

app = FastAPI()

NETWORK = NetworkConfig(
    chain_id=os.getenv('CHAIN_ID', 'cosmoshub-4'),
    url=os.getenv('RPC_ENDPOINT', 'https://rpc.cosmos.network:443')
)
client = LedgerClient(NETWORK)


@app.get('/api/check_tx')
async def check_tx(tx_hash: str, timeout: int = 30):
    """Return the tx result as soon as it is committed or raise 404 after `timeout` seconds."""
    try:
        for _ in range(timeout):
            try:
                return client.tx_by_hash(tx_hash)
            except Exception:
                await asyncio.sleep(1)
        raise HTTPException(status_code=404, detail='Transaction not found within timeout window.')
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:1 file: get_details_of_the_latest_block_with_cast_block_latest
import os
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

# Read the RPC URL from an environment variable or fallback to local default
COSMOS_RPC_URL = os.getenv("COSMOS_RPC_URL", "http://localhost:26657")

@app.get("/api/latest_block")
async def get_latest_block():
    """Fetch the latest block from the configured Cosmos RPC node."""
    try:
        rpc_endpoint = f"{COSMOS_RPC_URL.rstrip('/')}/block"  # no height param = latest
        response = requests.get(rpc_endpoint, timeout=10)
        response.raise_for_status()
        return response.json()  # CometBFT returns `{ "result": { "block": {...} } }`
    except requests.exceptions.RequestException as err:
        raise HTTPException(status_code=502, detail=f"Failed to fetch latest block: {err}")


# step:1 file: compile_all_workspace_contracts_with_workspace-optimizer
import subprocess
import os
import sys


def run_workspace_optimizer(repo_root: str = ".") -> None:
    """
    Run the CosmWasm workspace-optimizer Docker image to compile optimized
    Wasm binaries for every smart contract found in the Cargo workspace.

    Args:
        repo_root: Path to the repository root (defaults to current dir).
    Raises:
        FileNotFoundError: If Docker is not installed.
        subprocess.CalledProcessError: If the Docker command exits non-zero.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-v",
        f"{os.path.abspath(repo_root)}:/code",
        "--mount",
        "type=volume,source=registry_cache,target=/usr/local/cargo/registry",
        "cosmwasm/workspace-optimizer:0.12.11",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅  Workspace optimizer completed successfully.")
    except FileNotFoundError:
        print("🛑  Docker is not installed or could not be found in $PATH.", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        print(f"🛑  Workspace optimizer failed with exit code {e.returncode}.", file=sys.stderr)
        raise


if __name__ == "__main__":
    # Allow CLI execution: `python run_workspace_optimizer.py` from repo root
    run_workspace_optimizer()



# step:2 file: compile_all_workspace_contracts_with_workspace-optimizer
import os
import sys
import glob
from typing import List

try:
    import toml  # Light-weight parser for Cargo.toml
except ImportError:
    print("The 'toml' package is required. Install it with `pip install toml`.", file=sys.stderr)
    raise


def _get_workspace_members(repo_root: str = ".") -> List[str]:
    """Return the list of workspace member directories from Cargo.toml."""
    cargo_path = os.path.join(repo_root, "Cargo.toml")
    if not os.path.exists(cargo_path):
        raise FileNotFoundError("Cargo.toml not found at repository root.")
    data = toml.load(cargo_path)
    return data.get("workspace", {}).get("members", [])


def verify_workspace_artifacts(repo_root: str = ".") -> None:
    """
    Ensure that `artifacts/` contains one optimized `.wasm` file for every
    contract in the workspace.
    """
    members = _get_workspace_members(repo_root)
    artifact_dir = os.path.join(repo_root, "artifacts")

    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError("artifacts/ directory not found. Did you run the optimizer?")

    wasm_files = glob.glob(os.path.join(artifact_dir, "*.wasm"))
    wasm_basenames = {os.path.splitext(os.path.basename(p))[0] for p in wasm_files}

    missing_contracts = []
    for member in members:
        contract_name = os.path.basename(member)
        if contract_name not in wasm_basenames:
            missing_contracts.append(contract_name)

    if missing_contracts:
        raise RuntimeError(
            "Missing optimized contracts: " + ", ".join(missing_contracts)
        )

    print("✅  All workspace contracts have corresponding optimized Wasm artifacts.")


if __name__ == "__main__":
    # Allow CLI execution: `python verify_workspace_artifacts.py`
    verify_workspace_artifacts()



# step:5 file: open_a_5×_leveraged_loop_position_with_1_maxbtc_on_amber
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import Wallet
from cosmpy.aerial.tx import Transaction, SigningCfg

AMBER_CONTRACT = 'neutron1ambercontractaddressxxxxxxxxxxxx'
RPC_ENDPOINT = 'https://rpc-neutron.keplr.app'
CHAIN_ID = 'neutron-1'

app = FastAPI()

class OpenPositionRequest(BaseModel):
    mnemonic: str                       # !! Only for demo purposes !!
    open_position_msg: dict             # MsgExecuteContract generated in Step 4
    gas_limit: int = 250000             # conservative default
    gas_price: float = 0.025            # NTRN per gas unit

@app.post('/api/open_position')
async def open_position(req: OpenPositionRequest):
    try:
        # 1. Build client & wallet
        net_cfg = NetworkConfig(
            chain_id=CHAIN_ID,
            url=RPC_ENDPOINT,
            fee_minimum_gas_price=req.gas_price,
            fee_denomination='untrn'
        )
        client = LedgerClient(net_cfg)
        wallet = Wallet(req.mnemonic)

        # 2. Craft the transaction
        tx = (
            Transaction()
            .with_messages(req.open_position_msg)
            .with_sequence(client.query_account_sequence(wallet.address()))
            .with_account_num(client.query_account_number(wallet.address()))
            .with_gas(req.gas_limit)
            .with_chain_id(net_cfg.chain_id)
        )

        # 3. Sign & broadcast
        signed_tx = wallet.sign(tx)
        tx_response = client.broadcast_tx_block(signed_tx)

        if tx_response.is_error:
            raise HTTPException(400, f'Broadcast failed: {tx_response.log}')
        return {"tx_hash": tx_response.tx_hash, "height": tx_response.height}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:1 file: add_transaction_sign-mode_textual_support
#!/usr/bin/env bash
# scripts/update_sdk.sh
# ---------------------------------------------
# Updates Cosmos-SDK to v0.47.x and runs `go mod tidy`.
# Exits on any failure and prints a helpful message.
# ---------------------------------------------
set -euo pipefail

SDK_VERSION="v0.47.5"   # <- Pin the exact version you wish to use

printf "\n🚀  Updating Cosmos-SDK to %s …\n" "$SDK_VERSION"

go get github.com/cosmos/cosmos-sdk@${SDK_VERSION}

echo "🔄  Tidying go.mod / go.sum …"
go mod tidy

echo "✅  go.mod successfully updated to Cosmos-SDK ${SDK_VERSION}"


# step:2 file: add_transaction_sign-mode_textual_support
// app/encoding.go
package app

import (
    "github.com/cosmos/cosmos-sdk/client"
    "github.com/cosmos/cosmos-sdk/codec"
    "github.com/cosmos/cosmos-sdk/codec/types"
    "github.com/cosmos/cosmos-sdk/simapp"
    sdk "github.com/cosmos/cosmos-sdk/types"
    tx "github.com/cosmos/cosmos-sdk/types/tx"
    "github.com/cosmos/cosmos-sdk/types/tx/signing"
)

// MakeEncodingConfig returns an EncodingConfig with TEXTUAL sign-mode enabled.
func MakeEncodingConfig() simapp.EncodingConfig {
    cfg := simapp.MakeTestEncodingConfig()

    // ---------------------------------------------------
    // Enable SIGN_MODE_TEXTUAL so wallets / CLI can use it
    // ---------------------------------------------------
    signing.RegisterSignModeHandler(
        signing.SignMode_SIGN_MODE_TEXTUAL,
        func() client.TxConfig { return cfg.TxConfig }, // provide underlying TxConfig
        nil, // no custom options needed for defaults
    )

    // (Optional) You can also restrict available sign-modes explicitly:
    // cfg.TxConfig = tx.NewTxConfig(cfg.Codec, tx.ConfigOptions{
    //     EnabledSignModes: []signing.SignMode{
    //         signing.SignMode_SIGN_MODE_TEXTUAL,
    //         signing.SignMode_SIGN_MODE_DIRECT,
    //     },
    // })

    return cfg
}


# step:3 file: add_transaction_sign-mode_textual_support
// cmd/root.go
package cmd

import (
    "github.com/spf13/cobra"
    "github.com/cosmos/cosmos-sdk/client/flags"
)

// NewRootCmd constructs the root command for your daemon.
func NewRootCmd() *cobra.Command {
    rootCmd := &cobra.Command{
        Use:   "appd",
        Short: "My Cosmos SDK Application",
    }

    // OPTIONAL: If you build custom tx sub-commands, be sure to add tx flags.
    rootCmd.PersistentFlags().String(flags.FlagSignMode, "", "Choose sign mode (e.g. 'textual', 'direct')")

    // If you import the default tx commands from the SDK they already carry this flag;
    // the above keeps it when you wrap or replace them.
    return rootCmd
}


# step:4 file: add_transaction_sign-mode_textual_support
#!/usr/bin/env bash
# scripts/build.sh
set -euo pipefail

BINARY_NAME="appd"

printf "\n🔧  Building %s …\n" "$BINARY_NAME"

go build -o "./build/${BINARY_NAME}" ./cmd/appd

echo "✅  Binary built at ./build/${BINARY_NAME}"


# step:5 file: add_transaction_sign-mode_textual_support
# Example shell command — replace values as needed
appd tx sign unsigned_tx.json \
    --from alice \
    --chain-id mychain-1 \
    --sign-mode textual \
    --output-document signed_tx.json

# The CLI will display a textual, human-readable representation automatically.
# You can also inspect the signed file:
cat signed_tx.json | jq '.'


# step:2 file: convert_an_ether_value_to_wei_(cast_to-wei)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import subprocess

app = FastAPI()

class AmountIn(BaseModel):
    amount: str

    @validator('amount')
    def validate_amount(cls, v):
        try:
            if float(v) < 0:
                raise ValueError
            return v
        except ValueError:
            raise ValueError('`amount` must be a positive number represented as a string')

@app.post('/api/to_wei')
async def to_wei(payload: AmountIn):
    """Convert an Ether amount to Wei using the `cast` CLI.
    Foundry’s `cast` must be installed and present in the PATH.
    """
    try:
        # Build and execute the CLI command safely
        cmd = ['cast', 'to-wei', payload.amount, 'ether']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        wei_value = result.stdout.strip()
        return { 'wei': wei_value }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'cast error: {e.stderr.strip()}')
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail='`cast` CLI not found on the server')


# step:2 file: trace_a_block_by_hash_with_debug_traceblockbyhash
############################################
# file: app.py (Flask backend for Mintlify) #
############################################
import os
import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/traceBlockByHash', methods=['POST'])
def api_trace_block_by_hash():
    """Proxy for EVM `debug_traceBlockByHash`."""
    data = request.get_json(force=True)
    rpc_url: str | None = data.get('rpc_url')
    block_hash: str | None = data.get('block_hash')

    # -------------------------
    # Basic input validation
    # -------------------------
    if not rpc_url or not rpc_url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Valid rpc_url is required'}), 400

    if not block_hash or not block_hash.startswith('0x') or len(block_hash) != 66:
        return jsonify({'error': 'block_hash must be a 0x-prefixed 32-byte hash'}), 400

    payload = {
        'method': 'debug_traceBlockByHash',
        'params': [block_hash, {}],  # Empty tracer object → default parity-style trace
        'id': 1,
        'jsonrpc': '2.0'
    }

    try:
        # Forward the request to the given RPC endpoint
        res = requests.post(rpc_url, json=payload, timeout=20)
        res.raise_for_status()
        body = res.json()

        # Check for JSON-RPC error objects
        if 'error' in body:
            return jsonify({'error': body['error']}), 502  # Bad gateway

        return jsonify({'trace': body['result']}), 200

    except requests.exceptions.RequestException as err:
        # Network-level or HTTP-level error
        return jsonify({'error': str(err)}), 502


if __name__ == '__main__':
    # Bind on 0.0.0.0 so Mintlify can expose the port
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)))


# step:1 file: import_a_raw_private_key_into_the_cosmos_keyring
import tempfile
import os
import stat


def save_privkey_to_file(privkey_hex: str) -> str:
    """
    Persist a hex-encoded private key in a temporary file that is readable only by
    the current user. The function returns the absolute file path so that it can
    be consumed by the next step.
    """
    # Basic validation — ensure every character is hexadecimal
    if not privkey_hex or any(c not in "0123456789abcdefABCDEF" for c in privkey_hex.strip()):
        raise ValueError("Invalid hex-encoded private key provided.")

    # Create a secure temp file that survives after close (delete=False)
    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".hex", delete=False)
    try:
        tmp_file.write(privkey_hex.strip())
        tmp_file.flush()
        tmp_file.close()

        # Restrict permissions to 0o600 (read/write by owner only)
        os.chmod(tmp_file.name, stat.S_IRUSR | stat.S_IWUSR)
        return tmp_file.name
    except Exception as err:
        # Best-effort cleanup on failure
        tmp_file.close()
        if os.path.exists(tmp_file.name):
            os.remove(tmp_file.name)
        raise err


# step:2 file: import_a_raw_private_key_into_the_cosmos_keyring
import subprocess
import shlex
import os


def import_key(name: str, privkey_file: str, passphrase: str) -> str:
    """
    Imports the private key into the local `simd` keyring (file backend).
    The CLI prompts twice for a passphrase; both prompts are satisfied by
    piping the provided `passphrase` via STDIN.

    Returns the command's STDOUT on success.
    """
    if not os.path.exists(privkey_file):
        raise FileNotFoundError(f"Private key file {privkey_file} does not exist.")

    # Construct the CLI command safely
    cmd = f"simd keys import {shlex.quote(name)} {shlex.quote(privkey_file)} --keyring-backend file"

    # Provide the passphrase twice, each followed by a newline
    pass_input = f"{passphrase}\n{passphrase}\n".encode()

    proc = subprocess.run(
        shlex.split(cmd),
        input=pass_input,
        capture_output=True,
    )

    # Always wipe the temporary file after use
    try:
        os.remove(privkey_file)
    except OSError:
        pass

    if proc.returncode != 0:
        raise RuntimeError(f"simd key import failed: {proc.stderr.decode()}")

    return proc.stdout.decode()


# step:3 file: import_a_raw_private_key_into_the_cosmos_keyring
import subprocess
import shlex


def show_address(name: str) -> str:
    """
    Retrieves the address of the imported key from the `simd` keyring.
    Returns the bech32 address as a string.
    """
    cmd = f"simd keys show {shlex.quote(name)} --keyring-backend file --address"
    proc = subprocess.run(shlex.split(cmd), capture_output=True)

    if proc.returncode != 0:
        raise RuntimeError(f"simd keys show failed: {proc.stderr.decode()}")

    return proc.stdout.decode().strip()


# step:1 file: write_a_memory_allocation_profile_to_mem.prof_on_a_cosmos-evm_node
# config.py
import os

def get_rpc_endpoint() -> str:
    """
    Return the JSON-RPC endpoint of the Cosmos-EVM full node.
    The URL is resolved from the COSMOS_EVM_RPC environment variable and
    falls back to 'http://localhost:8545' if not set.
    """
    rpc = os.getenv("COSMOS_EVM_RPC", "http://localhost:8545")
    if not rpc.startswith(("http://", "https://")):
        raise ValueError(f"Invalid RPC endpoint: {rpc}")
    return rpc



# step:2 file: write_a_memory_allocation_profile_to_mem.prof_on_a_cosmos-evm_node
# auth.py
import os
import base64
from typing import Dict

def get_auth_headers() -> Dict[str, str]:
    """
    Build Authorization header for the Cosmos-EVM node.

    1. If COSMOS_EVM_JWT is set, use Bearer <token>.
    2. Else, if COSMOS_EVM_USER and COSMOS_EVM_PASSWORD are set, use Basic Auth.
    3. Otherwise, return empty dict (assumes node does not require auth).
    """
    jwt_token = os.getenv("COSMOS_EVM_JWT")
    if jwt_token:
        return {"Authorization": f"Bearer {jwt_token}"}

    user = os.getenv("COSMOS_EVM_USER")
    password = os.getenv("COSMOS_EVM_PASSWORD")
    if user and password:
        credentials = f"{user}:{password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}

    # No credentials provided
    return {}



# step:3 file: write_a_memory_allocation_profile_to_mem.prof_on_a_cosmos-evm_node
# rpc_debug.py
import json
import requests
from fastapi import APIRouter, HTTPException
from config import get_rpc_endpoint
from auth import get_auth_headers

router = APIRouter()

@router.post("/api/debug/write_mem_profile")
def write_mem_profile(filename: str = "mem.prof"):
    """
    Calls the debug_writeMemProfile method on the Cosmos-EVM node.
    By default writes to 'mem.prof'.
    """
    url = get_rpc_endpoint()
    headers = {"Content-Type": "application/json"}
    headers.update(get_auth_headers())

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "debug_writeMemProfile",
        "params": [filename]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach node: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Node responded with HTTP {resp.status_code}: {resp.text}")

    try:
        response_json = resp.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Non-JSON response returned from node.")

    if "error" in response_json:
        raise HTTPException(status_code=500, detail=response_json["error"])

    # success: result field may be None or some output
    return {"success": True, "result": response_json.get("result")}



# step:1 file: enable_unsafe_cors_on_gaia’s_rest_(api)_server
import os
from pathlib import Path

CONFIG_PATH = Path.home() / ".gaia" / "config" / "app.toml"


def open_config_file() -> list[str]:
    """Read the Gaia app.toml file and return a list of its lines."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            return lines
    except Exception as err:
        # Surface a more readable error for callers
        raise RuntimeError(f"Failed to read {CONFIG_PATH}: {err}") from err


# step:2 file: enable_unsafe_cors_on_gaia’s_rest_(api)_server
import re

SECTION_HEADERS = {"[api]", "[rpc]", "[grpc-web]"}


def modify_parameter(lines: list[str]) -> list[str]:
    """Return a new list of lines where `unsafe-cors = true` is guaranteed."""

    modified_lines = []
    in_relevant_section = False
    parameter_set = False

    for line in lines:
        stripped = line.strip().lower()

        # Track which section we are in
        if stripped.startswith("[") and stripped.endswith("]"):
            in_relevant_section = stripped in SECTION_HEADERS

        if in_relevant_section and stripped.startswith("unsafe-cors"):
            # Replace whatever value was there with `true`
            modified_lines.append("unsafe-cors = true\n")
            parameter_set = True
        else:
            modified_lines.append(line)

    # If the flag did not previously exist, inject it into the first relevant section
    if not parameter_set:
        for i, line in enumerate(modified_lines):
            if line.strip().lower() in SECTION_HEADERS:
                # Insert immediately after the section header
                modified_lines.insert(i + 1, "unsafe-cors = true\n")
                parameter_set = True
                break

    if not parameter_set:
        # No relevant section found; fall back to appending under [api]
        modified_lines.append("\n[api]\n")
        modified_lines.append("unsafe-cors = true\n")

    return modified_lines


# step:3 file: enable_unsafe_cors_on_gaia’s_rest_(api)_server
import shutil
import datetime as _dt


def save_and_close_file(modified_lines: list[str]):
    """Persist the modified app.toml, creating a backup of the original file."""
    timestamp = _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup_path = CONFIG_PATH.with_suffix(f".bak.{timestamp}")

    try:
        shutil.copy2(CONFIG_PATH, backup_path)
    except Exception as err:
        raise RuntimeError(f"Unable to create backup file {backup_path}: {err}") from err

    try:
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            f.writelines(modified_lines)
    except Exception as err:
        raise RuntimeError(f"Failed to write new configuration: {err}") from err


# step:4 file: enable_unsafe_cors_on_gaia’s_rest_(api)_server
import subprocess
import signal
import psutil  # type: ignore  # Requires `pip install psutil`


def restart_gaiad():
    """Restart the Gaia daemon to apply configuration changes."""
    # Preferred: use systemctl when it exists
    try:
        subprocess.check_call(["systemctl", "restart", "gaiad"], stderr=subprocess.STDOUT)
        return "Restarted via systemctl"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back: find the gaiad process and send SIGHUP
        for proc in psutil.process_iter(["name"]):
            if proc.info.get("name", "").startswith("gaiad"):
                proc.send_signal(signal.SIGHUP)
                return f"Sent SIGHUP to pid {proc.pid}"
        raise RuntimeError("gaiad process not found; manual restart required")


# step:5 file: enable_unsafe_cors_on_gaia’s_rest_(api)_server
import requests

DEFAULT_REST_ENDPOINT = "http://localhost:1317/cosmos/base/tendermint/v1beta1/blocks/latest"


def verify_cors_header(rest_endpoint: str | None = None, origin: str = "http://example.com") -> bool:
    """Return True if Access-Control-Allow-Origin is `*` or matches the supplied origin."""
    url = rest_endpoint or DEFAULT_REST_ENDPOINT
    try:
        response = requests.get(url, headers={"Origin": origin}, timeout=5)
        allowed = response.headers.get("Access-Control-Allow-Origin", "")
        return allowed == "*" or allowed == origin
    except requests.RequestException as err:
        raise RuntimeError(f"Failed to query REST endpoint {url}: {err}") from err


# step:1 file: enable_ufw_firewall_on_the_validator_server
########################################################
# backend/ufw_setup.py  ▸  STEP 1                      #
########################################################
from fastapi import FastAPI, HTTPException
import shutil

app = FastAPI()


def _detect_package_manager() -> str:
    """Return the first supported package-manager binary found in PATH."""
    for manager in ("apt-get", "yum", "dnf", "pacman", "apk"):
        if shutil.which(manager):
            return manager
    raise FileNotFoundError("No supported package manager found on this host.")


@app.get("/api/detect-package-manager")
async def detect_package_manager():
    """HTTP GET → { "package_manager": "apt-get" }"""
    try:
        manager = _detect_package_manager()
        return {"package_manager": manager}
    except FileNotFoundError as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:2 file: enable_ufw_firewall_on_the_validator_server
########################################################
# backend/ufw_setup.py  ▸  STEP 2                      #
########################################################
import subprocess
from fastapi import HTTPException

# Map managers → cache-refresh commands
CMD_CACHE = {
    "apt-get": ["sudo", "apt-get", "update", "-y"],
    "yum":     ["sudo", "yum", "makecache"],
    "dnf":     ["sudo", "dnf", "makecache"],
    "pacman":  ["sudo", "pacman", "-Sy"],
    "apk":     ["sudo", "apk", "update"],
}

@app.post("/api/update-package-cache")
async def update_package_cache(manager: str):
    """HTTP POST body ⇒ {"manager":"apt-get"}. Runs the correct cache update command."""
    if manager not in CMD_CACHE:
        raise HTTPException(status_code=400, detail=f"Unsupported package manager: {manager}")
    try:
        proc = subprocess.run(CMD_CACHE[manager], check=True, capture_output=True, text=True)
        return {"stdout": proc.stdout}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=err.stderr or str(err))


# step:3 file: enable_ufw_firewall_on_the_validator_server
########################################################
# backend/ufw_setup.py  ▸  STEP 3                      #
########################################################
# Map managers → install-UFW commands
CMD_INSTALL_UFW = {
    "apt-get": ["sudo", "apt-get", "install", "ufw", "-y"],
    "yum":     ["sudo", "yum", "install", "ufw", "-y"],
    "dnf":     ["sudo", "dnf", "install", "ufw", "-y"],
    "pacman":  ["sudo", "pacman", "-S", "ufw", "--noconfirm"],
    "apk":     ["sudo", "apk", "add", "ufw"],
}

@app.post("/api/install-ufw")
async def install_ufw(manager: str):
    """HTTP POST body ⇒ {"manager":"apt-get"}. Installs UFW."""
    if manager not in CMD_INSTALL_UFW:
        raise HTTPException(status_code=400, detail=f"Unsupported package manager: {manager}")
    try:
        proc = subprocess.run(CMD_INSTALL_UFW[manager], check=True, capture_output=True, text=True)
        return {"stdout": proc.stdout}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=err.stderr or str(err))


# step:4 file: enable_ufw_firewall_on_the_validator_server
########################################################
# backend/ufw_setup.py  ▸  STEP 4                      #
########################################################
CMD_ENABLE = ["sudo", "ufw", "enable"]

@app.post("/api/enable-firewall")
async def enable_firewall():
    """HTTP POST ⇒ enables UFW."""
    try:
        proc = subprocess.run(CMD_ENABLE, check=True, capture_output=True, text=True)
        return {"stdout": proc.stdout}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=err.stderr or str(err))


# step:5 file: enable_ufw_firewall_on_the_validator_server
########################################################
# backend/ufw_setup.py  ▸  STEP 5                      #
########################################################
CMD_STATUS = ["sudo", "ufw", "status", "verbose"]

@app.get("/api/firewall-status")
async def firewall_status():
    """HTTP GET → returns `ufw status verbose`."""
    try:
        proc = subprocess.run(CMD_STATUS, check=True, capture_output=True, text=True)
        return {"status": proc.stdout}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=err.stderr or str(err))


# step:1 file: query_an_account’s_balance_with_cast
import subprocess


def verify_foundry_installation():
    '''
    Ensures Foundry's cast CLI is installed and returns its version.
    '''
    try:
        result = subprocess.run(['cast', '--version'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except FileNotFoundError:
        raise EnvironmentError('Foundry cast CLI is not installed or not in PATH.')
    except subprocess.CalledProcessError as e:
        raise EnvironmentError(f'Error executing cast --version: {e.stderr}')


# step:3 file: query_an_account’s_balance_with_cast
from fastapi import FastAPI, HTTPException, Query
from decimal import Decimal
import subprocess

from verify_foundry import verify_foundry_installation  # Step 1 helper

app = FastAPI()

@app.get('/api/balance')
async def get_account_balance(
    address: str = Query(..., description='EVM account address starting with 0x'),
    rpc_url: str = Query(..., description='HTTPS JSON-RPC endpoint for the target chain')
):
    '''
    Uses Foundry's cast CLI to fetch an EVM account balance and returns both wei and ether values.
    '''
    # Ensure Foundry is installed
    try:
        verify_foundry_installation()
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Simple address validation
    if not address.startswith('0x') or len(address) != 42:
        raise HTTPException(status_code=400, detail='Invalid Ethereum address format')

    try:
        result = subprocess.run(
            ['cast', 'balance', address, '--rpc-url', rpc_url],
            capture_output=True,
            text=True,
            check=True
        )
        balance_wei = result.stdout.strip()
        balance_ether = str(Decimal(balance_wei) / Decimal(10 ** 18))

        return {
            'address': address,
            'balance_wei': balance_wei,
            'balance_ether': balance_ether,
            'rpc_url': rpc_url
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'Cast query failed: {e.stderr}')


# step:1 file: add_a_circuit_breaker_to_the_application_baseapp
#!/usr/bin/env python3
"""
search_circuit_docs.py
Simple helper script to search the official Cosmos documentation for any term.
"""
import json
import sys
import urllib.parse
import requests

BASE_URL = "https://docs.cosmos.network/api/search"

def search_circuit_docs(query: str) -> dict:
    """Return search results from the Cosmos docs search API."""
    try:
        url = f"{BASE_URL}?q={urllib.parse.quote_plus(query)}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"[ERROR] Documentation search failed: {exc}")
        return {}

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Cosmos SDK circuit breaker module integration BaseApp"
    results = search_circuit_docs(query)
    print(json.dumps(results, indent=2))


# step:2 file: add_a_circuit_breaker_to_the_application_baseapp
diff --git a/app/app.go b/app/app.go
@@
-import (
-    // existing imports
-)
+import (
+    // existing imports
+    circuitmodule "github.com/cosmos/cosmos-sdk/x/circuit"
+)
@@
-app.ModuleBasics = module.NewBasicManager(
-    /* other module basics */
-)
+app.ModuleBasics = module.NewBasicManager(
+    /* other module basics */
+    circuitmodule.AppModuleBasic{},
+)



# step:3 file: add_a_circuit_breaker_to_the_application_baseapp
diff --git a/app/app.go b/app/app.go
@@
 type App struct {
     // existing keepers ...
+    CircuitKeeper circuitkeeper.Keeper
 }
@@
 func NewApp(/* params */) *App {
     // existing keeper initialisations...
+
+    // --- Circuit Keeper ------------------------------------
+    circuitKeeper := circuitkeeper.NewKeeper(
+        appCodec,
+        keys[circuittypes.StoreKey],
+        subspaces[circuittypes.ModuleName],
+    )
+    app.CircuitKeeper = *circuitKeeper
+    // --------------------------------------------------------
@@
-    anteHandler, err := ante.NewAnteHandler(ante.HandlerOptions{
-        AccountKeeper: app.AccountKeeper,
-        BankKeeper:    app.BankKeeper,
-        // ...
-    })
+    anteHandler, err := ante.NewAnteHandler(ante.HandlerOptions{
+        AccountKeeper:        app.AccountKeeper,
+        BankKeeper:           app.BankKeeper,
+        // ... other keepers ...
+        CircuitBreakerKeeper: &app.CircuitKeeper,
+    })
     if err != nil {
         panic(err)
     }
 }



# step:4 file: add_a_circuit_breaker_to_the_application_baseapp
diff --git a/app/app.go b/app/app.go
@@
 app.mm = module.NewManager(
     // existing modules...
+    circuitmodule.NewAppModule(appCodec, app.CircuitKeeper),
 )
@@
 app.mm.SetOrderBeginBlockers(
     upgradetypes.ModuleName,
+    circuittypes.ModuleName,
     // existing modules...
 )



# step:5 file: add_a_circuit_breaker_to_the_application_baseapp
#!/usr/bin/env bash
set -euo pipefail

# Compile the application; fails fast on compilation errors.
make build  # alternatively: go build ./...

echo "✅ Application builds successfully with circuit module."



# step:6 file: add_a_circuit_breaker_to_the_application_baseapp
#!/usr/bin/env bash
set -euo pipefail

# Initialize data directory if it doesn't exist yet
if [ ! -d \"$HOME/.myapp\" ]; then
  myappd init local --chain-id localnet
fi

# Start the node
myappd start --home $HOME/.myapp --log_level info 2>&1 | tee node.log



# step:1 file: install_the_foundry_toolchain_using_foundryup
import shutil
import subprocess


def check_system_prerequisites():
    '''Verify that curl, bash, and git are installed and git is up to date (>= 2.30).'''
    required_bins = ['curl', 'bash', 'git']
    missing = [bin for bin in required_bins if shutil.which(bin) is None]
    if missing:
        raise EnvironmentError(f"Missing required binaries: {', '.join(missing)}")

    # Check git version
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True, check=True)
        version_str = result.stdout.strip()  # e.g., 'git version 2.35.1'
        version_number = version_str.split()[-1]
        major, minor, *_ = map(int, version_number.split('.'))
        if (major, minor) < (2, 30):
            raise EnvironmentError(f'git version 2.30 or higher is required, found {version_number}')
    except subprocess.CalledProcessError as e:
        raise EnvironmentError(f'Could not determine git version: {e}')

    return {
        'curl_path': shutil.which('curl'),
        'bash_path': shutil.which('bash'),
        'git_version': version_number,
    }



# step:2 file: install_the_foundry_toolchain_using_foundryup
import subprocess
import os
from pathlib import Path

def download_foundryup():
    '''Download and run the Foundry installation script.'''
    cmd = 'curl -L https://foundry.paradigm.xyz | bash'
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Failed to download and install foundryup: {e}')

    # Add ~/.foundry/bin to PATH for current process so subsequent steps can find foundryup
    foundry_bin = Path.home() / '.foundry' / 'bin'
    os.environ['PATH'] += os.pathsep + str(foundry_bin)
    return str(foundry_bin)



# step:3 file: install_the_foundry_toolchain_using_foundryup
import os
from pathlib import Path

def source_shell_profile(profile_path: str = None):
    '''Reload shell profile so that Foundry tools are in PATH. If sourcing in the current Python process is not possible, update PATH manually as fallback.'''
    # Determine which profile to source
    if profile_path is None:
        bashrc = Path.home() / '.bashrc'
        zshrc = Path.home() / '.zshrc'
        profile_path = zshrc if zshrc.exists() else bashrc

    if not Path(profile_path).exists():
        raise FileNotFoundError(f'Shell profile {profile_path} does not exist.')

    # Since "source" only affects the current shell, we emulate by appending Foundry bin to PATH.
    foundry_bin = Path.home() / '.foundry' / 'bin'
    os.environ['PATH'] += os.pathsep + str(foundry_bin)
    return str(profile_path)



# step:4 file: install_the_foundry_toolchain_using_foundryup
import subprocess

def run_foundryup_install():
    '''Run \"foundryup\" to install or update the Foundry toolchain.'''
    try:
        subprocess.run(['foundryup'], check=True)
    except FileNotFoundError:
        raise RuntimeError('foundryup not found in PATH. Did you run download_foundryup()?')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'foundryup execution failed: {e}')
    return True



# step:5 file: install_the_foundry_toolchain_using_foundryup
import subprocess

def allow_cosmos_prometheus_port():
    """Allow TCP traffic on port 26660 with a descriptive comment."""
    cmd = [
        "sudo",
        "ufw",
        "allow",
        "26660/tcp",
        "comment",
        "Cosmos Prometheus"
    ]

    try:
        # Run the UFW command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            "status": "success",
            "output": result.stdout.strip() or "Rule applied successfully."
        }
    except subprocess.CalledProcessError as exc:
        # Return error information without crashing the application
        return {
            "status": "error",
            "error": exc.stderr.strip(),
            "code": exc.returncode
        }



# step:2 file: allow_prometheus_port_26660_through_ufw
import subprocess


def reload_firewall():
    """Reload the UFW firewall to activate any pending rule changes."""
    cmd = ["sudo", "ufw", "reload"]

    try:
        # Run the UFW reload command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            "status": "success",
            "output": result.stdout.strip() or "Firewall reloaded successfully."
        }
    except subprocess.CalledProcessError as exc:
        # Return error details for debugging
        return {
            "status": "error",
            "error": exc.stderr.strip(),
            "code": exc.returncode
        }



# step:1 file: import_openzeppelin_contracts_into_the_project
from fastapi import FastAPI, HTTPException
import subprocess
import pathlib

app = FastAPI()

@app.post("/install_openzeppelin")
async def install_openzeppelin(project_path: str = "."):
    """Install OpenZeppelin Contracts with npm."""
    try:
        # Ensure we are operating in an absolute path for safety
        project_dir = pathlib.Path(project_path).resolve()
        subprocess.run(
            ["npm", "install", "@openzeppelin/contracts"],
            cwd=project_dir,
            check=True,
        )
        return {"status": "success", "message": "@openzeppelin/contracts installed."}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"npm install failed: {err}")


# step:2 file: import_openzeppelin_contracts_into_the_project
from fastapi import FastAPI, HTTPException
import pathlib

app = FastAPI()

SAMPLE_ERC20 = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import \"@openzeppelin/contracts/token/ERC20/ERC20.sol\";

contract MyToken is ERC20 {
    constructor(uint256 initialSupply) ERC20(\"MyToken\", \"MTK\") {
        _mint(msg.sender, initialSupply);
    }
}
"""

@app.post("/create_contract")
async def create_contract(project_path: str = ".", filename: str = "MyToken.sol"):
    """Create a Solidity file with OpenZeppelin imports."""
    try:
        contracts_dir = pathlib.Path(project_path).resolve() / "contracts"
        contracts_dir.mkdir(parents=True, exist_ok=True)
        contract_file = contracts_dir / filename
        contract_file.write_text(SAMPLE_ERC20)
        return {"status": "success", "message": f"Contract written to {contract_file}"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Unable to write contract: {err}")


# step:3 file: import_openzeppelin_contracts_into_the_project
from fastapi import FastAPI, HTTPException
import subprocess
import pathlib

app = FastAPI()

@app.post("/compile_contracts")
async def compile_contracts(project_path: str = "."):
    """Compile Solidity contracts with Hardhat."""
    try:
        project_dir = pathlib.Path(project_path).resolve()
        subprocess.run(["npx", "hardhat", "compile"], cwd=project_dir, check=True)
        return {"status": "success", "message": "Hardhat compilation completed."}
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"Hardhat compile failed: {err}")


# step:1 file: add_cw-orch_as_an_optional_dependency_in_cargo.toml
from pathlib import Path
import toml

# Absolute path to Cargo.toml (assumed to be in project root)
CARGO_TOML_PATH = Path(__file__).resolve().parent / 'Cargo.toml'

def open_cargo_toml():
    """Load Cargo.toml and return a Python dict representation."""
    if not CARGO_TOML_PATH.exists():
        raise FileNotFoundError(f'Cargo.toml not found at: {CARGO_TOML_PATH}')
    try:
        content = CARGO_TOML_PATH.read_text()
        return toml.loads(content)
    except toml.TomlDecodeError as err:
        raise ValueError(f'Invalid TOML format: {err}')


# step:2 file: add_cw-orch_as_an_optional_dependency_in_cargo.toml
def insert_optional_dependency(
    manifest: dict,
    dep_name: str = 'cw-orch',
    version: str = '*',
    section: str = "target.'cfg(not(target_arch = \"wasm32\"))'.dependencies"
) -> dict:
    """Insert/Update an optional dependency inside the specified Cargo.toml section."""
    # Walk (or create) the nested section path
    levels = [lvl.strip("'") for lvl in section.split('.')]
    cursor = manifest
    for lvl in levels:
        cursor = cursor.setdefault(lvl, {})
    # Finally, insert the dependency specification
    cursor[dep_name] = {"version": version, "optional": True}
    return manifest


# step:3 file: add_cw-orch_as_an_optional_dependency_in_cargo.toml
from pathlib import Path
import toml

CARGO_TOML_PATH = Path(__file__).resolve().parent / 'Cargo.toml'

def save_manifest(manifest: dict):
    """Serialize the in-memory manifest back to Cargo.toml on disk."""
    CARGO_TOML_PATH.write_text(toml.dumps(manifest))


# step:4 file: add_cw-orch_as_an_optional_dependency_in_cargo.toml
import subprocess

def cargo_check() -> str:
    """Run `cargo check` and return its standard output. Raises if the command fails."""
    proc = subprocess.run(['cargo', 'check'], capture_output=True, text=True)
    if proc.returncode != 0:
        # Forward stderr so callers can see why the graph failed to resolve
        raise RuntimeError(proc.stderr)
    return proc.stdout


# step:1 file: set_minimum_gas_prices_to_0.01token_in_app.toml
import os
from pathlib import Path


def locate_config_file(node_home: str) -> str:
    '''
    Locate the app.toml file inside a Cosmos-SDK node home directory.

    Args:
        node_home (str): Absolute path to the node's home (e.g. '/home/ubuntu/.appd')
    Returns:
        str: Absolute path string to app.toml
    Raises:
        FileNotFoundError: If the file cannot be found
    '''
    config_path = Path(node_home) / 'config' / 'app.toml'
    if not config_path.exists():
        raise FileNotFoundError(f'Configuration file not found at {config_path}')
    return str(config_path)



# step:2 file: set_minimum_gas_prices_to_0.01token_in_app.toml
import os


def load_config_file(file_path: str) -> dict:
    '''
    Load a TOML configuration file and return its contents as a Python dict.
    Prefers the built-in `tomllib` (Python 3.11+); falls back to the external
    `toml` package when necessary.
    '''
    try:
        import tomllib  # Python 3.11+
        with open(file_path, 'rb') as fp:
            return tomllib.load(fp)
    except ModuleNotFoundError:
        try:
            import toml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError('`toml` library required. Install via `pip install toml`.') from exc
        with open(file_path, 'r', encoding='utf-8') as fp:
            return toml.load(fp)



# step:3 file: set_minimum_gas_prices_to_0.01token_in_app.toml
def update_toml_value(config_data: dict, key: str = 'minimum-gas-prices', value: str = '0.01token') -> dict:
    '''
    Update (or add) a key in the TOML configuration dictionary.
    '''
    config_data[key] = value
    return config_data



# step:4 file: set_minimum_gas_prices_to_0.01token_in_app.toml
def save_file(config_data: dict, file_path: str) -> None:
    '''
    Persist the in-memory TOML dictionary back to disk.
    Uses the `toml` library for serialization.
    '''
    try:
        import toml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('`toml` library required. Install via `pip install toml`.') from exc

    with open(file_path, 'w', encoding='utf-8') as fp:
        toml.dump(config_data, fp)



# step:5 file: set_minimum_gas_prices_to_0.01token_in_app.toml
import subprocess


def restart_node_process(service_name: str) -> None:
    '''
    Restart the Cosmos-SDK node systemd service so configuration changes take effect.
    '''
    try:
        subprocess.run(['systemctl', 'restart', service_name], check=True, capture_output=True)
        print(f'[✓] Service {service_name} restarted')
    except subprocess.CalledProcessError as err:
        stderr = err.stderr.decode() if err.stderr else str(err)
        raise RuntimeError(f'Failed to restart {service_name}: {stderr}') from err



# step:6 file: set_minimum_gas_prices_to_0.01token_in_app.toml
import json
import subprocess
import time


def verify_config_value(binary: str, expected_value: str = '0.01token', retries: int = 5, delay: int = 3) -> bool:
    '''
    Poll the node CLI until it reports the desired minimum-gas-prices value.
    '''
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run([binary, 'status'], capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            mgp = (
                data.get('MinimumGasPrices')
                or data.get('minimum_gas_prices')
                or data.get('NodeInfo', {}).get('minimum_gas_price')
            )

            if mgp == expected_value:
                print(f'[✓] minimum-gas-prices verified as {mgp}')
                return True

            print(f'[Attempt {attempt}] Current value {mgp} != expected {expected_value}')

        except Exception as exc:
            print(f'[Attempt {attempt}] Error verifying min gas price: {exc}')

        time.sleep(delay)

    raise RuntimeError('Unable to confirm minimum-gas-prices change after multiple attempts')



# step:1 file: add_an_“interface”_feature_flag_in_cargo.toml_to_enable_cw-orch_for_a_cosmwasm_contract
import os
import toml


def load_cargo_toml(path: str = "Cargo.toml") -> dict:
    """Read Cargo.toml from disk and return its contents as a Python dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Make sure you are in the project root.")

    with open(path, "r", encoding="utf-8") as fp:
        cargo_data = toml.load(fp)

    return cargo_data


# step:2 file: add_an_“interface”_feature_flag_in_cargo.toml_to_enable_cw-orch_for_a_cosmwasm_contract
def add_cw_orch_dependency(cargo_data: dict, version: str = "*", optional: bool = True) -> dict:
    """Insert or update the cw-orch dependency in the [dependencies] table."""
    deps = cargo_data.setdefault("dependencies", {})

    existing = deps.get("cw-orch")
    if existing is None:
        # Dependency is missing; create it.
        deps["cw-orch"] = {"version": version, "optional": optional}
    elif isinstance(existing, str):
        # Dependency is a simple version string; replace with full table.
        deps["cw-orch"] = {"version": version, "optional": optional}
    else:
        # Dependency exists as a table; update fields without clobbering others.
        existing["version"] = version
        existing["optional"] = optional

    return cargo_data


# step:3 file: add_an_“interface”_feature_flag_in_cargo.toml_to_enable_cw-orch_for_a_cosmwasm_contract
def append_interface_feature(cargo_data: dict) -> dict:
    """Guarantee that the [features] table contains interface = [\"dep:cw-orch\"]."""
    features = cargo_data.setdefault("features", {})
    interface = features.setdefault("interface", [])

    if "dep:cw-orch" not in interface:
        interface.append("dep:cw-orch")

    return cargo_data


# step:4 file: add_an_“interface”_feature_flag_in_cargo.toml_to_enable_cw-orch_for_a_cosmwasm_contract
def save_cargo_toml(cargo_data: dict, path: str = "Cargo.toml") -> None:
    """Persist the modified Cargo.toml dictionary to disk."""
    import toml

    with open(path, "w", encoding="utf-8") as fp:
        toml.dump(cargo_data, fp)

    print("Cargo.toml saved ✅")


# step:5 file: add_an_“interface”_feature_flag_in_cargo.toml_to_enable_cw-orch_for_a_cosmwasm_contract
def cargo_build_with_interface() -> None:
    """Execute `cargo build --features interface` and surface any compilation errors."""
    import subprocess

    cmd = ["cargo", "build", "--features", "interface"]
    try:
        subprocess.run(cmd, check=True)
        print("Cargo build succeeded ✅")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Cargo build failed with exit code {err.returncode}")


# step:1 file: verify_the_installed_versions_of_foundry_components_(forge,_cast,_anvil)
# foundry_backend.py
from fastapi import FastAPI, HTTPException
import subprocess
import shutil

app = FastAPI()

@app.post("/api/foundry/update")
async def foundryup_update():
    """Run `foundryup` to update Foundry. Returns the CLI output."""
    # Make sure the binary exists on the host machine.
    if not shutil.which("foundryup"):
        raise HTTPException(status_code=404, detail="`foundryup` binary not found. Please install Foundry first.")

    try:
        result = subprocess.run(
            ["foundryup"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        # Surface the actual CLI error back to the caller.
        raise HTTPException(status_code=500, detail=e.stderr or "`foundryup` failed with an unknown error.")


# step:2 file: verify_the_installed_versions_of_foundry_components_(forge,_cast,_anvil)
# foundry_backend.py (append below previous code)
from fastapi import FastAPI, HTTPException
import subprocess
import shutil

app = FastAPI()

@app.get("/api/foundry/forge/version")
async def forge_version():
    """Return the installed Forge version."""
    if not shutil.which("forge"):
        raise HTTPException(status_code=404, detail="`forge` binary not found. Have you installed Foundry?")

    try:
        result = subprocess.run(
            ["forge", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"version": result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr or "`forge --version` failed.")


# step:3 file: verify_the_installed_versions_of_foundry_components_(forge,_cast,_anvil)
# foundry_backend.py (append below previous code)
from fastapi import FastAPI, HTTPException
import subprocess
import shutil

app = FastAPI()

@app.get("/api/foundry/cast/version")
async def cast_version():
    """Return the installed Cast version."""
    if not shutil.which("cast"):
        raise HTTPException(status_code=404, detail="`cast` binary not found. Have you installed Foundry?")

    try:
        result = subprocess.run(
            ["cast", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"version": result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr or "`cast --version` failed.")


# step:4 file: verify_the_installed_versions_of_foundry_components_(forge,_cast,_anvil)
# foundry_backend.py (append below previous code)
from fastapi import FastAPI, HTTPException
import subprocess
import shutil

app = FastAPI()

@app.get("/api/foundry/anvil/version")
async def anvil_version():
    """Return the installed Anvil version."""
    if not shutil.which("anvil"):
        raise HTTPException(status_code=404, detail="`anvil` binary not found. Have you installed Foundry?")

    try:
        result = subprocess.run(
            ["anvil", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"version": result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr or "`anvil --version` failed.")


# step:1 file: enable_state_sync_snapshots_with_interval_1000
import subprocess


def stop_node_service(service_name: str = "cosmosd") -> None:
    """Stops a systemd-managed Cosmos SDK node.

    Args:
        service_name: The name of the systemd service running the node.
    Raises:
        RuntimeError: If the service fails to stop.
    """
    try:
        # `systemctl is-active` returns 0 if active, non-zero otherwise.
        status = subprocess.run(["systemctl", "is-active", service_name], capture_output=True, text=True)
        if status.returncode != 0:
            print(f"[INFO] Service '{service_name}' is already stopped.")
            return

        print(f"[INFO] Stopping service '{service_name}'…")
        stop = subprocess.run(["sudo", "systemctl", "stop", service_name], check=True)
        print(f"[SUCCESS] Service '{service_name}' stopped.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to stop service '{service_name}': {err}") from err


# step:2 file: enable_state_sync_snapshots_with_interval_1000
import os
import re


def edit_app_toml_snapshot_section(
    chain_id: str,
    snapshot_interval: int = 1000,
    snapshot_keep_recent: int = 2,
) -> None:
    """Updates snapshot settings inside $HOME/.<chain-id>/config/app.toml.

    The function searches for existing `snapshot-interval` and `snapshot-keep-recent` lines
    and replaces their values. If a key is missing, it appends the config to the end of the file.
    """
    path = os.path.expanduser(f"~/.{chain_id}/config/app.toml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"app.toml not found at {path}")

    with open(path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    def upsert(key: str, value: int, lines: list[str]) -> list[str]:
        pattern = re.compile(rf"^\s*{key}\s*=\s*\d+")
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{key} = {value}\n"
                return lines
        # Key not found; append.
        lines.append(f"{key} = {value}\n")
        return lines

    lines = upsert("snapshot-interval", snapshot_interval, lines)
    lines = upsert("snapshot-keep-recent", snapshot_keep_recent, lines)

    with open(path, "w", encoding="utf-8") as fp:
        fp.writelines(lines)

    print(
        f"[SUCCESS] Updated snapshot settings in {path}: interval={snapshot_interval}, keep_recent={snapshot_keep_recent}"
    )


# step:3 file: enable_state_sync_snapshots_with_interval_1000
import subprocess


def start_node_service(service_name: str = "cosmosd") -> None:
    """Starts (or restarts) the Cosmos SDK node using systemd."""
    try:
        print(f"[INFO] Starting service '{service_name}'…")
        subprocess.run(["sudo", "systemctl", "start", service_name], check=True)
        print(f"[SUCCESS] Service '{service_name}' started.")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to start service '{service_name}': {err}") from err


# step:4 file: enable_state_sync_snapshots_with_interval_1000
import os
import time
from pathlib import Path


def verify_snapshot_creation(chain_id: str, check_interval: int = 30) -> None:
    """Polls the snapshot directory until at least one snapshot appears.

    Args:
        chain_id: The chain ID used to locate the data directory.
        check_interval: Seconds between directory scans.
    """
    snap_dir = Path(os.path.expanduser(f"~/.{chain_id}/data/snapshots"))
    if not snap_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")

    print(f"[INFO] Watching {snap_dir} for new snapshot files… (Ctrl+C to exit)")
    try:
        already_seen = {p.name for p in snap_dir.iterdir() if p.is_dir() or p.is_file()}
        while True:
            current = {p.name for p in snap_dir.iterdir() if p.is_dir() or p.is_file()}
            new_files = current - already_seen
            if new_files:
                for f in new_files:
                    print(f"[SUCCESS] New snapshot detected: {f}")
                already_seen = current
            time.sleep(check_interval)
    except KeyboardInterrupt:
        print("[INFO] Stopped watching snapshot directory.")


# step:5 file: enable_state_sync_snapshots_with_interval_1000
import requests


def serve_snapshot_rpc(rpc_url: str = "http://localhost:26657/snapshot/info") -> dict:
    """Checks that the node’s RPC endpoint responds with snapshot info.

    Args:
        rpc_url: Full URL to the /snapshot/info endpoint.
    Returns:
        Parsed JSON response if successful.
    Raises:
        RuntimeError: If the RPC endpoint is unreachable or returns a non-200 code.
    """
    try:
        print(f"[INFO] Querying snapshot info at {rpc_url}…")
        resp = requests.get(rpc_url, timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"Endpoint returned HTTP {resp.status_code}")
        json_resp = resp.json()
        print("[SUCCESS] Snapshot RPC is live. Response snippet:", json_resp.get("result", {})[:1])
        return json_resp
    except (requests.ConnectionError, requests.Timeout) as err:
        raise RuntimeError(f"Failed to reach snapshot RPC at {rpc_url}: {err}") from err


# step:3 file: deposit_3_ebtc_into_the_maxbtc_ebtc_supervault
### supervault_bff.py
"""Minimal FastAPI BFF that exposes vault metadata.
Run:  uvicorn supervault_bff:app --reload
"""
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Supervault BFF")

# Static reference table — keep this up-to-date from docs/btc-summer/technical/reference
SUPERVAULTS = {
    "ebtc": {
        "contract": "neutron1s8k6gcrnsfrs9rj3j8757w4e0ttmzsdmjvwfwxruhu2t8xjgwxaqegzjgt",
        "single_sided": True
    },
    # Add other assets here …
}

@app.get("/api/supervault/{asset}")
async def get_vault(asset: str):
    asset = asset.lower()
    if asset not in SUPERVAULTS:
        raise HTTPException(status_code=404, detail="Unsupported asset")
    return SUPERVAULTS[asset]


# step:4 file: deposit_3_ebtc_into_the_maxbtc_ebtc_supervault
### tx_builder.py
"""Utility that constructs a deposit tx for eBTC Supervault using cosmpy.
The mnemonic / private key should be provided via environment variable to keep secrets off the frontend.
"""
import os, base64, json
from datetime import timedelta
from cosmpy.crypto.keypairs import PrivateKey
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import LocalWallet

# Neutron main-net RPC + chain-id
NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url="https://rpc-kralum.neutron-1.nomusa.xyz",
    fee_minimum_gas_price=0.025,
    fee_denom="untrn"
)

EBTC_DENOM = "ibc/E2A000FD3EDD91C9429B473995CE2C7C555BCC8CFC1D0A3D02F514392B7A80E8"

client = LedgerClient(NETWORK)

MNEMONIC = os.getenv("BFF_MNEMONIC")
if not MNEMONIC:
    raise RuntimeError("BFF_MNEMONIC env var not set – cannot sign tx")

wallet = LocalWallet.from_mnemonic(MNEMONIC)


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



# step:5 file: deposit_3_ebtc_into_the_maxbtc_ebtc_supervault
### broadcaster.py
"""Sign + broadcast wrapper. Separating concerns keeps the builder reusable."""
import json
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.client import NetworkConfig, LedgerClient
from tx_builder import build_deposit_tx, wallet, client


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



# step:2 file: provide_new_liquidity_to_the_wbtc_lbtc_supervault_with_1_wbtc_and_1_lbtc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.contract import SmartContract
import os

app = FastAPI()

LCD_URL = os.getenv('LCD_URL', 'https://rest-kralum.neutron.org')
CHAIN_ID = os.getenv('CHAIN_ID', 'neutron-1')
WBTC_CONTRACT = os.getenv('WBTC_CONTRACT', 'neutron1wbtcxxxxxxxxxxxxxxxxxxxxxx')
LBTC_CONTRACT = os.getenv('LBTC_CONTRACT', 'neutron1lbtcxxxxxxxxxxxxxxxxxxxxxx')
MICRO_FACTOR = 1_000_000  # 1 token = 1_000_000 micro-units (example)

network_cfg = NetworkConfig(chain_id=CHAIN_ID, url=LCD_URL)
client = LedgerClient(network_cfg)

class BalanceStatus(BaseModel):
    has_wbtc: bool
    has_lbtc: bool

def cw20_balance(contract: str, addr: str) -> int:
    """Query CW20 balance via the contract's `balance` endpoint."""
    sc = SmartContract(contract, client)
    try:
        resp = sc.query({"balance": {"address": addr}})
        return int(resp.get('balance', '0'))
    except Exception:
        # If the query fails treat balance as zero
        return 0

@app.get('/api/validate_balances', response_model=BalanceStatus)
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


# step:3 file: provide_new_liquidity_to_the_wbtc_lbtc_supervault_with_1_wbtc_and_1_lbtc
from fastapi import FastAPI
import os

app = FastAPI()

@app.get('/api/supervault_address')
async def get_supervault_contract_address():
    """Simple helper so the frontend can discover the Supervault contract."""
    return {
        'supervault_address': os.getenv('SUPERVAULT_CONTRACT', 'neutron1supervaultxxxxxxxxxxxxxxxxxxxxxx')
    }


# step:4 file: provide_new_liquidity_to_the_wbtc_lbtc_supervault_with_1_wbtc_and_1_lbtc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, base64, json

from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Transaction

app = FastAPI()

LCD_URL = os.getenv('LCD_URL', 'https://rest-kralum.neutron.org')
CHAIN_ID = os.getenv('CHAIN_ID', 'neutron-1')
WBTC_CONTRACT = os.getenv('WBTC_CONTRACT', 'neutron1wbtcxxxxxxxxxxxxxxxxxxxxxx')
LBTC_CONTRACT = os.getenv('LBTC_CONTRACT', 'neutron1lbtcxxxxxxxxxxxxxxxxxxxxxx')
SUPERVAULT_CONTRACT = os.getenv('SUPERVAULT_CONTRACT', 'neutron1supervaultxxxxxxxxxxxxxxxxxxxxxx')
MICRO_FACTOR = 1_000_000

network_cfg = NetworkConfig(chain_id=CHAIN_ID, url=LCD_URL)
client = LedgerClient(network_cfg)

class TxBytes(BaseModel):
    tx_bytes: str  # base64-encoded unsigned Tx body (returned to caller)

@app.post('/api/construct_tx', response_model=TxBytes)
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


# step:5 file: provide_new_liquidity_to_the_wbtc_lbtc_supervault_with_1_wbtc_and_1_lbtc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, base64

from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import PrivateKeyWallet
from cosmpy.aerial.tx import Transaction

app = FastAPI()

LCD_URL = os.getenv('LCD_URL', 'https://rest-kralum.neutron.org')
CHAIN_ID = os.getenv('CHAIN_ID', 'neutron-1')
network_cfg = NetworkConfig(chain_id=CHAIN_ID, url=LCD_URL)
client = LedgerClient(network_cfg)

# IMPORTANT: Store this mnemonic securely (e.g., Hashicorp Vault, AWS Secrets Manager)
MNEMONIC = os.getenv('SIGNING_MNEMONIC')
if not MNEMONIC:
    raise RuntimeError('SIGNING_MNEMONIC environment variable is missing')
wallet = PrivateKeyWallet.from_mnemonic(MNEMONIC)

class BroadcastReq(BaseModel):
    tx_bytes: str  # base64-encoded unsigned Tx

@app.post('/api/sign_and_broadcast')
async def sign_and_broadcast_tx(req: BroadcastReq):
    """Signs the provided Tx bytes and broadcasts them to Neutron."""
    try:
        raw = base64.b64decode(req.tx_bytes)
        tx = Transaction.load(raw)
        signed_tx = tx.sign(wallet)
        response = client.broadcast_tx(signed_tx)

        # cosmpy returns an object that can be inspected for errors
        if response.is_err():
            raise HTTPException(status_code=500, detail=response.raw_log)

        return {"tx_hash": response.tx_hash}

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:1 file: enable_grpc_and_swagger_endpoints_in_app.toml
import os
import toml

def load_app_config(chain_id: str):
    """Load ~/.<chain-id>/config/app.toml and return a (config_dict, file_path) tuple."""
    config_path = os.path.expanduser(f"~/.{chain_id}/config/app.toml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"app.toml not found at {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as fp:
            config = toml.load(fp)
    except Exception as err:
        raise RuntimeError(f"Failed to parse {config_path}: {err}")

    return config, config_path


# step:2 file: enable_grpc_and_swagger_endpoints_in_app.toml
def enable_grpc(chain_id: str, address: str = ":9090"):
    """Enable gRPC and set its listen address (default :9090)."""
    config, path = load_app_config(chain_id)

    grpc_cfg = config.get("grpc", {})
    grpc_cfg["enable"] = True
    if address:
        grpc_cfg["address"] = address
    config["grpc"] = grpc_cfg

    # Persist changes back to disk
    with open(path, "w", encoding="utf-8") as fp:
        toml.dump(config, fp)

    return {"status": "success", "path": path, "grpc": grpc_cfg}


# step:3 file: enable_grpc_and_swagger_endpoints_in_app.toml
def enable_swagger(chain_id: str, address: str = "tcp://0.0.0.0:1317"):
    """Enable Swagger UI and REST API address."""
    config, path = load_app_config(chain_id)

    api_cfg = config.get("api", {})
    api_cfg["swagger"] = True
    api_cfg["address"] = address
    config["api"] = api_cfg

    with open(path, "w", encoding="utf-8") as fp:
        toml.dump(config, fp)

    return {"status": "success", "path": path, "api": api_cfg}


# step:4 file: enable_grpc_and_swagger_endpoints_in_app.toml
import subprocess

def restart_node(service_name: str):
    """Restart the node via systemctl (e.g. service_name='neutrond')."""
    try:
        subprocess.run(["systemctl", "restart", service_name], check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Failed to restart {service_name}: {err}")
    return {"status": "restarted", "service": service_name}


# step:5 file: enable_grpc_and_swagger_endpoints_in_app.toml
import subprocess
import requests


def check_endpoints(grpc_addr: str = "localhost:9090", swagger_url: str = "http://localhost:1317/swagger/"):
    """Return a list of gRPC services and whether the Swagger UI is reachable."""
    # Check gRPC services
    try:
        result = subprocess.run([
            "grpcurl", "-plaintext", grpc_addr, "list"
        ], capture_output=True, text=True, check=True)
        grpc_services = result.stdout.strip().splitlines()
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"grpcurl failed: {err.stderr}")

    # Check Swagger UI
    try:
        resp = requests.get(swagger_url, timeout=5)
        swagger_ok = resp.status_code == 200
    except requests.RequestException:
        swagger_ok = False

    return {
        "grpc_services": grpc_services,
        "swagger_reachable": swagger_ok
    }


# step:3 file: create_an_ibc_transfer_on_an_unordered_channel_with_a_unique_timeout_timestamp
# backend/ibc_transfer.py
'''
FastAPI route that signs and broadcasts an IBC (ICS-20) transfer using cosmpy.
The mnemonic is read from the MNEMONIC environment variable to keep secrets off the
frontend. Adjust NETWORKS mapping for your target chains.
'''
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.ibc.applications.transfer.v1.tx_pb2 import MsgTransfer
from cosmpy.protos.ibc.core.client.v1.client_pb2 import Height

app = FastAPI()

NETWORKS = {
    'cosmoshub-4': NetworkConfig(
        chain_id='cosmoshub-4',
        url='https://rpc.cosmos.directory/cosmoshub',
        fee_minimum_gas_price=0.025,
        fee_denomination='uatom',
        staking_denomination='uatom',
        address_prefix='cosmos',
    )
}

class IbcTransferRequest(BaseModel):
    port_id: str = 'transfer'
    channel_id: str
    amount: int           # base-denom amount, e.g. 1000000 uatom
    denom: str            # base denom, e.g. "uatom"
    receiver: str         # bech32 address on destination chain
    timeout_timestamp: int
    chain_id: str

@app.post('/api/ibc_transfer')
async def ibc_transfer(req: IbcTransferRequest):
    try:
        net_cfg = NETWORKS.get(req.chain_id)
        if not net_cfg:
            raise HTTPException(status_code=400, detail=f'Unsupported chain_id {req.chain_id}')

        mnemonic = os.getenv('MNEMONIC')
        if not mnemonic:
            raise HTTPException(status_code=500, detail='Backend mis-configuration: MNEMONIC environment variable not set')

        wallet = LocalWallet.from_mnemonic(mnemonic, prefix=net_cfg.address_prefix)
        client = LedgerClient(net_cfg)

        msg = MsgTransfer(
            source_port=req.port_id,
            source_channel=req.channel_id,
            token={'denom': req.denom, 'amount': str(req.amount)},
            sender=wallet.address(),
            receiver=req.receiver,
            timeout_height=Height(revision_number=0, revision_height=0),
            timeout_timestamp=req.timeout_timestamp,
        )

        tx = Transaction()
        tx.add_message(msg)
        tx.seal(client, wallet)
        tx.sign(wallet)

        broadcast_result = client.broadcast_transaction(tx)
        if broadcast_result.code != 0:
            raise HTTPException(
                status_code=400,
                detail=f'Broadcast failed (code={broadcast_result.code}): {broadcast_result.raw_log}',
            )

        return {'tx_hash': broadcast_result.tx_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:5 file: create_an_ibc_transfer_on_an_unordered_channel_with_a_unique_timeout_timestamp
# backend/tx_monitor.py
'''
Simple endpoint that polls a Cosmos RPC node for the transaction until it is confirmed.
'''
import asyncio
from fastapi import HTTPException
from cosmpy.aerial.client import LedgerClient

async def _wait_for_tx(client: LedgerClient, tx_hash: str, poll_interval: int = 5, max_attempts: int = 60):
    attempts = 0
    while attempts < max_attempts:
        tx_resp = client.query_tx(tx_hash)
        if tx_resp:
            if tx_resp.code == 0:
                return {'status': 'success', 'tx_response': tx_resp}
            else:
                raise HTTPException(status_code=400, detail=f'Transaction failed (code={tx_resp.code}): {tx_resp.raw_log}')
        await asyncio.sleep(poll_interval)
        attempts += 1
    raise HTTPException(status_code=504, detail='Timed out waiting for transaction confirmation')

@app.get('/api/tx_status/{chain_id}/{tx_hash}')
async def tx_status(chain_id: str, tx_hash: str):
    net_cfg = NETWORKS.get(chain_id)
    if not net_cfg:
        raise HTTPException(status_code=400, detail=f'Unsupported chain {chain_id}')
    client = LedgerClient(net_cfg)
    return await _wait_for_tx(client, tx_hash)


# step:1 file: load_an_external_snapshot_archive_into_the_node_s_store
import os


def get_node_home():
    """
    Determine the simd node's home directory.

    Priority order:
      1. SIMD_HOME environment variable
      2. Default to ~/.simapp

    Returns:
        str: Absolute path to the node home directory.

    Raises:
        FileNotFoundError: If the directory cannot be located.
    """
    # Use environment override when provided
    home_dir = os.environ.get("SIMD_HOME", os.path.expanduser("~/.simapp"))
    if not os.path.isdir(home_dir):
        raise FileNotFoundError(f"Unable to locate node home directory at {home_dir}")
    return home_dir


# step:2 file: load_an_external_snapshot_archive_into_the_node_s_store
import subprocess
import psutil


def stop_node_service():
    """
    Gracefully stop the running simd service or process.

    Returns:
        str: A human-readable message describing how the node was stopped.

    Raises:
        RuntimeError: If no running simd process could be found or stopped.
    """
    # Try systemd first
    try:
        systemctl_check = subprocess.run(["systemctl", "is-active", "--quiet", "simd"], check=False)
        if systemctl_check.returncode == 0:
            result = subprocess.run(["sudo", "systemctl", "stop", "simd"], capture_output=True, text=True)
            if result.returncode == 0:
                return "simd service stopped via systemctl"
    except FileNotFoundError:
        # systemctl not installed; fall back
        pass

    # Fallback: manually kill the simd process
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == "simd":
            proc.terminate()
            try:
                proc.wait(timeout=30)
                return f"simd process {proc.pid} terminated"
            except psutil.TimeoutExpired:
                proc.kill()
                return f"simd process {proc.pid} killed"

    raise RuntimeError("No running simd process or service found.")


# step:3 file: load_an_external_snapshot_archive_into_the_node_s_store
import hashlib
import tarfile
import json
from pathlib import Path


def validate_snapshot_archive(archive_path: str, expected_checksum: str = None, expected_chain_id: str = None):
    """
    Validate a snapshot archive by checking file existence, SHA-256 checksum, and chain-id.

    Args:
        archive_path (str): Path to the snapshot .tar.gz file.
        expected_checksum (str, optional): Expected SHA-256 hex string.
        expected_chain_id (str, optional): Expected chain-id to compare with the genesis inside the archive.

    Returns:
        dict: Validation summary containing the computed checksum.

    Raises:
        FileNotFoundError: If the archive is missing.
        ValueError: On checksum or chain-id mismatch.
    """
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(f"Snapshot archive {archive} not found.")

    # Compute SHA-256
    sha256 = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    computed_checksum = sha256.hexdigest()

    if expected_checksum and computed_checksum.lower() != expected_checksum.lower():
        raise ValueError(f"Checksum mismatch. Expected {expected_checksum}, got {computed_checksum}")

    if expected_chain_id:
        # Peek inside the archive for a genesis.json file
        try:
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("genesis.json"):
                        f = tar.extractfile(member)
                        if f:
                            genesis = json.load(f)
                            chain_id = genesis.get("chain_id")
                            if chain_id != expected_chain_id:
                                raise ValueError(f"Chain-ID mismatch. Expected {expected_chain_id}, got {chain_id}")
                            break
        except tarfile.TarError as err:
            raise ValueError(f"Unable to inspect archive: {err}")

    return {
        "archive": str(archive),
        "checksum": computed_checksum,
        "valid": True
    }


# step:4 file: load_an_external_snapshot_archive_into_the_node_s_store
import shutil
from pathlib import Path
from datetime import datetime


def backup_and_remove_data(node_home: str, backup_root: str = None):
    """
    Copy `<home>/data` to a timestamped backup directory and then remove the original.

    Args:
        node_home (str): Path returned by `get_node_home()`.
        backup_root (str, optional): Directory where backups will be stored. Defaults to `<home>/backup-<timestamp>`.

    Returns:
        str: Path to the backup directory that was created.
    """
    data_dir = Path(node_home) / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup_root = Path(backup_root) if backup_root else Path(node_home) / f"backup-{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=True)

    shutil.copytree(data_dir, backup_root / "data", dirs_exist_ok=True)
    # Remove original data directory after successful copy
    shutil.rmtree(data_dir)

    return str(backup_root)


# step:5 file: load_an_external_snapshot_archive_into_the_node_s_store
import subprocess


def restore_snapshot(archive_path: str, node_home: str):
    """
    Execute `simd snapshot restore` to populate the `<home>/data` directory from a snapshot.
    Raises an exception if the command returns a non-zero exit status.
    """
    cmd = ["simd", "snapshot", "restore", archive_path, f"--home={node_home}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Snapshot restore failed:\n{result.stderr}")
    return result.stdout.strip()


# step:6 file: load_an_external_snapshot_archive_into_the_node_s_store
def construct_simd_start_command(node_home: str, extra_flags: str = "") -> str:
    """
    Build a fully-qualified `simd start` command string.
    """
    base_cmd = f"simd start --home={node_home}"
    full_cmd = f"{base_cmd} {extra_flags}".strip()
    return full_cmd


# step:7 file: load_an_external_snapshot_archive_into_the_node_s_store
import subprocess


def start_node_service(node_home: str, use_systemd: bool = True):
    """
    Start the simd node after the snapshot restore.

    Args:
        node_home (str): Path to the node's home directory.
        use_systemd (bool): Whether to attempt `systemctl start simd` first.

    Returns:
        str: Message describing the action taken.
    """
    if use_systemd:
        result = subprocess.run(["sudo", "systemctl", "start", "simd"], capture_output=True, text=True)
        if result.returncode == 0:
            return "simd service started via systemd"
        # fall back to manual start if systemd fails

    # Manual background start
    cmd = ["simd", "start", f"--home={node_home}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return f"simd node started with PID {proc.pid}"


# step:8 file: load_an_external_snapshot_archive_into_the_node_s_store
import requests


def query_node_status(rpc_url: str = "http://localhost:26657"):
    """
    Fetch `/status` from the local Tendermint RPC and return selected fields.
    """
    try:
        response = requests.get(f"{rpc_url}/status", timeout=5)
        response.raise_for_status()
        data = response.json()
        info = data["result"]
        return {
            "network": info["node_info"]["network"],
            "latest_height": int(info["sync_info"]["latest_block_height"]),
            "catching_up": info["sync_info"]["catching_up"]
        }
    except Exception as err:
        raise RuntimeError(f"Failed to query node status: {err}")


# step:2 file: withdraw_10_%_of_the_user’s_shares_from_the_maxbtc_solvbtc_supervault
from fastapi import APIRouter, HTTPException
import os
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.utils import NETWORKS

router = APIRouter()

SUPER_VAULT_CONTRACT = os.getenv('SUPER_VAULT_CONTRACT')  # e.g. 'neutron1abc...'
CHAIN_ID = os.getenv('CHAIN_ID', 'neutron-1')

@router.get('/supervault/share-balance')
async def get_supervault_share_balance(address: str):
    """Return the amount of Supervault shares owned by `address`."""
    try:
        if not SUPER_VAULT_CONTRACT:
            raise ValueError('SUPER_VAULT_CONTRACT env var not set')

        # Connect to public Neutron endpoints
        client = LedgerClient(NETWORKS[CHAIN_ID])

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


# step:4 file: withdraw_10_%_of_the_user’s_shares_from_the_maxbtc_solvbtc_supervault
import os, base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NETWORKS
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.contract import MsgExecuteContract

router = APIRouter()
CHAIN_ID = os.getenv('CHAIN_ID', 'neutron-1')
SUPER_VAULT_CONTRACT = os.getenv('SUPER_VAULT_CONTRACT')

class PrepareWithdrawRequest(BaseModel):
    address: str
    shares_to_withdraw: int

class PrepareWithdrawResponse(BaseModel):
    body_bytes: str
    auth_info_bytes: str
    account_number: int
    chain_id: str

@router.post('/supervault/prepare-withdraw', response_model=PrepareWithdrawResponse)
async def prepare_withdraw(req: PrepareWithdrawRequest):
    """Returns components required for a DIRECT-SIGN transaction."""
    try:
        if not SUPER_VAULT_CONTRACT:
            raise ValueError('SUPER_VAULT_CONTRACT env var not set')

        ledger = LedgerClient(NETWORKS[CHAIN_ID])
        acct = ledger.query_account(req.address)

        # Contract-level execute message
        execute_msg = {
            'withdraw': {
                'shares': str(req.shares_to_withdraw)
            }
        }

        msg = MsgExecuteContract(
            sender=req.address,
            contract_address=SUPER_VAULT_CONTRACT,
            msg=execute_msg,
            funds=[],
        )

        tx = (
            Transaction()
            .with_messages(msg)
            .with_sequence(acct.sequence)
            .with_account_number(acct.account_number)
            .with_chain_id(CHAIN_ID)
            .with_gas(300000)
            .with_fee('2000untrn')  # Adjust fee & gas to your needs
        )

        body_bytes, auth_info_bytes, _ = tx.to_sign_doc()

        return PrepareWithdrawResponse(
            body_bytes=base64.b64encode(body_bytes).decode(),
            auth_info_bytes=base64.b64encode(auth_info_bytes).decode(),
            account_number=acct.account_number,
            chain_id=CHAIN_ID,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:6 file: withdraw_10_%_of_the_user’s_shares_from_the_maxbtc_solvbtc_supervault
import base64, os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from cosmpy.protos.cosmos.tx.v1beta1.tx_pb2 import TxRaw
from cosmpy.aerial.client import LedgerClient, NETWORKS

router = APIRouter()
CHAIN_ID = os.getenv('CHAIN_ID', 'neutron-1')

class BroadcastWithdrawRequest(BaseModel):
    body_bytes: str
    auth_info_bytes: str
    signature: str

@router.post('/supervault/broadcast-withdraw')
async def broadcast_withdraw(req: BroadcastWithdrawRequest):
    """Takes the signed tx fragments, creates TxRaw, broadcasts, and returns the tx-hash."""
    try:
        body_bytes = base64.b64decode(req.body_bytes)
        auth_info_bytes = base64.b64decode(req.auth_info_bytes)
        signature = base64.b64decode(req.signature)

        tx_raw = TxRaw(
            body_bytes=body_bytes,
            auth_info_bytes=auth_info_bytes,
            signatures=[signature],
        )

        ledger = LedgerClient(NETWORKS[CHAIN_ID])
        result = ledger.broadcast_tx(tx_raw.SerializeToString())

        if result.code != 0:
            raise ValueError(f'Tx failed with code {result.code}: {result.raw_log}')

        return {'txhash': result.txhash}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:3 file: view_available_supervault_positions_eligible_for_bitcoin_summer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.client import LedgerClient, NetworkConfig

app = FastAPI()

class PositionsRequest(BaseModel):
    user_address: str
    contract_address: str

@app.post('/api/supervault/positions')
async def supervault_positions(req: PositionsRequest):
    """Query Supervault for user positions via WASM smart-contract call."""
    try:
        # Public Neutron main-net endpoints (no secrets required)
        cfg = NetworkConfig(
            chain_id='neutron-1',
            lcd_url='https://rest-kralum.neutron-1.neutron.org',
            grpc_url='grpc://grpc-kralum.neutron-1.neutron.org:443'
        )

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


# step:1 file: generate_and_wire_up_a_default_simulation_manager
// tools/sim.go
//go:build tools
// +build tools

// This file guarantees simulation packages are kept in the module graph even
// though they are not referenced in production code.
package tools

import (
    _ "github.com/cosmos/cosmos-sdk/x/auth/simulation"
    _ "github.com/cosmos/cosmos-sdk/x/bank/simulation"
    // _ "github.com/cosmos/cosmos-sdk/x/staking/simulation" // add more as needed
)



# step:2 file: generate_and_wire_up_a_default_simulation_manager
// app/simulation.go
package app

import (
    "github.com/cosmos/cosmos-sdk/types/module"
    authmodule "github.com/cosmos/cosmos-sdk/x/auth"
    bankmodule "github.com/cosmos/cosmos-sdk/x/bank"
    // import additional modules here
)

// newSimulationManager assembles the SimulationManager once and caches it.
func (app *App) newSimulationManager() *module.SimulationManager {
    if app.sm != nil {
        return app.sm
    }

    simManager := module.NewSimulationManager(
        authmodule.NewAppModule(app.AppCodec(), app.AccountKeeper, nil),
        bankmodule.NewAppModule(app.AppCodec(), app.BankKeeper, app.AccountKeeper),
        // add further AppModule instances here
    )

    app.sm = simManager
    return simManager
}



# step:3 file: generate_and_wire_up_a_default_simulation_manager
// app/simulation.go (continued)

func (app *App) registerStoreDecoders() {
    if app.sm == nil {
        panic("simulation manager has not been initialised")
    }

    // This enables pretty-printing of store keys during simulation runs.
    app.sm.RegisterStoreDecoders()
}



# step:4 file: generate_and_wire_up_a_default_simulation_manager
// app/app.go (excerpt)

func NewApp(/* existing params */) *App {
    // ... keeper and module wiring ...

    // Build & attach the simulation manager.
    simMgr := app.newSimulationManager()
    app.registerStoreDecoders()
    app.SetSimulationManager(simMgr)

    return app
}



# step:5 file: generate_and_wire_up_a_default_simulation_manager
#!/usr/bin/env bash
# Run application simulations with verbose output

go test ./... -run TestFullAppSimulation -v



# step:1 file: launch_a_simd_node_with_an_unlimited_mempool_(max-txs_=_-1)
# backend/operations/simd_checks.py

import shutil
import subprocess
from fastapi import APIRouter, HTTPException

router = APIRouter()

def _check_simd_binary():
    '''Verify that the `simd` binary exists in the user PATH and is executable.'''
    simd_path = shutil.which("simd")
    if simd_path is None:
        raise FileNotFoundError("`simd` binary not found in your PATH. Have you installed simd?")
    try:
        result = subprocess.run(
            [simd_path, "version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"`simd` binary found but not executable: {exc}")
    return {"simd_path": simd_path, "version": result.stdout.strip()}

@router.get("/api/check_simd")
async def check_simd():
    '''HTTP endpoint that returns the path and version of the `simd` binary.'''
    try:
        return _check_simd_binary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:3 file: launch_a_simd_node_with_an_unlimited_mempool_(max-txs_=_-1)
# backend/operations/simd_start.py

import os
import shlex
from fastapi import APIRouter, HTTPException

router = APIRouter()

def _construct_start_command(home_dir: str) -> str:
    '''Return a safe string for starting simd with custom flags.'''
    if not home_dir:
        raise ValueError('home_dir parameter is required')
    expanded_home = os.path.expanduser(home_dir)
    # Ensure path exists
    if not os.path.isdir(expanded_home):
        raise FileNotFoundError(f'Provided home directory does not exist: {expanded_home}')
    # Quote the directory to avoid shell injection
    quoted_home = shlex.quote(expanded_home)
    return f'simd start --mempool.max-txs=-1 --home={quoted_home}'

@router.post('/api/build_start_cmd')
async def build_start_cmd(payload: dict):
    '''Endpoint that returns the full `simd start` command string.'''
    try:
        home_dir = payload.get('home')
        cmd = _construct_start_command(home_dir)
        return {'command': cmd}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# step:4 file: launch_a_simd_node_with_an_unlimited_mempool_(max-txs_=_-1)
# backend/operations/simd_process.py

import shlex
import subprocess
from fastapi import APIRouter, HTTPException

router = APIRouter()
_process_holder = {'proc': None}

def _start_simd_process(command: str):
    if _process_holder['proc'] is not None:
        raise RuntimeError('A simd process is already running.')
    args = shlex.split(command)
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    _process_holder['proc'] = proc
    return {'pid': proc.pid}

@router.post('/api/start_simd')
async def start_simd(payload: dict):
    try:
        command = payload.get('command')
        return _start_simd_process(command)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:5 file: launch_a_simd_node_with_an_unlimited_mempool_(max-txs_=_-1)
# backend/operations/simd_logs.py

import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from .simd_process import _process_holder

router = APIRouter()

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

@router.get('/api/simd_logs')
async def simd_logs():
    return StreamingResponse(_log_streamer(), media_type='text/event-stream')


# step:6 file: launch_a_simd_node_with_an_unlimited_mempool_(max-txs_=_-1)
# backend/operations/simd_status.py

import subprocess
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()

def _cli_status():
    res = subprocess.run(['simd', 'status'], capture_output=True, text=True, check=True)
    return res.stdout

def _rpc_status(rpc_url: str):
    r = requests.get(f'{rpc_url.rstrip('/')}/status')
    r.raise_for_status()
    return r.json()

@router.get('/api/simd_status')
async def simd_status(rpc: str = 'http://localhost:26657'):
    try:
        try:
            return _rpc_status(rpc)
        except Exception:
            # Fallback to CLI if RPC fails
            return {'raw': _cli_status()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:2 file: instantiate_the_contract_with_code_id_13_and_init_message_{_count_:0}
########################  instantiate.py  ########################
"""Backend-for-Frontend (BFF) service that instantiates a CosmWasm contract.

Exposes POST /instantiate which expects:
  {
    "code_id": 13,
    "init_msg": "{\"count\":0}",
    "label": "counter-v1",
    "admin": "juno1..."        // optional, may be null/empty
  }
Returns JSON { "tx_hash": "..." }
"""
import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgInstantiateContract
from cosmpy.aerial.tx import Transaction

###########################################################################
# Network / key setup ——— use ENV so secrets never reach the frontend      #
###########################################################################

RPC_ENDPOINT = os.getenv("JUNO_RPC", "https://rpc.juno.omniflix.co:443")
CHAIN_ID = os.getenv("JUNO_CHAIN", "juno-1")
FEE_DENOM = os.getenv("FEE_DENOM", "ujuno")
GAS_PRICE = float(os.getenv("GAS_PRICE", "0.075"))  # ujuno per gas unit

# Load the mnemonic from an environment variable (⚠️ NEVER check secrets in)
MNEMONIC = os.getenv("BACKEND_MNEMONIC")
if MNEMONIC is None:
    raise RuntimeError("Missing BACKEND_MNEMONIC environment variable")

wallet = LocalWallet.create_from_mnemonic(MNEMONIC)

cfg = NetworkConfig(
    chain_id=CHAIN_ID,
    url=RPC_ENDPOINT,
    fee_minimum_gas_price=f"{GAS_PRICE}{FEE_DENOM}",
    fee_denomination=FEE_DENOM,
)
client = LedgerClient(cfg)

###########################################################################
# FastAPI Schemas                                                          #
###########################################################################

class InstantiatePayload(BaseModel):
    code_id: int = Field(..., example=13)
    init_msg: str = Field(..., description="JSON string of the instantiate message")
    label: str = Field(..., example="counter-v1")
    admin: Optional[str] = Field(None, description="(optional) admin address")

    @validator("init_msg")
    def init_msg_must_be_valid_json(cls, v):
        try:
            json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"init_msg is not valid JSON: {e}") from e
        return v

app = FastAPI()

###########################################################################
# Helper: Build & broadcast instantiate tx                                 #
###########################################################################

def _broadcast_instantiate(payload: InstantiatePayload) -> str:
    """Builds, signs, and broadcasts a MsgInstantiateContract. Returns TX hash."""

    # Convert JSON string → bytes for protobuf msg field
    init_bytes = json.dumps(json.loads(payload.init_msg)).encode()

    # Build proto message --------------------------------------------------
    msg = MsgInstantiateContract(
        sender=wallet.address(),
        admin=payload.admin or "",
        code_id=payload.code_id,
        label=payload.label,
        msg=init_bytes,
        funds=[],  # no native tokens sent along
    )

    # Create TX wrapper ----------------------------------------------------
    tx = Transaction()
    tx.add_message(msg)
    tx.seal(wallet)  # add signer info & account sequence

    # Estimate gas automatically (client-side simulation)
    gas_estimate = client.estimate_gas(tx)
    tx.set_gas(gas_estimate * 130 // 100)  # 30% safety margin

    # Set fee based on gas x gas_price
    fee_amount = int(tx.gas_limit * GAS_PRICE)
    tx.set_fee([(fee_amount, FEE_DENOM)])

    # Sign & broadcast -----------------------------------------------------
    tx.sign(wallet)
    resp = client.broadcast(tx)

    if resp.tx_response.code != 0:
        raise RuntimeError(f"Tx failed: {resp.tx_response.raw_log}")

    return resp.tx_response.txhash

###########################################################################
# Route                                                                    #
###########################################################################

@app.post("/instantiate")
async def instantiate_contract(payload: InstantiatePayload):
    try:
        tx_hash = _broadcast_instantiate(payload)
        return {"tx_hash": tx_hash}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# step:3 file: instantiate_the_contract_with_code_id_13_and_init_message_{_count_:0}
########################  tx_status.py  ########################
"""Wait for a TX to commit and return the contract address (if any)."""
import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException

router = APIRouter()

# Re-use `client` from instantiate.py via import (singleton pattern)
from instantiate import client  # noqa: E402

# Constants --------------------------------------------------------------
POLL_INTERVAL = 2.0   # seconds
MAX_BLOCKS = 15       # safety limit (~1 min on Juno)

async def _extract_contract_address(tx_hash: str) -> Optional[str]:
    """Poll the node until TX found in a block, then parse logs for _contract_address."""
    for _ in range(MAX_BLOCKS):
        tx = client.tx(tx_hash)
        if tx is not None and tx.tx_response and tx.tx_response.height > 0:
            # TX was found — parse logs
            try:
                logs = json.loads(tx.tx_response.raw_log)
            except json.JSONDecodeError:
                raise RuntimeError("Cannot decode tx logs")

            for event in logs[0].get("events", []):
                if event.get("type") in ("instantiate", "instantiate_contract", "wasm"):  # chain-specific variants
                    for attr in event.get("attributes", []):
                        if attr.get("key") in ("_contract_address", "contract_address"):
                            return attr.get("value")
            return None  # tx succeeded but no address (unexpected)
        await asyncio.sleep(POLL_INTERVAL)
    raise RuntimeError("Timed out waiting for transaction to be included in a block")

@router.get("/tx_status/{tx_hash}")
async def find_block(timestamp: int, rpc_url: str):
    """Return the block number closest to a given Unix timestamp."""

    # Basic sanity-checks -----------------------------------------------------
    if timestamp <= 0:
        raise HTTPException(status_code=400, detail='Timestamp must be > 0')
    if not re.match(r'^https?://', rpc_url):
        raise HTTPException(status_code=400, detail='rpc_url must start with http(s)://')

    # Build the cast command --------------------------------------------------
    cmd = [
        'cast', 'find-block',
        '--timestamp', str(timestamp),
        '--rpc-url', rpc_url.strip()
    ]

    try:
        # Run the process with a 10-second safety timeout
        proc = run(cmd, capture_output=True, text=True, check=True, timeout=10)
        block_no_str = proc.stdout.strip()

        # cast normally returns a decimal block number, but guard against hex
        try:
            block_no = int(block_no_str, 0)
        except ValueError:
            block_no = block_no_str  # keep as string if it cannot be parsed

        return JSONResponse({
            'timestamp': timestamp,
            'rpc_url': rpc_url,
            'block_number': block_no
        })

    except CalledProcessError as e:
        # CLI returned non-zero exit status
        raise HTTPException(status_code=500, detail=f"cast error: {e.stderr.strip()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:1 file: configure_hardhat_networks_for_cosmos_evm_in_hardhat.config.ts
/* hardhat.config.ts */
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox"; // Hardhat plugins & ethers helpers
import * as dotenv from "dotenv";

dotenv.config(); // Loads .env variables into process.env

/*
 * Replace the solidity version or any other default fields to fit your
 * project. Only the networks.cosmos section is strictly required for
 * this workflow.
 */
const config: HardhatUserConfig = {
  solidity: "0.8.17",
  networks: {
    /*
     * Cosmos-SDK EVM compatible chain.
     * - url:   Public or private RPC endpoint
     * - chainId: Integer EVM chain-id (not the Cosmos-SDK chain-id)
     * - accounts: List of private keys that Hardhat will use to sign txs.
     */
    cosmos: {
      url: process.env.COSMOS_RPC || "https://rpc.cosmos.network",
      chainId: Number(process.env.COSMOS_CHAIN_ID) || 118,
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []
    }
    // ...existing networks can stay untouched
  }
};

export default config;


# step:2 file: configure_hardhat_networks_for_cosmos_evm_in_hardhat.config.ts
#!/usr/bin/env bash
# install_deps.sh – run once
set -euo pipefail

# Installs peer-dependencies as devDependencies (-D)
# so they don't ship with production builds.

npm i -D @nomicfoundation/hardhat-toolbox dotenv


# step:3 file: configure_hardhat_networks_for_cosmos_evm_in_hardhat.config.ts
#!/usr/bin/env bash
# compile.sh
set -euo pipefail

npx hardhat compile


# step:4 file: configure_hardhat_networks_for_cosmos_evm_in_hardhat.config.ts
/* scripts/blockNumber.ts */
import { ethers } from "hardhat";

async function main() {
  try {
    // ethers.provider is automatically configured to use the selected network
    const blockNumber = await ethers.provider.getBlockNumber();
    console.log(`Current block number on '${ethers.provider.network.name}':`, blockNumber);
  } catch (err) {
    console.error("Failed to fetch block number:", err);
    process.exit(1);
  }
}

main();


# step:5 file: configure_hardhat_networks_for_cosmos_evm_in_hardhat.config.ts
#!/usr/bin/env bash
# healthcheck.sh – exits non-zero if either compilation or RPC call fail
set -euo pipefail

echo "[1/2] ▶ Compiling workspace" && npx hardhat compile

echo "[2/2] ▶ Pinging Cosmos RPC for latest block number" && npx hardhat run --network cosmos scripts/blockNumber.ts


# step:1 file: run_fuzz_tests_with_forge_specifying_the_number_of_fuzz_runs
import subprocess
from typing import Dict

def compile_tests() -> Dict[str, str]:
    """Run `forge build` to compile the project’s contracts.

    Returns
    -------
    Dict[str, str]
        success : bool   - True if the build succeeded (exit-code 0).
        stdout  : str    - Standard output from the CLI.
        stderr  : str    - Standard error from the CLI (compiler warnings / errors).
    """
    try:
        result = subprocess.run(
            ["forge", "build"],
            capture_output=True,
            text=True,
            check=False  # Allow us to capture non-zero exit codes instead of raising.
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except FileNotFoundError:
        # The user does not have Foundry installed or forge is not in PATH.
        return {
            "success": False,
            "stdout": "",
            "stderr": "Forge CLI not found. Please install Foundry and ensure `forge` is available in PATH.",
        }
    except Exception as exc:
        # Catch-all for any other unexpected error.
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
        }


# step:2 file: run_fuzz_tests_with_forge_specifying_the_number_of_fuzz_runs
import subprocess
from typing import Dict

def execute_forge_test_fuzz(fuzz_runs: int = 500) -> Dict[str, str]:
    """Run `forge test --fuzz-runs <N>`.

    Parameters
    ----------
    fuzz_runs : int
        Number of fuzz iterations you wish to run. Must be > 0.

    Returns
    -------
    Dict[str, str]
        success : bool   - True if all tests passed (exit-code 0).
        stdout  : str    - Standard output of the test run.
        stderr  : str    - Standard error of the test run.
    """
    if fuzz_runs <= 0:
        return {
            "success": False,
            "stdout": "",
            "stderr": "`fuzz_runs` must be a positive integer.",
        }

    cmd = ["forge", "test", "--fuzz-runs", str(fuzz_runs)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Forge CLI not found. Please install Foundry and ensure `forge` is available in PATH.",
        }
    except Exception as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
        }


# step:3 file: run_fuzz_tests_with_forge_specifying_the_number_of_fuzz_runs
import re
from typing import Dict, List

def review_test_output(forge_stdout: str) -> Dict[str, object]:
    """Analyse `forge test` output and summarise failures.

    Parameters
    ----------
    forge_stdout : str
        Raw standard-output from a Forge test run.

    Returns
    -------
    Dict[str, object]
        failed      : bool     - True if any test failed.
        fail_cases  : List[str]- Lines that hint at failures or reverts.
        raw_output  : str      - Echo back the original output for reference.
    """
    failure_regex = re.compile(r"(FAIL|revert|Assertion|Error)", re.IGNORECASE)
    fail_cases: List[str] = []

    for line in forge_stdout.splitlines():
        if failure_regex.search(line):
            fail_cases.append(line.strip())

    return {
        "failed": len(fail_cases) > 0,
        "fail_cases": fail_cases,
        "raw_output": forge_stdout,
    }


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


# step:1 file: List all existing cron schedules
import requests


def query_cron_list_schedules(node_url: str = "https://rest.kralum.neutron-1.neutron.org") -> list:
    """Fetch the list of cron schedules from a Neutron REST endpoint.

    Args:
        node_url: Base URL of the Neutron REST API (without a trailing slash).

    Returns:
        A list of schedule objects returned by the chain.

    Raises:
        RuntimeError: If the request fails or the response cannot be parsed.
    """
    # Ensure the base URL has no trailing slash to avoid errors when composing the endpoint.
    base_url = node_url.rstrip("/")
    endpoint = f"{base_url}/neutron/cron/schedules"

    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to query cron schedules: {exc}") from exc

    try:
        json_data = response.json()
    except ValueError as exc:
        raise RuntimeError("Received invalid JSON from the Neutron REST endpoint") from exc

    # The REST API generally returns `{ \"schedules\": [...] }`.
    return json_data.get("schedules", [])



# step:4 file: Remove the cron schedule named "daily_rewards"
import asyncio
from cosmpy.aerial.client import LedgerClient, NetworkConfig

NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url="https://rpc-kralum.neutron.org",
)
client = LedgerClient(NETWORK)

async def monitor_proposal_status(proposal_module: str, proposal_id: int, poll: int = 10):
    """Waits until the proposal passes & is executed."""
    while True:
        try:
            state = client.query_contract_state_smart(
                proposal_module,
                {"proposal": {"proposal_id": proposal_id}}
            )
            status = state.get("proposal", {}).get("status")
            if status in ("passed", "executed"):
                return state
            print(f"Proposal {proposal_id} still {status}; sleeping {poll}s …")
        except Exception as err:
            print(f"Error while querying proposal: {err}")
        await asyncio.sleep(poll)


# step:5 file: Remove the cron schedule named "daily_rewards"
import subprocess

def schedule_removed(name: str) -> bool:
    """Returns True only if the schedule no longer exists."""
    try:
        # neutrond must be in $PATH and already configured for neutron-1
        out = subprocess.run(
            [
                "neutrond",
                "query",
                "cron",
                "show-schedule",
                name,
                "--output=json"
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # If the command succeeds, the schedule is still present
        print(f"Schedule still exists: {out.stdout}")
        return False
    except subprocess.CalledProcessError as err:
        # Cron module returns non-zero + "not found" when the schedule is gone
        if "not found" in err.stderr.lower():
            return True
        raise


# step:3 file: List all smart contracts deployed by my account
"""query_contracts.py
Python utility for querying contracts by creator via Neutron's LCD (REST) endpoint.
Requires the `httpx` dependency:  pip install httpx[http2]
"""
from typing import Optional, Dict, Any
import httpx

LCD_URL = "https://lcd.neutron.org"  # Change to your preferred public or self-hosted LCD

async def query_contracts_by_creator(
    creator_address: str,
    limit: int = 1000,
    pagination_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Query one page of contracts created by `creator_address`.

    Args:
        creator_address (str): Bech32 Neutron address.
        limit (int, optional): Maximum results per page. Defaults to 1000.
        pagination_key (str, optional): The opaque `next_key` from the previous
            response. If provided, the query continues from that key.

    Returns:
        Dict[str, Any]: JSON response from the LCD containing contracts and pagination data.
    """
    params = {
        "creator": creator_address,
        "pagination.limit": str(limit),
    }
    if pagination_key:
        params["pagination.key"] = pagination_key

    url = f"{LCD_URL}/cosmwasm/wasm/v1/contracts"

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()  # Raises if HTTP != 200
        return response.json()



# step:4 file: List all smart contracts deployed by my account
"""pagination_helper.py
Combines all pages from the `query_contracts_by_creator` helper.
"""
from typing import List, Dict, Any, Optional
import asyncio
from query_contracts import query_contracts_by_creator

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



# step:2 file: publish_a_raw_signed_transaction_with_cast_publish
from fastapi import FastAPI, HTTPException, Request
import os
import requests

app = FastAPI()

# RPC endpoint for an EVM-compatible Cosmos chain (e.g., Evmos, Cronos, etc.)
RPC_URL = os.getenv('COSMOS_RPC_URL', 'https://evmos-evm.publicnode.com')

@app.post('/api/broadcast_raw_tx')
async def broadcast_raw_tx(request: Request):
    """Broadcast an already-signed RLP-encoded transaction and return its tx-hash."""
    body = await request.json()
    raw_tx = body.get('raw_tx')

    # Basic validation.
    if not raw_tx or not raw_tx.startswith('0x'):
        raise HTTPException(status_code=400, detail='Invalid raw transaction hex string.')

    # Compose the JSON-RPC payload.
    payload = {
        'jsonrpc': '2.0',
        'method': 'eth_sendRawTransaction',
        'params': [raw_tx],
        'id': 1
    }

    try:
        rpc_resp = requests.post(RPC_URL, json=payload, timeout=30)
        rpc_resp.raise_for_status()
        data = rpc_resp.json()

        # Handle any RPC-level error returned by the node.
        if 'error' in data:
            raise HTTPException(status_code=500, detail=data['error']['message'])

        # Success: return the tx-hash to the caller.
        return {'tx_hash': data['result']}
    except requests.RequestException as exc:
        # Network-level failure.
        raise HTTPException(status_code=500, detail=str(exc))


# step:4 file: publish_a_raw_signed_transaction_with_cast_publish
from fastapi import HTTPException
import time

@app.get('/api/tx_receipt/{tx_hash}')
def update_node_start_flags(service_name: str = SERVICE_NAME, flag: str = CPU_PROFILE_FLAG):
    """Append a CPU-profiling flag to the ExecStart line of a systemd service.

    1. Backs up the original unit file to `/etc/systemd/system/<service>.bak`.
    2. Appends the profiling flag if it is not already present.
    3. Reloads systemd so the change takes effect.
    """
    unit_path = Path("/etc/systemd/system") / service_name
    backup_path = unit_path.with_suffix(".bak")

    if not unit_path.exists():
        raise FileNotFoundError(f"Service file {unit_path} not found. Run on the host where the node is installed.")

    # Backup first
    if not backup_path.exists():
        shutil.copy(unit_path, backup_path)

    # Read & update
    with unit_path.open("r") as f:
        lines = f.readlines()

    updated_lines = []
    exec_found = False
    for line in lines:
        if line.startswith("ExecStart="):
            exec_found = True
            if flag not in line:
                # Insert the profiling flag just before the linebreak
                parts = line.strip().split(" ")
                parts.insert(-1, flag)
                line = " ".join(parts) + "\n"
        updated_lines.append(line)

    if not exec_found:
        raise ValueError("No ExecStart entry found in service file. Manual intervention required.")

    with unit_path.open("w") as f:
        f.writelines(updated_lines)

    # Reload systemd
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    print(f"[✔] {service_name} updated with CPU profiling flag.")

# Example direct call (uncomment when running as root)
# update_node_start_flags()


# step:2 file: start_indefinite_cpu_profiling_to_cpu.prof
import subprocess

SERVICE_NAME = "cosmosd.service"  # keep consistent with step-1


def restart_node(service_name: str = SERVICE_NAME):
    """Safely restarts the cosmos node service."""
    try:
        subprocess.run(["systemctl", "restart", service_name], check=True)
        subprocess.run(["systemctl", "is-active", "--quiet", service_name], check=True)
        print(f"[✔] {service_name} restarted successfully.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to restart {service_name}: {e}")

# Example direct call
# restart_node()


# step:3 file: start_indefinite_cpu_profiling_to_cpu.prof
import os
import time
from pathlib import Path

PROFILE_PATH = Path.home() / ".cosmos" / "cpu.prof"  # adjust path if different


def check_profile_growth(profile_path: Path = PROFILE_PATH, wait_seconds: int = 5):
    """Checks that the profile file exists and increases in size over a short interval."""
    if not profile_path.exists():
        raise FileNotFoundError(f"{profile_path} does not exist. Did you enable profiling?")

    size_before = profile_path.stat().st_size
    time.sleep(wait_seconds)
    size_after = profile_path.stat().st_size

    if size_after > size_before:
        print(f"[✔] cpu.prof is growing (size {size_before} → {size_after} bytes).")
    else:
        raise RuntimeError("cpu.prof did not grow; profiling may not be active.")

# Example direct call
# check_profile_growth()


# step:2 file: make_a_raw_json-rpc_call_with_cast_rpc
# rpc_backend.py
import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()

RPC_URL = os.getenv("RPC_URL", "https://mainnet.infura.io/v3/YOUR_INFURA_KEY")

class RpcRequest(BaseModel):
    method: str
    params: list

@app.post("/api/rpc")
async def proxy_rpc(request: RpcRequest):
    # Build a standard JSON-RPC 2.0 payload
    payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": request.method,
        "params": request.params,
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(RPC_URL, json=payload, timeout=15)
            resp.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            # Surface network/HTTP errors to the caller.
            raise HTTPException(status_code=502, detail=str(e))

    # Forward the JSON-RPC response (success or error) verbatim.
    return resp.json()



# step:1 file: fetch_the_current_chain_id_using_cast
from fastapi import APIRouter, HTTPException
import subprocess

router = APIRouter()

@router.get('/api/verify_foundry')
async def verify_foundry_installation():
    """Check if the `cast` executable is available in $PATH and return its version."""
    try:
        result = subprocess.run(
            ['cast', '--version'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return {
            'installed': True,
            'version': result.stdout.strip(),
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Foundry 'cast' executable was not found in $PATH."
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail='Timeout while attempting to execute cast.'
        )
    except subprocess.CalledProcessError as err:
        raise HTTPException(
            status_code=500,
            detail=f'Error executing cast: {err.stderr}'
        )


# step:2 file: fetch_the_current_chain_id_using_cast
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Static mapping of chain → RPC; customise as required.
RPC_ENDPOINTS = {
    'evmos': 'https://eth.bd.evmos.org:8545',
    'canto': 'https://canto.slingshot.finance',
    'injective': 'https://ethereum.injective.network',
    'ki': 'https://rpc-mainnet.ki.chainstacklabs.com',
}

@router.get('/api/get_rpc_url')
async def get_rpc_url(chain: str):
    """Return a pre-configured JSON-RPC endpoint for the requested chain."""
    if chain not in RPC_ENDPOINTS:
        raise HTTPException(status_code=400, detail=f'Unsupported chain: {chain}')
    return {
        'chain': chain,
        'rpc_url': RPC_ENDPOINTS[chain],
    }


# step:3 file: fetch_the_current_chain_id_using_cast
from fastapi import APIRouter, HTTPException
import subprocess

router = APIRouter()

@router.get('/api/get_chain_id')
async def get_chain_id(rpc_url: str):
    """Return the EVM chain-ID reported by the given RPC URL using `cast`."""
    cmd = ['cast', 'chain-id', '--rpc-url', rpc_url]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        # `cast chain-id` prints the ID as plain text; convert to int for clarity.
        chain_id = int(result.stdout.strip())
        return {
            'rpc_url': rpc_url,
            'chain_id': chain_id,
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Foundry 'cast' executable was not found in $PATH."
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail='Timeout while querying the chain-ID.'
        )
    except subprocess.CalledProcessError as err:
        raise HTTPException(
            status_code=500,
            detail=f'Failed to fetch chain-ID: {err.stderr}'
        )
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail='Non-numeric chain-ID returned by cast.'
        )


# step:1 file: trigger_prepareproposal_handler_override
import requests
from typing import Dict, Any

COSMOS_DOCS_SEARCH_URL = "https://docs.cosmos.network/api/search"  # fictitious public endpoint for illustration

def search_cosmos_docs(query: str) -> Dict[str, Any]:
    """Search the Cosmos docs site and return JSON results.

    Args:
        query (str): Free-form search text.

    Returns:
        Dict[str, Any]: JSON response containing documentation matches or an error description.
    """
    try:
        resp = requests.get(COSMOS_DOCS_SEARCH_URL, params={"q": query}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        # Gracefully surface the failure to callers
        return {"error": str(exc), "results": []}


# step:2 file: trigger_prepareproposal_handler_override
// File: app/prepare_proposal.go
package app

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
    abci "github.com/cometbft/cometbft/abci/types"
)

// PrepareProposal overrides the default BaseApp.PrepareProposal implementation.
// It illustrates how to inspect and modify the list of TXs before the block
// proposal is finalised by CometBFT.
func (app *App) PrepareProposal(
    ctx sdk.Context,
    req *abci.RequestPrepareProposal,
) (*abci.ResponsePrepareProposal, error) {
    ctx.Logger().Info("[custom] PrepareProposal invoked", "txs", len(req.Txs))

    // ————————————————————————————————————————————————
    // Example Rule: Keep only the first 100 txs to cap block size.
    // ————————————————————————————————————————————————
    const maxTxs = 100
    trimmedTxs := req.Txs
    if len(req.Txs) > maxTxs {
        trimmedTxs = req.Txs[:maxTxs]
        ctx.Logger().Info("[custom] Trimmed transactions", "kept", maxTxs, "discarded", len(req.Txs)-maxTxs)
    }

    // Return the proposal with our (possibly) modified list of txs.
    return &abci.ResponsePrepareProposal{Txs: trimmedTxs}, nil
}


# step:3 file: trigger_prepareproposal_handler_override
// File: app/app.go  (excerpt)

// … existing imports …
import (
    baseapp "github.com/cosmos/cosmos-sdk/baseapp"
)

func NewApp(/* existing params */) *App {
    // Instantiate the BaseApp and other keepers as usual.
    app := &App{
        BaseApp: baseapp.NewBaseApp(
            appName,
            logger,
            db,
            encodingConfig.TxConfig,
            // … other opts …
        ),
        // … other struct fields …
    }

    // ————————————————————————————————————————————————
    // Register the override *before* mounting stores/startup.
    // ————————————————————————————————————————————————
    app.SetPrepareProposal(app.PrepareProposal)

    // … rest of NewApp (mount stores, register modules, etc.) …

    return app
}


# step:4 file: trigger_prepareproposal_handler_override
import subprocess, sys

def compile_application():
    """Rebuild the chain binary with the freshly-added ABCI override."""
    try:
        # `make install` compiles and copies the binary into $GOBIN.
        subprocess.run(["make", "install"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("✅  Binary compiled and installed to $GOBIN.")
    except subprocess.CalledProcessError as exc:
        print("❌  Compilation failed:\n", exc.stdout.decode())
        sys.exit(1)

if __name__ == "__main__":
    compile_application()


# step:5 file: trigger_prepareproposal_handler_override
import os, subprocess, signal, time

BINARY = os.getenv("CHAIN_BIN", "mychaind")  # replace with your binary name
HOME = os.getenv("CHAIN_HOME", os.path.expanduser("~/.mychaind"))


def start_node():
    """Starts the node in a subprocess and streams logs to a file."""
    log_file = open("node.log", "w")
    try:
        # Initialise if the home directory does not yet exist
        if not os.path.isdir(HOME):
            subprocess.run([BINARY, "init", "local-test", "--chain-id", "local-1"], check=True)

        # Start the chain and write logs to file
        proc = subprocess.Popen(
            [BINARY, "start", "--home", HOME, "--log_format", "plain"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        print("🔄  Node started (PID %d). Producing blocks …" % proc.pid)
        time.sleep(10)  # Let it run long enough to mint a block.
    finally:
        # Cleanly terminate and close file
        proc.send_signal(signal.SIGINT)
        proc.wait()
        log_file.close()
        print("🛑  Node stopped; logs available in node.log")

if __name__ == "__main__":
    start_node()


# step:6 file: trigger_prepareproposal_handler_override
import re, sys

LOG_PATH = "node.log"
PATTERN = re.compile(r"Custom PrepareProposal invoked", re.IGNORECASE)


def monitor_logs():
    try:
        with open(LOG_PATH, "r") as fh:
            matches = [line.strip() for line in fh if PATTERN.search(line)]
        if matches:
            print("🎉  Found %d PrepareProposal invocations:" % len(matches))
            for ln in matches:
                print("   ", ln)
        else:
            print("⚠️   No PrepareProposal logs detected. Ensure the node ran long enough to produce a block and that logging level isn't filtering them out.")
    except FileNotFoundError:
        print("node.log not found. Did you run Step 5?")
        sys.exit(1)

if __name__ == "__main__":
    monitor_logs()


# step:2 file: decode_protobuf_transaction_bytes_to_json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json
import base64
import re

app = FastAPI()

class DecodeRequest(BaseModel):
    raw_tx: str  # base64-encoded TxRaw bytes

def is_valid_base64(data: str) -> bool:
    """Performs a strict base64 validation (length & charset)."""
    base64_regex = re.compile(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$")
    if not base64_regex.match(data):
        return False
    try:
        base64.b64decode(data, validate=True)
        return True
    except Exception:
        return False

@app.post("/api/decode_tx")
async def decode_tx(request: DecodeRequest):
    # Step 1: sanity-check inside backend for extra safety
    if not is_valid_base64(request.raw_tx):
        raise HTTPException(status_code=400, detail="Invalid base64 provided.")

    # Step 2: call `simd tx decode <base64>` via subprocess
    try:
        result = subprocess.run(
            ["simd", "tx", "decode", request.raw_tx],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="simd binary not found on server.")
    except subprocess.CalledProcessError as err:
        raise HTTPException(status_code=500, detail=f"simd error: {err.stderr.strip()}")

    # Step 3: ensure stdout is valid JSON
    try:
        decoded_json = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="simd returned non-JSON output.")

    return decoded_json


# step:1 file: list_local_snapshots_available_to_the_simd_node
# backend/snapshot_api.py

import os
from fastapi import FastAPI, HTTPException

app = FastAPI()

def get_node_home() -> str:
    """
    Determine the node's home directory.
    1. Use the `SIMD_HOME` environment variable if it exists.
    2. Otherwise, default to `~/.simapp`.
    """
    home = os.getenv("SIMD_HOME", os.path.expanduser("~/.simapp"))
    if not os.path.isdir(home):
        raise FileNotFoundError(f"Home directory not found: {home}")
    return home

@app.get("/api/get_node_home")
async def api_get_node_home():
    """Return `{ "home": "/path/to/home" }` or HTTP 500 on error."""
    try:
        return {"home": get_node_home()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:2 file: list_local_snapshots_available_to_the_simd_node
# backend/snapshot_api.py (continued)

import subprocess

@app.get("/api/list_snapshots")
async def api_list_snapshots(home: str):
    """Return `{ "raw_output": "<stdout>" }` containing the snapshot list."""
    try:
        raw_output = _list_snapshots(home)
        return {"raw_output": raw_output}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

def _list_snapshots(home: str) -> str:
    """Helper that actually executes the shell commands."""
    commands = [
        ["simd", "snapshot", "list", f"--home={home}"],
        ["ls", "-lh", f"{home}/data/snapshots"]
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout.strip():
                return result.stdout
        except FileNotFoundError:
            # Command not available; try next
            continue
        except subprocess.CalledProcessError:
            # Command failed; try next
            continue
    raise RuntimeError("Unable to list snapshots. Ensure `simd` is installed or snapshots directory exists.")


# step:3 file: list_local_snapshots_available_to_the_simd_node
# backend/snapshot_api.py (continued)

import re
from typing import List, Dict

SNAPSHOT_REGEX = re.compile(
    r"height[=:](?P<height>\d+)\s+format[=:](?P<format>\d+)\s+hash[=:]?\s*(?P<hash>[A-Fa-f0-9]+)",
    re.IGNORECASE,
)

@app.post("/api/parse_snapshots")
async def api_parse_snapshots(data: Dict[str, str]):
    """POST `{ "raw_output": "..." }` → `{ "snapshots": [...] }`."""
    try:
        raw_output = data.get("raw_output", "")
        snapshots = _parse_snapshot_output(raw_output)
        return {"snapshots": snapshots}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

def _parse_snapshot_output(raw_output: str) -> List[Dict]:
    """Convert CLI output into structured objects."""
    snapshots = []
    for line in raw_output.splitlines():
        match = SNAPSHOT_REGEX.search(line)
        if match:
            snapshots.append({
                "height": int(match.group("height")),
                "format": match.group("format"),
                "hash": match.group("hash")
            })
    if not snapshots:
        raise ValueError("No snapshots found in provided output.")
    return snapshots


# step:1 file: enable_only_the_eth,_web3,_net,_and_txpool_namespaces_on_json-rpc
from fastapi import APIRouter, HTTPException
import subprocess
import logging

router = APIRouter(prefix="/api/evmd")

@router.post("/stop")
async def stop_evmd():
    """
    Attempt to stop the evmd process gracefully. If systemd is not being used,
    fall back to killing the process by name.
    """
    try:
        # Preferred: managed by systemd
        subprocess.run(["systemctl", "stop", "evmd"], check=True)
        return {"status": "evmd stopped via systemctl"}
    except subprocess.CalledProcessError:
        try:
            # Fallback for non-systemd environments
            subprocess.run(["pkill", "-f", "evmd"], check=True)
            return {"status": "evmd stopped via pkill"}
        except subprocess.CalledProcessError as err:
            logging.exception("Unable to stop evmd: %s", err)
            raise HTTPException(status_code=500, detail="Unable to stop evmd process")


# step:2 file: enable_only_the_eth,_web3,_net,_and_txpool_namespaces_on_json-rpc
from fastapi import APIRouter, HTTPException
import os
import toml

router = APIRouter(prefix="/api/evmd/config")
APP_TOML_PATH = os.path.expanduser("~/.evmd/config/app.toml")

@router.get("/")
async def read_app_toml():
    """Return the full app.toml contents so it can be inspected in the UI."""
    try:
        with open(APP_TOML_PATH, "r") as file:
            cfg = toml.load(file)
        return cfg
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"app.toml not found at {APP_TOML_PATH}")
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:3 file: enable_only_the_eth,_web3,_net,_and_txpool_namespaces_on_json-rpc
from fastapi import APIRouter, HTTPException
import os
import toml
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/evmd/config")
APP_TOML_PATH = os.path.expanduser("~/.evmd/config/app.toml")

class Namespaces(BaseModel):
    namespaces: list[str] = Field(..., description="JSON-RPC namespaces to expose, e.g. ['eth','web3','net','txpool']")

@router.post("/json-rpc")
async def update_json_rpc(namespaces: Namespaces):
    """
    Overwrite the `api` field under the [json-rpc] section with a comma-separated
    list of namespaces supplied by the caller.
    """
    try:
        with open(APP_TOML_PATH, "r") as file:
            cfg = toml.load(file)

        # Ensure the section exists and update it
        cfg.setdefault("json-rpc", {})["api"] = ",".join(namespaces.namespaces)

        with open(APP_TOML_PATH, "w") as file:
            toml.dump(cfg, file)

        return {"status": "updated", "api": cfg["json-rpc"]["api"]}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:4 file: enable_only_the_eth,_web3,_net,_and_txpool_namespaces_on_json-rpc
from fastapi import APIRouter, HTTPException
import os
import toml

router = APIRouter(prefix="/api/evmd/config")
APP_TOML_PATH = os.path.expanduser("~/.evmd/config/app.toml")

@router.get("/json-rpc")
async def get_json_rpc_api():
    """Read back the api field to verify that the previous write persisted."""
    try:
        with open(APP_TOML_PATH, "r") as file:
            cfg = toml.load(file)
        return {"api": cfg.get("json-rpc", {}).get("api", "")}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# step:5 file: enable_only_the_eth,_web3,_net,_and_txpool_namespaces_on_json-rpc
from fastapi import APIRouter, HTTPException
import subprocess
import logging

router = APIRouter(prefix="/api/evmd")

@router.post("/start")
async def start_evmd():
    """
    Start (or restart) the evmd process after configuration changes.
    """
    try:
        subprocess.run(["systemctl", "start", "evmd"], check=True)
        return {"status": "evmd restarted"}
    except subprocess.CalledProcessError as err:
        logging.exception("Unable to start evmd: %s", err)
        raise HTTPException(status_code=500, detail="Unable to start evmd")


# step:1 file: query_the_bank_balance_of_the_address_associated_with_key_my_validator
import subprocess


def get_key_address(key_name: str, keyring_backend: str = "test") -> str:
    """Return the bech32 address for a key in the local key-ring."""
    try:
        cmd = [
            "simd", "keys", "show", key_name,
            "-a", "--keyring-backend", keyring_backend,
        ]
        address = subprocess.check_output(cmd, text=True).strip()
        if not address:
            raise ValueError("No address returned by simd command.")
        return address
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to retrieve address for {key_name}: {e}") from e


# step:2 file: query_the_bank_balance_of_the_address_associated_with_key_my_validator
import subprocess, json


def query_bank_balances(address: str) -> dict:
    """Run `simd q bank balances` and return the parsed JSON payload."""
    if not address:
        raise ValueError("Address is required.")
    try:
        cmd = [
            "simd", "q", "bank", "balances", address,
            "--output", "json",
        ]
        output = subprocess.check_output(cmd, text=True)
        return json.loads(output)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to query balances for {address}: {e}") from e


# step:3 file: query_the_bank_balance_of_the_address_associated_with_key_my_validator
def parse_balances(balances_json: dict) -> list:
    """Extract denomination and amount pairs from a simd balance query response."""
    if not isinstance(balances_json, dict):
        raise TypeError("balances_json must be a dict")
    tokens = []
    for coin in balances_json.get("balances", []):
        denom = coin.get("denom")
        amount = coin.get("amount")
        if denom and amount:
            tokens.append({"denom": denom, "amount": amount})
    return tokens


# step:2 file: provide_liquidity_to_the_maxbtc_unibtc_supervault_using_1_maxbtc_and_1_unibtc
from fastapi import FastAPI, HTTPException
import os
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from pydantic import BaseModel

app = FastAPI()

# ----------  Chain / network configuration  ----------
NEUTRON_LCD = os.getenv('NEUTRON_LCD', 'https://lcd-kralum.neutron.org')
NETWORK = NetworkConfig(
    chain_id='neutron-1',
    url=NEUTRON_LCD,
    fee_minimum_gas_price=0.025,
    fee_denomination='untrn',
)
client = LedgerClient(NETWORK)

# ----------  CW-20 contract addresses (replace with real ones)  ----------
CW20_MAXBTC = os.getenv('CW20_MAXBTC', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
CW20_UNIBTC = os.getenv('CW20_UNIBTC', 'neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
REQUIRED_AMOUNT = 1  # whole-token requirement

class BalanceResponse(BaseModel):
    maxbtc: int
    unibtc: int
    eligible: bool

@app.get('/api/check-balance/{address}', response_model=BalanceResponse)
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


# step:3 file: provide_liquidity_to_the_maxbtc_unibtc_supervault_using_1_maxbtc_and_1_unibtc
import os
from fastapi import HTTPException

# Add to the same FastAPI instance defined earlier
@app.get('/api/supervault/details')
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


# step:4 file: provide_liquidity_to_the_maxbtc_unibtc_supervault_using_1_maxbtc_and_1_unibtc
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.contract import MsgExecuteContract
from pydantic import BaseModel

class BuildDepositRequest(BaseModel):
    sender: str  # user wallet address
    amount_maxbtc: int = 1
    amount_unibtc: int = 1

class BuildDepositResponse(BaseModel):
    tx_bytes: str  # hex-encoded, unsigned
    body: dict     # human-readable body for inspection/debug

@app.post('/api/supervault/build-deposit', response_model=BuildDepositResponse)
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


# step:5 file: provide_liquidity_to_the_maxbtc_unibtc_supervault_using_1_maxbtc_and_1_unibtc
from cosmpy.aerial.wallet import PrivateKey
from cosmpy.aerial.tx import Transaction
from pydantic import BaseModel

class BroadcastRequest(BaseModel):
    tx_bytes: str  # hex-encoded unsigned tx

class BroadcastResponse(BaseModel):
    tx_hash: str
    height: int

@app.post('/api/supervault/broadcast', response_model=BroadcastResponse)
async def broadcast(req: BroadcastRequest):
    try:
        # ------------  Recover server key (DO NOT USE IN PRODUCTION)  ------------
        mnemonic = os.getenv('SERVER_MNEMONIC')
        if not mnemonic:
            raise ValueError('SERVER_MNEMONIC is not set in the environment.')
        wallet = PrivateKey.from_mnemonic(mnemonic)

        # ------------  Re-hydrate tx and sign  ------------
        tx = Transaction(tx_bytes=bytes.fromhex(req.tx_bytes))
        tx.sign(wallet)
        signed_bytes = tx.get_tx_bytes()

        # ------------  Broadcast  ------------
        result = client.broadcast_tx_block(signed_bytes)
        return BroadcastResponse(tx_hash=result.tx_hash, height=result.height)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Broadcast failed: {e}')


# step:1 file: Upload the example contract WASM code
from pathlib import Path

# Locate the compiled CosmWasm binary (.wasm) and verify that it exists.
# Returns an absolute Path object so the next steps can reliably read the file.
def get_wasm_file_path(relative_path: str) -> Path:
    path = Path(relative_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'WASM file not found at {path}')
    return path


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


# step:3 file: Upload the example contract WASM code
from pathlib import Path
from cosmpy.aerial.tx import Transaction
from cosmpy.protos.cosmwasm.wasm.v1 import tx_pb2 as wasm_tx_pb2, types_pb2

# Build a MsgStoreCode transaction containing the wasm binary.
# Returns an unsigned Transaction object ready for signing.
def construct_tx_store_code(wasm_path: Path, sender_address: str) -> Transaction:
    wasm_bytes = wasm_path.read_bytes()

    access_config = types_pb2.AccessConfig(
        permission=types_pb2.AccessType.ACCESS_TYPE_EVERYBODY,
        address=''  # empty when permission is Everybody
    )

    msg = wasm_tx_pb2.MsgStoreCode(
        sender=sender_address,
        wasm_byte_code=wasm_bytes,
        instantiate_permission=access_config
    )

    tx = Transaction()
    tx.add_message(msg)
    # You can still tweak gas / fee before signing, e.g.:
    # tx = tx.with_gas(1_500_000).with_fee('5000untrn')
    return tx


# step:4 file: Upload the example contract WASM code
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.clients import LedgerClient
from cosmpy.aerial.tx import Transaction

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


# step:1 file: change_the_chain-id_in_genesis.json_to__testing_
import os
import subprocess

def get_chain_home() -> str:
    """Detect the simd node's home directory.

    Priority order:
      1. `SIMD_HOME` environment variable
      2. `simd config home` CLI output
      3. Fallback to `~/.simapp`
    """
    # 1. Environment variable
    env_home = os.getenv("SIMD_HOME")
    if env_home and os.path.isdir(os.path.expanduser(env_home)):
        return os.path.expanduser(env_home)

    # 2. CLI query (may not exist on older versions)
    try:
        result = subprocess.run(["simd", "config", "home"], capture_output=True, text=True, check=True)
        cli_home = result.stdout.strip()
        if cli_home and os.path.isdir(os.path.expanduser(cli_home)):
            return os.path.expanduser(cli_home)
    except (subprocess.SubprocessError, FileNotFoundError):
        # CLI not available or failed; continue
        pass

    # 3. Fallback default
    default_home = os.path.expanduser("~/.simapp")
    if os.path.isdir(default_home):
        return default_home

    raise FileNotFoundError("Unable to determine simd home directory. Set 'SIMD_HOME' or install simd CLI.")


# step:2 file: change_the_chain-id_in_genesis.json_to__testing_
import os

def locate_genesis_file(chain_home: str) -> str:
    """Return the absolute path to config/genesis.json, ensuring it exists."""
    genesis_path = os.path.join(chain_home, "config", "genesis.json")
    if not os.path.isfile(genesis_path):
        raise FileNotFoundError(f"genesis.json not found at {genesis_path}")
    return genesis_path


# step:3 file: change_the_chain-id_in_genesis.json_to__testing_
import shutil

def backup_genesis(genesis_path: str) -> str:
    """Copy genesis.json to genesis.json.bak and return the backup path."""
    backup_path = genesis_path + ".bak"
    shutil.copy2(genesis_path, backup_path)
    return backup_path


# step:4 file: change_the_chain-id_in_genesis.json_to__testing_
import json

def update_chain_id(genesis_path: str, new_chain_id: str = "testing") -> dict:
    """Return a modified genesis dict with the chain_id changed."""
    with open(genesis_path, "r", encoding="utf-8") as f:
        genesis_data = json.load(f)

    genesis_data["chain_id"] = new_chain_id
    return genesis_data


# step:5 file: change_the_chain-id_in_genesis.json_to__testing_
import json
import os
import shutil
import tempfile

def save_genesis(genesis_path: str, genesis_data: dict) -> None:
    """Safely overwrite genesis.json with the updated content."""
    dir_name = os.path.dirname(genesis_path)
    # Write to a temp file first
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tmp:
        json.dump(genesis_data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    # Atomically replace the original file
    shutil.move(temp_name, genesis_path)


# step:6 file: change_the_chain-id_in_genesis.json_to__testing_
import subprocess

def validate_genesis(chain_home: str) -> bool:
    """Return True if `simd validate-genesis` succeeds, otherwise False."""
    try:
        subprocess.run([
            "simd",
            "validate-genesis",
            "--home",
            chain_home,
        ], check=True)
        return True
    except subprocess.CalledProcessError as err:
        print("Genesis validation failed:", err)
        return False


# step:2 file: generate_an_unsigned_transaction_that_sends_1000stake_from_address_a_to_address_b
'''
backend/account_info.py
FastAPI endpoint to fetch account metadata using the Cosmos gRPC-Gateway (REST).
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()

NODE_RPC = "https://rpc.cosmos.network"  # <-- Change to your chain’s public RPC/gRPC-gateway

class AccountInfoResponse(BaseModel):
    address: str
    account_number: int
    sequence: int

@app.get("/api/account_info", response_model=AccountInfoResponse)
async def get_account_info(address: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            url = f"{NODE_RPC}/cosmos/auth/v1beta1/accounts/{address}"
            res = await client.get(url)
        res.raise_for_status()
        account_data = res.json()

        # Handle vesting / module accounts by drilling down into nested structures.
        base_account = (account_data.get("account", {})
                        .get("base_account") or account_data.get("account", {}))

        account_number = int(base_account.get("account_number", "0"))
        sequence = int(base_account.get("sequence", "0"))

        return AccountInfoResponse(
            address=address,
            account_number=account_number,
            sequence=sequence,
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"RPC error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# step:3 file: generate_an_unsigned_transaction_that_sends_1000stake_from_address_a_to_address_b
'''
backend/generate_send_tx.py
FastAPI endpoint to generate an unsigned Cosmos-SDK bank send transaction.
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.client import NetworkConfig
from cosmpy.protos.cosmos.bank.v1beta1.tx_pb2 import MsgSend
from cosmpy.protos.cosmos.base.v1beta1.coin_pb2 import Coin
import json

app = FastAPI()

CHAIN_ID = "cosmoshub-4"  # Replace with target chain-id
RPC_URL  = "https://rpc.cosmos.network:443"
GAS_LIMIT_DEFAULT = 200000

class GenerateTxRequest(BaseModel):
    sender: str
    recipient: str
    amount: int  # micro-denom
    denom: str   # e.g. "uatom"
    account_number: int
    sequence: int

class GenerateTxResponse(BaseModel):
    unsigned_tx: str  # JSON-encoded Tx document

@app.post("/api/tx/generate-send", response_model=GenerateTxResponse)
async def generate_unsigned_send_tx(payload: GenerateTxRequest):
    try:
        # 1. Build message
        msg = MsgSend(
            from_address=payload.sender,
            to_address=payload.recipient,
            amount=[Coin(denom=payload.denom, amount=str(payload.amount))],
        )

        # 2. Initialise Transaction object (no private key → no signatures)
        cfg = NetworkConfig(
            chain_id=CHAIN_ID,
            url=RPC_URL,
            fee_minimum_gas_price=0,
        )
        tx = Transaction(cfg)
        tx.add_message(msg)
        tx.fee = 0  # Fee left zero; user/wallet can update later.
        tx.gas_limit = GAS_LIMIT_DEFAULT

        # Important: We do *not* call tx.sign() → therefore signatures remain empty.

        # 3. Manually patch account_number & sequence so downstream wallets match CLI `--generate-only` output.
        raw = tx._proto
        raw.auth_info.signer_infos.clear()  # Ensure no signer info (generate-only)
        raw.body.memo = ""
        tx_bytes = raw.SerializeToString()

        # 4. Return JSON (base64 encoded TxRaw) matching `--output json` semantics
        unsigned_tx_json = json.dumps({
            "body": json.loads(raw.body.SerializeToString().hex()),
            "auth_info": json.loads(raw.auth_info.SerializeToString().hex()),
            "signatures": [],
            "account_number": str(payload.account_number),
            "sequence": str(payload.sequence),
            "chain_id": CHAIN_ID,
        })

        return GenerateTxResponse(unsigned_tx=unsigned_tx_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not build unsigned tx: {e}")


# step:1 file: initialize_and_populate_the_snapshot_store_at_data_snapshots
from pathlib import Path
import os
import subprocess
from fastapi import FastAPI, HTTPException

app = FastAPI()


def detect_node_home_and_binary():
    """Return a tuple of (home_path: Path, binary_name: str)."""
    # 1. Environment overrides -------------------------------------------------
    env_home = os.getenv("COSMOS_HOME")
    env_bin  = os.getenv("COSMOS_BINARY")
    if env_home and env_bin:
        return Path(env_home).expanduser(), env_bin

    # 2. Best-effort process scan ---------------------------------------------
    try:
        pgrep_output = subprocess.check_output(["pgrep", "-fa", "d$"]).decode()
        # Pick the first matching daemon process
        line        = pgrep_output.splitlines()[0]
        _pid, *cmd  = line.strip().split()
        binary_path = Path(cmd[0])
        binary_name = binary_path.name
        # Attempt to extract an explicit --home flag
        home_path = None
        if "--home" in cmd:
            idx = cmd.index("--home")
            if idx + 1 < len(cmd):
                home_path = Path(cmd[idx + 1]).expanduser()
        if not home_path:
            home_path = Path.home() / f".{binary_name}"
        return home_path, binary_name
    except Exception:
        # 3. Conservative fallback --------------------------------------------
        return Path.home() / ".appd", "appd"


@app.get("/api/node-info")
def api_node_info():
    """Return `{ home: str, binary: str }` or HTTP 500 on failure."""
    try:
        home, binary = detect_node_home_and_binary()
        return {"home": str(home), "binary": binary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:2 file: initialize_and_populate_the_snapshot_store_at_data_snapshots
import os
import signal
import subprocess
import time
from fastapi import FastAPI, HTTPException

app = FastAPI()

TIMEOUT_SEC = 30  # wait time for graceful shutdown


def stop_node_process(binary_name: str, timeout: int = TIMEOUT_SEC) -> bool:
    """Return True on graceful shutdown, False if SIGKILL had to be used."""
    try:
        pid_bytes = subprocess.check_output(["pgrep", "-f", binary_name])
        pids      = [int(pid) for pid in pid_bytes.decode().split()]
    except subprocess.CalledProcessError:
        raise RuntimeError(f"No running process found for '{binary_name}'.")

    # Send SIGTERM to every matching PID
    for pid in pids:
        os.kill(pid, signal.SIGTERM)

    # Wait until the first PID exits (others should follow)
    start = time.time()
    while time.time() - start < timeout:
        try:
            os.kill(pids[0], 0)  # check still alive
            time.sleep(1)
        except OSError:  # process is gone
            return True

    # Escalate to SIGKILL if we’re still here
    for pid in pids:
        os.kill(pid, signal.SIGKILL)
    return False


@app.post("/api/node-stop")
def api_node_stop(binary: str):
    try:
        graceful = stop_node_process(binary)
        return {"graceful": graceful}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:3 file: initialize_and_populate_the_snapshot_store_at_data_snapshots
from pathlib import Path
from fastapi import FastAPI, HTTPException

app = FastAPI()


def ensure_snapshot_dir(home_path: str) -> Path:
    snap_dir = Path(home_path).expanduser() / "data" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    return snap_dir


@app.post("/api/snapshot/ensure-dir")
def api_ensure_snapshot_dir(home: str):
    try:
        path = ensure_snapshot_dir(home)
        return {"snapshot_dir": str(path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:4 file: initialize_and_populate_the_snapshot_store_at_data_snapshots
import subprocess
from fastapi import FastAPI, HTTPException

app = FastAPI()


def export_snapshot(binary: str, home_path: str) -> str:
    cmd = [binary, "snapshots", "export", "--home", home_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


@app.post("/api/snapshot/export")
def api_export_snapshot(binary: str, home: str):
    try:
        output = export_snapshot(binary, home)
        return {"status": "exported", "cli_output": output}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:5 file: initialize_and_populate_the_snapshot_store_at_data_snapshots
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException

app = FastAPI()


def verify_snapshot(binary: str, home_path: str):
    snap_dir   = Path(home_path).expanduser() / "data" / "snapshots"
    archives   = [p.name for p in snap_dir.glob("*.tar*")]

    list_cmd   = [binary, "snapshots", "list", "--home", home_path]
    list_proc  = subprocess.run(list_cmd, capture_output=True, text=True)
    if list_proc.returncode != 0:
        raise RuntimeError(list_proc.stderr.strip())

    return {"archives_on_disk": archives, "cli_output": list_proc.stdout.strip()}


@app.get("/api/snapshot/verify")
def api_verify_snapshot(binary: str, home: str):
    try:
        res = verify_snapshot(binary, home)
        if not res["archives_on_disk"]:
            raise RuntimeError("Snapshot directory is empty.")
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# step:1 file: add_ens_support_when_querying_balances_with_foundry’s_cast_cli
import os
from flask import Flask, request, jsonify
from web3 import Web3

app = Flask(__name__)

# RPC endpoint; set ENV var ETH_RPC_URL or fallback to a public one
RPC_URL = os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com")
w3 = Web3(Web3.HTTPProvider(RPC_URL))

@app.route('/api/resolve_ens', methods=['GET'])
def resolve_ens():
    """Resolve an ENS name to a checksummed Ethereum address."""
    ens_name = request.args.get('ens')
    if not ens_name:
        return jsonify({"error": "Query param 'ens' is required."}), 400

    try:
        address = w3.ens.address(ens_name)
        if address is None:
            return jsonify({"error": f"ENS name '{ens_name}' not found."}), 404
        return jsonify({"address": Web3.toChecksumAddress(address)})
    except Exception as e:
        # Catch anything unexpected (e.g., network issues)
        return jsonify({"error": str(e)}), 500

# Allow this file to be run directly for local dev
autostart = os.getenv("FLASK_AUTOSTART", "true").lower() == "true"
if __name__ == '__main__' and autostart:
    app.run(host='0.0.0.0', port=8000, debug=True)


# step:2 file: add_ens_support_when_querying_balances_with_foundry’s_cast_cli
import os
from flask import Flask, request, jsonify
from web3 import Web3

app = Flask(__name__)

RPC_URL = os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com")
w3 = Web3(Web3.HTTPProvider(RPC_URL))

@app.route('/api/eth_balance', methods=['GET'])
def eth_balance():
    """Return balance for an address OR ENS name provided via ?value."""
    value = request.args.get('value')  # can be address or ENS name
    if not value:
        return jsonify({"error": "Query param 'value' is required."}), 400

    # Resolve ENS if necessary
    try:
        if "." in value:  # naïve ENS check
            resolved = w3.ens.address(value)
            if resolved is None:
                return jsonify({"error": f"ENS name '{value}' not found."}), 404
            address = Web3.toChecksumAddress(resolved)
        else:
            address = Web3.toChecksumAddress(value)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Query balance
    try:
        balance_wei = w3.eth.get_balance(address)
        balance_eth = w3.fromWei(balance_wei, 'ether')
        return jsonify({"address": address, "balance": str(balance_eth)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Allow this file to be run directly for local dev
autostart = os.getenv("FLASK_AUTOSTART", "true").lower() == "true"
if __name__ == '__main__' and autostart:
    app.run(host='0.0.0.0', port=8000, debug=True)


# step:2 file: save_a_remix_ide_workspace_to_a_github_repository
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import httpx, os, secrets

app = FastAPI()

# Load your GitHub OAuth app credentials from environment variables
GITHUB_CLIENT_ID = os.getenv('GITHUB_CLIENT_ID')
GITHUB_CLIENT_SECRET = os.getenv('GITHUB_CLIENT_SECRET')

# Very simple in-memory state store (replace with a DB or session store in production)
_state_store = {}

@app.get('/api/github/login')
async def github_login():
    """Redirect the browser to GitHub’s OAuth consent page."""
    if not GITHUB_CLIENT_ID:
        raise HTTPException(500, 'GITHUB_CLIENT_ID is missing.')

    state = secrets.token_hex(16)
    _state_store[state] = True  # Prevent CSRF

    auth_url = (
        'https://github.com/login/oauth/authorize'
        f'?client_id={GITHUB_CLIENT_ID}'
        '&scope=repo'
        f'&state={state}'
    )
    return RedirectResponse(auth_url)

@app.get('/api/github/callback')
async def github_callback(code: str, state: str):
    """Exchange the temporary code for a permanent GitHub access token."""
    if state not in _state_store:
        raise HTTPException(400, 'Invalid OAuth state.')
    _state_store.pop(state, None)

    if not GITHUB_CLIENT_SECRET:
        raise HTTPException(500, 'GITHUB_CLIENT_SECRET is missing.')

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            'https://github.com/login/oauth/access_token',
            headers={
                'Accept': 'application/json'
            },
            data={
                'client_id': GITHUB_CLIENT_ID,
                'client_secret': GITHUB_CLIENT_SECRET,
                'code': code,
                'state': state,
            }
        )
        token_resp.raise_for_status()
        token_json = token_resp.json()

    access_token = token_json.get('access_token')
    if not access_token:
        raise HTTPException(400, 'GitHub did not return an access token.')

    # In production you would typically set a secure session cookie here.
    return JSONResponse({'access_token': access_token})


# step:3 file: save_a_remix_ide_workspace_to_a_github_repository
from fastapi import FastAPI, Body, HTTPException
from typing import Dict
import httpx, base64

app = FastAPI()

@app.post('/api/github/push')
async def github_push(
    access_token: str = Body(..., embed=True),
    repo_name: str = Body(..., embed=True),
    commit_message: str = Body(..., embed=True),
    files: Dict[str, str] = Body(..., description='Mapping: filepath → file contents (utf-8 strings)')
):
    """Create the repository (if needed) and push the provided files as a single commit."""

    headers = {
        'Authorization': f'token {access_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    async with httpx.AsyncClient(headers=headers) as client:
        # 1️⃣  Get the authenticated user’s login name
        user_resp = await client.get('https://api.github.com/user')
        user_resp.raise_for_status()
        owner = user_resp.json()['login']

        # 2️⃣  Ensure the repository exists (create it if it does not)
        repo_resp = await client.get(f'https://api.github.com/repos/{owner}/{repo_name}')
        if repo_resp.status_code == 404:
            create_resp = await client.post('https://api.github.com/user/repos', json={'name': repo_name})
            create_resp.raise_for_status()
        elif repo_resp.status_code != 200:
            raise HTTPException(repo_resp.status_code, f'GitHub error: {repo_resp.text}')

        # 3️⃣  Upload each file via the "contents" API
        for path, content in files.items():
            encoded = base64.b64encode(content.encode()).decode()
            put_resp = await client.put(
                f'https://api.github.com/repos/{owner}/{repo_name}/contents/{path}',
                json={
                    'message': commit_message,
                    'content': encoded
                }
            )
            if put_resp.status_code not in (200, 201):
                raise HTTPException(put_resp.status_code, f'GitHub error: {put_resp.text}')

    return {
        'status': 'success',
        'repository_url': f'https://github.com/{owner}/{repo_name}'
    }


# step:1 file: set_custom_ante_handler_for_my_app
#!/usr/bin/env bash
set -e

# Locate app.go file within the repository
APP_FILE=$(git ls-files | grep -E '(^|/)app.go$' || true)

if [ -z "$APP_FILE" ]; then
  echo "app.go not found in repository."
  exit 1
fi

echo "Found app.go at: $APP_FILE"


# step:2 file: set_custom_ante_handler_for_my_app
package ante

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
    sdkerrors "github.com/cosmos/cosmos-sdk/types/errors"
    "github.com/cosmos/cosmos-sdk/x/auth/ante"
    "github.com/cosmos/cosmos-sdk/x/auth/signing"
)

// CustomAnteHandlerOptions holds configuration for our custom ante handler.
type CustomAnteHandlerOptions struct {
    ante.HandlerOptions

    // MaxSignatures sets an upper bound on allowed signatures per transaction.
    MaxSignatures int
    // MinFee defines the mandatory minimum fee.
    MinFee sdk.Coin
}

// NewCustomAnteHandler returns an sdk.AnteHandler that wraps the default handler
// and adds fee + signature checks.
func NewCustomAnteHandler(opts CustomAnteHandlerOptions) sdk.AnteHandler {
    defaultHandler := ante.NewAnteHandler(opts.HandlerOptions)

    return func(ctx sdk.Context, tx sdk.Tx, simulate bool) (sdk.Context, error) {
        // Execute default ante checks first.
        newCtx, err := defaultHandler(ctx, tx, simulate)
        if err != nil {
            return newCtx, err
        }

        // ---------- Custom Fee Check ----------
        feeTx, ok := tx.(sdk.FeeTx)
        if !ok {
            return newCtx, sdkerrors.Wrap(sdkerrors.ErrTxDecode, "tx does not implement FeeTx")
        }

        fees := feeTx.GetFee()
        if !fees.AmountOf(opts.MinFee.Denom).IsGTE(opts.MinFee.Amount) {
            return newCtx, sdkerrors.Wrapf(sdkerrors.ErrInsufficientFee,
                "minimum fee %s required", opts.MinFee.String())
        }

        // ---------- Signature Limit ----------
        sigTx, ok := tx.(signing.SigVerifiableTx)
        if !ok {
            return newCtx, sdkerrors.Wrap(sdkerrors.ErrTxDecode, "tx does not implement SigVerifiableTx")
        }

        if len(sigTx.GetSignaturesV2()) > opts.MaxSignatures {
            return newCtx, sdkerrors.Wrapf(sdkerrors.ErrUnauthorized,
                "transaction has %d signatures; maximum allowed is %d",
                len(sigTx.GetSignaturesV2()), opts.MaxSignatures)
        }

        // All good!
        return newCtx, nil
    }
}


# step:3 file: set_custom_ante_handler_for_my_app
// Inside NewMyApp() after keepers are initialized and before returning the app instance.

minGasPrices, _ := sdk.ParseDecCoins("0.01uatom") // adjust denom & value as needed
customAnteOpts := ante.CustomAnteHandlerOptions{
    HandlerOptions: ante.HandlerOptions{
        AccountKeeper:   app.AccountKeeper,
        BankKeeper:      app.BankKeeper,
        SignModeHandler: txCfg.SignModeHandler(),
        FeegrantKeeper:  app.FeeGrantKeeper,
        SigGasConsumer:  ante.DefaultSigVerificationGasConsumer,
        TxFeeChecker:    ante.NewMempoolFeeChecker(app.BaseApp, minGasPrices),
    },
    MaxSignatures: 2,
    MinFee:        sdk.NewInt64Coin("uatom", 1000), // 1,000 uatom minimum fee
}

app.SetAnteHandler(ante.NewCustomAnteHandler(customAnteOpts))


# step:4 file: set_custom_ante_handler_for_my_app
#!/usr/bin/env bash
set -e

echo "Running unit tests..."
go test ./...

echo "Building application binary..."
if [ -f Makefile ] && grep -q "install:" Makefile; then
  make install
else
  go build -o build/myappd ./cmd/myappd
fi

echo "✅  All tests passed and binary built."


# step:5 file: set_custom_ante_handler_for_my_app
#!/usr/bin/env bash
set -e

APP_CMD="myappd"             # Update to your binary name
CHAIN_ID="localnet-1"
HOME_DIR="$HOME/.myapp"
DENOM="uatom"

# Kill any existing process
pkill -f "$APP_CMD" || true
rm -rf "$HOME_DIR"

# Initialise chain
$APP_CMD init tester --chain-id $CHAIN_ID --home $HOME_DIR

# Add key and grab its address
$APP_CMD keys add tester --keyring-backend test --home $HOME_DIR
ADDRESS=$($APP_CMD keys show tester -a --keyring-backend test --home $HOME_DIR)

# Allocate large balance in genesis (1 B uatom)
cat $HOME_DIR/config/genesis.json | \
  jq --arg ADDRESS $ADDRESS '.app_state.bank.balances += [{"address":$ADDRESS,"coins":[{"denom":"uatom","amount":"1000000000"}]}]' \
  > tmp_genesis.json && mv tmp_genesis.json $HOME_DIR/config/genesis.json

# Start node (background)
$APP_CMD start --home $HOME_DIR --log_level info --pruning=nothing > node.log 2>&1 &
NODE_PID=$!
trap "kill $NODE_PID" EXIT
sleep 8

# Broadcast test tx (meets fee requirement)
$APP_CMD tx bank send $ADDRESS $ADDRESS 1$DENOM --fees 2000$DENOM \
  --chain-id $CHAIN_ID --keyring-backend test --home $HOME_DIR --yes --broadcast-mode block

echo "Tail of node log (look for ante-handler messages) ➡️"
tail -20 node.log | grep -E "insufficient fee|signatures; maximum" || true


# step:1 file: list_all_wallets_managed_by_the_node
import os\nimport httpx\nfrom fastapi import FastAPI, HTTPException\n\napp = FastAPI()\n\n# RPC configuration is provided via environment variables so that secrets never reach the browser.\nRPC_ENDPOINT = os.getenv(\"RPC_ENDPOINT\", \"http://localhost:8545\")\nRPC_USERNAME = os.getenv(\"RPC_USERNAME\")\nRPC_PASSWORD = os.getenv(\"RPC_PASSWORD\")\n\nasync def _rpc_call(payload: dict):\n    \"\"\"Internal helper to perform an authenticated JSON-RPC call.\"\"\"\n    auth = None\n    if RPC_USERNAME and RPC_PASSWORD:\n        auth = (RPC_USERNAME, RPC_PASSWORD)  # Basic-Auth tuple for httpx\n\n    try:\n        async with httpx.AsyncClient(auth=auth, timeout=10) as client:\n            response = await client.post(RPC_ENDPOINT, json=payload)\n            response.raise_for_status()\n    except httpx.HTTPError as http_err:\n        raise HTTPException(status_code=502, detail=f\"RPC connection failed: {http_err}\")\n    except Exception as err:\n        raise HTTPException(status_code=500, detail=f\"Unexpected error: {err}\")\n\n    # Parse JSON-RPC response and bubble up any RPC-level errors.\n    data = response.json()\n    if \"error\" in data and data[\"error\"] is not None:\n        raise HTTPException(status_code=500, detail=data[\"error\"])\n\n    return data.get(\"result\")


# step:2 file: list_all_wallets_managed_by_the_node
from fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.get(\"/api/personal_list_wallets\")\nasync def personal_list_wallets():\n    \"\"\"Return the list of wallets managed by the node.\"\"\"\n    payload = {\n        \"jsonrpc\": \"2.0\",\n        \"method\": \"personal_listWallets\",\n        \"params\": [],\n        \"id\": 1\n    }\n    result = await _rpc_call(payload)\n    return {\"wallets\": result}\n\n# Remember to include the router in your FastAPI application\n# app.include_router(router)


# step:1 file: start_the_node_with_the_application_and_cometbft_processes_split
import os
import subprocess


def start_abci_app(home_dir: str = os.path.expanduser("~/.simapp")):
    """Spawn `simd start --with-comet=false` and pipe its output."""
    cmd = [
        "simd",
        "start",
        "--with-comet=false",
        "--home",
        home_dir,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,          # decode bytes → str automatically
            bufsize=1,          # line-buffered
        )
    except FileNotFoundError:
        raise RuntimeError("`simd` binary not found in PATH. Make sure it is installed and executable.")

    print(f"Started simd (ABCI-only). PID={proc.pid}")
    return proc


# step:2 file: start_the_node_with_the_application_and_cometbft_processes_split
import time


def wait_for_abci_ready(proc, timeout: int = 60):
    """Block until the given `simd` process outputs `ABCI server started`."""
    deadline = time.time() + timeout
    if proc.stdout is None:
        raise RuntimeError("Process stdout pipe is not available.")

    for line in iter(proc.stdout.readline, ""):
        print(line.rstrip())  # optional: stream logs to console
        if "ABCI server started" in line:
            print("✅ ABCI application is up and listening on tcp://localhost:26658")
            return True
        if time.time() > deadline:
            proc.terminate()
            raise TimeoutError("Timed out waiting for ABCI server to start.")
    raise RuntimeError("simd process exited unexpectedly before readiness.")


# step:3 file: start_the_node_with_the_application_and_cometbft_processes_split
def start_cometbft(home_dir: str = os.path.expanduser("~/.simapp"), proxy_addr: str = "tcp://127.0.0.1:26658"):
    """Spawn `cometbft start` that points to the ABCI app via `--proxy_app`."""
    cmd = [
        "cometbft",
        "start",
        "--home",
        home_dir,
        "--proxy_app",
        proxy_addr,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        raise RuntimeError("`cometbft` binary not found in PATH. Install CometBFT or adjust PATH.")

    print(f"Started CometBFT. PID={proc.pid}")
    return proc


# step:4 file: start_the_node_with_the_application_and_cometbft_processes_split
import requests
import time


def wait_for_cometbft_rpc(rpc_url: str = "http://localhost:26657/status", timeout: int = 60, interval: float = 2.0):
    """Repeatedly query the RPC `/status` endpoint until it responds with HTTP 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(rpc_url, timeout=5)
            if resp.status_code == 200:
                print("✅ CometBFT RPC is live on tcp://localhost:26657")
                return resp.json()
        except requests.RequestException:
            pass  # endpoint not ready yet
        print("Waiting for CometBFT RPC endpoint …")
        time.sleep(interval)
    raise TimeoutError("Timed out waiting for CometBFT RPC endpoint to become available.")


# step:1 file: expose_the_websocket_json-rpc_endpoint_on_port_8546
import subprocess

SERVICE_NAME = "evmd"

def stop_node(service_name: str = SERVICE_NAME):
    """Stops the evmd node via systemd (or falls back to pkill)."""
    try:
        # Attempt to stop via systemd first
        result = subprocess.run([
            "systemctl",
            "stop",
            service_name
        ], check=True, capture_output=True)
        print(result.stdout.decode() or f"{service_name} stopped via systemd.")
    except subprocess.CalledProcessError as sysd_err:
        print("systemctl stop failed, attempting pkill …")
        try:
            subprocess.run(["pkill", "-f", service_name], check=True)
            print(f"{service_name} killed via pkill.")
        except subprocess.CalledProcessError as pkill_err:
            raise RuntimeError(
                f"Unable to stop {service_name}. systemd error: {sysd_err}. pkill error: {pkill_err}."
            )

if __name__ == "__main__":
    stop_node()


# step:2 file: expose_the_websocket_json-rpc_endpoint_on_port_8546
import os
import toml

DEFAULT_APP_TOML = os.path.expanduser("~/.evmd/config/app.toml")

def open_app_toml(path: str = DEFAULT_APP_TOML):
    """Returns the app.toml contents as a Python dict and the resolved path."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"app.toml not found at {path}. Did you install evmd?")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = toml.load(fh)
        return cfg, path
    except Exception as err:
        raise RuntimeError(f"Unable to load app.toml: {err}")

if __name__ == "__main__":
    cfg, _ = open_app_toml()
    print("[json-rpc] section before edit:")
    print(cfg.get("json-rpc", {}))


# step:3 file: expose_the_websocket_json-rpc_endpoint_on_port_8546
def modify_json_rpc_field(cfg: dict, ws_address: str = "0.0.0.0:8546") -> dict:
    """Updates the JSON-RPC settings in-memory."""
    if "json-rpc" not in cfg:
        cfg["json-rpc"] = {}
    cfg["json-rpc"]["enable"] = True
    cfg["json-rpc"]["ws-address"] = ws_address
    return cfg

if __name__ == "__main__":
    from pprint import pprint
    cfg, _ = open_app_toml()
    cfg = modify_json_rpc_field(cfg)
    pprint(cfg["json-rpc"])


# step:4 file: expose_the_websocket_json-rpc_endpoint_on_port_8546
import shutil

def save_and_close_file(cfg: dict, path: str):
    """Saves the modified app.toml, creating a .bak backup first."""
    backup_path = f"{path}.bak"
    try:
        shutil.copy2(path, backup_path)
        with open(path, "w", encoding="utf-8") as fh:
            toml.dump(cfg, fh)
        print(f"Configuration saved to {path}. Backup stored at {backup_path}.")
    except Exception as err:
        raise RuntimeError(f"Failed to write app.toml: {err}")

if __name__ == "__main__":
    cfg, path = open_app_toml()
    cfg = modify_json_rpc_field(cfg)
    save_and_close_file(cfg, path)


# step:5 file: expose_the_websocket_json-rpc_endpoint_on_port_8546
import subprocess

SERVICE_NAME = "evmd"

async def create_wallet():
    """Create a new key-pair and return { mnemonic, address }."""
    try:
        # 1. Generate random private key (secure entropy)
        priv_key = PrivateKey()

        # 2. Obtain the 24-word BIP-39 mnemonic from the key
        mnemonic = priv_key.to_mnemonic()

        # 3. Derive the bech32 address (default prefix: 'cosmos')
        address = priv_key.public_key.address()

        # SECURITY: do *not* store mnemonic or private key
        return {"mnemonic": mnemonic, "address": address}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


