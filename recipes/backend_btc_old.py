# step:2 file: increase_the_user’s_deposit_in_the_wbtc_usdc_supervault_by_0.2_wbtc_and_12_000_usdc
# validate_token_balances.py
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.config import NetworkConfig
from typing import Dict

# Constants converted to micro-denom amounts
MIN_WBTC = 20_000_000          # 0.2  * 10^8
MIN_USDC = 12_000_000_000      # 12000 * 10^6

async def validate_token_balances(
    rpc_endpoint: str,
    address: str,
    wbtc_contract: str,
    usdc_contract: str,
) -> Dict:
    """Raises ValueError if either balance is below the required threshold."""

    cfg = NetworkConfig(chain_id="neutron-1", url=rpc_endpoint)
    client = LedgerClient(cfg)

    def cw20_balance(contract: str) -> int:
        query = {"balance": {"address": address}}
        result = client.query_contract_smart(contract, query)
        return int(result.get("balance", 0))

    wbtc_bal = cw20_balance(wbtc_contract)
    usdc_bal = cw20_balance(usdc_contract)

    if wbtc_bal < MIN_WBTC:
        raise ValueError(f"Insufficient WBTC balance (have {wbtc_bal}, need {MIN_WBTC}).")
    if usdc_bal < MIN_USDC:
        raise ValueError(f"Insufficient USDC balance (have {usdc_bal}, need {MIN_USDC}).")

    return {
        "success": True,
        "balances": {"wbtc": wbtc_bal, "usdc": usdc_bal},
    }


# step:3 file: increase_the_user’s_deposit_in_the_wbtc_usdc_supervault_by_0.2_wbtc_and_12_000_usdc
# get_supervault_contract_address.py
import os

def get_supervault_contract_address(pair: str = "WBTC_USDC") -> str:
    """Return the Supervault contract address for the requested pair."""
    vault_registry = {
        "WBTC_USDC": os.getenv("WBTC_USDC_SUPERVAULT", "neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    }
    try:
        return vault_registry[pair]
    except KeyError:
        raise ValueError(f"No Supervault registered for pair {pair}.")


# step:2 file: redeem_lp_shares_from_the_maxbtc_ebtc_supervault
from typing import Tuple
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.contract import SmartContract

RPC_ENDPOINT = 'https://rpc-palvus.neutron.org:443'
CHAIN_ID = 'neutron-1'

# In production, fetch these dynamically (e.g. from a registry contract)
SUPERVAULT_REGISTRY = {
    'supervault_maxBTC_eBTC': 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
}

def query_supervault_balance(user_address: str, vault_key: str) -> Tuple[str, int]:
    """Return (contract_address, share_balance) for the given vault key."""
    if vault_key not in SUPERVAULT_REGISTRY:
        raise ValueError(f'Unknown vault key: {vault_key}')

    contract_address = SUPERVAULT_REGISTRY[vault_key]
    cfg = NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT, fee_minimum_gas_price=0)
    client = LedgerClient(cfg)
    contract = SmartContract(contract_address)

    try:
        # Smart-query: { balance: { address: <user_address> } }
        query_msg = {'balance': {'address': user_address}}
        result = client.query_contract(contract, query_msg)
        share_balance = int(result.get('balance', 0))
    except Exception as exc:
        raise RuntimeError(f'Contract query failed: {exc}') from exc

    return contract_address, share_balance



# step:3 file: redeem_lp_shares_from_the_maxbtc_ebtc_supervault
def validate_redeem_amount(requested: int, available: int) -> None:
    """Raise an error if the requested amount is invalid."""
    if requested <= 0:
        raise ValueError('Redeem amount must be greater than zero.')
    if requested > available:
        raise ValueError(f'Requested {requested} shares exceeds available balance {available}.')
    # No return value needed on success



# step:2 file: claim_my_ntrn_rewards_with_standard_vesting
import os
from cosmpy.aerial.client import LedgerClient, NetworkConfig

RPC_ENDPOINT = os.getenv('NEUTRON_RPC', 'https://rpc-kralum.neutron.org:443')
CHAIN_ID = os.getenv('NEUTRON_CHAIN_ID', 'neutron-1')
CLAIM_CONTRACT = os.getenv('CLAIM_CONTRACT_ADDRESS')  # e.g. 'neutron1xxxxx...'

if not CLAIM_CONTRACT:
    raise EnvironmentError('CLAIM_CONTRACT_ADDRESS environment variable must be set.')

cfg = NetworkConfig(
    chain_id=CHAIN_ID,
    rpc_endpoint=RPC_ENDPOINT,
)
client = LedgerClient(cfg)

def query_pending_rewards(user_address: str) -> int:
    """Return the amount (in uNTRN) of pending rewards or raise if none."""
    try:
        response = client.query_contract_smart(CLAIM_CONTRACT, {
            "pending_rewards": {"address": user_address}
        })
        rewards = int(response.get("rewards", 0))
        if rewards <= 0:
            raise ValueError(f"User {user_address} has no pending rewards.")
        return rewards
    except Exception as err:
        raise RuntimeError(f"Failed to query pending rewards: {err}") from err


# step:5 file: claim_my_ntrn_rewards_with_standard_vesting
def query_vesting_schedule(user_address: str):
    """Return the vesting schedule for a user after claiming."""
    try:
        schedule = client.query_contract_smart(CLAIM_CONTRACT, {
            "vesting_schedule": {"address": user_address}
        })
        return schedule
    except Exception as err:
        raise RuntimeError(f"Unable to fetch vesting schedule: {err}") from err


# step:2 file: hold_maxbtc_in_my_wallet_for_base_1x_rewards
import base64
import json
import requests
from typing import Dict


def query_cw20_balance(lcd_url: str, contract_address: str, user_address: str) -> Dict[str, int]:
    """Query the CW20 balance for a user with Neutron's LCD endpoint.

    Args:
        lcd_url (str): Base LCD URL, e.g., "https://rest.neutron.org".
        contract_address (str): CW20 contract address for maxBTC.
        user_address (str): Wallet address whose balance is requested.

    Returns:
        Dict[str, int]: {"raw_balance": <int>}
    """
    # CW20 exposes {"balance": {"address": "..."}} query messages
    query_msg = {"balance": {"address": user_address}}
    encoded_msg = base64.b64encode(json.dumps(query_msg).encode()).decode()
    url = f"{lcd_url}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{encoded_msg}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # The contract response returns numeric string under data.balance
        raw_balance = int(data.get("data", {}).get("balance", "0"))
        return {"raw_balance": raw_balance}
    except (requests.RequestException, ValueError, KeyError) as err:
        raise RuntimeError(f"Unable to query CW20 balance: {err}") from err


# step:3 file: hold_maxbtc_in_my_wallet_for_base_1x_rewards
def validate_balance(raw_balance: int) -> dict:
    """Determine if the supplied balance is greater than zero.

    Args:
        raw_balance (int): Balance value from the CW20 query.

    Returns:
        dict: {"is_eligible": bool, "raw_balance": int}
    """
    try:
        balance_int = int(raw_balance)
    except (TypeError, ValueError):
        raise ValueError('Provided balance must be an integer-convertible value.')

    is_eligible = balance_int > 0
    return {"is_eligible": is_eligible, "raw_balance": balance_int}


# step:5 file: bridge_1_wbtc_from_ethereum_to_neutron
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from typing import Optional
import time

# ---------------- Configuration ----------------
NEUTRON_RPC_ENDPOINT = 'https://rpc-kralum.neutron.org:443'  # Replace if custom endpoint
CHAIN_ID = 'neutron-1'
FEE_GAS_PRICE = 0.025  # NTRN per gas unit


def wait_for_ibc_transfer(
    neutron_address: str,
    ibc_denom: str,
    expected_amount: int,
    timeout_sec: int = 1800,
    poll_interval_sec: int = 10,
) -> Optional[int]:
    """Blocks until the expected IBC transfer arrives or the timeout is hit.

    Args:
        neutron_address: Recipient address on Neutron.
        ibc_denom: IBC denom (for example, 'ibc/…wbtc').
        expected_amount: Raw integer amount (8-decimals for WBTC).
    Returns:
        The final balance when detected.
    Raises:
        TimeoutError if the transfer is not seen within `timeout_sec`.
    """
    cfg = NetworkConfig(chain_id=CHAIN_ID, url=NEUTRON_RPC_ENDPOINT, fee_minimum_gas_price=FEE_GAS_PRICE)
    client = LedgerClient(cfg)

    original_balance = int(client.query_bank_balance(neutron_address, denom=ibc_denom))
    start = time.time()

    while time.time() - start < timeout_sec:
        current_balance = int(client.query_bank_balance(neutron_address, denom=ibc_denom))
        if current_balance - original_balance >= expected_amount:
            print('IBC relay complete — funds arrived on Neutron.')
            return current_balance
        print('Waiting for IBC transfer…')
        time.sleep(poll_interval_sec)

    raise TimeoutError('IBC transfer not detected within allotted time.')


# step:6 file: bridge_1_wbtc_from_ethereum_to_neutron
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from decimal import Decimal

NEUTRON_RPC_ENDPOINT = 'https://rpc-kralum.neutron.org:443'
CHAIN_ID = 'neutron-1'
FEE_GAS_PRICE = 0.025


def query_neutron_bank_balance(neutron_address: str, ibc_denom: str) -> str:
    """Returns the WBTC balance (human-readable) held by `neutron_address`."""
    cfg = NetworkConfig(chain_id=CHAIN_ID, url=NEUTRON_RPC_ENDPOINT, fee_minimum_gas_price=FEE_GAS_PRICE)
    client = LedgerClient(cfg)
    try:
        raw_balance = int(client.query_bank_balance(neutron_address, denom=ibc_denom))
        # WBTC uses 8 decimals
        display = str(Decimal(raw_balance) / (10 ** 8)) + ' WBTC'
        return display
    except Exception as e:
        raise RuntimeError(f'Failed to query Neutron bank balance: {e}')


# step:2 file: provide_paired_liquidity_of_1_wbtc_and_60,000_usdc_to_the_wbtc_usdc_supervault
from decimal import Decimal
from typing import Dict
from cosmpy.aerial.client import LedgerClient, NetworkConfig
# RPC endpoint and network configuration\nCONFIG = NetworkConfig(\n    chain_id='neutron-1',\n    url='https://rpc.mainnet.neutron-1.neutron.org',\n    fee_minimum_gas_price=0.005,\n    fee_denomination='untrn'\n)\n\n# Denoms for WBTC and USDC on Neutron (ICS20 / factory denoms used as placeholders)\nWBTC_DENOM = 'ibc/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'\nUSDC_DENOM = 'ibc/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy'\n\nclient = LedgerClient(CONFIG)\n\n\ndef check_token_balance(address: str,\n                        min_wbtc: Decimal = Decimal('1'),\n                        min_usdc: Decimal = Decimal('60000')) -> Dict[str, str]:\n    """Verify the wallet holds at least the requested WBTC/USDC amounts."""\n    try:\n        wbtc_balance = Decimal(client.query_bank_balance(address, denom=WBTC_DENOM))\n        usdc_balance = Decimal(client.query_bank_balance(address, denom=USDC_DENOM))\n    except Exception as exc:\n        raise RuntimeError(f'Unable to query balances: {exc}')\n\n    if wbtc_balance < min_wbtc or usdc_balance < min_usdc:\n        need_wbtc = (min_wbtc - wbtc_balance) if wbtc_balance < min_wbtc else Decimal(0)\n        need_usdc = (min_usdc - usdc_balance) if usdc_balance < min_usdc else Decimal(0)\n        raise ValueError(\n            f'Insufficient balances — missing: {need_wbtc} WBTC, {need_usdc} USDC. Please bridge or swap before retrying.'\n        )\n\n    return {\n        'wbtc_balance': str(wbtc_balance),\n        'usdc_balance': str(usdc_balance)\n    }


# step:3 file: provide_paired_liquidity_of_1_wbtc_and_60,000_usdc_to_the_wbtc_usdc_supervault
import json
from cosmpy.aerial.client import LedgerClient
# Re-use the LedgerClient from step 2 (named `client`)\nSUPER_VAULT_CONTRACT = 'neutron1vaultxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'  # TODO: replace with the real main-net address\n\n\ndef query_supervault_details() -> dict:\n    """Retrieve Supervault configuration such as accepted denoms and LP-token."""\n    try:\n        response_bytes = client.query_contract(SUPER_VAULT_CONTRACT, {'config': {}})\n    except Exception as exc:\n        raise RuntimeError(f'Failed to query supervault details: {exc}')\n\n    # The SDK normally returns bytes; decode if needed\n    if isinstance(response_bytes, (bytes, bytearray)):\n        response = json.loads(response_bytes.decode())\n    else:\n        response = response_bytes\n\n    return {\n        'vault_contract': SUPER_VAULT_CONTRACT,\n        'accepted_denoms': response.get('accepted_denoms', []),\n        'lp_token': response.get('lp_token')\n    }


# step:3 file: deposit_1_wbtc_into_the_wbtc_usdc_supervault
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.contract import Contract
from typing import Optional

def get_supervault_contract(registry_contract: str, wbtc_denom: str, usdc_denom: str) -> str:
    client = LedgerClient(NETWORK)
    registry = Contract(registry_contract, client)

    # The query msg structure is registry-specific; adjust to actual schema.
    query_msg = {
        "vault": {
            "base_asset": wbtc_denom,
            "quote_asset": usdc_denom,
        }
    }

    response: Optional[dict] = registry.query(query_msg)
    if not response or "address" not in response:
        raise ValueError("Supervault address not found for WBTC/USDC pair.")

    return response["address"]


# step:2 file: check_my_health_factor_on_amber_finance
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from typing import List, Dict


def query_amber_positions(
    address: str,
    contract_address: str,
    lcd_endpoint: str = 'https://rest-kralum.neutron.org',
    rpc_endpoint: str = 'https://rpc-kralum.neutron.org'
) -> List[Dict]:
    """Query the Amber contract for positions owned by *address*.

    :param address:  Bech32 Neutron address of the wallet
    :param contract_address: Deployed Amber contract address
    :param lcd_endpoint:  LCD URL (optional override)
    :param rpc_endpoint:  RPC URL (optional override)
    :returns: List of position dictionaries (empty list if none found)
    :raises RuntimeError: On any networking or contract error
    """
    try:
        # Configure a lightweight client for read-only queries (no signing needed)
        cfg = NetworkConfig(
            chain_id='neutron-1',
            url=rpc_endpoint,
            lcd_url=lcd_endpoint,
            fee_minimum_gas_price=0.002,
            fee_denom='untrn',
            max_gas=2_000_000,
        )
        client = LedgerClient(cfg)

        # Amber-specific query message
        query_msg = {
            'positions_by_owner': {
                'owner': address
            }
        }

        # Perform the smart-contract query (no gas cost)
        raw_response = client.query_contract(contract_address, query_msg)

        # Contract is expected to return: { "positions": [ ... ] }
        return raw_response.get('positions', [])

    except Exception as exc:
        # Wrap lower-level errors so callers have uniform failure type
        raise RuntimeError(f'Error querying Amber contract: {exc}') from exc


# step:3 file: check_my_health_factor_on_amber_finance
from typing import List, Dict


def calculate_health_factors(positions: List[Dict]) -> List[Dict]:
    """Compute (or read) the health-factor for every Amber position.

    If the contract already delivers a `health_factor` field it is preserved.
    Otherwise, `health_factor = collateral_value / debt_value` (∞ if debt = 0).
    """
    results: List[Dict] = []

    for pos in positions:
        collateral = float(pos.get('collateral_value', 0))
        debt = float(pos.get('debt_value', 0))

        # Use on-chain value if available, else compute
        health_factor = pos.get('health_factor')
        if health_factor is None:
            health_factor = float('inf') if debt == 0 else collateral / debt

        results.append({
            'id': pos.get('id'),
            'health_factor': health_factor,
            'collateral': collateral,
            'debt': debt,
        })

    return results


# step:2 file: close_my_leveraged_loop_position_on_amber
# backend/amber_queries.py
import json
import requests
from typing import Dict, Any

LCD_ENDPOINT = "https://lcd-neutron.blockchain.com"  # ← Replace with your preferred LCD
AMBER_CONTRACT = "neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # ← Amber contract address

class AmberError(Exception):
    pass

def query_position_status(user: str) -> Dict[str, Any]:
    """Returns the current Amber position information for a wallet address."""
    query_msg = {
        "position_by_owner": {  # ↳ adapt to Amber’s exact query schema
            "owner": user
        }
    }

    try:
        url = f"{LCD_ENDPOINT}/cosmwasm/wasm/v1/contract/{AMBER_CONTRACT}/smart/{requests.utils.quote(json.dumps(query_msg))}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as err:
        raise AmberError(f"Network error while querying Amber: {err}") from err

    data = resp.json()
    if "data" not in data:
        raise AmberError("Unexpected Amber response format: missing `data` field")

    # The response’s internal structure depends on Amber’s implementation.
    # Here we assume `data` contains `position_id`, `debt`, and `collateral`.
    return {
        "position_id": data["data"].get("position_id"),
        "debt": data["data"].get("debt"),
        "collateral": data["data"].get("collateral"),
    }


# step:5 file: close_my_leveraged_loop_position_on_amber
# backend/confirm_close.py
from amber_queries import query_position_status, AmberError

def confirm_position_closed(user: str) -> bool:
    """Returns True if the user has no outstanding Amber position."""
    try:
        status = query_position_status(user)
    except AmberError as err:
        # Depending on Amber’s implementation, it may throw when a position no longer exists.
        print(f"Amber query failed (likely closed): {err}")
        return True

    # Heuristic: closed when debt == 0 or position_id is None
    return not status.get("position_id") or str(status.get("debt", "0")) == "0"


# step:3 file: lend_1_wbtc_on_amber_finance
import os
from typing import Dict

def get_amber_lending_pool_contract(denom: str) -> Dict[str, str]:
    '''
    Returns the contract address for the Amber Finance lending pool that accepts a given denom.
    In production this could be a DB lookup, config file, or on-chain registry query.
    '''
    contract_map = {
        'WBTC': os.getenv('AMBER_WBTC_POOL_CONTRACT', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    }

    if denom not in contract_map:
        raise ValueError(f'No pool contract configured for {denom}')

    return {'contract_address': contract_map[denom]}


# step:5 file: provide_single-sided_liquidity_of_2_wbtc_to_the_maxbtc_usdc_supervault
# ---------------------------------------------
# query_receipt.py
# ---------------------------------------------
import asyncio
from typing import Dict, Any
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.tx import Tx

SUPERVALUT_EVENT_KEY_SHARE = "share_amount"  # Adjust key names to real contract events
SUPERVALUT_EVENT_KEY_POSITION = "position_id"

async def query_contract_receipt(tx_hash: str, rpc_endpoint: str) -> Dict[str, Any]:
    """Retrieve tx details and parse wasm events for share tokens / position IDs."""
    try:
        network_cfg = NetworkConfig(
            chain_id="neutron-1",
            url=rpc_endpoint,
        )
        client = LedgerClient(network_cfg)

        # Blocking query; wrap in thread executor if running in non-async framework
        tx: Tx = client.query_tx(tx_hash)
        if not tx:
            raise RuntimeError(f"Transaction {tx_hash} not found on chain.")

        parsed: Dict[str, Any] = {}
        for log in tx.logs:  # type: ignore[attr-defined]
            for event in log.get("events", []):
                if event.get("type") == "wasm":
                    for attr in event.get("attributes", []):
                        key = attr.get("key")
                        val = attr.get("value")
                        if key == SUPERVALUT_EVENT_KEY_SHARE:
                            parsed["vault_shares"] = val
                        if key == SUPERVALUT_EVENT_KEY_POSITION:
                            parsed["position_id"] = val
        return parsed if parsed else {"message": "No share or position information found."}

    except Exception as exc:
        raise RuntimeError(f"Failed to query receipt: {exc}") from exc

# For synchronous frameworks wrap via asyncio.run
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python query_receipt.py <tx_hash> <rpc_endpoint>")
        sys.exit(1)
    txh, rpc = sys.argv[1:]
    print(asyncio.run(query_contract_receipt(txh, rpc)))


# step:2 file: retrieve_projected_ntrn_rewards_based_on_current_point_total
import base64
import json
from typing import Dict, Any
import aiohttp

CONTRACT_ADDRESS = 'neutron1yu55umrtnna36vyjvhexp6q2ktljunukzxp9vptsfnylequg7gvqrcqf42'
LCD_URL = 'https://lcd.neutron.org'  # Replace with your preferred endpoint if needed.

async def query_points_contract(user_address: str,
                                contract_address: str = CONTRACT_ADDRESS,
                                lcd_url: str = LCD_URL) -> Dict[str, Any]:
    """Query the Points smart contract to obtain the user’s point total."""

    # Build the query message expected by the contract
    query_msg = {"points": {"user": user_address}}
    query_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
    endpoint = f"{lcd_url}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{query_b64}"

    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                raise RuntimeError(f"Contract query failed ({resp.status}): {error_body}")
            result = await resp.json()

    # Expected shape: {"data": {"points": "12345"}}
    try:
        points_str = result.get("data", {}).get("points", "0")
        points = int(points_str)
    except (AttributeError, ValueError):
        raise RuntimeError(f"Unexpected response from contract: {result}")

    return {"user_address": user_address, "points": points}


# step:3 file: retrieve_projected_ntrn_rewards_based_on_current_point_total
import base64
import json
from typing import Dict, Any
import aiohttp

CONTRACT_ADDRESS = 'neutron1yu55umrtnna36vyjvhexp6q2ktljunukzxp9vptsfnylequg7gvqrcqf42'
LCD_URL = 'https://lcd.neutron.org'

async def fetch_phase_reward_params(contract_address: str = CONTRACT_ADDRESS,
                                    lcd_url: str = LCD_URL) -> Dict[str, Any]:
    """Retrieve campaign parameters (total NTRN, phase length, per-point rate, etc.)."""

    # Fallback values in case on-chain query fails
    default_params: Dict[str, Any] = {
        "total_ntrn_allocated": 1_000_000,                 # 1 000 000 NTRN
        "phase_length_seconds": 28 * 24 * 60 * 60,         # 28-day phase
        "per_point_rate": 0.0001,                          # 0.0001 NTRN per point
        "multiplier": 1.0                                  # Base multiplier
    }

    query_msg = {"config": {}}
    query_b64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
    endpoint = f"{lcd_url}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{query_b64}"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Non-200 response: {resp.status}")
                raw = await resp.json()
                on_chain_cfg = raw.get("data", {})
                # Validate required fields
                for key in ["total_ntrn_allocated", "phase_length_seconds", "per_point_rate"]:
                    if key not in on_chain_cfg:
                        raise KeyError(f"Missing key {key} in on-chain config")
                return on_chain_cfg
        except Exception as err:
            # Log the error and supply defaults
            print(f"Warning: using default reward params due to error: {err}")
            return default_params


# step:4 file: retrieve_projected_ntrn_rewards_based_on_current_point_total
from typing import Dict, Any

def calculate_projected_rewards(points: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute projected end-of-phase rewards for the user."""

    per_point_rate: float = float(params.get("per_point_rate", 0))
    multiplier: float = float(params.get("multiplier", 1))

    projected_ntrn = points * per_point_rate * multiplier

    return {
        "points": points,
        "per_point_rate": per_point_rate,
        "multiplier": multiplier,
        "projected_ntrn": projected_ntrn
    }


# step:2 file: boost_my_rewards_to_3x_by_locking_additional_ntrn_equal_to_my_tvl
from decimal import Decimal
from typing import Dict
from cosmpy.aerial.client import LedgerClient, NetworkConfig

# RPC / network configuration
NETWORK_CFG = NetworkConfig(
    chain_id="neutron-1",
    url="https://rpc-kralum.neutron.org",
    fee_minimum_gas_price=0.025,
    fee_denomination="untrn",
    staking_denomination="untrn",
)

# List the vault/venue contracts to inspect.  Update as new venues roll out.
VENUES = [
    {
        "name": "Amber Finance",
        "contract": "neutron1ambercontract...",
        "query_fn": lambda addr: {"user_info": {"wallet": addr}},
        "balance_key": "deposit_amount",
    },
    {
        "name": "Supervaults",
        "contract": "neutron1supervault...",
        "query_fn": lambda addr: {"position": {"address": addr}},
        "balance_key": "amount",
    },
    # Add additional venues here as needed
]

async def query_user_tvl(wallet_address: str) -> Decimal:
    """Return total TVL (in micro-NTRN) held by the wallet across all venues."""
    client = LedgerClient(NETWORK_CFG)
    total_tvl = Decimal(0)

    try:
        for venue in VENUES:
            result: Dict = client.query_contract(
                venue["contract"],
                venue["query_fn"](wallet_address),
            )
            raw_amount = result.get(venue["balance_key"], "0")
            total_tvl += Decimal(raw_amount)
    except Exception as err:
        raise RuntimeError(f"TVL query failed: {err}")

    return total_tvl


# step:3 file: boost_my_rewards_to_3x_by_locking_additional_ntrn_equal_to_my_tvl
from decimal import Decimal

BOOST_TARGET = Decimal(3)  # Desired total boost (1× base + 2× extra)

def calculate_required_amount(current_tvl: Decimal) -> Decimal:
    """Compute NTRN needed to reach a 3× boost for the given TVL."""
    if current_tvl <= 0:
        return Decimal(0)

    additional_multiplier = BOOST_TARGET - Decimal(1)  # Only extra boost needed
    required = current_tvl * additional_multiplier

    # Round to whole micro-denom units (no fractions of micro-NTRN)
    return required.quantize(Decimal(0))


# step:4 file: boost_my_rewards_to_3x_by_locking_additional_ntrn_equal_to_my_tvl
from decimal import Decimal
from cosmpy.aerial.client import LedgerClient

CW20_NTRN_CONTRACT = "neutron1ntrncw20..."  # Update with actual contract address

async def query_cw20_balance(wallet_address: str) -> Decimal:
    """Return cw20 NTRN balance (in micro-NTRN) for a wallet."""
    client = LedgerClient(NETWORK_CFG)
    try:
        response = client.query_contract(
            CW20_NTRN_CONTRACT,
            {"balance": {"address": wallet_address}},
        )
        return Decimal(response.get("balance", "0"))
    except Exception as err:
        raise RuntimeError(f"Unable to query cw20 balance: {err}")


# step:6 file: set_my_boost_target_to_my_ethereum_address
from cosmpy.aerial.client import LedgerClient, NetworkConfig


def query_boost_target(contract_address: str, rpc_endpoint: str = 'https://rpc-kralum.neutron.org') -> str:
    """Return the EVM address currently set as target in the Boost contract."""
    try:
        cfg = NetworkConfig(
            chain_id='neutron-1',
            url=rpc_endpoint,
            fee_minimum_gas_price=0,
            fee_denomination='untrn',
            staking_denomination='untrn',
        )

        client = LedgerClient(cfg)
        query_msg = {"target": {}}
        response = client.query_contract(contract_address, query_msg)

        evm_address = response.get('evm_address')
        if evm_address is None:
            raise ValueError("Response did not contain 'evm_address'.")

        return evm_address
    except Exception as err:
        print(f'Error querying boost contract: {err}')
        raise


# step:2 file: show_my_total_bitcoin_summer_points_earned_in_the_current_phase
import os
from typing import Optional
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.config import NetworkConfig

# ---------- Neutron client bootstrap ----------
RPC_ENDPOINT = os.getenv("NEUTRON_RPC_URL", "https://rpc-kralum.neutron.org:443")
NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url=RPC_ENDPOINT,
    fee_minimum_gas_price=0.025,  # not used for queries but required by config
    fee_denomination="untrn",
)
CLIENT = LedgerClient(NETWORK)
# ---------------------------------------------

def fetch_current_campaign_phase(campaign_contract_addr: str) -> Optional[int]:
    """Query BTC Summer Campaign contract for the active phase ID.

    Args:
        campaign_contract_addr: On-chain address of the campaign contract.

    Returns:
        The `phase_id` (int) if present, otherwise `None`.
    """
    try:
        query_msg = {"get_current_phase": {}}
        response = CLIENT.query_contract(campaign_contract_addr, query_msg)
        # Expected response shape: {"phase_id": 3}
        return response.get("phase_id")
    except Exception as err:
        # Log for observability; propagate upwards for the API layer to handle.
        print(f"[ERROR] Campaign phase query failed: {err}")
        raise


# step:3 file: show_my_total_bitcoin_summer_points_earned_in_the_current_phase
def query_phase_points(points_contract_addr: str, user_addr: str, phase_id: int) -> int:
    """Return the amount of points a user has accumulated in a particular phase.

    Args:
        points_contract_addr: Address of the Points contract.
        user_addr:        Neutron address of the user.
        phase_id:         Phase ID supplied by `fetch_current_campaign_phase`.
    Returns:
        The integer number of points (0 if contract returns empty).
    """
    try:
        query_msg = {
            "get_phase_points": {
                "address": user_addr,
                "phase_id": phase_id
            }
        }
        response = CLIENT.query_contract(points_contract_addr, query_msg)
        # Expected response: {"points": "12345"}
        return int(response.get("points", "0"))
    except Exception as err:
        print(f"[ERROR] Phase points query failed: {err}")
        raise


# step:1 file: list_current_amber_lending_markets_and_apys
import os
from cosmpy.aerial.client import LedgerClient  # Only imported to show it could be used later if on‑chain discovery is required


def get_controller_address(network: str = "neutron") -> str:
    """Return the Lens (controller) contract address.

    Priority order:
      1. `AMBER_LENS_ADDRESS` environment variable (useful for dev / test).
      2. Hard‑coded list of known addresses, keyed by network.

    Raises
    ------
    ValueError
        If the address cannot be determined.
    """

    # 1. Allow runtime override via env variable
    env_key = "AMBER_LENS_ADDRESS"
    if env_key in os.environ and os.environ[env_key].strip():
        return os.environ[env_key].strip()

    # 2. Fallback map of known addresses per network
    known_addresses = {
        "neutron": "neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # <-- replace with real mainnet address
        "pion-1": "neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"   # <-- replace with real testnet address
    }

    address = known_addresses.get(network)
    if not address:
        raise ValueError(
            f"No known Amber Lens address for network '{network}'. "
            f"Set the {env_key} environment variable to override."
        )
    return address


# step:2 file: list_current_amber_lending_markets_and_apys
from typing import List, Dict
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.config import NetworkConfig


async def query_markets(contract_address: str,
                        rpc_endpoint: str = "https://rpc-kralum.neutron.org",
                        chain_id: str = "neutron-1") -> List[Dict]:
    """Return the array of market objects exposed by the Lens contract."""
    cfg = NetworkConfig(chain_id=chain_id, url=rpc_endpoint)
    client = LedgerClient(cfg)

    query_msg = {"markets": {}}
    try:
        response = client.query_contract_smart(contract_address, query_msg)
    except Exception as err:
        raise RuntimeError(f"Failed to query markets from {contract_address}: {err}")

    # The contract usually replies with `{ "markets": [ ... ] }`
    markets = response.get("markets", [])
    if not markets:
        raise ValueError("No markets returned by the contract.")

    return markets


# step:3 file: list_current_amber_lending_markets_and_apys
from typing import List, Dict
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.config import NetworkConfig


def query_market_states(contract_address: str,
                        markets: List[Dict],
                        rpc_endpoint: str = "https://rpc-kralum.neutron.org",
                        chain_id: str = "neutron-1") -> List[Dict]:
    """Return each market enriched with its current state (rates, factors, etc.)."""

    cfg = NetworkConfig(chain_id=chain_id, url=rpc_endpoint)
    client = LedgerClient(cfg)

    enriched: List[Dict] = []
    for market in markets:
        # Try the most common key names for ID/symbol.
        market_id = market.get("market_id") or market.get("id") or market.get("symbol")
        if market_id is None:
            # Skip if we cannot locate an identifier.
            print(f"[WARN] Missing market identifier in: {market}")
            continue

        query_msg = {"market_state": {"market_id": market_id}}
        try:
            state = client.query_contract_smart(contract_address, query_msg)
        except Exception as err:
            print(f"[ERROR] Could not fetch state for market '{market_id}': {err}")
            continue

        # Merge base market data with the dynamic state.
        market_full = {**market, **state}
        enriched.append(market_full)

    if not enriched:
        raise RuntimeError("No market states could be fetched; aborting.")

    return enriched


# step:4 file: list_current_amber_lending_markets_and_apys
from typing import List, Dict

SECONDS_PER_YEAR = 365 * 24 * 60 * 60  # 31,536,000


def rate_to_apy(rate_per_second: float) -> float:
    """Convert a discrete per-second rate into an APY (compounded every second)."""
    # Guard against nonsensical negative rates.
    rate_per_second = max(rate_per_second, 0.0)
    return (1.0 + rate_per_second) ** SECONDS_PER_YEAR - 1.0


def calculate_market_apys(market_states: List[Dict]) -> List[Dict]:
    """Append `supply_apy` and `borrow_apy` fields to each market in the list."""
    results: List[Dict] = []
    for m in market_states:
        supply_rate = float(m.get("supply_rate_per_second", 0))
        borrow_rate = float(m.get("borrow_rate_per_second", 0))

        m["supply_apy"] = rate_to_apy(supply_rate)
        m["borrow_apy"] = rate_to_apy(borrow_rate)
        results.append(m)

    return results


# step:4 file: transfer_a_boost_receipt_nft_to_another_address
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from typing import Dict

NETWORK_CONFIG = NetworkConfig(
    chain_id='neutron-1',
    url='https://rpc-kralum.neutron.org',  # Update to a reliable RPC
    fee_minimum_gas_price=0.0025,
    fee_denomination='untrn',
)

CONTRACT_ADDRESS_RECEIPT_NFT = '<YOUR_CONTRACT_ADDRESS>'

client = LedgerClient(NETWORK_CONFIG)

def assert_nft_owner(token_id: str, expected_owner: str) -> Dict:
    '''
    Queries the CW721 contract for ownership of token_id and asserts the
    expected_owner matches the current owner.
    '''
    try:
        query = {'owner_of': {'token_id': token_id}}
        response = client.query_contract_smart(CONTRACT_ADDRESS_RECEIPT_NFT, query)
        current_owner = response.get('owner')
        if current_owner != expected_owner:
            raise ValueError(
                f'Ownership mismatch: expected {expected_owner}, got {current_owner}'
            )
        return {'token_id': token_id, 'owner': current_owner}
    except Exception as e:
        raise RuntimeError(f'Failed to verify NFT ownership: {str(e)}') from e


# step:2 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.config import NetworkConfig
import json

RPC_ENDPOINT = "https://rpc.neutron.org"
CHAIN_ID = "neutron-1"
VESTING_CONTRACT = "neutron1dz57hjkdytdshl2uyde0nqvkwdww0ckx7qfe05raz4df6m3khfyqfnj0nr"

def query_claimable_rewards(address: str) -> int:
    """Return claimable rewards for the given address in micro-denom (uNTRN)."""
    try:
        cfg = NetworkConfig(
            chain_id=CHAIN_ID,
            url=RPC_ENDPOINT,
            fee_denom="untrn",
            gas_price=0.025,
        )
        client = LedgerClient(cfg)
        query_msg = {"claimable_rewards": {"address": address}}
        # cosmpy automatically JSON-encodes the query
        result = client.query_contract_json(VESTING_CONTRACT, query_msg)
        # Expected result shape: {"claimable": "12345"}
        return int(result.get("claimable", "0"))
    except Exception as exc:
        raise RuntimeError(f"Failed to query claimable rewards: {exc}")


# step:3 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
def validate_claimable_amount(claimable: int) -> None:
    """Abort the workflow if the claimable amount is zero or negative."""
    if claimable <= 0:
        raise ValueError("Claimable amount is zero; nothing to vest.")


# step:6 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.config import NetworkConfig

RPC_ENDPOINT = "https://rpc.neutron.org"
CHAIN_ID = "neutron-1"
VESTING_CONTRACT = "neutron1dz57hjkdytdshl2uyde0nqvkwdww0ckx7qfe05raz4df6m3khfyqfnj0nr"

def query_vesting_schedule(address: str) -> dict:
    """Return the (possibly newly-created) vesting schedule for the address."""
    try:
        cfg = NetworkConfig(
            chain_id=CHAIN_ID,
            url=RPC_ENDPOINT,
            fee_denom="untrn",
            gas_price=0.025,
        )
        client = LedgerClient(cfg)
        query_msg = {"vesting_schedule": {"address": address}}
        return client.query_contract_json(VESTING_CONTRACT, query_msg)
    except Exception as exc:
        raise RuntimeError(f"Failed to query vesting schedule: {exc}")


# step:1 file: enable_usdc_gas_payments_for_my_next_transaction
import os
import httpx
from typing import List

NEUTRON_LCD_URL = os.getenv('NEUTRON_LCD_URL', 'https://rest-kralum.neutron.org')

async def query_dynamic_fees_supported_assets(denom: str = 'uusdc') -> bool:
    '''Return True if the provided denom is present in the dynamic fee params.'''
    endpoint = f'{NEUTRON_LCD_URL}/neutron/dynamicfees/params'
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            ntrn_prices: List[dict] = data.get('params', {}).get('ntrn_prices', [])
            return any(price.get('denom') == denom for price in ntrn_prices)
        except httpx.HTTPError as err:
            raise RuntimeError(f'Failed to query dynamic fee params: {err}') from err


# step:2 file: enable_usdc_gas_payments_for_my_next_transaction
import os
import httpx

NEUTRON_LCD_URL = os.getenv('NEUTRON_LCD_URL', 'https://rest-kralum.neutron.org')

async def query_min_gas_price(denom: str = 'uusdc') -> float:
    '''Returns the minimum gas price for the given denom as a float.'''
    endpoint = f'{NEUTRON_LCD_URL}/neutron/globalfee/min_gas_prices'
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            prices = data.get('min_gas_prices', [])
            for price in prices:
                if price.get('denom') == denom:
                    return float(price.get('amount', '0'))
            raise ValueError(f'Denom {denom} not found in min_gas_prices')
        except httpx.HTTPError as err:
            raise RuntimeError(f'Failed to query global fee: {err}') from err


# step:2 file: instantly_claim_50%_of_my_ntrn_staking_rewards
import asyncio
from typing import List, Dict
from cosmpy.aerial.client import LedgerClient, NetworkConfig

# Neutron network configuration
NETWORK = NetworkConfig(
    chain_id='neutron-1',
    url='https://rpc-kralum.neutron.org/',
    fee_minimum_gas_price=0.025,
    fee_denomination='untrn'
)

async def query_pending_staking_rewards(delegator_address: str) -> List[Dict]:
    """Return a list of dictionaries with rewards per validator."""
    client = LedgerClient(NETWORK)
    try:
        resp = client.distribution.get_total_rewards(delegator_address)
        rewards: List[Dict] = []
        for entry in resp.rewards:
            # Each entry.reward is a list of Coin objects; pick the untrn coin
            coin = next((c for c in entry.reward if c.denom == 'untrn'), None)
            amount_int = int(coin.amount) if coin else 0
            rewards.append({
                'validator_address': entry.validator_address,
                'amount': amount_int,  # micro-denom (untrn)
                'denom': 'untrn'
            })
        return rewards
    except Exception as e:
        raise RuntimeError(f'Error while querying rewards: {e}')


# step:3 file: instantly_claim_50%_of_my_ntrn_staking_rewards
from typing import List, Dict

def calculate_partial_rewards(rewards: List[Dict], ratio: float = 0.5) -> List[Dict]:
    # Validate ratio input
    if not 0 < ratio <= 1:
        raise ValueError('ratio must be between 0 and 1')

    partial: List[Dict] = []
    for r in rewards:
        partial.append({
            'validator_address': r['validator_address'],
            'amount': int(r['amount'] * ratio),
            'denom': r['denom']
        })
    return partial


# step:2 file: check_my_current_bitcoin_summer_boost_multiplier
import json
from typing import Dict, Any

from cosmpy.aerial.client import LedgerClient, NetworkConfig

# --- Network configuration (adjust RPC URL if needed) ---
NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url="https://rpc.kralum.neutron.org",   # Public Neutron RPC; replace with your preferred endpoint
    fee_minimum_gas_price=0.025,
    fee_denomination="untrn",
)

# Address of the Bitcoin Summer boost contract (replace with the actual address)
BOOST_CONTRACT_ADDRESS = "neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

def query_multiplier(user_address: str) -> Dict[str, Any]:
    """Return raw JSON response from the contract’s get_multiplier query."""
    try:
        client = LedgerClient(NETWORK)
        query_msg = {
            "get_multiplier": {
                "address": user_address
            }
        }

        # Query the contract; cosmpy returns bytes
        response_bytes = client.query_contract_smart(BOOST_CONTRACT_ADDRESS, query_msg)
        response: Dict[str, Any] = json.loads(response_bytes)
        return response

    except Exception as err:
        # Wrap and re-raise for upstream handling/logging
        raise RuntimeError(f"Contract query failed: {err}") from err


# step:3 file: check_my_current_bitcoin_summer_boost_multiplier
def parse_multiplier_response(response: dict) -> float:
    """Extracts the multiplier (e.g., 1.25) from the contract response."""
    try:
        # Common response shapes:
        # {"multiplier": "1.25"}  OR  {"data": {"multiplier": "1.25"}}
        multiplier_str = response.get("multiplier") or response.get("data", {}).get("multiplier")
        if multiplier_str is None:
            raise KeyError("multiplier field missing")
        return float(multiplier_str)
    except (ValueError, KeyError) as err:
        raise ValueError(f"Unable to parse multiplier: {err}") from err


# step:1 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
import os
from bitcoinlib.wallets import wallet_exists, HDWallet, Wallet


def get_btc_wallet_address(wallet_name: str = "btc_source") -> str:
    """Retrieve a Bitcoin address that will source the 1 BTC. If the wallet
    does not yet exist it is created automatically (main-net, SegWit)."""
    try:
        if not wallet_exists(wallet_name):
            # NOTE: In production import from a mnemonic or xpriv instead of
            # generating a fresh wallet on the fly.
            wallet = HDWallet.create(wallet_name, witness_type="segwit", network="bitcoin")
        else:
            wallet = Wallet(wallet_name)
        key = wallet.get_key()  # first external key
        return key.address
    except Exception as e:
        raise RuntimeError(f"Unable to obtain BTC wallet address: {e}")


# step:2 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
import os
import requests

SOLV_API_ENDPOINT = os.getenv("SOLV_GATEWAY_API", "https://api.solv.finance/gateway")
SOLV_API_KEY = os.getenv("SOLV_GATEWAY_KEY")


def generate_solvbtc_deposit_address(eth_destination: str) -> str:
    """Return a deposit address for exactly 1 BTC that is mapped to
    `eth_destination`. The Solv backend will watch that address and mint
    solvBTC once the deposit is finalised."""
    payload = {
        "asset": "BTC",
        "amount": "1",
        "destEvmAddress": eth_destination
    }
    headers = {
        "Authorization": f"Bearer {SOLV_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(f"{SOLV_API_ENDPOINT}/deposit-address", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["depositAddress"]
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to obtain deposit address: {e}")


# step:3 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
from decimal import Decimal
from bitcoinlib.wallets import Wallet

SATOSHI = 100_000_000  # 1 BTC in sats


def construct_btc_tx(wallet_name: str, destination: str, amount_btc: Decimal = Decimal("1")) -> str:
    """Build and sign the TX but do *not* broadcast it yet. Returns hex-encoded
    transaction ready for broadcasting."""
    try:
        wallet = Wallet(wallet_name)
        amount_sat = int(amount_btc * SATOSHI)
        # `offline=True` so nothing is pushed to the network yet
        tx = wallet.send_to(destination, amount_sat, network_fee="fast", offline=True, replace_by_fee=False)
        wallet.sign_transaction(tx)
        return tx.raw_hex()
    except Exception as e:
        raise RuntimeError(f"Could not construct BTC transaction: {e}")


# step:4 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
import json
import requests
import os

BTC_RPC_URL = os.getenv("BITCOIN_RPC_URL", "http://user:password@127.0.0.1:8332")


def broadcast_btc_tx(raw_tx_hex: str) -> str:
    """Push `raw_tx_hex` to the Bitcoin node via JSON-RPC and return the TXID."""
    try:
        payload = json.dumps({
            "jsonrpc": "1.0",
            "id": "broadcast",
            "method": "sendrawtransaction",
            "params": [raw_tx_hex]
        })
        res = requests.post(BTC_RPC_URL, data=payload, headers={"Content-Type": "application/json"})
        res.raise_for_status()
        return res.json()["result"]
    except requests.RequestException as e:
        raise RuntimeError(f"Broadcast failed: {e}")


# step:5 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
import json, time, os, requests

RPC_URL = os.getenv("BITCOIN_RPC_URL", "http://user:password@127.0.0.1:8332")
TARGET_CONF = int(os.getenv("BTC_CONFIRMATIONS", "6"))
POLL_INTERVAL = 60  # seconds


def _rpc(method: str, params: list):
    payload = json.dumps({"jsonrpc": "1.0", "id": "rpc", "method": method, "params": params})
    r = requests.post(RPC_URL, data=payload, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()["result"]


def get_tx_confirmations(txid: str) -> int:
    tx_data = _rpc("gettransaction", [txid])
    return tx_data.get("confirmations", 0)


def wait_for_confirmations(txid: str, target: int = TARGET_CONF):
    """Block until `txid` reaches `target` confirmations."""
    try:
        while True:
            confs = get_tx_confirmations(txid)
            print(f"TX {txid} ➜ {confs}/{target} confirmations")
            if confs >= target:
                return confs
            time.sleep(POLL_INTERVAL)
    except Exception as e:
        raise RuntimeError(f"Error polling confirmations: {e}")


# step:6 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
import os, json, requests
from web3 import Web3

INFURA_URL = os.getenv("ETH_RPC_URL", "https://mainnet.infura.io/v3/your_key")
PRIVATE_KEY = os.getenv("ETH_PRIVATE_KEY")
SOLV_MINT_CONTRACT = os.getenv("SOLV_MINT_CONTRACT", "0xYourMintContract")
SOLV_MINT_ABI_PATH = os.getenv("SOLV_MINT_ABI", "./SolvMint.json")

w3 = Web3(Web3.HTTPProvider(INFURA_URL))


def attest_and_mint_solvbtc(btc_txid: str, eth_destination: str) -> str:
    """Call the mint() method on Solv’s contract with the on-chain BTC proof."""
    # 1. Obtain the BTC-to-Ethereum proof from Solv Gateway
    proof_resp = requests.get(f"{SOLV_API_ENDPOINT}/proof/{btc_txid}", headers={"Authorization": f"Bearer {SOLV_API_KEY}"})
    proof_resp.raise_for_status()
    proof_blob = proof_resp.json()["proof"]

    # 2. Load contract interface
    with open(SOLV_MINT_ABI_PATH, "r", encoding="utf-8") as f:
        mint_abi = json.load(f)
    contract = w3.eth.contract(address=SOLV_MINT_CONTRACT, abi=mint_abi)

    # 3. Build transaction
    account = w3.eth.account.from_key(PRIVATE_KEY)
    tx = contract.functions.mint(eth_destination, proof_blob).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 300_000,
        "gasPrice": w3.to_wei("30", "gwei")
    })

    # 4. Sign & send
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if receipt.status != 1:
        raise RuntimeError("Minting transaction failed on Ethereum.")
    return receipt.transactionHash.hex()


# step:7 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
import os, time
from axelartools import AxelarClient  # third-party helper sdk (pip install axelartools)

AXELAR_API = os.getenv("AXELAR_API", "https://axelar-api.axelar.dev")
TOKEN_SYMBOL = "solvBTC"
SRC_CHAIN = "Ethereum"
DST_CHAIN = "neutron-1"


def bridge_asset_to_neutron(eth_sender: str, neutron_receiver: str, reference_tx: str):
    """Initiate an Axelar transfer of 1 solvBTC and wait for completion."""
    client = AxelarClient(base_url=AXELAR_API)
    try:
        transfer = client.create_transfer(
            from_chain=SRC_CHAIN,
            to_chain=DST_CHAIN,
            asset=TOKEN_SYMBOL,
            amount="100000000",  # 1 token, assuming 8 decimals
            from_address=eth_sender,
            to_address=neutron_receiver,
            reference_tx=reference_tx
        )
        # Wait for status = completed
        while True:
            info = client.get_transfer(transfer["id"])
            if info["status"] == "completed":
                return info
            if info["status"] == "failed":
                raise RuntimeError(f"Axelar bridge failed: {info}")
            time.sleep(30)
    except Exception as e:
        raise RuntimeError(f"Bridge error: {e}")


# step:8 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
import os
from cosmpy.aerial.client import LedgerClient, NetworkConfig

NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url=os.getenv("NEUTRON_LCD", "https://lcd-kralum.neutron.org:443"),
    fee_minimum_gas_price=0.025,
)


def query_balance_neutron(address: str, denom: str = "asolvBTC") -> int:
    """Return the balance (in micro-units) of the `denom` held by `address`."""
    try:
        client = LedgerClient(NETWORK)
        return client.query_bank_balance(address, denom=denom)
    except Exception as e:
        raise RuntimeError(f"Neutron balance query failed: {e}")


# step:2 file: execute_an_emergency_withdrawal_for_the_user’s_amber_trading_position
from typing import List, Dict

from cosmpy.aerial.client import LCDClient
from cosmpy.aerial.config import NetworkConfig

# ----------------------------
# Configuration (edit to suit)
# ----------------------------
AMBER_CONTRACT_ADDRESS = "neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # TODO: put the real contract address here
NETWORK = NetworkConfig(
    chain_id="neutron-1",
    url="https://rest-kralum.neutron-1.neutron.org"
)

lcd = LCDClient(NETWORK)


def query_amber_positions(owner: str) -> List[Dict]:
    """Return every Amber position belonging to `owner`. Raises on error."""
    try:
        query_msg = {"positions": {"owner": owner}}
        result: Dict = lcd.wasm.contract_query(AMBER_CONTRACT_ADDRESS, query_msg)
        # The Amber contract is expected to return `{ "positions": [ ... ] }`
        return result.get("positions", [])
    except Exception as exc:
        raise RuntimeError(f"Failed to query Amber positions for {owner}: {exc}") from exc


# Optional: expose via FastAPI for the frontend
# ---------------------------------------------
# from fastapi import FastAPI, HTTPException
# app = FastAPI()
#
# @app.get("/api/amber/positions")
# def get_positions(address: str):
#     try:
#         return query_amber_positions(address)
#     except RuntimeError as e:
#         raise HTTPException(status_code=500, detail=str(e))


# step:2 file: lend_2_unibtc_on_amber_finance
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.config import network_config

UNIBTC_CONTRACT_ADDRESS = 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
REQUIRED_AMOUNT = 2 * 1_000_000  # 2 uniBTC in micro-units


def check_token_balance(address: str, rpc_endpoint: str = 'https://rpc-kralum.neutron-1.neutron.org') -> bool:
    """Raise if the CW20 balance of `address` is below REQUIRED_AMOUNT."""
    try:
        cfg = network_config('neutron-1', rpc_endpoint)
        client = LedgerClient(cfg)
        query_msg = {"balance": {"address": address}}
        resp = client.query_contract_smart(UNIBTC_CONTRACT_ADDRESS, query_msg)
        balance = int(resp.get('balance', '0'))
        if balance < REQUIRED_AMOUNT:
            raise ValueError(f'Insufficient uniBTC: have {balance}, need {REQUIRED_AMOUNT}')
        return True
    except Exception as exc:
        raise RuntimeError(f'Error querying uniBTC balance: {exc}')


# step:2 file: cancel_(unlock)_the_user’s_ntrn_stake_lock_once_the_vesting_period_has_ended
# ---------- step-2: query_stake_lock_status.py ----------
import httpx, datetime, os
from cosmpy.aio.client import LedgerClient

LCD_ENDPOINT = os.getenv("LCD_ENDPOINT", "https://lcd-kralum.neutron.org")

class LockNotFound(Exception):
    pass

async def query_stake_lock_status(lock_id: int, address: str) -> dict:
    """Return detailed lock information and boolean `can_cancel`."""
    async with httpx.AsyncClient(base_url=LCD_ENDPOINT, timeout=10) as client:
        # NOTE: The exact LCD path may differ depending on the module name.
        # Replace `/neutron/vesting/locks/` with the actual REST route if different.
        response = await client.get(f"/neutron/vesting/locks/{lock_id}")
        if response.status_code == 404:
            raise LockNotFound(f"Lock id {lock_id} not found on-chain.")
        response.raise_for_status()
        data = response.json()

    lock_owner = data.get("owner")
    if lock_owner != address:
        raise ValueError(f"Lock #{lock_id} is owned by {lock_owner}, not {address}.")

    unlock_time_str = data.get("unlock_time")  # ISO time string expected
    unlock_time = datetime.datetime.fromisoformat(unlock_time_str.replace("Z", "+00:00"))
    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    can_cancel = now >= unlock_time

    return {
        "lock_id": lock_id,
        "owner": lock_owner,
        "amount": data.get("amount"),
        "unlock_time": unlock_time_str,
        "can_cancel": can_cancel,
    }


# step:3 file: open_a_5×_leveraged_loop_position_with_1_maxbtc_on_amber
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from typing import Dict, Any

AMBER_CONTRACT = "neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"

async def query_amber_market_parameters() -> Dict[str, Any]:
    """Fetch market parameters (max leverage, collateral factors, rates) from Amber Finance."""
    config = NetworkConfig(
        chain_id="neutron-1",
        url="https://rest-kralum.neutron.org",  # LCD endpoint
        fee_denom="untrn",
        gas_price=0.01,
    )

    client = LedgerClient(config)

    try:
        params = client.query_contract_smart(
            AMBER_CONTRACT,
            {"market_params": {}}
        )
        # Example structure ⇒ {"max_leverage": "6x", "collateral_factor": "0.9", ...}
        if float(params.get("max_leverage", "0x").rstrip("x")) < 5:
            raise ValueError("5× leverage exceeds Amber’s max leverage setting.")
        return params
    except Exception as exc:
        raise RuntimeError(f"Unable to fetch Amber market parameters: {exc}") from exc


# step:6 file: open_a_5×_leveraged_loop_position_with_1_maxbtc_on_amber
import asyncio
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from typing import Dict, Any

AMBER_CONTRACT = "neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"

async def query_position_status(user_address: str, poll_interval: float = 3.0, max_attempts: int = 20) -> Dict[str, Any]:
    """Repeatedly query Amber Finance until the user’s newest position is found."""
    config = NetworkConfig(
        chain_id="neutron-1",
        url="https://rest-kralum.neutron.org",
        fee_denom="untrn",
        gas_price=0.01,
    )
    client = LedgerClient(config)

    for attempt in range(max_attempts):
        try:
            response = client.query_contract_smart(
                AMBER_CONTRACT,
                {"positions": {"address": user_address}}
            )
            # Assume response ⇒ {"positions": [{"id": 42, "borrowed": "4.0 maxBTC", "health": "0.85"}, ...]}
            if response.get("positions"):
                # Pick the most recent (highest ID) position
                latest = sorted(response["positions"], key=lambda p: p["id"], reverse=True)[0]
                return latest
        except Exception as exc:
            # Log and continue polling
            print(f"Attempt {attempt+1}: Amber query failed → {exc}")

        await asyncio.sleep(poll_interval)

    raise TimeoutError("Position not found within polling window.")


# step:1 file: confirm_forfeitable_reward_structure_for_my_current_vault
async def get_vault_contract_address(user_address: str, lcd_endpoint: str, factory_address: str) -> str:
    """
    Look up the vault contract address for a given user by querying the on-chain vault-factory contract.

    :param user_address: User's Bech32 Neutron address.
    :param lcd_endpoint: Public LCD/REST endpoint for Neutron (e.g., "https://rest-kralum.neutron-1.neutron.org").
    :param factory_address: Bech32 address of the vault factory contract.
    :return: Vault contract address (Bech32).
    :raises RuntimeError: If the query fails or returns an unexpected result.
    """
    from cosmpy.aerial.client import LedgerClient
    from cosmpy.aerial.config import NetworkConfig
    from cosmpy.aerial.exceptions import LedgerQueryException

    cfg = NetworkConfig(
        chain_id="neutron-1",
        url=lcd_endpoint,
        fee_minimum_gas_price="0.0025untrn"
    )
    client = LedgerClient(cfg)

    # Factory contract is expected to expose a `{ vault_address: { owner } }` query.
    query_msg = {"vault_address": {"owner": user_address}}

    try:
        resp = client.query_contract(factory_address, query_msg)
        return resp["vault_address"]
    except (KeyError, LedgerQueryException) as exc:
        raise RuntimeError(f"Unable to obtain vault address for {user_address}: {exc}")


# step:2 file: confirm_forfeitable_reward_structure_for_my_current_vault
async def query_vault_config(vault_contract_address: str, lcd_endpoint: str) -> dict:
    """Returns the full config object for a vault contract."""
    from cosmpy.aerial.client import LedgerClient
    from cosmpy.aerial.config import NetworkConfig
    from cosmpy.aerial.exceptions import LedgerQueryException

    cfg = NetworkConfig(
        chain_id="neutron-1",
        url=lcd_endpoint,
        fee_minimum_gas_price="0.0025untrn"
    )
    client = LedgerClient(cfg)

    query_msg = {"config": {}}

    try:
        return client.query_contract(vault_contract_address, query_msg)
    except LedgerQueryException as exc:
        raise RuntimeError(f"Failed to query config for {vault_contract_address}: {exc}")


# step:3 file: confirm_forfeitable_reward_structure_for_my_current_vault
def parse_reward_policy(config: dict) -> dict:
    """Extracts `forfeitable_rewards` and `early_exit_penalty` from the config dict."""
    forfeitable = bool(config.get("forfeitable_rewards", False))
    penalty_schedule = config.get("early_exit_penalty")  # May be `None` or a struct

    return {
        "forfeitable_rewards": forfeitable,
        "early_exit_penalty": penalty_schedule
    }


# step:2 file: stake_1000_ntrn_for_48_months_to_maximize_my_boost
import asyncio
from cosmpy.aio.client import LedgerClient
from cosmpy.aio.grpc import NetworkConfig

async def validate_asset_balance(address: str, required_untrn: int = 1_000_000_000, buffer_untrn: int = 50_000) -> bool:
    """Raise an exception if `address` has an insufficient balance of uNTRN."""
    try:
        network = NetworkConfig.fetch_network_config('neutron-1')
        async with LedgerClient(network) as client:
            balance_resp = await client.bank.balance(address, 'untrn')
            balance = int(balance_resp.amount)
            needed = required_untrn + buffer_untrn
            if balance < needed:
                raise ValueError(
                    f'Insufficient balance: {balance} uNTRN available, {needed} uNTRN required.'
                )
            return True
    except Exception as e:
        # Re-raise so the caller can handle/log
        raise e

# Quick CLI helper (optional)
if __name__ == '__main__':
    import sys, json
    ok = asyncio.run(validate_asset_balance(sys.argv[1]))
    print(json.dumps({'ok': ok}))


# step:3 file: stake_1000_ntrn_for_48_months_to_maximize_my_boost
import os

def get_ntrn_boost_lock_contract() -> str:
    """Return the NTRN Boost/Lock contract address, validating its format."""
    contract_addr = os.getenv('NTRN_LOCK_CONTRACT', 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    if not contract_addr or not contract_addr.startswith('neutron1'):
        raise ValueError('Invalid or missing NTRN Lock contract address.')
    return contract_addr


# step:3 file: deposit_3_ebtc_into_the_maxbtc_ebtc_supervault
import asyncio
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.contract import SmartContract

RPC_ENDPOINT = "https://rpc-kralum.neutron-1.neutron.org:443"
CHAIN_ID = "neutron-1"
REGISTRY_CONTRACT = "neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # TODO: replace with actual address

async def query_supervault_details(denom: str = "eBTC") -> dict:
    """Return a dict with the Supervault address for the given denom if single-sided deposits are allowed."""

    cfg = NetworkConfig(
        chain_id=CHAIN_ID,
        url=RPC_ENDPOINT,
        fee_minimum_gas_price=0.05,
        fee_denom="untrn",
    )

    client = LedgerClient(cfg)
    registry = SmartContract(REGISTRY_CONTRACT, client)

    try:
        vault_info = registry.query({"vault_for": {"denom": denom}})
    except Exception as exc:
        raise RuntimeError(f"Unable to query vault registry: {exc}")

    vault_addr = vault_info.get("vault_address")
    allows_single_sided = vault_info.get("allows_single_sided", False)

    if not vault_addr:
        raise RuntimeError(f"No Supervault found for denom {denom}.")

    if not allows_single_sided:
        raise RuntimeError(f"Vault {vault_addr} does not allow single-sided deposits with {denom}.")

    return {"vault_address": vault_addr}

# Example manual invocation
if __name__ == "__main__":
    print(asyncio.run(query_supervault_details()))


# step:2 file: provide_new_liquidity_to_the_wbtc_lbtc_supervault_with_1_wbtc_and_1_lbtc
import os
import json
from typing import Dict
from cosmpy.aerial.client import LedgerClient, NetworkConfig

RPC_ENDPOINT = os.getenv('NEUTRON_RPC', 'https://rpc-kralum.neutron.org:443')
CHAIN_ID = os.getenv('CHAIN_ID', 'neutron-1')
WBTC_CONTRACT = os.getenv('WBTC_CONTRACT')  # e.g. 'neutron1wbtc…'
LBTC_CONTRACT = os.getenv('LBTC_CONTRACT')  # e.g. 'neutron1lbtc…'

network_cfg = NetworkConfig(
    chain_id=CHAIN_ID,
    url=RPC_ENDPOINT,
    fee_minimum_gas_price='0.05',
    fee_denom='untrn',
)
client = LedgerClient(network_cfg)

def _cw20_balance(address: str, contract: str) -> int:
    """Internal helper – returns the CW20 balance for `address`."""
    query_msg = json.dumps({'balance': {'address': address}}).encode()
    try:
        result = client.query_contract_smart(contract, query_msg)
    except Exception as err:
        raise RuntimeError(f'Contract query failed: {err}') from err
    return int(json.loads(result)["balance"])

def validate_token_balances(address: str, min_wbtc: int = 1, min_lbtc: int = 1) -> Dict[str, int]:
    """Throws if the address does not meet the minimum WBTC/LBTC balance requirements."""
    if not WBTC_CONTRACT or not LBTC_CONTRACT:
        raise RuntimeError('Token contract addresses are not configured')

    wbtc_balance = _cw20_balance(address, WBTC_CONTRACT)
    lbtc_balance = _cw20_balance(address, LBTC_CONTRACT)

    if wbtc_balance < min_wbtc:
        raise ValueError(f'Insufficient WBTC: {wbtc_balance} < {min_wbtc}')
    if lbtc_balance < min_lbtc:
        raise ValueError(f'Insufficient LBTC: {lbtc_balance} < {min_lbtc}')

    return {'wbtc_balance': wbtc_balance, 'lbtc_balance': lbtc_balance}


# step:3 file: provide_new_liquidity_to_the_wbtc_lbtc_supervault_with_1_wbtc_and_1_lbtc
import os

def get_supervault_contract_address() -> str:
    """Returns the Supervault contract address from environment variables or other config."""
    addr = os.getenv('WBTC_LBTC_SUPERVAULT_ADDRESS')
    if not addr:
        raise RuntimeError('WBTC_LBTC_SUPERVAULT_ADDRESS environment variable not set')
    return addr


# step:4 file: provide_new_liquidity_to_the_wbtc_lbtc_supervault_with_1_wbtc_and_1_lbtc
import base64
import json
from typing import List, Dict


def construct_tx_supervault_deposit(sender: str,
                                     wbtc_amount: int,
                                     lbtc_amount: int,
                                     wbtc_contract: str,
                                     lbtc_contract: str,
                                     supervault_contract: str) -> List[Dict]:
    """Constructs the CW20 `send` messages (with hook) that perform the Supervault deposit."""
    hook_msg = {"deposit": {}}
    hook_b64 = base64.b64encode(json.dumps(hook_msg).encode()).decode()

    msgs: List[Dict] = []
    for contract, amount in ((wbtc_contract, wbtc_amount), (lbtc_contract, lbtc_amount)):
        msgs.append({
            'typeUrl': '/cosmwasm.wasm.v1.MsgExecuteContract',
            'value': {
                'sender': sender,
                'contract': contract,
                'msg': {
                    'send': {
                        'contract': supervault_contract,
                        'amount': str(amount),
                        'msg': hook_b64,
                    }
                },
                'funds': [],
            },
        })

    return msgs


# step:2 file: withdraw_10_%_of_the_user’s_shares_from_the_maxbtc_solvbtc_supervault
from typing import Any

# pip install cosmpy-aerial

def query_supervault_share_balance(
    address: str,
    contract_address: str,
    lcd_url: str = "https://rest-kralum.neutron-1.neutron.org"
) -> int:
    """Return the amount of Super-vault shares held by *address*."""
    from cosmpy.aerial.client import LedgerClient, NetworkConfig
    from cosmpy.aerial.exceptions import LedgerError

    cfg = NetworkConfig(
        chain_id="neutron-1",
        url=lcd_url,
        fee_minimum_gas_price=0,
        fee_denomination="untrn",
    )
    client = LedgerClient(cfg)

    query_msg: dict[str, Any] = {"share_balance": {"address": address}}
    try:
        response: dict[str, Any] = client.query_contract_smart(contract_address, query_msg)
        balance_str: str = response.get("balance", "0")
        return int(balance_str)
    except LedgerError as err:
        raise RuntimeError(f"WASM query failed: {err}") from err


# step:3 file: withdraw_10_%_of_the_user’s_shares_from_the_maxbtc_solvbtc_supervault
def calculate_shares_to_withdraw(total_shares: int, percentage: float = 0.10) -> int:
    """Calculate an integer amount of shares corresponding to *percentage* of the
    user’s current holdings. Fractional shares are truncated."""
    if total_shares <= 0:
        raise ValueError("total_shares must be positive.")

    shares_to_withdraw: int = int(total_shares * percentage)
    if shares_to_withdraw == 0:
        raise ValueError("Calculated withdrawal amount is zero; balance may be too small.")
    return shares_to_withdraw


# step:2 file: view_accrued_defi_yields_in_the_wbtc_usdc_supervault
import os
import logging
from typing import Dict
from cosmpy.aerial.client import LedgerClient, NetworkConfig

# Network / contract configuration (override with environment variables in production)
RPC_ENDPOINT = os.getenv('NEUTRON_RPC', 'https://rpc-kralum.neutron-1.neutron.org')
CHAIN_ID = os.getenv('NEUTRON_CHAIN_ID', 'neutron-1')
CONTRACT_ADDRESS_SUPERVAULT = os.getenv('CONTRACT_ADDRESS_SUPERVAULT')

if not CONTRACT_ADDRESS_SUPERVAULT:
    raise EnvironmentError('CONTRACT_ADDRESS_SUPERVAULT environment variable must be set.')

# Re-usable network client
_network_cfg = NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT)
_client = LedgerClient(_network_cfg)

def query_pending_yield(sender: str) -> Dict:
    '''Return the pending yield for `sender` from the Supervault contract.'''
    query_msg = {
        'pending_yield': {
            'address': sender,
        }
    }
    try:
        response = _client.query_contract(CONTRACT_ADDRESS_SUPERVAULT, query_msg)
        # Example response => {'pending_yield': [{'denom': 'uUSDC', 'amount': '12345678'}]}
        return response
    except Exception as exc:
        logging.exception('Failed to query pending yield: %s', exc)
        raise


# step:3 file: view_accrued_defi_yields_in_the_wbtc_usdc_supervault
from decimal import Decimal
from typing import Dict, List

# Mapping from on-chain denom to display symbol & decimals
DENOM_METADATA = {
    'uUSDC': {'symbol': 'USDC', 'decimals': 6},  # 1 USDC = 10^6 uUSDC
    'uWBTC': {'symbol': 'WBTC', 'decimals': 8},  # 1 WBTC = 10^8 uWBTC
    # Add additional denoms as necessary
}

def _format_coin(coin: Dict) -> Dict:
    denom = coin['denom']
    amount_micro = int(coin['amount'])
    meta = DENOM_METADATA.get(denom)

    # Fallback for unknown denominations
    if not meta:
        return {
            'symbol': denom,
            'amount': str(amount_micro),
        }

    human_amount = Decimal(amount_micro) / Decimal(10 ** meta['decimals'])
    return {
        'symbol': meta['symbol'],
        'amount': f'{human_amount.normalize():f}',
    }

def format_amounts(pending_yield_resp: Dict) -> Dict[str, str]:
    '''Aggregate and format pending yields by symbol.'''
    formatted: Dict[str, Decimal] = {}
    coins: List[Dict] = pending_yield_resp.get('pending_yield', [])

    for coin in coins:
        converted = _format_coin(coin)
        symbol = converted['symbol']
        amount = Decimal(converted['amount'])
        formatted[symbol] = formatted.get(symbol, Decimal(0)) + amount

    # Convert Decimal values back to strings for JSON serialization
    return {symbol: str(amount.normalize()) for symbol, amount in formatted.items()}


# step:2 file: provide_liquidity_to_the_maxbtc_unibtc_supervault_using_1_maxbtc_and_1_unibtc
import aiohttp
import asyncio
from typing import Dict

LCD_URL = "https://rest-kralum.neutron-1.neutron.org"

async def check_token_balance(address: str, denoms: list[str], lcd_url: str = LCD_URL) -> Dict[str, int]:
    """Return balances for the requested denoms and raise if any are missing."""
    url = f"{lcd_url}/cosmos/bank/v1beta1/balances/{address}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                raise ValueError(f"LCD returned {resp.status}: {await resp.text()}")
            data = await resp.json()

    balance_map = {coin["denom"]: int(coin["amount"]) for coin in data.get("balances", [])}

    # Require at least 1 full token (assume 1e8 micro-units for BTC-like denoms)
    missing = [d for d in denoms if balance_map.get(d, 0) < 100000000]
    if missing:
        raise ValueError(f"Insufficient balance for: {', '.join(missing)} (need ≥1 token each)")

    return {d: balance_map[d] for d in denoms}

# Example usage inside an async context:
# balances = await check_token_balance(address, ["amaxbtc", "aunibtc"])


# step:3 file: provide_liquidity_to_the_maxbtc_unibtc_supervault_using_1_maxbtc_and_1_unibtc
import json
import aiohttp
from typing import Dict

async def query_supervault_details(contract_address: str, lcd_url: str = LCD_URL) -> Dict:
    """Returns the Supervault configuration via a smart-query call."""
    query_msg = {"config": {}}
    # Encode the query_msg as URL-safe base64-ish per CosmWasm LCD spec
    encoded_msg = json.dumps(query_msg).encode("utf-8").hex()
    url = f"{lcd_url}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{encoded_msg}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                raise ValueError(f"Failed to query Supervault config: {resp.status}")
            data = await resp.json()

    # LCD wraps smart-query response under "data"
    return data.get("data", {})

# Example:
# config = await query_supervault_details(supervault_address)