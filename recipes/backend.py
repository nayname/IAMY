# Minimal Backend.py containing only the functions called in the intents implementation

import subprocess
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from decimal import Decimal, ROUND_UP, InvalidOperation
import base64
import re
import requests
import json
import os
import hashlib
import math
import asyncio
from dataclasses import dataclass, field
from urllib.parse import quote
from collections.abc import MutableMapping, MutableSequence

# External libraries imports for type hints and functionality
try:
    from bech32 import bech32_encode, convertbits, bech32_decode
except ImportError:
    pass
try:
    from ecdsa import SECP256k1, SigningKey, util as ecdsa_util
except ImportError:
    pass
try:
    from Crypto.Hash import RIPEMD160
except ImportError:
    pass
try:
    from bip_utils import Bip39MnemonicGenerator, Bip39WordsNum, Bip39SeedGenerator, Bip32Slip10Secp256k1
except ImportError:
    pass
try:
    from tomlkit import parse, table, array, dumps
except ImportError:
    pass
try:
    import httpx
except ImportError:
    pass
try:
    from cosmpy.protos.cosmos.auth.v1beta1.auth_pb2 import BaseAccount as BaseAccountProto
except ImportError:
    pass
try:
    from cosmpy.protos.cosmwasm.wasm.v1.tx_pb2 import MsgStoreCode as MsgStoreCodeProto
except ImportError:
    pass
try:
    from cosmpy.protos.cosmos.tx.v1beta1.tx_pb2 import TxBody as TxBodyProto, AuthInfo as AuthInfoProto, Tx as TxProto, Fee as FeeProto
except ImportError:
    pass
try:
    from cosmpy.protos.cosmos.base.v1beta1.coin_pb2 import Coin as CoinProto
except ImportError:
    pass
try:
    from cosmpy.protos.cosmos.crypto.secp256k1.keys_pb2 import PubKey as Secp256k1PubKeyProto
except ImportError:
    pass
try:
    from cosmpy.protos.cosmos.tx.signing.v1beta1.signing_pb2 import SignMode as SignModeProto, SignDoc as SignDocProto, SignerInfo as SignerInfoProto
except ImportError:
    pass
try:
    from cosmpy.aerial.wallet import LocalWallet
    from cosmpy.aerial.client import LedgerClient, NetworkConfig
    from cosmpy.crypto.keypairs import PrivateKey
except ImportError:
    pass
try:
    from google.protobuf.any_pb2 import Any as AnyProto
except ImportError:
    pass
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib # type: ignore
try:
    from cosmospy_protobuf.cosmos.bank.v1beta1.tx_pb2 import MsgSend as MsgSendProto
    from cosmospy_protobuf.cosmos.base.v1beta1.coin_pb2 import Coin as CoinMsgProto
    from cosmospy_protobuf.cosmos.tx.v1beta1.tx_pb2 import TxBody as TxBodyMsgProto, AuthInfo as AuthInfoMsgProto, Tx as TxMsgProto, Fee as FeeMsgProto
    from cosmospy_protobuf.cosmos.auth.v1beta1.auth_pb2 import BaseAccount as BaseAccountMsgProto
    from cosmospy_protobuf.cosmos.tx.v1beta1.tx_pb2 import Tx as TxMsgProto
    from cosmospy_protobuf.cosmos.tx.signing.v1beta1.signing_pb2 import SignerInfo as SignerInfoMsgProto, SignMode as SignModeMsgProto
    from cosmospy_protobuf.cosmos.crypto.secp256k1.keys_pb2 import PubKey as Secp256k1PubKeyMsgProto
    from cosmospy_protobuf.cosmos.tx.v1beta1.tx_pb2 import SignDoc as SignDocMsgProto
    from cosmospy_protobuf.cosmwasm.wasm.v1.tx_pb2 import MsgExecuteContract as MsgExecuteContractProto
    from cosmospy_protobuf.cosmos.base.v1beta1.coin_pb2 import Coin as CosmosCoinProto
    from cosmos_sdk_proto.cosmos.tx.v1beta1.tx_pb2 import TxRaw as TxRawProto
except ImportError:
    pass
try:
    # Requires the 'cryptography' package
    from cryptography.fernet import Fernet
except ImportError:
    pass

# --- Constants ---
LCD_BASE_URL = 'https://lcd-archive.junonetwork.io'
JUNO_HRP = "juno"

# --- Data Classes ---

@dataclass
class NodeConfig:
    lcd_url: str = LCD_BASE_URL
    rpc_url: Optional[str] = None
    chain_id: Optional[str] = None
    keyring_backend: Optional[str] = None

@dataclass
class SmartQueryParameters:
    contract_address: str
    query: Dict[str, Any]
    node: NodeConfig

@dataclass
class WasmChecksum:
    digest: bytes
    hex: str

@dataclass
class ChainAccountInfo:
    chain_id: str
    account_number: int
    sequence: int

@dataclass
class PaginationState:
    page_limit: int = 100
    start_after: Optional[str] = None
    token_ids: List[str] = field(default_factory=list)

# --- Exceptions ---

class LcdRequestError(Exception):
    pass
class ContractAddressNotFoundError(Exception):
    pass
class ContractNotFoundOnChainError(Exception):
    pass
class ProjectRootError(Exception):
    pass
class CargoReadError(Exception):
    pass
class RustOptimizerError(Exception):
    pass
class WasmArtifactError(Exception):
    pass
class AmountParseError(ValueError):
    pass
class InsufficientBalanceError(ValueError):
    pass
class TxConstructionError(RuntimeError):
    pass
class TxSimulationError(RuntimeError):
    pass
class TxBroadcastError(RuntimeError):
    pass
class FundsVerificationError(RuntimeError):
    pass
class ContractAddressValidationError(ValueError):
    pass
class QueryBuildError(ValueError):
    pass
class QueryEncodingError(ValueError):
    pass
class SmartQueryHttpError(RuntimeError):
    pass
class SmartQueryParseError(ValueError):
    pass
class ContractResponseDecodeError(ValueError):
    pass
class WasmArtifactError(Exception):
    pass
class ChainQueryError(Exception):
    pass
class SimulationError(Exception):
    pass
class SigningError(Exception):
    pass
class BroadcastError(Exception):
    pass
class CodeIdNotFoundError(Exception):
    pass
class CodeHashMismatchError(Exception):
    pass
class WorkspaceNotFoundError(Exception):
    pass
class WorkspaceMemberError(Exception):
    pass
class OptimizerError(Exception):
    pass
class ArtifactVerificationError(Exception):
    pass
class WasmArtifactError(Exception):
    pass
class JunodTxTestError(Exception):
    pass
class LCDQueryError(Exception):
    pass
class InvalidJunoAddressError(ValueError):
    pass
class NftNotFoundError(Exception):
    pass
class ValidationError(Exception):
    pass


# --- Utility Helpers (Internal to the Minimal Backend) ---

def _load_toml(path: Path) -> dict:
    '''Load a TOML file using tomllib or fallback.'''
    if not path.is_file():
        raise FileNotFoundError(f'TOML file not found: {path}')

    data = path.read_bytes()
    if tomllib is None:
        import toml # type: ignore
        return toml.loads(data.decode('utf-8'))
    return tomllib.loads(data.decode('utf-8'))

def _resolve_workspace_root(workspace_root: Optional[str] = None) -> Path:
    '''Resolve the workspace root directory, using detect_workspace_root if needed.'''
    if workspace_root:
        root = Path(workspace_root).expanduser().resolve()
    else:
        info = detect_workspace_root()
        root = Path(info['workspace_root']).resolve()

    if not root.is_dir():
        raise FileNotFoundError(f'Workspace root directory does not exist: {root}')
    return root

def _is_valid_wasm(path: Path) -> bool:
    '''Basic WASM validation check.'''
    if not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False
    try:
        with path.open('rb') as f:
            magic = f.read(4)
        return magic == b'\0asm'
    except Exception:
        return False

def _parse_gas_price_entry(entry: str):
    """Splits an entry like "0.0025ujuno" into (Decimal('0.0025'), 'ujuno')."""
    entry = entry.strip()
    if not entry:
        raise ValueError('Empty gas price entry.')
    i = 0
    while i < len(entry) and (entry[i].isdigit() or entry[i] == '.'):
        i += 1
    if i == 0:
        raise ValueError(f'Gas price entry has no numeric prefix: {entry}')
    amount_str = entry[:i]
    denom_str = entry[i:]
    return Decimal(amount_str), denom_str

def _get_fernet() -> Fernet:
    key = os.environ.get("WALLET_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("WALLET_ENCRYPTION_KEY environment variable must be set to a Fernet key.")
    return Fernet(key.encode("utf-8"))

def _extract_network_field(data: Any) -> str:
    '''Internal helper to robustly extract the network or chain-id field from junod status JSON.'''
    if not isinstance(data, dict):
        return ''
    candidate_paths = [('node_info', 'network'), ('NodeInfo', 'network'), ('result', 'node_info', 'network'), ('Result', 'node_info', 'network')]
    for path in candidate_paths:
        cursor: Any = data
        for key in path:
            if isinstance(cursor, dict) and key in cursor:
                cursor = cursor[key]
            else:
                cursor = None
                break
        if isinstance(cursor, str) and cursor:
            return cursor
    return ''

def _contract_smart_query(address: str, query_msg: Dict[str, Any]) -> Dict[str, Any]:
    query_bytes = json.dumps(query_msg, separators=(",", ":")).encode("utf-8")
    query_b64 = base64.b64encode(query_bytes).decode("utf-8")
    url = f"{LCD_BASE_URL}/cosmwasm/wasm/v1/contract/{address}/smart/{query_b64}"

    with requests.Client(timeout=10.0) as client:
        resp = client.get(url)

    if resp.status_code != 200:
        raise RuntimeError(f"Smart query failed with status {resp.status_code}: {resp.text}")

    body = resp.json()
    data_b64 = body.get("data")
    if not data_b64:
        raise RuntimeError(f"Malformed smart query response: {body}")

    try:
        decoded = base64.b64decode(data_b64)
        return json.loads(decoded.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to decode smart query response: {exc}") from exc

def get_current_count(contract_address: str) -> int:
    result = _contract_smart_query(contract_address, {"get_count": {}})
    if "count" not in result:
        raise RuntimeError(f"'count' field missing in contract response: {result}")
    return int(result["count"])

def decimal_to_str(value: Decimal) -> str:
    s = format(value, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

# --- Imported Functions ---

# [cite_start]Compile a CosmWasm contract for ARM64 using the rust-optimizer-arm64 Docker image [cite: 1]
def verify_docker_installed() -> Dict[str, str]:
    try:
        completed = subprocess.run(['docker', 'version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Docker is not installed or not available in PATH.') from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('Docker is installed but the daemon may not be running or is misconfigured.') from exc
    return {'status': 'ok', 'message': 'Docker is installed and responding.', 'stdout': completed.stdout}
def verify_contract_builds_locally(project_root: str) -> Dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    cargo_toml = root / 'Cargo.toml'
    if not cargo_toml.is_file():
        raise FileNotFoundError(f'No Cargo.toml found in {root}; ensure project_root is the contract root directory.')
    try:
        completed = subprocess.run(['cargo', 'check'], cwd=str(root), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Cargo is not installed or not available in PATH.') from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('cargo check failed. Fix the compilation errors before running the optimizer.') from exc
    return {'status': 'ok', 'message': 'cargo check completed successfully.', 'stdout': completed.stdout}
def run_rust_optimizer_arm64(project_root: str, tag: str = 'latest') -> Dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    if not (root / 'Cargo.toml').is_file():
        raise FileNotFoundError(f'No Cargo.toml found in {root}; ensure project_root is the contract root directory.')
    cache_volume = f'{root.name}_cache'
    image = f'cosmwasm/rust-optimizer-arm64:{tag}'
    cmd = ['docker', 'run', '--rm', '-v', f'{root}:/code', '--mount', f'type=volume,source={cache_volume},target=/code/target', image]
    try:
        completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Docker is not installed or not available in PATH. Install Docker and try again.') from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('rust-optimizer Docker run failed.') from exc
    return {'status': 'ok', 'message': f'Optimization completed using image {image}.', 'stdout': completed.stdout}
def collect_optimized_wasm_artifacts(project_root: str) -> List[str]:
    root = Path(project_root).expanduser().resolve()
    artifacts_dir = root / 'artifacts'
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f'Artifacts directory not found at {artifacts_dir}. Have you run the rust-optimizer?')
    wasm_files = sorted(str(p) for p in artifacts_dir.glob('*.wasm'))
    if not wasm_files:
        raise RuntimeError(f'No .wasm files found in {artifacts_dir}. Ensure the optimizer completed successfully.')
    return wasm_files

# [cite_start]Compile the clock_example CosmWasm contract and upload clock_example.wasm to the Juno testnet (chain-id uni-6). [cite: 1, 2, 3]
def compile_clock_example(contract_root: str = '.') -> str:
    root_path = Path(contract_root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f'Contract root directory not found: {root_path}')
    artifacts_dir = root_path / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    cmd = ['docker', 'run', '--rm', '-v', f'{root_path}:/code', '--mount', 'type=volume,source=clock_example_cache,target=/code/target', 'cosmwasm/rust-optimizer:0.12.11']
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Docker is not installed or not found in PATH.') from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'cosmwasm/rust-optimizer failed: {exc.stderr}') from exc
    wasm_path = artifacts_dir / 'clock_example.wasm'
    if not wasm_path.exists():
        raise FileNotFoundError(f'Expected artifact not found at {wasm_path}')
    return str(wasm_path)
def load_wasm_artifact_bytes(wasm_path: str) -> bytes:
    path = Path(wasm_path)
    if not path.is_file():
        raise FileNotFoundError(f'WASM artifact not found at {path}')
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f'WASM artifact at {path} is empty.')
    with path.open('rb') as f:
        data = f.read()
    if not data:
        raise ValueError('Failed to read WASM artifact or file is empty.')
    return data
def fetch_minimum_gas_price(base_url: str = LCD_BASE_URL) -> Tuple[Decimal, str]:
    url = f'{base_url}/cosmos/base/node/v1beta1/config'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to fetch node config from {url}: {exc}') from exc
    data = resp.json()
    mgp = data.get('minimum_gas_price')
    if not mgp:
        raise ValueError('minimum_gas_price not present in node config response.')
    match = re.match(r'^([0-9.]+)([a-zA-Z/]+)$', mgp)
    if not match:
        raise ValueError(f'Unexpected minimum_gas_price format: {mgp}')
    amount_str, denom = match.groups()
    try:
        amount = Decimal(amount_str)
    except Exception as exc:
        raise ValueError(f'Invalid gas price amount {amount_str!r}: {exc}') from exc
    return amount, denom
def fetch_sender_account_state(address: str, base_url: str = LCD_BASE_URL, fee_denom: str = 'ujunox', required_balance: Optional[Decimal] = None) -> Dict[str, Any]:
    account_url = f'{base_url}/cosmos/auth/v1beta1/accounts/{address}'
    balances_url = f'{base_url}/cosmos/bank/v1beta1/spendable_balances/{address}'
    try:
        acc_resp = requests.get(account_url, timeout=10)
        acc_resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to fetch account data: {exc}') from exc
    acc_data = acc_resp.json()
    account_any = acc_data.get('account')
    if not account_any or 'value' not in account_any:
        raise ValueError('Account response missing account.value field.')
    try:
        raw = base64.b64decode(account_any['value'])
        base_account = BaseAccountProto.FromString(raw)
    except Exception as exc:
        raise RuntimeError(f'Failed to decode BaseAccount protobuf: {exc}') from exc
    account_number = int(base_account.account_number)
    sequence = int(base_account.sequence)
    try:
        bal_resp = requests.get(balances_url, timeout=10)
        bal_resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to fetch spendable balances: {exc}') from exc
    bal_data = bal_resp.json()
    balances = bal_data.get('balances', [])
    spendable = Decimal(0)
    for coin in balances:
        if coin.get('denom') == fee_denom:
            spendable = Decimal(coin.get('amount', '0'))
            break
    if required_balance is not None and spendable < required_balance:
        raise ValueError(f'Insufficient balance: have {spendable} {fee_denom}, required at least {required_balance}.')
    return {'account_number': account_number, 'sequence': sequence, 'balances': balances, 'fee_denom': fee_denom, 'spendable_for_fee': str(spendable)}
def construct_msg_store_code_tx_for_simulation(sender: str, wasm_bytes: bytes, gas_price_amount: Decimal, gas_price_denom: str, initial_gas_limit: int = 2_000_000, chain_id: str = 'uni-6') -> Dict[str, Any]:
    msg = MsgStoreCodeProto(sender=sender, wasm_byte_code=wasm_bytes)
    msg_any = AnyProto()
    msg_any.Pack(msg)
    tx_body = TxBodyProto(messages=[msg_any], memo='', timeout_height=0)
    fee_amount = int((gas_price_amount * Decimal(initial_gas_limit)).to_integral_value(rounding=ROUND_UP))
    fee = FeeProto(amount=[CoinProto(denom=gas_price_denom, amount=str(fee_amount))], gas_limit=initial_gas_limit)
    auth_info = AuthInfoProto(signer_infos=[], fee=fee)
    tx = TxProto(body=tx_body, auth_info=auth_info, signatures=[])
    tx_bytes = tx.SerializeToString()
    return {'tx_bytes': tx_bytes, 'tx_body': tx_body, 'auth_info': auth_info, 'initial_gas_limit': initial_gas_limit, 'fee_amount': fee_amount, 'gas_price_amount': str(gas_price_amount), 'gas_price_denom': gas_price_denom, 'chain_id': chain_id}
def simulate_store_code_tx(tx_bytes: bytes, base_url: str = LCD_BASE_URL) -> Dict[str, Any]:
    url = f'{base_url}/cosmos/tx/v1beta1/simulate'
    payload = {'tx_bytes': base64.b64encode(tx_bytes).decode('ascii')}
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Simulation request failed: {exc}') from exc
    data = resp.json()
    gas_info = data.get('gas_info') or {}
    gas_used_str = gas_info.get('gas_used')
    if gas_used_str is None:
        raise ValueError(f'Simulation response missing gas_info.gas_used: {data}')
    try:
        gas_used = int(gas_used_str)
    except ValueError as exc:
        raise ValueError(f'Invalid gas_used value {gas_used_str!r}: {exc}') from exc
    safety_factor = 1.3
    adjusted_gas_limit = int(gas_used * safety_factor)
    return {'raw_response': data, 'gas_used': gas_used, 'adjusted_gas_limit': adjusted_gas_limit}
def sign_store_code_tx(sender: str, wasm_bytes: bytes, gas_price_amount: Decimal, gas_price_denom: str, gas_limit: int, account_number: int, sequence: int, chain_id: str, privkey_hex: str) -> Dict[str, Any]:
    msg = MsgStoreCodeProto(sender=sender, wasm_byte_code=wasm_bytes)
    msg_any = AnyProto()
    msg_any.Pack(msg)
    tx_body = TxBodyProto(messages=[msg_any], memo='', timeout_height=0)
    fee_amount = int((gas_price_amount * Decimal(gas_limit)).to_integral_value(rounding=ROUND_UP))
    fee = FeeProto(amount=[CoinProto(denom=gas_price_denom, amount=str(fee_amount))], gas_limit=gas_limit)
    try:
        privkey_bytes = bytes.fromhex(privkey_hex)
    except ValueError as exc:
        raise ValueError('Private key must be a 64-character hex string.') from exc
    if len(privkey_bytes) != 32:
        raise ValueError('Private key must be 32 bytes (64 hex characters).')
    sk = SigningKey.from_string(privkey_bytes, curve=SECP256k1)
    vk = sk.get_verifying_key()
    try:
        pubkey_bytes = vk.to_string('compressed')
    except TypeError as exc:
        raise RuntimeError('ecdsa library is too old; upgrade to a version that supports compressed keys.') from exc
    pk_msg = Secp256k1PubKeyProto(key=pubkey_bytes)
    pk_any = AnyProto()
    pk_any.Pack(pk_msg)
    pk_any.type_url = '/cosmos.crypto.secp256k1.PubKey'
    mode_info = SignerInfoProto(mode_info=SignerInfoProto.ModeInfo(single=SignerInfoProto.ModeInfo.Single(mode=SignModeProto.SIGN_MODE_DIRECT)))
    signer_info = SignerInfoProto(public_key=pk_any, mode_info=mode_info.mode_info, sequence=sequence)
    auth_info = AuthInfoProto(signer_infos=[signer_info], fee=fee)
    sign_doc = SignDocProto(body_bytes=tx_body.SerializeToString(), auth_info_bytes=auth_info.SerializeToString(), chain_id=chain_id, account_number=account_number)
    sign_bytes = sign_doc.SerializeToString()
    signature = sk.sign_deterministic(sign_bytes, hashfunc=hashlib.sha256)
    tx = TxProto(body=tx_body, auth_info=auth_info, signatures=[signature])
    tx_bytes = tx.SerializeToString()
    return {'tx': tx, 'tx_bytes': tx_bytes, 'fee_amount': fee_amount}
def broadcast_store_code_tx(tx_bytes: bytes, base_url: str = LCD_BASE_URL) -> Dict[str, Any]:
    url = f'{base_url}/cosmos/tx/v1beta1/txs'
    payload = {'tx_bytes': base64.b64encode(tx_bytes).decode('ascii'), 'mode': 'BROADCAST_MODE_BLOCK'}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Broadcast failed: {exc}') from exc
    data = resp.json()
    tx_response = data.get('tx_response') or {}
    if not tx_response:
        raise ValueError(f'Broadcast response missing tx_response field: {data}')
    code = tx_response.get('code', 0)
    if code != 0:
        raw_log = tx_response.get('raw_log')
        raise RuntimeError(f'Transaction failed with code {code}: {raw_log}')
    return tx_response
def extract_code_id_from_logs(tx_response: Dict[str, Any]) -> str:
    logs = tx_response.get('logs', [])
    for log in logs:
        events = log.get('events', [])
        for event in events:
            attributes = event.get('attributes', [])
            for attr in attributes:
                key = attr.get('key')
                value = attr.get('value')
                if key in ('code_id', 'codeID'):
                    return str(value)
    raise ValueError(f'code_id not found in transaction logs: {logs}')
def verify_code_uploaded_on_chain(code_id: str, expected_creator: str, base_url: str = LCD_BASE_URL) -> Dict[str, Any]:
    url = f'{base_url}/cosmwasm/wasm/v1/code/{code_id}'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to query code_id {code_id}: {exc}') from exc
    data = resp.json()
    code_info = data.get('code_info')
    if not code_info:
        raise ValueError(f'No code_info returned for code_id {code_id}: {data}')
    creator = code_info.get('creator')
    if creator != expected_creator:
        raise ValueError(f'Code creator mismatch for code_id {code_id}: expected {expected_creator}, got {creator}')
    code_id_in_payload = code_info.get('code_id')
    if str(code_id_in_payload) != str(code_id):
        raise ValueError(f'code_id mismatch in response: path={code_id}, payload={code_id_in_payload}')
    return {'code_info': code_info, 'verified': True}

# [cite_start]Send 1,000,000 ujuno to a specified contract address on Juno by constructing, simulating, signing, and broadcasting a bank send transaction via the LCD broadcast endpoint. [cite: 4, 5, 6]
async def get_sender_wallet(env_var: str = 'JUNO_SENDER_PRIVATE_KEY') -> dict:
    DENOM = 'ujuno'
    CHAIN_ID = 'juno-1'
    try:
        hex_key = os.environ.get(env_var)
        if not hex_key:
            raise RuntimeError(f'Private key environment variable {env_var} is not set.')
        hex_key = hex_key.strip().lower().replace('0x', '')
        if len(hex_key) != 64:
            raise ValueError('Private key must be 32 bytes (64 hex characters).')
        priv_bytes = bytes.fromhex(hex_key)
        sk = SigningKey.from_string(priv_bytes, curve=SECP256k1)
        vk = sk.get_verifying_key()
        compressed_pub = vk.to_string('compressed')
        sha = hashlib.sha256(compressed_pub).digest()
        ripemd = RIPEMD160.new()
        ripemd.update(sha)
        ripemd_digest = ripemd.digest()
        five_bit_r = convertbits(list(ripemd_digest), 8, 5, True)
        bech32_addr = bech32_encode('juno', five_bit_r)
        return {'address': bech32_addr, 'private_key_hex': priv_bytes.hex(), 'public_key_bytes': compressed_pub}
    except Exception as e:
        raise RuntimeError(f'Failed to load or derive sender wallet: {e}') from e
async def validate_contract_address(contract_address: str, check_on_chain: bool = True) -> bool:
    try:
        hrp, data = bech32_decode(contract_address)
        if data is None or hrp != 'juno':
            raise ValueError('Invalid bech32 address or wrong prefix (expected "juno").')
    except Exception as e:
        raise ValueError(f'Invalid contract address format: {e}') from e
    if not check_on_chain:
        return True
    url = f'{LCD_BASE_URL}/cosmos/auth/v1beta1/accounts/{contract_address}'
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                raise ValueError('Address not found on-chain (404 from auth module).')
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f'HTTP error while validating contract address: {e}') from e
    if 'account' not in data:
        raise RuntimeError('LCD response does not contain an "account" field; cannot confirm account.')
    return True
async def check_sender_balance(sender_address: str, required_amount: int = 1_000_000, fee_buffer: int = 200_000) -> dict:
    DENOM = 'ujuno'
    url = f'{LCD_BASE_URL}/cosmos/bank/v1beta1/balances/{sender_address}/by_denom'
    params = {'denom': DENOM}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f'HTTP error while querying sender balance: {e}') from e
    balance_str = data.get('balance', {}).get('amount')
    if balance_str is None:
        raise RuntimeError('LCD response missing balance.amount for ujuno.')
    try:
        balance = int(balance_str)
    except ValueError:
        raise RuntimeError(f'Invalid balance amount returned: {balance_str}')
    needed = required_amount + fee_buffer
    if balance < needed:
        raise ValueError(f'Insufficient ujuno balance. Have {balance}, but need at least {needed} (1,000,000 for send + buffer).')
    return {'balance': balance, 'required_total': needed}
def construct_msg_send(from_address: str, to_address: str, amount: int = 1_000_000) -> AnyProto:
    DENOM = 'ujuno'
    if amount <= 0:
        raise ValueError('Amount must be a positive integer of ujuno.')
    msg = MsgSendProto(from_address=from_address, to_address=to_address, amount=[CoinMsgProto(denom=DENOM, amount=str(amount))])
    any_msg = AnyProto()
    any_msg.Pack(msg)
    any_msg.type_url = '/cosmos.bank.v1beta1.MsgSend'
    return any_msg
def build_unsigned_tx(msg_any, memo: str = '', initial_gas_limit: int = 200_000) -> TxMsgProto:
    if initial_gas_limit <= 0:
        raise ValueError('initial_gas_limit must be positive.')
    tx_body = TxBodyMsgProto(messages=[msg_any], memo=memo)
    fee = FeeMsgProto(amount=[], gas_limit=initial_gas_limit, payer='', granter='')
    auth_info = AuthInfoMsgProto(signer_infos=[], fee=fee)
    tx = TxMsgProto(body=tx_body, auth_info=auth_info, signatures=[])
    return tx
async def simulate_tx_for_gas(tx: TxMsgProto, gas_adjustment: float = 1.2) -> int:
    if gas_adjustment <= 0:
        raise ValueError('gas_adjustment must be positive.')
    tx_bytes = tx.SerializeToString()
    payload = {'tx_bytes': base64.b64encode(tx_bytes).decode('utf-8')}
    url = f'{LCD_BASE_URL}/cosmos/tx/v1beta1/simulate'
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f'HTTP error during gas simulation: {e}') from e
    gas_used_str = data.get('gas_info', {}).get('gas_used')
    if gas_used_str is None:
        raise RuntimeError('Simulation response missing gas_info.gas_used.')
    try:
        gas_used = int(gas_used_str)
    except ValueError:
        raise RuntimeError(f'Invalid gas_used value returned from simulate: {gas_used_str}')
    gas_limit = math.ceil(gas_used * gas_adjustment)
    tx.auth_info.fee.gas_limit = gas_limit
    return gas_limit
async def apply_min_gas_price_fee(tx: TxMsgProto, denom: str = 'ujuno') -> str:
    url = f'{LCD_BASE_URL}/cosmos/base/node/v1beta1/config'
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f'HTTP error while fetching minimum_gas_price: {e}') from e
    min_gas_price_str = data.get('minimum_gas_price')
    if not min_gas_price_str:
        raise RuntimeError('minimum_gas_price is empty or missing in node config.')
    selected_price = None
    for part in min_gas_price_str.split(','):
        amount_dec, denom_str = _parse_gas_price_entry(part)
        if denom_str == denom:
            selected_price = amount_dec
            break
    if selected_price is None:
        raise RuntimeError(f'No gas price entry for denom {denom} found in "{min_gas_price_str}".')
    gas_limit = tx.auth_info.fee.gas_limit
    if gas_limit <= 0:
        raise RuntimeError('tx.auth_info.fee.gas_limit must be set before computing fees.')
    fee_amount = (selected_price * Decimal(gas_limit)).quantize(Decimal('1'), rounding=ROUND_UP)
    tx.auth_info.fee.amount[:] = [CoinMsgProto(denom=denom, amount=str(fee_amount))]
    return str(fee_amount)
async def fetch_account_number_and_sequence(address: str) -> dict:
    url = f'{LCD_BASE_URL}/cosmos/auth/v1beta1/accounts/{address}'
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f'HTTP error while fetching account data: {e}') from e
    account_any = data.get('account')
    if not account_any or 'value' not in account_any:
        raise RuntimeError('LCD response missing account.value; cannot decode account.')
    raw = base64.b64decode(account_any['value'])
    any_msg = AnyProto()
    any_msg.ParseFromString(raw)
    base_account = None
    if any_msg.type_url.endswith('BaseAccount'):
        base_account = BaseAccountMsgProto()
        any_msg.Unpack(base_account)
    elif any_msg.type_url.endswith('ContinuousVestingAccount'):
        from cosmospy_protobuf.cosmos.vesting.v1beta1.vesting_pb2 import ContinuousVestingAccount as ContinuousVestingAccountMsgProto
        vesting = ContinuousVestingAccountMsgProto()
        any_msg.Unpack(vesting)
        base_account = vesting.base_account
    elif any_msg.type_url.endswith('DelayedVestingAccount'):
        from cosmospy_protobuf.cosmos.vesting.v1beta1.vesting_pb2 import DelayedVestingAccount as DelayedVestingAccountMsgProto
        vesting = DelayedVestingAccountMsgProto()
        any_msg.Unpack(vesting)
        base_account = vesting.base_account
    else:
        raise RuntimeError(f'Unsupported account type_url: {any_msg.type_url}')
    return {'account_number': int(base_account.account_number), 'sequence': int(base_account.sequence)}
def sign_tx_with_sender_key(tx: TxMsgProto, private_key_hex: str, account_number: int, sequence: int, chain_id: str = 'juno-1') -> TxMsgProto:
    if not private_key_hex:
        raise ValueError('private_key_hex must be provided.')
    priv_hex = private_key_hex.lower().replace('0x', '').strip()
    if len(priv_hex) != 64:
        raise ValueError('private_key_hex must be 32 bytes (64 hex characters).')
    priv_bytes = bytes.fromhex(priv_hex)
    sk = SigningKey.from_string(priv_bytes, curve=SECP256k1)
    vk = sk.get_verifying_key()
    compressed_pub = vk.to_string('compressed')
    pk = Secp256k1PubKeyMsgProto(key=compressed_pub)
    pk_any = AnyProto()
    pk_any.Pack(pk)
    pk_any.type_url = '/cosmos.crypto.secp256k1.PubKey'
    mode_info = TxMsgProto.ModeInfo(single=TxMsgProto.ModeInfo.Single(mode=SignModeMsgProto.SIGN_MODE_DIRECT))
    signer_info = SignerInfoMsgProto(public_key=pk_any, mode_info=mode_info, sequence=sequence)
    tx.auth_info.signer_infos[:] = [signer_info]
    sign_doc = SignDocMsgProto(body_bytes=tx.body.SerializeToString(), auth_info_bytes=tx.auth_info.SerializeToString(), chain_id=chain_id, account_number=account_number)
    sign_doc_bytes = sign_doc.SerializeToString()
    sign_bytes = hashlib.sha256(sign_doc_bytes).digest()
    signature = sk.sign_deterministic(sign_bytes, hashfunc=hashlib.sha256, sigencode=ecdsa_util.sigencode_string)
    tx.signatures[:] = [signature]
    return tx
def encode_tx_to_bytes(tx: TxMsgProto) -> str:
    if not tx.signatures:
        raise RuntimeError('Tx has no signatures; call sign_tx_with_sender_key() first.')
    tx_bytes = tx.SerializeToString()
    return base64.b64encode(tx_bytes).decode('utf-8')
async def broadcast_tx_via_lcd(tx_bytes_b64: str, mode: str = 'BROADCAST_MODE_SYNC') -> dict:
    valid_modes = {'BROADCAST_MODE_SYNC', 'BROADCAST_MODE_BLOCK', 'BROADCAST_MODE_ASYNC', 'BROADCAST_MODE_UNSPECIFIED'}
    if mode not in valid_modes:
        raise ValueError(f'Invalid broadcast mode: {mode}')
    url = f'{LCD_BASE_URL}/cosmos/tx/v1beta1/txs'
    payload = {'tx_bytes': tx_bytes_b64, 'mode': mode}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f'HTTP error while broadcasting transaction: {e}') from e
    tx_response = data.get('tx_response')
    if tx_response is None:
        raise RuntimeError('LCD broadcast response missing "tx_response" field.')
    return tx_response
def verify_broadcast_result(tx_response: dict) -> dict:
    if tx_response is None:
        raise ValueError('tx_response is None; cannot verify broadcast result.')
    if 'code' not in tx_response:
        raise RuntimeError('tx_response missing "code" field.')
    code = tx_response['code']
    raw_log = tx_response.get('raw_log', '')
    if code != 0:
        raise RuntimeError(f'Transaction failed with code {code}: {raw_log}')
    return {'txhash': tx_response.get('txhash'), 'height': tx_response.get('height'), 'raw_log': raw_log}

# [cite_start]Connect a SigningCosmWasmClient to RPC https://rpc.juno.strange.love and execute increment on the contract [cite: 7]
def prepare_backend_signer() -> LocalWallet:
    mnemonic = os.getenv("JUNO_MNEMONIC")
    if not mnemonic:
        raise RuntimeError("JUNO_MNEMONIC environment variable is not set.")
    try:
        wallet = LocalWallet.from_mnemonic(mnemonic, prefix="juno")
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize LocalWallet: {exc}") from exc
    return wallet
def init_signing_cosmwasm_client(wallet: LocalWallet) -> LedgerClient:
    RPC_ENDPOINT = "https://rpc.juno.strange.love"
    CHAIN_ID = "juno-1"
    FEE_DENOM = "ujuno"
    try:
        network_cfg = NetworkConfig(chain_id=CHAIN_ID, url=RPC_ENDPOINT, fee_minimum_gas_price=0.025, fee_denomination=FEE_DENOM, staking_denomination=FEE_DENOM)
        client = LedgerClient(network_cfg)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Juno LedgerClient: {exc}") from exc
    return client
def resolve_contract_address() -> str:
    addr = os.getenv("CONTRACT_ADDRESS")
    if not addr:
        raise RuntimeError("CONTRACT_ADDRESS environment variable is not set. Set it to the deployed counter contract address.")
    if not addr.startswith("juno1"):
        raise ValueError(f"CONTRACT_ADDRESS does not look like a Juno address: {addr}")
    return addr
def execute_increment_msg_over_rpc(client: LedgerClient, wallet: LocalWallet, contract_address: str, memo: Optional[str] = None) -> Dict[str, Any]:
    msg = {"increment": {}}
    try:
        result = client.execute_contract(wallet, contract_address, msg, funds=[], memo=memo or "", gas_limit=200_000)
    except Exception as exc:
        raise RuntimeError(f"Failed to execute increment on contract: {exc}") from exc
    txhash = getattr(result, "txhash", None) or getattr(result, "tx_hash", None)
    if txhash is None and isinstance(result, dict):
        txhash = result.get("txhash") or result.get("tx_hash")
    if not txhash:
        raise RuntimeError(f"Could not determine transaction hash from result: {result}")
    return {"txhash": txhash, "raw_result": result}
def confirm_tx_via_lcd(txhash: str, max_attempts: int = 12, poll_interval: float = 2.0) -> Dict[str, Any]:
    LCD_BASE_URL = "https://lcd-archive.junonetwork.io"
    url = f"{LCD_BASE_URL}/cosmos/tx/v1beta1/txs/{txhash}"
    with requests.Client(timeout=10.0) as client:
        for attempt in range(max_attempts):
            try:
                resp = client.get(url)
            except requests.RequestException as exc:
                raise RuntimeError(f"Error querying LCD for tx {txhash}: {exc}") from exc
            if resp.status_code == 404:
                time.sleep(poll_interval)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"Unexpected status while fetching tx {txhash}: {resp.status_code} {resp.text}")
            body = resp.json()
            tx_response = body.get("tx_response")
            if not tx_response:
                raise RuntimeError(f"Malformed LCD response for tx {txhash}: {body}")
            code = int(tx_response.get("code", 0))
            if code != 0:
                raise RuntimeError(f"Transaction {txhash} failed with code={code}, raw_log={tx_response.get('raw_log')}")
            return tx_response
    raise RuntimeError(f"Transaction {txhash} not found in LCD after {max_attempts} attempts.")
def verify_incremented_count(contract_address: str, before: Optional[int]) -> Dict[str, Any]:
    after = get_current_count(contract_address)
    verified = before is not None and after == before + 1
    return {"before": before, "after": after, "verified": verified}

# [cite_start]Automatically extract the contract address from instantiate txhash [cite: 8]
async def lcd_get_tx_by_hash(tx_hash: str) -> Dict[str, Any]:
    if not tx_hash:
        raise ValueError("tx_hash is required")
    url = f"{LCD_BASE_URL}/cosmos/tx/v1beta1/txs/{tx_hash}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
        except httpx.RequestError as exc:
            raise LcdRequestError(f"Error while requesting {url}: {exc}") from exc
    if response.status_code != 200:
        raise LcdRequestError(f"LCD returned non-200 status {response.status_code}: {response.text}")
    data = response.json()
    tx_response = data.get("tx_response")
    if tx_response is None:
        raise LcdRequestError("LCD response missing 'tx_response' field")
    return tx_response
def parse_tx_logs_for_contract_address(tx_response: Dict[str, Any]) -> str:
    logs: List[Dict[str, Any]] = tx_response.get("logs", [])
    if not logs:
        raise ContractAddressNotFoundError("CONTRACT_ADDRESS_NOT_FOUND_IN_TX_LOGS")
    target_keys = {"_contract_address", "contract_address"}
    instantiate_event_types = {"instantiate", "wasm", "instantiate_contract"}
    for log in logs:
        for event in log.get("events", []):
            if event.get("type") in instantiate_event_types:
                for attr in event.get("attributes", []):
                    key = attr.get("key")
                    if key in target_keys:
                        value = attr.get("value")
                        if value:
                            return value
    for log in logs:
        for event in log.get("events", []):
            for attr in event.get("attributes", []):
                key = attr.get("key")
                if key in target_keys:
                    value = attr.get("value")
                    if value:
                        return value
    raise ContractAddressNotFoundError("CONTRACT_ADDRESS_NOT_FOUND_IN_TX_LOGS")
async def validate_contract_via_lcd(contract_address: str) -> Dict[str, Any]:
    if not contract_address:
        raise ValueError("contract_address is required")
    url = f"{LCD_BASE_URL}/cosmwasm/wasm/v1/contract/{contract_address}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
        except httpx.RequestError as exc:
            raise LcdRequestError(f"Error while requesting {url}: {exc}") from exc
    if response.status_code != 200:
        raise ContractNotFoundOnChainError("CONTRACT_NOT_FOUND_ON_CHAIN")
    data = response.json()
    contract_info = data.get("contract_info")
    if not isinstance(contract_info, dict):
        raise ContractNotFoundOnChainError("CONTRACT_NOT_FOUND_ON_CHAIN")
    if not contract_info.get("code_id") or not contract_info.get("creator"):
        raise ContractNotFoundOnChainError("CONTRACT_NOT_FOUND_ON_CHAIN")
    return data
def _build_error_payload(code: str, details: Optional[str] = None) -> Dict[str, Any]:
    ERROR_MESSAGES = {"CONTRACT_ADDRESS_NOT_FOUND_IN_TX_LOGS": "No CosmWasm contract address was found in the transaction logs.", "CONTRACT_NOT_FOUND_ON_CHAIN": "The extracted address is not a valid CosmWasm contract on-chain.", "LCD_REQUEST_FAILED": "Failed to query the Juno LCD.", "UNEXPECTED_ERROR": "An unexpected error occurred while resolving the contract address."}
    return {"ok": False, "error": {"code": code, "message": ERROR_MESSAGES.get(code, "Unexpected error."), "details": details}}

# Add cw-orch as an optional dependency in a CosmWasm contract's Cargo.toml
def load_cargo_toml(project_root: str = '.') -> Dict[str, Any]:
    path = Path(project_root).resolve() / 'Cargo.toml'
    if not path.is_file():
        raise FileNotFoundError(f'Cargo.toml not found at {path}')
    try:
        with path.open('r', encoding='utf-8') as fp:
            import toml # type: ignore
            data = toml.load(fp)
    except Exception as exc:
        raise RuntimeError(f'Failed to parse {path}: {exc}') from exc
    return data
def save_cargo_toml(data: Dict[str, Any], project_root: str = '.') -> None:
    path = Path(project_root).resolve() / 'Cargo.toml'
    try:
        with path.open('w', encoding='utf-8') as fp:
            import toml # type: ignore
            toml.dump(data, fp)
    except Exception as exc:
        raise RuntimeError(f'Failed to write {path}: {exc}') from exc
def add_cw_orch_optional_dependency(project_root: str = '.', version: str = '0.18.0') -> None:
    cargo = load_cargo_toml(project_root)
    dependencies = cargo.setdefault('dependencies', {})
    existing = dependencies.get('cw-orch')
    desired = {'version': version, 'optional': True}
    if existing is None:
        dependencies['cw-orch'] = desired
    elif isinstance(existing, str):
        desired['version'] = version or existing
        dependencies['cw-orch'] = desired
    elif isinstance(existing, dict):
        existing.update(desired)
    else:
        raise TypeError('Unexpected type for cw-orch dependency in Cargo.toml')
    save_cargo_toml(cargo, project_root)
def configure_cw_orch_feature(project_root: str = '.') -> None:
    cargo = load_cargo_toml(project_root)
    features = cargo.setdefault('features', {})
    default_feature = features.get('default')
    if default_feature is None:
        features['default'] = []
    elif not isinstance(default_feature, list):
        raise TypeError('The default feature in Cargo.toml must be a list')
    cw_orch_feature = features.get('cw-orch')
    if cw_orch_feature is None:
        features['cw-orch'] = ['dep:cw-orch']
    elif isinstance(cw_orch_feature, list):
        if 'dep:cw-orch' not in cw_orch_feature:
            cw_orch_feature.append('dep:cw-orch')
    else:
        raise TypeError('The cw-orch feature in Cargo.toml must be a list')
    save_cargo_toml(cargo, project_root)
def verify_cargo_with_cw_orch(project_root: str = '.', cargo_subcommand: str = 'check') -> Dict[str, Any]:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Project root directory not found: {root}')
    cmd = ['cargo', cargo_subcommand, '--features', 'cw-orch']
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError('cargo executable not found. Is Rust installed and on PATH?') from exc
    result: Dict[str, Any] = {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr, 'command': ' '.join(cmd), 'cwd': str(root)}
    if proc.returncode != 0:
        raise RuntimeError(f'Command failed with exit code {proc.returncode}. Stderr: {proc.stderr}')
    return result

# [cite_start]Get current count from contract CONTRACT_ADDRESS [cite: 9]
def resolve_contract_address(explicit_address: Optional[str] = None, env_var_name: str = 'CONTRACT_ADDRESS') -> str:
    if explicit_address:
        contract_address = explicit_address
    else:
        contract_address = os.getenv(env_var_name)
    if not contract_address:
        raise ValueError(f'No contract address provided and environment variable {env_var_name} is not set.')
    if not contract_address.startswith('juno1'):
        raise ValueError(f'Resolved contract address {contract_address} does not look like a Juno bech32 address.')
    return contract_address
def build_get_count_query_payload() -> str:
    query_obj = {'get_count': {}}
    try:
        json_bytes = json.dumps(query_obj, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    except (TypeError, ValueError) as e:
        raise ValueError(f'Failed to serialize get_count query JSON: {e}') from e
    encoded = base64.b64encode(json_bytes).decode('ascii')
    return encoded
def lcd_query_wasm_smart(contract_address: str, query_data_b64: str, timeout: float = 10.0) -> Dict[str, Any]:
    if not contract_address:
        raise ValueError('contract_address is required')
    if not query_data_b64:
        raise ValueError('query_data_b64 is required')
    query_data_escaped = quote(query_data_b64, safe='')
    url = f'{LCD_BASE_URL}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{query_data_escaped}'
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f'Failed to reach LCD endpoint {url}: {e}') from e
    if response.status_code != 200:
        raise RuntimeError(f'LCD query failed with status {response.status_code}: {response.text}')
    try:
        data = response.json()
    except ValueError as e:
        raise RuntimeError(f'LCD response is not valid JSON: {e}. Raw body: {response.text}') from e
    if 'data' not in data:
        raise KeyError(f"LCD JSON response is missing required 'data' field: {data}")
    return data
def decode_and_extract_count(lcd_response: Dict[str, Any]) -> int:
    if 'data' not in lcd_response:
        raise KeyError("lcd_response does not contain required 'data' field")
    data_b64 = lcd_response['data']
    if not isinstance(data_b64, str):
        raise TypeError(f"lcd_response['data'] must be a base64-encoded string, got {type(data_b64)}")
    try:
        raw_bytes = base64.b64decode(data_b64)
    except (base64.binascii.Error, ValueError) as e:
        raise ValueError(f"Failed to base64-decode lcd_response['data']: {e}") from e
    try:
        payload = json.loads(raw_bytes.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f'Decoded contract response is not valid UTF-8 JSON: {e}. Raw bytes: {raw_bytes!r}') from e
    if 'count' not in payload:
        raise KeyError(f"Contract query result does not contain 'count' field: {payload}")
    count_value = payload['count']
    try:
        return int(count_value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Contract 'count' field is not an integer: {count_value!r}") from e

# [cite_start]Query detailed CW721 nft_info for token ID 8 from a given NFT contract on Juno using a CosmWasm smart query. [cite: 10]
def validate_contract_and_token(contract_address: str, token_id: Union[str, int]) -> Tuple[str, str]:
    JUNO_ADDRESS_REGEX = re.compile(r'^juno1[0-9a-z]{38}$')
    if not isinstance(contract_address, str) or not JUNO_ADDRESS_REGEX.match(contract_address):
        raise ValidationError(f'Invalid Juno contract address: {contract_address!r}')
    if isinstance(token_id, int):
        if token_id < 0:
            raise ValidationError('token_id must be non-negative.')
        token_id_str = str(token_id)
    elif isinstance(token_id, str):
        token_id_str = token_id.strip()
        if not token_id_str:
            raise ValidationError('token_id string cannot be empty.')
    else:
        raise ValidationError('token_id must be either a string or an integer.')
    return contract_address, token_id_str
def build_nft_info_query_json(token_id: str) -> Dict:
    if not isinstance(token_id, str) or not token_id:
        raise ValueError('token_id must be a non-empty string.')
    query = {'nft_info': {'token_id': token_id}}
    return query
def encode_query_to_base64(query: Dict) -> str:
    try:
        json_bytes = json.dumps(query, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Failed to serialize query to JSON: {exc}') from exc
    encoded = base64.b64encode(json_bytes).decode('ascii')
    return encoded
def lcd_smart_query_nft_info(contract_address: str, query_data_b64: str, timeout: float = 10.0) -> Dict[str, Any]:
    class ContractQueryError(Exception):
        pass
    if not isinstance(query_data_b64, str) or not query_data_b64:
        raise ValueError('query_data_b64 must be a non-empty base64 string.')
    encoded_query_data = quote(query_data_b64, safe='')
    url = f'{LCD_BASE_URL}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{encoded_query_data}'
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise ContractQueryError(f'Network error while querying contract: {exc}') from exc
    if response.status_code != 200:
        try:
            err_json = response.json()
            message = err_json.get('message') or str(err_json)
        except ValueError:
            message = response.text
        raise ContractQueryError(f'LCD returned HTTP {response.status_code} for smart query: {message}')
    try:
        lcd_json = response.json()
    except ValueError as exc:
        raise ContractQueryError(f'Failed to parse LCD JSON response: {exc}') from exc
    if 'data' not in lcd_json:
        raise ContractQueryError(f'LCD response did not contain expected \'data\' field: {lcd_json}')
    return lcd_json
def decode_response_data(lcd_response: Dict[str, Any]) -> Dict[str, Any]:
    class ResponseDecodeError(Exception):
        pass
    if 'data' not in lcd_response:
        raise ResponseDecodeError('LCD response does not contain \'data\' field.')
    data_field = lcd_response['data']
    if not isinstance(data_field, str):
        raise ResponseDecodeError(f'Expected \'data\' to be a base64 string, got {type(data_field)} instead.')
    try:
        raw_bytes = base64.b64decode(data_field)
    except (ValueError, TypeError) as exc:
        raise ResponseDecodeError(f'Failed to base64-decode LCD data field: {exc}') from exc
    try:
        decoded_json = json.loads(raw_bytes.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponseDecodeError(f'Failed to decode JSON from contract data: {exc}') from exc
    return decoded_json
def return_nft_metadata(contract_address: str, token_id: Union[str, int]) -> Dict[str, Any]:
    class ContractQueryError(Exception):
        pass
    normalized_address, token_id_str = validate_contract_and_token(contract_address, token_id)
    query_json = build_nft_info_query_json(token_id_str)
    query_b64 = encode_query_to_base64(query_json)
    try:
        lcd_response = lcd_smart_query_nft_info(normalized_address, query_b64)
    except ContractQueryError as exc:
        msg = str(exc).lower()
        if 'not found' in msg or 'no token' in msg or ('token_id' in msg and 'does not exist' in msg):
            raise NftNotFoundError(f'NFT with token_id {token_id_str} does not exist.') from exc
        raise
    nft_info = decode_response_data(lcd_response)
    return nft_info

# [cite_start]Execute reset on contract CONTRACT_ADDRESS setting count to 0 [cite: 11, 12]
def get_sender_address() -> str:
    address = os.getenv('JUNO_SENDER_ADDRESS')
    if not address:
        raise RuntimeError('Environment variable JUNO_SENDER_ADDRESS is not set.')
    if not address.startswith('juno1'):
        raise ValueError(f'JUNO_SENDER_ADDRESS must start with prefix juno1. Got: {address}')
    return address
def get_account_info(sender_address: str):
    url = f'{LCD_BASE_URL}/cosmos/auth/v1beta1/accounts/{sender_address}'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f'Error querying account info: {e}') from e
    data = resp.json()
    account_any = data.get('account')
    if not account_any:
        raise RuntimeError(f'No account field in LCD response for {sender_address}')
    value_b64 = account_any.get('value')
    if not value_b64:
        raise RuntimeError('LCD account response missing protobuf-encoded BaseAccount value.')
    try:
        raw = base64.b64decode(value_b64)
    except (base64.binascii.Error, TypeError) as e:
        raise RuntimeError(f'Failed to base64-decode account value: {e}') from e
    base_account = BaseAccountProto()
    base_account.ParseFromString(raw)
    if not base_account.address:
        raise RuntimeError('Decoded BaseAccount has empty address.')
    if base_account.address != sender_address:
        raise RuntimeError(f'Address mismatch: requested {sender_address}, got {base_account.address} from LCD.')
    if not base_account.pub_key.type_url:
        raise RuntimeError('Account has no public key set on-chain (likely never used). Cannot construct SignerInfo without a pub_key.')
    return {'account_number': int(base_account.account_number), 'sequence': int(base_account.sequence), 'pub_key': base_account.pub_key}
def check_spendable_balance(sender_address: str, required_amount: int, denom: str = 'ujuno') -> int:
    url = f'{LCD_BASE_URL}/cosmos/bank/v1beta1/spendable_balances/{sender_address}'
    params = {'pagination.limit': '1000'}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f'Error querying spendable balances: {e}') from e
    data = resp.json()
    balances = data.get('balances', [])
    balance_amount = 0
    for coin in balances:
        if coin.get('denom') == denom:
            try:
                balance_amount = int(coin.get('amount', '0'))
            except ValueError:
                bad = coin.get('amount')
                raise RuntimeError(f'Invalid amount for {denom} in LCD response: {bad}')
            break
    if balance_amount == 0:
        raise RuntimeError(f'No spendable balance found for denom {denom} on address {sender_address}.')
    if balance_amount < required_amount:
        raise RuntimeError(f'Insufficient {denom} balance. Required: {required_amount}, available: {balance_amount}.')
    return balance_amount
def construct_msg_execute_reset(sender_address: str, contract_address: str) -> AnyProto:
    execute_payload = {'reset': {'count': 0}}
    msg = MsgExecuteContractProto(sender=sender_address, contract=contract_address, msg=json.dumps(execute_payload).encode('utf-8'), funds=[])
    any_msg = AnyProto()
    any_msg.Pack(msg, type_url_prefix='/')
    if any_msg.type_url != '/cosmwasm.wasm.v1.MsgExecuteContract':
        raise RuntimeError(f'Unexpected type_url for MsgExecuteContract: {any_msg.type_url}')
    return any_msg
def build_unsigned_tx(execute_msg_any: AnyProto, sequence: int, pub_key_any: AnyProto, gas_limit: int = 200000, fee_amount: int = 50000, fee_denom: str = 'ujuno', memo: str = '') -> TxProto:
    if gas_limit <= 0:
        raise ValueError('gas_limit must be positive.')
    if fee_amount < 0:
        raise ValueError('fee_amount cannot be negative.')
    tx_body = TxBodyProto(messages=[execute_msg_any], memo=memo)
    fee = FeeProto(amount=[CoinProto(denom=fee_denom, amount=str(fee_amount))], gas_limit=gas_limit)
    mode_info = TxProto.ModeInfo(single=TxProto.ModeInfo.Single(mode=SignModeProto.SIGN_MODE_DIRECT))
    signer_info = SignerInfoProto(public_key=pub_key_any, mode_info=mode_info, sequence=sequence)
    auth_info = AuthInfoProto(signer_infos=[signer_info], fee=fee)
    tx = TxProto(body=tx_body, auth_info=auth_info, signatures=[])
    return tx
def simulate_tx_and_update_fee(tx: TxProto, gas_price: Decimal = Decimal('0.025'), gas_adjustment: float = 1.2, fee_denom: str = 'ujuno') -> dict:
    if gas_adjustment <= 0:
        raise ValueError('gas_adjustment must be positive.')
    tx_bytes = tx.SerializeToString()
    tx_bytes_b64 = base64.b64encode(tx_bytes).decode('utf-8')
    url = f'{LCD_BASE_URL}/cosmos/tx/v1beta1/simulate'
    payload = {'tx_bytes': tx_bytes_b64}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f'Error simulating transaction: {e}') from e
    sim = resp.json()
    gas_info = sim.get('gas_info') or {}
    gas_used_str = gas_info.get('gas_used')
    if gas_used_str is None:
        raise RuntimeError(f'Simulation response missing gas_used: {sim}')
    try:
        gas_used = int(gas_used_str)
    except ValueError:
        raise RuntimeError(f'Invalid gas_used in simulation response: {gas_used_str}')
    gas_limit_decimal = Decimal(gas_used) * Decimal(str(gas_adjustment))
    gas_limit = int(gas_limit_decimal.quantize(Decimal('1'), rounding=ROUND_UP))
    fee_amount_decimal = Decimal(gas_limit) * gas_price
    fee_amount = int(fee_amount_decimal.quantize(Decimal('1'), rounding=ROUND_UP))
    tx.auth_info.fee.gas_limit = gas_limit
    tx.auth_info.fee.amount[:] = [CoinProto(denom=fee_denom, amount=str(fee_amount))]
    return {'gas_used': gas_used, 'gas_limit': gas_limit, 'fee_amount': fee_amount, 'fee_denom': fee_denom}
def sign_tx(tx: TxProto, account_number: int, chain_id: str = 'juno-1', private_key_hex: str = None) -> dict:
    if private_key_hex is None:
        private_key_hex = os.getenv('JUNO_PRIVATE_KEY_HEX')
    if not private_key_hex:
        raise RuntimeError('Missing private key. Set JUNO_PRIVATE_KEY_HEX or pass private_key_hex explicitly.')
    try:
        privkey_bytes = bytes.fromhex(private_key_hex)
    except ValueError as e:
        raise RuntimeError('JUNO_PRIVATE_KEY_HEX must be a valid hex string.') from e
    privkey = PrivateKey(privkey_bytes)
    body_bytes = tx.body.SerializeToString()
    auth_info_bytes = tx.auth_info.SerializeToString()
    sign_doc = SignDocProto(body_bytes=body_bytes, auth_info_bytes=auth_info_bytes, chain_id=chain_id, account_number=account_number)
    sign_bytes = sign_doc.SerializeToString()
    signature = privkey.sign(sign_bytes)
    tx.signatures[:] = [signature]
    raw_tx_bytes = tx.SerializeToString()
    tx_bytes_b64 = base64.b64encode(raw_tx_bytes).decode('utf-8')
    return {'tx': tx, 'raw_tx_bytes': raw_tx_bytes, 'tx_bytes_b64': tx_bytes_b64}
def broadcast_tx(tx_bytes_b64: str, mode: str = 'BROADCAST_MODE_SYNC') -> dict:
    url = f'{LCD_BASE_URL}/cosmos/tx/v1beta1/txs'
    payload = {'tx_bytes': tx_bytes_b64, 'mode': mode}
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f'Error broadcasting transaction: {e}') from e
    data = resp.json()
    tx_response = data.get('tx_response')
    if not tx_response:
        raise RuntimeError(f'LCD broadcast response missing tx_response: {data}')
    code = int(tx_response.get('code', 0))
    if code != 0:
        raw_log = tx_response.get('raw_log')
        raise RuntimeError(f'Transaction failed with code {code}: {raw_log}')
    return tx_response
def verify_reset_effect(contract_address: str, expected_count: int = 0) -> int:
    query_msg = {'get_count': {}}
    query_b64 = base64.b64encode(json.dumps(query_msg).encode('utf-8')).decode('utf-8')
    url = f'{LCD_BASE_URL}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{query_b64}'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f'Error querying contract state: {e}') from e
    data = resp.json()
    data_b64 = data.get('data')
    if data_b64 is None:
        raise RuntimeError(f'Unexpected smart query response (missing data field): {data}')
    try:
        raw = base64.b64decode(data_b64)
        decoded = json.loads(raw.decode('utf-8'))
    except Exception as e:
        raise RuntimeError(f'Failed to decode smart query response payload: {e}') from e
    count = decoded.get('count')
    if count != expected_count:
        raise RuntimeError(f'Contract count mismatch. Expected {expected_count}, got {count}.')
    return count

# [cite_start]Execute increment on contract address CONTRACT_ADDRESS with 10ujuno [cite: 13, 14]
def lcd_verify_contract_exists(contract_address: str) -> dict:
    if not contract_address:
        raise ValueError('contract_address must be a non-empty Bech32 address.')
    url = f'{LCD_BASE_URL}/cosmwasm/wasm/v1/contract/{contract_address}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to query contract info from LCD: {exc}') from exc
    data = response.json()
    if 'contract_info' not in data:
        raise ValueError(f'LCD response does not contain contract_info for {contract_address}: {data}')
    return data
def bff_prepare_execute_msg_increment() -> Tuple[Dict[str, Any], bytes]:
    msg = {'increment': {}}
    try:
        json_str = json.dumps(msg, separators=(',', ':'), sort_keys=True)
        msg_bytes = json_str.encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f'Failed to encode execute message as JSON: {exc}') from exc
    return msg, msg_bytes
def bff_get_chain_and_account_info(address: str) -> Dict[str, Any]:
    if not address:
        raise ValueError('address must be a non-empty Bech32 string.')
    node_info_url = f'{LCD_BASE_URL}/cosmos/base/tendermint/v1beta1/node_info'
    try:
        node_resp = requests.get(node_info_url, timeout=10)
        node_resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to query node_info from LCD: {exc}') from exc
    node_data = node_resp.json()
    default_node_info = node_data.get('default_node_info') or {}
    chain_id = default_node_info.get('network')
    if not chain_id:
        raise RuntimeError(f'Could not determine chain_id from node_info response: {node_data}')
    acct_url = f'{LCD_BASE_URL}/cosmos/auth/v1beta1/accounts/{address}'
    try:
        acct_resp = requests.get(acct_url, timeout=10)
        acct_resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to query account data from LCD: {exc}') from exc
    acct_data = acct_resp.json()
    account = acct_data.get('account')
    if not account:
        raise RuntimeError(f'No account data found for address {address}: {acct_data}')
    account_number_str = account.get('account_number')
    sequence_str = account.get('sequence')
    if account_number_str is None or sequence_str is None:
        raise RuntimeError(f'Account number or sequence missing in LCD response for {address}: {account}')
    try:
        account_number = int(account_number_str)
        sequence = int(sequence_str)
    except ValueError as exc:
        raise RuntimeError(f'Failed to parse account_number/sequence as integers: {account_number_str}/{sequence_str}') from exc
    return {'chain_id': chain_id, 'account_number': account_number, 'sequence': sequence, 'raw_account': account}
def lcd_check_spendable_balance_for_ujuno(address: str, required_ujuno: int) -> Dict[str, Any]:
    if not address:
        raise ValueError('address must be a non-empty Bech32 string.')
    if required_ujuno <= 0:
        raise ValueError('required_ujuno must be a positive integer representing ujuno.')
    url = f'{LCD_BASE_URL}/cosmos/bank/v1beta1/balances/{address}/by_denom'
    params = {'denom': 'ujuno'}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to query ujuno balance from LCD: {exc}') from exc
    data = resp.json()
    balance_obj = data.get('balance') or {}
    amount_str = balance_obj.get('amount', '0')
    try:
        available = int(amount_str)
    except ValueError as exc:
        raise RuntimeError(f'LCD returned non-integer balance amount: {amount_str}') from exc
    if available < required_ujuno:
        raise RuntimeError(f'Insufficient ujuno balance. Required {required_ujuno}, available {available}.')
    return {'available_ujuno': available, 'required_ujuno': required_ujuno}
def bff_construct_execute_contract_tx(sender: str, contract_address: str, msg_bytes: bytes, funds_ujuno: int = 10, initial_gas_limit: int = 200000, gas_price_ujuno: Decimal = Decimal('0.025')) -> TxRawProto:
    if not sender or not contract_address:
        raise ValueError('sender and contract_address must be non-empty strings.')
    if initial_gas_limit <= 0:
        raise ValueError('initial_gas_limit must be positive.')
    exec_msg = MsgExecuteContractProto(sender=sender, contract=contract_address, msg=msg_bytes, funds=[CosmosCoinProto(denom='ujuno', amount=str(funds_ujuno))])
    any_msg = AnyProto(type_url='/cosmwasm.wasm.v1.MsgExecuteContract', value=exec_msg.SerializeToString())
    tx_body = TxBodyProto(messages=[any_msg], memo='', timeout_height=0)
    body_bytes = tx_body.SerializeToString()
    fee_amount = (Decimal(initial_gas_limit) * gas_price_ujuno).to_integral_value(rounding=ROUND_UP)
    fee = FeeProto(amount=[CosmosCoinProto(denom='ujuno', amount=str(fee_amount))], gas_limit=initial_gas_limit, payer='', granter='')
    auth_info = AuthInfoProto(signer_infos=[], fee=fee)
    auth_info_bytes = auth_info.SerializeToString()
    tx_raw = TxRawProto(body_bytes=body_bytes, auth_info_bytes=auth_info_bytes, signatures=[])
    return tx_raw
def lcd_simulate_tx(tx_raw: TxRawProto, gas_adjustment: float = 1.3, gas_price_ujuno: Decimal = Decimal('0.025')) -> Dict[str, Any]:
    if gas_adjustment <= 0:
        raise ValueError('gas_adjustment must be positive.')
    tx_bytes = tx_raw.SerializeToString()
    payload = {'tx_bytes': base64.b64encode(tx_bytes).decode('ascii')}
    url = f'{LCD_BASE_URL}/cosmos/tx/v1beta1/simulate'
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to simulate transaction via LCD: {exc}') from exc
    sim_result = resp.json()
    gas_info = sim_result.get('gas_info') or {}
    gas_used_str = gas_info.get('gas_used')
    if gas_used_str is None:
        raise RuntimeError(f'LCD simulation response missing gas_used: {sim_result}')
    try:
        gas_used = int(gas_used_str)
    except ValueError as exc:
        raise RuntimeError(f'Invalid gas_used value: {gas_used_str}') from exc
    recommended_gas_limit = int(gas_used * gas_adjustment)
    fee_amount = (Decimal(recommended_gas_limit) * gas_price_ujuno).to_integral_value(rounding=ROUND_UP)
    return {'gas_used': gas_used, 'recommended_gas_limit': recommended_gas_limit, 'recommended_fee_ujuno': int(fee_amount), 'raw_response': sim_result}
def bff_sign_execute_tx(unsigned_tx: TxRawProto, chain_id: str, account_number: int, sequence: int, fee_amount_ujuno: int, gas_limit: int, private_key_hex: str) -> bytes:
    if fee_amount_ujuno < 0:
        raise ValueError('fee_amount_ujuno must be non-negative.')
    if gas_limit <= 0:
        raise ValueError('gas_limit must be positive.')
    if not chain_id:
        raise ValueError('chain_id must be a non-empty string.')
    try:
        privkey = PrivateKey(bytes.fromhex(private_key_hex))
    except Exception as exc:
        raise RuntimeError('Failed to load private key from hex.') from exc
    pubkey = Secp256k1PubKeyProto(key=privkey.public_key.bytes)
    any_pubkey = AnyProto(type_url='/cosmos.crypto.secp256k1.PubKey', value=pubkey.SerializeToString())
    mode_info = TxProto.ModeInfo(single=TxProto.ModeInfo.Single(mode=SignModeProto.SIGN_MODE_DIRECT))
    signer_info = SignerInfoProto(public_key=any_pubkey, mode_info=mode_info, sequence=sequence)
    fee = FeeProto(amount=[CosmosCoinProto(denom='ujuno', amount=str(fee_amount_ujuno))], gas_limit=gas_limit, payer='', granter='')
    auth_info = AuthInfoProto(signer_infos=[signer_info], fee=fee)
    auth_info_bytes = auth_info.SerializeToString()
    sign_doc = SignDocProto(body_bytes=unsigned_tx.body_bytes, auth_info_bytes=auth_info_bytes, chain_id=chain_id, account_number=account_number)
    sign_doc_bytes = sign_doc.SerializeToString()
    signature = privkey.sign(sign_doc_bytes)
    signed_tx = TxRawProto(body_bytes=unsigned_tx.body_bytes, auth_info_bytes=auth_info_bytes, signatures=[signature])
    return signed_tx.SerializeToString()
def lcd_broadcast_tx(tx_bytes: bytes, mode: str = 'BROADCAST_MODE_BLOCK') -> Dict[str, Any]:
    if not tx_bytes:
        raise ValueError('tx_bytes must not be empty.')
    payload = {'tx_bytes': base64.b64encode(tx_bytes).decode('ascii'), 'mode': mode}
    url = f'{LCD_BASE_URL}/cosmos/tx/v1beta1/txs'
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f'Failed to broadcast transaction via LCD: {exc}') from exc
    data = resp.json()
    tx_response = data.get('tx_response')
    if tx_response is None:
        raise RuntimeError(f'LCD broadcast response missing tx_response: {data}')
    return tx_response
def lcd_verify_execute_effect(contract_address: str, tx_response: Dict[str, Any]) -> Dict[str, Any]:
    if tx_response is None:
        raise ValueError('tx_response must not be None.')
    code = int(tx_response.get('code', 0))
    if code != 0:
        raise RuntimeError(f'Execute transaction failed with code {code}: {tx_response.get("raw_log")}')
    wasm_events = []
    for log in tx_response.get('logs', []):
        for event in log.get('events', []):
            if event.get('type') == 'wasm':
                wasm_events.append(event)
    if not wasm_events:
        raise RuntimeError('No wasm events found in tx_response logs; execute may not have run as expected.')
    return {'success': True, 'txhash': tx_response.get('txhash'), 'height': tx_response.get('height'), 'wasm_events': wasm_events}

# [cite_start]Query the smart (contract-state smart) view of a CosmWasm contract on Juno, using junod CLI and/or the LCD endpoint. [cite: 15]
def collect_query_parameters(payload: Dict[str, Any]) -> SmartQueryParameters:
    if not isinstance(payload, dict):
        raise ValueError('payload must be a dict')
    contract_address = payload.get('contract_address')
    if not isinstance(contract_address, str) or not contract_address:
        raise ValueError('contract_address (bech32) is required and must be a non-empty string')
    raw_query = payload.get('query')
    if raw_query is None:
        raise ValueError('query is required (e.g. {config: {}})')
    if isinstance(raw_query, str):
        try:
            query_obj = json.loads(raw_query)
        except json.JSONDecodeError as exc:
            raise ValueError(f'query string must be valid JSON: {exc}') from exc
    elif isinstance(raw_query, dict):
        query_obj = raw_query
    else:
        raise ValueError('query must be either a dict or a JSON string')
    lcd_url = payload.get('lcd_url') or os.getenv('JUNO_LCD_URL') or LCD_BASE_URL
    rpc_url = payload.get('rpc_url') or os.getenv('JUNO_RPC_URL')
    chain_id = payload.get('chain_id') or os.getenv('JUNO_CHAIN_ID')
    keyring_backend = payload.get('keyring_backend') or os.getenv('JUNO_KEYRING_BACKEND')
    node_cfg = NodeConfig(lcd_url=lcd_url, rpc_url=rpc_url, chain_id=chain_id, keyring_backend=keyring_backend)
    return SmartQueryParameters(contract_address=contract_address, query=query_obj, node=node_cfg)
def encode_smart_query_for_lcd(query: Dict[str, Any]) -> str:
    if not isinstance(query, dict):
        raise ValueError('query must be a dict representing the JSON payload')
    try:
        json_str = json.dumps(query, separators=(',', ':'), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'query could not be serialized to JSON: {exc}') from exc
    json_bytes = json_str.encode('utf-8')
    b64 = base64.b64encode(json_bytes).decode('ascii')
    return b64
def http_get_contract_smart_state(contract_address: str, query_data_b64: str, lcd_url: str = LCD_BASE_URL, timeout: float = 10.0) -> Dict[str, Any]:
    if not contract_address or not isinstance(contract_address, str):
        raise ValueError('contract_address must be a non-empty string')
    if not query_data_b64 or not isinstance(query_data_b64, str):
        raise ValueError('query_data_b64 must be a non-empty base64 string')
    base = lcd_url.rstrip('/')
    encoded_addr = quote(contract_address, safe='')
    encoded_query = quote(query_data_b64, safe='')
    url = f'{base}/cosmwasm/wasm/v1/contract/{encoded_addr}/smart/{encoded_query}'
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise requests.RequestException(f'Failed GET {url}: {exc}') from exc
    try:
        body = resp.json()
    except ValueError as exc:
        raise ValueError(f'LCD response was not valid JSON: {exc}; raw={resp.text!r}') from exc
    if 'data' not in body:
        raise ValueError(f'Unexpected LCD response; missing data field: {body}')
    return body
def decode_lcd_smart_query_response(lcd_response: Dict[str, Any]) -> Union[Any, bytes]:
    if not isinstance(lcd_response, dict):
        raise ValueError('lcd_response must be a dict')
    if 'data' not in lcd_response:
        raise KeyError('lcd_response is missing required data field')
    b64_data = lcd_response['data']
    if not isinstance(b64_data, str):
        raise ValueError('data field must be a base64 string')
    try:
        raw_bytes = base64.b64decode(b64_data)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Failed to base64-decode LCD data field: {exc}') from exc
    try:
        text = raw_bytes.decode('utf-8')
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw_bytes
def execute_junod_cli_smart_query(contract_address: str, query: Dict[str, Any], rpc_endpoint: Optional[str], chain_id: Optional[str], junod_binary: str = 'junod') -> Dict[str, Any]:
    if not contract_address or not isinstance(contract_address, str):
        raise ValueError('contract_address must be a non-empty string')
    if not isinstance(query, dict):
        raise ValueError('query must be a dict representing the JSON payload')
    if not rpc_endpoint or not chain_id:
        raise ValueError('rpc_endpoint and chain_id are required to run junod CLI queries')
    try:
        query_str = json.dumps(query, separators=(',', ':'), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'query could not be serialized to JSON: {exc}') from exc
    cmd = [junod_binary, 'query', 'wasm', 'contract-state', 'smart', contract_address, query_str, '--node', rpc_endpoint, '--chain-id', chain_id, '--output', 'json']
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f'junod binary not found: {junod_binary}') from exc
    except Exception as exc:
        raise RuntimeError(f'Failed to execute junod CLI: {exc}') from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        stdout_display = stdout or '<empty>'
        stderr_display = stderr or '<empty>'
        raise RuntimeError(f'junod query failed with exit code {completed.returncode}. STDOUT: {stdout_display} STDERR: {stderr_display}')
    try:
        cli_output = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'Failed to parse junod JSON output: {exc}; raw={completed.stdout!r}') from exc
    return cli_output
def compare_cli_and_lcd_results(lcd_decoded_result: Any, lcd_raw_response: Dict[str, Any], cli_response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cli_response, dict):
        raise ValueError('cli_response must be a dict')
    if 'data' not in cli_response:
        raise ValueError(f'CLI response is missing data field: {cli_response}')
    data_field = cli_response['data']
    if not isinstance(data_field, str):
        raise ValueError('data field of CLI response must be a base64 string')
    try:
        cli_bytes = base64.b64decode(data_field)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Failed to base64-decode CLI data field: {exc}') from exc
    try:
        cli_text = cli_bytes.decode('utf-8')
        cli_decoded = json.loads(cli_text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        cli_decoded = cli_bytes
    if lcd_decoded_result != cli_decoded:
        raise ValueError('LCD and junod CLI smart query results differ. Decoded LCD result: {lcd_decoded_result!r} Decoded CLI result: {cli_decoded!r}')
    return {'match': True, 'lcd': lcd_decoded_result, 'cli': cli_decoded}

# [cite_start]Compile the current CosmWasm smart contract using rust-optimizer [cite: 16]
def detect_contract_project_root(start_dir: Optional[str] = None) -> Path:
    start_path = Path(start_dir).resolve() if start_dir else Path.cwd().resolve()
    current = start_path
    for candidate in [current] + list(current.parents):
        cargo_toml = candidate / 'Cargo.toml'
        if cargo_toml.is_file():
            cargo_data = _load_toml(cargo_toml)
            if 'workspace' in cargo_data:
                raise ProjectRootError('Found Cargo.toml but it defines a [workspace]. This helper expects to run inside a single-contract project.')
            if 'package' not in cargo_data:
                raise ProjectRootError('Cargo.toml does not contain a [package] section.')
            return candidate
    raise ProjectRootError(f'Could not locate a suitable Cargo.toml when walking up from {start_path}.')
def read_contract_name_from_cargo(project_root: str) -> Tuple[str, Path]:
    root = Path(project_root).resolve()
    cargo_toml = root / 'Cargo.toml'
    if not cargo_toml.is_file():
        raise CargoReadError(f'Cargo.toml not found at {cargo_toml}')
    try:
        cargo_data = _load_toml(cargo_toml)
        package_table = cargo_data['package']
        contract_name = package_table['name']
    except KeyError as exc:
        raise CargoReadError(f'Cargo.toml at {cargo_toml} is missing the [package] name field.') from exc
    artifacts_dir = root / 'artifacts'
    artifact_path = artifacts_dir / f'{contract_name}.wasm'
    return contract_name, artifact_path
def run_rust_optimizer(project_root: str, use_docker: bool = True, docker_image: str = 'cosmwasm/rust-optimizer:0.14.0', timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    if not (root / 'Cargo.toml').is_file():
        raise RustOptimizerError(f'No Cargo.toml found under {root}. Is this a valid CosmWasm contract project?')
    if use_docker:
        target_dir = root / 'target'
        cmd = ['docker', 'run', '--rm', '-v', f'{root}:/code', '-v', f'{target_dir}:/target', '-v', 'rust-optimizer-cache:/usr/local/cargo/registry', docker_image]
        working_dir = root
    else:
        cmd = ['cargo', 'wasm', '--locked', '--release']
        working_dir = root
    try:
        completed = subprocess.run(cmd, cwd=working_dir, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds)
    except FileNotFoundError as exc:
        raise RustOptimizerError(f'Failed to execute {cmd[0]!r}. Ensure it is installed and available on PATH.') from exc
    except subprocess.TimeoutExpired as exc:
        raise RustOptimizerError(f'rust-optimizer command timed out after {timeout_seconds} seconds.') from exc
    if completed.returncode != 0:
        raise RustOptimizerError(f'rust-optimizer command failed with non-zero exit code {completed.returncode}.\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}')
    return {'stdout': completed.stdout, 'stderr': completed.stderr, 'returncode': completed.returncode, 'command': cmd}
def verify_wasm_artifact(artifact_path: str, optimizer_stdout: Optional[str] = None, optimizer_stderr: Optional[str] = None, run_wasm_validator: bool = True, wasm_validator_cmd: Optional[list] = None) -> Dict[str, Any]:
    path = Path(artifact_path).resolve()
    if not path.is_file():
        msg = f'WASM artifact not found at {path}.'
        if optimizer_stdout or optimizer_stderr: msg += '\n\nOptimizer STDOUT:\n' + (optimizer_stdout or '') + '\n\nOptimizer STDERR:\n' + (optimizer_stderr or '')
        raise WasmArtifactError(msg)
    size = path.stat().st_size
    if size == 0:
        msg = f'WASM artifact at {path} is empty (0 bytes).'
        if optimizer_stdout or optimizer_stderr: msg += '\n\nOptimizer STDOUT:\n' + (optimizer_stdout or '') + '\n\nOptimizer STDERR:\n' + (optimizer_stderr or '')
        raise WasmArtifactError(msg)
    with path.open('rb') as f:
        magic = f.read(4)
    if magic != b'\x00asm':
        msg = f'File at {path} does not appear to be a valid WebAssembly module. Expected magic bytes 0x00 0x61 0x73 0x6d, got {magic!r}. '
        if optimizer_stdout or optimizer_stderr: msg += '\n\nOptimizer STDOUT:\n' + (optimizer_stdout or '') + '\n\nOptimizer STDERR:\n' + (optimizer_stderr or '')
        raise WasmArtifactError(msg)
    validator_output: Dict[str, Any] = {}
    return {'artifact_path': str(path), 'size_bytes': size, 'magic_bytes': list(magic), 'validator': validator_output}

# [cite_start]Execute a CosmWasm contract while attaching tokens, equivalent to using the `--amount` flag in `junod tx wasm execute`. [cite: 17, 18]
def parse_and_validate_amounts(coin_str: str, allowed_denoms: Optional[List[str]] = None) -> List[Dict[str, str]]:
    COIN_RE = re.compile(r'^([0-9]+)([a-zA-Z0-9/]+)$')
    if not coin_str:
        raise AmountParseError('Coin string must not be empty.')
    if allowed_denoms is None:
        allowed_denoms = ['ujuno', 'ujunox']
    coins: List[Dict[str, str]] = []
    for part in coin_str.split(','):
        token = part.strip()
        if not token: continue
        m = COIN_RE.match(token)
        if not m: raise AmountParseError(f'Invalid coin segment {token}. Expected format like 100000ujuno.')
        amount_str, denom = m.groups()
        if not amount_str.isdigit(): raise AmountParseError(f'Amount {amount_str} in segment {token} is not a valid integer.')
        if amount_str == '0': raise AmountParseError('Amounts must be strictly positive.')
        if len(amount_str) > 1 and amount_str.startswith('0'): raise AmountParseError(f'Amount {amount_str} must not contain leading zeros.')
        if denom not in allowed_denoms: raise AmountParseError(f'Denom {denom} is not in the list of allowed Juno denoms: {allowed_denoms}.')
        coins.append({'denom': denom, 'amount': amount_str})
    if not coins:
        raise AmountParseError('No valid coin segments were found in input.')
    return coins
async def check_spendable_balances_for_sender(sender_address: str, required_funds: List[Dict[str, str]], expected_fee_per_denom: Optional[Dict[str, int]] = None, lcd_url: str = LCD_BASE_URL) -> Dict[str, Dict[str, int]]:
    if expected_fee_per_denom is None:
        expected_fee_per_denom = {}
    url = f'{lcd_url}/cosmos/bank/v1beta1/spendable_balances/{sender_address}'
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f'Error querying spendable balances from LCD: {exc}') from exc
    data = resp.json()
    spendable_map: Dict[str, int] = {}
    for coin in data.get('balances', []):
        denom = coin.get('denom')
        amount_str = coin.get('amount', '0')
        if denom is None: continue
        try:
            spendable_map[denom] = int(amount_str)
        except ValueError:
            raise RuntimeError(f'LCD returned a non-integer amount {amount_str} for denom {denom}.')
    for coin in required_funds:
        denom = coin['denom']
        raw_amount = coin['amount']
        try:
            required_amount = int(raw_amount)
        except ValueError:
            raise ValueError(f'Required amount {raw_amount} for denom {denom} is not a valid integer.')
        fee_buffer = int(expected_fee_per_denom.get(denom, 0))
        total_required = required_amount + fee_buffer
        available = spendable_map.get(denom, 0)
        if available < total_required:
            raise InsufficientBalanceError(f'Insufficient spendable balance for denom {denom}. Required including fees: {total_required}, available: {available}.')
    return {'spendable_balances': spendable_map}
def build_execute_contract_msg_with_funds(sender: str, contract: str, execute_msg: Dict, funds: List[Dict[str, str]]) -> MsgExecuteContractProto:
    if not isinstance(execute_msg, dict):
        raise ExecuteMsgBuildError('execute_msg must be a JSON-serializable dict.')
    if not funds:
        raise ExecuteMsgBuildError('funds must be a non-empty list when attaching tokens.')
    try:
        msg_bytes = json.dumps(execute_msg, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ExecuteMsgBuildError(f'Failed to JSON-encode execute_msg: {exc}') from exc
    proto_funds = []
    for coin in funds:
        denom = coin['denom']
        amount = coin['amount']
        if not denom or not amount:
            raise ExecuteMsgBuildError(f'Invalid fund entry: {coin}.')
        proto_funds.append(CosmosCoinProto(denom=denom, amount=str(amount)))
    msg = MsgExecuteContractProto(sender=sender, contract=contract, msg=msg_bytes, funds=proto_funds)
    return msg
async def _get_minimum_gas_price(lcd_url: str = LCD_BASE_URL) -> Tuple[Decimal, str]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f'{lcd_url}/cosmos/base/node/v1beta1/config')
        resp.raise_for_status()
    cfg = resp.json()
    mgp = cfg.get('minimum_gas_price')
    if not mgp: raise TxConstructionError('LCD /config did not return minimum_gas_price.')
    i = 0
    while i < len(mgp) and (mgp[i].isdigit() or mgp[i] == '.'): i += 1
    amount_str = mgp[:i]
    denom = mgp[i:]
    if not amount_str or not denom: raise TxConstructionError(f'Could not parse minimum_gas_price value {mgp}.')
    return Decimal(amount_str), denom
async def construct_execute_tx(msg: MsgExecuteContractProto, public_key_bytes: bytes, sequence: int, gas_limit: int = 200_000, gas_adjustment: float = 1.0, lcd_url: str = LCD_BASE_URL) -> Tuple[TxProto, int, CosmosCoinProto]:
    if gas_limit <= 0: raise TxConstructionError('gas_limit must be positive.')
    effective_gas_limit = int(gas_limit * gas_adjustment)
    try:
        gas_price_amount, gas_price_denom = await _get_minimum_gas_price(lcd_url)
    except httpx.HTTPError as exc:
        raise TxConstructionError(f'Failed to fetch minimum_gas_price from LCD: {exc}') from exc
    fee_decimal = (gas_price_amount * Decimal(effective_gas_limit)).to_integral_value(rounding=ROUND_UP)
    fee_amount = int(fee_decimal)
    fee_coin = CosmosCoinProto(denom=gas_price_denom, amount=str(fee_amount))
    msg_any = AnyProto(type_url='/cosmwasm.wasm.v1.MsgExecuteContract', value=msg.SerializeToString())
    tx_body = TxBodyProto(messages=[msg_any], memo='', timeout_height=0)
    pubkey = Secp256k1PubKeyProto(key=public_key_bytes)
    pubkey_any = AnyProto(type_url='/cosmos.crypto.secp256k1.PubKey', value=pubkey.SerializeToString())
    mode_info = TxProto.ModeInfo(single=TxProto.ModeInfo.Single(mode=SignModeProto.SIGN_MODE_DIRECT))
    signer_info = SignerInfoProto(public_key=pubkey_any, mode_info=mode_info, sequence=sequence)
    auth_info = AuthInfoProto(signer_infos=[signer_info], fee=FeeProto(amount=[fee_coin], gas_limit=effective_gas_limit, payer='', granter=''))
    unsigned_tx = TxProto(body=tx_body, auth_info=auth_info, signatures=[])
    return unsigned_tx, effective_gas_limit, fee_coin
async def simulate_execute_tx(unsigned_tx: TxProto, gas_adjustment: float = 1.2, lcd_url: str = LCD_BASE_URL) -> Tuple[int, int]:
    if gas_adjustment <= 0: raise ValueError('gas_adjustment must be positive.')
    tx_bytes = unsigned_tx.SerializeToString()
    payload = {'tx_bytes': base64.b64encode(tx_bytes).decode('utf-8')}
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.post(f'{lcd_url}/cosmos/tx/v1beta1/simulate', json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise TxSimulationError(f'Error calling /cosmos/tx/v1beta1/simulate: {exc}') from exc
    data = resp.json()
    gas_info = data.get('gas_info') or {}
    gas_used_str = gas_info.get('gas_used')
    if gas_used_str is None: raise TxSimulationError(f'Simulation response missing field gas_used: {data}')
    try:
        gas_used = int(gas_used_str)
    except ValueError:
        raise TxSimulationError(f'Simulation returned non-integer gas_used {gas_used_str}.')
    adjusted_gas_limit = int(gas_used * gas_adjustment)
    return gas_used, adjusted_gas_limit
async def sign_and_broadcast_execute_tx(unsigned_tx: TxProto, privkey_hex: str, account_number: int, sequence: int, chain_id: str, lcd_url: str = LCD_BASE_URL) -> Tuple[str, dict]:
    key_hex = privkey_hex.lower().replace('0x', '')
    if len(key_hex) != 64: raise ValueError('Private key must be 32 bytes (64 hex characters).')
    try:
        privkey = PrivateKey(bytes.fromhex(key_hex))
    except ValueError as exc:
        raise ValueError(f'Invalid private key hex: {exc}') from exc
    wallet = LocalWallet(privkey)
    signed_tx: TxProto = wallet.sign_tx(unsigned_tx, account_number=account_number, sequence=sequence, chain_id=chain_id)
    tx_bytes = signed_tx.SerializeToString()
    payload = {'tx_bytes': base64.b64encode(tx_bytes).decode('utf-8'), 'mode': 'BROADCAST_MODE_BLOCK'}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f'{lcd_url}/cosmos/tx/v1beta1/txs', json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise TxBroadcastError(f'Error broadcasting transaction: {exc}') from exc
    data = resp.json()
    tx_response = data.get('tx_response') or {}
    txhash = tx_response.get('txhash')
    if not txhash: raise TxBroadcastError(f'LCD did not return txhash. Full response: {data}')
    code = tx_response.get('code', 0)
    if code != 0:
        raw_log = tx_response.get('raw_log', '')
        raise TxBroadcastError(f'Transaction failed with code {code}. txhash={txhash}, raw_log={raw_log}')
    return txhash, tx_response
async def _get_balance_by_denom(address: str, denom: str, lcd_url: str = LCD_BASE_URL) -> int:
    url = f'{lcd_url}/cosmos/bank/v1beta1/balances/{address}/by_denom'
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params={'denom': denom})
        resp.raise_for_status()
    data = resp.json()
    balance = data.get('balance') or {}
    amount_str = balance.get('amount', '0')
    try:
        return int(amount_str)
    except ValueError:
        raise RuntimeError(f'LCD returned non-integer balance {amount_str} for {address} denom {denom}.')
async def verify_funds_transferred_to_contract(contract_address: str, sender_address: str, attached_funds: List[Dict[str, str]], before_contract_balances: Optional[Dict[str, int]] = None, before_sender_balances: Optional[Dict[str, int]] = None, lcd_url: str = LCD_BASE_URL) -> Dict[str, Dict[str, int]]:
    after_contract: Dict[str, int] = {}
    after_sender: Dict[str, int] = {}
    for coin in attached_funds:
        denom = coin['denom']
        after_contract[denom] = await _get_balance_by_denom(contract_address, denom, lcd_url)
        after_sender[denom] = await _get_balance_by_denom(sender_address, denom, lcd_url)
    contract_deltas: Dict[str, int] = {}
    sender_deltas: Dict[str, int] = {}
    for coin in attached_funds:
        denom = coin['denom']
        sent_amount = int(coin['amount'])
        if before_contract_balances is not None:
            before_c = int(before_contract_balances.get(denom, 0))
            delta_c = after_contract[denom] - before_c
            contract_deltas[denom] = delta_c
            if delta_c < sent_amount:
                raise FundsVerificationError(f'Contract balance for {denom} increased by {delta_c}, expected at least {sent_amount}.')
        if before_sender_balances is not None:
            before_s = int(before_sender_balances.get(denom, 0))
            delta_s = after_sender[denom] - before_s
            sender_deltas[denom] = delta_s
            if delta_s > -sent_amount:
                raise FundsVerificationError(f'Sender balance for {denom} decreased by {-delta_s}, expected at least {sent_amount}.')
    return {'after_contract_balances': after_contract, 'after_sender_balances': after_sender, 'contract_deltas': contract_deltas, 'sender_deltas': sender_deltas}

# [cite_start]Configure the junod CLI to use node NODE_URL and chain-id uni-6 [cite: 19]
def check_junod_installed() -> Dict[str, str]:
    try:
        result = subprocess.run(['junod', 'version', '--long'], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError('The junod binary was not found on the host system.')
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'junod version --long failed with exit code {exc.returncode}: {exc.stderr.strip()}')
    return {'installed': True, 'output': result.stdout.strip()}
def set_junod_node(node_url: str) -> Dict[str, str]:
    if not node_url or not isinstance(node_url, str):
        raise ValueError('node_url must be a non-empty string')
    try:
        result = subprocess.run(['junod', 'config', 'node', node_url], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError('The junod binary was not found. Run check_junod_installed() first and ensure junod is on the PATH.')
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'junod config node {node_url} failed with exit code {exc.returncode}: {exc.stderr.strip()}')
    return {'node_url': node_url, 'stdout': result.stdout.strip()}
def set_junod_chain_id(chain_id: str = 'uni-6') -> Dict[str, str]:
    if not chain_id or not isinstance(chain_id, str):
        raise ValueError('chain_id must be a non-empty string')
    try:
        result = subprocess.run(['junod', 'config', 'chain-id', chain_id], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError('The junod binary was not found. Run check_junod_installed() first and ensure junod is on the PATH.')
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'junod config chain-id {chain_id} failed with exit code {exc.returncode}: {exc.stderr.strip()}')
    return {'chain_id': chain_id, 'stdout': result.stdout.strip()}
def set_junod_output_json() -> Dict[str, str]:
    try:
        result = subprocess.run(['junod', 'config', 'output', 'json'], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError('The junod binary was not found. Run check_junod_installed() first and ensure junod is on the PATH.')
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'junod config output json failed with exit code {exc.returncode}: {exc.stderr.strip()}')
    return {'output_format': 'json', 'stdout': result.stdout.strip()}
def test_junod_connectivity(node_url: str, expected_chain_id: str = 'uni-6') -> Dict[str, Any]:
    if not node_url or not isinstance(node_url, str):
        raise ValueError('node_url must be a non-empty string')
    try:
        result = subprocess.run(['junod', 'status', '--node', node_url], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError('The junod binary was not found. Run check_junod_installed() first and ensure junod is on the PATH.')
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'junod status --node {node_url} failed with exit code {exc.returncode}: {exc.stderr.strip()}')
    raw = result.stdout.strip()
    try:
        status_json = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'Failed to decode junod status output as json: {exc}. Raw output was: {raw}')
    network = _extract_network_field(status_json)
    if not network:
        raise RuntimeError('Could not locate node_info.network or equivalent field in junod status output.')
    if network != expected_chain_id:
        raise RuntimeError(f'Connected node reports chain-id {network}, but expected {expected_chain_id}.')
    return {'ok': True, 'node_url': node_url, 'reported_chain_id': network, 'raw_status': status_json}

# [cite_start]Look up a Juno transaction by hash and extract the CosmWasm code_id from its events. [cite: 20]
def validate_tx_hash(tx_hash: str) -> str:
    if not isinstance(tx_hash, str):
        raise TxValidationError("Transaction hash must be a string.")
    normalized = tx_hash.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", normalized):
        raise TxValidationError("Invalid transaction hash format. Expected a 64-character hex string.")
    return normalized.lower()
def fetch_tx_by_hash(tx_hash: str) -> Dict[str, Any]:
    normalized_hash = validate_tx_hash(tx_hash)
    url = f"{LCD_BASE_URL}/cosmos/tx/v1beta1/txs/{normalized_hash}"
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        raise LcdRequestError(f"Failed to reach LCD endpoint: {exc}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise LcdRequestError(f"LCD returned non-JSON response with status {response.status_code}.") from exc
    return {"status_code": response.status_code, "data": payload}
def check_tx_found_and_success(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    class TxQueryError(RuntimeError): pass
    if status_code != 200:
        message = payload.get("message") if isinstance(payload, dict) else None
        raise TxQueryError(f"LCD query failed with HTTP status {status_code}. Message: {message or payload}")
    tx_response = payload.get("tx_response") if isinstance(payload, dict) else None
    if tx_response is None:
        raise TxQueryError("LCD response missing 'tx_response' field.")
    code = tx_response.get("code")
    raw_log = tx_response.get("raw_log", "")
    if code is None:
        raise TxQueryError("'tx_response.code' is missing in LCD response.")
    if code != 0:
        raise TxQueryError(f"Transaction execution failed with code {code}. raw_log: {raw_log}")
    return tx_response
def extract_code_id_from_events(tx_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    logs = tx_response.get("logs") or []
    if not isinstance(logs, list): return candidates
    for log_index, log in enumerate(logs):
        events = log.get("events") or []
        if not isinstance(events, list): continue
        for event_index, event in enumerate(events):
            event_type = event.get("type")
            if event_type not in ("store_code", "instantiate"): continue
            attributes = event.get("attributes") or []
            if not isinstance(attributes, list): continue
            for attribute_index, attr in enumerate(attributes):
                key = attr.get("key")
                if key != "code_id": continue
                value = attr.get("value")
                if value is None: continue
                candidates.append({"code_id": str(value), "event_type": event_type, "log_index": log_index, "event_index": event_index, "attribute_index": attribute_index})
    return candidates
def fallback_parse_raw_log_for_code_id(tx_response: Dict[str, Any]) -> Optional[str]:
    raw_log = tx_response.get("raw_log", "")
    if not raw_log or not isinstance(raw_log, str): return None
    pattern = re.compile(r"code_id[\"']?\s*[:=]\s*\"?(\d+)\"?", re.IGNORECASE)
    match = pattern.search(raw_log)
    if not match: return None
    return match.group(1)
def return_code_id(event_candidates: List[Dict[str, Any]], fallback_code_id: Optional[str]) -> Dict[str, Any]:
    unique_ids = list({c["code_id"] for c in event_candidates})
    if unique_ids:
        if len(unique_ids) == 1:
            return {"code_id": unique_ids[0], "source": "events"}
        else:
            return {"code_ids": unique_ids, "source": "events", "details": event_candidates}
    if fallback_code_id:
        return {"code_id": fallback_code_id, "source": "raw_log"}
    class CodeIdNotFoundError(RuntimeError): pass
    raise CodeIdNotFoundError("This transaction does not include a CosmWasm store_code or instantiate event exposing a code_id.")

# [cite_start]Compile all workspace contracts with workspace-optimizer [cite: 21]
def detect_workspace_root(start_dir: Optional[str] = None) -> dict:
    if start_dir is None:
        current = Path.cwd().resolve()
    else:
        current = Path(start_dir).expanduser().resolve()
    for directory in [current] + list(current.parents):
        cargo_toml = directory / 'Cargo.toml'
        if not cargo_toml.is_file(): continue
        try: data = _load_toml(cargo_toml)
        except Exception as exc: raise RuntimeError(f'Failed to parse {cargo_toml}: {exc}') from exc
        if 'workspace' in data: return {'workspace_root': str(directory)}
    raise WorkspaceNotFoundError(f'No Cargo workspace root found starting from {current}')
def list_workspace_members(workspace_root: Optional[str] = None) -> dict:
    root = _resolve_workspace_root(workspace_root)
    cargo_toml = root / 'Cargo.toml'
    try: data = _load_toml(cargo_toml)
    except Exception as exc: raise RuntimeError(f'Failed to parse root Cargo.toml at {cargo_toml}: {exc}') from exc
    workspace = data.get('workspace')
    if not isinstance(workspace, dict): raise WorkspaceMemberError('Root Cargo.toml does not contain a valid [workspace] table.')
    member_patterns = workspace.get('members', [])
    if not isinstance(member_patterns, list) or not member_patterns: raise WorkspaceMemberError('Workspace has no members defined under [workspace].')
    valid_members: List[Dict[str, str]] = []
    invalid_members: List[Dict[str, str]] = []
    for pattern in member_patterns:
        if not isinstance(pattern, str):
            invalid_members.append({'pattern': repr(pattern), 'reason': 'Workspace member pattern is not a string.'})
            continue
        matches = list(root.glob(pattern))
        if not matches:
            invalid_members.append({'pattern': pattern, 'reason': 'Workspace member pattern did not match any paths.'})
            continue
        for member_dir in matches:
            if not member_dir.is_dir():
                invalid_members.append({'path': str(member_dir), 'reason': 'Matched workspace member is not a directory.'})
                continue
            member_cargo = member_dir / 'Cargo.toml'
            if not member_cargo.is_file():
                invalid_members.append({'path': str(member_dir), 'reason': 'Workspace member is missing Cargo.toml.'})
                continue
            try: member_data = _load_toml(member_cargo)
            except Exception as exc:
                invalid_members.append({'path': str(member_dir), 'reason': f'Failed to parse Cargo.toml: {exc}'})
                continue
            pkg = member_data.get('package', {})
            name = pkg.get('name', member_dir.name)
            lib_cfg = member_data.get('lib')
            is_valid_contract = False
            failure_reason = ''
            if isinstance(lib_cfg, dict):
                crate_type = lib_cfg.get('crate-type') or lib_cfg.get('crate_type')
                if isinstance(crate_type, list) and 'cdylib' in crate_type: is_valid_contract = True
                else: failure_reason = 'lib.crate-type must contain cdylib for CosmWasm.'
            else: failure_reason = 'Missing [lib] section in contract Cargo.toml.'
            if is_valid_contract: valid_members.append({'name': name, 'path': str(member_dir)})
            else: invalid_members.append({'name': name, 'path': str(member_dir), 'reason': failure_reason or 'Invalid CosmWasm contract configuration.'})
    return {'workspace_root': str(root), 'valid_members': valid_members, 'invalid_members': invalid_members}
def run_workspace_optimizer(workspace_root: Optional[str] = None, optimizer_image: str = 'cosmwasm/workspace-optimizer:0.13.0', timeout: int = 1800) -> dict:
    root = _resolve_workspace_root(workspace_root)
    try:
        docker_check = subprocess.run(['docker', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError as exc: raise OptimizerError('Docker is not installed or not available in PATH.') from exc
    except Exception as exc: raise OptimizerError(f'Failed to run docker --version: {exc}') from exc
    if docker_check.returncode != 0: raise OptimizerError(f'Docker is not available: {docker_check.stderr.strip()}')
    cmd = ['docker', 'run', '--rm', '-v', f'{str(root)}:/code', '--mount', f'type=volume,source={root.name}_cache,target=/code/target', '--mount', 'type=volume,source=registry_cache,target=/usr/local/cargo/registry', optimizer_image]
    try:
        completed = subprocess.run(cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc: raise OptimizerError(f'workspace-optimizer timed out after {timeout} seconds.') from exc
    except Exception as exc: raise OptimizerError(f'workspace-optimizer failed to start: {exc}') from exc
    if completed.returncode != 0: raise OptimizerError(f'workspace-optimizer failed with exit code {completed.returncode}: {completed.stderr.strip()}')
    return {'workspace_root': str(root), 'optimizer_image': optimizer_image, 'stdout': completed.stdout, 'stderr': completed.stderr, 'return_code': completed.returncode}
def collect_and_verify_wasm_outputs(workspace_root: Optional[str] = None, members: Optional[List[Dict[str, Any]]] = None, artifacts_dir_name: str = 'artifacts') -> dict:
    root = _resolve_workspace_root(workspace_root)
    artifacts_dir = root / artifacts_dir_name
    if not artifacts_dir.is_dir(): raise ArtifactVerificationError(f'Artifacts directory does not exist: {artifacts_dir}')
    if members is None: members = list_workspace_members(str(root)).get('valid_members', [])
    verified: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for member in members:
        name = member.get('name')
        if not name: failed.append({'member': member, 'reason': 'Missing contract name in member metadata.'}); continue
        artifact_path = artifacts_dir / f'{name}.wasm'
        if not artifact_path.is_file(): failed.append({'member': member, 'artifact_path': str(artifact_path), 'reason': 'Wasm artifact file is missing.'}); continue
        if not _is_valid_wasm(artifact_path): failed.append({'member': member, 'artifact_path': str(artifact_path), 'reason': 'Wasm artifact failed basic validation (magic header or size).'}); continue
        verified.append({'member': member, 'artifact_path': str(artifact_path), 'size_bytes': artifact_path.stat().st_size})
    return {'workspace_root': str(root), 'artifacts_dir': str(artifacts_dir), 'verified_contracts': verified, 'failed_contracts': failed}

# [cite_start]Query a CosmWasm smart contract on Juno via REST using a base64-encoded smart query. [cite: 22]
def validate_contract_address_format(contract_address: str) -> str:
    pattern = re.compile(r'^juno1[0-9a-z]{38}$')
    if contract_address is None: raise ContractAddressValidationError('Contract address is required.')
    contract_address = contract_address.strip()
    if not contract_address: raise ContractAddressValidationError('Contract address cannot be empty or whitespace.')
    if not pattern.fullmatch(contract_address): raise ContractAddressValidationError('Invalid Juno contract address format. Expected a bech32 address starting with prefix juno1.')
    return contract_address
def build_query_json_string(query: Union[str, dict]) -> str:
    try:
        if isinstance(query, str): parsed = json.loads(query)
        elif isinstance(query, dict): parsed = query
        else: raise QueryBuildError('Query must be a dict or JSON string.')
    except (json.JSONDecodeError, TypeError) as exc:
        raise QueryBuildError(f'Invalid query JSON: {exc}') from exc
    return json.dumps(parsed, separators=(',', ':'), ensure_ascii=False)
def encode_query_to_base64(json_query: str) -> str:
    if not isinstance(json_query, str) or not json_query: raise QueryEncodingError('JSON query must be a non-empty string.')
    try:
        query_bytes = json_query.encode('utf-8')
        b64_bytes = base64.b64encode(query_bytes)
        b64_str = b64_bytes.decode('ascii')
        return urllib.parse.quote(b64_str, safe='')
    except Exception as exc:
        raise QueryEncodingError(f'Failed to encode query to base64: {exc}') from exc
async def http_get_lcd_smart_query(contract_address: str, encoded_query: str) -> dict:
    if not contract_address or not encoded_query: raise SmartQueryHttpError('Both contract_address and encoded_query are required.')
    url = f'{LCD_BASE_URL}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{encoded_query}'
    headers = {'Accept': 'application/json'}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
    except httpx.RequestError as exc:
        raise SmartQueryHttpError(f'Network error while querying LCD: {exc}') from exc
    if response.status_code != 200:
        body_text = response.text
        raise SmartQueryHttpError(f'LCD smart query failed with status {response.status_code}: {body_text}')
    try: return response.json()
    except ValueError as exc: raise SmartQueryHttpError(f'Failed to parse LCD JSON response: {exc}') from exc
def parse_lcd_smart_query_response(response_json: Dict[str, Any]) -> str:
    if response_json is None: raise SmartQueryParseError('LCD response JSON is required.')
    if not isinstance(response_json, dict): raise SmartQueryParseError('LCD response JSON must be a dict.')
    if 'data' not in response_json: raise SmartQueryParseError('LCD smart query response missing required field data.')
    data_field = response_json['data']
    if not isinstance(data_field, str) or not data_field: raise SmartQueryParseError('LCD smart query data field must be a non-empty base64 string.')
    return data_field
def decode_contract_response_data(data_b64: str) -> Any:
    if not isinstance(data_b64, str) or not data_b64: raise ContractResponseDecodeError('data_b64 must be a non-empty base64 string.')
    try: raw_bytes = base64.b64decode(data_b64)
    except Exception as exc: raise ContractResponseDecodeError(f'Failed to base64-decode contract data: {exc}') from exc
    try: text = raw_bytes.decode('utf-8')
    except UnicodeDecodeError as exc: raise ContractResponseDecodeError(f'Contract data is not valid UTF-8: {exc}') from exc
    try: return json.loads(text)
    except json.JSONDecodeError as exc: raise ContractResponseDecodeError(f'Contract data is not valid JSON: {exc}') from exc

# [cite_start]Upload the compiled CosmWasm wasm file artifacts/CONTRACT_NAME.wasm to the Juno chain [cite: 23, 24]
def read_and_validate_wasm_artifact(contract_name: str, artifacts_dir: str = "artifacts") -> bytes:
    WASM_MAGIC = b"\x00asm"
    wasm_path = Path(artifacts_dir) / f"{contract_name}.wasm"
    if not wasm_path.exists() or not wasm_path.is_file(): raise WasmArtifactError(f"Wasm artifact not found at {wasm_path!s}")
    try: wasm_bytes = wasm_path.read_bytes()
    except OSError as exc: raise WasmArtifactError(f"Failed to read wasm artifact: {exc}") from exc
    if not wasm_bytes: raise WasmArtifactError(f"Wasm artifact {wasm_path!s} is empty")
    if wasm_bytes[:4] != WASM_MAGIC: raise WasmArtifactError(f"Invalid wasm artifact {wasm_path!s}: missing magic bytes 0x00 0x61 0x73 0x6d")
    return wasm_bytes
def compute_wasm_checksum(wasm_bytes: bytes) -> WasmChecksum:
    if not isinstance(wasm_bytes, (bytes, bytearray)) or not wasm_bytes: raise ValueError("wasm_bytes must be non-empty bytes")
    sha = hashlib.sha256()
    sha.update(wasm_bytes)
    digest = sha.digest()
    hex_digest = sha.hexdigest()
    return WasmChecksum(digest=digest, hex=hex_digest)
async def get_chain_and_account_info(address: str, lcd_url: str = LCD_BASE_URL) -> ChainAccountInfo:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            node_resp = await client.get(f"{lcd_url}/cosmos/base/tendermint/v1beta1/node_info")
            node_resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ChainQueryError(f"Failed to fetch node_info: {exc}") from exc
        node_data = node_resp.json()
        try:
            chain_id = node_data["default_node_info"]["network"]
        except (KeyError, TypeError) as exc:
            raise ChainQueryError(f"Unexpected node_info format: {node_data}") from exc
        try:
            acct_resp = await client.get(f"{lcd_url}/cosmos/auth/v1beta1/accounts/{address}")
            acct_resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ChainQueryError(f"Failed to fetch account info: {exc}") from exc
        acct_data = acct_resp.json()
        try:
            any_account = acct_data["account"]
            value_b64 = any_account["value"]
        except (KeyError, TypeError) as exc:
            raise ChainQueryError(f"Unexpected account response format: {acct_data}") from exc
        try:
            raw_bytes = base64.b64decode(value_b64)
            base_account = BaseAccountProto()
            base_account.ParseFromString(raw_bytes)
        except Exception as exc:
            raise ChainQueryError(f"Failed to decode BaseAccount from Any.value: {exc}") from exc
        return ChainAccountInfo(chain_id=chain_id, account_number=int(base_account.account_number), sequence=int(base_account.sequence))
def construct_store_code_tx(sender_address: str, wasm_bytes: bytes, gas_limit: int = 2_000_000, fee_amount: str = "500000", fee_denom: str = "ujuno", memo: str = "store wasm code") -> Tuple[TxProto, str]:
    if not wasm_bytes: raise ValueError("wasm_bytes must be non-empty")
    msg = MsgStoreCodeProto(sender=sender_address, wasm_byte_code=wasm_bytes)
    msg_any = AnyProto(type_url="/cosmwasm.wasm.v1.MsgStoreCode", value=msg.SerializeToString())
    tx_body = TxBodyProto(messages=[msg_any], memo=memo)
    fee_coin = CoinProto(denom=fee_denom, amount=fee_amount)
    fee = FeeProto(amount=[fee_coin], gas_limit=gas_limit)
    auth_info = AuthInfoProto(fee=fee)
    tx = TxProto(body=tx_body, auth_info=auth_info, signatures=[])
    tx_bytes = tx.SerializeToString()
    tx_bytes_b64 = base64.b64encode(tx_bytes).decode()
    return tx, tx_bytes_b64
async def simulate_and_update_fee(tx: TxProto, lcd_url: str = LCD_BASE_URL, gas_adjustment: float = 1.3, gas_price_ujuno: Decimal = Decimal("0.075"), fee_denom: str = "ujuno") -> Tuple[TxProto, int, int]:
    tx_bytes = tx.SerializeToString()
    tx_b64 = base64.b64encode(tx_bytes).decode()
    payload = {"tx_bytes": tx_b64}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(f"{lcd_url}/cosmos/tx/v1beta1/simulate", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SimulationError(f"Simulation HTTP error: {exc}") from exc
    data = resp.json()
    try:
        gas_used_str = data["gas_info"]["gas_used"]
        gas_used = int(gas_used_str)
    except (KeyError, TypeError, ValueError) as exc:
        raise SimulationError(f"Unexpected simulate response: {data}") from exc
    gas_limit = math.ceil(gas_used * gas_adjustment)
    fee_amount_int = int((Decimal(gas_limit) * gas_price_ujuno).to_integral_value(rounding=ROUND_UP))
    fee_coin = CoinProto(denom=fee_denom, amount=str(fee_amount_int))
    if not tx.auth_info.fee: tx.auth_info.fee.CopyFrom(FeeProto())
    tx.auth_info.fee.gas_limit = gas_limit
    tx.auth_info.fee.amount.clear()
    tx.auth_info.fee.amount.append(fee_coin)
    return tx, gas_used, gas_limit
def sign_store_code_tx_upload(tx: TxProto, chain_id: str, account_number: int, sequence: int, private_key_hex: str) -> str:
    try:
        priv_key_bytes = bytes.fromhex(private_key_hex)
        priv_key = PrivateKey(priv_key_bytes)
        pubkey_bytes = priv_key.public_key.bytes
    except Exception as exc: raise SigningError(f"Invalid private key: {exc}") from exc
    proto_pubkey = Secp256k1PubKeyProto(key=pubkey_bytes)
    pubkey_any = AnyProto(type_url="/cosmos.crypto.secp256k1.PubKey", value=proto_pubkey.SerializeToString())
    mode_info = TxProto.ModeInfo(single=TxProto.ModeInfo.Single(mode=SignModeProto.SIGN_MODE_DIRECT))
    signer_info = SignerInfoProto(public_key=pubkey_any, mode_info=mode_info, sequence=int(sequence))
    if not tx.auth_info: tx.auth_info.CopyFrom(AuthInfoProto())
    tx.auth_info.signer_infos.clear()
    tx.auth_info.signer_infos.append(signer_info)
    sign_doc = SignDocProto(body_bytes=tx.body.SerializeToString(), auth_info_bytes=tx.auth_info.SerializeToString(), chain_id=chain_id, account_number=int(account_number))
    sign_doc_bytes = sign_doc.SerializeToString()
    try: signature = priv_key.sign(sign_doc_bytes)
    except Exception as exc: raise SigningError(f"Failed to sign SignDoc: {exc}") from exc
    tx.signatures.clear()
    tx.signatures.append(signature)
    tx_bytes = tx.SerializeToString()
    return base64.b64encode(tx_bytes).decode()
async def broadcast_signed_tx(signed_tx_base64: str, lcd_url: str = LCD_BASE_URL, mode: str = "BROADCAST_MODE_BLOCK") -> dict:
    payload = {"tx_bytes": signed_tx_base64, "mode": mode}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(f"{lcd_url}/cosmos/tx/v1beta1/txs", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BroadcastError(f"Broadcast HTTP error: {exc}") from exc
    data = resp.json()
    try:
        tx_response = data["tx_response"]
        _ = tx_response["txhash"]
        _ = tx_response["code"]
    except (KeyError, TypeError) as exc:
        raise BroadcastError(f"Unexpected broadcast response format: {data}") from exc
    return tx_response
async def fetch_tx_and_extract_code_id(txhash: str, lcd_url: str = LCD_BASE_URL) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{lcd_url}/cosmos/tx/v1beta1/txs/{txhash}")
        resp.raise_for_status()
    data = resp.json()
    tx_response = data.get("tx_response", {})
    logs = tx_response.get("logs", [])
    for log in logs:
        for event in log.get("events", []):
            ev_type = event.get("type", "")
            if ev_type not in ("store_code", "wasm"): continue
            for attr in event.get("attributes", []):
                if attr.get("key") == "code_id":
                    code_id = attr.get("value")
                    if code_id: return code_id
    raise CodeIdNotFoundError(f"No code_id attribute found in tx events for txhash={txhash}. Response: {tx_response}")
async def verify_uploaded_code_hash(code_id: str, local_checksum_bytes: bytes, lcd_url: str = LCD_BASE_URL) -> bool:
    if not local_checksum_bytes: raise ValueError("local_checksum_bytes must be non-empty")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{lcd_url}/cosmwasm/wasm/v1/code/{code_id}")
        resp.raise_for_status()
    data = resp.json()
    try:
        code_info = data["code_info"]
        data_hash_b64 = code_info["data_hash"]
    except (KeyError, TypeError) as exc:
        raise CodeHashMismatchError(f"Unexpected code info response: {data}") from exc
    try:
        onchain_hash = base64.b64decode(data_hash_b64)
    except Exception as exc:
        raise CodeHashMismatchError(f"Failed to decode on-chain data_hash: {exc}") from exc
    if onchain_hash != local_checksum_bytes:
        raise CodeHashMismatchError(f"On-chain data_hash ({onchain_hash.hex()}) does not match local checksum ({local_checksum_bytes.hex()})")
    return True

# [cite_start]Enable cw-orch integration by adding an 'interface' feature flag in Cargo.toml for a CosmWasm contract. [cite: 25]
def load_cargo_toml_orch(path: str = 'Cargo.toml'):
    cargo_path = Path(path)
    if not cargo_path.is_file(): raise FileNotFoundError(f'Cargo.toml not found at: {cargo_path}')
    try: content = cargo_path.read_text(encoding='utf-8')
    except OSError as exc: raise RuntimeError(f'Failed to read {cargo_path}: {exc}') from exc
    try: doc = parse(content)
    except Exception as exc: raise ValueError(f'Failed to parse {cargo_path} as TOML: {exc}') from exc
    return doc
def ensure_cw_orch_dependency(doc, version: str):
    if not isinstance(doc, MutableMapping): raise TypeError('Expected a TOML document mapping for doc')
    if 'dependencies' not in doc or doc['dependencies'] is None: doc['dependencies'] = table()
    deps = doc['dependencies']
    if not isinstance(deps, MutableMapping): raise TypeError('The [dependencies] section is not a table')
    if 'cw-orch' not in deps: deps['cw-orch'] = {'version': version, 'optional': True}
    return doc
def ensure_features_table_exists(doc):
    if not isinstance(doc, MutableMapping): raise TypeError('Expected a TOML document mapping for doc')
    if 'features' not in doc or doc['features'] is None: doc['features'] = table()
    else:
        features = doc['features']
        if not isinstance(features, MutableMapping): raise TypeError('The [features] section exists but is not a table')
    return doc
def add_interface_feature_flag(doc):
    if not isinstance(doc, MutableMapping): raise TypeError('Expected a TOML document mapping for doc')
    if 'features' not in doc or doc['features'] is None: raise KeyError('No [features] table found; call ensure_features_table_exists first')
    features = doc['features']
    if not isinstance(features, MutableMapping): raise TypeError('The [features] section is not a table')
    if 'interface' not in features or features['interface'] is None:
        vals = array()
        vals.append('dep:cw-orch')
        features['interface'] = vals
    else:
        vals = features['interface']
        if not isinstance(vals, MutableSequence): raise TypeError('features.interface must be an array of feature flags')
        existing = [str(v) for v in vals]
        if 'dep:cw-orch' not in existing: vals.append('dep:cw-orch')
    return doc
def ensure_default_includes_interface(doc):
    if not isinstance(doc, MutableMapping): raise TypeError('Expected a TOML document mapping for doc')
    if 'features' not in doc or doc['features'] is None: raise KeyError('No [features] table found; call ensure_features_table_exists first')
    features = doc['features']
    if not isinstance(features, MutableMapping): raise TypeError('The [features] section is not a table')
    if 'default' not in features or features['default'] is None:
        vals = array()
        vals.append('interface')
        features['default'] = vals
    else:
        vals = features['default']
        if not isinstance(vals, MutableSequence): raise TypeError('features.default must be an array of feature flags')
        existing = [str(v) for v in vals]
        if 'interface' not in existing: vals.append('interface')
    return doc
def write_cargo_toml(doc, path: str = 'Cargo.toml'):
    cargo_path = Path(path)
    try:
        content = dumps(doc)
        cargo_path.write_text(content, encoding='utf-8')
    except OSError as exc: raise RuntimeError(f'Failed to write {cargo_path}: {exc}') from exc
    return str(cargo_path)
def cargo_check_with_interface_feature(project_root: str = '.'):
    root = Path(project_root)
    if not root.is_dir(): raise NotADirectoryError(f'Not a directory: {root}')
    cmd = ['cargo', 'check', '--features', 'interface']
    try:
        result = subprocess.run(cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError as exc: raise RuntimeError('cargo executable not found; ensure Rust is installed and on PATH') from exc
    except Exception as exc: raise RuntimeError(f'Failed to run cargo check: {exc}') from exc
    if result.returncode != 0: raise RuntimeError(f'cargo check --features interface failed with exit code {result.returncode}. STDOUT: {result.stdout} STDERR: {result.stderr}')
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}

# [cite_start]Query the CW721 all_tokens list from a given NFT contract on Juno using CosmWasm smart queries. [cite: 26, 27]
def validate_contract_address(contract_address: str) -> str:
    if not isinstance(contract_address, str) or not contract_address: raise InvalidContractAddress("Contract address must be a non-empty string.")
    hrp, data = bech32_decode(contract_address)
    if hrp != JUNO_HRP or data is None: raise InvalidContractAddress(f"Invalid Juno bech32 address: {contract_address}")
    return contract_address
def initialize_pagination_state(page_limit: int = 100) -> PaginationState:
    if not isinstance(page_limit, int) or page_limit <= 0: raise ValueError("page_limit must be a positive integer.")
    return PaginationState(page_limit=page_limit)
def build_all_tokens_query_json(page_limit: int, start_after: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(page_limit, int) or page_limit <= 0: raise ValueError("page_limit must be a positive integer.")
    all_tokens: Dict[str, Any] = {"limit": page_limit}
    if start_after: all_tokens["start_after"] = start_after
    return {"all_tokens": all_tokens}
def encode_query_to_base64(query: Dict[str, Any]) -> str:
    try: json_bytes = json.dumps(query, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as e: raise ValueError(f"Failed to serialize query to JSON: {e}") from e
    encoded = base64.urlsafe_b64encode(json_bytes).decode("ascii")
    return encoded
async def lcd_smart_query_all_tokens(contract_address: str, query_data_b64: str, timeout: float = 10.0) -> Dict[str, Any]:
    url = f"{LCD_BASE_URL}/cosmwasm/wasm/v1/contract/{contract_address}/smart/{query_data_b64}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"LCD smart query failed with status {e.response.status_code}: {e.response.text}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Network error while calling LCD: {e}") from e
    try: data = response.json()
    except ValueError as e: raise RuntimeError(f"Failed to parse LCD response as JSON: {e}") from e
    if "data" not in data: raise RuntimeError(f"LCD response missing 'data' field: {data}")
    return data
def decode_all_tokens_response(lcd_response: Dict[str, Any]) -> List[str]:
    if "data" not in lcd_response: raise KeyError("LCD response dict does not contain 'data' field.")
    b64_data = lcd_response["data"]
    if not isinstance(b64_data, str): raise TypeError("'data' field must be a base64-encoded string.")
    try:
        try: raw_bytes = base64.b64decode(b64_data)
        except Exception: raw_bytes = base64.urlsafe_b64decode(b64_data)
    except Exception as e: raise ValueError(f"Failed to base64-decode response 'data': {e}") from e
    try: decoded_json = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e: raise ValueError(f"Failed to decode response JSON: {e}") from e
    if not isinstance(decoded_json, dict): raise TypeError(f"Decoded response JSON is not an object: {decoded_json}")
    tokens = decoded_json.get("tokens")
    if tokens is None: raise KeyError(f"Decoded response does not contain 'tokens' field: {decoded_json}")
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens): raise TypeError(f"'tokens' must be a list of strings, got: {tokens}")
    return tokens
async def fetch_all_cw721_token_ids(contract_address: str, page_limit: int = 100) -> List[str]:
    validate_contract_address(contract_address)
    state = initialize_pagination_state(page_limit=page_limit)
    while True:
        query_json = build_all_tokens_query_json(page_limit=state.page_limit, start_after=state.start_after)
        query_b64 = encode_query_to_base64(query_json)
        lcd_response = await lcd_smart_query_all_tokens(contract_address=contract_address, query_data_b64=query_b64)
        page_tokens = decode_all_tokens_response(lcd_response)
        if not page_tokens: break
        state.token_ids.extend(page_tokens)
        if len(page_tokens) < state.page_limit: break
        state.start_after = page_tokens[-1]
    return state.token_ids

# [cite_start]Claim JUNOX test tokens from the Juno faucet for a given address and verify receipt on-chain [cite: 28]
def validate_juno_address_faucet(address: str, expected_prefix: str = "juno") -> Dict[str, str]:
    result = {"is_valid": "false", "error": ""}
    if not isinstance(address, str) or not address: result["error"] = "Address must be a non-empty string."; return result
    if len(address) < 10 or len(address) > 90: result["error"] = "Address length is not within the expected range."; return result
    try: hrp, data = bech32_decode(address)
    except Exception as e: result["error"] = f"Invalid bech32 encoding: {e}"; return result
    if hrp is None or data is None: result["error"] = "Invalid bech32 address or checksum."; return result
    if hrp != expected_prefix: result["error"] = f"Invalid address prefix: expected '{expected_prefix}', got '{hrp}'."; return result
    if len(data) == 0: result["error"] = "Bech32 data part is empty; address is malformed."; return result
    result["is_valid"] = "true"; result["error"] = ""; return result
async def query_junox_balance_before_faucet(address: str, denom: str = "ujunox", lcd_base: str = LCD_BASE_URL) -> Dict:
    if not address: raise ValueError("Address is required to query balance.")
    url = f"{lcd_base}/cosmos/bank/v1beta1/balances/{address}/by_denom"
    params = {"denom": denom}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
    except httpx.RequestError as e: raise RuntimeError(f"Error querying Juno LCD: {e}") from e
    if resp.status_code != 200: raise RuntimeError(f"LCD returned HTTP {resp.status_code} when querying balance: {resp.text}")
    data = resp.json()
    balance = data.get("balance")
    if balance is None: amount_str = "0"; resp_denom = denom
    else: amount_str = balance.get("amount", "0"); resp_denom = balance.get("denom", denom)
    if resp_denom != denom: raise RuntimeError(f"Unexpected denom in LCD response: {resp_denom} (expected {denom})")
    try: amount_int = int(amount_str)
    except ValueError as e: raise RuntimeError(f"Invalid amount format in LCD response: {amount_str}") from e
    return {"address": address, "denom": denom, "amount": amount_int}
async def poll_faucet_tx_until_final(tx_hash: str, lcd_base: str = LCD_BASE_URL, timeout_seconds: float = 60.0, poll_interval_seconds: float = 2.0) -> Dict[str, Any]:
    if not tx_hash: raise ValueError("tx_hash is required to poll transaction status.")
    url = f"{lcd_base}/cosmos/tx/v1beta1/txs/{tx_hash}"
    end_time = asyncio.get_event_loop().time() + timeout_seconds
    last_response: Any = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try: resp = await client.get(url)
            except httpx.RequestError as e: last_response = {"error": f"Request error: {e}"}
            else:
                if resp.status_code == 200:
                    data = resp.json(); last_response = data; tx_resp = data.get("tx_response")
                    if tx_resp is not None:
                        try: code = int(tx_resp.get("code", 0))
                        except (TypeError, ValueError): code = 0
                        if code == 0: return {"status": "success", "response": data}
                        else: return {"status": "failed", "response": data}
                else: last_response = {"http_status": resp.status_code, "body": resp.text}
            if asyncio.get_event_loop().time() >= end_time: return {"status": "timeout", "response": last_response}
            await asyncio.sleep(poll_interval_seconds)
async def _query_junox_balance(address: str, denom: str = "ujunox", lcd_base: str = LCD_BASE_URL) -> int:
    url = f"{lcd_base}/cosmos/bank/v1beta1/balances/{address}/by_denom"; params = {"denom": denom}
    async with httpx.AsyncClient(timeout=10.0) as client: resp = await client.get(url, params=params)
    if resp.status_code != 200: raise RuntimeError(f"LCD returned HTTP {resp.status_code} when querying balance: {resp.text}")
    data = resp.json(); balance = data.get("balance")
    if balance is None: amount_str = "0"; resp_denom = denom
    else: amount_str = balance.get("amount", "0"); resp_denom = balance.get("denom", denom)
    if resp_denom != denom: raise RuntimeError(f"Unexpected denom in LCD response: {resp_denom} (expected {denom})")
    try: return int(amount_str)
    except ValueError as e: raise RuntimeError(f"Invalid amount format in LCD response: {amount_str}") from e
async def compare_junox_balance_after_faucet(address: str, pre_faucet_amount: int, denom: str = "ujunox", lcd_base: str = LCD_BASE_URL) -> Dict:
    if pre_faucet_amount is None: raise ValueError("pre_faucet_amount must be provided (from step 2).")
    post_amount = await _query_junox_balance(address, denom=denom, lcd_base=lcd_base)
    delta = post_amount - int(pre_faucet_amount)
    return {"address": address, "denom": denom, "pre_amount": int(pre_faucet_amount), "post_amount": post_amount, "delta": delta}

# [cite_start]Store the returned code id in shell variable CODE_ID [cite: 29]
def retrieve_last_tx_output(raw_output: str) -> dict:
    if not raw_output or not raw_output.strip(): raise ValueError('raw_output is empty; provide the JSON string from the last transaction.')
    try: tx_output = json.loads(raw_output)
    except json.JSONDecodeError as exc: raise ValueError(f'Failed to parse transaction JSON output: {exc}') from exc
    if not isinstance(tx_output, dict): raise ValueError('Parsed transaction output is not a JSON object (dict).')
    return tx_output
def parse_code_id_from_output(tx_output: dict) -> str:
    if not isinstance(tx_output, dict): raise ValueError('tx_output must be a dictionary.')
    code_id = None
    if 'code_id' in tx_output: code_id = tx_output['code_id']
    if not code_id:
        tx_response = tx_output.get('tx_response') or tx_output.get('result') or {}
        logs = tx_response.get('logs') or []
        for log in logs:
            events = log.get('events') or []
            for event in events:
                if event.get('type') == 'store_code':
                    for attr in event.get('attributes') or []:
                        key = attr.get('key')
                        if key in ('code_id', 'codeID'):
                            code_id = attr.get('value'); break
                if code_id: break
            if code_id: break
    if code_id is None: raise ValueError('Could not find code_id in transaction output; ensure this is a wasm store transaction JSON.')
    code_id_str = str(code_id).strip()
    if not code_id_str.isdigit(): raise ValueError(f'Extracted code_id {code_id_str} is not purely numeric.')
    return code_id_str
def export_shell_variable_CODE_ID(code_id: str) -> str:
    if not code_id or not str(code_id).strip().isdigit(): raise ValueError('code_id must be a non-empty numeric string.')
    sanitized = str(code_id).strip()
    return f'export CODE_ID={sanitized}'

# [cite_start]Create a new Juno wallet named "MyWalletName" by generating a mnemonic and deriving a Juno-compatible address. [cite: 30]
def select_juno_chain_key_params() -> Dict[str, Any]:
    return {"chain_id": "juno-1", "bip44_coin_type": 118, "derivation_path": "m/44'/118'/0'/0/0", "bech32_prefix": "juno"}
def generate_bip39_mnemonic(words: int = 24) -> str:
    if words not in (12, 24): raise ValueError("words must be either 12 or 24.")
    words_num = Bip39WordsNum.WORDS_NUM_12 if words == 12 else Bip39WordsNum.WORDS_NUM_24
    try: mnemonic = Bip39MnemonicGenerator().FromWordsNumber(words_num)
    except Exception as exc: raise RuntimeError("Failed to generate mnemonic using BIP-39 generator.") from exc
    return str(mnemonic)
def derive_hd_key_from_mnemonic(mnemonic: str, derivation_path: str = "m/44'/118'/0'/0/0") -> bytes:
    if not isinstance(mnemonic, str) or not mnemonic.strip(): raise ValueError("mnemonic must be a non-empty string.")
    if not isinstance(derivation_path, str) or not derivation_path.startswith("m/"): raise ValueError("derivation_path must be a BIP-32 style string starting with 'm/'.")
    try: seed_bytes = Bip39SeedGenerator(mnemonic).Generate()
    except Exception as exc: raise ValueError("Failed to convert mnemonic to seed; is the mnemonic valid BIP-39?") from exc
    try:
        root = Bip32Slip10Secp256k1.FromSeed(seed_bytes)
        derived = root.DerivePath(derivation_path)
        private_key_bytes = derived.PrivateKey().Raw().ToBytes()
    except Exception as exc: raise RuntimeError(f"Failed to derive private key for path {derivation_path}.") from exc
    if len(private_key_bytes) != 32: raise RuntimeError("Derived private key is not 32 bytes; derivation may be incorrect.")
    return private_key_bytes
def derive_public_key_and_address(private_key: bytes, bech32_prefix: str = "juno") -> Dict[str, Any]:
    if not isinstance(private_key, (bytes, bytearray)): raise TypeError("private_key must be bytes.")
    if len(private_key) != 32: raise ValueError("private_key must be exactly 32 bytes for secp256k1.")
    try:
        sk = SigningKey.from_string(bytes(private_key), curve=SECP256k1)
        try: public_key_bytes = sk.get_verifying_key().to_string("compressed")
        except TypeError:
            vk = sk.get_verifying_key(); uncompressed = vk.to_string(); x_bytes, y_bytes = uncompressed[:32], uncompressed[32:]; prefix = b"\x02" if (y_bytes[-1] % 2 == 0) else b"\x03"; public_key_bytes = prefix + x_bytes
    except Exception as exc: raise RuntimeError("Failed to derive public key from private key.") from exc
    sha256_digest = hashlib.sha256(public_key_bytes).digest()
    ripemd160 = RIPEMD160.new()
    ripemd160.update(sha256_digest)
    pubkey_hash = ripemd160.digest()
    data5 = convertbits(pubkey_hash, 8, 5, True)
    if data5 is None: raise RuntimeError("Failed to convert pubkey hash to 5-bit words for bech32 encoding.")
    address = bech32_encode(bech32_prefix, data5)
    if address is None: raise RuntimeError("bech32_encode returned None; invalid data or prefix.")
    return {"public_key_bytes": public_key_bytes, "public_key_hex": public_key_bytes.hex(), "address": address}
def persist_wallet_metadata(wallet_name: str, address: str, public_key_hex: str, private_key: bytes) -> Dict[str, Any]:
    WALLET_STORE_PATH = os.environ.get("WALLET_STORE_PATH", "wallet_store.json")
    if not wallet_name: raise ValueError("wallet_name is required.")
    if not address: raise ValueError("address is required.")
    if not isinstance(private_key, (bytes, bytearray)) or len(private_key) != 32: raise ValueError("private_key must be a 32-byte value.")
    fernet = _get_fernet()
    encrypted_private_key = fernet.encrypt(bytes(private_key)).decode("utf-8")
    record: Dict[str, Any] = {"wallet_name": wallet_name, "address": address, "public_key_hex": public_key_hex, "encrypted_private_key": encrypted_private_key}
    try:
        if os.path.exists(WALLET_STORE_PATH):
            with open(WALLET_STORE_PATH, "r", encoding="utf-8") as f: store = json.load(f)
        else: store = {}
    except (OSError, json.JSONDecodeError) as exc: raise RuntimeError(f"Failed to load wallet store from {WALLET_STORE_PATH}.") from exc
    if address in store: raise RuntimeError(f"A wallet with address {address} already exists in the keyring.")
    store[address] = record
    try:
        with open(WALLET_STORE_PATH, "w", encoding="utf-8") as f: json.dump(store, f, indent=2)
    except OSError as exc: raise RuntimeError(f"Failed to persist wallet metadata to {WALLET_STORE_PATH}.") from exc
    return record
def optional_verify_address_on_chain(expected_network: str = "juno-1") -> Dict[str, Any]:
    url = f"{LCD_BASE_URL}/cosmos/base/tendermint/v1beta1/node_info"
    try:
        response = requests.get(url, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc: raise RuntimeError(f"Failed to reach Juno LCD at {url}: {exc}") from exc
    try: data = response.json()
    except ValueError as exc: raise RuntimeError("LCD node_info response was not valid JSON.") from exc
    default_info = data.get("default_node_info", {}) or {}
    network = default_info.get("network")
    matches_expected = expected_network is None or network == expected_network
    return {"raw": data, "network": network, "matches_expected": matches_expected}

# [cite_start]Set TXFLAGS environment variable with gas settings [cite: 31]
def get_minimum_gas_price():
    url = f'{LCD_BASE_URL}/cosmos/base/node/v1beta1/config'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc: raise LCDQueryError(f'Failed to query node config: {exc}') from exc
    data = response.json()
    minimum_gas_price = data.get('minimum_gas_price')
    if not minimum_gas_price: raise LCDQueryError('minimum_gas_price field missing in LCD response')
    first_entry = minimum_gas_price.split(',')[0].strip()
    split_index = 0
    while split_index < len(first_entry) and (first_entry[split_index].isdigit() or first_entry[split_index] == '.'): split_index += 1
    if split_index == 0 or split_index == len(first_entry): raise LCDQueryError(f'Unexpected minimum_gas_price format: {first_entry!r}')
    amount = first_entry[:split_index]; denom = first_entry[split_index:]
    return {'raw': minimum_gas_price, 'amount': amount, 'denom': denom}
def get_chain_id():
    node_info_url = f'{LCD_BASE_URL}/cosmos/base/tendermint/v1beta1/node_info'
    try:
        response = requests.get(node_info_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc: raise LCDQueryError(f'Failed to query node_info: {exc}') from exc
    data = response.json()
    default_node_info = data.get('default_node_info') or {}
    chain_id = default_node_info.get('network')
    if chain_id: return chain_id
    latest_block_url = f'{LCD_BASE_URL}/cosmos/base/tendermint/v1beta1/blocks/latest'
    try:
        response = requests.get(latest_block_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc: raise LCDQueryError(f'Failed to query latest block for chain-id fallback: {exc}') from exc
    block_data = response.json()
    header = (block_data.get('block') or {}).get('header') or {}
    chain_id = header.get('chain_id')
    if not chain_id: raise LCDQueryError('Could not determine chain-id from LCD responses')
    return chain_id
def build_txflags_string(minimum_gas_price: str, chain_id: str, gas_adjustment: float = 1.3, rpc_endpoint: str = 'https://rpc.junonetwork.io') -> str:
    if not minimum_gas_price or not chain_id: raise ValueError('minimum_gas_price and chain_id must be non-empty')
    if gas_adjustment <= 0: raise ValueError('gas_adjustment must be positive')
    flags = (f'--gas=auto --gas-adjustment={gas_adjustment} --gas-prices={minimum_gas_price} --chain-id={chain_id} --node={rpc_endpoint}')
    return flags
def set_txflags_env(txflags: str) -> None:
    if not txflags or not txflags.strip(): raise ValueError('txflags must be a non-empty string')
    os.environ['TXFLAGS'] = txflags
def test_junod_tx_with_txflags(from_identifier: str, amount: str = '1', denom: str = 'ujuno', simulate_only: bool = True, home: Optional[str] = None) -> Dict[str, Any]:
    import shlex
    txflags = os.environ.get('TXFLAGS')
    if not txflags: raise JunodTxTestError('TXFLAGS environment variable is not set. Call set_txflags_env(...) first.')
    base_cmd = (f'junod tx bank send {from_identifier} {from_identifier} {amount}{denom} --from {from_identifier} --yes')
    if simulate_only: base_cmd += ' --dry-run'
    if home: base_cmd += f' --home {home}'
    full_cmd = f'{base_cmd} {txflags}'
    try:
        completed = subprocess.run(shlex.split(full_cmd), capture_output=True, text=True, check=False)
    except OSError as exc: raise JunodTxTestError(f'Failed to execute junod command: {exc}') from exc
    result: Dict[str, Any] = {'command': full_cmd, 'stdout': completed.stdout, 'stderr': completed.stderr, 'returncode': completed.returncode}
    if completed.returncode != 0: raise JunodTxTestError(f'junod tx command failed with code {completed.returncode}: {completed.stderr or completed.stdout}')
    return result

# [cite_start]Check the balance of a given wallet address on Juno [cite: 32]
def validate_juno_address(address: str) -> str:
    if not isinstance(address, str): raise InvalidJunoAddressError('Address must be a string.')
    address = address.strip()
    if not address: raise InvalidJunoAddressError('Address must not be empty.')
    if not address.startswith('juno1'): raise InvalidJunoAddressError('Address must start with prefix juno1.')
    hrp, data = bech32_decode(address)
    if hrp != 'juno' or data is None: raise InvalidJunoAddressError('Invalid Juno bech32 address: bad checksum or wrong prefix.')
    return address
async def fetch_all_balances_balance(address: str) -> List[Dict[str, Any]]:
    url = f'{LCD_BASE_URL}/cosmos/bank/v1beta1/balances/{address}'
    balances: List[Dict[str, Any]] = []; next_key: str | None = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            params: Dict[str, Any] = {};
            if next_key: params['pagination.key'] = next_key
            try:
                response = await client.get(url, params=params); response.raise_for_status()
            except httpx.HTTPError as exc: raise RuntimeError(f'Error querying Juno LCD balances: {exc}') from exc
            data = response.json() or {}; balances.extend(data.get('balances', []))
            pagination = data.get('pagination') or {}; next_key = pagination.get('next_key')
            if not next_key: break
    return balances
async def fetch_spendable_balances_balance(address: str) -> List[Dict[str, Any]]:
    url = f'{LCD_BASE_URL}/cosmos/bank/v1beta1/spendable_balances/{address}'
    balances: List[Dict[str, Any]] = []; next_key: str | None = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            params: Dict[str, Any] = {};
            if next_key: params['pagination.key'] = next_key
            try:
                response = await client.get(url, params=params); response.raise_for_status()
            except httpx.HTTPError as exc: raise RuntimeError(f'Error querying Juno LCD spendable balances: {exc}') from exc
            data = response.json() or {}; balances.extend(data.get('balances', []))
            pagination = data.get('pagination') or {}; next_key = pagination.get('next_key')
            if not next_key: break
    return balances
async def fetch_denoms_metadata() -> List[Dict[str, Any]]:
    url = f'{LCD_BASE_URL}/cosmos/bank/v1beta1/denoms_metadata'
    metadatas: List[Dict[str, Any]] = []; next_key: str | None = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            params: Dict[str, Any] = {};
            if next_key: params['pagination.key'] = next_key
            try:
                response = await client.get(url, params=params); response.raise_for_status()
            except httpx.HTTPError as exc: raise RuntimeError(f'Error querying Juno LCD denoms metadata: {exc}') from exc
            data = response.json() or {}; metadatas.extend(data.get('metadatas', []))
            pagination = data.get('pagination') or {}; next_key = pagination.get('next_key')
            if not next_key: break
    return metadatas
def format_balances(all_balances: List[Dict[str, Any]], spendable_balances: Optional[List[Dict[str, Any]]], metadata_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    spendable_map: Dict[str, str] = {}
    if spendable_balances:
        for coin in spendable_balances:
            denom = coin.get('denom'); amount = coin.get('amount')
            if denom is not None and amount is not None: spendable_map[denom] = amount
    result: List[Dict[str, Any]] = []
    for coin in all_balances:
        denom = coin.get('denom'); amount_str = coin.get('amount')
        if denom is None or amount_str is None: continue
        meta = metadata_index.get(denom); display_denom = denom; exponent = 0
        if meta:
            display_denom = meta.get('display') or denom
            for unit in meta.get('denom_units', []):
                if unit.get('denom') == display_denom:
                    try: exponent = int(unit.get('exponent', 0))
                    except (TypeError, ValueError): exponent = 0
                    break
        try: base_amount = Decimal(amount_str)
        except (InvalidOperation, TypeError): base_amount = Decimal(0)
        if exponent > 0: factor = Decimal(10) ** exponent; display_amount = base_amount / factor
        else: display_amount = base_amount
        spendable_amount_str = spendable_map.get(denom); spendable_entry: Optional[Dict[str, Any]] = None
        if spendable_amount_str is not None:
            try: spendable_base = Decimal(spendable_amount_str)
            except (InvalidOperation, TypeError): spendable_base = Decimal(0)
            if exponent > 0: factor = Decimal(10) ** exponent; spendable_display = spendable_base / factor
            else: spendable_display = spendable_base
            spendable_entry = {'amount': spendable_amount_str, 'display_amount': decimal_to_str(spendable_display)}
        result.append({'base_denom': denom, 'display_denom': display_denom, 'exponent': exponent, 'total': {'amount': amount_str, 'display_amount': decimal_to_str(display_amount)}, 'spendable': spendable_entry})
    result.sort(key=lambda entry: entry['base_denom'])
    return result