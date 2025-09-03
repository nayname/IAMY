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


# step:1 file: Query the cron schedule named "daily_maintenance"
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


# step:2 file: Query the cron schedule named "daily_maintenance"
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


# step:1 file: Show details for the cron schedule "protocol_update"
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


