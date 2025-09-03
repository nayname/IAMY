# ===================================================================================
# == Imports
# ===================================================================================
import json
import subprocess
import time
import base64
from datetime import datetime
from typing import Dict, Any, List

import requests
from cosmpy.aerial.client import LedgerClient, NetworkConfig
from cosmpy.aerial.exceptions import CosmPyException
from cosmpy.aerial.tx import Transaction
from cosmpy.aerial.wallet import LocalWallet
# Note: In a real project, these proto imports would point to your generated files.
# from neutron_proto.cron import MsgAddSchedule
# from cosmos_proto.cosmos.gov.v1 import MsgSubmitProposal, TextProposal
# from google.protobuf.any_pb2 import Any as Any_pb

# ===================================================================================
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

def query_cron_params(chain_id: str = "neutron-1", node: str = "https://rpc-kralum.neutron.org:443") -> dict:
    """Fetches the current Cron module parameters via CLI."""
    proc = subprocess.run(
        ["neutrond", "query", "cron", "params", "--chain-id", chain_id, "--node", node, "--output", "json"],
        capture_output=True, text=True, check=True
    )
    return json.loads(proc.stdout).get("params", {})

def query_cron_schedule(schedule_name: str, node: str = "https://rpc.neutron.org:443") -> Dict:
    """Fetch schedule metadata from the Neutron Cron module via `neutrond` CLI."""
    try:
        cmd = ["neutrond", "query", "cron", "schedule", schedule_name, "--output", "json", "--node", node]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to query schedule '{schedule_name}': {exc.stderr.strip()}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Received non-JSON response from neutrond CLI") from exc

def query_all_cron_schedules(limit: int = 1000) -> List[Dict]:
    """Return every cron schedule on-chain, handling pagination."""
    schedules: List[Dict] = []
    next_key: str = ""
    while True:
        cmd = ["neutrond", "query", "cron", "schedules", "--limit", str(limit), "--output", "json"]
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

