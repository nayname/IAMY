{
    "label": "send",
    "workflow": [
        {
            "step": 1,
            "label": "backend",
            "introduction": "Initialises a FastAPI backend, connects to the Cosmos-EVM RPC endpoint and exposes an endpoint that returns the wallet address derived from the server-side PRIVATE_KEY.",
            "code": "from fastapi import FastAPI, HTTPException\nfrom os import getenv\nfrom web3 import Web3\nfrom web3.middleware import geth_poa_middleware\n\napp = FastAPI()\n\n# --- RPC and Key Configuration ---\nRPC_URL = getenv(\"COSMOS_EVM_RPC\", \"https://rpc.evmos.org:8545\")  # Override in production\nPRIVATE_KEY = getenv(\"PRIVATE_KEY\")  # NEVER commit this value; set as an env var\n\n# --- Web3 Client ---\nw3 = Web3(Web3.HTTPProvider(RPC_URL))\n# Inject PoA middleware (many Cosmos EVM chains like Evmos, Canto, etc. use PoA)\nw3.middleware_onion.inject(geth_poa_middleware, layer=0)\n\n@app.get(\"/api/instantiate_wallet\")\nasync def instantiate_wallet():\n    \"\"\"Creates a wallet instance from PRIVATE_KEY and returns its address.\"\"\"\n    if not PRIVATE_KEY:\n        raise HTTPException(status_code=500, detail=\"PRIVATE_KEY is not set in environment variables\")\n    acct = w3.eth.account.from_key(PRIVATE_KEY)\n    return {\"address\": acct.address}",
            "usage": "const address = await fetch('/api/instantiate_wallet').then(r => r.json()).then(j => j.address);"
        },
        {
            "step": 2,
            "label": "backend",
            "introduction": "Fetches dynamic EIP-1559 fee data (base fee, priority fee, max fee) from the latest block.",
            "code": "from fastapi import HTTPException\n\n@app.get(\"/api/fee_data\")\nasync def fee_data():\n    \"\"\"Returns baseFeePerGas, maxPriorityFeePerGas and a recommended maxFeePerGas.\"\"\"\n    try:\n        latest_block = w3.eth.get_block('latest')\n        base_fee = latest_block['baseFeePerGas']\n        max_priority_fee = w3.eth.max_priority_fee  # Recommended priority tip\n        # Heuristic: maxFeePerGas = baseFee + 2 * maxPriorityFee\n        max_fee_per_gas = base_fee + 2 * max_priority_fee\n        return {\n            \"baseFeePerGas\": base_fee,\n            \"maxPriorityFeePerGas\": max_priority_fee,\n            \"maxFeePerGas\": max_fee_per_gas\n        }\n    except Exception as e:\n        raise HTTPException(status_code=500, detail=str(e))",
            "usage": "const fees = await fetch('/api/fee_data').then(r => r.json());"
        },
        {
            "step": 3,
            "label": "backend",
            "introduction": "Builds a complete EIP-1559 transaction object, ready for signing, by merging user-supplied fields with network fee data and account nonce.",
            "code": "from pydantic import BaseModel\nfrom fastapi import HTTPException\n\nclass BuildTxParams(BaseModel):\n    to: str\n    value: int = 0            # Amount in wei\n    data: str = \"0x\"          # Hex-encoded calldata\n    gasLimit: int = 21000     # Conservative default\n\n@app.post(\"/api/build_tx\")\nasync def build_tx(params: BuildTxParams):\n    \"\"\"Constructs an unsigned EIP-1559 transaction object.\"\"\"\n    try:\n        fee_info = await fee_data()          # Dynamic fee suggestions\n        wallet_info = await instantiate_wallet()\n        sender = wallet_info['address']\n\n        nonce = w3.eth.get_transaction_count(sender)\n        chain_id = w3.eth.chain_id\n\n        tx = {\n            \"to\": params.to,\n            \"value\": params.value,\n            \"data\": params.data,\n            \"gas\": params.gasLimit,\n            \"maxFeePerGas\": fee_info['maxFeePerGas'],\n            \"maxPriorityFeePerGas\": fee_info['maxPriorityFeePerGas'],\n            \"nonce\": nonce,\n            \"chainId\": chain_id,\n            \"type\": 2  # EIP-1559\n        }\n        return tx\n    except Exception as e:\n        raise HTTPException(status_code=500, detail=str(e))",
            "usage": "const txObject = await fetch('/api/build_tx', {\n  method: 'POST',\n  headers: { 'Content-Type': 'application/json' },\n  body: JSON.stringify({ to: '0xDestinationAddress', value: 0, data: '0x', gasLimit: 50000 })\n}).then(r => r.json());"
        },
        {
            "step": 4,
            "label": "backend",
            "introduction": "Signs the built transaction with the backend\u2019s PRIVATE_KEY and broadcasts it to the Cosmos-EVM network.",
            "code": "from pydantic import BaseModel\nfrom fastapi import HTTPException\n\nclass TxObject(BaseModel):\n    to: str\n    value: int\n    data: str\n    gas: int\n    maxFeePerGas: int\n    maxPriorityFeePerGas: int\n    nonce: int\n    chainId: int\n    type: int\n\n@app.post(\"/api/sign_and_send\")\nasync def sign_and_send(tx: TxObject):\n    \"\"\"Signs an EIP-1559 tx using PRIVATE_KEY and broadcasts it, returning the tx hash.\"\"\"\n    if not PRIVATE_KEY:\n        raise HTTPException(status_code=500, detail=\"Server missing PRIVATE_KEY\")\n    try:\n        signed_tx = w3.eth.account.sign_transaction(tx.dict(), private_key=PRIVATE_KEY)\n        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)\n        return {\"txHash\": tx_hash.hex()}\n    except Exception as e:\n        raise HTTPException(status_code=500, detail=str(e))",
            "usage": "const receipt = await fetch('/api/sign_and_send', {\n  method: 'POST',\n  headers: { 'Content-Type': 'application/json' },\n  body: JSON.stringify(txObject)\n}).then(r => r.json());\nconsole.log('Broadcasted Tx Hash:', receipt.txHash);"
        }
    ]
}