{
    "tools": [
        {
            "step": 1,
            "label": "frontend",
            "introduction": "connectEthWallet opens the user\u2019s MetaMask (or compatible) extension and returns the active account once the wallet is connected to the EVM RPC.",
            "function": "connectEthWallet()",
            "usage": "const { account } = await connectEthWallet();"
        },
        {
            "step": 2,
            "label": "backend",
            "introduction": "json_rpc_call performs a raw JSON-RPC request (e.g. eth_feeHistory) so you can retrieve the current EIP-1559 fee suggestions from the node.",
            "function": "json_rpc_call(method, params, endpoint)",
            "usage": "const feeData = json_rpc_call(\"eth_feeHistory\", [5, \"latest\", [50]]);"
        },
        {
            "step": 3,
            "label": "frontend",
            "introduction": "undef",
            "function": "undef",
            "usage": "undef"
        },
        {
            "step": 4,
            "label": "frontend",
            "introduction": "approveErc20Spend shows how to sign and broadcast an Ethereum transaction through window.ethereum.request('eth_sendTransaction'), mirroring wallet.sendTransaction for EIP-1559 flows.",
            "function": "approveErc20Spend({ ownerAddress, bridgeAddress, amountSats })",
            "usage": "const txHash = await approveErc20Spend({ ownerAddress, bridgeAddress, amountSats: \"100000000\" });"
        }
    ],
    "frontend": [
        "const { account } = await connectEthWallet();//step: 1 Tool: instantiate_wallet Desciption: Create an `ethers.Wallet` (or `JsonRpcSigner`) connected to the desired Cosmos EVM RPC endpoint.",
        "undef//step: 3 Tool: build_tx_object Desciption: Construct the transaction object including `to`, `value`, `data`, `gasLimit`, plus EIP-1559 fields `maxFeePerGas` and `maxPriorityFeePerGas`.",
        "const txHash = await approveErc20Spend({ ownerAddress, bridgeAddress, amountSats: \"100000000\" });//step: 4 Tool: sign_and_send_tx Desciption: Use `wallet.sendTransaction(txObject)` to sign and broadcast the EIP-1559 transaction."
    ],
    "backend": [
        "const feeData = json_rpc_call(\"eth_feeHistory\", [5, \"latest\", [50]]);#step: 2 Tool: fetch_fee_data Desciption: Call `provider.getFeeData()` (or `provider.send('eth_feeHistory', ...)`) to obtain current `maxPriorityFeePerGas` and `maxFeePerGas` suggestions."
    ],
    "intent": "Configure EIP-1559 gas parameters when sending a transaction with ethers.js",
    "workflow": [
        {
            "step": 1,
            "tool": "instantiate_wallet",
            "description": "Create an `ethers.Wallet` (or `JsonRpcSigner`) connected to the desired Cosmos EVM RPC endpoint."
        },
        {
            "step": 2,
            "tool": "fetch_fee_data",
            "description": "Call `provider.getFeeData()` (or `provider.send('eth_feeHistory', ...)`) to obtain current `maxPriorityFeePerGas` and `maxFeePerGas` suggestions."
        },
        {
            "step": 3,
            "tool": "build_tx_object",
            "description": "Construct the transaction object including `to`, `value`, `data`, `gasLimit`, plus EIP-1559 fields `maxFeePerGas` and `maxPriorityFeePerGas`."
        },
        {
            "step": 4,
            "tool": "sign_and_send_tx",
            "description": "Use `wallet.sendTransaction(txObject)` to sign and broadcast the EIP-1559 transaction."
        }
    ],
    "outcome_checks": [
        "Verify the transaction\u2019s `type` is `0x2` (EIP-1559) in a block explorer.",
        "Ensure the transaction was mined without being dropped for insufficient fees."
    ]
}