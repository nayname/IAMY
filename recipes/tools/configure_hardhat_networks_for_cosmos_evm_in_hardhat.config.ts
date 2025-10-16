{
    "tools": [
        {
            "step": 1,
            "label": "backend",
            "introduction": "Reads the specified configuration file and returns its lines so they can be edited.",
            "function": "open_config_file(config_path=None)",
            "usage": "lines = open_config_file(\"hardhat.config.ts\")"
        },
        {
            "step": 2,
            "label": "backend",
            "introduction": "Updates (or appends) a configuration block within the file\u2019s lines before the file is saved back to disk.",
            "function": "modify_minimum_gas_prices(lines, new_value=\"0stake\")",
            "usage": "updated_lines = modify_minimum_gas_prices(lines, new_value=\"cosmos_network_entry\")"
        },
        {
            "step": 3,
            "label": "backend",
            "introduction": "Runs an npm install command that pulls in Hardhat, the toolbox and other dev-dependencies.",
            "function": "install()",
            "usage": "install()"
        },
        {
            "step": 4,
            "label": "backend",
            "introduction": "Invokes Hardhat\u2019s compile task to ensure the workspace builds successfully.",
            "function": "compile()",
            "usage": "compile()"
        },
        {
            "step": 5,
            "label": "backend",
            "introduction": "Checks that the configured RPC endpoint is reachable, effectively \u2018pinging\u2019 the Cosmos EVM network.",
            "function": "connect_rpc_endpoint(rpc_endpoint='https://rpc-kralum.neutron.org')",
            "usage": "connect_rpc_endpoint(process.env.COSMOS_RPC or \"https://rpc.cosmos.network\")"
        }
    ],
    "frontend": [],
    "backend": [
        "lines = open_config_file(\"hardhat.config.ts\")#step: 1 Tool: open_file Desciption: Open hardhat.config.ts so the Cosmos EVM network definition can be added.",
        "updated_lines = modify_minimum_gas_prices(lines, new_value=\"cosmos_network_entry\")#step: 2 Tool: update_file Desciption: Inside the exported HardhatUserConfig object, append a new entry under networks:\n\nnetworks: {\n  cosmos: {\n    url: process.env.COSMOS_RPC || \"https://rpc.cosmos.network\",   // RPC endpoint for the target Cosmos-SDK EVM chain\n    chainId: Number(process.env.COSMOS_CHAIN_ID) || 118,           // Replace with the chain\u2019s EVM chainId\n    accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []\n  },\n  /* existing networks \u2026 */\n}",
        "install()#step: 3 Tool: install_npm_packages Desciption: Install any missing dependencies used in the config (e.g. dotenv and Hardhat toolbox):\n  npm i -D @nomicfoundation/hardhat-toolbox dotenv",
        "compile()#step: 4 Tool: run_command Desciption: Compile the workspace to confirm the config is syntactically correct:\n  npx hardhat compile",
        "connect_rpc_endpoint(process.env.COSMOS_RPC or \"https://rpc.cosmos.network\")#step: 5 Tool: run_command Desciption: Ping the configured network to ensure Hardhat can reach it:\n  npx hardhat --network cosmos block-number"
    ],
    "intent": "Configure Hardhat networks for Cosmos EVM in hardhat.config.ts",
    "workflow": [
        {
            "step": 1,
            "tool": "open_file",
            "description": "Open hardhat.config.ts so the Cosmos EVM network definition can be added."
        },
        {
            "step": 2,
            "tool": "update_file",
            "description": "Inside the exported HardhatUserConfig object, append a new entry under networks:\n\nnetworks: {\n  cosmos: {\n    url: process.env.COSMOS_RPC || \"https://rpc.cosmos.network\",   // RPC endpoint for the target Cosmos-SDK EVM chain\n    chainId: Number(process.env.COSMOS_CHAIN_ID) || 118,           // Replace with the chain\u2019s EVM chainId\n    accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []\n  },\n  /* existing networks \u2026 */\n}"
        },
        {
            "step": 3,
            "tool": "install_npm_packages",
            "description": "Install any missing dependencies used in the config (e.g. dotenv and Hardhat toolbox):\n  npm i -D @nomicfoundation/hardhat-toolbox dotenv"
        },
        {
            "step": 4,
            "tool": "run_command",
            "description": "Compile the workspace to confirm the config is syntactically correct:\n  npx hardhat compile"
        },
        {
            "step": 5,
            "tool": "run_command",
            "description": "Ping the configured network to ensure Hardhat can reach it:\n  npx hardhat --network cosmos block-number"
        }
    ],
    "outcome_checks": [
        "Compilation finishes with no errors.",
        "The command in step 5 returns a positive block number, confirming connectivity to the Cosmos EVM RPC."
    ]
}