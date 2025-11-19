{
    "label": "others",
    "workflow": [
        {
            "step": 1,
            "label": "backend",
            "introduction": "Create or overwrite hardhat.config.ts with a Cosmos-EVM network definition. This file is read by the Hardhat CLI when you execute any command.",
            "code": "/* hardhat.config.ts */\nimport { HardhatUserConfig } from \"hardhat/config\";\nimport \"@nomicfoundation/hardhat-toolbox\"; // Hardhat plugins & ethers helpers\nimport * as dotenv from \"dotenv\";\n\ndotenv.config(); // Loads .env variables into process.env\n\n/*\n * Replace the solidity version or any other default fields to fit your\n * project. Only the networks.cosmos section is strictly required for\n * this workflow.\n */\nconst config: HardhatUserConfig = {\n  solidity: \"0.8.17\",\n  networks: {\n    /*\n     * Cosmos-SDK EVM compatible chain.\n     * - url:   Public or private RPC endpoint\n     * - chainId: Integer EVM chain-id (not the Cosmos-SDK chain-id)\n     * - accounts: List of private keys that Hardhat will use to sign txs.\n     */\n    cosmos: {\n      url: process.env.COSMOS_RPC || \"https://rpc.cosmos.network\",\n      chainId: Number(process.env.COSMOS_CHAIN_ID) || 118,\n      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []\n    }\n    // ...existing networks can stay untouched\n  }\n};\n\nexport default config;",
            "usage": "Save the file, or copy the snippet into the existing hardhat.config.ts so the `cosmos` network becomes available to Hardhat."
        },
        {
            "step": 2,
            "label": "backend",
            "introduction": "Install any missing development dependencies declared in the configuration (dotenv & Hardhat toolbox).",
            "code": "#!/usr/bin/env bash\n# install_deps.sh \u2013 run once\nset -euo pipefail\n\n# Installs peer-dependencies as devDependencies (-D)\n# so they don't ship with production builds.\n\nnpm i -D @nomicfoundation/hardhat-toolbox dotenv",
            "usage": "chmod +x install_deps.sh && ./install_deps.sh"
        },
        {
            "step": 3,
            "label": "backend",
            "introduction": "Compile the project to verify that the updated configuration is syntactically correct and that your contracts still build.",
            "code": "#!/usr/bin/env bash\n# compile.sh\nset -euo pipefail\n\nnpx hardhat compile",
            "usage": "./compile.sh"
        },
        {
            "step": 4,
            "label": "backend",
            "introduction": "Create a small helper script that queries the latest block number from the configured Cosmos network. This proves Hardhat can reach the RPC endpoint.",
            "code": "/* scripts/blockNumber.ts */\nimport { ethers } from \"hardhat\";\n\nasync function main() {\n  try {\n    // ethers.provider is automatically configured to use the selected network\n    const blockNumber = await ethers.provider.getBlockNumber();\n    console.log(`Current block number on '${ethers.provider.network.name}':`, blockNumber);\n  } catch (err) {\n    console.error(\"Failed to fetch block number:\", err);\n    process.exit(1);\n  }\n}\n\nmain();",
            "usage": "npx hardhat run --network cosmos scripts/blockNumber.ts"
        },
        {
            "step": 5,
            "label": "backend",
            "introduction": "Optional health-check bash wrapper that compiles and then pings the Cosmos network in one go. It can be wired into CI pipelines.",
            "code": "#!/usr/bin/env bash\n# healthcheck.sh \u2013 exits non-zero if either compilation or RPC call fail\nset -euo pipefail\n\necho \"[1/2] \u25b6 Compiling workspace\" && npx hardhat compile\n\necho \"[2/2] \u25b6 Pinging Cosmos RPC for latest block number\" && npx hardhat run --network cosmos scripts/blockNumber.ts",
            "usage": "chmod +x healthcheck.sh && ./healthcheck.sh"
        }
    ]
}