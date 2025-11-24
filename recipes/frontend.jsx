// step:1 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* scripts/ensureSchemas.js */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

/**
 * Ensure that CosmWasm JSON schema files exist for a contract.
 *
 * - Runs `cargo schema` in the contract directory if the schema folder is missing or empty.
 * - Verifies that execute, query, and instantiate schema files are present.
 *
 * @param {string} contractRootDir - Path to the contract's root directory (where Cargo.toml lives).
 * @param {string} [schemaDirName='schema'] - Name of the directory containing JSON schemas.
 */
function ensureContractSchemas(contractRootDir, schemaDirName = 'schema') {
  const schemaDir = path.join(contractRootDir, schemaDirName);

  try {
    // Run `cargo schema` if schema directory doesn't exist or is empty.
    const schemaDirExists = fs.existsSync(schemaDir);
    const schemaFiles = schemaDirExists ? fs.readdirSync(schemaDir) : [];

    if (!schemaDirExists || schemaFiles.length === 0) {
      console.log(`[ensureSchemas] Running 'cargo schema' in ${contractRootDir}...`);
      execSync('cargo schema', {
        cwd: contractRootDir,
        stdio: 'inherit',
      });
    } else {
      console.log(`[ensureSchemas] Found existing schema directory at ${schemaDir}.`);
    }

    // Re-read directory after possible generation.
    const finalSchemaFiles = fs.readdirSync(schemaDir);

    const requiredSchemas = [
      'instantiate_msg.json',
      'execute_msg.json',
      'query_msg.json',
    ];

    const missing = requiredSchemas.filter(
      (file) => !finalSchemaFiles.includes(file),
    );

    if (missing.length > 0) {
      throw new Error(
        `[ensureSchemas] Missing expected schema files in ${schemaDir}: ${missing.join(', ')}.`,
      );
    }

    console.log(
      `[ensureSchemas] All required schema files present in ${schemaDir}.`,
    );
    return { schemaDir, files: finalSchemaFiles };
  } catch (error) {
    console.error('[ensureSchemas] Failed to ensure contract schemas:', error);
    throw error;
  }
}

// Allow running as a standalone CLI script.
if (require.main === module) {
  const contractRootDir = process.argv[2] || process.cwd();
  const schemaDirName = process.argv[3] || 'schema';

  try {
    const result = ensureContractSchemas(contractRootDir, schemaDirName);
    console.log(
      `[ensureSchemas] Success. Schema directory: ${result.schemaDir}`,
    );
  } catch (error) {
    process.exit(1);
  }
}

module.exports = { ensureContractSchemas };



// step:2 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* ts-codegen.config.ts */

// Minimal contract list configuration for @cosmwasm/ts-codegen.
const contracts = [
  {
    // Human-readable name for your contract. Used in generated class/type names.
    name: 'MyContract',
    // Relative path to the directory containing JSON schema files.
    dir: './contracts/my-contract/schema',
  },
];

/**
 * Main ts-codegen configuration object.
 * The CLI will import the default export from this file.
 */
const config = {
  contracts,
  // Directory where generated TypeScript code will be written.
  outPath: './src/contracts',
};

export default config;

/**
 * Optional: shared Juno network configuration used by tests or application code.
 * This is not consumed by ts-codegen directly, but you can import it in
 * your own scripts (for example, the smoke test in step 6).
 */
export const junoNetwork = {
  chainId: 'juno-1',
  // Use the documented LCD archive endpoint for Juno.
  lcdEndpoint:
    process.env.JUNO_LCD_ENDPOINT || 'https://lcd-archive.junonetwork.io',
  // RPC endpoint is intentionally left to be provided via environment.
  // See https://cosmos.directory/juno/nodes for public nodes or run your own.
  rpcEndpoint: process.env.JUNO_RPC_ENDPOINT || 'http://localhost:26657',
};



// step:3 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* scripts/installTsCodegen.js */

const { execSync } = require('child_process');

/**
 * Installs @cosmwasm/ts-codegen as a devDependency using npm or yarn.
 *
 * By default this script tries `npm`, and falls back to `yarn` if npm fails.
 */
function installTsCodegen(version = 'latest') {
  try {
    console.log(
      `[installTsCodegen] Installing @cosmwasm/ts-codegen@${version} as devDependency...`,
    );
    execSync(`npm install --save-dev @cosmwasm/ts-codegen@${version}`, {
      stdio: 'inherit',
    });
    console.log('[installTsCodegen] Installation via npm completed.');
  } catch (npmError) {
    console.warn(
      '[installTsCodegen] npm install failed, trying yarn add --dev...',
    );
    try {
      execSync(`yarn add --dev @cosmwasm/ts-codegen@${version}`, {
        stdio: 'inherit',
      });
      console.log('[installTsCodegen] Installation via yarn completed.');
    } catch (yarnError) {
      console.error(
        '[installTsCodegen] Failed to install @cosmwasm/ts-codegen with npm and yarn.',
      );
      console.error('npm error:', npmError.message);
      console.error('yarn error:', yarnError.message);
      throw yarnError;
    }
  }
}

// Allow running as a standalone CLI script.
if (require.main === module) {
  const version = process.argv[2] || 'latest';
  try {
    installTsCodegen(version);
  } catch (error) {
    process.exit(1);
  }
}

module.exports = { installTsCodegen };



// step:4 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* scripts/generateTsClient.js */

const { execSync } = require('child_process');
const path = require('path');

/**
 * Runs the @cosmwasm/ts-codegen CLI with the provided config file.
 *
 * @param {string} [configPath='ts-codegen.config.ts'] - Path to the ts-codegen config file.
 */
function generateTsClient(configPath = 'ts-codegen.config.ts') {
  const resolvedConfigPath = path.resolve(configPath);

  try {
    console.log(
      `[generateTsClient] Generating TypeScript client using config: ${resolvedConfigPath}`,
    );
    // Use npx so that the local devDependency CLI is picked up automatically.
    execSync(
      `npx @cosmwasm/ts-codegen generate-ts --config "${resolvedConfigPath}"`,
      {
        stdio: 'inherit',
      },
    );
    console.log(
      '[generateTsClient] ts-codegen completed successfully. Check the configured outPath for generated files.',
    );
  } catch (error) {
    console.error(
      '[generateTsClient] ts-codegen generation failed. See output above for details.',
    );
    throw error;
  }
}

// Allow running as a standalone CLI script.
if (require.main === module) {
  const configPath = process.argv[2] || 'ts-codegen.config.ts';
  try {
    generateTsClient(configPath);
  } catch (error) {
    process.exit(1);
  }
}

module.exports = { generateTsClient };



// step:5 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* scripts/typecheckGeneratedClient.js */

const { execSync } = require('child_process');

/**
 * Runs the TypeScript compiler in "noEmit" mode to type-check the project,
 * including the generated client code.
 *
 * @param {string[]} [extraArgs=[]] - Extra arguments passed to `tsc`.
 */
function typecheckGeneratedClient(extraArgs = []) {
  const baseCmd = ['npx', 'tsc', '--noEmit', ...extraArgs].join(' ');
  try {
    console.log(`[typecheckGeneratedClient] Running: ${baseCmd}`);
    execSync(baseCmd, { stdio: 'inherit' });
    console.log('[typecheckGeneratedClient] Type-check completed successfully.');
  } catch (error) {
    console.error(
      '[typecheckGeneratedClient] Type-check failed. Fix TypeScript errors above.',
    );
    throw error;
  }
}

// Allow running as a standalone CLI script.
if (require.main === module) {
  const extraArgs = process.argv.slice(2);
  try {
    typecheckGeneratedClient(extraArgs);
  } catch (error) {
    process.exit(1);
  }
}

module.exports = { typecheckGeneratedClient };



// step:6 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* tests/smokeTestMyContract.ts */

import { junoNetwork } from '../ts-codegen.config';
// Adjust the import path and type name based on what ts-codegen generated for your contract.
import type { QueryMsg } from '../src/contracts/MyContract.types';

// Use Node 18+ global fetch or bring your own polyfill (for example, node-fetch).
const fetchFn: typeof fetch = (globalThis as any).fetch;

/**
 * Encodes a JSON query message to the base64-encoded string that the
 * Juno LCD smart-contract query endpoint expects.
 */
function encodeQueryMsgToBase64(msg: QueryMsg): string {
  const json = JSON.stringify(msg);
  return Buffer.from(json, 'utf8').toString('base64');
}

/**
 * Perform a simple smart-contract query against Juno using the LCD endpoint.
 *
 * This function assumes your contract exposes a `config` query; adjust the
 * `queryMsg` shape to match your actual schema if needed.
 */
export async function smokeTestMyContractQuery() {
  const lcdBaseUrl =
    junoNetwork.lcdEndpoint || 'https://lcd-archive.junonetwork.io';
  const contractAddress = process.env.MY_CONTRACT_ADDRESS;

  if (!contractAddress) {
    throw new Error(
      'Missing MY_CONTRACT_ADDRESS environment variable (juno1... contract address).',
    );
  }

  if (!fetchFn) {
    throw new Error(
      'Global fetch is not available. Run this with Node 18+ or provide a fetch polyfill.',
    );
  }

  // Example query message: { config: {} }
  const queryMsg: QueryMsg = {
    // Replace this with the actual shape of your query.
    // For example, if your schema has { "config": {} }, this is correct.
    // @ts-expect-error - will be narrowed once you use the real QueryMsg type.
    config: {},
  };

  const queryDataBase64 = encodeQueryMsgToBase64(queryMsg);
  const url = `${lcdBaseUrl}/cosmwasm/wasm/v1/contract/${contractAddress}/smart/${encodeURIComponent(
    queryDataBase64,
  )}`;

  console.log('[smokeTestMyContractQuery] Querying URL:', url);

  const response = await fetchFn(url);

  if (!response.ok) {
    const bodyText = await response.text();
    throw new Error(
      `[smokeTestMyContractQuery] LCD request failed: ${response.status} ${response.statusText} - ${bodyText}`,
    );
  }

  const body = (await response.json()) as { data: string };

  if (!body || typeof body.data !== 'string') {
    throw new Error(
      `[smokeTestMyContractQuery] Unexpected LCD response shape: ${JSON.stringify(
        body,
      )}`,
    );
  }

  // The `data` field is base64-encoded JSON returned by the contract.
  const decodedJson = Buffer.from(body.data, 'base64').toString('utf8');

  let parsed: unknown;
  try {
    parsed = JSON.parse(decodedJson);
  } catch (e) {
    throw new Error(
      `[smokeTestMyContractQuery] Failed to parse contract response JSON: ${decodedJson}`,
    );
  }

  console.log(
    '[smokeTestMyContractQuery] Contract query successful. Parsed response:',
  );
  // eslint-disable-next-line no-console
  console.dir(parsed, { depth: null });

  return parsed;
}

// Allow running as a standalone script: `npx ts-node tests/smokeTestMyContract.ts`
if (require.main === module) {
  smokeTestMyContractQuery().catch((error) => {
    console.error('[smokeTestMyContractQuery] Error:', error);
    process.exit(1);
  });
}



// step:3 file: claim_junox_test_tokens_from_the_juno_faucet_for_a_given_address_and_verify_receipt_on-chain
export const openJunoFaucetUI = (faucetUrl) => {
  // 'faucetUrl' should be configured in your environment, e.g.
  // process.env.NEXT_PUBLIC_JUNO_UNI6_FAUCET_URL
  if (typeof window === 'undefined') {
    throw new Error('Cannot open faucet UI: window is not available (are you on the server?).');
  }

  if (!faucetUrl || typeof faucetUrl !== 'string') {
    throw new Error('Faucet URL is not configured or is invalid.');
  }

  try {
    const newWindow = window.open(faucetUrl, '_blank', 'noopener,noreferrer');

    if (!newWindow) {
      // Pop-up may be blocked by the browser
      throw new Error('Failed to open the faucet UI. Please allow pop-ups for this site.');
    }
  } catch (err) {
    console.error('Error opening Juno faucet UI:', err);
    throw err;
  }
};



// step:4 file: claim_junox_test_tokens_from_the_juno_faucet_for_a_given_address_and_verify_receipt_on-chain
export const instructUserToRequestFaucetTokens = (address, denomLabel = 'JUNOX') => {
  if (typeof window === 'undefined') {
    throw new Error('Cannot show faucet instructions: window is not available (are you on the server?).');
  }

  if (!address || typeof address !== 'string') {
    throw new Error('A valid Juno address is required before requesting faucet tokens.');
  }

  // This function does not attempt to automate the official faucet UI,
  // because it is a separate web application and its implementation
  // can change. Instead, we give clear instructions to the user.
  const message = [
    'A faucet window/tab should now be open.',
    `1. Paste this address into the faucet address field: ${address}`,
    `2. Select the token type (e.g., ${denomLabel}).`,
    '3. Submit the request and wait for the faucet to confirm or show an error.',
    '4. If you see errors such as rate limiting or daily limits, follow the faucet instructions or try again later.'
  ].join('\n');

  // You can replace this alert with rendering a modal or in-app message.
  window.alert(message);
};



// step:1 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* scripts/ensureSchemas.js */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

/**
 * Ensure that CosmWasm JSON schema files exist for a contract.
 *
 * - Runs `cargo schema` in the contract directory if the schema folder is missing or empty.
 * - Verifies that execute, query, and instantiate schema files are present.
 *
 * @param {string} contractRootDir - Path to the contract's root directory (where Cargo.toml lives).
 * @param {string} [schemaDirName='schema'] - Name of the directory containing JSON schemas.
 */
function ensureContractSchemas(contractRootDir, schemaDirName = 'schema') {
  const schemaDir = path.join(contractRootDir, schemaDirName);

  try {
    // Run `cargo schema` if schema directory doesn't exist or is empty.
    const schemaDirExists = fs.existsSync(schemaDir);
    const schemaFiles = schemaDirExists ? fs.readdirSync(schemaDir) : [];

    if (!schemaDirExists || schemaFiles.length === 0) {
      console.log(`[ensureSchemas] Running 'cargo schema' in ${contractRootDir}...`);
      execSync('cargo schema', {
        cwd: contractRootDir,
        stdio: 'inherit',
      });
    } else {
      console.log(`[ensureSchemas] Found existing schema directory at ${schemaDir}.`);
    }

    // Re-read directory after possible generation.
    const finalSchemaFiles = fs.readdirSync(schemaDir);

    const requiredSchemas = [
      'instantiate_msg.json',
      'execute_msg.json',
      'query_msg.json',
    ];

    const missing = requiredSchemas.filter(
      (file) => !finalSchemaFiles.includes(file),
    );

    if (missing.length > 0) {
      throw new Error(
        `[ensureSchemas] Missing expected schema files in ${schemaDir}: ${missing.join(', ')}.`,
      );
    }

    console.log(
      `[ensureSchemas] All required schema files present in ${schemaDir}.`,
    );
    return { schemaDir, files: finalSchemaFiles };
  } catch (error) {
    console.error('[ensureSchemas] Failed to ensure contract schemas:', error);
    throw error;
  }
}

// Allow running as a standalone CLI script.
if (require.main === module) {
  const contractRootDir = process.argv[2] || process.cwd();
  const schemaDirName = process.argv[3] || 'schema';

  try {
    const result = ensureContractSchemas(contractRootDir, schemaDirName);
    console.log(
      `[ensureSchemas] Success. Schema directory: ${result.schemaDir}`,
    );
  } catch (error) {
    process.exit(1);
  }
}

module.exports = { ensureContractSchemas };



// step:2 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* ts-codegen.config.ts */

// Minimal contract list configuration for @cosmwasm/ts-codegen.
const contracts = [
  {
    // Human-readable name for your contract. Used in generated class/type names.
    name: 'MyContract',
    // Relative path to the directory containing JSON schema files.
    dir: './contracts/my-contract/schema',
  },
];

/**
 * Main ts-codegen configuration object.
 * The CLI will import the default export from this file.
 */
const config = {
  contracts,
  // Directory where generated TypeScript code will be written.
  outPath: './src/contracts',
};

export default config;

/**
 * Optional: shared Juno network configuration used by tests or application code.
 * This is not consumed by ts-codegen directly, but you can import it in
 * your own scripts (for example, the smoke test in step 6).
 */
export const junoNetwork = {
  chainId: 'juno-1',
  // Use the documented LCD archive endpoint for Juno.
  lcdEndpoint:
    process.env.JUNO_LCD_ENDPOINT || 'https://lcd-archive.junonetwork.io',
  // RPC endpoint is intentionally left to be provided via environment.
  // See https://cosmos.directory/juno/nodes for public nodes or run your own.
  rpcEndpoint: process.env.JUNO_RPC_ENDPOINT || 'http://localhost:26657',
};



// step:3 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* scripts/installTsCodegen.js */

const { execSync } = require('child_process');

/**
 * Installs @cosmwasm/ts-codegen as a devDependency using npm or yarn.
 *
 * By default this script tries `npm`, and falls back to `yarn` if npm fails.
 */
function installTsCodegen(version = 'latest') {
  try {
    console.log(
      `[installTsCodegen] Installing @cosmwasm/ts-codegen@${version} as devDependency...`,
    );
    execSync(`npm install --save-dev @cosmwasm/ts-codegen@${version}`, {
      stdio: 'inherit',
    });
    console.log('[installTsCodegen] Installation via npm completed.');
  } catch (npmError) {
    console.warn(
      '[installTsCodegen] npm install failed, trying yarn add --dev...',
    );
    try {
      execSync(`yarn add --dev @cosmwasm/ts-codegen@${version}`, {
        stdio: 'inherit',
      });
      console.log('[installTsCodegen] Installation via yarn completed.');
    } catch (yarnError) {
      console.error(
        '[installTsCodegen] Failed to install @cosmwasm/ts-codegen with npm and yarn.',
      );
      console.error('npm error:', npmError.message);
      console.error('yarn error:', yarnError.message);
      throw yarnError;
    }
  }
}

// Allow running as a standalone CLI script.
if (require.main === module) {
  const version = process.argv[2] || 'latest';
  try {
    installTsCodegen(version);
  } catch (error) {
    process.exit(1);
  }
}

module.exports = { installTsCodegen };



// step:4 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* scripts/generateTsClient.js */

const { execSync } = require('child_process');
const path = require('path');

/**
 * Runs the @cosmwasm/ts-codegen CLI with the provided config file.
 *
 * @param {string} [configPath='ts-codegen.config.ts'] - Path to the ts-codegen config file.
 */
function generateTsClient(configPath = 'ts-codegen.config.ts') {
  const resolvedConfigPath = path.resolve(configPath);

  try {
    console.log(
      `[generateTsClient] Generating TypeScript client using config: ${resolvedConfigPath}`,
    );
    // Use npx so that the local devDependency CLI is picked up automatically.
    execSync(
      `npx @cosmwasm/ts-codegen generate-ts --config "${resolvedConfigPath}"`,
      {
        stdio: 'inherit',
      },
    );
    console.log(
      '[generateTsClient] ts-codegen completed successfully. Check the configured outPath for generated files.',
    );
  } catch (error) {
    console.error(
      '[generateTsClient] ts-codegen generation failed. See output above for details.',
    );
    throw error;
  }
}

// Allow running as a standalone CLI script.
if (require.main === module) {
  const configPath = process.argv[2] || 'ts-codegen.config.ts';
  try {
    generateTsClient(configPath);
  } catch (error) {
    process.exit(1);
  }
}

module.exports = { generateTsClient };



// step:5 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* scripts/typecheckGeneratedClient.js */

const { execSync } = require('child_process');

/**
 * Runs the TypeScript compiler in "noEmit" mode to type-check the project,
 * including the generated client code.
 *
 * @param {string[]} [extraArgs=[]] - Extra arguments passed to `tsc`.
 */
function typecheckGeneratedClient(extraArgs = []) {
  const baseCmd = ['npx', 'tsc', '--noEmit', ...extraArgs].join(' ');
  try {
    console.log(`[typecheckGeneratedClient] Running: ${baseCmd}`);
    execSync(baseCmd, { stdio: 'inherit' });
    console.log('[typecheckGeneratedClient] Type-check completed successfully.');
  } catch (error) {
    console.error(
      '[typecheckGeneratedClient] Type-check failed. Fix TypeScript errors above.',
    );
    throw error;
  }
}

// Allow running as a standalone CLI script.
if (require.main === module) {
  const extraArgs = process.argv.slice(2);
  try {
    typecheckGeneratedClient(extraArgs);
  } catch (error) {
    process.exit(1);
  }
}

module.exports = { typecheckGeneratedClient };



// step:6 file: generate_a_typescript_client_for_a_cosmwasm_contract_using_ts-codegen.
/* tests/smokeTestMyContract.ts */

import { junoNetwork } from '../ts-codegen.config';
// Adjust the import path and type name based on what ts-codegen generated for your contract.
import type { QueryMsg } from '../src/contracts/MyContract.types';

// Use Node 18+ global fetch or bring your own polyfill (for example, node-fetch).
const fetchFn: typeof fetch = (globalThis as any).fetch;

/**
 * Encodes a JSON query message to the base64-encoded string that the
 * Juno LCD smart-contract query endpoint expects.
 */
function encodeQueryMsgToBase64(msg: QueryMsg): string {
  const json = JSON.stringify(msg);
  return Buffer.from(json, 'utf8').toString('base64');
}

/**
 * Perform a simple smart-contract query against Juno using the LCD endpoint.
 *
 * This function assumes your contract exposes a `config` query; adjust the
 * `queryMsg` shape to match your actual schema if needed.
 */
export async function smokeTestMyContractQuery() {
  const lcdBaseUrl =
    junoNetwork.lcdEndpoint || 'https://lcd-archive.junonetwork.io';
  const contractAddress = process.env.MY_CONTRACT_ADDRESS;

  if (!contractAddress) {
    throw new Error(
      'Missing MY_CONTRACT_ADDRESS environment variable (juno1... contract address).',
    );
  }

  if (!fetchFn) {
    throw new Error(
      'Global fetch is not available. Run this with Node 18+ or provide a fetch polyfill.',
    );
  }

  // Example query message: { config: {} }
  const queryMsg: QueryMsg = {
    // Replace this with the actual shape of your query.
    // For example, if your schema has { "config": {} }, this is correct.
    // @ts-expect-error - will be narrowed once you use the real QueryMsg type.
    config: {},
  };

  const queryDataBase64 = encodeQueryMsgToBase64(queryMsg);
  const url = `${lcdBaseUrl}/cosmwasm/wasm/v1/contract/${contractAddress}/smart/${encodeURIComponent(
    queryDataBase64,
  )}`;

  console.log('[smokeTestMyContractQuery] Querying URL:', url);

  const response = await fetchFn(url);

  if (!response.ok) {
    const bodyText = await response.text();
    throw new Error(
      `[smokeTestMyContractQuery] LCD request failed: ${response.status} ${response.statusText} - ${bodyText}`,
    );
  }

  const body = (await response.json()) as { data: string };

  if (!body || typeof body.data !== 'string') {
    throw new Error(
      `[smokeTestMyContractQuery] Unexpected LCD response shape: ${JSON.stringify(
        body,
      )}`,
    );
  }

  // The `data` field is base64-encoded JSON returned by the contract.
  const decodedJson = Buffer.from(body.data, 'base64').toString('utf8');

  let parsed: unknown;
  try {
    parsed = JSON.parse(decodedJson);
  } catch (e) {
    throw new Error(
      `[smokeTestMyContractQuery] Failed to parse contract response JSON: ${decodedJson}`,
    );
  }

  console.log(
    '[smokeTestMyContractQuery] Contract query successful. Parsed response:',
  );
  // eslint-disable-next-line no-console
  console.dir(parsed, { depth: null });

  return parsed;
}

// Allow running as a standalone script: `npx ts-node tests/smokeTestMyContract.ts`
if (require.main === module) {
  smokeTestMyContractQuery().catch((error) => {
    console.error('[smokeTestMyContractQuery] Error:', error);
    process.exit(1);
  });
}



// step:3 file: claim_junox_test_tokens_from_the_juno_faucet_for_a_given_address_and_verify_receipt_on-chain
export const openJunoFaucetUI = (faucetUrl) => {
  // 'faucetUrl' should be configured in your environment, e.g.
  // process.env.NEXT_PUBLIC_JUNO_UNI6_FAUCET_URL
  if (typeof window === 'undefined') {
    throw new Error('Cannot open faucet UI: window is not available (are you on the server?).');
  }

  if (!faucetUrl || typeof faucetUrl !== 'string') {
    throw new Error('Faucet URL is not configured or is invalid.');
  }

  try {
    const newWindow = window.open(faucetUrl, '_blank', 'noopener,noreferrer');

    if (!newWindow) {
      // Pop-up may be blocked by the browser
      throw new Error('Failed to open the faucet UI. Please allow pop-ups for this site.');
    }
  } catch (err) {
    console.error('Error opening Juno faucet UI:', err);
    throw err;
  }
};



// step:4 file: claim_junox_test_tokens_from_the_juno_faucet_for_a_given_address_and_verify_receipt_on-chain
export const instructUserToRequestFaucetTokens = (address, denomLabel = 'JUNOX') => {
  if (typeof window === 'undefined') {
    throw new Error('Cannot show faucet instructions: window is not available (are you on the server?).');
  }

  if (!address || typeof address !== 'string') {
    throw new Error('A valid Juno address is required before requesting faucet tokens.');
  }

  // This function does not attempt to automate the official faucet UI,
  // because it is a separate web application and its implementation
  // can change. Instead, we give clear instructions to the user.
  const message = [
    'A faucet window/tab should now be open.',
    `1. Paste this address into the faucet address field: ${address}`,
    `2. Select the token type (e.g., ${denomLabel}).`,
    '3. Submit the request and wait for the faucet to confirm or show an error.',
    '4. If you see errors such as rate limiting or daily limits, follow the faucet instructions or try again later.'
  ].join('\n');

  // You can replace this alert with rendering a modal or in-app message.
  window.alert(message);
};



