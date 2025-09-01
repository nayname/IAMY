// step:1 file: Query the global counter value
export const loadContractAddress = () => {
  // It’s recommended to store contract addresses in environment variables
  // so you can easily switch between testnet, mainnet, and localnet.
  const address =
    import.meta.env.VITE_TEMPLATE_CONTRACT_ADDRESS ||
    process.env.NEXT_PUBLIC_TEMPLATE_CONTRACT_ADDRESS;

  if (!address) {
    throw new Error(
      'NeutronTemplate contract address is not defined in environment variables.'
    );
  }

  // Basic sanity-check: Neutron bech32 addresses start with `ntrn1` and are 43 chars long.
  if (!/^ntrn1[0-9a-z]{38}$/.test(address)) {
    throw new Error('Invalid Neutron contract address format.');
  }

  return address;
};


// step:2 file: Query the global counter value
export const constructWasmQueryMsg = () => {
  // Message schema follows the contract’s public interface.
  return {
    get_global_counter: {}
  };
};


// step:3 file: Query the global counter value
import { StargateClient } from '@cosmjs/stargate';
import { loadContractAddress } from './loadContractAddress.js';
import { constructWasmQueryMsg } from './constructWasmQueryMsg.js';

/**
 * Queries the NeutronTemplate contract for its global counter value.
 * @param {string} rpcEndpoint - Full RPC URL, e.g. "https://rpc-kralum.neutron-1.neutron.org".
 * @returns {Promise<number>} - The current counter value stored on-chain.
 */
export const queryContractGlobalCounter = async (rpcEndpoint) => {
  try {
    // 1. Connect to the chain.
    const client = await StargateClient.connect(rpcEndpoint);

    // 2. Prepare contract address & query msg.
    const contractAddress = loadContractAddress();
    const queryMsg = constructWasmQueryMsg();

    // 3. Execute the smart query.
    const response = await client.queryContractSmart(contractAddress, queryMsg);
    // Expected response shape: { count: <number> }

    if (!response || typeof response.count !== 'number') {
      throw new Error('Unexpected response format from contract.');
    }

    return response.count;
  } catch (error) {
    console.error('Failed to query global counter:', error);
    throw error;
  }
};


// step:1 file: Query my personal counter value
export const getSenderAddress = async () => {
  const chainId = "neutron-1"; // change to testnet chain-id if needed

  // Make sure the browser wallet is available
  const keplr = window.keplr;
  if (!keplr) {
    throw new Error("Keplr wallet is not installed. Please install it and retry.");
  }

  // Ask the wallet to enable the Neutron chain and return an OfflineSigner
  await keplr.enable(chainId);
  const offlineSigner = keplr.getOfflineSigner(chainId);

  // Extract the first account (there is at least one by spec if enable() succeeded)
  const accounts = await offlineSigner.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error("No accounts found in the connected wallet.");
  }

  // Return both the bech32 address and the signer in case later logic needs signing capability
  return {
    address: accounts[0].address,
    signer: offlineSigner
  };
};


// step:2 file: Query my personal counter value
export const loadContractAddress = (network = "mainnet") => {
  // NOTE: In production you might fetch this value from your backend or .env file.
  const CONTRACTS = {
    mainnet: {
      neutronTemplate: "neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" // ← replace with real mainnet addr
    },
    testnet: {
      neutronTemplate: "neutron1yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy" // ← replace with real testnet addr
    }
  };

  const address = CONTRACTS[network]?.neutronTemplate;
  if (!address) {
    throw new Error(`NeutronTemplate contract address for network '${network}' not found.`);
  }

  return address;
};


// step:3 file: Query my personal counter value
export const constructWasmQueryMsg = (senderAddress) => {
  if (!senderAddress) {
    throw new Error("Sender address is required to construct the query message.");
  }

  // Build the query message expected by the NeutronTemplate contract
  return {
    get_personal_counter: {
      address: senderAddress
    }
  };
};


// step:4 file: Query my personal counter value
import { CosmWasmClient } from "@cosmjs/cosmwasm-stargate";

/**
 * Queries the NeutronTemplate contract for the caller's personal counter.
 * @param {string} rpcEndpoint - Public RPC endpoint, e.g. "https://rpc.ntrn.io".
 * @param {string} contractAddress - Deployed NeutronTemplate contract address.
 * @param {object} queryMsg - The query payload produced in Step 3.
 * @returns {Promise<object>} - The raw JSON response from the contract (e.g. { counter: "7" }).
 */
export const queryPersonalCounter = async (rpcEndpoint, contractAddress, queryMsg) => {
  try {
    // Initialise a readonly CosmWasm client (no signer required for queries)
    const client = await CosmWasmClient.connect(rpcEndpoint);

    // Execute the smart query
    const response = await client.queryContractSmart(contractAddress, queryMsg);

    return response; // e.g. { counter: 7 }
  } catch (error) {
    // Forward the error after logging for debugging purposes
    console.error("Contract smart-query failed:", error);
    throw error;
  }
};


// step:1 file: Query a contract’s NTRN balance
export const getContractAddress = () => {
  // Attempt to read the contract address from an input with id 'contract-address-input'
  const inputEl = document.getElementById('contract-address-input');
  if (!inputEl) {
    throw new Error('Element with id "contract-address-input" not found in the DOM.');
  }
  const address = inputEl.value.trim();
  if (!address) {
    throw new Error('Contract address cannot be empty.');
  }
  return address;
};


// step:2 file: Query a contract’s NTRN balance
import { bech32 } from 'bech32';

export const validateAddressFormat = (address) => {
  try {
    const decoded = bech32.decode(address);
    if (decoded.prefix !== 'neutron') {
      throw new Error('Invalid bech32 prefix: expected neutron, got ' + decoded.prefix);
    }
    // Re-encode to verify checksum integrity
    const recoded = bech32.encode(decoded.prefix, decoded.words);
    if (recoded !== address.toLowerCase()) {
      throw new Error('Checksum mismatch.');
    }
    return true;
  } catch (err) {
    throw new Error('Address validation failed: ' + err.message);
  }
};


// step:1 file: Deposit 100 NTRN into a smart contract
export const ensureWalletConnected = async (chainId = 'neutron-1') => {
  // Verify Keplr is installed
  if (!window || !window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }
  try {
    // Request access to the Neutron chain
    await window.keplr.enable(chainId);
    // Obtain an OfflineSigner (Amino/Direct signer auto-selected)
    const offlineSigner = await window.keplr.getOfflineSignerAuto(chainId);
    return offlineSigner;
  } catch (error) {
    console.error('Failed to connect to Keplr:', error);
    throw error;
  }
};


// step:2 file: Deposit 100 NTRN into a smart contract
export const getSenderAddress = async (signer) => {
  const accounts = await signer.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in signer.');
  }
  return accounts[0].address;
};


// step:3 file: Deposit 100 NTRN into a smart contract
import { fromBech32 } from '@cosmjs/encoding';

export const validateContractAddress = (address, expectedPrefix = 'neutron') => {
  try {
    const { prefix } = fromBech32(address);
    if (prefix !== expectedPrefix) {
      throw new Error(`Prefix mismatch: expected ${expectedPrefix}, got ${prefix}`);
    }
    return true;
  } catch (error) {
    console.error('Invalid contract address:', error);
    throw new Error('Provided contract address is invalid.');
  }
};


// step:4 file: Deposit 100 NTRN into a smart contract
export const convertToBaseUnits = (amountNTRN) => {
  if (isNaN(amountNTRN)) {
    throw new Error('Amount must be numeric.');
  }
  const MICRO_FACTOR = 1_000_000; // 1 NTRN = 1,000,000 untrn
  const microAmount = BigInt(Math.floor(Number(amountNTRN) * MICRO_FACTOR));
  return microAmount.toString();
};


// step:5 file: Deposit 100 NTRN into a smart contract
import { coin } from '@cosmjs/stargate';

export const constructTxWasmExecute = (sender, contract, depositMicro) => {
  const executeMsg = { deposit: {} }; // Payload expected by the contract
  const funds = [coin(depositMicro, 'untrn')];
  return {
    sender,
    contract,
    msg: executeMsg,
    funds,
  };
};


// step:6 file: Deposit 100 NTRN into a smart contract
import { SigningCosmWasmClient } from '@cosmjs/cosmwasm-stargate';

export const signAndBroadcastTx = async (signer, executeMsg, rpcEndpoint = 'https://rpc-kralum.neutron.org') => {
  try {
    const client = await SigningCosmWasmClient.connectWithSigner(rpcEndpoint, signer);
    // Use 'auto' for fee estimation or replace with a custom fee object.
    const fee = 'auto';
    const result = await client.execute(
      executeMsg.sender,
      executeMsg.contract,
      executeMsg.msg,
      fee,
      undefined,
      executeMsg.funds,
    );
    console.log('Transaction broadcasted. Hash:', result.transactionHash);
    return result;
  } catch (error) {
    console.error('Failed to sign/broadcast transaction:', error);
    throw error;
  }
};


// step:1 file: Withdraw 50 NTRN from the smart contract
/* Step 1 – Ensure wallet connection */
export const ensureWalletConnected = async () => {
  const chainId = 'neutron-1';               // Neutron main-net chain-id
  const keplr = window.keplr;

  if (!keplr) {
    // Keplr wallet extension not found
    throw new Error('Keplr wallet is not installed.');
  }

  try {
    // Ask Keplr to enable the chain (will prompt user if not yet added)
    await keplr.enable(chainId);
  } catch (err) {
    throw new Error(`User rejected wallet connection: ${err.message}`);
  }

  // Return OfflineSigner for use with CosmJS
  return window.getOfflineSigner(chainId);
};


// step:2 file: Withdraw 50 NTRN from the smart contract
/* Step 2 – Validate contract address */
import { bech32 } from 'bech32';

export const validateContractAddress = (contractAddress) => {
  const expectedPrefix = 'neutron';

  try {
    const { prefix } = bech32.decode(contractAddress);
    if (prefix !== expectedPrefix) {
      throw new Error(
        `Invalid prefix: expected "${expectedPrefix}", got "${prefix}"`
      );
    }
  } catch (err) {
    throw new Error(`Malformed Bech32 address: ${err.message}`);
  }

  return true; // address is valid
};


// step:3 file: Withdraw 50 NTRN from the smart contract
/* Step 3 – Convert 50 NTRN to micro-denom (untrn) */
export const convertToBaseUnits = (amountNtrn) => {
  if (typeof amountNtrn !== 'number' || amountNtrn <= 0) {
    throw new Error('Amount must be a positive number.');
  }

  const MICRO_FACTOR = 1_000_000; // 10^6
  return (amountNtrn * MICRO_FACTOR).toString(); // "50000000"
};


// step:4 file: Withdraw 50 NTRN from the smart contract
/* Step 4 – Build MsgExecuteContract for the withdraw */
import { toUtf8 } from '@cosmjs/encoding';

export const constructTxWasmExecute = (
  senderAddress,
  contractAddress,
  amountBaseUnits
) => {
  // Build the JSON message expected by the contract
  const executeMsg = {
    withdraw: {
      amount: amountBaseUnits, // already a string (e.g. "50000000")
    },
  };

  // Return message object compatible with SigningCosmWasmClient.signAndBroadcast()
  return {
    typeUrl: '/cosmwasm.wasm.v1.MsgExecuteContract',
    value: {
      sender: senderAddress,
      contract: contractAddress,
      msg: toUtf8(JSON.stringify(executeMsg)),
      funds: [], // no attached funds
    },
  };
};


// step:5 file: Withdraw 50 NTRN from the smart contract
/* Step 5 – Sign & broadcast the transaction */
import { SigningCosmWasmClient } from '@cosmjs/cosmwasm-stargate';

export const signAndBroadcastTx = async (signer, msg) => {
  const rpcEndpoint = 'https://rpc-kralum.neutron.org'; // public Neutron RPC
  const fee = 'auto'; // let CosmJS estimate gas & fee

  // Derive sender address from signer
  const [{ address: senderAddress }] = await signer.getAccounts();

  // Instantiate client
  const client = await SigningCosmWasmClient.connectWithSigner(
    rpcEndpoint,
    signer
  );

  // Sign & broadcast
  const result = await client.signAndBroadcast(senderAddress, [msg], fee);

  if (result.code !== 0) {
    throw new Error(`Tx failed (code: ${result.code}): ${result.rawLog}`);
  }

  return result; // contains transactionHash, height, etc.
};


// step:1 file: Connect Leap wallet
export const detectLeapProvider = async () => {
  // Ensure we're in a browser environment
  if (typeof window === 'undefined') {
    throw new Error('This function must be run in a browser context.');
  }

  // 1) Check if the Leap extension has already injected itself
  if (window.leap) {
    return window.leap;
  }

  // 2) Fallback: attempt to dynamically import the Cosmos Kit Leap adapter
  try {
    const { LeapWallet } = await import('@cosmos-kit/leap');
    const leapAdapter = new LeapWallet();

    // The adapter offers helper methods to check installation status
    if (await leapAdapter.isInstalled?.()) {
      return leapAdapter;
    }
  } catch (err) {
    console.error('Failed to load @cosmos-kit/leap adapter:', err);
  }

  // If we reach here, Leap is not available
  throw new Error('Leap Wallet provider not found. Please install the Leap browser extension.');
};


// step:2 file: Connect Leap wallet
export const suggestNeutronChain = async (leap) => {
  const chainConfig = {
    // Basic chain info
    chainId: 'neutron-1',
    chainName: 'Neutron',

    // Public RPC & REST endpoints (replace with your own if self-hosting)
    rpc: 'https://rpc-kralum.neutron.org',
    rest: 'https://lcd-kralum.neutron.org',

    // BIP-44 & Bech32 configuration
    bip44: { coinType: 118 },
    bech32Config: {
      bech32PrefixAccAddr: 'neutron',
      bech32PrefixAccPub: 'neutronpub',
      bech32PrefixValAddr: 'neutronvaloper',
      bech32PrefixValPub: 'neutronvaloperpub',
      bech32PrefixConsAddr: 'neutronvalcons',
      bech32PrefixConsPub: 'neutronvalconspub',
    },

    // Denomination definitions
    stakeCurrency: {
      coinDenom: 'NTRN',
      coinMinimalDenom: 'untrn',
      coinDecimals: 6,
    },
    currencies: [
      {
        coinDenom: 'NTRN',
        coinMinimalDenom: 'untrn',
        coinDecimals: 6,
      },
    ],
    feeCurrencies: [
      {
        coinDenom: 'NTRN',
        coinMinimalDenom: 'untrn',
        coinDecimals: 6,
        gasPriceStep: {
          low: 0.01,
          average: 0.025,
          high: 0.04,
        },
      },
    ],

    // Feature flags recognised by Leap
    features: ['stargate', 'ibc-transfer', 'cosmwasm'],
  };

  try {
    await leap.experimentalSuggestChain(chainConfig);
  } catch (error) {
    console.error('Failed to suggest Neutron chain to Leap:', error);
    throw new Error('Chain suggestion rejected or failed.');
  }
};


// step:3 file: Connect Leap wallet
export const enableNeutronChain = async (leap) => {
  const chainId = 'neutron-1';
  try {
    await leap.enable(chainId);
  } catch (error) {
    console.error('User rejected or failed to enable Neutron chain:', error);
    throw new Error('Failed to enable Neutron chain in Leap.');
  }
};


// step:4 file: Connect Leap wallet
export const retrieveLeapAccounts = async (leap) => {
  const chainId = 'neutron-1';

  try {
    // Prefer the more generic getOfflineSigner if available
    const offlineSigner = leap.getOfflineSigner?.(chainId) || leap.getOfflineSignerOnlyAmino?.(chainId);

    if (!offlineSigner) {
      throw new Error('Could not obtain an OfflineSigner from Leap.');
    }

    const accounts = await offlineSigner.getAccounts();
    if (!accounts || accounts.length === 0) {
      throw new Error('No accounts returned by Leap.');
    }

    return {
      offlineSigner,
      address: accounts[0].address,
    };
  } catch (error) {
    console.error('Error while retrieving Leap accounts:', error);
    throw error;
  }
};


// step:1 file: Query the connected wallet’s NTRN balance
export const ensureWalletConnected = async () => {
  // Neutron main-net chain-ID
  const chainId = "neutron-1";

  // Detect a supported browser wallet
  const wallet = window.keplr || window.leap;
  if (!wallet) {
    throw new Error("Keplr or Leap wallet extension is not installed.");
  }

  try {
    // Ask the wallet to connect / enable the requested chain
    await wallet.enable(chainId);
  } catch (err) {
    throw new Error(`Wallet connection request was rejected: ${err?.message ?? err}`);
  }

  // Return an OfflineSigner that grants access to the user accounts
  return wallet.getOfflineSigner ? wallet.getOfflineSigner(chainId) : window.getOfflineSigner(chainId);
};


// step:2 file: Query the connected wallet’s NTRN balance
export const getWalletAddress = async (signer) => {
  const accounts = await signer.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error("Unable to retrieve an account from the signer.");
  }

  // Typically the first account is the active one
  return accounts[0].address;
};


// step:1 file: Set the user's own address as the admin of a smart contract
export const getUserAddress = async () => {
  const chainId = 'neutron-1';
  const keplr = window.keplr;

  // Ensure Keplr is installed
  if (!keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  // Request wallet connection (prompts user approval)
  await keplr.enable(chainId);

  // Retrieve the signer and the account list
  const signer = window.getOfflineSigner(chainId);
  const accounts = await signer.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in the signer.');
  }

  // Return the first account’s bech32 address
  return accounts[0].address;
};


// step:2 file: Set the user's own address as the admin of a smart contract
export const promptContractAddress = () => {
  // Browser prompt for simplicity; replace with a nicer UI as needed
  const contractAddr = prompt('Enter the CosmWasm contract address to update admin for:');

  if (!contractAddr) {
    throw new Error('Contract address is required.');
  }

  const trimmed = contractAddr.trim();

  // Basic bech32 length sanity-check (optional: use full bech32 validation)
  if (trimmed.length < 40 || trimmed.length > 90) {
    throw new Error('Invalid contract address length.');
  }

  return trimmed;
};


// step:1 file: Connect a user’s wallet to the dApp
/*
 * connectWallet()
 * Detects an injected Cosmos-SDK wallet (Keplr or Leap) or a WalletConnect provider
 * and requests access to the user’s accounts.
 */
export const connectWallet = async () => {
  const chainId = 'neutron-1';

  // Detect supported wallets in order of preference
  const wallet = (window.keplr || window.leap || window.walletConnect) ?? null;
  if (!wallet) {
    throw new Error('No supported wallet found. Please install Keplr or Leap, or connect via WalletConnect.');
  }

  try {
    // Ask the wallet to enable access to Neutron
    await wallet.enable(chainId);

    // Retrieve an OfflineSigner that will be used in later steps
    const signer = await wallet.getOfflineSignerAuto(chainId);

    return { wallet, signer };
  } catch (error) {
    console.error('connectWallet() failed:', error);
    throw new Error('Wallet connection was rejected or failed.');
  }
};


// step:2 file: Connect a user’s wallet to the dApp
/*
 * ensureNetworkNeutron(wallet)
 * Makes sure the wallet is configured for Neutron mainnet. If the chain is not
 * present, it sends an experimentalSuggestChain() request with recommended
 * parameters. Finally, it enables the chain again to refresh permissions.
 */
export const ensureNetworkNeutron = async (wallet) => {
  const chainId = 'neutron-1';

  // Neutron chain parameters — tweak RPC/REST endpoints as needed for production
  const neutronChainInfo = {
    chainId: 'neutron-1',
    chainName: 'Neutron',
    rpc: 'https://rpc-kralum.neutron-1.neutron.org',
    rest: 'https://rest-kralum.neutron-1.neutron.org',
    stakeCurrency: { coinDenom: 'NTRN', coinMinimalDenom: 'untrn', coinDecimals: 6 },
    bip44: { coinType: 118 },
    bech32Config: {
      bech32PrefixAccAddr: 'neutron',
      bech32PrefixAccPub: 'neutronpub',
      bech32PrefixValAddr: 'neutronvaloper',
      bech32PrefixValPub: 'neutronvaloperpub',
      bech32PrefixConsAddr: 'neutronvalcons',
      bech32PrefixConsPub: 'neutronvalconspub'
    },
    currencies: [
      { coinDenom: 'NTRN', coinMinimalDenom: 'untrn', coinDecimals: 6 }
    ],
    feeCurrencies: [
      { coinDenom: 'NTRN', coinMinimalDenom: 'untrn', coinDecimals: 6, gasPriceStep: { low: 0.01, average: 0.025, high: 0.04 } }
    ],
    features: ['stargate', 'ibc-transfer', 'cosmwasm']
  };

  try {
    // Attempt to query the chain; if it throws, the chain likely isn't added yet
    await wallet.getKey(chainId);
  } catch (_) {
    // Use Keplr/Leap experimental API to suggest the chain
    if (wallet.experimentalSuggestChain) {
      await wallet.experimentalSuggestChain(neutronChainInfo);
    } else {
      throw new Error('The connected wallet does not support adding new chains.');
    }
  }

  // Re-enable to make sure we have permission on the (new) chain
  await wallet.enable(chainId);
};


// step:3 file: Connect a user’s wallet to the dApp
/*
 * storeSessionAccount(signer)
 * Extracts the first account (address+pubkey) from the provided signer and
 * persists it into sessionStorage under the key `neutron_session_account`.
 */
export const storeSessionAccount = async (signer) => {
  try {
    const accounts = await signer.getAccounts();
    if (!accounts || accounts.length === 0) {
      throw new Error('No accounts returned by signer.');
    }
    const { address, pubkey } = accounts[0];

    // Save as JSON for later retrieval
    const sessionData = JSON.stringify({ address, pubkey: Buffer.from(pubkey).toString('base64') });
    sessionStorage.setItem('neutron_session_account', sessionData);

    return { address, pubkey };
  } catch (error) {
    console.error('storeSessionAccount() failed:', error);
    throw new Error('Unable to retrieve or store account information.');
  }
};


// step:1 file: Query my deposited NTRN amount
// src/utils/getSenderAddress.js
export const getSenderAddress = async () => {
  const chainId = 'neutron-1';

  // Ensure Keplr is available in the browser
  if (!window || !window.keplr) {
    throw new Error('Keplr wallet extension is not installed.');
  }

  try {
    // Request wallet access for Neutron
    await window.keplr.enable(chainId);

    // Get an offline signer for the chain
    const signer = window.getOfflineSigner(chainId);
    const accounts = await signer.getAccounts();

    if (!accounts || accounts.length === 0) {
      throw new Error('No accounts found in the connected wallet.');
    }

    // Return the first account’s address
    return accounts[0].address;
  } catch (err) {
    console.error('Error while fetching sender address:', err);
    throw err;
  }
};


// step:2 file: Query my deposited NTRN amount
// src/config/loadContractAddress.js
export const loadContractAddress = () => {
  // BEST PRACTICE: Keep contract addresses in env variables or config files, not hard-coded.
  const contractAddress = process.env.REACT_APP_TEMPLATE_CONTRACT_ADDRESS;

  if (!contractAddress) {
    throw new Error('NeutronTemplate contract address is not configured.');
  }

  return contractAddress;
};


// step:3 file: Query my deposited NTRN amount
// src/utils/constructWasmQueryMsg.js
/**
 * Creates the query message for `{ get_deposit: { address: <sender> } }`.
 * @param {string} senderAddress - Bech32 Neutron account address whose deposit is queried.
 * @returns {object} A properly-shaped query message ready for CosmWasmClient.
 */
export const constructWasmQueryMsg = (senderAddress) => {
  if (!senderAddress) {
    throw new Error('Sender address must be provided to construct the query.');
  }

  return {
    get_deposit: {
      address: senderAddress,
    },
  };
};


// step:4 file: Query my deposited NTRN amount
// src/queries/queryContractSmart.js
import { CosmWasmClient } from '@cosmjs/cosmwasm-stargate';

/**
 * Queries a CosmWasm contract on Neutron for a given message.
 * @param {string} rpcEndpoint - Full RPC endpoint, e.g. 'https://rpc-kralum.neutron-1.neutron.org:443'.
 * @param {string} contractAddress - Bech32 address of the NeutronTemplate contract.
 * @param {object} queryMsg - JSON query message constructed in Step 3.
 * @returns {Promise<any>} Result returned by the contract.
 */
export const queryContractSmart = async (rpcEndpoint, contractAddress, queryMsg) => {
  if (!rpcEndpoint || !contractAddress || !queryMsg) {
    throw new Error('RPC endpoint, contract address, and query message are all required.');
  }

  try {
    // Create a read-only CosmWasm client
    const client = await CosmWasmClient.connect(rpcEndpoint);

    // Execute the smart query
    const result = await client.queryContractSmart(contractAddress, queryMsg);

    return result; // Expected shape: { amount: "<uNTRN>" }
  } catch (err) {
    console.error('Smart-contract query failed:', err);
    throw err;
  }
};


// step:1 file: Query the code hash of a specific smart contract
/*
 * validateContractAddress.js
 * Checks if a contract address is a valid Neutron Bech32 address.
 */
import { fromBech32 } from "@cosmjs/encoding";

/**
 * Validate a Bech32 contract address.
 * @param {string} address - The contract address to validate.
 * @param {string} [expectedPrefix="neutron"] - Expected Bech32 prefix.
 * @throws {Error} If the address is malformed or has the wrong prefix.
 * @returns {boolean} true if validation succeeds.
 */
export const validateContractAddress = (address, expectedPrefix = "neutron") => {
  try {
    const { prefix } = fromBech32(address);
    if (prefix !== expectedPrefix) {
      throw new Error(`Invalid prefix: expected '${expectedPrefix}', got '${prefix}'`);
    }
    return true;
  } catch (err) {
    // Wrap and re-throw to keep a clean error surface for callers
    throw new Error(`Invalid contract address: ${err.message}`);
  }
};


// step:1 file: Connect Keplr wallet
export const ensureKeplrInstalled = async () => {
  // Verifies Keplr is injected into the browser window.
  if (typeof window === 'undefined' || !window.keplr) {
    // If not installed, open the official download page and throw an error.
    window.open('https://www.keplr.app/download', '_blank');
    throw new Error('Keplr extension is not installed.');
  }

  // At this point Keplr exists; returning it allows subsequent steps to use the instance.
  return window.keplr;
};


// step:2 file: Connect Keplr wallet
export const suggestNeutronChain = async (keplr, network = 'mainnet') => {
  if (!keplr || !keplr.experimentalSuggestChain) {
    throw new Error('Keplr experimentalSuggestChain API is unavailable.');
  }

  // Define minimal, reliable endpoints for both mainnet and testnet.
  const ENDPOINTS = {
    mainnet: {
      chainId: 'neutron-1',
      rpc: 'https://rpc-kralum.neutron-1.neutron.org',
      rest: 'https://rest-kralum.neutron-1.neutron.org'
    },
    testnet: {
      chainId: 'pion-1',
      rpc: 'https://rpc-palvus.pion-1.ntrn.tech',
      rest: 'https://rest-palvus.pion-1.ntrn.tech'
    }
  };

  const cfg = ENDPOINTS[network];
  if (!cfg) {
    throw new Error(`Unsupported network tag: ${network}`);
  }

  const chainConfig = {
    chainId: cfg.chainId,
    chainName: network === 'mainnet' ? 'Neutron' : 'Neutron Testnet',
    rpc: cfg.rpc,
    rest: cfg.rest,
    bip44: { coinType: 118 },
    bech32Config: {
      bech32PrefixAccAddr: 'neutron',
      bech32PrefixAccPub: 'neutronpub',
      bech32PrefixValAddr: 'neutronvaloper',
      bech32PrefixValPub: 'neutronvaloperpub',
      bech32PrefixConsAddr: 'neutronvalcons',
      bech32PrefixConsPub: 'neutronvalconspub'
    },
    currencies: [
      {
        coinDenom: 'NTRN',
        coinMinimalDenom: 'untrn',
        coinDecimals: 6,
        coinGeckoId: 'neutron'
      }
    ],
    feeCurrencies: [
      {
        coinDenom: 'NTRN',
        coinMinimalDenom: 'untrn',
        coinDecimals: 6,
        coinGeckoId: 'neutron',
        gasPriceStep: { low: 0.01, average: 0.025, high: 0.04 }
      }
    ],
    stakeCurrency: {
      coinDenom: 'NTRN',
      coinMinimalDenom: 'untrn',
      coinDecimals: 6,
      coinGeckoId: 'neutron'
    },
    features: ['stargate', 'ibc-transfer', 'cosmwasm']
  };

  try {
    await keplr.experimentalSuggestChain(chainConfig);
  } catch (err) {
    console.error('Failed to suggest Neutron chain:', err);
    throw err;
  }
};


// step:3 file: Connect Keplr wallet
export const enableNeutron = async (chainId = 'neutron-1') => {
  if (!window.keplr) {
    throw new Error('Keplr extension not detected.');
  }
  try {
    await window.keplr.enable(chainId); // Opens the Keplr approval popup.
    return true; // Success indicates the site now has access to the chain.
  } catch (err) {
    console.error(`User rejected enabling ${chainId}:`, err);
    throw err;
  }
};


// step:4 file: Connect Keplr wallet
export const getOfflineSignerAndAddress = async (chainId = 'neutron-1') => {
  if (!window.keplr) {
    throw new Error('Keplr extension not found.');
  }

  // Obtain the signer (supports Amino & Direct)
  const offlineSigner = window.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No accounts returned from Keplr.');
  }

  return {
    signer: offlineSigner,
    address: accounts[0].address
  };
};


// step:1 file: Migrate an existing smart contract to a new code ID
import { bech32 } from "@cosmjs/encoding";

/**
 * Prompt the user for a contract address and validate it.
 * @returns {string} A valid Neutron contract address.
 * @throws Will throw if the address is missing or malformed.
 */
export const getContractAddress = () => {
  const address = window.prompt("Enter the CosmWasm contract address to migrate:");
  if (!address) {
    throw new Error("No address supplied.");
  }

  try {
    const { prefix } = bech32.decode(address);
    if (prefix !== "neutron") {
      throw new Error("Address prefix must be 'neutron'.");
    }
  } catch (err) {
    throw new Error(`Invalid bech32 address: ${err.message}`);
  }

  return address;
};


// step:3 file: Migrate an existing smart contract to a new code ID
/**
 * Ask the user for a JSON-formatted migrate message and return it as an object.
 * Defaults to an empty object if the user provides no input.
 */
export const collectMigrateMessage = () => {
  let msgInput = window.prompt("Enter the migrate message in JSON (default {}):", "{}");
  if (!msgInput || msgInput.trim() === "") {
    msgInput = "{}";
  }
  try {
    const msg = JSON.parse(msgInput);
    return msg;
  } catch (err) {
    throw new Error("Invalid JSON supplied for migrate message.");
  }
};


// step:4 file: Migrate an existing smart contract to a new code ID
import { SigningCosmWasmClient, GasPrice, calculateFee } from "@cosmjs/cosmwasm-stargate";
import { toUtf8 } from "@cosmjs/encoding";
import Long from "long";

/**
 * Build a MsgMigrateContract and estimate fee.
 *
 * @param {Object} params - Parameters for building the transaction.
 * @param {string} params.rpcEndpoint - RPC endpoint.
 * @param {OfflineSigner} params.signer - OfflineSigner returned from wallet.
 * @param {string} params.senderAddress - Admin address executing the migration.
 * @param {string} params.contractAddress - Address of the contract to migrate.
 * @param {number} params.newCodeId - Target code ID.
 * @param {Object} params.migrateMsg - JSON migrate message.
 * @param {string} [params.gasPrice="0.025untrn"] - Gas price to use.
 * @param {number} [params.gasLimit=300000] - Gas limit.
 * @returns {Promise<{client: SigningCosmWasmClient, msg: any, fee: StdFee}>}
 */
export const constructMigrateTx = async ({
  rpcEndpoint,
  signer,
  senderAddress,
  contractAddress,
  newCodeId,
  migrateMsg,
  gasPrice = "0.025untrn",
  gasLimit = 300000,
}) => {
  try {
    const client = await SigningCosmWasmClient.connectWithSigner(rpcEndpoint, signer, {
      gasPrice: GasPrice.fromString(gasPrice),
    });

    const msg = {
      typeUrl: "/cosmwasm.wasm.v1.MsgMigrateContract",
      value: {
        sender: senderAddress,
        contract: contractAddress,
        codeId: Long.fromNumber(newCodeId),
        msg: toUtf8(JSON.stringify(migrateMsg)),
      },
    };

    const fee = calculateFee(gasLimit, gasPrice);

    return { client, msg, fee };
  } catch (error) {
    console.error(error);
    throw new Error("Failed to construct migrate transaction.");
  }
};


// step:5 file: Migrate an existing smart contract to a new code ID
/**
 * Sign and broadcast a previously-built MsgMigrateContract.
 *
 * @param {SigningCosmWasmClient} client - Connected signing client.
 * @param {string} senderAddress - Admin address (must match signer).
 * @param {any} msg - MsgMigrateContract created in Step 4.
 * @param {StdFee} fee - Fee object calculated in Step 4.
 * @returns {Promise<TxRaw>} The broadcast result.
 */
export const signAndBroadcast = async (client, senderAddress, msg, fee) => {
  try {
    const result = await client.signAndBroadcast(senderAddress, [msg], fee);
    if (result.code !== 0) {
      throw new Error(`Broadcast failed with code ${result.code}: ${result.rawLog}`);
    }
    console.info(`Migration successful. Tx hash: ${result.transactionHash}`);
    return result;
  } catch (error) {
    console.error(error);
    throw new Error("Failed to sign and broadcast the migrate transaction.");
  }
};


// step:1 file: Increment the global counter
/* getSenderAddress.js */
export const getSenderAddress = async () => {
  const chainId = 'neutron-1';

  // 1. Ensure Keplr is installed
  if (!window || !window.keplr) {
    throw new Error('Keplr wallet extension is not installed.');
  }

  // 2. Ask Keplr to enable the Neutron chain
  await window.keplr.enable(chainId);

  // 3. Obtain the OfflineSigner for signing transactions
  const signer = window.getOfflineSigner(chainId);

  // 4. Fetch the user’s account address
  const accounts = await signer.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in the connected wallet.');
  }

  return {
    signer,                // OfflineSigner instance
    address: accounts[0].address // Bech32 Neutron address
  };
};


// step:2 file: Increment the global counter
/* loadContractAddress.js */
export const loadContractAddress = () => {
  // Recommended: expose this via Vite/Next.js env variable or a static config file
  const contractAddress = import.meta.env.VITE_TEMPLATE_CONTRACT_ADDRESS;

  if (!contractAddress) {
    throw new Error('Contract address not found. Make sure VITE_TEMPLATE_CONTRACT_ADDRESS is set.');
  }

  return contractAddress;
};


// step:3 file: Increment the global counter
/* constructWasmExecuteMsg.js */
export const constructWasmExecuteMsg = () => {
  // According to the NeutronTemplate contract schema, this message increments a global counter
  return {
    increment_global: {}
  };
};


// step:4 file: Increment the global counter
/* signAndBroadcastTx.js */
import { SigningCosmWasmClient } from '@cosmjs/cosmwasm-stargate';

export const signAndBroadcastTx = async ({
  signer,
  senderAddress,
  contractAddress,
  msg
}) => {
  // RPC endpoint for Neutron mainnet; replace with your preferred provider if needed
  const rpcEndpoint = 'https://rpc-kralum.neutron-1.neutron.org';

  // 1. Connect the client with the signer
  const client = await SigningCosmWasmClient.connectWithSigner(rpcEndpoint, signer);

  // 2. Prepare an appropriate fee. Adjust gas & amount for your use-case.
  const fee = {
    amount: [
      {
        denom: 'untrn', // micro-NTRN denom on Neutron
        amount: '5000'  // ~0.005 NTRN; change if needed
      }
    ],
    gas: '200000'
  };

  try {
    // 3. Execute the contract message
    const result = await client.execute(
      senderAddress,      // The wallet paying the fee
      contractAddress,    // Target contract address
      msg,                // { increment_global: {} }
      fee                 // Fee object
    );

    console.info('✅ Contract executed. Transaction hash:', result.transactionHash);
    return result.transactionHash;
  } catch (err) {
    // Catch & rethrow for downstream UI handling
    console.error('❌ Failed to execute increment_global:', err);
    throw err;
  }
};


// step:1 file: Transfer contract admin rights to another address
// Step 1 – Fetch current admin address
// Requires: @cosmjs/cosmwasm-stargate
import { CosmWasmClient } from "@cosmjs/cosmwasm-stargate";

/**
 * fetchCurrentAdmin queries a contract’s metadata and returns its admin address.
 * @param {string} contractAddress – Bech32 address of the contract
 * @param {string} [rpcEndpoint]   – RPC endpoint, defaults to Neutron main-net
 * @returns {Promise<string>}      – Current admin address (or empty string if none)
 */
export const fetchCurrentAdmin = async (
  contractAddress,
  rpcEndpoint = "https://rpc-kralum.neutron-1.neutron.org:443"
) => {
  try {
    const client = await CosmWasmClient.connect(rpcEndpoint);
    const info = await client.getContract(contractAddress);
    return info.admin || ""; // empty string means no admin set
  } catch (error) {
    console.error("[fetchCurrentAdmin]", error);
    throw new Error("Unable to fetch contract admin: " + error.message);
  }
};


// step:2 file: Transfer contract admin rights to another address
// Step 2 – Validate new admin address
// Requires: bech32 (npm i bech32)
import { bech32 } from "bech32";

/**
 * validateNewAdminAddress ensures the provided address is valid Bech32 with the
 * correct prefix (default "neutron").
 * @param {string} address     – Proposed new admin address
 * @param {string} prefix      – Expected prefix, defaults to "neutron"
 * @throws {Error}             – If the address is invalid
 */
export const validateNewAdminAddress = (address, prefix = "neutron") => {
  try {
    const { prefix: addrPrefix } = bech32.decode(address);
    if (addrPrefix !== prefix) {
      throw new Error(`Invalid prefix: expected '${prefix}', got '${addrPrefix}'`);
    }
    return true;
  } catch (err) {
    console.error("[validateNewAdminAddress]", err);
    throw new Error("Provided new admin address is not a valid Bech32 string.");
  }
};


// step:3 file: Transfer contract admin rights to another address
// Step 3 – Obtain contract address from the user or UI element

/**
 * getContractAddress reads the contract address from an HTML input element or
 * any other source you prefer.
 * @param {string} [inputId] – DOM element id, defaults to 'contractAddressInput'
 * @returns {string}         – The contract address string
 */
export const getContractAddress = (inputId = "contractAddressInput") => {
  const el = document.getElementById(inputId);
  if (!el || !el.value) {
    throw new Error("Contract address not found in the input element.");
  }
  return el.value.trim();
};


// step:4 file: Transfer contract admin rights to another address
// Step 4 – Construct MsgUpdateAdmin
import { coins } from "@cosmjs/amino"; // only for fee helper (optional)

/**
 * buildUpdateAdminMsg creates the message object required by CosmJS for
 * /cosmwasm.wasm.v1.MsgUpdateAdmin.
 * @param {string} sender         – Current admin (wallet) address
 * @param {string} contract       – Contract whose admin you’re updating
 * @param {string} newAdmin       – New admin address
 * @returns {object}              – CosmJS-ready message object
 */
export const buildUpdateAdminMsg = (sender, contract, newAdmin) => {
  return {
    typeUrl: "/cosmwasm.wasm.v1.MsgUpdateAdmin",
    value: {
      sender: sender,
      newAdmin: newAdmin,
      contract: contract
    }
  };
};


// step:5 file: Transfer contract admin rights to another address
// Step 5 – Sign and broadcast update-admin transaction
// Requires: @cosmjs/cosmwasm-stargate
import { SigningCosmWasmClient, GasPrice } from "@cosmjs/cosmwasm-stargate";

/**
 * signAndBroadcastUpdateAdminMsg signs and broadcasts the update-admin message.
 * @param {OfflineSigner} signer  – Obtained from Keplr or Leap wallet
 * @param {string} sender         – Current admin wallet address (signer’s addr)
 * @param {object} msg            – Message object from buildUpdateAdminMsg
 * @param {string} [rpcEndpoint]  – RPC endpoint, defaults to Neutron mainnet
 * @param {string} [memo]         – Optional memo for the tx
 * @returns {Promise<object>}     – Result from the broadcast
 */
export const signAndBroadcastUpdateAdminMsg = async (
  signer,
  sender,
  msg,
  rpcEndpoint = "https://rpc-kralum.neutron-1.neutron.org:443",
  memo = ""
) => {
  try {
    const gasPrice = GasPrice.fromString("0.05untrn");
    const client = await SigningCosmWasmClient.connectWithSigner(
      rpcEndpoint,
      signer,
      { gasPrice }
    );

    // Estimate gas or set a flat fee
    const fee = {
      amount: coins(0, "untrn"), // 0-fee, gasPrice will calculate final fee
      gas: "250000"               // adjust based on contract complexity
    };

    const result = await client.signAndBroadcast(sender, [msg], fee, memo);
    if (result.code !== 0) {
      throw new Error(`Broadcast failed with code ${result.code}: ${result.rawLog}`);
    }
    return result;
  } catch (error) {
    console.error("[signAndBroadcastUpdateAdminMsg]", error);
    throw new Error("Transaction failed: " + error.message);
  }
};


// step:1 file: Send 10 NTRN to a specified recipient address
export const ensureWalletConnected = async (chainId = 'neutron-1') => {
  try {
    const { keplr } = window;
    if (!keplr) {
      throw new Error('Keplr wallet is not installed.');
    }
    // Ask Keplr to enable (or add) Neutron
    await keplr.enable(chainId);

    // getOfflineSignerAuto works for both Amino & Direct protobuf signing
    const offlineSigner = await keplr.getOfflineSignerAuto(chainId);
    return offlineSigner;
  } catch (error) {
    console.error('Failed to connect wallet:', error);
    throw error;
  }
};


// step:2 file: Send 10 NTRN to a specified recipient address
export const getSenderAddress = async (signer) => {
  const accounts = await signer.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error('No account found for the connected wallet.');
  }
  return accounts[0].address;
};


// step:3 file: Send 10 NTRN to a specified recipient address
import { Bech32 } from '@cosmjs/encoding';

export const validateRecipientAddress = (address, expectedPrefix = 'neutron') => {
  try {
    const decoded = Bech32.decode(address);
    if (decoded.prefix !== expectedPrefix) {
      throw new Error(`Invalid Bech32 prefix: expected ${expectedPrefix}, got ${decoded.prefix}`);
    }
    return true;
  } catch (error) {
    console.error('Address validation failed:', error);
    throw new Error('Provided recipient address is invalid.');
  }
};


// step:4 file: Send 10 NTRN to a specified recipient address
export const convertToBaseUnits = (amount, decimals = 6) => {
  const numericAmount = Number(amount);
  if (!Number.isFinite(numericAmount) || numericAmount <= 0) {
    throw new Error('Amount must be a positive number.');
  }
  const factor = Math.pow(10, decimals);
  return String(Math.round(numericAmount * factor));
};


// step:5 file: Send 10 NTRN to a specified recipient address
import { coin } from '@cosmjs/stargate';

export const constructTxBankSend = (sender, recipient, amountMicro, denom = 'untrn') => {
  return {
    typeUrl: '/cosmos.bank.v1beta1.MsgSend',
    value: {
      fromAddress: sender,
      toAddress: recipient,
      amount: [coin(amountMicro, denom)],
    },
  };
};


// step:6 file: Send 10 NTRN to a specified recipient address
import { SigningStargateClient, GasPrice } from '@cosmjs/stargate';

export const signAndBroadcastTx = async (
  signer,
  messages,
  {
    rpcEndpoint = 'https://rpc-kralum.neutron.org',
    gasPrice   = GasPrice.fromString('0.025untrn'),
    memo       = '',
  } = {},
) => {
  try {
    const client = await SigningStargateClient.connectWithSigner(rpcEndpoint, signer, { gasPrice });
    const [account] = await signer.getAccounts();
    const result = await client.signAndBroadcast(account.address, messages, 'auto', memo);

    if (result.code !== 0) {
      throw new Error(`Transaction failed with code ${result.code}: ${result.rawLog}`);
    }
    return result.transactionHash;
  } catch (error) {
    console.error('Broadcast error:', error);
    throw error;
  }
};


// step:1 file: Increment my personal counter
export const getSenderAddress = async (chainId = 'neutron-1') => {
  // Ensure Keplr wallet is installed in the browser
  if (typeof window === 'undefined' || !window.keplr) {
    throw new Error('Keplr wallet extension is not available.');
  }

  // Request the user to unlock/select the Neutron chain
  await window.keplr.enable(chainId);

  // Obtain the signer (OfflineSigner) from Keplr
  const signer = window.getOfflineSigner(chainId);
  const accounts = await signer.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in Keplr.');
  }

  // Return both the address and the signer for later use
  return {
    address: accounts[0].address,
    signer,
  };
};


// step:2 file: Increment my personal counter
export const loadContractAddress = () => {
  /*
    The contract address can live in an environment variable so that it can be
    configured per network (testnet, mainnet, localnet). This keeps the source
    code network agnostic.
  */
  const address = process.env.NEXT_PUBLIC_NEUTRON_TEMPLATE_ADDRESS;

  if (!address || address.length === 0) {
    throw new Error('NeutronTemplate contract address is not set.');
  }

  return address;
};


// step:3 file: Increment my personal counter
export const constructIncrementMsg = () => {
  // The NeutronTemplate contract expects an execute payload of the form:
  // {
  //   increment_personal: {}
  // }
  return {
    increment_personal: {},
  };
};


// step:4 file: Increment my personal counter
import { SigningCosmWasmClient } from '@cosmjs/cosmwasm-stargate';

/**
 * signAndBroadcastTx signs and submits an execute transaction to the Neutron network.
 *
 * @param {string} rpcEndpoint - Public RPC endpoint for the network (e.g. 'https://rpc.neutron.org').
 * @param {OfflineSigner} signer - The signer object obtained from Keplr.
 * @param {string} senderAddress - Bech32 address of the user executing the contract.
 * @param {string} contractAddress - Address of the NeutronTemplate contract.
 * @param {object} msg - Execute message, e.g. { increment_personal: {} }.
 * @param {object} [fee] - Optional custom fee. Defaults to 200k gas & 0.2 NTRN.
 * @returns {Promise<object>} - The deliverTxResponse from CosmJS.
 */
export const signAndBroadcastTx = async ({
  rpcEndpoint = 'https://rpc-kralum.neutron-1.neutron.org',
  signer,
  senderAddress,
  contractAddress,
  msg,
  fee = {
    amount: [{ denom: 'untrn', amount: '200000' }], // 0.2 NTRN
    gas: '200000',
  },
}) => {
  try {
    if (!signer) {
      throw new Error('Signer is required to broadcast transactions.');
    }

    // Initialize CosmWasm client with the signer
    const client = await SigningCosmWasmClient.connectWithSigner(rpcEndpoint, signer);

    // Execute the contract method
    const result = await client.execute(senderAddress, contractAddress, msg, fee);

    console.log('Tx hash:', result.transactionHash);
    return result;
  } catch (error) {
    console.error('Failed to execute contract:', error);
    throw error;
  }
};


// step:1 file: List all smart contracts deployed by my account
/* getCreatorAddress.js */
// Returns a promise that resolves with the connected wallet address (bech32 string).
export const getCreatorAddress = async (chainId = 'neutron-1') => {
  // Make sure the browser has access to the Keplr extension
  const { keplr } = window;
  if (!keplr) {
    throw new Error('Keplr extension was not detected. Please install/enable Keplr.');
  }

  // Ask Keplr to enable the Neutron chain if it is not already enabled
  await keplr.enable(chainId);

  // Obtain the OfflineSigner, then the account list
  const offlineSigner = keplr.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No accounts found in the connected wallet.');
  }

  // The first account will be used as the creator address
  return accounts[0].address;
};


// step:2 file: List all smart contracts deployed by my account
/* validateAddress.js */
import { Bech32 } from '@cosmjs/encoding';

// Throws an error if the provided address is not a valid Neutron bech32 string.
export const validateCreatorAddress = (address, expectedPrefix = 'neutron') => {
  try {
    // Decode the bech32 string; will throw if invalid
    const { prefix } = Bech32.decode(address);

    // Ensure the address uses the expected bech32 prefix ("neutron")
    if (prefix !== expectedPrefix) {
      throw new Error(`Invalid bech32 prefix: expected \"${expectedPrefix}\", got \"${prefix}\".`);
    }

    return true; // valid
  } catch (err) {
    throw new Error(`Invalid bech32 address: ${err.message}`);
  }
};


// step:1 file: Enable mobile wallet support
import React from 'react';
import { ChainProvider } from '@cosmos-kit/react';
import { wallets as keplrWallets } from '@cosmos-kit/keplr';
import { wallets as leapWallets } from '@cosmos-kit/leap';
import { wallets as cosmostationWallets } from '@cosmos-kit/cosmostation';
import { walletconnect } from '@cosmos-kit/walletconnect';
import { Chain } from '@chain-registry/types';

// 1. Neutron chain configuration (simplified – you can also pull this from @chain-registry)
const neutronChain: Chain = {
  chain_name: 'neutron',
  status: 'live',
  chain_id: 'neutron-1',
  bech32_prefix: 'neutron',
  pretty_name: 'Neutron',
  fees: {
    fee_tokens: [
      {
        denom: 'untrn',
        fixed_min_gas_price: 0.025,
        low_gas_price: 0.015,
        average_gas_price: 0.025,
        high_gas_price: 0.03,
      },
    ],
  },
  apis: {
    rpc: [{ address: 'https://rpc-kralum.neutron.org', provider: 'Neutron Foundation' }],
    rest: [{ address: 'https://rest-kralum.neutron.org', provider: 'Neutron Foundation' }],
  },
};

// 2. WalletConnect v2 client options (replace NEXT_PUBLIC_WC_PROJECT_ID with your own key)
const wcOptions = {
  projectId: process.env.NEXT_PUBLIC_WC_PROJECT_ID || '',
  relayUrl: 'wss://relay.walletconnect.com',
  metadata: {
    name: 'NeutronTemplate',
    description: 'Neutron dApp template with WalletConnect support',
    url: 'https://your-dapp.com',
    icons: ['https://your-dapp.com/icon.png'],
  },
};

// 3. Desktop + mobile wallets we want to offer
const wallets = [...keplrWallets, ...leapWallets, ...cosmostationWallets];

// 4. Re-usable React provider that wires everything together
export const WalletProvider: React.FC<React.PropsWithChildren> = ({ children }) => (
  <ChainProvider
    chains={[neutronChain]}
    assetLists={[]}
    wallets={wallets}
    walletConnectOptions={wcOptions}
    walletConnect={walletconnect}
    signerOptions={{
      signingStargate: {
        preferredSignType: 'direct',
      },
    }}
  >
    {children}
  </ChainProvider>
);


// step:2 file: Enable mobile wallet support
import { WalletConnectOptions } from '@cosmos-kit/core';

export const walletConnectV2Config: WalletConnectOptions = {
  signClient: {
    projectId: process.env.NEXT_PUBLIC_WC_PROJECT_ID || '',
    relayUrl: 'wss://relay.walletconnect.com',
    metadata: {
      name: 'NeutronTemplate',
      description: 'Neutron Template WalletConnect Integration',
      url: 'https://your-dapp.com',
      icons: ['https://your-dapp.com/icon.png'],
    },
  },
  namespaces: {
    cosmos: {
      chains: ['cosmos:neutron-1'],
      methods: [
        'cosmos_getAccounts',
        'cosmos_signDirect',
        'cosmos_signAmino',
        'cosmos_sendTransaction',
      ],
      events: ['chainChanged', 'accountsChanged'],
    },
  },
};

// Mobile wallet IDs recognised by Cosmos Kit
export const supportedMobileWallets = [
  'keplr-mobile',
  'leap-mobile',
  'cosmostation-mobile',
];

// Helper that returns a ready-to-use mobile config bundle
export const getMobileWalletConfig = () => ({
  walletConnectOptions: walletConnectV2Config,
  wallets: supportedMobileWallets,
});


// step:3 file: Enable mobile wallet support
import React from 'react';
import { useWallet, WalletStatus } from '@cosmos-kit/react';
import { isMobile } from 'react-device-detect';

const shorten = (addr?: string) => (addr ? `${addr.slice(0, 6)}...${addr.slice(-4)}` : '');

export const ConnectWalletButton: React.FC = () => {
  const { connect, disconnect, status, address, viewWalletRepo } = useWallet();

  const handleClick = async () => {
    try {
      if (status === WalletStatus.Connected) {
        await disconnect();
        return;
      }

      if (isMobile) {
        // Mobile: open wallet list to launch deep-link
        viewWalletRepo();
      } else {
        // Desktop: opens Cosmos Kit QR modal automatically
        await connect();
      }
    } catch (err) {
      console.error('Wallet connect error:', err);
      alert(`Wallet connection failed: ${(err as Error).message}`);
    }
  };

  return (
    <button onClick={handleClick} className='px-4 py-2 rounded bg-indigo-600 text-white'>
      {status === WalletStatus.Connected ? shorten(address) : 'Connect Wallet'}
    </button>
  );
};


// step:4 file: Enable mobile wallet support
import { useEffect } from 'react';
import { useWallet } from '@cosmos-kit/react';

const WC_SESSION_KEY = 'ntrn_wc_session_v2';

export const usePersistWcSession = () => {
  const { client, status } = useWallet();

  // Save active session whenever it changes
  useEffect(() => {
    if (status === 'Connected' && client?.session) {
      try {
        localStorage.setItem(WC_SESSION_KEY, JSON.stringify(client.session));
      } catch (err) {
        console.warn('Unable to persist WalletConnect session', err);
      }
    }
  }, [client, status]);

  // Attempt to restore a previous session on first render
  useEffect(() => {
    const restore = async () => {
      try {
        const raw = localStorage.getItem(WC_SESSION_KEY);
        if (raw && client?.restoreSession && status !== 'Connected') {
          await client.restoreSession(JSON.parse(raw));
        }
      } catch (err) {
        console.error('Failed to restore WalletConnect session', err);
      }
    };

    restore();
  }, [client]);
};


