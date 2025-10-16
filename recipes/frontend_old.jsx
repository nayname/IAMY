export const ensureWalletConnected = async () => {
  try {
    const chainId = 'neutron-1';
    const keplr = window.keplr;

    // Execute the full workflow
    if (!keplr) {
      throw new Error('Keplr wallet is not installed.');
    }

    await keplr.enable(chainId);
    const signer = window.getOfflineSigner(chainId);

    return signer;

  } catch (err) {
    alert(err.message);
//     setError(err.message); // Update the error state
  }
};

export const getWalletAddress = async (signer) => {
  try {
      const accounts = await signer.getAccounts();
      if (!accounts || accounts.length === 0) {
        throw new Error('No account found in the signer.');
      }

      const address = accounts[0].address;
      return address;
  } catch (err) {
    alert(err.message);
//     setError(err.message); // Update the error state
  }
};


export const connectWallet = async (preferredWallet = 'keplr') => {
  /*
   * Attempt to detect and connect to the requested wallet extension.
   * Currently supports Keplr and Leap; extend this switch-case to add more wallets.
   */
  let wallet;
  switch (preferredWallet.toLowerCase()) {
    case 'keplr':
      wallet = window.keplr;
      break;
    case 'leap':
      wallet = window.leap;
      break;
    default:
      throw new Error(`${preferredWallet} wallet is not supported by this dApp.`);
  }

  if (!wallet) {
    throw new Error(`${preferredWallet} extension not found. Please install it and refresh the page.`);
  }

  try {
    // Ask the user to approve connection permissions (UI popup in the wallet).
    await wallet.enable('neutron-1');
    // Return an OfflineSigner required by CosmJS.
    return wallet.getOfflineSigner('neutron-1');
  } catch (err) {
    console.error('Wallet connection failed:', err);
    throw new Error('User rejected the wallet connection request or another error occurred.');
  }
};

export const ensureNeutronNetwork = async () => {
  const chainId = 'neutron-1';
  const keplr = window.keplr || window.leap;
  if (!keplr) throw new Error('No compatible wallet detected.');

  try {
    // First try to enable Neutron if it already exists in the wallet.
    await keplr.enable(chainId);
    return true;
  } catch (enableErr) {
    console.warn('Neutron chain not yet added in the wallet, attempting experimentalSuggestChain');

    // Fallback: suggest chain (only works if wallet supports the experimental API).
    if (!keplr.experimentalSuggestChain) {
      throw new Error('Wallet does not support chain suggestions. Please add Neutron manually.');
    }

    // Minimal and up-to-date Neutron chain configuration.
    const neutronChainInfo = {
      chainId,
      chainName: 'Neutron',
      rpc: 'https://rpc-kralum.neutron.org',
      rest: 'https://api-kralum.neutron.org',
      bip44: { coinType: 118 },
      bech32Config: {
        bech32PrefixAccAddr: 'neutron',
        bech32PrefixAccPub: 'neutronpub',
        bech32PrefixValAddr: 'neutronvaloper',
        bech32PrefixValPub: 'neutronvaloperpub',
        bech32PrefixConsAddr: 'neutronvalcons',
        bech32PrefixConsPub: 'neutronvalconspub'
      },
      currencies: [{ coinDenom: 'NTRN', coinMinimalDenom: 'untrn', coinDecimals: 6 }],
      feeCurrencies: [{ coinDenom: 'NTRN', coinMinimalDenom: 'untrn', coinDecimals: 6 }],
      stakeCurrency: { coinDenom: 'NTRN', coinMinimalDenom: 'untrn', coinDecimals: 6 },
      gasPriceStep: { low: 0.01, average: 0.025, high: 0.04 }
    };

    try {
      await keplr.experimentalSuggestChain(neutronChainInfo);
      // Chain suggested successfully; enable it now.
      await keplr.enable(chainId);
      return true;
    } catch (suggestErr) {
      console.error('Failed to suggest Neutron chain:', suggestErr);
      throw new Error('Unable to add Neutron network automatically. Please add it to your wallet manually.');
    }
  }
};

export const storeSessionAccount = async (signer) => {
  if (!signer) throw new Error('Signer instance is required.');

  // CosmJS signers expose getAccounts() which returns an array of accounts.
  const accounts = await signer.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error('No accounts found in the signer.');
  }

  const { address, pubkey } = accounts[0];

  const pubkeyBase64 = btoa(String.fromCharCode.apply(null, pubkey));

  const accountInfo = {
    address,
    pubkey: pubkeyBase64 // Use the browser-safe Base64 string
  };

  try {
    // Persist to the browser session (cleared on tab close).
    sessionStorage.setItem('neutron_account', JSON.stringify(accountInfo));
    return accountInfo;
  } catch (err) {
    console.error('Failed to write account info to sessionStorage:', err);
    throw new Error('Unable to store account data locally.');
  }
};

/**
 * @file This file contains a curated set of self-contained, vanilla JavaScript functions
 * for interacting with a web-based blockchain application.
 *
 * It has been cleaned of duplicates and functions that require external NPM libraries
 * (like @cosmjs), making it suitable for static environments. Redundant implementations
 * have been removed, leaving one canonical version of each function.
 *
 * For complex operations like querying the chain or broadcasting transactions,
 * this file adopts a backend-for-frontend pattern. The functions make calls
 * to a backend API, and comments describe the expected implementation of those endpoints.
 */

// ===================================================================================
// == Core Wallet & User Interaction (Vanilla JS)
// ===================================================================================

/**
 * Connects to a browser wallet (Keplr or Leap) and returns the signer and address.
 * This is the canonical version, replacing multiple redundant implementations.
 * @param {string} [chainId='neutron-1'] - The identifier of the chain to connect to.
 * @returns {Promise<{address: string, signer: object}>} A promise that resolves to an object
 * containing the user's bech32 address and the offline signer.
 * @throws {Error} If a wallet is not installed or the user denies the connection.
 */
export const getOfflineSignerAndAddress = async (chainId = 'neutron-1') => {
    if (typeof window === 'undefined') {
        throw new Error('This function must be run in a browser.');
    }
    const wallet = window.keplr || window.leap;
    if (!wallet) {
        throw new Error('Keplr or Leap wallet is not installed.');
    }
    await wallet.enable(chainId);
    const signer = wallet.getOfflineSigner(chainId);
    const accounts = await signer.getAccounts();
    if (!accounts || accounts.length === 0) {
        throw new Error('No accounts found in the connected wallet.');
    }
    return {
        address: accounts[0].address,
        signer,
    };
};

/**
 * Loads a contract address from environment variables.
 * @returns {string} The contract address.
 * @throws {Error} If the address is not defined or has an invalid format.
 */
export const loadContractAddress = () => {
    const address =
        import.meta.env.VITE_TEMPLATE_CONTRACT_ADDRESS ||
        process.env.NEXT_PUBLIC_TEMPLATE_CONTRACT_ADDRESS;
    if (!address) {
        throw new Error(
            'Contract address is not defined in environment variables.'
        );
    }
    if (!/^neutron1[0-9a-z]{38}$/.test(address)) {
        throw new Error('Invalid Neutron contract address format.');
    }
    return address;
};

/**
 * Gets a contract address from a DOM input element.
 * @param {string} [elementId='contract-address-input'] - The ID of the input element.
 * @returns {string} The trimmed contract address from the input value.
 * @throws {Error} If the element is not found or the input is empty.
 */
export const getContractAddress = (elementId = 'contract-address-input') => {
    const inputEl = document.getElementById(elementId);
    if (!inputEl) {
        throw new Error(`Element with id "${elementId}" not found in the DOM.`);
    }
    const address = inputEl.value.trim();
    if (!address) {
        throw new Error('Contract address cannot be empty.');
    }
    return address;
};

// ===================================================================================
// == Blockchain Interaction (Delegated to Backend)
// ===================================================================================

/**
 * Queries a smart contract by sending the request to a secure backend endpoint.
 * @param {string} contractAddress - The bech32 address of the contract.
 * @param {object} queryMsg - The JSON query message for the contract.
 * @returns {Promise<any>} The JSON response from the contract.
 */
export const queryContractSmart = async (contractAddress, queryMsg) => {
    const response = await fetch('/api/query-contract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contractAddress, query: queryMsg }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to query contract.');
    }
    return response.json();
    /*
     * == BACKEND IMPLEMENTATION NOTE (/api/query-contract) ==
     *
     * 1. The backend receives `{ contractAddress, query }` in the request body.
     * 2. It uses `@cosmjs/cosmwasm-stargate`'s `CosmWasmClient.connect(RPC_ENDPOINT)`.
     * 3. It calls `client.queryContractSmart(contractAddress, query)`.
     * 4. It returns the result as JSON to the frontend.
     */
};

/**
 * Validates a bech32 address using a backend endpoint.
 * @param {string} address - The address to validate.
 * @returns {Promise<boolean>} A promise that resolves to true if the address is valid.
 * @throws {Error} If the backend reports the address is invalid.
 */
export const validateAddressFormat = async (address) => {
    const response = await fetch('/api/validate-address', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address }),
    });
    const result = await response.json();
    if (!response.ok || !result.isValid) {
        throw new Error(result.message || 'Invalid address.');
    }
    return true;
    /*
     * == BACKEND IMPLEMENTATION NOTE (/api/validate-address) ==
     *
     * 1. The backend receives `{ address }` in the request body.
     * 2. It uses the `bech32` or `@cosmjs/encoding` library to decode the address.
     * 3. It checks for decoding errors and verifies the bech32 prefix (e.g., 'neutron').
     * 4. It returns `{ isValid: true }` or `{ isValid: false, message: '...' }`.
     */
};

/**
 * Sends a pre-signed transaction to a backend relayer for broadcasting.
 * @param {object} signer - The OfflineSigner from `getOfflineSignerAndAddress`.
 * @param {string} senderAddress - The sender's bech32 address.
 * @param {Array<object>} messages - An array of message objects for the transaction.
 * @param {object|string} fee - The fee object or "auto".
 * @param {string} [memo=''] - An optional memo for the transaction.
 * @returns {Promise<string>} The transaction hash.
 */
export const signAndBroadcast = async (signer, senderAddress, messages, fee, memo = '') => {
    // NOTE: A real implementation requires a library like @cosmjs/stargate to sign.
    // This function demonstrates the pattern of signing on the client and sending
    // the signed bytes to a backend for broadcasting.
    const signedTxBytes = await "/* (Use a library like @cosmjs/stargate to create signed transaction bytes here) */";

    const response = await fetch('/api/broadcast-tx', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signedTxBytes }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to broadcast transaction.');
    }

    const result = await response.json();
    return result.transactionHash;
    /*
     * == BACKEND IMPLEMENTATION NOTE (/api/broadcast-tx) ==
     *
     * 1. The backend receives the raw, signed transaction bytes.
     * 2. It connects to an RPC endpoint using `StargateClient.connect(RPC_ENDPOINT)`.
     * 3. It calls `client.broadcastTx(signedTxBytes)` to submit the transaction.
     * 4. It returns `{ transactionHash: '...' }` on success or an error message on failure.
     */
};


// ===================================================================================
// == Message Constructors & Utility Helpers (Vanilla JS)
// ===================================================================================

/**
 * Constructs a query message object for a CosmWasm smart contract.
 * @param {string} senderAddress - The bech32 address for the query, if required.
 * @returns {object} A query message object.
 */
export const constructWasmQueryMsg = (senderAddress) => {
    // This example is specific to the `get_personal_counter` query.
    // In a real app, you might have multiple, more specific constructors.
    if (!senderAddress) {
        return { get_global_counter: {} };
    }
    return {
        get_personal_counter: { address: senderAddress },
    };
};

/**
 * Constructs an execute message object for a CosmWasm smart contract.
 * @param {string} senderAddress - The sender's address.
 * @param {string} contractAddress - The contract's address.
 * @param {object} msg - The core message payload.
 * @param {Array<object>} [funds=[]] - Any funds to attach to the message.
 * @returns {object} An execute message object.
 */
export const constructTxWasmExecute = (senderAddress, contractAddress, msg, funds = []) => {
    // This function returns a generic structure. The specific `msg` payload
    // would be created separately, e.g., `{ deposit: {} }`.
    return {
        typeUrl: '/cosmwasm.wasm.v1.MsgExecuteContract',
        value: {
            sender: senderAddress,
            contract: contractAddress,
            msg: new TextEncoder().encode(JSON.stringify(msg)),
            funds: funds,
        },
    };
};

/**
 * Converts a human-readable token amount to its smallest denomination (base units).
 * @param {string|number} amount - The amount of tokens to convert.
 * @param {number} [decimals=6] - The number of decimal places for the token.
 * @returns {string} The amount in its smallest unit as a string.
 */
export const convertToBaseUnits = (amount, decimals = 6) => {
    const numericAmount = Number(amount);
    if (!Number.isFinite(numericAmount) || numericAmount <= 0) {
        throw new Error('Amount must be a positive number.');
    }
    const factor = 10 ** decimals;
    return String(Math.floor(numericAmount * factor));
};

/**
 * Prompts the user's wallet to add the Neutron chain configuration.
 * @param {object} wallet - The wallet object from `window.keplr` or `window.leap`.
 */
export const suggestNeutronChain = async (wallet) => {
    if (!wallet || !wallet.experimentalSuggestChain) {
        throw new Error('Wallet does not support suggesting new chains.');
    }
    const chainConfig = {
        chainId: 'neutron-1',
        chainName: 'Neutron',
        rpc: 'https://rpc-kralum.neutron-1.neutron.org',
        rest: 'https://rest-kralum.neutron-1.neutron.org',
        bip44: { coinType: 118 },
        bech32Config: { bech32PrefixAccAddr: 'neutron' },
        currencies: [{ coinDenom: 'NTRN', coinMinimalDenom: 'untrn', coinDecimals: 6 }],
        feeCurrencies: [{ coinDenom: 'NTRN', coinMinimalDenom: 'untrn', coinDecimals: 6 }],
        stakeCurrency: { coinDenom: 'NTRN', coinMinimalDenom: 'untrn', coinDecimals: 6 },
    };
    await wallet.experimentalSuggestChain(chainConfig);
};


// // ===================================================================================
// // == Not Implemented (UI Components / Hooks)
// // ===================================================================================
//
// /**
//  * Placeholder for a React Context Provider for wallet state management.
//  * Original implementation used @cosmos-kit/react.
//  */
// export const WalletProvider = () => {
//   alert('Function is not implemented');
// };
//
// /**
//  * Placeholder for a React button component to connect/disconnect a wallet.
//  * Original implementation used @cosmos-kit/react.
//  */
// export const ConnectWalletButton = () => {
//   alert('Function is not implemented');
// };
//
// /**
//  * Placeholder for a React Hook to persist WalletConnect sessions.
//  * Original implementation used @cosmos-kit/react.
//  */
// export const usePersistWcSession = () => {
//   alert('Function is not implemented');
// };


// ===================================================================================
// == BTC
// ===================================================================================

// step:1 file: increase_the_user’s_deposit_in_the_wbtc_usdc_supervault_by_0.2_wbtc_and_12_000_usdc
/* src/utils/wallet.js */
export const getUserWalletAddress = async () => {
  const chainId = 'neutron-1';

  // 1. Ensure Keplr is injected in the browser
  if (!window.keplr) {
    throw new Error('Keplr wallet not found. Please install the Keplr browser extension.');
  }

  // 2. Ask Keplr to enable the Neutron chain
  await window.keplr.enable(chainId);

  // 3. Retrieve the OfflineSigner and account list
  const signer = window.getOfflineSigner(chainId);
  const accounts = await signer.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No accounts detected for the selected chain.');
  }

  // 4. Return the first account address (default behaviour for Keplr)
  return accounts[0].address;
};


// step:1 file: redeem_lp_shares_from_the_maxbtc_ebtc_supervault
export const getUserAddress = async (chainId = 'neutron-1') => {
  // Check that Keplr is available in the browser
  if (!window.keplr) {
    throw new Error('Keplr wallet not found. Please install or unlock the Keplr browser extension.');
  }

  // Ask Keplr to enable the target chain (this may prompt the user)
  await window.keplr.enable(chainId);

  // Obtain an OfflineSigner instance for the chain
  const signer = window.getOfflineSigner(chainId);
  const accounts = await signer.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No accounts detected in Keplr for the selected chain.');
  }

  // Return signer (for later use) and address
  return {
    signer,
    address: accounts[0].address,
  };
};


// step:2 file: redeem_lp_shares_from_the_maxbtc_ebtc_supervault
export const queryShareBalance = async (restEndpoint, contractAddress, userAddress) => {
  // The exact query key ("balance") should match the Supervault contract’s API.
  const queryPayload = { "balance": { "address": userAddress } };

  // CosmWasm REST endpoints expect the query JSON to be base64-encoded.
  const base64Query = btoa(JSON.stringify(queryPayload));
  const url = `${restEndpoint}/cosmwasm/wasm/v1/contract/${contractAddress}/smart/${base64Query}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Contract query failed: ${response.status} ${response.statusText}`);
  }

  const { data } = await response.json();
  // Assume the contract returns `{ balance: "<amount>" }`. Adjust as needed.
  return data?.balance || '0';
};


// step:3 file: redeem_lp_shares_from_the_maxbtc_ebtc_supervault
export const validateRedeemAmount = (requestedAmount, availableShares) => {
  const req = BigInt(requestedAmount);
  const avail = BigInt(availableShares);

  if (req <= 0n) {
    throw new Error('Redeem amount must be greater than zero.');
  }
  if (req > avail) {
    throw new Error('Redeem amount exceeds the available share balance.');
  }
  // Validation successful
  return true;
};


// step:1 file: bridge_1_wbtc_from_ethereum_to_neutron
/* connectEthWallet.js */
export const connectEthWallet = async () => {
  // --- Constants -----------------------------------------------------------
  const MIN_ETH_WEI = 10n ** 16n;            // ≈ 0.01 ETH for gas
  const MIN_WBTC_SATS = 100000000n;          // 1 WBTC (8 dp)
  const WBTC_ADDRESS = '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599'; // main-net
  const BALANCE_OF_SELECTOR = '0x70a08231';  // keccak256('balanceOf(address)')[0:4]

  // --- Pre-checks ----------------------------------------------------------
  if (typeof window === 'undefined' || !window.ethereum) {
    throw new Error('MetaMask (or compatible) wallet is not installed.');
  }

  // --- Request account -----------------------------------------------------
  const [account] = await window.ethereum.request({ method: 'eth_requestAccounts' });
  if (!account) throw new Error('No Ethereum account returned by MetaMask.');

  // --- Check ETH balance ---------------------------------------------------
  const ethBalanceHex = await window.ethereum.request({
    method: 'eth_getBalance',
    params: [account, 'latest']
  });
  const ethBalanceWei = BigInt(ethBalanceHex);
  if (ethBalanceWei < MIN_ETH_WEI) {
    throw new Error('Insufficient ETH for gas (need at least ≈0.01 ETH).');
  }

  // --- Check WBTC balance --------------------------------------------------
  const paddedAcct = account.slice(2).padStart(64, '0');
  const data = BALANCE_OF_SELECTOR + paddedAcct;
  const wbtcBalanceHex = await window.ethereum.request({
    method: 'eth_call',
    params: [{ to: WBTC_ADDRESS, data }, 'latest']
  });
  const wbtcBalance = BigInt(wbtcBalanceHex);
  if (wbtcBalance < MIN_WBTC_SATS) {
    throw new Error('At least 1 WBTC is required to continue.');
  }

  // --- Return account details ---------------------------------------------
  return { account, wbtcBalance: wbtcBalance.toString() };
};


// step:2 file: bridge_1_wbtc_from_ethereum_to_neutron
/* approveErc20Spend.js */
export const approveErc20Spend = async ({ ownerAddress, bridgeAddress, amountSats = '100000000' }) => {
  const WBTC_ADDRESS = '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599';
  const APPROVE_SELECTOR = '0x095ea7b3'; // keccak256('approve(address,uint256)')[0:4]

  // --- Encode parameters ---------------------------------------------------
  const spenderPadded = bridgeAddress.slice(2).padStart(64, '0');
  const amountHex = BigInt(amountSats).toString(16).padStart(64, '0');
  const data = APPROVE_SELECTOR + spenderPadded + amountHex;

  // --- Send tx via MetaMask -------------------------------------------------
  const txHash = await window.ethereum.request({
    method: 'eth_sendTransaction',
    params: [{
      from: ownerAddress,
      to: WBTC_ADDRESS,
      data,
      value: '0x0'
    }]
  });

  return txHash; // user can track this tx for confirmation
};


// step:3 file: bridge_1_wbtc_from_ethereum_to_neutron
/* depositWbtcToBridge.js */
export const depositWbtcToBridge = async ({
  ownerAddress,
  bridgeAddress,
  neutronAddress,
  amountSats = '100000000'
}) => {
  /*
    NOTE: Every bridge has its own ABI.
    Adjust `DEPOSIT_SELECTOR` and encoding if your bridge differs.
    Example ABI (pseudo):
      function deposit(address token, uint256 amount, bytes destination) external;
    Keccak-256 selector => 0xb6b55f25 (placeholder here).
  */
  const TOKEN_ADDRESS = '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599';
  const DEPOSIT_SELECTOR = '0xb6b55f25'; // placeholder selector – update to real one!

  // --- Encode parameters ---------------------------------------------------
  const pad = (hex, bytes = 64) => hex.replace(/^0x/, '').padStart(bytes, '0');

  const tokenParam   = pad(TOKEN_ADDRESS);
  const amountParam  = pad(BigInt(amountSats).toString(16));

  // Destination (Neutron bech32) converted to raw UTF-8 hex -----------------
  const destUtf8Hex  = Buffer.from(neutronAddress, 'utf8').toString('hex');
  const destLen      = pad(Number(destUtf8Hex.length / 2).toString(16));
  const destParam    = destUtf8Hex.padEnd(64, '0'); // right-pad to 32B

  const data = DEPOSIT_SELECTOR + tokenParam + amountParam + destLen + destParam;

  // --- Send tx via MetaMask -------------------------------------------------
  const txHash = await window.ethereum.request({
    method: 'eth_sendTransaction',
    params: [{
      from: ownerAddress,
      to: bridgeAddress,
      data,
      value: '0x0'
    }]
  });

  return txHash;
};


// step:2 file: opt_in_to_partner_airdrops_for_my_vault_deposits
export const getVaultContractAddress = () => {
  // In production you might fetch this from an API or .env file.
  // Hard-coded here for demo purposes.
  return 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx';
};


// step:3 file: opt_in_to_partner_airdrops_for_my_vault_deposits
export const buildOptInAirdropsMsg = (partnerId = 'all') => {
  return {
    opt_in_airdrops: {
      partner_id: partnerId
    }
  };
};


// step:5 file: opt_in_to_partner_airdrops_for_my_vault_deposits
export const queryAirdropStatus = async (
  contractAddress,
  userAddress,
  lcdEndpoint = 'https://rest-kralum.neutron-1.neutron.org'
) => {
  // Build the query `{ airdrop_status: { address: <USER_ADDR> } }`
  const query = {
    airdrop_status: {
      address: userAddress,
    },
  };

  // The LCD expects the query message to be base64-encoded
  const base64Query = btoa(JSON.stringify(query));

  const url = `${lcdEndpoint}/cosmwasm/wasm/v1/contract/${contractAddress}/smart/${base64Query}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`LCD query failed with status ${response.status}`);
  }

  const result = await response.json();
  return result.data; // `data` holds the smart-query response
};


// step:3 file: check_my_health_factor_on_amber_finance
/*
 * positions: Array<{ id: string | number, collateral: string, debt: string, health_factor?: string }>
 * All monetary fields are expected in micro-denom (e.g. `untrn`).
 */
export const calculateHealthFactor = (positions) => {
  if (!Array.isArray(positions)) {
    throw new Error('Invalid positions array received.');
  }

  return positions.map((p) => {
    // Attempt to use the pre-computed value first
    if (p.health_factor !== undefined) {
      return {
        id: p.id,
        collateral: Number(p.collateral),
        debt: Number(p.debt),
        healthFactor: Number(p.health_factor)
      };
    }

    const collateral = Number(p.collateral);
    const debt = Number(p.debt);

    // Protect against division by zero
    const healthFactor = debt === 0 ? Infinity : collateral / debt;

    return {
      id: p.id,
      collateral,
      debt,
      healthFactor
    };
  });
};


// step:4 file: check_my_health_factor_on_amber_finance
export const presentResults = (computedPositions) => {
  if (!Array.isArray(computedPositions)) {
    throw new Error('Expected an array from calculateHealthFactor().');
  }

  return computedPositions.map((p) => {
    const fmt = (v) => (v / 1e6).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const hf   = p.healthFactor === Infinity ? '∞' : p.healthFactor.toFixed(2);

    return `Position #${p.id} → HF: ${hf}, Collateral: ${fmt(p.collateral)} NTRN, Debt: ${fmt(p.debt)} NTRN`;
  }).join('\n');
};


// step:2 file: lock_2000_ntrn_for_3_months_to_obtain_a_1.2×_btc_summer_boost
export const fetchNtrnBalance = async (address) => {
  // Public LCD endpoint — replace with your preferred endpoint or a proxy if needed
  const LCD = 'https://lcd-neutron.blockpane.com';
  const denom = 'untrn';

  try {
    const res = await fetch(`${LCD}/cosmos/bank/v1beta1/balances/${address}`);
    if (!res.ok) {
      throw new Error(`LCD error: ${res.status} ${res.statusText}`);
    }
    const data = await res.json();

    /* The response shape is:
        {
          "balances": [ { "denom": "untrn", "amount": "12345" }, ... ],
          ...
        }
    */
    const coin = (data.balances || []).find((c) => c.denom === denom);
    const amount = coin ? Number(coin.amount) : 0;
    return amount; // returns micro-denom amount (e.g. 2 000 000 000 for 2 000 NTRN)
  } catch (err) {
    console.error('[fetchNtrnBalance] ', err);
    throw err;
  }
};


// step:3 file: lock_2000_ntrn_for_3_months_to_obtain_a_1.2×_btc_summer_boost
export const validateLockAmount = (rawBalance, amountToLock = 2_000_000_000) => {
  if (rawBalance < amountToLock) {
    throw new Error('Insufficient spendable NTRN balance (need ≥ 2,000 NTRN).');
  }
  /*
    NOTE: Detecting whether funds are already vested or locked normally requires
    contract-specific queries that are out of scope for a client-side snippet.
    For simple front-end validation we only check spendable balance.
  */
  return true;
};


// step:4 file: lock_2000_ntrn_for_3_months_to_obtain_a_1.2×_btc_summer_boost
export const calculateUnlockTimestamp = () => {
  const NOW_SEC = Math.floor(Date.now() / 1000); // JS Date gives ms
  const LOCK_DURATION = 7_776_000; // 90 days in seconds
  return NOW_SEC + LOCK_DURATION;
};


// step:5 file: lock_2000_ntrn_for_3_months_to_obtain_a_1.2×_btc_summer_boost
export const constructLockExecuteMsg = ({ sender, amount = '2000000000', durationSeconds = 7_776_000 }) => {
  if (!sender) throw new Error('`sender` is required');

  const executeMsg = {
    lock: {
      duration_seconds: durationSeconds.toString()
    }
  };

  return {
    contract_address: 'neutron14lnmj4k0tqsfn3x8kmnmacg64ct2utyz0aaxtm5g3uwwp8kk4f6shcgrtt',
    sender,
    msg: executeMsg,
    funds: [
      {
        denom: 'untrn',
        amount: amount.toString()
      }
    ]
  };
};


// step:7 file: lock_2000_ntrn_for_3_months_to_obtain_a_1.2×_btc_summer_boost
export const queryBoostMultiplier = async (address) => {
  const BOOST_POINTER_CONTRACT = 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxx'; // TODO: replace with real addr
  const queryMsg = {
    multiplier: {
      address
    }
  };

  // LCD expects the smart-query to be Base64-encoded
  const base64Query = btoa(JSON.stringify(queryMsg));
  const LCD = 'https://lcd-neutron.blockpane.com';

  try {
    const url = `${LCD}/cosmwasm/wasm/v1/contract/${BOOST_POINTER_CONTRACT}/smart/${base64Query}`;
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`LCD error: ${res.status} ${res.statusText}`);
    }
    const data = await res.json();
    /* Expected shape (example):
        {
          "data": {
            "multiplier": "1.25"
          }
        }
    */
    return data.data?.multiplier ?? null;
  } catch (err) {
    console.error('[queryBoostMultiplier] ', err);
    throw err;
  }
};


// step:4 file: close_my_leveraged_loop_position_on_amber
// File: src/utils/amber.js
import { getUserAddress as getUserAddress_1 } from './wallet';

// Helper — base64 → Uint8Array
const b64ToUint8 = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

export const signAndBroadcastClosePosition = async ({
  chainId           = 'neutron-1',
  signDocBase64,               // from step 3
  backendBroadcastUrl = '/api/amber/broadcast_signed_tx'
}) => {
  try {
    const address     = await getUserAddress_1(chainId);
    const signDocBytes = b64ToUint8(signDocBase64);

    // Keplr — sign the SignDoc using signDirect
    const { signed, signature } = await window.keplr.signDirect(
      chainId,
      address,
      { typeUrl: '/cosmos.tx.v1beta1.SignDoc', value: signDocBytes }
    );

    // Convert binary blobs → base64 so they can be sent over HTTP
    const bodyB64       = btoa(String.fromCharCode(...signed.bodyBytes));
    const authInfoB64   = btoa(String.fromCharCode(...signed.authInfoBytes));
    const sigB64        = btoa(String.fromCharCode(...signature.signature));

    const res = await fetch(backendBroadcastUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        body_bytes: bodyB64,
        auth_info_bytes: authInfoB64,
        signatures: [sigB64]
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Broadcast failed');
    }

    return await res.json(); // { txhash, height, ... }
  } catch (err) {
    console.error('[signAndBroadcastClosePosition] error:', err);
    throw err;
  }
};


// step:5 file: retrieve_projected_ntrn_rewards_based_on_current_point_total
export const fetchProjectionAndDisplay = async (address) => {
  try {
    const res = await fetch(`/api/projection?address=${address}`);
    if (!res.ok) {
      throw new Error(`Backend responded with status ${res.status}`);
    }

    const data = await res.json();
    const { points, projected_reward_ntrn, assumptions } = data;

    const message = `With ${points} points and a per-point rate of ${assumptions.per_point_rate / 1_000_000} NTRN, you are projected to earn ≈ ${projected_reward_ntrn} NTRN this phase.`;

    // Display the message however your UI prefers. Here we log to console.
    console.log(message);
    return message;
  } catch (error) {
    console.error('Failed to fetch projection:', error);
    return 'Unable to compute projection at this time.';
  }
};


// step:4 file: swap_1_ebtc_for_unibtc_on_neutron_dex
export const constructSwapMsg = ({
  sender,
  contractAddress,
  offerDenom = 'eBTC',
  offerAmount = '1000000', // 1 eBTC in micro-units
  askDenom = 'uniBTC',
  maxSlippage = '0.005' // 0.5%
}) => {
  // Astroport-style swap execute message
  const execMsg = {
    swap: {
      offer_asset: {
        info: { native_token: { denom: offerDenom } },
        amount: offerAmount
      },
      max_slippage: maxSlippage
    }
  };

  // Construct the protobuf-ready envelope that cosmpy will later consume
  return {
    typeUrl: '/cosmwasm.wasm.v1.MsgExecuteContract',
    value: {
      sender,
      contract: contractAddress,
      msg: btoa(JSON.stringify(execMsg)), // base64-encoded JSON for CosmWasm
      funds: [
        { denom: offerDenom, amount: offerAmount }
      ]
    }
  };
};


// step:2 file: set_my_boost_target_to_my_ethereum_address
export const getUserEvmAddressInput = () => {
  const input = prompt('Enter the destination Ethereum (EVM) address (0x…)');
  if (!input) {
    throw new Error('No Ethereum address supplied by user.');
  }
  return input.trim();
};


// step:3 file: set_my_boost_target_to_my_ethereum_address
export const validateEthereumAddress = (evmAddress) => {
  const regex = /^0x[a-fA-F0-9]{40}$/;
  if (!regex.test(evmAddress)) {
    throw new Error('Invalid Ethereum address format.');
  }
  return true;
};


// step:4 file: set_my_boost_target_to_my_ethereum_address
export const constructSetTargetMsg = ({
  contractAddress,
  senderAddress,
  evmAddress,
}) => {
  // Contract-level JSON payload
  const payload = { set_target: { evm_address: evmAddress } };

  // CosmWasm execute envelope
  return {
    typeUrl: '/cosmwasm.wasm.v1.MsgExecuteContract',
    value: {
      sender: senderAddress,
      contract: contractAddress,
      msg: Array.from(
        new TextEncoder().encode(JSON.stringify(payload))
      ),
      funds: [],
    },
  };
};


// step:6 file: set_my_boost_target_to_my_ethereum_address
export const queryBoostTarget = async (contractAddress) => {
  try {
    const queryMsg = { target: {} };
    const encoded = btoa(JSON.stringify(queryMsg)); // base64-encode the query JSON

    const endpoint = `https://rest-kralum.neutron.org/cosmwasm/wasm/v1/contract/${contractAddress}/smart/${encoded}`;
    const resp = await fetch(endpoint);

    if (!resp.ok) {
      throw new Error(`Query failed with ${resp.status}: ${resp.statusText}`);
    }

    const result = await resp.json();
    return result; // Expected shape: { data: { evm_address: '0x...' } }
  } catch (err) {
    console.error(err);
    throw err;
  }
};


// step:1 file: show_my_total_bitcoin_summer_points_earned_in_the_current_phase
export const getNeutronAddress = async () => {
  const chainId = 'neutron-1';

  // Detect a compatible browser extension
  const wallet = window.keplr || window.leap;
  if (!wallet) {
    throw new Error('No supported Neutron wallet extension (Keplr or Leap) found.');
  }

  // Request connection to the chain
  try {
    await wallet.enable(chainId);
  } catch (error) {
    throw new Error(`Wallet connection rejected or chain not supported: ${error.message}`);
  }

  // Obtain signer and account list
  const offlineSigner = wallet.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error('Unable to fetch an account from the wallet signer.');
  }

  // Return the first available address
  return accounts[0].address;
};


// step:4 file: show_my_total_bitcoin_summer_points_earned_in_the_current_phase
export const displayPoints = (points) => {
  // Locate—or create—the DOM element for displaying points
  let container = document.getElementById('points-display');
  if (!container) {
    container = document.createElement('div');
    container.id = 'points-display';
    document.body.appendChild(container);
  }
  container.textContent = `You have ${points} point${points === 1 ? '' : 's'} in the current campaign phase.`;
};


// step:4 file: list_current_amber_lending_markets_and_apys
export const SECONDS_PER_YEAR = 60 * 60 * 24 * 365;

export const rateToAPY = (ratePerSecond) => {
  const r = Number(ratePerSecond);
  if (isNaN(r)) {
    throw new Error('rateToAPY received an invalid number');
  }
  const apy = (Math.pow(1 + r, SECONDS_PER_YEAR) - 1) * 100; // convert to %
  return Number(apy.toFixed(2));
};


// step:5 file: list_current_amber_lending_markets_and_apys
import React, { useEffect, useState } from 'react';
import { rateToAPY as rateToAPY_1 } from './rateToAPY';

const MarketTable = () => {
  const [markets, setMarkets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const env = 'mainnet';

        // Step 1: controller address (validates backend availability)
        const addrRes = await fetch(`/api/amber/controller-address?env=${env}`);
        if (!addrRes.ok) throw new Error(await addrRes.text());
        await addrRes.json();

        // Step 2: market list
        const marketsRes = await fetch(`/api/amber/markets?env=${env}`);
        if (!marketsRes.ok) throw new Error(await marketsRes.text());
        const marketList = await marketsRes.json();

        // Step 3: for each market fetch state in parallel
        const enriched = await Promise.all(
          marketList.map(async (m) => {
            const stateRes = await fetch(`/api/amber/market-state?env=${env}&market_id=${m.id}`);
            if (!stateRes.ok) throw new Error(await stateRes.text());
            const state = await stateRes.json();

            return {
              id: m.id,
              symbol: m.symbol,
              collateralFactor: Number(m.collateral_factor),
              supplyAPY: rateToAPY_1(state.supply_rate_per_second),
              borrowAPY: rateToAPY_1(state.borrow_rate_per_second)
            };
          })
        );

        setMarkets(enriched);
      } catch (e) {
        console.error(e);
        setError(e.message || 'Could not load market data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <p>Loading markets…</p>;
  if (error) return <p style={{ color: 'red' }}>{error}</p>;

  return (
    <table>
      <thead>
        <tr>
          <th>Symbol</th>
          <th>Collateral Factor</th>
          <th>Supply APY (%)</th>
          <th>Borrow APY (%)</th>
        </tr>
      </thead>
      <tbody>
        {markets.map((m) => (
          <tr key={m.id}>
            <td>{m.symbol}</td>
            <td>{(m.collateralFactor * 100).toFixed(0)}%</td>
            <td>{m.supplyAPY}</td>
            <td>{m.borrowAPY}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default MarketTable;


// step:1 file: enable_usdc_gas_payments_for_my_next_transaction
/*
 * utils/fees.js
 * Checks whether a denom is present in `ntrn_prices` via the `/neutron/dynamicfees/params` REST endpoint.
 */
export const REST_ENDPOINT = "https://rest-kralum.neutron-1.neutron.org"; // Replace with your preferred REST endpoint

export const isFeeDenomEligible = async (denom = "uusdc", restEndpoint = REST_ENDPOINT) => {
  try {
    const res = await fetch(`${restEndpoint}/neutron/dynamicfees/params`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const json = await res.json();
    const prices = json?.params?.ntrn_prices ?? [];
    const eligible = prices.some((d) => d.denom === denom);

    if (!eligible) {
      throw new Error(`${denom} is not found in ntrn_prices ‑ it cannot be used to pay fees.`);
    }

    return {
      eligible: true,
      raw: json
    };
  } catch (err) {
    console.error("Dynamic-fees query failed", err);
    throw err;
  }
};


// step:2 file: enable_usdc_gas_payments_for_my_next_transaction
/*
 * utils/fees.js (continued)
 * Returns the min-gas-price (as a string) for a given denom.
 */
export const getMinGasPrice = async (denom = "uusdc", restEndpoint = REST_ENDPOINT) => {
  try {
    const res = await fetch(`${restEndpoint}/neutron/globalfee/min_gas_prices`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const json = await res.json(); // [{ denom: "untrn", amount: "0.015" }, ...]
    const entry = (json || []).find((e) => e.denom === denom);
    if (!entry) throw new Error(`No gas-price entry for denom ${denom}`);

    return entry.amount; // string, e.g. "0.07"
  } catch (err) {
    console.error("Global-fee query failed", err);
    throw err;
  }
};


// step:3 file: enable_usdc_gas_payments_for_my_next_transaction
/*
 * utils/fees.js (continued)
 * Stores the chosen fee denom in `localStorage`. The UI can read this value when constructing txs.
 */
export const setDefaultFeeDenom = (denom = "uusdc") => {
  try {
    localStorage.setItem("NEUTRON_FEE_DENOM", denom);
  } catch (err) {
    console.warn("Unable to write NEUTRON_FEE_DENOM to localStorage", err);
  }
};


// step:2 file: instantly_claim_50%_of_my_ntrn_staking_rewards
export const queryPendingStakingRewards = async (
  delegatorAddress,
  restEndpoint = 'https://rest-kralum.neutron-1.neutron.org'
) => {
  const url = `${restEndpoint}/cosmos/distribution/v1beta1/delegators/${delegatorAddress}/rewards`;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Distribution query failed :: ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    console.error('[Query-Rewards] ::', error);
    throw error;
  }
};


// step:3 file: instantly_claim_50%_of_my_ntrn_staking_rewards
export const calculatePartialRewards = (
  rewardsResponse,
  fraction = 0.5,
  denom = 'untrn'
) => {
  if (!rewardsResponse || !Array.isArray(rewardsResponse.rewards)) {
    throw new Error('Malformed rewards response.');
  }

  const partial = rewardsResponse.rewards
    .map((entry) => {
      const coin = (entry.reward || []).find((c) => c.denom === denom);
      const rawAmount = coin ? Number(coin.amount) : 0;
      const half = Math.floor(rawAmount * fraction);
      return {
        validator_address: entry.validator_address,
        amount: half.toString(),
        denom,
      };
    })
    .filter((c) => Number(c.amount) > 0);

  return partial;
};


// step:5 file: instantly_claim_50%_of_my_ntrn_staking_rewards
import { Buffer } from 'buffer';

export const signAndBroadcastWithdrawal = async (
  signerAddress,
  signDoc,
  apiUrl = '/api/broadcast_tx'
) => {
  if (!window.keplr) {
    throw new Error('Keplr wallet missing.');
  }

  // Re-build the object expected by signDirect
  const directSignDoc = {
    bodyBytes: Uint8Array.from(atob(signDoc.body_bytes), (c) => c.charCodeAt(0)),
    authInfoBytes: Uint8Array.from(atob(signDoc.auth_info_bytes), (c) => c.charCodeAt(0)),
    chainId: signDoc.chain_id,
    accountNumber: BigInt(signDoc.account_number),
  };

  // -------------------
  // 1) Sign the TX bytes
  // -------------------
  const { signature } = await window.keplr.signDirect(
    signDoc.chain_id,
    signerAddress,
    directSignDoc
  );

  // ------------------------------------------------
  // 2) Send the signed payload to the backend (BFF)
  // ------------------------------------------------
  const res = await fetch(apiUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      body_bytes: signDoc.body_bytes,
      auth_info_bytes: signDoc.auth_info_bytes,
      signature: Buffer.from(signature.signature).toString('base64'),
    }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Broadcast failed');
  }

  return await res.json(); // { txhash: '...' }
};


// step:1 file: convert_1_btc_into_solvbtc_and_bridge_it_to_neutron_for_deposit
export const getBTCWalletAddress = async () => {
  // Attempt to connect to Unisat (or any extension that injects `window.unisat`)
  if (window.unisat && typeof window.unisat.requestAccounts === 'function') {
    try {
      const accounts = await window.unisat.requestAccounts();
      if (accounts && accounts.length > 0) {
        return accounts[0];
      }
    } catch (err) {
      console.error('Failed to fetch address from Unisat:', err);
    }
  }

  // Fallback: ask user to type it in
  const address = prompt('Please enter the Bitcoin address that will fund 1 BTC:');
  if (!address || address.trim() === '') {
    throw new Error('A valid Bitcoin address is required.');
  }
  return address.trim();
};


// step:4 file: execute_an_emergency_withdrawal_for_the_user’s_amber_trading_position
/* src/utils/tx.js */
export const signAndBroadcastEmergencyWithdraw = async (signDocFromBackend, chainId = 'neutron-1') => {
  const { keplr } = window;
  if (!keplr) throw new Error('Keplr extension not available.');

  // Decode base64-encoded fields returned by the backend
  const toUint8Array = (b64) => Uint8Array.from(atob(b64), c => c.charCodeAt(0));

  const signDoc = {
    bodyBytes:     toUint8Array(signDocFromBackend.bodyBytes),
    authInfoBytes: toUint8Array(signDocFromBackend.authInfoBytes),
    chainId:       signDocFromBackend.chainId,
    accountNumber: Number(signDocFromBackend.accountNumber)
  };

  // Fetch the signer address again (defensive)
  const offlineSigner = keplr.getOfflineSigner(chainId);
  const [account]     = await offlineSigner.getAccounts();

  // 1. Sign the proto‐SignDoc (DIRECT mode)
  const { signature } = await keplr.signDirect(chainId, account.address, signDoc);

  // 2. POST the signed doc + signature back to the backend for final assembly & broadcast
  const res = await fetch('/api/tx/broadcast', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      bodyBytes:     signDocFromBackend.bodyBytes,
      authInfoBytes: signDocFromBackend.authInfoBytes,
      signature:     Buffer.from(signature.signature, 'base64').toString('base64')
    })
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Broadcast failed: ${text}`);
  }

  const { txhash } = await res.json();
  return txhash;
};


// step:2 file: lock_an_additional_500_ntrn_for_24_months_(boost)
export const hasMinBalance = async (address, minAmount = 500000000) => {
  const REST_ENDPOINT = 'https://rest-kralum.neutron.org';
  const url = `${REST_ENDPOINT}/cosmos/bank/v1beta1/balances/${address}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`LCD error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  const balanceEntry = data.balances?.find((b) => b.denom === 'untrn');
  const balance = balanceEntry ? parseInt(balanceEntry.amount, 10) : 0;
  return balance >= minAmount;
};


// step:3 file: lock_an_additional_500_ntrn_for_24_months_(boost)
export const constructBoostLockMsg = (amount = '500000000', durationMonths = 24) => {
  const msg = {
    lock: {
      amount,
      duration: `${durationMonths}_months`,
    },
  };
  return msg;
};

export const encodeMsgForContract = (msg) => window.btoa(JSON.stringify(msg));


// step:6 file: lock_an_additional_500_ntrn_for_24_months_(boost)
export const queryBoostPosition = async (address) => {
  const BOOST_CONTRACT_ADDRESS = 'neutron1boostcontractaddress…'; // TODO: replace
  const REST_ENDPOINT = 'https://rest-kralum.neutron.org';

  const queryMsg = { position: { address } };
  const encoded = window.btoa(JSON.stringify(queryMsg));
  const url = `${REST_ENDPOINT}/cosmwasm/wasm/v1/contract/${BOOST_CONTRACT_ADDRESS}/smart/${encoded}`;

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Contract query failed: ${res.status}`);
  }
  const data = await res.json();
  return data.data; // The exact schema depends on the contract implementation
};


// step:2 file: lend_2_unibtc_on_amber_finance
export const checkTokenBalance = async (address) => {
  try {
    // TODO: replace with the real uniBTC CW20 contract address
    const cw20Contract = 'neutron1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx';
    const restEndpoint = 'https://rest-kralum.neutron.org';

    // Build the CW20 balance query and base64-encode it for the REST endpoint
    const query = { balance: { address } };
    const encodedQuery = btoa(JSON.stringify(query));

    const url = `${restEndpoint}/cosmwasm/wasm/v1/contract/${cw20Contract}/smart/${encodedQuery}`;
    const res = await fetch(url);

    if (!res.ok) {
      throw new Error(`REST API returned ${res.status}`);
    }

    const json = await res.json();
    const microAmount = BigInt(json?.data?.balance || 0n);

    // Assume uniBTC uses 6 decimals on Neutron
    const displayAmount = Number(microAmount) / 1_000_000;

    return {
      ok: displayAmount >= 2,
      amount: displayAmount
    };
  } catch (err) {
    console.error('Balance check failed', err);
    throw err;
  }
};


// step:3 file: cancel_(unlock)_the_user’s_ntrn_stake_lock_once_the_vesting_period_has_ended
// src/tx/constructUnlockTx.js
export const constructUnlockTx = async ({
  chainId = 'neutron-1',
  senderAddress,
  contractAddress,
  lockId,
  fee = {
    amount: [{ amount: '6000', denom: 'untrn' }],
    gas: '300000'
  },
  memo = ''
}) => {
  // Cosmos SDK Tx requires accountNumber & sequence → fetch from LCD
  const LCD_URL = 'https://rest-kralum.neutron-1.neutron.org';
  const accountRes = await fetch(
    `${LCD_URL}/cosmos/auth/v1beta1/accounts/${senderAddress}`
  ).then((r) => r.json());

  const baseAccount =
    accountRes.account.base_account || accountRes.account; // handles vesting / eth-addr

  const accountNumber = String(baseAccount.account_number);
  const sequence = String(baseAccount.sequence);

  // MsgExecuteContract to cancel the lock
  const msg = {
    type: 'wasm/MsgExecuteContract',
    value: {
      sender: senderAddress,
      contract: contractAddress,
      msg: {
        cancel_lock: { lock_id: lockId }
      },
      funds: []
    }
  };

  // Build the amino-compatible StdSignDoc
  const signDoc = {
    chain_id: chainId,
    account_number: accountNumber,
    sequence: sequence,
    fee,
    msgs: [msg],
    memo
  };

  return {
    signDoc,
    msg,
    fee,
    memo
  };
};


// step:4 file: cancel_(unlock)_the_user’s_ntrn_stake_lock_once_the_vesting_period_has_ended
// src/tx/signAndBroadcast.js
import { toHex } from "./utils"; // helper functions to convert to hex if desired (implementation omitted for brevity)

export const signAndBroadcastTx = async ({
  chainId = 'neutron-1',
  signDoc,
  offlineSigner
}) => {
  try {
    if (!window || !window.keplr) {
      throw new Error('Keplr wallet extension not found');
    }

    // Sign using Keplr's amino signer
    const { signature, signed } = await window.keplr.signAmino(
      chainId,
      signDoc.msgs[0].value.sender,
      signDoc,
      {
        // Signer data → indicates we don’t want Keplr to override fields
        preferNoSetFee: true,
        preferNoSetMemo: true
      }
    );

    // Build StdTx (amino)
    const stdTx = {
      msg: signed.msgs,
      fee: signed.fee,
      signatures: [
        {
          pub_key: signature.pub_key,
          signature: signature.signature,
          account_number: signed.account_number,
          sequence: signed.sequence
        }
      ],
      memo: signed.memo
    };

    // Marshal to bytes → Use Keplr to send raw bytes without CosmJS
    const txBytes = Uint8Array.from(
      Buffer.from(JSON.stringify(stdTx))
    );

    const result = await window.keplr.sendTx(chainId, txBytes, "sync");

    return {
      txhash: toHex(result),
      raw_log: 'Check Neutron explorer for details'
    };
  } catch (err) {
    console.error('[signAndBroadcastTx] ', err);
    throw err;
  }
};


// step:2 file: open_a_5×_leveraged_loop_position_with_1_maxbtc_on_amber
export const validateTokenBalance = async (
  address,
  {
    min = BigInt(1_000000), // 1.0 maxBTC in micro-denom units (example: 1e6)
    denom = 'amaxbtc',      // replace with the exact on-chain denom for maxBTC
    restEndpoint = 'https://rest-kralum.neutron.org' // example REST endpoint
  } = {}
) => {
  try {
    const url = `${restEndpoint}/cosmos/bank/v1beta1/balances/${address}/${denom}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`LCD error: ${res.status}`);

    const { balance } = await res.json();
    const amount = BigInt(balance?.amount || 0);

    if (amount < min) {
      throw new Error(`Insufficient balance: need ≥ ${min} ${denom}, have ${amount}`);
    }
    return { ok: true, amount };
  } catch (err) {
    console.error(err);
    return { ok: false, reason: err.message };
  }
};


// step:3 file: open_a_5×_leveraged_loop_position_with_1_maxbtc_on_amber
export const queryAmberMarketParameters = async (
  {
    contract = 'neutron1ambercontractaddressxxxxxxxxxxxx', // Amber contract address
    restEndpoint = 'https://rest-kralum.neutron.org'
  } = {}
) => {
  // The Amber contract is assumed to expose `{ "config": {} }` or similar.
  const query = { market_params: {} };
  const encoded = btoa(JSON.stringify(query));
  const url = `${restEndpoint}/cosmwasm/wasm/v1/contract/${contract}/smart/${encoded}`;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Amber query failed: ${res.status}`);

  return await res.json(); // → { data: { max_leverage: '6', collateral_factor: '0.8', ... } }
};


// step:4 file: open_a_5×_leveraged_loop_position_with_1_maxbtc_on_amber
export const constructOpenLeverageMsg = (
  {
    sender,
    collateralAmount = '1000000',           // 1.0 maxBTC in micro-units
    collateralDenom = 'amaxbtc',            // actual on-chain denom
    leverage = '5',
    contract = 'neutron1ambercontractaddressxxxxxxxxxxxx'
  }
) => {
  if (!sender) throw new Error('`sender` (wallet address) is required');

  // Amber-specific execute message
  const executeMsg = {
    open_position: {
      collateral: {
        denom: collateralDenom,
        amount: collateralAmount
      },
      leverage
    }
  };

  // Standard CosmWasm MsgExecuteContract to be signed later
  return {
    typeUrl: '/cosmwasm.wasm.v1.MsgExecuteContract',
    value: {
      sender,
      contract,
      msg: new TextEncoder().encode(JSON.stringify(executeMsg)),
      funds: [{ denom: collateralDenom, amount: collateralAmount }]
    }
  };
};


// step:6 file: open_a_5×_leveraged_loop_position_with_1_maxbtc_on_amber
export const queryPositionStatus = async (
  {
    owner,
    contract = 'neutron1ambercontractaddressxxxxxxxxxxxx',
    restEndpoint = 'https://rest-kralum.neutron.org',
    retries = 10,
    delayMs = 3000
  }
) => {
  if (!owner) throw new Error('Owner address is required');

  const query = { positions_by_owner: { owner } };
  const encoded = btoa(JSON.stringify(query));
  const url = `${restEndpoint}/cosmwasm/wasm/v1/contract/${contract}/smart/${encoded}`;

  for (let i = 0; i < retries; i++) {
    const res = await fetch(url);
    if (res.ok) {
      const data = await res.json();
      if (data?.data?.positions?.length) {
        return data.data.positions[0]; // Return the first position found
      }
    }
    await new Promise(r => setTimeout(r, delayMs));
  }
  throw new Error('Position not found within timeout window');
};


// step:1 file: confirm_forfeitable_reward_structure_for_my_current_vault
export const getVaultContractAddress = async (userAddress, registryBaseUrl = "https://indexer.neutron.org") => {
  // Example endpoint:  GET {registryBaseUrl}/api/v1/vaults/{userAddress}
  try {
    const res = await fetch(`${registryBaseUrl}/api/v1/vaults/${userAddress}`);

    if (!res.ok) {
      throw new Error(`Failed to fetch vault address: ${res.status} ${res.statusText}`);
    }

    const { vault_address } = await res.json();

    if (!vault_address) {
      throw new Error("Vault address not found in response.");
    }

    return vault_address;
  } catch (err) {
    console.error("[getVaultContractAddress]", err);
    throw err;
  }
};


// step:2 file: confirm_forfeitable_reward_structure_for_my_current_vault
export const queryVaultConfig = async (vaultAddress, lcdUrl = "https://rest.neutron-1.neutron.org") => {
  /**
   * Helper to base64-encode the JSON query in both browser and Node environments.
   */
  const base64Encode = (obj) => {
    const jsonStr = JSON.stringify(obj);
    if (typeof window !== "undefined" && window.btoa) {
      return window.btoa(jsonStr);
    }
    return Buffer.from(jsonStr).toString("base64");
  };

  try {
    const queryMsg = { config: {} };        // ← { "config": {} }
    const encoded = base64Encode(queryMsg); // base64

    const endpoint = `${lcdUrl}/cosmwasm/wasm/v1/contract/${vaultAddress}/smart/${encoded}`;
    const res = await fetch(endpoint);

    if (!res.ok) {
      throw new Error(`Contract query failed: ${res.status} ${res.statusText}`);
    }

    // Depending on LCD version the JSON key can be `data` or `result`.
    const json = await res.json();
    const config = json.data ?? json.result ?? json;

    return config; // returns the full config object
  } catch (err) {
    console.error("[queryVaultConfig]", err);
    throw err;
  }
};


// step:3 file: confirm_forfeitable_reward_structure_for_my_current_vault
export const parseRewardPolicy = (config) => {
  if (!config || typeof config !== "object") {
    throw new Error("Invalid or empty config object supplied.");
  }

  const forfeitableRewards = config.forfeitable_rewards ?? null;
  const earlyExitPenalty = config.early_exit_penalty ?? null;

  return {
    forfeitableRewards,
    earlyExitPenalty,
    // Convenience flag
    isForfeitable: Boolean(forfeitableRewards) || Boolean(earlyExitPenalty)
  };
};


// step:4 file: confirm_forfeitable_reward_structure_for_my_current_vault
export const displayRewardPolicy = (policy) => {
  try {
    if (!policy || !policy.isForfeitable) {
      return "No early withdrawal penalties. All rewards are fully claimable.";
    }

    let html = "<h4>Early Withdrawal Policy</h4>";

    if (policy.earlyExitPenalty && typeof policy.earlyExitPenalty === "object") {
      html += "<ul>";
      for (const [period, penalty] of Object.entries(policy.earlyExitPenalty)) {
        html += `<li>Within ${period}: ${penalty}% penalty</li>`;
      }
      html += "</ul>";
    } else if (policy.forfeitableRewards !== null) {
      html += `<p>${policy.forfeitableRewards}% of accumulated rewards are forfeited on early exit.</p>`;
    }

    return html;
  } catch (err) {
    console.error("[displayRewardPolicy]", err);
  }
};


// step:2 file: deposit_3_ebtc_into_the_maxbtc_ebtc_supervault
/*
 * checkEbtcBalance.js
 * Simple REST call to the Neutron LCD to verify the user has ≥ minAmount (micro-denom) eBTC.
 */
export const checkEbtcBalance = async (address, minAmountMicro = '3000000') => {
  try {
    // Public Neutron LCD endpoint (you may replace with your own)
    const lcd = 'https://lcd-kralum.neutron-1.nomusa.xyz';

    // eBTC IBC denom on Neutron (see docs/btc-summer/technical/reference)
    const EBTC_DENOM = 'ibc/E2A000FD3EDD91C9429B473995CE2C7C555BCC8CFC1D0A3D02F514392B7A80E8';

    const resp   = await fetch(`${lcd}/cosmos/bank/v1beta1/balances/${address}`);
    if (!resp.ok) throw new Error(`LCD error ${resp.status}`);

    const { balances } = await resp.json();
    const coin = balances.find((c) => c.denom === EBTC_DENOM);
    const amount = coin ? coin.amount : '0';

    if (BigInt(amount) < BigInt(minAmountMicro)) {
      throw new Error(`Insufficient eBTC balance. Need ≥ ${minAmountMicro}, have ${amount}`);
    }

    return {
      ok: true,
      amountMicro: amount
    };
  } catch (err) {
    console.error('[checkEbtcBalance] →', err);
    throw err;
  }
};


// step:3 file: withdraw_10_%_of_the_user’s_shares_from_the_maxbtc_solvbtc_supervault
export const calculateSharesToWithdraw = (totalShares) => {
  const numericShares = Number(totalShares);
  if (!Number.isFinite(numericShares) || numericShares <= 0) {
    throw new Error('Invalid share balance.');
  }

  // Use Math.floor to avoid fractional shares (contracts expect integers)
  const sharesToWithdraw = Math.floor(numericShares * 0.10);
  if (sharesToWithdraw === 0) {
    throw new Error('Calculated shares_to_withdraw equals zero.');
  }

  return sharesToWithdraw;
};


// step:5 file: withdraw_10_%_of_the_user’s_shares_from_the_maxbtc_solvbtc_supervault
export const signAndBroadcastWithdrawTx = async ({ address, signPayload }) => {
  const { body_bytes, auth_info_bytes, account_number, chain_id } = signPayload;

  // Helper to convert base64 → Uint8Array
  const toUint8Array = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

  const signDoc = {
    bodyBytes: toUint8Array(body_bytes),
    authInfoBytes: toUint8Array(auth_info_bytes),
    chainId: chain_id,
    accountNumber: account_number,
  };

  // 1. Ask Keplr to sign the Tx
  const { signature } = await window.keplr.signDirect(chain_id, address, signDoc);

  // 2. Hand the signature back to the backend so it can assemble & broadcast
  const res = await fetch('/api/supervault/broadcast-withdraw', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      body_bytes: body_bytes,
      auth_info_bytes: auth_info_bytes,
      signature: signature.signature, // already base64
    }),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || 'Broadcast failed');
  }

  return data.txhash;
};


// step:2 file: view_available_supervault_positions_eligible_for_bitcoin_summer
export const getSupervaultContractAddress = () => {
  // TODO: Replace the placeholder with the real Supervault address
  const CONTRACT_ADDRESS = 'neutron1supervaultcontractaddressxxx';

  if (!CONTRACT_ADDRESS || !CONTRACT_ADDRESS.startsWith('neutron')) {
    throw new Error('Invalid or missing Supervault contract address configuration.');
  }

  return CONTRACT_ADDRESS;
};


// step:4 file: view_available_supervault_positions_eligible_for_bitcoin_summer
export const filterPositionsByCampaign = (positions, campaignName = 'Bitcoin Summer') => {
  if (!Array.isArray(positions)) {
    throw new Error('Expected positions to be an array.');
  }

  return positions.filter((position) => {
    const campaigns = position.eligible_campaigns || [];
    return campaigns.includes(campaignName);
  });
};


// step:5 file: view_available_supervault_positions_eligible_for_bitcoin_summer
export const PositionsTable = ({ positions }) => {
  if (!positions || positions.length === 0) {
    return <p>No Bitcoin Summer positions found.</p>;
  }

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <th style={{ borderBottom: '1px solid #ccc' }}>Position ID</th>
          <th style={{ borderBottom: '1px solid #ccc' }}>Deposit Amount</th>
          <th style={{ borderBottom: '1px solid #ccc' }}>Rewards Status</th>
        </tr>
      </thead>
      <tbody>
        {positions.map((p) => (
          <tr key={p.position_id}>
            <td style={{ padding: '4px 8px' }}>{p.position_id}</td>
            <td style={{ padding: '4px 8px' }}>{p.deposit_amount}</td>
            <td style={{ padding: '4px 8px' }}>{p.rewards_status}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};


// step:1 file: set_boost_lock_duration_preference_to_48_months
/*
 * Reads the value from an <input type="range"> element and returns it as an integer.
 * @param {string} sliderElementId – The DOM id of the slider.
 * @returns {number} – The selected number of months.
 * @throws {Error} – If the element cannot be found or the value is invalid.
 */
export const captureSliderInput = (sliderElementId) => {
  const slider = document.getElementById(sliderElementId);
  if (!slider) {
    throw new Error(`Slider element with id "${sliderElementId}" not found.`);
  }

  const months = parseInt(slider.value, 10);
  if (Number.isNaN(months) || months <= 0) {
    throw new Error(`Invalid slider value: ${slider.value}`);
  }

  return months;
};


// step:2 file: set_boost_lock_duration_preference_to_48_months
/*
 * Calculates the boost multiplier based on the staking duration.
 * Formula: 1x at 0 months → 2x at 48 months (linear scaling).
 * @param {number} months – The staking duration in months.
 * @returns {number} – The boost multiplier rounded to two decimals.
 */
export const calculateBoostMultiplier = (months) => {
  const MAX_MONTHS = 48;
  const multiplier = 1 + months / MAX_MONTHS; // 48 months → 1 + 48/48 = 2.0×
  return +multiplier.toFixed(2);
};


// step:3 file: set_boost_lock_duration_preference_to_48_months
/*
 * Persists the preferred lock duration to localStorage.
 * @param {number} months – The lock duration in months.
 */
export const saveLockDurationPreference = (months) => {
  try {
    localStorage.setItem('preferredLockDurationMonths', months.toString());
  } catch (err) {
    // localStorage can throw (e.g., in private mode)
    console.error('Failed to store lock duration preference', err);
  }
};


// step:4 file: set_boost_lock_duration_preference_to_48_months
/*
 * Updates a DOM element with a preview of the boost multiplier and minimum lock.
 * @param {Object} params
 * @param {number} params.months – Number of months selected.
 * @param {number} params.multiplier – Calculated multiplier from Step 2.
 * @param {number} [params.minLockBase=100] – Base amount used to derive minimum lock (editable).
 */
export const displayMultiplierPreview = ({ months, multiplier, minLockBase = 100 }) => {
  const previewEl = document.getElementById('multiplierPreview');
  if (!previewEl) {
    console.warn('Preview element #multiplierPreview not found in DOM.');
    return;
  }

  // Example heuristic: scale the base amount linearly with months.
  const minLockAmount = ((minLockBase * months) / 48).toFixed(2);
  previewEl.textContent = `Boost: ${multiplier}×  |  Minimum lock: ${minLockAmount} NTRN`;
};