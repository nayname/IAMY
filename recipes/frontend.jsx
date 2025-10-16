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
    const coin = (data.balances || []).find((c) => c.denom === denom);
    const amount = coin ? Number(coin.amount) : 0;
    return amount;
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
  return true;
};

// step:4 file: lock_2000_ntrn_for_3_months_to_obtain_a_1.2×_btc_summer_boost
export const calculateUnlockTimestamp = () => {
  const NOW_SEC = Math.floor(Date.now() / 1000);
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

  const base64Query = btoa(JSON.stringify(queryMsg));
  const LCD = 'https://lcd-neutron.blockpane.com';

  try {
    const url = `${LCD}/cosmwasm/wasm/v1/contract/${BOOST_POINTER_CONTRACT}/smart/${base64Query}`;
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`LCD error: ${res.status} ${res.statusText}`);
    }
    const data = await res.json();
    return data.data?.multiplier ?? null;
  } catch (err) {
    console.error('[queryBoostMultiplier] ', err);
    throw err;
  }
};

// step:4 file: close_my_leveraged_loop_position_on_amber
// File: src/utils/amber.js
const b64ToUint8 = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

export const signAndBroadcastClosePosition = async ({
  chainId           = 'neutron-1',
  signDocBase64,
  backendBroadcastUrl = '/api/amber/broadcast_signed_tx'
}) => {
  try {
    const address     = await getUserAddress(chainId);
    const signDocBytes = b64ToUint8(signDocBase64);

    const { signed, signature } = await window.keplr.signDirect(
      chainId,
      address,
      { typeUrl: '/cosmos.tx.v1beta1.SignDoc', value: signDocBytes }
    );

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

    return await res.json();
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
  offerAmount = '1000000',
  askDenom = 'uniBTC',
  maxSlippage = '0.005'
}) => {
  const execMsg = {
    swap: {
      offer_asset: {
        info: { native_token: { denom: offerDenom } },
        amount: offerAmount
      },
      max_slippage: maxSlippage
    }
  };

  return {
    typeUrl: '/cosmwasm.wasm.v1.MsgExecuteContract',
    value: {
      sender,
      contract: contractAddress,
      msg: btoa(JSON.stringify(execMsg)),
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
  const payload = { set_target: { evm_address: evmAddress } };

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
    const encoded = btoa(JSON.stringify(queryMsg));

    const endpoint = `https://rest-kralum.neutron.org/cosmwasm/wasm/v1/contract/${contractAddress}/smart/${encoded}`;
    const resp = await fetch(endpoint);

    if (!resp.ok) {
      throw new Error(`Query failed with ${resp.status}: ${resp.statusText}`);
    }

    const result = await resp.json();
    return result;
  } catch (err) {
    console.error(err);
    throw err;
  }
};

// step:1 file: show_my_total_bitcoin_summer_points_earned_in_the_current_phase
export const getNeutronAddress = async () => {
  const chainId = 'neutron-1';

  const wallet = window.keplr || window.leap;
  if (!wallet) {
    throw new Error('No supported Neutron wallet extension (Keplr or Leap) found.');
  }

  try {
    await wallet.enable(chainId);
  } catch (error) {
    throw new Error(`Wallet connection rejected or chain not supported: ${error.message}`);
  }

  const offlineSigner = wallet.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error('Unable to fetch an account from the wallet signer.');
  }

  return accounts[0].address;
};

// step:4 file: show_my_total_bitcoin_summer_points_earned_in_the_current_phase
export const displayPoints = (points) => {
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
  const apy = (Math.pow(1 + r, SECONDS_PER_YEAR) - 1) * 100;
  return Number(apy.toFixed(2));
};

// step:5 file: list_current_amber_lending_markets_and_apys
const MarketTable = () => {
  const [markets, setMarkets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const env = 'mainnet';
        const addrRes = await fetch(`/api/amber/controller-address?env=${env}`);
        if (!addrRes.ok) throw new Error(await addrRes.text());
        await addrRes.json();

        const marketsRes = await fetch(`/api/amber/markets?env=${env}`);
        if (!marketsRes.ok) throw new Error(await marketsRes.text());
        const marketList = await marketsRes.json();

        const enriched = await Promise.all(
          marketList.map(async (m) => {
            const stateRes = await fetch(`/api/amber/market-state?env=${env}&market_id=${m.id}`);
            if (!stateRes.ok) throw new Error(await stateRes.text());
            const state = await stateRes.json();

            return {
              id: m.id,
              symbol: m.symbol,
              collateralFactor: Number(m.collateral_factor),
              supplyAPY: rateToAPY(state.supply_rate_per_second),
              borrowAPY: rateToAPY(state.borrow_rate_per_second)
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

// step:1 file: lock_an_account_so_it_can_no_longer_send_transactions
/* wallet.js */
export const getAccountAddress = async () => {
  if (typeof window === "undefined" || !window.ethereum) {
    throw new Error("No Ethereum provider found. Make sure MetaMask is installed.");
  }

  try {
    const accounts = await window.ethereum.request({
      method: "eth_requestAccounts"
    });

    if (!accounts || accounts.length === 0) {
      throw new Error("No accounts returned from provider.");
    }
    return accounts[0];
  } catch (err) {
    console.error("Failed to fetch account address", err);
    throw new Error("Could not obtain account address. Check console for details.");
  }
};