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



