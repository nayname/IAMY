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
export const displayRewardPolicy = (policy, targetElementId = "reward-policy") => {
  try {
    const el = document.getElementById(targetElementId);

    // Fallback if no DOM target is available
    const output = (msg) => {
      if (el) {
        el.innerHTML = msg;
      } else {
        console.log(msg);
      }
    };

    if (!policy || !policy.isForfeitable) {
      output("No early withdrawal penalties. All rewards are fully claimable.");
      return;
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

    output(html);
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
};// step:1 file: query_transaction_details_with_cast_tx
export const getValidatedTxHash = (inputHash) => {
  // Accept either `0x`-prefixed or plain 64-char hex
  const pattern = /^0x?[0-9a-fA-F]{64}$/;
  if (!pattern.test(inputHash)) {
    throw new Error('Invalid transaction hash supplied.');
  }
  // Normalise to lowercase 0x-prefixed form before sending to backend
  const cleaned = inputHash.toLowerCase();
  return cleaned.startsWith('0x') ? cleaned : `0x${cleaned}`;
};


// step:2 file: sign_an_arbitrary_message_using_eth_sign_on_a_cosmos-evm_(ethermint_evmos)_json-rpc_endpoint
export const toHexMessage = (message) => {
  if (typeof message !== 'string') {
    throw new Error('Message must be a string');
  }
  // Convert each character to its UTF-8 char code, then to a two-digit hex value
  const hexBody = Array.from(message)
    .map(char => char.charCodeAt(0).toString(16).padStart(2, '0'))
    .join('');

  return '0x' + hexBody;
};


// step:1 file: trace_all_transactions_in_block_number_0xe_using_debug_traceblockbynumber
export const getRpcEndpoint = () => {
  // Use an environment variable if available, otherwise fallback to a public RPC.
  return import.meta.env.VITE_COSMOS_EVM_RPC_ENDPOINT || "https://rpc.evmos.org:8545";
};


// step:2 file: trace_all_transactions_in_block_number_0xe_using_debug_traceblockbynumber
export const debugTraceBlockByNumber = async (
  blockNumberHex = "0xe", // default block number in hex
  endpoint = getRpcEndpoint()
) => {
  try {
    const payload = {
      jsonrpc: "2.0",
      id: 1,
      method: "debug_traceBlockByNumber",
      params: [blockNumberHex, {}]
    };

    // Send the JSON-RPC request
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    // Basic HTTP-level error handling
    if (!response.ok) {
      throw new Error(`Network error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    // JSON-RPC-level error handling
    if (data.error) {
      throw new Error(`RPC Error: ${data.error.code} – ${data.error.message}`);
    }

    return data.result; // the actual trace data
  } catch (error) {
    console.error("Failed to trace block:", error);
    throw error;
  }
};


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


// step:1 file: create_a_filter_that_tracks_new_blocks_using_eth_newblockfilter
/*
 * connectRpc.js
 * Checks connectivity to an Ethereum-compatible JSON-RPC endpoint.
 */
export const connectRpc = async (rpcUrl = 'http://localhost:8545') => {
  const payload = {
    jsonrpc: '2.0',
    id: 1,
    method: 'web3_clientVersion',
    params: []
  };

  try {
    const response = await fetch(rpcUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`RPC connection failed with status ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(`RPC error: ${data.error.message}`);
    }

    console.info('JSON-RPC connection established:', data.result);
    return rpcUrl; // Return the verified RPC URL for downstream steps
  } catch (err) {
    console.error('Failed to connect to JSON-RPC endpoint:', err);
    throw err; // Re-throw so callers can handle it
  }
};


// step:2 file: create_a_filter_that_tracks_new_blocks_using_eth_newblockfilter
/*
 * createBlockFilter.js
 * Requests a new block filter and returns its ID.
 */
export const createBlockFilter = async (rpcUrl) => {
  const payload = {
    jsonrpc: '2.0',
    id: 1,
    method: 'eth_newBlockFilter',
    params: []
  };

  try {
    const response = await fetch(rpcUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`RPC request failed with status ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(`RPC error: ${data.error.message}`);
    }

    const filterId = data.result;
    if (!filterId) {
      throw new Error('eth_newBlockFilter did not return a filter ID');
    }

    console.info('Created block filter:', filterId);
    return filterId;
  } catch (err) {
    console.error('Failed to create block filter:', err);
    throw err;
  }
};


// step:3 file: create_a_filter_that_tracks_new_blocks_using_eth_newblockfilter
/*
 * storeFilterId.js
 * Simple helpers to save and retrieve a filter ID from localStorage.
 */
export const storeFilterId = (filterId) => {
  try {
    localStorage.setItem('blockFilterId', filterId);
    console.info('Filter ID stored in localStorage');
  } catch (err) {
    console.warn('Unable to write filter ID to localStorage:', err);
  }
};

export const getStoredFilterId = () => {
  try {
    return localStorage.getItem('blockFilterId') || null;
  } catch {
    return null;
  }
};


// step:4 file: create_a_filter_that_tracks_new_blocks_using_eth_newblockfilter
/*
 * pollFilterChanges.js
 * Periodically calls eth_getFilterChanges and delivers new block hashes via callback.
 */
export const pollFilterChanges = (
  rpcUrl,
  filterId,
  onNewBlocks = (hashArray) => {},
  pollIntervalMs = 5000
) => {
  if (!filterId) {
    throw new Error('pollFilterChanges requires a valid filterId');
  }

  let timerId = null;

  const payload = {
    jsonrpc: '2.0',
    id: 1, // the ID can be arbitrary per request
    method: 'eth_getFilterChanges',
    params: [filterId]
  };

  const poll = async () => {
    try {
      const response = await fetch(rpcUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        console.error(`eth_getFilterChanges failed: ${response.status}`);
        return; // keep polling despite transient errors
      }

      const data = await response.json();
      if (data.error) {
        console.error('RPC error from eth_getFilterChanges:', data.error.message);
        return;
      }

      const hashes = data.result || [];
      if (hashes.length > 0) {
        onNewBlocks(hashes);
      }
    } catch (err) {
      console.error('Error while polling eth_getFilterChanges:', err);
    }
  };

  // Start the interval polling
  timerId = setInterval(poll, pollIntervalMs);
  console.info(`Started polling filter ${filterId} every ${pollIntervalMs} ms`);

  // Return a stop function so the caller can cancel polling
  return () => {
    if (timerId) {
      clearInterval(timerId);
      console.info('Stopped polling filter changes');
    }
  };
};


// step:1 file: get_block_number_0x1_with_the_full_list_of_transactions
export const getBlockByNumber = async ({
  rpcEndpoint,
  blockNumber = '0x1',       // Hex string or tags like 'latest'
  includeTxObjects = true,   // true: full tx objects, false: only tx hashes
  fetchOptions = {}          // optional: extra init values for fetch (e.g., mode, credentials)
} = {}) => {
  if (!rpcEndpoint) {
    throw new Error('Parameter "rpcEndpoint" is required.');
  }

  // Compose the JSON-RPC payload
  const payload = {
    jsonrpc: '2.0',
    id: 1,
    method: 'eth_getBlockByNumber',
    params: [blockNumber, includeTxObjects]
  };

  try {
    const response = await fetch(rpcEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      ...fetchOptions
    });

    // Network-level error handling
    if (!response.ok) {
      throw new Error(`JSON-RPC call failed with HTTP status ${response.status}`);
    }

    const body = await response.json();

    // JSON-RPC-level error handling
    if (body.error) {
      const message = body.error.message || JSON.stringify(body.error);
      throw new Error(`RPC Error: ${message}`);
    }

    return body.result; // the block object
  } catch (err) {
    console.error('[getBlockByNumber] RPC request failed:', err);
    throw err;
  }
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


// step:1 file: create_a_new_key_(account)_secured_with_a_passphrase
export const requestUniqueKeyName = async () => {
  try {
    let name = '';

    /* Loop until the user provides a unique name */
    while (true) {
      name = prompt('Enter a unique key name for your wallet:');
      if (!name) throw new Error('Key name is required.');

      // Ask the backend if the name is available
      const resp = await fetch(`/api/keys/validate?name=${encodeURIComponent(name)}`);
      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`Validation request failed: ${errText}`);
      }
      const { available } = await resp.json();

      if (available) {
        alert(`Great! “${name}” is available.`);
        return name;
      }

      alert(`The key name “${name}” already exists. Please choose another.`);
    }
  } catch (err) {
    console.error(err);
    throw err;
  }
};


// step:4 file: create_a_new_key_(account)_secured_with_a_passphrase
export const createKeyAndShowMnemonic = async (name) => {
  try {
    const passphrase = prompt('Create a passphrase (min 8 chars) to secure your key:');
    if (!passphrase || passphrase.length < 8) {
      throw new Error('Passphrase must be at least 8 characters long.');
    }

    const res = await fetch('/api/keys/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, passphrase })
    });

    if (!res.ok) {
      const errTxt = await res.text();
      throw new Error(`Key creation failed: ${errTxt}`);
    }

    const { address, mnemonic } = await res.json();

    /* Display mnemonic exactly once so the user can back it up */
    alert(`IMPORTANT — BACKUP NOW!\n\nAddress: ${address}\n\n24-word mnemonic:\n${mnemonic}`);

    // Return values for further programmatic use if needed
    return { address, mnemonic };
  } catch (err) {
    console.error(err);
    throw err;
  }
};


// step:1 file: provide_paired_liquidity_of_1_wbtc_and_60,000_usdc_to_the_wbtc_usdc_supervault
export const getSenderAddress = async () => {
  const chainId = 'neutron-1';

  // Check that Keplr is installed
  if (!window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  // Ask Keplr to enable the chain
  await window.keplr.enable(chainId);

  // Get an OfflineSigner to access the user’s account(s)
  const offlineSigner = window.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No accounts found in the connected wallet.');
  }

  // Return the first account (default behaviour for most dApps)
  return accounts[0].address;
};


// step:1 file: opt_in_to_partner_airdrops_for_my_vault_deposits
export const getUserAddress = async () => {
  // Neutron main-net chain-id
  const chainId = 'neutron-1';

  // 1. Make sure Keplr is injected
  if (!window.keplr) {
    throw new Error('Keplr wallet not found. Please install or enable it.');
  }

  // 2. Request wallet access for the given chain
  await window.keplr.enable(chainId);

  // 3. Retrieve the signer and the first account
  const signer = window.getOfflineSigner(chainId);
  const accounts = await signer.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No accounts were returned by the wallet.');
  }

  // 4. Return the Bech32 address
  return accounts[0].address;
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


// step:2 file: estimate_gas_for_a_contract_call_with_cast_estimate
// frontend/gatherCallParameters.js

/**
 * Reads inputs from the page and bundles them for the backend.
 * Required element IDs:
 *   #rpcUrl, #contractAddress, #functionSignature, #abiArgs (optional, comma-separated)
 */
export const gatherCallParameters = () => {
  try {
    const rpcUrl = document.getElementById('rpcUrl').value.trim();
    const contractAddress = document.getElementById('contractAddress').value.trim();
    const functionSignature = document.getElementById('functionSignature').value.trim();
    const abiArgsRaw = (document.getElementById('abiArgs')?.value || '').trim();

    if (!rpcUrl || !contractAddress || !functionSignature) {
      throw new Error('RPC URL, contract address, and function signature are required.');
    }

    const args = abiArgsRaw.length ? abiArgsRaw.split(',').map(arg => arg.trim()) : [];

    return {
      rpc_url: rpcUrl,
      contract_address: contractAddress,
      function_signature: functionSignature,
      args,
    };
  } catch (err) {
    console.error(err);
    alert(err.message);
    throw err;
  }
};


// step:4 file: estimate_gas_for_a_contract_call_with_cast_estimate
// frontend/interpretGasCost.js

/**
 * Computes total gas cost.
 * @param {number} gasUnits       – integer from backend.
 * @param {number} gasPriceGwei   – gas price in Gwei (default = 20).
 * @returns {{ gasUnits, gasPriceGwei, weiCost: string, etherCost: number }}
 */
export const interpretGasCost = (gasUnits, gasPriceGwei = 20) => {
  if (typeof gasUnits !== 'number' || gasUnits <= 0) throw new Error('Invalid gasUnits.');
  if (typeof gasPriceGwei !== 'number' || gasPriceGwei <= 0) throw new Error('Invalid gasPriceGwei.');

  const GWEI_TO_WEI = 1_000_000_000n;       // 1e9
  const ETHER_TO_WEI = 1_000_000_000_000_000_000n; // 1e18

  const weiCostBig = BigInt(gasUnits) * (BigInt(Math.round(gasPriceGwei * 1e9)));
  const etherCost = Number(weiCostBig) / Number(ETHER_TO_WEI);

  return {
    gasUnits,
    gasPriceGwei,
    weiCost: weiCostBig.toString(),
    etherCost,
  };
};


// step:1 file: retrieve_full_transaction_details_from_a_cosmos-evm_chain_using_an_ethereum_transaction_hash
/* utils/validateTxHash.js */
export const validateTxHash = (txHash) => {
  // A valid hash looks like: 0x followed by exactly 64 hex characters
  const HASH_REGEX = /^0x([A-Fa-f0-9]{64})$/;

  if (typeof txHash !== 'string' || !HASH_REGEX.test(txHash)) {
    throw new Error('Invalid transaction hash. Expecting a 0x-prefixed, 64-character hexadecimal string.');
  }

  // Return the normalized hash (lower-case) for consistency
  return txHash.toLowerCase();
};


// step:1 file: estimate_gas_fees_for_eip-1559_transactions
export const collectTxFields = async ({ to, value = "0x0", data = "0x", gasLimit }) => {
  // 1. Ensure an EVM wallet is available in the browser
  if (typeof window === "undefined" || !window.ethereum) {
    throw new Error("No EVM wallet detected. Please install MetaMask or another Web3 wallet.");
  }

  // 2. Ask the user to connect the wallet and fetch the first account
  const [from] = await window.ethereum.request({ method: "eth_requestAccounts" });

  // 3. Verify the user is on the expected Cosmos-EVM chain (replace the ID below)
  const expectedChainIdHex = "0x2323"; // example: 0x2323 == 8995
  const chainIdHex = await window.ethereum.request({ method: "eth_chainId" });
  if (chainIdHex !== expectedChainIdHex) {
    throw new Error(`Wrong network. Please switch your wallet to chainId ${expectedChainIdHex}.`);
  }

  // 4. Build the transaction object
  const tx = {
    from,
    to,
    value,        // amount in WEI, hex-prefixed string
    data,         // calldata, hex-prefixed string
    gasLimit,     // optional gas limit (hex string)
    chainId: parseInt(chainIdHex, 16)
  };

  return tx;
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


// step:1 file: create_a_shell_alias_for_cast_with_a_default_rpc_endpoint
export const getRpcEndpoint = () => {
  // Attempt to read from a build-time environment variable first
  const envUrl = process.env.NEXT_PUBLIC_RPC_URL || '';
  if (envUrl) {
    localStorage.setItem('COSMOS_RPC_URL', envUrl);
    return envUrl;
  }

  // Fallback: ask the user
  const url = window.prompt(
    'Enter the HTTP JSON-RPC URL of your Cosmos EVM node (e.g., http://localhost:8545)',
    'http://localhost:8545'
  );

  if (!url) {
    throw new Error('RPC URL is required to proceed.');
  }

  // Persist so later steps can reuse it
  localStorage.setItem('COSMOS_RPC_URL', url);
  return url;
};


// step:2 file: create_a_shell_alias_for_cast_with_a_default_rpc_endpoint
export const validateRpcEndpoint = async (rpcUrl) => {
  const payload = {
    jsonrpc: '2.0',
    id: 1,
    method: 'eth_blockNumber',
    params: []
  };

  try {
    const response = await fetch(rpcUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Endpoint returned HTTP ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(`RPC error: ${data.error.message}`);
    }

    // Convert the hex string (e.g., "0x10d4f") to an integer
    const blockNumber = parseInt(data.result, 16);
    console.info(`RPC endpoint is healthy. Current block: ${blockNumber}`);
    return blockNumber;
  } catch (error) {
    console.error(error);
    throw new Error(`Failed to validate RPC endpoint: ${error.message}`);
  }
};


// step:1 file: retrieve_the_current_base_gas_price_(eth_gasprice)
/*
 * Creates a JSON-RPC compliant payload for fetching the current gas price
 * from an Ethereum-compatible node.
 */
export const prepareGasPricePayload = (id = 1) => {
  return {
    jsonrpc: "2.0",
    method: "eth_gasPrice",
    params: [],
    id
  };
};


// step:2 file: retrieve_the_current_base_gas_price_(eth_gasprice)
/*
 * Executes a POST request against an Ethereum JSON-RPC endpoint.
 * Defaults to a local node at http://localhost:8545 but can be overridden.
 */
export const postJsonRpc = async (payload, endpoint = "http://localhost:8545") => {
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      // Surface HTTP-level errors
      const errText = await res.text();
      throw new Error(`HTTP ${res.status}: ${errText}`);
    }

    return await res.json();
  } catch (err) {
    console.error("JSON-RPC request failed", err);
    throw err;
  }
};


// step:3 file: retrieve_the_current_base_gas_price_(eth_gasprice)
/*
 * Parses the `eth_gasPrice` response and converts the result from hex-encoded
 * wei to human-readable formats.
 */
export const parseGasPriceResponse = (response) => {
  if (!response || typeof response.result !== "string") {
    throw new Error("Malformed JSON-RPC response: missing 'result' field.");
  }

  // The result is a hex string (e.g., "0x91b1d9f00"). Convert to BigInt.
  const weiBigInt = BigInt(response.result);
  const wei = weiBigInt.toString(10); // decimal string

  // Helper conversions
  const gwei = (weiBigInt / 1_000_000_000n).toString(10);
  const ether = Number(weiBigInt) / 1e18; // may lose precision but fine for display

  return { wei, gwei, ether };
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


// step:1 file: check_my_health_factor_on_amber_finance
export const getUserAddress = async () => {
  const chainId = 'neutron-1';

  // Make sure the browser has access to a Keplr-compatible wallet
  if (!window || !window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  // Ask the wallet for permission to access the chain
  await window.keplr.enable(chainId);

  // Retrieve an OfflineSigner, then the actual account list
  const signer = window.getOfflineSigner(chainId);
  const accounts = await signer.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in the connected wallet.');
  }

  // The first account is assumed to be the active one
  return accounts[0].address;
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


// step:1 file: fetch_the_current_nonce_(transaction_count)_for_an_evm_address
/*
 * validateEvmAddress.js
 * Utility to validate a caller-supplied EVM address.
 * Performs a basic 0x-prefixed, 40-hex-character length check.
 */
export const validateEvmAddress = (address) => {
  // Ensure a string was provided
  if (typeof address !== "string") {
    throw new Error("Address must be a string.");
  }

  // Regex: 0x + 40 hexadecimal chars (case-insensitive)
  const re = /^0x[a-fA-F0-9]{40}$/;

  if (!re.test(address)) {
    throw new Error("Invalid EVM address format.");
  }

  // If the check passes, return the checksummed address (lower-cased here)
  return address.toLowerCase();
};


// step:2 file: fetch_the_current_nonce_(transaction_count)_for_an_evm_address
/*
 * getTransactionCount.js
 * Queries eth_getTransactionCount from a Cosmos-EVM JSON-RPC endpoint.
 */
export const getTransactionCount = async ({ address, rpcEndpoint }) => {
  // Validate parameters up-front
  if (!address) throw new Error("'address' is required");
  if (!rpcEndpoint) throw new Error("'rpcEndpoint' is required");

  // JSON-RPC payload
  const payload = {
    jsonrpc: "2.0",
    id: 1,
    method: "eth_getTransactionCount",
    params: [address, "latest"]
  };

  try {
    const res = await fetch(rpcEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`RPC HTTP error: ${res.status} ${res.statusText}`);
    }

    const json = await res.json();

    // Handle JSON-RPC error object
    if (json.error) {
      throw new Error(`RPC returned error: ${json.error.message || JSON.stringify(json.error)}`);
    }

    // json.result is a hex string, e.g. "0x10"
    const nonceHex = json.result;
    const nonce = parseInt(nonceHex, 16);

    if (Number.isNaN(nonce)) {
      throw new Error(`Unable to parse nonce from result: ${nonceHex}`);
    }

    return nonce;
  } catch (err) {
    // Re-throw with context so caller can surface it in UI
    throw new Error(`Failed to fetch transaction count: ${err.message}`);
  }
};


// step:1 file: lock_2000_ntrn_for_3_months_to_obtain_a_1.2×_btc_summer_boost
export const getWalletAddress = async () => {
  const chainId = 'neutron-1';

  if (!window || !window.keplr) {
    throw new Error('Keplr wallet is not installed in this browser.');
  }

  // Ask Keplr to enable the Neutron chain (will prompt the user on first run)
  await window.keplr.enable(chainId);

  // Retrieve the key information for this chain
  const key = await window.keplr.getKey(chainId);

  if (!key || !key.bech32Address) {
    throw new Error('Failed to obtain a bech32 address from Keplr.');
  }

  return key.bech32Address; // e.g. neutron1...
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

    /*  The response shape is:
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
    /*  Expected shape (example):
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


// step:1 file: close_my_leveraged_loop_position_on_amber
export const getUserAddress = async (chainId = 'neutron-1') => {
  try {
    // Make sure Keplr is installed
    if (!window.keplr) {
      throw new Error('Keplr wallet is not installed.');
    }

    // Ask Keplr to enable the selected chain
    await window.keplr.enable(chainId);

    // Obtain the OfflineSigner and read the account
    const offlineSigner = window.getOfflineSigner(chainId);
    const accounts = await offlineSigner.getAccounts();

    if (!accounts || accounts.length === 0) {
      throw new Error('No account found in Keplr.');
    }

    return accounts[0].address; // <- Neutron bech32 address
  } catch (err) {
    console.error('[getUserAddress] error:', err);
    throw err;
  }
};


// step:4 file: close_my_leveraged_loop_position_on_amber
// File: src/utils/amber.js
import { getUserAddress } from './wallet';

// Helper — base64 → Uint8Array
const b64ToUint8 = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

export const signAndBroadcastClosePosition = async ({
  chainId           = 'neutron-1',
  signDocBase64,               // from step 3
  backendBroadcastUrl = '/api/amber/broadcast_signed_tx'
}) => {
  try {
    const address     = await getUserAddress(chainId);
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


// step:1 file: retrieve_projected_ntrn_rewards_based_on_current_point_total
export const getWalletAddress = async (chainId = 'neutron-1') => {
  // Ensure the wallet extension is available
  if (!window.keplr) {
    throw new Error('Keplr wallet extension is not installed.');
  }

  // Request access to the given chain from the wallet
  await window.keplr.enable(chainId);

  // Obtain the signer & the user’s accounts
  const offlineSigner = window.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in the connected wallet.');
  }

  // Return the first address (most wallets only expose one)
  return accounts[0].address;
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


// step:1 file: swap_1_ebtc_for_unibtc_on_neutron_dex
export const getUserAddress = async () => {
  const chainId = 'neutron-1'; // change to testnet ID if needed

  // 1. Make sure Keplr is available
  if (!window.keplr) {
    throw new Error('Keplr wallet extension is not installed.');
  }

  // 2. Ask Keplr to enable the requested chain
  await window.keplr.enable(chainId);

  // 3. Obtain the offline signer
  const offlineSigner = window.keplr.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();

  // 4. Basic safety check
  if (!accounts || accounts.length === 0) {
    throw new Error('No account detected in Keplr.');
  }

  // 5. Return the first available address
  return accounts[0].address; // e.g. neutron1...
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


// step:1 file: use_wagmi_react_hooks_to_connect_a_wallet_in_a_dapp
# Using npm
npm install wagmi @rainbow-me/rainbowkit viem react react-dom

# Or, if you prefer yarn
yarn add wagmi @rainbow-me/rainbowkit viem react react-dom


// step:2 file: use_wagmi_react_hooks_to_connect_a_wallet_in_a_dapp
// src/lib/wagmi.ts
import { configureChains } from "wagmi";
import { evmos } from "wagmi/chains";                // Cosmos-EVM example
import { jsonRpcProvider } from "wagmi/providers/jsonRpc";
import { publicProvider } from "wagmi/providers/public";

// 1️⃣  Declare which chains your dApp supports
const supportedChains = [evmos];

// 2️⃣  Point Wagmi at at least one RPC provider
export const { chains, publicClient, webSocketPublicClient } = configureChains(
  supportedChains,
  [
    jsonRpcProvider({
      rpc: () => ({
        http: "https://eth.bd.evmos.org:8545"      // Public Evmos RPC
      })
    }),
    publicProvider()                                 // Fallback public provider
  ]
);



// step:3 file: use_wagmi_react_hooks_to_connect_a_wallet_in_a_dapp
// src/App.tsx
import React from "react";
import "@rainbow-me/rainbowkit/styles.css";

import { WagmiConfig, createConfig } from "wagmi";
import { RainbowKitProvider, getDefaultWallets } from "@rainbow-me/rainbowkit";

import { chains, publicClient } from "./lib/wagmi";   // Step-2 exports
import ConnectWallet from "./components/ConnectWallet"; // Step-4 component

// Generate default connectors (MetaMask, WalletConnect, etc.)
const { connectors } = getDefaultWallets({
  appName: "My Cosmos/EVM dApp",
  projectId: "YOUR_WALLETCONNECT_PROJECT_ID",       // ← Replace with real ID
  chains
});

// Create Wagmi client instance
export const wagmiConfig = createConfig({
  autoConnect: true,
  connectors,
  publicClient
});

function App() {
  return (
    <WagmiConfig config={wagmiConfig}>
      <RainbowKitProvider chains={chains}>
        {/* Your routes / components go here */}
        <ConnectWallet />
      </RainbowKitProvider>
    </WagmiConfig>
  );
}

export default App;



// step:4 file: use_wagmi_react_hooks_to_connect_a_wallet_in_a_dapp
// src/components/ConnectWallet.tsx
import React from "react";
import { ConnectButton } from "@rainbow-me/rainbowkit";
import { useAccount } from "wagmi";

const ConnectWallet: React.FC = () => {
  const { address, isConnected } = useAccount();

  return (
    <div>
      {/* RainbowKit component opens the wallet modal */}
      <ConnectButton />

      {/* Optional UI feedback */}
      {isConnected && (
        <p style={{ marginTop: "1rem" }}>
          Connected address: {address}
        </p>
      )}
    </div>
  );
};

export default ConnectWallet;



// step:1 file: delegate_500stake_from_recipient_to_my_validator_validator
export const validateAddresses = (delegatorAddress, validatorAddress) => {
  // Generic Bech32 address shape: <prefix>1<38-char lowercase bech32 body>
  const buildRegex = (prefix) => new RegExp(`^${prefix}1[0-9a-z]{38}$`);

  if (!buildRegex('cosmos').test(delegatorAddress)) {
    throw new Error('Invalid delegator Bech32 address');
  }
  if (!buildRegex('cosmosvaloper').test(validatorAddress)) {
    throw new Error('Invalid validator Bech32 address');
  }
  return true; // everything looks good
};


// step:2 file: delegate_500stake_from_recipient_to_my_validator_validator
export const checkBalance = async (
  address,
  lcdEndpoint = 'https://lcd.cosmos.directory/gaia',
  minAmount = 500 /* stake */
) => {
  const url = `${lcdEndpoint}/cosmos/bank/v1beta1/balances/${address}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`LCD error: ${res.status} ${res.statusText}`);
  }

  const { balances } = await res.json();
  const stakeObj = balances.find((b) => b.denom === 'stake');
  const available = stakeObj ? BigInt(stakeObj.amount) : 0n;

  const estimatedFee = 5000n; // tweak for your chain
  if (available < BigInt(minAmount) + estimatedFee) {
    throw new Error('Insufficient balance for 500stake delegation plus fees');
  }
  return Number(available);
};


// step:1 file: query_allbalances_for_an_address_at_block_height_123_via_grpc
/* utils/address.js */
export const validateBech32Address = (address, prefix = 'cosmos') => {
  if (typeof address !== 'string') return false;

  // Bech32 addresses are conventionally lowercase
  if (address !== address.toLowerCase()) return false;

  // Basic structural rule: <prefix>1<38 bech32 chars (excluding "1", "b", "i", "o")
  const regex = new RegExp('^' + prefix + '1[ac-hj-np-z02-9]{38}$');
  return regex.test(address);
};


// step:2 file: add_a_cosmos_evm_network_to_metamask
// addEvmosNetwork.js
export const addEvmosNetwork = async () => {
  // 1. Grab config from backend
  const res = await fetch('/api/network/evmos');
  if (!res.ok) {
    throw new Error(`Could not fetch network config: ${res.statusText}`);
  }
  const config = await res.json();

  // 2. Ensure MetaMask is present
  if (typeof window === 'undefined' || !window.ethereum) {
    throw new Error('MetaMask is not installed or window.ethereum is undefined.');
  }

  try {
    // 3. Ask MetaMask to add the chain (opens confirmation modal for the user)
    await window.ethereum.request({
      method: 'wallet_addEthereumChain',
      params: [config]
    });

    // 4. Optionally switch to the newly-added chain
    await window.ethereum.request({
      method: 'wallet_switchEthereumChain',
      params: [{ chainId: config.chainId }]
    });

    return 'Network added & switched successfully';
  } catch (error) {
    console.error('MetaMask network setup failed', error);
    throw error; // Propagate so the caller can handle (e.g., display toast)
  }
};


// step:1 file: hash_arbitrary_data_using_keccak-256_(cast_keccak)
export const getKeccakHashFromInput = async () => {
  try {
    // Ask the user for the data to hash
    const data = window.prompt(
      "Enter data (plain text or 0x-prefixed hex) to hash with Keccak-256:"
    );

    if (!data) {
      throw new Error("No input provided.");
    }

    // Call the backend
    const response = await fetch("/api/keccak", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ data })
    });

    if (!response.ok) {
      // Surface backend error details if available
      const err = await response.json().catch(() => ({ error: "Unknown error" }));
      throw new Error(err.error || "Failed to compute Keccak hash.");
    }

    const { hash } = await response.json();
    return hash; // e.g., 0x…
  } catch (error) {
    console.error(error);
    alert(`Error: ${error.message}`);
    return null;
  }
};


// step:1 file: connect_a_user_s_wallet_to_the_dapp
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


// step:2 file: connect_a_user_s_wallet_to_the_dapp
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


// step:3 file: connect_a_user_s_wallet_to_the_dapp
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


// step:2 file: decode_binary_data_to_hexadecimal_using_foundry’s_cast_from-bin
export const prepareBinaryInput = async (file) => {
  if (!file) {
    throw new Error('No file supplied for conversion.');
  }

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/cast/from-bin', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Backend error: ${errorText}`);
  }

  const { hex } = await response.json();
  return hex; // Forward to the caller for further handling
};


// step:4 file: decode_binary_data_to_hexadecimal_using_foundry’s_cast_from-bin
export const captureOutput = async (hex) => {
  if (typeof hex !== 'string' || !hex.startsWith('0x')) {
    throw new Error('Invalid hex string supplied.');
  }
  try {
    await navigator.clipboard.writeText(hex);
    console.log('Hex string copied to clipboard');
  } catch (err) {
    console.error('Clipboard write failed:', err);
  }
  return hex; // Return for optional chaining
};


// step:1 file: retrieve_the_smart-contract_bytecode_deployed_at_a_given_evm_address_(latest_block)
export const validateEvmAddress = (address) => {
  // Ensure the value is a string
  if (typeof address !== 'string') {
    throw new Error('Address must be a string.');
  }

  // Basic regex: starts with 0x followed by exactly 40 hex chars (case-insensitive)
  const evmRegex = /^0x[a-fA-F0-9]{40}$/;

  if (!evmRegex.test(address)) {
    throw new Error('Invalid EVM address. Expect 0x + 40 hex characters.');
  }

  // All good — return true so calling code can proceed
  return true;
};


// step:2 file: run_mutex_profiling_for_10_seconds_and_write_output_to_mutex.prof
/* frontend/profileParams.js */

/**
 * Prepare the parameters needed for the mutex profiler.
 * Values are hard-coded per the workflow but could be made dynamic.
 */
export const prepareProfileParameters = () => ({
  duration: 10,          // seconds
  outputPath: 'mutex.prof'
});


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


// step:1 file: set_my_boost_target_to_my_ethereum_address
export const getNeutronAddress = async () => {
  const chainId = 'neutron-1'; // main-net chain ID
  try {
    if (!window?.keplr) {
      throw new Error('Keplr wallet is not installed.');
    }

    // Request wallet connection for the chain
    await window.keplr.enable(chainId);

    // Retrieve the signer and the account list
    const signer = window.getOfflineSigner(chainId);
    const accounts = await signer.getAccounts();

    if (!accounts.length) {
      throw new Error('No account found inside Keplr.');
    }

    return accounts[0].address;
  } catch (err) {
    console.error(err);
    throw err;
  }
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
import { rateToAPY } from './rateToAPY';

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

export default MarketTable;


// step:1 file: lock_a_specific_account_so_it_can_no_longer_send_transactions
/* wallet.js */
export const getAccountAddress = async () => {
  if (typeof window === "undefined" || !window.ethereum) {
    throw new Error("No Ethereum provider found. Make sure MetaMask is installed.");
  }

  try {
    // Prompt user to connect their wallet
    const accounts = await window.ethereum.request({
      method: "eth_requestAccounts"
    });

    if (!accounts || accounts.length === 0) {
      throw new Error("No accounts returned from provider.");
    }

    // Return the first account by convention
    return accounts[0];
  } catch (err) {
    console.error("Failed to fetch account address", err);
    throw new Error("Could not obtain account address. Check console for details.");
  }
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


// step:1 file: subscribe_to_hashes_of_new_pending_transactions_via_websocket
export const openWebsocketConnection = (url = 'ws://localhost:8546') => {
  return new Promise((resolve, reject) => {
    try {
      const ws = new WebSocket(url);

      // Resolve when the connection opens successfully
      ws.onopen = () => {
        console.log(`WebSocket connected to ${url}`);
        resolve(ws);
      };

      // Reject if an error occurs while connecting
      ws.onerror = (err) => {
        console.error('WebSocket connection error:', err);
        reject(new Error(`Unable to connect to ${url}`));
      };
    } catch (error) {
      reject(error);
    }
  });
};


// step:2 file: subscribe_to_hashes_of_new_pending_transactions_via_websocket
export const ethSubscribePendingTxs = (ws, requestId = 1) => {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    throw new Error('WebSocket is not open.');
  }

  const payload = {
    id: requestId,
    method: 'eth_subscribe',
    params: ['newPendingTransactions']
  };

  ws.send(JSON.stringify(payload));

  return requestId; // Return the request id used
};


// step:3 file: subscribe_to_hashes_of_new_pending_transactions_via_websocket
export const storeSubscriptionId = (ws, callback) => {
  const handler = (event) => {
    try {
      const msg = JSON.parse(event.data);

      // The response to our subscription request will have the same id we sent (1 by default) and a 'result' field.
      if (msg.id === 1 && msg.result) {
        const subscriptionId = msg.result;
        console.log('Received subscription id:', subscriptionId);

        // Persist for later use
        localStorage.setItem('pendingTxSubscriptionId', subscriptionId);

        // Stop listening for the acknowledgement once received
        ws.removeEventListener('message', handler);

        if (typeof callback === 'function') {
          callback(subscriptionId);
        }
      }
    } catch (err) {
      console.error('Failed to parse subscription response:', err);
    }
  };

  ws.addEventListener('message', handler);
};


// step:4 file: subscribe_to_hashes_of_new_pending_transactions_via_websocket
export const listenForPendingTxNotifications = (ws, callback) => {
  ws.addEventListener('message', (event) => {
    try {
      const msg = JSON.parse(event.data);

      // A notification will contain the method 'eth_subscription'
      if (msg.method === 'eth_subscription' && msg.params) {
        const { subscription, result: txHash } = msg.params;

        if (typeof callback === 'function') {
          callback({ subscriptionId: subscription, txHash });
        }
      }
    } catch (err) {
      // Non-JSON or unrelated messages can be safely ignored
    }
  });
};


// step:1 file: broadcast_a_raw,_rlp-encoded_signed_transaction_to_the_network
export const getSignedRawTx = async () => {
  // Prompt the user to paste a raw, signed transaction.
  // In production you would integrate with the wallet’s SDK instead of using window.prompt().
  const rawTx = window.prompt('Please paste the 0x-prefixed, RLP-encoded transaction:');

  // Basic validation
  if (!rawTx) {
    throw new Error('Transaction input cancelled by user.');
  }
  if (!rawTx.startsWith('0x')) {
    throw new Error('Raw transaction must start with 0x.');
  }
  if (rawTx.length < 10) {
    throw new Error('Raw transaction appears too short.');
  }

  return rawTx.trim();
};


// step:4 file: broadcast_a_raw,_rlp-encoded_signed_transaction_to_the_network
export const captureTxHash = (txHash) => {
  if (!txHash || !txHash.startsWith('0x')) {
    throw new Error('Invalid transaction hash received.');
  }
  const storageKey = 'sentTxs';
  const existing = JSON.parse(localStorage.getItem(storageKey) || '[]');
  existing.push({ hash: txHash, timestamp: Date.now() });
  localStorage.setItem(storageKey, JSON.stringify(existing));
  return txHash;
};


// step:2 file: send_a_signed_transaction_with_cast
export const getRpcUrl = () => {
  // You can hard-code, pull from an env variable, or prompt the user.
  const defaultRpc = 'https://rpc.evmos.dev'; // Example RPC; replace for your chain
  const rpc = window.prompt('Enter a Cosmos-EVM RPC URL', defaultRpc);
  if (!rpc) throw new Error('RPC URL is required.');
  return rpc.trim();
};


// step:4 file: send_a_signed_transaction_with_cast
export const constructTxParams = ({ recipient, amount, gasPrice, gasLimit, data = '' }) => {
  if (!recipient || !amount || !gasPrice || !gasLimit) {
    throw new Error('Missing required transaction parameter(s).');
  }
  return {
    recipient,
    amount,       // Value expressed in wei (e.g., '1000000000000000000' for 1 ETH-denom token)
    gas_price: gasPrice, // also in wei
    gas_limit: gasLimit, // integer
    data          // optional hex data string
  };
};


// step:1 file: mint_a_boost-receipt_nft_by_staking_250_ntrn_for_12_months
export const connectNeutronWallet = async (chainId = 'neutron-1') => {
  // Ensure Keplr is available
  if (typeof window === 'undefined' || !window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  try {
    await window.keplr.enable(chainId);
    const signer = window.keplr.getOfflineSigner(chainId);
    const accounts = await signer.getAccounts();
    if (!accounts || accounts.length === 0) {
      throw new Error('No account found in Keplr.');
    }
    return { signer, address: accounts[0].address };
  } catch (err) {
    console.error('Failed to connect to Keplr:', err);
    throw err;
  }
};


// step:1 file: initiate_standard_vesting_for_any_unclaimed_ntrn_rewards
export const getWalletAddress = async () => {
  const chainId = 'neutron-1';

  // Make sure Keplr is injected
  if (!window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  // Request connection to Neutron
  await window.keplr.enable(chainId);

  // Get the offline signer & account list
  const offlineSigner = window.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in the signer.');
  }

  // Return the first account’s address
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


// step:3 file: simulate_the_signed_transaction_to_estimate_gas
export const extractGasInfo = (simulationResponse) => {
  if (!simulationResponse || !simulationResponse.gas_info) {
    throw new Error('Missing gas_info in simulation response');
  }
  const { gas_used, gas_wanted } = simulationResponse.gas_info;
  return {
    gasUsed: Number(gas_used),
    gasWanted: Number(gas_wanted),
  };
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


// step:1 file: query_a_wallet’s_bank_balances_via_the_rest_api
/* utils/wallet.js */
export const collectTargetAddress = () => {
  // Assuming there is an <input id="address-input" /> in the DOM
  const raw = document.getElementById('address-input')?.value?.trim();
  // Basic sanity checks; for production use a bech32 library
  if (!raw || !/^([a-z0-9]+)1[0-9a-z]{38}$/.test(raw)) {
    throw new Error('Invalid bech32 address supplied.');
  }
  return raw;
};


// step:3 file: query_a_wallet’s_bank_balances_via_the_rest_api
/* services/balances.js */
export const fetchAndParseBalances = async (address) => {
  const res = await fetch(`/api/balances?address=${encodeURIComponent(address)}`);
  if (!res.ok) {
    const { detail } = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(`Backend error: ${detail}`);
  }
  const data = await res.json();
  // Ensure we always return an array even if no balances are found
  return (data.balances || []).map(({ denom, amount }) => ({ denom, amount }));
};


// step:1 file: inspect_full_txpool_contents_for_debugging_purposes
/* utils/txpool.js */
export const createTxpoolPayload = (depth = 'inspect') => {
  // Acceptable values are 'inspect' or 'content'. Default = 'inspect'.
  const allowed = ['inspect', 'content'];
  if (!allowed.includes(depth)) {
    throw new Error(`Invalid depth: ${depth}. Expected 'inspect' or 'content'.`);
  }

  const method = depth === 'content' ? 'txpool_content' : 'txpool_inspect';

  return {
    jsonrpc: '2.0',
    method,
    params: [],
    id: Date.now() // simple unique identifier
  };
};


// step:1 file: instantly_claim_50%_of_my_ntrn_staking_rewards
export const getDelegatorAddress = async (chainId = 'neutron-1') => {
  if (!window || !window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  try {
    // Prompt wallet connection / network enable
    await window.keplr.enable(chainId);

    // Retrieve signer & accounts
    const signer = window.getOfflineSigner(chainId);
    const accounts = await signer.getAccounts();

    if (!accounts.length) {
      throw new Error('No accounts found in the connected wallet.');
    }

    return {
      address: accounts[0].address,
      signer,
    };
  } catch (err) {
    console.error('[Keplr-Connect] ::', err);
    throw new Error('Failed to connect the wallet.');
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


// step:5 file: Set cron execution stage to BEGIN_BLOCKER for schedule health_check
import { SigningCosmWasmClient } from "@cosmjs/cosmwasm-stargate";

const rpcEndpoint = "https://rpc.neutron.org";
const daoAddress = "<MAIN_DAO_CONTRACT_ADDRESS>"; // replace with actual address

/**
 * Submit a proposal to the Main DAO.
 * @param {OfflineSigner} offlineSigner – Keplr/Leap signer already connected to the chain.
 * @param {string} sender – Bech32 address of the proposer.
 * @param {object} proposal – Parsed JSON { title, description, messages }.
 */
export const submitProposalToMainDao = async (offlineSigner, sender, proposal) => {
  try {
    const client = await SigningCosmWasmClient.connectWithSigner(rpcEndpoint, offlineSigner);

    // Wrap the Cron messages in a CW-dao single-choice proposal
    const execMsg = {
      propose: {
        msg: {
          propose_single: {
            title: proposal.title,
            description: proposal.description,
            msgs: proposal.messages,
          },
        },
      },
    };

    const fee = "auto";
    const result = await client.execute(sender, daoAddress, execMsg, fee);
    return result;
  } catch (error) {
    console.error("Failed to submit proposal", error);
    throw error;
  }
};


// step:1 file: submit_a_governance_text_proposal
/* utils/wallet.js */
export const ensureWalletConnected = async () => {
  try {
    const chainId = 'cosmoshub-4';            // ‼️ Replace with your target chain
    if (!window.keplr) {
      throw new Error('Keplr extension not found. Please install Keplr and refresh.');
    }

    // Request wallet connection
    await window.keplr.enable(chainId);

    // Get an OfflineSigner and extract the first account address
    const signer = window.getOfflineSigner(chainId);
    const [account] = await signer.getAccounts();

    if (!account?.address) {
      throw new Error('Unable to fetch wallet address.');
    }

    return account.address; // Bech32 address of the proposer
  } catch (error) {
    console.error('[ensureWalletConnected] ❌', error);
    throw error;
  }
};


// step:2 file: submit_a_governance_text_proposal
/* utils/metadata.js */
export const gatherProposalMetadata = ({
  title,
  description,
  summary,
  depositAmount,
  depositDenom
}) => {
  // Basic validation
  if (!title || !description || !summary) {
    throw new Error('Title, description, and summary are required.');
  }
  if (!/^[0-9]+$/.test(depositAmount)) {
    throw new Error('Deposit amount must be an integer string (micro-denom).');
  }
  if (!depositDenom) {
    throw new Error('Deposit denom is required (e.g., "uatom").');
  }

  return {
    title: title.trim(),
    description: description.trim(),
    summary: summary.trim(),
    deposit_amount: depositAmount.trim(),
    deposit_denom: depositDenom.trim()
  };
};


// step:3 file: submit_a_governance_text_proposal
/* services/proposal.js */
export const submitTextProposal = async ({ proposerAddress, metadata }) => {
  try {
    const res = await fetch('/api/proposal/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ proposer_address: proposerAddress, ...metadata })
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Backend error: ${err}`);
    }

    const data = await res.json();
    /* data = {
         tx_hash: 'ABC123...',
         proposal_id: 42
       }
    */
    return data;
  } catch (error) {
    console.error('[submitTextProposal] ❌', error);
    throw error;
  }
};


// step:3 file: Show last execution height for schedule daily_rewards
export const displayLastExecutionHeight = (height) => {
  if (height === undefined || height === null) {
    console.error('Height is not provided.');
    return;
  }
  console.log(`Last execution height: ${height}`);
  // You can additionally inject this into the DOM, e.g.,
  // document.getElementById('last-height').textContent = `Last execution height: ${height}`;
};


// step:3 file: delete_a_local_snapshot_stored_by_the_node
// frontend/snapshots.js

/**
 * fetchSnapshots — retrieves the current snapshot list from the backend.
 * After deleting a snapshot, call this again to verify the ID is gone.
 */
export const fetchSnapshots = async () => {
  try {
    const res = await fetch('/api/snapshots');
    if (!res.ok) {
      throw new Error(`Server responded with ${res.status}`);
    }
    const json = await res.json();
    return json.snapshots;
  } catch (err) {
    console.error('Failed to fetch snapshots', err);
    throw err;
  }
};


// step:1 file: fetch_the_current_evm_chain_id_via_json-rpc
export const prepareJsonRpcPayload = (
  method = "eth_chainId", // default method
  params = [],             // default empty params
  id = 1                   // default request id
) => {
  /*
    Returns a well-formed JSON-RPC 2.0 payload, e.g.
    {
      jsonrpc: "2.0",
      method: "eth_chainId",
      params: [],
      id: 1
    }
  */
  return {
    jsonrpc: "2.0",
    method,
    params,
    id
  };
};


// step:2 file: fetch_the_current_evm_chain_id_via_json-rpc
export const sendRpcRequest = async (
  endpoint = "http://localhost:8545", // default local endpoint
  payload
) => {
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      // Any non-2xx HTTP status is treated as an error
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const json = await response.json();
    return json;
  } catch (error) {
    console.error("Error sending RPC request:", error);
    throw error; // propagate so caller can handle
  }
};


// step:3 file: fetch_the_current_evm_chain_id_via_json-rpc
export const parseChainIdFromResponse = (
  rpcResponse,
  asDecimal = false // set true if you want decimal output
) => {
  // Handle JSON-RPC error object if present
  if (rpcResponse.error) {
    throw new Error(`RPC Error: ${rpcResponse.error.message || "Unknown error"}`);
  }

  const chainIdHex = rpcResponse.result;

  if (typeof chainIdHex !== "string") {
    throw new Error("Invalid RPC response: 'result' field is not a string");
  }

  // Return either hex (default) or decimal representation
  return asDecimal ? parseInt(chainIdHex, 16) : chainIdHex;
};


// step:1 file: query_the_balance_of_an_evm_address_at_the_latest_block
export const validateEthAddress = (address) => {
  const regex = /^0x[0-9a-fA-F]{40}$/;
  if (!regex.test(address)) {
    throw new Error('Invalid Ethereum address format.');
  }
  return true; // address is valid
};


// step:2 file: query_the_balance_of_an_evm_address_at_the_latest_block
export const selectRpcEndpoint = () => {
  // A small pool of free public RPC providers
  const endpoints = [
    'https://cloudflare-eth.com',
    'https://rpc.flashbots.net'
  ];
  return endpoints[Math.floor(Math.random() * endpoints.length)];
};


// step:4 file: query_the_balance_of_an_evm_address_at_the_latest_block
export const parseBalanceHexToDecimal = (balanceHex) => {
  if (!balanceHex || typeof balanceHex !== 'string') {
    throw new Error('balanceHex must be a non-empty hex string.');
  }
  try {
    const weiBigInt = BigInt(balanceHex); // BigInt handles the 0x prefix
    const weiStr = weiBigInt.toString(10);
    const etherStr = (weiBigInt / 10n ** 18n).toString(10);
    return { wei: weiStr, ether: etherStr };
  } catch (err) {
    throw new Error('Failed to parse balance: ' + err.message);
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


// step:1 file: Create a cron schedule named "daily_rewards" that distributes rewards every 7,200 blocks at END_BLOCKER
/* gatherScheduleInputs.js
 * Helper that can be wired to a form or wizard.
 */
export const gatherScheduleInputs = () => {
  // In a real app you would read these from form fields or a config file.
  const scheduleName = "daily_rewards";            // Unique schedule identifier
  const period = 7200;                              // Blocks between executions
  const executionStage = "EXECUTION_STAGE_END_BLOCKER"; // When to fire (Begin/End block)
  const targetContract = "neutron1contract...";     // Rewards contract address

  // CosmWasm execute payload that the cron job will run each period
  const rewardsMsg = {
    distribute: {}
  };

  // MsgExecuteContract that the Cron module will invoke
  const compiledExecuteMsg = {
    "@type": "/cosmwasm.wasm.v1.MsgExecuteContract",
    "sender": targetContract,         // will be overwritten by Cron when executed
    "contract": targetContract,
    "msg": Buffer.from(JSON.stringify(rewardsMsg)).toString("base64"),
    "funds": []
  };

  return {
    scheduleName,
    period,
    executionStage,
    authority: "neutron1mainDAOaddress...", // DAO (gov) address that controls Cron
    msgs: [compiledExecuteMsg]
  };
};


// step:4 file: Create a cron schedule named "daily_rewards" that distributes rewards every 7,200 blocks at END_BLOCKER
/* signAndBroadcastTx.js */
import { SigningStargateClient, GasPrice } from "@cosmjs/stargate";

export const signAndBroadcastTx = async (
  offlineSigner,
  rpcEndpoint,
  messages,
  memo = "",
) => {
  try {
    const client = await SigningStargateClient.connectWithSigner(
      rpcEndpoint,
      offlineSigner,
      { gasPrice: GasPrice.fromString("0.025untrn") }
    );

    const [{ address }] = await offlineSigner.getAccounts();

    // Auto estimate fee; fallback to explicit if needed
    const fee = await client.simulate(address, messages, memo).then((gas) => ({
      amount: [{ denom: "untrn", amount: "0" }],
      gas: String(Math.round(gas * 1.3)), // 30% safety margin
    }));

    const txResult = await client.signAndBroadcast(address, messages, fee, memo);

    if (txResult.code !== 0) throw new Error(`Tx failed: ${txResult.rawLog}`);
    return txResult.transactionHash;
  } catch (err) {
    console.error("signAndBroadcastTx error", err);
    throw err;
  }
};


// step:1 file: convert_a_wei_value_to_ether_(cast_from-wei)
/* utils/weiPrompt.js */
export const promptWeiAmount = () => {
  return new Promise((resolve, reject) => {
    try {
      // Ask the user for a Wei amount
      const wei = window.prompt(
        "Enter a Wei amount (e.g., 420000000000000000):"
      );

      // Basic validations ---------------------------------------------------
      if (wei === null) {
        return reject(new Error("Prompt was cancelled by the user."));
      }
      const trimmed = wei.trim();
      if (!/^\d+$/.test(trimmed)) {
        return reject(
          new Error("Wei amount must be a non-negative integer in string form.")
        );
      }

      // Return the clean, validated value
      resolve(trimmed);
    } catch (err) {
      reject(err);
    }
  });
};


// step:1 file: read_a_smart-contract’s_state_on_a_cosmos-sdk_ethermint_chain_using_the_viem_library
// rpc.js
export const getRpcEndpoint = () => {
  /*
   * Change this value to your own full-node URL or an official public RPC.
   * The example below targets the Evmos mainnet.
   */
  return 'https://evmos-rpc.polkachu.com';
};


// step:3 file: read_a_smart-contract’s_state_on_a_cosmos-sdk_ethermint_chain_using_the_viem_library
// abi.js
export const myContractAbi = [
  {
    inputs: [
      { internalType: 'address', name: 'account', type: 'address' }
    ],
    name: 'balanceOf',
    outputs: [
      { internalType: 'uint256', name: '', type: 'uint256' }
    ],
    stateMutability: 'view',
    type: 'function'
  }
  // 👉 add more fragments as required
];


// step:5 file: read_a_smart-contract’s_state_on_a_cosmos-sdk_ethermint_chain_using_the_viem_library
// readContract.js
export const readContract = async ({ contractAddress, abi, functionName, args = [] }) => {
  try {
    const res = await fetch('/api/read-contract', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ contractAddress, abi, functionName, args }),
    });

    if (!res.ok) {
      const { detail } = await res.json();
      throw new Error(detail || 'Unexpected server error');
    }

    const { data } = await res.json();
    return data;
  } catch (error) {
    console.error('Failed to read contract', error);
    throw error;
  }
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


// step:1 file: subscribe_to_new_block_headers_over_websocket
// openWebsocketConnection.js
// Establishes a WebSocket connection and resolves with the open socket instance.
export const openWebsocketConnection = (endpoint = 'ws://localhost:8546') => {
  return new Promise((resolve, reject) => {
    try {
      const ws = new WebSocket(endpoint);

      // Resolve once the socket is open
      ws.onopen = () => resolve(ws);

      // Bubble up any connection errors
      ws.onerror = (err) => {
        reject(new Error(`WebSocket connection error: ${err.message || err}`));
      };
    } catch (error) {
      reject(error);
    }
  });
};


// step:2 file: subscribe_to_new_block_headers_over_websocket
// ethSubscribeNewHeads.js
// Sends the JSON-RPC subscribe call and resolves with the returned subscription ID.
export const ethSubscribeNewHeads = (ws) => {
  return new Promise((resolve, reject) => {
    // Use a timestamp as a simple unique id; production code may want a counter.
    const id = Date.now();

    const request = {
      id,
      jsonrpc: '2.0',
      method: 'eth_subscribe',
      params: ['newHeads']
    };

    // Handler waits for the matching response frame.
    const handleMessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.id === id) {
          ws.removeEventListener('message', handleMessage);
          if (data.error) {
            reject(new Error(`Subscription failed: ${data.error.message}`));
          } else {
            resolve(data.result); // This is the subscription ID
          }
        }
      } catch (_) {
        /* Ignore non-JSON frames */
      }
    };

    ws.addEventListener('message', handleMessage);

    // Fire the request
    ws.send(JSON.stringify(request));
  });
};


// step:3 file: subscribe_to_new_block_headers_over_websocket
// storeSubscriptionId.js
// Simple in-module variable holder; you could swap this for IndexedDB, Redux, etc.
let currentSubscriptionId = null;

export const storeSubscriptionId = (subId) => {
  if (!subId) throw new Error('No subscription ID provided.');
  currentSubscriptionId = subId;
  return currentSubscriptionId;
};

export const getSubscriptionId = () => currentSubscriptionId;


// step:4 file: subscribe_to_new_block_headers_over_websocket
// listenForNewBlocks.js
// Listens for eth_subscription notifications and invokes a user-supplied callback.
export const listenForNewBlocks = (ws, subId, onNewBlock) => {
  if (!subId) throw new Error('Subscription ID is required to listen for notifications.');
  if (typeof onNewBlock !== 'function') throw new Error('onNewBlock callback must be a function.');

  const handleMessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (
        data.method === 'eth_subscription' &&
        data.params &&
        data.params.subscription === subId
      ) {
        // Pass the block header (data.params.result) to the callback
        onNewBlock(data.params.result);
      }
    } catch (_) {
      /* Ignore frames we can’t parse */
    }
  };

  ws.addEventListener('message', handleMessage);

  // Return a disposer in case the caller wants to stop listening later.
  return () => ws.removeEventListener('message', handleMessage);
};


// step:1 file: initialize_a_new_wallet_at_a_provided_url
export const getWalletUrl = async () => {
  // Prompt the user for the keystore location
  const rawUrl = window.prompt('Enter the remote URL where the wallet keystore will be created:');
  if (!rawUrl) {
    throw new Error('URL cannot be empty.');
  }

  try {
    const url = new URL(rawUrl);

    // Simple protocol + hostname validation
    if (!['http:', 'https:'].includes(url.protocol)) {
      throw new Error('URL must start with http:// or https://');
    }
    if (!url.hostname) {
      throw new Error('URL must contain a valid hostname.');
    }

    return url.toString();
  } catch (err) {
    throw new Error(`Invalid URL provided: ${err.message}`);
  }
};


// step:1 file: retrieve_the_list_of_pending_(unconfirmed)_transactions_from_the_cosmos_mempool
/* cometRpc.js */

export let COMET_RPC_ENDPOINT = 'http://localhost:26657';

/**
 * Return the currently-configured RPC endpoint.
 * @returns {string}
 */
export const getCometRpcEndpoint = () => COMET_RPC_ENDPOINT;

/**
 * Update the global RPC endpoint in a type-safe way.
 * @param {string} url – Full URL including protocol, e.g. "http://my-node:26657".
 */
export const setCometRpcEndpoint = (url) => {
  try {
    const parsed = new URL(url);
    // Strip trailing slash so we can safely concatenate paths later
    COMET_RPC_ENDPOINT = parsed.href.replace(/\/$/, '');
  } catch (err) {
    console.error('Invalid RPC endpoint supplied:', err);
    throw new Error('Provided RPC endpoint is not a valid URL.');
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


// step:1 file: connect_remix_ide_to_cosmos_evm_via_injected_provider
export const ensureMetaMaskInstalledAndUnlocked = async () => {
  // Verify MetaMask extension is present
  if (typeof window === 'undefined' || !window.ethereum || !window.ethereum.isMetaMask) {
    throw new Error('MetaMask is not installed. Please install it from https://metamask.io/');
  }

  // Request account access (prompts unlock if locked)
  const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' }).catch((err) => {
    console.error(err);
    throw new Error('User denied account authorization or MetaMask is locked.');
  });

  if (!accounts || accounts.length === 0) {
    throw new Error('No MetaMask accounts found.');
  }

  return accounts[0]; // Return the active address
};


// step:2 file: connect_remix_ide_to_cosmos_evm_via_injected_provider
export const addCosmosEvmNetwork = async () => {
  if (!window.ethereum) throw new Error('MetaMask is not available');

  const evmosChain = {
    chainId: '0x2329', // 9001 in hex
    chainName: 'Evmos',
    nativeCurrency: {
      name: 'EVMOS',
      symbol: 'EVMOS',
      decimals: 18
    },
    rpcUrls: ['https://eth.bd.evmos.org:8545'],
    blockExplorerUrls: ['https://escan.live']
  };

  try {
    await window.ethereum.request({
      method: 'wallet_addEthereumChain',
      params: [evmosChain]
    });
  } catch (error) {
    console.error(error);
    throw new Error('Failed to add the Evmos network to MetaMask.');
  }
};


// step:3 file: connect_remix_ide_to_cosmos_evm_via_injected_provider
export const openRemixIde = () => {
  const remixUrl = 'https://remix.ethereum.org';
  // Open Remix; MetaMask will prompt for connection in the new tab
  window.open(remixUrl, '_blank', 'noopener,noreferrer');
};


// step:4 file: connect_remix_ide_to_cosmos_evm_via_injected_provider
export const verifyInjectedProvider = async () => {
  if (!window.ethereum) {
    throw new Error('MetaMask provider not found.');
  }

  const chainId = await window.ethereum.request({ method: 'eth_chainId' });
  console.log(`Connected to chainId: ${chainId}`);

  // Warn if user is on a different network
  if (chainId !== '0x2329') {
    console.warn('MetaMask is not connected to the expected Evmos network.');
  }

  return chainId;
};


// step:6 file: connect_remix_ide_to_cosmos_evm_via_injected_provider
export const deployContract = async ({ bytecode }) => {
  if (!window.ethereum) throw new Error('MetaMask not found in browser.');

  // Request the active account
  const [from] = await window.ethereum.request({ method: 'eth_requestAccounts' });

  // Raw deployment transaction parameters
  const txParams = {
    from,
    data: bytecode
  };

  // Estimate gas for deployment
  try {
    txParams.gas = await window.ethereum.request({
      method: 'eth_estimateGas',
      params: [txParams]
    });
  } catch (error) {
    console.warn('Gas estimation failed, defaulting to 3,000,000.', error);
    txParams.gas = '0x2DC6C0'; // 3,000,000 in hex
  }

  // Send deployment transaction (MetaMask will prompt for confirmation)
  const txHash = await window.ethereum.request({
    method: 'eth_sendTransaction',
    params: [txParams]
  });

  console.log('Deployment transaction hash:', txHash);
  return txHash;
};


// step:1 file: vote_yes_on_proposal_5
export const getVoterAddress = async () => {
  const chainId = 'cosmoshub-4'; // Change to your chain if different

  // 1. Check that Keplr is present
  if (!window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  // 2. Request access to the chain
  await window.keplr.enable(chainId);

  // 3. Obtain the signer & first account (default)
  const signer = window.keplr.getOfflineSigner(chainId);
  const accounts = await signer.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in the signer.');
  }

  // 4. Return the bech32 address
  return accounts[0].address;
};


// step:2 file: vote_yes_on_proposal_5
export const validateProposal = async (proposalId = 5) => {
  const lcdEndpoint = 'https://api.cosmos.network'; // Public LCD; replace if you run your own

  const resp = await fetch(`${lcdEndpoint}/cosmos/gov/v1beta1/proposals/${proposalId}`);
  if (!resp.ok) {
    throw new Error(`Proposal ${proposalId} not found (HTTP ${resp.status}).`);
  }

  const data = await resp.json();
  const status = data?.proposal?.status;
  if (status !== 'PROPOSAL_STATUS_VOTING_PERIOD') {
    throw new Error(`Proposal ${proposalId} is not in voting period. Current status: ${status}`);
  }

  return data.proposal; // Return the full proposal object if the check passes
};


// step:5 file: vote_yes_on_proposal_5
export const queryVoteRecord = async (proposalId, voterAddress) => {
  const lcdEndpoint = 'https://api.cosmos.network';

  // Each chain may differ slightly in path. The Cosmos Hub LCD supports:
  // /cosmos/gov/v1beta1/proposals/{proposal_id}/votes/{voter}
  const url = `${lcdEndpoint}/cosmos/gov/v1beta1/proposals/${proposalId}/votes/${voterAddress}`;
  const res = await fetch(url);

  if (!res.ok) {
    throw new Error(`Failed to fetch vote record (HTTP ${res.status}).`);
  }

  const data = await res.json();
  const option = data?.vote?.option;

  if (option !== 'VOTE_OPTION_YES') {
    throw new Error(`Vote not recorded as YES. Found: ${option ?? 'none'}`);
  }

  return data.vote; // Full vote object
};


// step:1 file: switch_metamask_to_the_correct_cosmos-evm_network_before_contract_deployment
export const fetchCosmosEvmChainParams = async (chainSlug) => {
  // Build the GitHub raw URL for the desired chain.json
  const url = `https://raw.githubusercontent.com/cosmos/chain-registry/master/${chainSlug}/chain.json`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Unable to fetch chain data: ${response.statusText}`);
    }

    const chainData = await response.json();

    // Extract useful EVM-specific fields with sensible fallbacks
    const evmRpc   = chainData?.evm?.json_rpc || chainData?.apis?.rpc?.[0]?.address;
    const evmChainId = chainData?.evm?.chain_id;
    const symbol   = chainData?.currencies?.[0]?.symbol || chainData?.native_currency?.symbol;
    const explorer = chainData?.explorers?.[0]?.url || null;

    if (!evmRpc || !evmChainId) {
      throw new Error('Fetched chain data is missing required EVM fields.');
    }

    return {
      chainSlug,
      evmRpc,
      evmChainId,
      symbol,
      explorer,
      raw: chainData // keep raw JSON for advanced use-cases
    };
  } catch (error) {
    console.error('fetchCosmosEvmChainParams error:', error);
    throw error;
  }
};


// step:2 file: switch_metamask_to_the_correct_cosmos-evm_network_before_contract_deployment
export const getMetaMaskCurrentChainId = async () => {
  if (!window.ethereum) {
    throw new Error('MetaMask is not installed');
  }
  // Returned value is hex (e.g., "0x2329" for 9001); convert to integer.
  const chainIdHex = await window.ethereum.request({ method: 'eth_chainId' });
  return parseInt(chainIdHex, 16);
};


// step:3 file: switch_metamask_to_the_correct_cosmos-evm_network_before_contract_deployment
export const switchToCosmosEvmNetwork = async (chainParams) => {
  if (!window.ethereum) {
    throw new Error('MetaMask is not installed');
  }

  const { evmChainId, symbol, evmRpc, explorer, raw } = chainParams;
  const chainIdHex = '0x' + evmChainId.toString(16);

  try {
    // First try to switch directly
    await window.ethereum.request({
      method: 'wallet_switchEthereumChain',
      params: [{ chainId: chainIdHex }]
    });
  } catch (switchError) {
    // Error code 4902 = chain not added yet
    if (switchError.code === 4902) {
      try {
        // Add the network
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [{
            chainId: chainIdHex,
            chainName: raw?.pretty_name || chainParams.chainSlug,
            nativeCurrency: {
              name: symbol,
              symbol: symbol,
              decimals: 18 // Most Cosmos EVM coins use 18 decimals
            },
            rpcUrls: [evmRpc],
            blockExplorerUrls: explorer ? [explorer] : []
          }]
        });
        // Then switch again
        await window.ethereum.request({
          method: 'wallet_switchEthereumChain',
          params: [{ chainId: chainIdHex }]
        });
      } catch (addError) {
        console.error('Error adding chain to MetaMask:', addError);
        throw addError;
      }
    } else {
      console.error('Error switching chain in MetaMask:', switchError);
      throw switchError;
    }
  }
};


// step:4 file: switch_metamask_to_the_correct_cosmos-evm_network_before_contract_deployment
export const verifyMetaMaskChainId = async (expectedChainId) => {
  const current = await getMetaMaskCurrentChainId();
  if (current !== expectedChainId) {
    throw new Error(`Chain ID verification failed. Expected ${expectedChainId}, got ${current}`);
  }
  return true; // Success
};


// step:1 file: execute_an_emergency_withdrawal_for_the_user’s_amber_trading_position
/* src/utils/wallet.js */
export const getUserWalletAddress = async (chainId = 'neutron-1') => {
  // Ensure Keplr is injected
  const { keplr } = window;
  if (!keplr) {
    throw new Error('Keplr extension not found. Install it first.');
  }

  // Ask Keplr to enable the Neutron network (prompts the user if not already enabled)
  await keplr.enable(chainId);

  // Obtain an OfflineSigner and read the first account
  const offlineSigner = keplr.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No Neutron account found in Keplr.');
  }

  return accounts[0].address; // Bech32 address (e.g. neutron1...)
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


// step:1 file: query_the_node’s_client_version_using_foundry’s_`cast`_cli
export const getRpcEndpoint = () => {
  // Read the value injected at build time by Mintlify (or any React-style env system)
  const rpcUrl = process.env.NEXT_PUBLIC_COSMOS_RPC_URL;

  if (!rpcUrl) {
    // Fallback to a well-known public RPC if the env var is missing
    console.warn(
      'NEXT_PUBLIC_COSMOS_RPC_URL is not defined. Falling back to a public mainnet RPC.'
    );
    return 'https://cosmos-rpc.polkachu.com:443';
  }

  return rpcUrl;
};


// step:2 file: query_the_node’s_client_version_using_foundry’s_`cast`_cli
export const fetchNodeVersion = async (rpcUrl) => {
  try {
    const response = await fetch(`${rpcUrl}/status`);

    if (!response.ok) {
      throw new Error(`RPC responded with HTTP status ${response.status}`);
    }

    const data = await response.json();

    // Tendermint RPC usually places the version in one of these fields
    const version =
      data?.result?.node_info?.version ||
      data?.result?.application_version?.version ||
      'Unknown';

    return {
      raw: data,
      version,
    };
  } catch (error) {
    console.error('Unable to fetch node version', error);
    throw error;
  }
};


// step:1 file: compute_a_create2_contract_address_(cast_compute-address)
/*
 * getCreate2Address()
 * Prompts the user for the required parameters and calls the backend
 * FastAPI endpoint `/api/compute_create2_address`.
 */
export const getCreate2Address = async () => {
  try {
    // Prompt the user for the three required parameters
    const deployer = prompt("Enter the deployer (factory) address (0x…)");
    const salt = prompt("Enter the 32-byte salt in hex (WITHOUT 0x)");
    const initCodeHash = prompt("Enter the contract init-code hash (0x…)");

    // Basic client-side validation
    if (!deployer || !salt || !initCodeHash) {
      throw new Error("All three fields are required.");
    }

    // Send the data to the backend for processing
    const res = await fetch("/api/compute_create2_address", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        deployer,
        salt,
        init_code_hash: initCodeHash
      })
    });

    if (!res.ok) {
      // Surface backend error message, if any
      const err = await res.json();
      throw new Error(err.detail || "Backend error while computing address.");
    }

    const { create2_address } = await res.json();
    alert(`Deterministic CREATE2 address: ${create2_address}`);
    return create2_address;
  } catch (error) {
    console.error(error);
    alert(error.message);
    return null;
  }
};


// step:1 file: fetch_a_transaction_receipt_with_cast_receipt
export const getValidatedTxHash = (rawHash) => {
  // Ensure the input is non-empty and a string
  if (!rawHash || typeof rawHash !== 'string') {
    throw new Error('Transaction hash is required.');
  }

  // Remove leading / trailing whitespace
  const txHash = rawHash.trim();

  // A 32-byte hash is 64 hex chars + the 0x prefix ⇒ 66 chars total
  const regex = /^0x([A-Fa-f0-9]{64})$/;
  if (!regex.test(txHash)) {
    throw new Error('Invalid transaction hash. Expected 32-byte hex string with a 0x prefix.');
  }

  return txHash;
};


// step:5 file: verify_a_smart_contract_on_an_etherscan-compatible_explorer_using_hardhat
export const checkVerificationStatus = async (network, contractAddress, apiKey) => {
  const baseUrlMap = {
    mainnet: 'https://api.etherscan.io/api',
    goerli: 'https://api-goerli.etherscan.io/api',
    sepolia: 'https://api-sepolia.etherscan.io/api',
  };

  const baseUrl = baseUrlMap[network];
  if (!baseUrl) throw new Error(`Unsupported network ${network}`);

  const url = `${baseUrl}?module=contract&action=getsourcecode&address=${contractAddress}&apikey=${apiKey}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Explorer responded with ${res.status}`);
    const data = await res.json();
    const verified = data.status === '1' && data.result && data.result.length > 0 && data.result[0].SourceCode !== '';
    return { verified, raw: data };
  } catch (err) {
    console.error(err);
    throw new Error('Failed to fetch verification status');
  }
};


// step:1 file: lock_an_additional_500_ntrn_for_24_months_(boost)
export const getNeutronAddress = async () => {
  const chainId = 'neutron-1';
  const { keplr } = window;

  if (!keplr) {
    throw new Error('Keplr wallet not found. Please install or unlock Keplr.');
  }

  // Ask the wallet to enable the chain and expose an OfflineSigner
  await keplr.enable(chainId);
  const signer = window.getOfflineSigner(chainId);

  const accounts = await signer.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error('No account detected in the wallet.');
  }

  return accounts[0].address; // Neutron bech32 address
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


// step:1 file: lend_2_unibtc_on_amber_finance
export const getSenderAddress = async () => {
  try {
    const chainId = 'neutron-1';
    if (!window?.keplr) {
      throw new Error('Keplr wallet is not installed');
    }

    // Request Keplr to enable the Neutron chain
    await window.keplr.enable(chainId);
    const offlineSigner = window.getOfflineSigner(chainId);
    const accounts = await offlineSigner.getAccounts();

    if (!accounts || accounts.length === 0) {
      throw new Error('No account found in Keplr');
    }

    return accounts[0].address;
  } catch (err) {
    console.error('Failed to get wallet address', err);
    throw err;
  }
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


// step:1 file: cancel_(unlock)_the_user’s_ntrn_stake_lock_once_the_vesting_period_has_ended
// src/utils/wallet.js
export const getUserWalletAddress = async (chainId = 'neutron-1') => {
  try {
    if (!window || !window.keplr) {
      throw new Error('Keplr wallet extension not found');
    }

    // Request wallet access
    await window.keplr.enable(chainId);

    // Obtain the signer and the first account
    const offlineSigner = window.getOfflineSigner(chainId);
    const accounts = await offlineSigner.getAccounts();

    if (!accounts || accounts.length === 0) {
      throw new Error('No account found in wallet');
    }

    return {
      address: accounts[0].address,
      offlineSigner
    };
  } catch (err) {
    console.error('[getUserWalletAddress] ', err);
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
import { toHex, fromAscii } from "./utils"; // helper functions to convert to hex if desired (implementation omitted for brevity)

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


// step:1 file: fetch_the_transaction_receipt_for_a_given_hash
export const validateTxHash = (txHash) => {
  // Ethereum-style transaction hashes are 0x-prefixed and 66 characters long
  const pattern = /^0x([A-Fa-f0-9]{64})$/;

  if (!pattern.test(txHash)) {
    throw new Error(
      'Invalid transaction hash. It must be 0x-prefixed and 66 characters long.'
    );
  }

  // Return the hash unchanged so it can be safely re-used downstream
  return txHash;
};


// step:2 file: fetch_the_transaction_receipt_for_a_given_hash
/* Public RPC URLs for EVM-compatible Cosmos chains (e.g., Evmos).  
 * Feel free to add/remove endpoints that suit your deployment.
 */
const RPC_ENDPOINTS = [
  'https://eth.bd.evmos.dev:8545/',
  'https://rpc-evmos.cosmos.directory/',
  'https://evmos-json-rpc.publicnode.com'
];

export const selectRpcEndpoint = () => {
  // Basic (pseudo) load-balancing via random selection
  const index = Math.floor(Math.random() * RPC_ENDPOINTS.length);
  return RPC_ENDPOINTS[index];
};


// step:5 file: fetch_the_transaction_receipt_for_a_given_hash
export const interpretReceipt = (receipt) => {
  if (!receipt) {
    throw new Error('No receipt supplied');
  }

  const {
    status,
    gasUsed,
    cumulativeGasUsed,
    effectiveGasPrice,
    blockNumber,
    transactionHash,
    logs,
  } = receipt;

  // Helper to convert hex strings (e.g., "0x1a") into decimal numbers
  const hexToInt = (hex) => parseInt(hex, 16);

  return {
    transactionHash,
    blockNumber: hexToInt(blockNumber),
    status: status === '0x1' ? 'Success' : 'Failure',
    gasUsed: hexToInt(gasUsed),
    cumulativeGasUsed: hexToInt(cumulativeGasUsed),
    effectiveGasPrice: hexToInt(effectiveGasPrice),
    logs, // Raw logs are surfaced for the caller to decode as needed
  };
};


// step:1 file: broadcast_a_msgsend_transaction_(bank_module)_via_cli
export const getSenderAddress = async (chainId = 'cosmoshub-4') => {
  try {
    if (!window.keplr) {
      throw new Error('Keplr wallet extension not found');
    }

    // Ask the extension to enable the selected chain.
    await window.keplr.enable(chainId);

    // Obtain an OfflineSigner and read the first account.
    const offlineSigner = window.getOfflineSigner(chainId);
    const accounts = await offlineSigner.getAccounts();

    if (!accounts || accounts.length === 0) {
      throw new Error('No account found for the given chain.');
    }

    return accounts[0].address;
  } catch (error) {
    console.error('getSenderAddress error:', error);
    throw error;
  }
};


// step:2 file: get_details_of_the_latest_block_with_cast_block_latest
export const fetchLatestBlock = async () => {
  const res = await fetch('/api/latest_block');
  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`);
  }
  const json = await res.json();
  return json.result?.block ?? json; // fallback if structure differs
};


// step:1 file: open_a_5×_leveraged_loop_position_with_1_maxbtc_on_amber
export const getUserAddress = async () => {
  const chainId = 'neutron-1'; // Main-net; replace with the test-net ID if needed

  // 1. Make sure Keplr is injected in the page
  if (!window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  // 2. Ask the wallet to enable the selected chain
  await window.keplr.enable(chainId);

  // 3. Fetch the first account from the signer
  const offlineSigner = window.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();
  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in Keplr.');
  }

  return accounts[0].address; // Neutron bech-32 address
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
export const displayRewardPolicy = (policy, targetElementId = "reward-policy") => {
  try {
    const el = document.getElementById(targetElementId);

    // Fallback if no DOM target is available
    const output = (msg) => {
      if (el) {
        el.innerHTML = msg;
      } else {
        console.log(msg);
      }
    };

    if (!policy || !policy.isForfeitable) {
      output("No early withdrawal penalties. All rewards are fully claimable.");
      return;
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

    output(html);
  } catch (err) {
    console.error("[displayRewardPolicy]", err);
  }
};


// step:1 file: convert_an_ether_value_to_wei_(cast_to-wei)
export const convertEtherToWei = async () => {
  // 1️⃣ Prompt the user for an Ether amount
  const amount = prompt('Enter Ether amount to convert into Wei (e.g., 0.42):');

  // Exit if the user cancels the prompt
  if (amount === null) {
    return null;
  }

  // 2️⃣ Basic validation
  if (amount.trim() === '' || isNaN(amount) || Number(amount) < 0) {
    alert('Please enter a valid positive number.');
    return null;
  }

  try {
    // 3️⃣ POST the amount to the backend
    const resp = await fetch('/api/to_wei', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount: amount.trim() })
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Conversion failed');
    }

    // 4️⃣ Parse and display the Wei result
    const data = await resp.json();
    alert(`${amount} ETH is ${data.wei} Wei`);
    return data.wei;
  } catch (error) {
    console.error('Error converting ETH to Wei:', error);
    alert(error.message);
    throw error;
  }
};


// step:1 file: trace_a_block_by_hash_with_debug_traceblockbyhash
/* eslint-disable no-undef */
// src/utils/traceBlock.js
// Helper to request an EVM block trace from the backend
export const traceBlockByHash = async ({ rpcUrl, blockHash }) => {
  // Basic argument validation
  if (!rpcUrl) throw new Error('Parameter "rpcUrl" is required');
  if (!blockHash || !blockHash.startsWith('0x')) {
    throw new Error('Parameter "blockHash" must be a 0x-prefixed hash');
  }

  // Send the request to the backend-for-frontend (BFF)
  const res = await fetch('/api/traceBlockByHash', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rpc_url: rpcUrl, block_hash: blockHash })
  });

  // Handle network-level errors
  if (!res.ok) {
    const { error } = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(error || 'Backend returned a non-200 status');
  }

  // Parse and return the trace result
  const { trace } = await res.json();
  return trace;
};


// step:1 file: estimate_the_gas_required_for_a_simple_native-token_transfer
/*
 * Collect and validate the basic transfer parameters.
 * @param {string} senderAddress  – 0x-prefixed hex address (40 chars)
 * @param {string} recipientAddress – 0x-prefixed hex address (40 chars)
 * @param {string|number|bigint} valueWei – amount in Wei
 * @returns {{senderAddress:string,recipientAddress:string,valueWei:string}}
 */
export const collectTransferParameters = (senderAddress, recipientAddress, valueWei) => {
  const ethAddrRegex = /^0x[a-fA-F0-9]{40}$/;

  if (!ethAddrRegex.test(senderAddress)) {
    throw new Error("Invalid sender address provided.");
  }
  if (!ethAddrRegex.test(recipientAddress)) {
    throw new Error("Invalid recipient address provided.");
  }
  if (valueWei === undefined || valueWei === null || valueWei === "") {
    throw new Error("Value (in Wei) is required.");
  }

  return {
    senderAddress,
    recipientAddress,
    valueWei: valueWei.toString() // always return as string for consistency
  };
};


// step:2 file: estimate_the_gas_required_for_a_simple_native-token_transfer
/*
 * Construct a minimal tx object for eth_estimateGas.
 * Adds 0x-prefixed hex values for the `value` field and blank `data`.
 */
export const constructTxObject = ({ senderAddress, recipientAddress, valueWei }) => {
  const valueHex = "0x" + BigInt(valueWei).toString(16);
  return {
    from: senderAddress,
    to: recipientAddress,
    value: valueHex,
    data: "0x" // empty data for a native transfer
  };
};


// step:3 file: estimate_the_gas_required_for_a_simple_native-token_transfer
/*
 * Return a public Ethereum JSON-RPC endpoint.
 * Replace with your preferred RPC service if desired.
 */
export const getRpcEndpoint = () => {
  // Cloudflare's public Ethereum gateway (Mainnet)
  return "https://cloudflare-eth.com";
};


// step:4 file: estimate_the_gas_required_for_a_simple_native-token_transfer
/*
 * Estimate gas by calling eth_estimateGas.
 * @param {object} txObject – built in Step 2
 * @param {string} rpcEndpoint – RPC URL obtained in Step 3
 * @returns {Promise<string>} – hex string like "0x5208"
 */
export const estimateGas = async (txObject, rpcEndpoint = getRpcEndpoint()) => {
  try {
    const payload = {
      jsonrpc: "2.0",
      id: 1,
      method: "eth_estimateGas",
      params: [txObject]
    };

    const res = await fetch(rpcEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`Network error: ${res.status} ${res.statusText}`);
    }

    const json = await res.json();

    if (json.error) {
      throw new Error(`RPC Error: ${json.error.message || "Unknown error"}`);
    }

    if (!json.result) {
      throw new Error("No result field returned from RPC response.");
    }

    return json.result; // gas in hexadecimal string
  } catch (err) {
    console.error("estimateGas() failed", err);
    throw err;
  }
};


// step:5 file: estimate_the_gas_required_for_a_simple_native-token_transfer
/*
 * Convert a hex gas value to decimal and add a safety buffer.
 * @param {string} gasHex – hex string from Step 4
 * @param {number} [bufferFraction=0.1] – e.g. 0.1 adds 10 % extra gas
 * @returns {{gas:string, adjustedGas:string}}
 */
export const parseGasHexToDecimal = (gasHex, bufferFraction = 0.1) => {
  if (!gasHex || typeof gasHex !== "string" || !gasHex.startsWith("0x")) {
    throw new Error("Invalid hex gas value provided.");
  }

  const gasBigInt = BigInt(gasHex);
  // Calculate buffer: gas * bufferFraction. Using integer arithmetic.
  const bufferGas = gasBigInt / BigInt(Math.round(1 / bufferFraction));
  const adjustedGas = gasBigInt + bufferGas;

  return {
    gas: gasBigInt.toString(),
    adjustedGas: adjustedGas.toString()
  };
};


// step:2 file: query_an_account’s_balance_with_cast
export const getRpcUrl = (chainName) => {
  /*
   * Returns a pre-defined HTTPS RPC endpoint for popular Cosmos EVM chains.
   */
  const RPC_ENDPOINTS = {
    evmos: 'https://evmos-evm.publicnode.com',
    cronos: 'https://evm.cronos.org',
    kava: 'https://evm.kava.io'
  };

  const endpoint = RPC_ENDPOINTS[chainName.toLowerCase()];
  if (!endpoint) {
    throw new Error(`No RPC endpoint configured for chain: ${chainName}`);
  }
  return endpoint;
};


// step:1 file: replace_(speed-up)_a_pending_transaction_by_resending_it_with_a_higher_gas_price
/* ---------------------------------------------
 * fetchTransactionByHash(txHash, rpcUrl)
 * ---------------------------------------------
 * Returns: the transaction JSON object or throws.
 */
export const fetchTransactionByHash = async (txHash, rpcUrl) => {
  const payload = {
    jsonrpc: "2.0",
    id: Date.now(),
    method: "eth_getTransactionByHash",
    params: [txHash]
  };

  const res = await fetch(rpcUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    throw new Error(`RPC call failed with status ${res.status}`);
  }

  const json = await res.json();
  if (json.error) throw new Error(json.error.message);
  if (!json.result) throw new Error("Transaction not found on RPC node");

  return json.result; // full tx object
};


// step:2 file: replace_(speed-up)_a_pending_transaction_by_resending_it_with_a_higher_gas_price
/* ---------------------------------------------
 * verifyTxPending(tx)
 * ---------------------------------------------
 * Throws if the tx is NOT replaceable.
 */
export const verifyTxPending = (tx) => {
  if (tx.blockNumber !== null) {
    throw new Error("Original transaction is already mined; cannot be replaced.");
  }
  return true;
};


// step:3 file: replace_(speed-up)_a_pending_transaction_by_resending_it_with_a_higher_gas_price
/* ----------------------------------------------------------
 * constructReplacementTx(originalTx, { multiplier })
 * ----------------------------------------------------------
 * - Keeps nonce, to, data, value, gas.
 * - Increases gasPrice (or maxFeePerGas / maxPriorityFeePerGas).
 * - Returns a txParams object ready for signing.
 */
export const constructReplacementTx = (
  originalTx,
  { multiplier = 1.2 } = {}
) => {
  const toHex = (bn) => "0x" + bn.toString(16);
  const bump = (valueHex) => {
    const orig = BigInt(valueHex);
    // ensure we bump by at least +10% even if multiplier is small
    const bumped = orig * BigInt(Math.round(multiplier * 100)) / BigInt(100);
    return bumped > orig ? bumped : orig + orig / BigInt(10);
  };

  const baseParams = {
    from: originalTx.from,
    to: originalTx.to,
    data: originalTx.input,
    gas: originalTx.gas,
    nonce: originalTx.nonce,
    value: originalTx.value
  };

  // Legacy gas pricing ---------------------------------------------------
  if (originalTx.gasPrice && originalTx.gasPrice !== "0x0") {
    baseParams.gasPrice = toHex(bump(originalTx.gasPrice));
  }

  // EIP-1559 style -------------------------------------------------------
  if (originalTx.maxFeePerGas) {
    baseParams.maxFeePerGas = toHex(bump(originalTx.maxFeePerGas));
    baseParams.maxPriorityFeePerGas = toHex(
      bump(originalTx.maxPriorityFeePerGas || "0x0")
    );
  }

  return baseParams;
};


// step:4 file: replace_(speed-up)_a_pending_transaction_by_resending_it_with_a_higher_gas_price
/* ----------------------------------------------------------
 * signReplacementTx(txParams, { chainId })
 * ----------------------------------------------------------
 * Returns rawTx hex *or* txHash if wallet automatically broadcasts.
 */
export const signReplacementTx = async (txParams, { chainId } = {}) => {
  if (!window.ethereum) throw new Error("No EVM-compatible wallet detected.");

  // Add chainId if provided (helps some wallets)
  if (chainId) txParams.chainId = chainId;

  try {
    // 1️⃣  Attempt to just sign (keeps rawTx for manual broadcast)
    const raw = await window.ethereum.request({
      method: "eth_signTransaction",
      params: [txParams]
    });
    if (!raw || !raw.raw) throw new Error("Wallet did not return raw transaction");
    return { rawTx: raw.raw };
  } catch (e) {
    // 2️⃣  Fallback: let the wallet sign *and* send (most common path)
    const txHash = await window.ethereum.request({
      method: "eth_sendTransaction",
      params: [txParams]
    });
    return { txHash }; // already broadcast, nothing left to do
  }
};


// step:5 file: replace_(speed-up)_a_pending_transaction_by_resending_it_with_a_higher_gas_price
/* ----------------------------------------------------------
 * sendRawTransaction(rawTxHex, rpcUrl)
 * ----------------------------------------------------------
 * Broadcasts and returns the new txHash.
 */
export const sendRawTransaction = async (rawTxHex, rpcUrl) => {
  const payload = {
    jsonrpc: "2.0",
    id: Date.now(),
    method: "eth_sendRawTransaction",
    params: [rawTxHex]
  };

  const res = await fetch(rpcUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!res.ok) throw new Error(`Broadcast failed – HTTP ${res.status}`);
  const json = await res.json();
  if (json.error) throw new Error(json.error.message);
  return json.result; // txHash
};


// step:1 file: query_current_txpool_status_(pending_vs._queued_counts)
/*
 * prepareJsonRpcPayload.js
 * -----------------------
 * Returns a JSON object ready to be sent to a JSON-RPC endpoint.
 */
export const prepareJsonRpcPayload = () => {
  return {
    jsonrpc: "2.0",
    method: "txpool_status",
    params: [],
    id: 1,
  };
};


// step:2 file: query_current_txpool_status_(pending_vs._queued_counts)
/*
 * postJsonRpc.js
 * --------------
 * Sends a POST request with a JSON-RPC payload.
 *
 * Parameters:
 *   endpoint (string) – Full URL to the JSON-RPC server.
 *   payload  (object) – Payload created in Step 1.
 *
 * Returns:
 *   The parsed JSON received from the server.
 */
export const postJsonRpc = async (endpoint, payload) => {
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      // Convert non-2xx HTTP codes into explicit errors.
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const json = await response.json();
    return json;
  } catch (error) {
    console.error("postJsonRpc error:", error);
    throw error;
  }
};


// step:3 file: query_current_txpool_status_(pending_vs._queued_counts)
/*
 * parseTxpoolStatus.js
 * --------------------
 * Extracts `pending` and `queued` values from the txpool_status response.
 */
export const parseTxpoolStatus = (jsonRpcResponse) => {
  if (!jsonRpcResponse) {
    throw new Error("Empty response object");
  }

  if (jsonRpcResponse.error) {
    // Surface the JSON-RPC error to the caller.
    throw new Error(`JSON-RPC Error ${jsonRpcResponse.error.code}: ${jsonRpcResponse.error.message}`);
  }

  const { result } = jsonRpcResponse;
  if (!result || typeof result.pending !== "string" || typeof result.queued !== "string") {
    throw new Error("Malformed txpool_status response");
  }

  // Convert hex strings (e.g., "0x4e") to decimal numbers.
  const pending = parseInt(result.pending, 16);
  const queued = parseInt(result.queued, 16);

  return { pending, queued };
};


// step:1 file: deposit_3_ebtc_into_the_maxbtc_ebtc_supervault
/*
 * ensureWalletAndGetAddress.js
 * Utility to make sure Keplr is installed/enabled and to fetch the active Neutron address.
 */
export const ensureWalletAndGetAddress = async () => {
  try {
    const chainId = 'neutron-1';

    // Basic Keplr presence check
    if (!window.keplr) {
      throw new Error('Keplr wallet is not installed. Please install it first.');
    }

    // Request connection to Neutron chain
    await window.keplr.enable(chainId);

    // Fetch the offline signer & accounts list
    const offlineSigner = window.keplr.getOfflineSigner(chainId);
    const accounts      = await offlineSigner.getAccounts();

    if (!accounts || accounts.length === 0) {
      throw new Error('No Neutron account found in Keplr.');
    }

    return accounts[0].address; // sender / depositor address
  } catch (err) {
    console.error('[ensureWalletAndGetAddress] →', err);
    throw err;
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


// step:1 file: retrieve_the_latest_block_number_from_the_cosmos_evm_json-rpc_endpoint
/* selectRpcEndpoint.js
 * Returns a JSON-RPC URL that any browser can reach.
 * Replace the placeholder with an actual public endpoint or an
 * environment-specific value if needed.
 */
export const selectRpcEndpoint = () => {
  // Fallback public RPC; change to your preferred provider.
  const defaultRpc = 'https://evmos-jsonrpc.publicnode.com';

  // Basic runtime validation.
  if (!defaultRpc.startsWith('http')) {
    throw new Error('Invalid RPC endpoint. Please supply a valid HTTP/WS URL.');
  }

  return defaultRpc;
};


// step:2 file: retrieve_the_latest_block_number_from_the_cosmos_evm_json-rpc_endpoint
/* fetchLatestBlockNumber.js
 * Issues an `eth_blockNumber` JSON-RPC call and returns the result as a hex string.
 */
export const fetchLatestBlockNumber = async (rpcUrl) => {
  const payload = {
    jsonrpc: '2.0',
    id: 1,
    method: 'eth_blockNumber',
    params: []
  };

  try {
    const response = await fetch(rpcUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    // Network-level error handling.
    if (!response.ok) {
      throw new Error(`RPC request failed with status ${response.status}`);
    }

    const data = await response.json();

    // JSON-RPC-level error handling.
    if (data.error) {
      throw new Error(`RPC error: ${data.error.message || JSON.stringify(data.error)}`);
    }

    return data.result; // e.g., "0x1a2b3c"
  } catch (err) {
    console.error('[fetchLatestBlockNumber] →', err);
    throw err;
  }
};


// step:3 file: retrieve_the_latest_block_number_from_the_cosmos_evm_json-rpc_endpoint
/* hexToDecimal.js
 * Safely converts a 0x-prefixed hex string to a JavaScript number.
 * Note: For extremely large block heights, use BigInt instead of Number.
 */
export const hexToDecimal = (hexString) => {
  if (typeof hexString !== 'string' || !/^0x[0-9a-fA-F]+$/.test(hexString)) {
    throw new Error('Invalid hex string supplied to hexToDecimal');
  }

  // Use BigInt for correctness on 64-bit+ values.
  return Number(BigInt(hexString));
};


// step:1 file: provide_new_liquidity_to_the_wbtc_lbtc_supervault_with_1_wbtc_and_1_lbtc
export const getUserWalletAddress = async () => {
  const chainId = 'neutron-1';

  // Make sure Keplr is available in the browser
  if (!window?.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  try {
    // Ask the extension to enable the Neutron chain
    await window.keplr.enable(chainId);

    // Obtain the OfflineSigner and fetch the first account
    const signer = window.keplr.getOfflineSigner(chainId);
    const accounts = await signer.getAccounts();

    if (!accounts || accounts.length === 0) {
      throw new Error('No account found in the signer.');
    }

    return accounts[0].address;
  } catch (error) {
    console.error('Failed to fetch wallet address', error);
    throw error;
  }
};


// step:1 file: create_an_ibc_transfer_on_an_unordered_channel_with_a_unique_timeout_timestamp
export const getUnorderedIBCChannels = async (lcdEndpoint) => {
  try {
    const response = await fetch(`${lcdEndpoint}/ibc/core/channel/v1beta1/channels?pagination.limit=1000`);
    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    if (!data.channels) {
      throw new Error('No channels data found.');
    }
    return data.channels
      .filter((ch) => {
        const ordering = (ch.ordering || ch.order || '').toUpperCase();
        return ordering === 'UNORDERED' || ordering === 'ORDER_UNORDERED';
      })
      .map((ch) => ({
        portId: ch.port_id,
        channelId: ch.channel_id,
      }));
  } catch (error) {
    console.error('[getUnorderedIBCChannels]', error);
    throw error;
  }
};


// step:2 file: create_an_ibc_transfer_on_an_unordered_channel_with_a_unique_timeout_timestamp
export const generateTimeoutTimestamp = (secondsAhead = 600) => {
  // Current Unix time in seconds
  const currentSeconds = Math.floor(Date.now() / 1000);
  const futureSeconds = currentSeconds + secondsAhead;
  // Convert to nanoseconds
  return (BigInt(futureSeconds) * 1000000000n).toString();
};


// step:4 file: create_an_ibc_transfer_on_an_unordered_channel_with_a_unique_timeout_timestamp
export const sendIbcTransfer = async (transferPayload) => {
  try {
    const response = await fetch('/api/ibc_transfer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(transferPayload),
    });
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`IBC transfer failed: ${errText}`);
    }
    const data = await response.json();
    return data.tx_hash;
  } catch (error) {
    console.error('[sendIbcTransfer]', error);
    throw error;
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


// step:1 file: get_an_account_merkle_proof_at_a_specific_block_(eth_getproof)
// utils/validateEthAddress.js
// Validates that the supplied string is a proper 0x-prefixed, 40-hex-character (20-byte) EVM address.
export const validateEthAddress = (address) => {
  // Basic format check: 0x followed by exactly 40 hexadecimal characters
  const regex = /^0x[a-fA-F0-9]{40}$/;
  if (!regex.test(address)) {
    throw new Error(
      'Invalid Ethereum address: expected 0x-prefixed, 40-character hexadecimal string.'
    );
  }
  return true; // Return true if the address is valid
};


// step:2 file: get_an_account_merkle_proof_at_a_specific_block_(eth_getproof)
// utils/validateBlockHeight.js
// Ensures the block height is a non-zero, positive hexadecimal string (e.g. '0x1').
export const validateBlockHeight = (blockHeight) => {
  if (typeof blockHeight !== 'string' || !/^0x[0-9a-fA-F]+$/.test(blockHeight)) {
    throw new Error('Block height must be a hex string like 0x1.');
  }

  const numericHeight = parseInt(blockHeight, 16);
  if (Number.isNaN(numericHeight) || numericHeight < 1) {
    throw new Error('Cosmos EVM rejects block heights < 1.');
  }
  return true; // Valid block height
};


// step:3 file: get_an_account_merkle_proof_at_a_specific_block_(eth_getproof)
// utils/constructJsonRpcPayload.js
// Builds the JSON-RPC request body for eth_getProof.
export const constructJsonRpcPayload = (
  account,
  blockHeight = '0x1', // Default block height
  storageKeys = []     // Optional array of storage keys; defaults to an empty list
) => {
  // Input validation (reuse Step 1 & Step 2 helpers)
  validateEthAddress(account);
  validateBlockHeight(blockHeight);

  return {
    jsonrpc: '2.0',
    method: 'eth_getProof',
    params: [account, storageKeys, blockHeight],
    id: 1
  };
};


// step:4 file: get_an_account_merkle_proof_at_a_specific_block_(eth_getproof)
// utils/httpPostRpc.js
// Executes a POST request against the JSON-RPC endpoint.
export const httpPostRpc = async (
  payload,
  endpoint = 'http://localhost:8545',
  timeoutMs = 10_000 // 10 second timeout
) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    if (!res.ok) {
      throw new Error(`HTTP error ${res.status}: ${res.statusText}`);
    }

    const json = await res.json();
    return json; // Raw JSON-RPC response
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error('RPC request timed out.');
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
};


// step:5 file: get_an_account_merkle_proof_at_a_specific_block_(eth_getproof)
// utils/parseRpcResponse.js
// Parses the JSON-RPC response from eth_getProof and extracts useful fields.
export const parseRpcResponse = (rpcResponse) => {
  if (!rpcResponse) {
    throw new Error('Empty RPC response.');
  }

  if (rpcResponse.error) {
    // Forward the RPC error message to the caller
    throw new Error(`RPC Error: ${rpcResponse.error.message}`);
  }

  const { result } = rpcResponse;
  if (!result) {
    throw new Error('RPC result field is missing.');
  }

  const { balance, nonce, storageHash, codeHash, proof } = result;
  return {
    balance,
    nonce,
    storageHash,
    codeHash,
    proof
  };
};


// step:1 file: withdraw_10_%_of_the_user’s_shares_from_the_maxbtc_solvbtc_supervault
export const getUserWalletAddress = async () => {
  const chainId = 'neutron-1';

  // 1. Make sure Keplr exists in the browser
  if (typeof window === 'undefined' || !window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  // 2. Ask the user to approve access to Neutron in Keplr
  await window.keplr.enable(chainId);

  // 3. Obtain the signer & accounts
  const offlineSigner = window.getOfflineSigner(chainId);
  const accounts = await offlineSigner.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in Keplr.');
  }

  // 4. Return both address and signer for downstream steps
  return {
    address: accounts[0].address,
    offlineSigner,
  };
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


// step:1 file: view_available_supervault_positions_eligible_for_bitcoin_summer
export const getUserAddress = async () => {
  const chainId = 'neutron-1';

  // Ensure a wallet is injected
  if (!window.keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  // Ask the wallet to enable the Neutron chain (prompts user if needed)
  await window.keplr.enable(chainId);

  // Retrieve an OfflineSigner for read-only or signing purposes
  const signer = window.getOfflineSigner(chainId);
  const accounts = await signer.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in the connected wallet.');
  }

  // Return the Bech32 Neutron address
  return accounts[0].address;
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
import React from 'react';

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


// step:2 file: launch_a_simd_node_with_an_unlimited_mempool_(max-txs_=_-1)
// frontend/utils/getNodeHome.js

export async function getNodeHome() {
  // Try to read a cached home directory from browser storage
  let homeDir = localStorage.getItem('simd_home');
  if (!homeDir) {
    // Prompt the user for the directory; default to ~/.simapp
    homeDir = prompt('Enter the directory for your simd node (default ~/.simapp):', '~/.simapp') || '~/.simapp';
    localStorage.setItem('simd_home', homeDir);
  }

  if (typeof homeDir !== 'string' || homeDir.trim() === '') {
    throw new Error('Invalid home directory supplied.');
  }

  return homeDir.trim();
}


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


// step:1 file: find_the_block_number_closest_to_a_given_timestamp_(cast_find-block)
/* -------------------------------------------------------------
 * promptUserForBlockSearch()
 * Prompts for timestamp & RPC URL, validates the inputs and
 * returns them in a normalised shape that the backend expects.
 * No external libraries are required, fully browser-native.
 * -----------------------------------------------------------*/
export const promptUserForBlockSearch = async () => {
  // Ask for a timestamp (UNIX seconds or RFC-3339 date-time string)
  const tsRaw = window.prompt(
    'Enter a UNIX timestamp (seconds) or an RFC-3339 date/time string:',
    ''
  );
  if (!tsRaw) {
    throw new Error('A timestamp value is required.');
  }

  // Ask for the RPC endpoint URL (e.g. https://rpc-evmos.example)
  const rpcUrl = window.prompt('Enter the RPC endpoint URL for the chain:', '');
  if (!rpcUrl) {
    throw new Error('An RPC URL is required.');
  }

  // Helper: convert a string that might be RFC-3339 into a UNIX timestamp
  const normaliseTimestamp = (input) => {
    // If it is only digits, treat as UNIX seconds already
    if (/^\d+$/.test(input.trim())) {
      return parseInt(input.trim(), 10);
    }
    // Otherwise attempt RFC-3339/ISO-8601 parse using Date()
    const d = new Date(input);
    if (isNaN(d.getTime())) {
      throw new Error('Invalid RFC-3339/ISO-8601 date-time string provided.');
    }
    return Math.floor(d.getTime() / 1000); // seconds
  };

  const timestamp = normaliseTimestamp(tsRaw);
  return { timestamp, rpcUrl: rpcUrl.trim() };
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


// step:1 file: Remove the cron schedule named "daily_rewards"
export const constructMsgRemoveSchedule = (authority, name = "daily_rewards") => {
  if (!authority) {
    throw new Error("Authority (DAO address) is required");
  }

  // EncodeObject compatible with CosmJS
  return {
    typeUrl: "/neutron.cron.MsgRemoveSchedule",
    value: {
      authority,
      name,
    },
  };
};


// step:2 file: Remove the cron schedule named "daily_rewards"
import { EncodeObject, Registry } from "@cosmjs/proto-signing";
import { toUtf8 } from "@cosmjs/encoding";

/**
 * Creates a MsgExecuteContract that submits a proposal to the DAO proposal module.
 * @param {Registry} registry               – Proto registry holding Neutron types.
 * @param {EncodeObject[]} msgs            – Array of on-chain messages to execute if the proposal passes.
 * @param {string} proposalModuleAddress   – Address of the DAO's proposal-single module.
 * @param {string} title                   – Proposal title.
 * @param {string} description             – Proposal description.
 * @returns {EncodeObject}                 – MsgExecuteContract ready to be signed & broadcast.
 */
export const wrapInDaoProposal = (
  registry,
  msgs,
  proposalModuleAddress,
  title = "Remove daily cron schedule",
  description = "This proposal removes the 'daily_rewards' cron schedule."
) => {
  if (!proposalModuleAddress) throw new Error("proposalModuleAddress is required");
  if (!msgs.length) throw new Error("At least one message must be supplied");

  // Convert each EncodeObject -> stargate format expected by cw-dao
  const encodedMsgs = msgs.map((msg) => {
    const binary = registry.encode(msg);
    return {
      stargate: {
        type_url: msg.typeUrl,
        value: Buffer.from(binary).toString("base64"),
      },
    };
  });

  // cw-proposal-single JSON payload
  const proposeMsg = {
    propose: {
      title,
      description,
      msgs: encodedMsgs,
    },
  };

  // Wrap into MsgExecuteContract (CosmWasm)
  return {
    typeUrl: "/cosmwasm.wasm.v1.MsgExecuteContract",
    value: {
      sender: "",                // filled automatically just before signing
      contract: proposalModuleAddress,
      msg: toUtf8(JSON.stringify(proposeMsg)),
      funds: [],
    },
  };
};


// step:3 file: Remove the cron schedule named "daily_rewards"
import { SigningStargateClient } from "@cosmjs/stargate";

/**
 * Signs & broadcasts a DAO proposal using Keplr.
 */
export const signAndBroadcastTx = async (
  executeMsg,
  rpcEndpoint = "https://rpc-kralum.neutron.org",
  chainId = "neutron-1",
  fee = "auto"
) => {
  const { keplr } = window;
  if (!keplr) throw new Error("Keplr wallet is not installed");

  // Enable chain & fetch signer
  await keplr.enable(chainId);
  const offlineSigner = keplr.getOfflineSigner(chainId);
  const [account] = await offlineSigner.getAccounts();

  // Patch sender field now that we know the address
  executeMsg.value.sender = account.address;

  // Broadcast
  const client = await SigningStargateClient.connectWithSigner(rpcEndpoint, offlineSigner);
  const res = await client.signAndBroadcast(account.address, [executeMsg], fee);
  if (res.code !== 0) {
    throw new Error(`Broadcast failed: ${res.rawLog}`);
  }
  return res.transactionHash;
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


// step:1 file: publish_a_raw_signed_transaction_with_cast_publish
export const promptRawTx = () => {
  // Ask the user to paste a raw, signed transaction.
  const rawTx = prompt('Paste the 0x-prefixed RLP-encoded signed transaction:');

  // Basic validation.
  if (!rawTx || !rawTx.trim().startsWith('0x')) {
    throw new Error('Invalid raw transaction hex string.');
  }

  return rawTx.trim();
};


// step:3 file: publish_a_raw_signed_transaction_with_cast_publish
export const broadcastRawTx = async (rawTx) => {
  const response = await fetch('/api/broadcast_raw_tx', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ raw_tx: rawTx })
  });

  if (!response.ok) {
    // Attempt to surface any backend-provided error message.
    let detail = 'Failed to broadcast transaction';
    try {
      const err = await response.json();
      detail = err.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }

  const { tx_hash } = await response.json();
  return tx_hash; // <- Store or display as needed.
};


// step:5 file: publish_a_raw_signed_transaction_with_cast_publish
export const waitForTxReceipt = async (txHash, interval = 5000) => {
  /*
   * Poll `/api/tx_receipt/:txHash` every `interval` ms.  When the node
   * returns a non-null receipt we resolve with that receipt.
   */
  while (true) {
    const resp = await fetch(`/api/tx_receipt/${txHash}`);

    if (!resp.ok) {
      let detail = 'Unable to fetch transaction receipt';
      try {
        const err = await resp.json();
        detail = err.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    const { receipt } = await resp.json();

    if (receipt) {
      return receipt; // 🎉  The transaction is now finalized on-chain.
    }

    // Wait before the next attempt.
    await new Promise((resolve) => setTimeout(resolve, interval));
  }
};


// step:1 file: retrieve_a_block_by_its_hash_without_transaction_bodies
/* getBlockByHash.js
 * Utility to query an Ethereum JSON-RPC endpoint for a block by hash.
 * This can safely run in a browser environment that supports `fetch`.
 */
export const getBlockByHash = async ({ rpcEndpoint, blockHash }) => {
  // Basic argument validation
  if (!rpcEndpoint || !blockHash) {
    throw new Error("Both 'rpcEndpoint' and 'blockHash' parameters are required.");
  }

  // Construct JSON-RPC payload
  const payload = {
    jsonrpc: "2.0",
    id: 1,
    method: "eth_getBlockByHash",
    params: [blockHash, false] // `false` => return block data without full transaction objects
  };

  try {
    // Issue POST request to the RPC endpoint
    const response = await fetch(rpcEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    // Throw if HTTP status is not OK (200–299)
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`HTTP ${response.status}: ${text}`);
    }

    // Parse the JSON-RPC response
    const data = await response.json();

    // Handle possible JSON-RPC errors
    if (data.error) {
      throw new Error(`RPC Error (${data.error.code}): ${data.error.message}`);
    }

    return data.result; // Block object
  } catch (err) {
    console.error("Failed to fetch block by hash:", err);
    throw err; // Propagate error to caller
  }
};


// step:1 file: make_a_raw_json-rpc_call_with_cast_rpc
// rpcClient.js
// Utility that asks the user for a JSON-RPC method and parameters,
// then POSTs the payload to our backend.
export async function requestRpcCall() {
  try {
    // 1. Prompt the user. Replace these prompts with a proper UI form in production.
    const method = prompt("Enter JSON-RPC method (e.g. eth_getBalance):");
    if (!method) throw new Error("Method name is required");

    const rawParams = prompt("Enter params as a JSON array (e.g. [\"0x…\",\"latest\"]):", "[]");
    const params = JSON.parse(rawParams || "[]");

    // 2. Send the request to the backend.
    const response = await fetch("/api/rpc", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method, params })
    });

    if (!response.ok) {
      // Non-2xx HTTP status
      const errorText = await response.text();
      throw new Error(`Backend error: ${errorText}`);
    }

    const data = await response.json();
    displayRpcResult(data);
  } catch (err) {
    console.error(err);
    alert(`Error: ${err.message}`);
  }
}

// Helper to render the backend result.
function displayRpcResult(result) {
  // In a real app, you would update the DOM. For docs, we just alert.
  alert(JSON.stringify(result, null, 2));
}



// step:3 file: make_a_raw_json-rpc_call_with_cast_rpc
// index.js
import { requestRpcCall } from "./rpcClient.js";

// Example: tie the function to a button click in your docs site.
document.getElementById("rpcButton").addEventListener("click", requestRpcCall);



// step:1 file: decode_protobuf_transaction_bytes_to_json
export const validateBase64Input = (inputStr) => {
  // Regular-expression check for valid base64 padding/charset
  const base64Regex = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
  if (!base64Regex.test(inputStr)) {
    throw new Error("Input is not valid base64.");
  }
  // Double-check by actually decoding; browsers throw on bad padding
  try {
    atob(inputStr);
  } catch (err) {
    throw new Error("Failed to decode base64 string: " + err.message);
  }
  return true; // Valid base64
};


// step:3 file: decode_protobuf_transaction_bytes_to_json
export const prettyPrintJSON = (jsonObj, downloadFileName = null) => {
  const pretty = JSON.stringify(jsonObj, null, 2); // 2-space indent for readability

  // Optionally offer a JSON file download to the user
  if (downloadFileName) {
    const blob = new Blob([pretty], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = downloadFileName;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return pretty; // Useful for placing in a <pre> tag or console.log
};


// step:1 file: provide_liquidity_to_the_maxbtc_unibtc_supervault_using_1_maxbtc_and_1_unibtc
export const connectWalletAndGetAddress = async () => {
  const chainId = 'neutron-1';
  const keplr = window.keplr;

  if (!keplr) {
    throw new Error('Keplr wallet is not installed.');
  }

  // Request wallet connection if it has not yet been approved
  await keplr.enable(chainId);

  // Get the offline-signer injected by Keplr
  const signer = window.getOfflineSigner(chainId);
  const accounts = await signer.getAccounts();

  if (!accounts || accounts.length === 0) {
    throw new Error('No account found in the wallet.');
  }

  return { signer, address: accounts[0].address };
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


// step:1 file: generate_an_unsigned_transaction_that_sends_1000stake_from_address_a_to_address_b
/* validateAddressFormat.js
 * Simple Bech32-style validation without external libraries.
 * NOTE: This is a lightweight check (length, prefix, and character set) and
 *       does NOT perform a full Bech32 checksum. For production-grade apps,
 *       do server-side or WASM-based checksum verification.
 */
export const validateAddressFormat = (address, expectedPrefix) => {
  // 1. Quick sanity checks
  if (typeof address !== 'string' || !address.includes('1')) {
    throw new Error('Address must be a Bech32 string that contains the separator “1”.');
  }

  // 2. Split prefix and data part
  const [prefix, data] = address.split('1');
  if (prefix !== expectedPrefix) {
    throw new Error(`Invalid prefix: expected “${expectedPrefix}”, received “${prefix}”.`);
  }

  // 3. Validate character set (qpzry9x8gf2tvdw0s3jn54khce6mua7l)
  const charset = /^[qpzry9x8gf2tvdw0s3jn54khce6mua7l]+$/;
  if (!charset.test(data)) {
    throw new Error('Invalid Bech32 character set in address.');
  }

  // 4. Minimum length check (Cosmos addresses are usually 39–45 chars)
  if (address.length < 39 || address.length > 90) {
    throw new Error('Address length outside expected Bech32 bounds.');
  }

  // Passed all checks
  return true;
};


// step:1 file: query_total_delegations_to_my_validator_validator
/*
 * fetchValidatorDelegations
 * -------------------------
 * @param {string}  validatorAddress  Bech32 validator operator address (e.g. "cosmosvaloper1...")
 * @param {string}  restEndpoint      Fully-qualified REST/LCD endpoint of the chain (defaults to Cosmos Hub main-net)
 * @returns {Promise<Array>}          Promise resolving to an array of delegation objects
 */
export const fetchValidatorDelegations = async (
  validatorAddress,
  restEndpoint = 'https://lcd.cosmos.directory/cosmoshub'
) => {
  if (!validatorAddress) {
    throw new Error('validatorAddress is required');
  }

  // Strip trailing slash for safe concatenation
  const baseUrl = restEndpoint.replace(/\/$/, '');

  const allDelegations = [];
  let nextKey = null;

  try {
    do {
      // Build URL with pagination key when present
      const params = new URLSearchParams();
      if (nextKey) params.append('pagination.key', nextKey);

      const url = `${baseUrl}/cosmos/staking/v1beta1/validators/${validatorAddress}/delegations?${params}`;
      const res = await fetch(url);

      if (!res.ok) {
        throw new Error(`Failed to fetch delegations: ${res.status} ${res.statusText}`);
      }

      const json = await res.json();
      const pageDelegations = json.delegation_responses || [];
      allDelegations.push(...pageDelegations);

      // Continue if the API returned a non-null pagination key
      nextKey = json.pagination && json.pagination.next_key;
    } while (nextKey);

    return allDelegations;
  } catch (err) {
    console.error('fetchValidatorDelegations error:', err);
    throw err; // Propagate so callers can handle it
  }
};


// step:2 file: query_total_delegations_to_my_validator_validator
/*
 * calculateTotalStake
 * -------------------
 * @param {Array}  delegations  Array returned by fetchValidatorDelegations()
 * @param {string} denom        Base denom to filter on (defaults to "uatom")
 * @param {number} microFactor  Conversion factor from micro-denom to main denom (1_000_000 for most chains)
 * @returns {{ denom: string, micro: string, human: string }}
 */
export const calculateTotalStake = (
  delegations,
  denom = 'uatom',
  microFactor = 1_000_000
) => {
  if (!Array.isArray(delegations)) {
    throw new TypeError('delegations must be an array');
  }

  // Aggregate the micro amounts
  const totalMicro = delegations.reduce((acc, d) => {
    const bal = d.balance;
    if (bal && bal.denom === denom) {
      return acc + Number(bal.amount || 0);
    }
    return acc;
  }, 0);

  // Convert to main denom (e.g. ATOM)
  const totalHuman = totalMicro / microFactor;

  return {
    denom,
    micro: totalMicro.toString(),
    human: totalHuman.toLocaleString(undefined, { maximumFractionDigits: 6 })
  };
};


// step:1 file: retrieve_intermediate_state_roots_for_a_transaction
export const selectRpcEndpoint = () => {
  // Prefer an endpoint supplied at runtime (e.g., window.RPC_ENDPOINT)
  // and fall back to a hard-coded default.
  const fallback = 'https://rpc.example.com';
  const endpoint = typeof window !== 'undefined' && window.RPC_ENDPOINT ? window.RPC_ENDPOINT : fallback;

  // Very basic sanity-check so we fail early if the URL is missing.
  if (!endpoint || typeof endpoint !== 'string') {
    throw new Error('RPC endpoint is not defined or is invalid.');
  }

  return endpoint;
};


// step:2 file: retrieve_intermediate_state_roots_for_a_transaction
export const debugIntermediateRoots = async (txHash, rpcEndpoint) => {
  if (!txHash || typeof txHash !== 'string') {
    throw new Error('`txHash` must be a non-empty string.');
  }

  // Build the JSON-RPC payload exactly as required by the node.
  const payload = {
    method: 'debug_intermediateRoots',
    params: [txHash],
    id: 1,
    jsonrpc: '2.0',
  };

  try {
    const response = await fetch(rpcEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`RPC call failed with HTTP status ${response.status}`);
    }

    const json = await response.json();

    if (json.error) {
      // Preserve the original error message from the node for easier debugging.
      throw new Error(`RPC error: ${json.error.message || JSON.stringify(json.error)}`);
    }

    return json.result;
  } catch (err) {
    console.error('[debugIntermediateRoots] Unexpected error:', err);
    throw err;
  }
};


// step:1 file: save_a_remix_ide_workspace_to_a_github_repository
export const activateRemixGithubPlugin = async () => {
  // Ensure we are running inside the Remix IDE
  if (!window.remix || typeof window.remix.call !== 'function') {
    throw new Error('Remix plugin API not detected. Are you running this code inside the Remix IDE?');
  }

  try {
    // Ask the Remix Plugin Manager to enable the GitHub plugin
    await window.remix.call('pluginManager', 'activatePlugin', 'github');
    return true;
  } catch (error) {
    console.error('Failed to activate the GitHub plugin:', error);
    throw error;
  }
};


// step:3 file: list_all_wallets_managed_by_the_node
export const getWalletList = async () => {\n  try {\n    const res = await fetch('/api/personal_list_wallets');\n    if (!res.ok) {\n      throw new Error(`Backend responded with ${res.status}`);\n    }\n    const data = await res.json();\n    return data.wallets;\n  } catch (err) {\n    console.error('Failed to fetch wallet list:', err);\n    throw err;\n  }\n};


