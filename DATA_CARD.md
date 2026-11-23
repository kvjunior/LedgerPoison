# Supplied data and permitted inference

## Inputs bundled for this workflow

Four real supplied ego-graph pickles: phishing/normal `ex_1` and `ex_2`, plus `datasets/open/ex1/phishing_address.csv`. These came from the supplied `eth-phishing-detection-main.zip`, not from newly scraped records. Both supplied Etherscan exports are present: `export-AvgGasPrice.csv` (wei converted to gwei by 1e-9) and `export-EtherPrice.csv` (USD/ETH). No ETH/USD constant is used in the supplied configurations. SHA256SUMS identifies the exact bytes. Only the data needed by this Ethereum workflow is bundled; unrelated paper PDFs, Bitcoin exports and unusable LFS pointers are excluded.

## Representation

An original directed edge is `(from,to,total_value_ETH,last_timestamp,count)`. `expand: last` creates `count` equal-value records at the last timestamp. This preserves the selected aggregate's count, total value and pair identity. It cannot recover original amounts, per-transaction timestamps, true first activity, exact fees, per-address balance history, contracts or historical label publication times. Selection of an earlier time window applies to aggregate last timestamps and is not equivalent to filtering raw transactions. Conflicting ego views use the original maximum-count conflict heuristic; it is logged and is not evidence of ground truth.

The nine input features are outgoing/incoming/total counts, in/out count ratio, outgoing/incoming/net value totals, distinct counterparties, and counterparty entropy based on reconstructed counts. They are valid statistics of the reconstructed benchmark. Even those counts/totals cannot be claimed to equal historical pre-cutoff quantities for collapsed aggregates whose original transfers span the cutoff.

Center labels are normalized to `1=phishing`, `0=normal`; contradictory center labels and phishing centers absent from the supplied phishing list are dropped in strict mode. Non-center accounts are not presumed benign. These are sampled labels and sampled ego neighborhoods, not a complete network or a representative prevalence sample.

## Declared simulation assumptions

Default missing label-arrival times are the first reconstructed observation plus 30 days. This is an assumption, not an observed forensic delay. Fresh buffers start with zero balance; each explicitly registered bankroll and each coalition account has a declared 100 ETH initial balance. These amounts are finite simulation inputs, not historical observed balances. Owned transfer amounts remain internal; transfer principal paid to non-owned recipients and all gas fees count toward budget. Opportunity cost of tied-up capital is not modeled.

The validator requires owned senders, funding before spending, chronological nonnegative balances, correct attack window, positive finite fee inputs and a total expenditure within budget. It does not verify signatures, transaction inclusion, sender EVM nonce/state, EIP-1559 fee caps, contract behavior or genuine counterparty consent. The generated nonce counts are simulator sequence counters only.

## Diagnostic scope

Executed end-to-end integration used ex_1 selected aggregate last timestamps 2021-01-01 through 2021-03-31: 41,542 reconstructed records, 12,369 observed addresses and 14,554 pairs. It uses one seed, three training epochs, nine input features and a 96-account holdout (40 phishing). 6,726 pairs have collapsed original transfers. First-seen is a left-truncated reconstruction proxy. This exercise validates code integration; it provides no publication-level effect estimate or novelty evidence.

## Other supplied collections

Elliptic is real Bitcoin data with anonymized features; those features cannot be regenerated from transfer plans. Supplied Elliptic++ CSVs were LFS pointers rather than downloaded content. Etherscan public-category labels do not establish benignness or complete fraud ground truth. Bitcoin mempool exports are not supported by this Ethereum account validator. They are not silently loaded, merged, or replaced with synthetic observations.
