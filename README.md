# LedgerPoison — 

The package contains the actual supplied Ethereum graph inputs and daily gas/ETH-USD files, regression tests, reproducible configurations, per-address predictions, model checkpoints, and generated diagnostic tables. It runs a finite funded transaction simulator over an aggregate reconstruction. It does not certify an on-chain attack, TDSC suitability, statistical significance, or camera-ready research results.

## Install on a Linux server

Python 3.12 and a CPU PyTorch build were tested. Start inside the extracted `ledgerpoison` directory. Use a virtual environment. Install CPU PyTorch first and then the other tested versions:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt
```

`requirements-tested.txt` records this execution environment; `requirements.txt` provides compatible minimum versions for other environments. Different versions/devices yield a different run fingerprint. CPU runs were tested; CUDA reproducibility and server hardware were not tested here. The full ex_2 graph with default 128-wide full-batch attention needs substantially more memory than the bounded integration configuration. Check available memory and use `--dry-run` to inspect the grid; dry-run loads data but does not estimate model memory.

## Run the complete engineering integration pipeline

```bash
bash run_server.sh integration
```

This runs regression tests with the bundled real-data adapter enabled, a 13-cell attack/variant/no-attack check and a 108-cell detector/defense/labeler matrix, verifies saved predictions/metrics/plans/checksums, and regenerates all analysis tables. It uses ex_1 aggregate edges whose **last timestamps** fall within 2021-01-01 through 2021-03-31, 3 epochs, 16 hidden units and one seed. Its fixed holdout comprises 96 accounts (40 phishing). These are diagnostic outputs. Newly written results are under `results/server_*`; original verified evidence is under `evidence/`.

A $200 diagnostic budget produces complete nonempty plans. Lower budgets may legitimately produce zero transactions because truncation keeps complete funded buffer blocks. Empty plans are recorded and must not be interpreted as successful attacks. `verify_results` is intentionally an engineering-integration verifier that requires nonempty attack plans; it is not suitable for a low-budget full campaign containing legitimate empty plans.

## Launch the research campaign on the complete supplied Ethereum inputs

```bash
bash run_server.sh gate
```

This uses ex_2 without the diagnostic window, five seeds, GAT, four budgets and matched camouflage/random arms (40 cells). It regenerates six-condition tables and checks a conservative prespecified Gate B: paired Student-t intervals, Bonferroni adjustment over scanned budgets, and a lower effect bound of at least 0.02 F1. A failed gate exits with status 2 and writes its decision. The interval is a working small-sample model over seeds, not a distribution-free assurance. No such full-graph campaign has been completed in this package.

After inspecting Gate B and the dataset limitations, the optional full grid is:

```bash
bash run_server.sh full
```

This command runs the configured full grid; it does not claim that a gate has already passed. The full grid contains 9,720 cells including matched random arms. Plan server resources before launching it. Sweep label-lag assumptions and aggregate conflict policies before making a robust scientific claim; those sensitivity campaigns are not automatically included. `full` is explicitly requested by the operator, and never launches as a side effect of integration or gate.

For external data paths, call `python -m experiments.run_grid configs/gate_b.yaml --data-root /path/to/data --out /path/to/results --threads 2`. That directory must contain `eth-phishing-detection-main/datasets/...` and both price CSVs, as the bundled `data/` directory does. Missing data raises an error; it never substitutes synthetic data. `--synthetic` is an explicit engineering-only option.

## What changed

- Only nine recoverable count, total-value and counterparty features remain. Within-pair amount dispersion, exact interarrival patterns and true ages cannot be recovered from aggregates.
- Clean retraining at the evaluation snapshot is an explicit control. Six scored conditions distinguish time/retraining changes from evasion and poisoning.
- Each random arm shares its treatment's owned endpoints and targets, funding edges, timestamps, amounts, gas, transfer count and cost; only non-owned recipient identities are randomized. This null isolates counterparty selection, not every possible strategic effect of timing or topology.
- Senders must be explicitly registered as owned. Bankrolls and coalition balances are finite declared assumptions; spending and fees are replayed chronologically. An `atk_` prefix grants no permission.
- Held-out validation truth is withheld from label propagation and pseudo-label training. Test identities are fixed from the clean evaluation graph. Detector inputs never contain the attacker provenance mask.
- Random streams are derived independently of execution order. Cached clean contexts include defense/configuration identity. Resumption checks source/data/configuration/environment fingerprints and checkpoint hashes.
- Analysis preserves parameter/labeler/control-family identities, rejects duplicate seed arms, and uses paired effects and Student-t confidence intervals. One-seed diagnostic confidence intervals are unavailable, not zero.
- Economic tables count exact per-address false-negative flips against clean retraining and report hypothetical benefits on non-owned holdouts. Gas sensitivity rescales fees only, not lost transfer principal.
- Accurate public names are `age_gate`, `variance_gcn`, and `aggregate_noise`. They are custom baselines; aggregate clipping/noise is not per-example DP-SGD or a faithful cited method implementation. `time_sage` is an optional age/recency model, not automatically label leakage or proof of leakage.

## Result integrity and interpretation

Every cell stores six prediction arrays, validation thresholds, test addresses/labels, complete injected transaction records, costs, initial/ending simulated balances, simulated sender sequence counters, three checkpoints and checkpoint hashes. Source/data/configuration/environment hashes prevent reuse across changed protocols. Interrupted cells leave no committed result JSON; checkpoints are trusted only with a matching committed result/hash. Failures are recorded and the process exits unsuccessfully.

Daily average gas prices approximate execution cost; actual EIP-1559 fees, contract execution, signatures, historical balances and historical nonces are not replayed. `origin` is a simulation annotation only. This version supports Ethereum account-plan validation; Bitcoin/Elliptic adapters are retained as import utilities and are excluded from this attack workflow. Elliptic++ Git LFS pointers are not usable data. These inputs must not be merged into the Ethereum attack graph.

Read `DATA_CARD.md`, `CHANGELOG_REVISION.md`, and `evidence/INTEGRATION_REPORT.md` before using numbers in a manuscript. SHA256SUMS covers the delivered package contents.
