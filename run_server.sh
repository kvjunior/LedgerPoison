#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mode="${1:-integration}"
case "$mode" in
  integration)
    GRABPHISHER_ROOT="$PWD/data/eth-phishing-detection-main" python -m pytest -q
    for config in integration integration_matrix; do
      python -m experiments.run_grid "configs/$config.yaml" --out "results/server_$config"
      python -m experiments.verify_results "results/server_$config" --out "results/server_$config/verification.json"
      python -m analysis.tables "results/server_$config" --out "results/server_$config/tables" --allow-diagnostic
    done
    ;;
  gate)
    python -m experiments.run_grid configs/gate_b.yaml --out results/server_gate
    python -m analysis.tables results/server_gate --out results/server_gate/tables
    python -m experiments.check_gate results/server_gate --out results/server_gate/gate_decision.json
    ;;
  full)
    python -m experiments.run_grid configs/full_grid.yaml --out results/server_full
    python -m analysis.tables results/server_full --out results/server_full/tables
    ;;
  *) printf '%s\n' 'Usage: bash run_server.sh {integration|gate|full}' >&2; exit 2 ;;
esac
