"""
Deterministic per-address features f: address -> R^d computed from the
ledger alone. Applied identically to every address, real or injected, so an
attacker's feature vector is a *consequence* of the transactions it submits.

The feature set mirrors what published Ethereum/Bitcoin detectors derive
from raw transactions (counts, value statistics, counterparty diversity,
activity span, burstiness, gas statistics). It is deliberately simple and
documented so that reviewers can audit each dimension.

All features are computed relative to a reference time `t_ref` (the block
height at which the detector is built) using only transactions with
timestamp <= t_ref, which prevents future leakage.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .ledger import Ledger

# B1: absolute-recency features (`age`, `active_span`, `frac_txs_last_day/week`)
# were removed because they encode "new account", which is exactly the test-set
# membership signal (test = accounts first seen in the evaluation window). Keeping
# them let a detector read the test label off an input. The remaining features are
# counts, value statistics, counterparty diversity, inter-event *shape*
# (mean_interval / cv_interval are relative, not absolute recency) and gas — all
# behavioural and comparable across account ages.
FEATURE_NAMES: List[str] = [
    "n_out", "n_in", "n_total", "in_out_ratio", "out_value_sum", "in_value_sum",
    "net_value", "n_unique_counterparties", "counterparty_entropy",
]
FEATURE_DIM = len(FEATURE_NAMES)

_DAY = 86_400.0
_WEEK = 7 * _DAY


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), 0.0)


def _group_stats(values: np.ndarray, groups: np.ndarray, n: int) -> Tuple[np.ndarray, ...]:
    """sum, mean, std, max of `values` grouped by `groups` in [0, n)."""
    cnt = np.bincount(groups, minlength=n).astype(np.float64)
    s = np.bincount(groups, weights=values, minlength=n)
    s2 = np.bincount(groups, weights=values * values, minlength=n)
    mean = _safe_div(s, cnt)
    var = np.maximum(_safe_div(s2, cnt) - mean ** 2, 0.0)
    mx = np.full(n, 0.0)
    if len(values):
        np.maximum.at(mx, groups, values)
    return cnt, s, mean, np.sqrt(var), mx


def compute_features(ledger: Ledger, t_ref: float, age_cap: float = None) -> np.ndarray:
    """Return an [num_addresses, FEATURE_DIM] float32 matrix for the ledger,
    using only transactions at or before t_ref. Addresses with no activity
    at or before t_ref get an all-zero row.

    age_cap (seconds): if set, the `age` feature is clipped to [0, age_cap].
    This prevents `age` from encoding "born after t_train", which would let a
    detector read test-set membership off an input (see harness test-set
    definition). The harness passes age_cap = t_eval - t_train."""
    n = ledger.num_addresses
    a = ledger.arrays()
    keep = a["timestamp"] <= t_ref
    src, dst = a["src"][keep], a["dst"][keep]
    val, ts, gas = a["value"][keep], a["timestamp"][keep], a["gas_used"][keep].astype(np.float64)

    X = np.zeros((n, FEATURE_DIM), dtype=np.float64)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}

    n_out, out_sum, out_mean, out_std, out_max = _group_stats(val, src, n)
    n_in, in_sum, in_mean, in_std, in_max = _group_stats(val, dst, n)
    X[:, col["n_out"]] = n_out
    X[:, col["n_in"]] = n_in
    X[:, col["n_total"]] = n_out + n_in
    X[:, col["in_out_ratio"]] = _safe_div(n_in, n_in + n_out)
    X[:, col["out_value_sum"]] = out_sum
    X[:, col["in_value_sum"]] = in_sum
    X[:, col["net_value"]] = in_sum - out_sum

    # counterparty diversity: for each address, distribution over counterparties
    nodes = np.concatenate([src, dst])
    parties = np.concatenate([dst, src])
    if len(nodes):
        pair = nodes.astype(np.int64) * n + parties.astype(np.int64)
        uniq, counts = np.unique(pair, return_counts=True)
        owner = uniq // n
        n_unique = np.bincount(owner, minlength=n).astype(np.float64)
        total = np.bincount(owner, weights=counts, minlength=n)
        p = counts / np.where(total[owner] > 0, total[owner], 1.0)
        ent = np.bincount(owner, weights=-p * np.log(p), minlength=n)
        X[:, col["n_unique_counterparties"]] = n_unique
        X[:, col["counterparty_entropy"]] = ent

    return X.astype(np.float32)


def standardize(X: np.ndarray, fit_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Log1p on heavy-tailed columns then z-score with statistics fitted on
    `fit_mask` rows only (train nodes), returned so the same transform can be
    applied to later snapshots."""
    if not fit_mask.any() or not np.isfinite(X).all():
        raise ValueError("finite features and nonempty fit mask required")
    Xl = np.log1p(np.abs(X)) * np.sign(X)
    mu = Xl[fit_mask].mean(0)
    sd = Xl[fit_mask].std(0) + 1e-6
    return ((Xl - mu) / sd).astype(np.float32), {"mu": mu, "sd": sd}


def apply_standardize(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    Xl = np.log1p(np.abs(X)) * np.sign(X)
    return ((Xl - stats["mu"]) / stats["sd"]).astype(np.float32)
