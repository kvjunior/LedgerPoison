"""
Data adapters. Each returns (ledger, labels, label_time, meta).

  * `synthetic_ledger`  — explicit synthetic generator for smoke tests and
    unit tests. `meta["synthetic"] = True`; the harness refuses to write
    such results into the paper tables.
  * `load_ethereum_csv` — generic raw-transaction export (XBlock-style):
    columns from,to,value,timestamp[,gas_used]; labels from a second CSV
    address,label[,label_time]. This is the intended primary path.
  * `load_elliptic` — Bitcoin Elliptic benchmark. Features are anonymised
    and cannot be regenerated from transactions, so `meta["features_derivable"]
    = False`; attacks that need feature derivation must refuse to run on it.
    Kept only as a secondary robustness check.

No adapter ever silently substitutes synthetic data for a missing file.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .ledger import Ledger, Tx

Meta = Dict[str, object]


def synthetic_ledger(n_addresses: int = 3000, n_txs: int = 20000, fraud_ratio: float = 0.05,
                     t0: float = 1.6e9, span: float = 90 * 86400.0, seed: int = 0,
                     label_lag_days: Tuple[float, float] = (10.0, 120.0)) -> Tuple[Ledger, Dict[str, int], Dict[str, float], Meta]:
    """Fraud rings: fraud addresses receive from many victims and pay out to
    a few collectors, with heavier tails and bursty timing. Honest activity is
    preferential-attachment with uniform timing. Labels arrive with a lag."""
    rng = np.random.default_rng(seed)
    addrs = [f"a{i:06d}" for i in range(n_addresses)]
    n_fraud = max(1, int(n_addresses * fraud_ratio))
    fraud = set(rng.choice(n_addresses, n_fraud, replace=False).tolist())
    w = rng.pareto(1.5, n_addresses) + 1.0
    w /= w.sum()
    txs = []
    for _ in range(n_txs):
        s = int(rng.choice(n_addresses, p=w))
        if s in fraud and rng.random() < 0.6:
            d = int(rng.choice(list(fraud)))                     # collector inside the ring
            value = float(rng.lognormal(3.0, 1.2))
            t = t0 + span * float(rng.beta(4, 1.5))              # bursty, late
        elif rng.random() < 0.08:                               # victim -> fraud inflow
            d = int(rng.choice(list(fraud)))
            value = float(rng.lognormal(2.0, 1.0))
            t = t0 + span * float(rng.random())
        else:
            d = int(rng.choice(n_addresses, p=w))
            value = float(rng.lognormal(1.0, 1.0))
            t = t0 + span * float(rng.random())
        if d == s:
            d = (d + 1) % n_addresses
        txs.append(Tx(addrs[s], addrs[d], value, t, gas_used=int(rng.choice([21000, 21000, 46000, 65000]))))
    ledger = Ledger(txs)
    # labels: all fraud addresses + a random 60% of honest addresses labelled licit
    labels: Dict[str, int] = {}
    label_time: Dict[str, float] = {}
    first_seen = ledger.first_seen()
    idx = ledger.address_index()
    lo, hi = label_lag_days
    for a in addrs:
        i = idx.get(a)
        if i is None:
            continue
        if int(a[1:]) in fraud:
            labels[a] = 1
        elif rng.random() < 0.6:
            labels[a] = 0
        else:
            continue
        label_time[a] = first_seen[i] + 86400.0 * float(rng.uniform(lo, hi))
    meta = {"synthetic": True, "features_derivable": True, "chain": "ethereum", "name": "synthetic"}
    return ledger, labels, label_time, meta


def load_ethereum_csv(tx_csv: str, label_csv: str, gas_default: int = 21_000,
                      value_scale: float = 1e-18) -> Tuple[Ledger, Dict[str, int], Dict[str, float], Meta]:
    """tx_csv columns: from,to,value,timestamp[,gas_used]; value in wei by
    default (value_scale converts to ETH). label_csv: address,label[,label_time]."""
    for p in (tx_csv, label_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found — no synthetic substitution is performed")
    df = pd.read_csv(tx_csv)
    cols = {c.lower(): c for c in df.columns}
    need = ["from", "to", "value", "timestamp"]
    for c in need:
        if c not in cols:
            raise ValueError(f"tx_csv missing column {c!r}; found {list(df.columns)}")
    for c in ('from','to'):
        if df[cols[c]].isna().any(): raise ValueError('missing transaction address')
        df[cols[c]]=df[cols[c]].astype(str).str.strip().str.lower()
        if (df[cols[c]]=='').any(): raise ValueError('empty transaction address')
    df = df[df[cols["from"]] != df[cols["to"]]].copy()
    gas = df[cols["gas_used"]].astype(int).values if "gas_used" in cols else np.full(len(df), gas_default)
    txs = (Tx(str(f), str(t), float(v) * value_scale, float(ts), int(g))
           for f, t, v, ts, g in zip(df[cols["from"]], df[cols["to"]], df[cols["value"]],
                                     df[cols["timestamp"]], gas))
    ledger = Ledger(txs)
    ldf = pd.read_csv(label_csv)
    lcols = {c.lower(): c for c in ldf.columns}
    if not {'address','label'}<=set(lcols): raise ValueError('missing address/label columns')
    if ldf[lcols['address']].isna().any(): raise ValueError('missing label address')
    ldf[lcols['address']]=ldf[lcols['address']].astype(str).str.strip().str.lower()
    if not ldf[lcols['label']].isin([0,1]).all(): raise ValueError('labels must be binary 0/1')
    if (ldf.groupby(lcols['address'])[lcols['label']].nunique()>1).any(): raise ValueError('conflicting labels')
    if 'label_time' in lcols:
        if not np.isfinite(ldf[lcols['label_time']].to_numpy(float)).all(): raise ValueError('invalid label times')
        if (ldf.groupby(lcols['address'])[lcols['label_time']].nunique()>1).any(): raise ValueError('conflicting label times')
    labels = {str(a): int(l) for a, l in zip(ldf[lcols["address"]], ldf[lcols["label"]])}
    if "label_time" in lcols:
        label_time = {str(a): float(t) for a, t in zip(ldf[lcols["address"]], ldf[lcols["label_time"]])}
    else:
        label_time = {a: 0.0 for a in labels}
    meta = {"synthetic": False, "features_derivable": True, "chain": "ethereum",
            "name": os.path.basename(tx_csv), "representation":"raw_transactions", "label_times_observed":"label_time" in lcols, "historical_state_verified":False}
    return ledger, labels, label_time, meta


def load_elliptic(root: str, step_seconds: float = 14 * 86400.0):
    """Returns a ready GraphSnapshot-compatible dict rather than a Ledger:
    Elliptic has no transaction values or gas and its 166 features are
    anonymised. Timestep k is mapped to time k*step_seconds."""
    from .graph import GraphSnapshot  # local import to avoid cycle
    feat = os.path.join(root, "elliptic_txs_features.csv")
    edges = os.path.join(root, "elliptic_txs_edgelist.csv")
    cls = os.path.join(root, "elliptic_txs_classes.csv")
    for p in (feat, edges, cls):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found")
    fdf = pd.read_csv(feat, header=None)
    ids = fdf.iloc[:, 0].values
    id2i = {int(v): i for i, v in enumerate(ids)}
    x = fdf.iloc[:, 2:].values.astype(np.float32)
    step = fdf.iloc[:, 1].values.astype(np.float64)
    edf = pd.read_csv(edges)
    e = edf.values
    keep = np.array([(a in id2i and b in id2i) for a, b in e])
    src = np.array([id2i[a] for a in e[keep, 0]], dtype=np.int64)
    dst = np.array([id2i[b] for b in e[keep, 1]], dtype=np.int64)
    cdf = pd.read_csv(cls)
    y = np.full(len(ids), -1, dtype=np.int64)
    for tid, c in zip(cdf.iloc[:, 0], cdf.iloc[:, 1].astype(str)):
        if int(tid) in id2i and c in ("1", "2"):
            y[id2i[int(tid)]] = 1 if c == "1" else 0
    t = step * step_seconds
    snap = GraphSnapshot(t_ref=float(t.max()), x=x, edge_index=np.stack([src, dst]),
                         edge_time=t[src], edge_value=np.zeros(len(src)), first_seen=t,
                         address_index={str(k): v for k, v in id2i.items()}, y=y,
                         label_time=np.zeros(len(ids)), attacker_mask=np.zeros(len(ids), bool),
                         attacker_edge=np.zeros(len(src), bool))
    meta = {"synthetic": False, "features_derivable": False, "chain": "bitcoin", "name": "elliptic"}
    return snap, meta


# ---------------------------------------------------------------------------
# GrabPhisher / eth-phishing-detection repository (github.com/YancyKahn/eth-phishing-detection)
# ---------------------------------------------------------------------------
# What the repository actually contains (verified 2026-10-07, see DATA_CARD.md):
#   datasets/etherscan/ex_{1,2}/{phishing,normal}_node_graphs.pkl : dict address -> networkx.DiGraph,
#     one EGO graph per labelled "center" account, built by src/utils/preprocess.py from the center's
#     last <=1000 external transactions. Edges are AGGREGATED per (From, To):
#         value = SUM of values (ETH), timestamp = MAX (last) timestamp, count = number of txs.
#   Label convention in the repository: phishing = 0, normal = 1 (inverted w.r.t. ours).
#   datasets/open/ex1/phishing_address.csv : Etherscan "phish-hack" label list (5,781 rows, 4,890 distinct).
# The raw per-transaction CSVs (original_data/...) are NOT in the repository.

_GRAB_ALLOW = {("networkx.classes.digraph", "DiGraph"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"), ("numpy", "dtype")}


def _safe_load_ego_pickle(path: str):
    """Unpickle only networkx.DiGraph and numpy scalars/dtypes; anything else raises.
    Pickles can execute code on load, so the allow-list is the security boundary."""
    import pickle

    class _Safe(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in _GRAB_ALLOW:
                raise pickle.UnpicklingError(f"blocked global {module}.{name}")
            if module.endswith("multiarray"):
                import numpy.core.multiarray as mm  # numpy>=2 re-exports under numpy.core
                return getattr(mm, name)
            return super().find_class(module, name)

    with open(path, "rb") as fh:
        return _Safe(fh).load()


def load_grabphisher_ego(root: str, variant: str = "ex_2", expand: str = "last",
                         strict_labels: bool = True, label_neighbors: bool = False,
                         label_time_policy: str = "none", lag_days: float = 30.0,
                         t_min: Optional[float] = None, t_max: Optional[float] = None
                         ) -> Tuple[Ledger, Dict[str, int], Dict[str, float], Meta]:
    """Build a Ledger from the repository's aggregated ego graphs.

    expand : how an aggregated (u,v,value_sum,t_last,count) edge becomes Tx records.
        "last"   -> `count` Txs, each value_sum/count, all at t_last (exact counts and totals;
                    timing of collapsed pairs is NOT recoverable and is pushed to the last tx).
        "single" -> one Tx with value_sum at t_last (exact pair existence and totals only).
    strict_labels : keep a phishing center only if it is on the phish-hack list, drop a normal
        center that is on the list, and drop centers labelled both ways.
    label_neighbors : if True, non-center addresses on the phish-hack list are labelled 1. Default
        False because non-center features are computed only from their edges to centers
        (ego-sampling) and are therefore not comparable to center features.
    label_time_policy : "none" (harness applies first_seen + lag), "first_seen", or
        "last_activity" (label assumed to arrive `lag_days` after the address's last
        transaction in the sample, i.e. flagged on abandonment). Both are ASSUMPTIONS.
    t_min, t_max : optional fixed time window (unix seconds) applied to all edges.
    """
    import pandas as pd

    gdir = os.path.join(root, "datasets", "etherscan", variant)
    lab_csv = os.path.join(root, "datasets", "open", "ex1", "phishing_address.csv")
    for p in (gdir, lab_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found — no synthetic substitution is performed")
    P = _safe_load_ego_pickle(os.path.join(gdir, "phishing_node_graphs.pkl"))
    N = _safe_load_ego_pickle(os.path.join(gdir, "normal_node_graphs.pkl"))
    phish_list = set(pd.read_csv(lab_csv)["address"].str.lower())

    # ---- labels (ours: 1 = phishing, 0 = normal) ---------------------------
    p_c = {a.lower() for a in P}
    n_c = {a.lower() for a in N}
    conflict = p_c & n_c
    stats = {"phishing_centers_raw": len(p_c), "normal_centers_raw": len(n_c), "conflicting_centers": len(conflict)}
    p_keep = p_c - conflict
    n_keep = n_c - conflict
    if strict_labels:
        stats["phishing_dropped_not_on_list"] = len(p_keep - phish_list)
        stats["normal_dropped_on_list"] = len(n_keep & phish_list)
        p_keep &= phish_list
        n_keep -= phish_list
    labels: Dict[str, int] = {a: 1 for a in p_keep}
    labels.update({a: 0 for a in n_keep})

    # ---- deduplicate aggregated edges across overlapping ego graphs ---------
    # Different centers cover different windows (last <=1000 txs each), so the same pair can
    # appear with different counts; keep the record with the largest count (widest window).
    pairs: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    n_conflicts = 0
    for G in list(P.values()) + list(N.values()):
        for u, v, a in G.edges(data=True):
            u, v = str(u).lower(), str(v).lower()
            if u == v:
                continue
            rec = (float(a["value"]), float(a["timestamp"]), int(a.get("count", 1)))
            if t_min is not None and rec[1] < t_min:
                continue
            if t_max is not None and rec[1] > t_max:
                continue
            old = pairs.get((u, v))
            if old is not None and old != rec:
                n_conflicts += 1
                if rec[2] <= old[2]:
                    continue
            pairs[(u, v)] = rec

    # ---- expand into Tx records ------------------------------------------------
    order = sorted(pairs.items(), key=lambda kv: kv[1][1])
    txs = []
    n_collapsed = 0
    for (u, v), (val, ts, cnt) in order:
        if cnt > 1:
            n_collapsed += 1
        if expand == "single" or cnt == 1:
            txs.append(Tx(u, v, max(val, 0.0), ts, 21_000, origin="chain" if cnt == 1 else "chain_agg"))
        elif expand == "last":
            per = max(val, 0.0) / cnt
            txs.extend(Tx(u, v, per, ts, 21_000, origin="chain_agg") for _ in range(cnt))
        else:
            raise ValueError(f"unknown expand mode {expand!r}")
    ledger = Ledger(txs)

    if label_neighbors:
        idx = ledger.address_index()
        for a in phish_list:
            if a in idx and a not in labels:
                labels[a] = 1

    # ---- label arrival times (assumption, never data) -------------------------
    label_time: Dict[str, float] = {}
    if label_time_policy != "none":
        idx = ledger.address_index()
        a = ledger.arrays()
        n = ledger.num_addresses
        last = np.full(n, -np.inf)
        np.maximum.at(last, a["src"], a["timestamp"])
        np.maximum.at(last, a["dst"], a["timestamp"])
        fs = ledger.first_seen()
        for addr in labels:
            i = idx.get(addr)
            if i is None:
                continue
            base = fs[i] if label_time_policy == "first_seen" else last[i]
            label_time[addr] = float(base + lag_days * 86400.0)

    lo, hi = ledger.time_range()
    meta: Meta = {
        "synthetic": False, "features_derivable": True, "chain": "ethereum",
        "name": f"grabphisher_{variant}", "source": "github.com/YancyKahn/eth-phishing-detection",
        "value_unit": "ETH", "gas_known": False, "timestamps_exact": n_collapsed == 0,
        "representation": "aggregate_reconstruction", "historical_state_verified": False,
        "within_pair_amounts_observed": False, "label_times_observed": False,
        "first_seen_semantics": "first selected aggregate last timestamp",
        "ego_sampled": True, "label_nodes": "centers" if not label_neighbors else "centers+list",
        "label_time_policy": label_time_policy, "lag_days": lag_days,
        "pairs": len(pairs), "collapsed_pairs": n_collapsed,
        "collapsed_pairs_frac": n_collapsed / max(len(pairs), 1),
        "pair_conflicts_resolved": n_conflicts, "n_txs": len(ledger),
        "time_range": (lo, hi), "n_labels": len(labels),
        "n_phishing": int(sum(labels.values())), **stats,
    }
    return ledger, labels, label_time, meta
