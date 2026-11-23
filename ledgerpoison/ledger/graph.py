"""
Build a detector-facing graph snapshot from the ledger at a reference time.

`GraphSnapshot` is pure NumPy so that the whole attack/label/economics side
of the pipeline runs without PyTorch; `detectors/` converts it to tensors.
Nothing in the snapshot reveals transaction provenance: `attacker_mask` is
carried separately and only the harness/analysis may read it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .features import compute_features
from .ledger import Ledger


@dataclass
class GraphSnapshot:
    t_ref: float
    x: np.ndarray                      # [N, d] raw features
    edge_index: np.ndarray             # [2, E] int64, one directed edge per tx
    edge_time: np.ndarray              # [E] timestamps
    edge_value: np.ndarray             # [E]
    first_seen: np.ndarray             # [N] seconds
    address_index: Dict[str, int]
    y: np.ndarray                      # [N] in {-1 unknown, 0 licit, 1 illicit}
    label_time: np.ndarray             # [N] time the label became available (inf if unknown)
    # --- analysis-only, must not be fed to detectors ---
    attacker_mask: np.ndarray = field(default=None)   # [N] address injected by attacker
    attacker_edge: np.ndarray = field(default=None)   # [E]

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def labels_known_at(self, t: float) -> np.ndarray:
        """Labels visible to the defender at time t (forensic lag applied)."""
        y = self.y.copy()
        y[self.label_time > t] = -1
        return y

    def age(self, t: Optional[float] = None) -> np.ndarray:
        t = self.t_ref if t is None else t
        return np.maximum(t - self.first_seen, 0.0)


def build_snapshot(ledger: Ledger, t_ref: float, labels: Dict[str, int],
                   label_time: Optional[Dict[str, float]] = None,
                   age_cap: Optional[float] = None) -> GraphSnapshot:
    """Construct the graph of all transactions with timestamp <= t_ref.

    `labels` maps address -> {0,1}; missing addresses are unknown (-1).
    `label_time` maps address -> time the forensic label became available;
    if absent, labels are treated as available at time 0 (no lag).
    `age_cap` is forwarded to compute_features (see B1 fix)."""
    snap_ledger = ledger.snapshot(t_ref)
    idx = snap_ledger.address_index()
    n = snap_ledger.num_addresses
    a = snap_ledger.arrays()
    x = compute_features(snap_ledger, t_ref, age_cap=age_cap)

    y = np.full(n, -1, dtype=np.int64)
    lt = np.full(n, np.inf, dtype=np.float64)
    for addr, i in idx.items():
        if addr in labels:
            y[i] = int(labels[addr])
            lt[i] = 0.0 if label_time is None else float(label_time.get(addr, 0.0))

    # attacker provenance (analysis only)
    attacker_addr = set(snap_ledger.addresses_of("attacker"))
    amask = np.zeros(n, dtype=bool)
    for addr in attacker_addr:
        amask[idx[addr]] = True

    return GraphSnapshot(
        t_ref=t_ref, x=x,
        edge_index=np.stack([a["src"], a["dst"]]).astype(np.int64),
        edge_time=a["timestamp"], edge_value=a["value"],
        first_seen=snap_ledger.first_seen(), address_index=idx,
        y=y, label_time=lt, attacker_mask=amask, attacker_edge=a["attacker_tx"],
    )
