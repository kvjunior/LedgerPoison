"""
Temporal Consistency Filtering (preprocessing defense).

Quarantines *young* nodes (age < age_threshold at retrain time) whose
total degree is anomalous (z-score above z_threshold against the
established-node degree distribution) or whose neighbourhood is mostly
young. Quarantined nodes stay in the graph for message passing but are
removed from the training mask. Returns the filtered mask and diagnostics
(how many attacker nodes were caught — analysis only, needs attacker_mask).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..ledger.graph import GraphSnapshot


def temporal_consistency_filter(snapshot: GraphSnapshot, train_mask: np.ndarray, t_now: float,
                                age_threshold: float = 14 * 86400.0, z_threshold: float = 2.0,
                                young_neighbor_frac: float = 0.5) -> Tuple[np.ndarray, Dict[str, float]]:
    n = snapshot.num_nodes
    src, dst = snapshot.edge_index
    deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    age = snapshot.age(t_now)
    young = age < age_threshold
    est = ~young
    mu, sd = (deg[est].mean(), deg[est].std() + 1e-6) if est.any() else (deg.mean(), deg.std() + 1e-6)
    z = (deg - mu) / sd
    nb_young = np.bincount(dst, weights=young[src].astype(float), minlength=n) + \
               np.bincount(src, weights=young[dst].astype(float), minlength=n)
    frac_young = nb_young / np.maximum(deg, 1)
    flagged = young & ((z > z_threshold) | (frac_young > young_neighbor_frac))
    new_mask = train_mask & ~flagged
    diag: Dict[str, float] = {"flagged": int(flagged.sum()), "flagged_in_train": int((flagged & train_mask).sum())}
    if snapshot.attacker_mask is not None:
        am = snapshot.attacker_mask
        diag["attacker_caught"] = int((flagged & am).sum())
        diag["attacker_total"] = int(am.sum())
        diag["honest_flagged"] = int((flagged & ~am).sum())
    return new_mask, diag
