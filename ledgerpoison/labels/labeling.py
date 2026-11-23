"""
The forensic-labelling pipeline as an explicit, ablatable component.

Two parts:

1. `visible_labels(snapshot, t)` — ground-truth forensic labels that have
   *arrived* by time t (lag modelled per address via `label_time`). This is
   the only source of true labels; attackers never write here.

2. `PseudoLabeler` — what the defender does with unlabelled nodes before
   retraining: nothing ("none"), majority-vote label propagation over the
   graph ("lp", vectorised with scipy.sparse), or confidence-thresholded
   self-training from a model's scores ("self_train"). Label-Delay
   Exploitation only exists if the defender uses (2); the paper must report
   results under each option.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..ledger.graph import GraphSnapshot


def visible_labels(snapshot: GraphSnapshot, t: float) -> np.ndarray:
    return snapshot.labels_known_at(t)


@dataclass
class PseudoLabeler:
    method: str = "none"           # "none" | "lp" | "self_train"
    iterations: int = 10
    min_known_neighbors: int = 1
    confidence: float = 0.9        # self-training threshold on P(illicit) or P(licit)

    def __call__(self, snapshot: GraphSnapshot, y_visible: np.ndarray,
                 scores: Optional[np.ndarray] = None) -> np.ndarray:
        if self.method == "none":
            return y_visible.copy()
        if self.method == "lp":
            return label_propagation(snapshot, y_visible, self.iterations, self.min_known_neighbors)
        if self.method == "self_train":
            if scores is None:
                raise ValueError("self_train needs model scores P(illicit) per node")
            y = y_visible.copy()
            unk = y < 0
            y[unk & (scores >= self.confidence)] = 1
            y[unk & (scores <= 1.0 - self.confidence)] = 0
            return y
        raise ValueError(f"unknown pseudo-labeler {self.method!r}")


def label_propagation(snapshot: GraphSnapshot, y_visible: np.ndarray, iterations: int = 10,
                      min_known_neighbors: int = 1) -> np.ndarray:
    """Iterative 1-hop majority vote over the undirected graph for unknown
    nodes. Ties go to licit (the conservative choice a real pipeline makes),
    which is exactly the behaviour Label-Delay Exploitation targets."""
    n = snapshot.num_nodes
    src, dst = snapshot.edge_index
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    y = y_visible.copy()
    for _ in range(iterations):
        known = y >= 0
        pos = A @ (known & (y == 1)).astype(np.float64)
        neg = A @ (known & (y == 0)).astype(np.float64)
        unk = ~known
        decidable = unk & ((pos + neg) >= min_known_neighbors)
        new = y.copy()
        new[decidable & (pos > neg)] = 1
        new[decidable & (pos <= neg)] = 0
        if np.array_equal(new, y):
            break
        y = new
    return y
