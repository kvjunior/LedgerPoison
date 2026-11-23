"""
Append-only transaction ledger.

Design invariant of the whole codebase: the ONLY way to change anything the
detector sees is to append transactions here. There is no delete, no modify,
no label API, and no feature API. Features, graphs and labels are *derived*
from the ledger by the modules in `features.py`, `graph.py` and `labels/`.

A `Tx` is the minimal chain-agnostic record: sender, receiver, value (in the
major currency unit: ETH here; adapters perform unit conversion), timestamp (unix seconds), and the
resource consumed for fee purposes (`gas_used` on Ethereum; `vbytes` on
Bitcoin). `origin` marks provenance for *analysis only*: attacks must never
read it, and the harness asserts that no detector input contains it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Tx:
    sender: str
    receiver: str
    value: float
    timestamp: float
    gas_used: int = 21_000          # Ethereum simple transfer; vbytes on Bitcoin
    origin: str = "chain"           # "chain" | "attacker" — analysis only

    def __post_init__(self) -> None:
        if not self.sender or not self.receiver:
            raise ValueError("empty address")
        if not np.isfinite(self.value) or not np.isfinite(self.timestamp):
            raise ValueError("non-finite transaction")
        if not isinstance(self.gas_used, (int, np.integer)):
            raise ValueError("gas_used must be an integer")
        if self.sender == self.receiver:
            raise ValueError("self-transfer not allowed as a graph edge")
        if self.value < 0:
            raise ValueError("negative value")
        if self.gas_used <= 0:
            raise ValueError("gas_used must be positive")


class Ledger:
    """Immutable-history, append-only store of transactions.

    Transactions are kept in a list in append order; `snapshot(t)` returns the
    prefix confirmed at or before time `t`. Address ids are assigned on first
    appearance (`address_index`), so injected addresses receive fresh ids and
    `first_seen` is a derived quantity, never attacker-settable.
    """

    def __init__(self, txs: Optional[Iterable[Tx]] = None) -> None:
        self._txs: List[Tx] = []
        self._addr_index: Dict[str, int] = {}
        self._first_seen: List[float] = []
        if txs is not None:
            self.append(txs)

    # ------------------------------------------------------------ mutation
    def append(self, txs: Iterable[Tx]) -> int:
        """Append transactions. Returns number appended. Never modifies
        existing records; ordering of *existing* records is preserved."""
        n = 0
        for tx in txs:
            if not isinstance(tx, Tx):
                raise TypeError("only Tx objects may be appended")
            for a in (tx.sender, tx.receiver):
                if a not in self._addr_index:
                    self._addr_index[a] = len(self._addr_index)
                    self._first_seen.append(tx.timestamp)
                else:
                    i = self._addr_index[a]
                    if tx.timestamp < self._first_seen[i]:
                        # first_seen is the earliest timestamp on chain
                        self._first_seen[i] = tx.timestamp
            self._txs.append(tx)
            n += 1
        return n

    # ------------------------------------------------------------ queries
    def __len__(self) -> int:
        return len(self._txs)

    @property
    def transactions(self) -> Tuple[Tx, ...]:
        return tuple(self._txs)

    @property
    def num_addresses(self) -> int:
        return len(self._addr_index)

    def address_index(self) -> Dict[str, int]:
        return dict(self._addr_index)

    def first_seen(self) -> np.ndarray:
        return np.asarray(self._first_seen, dtype=np.float64)

    def snapshot(self, t_max: float, t_min: float = -np.inf) -> "Ledger":
        """Ledger restricted to transactions with t_min <= timestamp <= t_max.
        Address ids are re-assigned in the snapshot (fresh Ledger)."""
        return Ledger(tx for tx in self._txs if t_min <= tx.timestamp <= t_max)

    def arrays(self) -> Dict[str, np.ndarray]:
        """Columnar view used by features/graph builders."""
        idx = self._addr_index
        src = np.fromiter((idx[t.sender] for t in self._txs), dtype=np.int64, count=len(self._txs))
        dst = np.fromiter((idx[t.receiver] for t in self._txs), dtype=np.int64, count=len(self._txs))
        val = np.fromiter((t.value for t in self._txs), dtype=np.float64, count=len(self._txs))
        ts = np.fromiter((t.timestamp for t in self._txs), dtype=np.float64, count=len(self._txs))
        gas = np.fromiter((t.gas_used for t in self._txs), dtype=np.int64, count=len(self._txs))
        attacker = np.fromiter((t.origin == "attacker" for t in self._txs), dtype=bool, count=len(self._txs))
        return {"src": src, "dst": dst, "value": val, "timestamp": ts,
                "gas_used": gas, "attacker_tx": attacker}

    def time_range(self) -> Tuple[float, float]:
        if not self._txs:
            return (0.0, 0.0)
        ts = [t.timestamp for t in self._txs]
        return (min(ts), max(ts))

    def addresses_of(self, origin: str) -> List[str]:
        """Analysis helper (not for attacks): addresses that only ever appear
        in transactions of the given origin."""
        seen: Dict[str, set] = {}
        for t in self._txs:
            for a in (t.sender, t.receiver):
                seen.setdefault(a, set()).add(t.origin)
        return [a for a, o in seen.items() if o == {origin}]
