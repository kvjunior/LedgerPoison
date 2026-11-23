"""
Fee model. The attacker's budget is denominated in USD; cost is charged
per *transaction*, never per address (creating an externally owned account
is free on Ethereum and a Bitcoin address is just a key).

Ethereum: fee = gas_used * gas_price(t) [gwei] * 1e-9 * ETH_USD(t).
Bitcoin:  fee = vbytes * fee_rate(t) [sat/vB] * 1e-8 * BTC_USD(t).

Price series are supplied as (timestamp, value) arrays sorted by time and
looked up by step function; constants are accepted for grid sweeps. The
ROI analysis in `economics/roi.py` sweeps the price grid explicitly rather
than baking one market point into the attack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .ledger import Tx


class PriceSeries:
    """Step-function price lookup. Constant if a scalar is given."""

    def __init__(self, value: float | Sequence[Tuple[float, float]]):
        if np.isscalar(value):
            self._t = None
            self._v = float(value)
            if not np.isfinite(self._v) or self._v <= 0: raise ValueError("positive finite price required")
        else:
            arr = np.asarray(sorted(value), dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2 or not len(arr) or not np.isfinite(arr).all() or (arr[:,1] <= 0).any():
                raise ValueError("positive finite price series required")
            self._t, self._v = arr[:, 0], arr[:, 1]

    def at(self, t: np.ndarray | float) -> np.ndarray:
        if self._t is None:
            return np.full_like(np.asarray(t, dtype=np.float64), self._v)
        i = np.searchsorted(self._t, np.asarray(t, dtype=np.float64), side="right") - 1
        i = np.clip(i, 0, len(self._t) - 1)
        return self._v[i]


@dataclass
class FeeModel:
    chain: str                          # "ethereum" | "bitcoin"
    unit_price: PriceSeries             # gwei per gas, or sat per vbyte
    fiat_price: PriceSeries             # USD per ETH / BTC
    simple_transfer_units: int = 21_000 # gas for a plain ETH transfer; ~140 vbytes for a 1-in-2-out BTC tx

    def __post_init__(self):
        if self.chain not in ('ethereum','bitcoin'): raise ValueError('unsupported fee chain')
        if self.simple_transfer_units<=0: raise ValueError('invalid transfer units')

    def tx_cost_usd(self, txs: Iterable[Tx]) -> np.ndarray:
        txs = list(txs)
        if not txs:
            return np.zeros(0)
        t = np.array([x.timestamp for x in txs])
        units = np.array([x.gas_used for x in txs], dtype=np.float64)
        scale = 1e-9 if self.chain == "ethereum" else 1e-8
        return units * self.unit_price.at(t) * scale * self.fiat_price.at(t)

    def total_usd(self, txs: Iterable[Tx]) -> float:
        return float(self.tx_cost_usd(txs).sum())

    def max_transactions(self, budget_usd: float, t: float) -> int:
        """How many simple transfers the budget buys at time t."""
        unit = self.simple_transfer_units * (1e-9 if self.chain == "ethereum" else 1e-8)
        per_tx = unit * float(self.unit_price.at(t)) * float(self.fiat_price.at(t))
        return int(np.floor(budget_usd / per_tx + 1e-9)) if per_tx > 0 else 0


def ethereum_default(gas_gwei: float = 30.0, eth_usd: float = 3500.0) -> FeeModel:
    return FeeModel("ethereum", PriceSeries(gas_gwei), PriceSeries(eth_usd), 21_000)


def bitcoin_default(sat_per_vb: float = 20.0, btc_usd: float = 60_000.0) -> FeeModel:
    return FeeModel("bitcoin", PriceSeries(sat_per_vb), PriceSeries(btc_usd), 140)


def load_etherscan_chart_csv(path: str, scale: float = 1.0) -> PriceSeries:
    """Etherscan chart export ("Date(UTC)","UnixTimeStamp","Value"), e.g.
    export-AvgGasPrice.csv (Value in wei -> pass scale=1e-9 for gwei) or
    export-EtherPrice.csv (Value in USD). Rows with Value <= 0 are dropped."""
    import csv
    pts = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            v = float(row["Value"] if "Value" in row else row["Value (Wei)"])
            if v > 0:
                pts.append((float(row["UnixTimeStamp"]), v * scale))
    if not pts:
        raise ValueError(f"no positive values in {path}")
    return PriceSeries(pts)


def ethereum_historical(gas_csv: str, eth_usd_csv: Optional[str] = None,
                        eth_usd_constant: Optional[float] = None) -> FeeModel:
    """Fee model from Etherscan's daily average gas price and ETH/USD series. If no
    ETH/USD series is supplied a constant must be given explicitly (and reported)."""
    gas = load_etherscan_chart_csv(gas_csv, scale=1e-9)
    if eth_usd_csv:
        fiat = load_etherscan_chart_csv(eth_usd_csv)
    elif eth_usd_constant is not None:
        fiat = PriceSeries(eth_usd_constant)
    else:
        raise ValueError("ETH/USD series missing: pass eth_usd_csv (export-EtherPrice.csv) or eth_usd_constant")
    return FeeModel("ethereum", gas, fiat, 21_000)
