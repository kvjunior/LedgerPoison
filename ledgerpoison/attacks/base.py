"""
Attack base class. An attack is a *transaction plan*: a list of `Tx` the
attacker will submit between `t_start` and `t_end`, priced by the fee model.
The attack never touches features, edges, labels or masks; the harness
appends the plan to the ledger and everything downstream is recomputed.

Attacker knowledge (Kerckhoffs-style, all public):
  * the ledger snapshot at t_start (structure, values, timestamps);
  * labels that are *publicly visible* at t_start (e.g. tagged exchanges,
    published blacklists) — `y_visible`;
  * its own coalition: the fraud addresses it controls (`coalition`), which
    is the only set it may target — it does not know other actors' labels;
  * a surrogate classifier it trains itself on public data (`Surrogate`).

Invariants enforced by `validate_plan` (and unit-tested):
  I1 every planned tx has t_start <= timestamp <= t_end (no rewriting history);
  I2 total fee <= budget_usd;
  I3 senders are either coalition addresses or fresh attacker addresses; the
     attacker cannot spend from addresses it does not control;
  I4 no Tx carries a label, feature or mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..ledger.cost import FeeModel
from ..ledger.features import apply_standardize, compute_features, standardize
from ..ledger.graph import GraphSnapshot
from ..ledger.ledger import Ledger, Tx


@dataclass
class AttackContext:
    snapshot: GraphSnapshot         # public state at t_start
    ledger: Ledger                  # public ledger at t_start (for value/timing empirics)
    y_visible: np.ndarray           # public labels at t_start ({-1,0,1})
    coalition: List[str]            # attacker-owned fraud addresses
    fee: FeeModel
    budget_usd: float
    t_start: float
    t_end: float
    rng: np.random.Generator
    edges_per_buffer: int = 5
    fund_buffers: bool = True       # fresh addresses need an on-chain funding tx
    surrogate: Optional["Surrogate"] = None
    dust_eth: float = 1e-6          # B2: structural edges move dust, not real value
    timing: str = "cluster"         # default timing policy, overridable per attack
    age_days: float = 0.0           # default ageing, overridable per attack

    owned_addresses: set = field(default_factory=set)
    initial_balances_eth: Dict[str, float] = field(default_factory=dict)
    bankroll_eth: float = 100.0  # explicit simulation assumption, never inferred from the data
    coalition_balance_eth: float = 100.0

    def __post_init__(self):
        self.owned_addresses.update(self.coalition)
        for a in self.coalition:
            self.initial_balances_eth.setdefault(a, self.coalition_balance_eth)

    def bankroll_address(self, tag: str) -> str:
        a = self.fresh_address(tag, 0)
        self.initial_balances_eth.setdefault(a, self.bankroll_eth)
        return a

    # ---- helpers available to all attacks --------------------------------
    def licit_pool(self) -> np.ndarray:
        """Publicly-licit addresses (ids) the attacker can use as credibility
        counterparties. Falls back to high-degree unlabeled addresses."""
        ids = np.where(self.y_visible == 0)[0]
        if len(ids) >= 10:
            return ids
        deg = np.bincount(self.snapshot.edge_index[1], minlength=self.snapshot.num_nodes)
        return np.argsort(-deg)[:max(10, self.snapshot.num_nodes // 20)]

    def sample_licit_value(self, k: int) -> np.ndarray:
        """Empirical outgoing values of publicly-licit senders."""
        src = self.snapshot.edge_index[0]
        lic = np.isin(src, self.licit_pool())
        vals = self.snapshot.edge_value[lic]
        if len(vals) == 0:
            vals = self.snapshot.edge_value
        if len(vals) == 0:
            return np.ones(k)
        return self.rng.choice(vals, size=k, replace=True)

    def id_to_addr(self) -> Dict[int, str]:
        return {i: a for a, i in self.snapshot.address_index.items()}

    def fresh_address(self, tag: str, k: int) -> str:
        a = f"atk_{tag}_{k:06d}"
        if a in self.snapshot.address_index:
            raise ValueError("fresh address collision")
        self.owned_addresses.add(a)
        return a

    def timestamps(self, k: int, mode: str = "cluster", cluster_frac: float = 0.1) -> np.ndarray:
        """Timing policy for injected txs inside [t_start, t_end].
        'cluster': in the last `cluster_frac` of the window (default attacker);
        'spread': uniform over the whole window (adaptive attacker)."""
        lo, hi = self.t_start, self.t_end
        if mode == "spread":
            return self.rng.uniform(lo, hi, size=k)
        return self.rng.uniform(hi - cluster_frac * (hi - lo), hi, size=k)


class Surrogate:
    """Attacker-side logistic regression on public features/labels. Used only
    to *evaluate* candidate behaviours (Label-Delay template search); the
    victim's model is never queried."""

    def __init__(self, snapshot: GraphSnapshot, y_visible: np.ndarray, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        known = y_visible >= 0
        self.Xs, self.stats = standardize(snapshot.x, known)
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        if known.sum() > 0 and len(np.unique(y_visible[known])) == 2:
            self.clf.fit(self.Xs[known], y_visible[known])
            self.ok = True
        else:
            self.ok = False

    def p_illicit(self, X_raw: np.ndarray) -> np.ndarray:
        if not self.ok:
            return np.full(len(X_raw), 0.5)
        return self.clf.predict_proba(apply_standardize(X_raw, self.stats))[:, 1]


class Attack:
    name = "base"

    def __init__(self, **params):
        self.params = params

    def plan(self, ctx: AttackContext) -> List[Tx]:
        raise NotImplementedError

    # ---- shared building blocks -------------------------------------------
    # B2: a buffer's outgoing transfers move DUST (ctx.dust_eth). Structural
    # attacks shape degree/timing, not value, so there is no reason to move
    # real ETH, and doing so made the attacks cost millions and bankrupt the
    # buffers. The funding transfer covers (dust * n_out + gas headroom) and is
    # always emitted BEFORE the buffer's first outgoing tx (causal order).

    def _gas_eth(self, ctx: AttackContext, t: float) -> float:
        return 21_000 * float(ctx.fee.unit_price.at(t)) * (1e-9 if ctx.fee.chain == "ethereum" else 1e-8)

    def _fund(self, ctx: AttackContext, buffer: str, n_out: int, t_fund: float, funder: str) -> List[Tx]:
        """One funding transfer sized to cover the buffer's dust outflows plus
        gas headroom, emitted at t_fund (before the buffer spends)."""
        if not ctx.fund_buffers:
            return []
        need = n_out * ctx.dust_eth + (n_out + 1) * self._gas_eth(ctx, t_fund) * 2.0
        return [Tx(funder, buffer, float(need), float(t_fund), 21_000, origin="attacker")]

    def _credibility(self, ctx: AttackContext, buffer: str, k: int, t_lo: float, t_hi: float) -> List[Tx]:
        """Dust transfers from `buffer` to publicly-licit nodes, spread over
        [t_lo, t_hi] (B4: timing/age respected, not hard-coded to 24h)."""
        if k <= 0:
            return []
        pool = ctx.licit_pool()
        i2a = ctx.id_to_addr()
        picks = ctx.rng.choice(pool, size=min(k, len(pool)), replace=False)
        lo = max(ctx.t_start, min(t_lo, t_hi))
        ts = ctx.rng.uniform(lo, t_hi, size=len(picks))
        return [Tx(buffer, i2a[int(p)], ctx.dust_eth, float(tt), 21_000, origin="attacker")
                for p, tt in zip(picks, sorted(ts))]


def validate_plan(plan: Sequence[Tx], ctx: AttackContext, check_balance: bool = True) -> Dict[str, float]:
    """Enforce invariants I1–I4 and (B2) economic feasibility; returns cost
    summary. Raises on violation.

      I1 timestamps inside [t_start, t_end]
      I2 gas cost <= budget
      I3 senders are coalition- or attacker-owned
      I4 every tx tagged origin='attacker'
      B2 every fresh attacker address is funded before it spends and never goes
         negative (coalition members are assumed pre-funded on-chain and skipped)
    """
    if ctx.fee.chain != "ethereum":
        raise ValueError("account-balance validator supports Ethereum simulation only")
    if not check_balance:
        raise ValueError("balance validation cannot be disabled")
    for tx in plan:
        if not (ctx.t_start <= tx.timestamp <= ctx.t_end):
            raise AssertionError("I1: transaction outside attack window")
        if tx.origin != "attacker":
            raise AssertionError("I4: invalid provenance tag")
        if tx.sender not in ctx.owned_addresses:
            raise AssertionError("I3: sender is not registered as attacker-owned")
    cost = ctx.fee.total_usd(plan)
    outsider_value = sum(t.value for t in plan if t.receiver not in ctx.owned_addresses)
    principal_usd = sum(t.value * float(ctx.fee.fiat_price.at(t.timestamp))
                        for t in plan if t.receiver not in ctx.owned_addresses)
    total = cost + principal_usd
    if not np.isfinite(total) or total > ctx.budget_usd + 1e-9:
        raise AssertionError("I2: fees plus lost principal exceed budget")
    balances = dict(ctx.initial_balances_eth)
    if any(not np.isfinite(v) or v < 0 for v in balances.values()):
        raise ValueError("invalid explicit initial balance")
    nonces = {a: 0 for a in ctx.owned_addresses}
    # Stable order for equal timestamps; a later funding record cannot backdate a debit.
    for _, tx in sorted(enumerate(plan), key=lambda p: (p[1].timestamp, p[0])):
        gas = tx.gas_used * float(ctx.fee.unit_price.at(tx.timestamp)) * 1e-9
        debit = tx.value + gas
        if balances.get(tx.sender, 0.0) + 1e-10 < debit:
            raise AssertionError("B2: sender spends above explicitly available funds")
        balances[tx.sender] = balances.get(tx.sender, 0.0) - debit
        balances[tx.receiver] = balances.get(tx.receiver, 0.0) + tx.value
        nonces[tx.sender] += 1
    fresh = {a for t in plan for a in (t.sender,t.receiver) if a in ctx.owned_addresses} - set(ctx.coalition)
    return {"n_txs": len(plan), "cost_usd": total, "fee_usd": cost,
            "lost_principal_usd": principal_usd, "n_fresh_addresses": len(fresh),
            "value_to_outsiders_eth": outsider_value, "neg_balance_addrs": 0,
            "initial_balances_eth": ctx.initial_balances_eth,
            "ending_owned_balances_eth": {a: balances.get(a,0.0) for a in ctx.owned_addresses},
            "simulated_sender_nonces": nonces,
            "validation_scope": "finite funded account-transfer simulation; not EVM replay"}


def truncate_to_budget(plan: List[Tx], ctx: AttackContext) -> List[Tx]:
    """Keep the longest BUFFER-ALIGNED prefix affordable under the gas budget.

    Cutting mid-buffer could drop a buffer's funding tx while keeping its
    spends (B2 violation), so truncation happens only at the boundaries marked
    by a top-level funding tx (funder -> buffer). Everything a buffer does
    stays together or is dropped together."""
    if not plan:
        return plan
    # boundaries: indices where a new buffer's funding tx starts
    fresh_receivers = set()
    bounds = []
    for i, tx in enumerate(plan):
        if tx.receiver in ctx.owned_addresses and tx.receiver not in ctx.coalition and tx.receiver not in fresh_receivers:
            bounds.append(i); fresh_receivers.add(tx.receiver)
    if not bounds or bounds[0] != 0:
        bounds = [0] + bounds
    bounds.append(len(plan))
    costs = ctx.fee.tx_cost_usd(plan) + np.array([
        0.0 if t.receiver in ctx.owned_addresses else t.value * float(ctx.fee.fiat_price.at(t.timestamp))
        for t in plan])
    kept, spent = 0, 0.0
    for a, b in zip(bounds, bounds[1:]):
        seg = float(costs[a:b].sum())
        if spent + seg > ctx.budget_usd + 1e-9:
            break
        spent += seg; kept = b
    return plan[:kept]
