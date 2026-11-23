"""
Concrete attacks. Each returns a list of Tx ordered so that any affordable
prefix is coherent: a fresh buffer is funded before it spends (B2), and all
value moved by the attacker is dust (ctx.dust_eth) so the modelled cost is
gas + dust ~= gas (the honest economic cost).

  RandomInjection          — matched-cost / matched-shape control. Given a
                             treatment profile (buffers, edges each, target
                             multiset) it injects the SAME counts with random
                             targets and timing, so treatment - control isolates
                             strategy, not spend or shape (B3).
  NeighborhoodCamouflage   — buffers transact to coalition targets and to
                             publicly-licit counterparties (dilute aggregation).
  TopologyManipulation     — pad degree / out-in ratio of coalition members
                             toward the licit profile with dust round-trips.
  LabelDelayExploitation   — fresh addresses whose derived features sit near the
                             surrogate boundary, acting inside the labelling lag.

Timing (`timing`: cluster|spread) and ageing (`age_days`) are attack params;
both are honoured by the funding time and the credibility-edge spread (B4).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..ledger.features import compute_features
from ..ledger.ledger import Ledger, Tx
from .base import Attack, AttackContext, truncate_to_budget


def _target_order(ctx: AttackContext, coalition_ids: np.ndarray) -> np.ndarray:
    """Highest in-degree coalition members first (largest payoff per $)."""
    indeg = np.bincount(ctx.snapshot.edge_index[1], minlength=ctx.snapshot.num_nodes)
    return coalition_ids[np.argsort(-indeg[coalition_ids])]


def _coalition_ids(ctx: AttackContext) -> np.ndarray:
    return np.array([ctx.snapshot.address_index[a] for a in ctx.coalition
                     if a in ctx.snapshot.address_index], dtype=np.int64)


def _act_time(ctx: AttackContext, timing: str) -> float:
    return float(ctx.timestamps(1, timing)[0])


class _BufferAttack(Attack):
    """Shared buffer lifecycle: fund (at t_act - age_days) then act, emitted in
    causal order; every outgoing edge is dust. Subclasses supply the edges."""

    def _buffer_plan(self, ctx, buffer, funder, edges_to, timing, age_days):
        """edges_to: list of (dst_addr, t) dust edges the buffer sends."""
        if not edges_to:
            return []
        t_first = min(t for _, t in edges_to)
        t_fund = max(ctx.t_start, t_first - age_days * 86400.0)
        plan = self._fund(ctx, buffer, len(edges_to), t_fund, funder)
        plan += [Tx(buffer, dst, ctx.dust_eth, float(t), 21_000, origin="attacker")
                 for dst, t in sorted(edges_to, key=lambda e: e[1])]
        return plan


class RandomInjection(_BufferAttack):
    name = "random"

    def __init__(self, timing: str = "cluster", age_days: float = 0.0, profile: Dict = None):
        super().__init__(timing=timing, age_days=age_days, profile=profile)

    def plan(self, ctx: AttackContext) -> List[Tx]:
        p = self.params
        timing = p.get("timing") or ctx.timing
        age_days = p.get("age_days", ctx.age_days)
        i2a = ctx.id_to_addr()
        funder = ctx.bankroll_address("rfund")
        # profile = {"n_buffers": int, "edges_per_buffer": int} matched to the treatment (B3);
        # without one, fall back to filling the gas budget.
        prof = p.get("profile")
        n_buffers = prof["n_buffers"] if prof else max(
            1, ctx.fee.max_transactions(ctx.budget_usd, ctx.t_end) // (ctx.edges_per_buffer + 1))
        m = prof["edges_per_buffer"] if prof else ctx.edges_per_buffer
        plan: List[Tx] = []
        for k in range(n_buffers):
            b = ctx.fresh_address("rand", k)
            recv = ctx.rng.choice(ctx.snapshot.num_nodes, size=min(m, ctx.snapshot.num_nodes), replace=False)
            edges = [(i2a[int(r)], _act_time(ctx, timing)) for r in recv]
            plan += self._buffer_plan(ctx, b, funder, edges, timing, age_days)
            if ctx.fee.total_usd(plan) > ctx.budget_usd:
                break
        return truncate_to_budget(plan, ctx)


class NeighborhoodCamouflage(_BufferAttack):
    name = "camouflage"

    def __init__(self, timing: str = "cluster", age_days: float = 0.0, buffers_per_target: int = 10):
        super().__init__(timing=timing, age_days=age_days, buffers_per_target=buffers_per_target)

    def plan(self, ctx: AttackContext) -> List[Tx]:
        p = self.params
        timing, age_days = p["timing"], p["age_days"]
        i2a = ctx.id_to_addr()
        coalition_ids = _coalition_ids(ctx)
        if len(coalition_ids) == 0:
            return []
        targets = _target_order(ctx, coalition_ids)
        funder = ctx.bankroll_address("cfund")
        plan: List[Tx] = []
        k = 0
        for tgt in targets:
            for _ in range(p["buffers_per_target"]):
                b = ctx.fresh_address("camo", k); k += 1
                t_act = _act_time(ctx, timing)
                # one dilution edge to the target + (edges_per_buffer-1) credibility edges
                edges = [(i2a[int(tgt)], t_act)]
                pool = ctx.licit_pool()
                picks = ctx.rng.choice(pool, size=min(ctx.edges_per_buffer - 1, len(pool)), replace=False)
                edges += [(i2a[int(pp)], _act_time(ctx, timing)) for pp in picks]
                plan += self._buffer_plan(ctx, b, funder, edges, timing, age_days)
            if ctx.fee.total_usd(plan) > ctx.budget_usd:
                break
        return truncate_to_budget(plan, ctx)


class TopologyManipulation(_BufferAttack):
    """Pad total degree and out/in ratio of coalition members toward the licit
    profile with dust round-trips. The member -> buffer leg is sent from the
    coalition member (allowed by I3, member is attacker-owned)."""
    name = "topology"

    def __init__(self, timing: str = "cluster", age_days: float = 0.0, max_pad_per_member: int = 50):
        super().__init__(timing=timing, age_days=age_days, max_pad_per_member=max_pad_per_member)

    def plan(self, ctx: AttackContext) -> List[Tx]:
        p = self.params
        timing, age_days = p["timing"], p["age_days"]
        snap = ctx.snapshot
        i2a = ctx.id_to_addr()
        out_d = np.bincount(snap.edge_index[0], minlength=snap.num_nodes).astype(float)
        in_d = np.bincount(snap.edge_index[1], minlength=snap.num_nodes).astype(float)
        deg = out_d + in_d
        lic = ctx.licit_pool()
        target_deg = float(deg[lic].mean()) if len(lic) else float(deg.mean())
        ratio_lic = float(np.median(out_d[lic] / np.maximum(in_d[lic], 1.0))) if len(lic) else 1.0
        coalition_ids = _coalition_ids(ctx)
        if len(coalition_ids) == 0:
            return []
        need_deg = np.ceil(target_deg - deg[coalition_ids])
        need_ratio = np.ceil(ratio_lic * np.maximum(in_d[coalition_ids], 1.0) - out_d[coalition_ids])
        need = np.clip(np.maximum(need_deg, need_ratio), 0, p["max_pad_per_member"]).astype(int)
        order = np.argsort(-need)
        funder = ctx.bankroll_address("tfund")
        plan: List[Tx] = []
        k = 0
        for j in order:
            tgt, n_pad = int(coalition_ids[j]), int(need[j])
            for _ in range(n_pad):
                b = ctx.fresh_address("topo", k); k += 1
                t_act = _act_time(ctx, timing)
                t_fund = max(ctx.t_start, t_act - age_days * 86400.0)
                # member -> buffer (dust, out-degree padding); fund the member leg's gas via funder->member? No:
                # the coalition member is attacker-owned and assumed funded on-chain already, so it can send dust.
                # buffer -> member (dust, total-degree padding): buffer must be funded first.
                buf_edges = [(i2a[tgt], min(ctx.t_end, t_act + 3600.0))]
                pool = ctx.licit_pool()
                picks = ctx.rng.choice(pool, size=min(max(ctx.edges_per_buffer - 2, 0), len(pool)), replace=False)
                buf_edges += [(i2a[int(pp)], _act_time(ctx, timing)) for pp in picks]
                plan.append(Tx(i2a[tgt], b, ctx.dust_eth, t_act, 21_000, origin="attacker"))   # member -> buffer
                plan += self._buffer_plan(ctx, b, funder, buf_edges, timing, age_days)
            if ctx.fee.total_usd(plan) > ctx.budget_usd:
                break
        return truncate_to_budget(plan, ctx)


class LabelDelayExploitation(_BufferAttack):
    name = "label_delay"

    def __init__(self, timing: str = "cluster", age_days: float = 0.0, lag_window_days: float = 7.0,
                 n_addresses: int = 50, template_grid: int = 12, target_p: float = 0.5):
        super().__init__(timing=timing, age_days=age_days, lag_window_days=lag_window_days,
                         n_addresses=n_addresses, template_grid=template_grid, target_p=target_p)

    def _borderline_template(self, ctx: AttackContext):
        """Search (n_txs, in_share) templates for the one whose DERIVED features
        sit nearest the surrogate boundary. Values are dust, so only structure
        and timing move the features (B2: no value search)."""
        if ctx.surrogate is None:
            return dict(n_txs=ctx.edges_per_buffer, in_share=0.5)
        best, best_gap = None, np.inf
        rng = np.random.default_rng(0)
        pool = ctx.licit_pool()
        i2a = ctx.id_to_addr()
        for n_txs in (3, 5, 8, 12):
            for in_share in (0.2, 0.5, 0.8):
                probe = "atk_probe"
                funder = "atk_probe_bankroll"
                t = ctx.t_end
                txs = [Tx(funder, probe, n_txs * (ctx.dust_eth + 42000 * float(ctx.fee.unit_price.at(t)) * 1e-9), ctx.t_start)]
                for _ in range(n_txs):
                    cp = i2a[int(rng.choice(pool))]
                    tt = max(ctx.t_start, t - rng.uniform(0, 3 * 86400.0))
                    txs.append(Tx(funder, probe, ctx.dust_eth, tt) if rng.random() < in_share
                               else Tx(probe, cp, ctx.dust_eth, tt))
                mini = Ledger(txs)
                X = compute_features(mini, ctx.t_end)
                x_probe = X[mini.address_index()[probe]][None, :]
                gap = abs(float(ctx.surrogate.p_illicit(x_probe)[0]) - self.params["target_p"])
                if gap < best_gap:
                    best_gap, best = gap, dict(n_txs=n_txs, in_share=in_share)
        return best

    def plan(self, ctx: AttackContext) -> List[Tx]:
        p = self.params
        timing, age_days = p["timing"], p["age_days"]
        tpl = self._borderline_template(ctx)
        i2a = ctx.id_to_addr()
        pool = ctx.licit_pool()
        funder = ctx.bankroll_address("lfund")
        lag_lo = max(ctx.t_start, ctx.t_end - p["lag_window_days"] * 86400.0)
        plan: List[Tx] = []
        for k in range(p["n_addresses"]):
            b = ctx.fresh_address("ldelay", k)
            t_act = float(ctx.rng.uniform(lag_lo, ctx.t_end)) if timing == "cluster" \
                else float(ctx.rng.uniform(ctx.t_start, ctx.t_end))
            picks = ctx.rng.choice(pool, size=min(tpl["n_txs"], len(pool)), replace=False)
            edges = []
            inflow = []
            for cp in picks:
                tt = float(ctx.rng.uniform(max(ctx.t_start, t_act - 3 * 86400.0), t_act))
                # in_share is realised by receiving from a bankroll funder address
                # (an attacker in-edge), giving the Carlini-style inflow signature
                # without leaving a prior buffer under-funded.
                if ctx.rng.random() < tpl["in_share"]:
                    inflow.append((ctx.bankroll_address("lfund"), tt))
                else:
                    edges.append((i2a[int(cp)], tt))
            plan += self._buffer_plan(ctx, b, funder, edges, timing, age_days)
            # inflow edges: bankroll funder -> this buffer (funder is pre-funded)
            plan += [Tx(src, b, ctx.dust_eth, float(tt), 21_000, origin="attacker") for src, tt in inflow]
            if ctx.fee.total_usd(plan) > ctx.budget_usd:
                break
        return truncate_to_budget(plan, ctx)


ATTACKS = {
    "random": RandomInjection,
    "camouflage": NeighborhoodCamouflage,
    "topology": TopologyManipulation,
    "label_delay": LabelDelayExploitation,
}


def treatment_profile(plan: List[Tx]) -> Dict:
    """Buffer/edge counts of a treatment plan, so the random control can be
    matched to the same shape (B3)."""
    out = {}
    for tx in plan:
        if tx.sender.startswith("atk_"):
            out.setdefault(tx.sender, 0)
            out[tx.sender] += 1
    bufs = [b for b in out if not b.startswith(("atk_rfund", "atk_cfund", "atk_tfund", "atk_lfund"))]
    n = len(bufs)
    epb = int(round(np.mean([out[b] for b in bufs]))) if bufs else 1
    return {"n_buffers": n, "edges_per_buffer": max(epb, 1)}


def matched_random_plan(reference, ctx):
    """Preserve every target/owned endpoint, amount, time, and gas unit.
    Randomize only public non-target counterparties; match their count/uniqueness.
    """
    from dataclasses import replace
    rng = ctx.rng
    existing = sorted(a for a in ctx.snapshot.address_index if a not in ctx.coalition)
    if not existing:
        raise ValueError("no public non-target counterparties")
    mapping = {}
    unique = sorted({t.receiver for t in reference if t.receiver not in ctx.owned_addresses
                     and t.receiver not in ctx.coalition})
    if len(unique) > len(existing): raise ValueError("not enough counterparties")
    for a,b in zip(unique,rng.choice(existing,len(unique),replace=False)):
        mapping[a] = str(b)
    return [replace(t,receiver=mapping.get(t.receiver,t.receiver)) for t in reference]
