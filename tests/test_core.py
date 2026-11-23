"""Torch-free tests: ledger invariants, feature determinism, fee model,
labelling, attack-plan invariants. Run: pytest tests/test_core.py -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ledgerpoison.attacks.base import AttackContext, Surrogate, validate_plan
from ledgerpoison.attacks.strategies import ATTACKS
from ledgerpoison.labels.labeling import PseudoLabeler, label_propagation
from ledgerpoison.ledger.adapters import synthetic_ledger
from ledgerpoison.ledger.cost import PriceSeries, ethereum_default
from ledgerpoison.ledger.features import FEATURE_DIM, FEATURE_NAMES, compute_features
from ledgerpoison.ledger.graph import build_snapshot
from ledgerpoison.ledger.ledger import Ledger, Tx


@pytest.fixture(scope="module")
def data():
    return synthetic_ledger(n_addresses=800, n_txs=5000, seed=1)


# ---------------------------------------------------------------- ledger
def test_ledger_append_only():
    L = Ledger([Tx("a", "b", 1.0, 10.0)])
    assert not hasattr(L, "delete") and not hasattr(L, "modify")
    exposed = L.transactions
    assert isinstance(exposed, tuple)
    with pytest.raises(TypeError):
        exposed[0] = Tx("x", "y", 0, 1)
    assert L.transactions[0] == Tx("a", "b", 1.0, 10.0)
    L2 = Ledger([Tx("a", "b", 1.0, 10.0)])
    n_before = len(L2)
    L2.append([Tx("c", "d", 2.0, 5.0)])
    assert len(L2) == n_before + 1
    assert L2.transactions[0] == Tx("a", "b", 1.0, 10.0)          # ordering preserved
    assert L2.first_seen()[L2.address_index()["a"]] == 10.0        # first_seen derived, not settable


def test_tx_validation():
    with pytest.raises(ValueError):
        Tx("a", "a", 1.0, 0.0)
    with pytest.raises(ValueError):
        Tx("a", "b", -1.0, 0.0)


def test_snapshot_respects_time(data):
    ledger = data[0]
    lo, hi = ledger.time_range()
    mid = (lo + hi) / 2
    s = ledger.snapshot(mid)
    assert all(t.timestamp <= mid for t in s.transactions)


# --------------------------------------------------------------- features
def test_feature_dim_and_determinism(data):
    ledger = data[0]
    t = ledger.time_range()[1]
    X1, X2 = compute_features(ledger, t), compute_features(ledger, t)
    assert X1.shape == (ledger.num_addresses, FEATURE_DIM) and np.array_equal(X1, X2)
    assert len(FEATURE_NAMES) == FEATURE_DIM


def test_injection_changes_only_endpoints(data):
    ledger = data[0]
    t = ledger.time_range()[1]
    X0 = compute_features(ledger, t)
    idx = ledger.address_index()
    a, b = "a000001", "a000002"
    L2 = Ledger(ledger.transactions)
    L2.append([Tx(a, b, 3.0, t - 1.0, origin="attacker")])
    X1 = compute_features(L2, t)
    changed = np.where(np.any(X1[: X0.shape[0]] != X0, axis=1))[0]
    assert set(changed) == {idx[a], idx[b]}


def test_no_future_leakage(data):
    ledger = data[0]
    lo, hi = ledger.time_range()
    mid = (lo + hi) / 2
    X_mid = compute_features(ledger, mid)
    X_mid_from_snapshot = compute_features(ledger.snapshot(mid), mid)
    # every address present in the snapshot has identical features either way
    idx_full, idx_snap = ledger.address_index(), ledger.snapshot(mid).address_index()
    for addr, i in idx_snap.items():
        assert np.allclose(X_mid[idx_full[addr]], X_mid_from_snapshot[i], atol=1e-5)


# -------------------------------------------------------------------- cost
def test_fee_model_charges_per_tx_not_per_address():
    fee = ethereum_default(30.0, 3500.0)
    txs = [Tx("x", f"y{i}", 1.0, 1e9, 21_000) for i in range(10)]
    assert abs(fee.total_usd(txs) - 10 * 21_000 * 30e-9 * 3500.0) < 1e-9
    assert fee.max_transactions(fee.total_usd(txs), 1e9) == 10


def test_price_series_step():
    ps = PriceSeries([(0.0, 10.0), (100.0, 20.0)])
    assert ps.at(50.0) == 10.0 and ps.at(150.0) == 20.0 and ps.at(-5.0) == 10.0


# ------------------------------------------------------------------ labels
def test_label_lag_and_propagation(data):
    ledger, labels, label_time, meta = data
    lo, hi = ledger.time_range()
    snap = build_snapshot(ledger, hi, labels, label_time)
    early = snap.labels_known_at(lo)
    late = snap.labels_known_at(hi + 365 * 86400.0)
    assert (early >= 0).sum() < (late >= 0).sum()               # lag hides labels
    y_lp = label_propagation(snap, early)
    assert ((y_lp >= 0).sum() >= (early >= 0).sum())            # LP only adds labels
    assert np.array_equal(y_lp[early >= 0], early[early >= 0])  # never overwrites known labels
    assert np.array_equal(PseudoLabeler("none")(snap, early), early)


# ----------------------------------------------------------------- attacks
def _ctx(data, attack, budget=50.0, seed=0):
    ledger, labels, label_time, meta = data
    lo, hi = ledger.time_range()
    t_train, t_eval = lo + 0.6 * (hi - lo), lo + 0.8 * (hi - lo)
    A = build_snapshot(ledger, t_train, labels, label_time)
    y_pub = A.labels_known_at(t_train)
    fraud = [a for a, i in A.address_index.items() if A.y[i] == 1][:10]
    return AttackContext(snapshot=A, ledger=ledger.snapshot(t_train), y_visible=y_pub, coalition=fraud,
                         fee=ethereum_default(), budget_usd=budget, t_start=t_train, t_end=t_eval,
                         rng=np.random.default_rng(seed),
                         surrogate=Surrogate(A, y_pub) if attack == "label_delay" else None)


@pytest.mark.parametrize("attack", list(ATTACKS))
def test_plan_invariants(data, attack):
    ctx = _ctx(data, attack)
    plan = ATTACKS[attack]().plan(ctx)
    stats = validate_plan(plan, ctx)           # raises on I1–I4 AND B2 balance violation
    assert stats["n_txs"] > 0 and stats["cost_usd"] <= ctx.budget_usd
    assert stats["neg_balance_addrs"] == 0     # B2: no fresh address ever goes negative
    for tx in plan:
        assert set(tx.__dataclass_fields__) == {"sender", "receiver", "value", "timestamp", "gas_used", "origin"}


def test_attacks_move_only_dust_to_outsiders(data):
    # B2: structural edges carry dust; no real ETH leaves the attacker's control
    for attack in ("camouflage", "topology", "label_delay"):
        ctx = _ctx(data, attack, budget=200)
        plan = ATTACKS[attack]().plan(ctx)
        stats = validate_plan(plan, ctx)
        assert stats["value_to_outsiders_eth"] <= ctx.dust_eth * stats["n_txs"] + 1e-9


def test_matched_cost_control_is_comparable(data):
    from ledgerpoison.attacks.strategies import matched_random_plan
    ctx=_ctx(data,'camouflage');pc=ATTACKS['camouflage']().plan(ctx)
    pr=matched_random_plan(pc,ctx)
    a,b=validate_plan(pc,ctx),validate_plan(pr,ctx)
    assert a['cost_usd']==pytest.approx(b['cost_usd'],abs=1e-10)
    assert len(pc)==len(pr)


def test_plan_cannot_rewrite_history(data):
    ctx = _ctx(data, "camouflage")
    plan = ATTACKS["camouflage"]().plan(ctx)
    assert min(t.timestamp for t in plan) >= ctx.t_start


def test_attacked_snapshot_marks_provenance_only_for_analysis(data):
    ledger, labels, label_time, meta = data
    ctx = _ctx(data, "topology")
    plan = ATTACKS["topology"]().plan(ctx)
    L2 = Ledger(ledger.transactions); L2.append(plan)
    P = build_snapshot(L2, ctx.t_end, labels, label_time, age_cap=ctx.t_end-ctx.t_start)
    assert P.attacker_mask.sum() == len({t.sender for t in plan if t.sender.startswith("atk_")} |
                                        {t.receiver for t in plan if t.receiver.startswith("atk_")})
    # injected addresses never carry a label the attacker set
    assert np.all(P.y[P.attacker_mask] == -1)


# ---- real-data adapter (skipped unless the repository is present) ----
import os
GRAB = os.environ.get("GRABPHISHER_ROOT", "data/eth-phishing-detection-main")


@pytest.mark.skipif(not os.path.exists(GRAB), reason="GrabPhisher repo not present")
def test_grabphisher_adapter_invariants():
    from ledgerpoison.ledger.adapters import load_grabphisher_ego
    L, labels, lt, meta = load_grabphisher_ego(GRAB, variant="ex_1")
    assert meta["features_derivable"] and not meta["synthetic"]
    assert set(labels.values()) <= {0, 1} and meta["n_phishing"] > 0
    # append-only: every tx timestamp within the ledger range, ETH values non-negative
    for tx in L.transactions[:1000]:
        assert tx.value >= 0 and tx.origin.startswith("chain")
