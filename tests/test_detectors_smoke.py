"""PyTorch smoke tests — MUST pass on the server before any model number is
used. Run: pytest tests/test_detectors_smoke.py -q  (CPU, ~1-3 min)"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ledgerpoison.detectors.gnn import (Detector, TrainConfig, sample_neighbors,  # noqa: E402
                                        segment_softmax, to_tensors)
from ledgerpoison.harness import CellConfig, run_cell                     # noqa: E402
from ledgerpoison.ledger.adapters import synthetic_ledger                 # noqa: E402
from ledgerpoison.ledger.cost import ethereum_default                     # noqa: E402
from ledgerpoison.ledger.graph import build_snapshot                      # noqa: E402


def test_segment_softmax_sums_to_one():
    idx = torch.tensor([0, 0, 1, 2, 2, 2])
    a = segment_softmax(torch.randn(6), idx, 3)
    s = torch.zeros(3).index_add_(0, idx, a)
    assert torch.allclose(s, torch.ones(3))


def test_sample_neighbors_shapes():
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 1, 1, 2, 2, 0]])
    idx, mask = sample_neighbors(ei, 6, 2, torch.Generator().manual_seed(0))
    assert idx.shape == (6, 2) and mask[1].sum() == 2 and mask[0].sum() == 1 and mask[3].sum() == 0


@pytest.mark.parametrize("kind", ["gcn", "sage", "gat", "time_sage", "provenance", "robust", "rgcn"])
def test_models_train_and_predict(kind):
    ledger, labels, label_time, meta = synthetic_ledger(n_addresses=400, n_txs=2500, seed=2)
    t = ledger.time_range()[1]
    s = build_snapshot(ledger, t, labels, label_time)
    y = s.y
    rng = np.random.default_rng(0)
    known = np.where(y >= 0)[0]; rng.shuffle(known)
    va = np.zeros(s.num_nodes, bool); va[known[: len(known) // 5]] = True
    tr = (y >= 0) & ~va
    det = Detector(kind, s.x.shape[1], seed=0, cfg=TrainConfig(epochs=15, patience=5))
    g = to_tensors(s.x, s.edge_index, s.first_seen, s.t_ref)
    det.fit(g, y, tr, va)
    p = det.predict_proba(g)
    assert p.shape == (s.num_nodes,) and np.all(np.isfinite(p)) and 0 <= p.min() and p.max() <= 1


def test_harness_six_cells_on_synthetic():
    ledger, labels, label_time, meta = synthetic_ledger(n_addresses=600, n_txs=4000, seed=3)
    t_train = ledger.time_range()[0] + 0.6 * (ledger.time_range()[1] - ledger.time_range()[0])
    A = build_snapshot(ledger, t_train, labels, label_time)
    coalition = [a for a, i in A.address_index.items() if A.y[i] == 1][:8]
    from ledgerpoison.ledger.ledger import Tx
    lo,hi=ledger.time_range()
    for k in range(12):
        addr=f"late_{k}"
        ledger.append([Tx(addr,"a000001",.001,lo+.7*(hi-lo)),Tx("a000001",addr,.001,lo+.71*(hi-lo))])
        labels[addr]=k%2;label_time[addr]=lo+.75*(hi-lo)
    cfg = CellConfig(attack="camouflage", budget_usd=30.0, detector="sage", train_cfg={"epochs": 10, "patience": 5}, split_mode="span")
    r = run_cell(ledger, labels, label_time, meta, ethereum_default(), coalition, cfg)
    assert set(r["cells"]) == {"clean_M0", "evasion_only", "clean_retrained", "clean_retrained_on_attacked", "poison_only", "combined"}
    assert r["meta"]["synthetic"] is True and r["plan"]["n_txs"] > 0
    assert r["n_test"] > 0


def test_feature_contract_excludes_unobserved_distribution_statistics():
    from ledgerpoison.ledger.features import FEATURE_NAMES
    assert set(FEATURE_NAMES)=={"n_out","n_in","n_total","in_out_ratio","out_value_sum","in_value_sum","net_value","n_unique_counterparties","counterparty_entropy"}
