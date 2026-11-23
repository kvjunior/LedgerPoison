"""Regression checks for funding, exact pairing, raw rows and clean retraining."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from ledgerpoison.attacks.base import AttackContext,validate_plan
from ledgerpoison.attacks.strategies import ATTACKS,matched_random_plan
from ledgerpoison.ledger.adapters import synthetic_ledger,load_ethereum_csv
from ledgerpoison.ledger.ledger import Tx
from ledgerpoison.ledger.graph import build_snapshot
from ledgerpoison.ledger.cost import ethereum_default
from ledgerpoison.harness import CellConfig,CleanContext,run_cell,identity
from ledgerpoison.economics.roi import effect_sizes,ci95


def fixture():
    L,labels,lt,meta=synthetic_ledger(n_addresses=200,n_txs=1200,seed=3)
    lo,hi=L.time_range()
    for k in range(12):
        a=f'late_{k}';L.append([Tx(a,'a000001',.001,lo+.70*(hi-lo)),Tx('a000001',a,.001,lo+.71*(hi-lo))]);labels[a]=k%2;lt[a]=lo+.75*(hi-lo)
    meta['input_fingerprint']='test-data-v2'
    cfg=CellConfig(attack='none',split_mode='span',detector='sage',train_cfg={'epochs':2,'patience':2},model_cfg={'hid':8,'layers':2},diagnostic_only=True)
    A=build_snapshot(L,lo+.6*(hi-lo),labels,lt);coal=[a for a,i in A.address_index.items() if A.y[i]==1][:3]
    return L,labels,lt,meta,cfg,coal


def context():
    L,labels,lt,meta,cfg,coal=fixture();A=build_snapshot(L,L.time_range()[0]+.6*(L.time_range()[1]-L.time_range()[0]),labels,lt)
    return AttackContext(A,L.snapshot(A.t_ref),A.y.copy(),coal,ethereum_default(),200,A.t_ref,A.t_ref+86400*5,np.random.default_rng(9))


def test_unregistered_prefix_and_finite_bankroll():
    ctx=context()
    with pytest.raises(AssertionError,match='registered'):validate_plan([Tx('atk_fake_fund','outsider',.001,ctx.t_start,origin='attacker')],ctx)
    fund=ctx.bankroll_address('fund');ctx.initial_balances_eth[fund]=0
    with pytest.raises(AssertionError,match='available funds'):validate_plan([Tx(fund,'outsider',.001,ctx.t_start,origin='attacker')],ctx)


@pytest.mark.parametrize('attack',['camouflage','topology','label_delay'])
def test_exact_paired_control(attack):
    ctx=context();plan=ATTACKS[attack]().plan(ctx);assert plan
    before=validate_plan(plan,ctx);control=matched_random_plan(plan,ctx);after=validate_plan(control,ctx)
    assert before['cost_usd']==pytest.approx(after['cost_usd'],abs=1e-10)
    assert len(plan)==len(control)
    for a,b in zip(plan,control):
        assert (a.sender,a.value,a.timestamp,a.gas_used)==(b.sender,b.value,b.timestamp,b.gas_used)
        if a.receiver in ctx.owned_addresses:assert a.receiver==b.receiver


def test_nonfinite_rejected():
    for value,t in [(float('nan'),0),(1,float('inf'))]:
        with pytest.raises(ValueError):Tx('a','b',value,t)


def test_raw_gas_rows_and_conflicting_labels(tmp_path):
    tx=tmp_path/'tx.csv';labels=tmp_path/'labels.csv'
    pd.DataFrame({'from':['A','B','C'],'to':['A','D','E'],'value':[1,2,3],'timestamp':[1,2,3],'gas_used':[21000,42000,63000]}).to_csv(tx,index=False)
    pd.DataFrame({'address':['B'],'label':[1]}).to_csv(labels,index=False)
    L,*_=load_ethereum_csv(str(tx),str(labels));assert [t.gas_used for t in L.transactions]==[42000,63000];assert L.transactions[0].sender=='b'
    pd.DataFrame({'address':['B','b'],'label':[1,0]}).to_csv(labels,index=False)
    with pytest.raises(ValueError,match='conflicting'):load_ethereum_csv(str(tx),str(labels))


def test_no_attack_matches_clean_retraining_and_withholds_validation():
    L,labels,lt,meta,cfg,coal=fixture();clean=CleanContext(L,labels,lt,meta,ethereum_default(),coal,cfg)
    assert not (clean.trC&clean.vaC).any();assert not (clean.trC&clean.test_C).any()
    r=run_cell(L,labels,lt,meta,ethereum_default(),coal,cfg,clean=clean)
    assert r['predictions']['combined']==r['predictions']['clean_retrained']
    assert r['predictions']['poison_only']==r['predictions']['clean_retrained']
    changed=copy.deepcopy(cfg);changed.train_cfg['epochs']=3
    with pytest.raises(ValueError,match='cached'):run_cell(L,labels,lt,meta,ethereum_default(),coal,changed,clean=clean)
    assert identity(cfg,meta)!=identity(cfg,dict(meta,input_fingerprint='changed'))


def test_small_sample_student_t():
    assert ci95([0,1])>1.96*.5
