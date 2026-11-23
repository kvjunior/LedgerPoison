"""Cross-file verification from saved plans, predictions, metrics and tables."""
import argparse,json
from pathlib import Path
import numpy as np
from analysis.tables import load_results,four_cell
from ledgerpoison.harness import _metrics
from ledgerpoison.economics.roi import effect_sizes,roi_table


def verify(root):
    root=Path(root);rs=load_results(root,True);status=json.loads((root/'status.json').read_text())
    assert status['complete'] and len(rs)==status['expected_cells']
    pairs={}
    for r in rs:
        y=np.array(r['test_y']);assert len(y)==r['n_test']==len(r['test_addresses'])
        assert r['plan']['cost_usd']<=r['config']['budget_usd']+1e-8
        assert r['plan']['n_txs']==len(r['plan_transactions'])
        assert r['plan']['neg_balance_addrs']==0
        assert r['meta']['input_fingerprint']==status['fingerprint']
        for name,p in r['predictions'].items():
            p=np.array(p);assert len(p)==len(y) and np.isfinite(p).all()
            metrics=_metrics(y,p,r['thresholds'][name])
            for k,v in metrics.items():assert abs(v-r['cells'][name][k])<1e-10
        import hashlib
        for name in ('M0','Mc','Mp'):
            checkpoint=root/f'{r["run_id"]}_{name}.pt'
            assert checkpoint.exists()
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256'][name]
        if r['config']['attack']=='none':assert r['predictions']['combined']==r['predictions']['clean_retrained']
        else:assert r['plan']['n_txs']>0,'integration must exercise nonempty injections'
        key=(r['control_family'],json.dumps(r['control_params'],sort_keys=True),r['config']['detector'],r['config']['defense'],r['config']['pseudo_labeler'],r['config']['seed'],r['config']['budget_usd'])
        pairs.setdefault(key,{})[r['config']['attack']]=r
    for key,g in pairs.items():
        if key[0]=='none':continue
        a=g[key[0]];b=g['random'];assert a['test_addresses']==b['test_addresses']
        owned=set(a['plan']['initial_balances_eth'])|set(a['plan']['ending_owned_balances_eth'])
        for x,z in zip(a['plan_transactions'],b['plan_transactions']):
            for k in ('sender','value','timestamp','gas_used'):assert x[k]==z[k]
            if x['receiver'] in owned:assert x['receiver']==z['receiver']
    tables={'six_cell_rows':len(four_cell(rs)),'paired_effect_rows':len(effect_sizes(rs)),'scenario_rows':len(roi_table(rs))}
    return dict(completed_cells=len(rs),nonempty_cells=sum(r['plan']['n_txs']>0 for r in rs),fixed_holdout_accounts=sorted(set(r['n_test'] for r in rs)),tables=tables,verified=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('results');ap.add_argument('--out');a=ap.parse_args();report=verify(a.results);print(json.dumps(report,indent=2))
    if a.out:Path(a.out).write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
