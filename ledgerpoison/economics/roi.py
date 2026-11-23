"""Paired, parameter-specific effects and explicitly hypothetical benefit scenarios."""
import json
import numpy as np
import pandas as pd
from scipy.stats import t


def ci95(values,alpha=.05):
    a=np.asarray(values,float)
    return float(t.ppf(1-alpha/2,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else np.nan


def effect_sizes(results,cell='combined'):
    pairs={};rows=[]
    for r in results:
        c=r['config']
        key=(r['meta']['input_fingerprint'],r['meta']['name'],r['control_family'],json.dumps(r['control_params'],sort_keys=True),c['budget_usd'],c['detector'],c['defense'],c['pseudo_labeler'],c['seed'])
        arm='random' if c['attack']=='random' else 'treatment'
        if arm in pairs.setdefault(key,{}): raise ValueError('duplicate seed/parameter arm')
        pairs[key][arm]=r
    for key,pair in pairs.items():
        if set(pair)!= {'random','treatment'}: continue
        a,b=pair['treatment'],pair['random']
        if a['test_addresses']!=b['test_addresses'] or a['test_y']!=b['test_y']: raise ValueError('unmatched holdouts')
        if abs(a['plan']['cost_usd']-b['plan']['cost_usd'])>1e-8: raise ValueError('unmatched costs')
        if a['plan']['n_txs']!=b['plan']['n_txs']: raise ValueError('unmatched tx counts')
        if abs(a['cells']['clean_retrained']['f1']-b['cells']['clean_retrained']['f1'])>1e-10: raise ValueError('unmatched clean models')
        rows.append(dict(zip(('fingerprint','dataset','attack','params','budget_usd','detector','defense','pseudo_labeler','seed'),key),effect=b['cells'][cell]['f1']-a['cells'][cell]['f1']))
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows);keys=[k for k in rows[0] if k not in ('seed','effect')]
    return df.groupby(keys).agg(n_seeds=('seed','nunique'),effect_mean=('effect','mean'),effect_ci95=('effect',ci95)).reset_index()


def roi_table(results,value_grid=(500,1000,5000,20000),gas_multipliers=(.5,1.,2.,5.),attacked_cell='combined'):
    rows=[]
    for r in results:
        y=np.array(r['test_y']);pc=np.array(r['predictions']['clean_retrained']);pa=np.array(r['predictions'][attacked_cell])
        fl=int(((y==1)&(pc>=r['thresholds']['clean_retrained'])&(pa<r['thresholds'][attacked_cell])).sum())
        for value in value_grid:
            for gm in gas_multipliers:
                cost=r['plan']['fee_usd']*gm+r['plan']['lost_principal_usd']
                rows.append(dict(dataset=r['meta']['name'],attack=r['config']['attack'],params=json.dumps(r['control_params'],sort_keys=True),detector=r['config']['detector'],defense=r['config']['defense'],labeler=r['config']['pseudo_labeler'],seed=r['config']['seed'],budget_usd=r['config']['budget_usd'],flipped=fl,gas_multiplier=gm,cost_usd=cost,assumed_value_per_holdout=value,hypothetical_benefit_usd=fl*value,hypothetical_roi=(fl*value-cost)/cost if cost>0 else np.nan,scope='sensitivity scenario on non-owned holdouts; no realized profit'))
    return pd.DataFrame(rows)
