"""Fingerprint-checked, paired experiment runner. All bounded runs are diagnostics."""
from __future__ import annotations
import argparse, hashlib, itertools, json, platform, sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import yaml
from ledgerpoison.harness import CellConfig, CleanContext, run_cell, identity, _split_times
from ledgerpoison.ledger.adapters import load_ethereum_csv, synthetic_ledger, load_grabphisher_ego
from ledgerpoison.ledger.cost import ethereum_default, ethereum_historical
from ledgerpoison.ledger.graph import build_snapshot


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def atomic(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)


def load_dataset(spec,synthetic=False):
    if synthetic or spec['kind']=='synthetic': return synthetic_ledger(seed=int(spec.get('seed',0)))
    if spec['kind']=='ethereum_csv': return load_ethereum_csv(spec['tx_csv'],spec['label_csv'])
    if spec['kind']=='grabphisher_ego':
        kw={k:spec[k] for k in ('variant','expand','strict_labels','label_time_policy','lag_days','t_min','t_max') if k in spec}
        return load_grabphisher_ego(spec['root'],**kw)
    raise ValueError('unsupported dataset kind')


def select_coalition(ledger,labels,label_time,size,select,seed,t_train):
    snap=build_snapshot(ledger,t_train,labels,label_time)
    fraud=sorted(a for a,i in snap.address_index.items() if snap.y[i]==1)
    if not fraud: raise ValueError('no active fraud coalition')
    if select=='random': return list(np.random.default_rng(seed).choice(fraud,min(size,len(fraud)),replace=False))
    if select!='top_indegree': raise ValueError('unknown coalition selection')
    deg=np.bincount(snap.edge_index[1],minlength=snap.num_nodes)
    fraud.sort(key=lambda a:(-deg[snap.address_index[a]],a))
    return fraud[:size]


def fingerprint(spec):
    root=Path(__file__).resolve().parents[1]
    files={str(f.relative_to(root)):sha(f) for folder in ('ledgerpoison','experiments','analysis') for f in sorted((root/folder).rglob('*.py'))}
    inputs={}
    def visit(v):
        if isinstance(v,dict):
            for x in v.values(): visit(x)
        elif isinstance(v,str) and Path(v).exists():
            x=Path(v)
            for f in ([x] if x.is_file() else sorted(x.rglob('*'))):
                if f.is_file() and f.suffix in ('.pkl','.pickle','.csv','.txt','.json'): inputs[str(f)]=sha(f)
    visit(spec['dataset']);visit(spec['fee'])
    import torch, scipy, sklearn, pandas, networkx
    versions={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pandas.__version__,'networkx':networkx.__version__}
    data={'sources':files,'inputs':inputs,'environment':versions,'settings':{k:v for k,v in spec.items() if k!='out_dir'}}
    data['fingerprint']=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    return data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('config');ap.add_argument('--data-root');ap.add_argument('--out');ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=2);ap.add_argument('--synthetic',action='store_true');ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    config=Path(args.config).resolve();spec=yaml.safe_load(config.read_text())
    data_root=Path(args.data_root).resolve() if args.data_root else config.parent.parent/'data'
    for section in ('dataset','fee'):
        node=spec[section].get('historical',spec[section])
        for k in ('root','tx_csv','label_csv','gas_csv','eth_usd_csv'):
            if k in node and not Path(node[k]).is_absolute(): node[k]=str(data_root/node[k])
    out=Path(args.out or spec.get('out_dir','results/run'));out.mkdir(parents=True,exist_ok=True)
    manifest=fingerprint(spec);atomic(out/'manifest.json',manifest)
    ledger,labels,label_time,meta=load_dataset(spec['dataset'],args.synthetic)
    meta['input_fingerprint']=manifest['fingerprint'];meta['diagnostic_only']=bool(args.synthetic or spec['dataset'].get('t_min') is not None or spec['dataset'].get('t_max') is not None or spec.get('diagnostic_only'))
    h=spec['fee'].get('historical')
    if spec['fee']['chain']!='ethereum': raise ValueError('this revision supports Ethereum account simulation only')
    fee=ethereum_historical(h['gas_csv'],h.get('eth_usd_csv'),h.get('eth_usd_constant')) if h else ethereum_default(spec['fee']['gas_gwei'],spec['fee']['fiat_usd'])
    defaults=dict(spec.get('cell',{}));defaults.update(device=args.device,diagnostic_only=meta['diagnostic_only'])
    g=spec['grid'];cells=[]
    for atk,budget,det,dfn,lab,seed in itertools.product(g['attack'],g['budget_usd'],g['detector'],g['defense'],g['pseudo_labeler'],g['seeds']):
        if atk=='random': continue # generated separately for each treatment
        for params in g.get('attack_params',{}).get(atk,[{}]):
            base=dict(defaults,attack=atk,attack_params=params,budget_usd=float(budget),detector=det,defense=dfn,pseudo_labeler=lab,seed=int(seed))
            cells.append(CellConfig(**base))
            if atk!='none': cells.append(CellConfig(**dict(base,attack='random',attack_params=dict(params,match_to=atk))))
    cells.sort(key=lambda c:(c.seed,c.detector,c.defense,c.pseudo_labeler,c.budget_usd,c.attack,json.dumps(c.attack_params,sort_keys=True)))
    print(f'{len(cells)} cells; diagnostic_only={meta["diagnostic_only"]}; {len(ledger)} records',flush=True)
    if args.dry_run: return
    t_train,_=_split_times(ledger,CellConfig(**defaults));coalitions={};cache_key=None;clean=None;completed=0;skipped=0
    for c in cells:
        rid=identity(c,meta);path=out/f'{rid}.json'
        if path.exists():
            r=json.loads(path.read_text())
            if r.get('run_id')==rid and len(r.get('predictions',{}))==6 and all((out/f'{rid}_{name}.pt').exists() and sha(out/f'{rid}_{name}.pt')==r.get('checkpoint_sha256',{}).get(name) for name in ('M0','Mc','Mp')):
                skipped+=1;continue
        try:
            if c.seed not in coalitions:
                coalitions[c.seed]=select_coalition(ledger,labels,label_time,**spec['coalition'],seed=c.seed,t_train=t_train)
                atomic(out/f'coalition_s{c.seed}.json',coalitions[c.seed])
            key=(c.seed,c.detector,c.defense,c.pseudo_labeler)
            if key!=cache_key:
                clean=None
                clean=CleanContext(ledger,labels,label_time,meta,fee,coalitions[c.seed],c);cache_key=key
            r=run_cell(ledger,labels,label_time,meta,fee,coalitions[c.seed],c,str(out),clean=clean)
            completed+=1
            print(f'{completed}: {c.attack}/{r["control_family"]} {c.detector} {c.defense} {c.pseudo_labeler} n={r["n_test"]} tx={r["plan"]["n_txs"]} cost={r["plan"]["cost_usd"]:.4f}',flush=True)
        except Exception as exc:
            atomic(out/f'{rid}.failure.json',{'run_id':rid,'config':asdict(c),'error':repr(exc)});raise
    atomic(out/'status.json',{'expected_cells':len(cells),'completed_this_run':completed,'skipped_verified':skipped,'complete':completed+skipped==len(cells),'fingerprint':manifest['fingerprint']})

if __name__=='__main__': main()
