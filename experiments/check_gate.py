"""Conservative Gate B: paired t intervals, Bonferroni over all scanned budgets."""
import argparse,json
from pathlib import Path
from scipy.stats import t
from analysis.tables import load_results
from ledgerpoison.economics.roi import effect_sizes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('results');ap.add_argument('--margin',type=float,default=.02);ap.add_argument('--out',required=True);a=ap.parse_args()
    rs=load_results(a.results) # refuses diagnostic and synthetic results
    status=json.loads((Path(a.results)/'status.json').read_text())
    if not status['complete']:raise ValueError('incomplete campaign')
    es=effect_sizes(rs)
    if es.empty:raise ValueError('no paired results')
    groups=es.groupby(['fingerprint','dataset','attack','params','detector','defense','pseudo_labeler'])
    decisions=[]
    for _,g in groups:
        m=len(g)
        for _,r in g.iterrows():
            n=int(r.n_seeds)
            corrected=float(r.effect_ci95*t.ppf(1-.05/(2*m),n-1)/t.ppf(.975,n-1)) if n>=5 else float('inf')
            lower=float(r.effect_mean)-corrected
            decisions.append(dict(attack=r.attack,params=r.params,budget_usd=float(r.budget_usd),n_seeds=n,effect_mean=float(r.effect_mean),corrected_lower=lower if lower!=float('-inf') else None,passes=bool(n>=5 and lower>=a.margin)))
    passed=any(x['passes'] for x in decisions)
    Path(a.out).write_text(json.dumps({'gate_passed':passed,'margin':a.margin,'method':'paired Student t; Bonferroni across scanned budgets within each prespecified treatment family','decisions':decisions},indent=2,allow_nan=False))
    print('PASS' if passed else 'FAIL')
    if not passed:raise SystemExit(2)
if __name__=='__main__': main()
