"""Protocol v2: fixed holdout identities, withheld validation seeds, clean and
attacked retraining, six scored conditions, finite funded transaction simulation,
paired controls and saved per-address predictions. Reconstructed inputs require
explicit opt-in; timestamp/label/balance assumptions remain visible in metadata.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .attacks.base import AttackContext, Surrogate, validate_plan
from .attacks.strategies import ATTACKS
from .defenses.temporal_consistency import temporal_consistency_filter
from .labels.labeling import PseudoLabeler, visible_labels
from .ledger.cost import FeeModel
from .ledger.features import apply_standardize, standardize
from .ledger.graph import GraphSnapshot, build_snapshot
from .ledger.ledger import Ledger


@dataclass
class CellConfig:
    attack: str = "camouflage"
    attack_params: Dict = field(default_factory=dict)
    budget_usd: float = 100.0
    detector: str = "gat"
    defense: str = "none"               # none | temporal | robust | age_gate | variance_gcn | aggregate_noise
    pseudo_labeler: str = "none"        # none | lp | self_train
    seed: int = 0
    edges_per_buffer: int = 5
    fund_buffers: bool = True
    train_frac_of_span: float = 0.6     # t_train position (fraction of span or tx-time quantile)
    eval_frac_of_span: float = 0.8      # t_eval position
    split_mode: str = "quantile"        # "quantile" of transaction times (default) | "span"
    val_frac: float = 0.2
    train_cfg: Dict = field(default_factory=dict)
    device: str = "cpu"
    forensic_lag_days: float = 30.0     # used ONLY when the dataset has no label_time column
    min_observed_txs: int = 2           # B6: train only on comparable-feature nodes
    allow_reconstructed: bool = False
    diagnostic_only: bool = False
    model_cfg: Dict = field(default_factory=dict)
    bankroll_eth: float = 100.0
    coalition_balance_eth: float = 100.0
    match_control: bool = True          # B3: match the random control to the treatment


def ensure_label_times(ledger: Ledger, labels: Dict[str, int], label_time: Dict[str, float],
                       lag_days: float) -> Dict[str, float]:
    """Datasets without per-address label arrival times get a stated
    assumption: the forensic label of an address arrives `lag_days` after
    its first on-chain activity. This assumption must be reported."""
    if label_time and any(v > 0 for v in label_time.values()):
        return label_time
    fs, idx = ledger.first_seen(), ledger.address_index()
    return {a: float(fs[idx[a]] + lag_days * 86400.0) for a in labels if a in idx}


def _split_times(ledger: Ledger, cfg: CellConfig):
    """Real ledgers are heavily back-loaded in time, so splitting by fraction of the span
    leaves most activity after t_train; the default splits at transaction-time quantiles."""
    if cfg.split_mode == "span":
        lo, hi = ledger.time_range()
        return lo + cfg.train_frac_of_span * (hi - lo), lo + cfg.eval_frac_of_span * (hi - lo)
    ts = ledger.arrays()["timestamp"]
    return float(np.quantile(ts, cfg.train_frac_of_span)), float(np.quantile(ts, cfg.eval_frac_of_span))


def _observed_tx_count(snapshot: GraphSnapshot) -> np.ndarray:
    """Number of transactions each node participates in (degree over the
    directed multigraph). Used to restrict training to comparable-feature
    nodes (B6: ego-sampled neighbours have too few observed txs)."""
    src, dst = snapshot.edge_index
    n = snapshot.num_nodes
    return np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)


def _test_mask(snapshot: GraphSnapshot, t_train: float, t_eval: float,
               coalition_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """B1: the test set is defined by ACTIVITY WINDOW, not by label arrival.

    A node is a test node iff (a) its true label is known to the experimenter
    (regardless of when the *defender* would learn it), and (b) its first
    on-chain activity falls in (t_train, t_eval]. These are the genuinely new
    accounts a production detector must score before forensics catch up, and
    membership is NOT encodable from the capped `age` feature. The attacker's
    own coalition addresses are excluded so the attack cannot score its own
    nodes."""
    first = snapshot.first_seen
    m = (snapshot.y >= 0) & (first > t_train) & (first <= t_eval)
    if coalition_ids is not None and len(coalition_ids):
        m[coalition_ids] = False
    return m


def _train_val_masks(snapshot: GraphSnapshot, t_now: float, cfg: CellConfig, rng: np.random.Generator,
                     labeler: PseudoLabeler, test_mask: np.ndarray,
                     scores: Optional[np.ndarray] = None):
    """Training labels the defender may use at t_now (visible + pseudo),
    split into train/val. Test nodes are held out of training regardless of
    label visibility. Training is restricted to comparable-feature nodes
    (>= cfg.min_observed_txs transactions)."""
    import hashlib
    y_vis = visible_labels(snapshot, t_now).copy()
    y_vis[test_mask] = -1
    comparable = _observed_tx_count(snapshot) >= cfg.min_observed_txs
    va = np.zeros(snapshot.num_nodes, bool)
    for label in (0,1):
        candidates = [(a,i) for a,i in snapshot.address_index.items()
                      if y_vis[i] == label and comparable[i] and not test_mask[i]]
        candidates.sort(key=lambda ai: hashlib.sha256(f"{cfg.seed}:validation:{ai[0]}".encode()).digest())
        nval = min(max(1,int(len(candidates)*cfg.val_frac)),max(0,len(candidates)-1))
        for _,i in candidates[:nval]: va[i] = True
    withheld = y_vis.copy(); withheld[va] = -1
    y = labeler(snapshot, withheld, scores)
    # Restore validation truth only for validation scoring, never as an LP seed.
    y[va] = y_vis[va]
    tr = (y >= 0) & comparable & ~test_mask & ~va
    if not tr.any(): raise ValueError("no eligible training labels")
    return y, tr, va


def _metrics(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum()); tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    auc = None; ap = None
    if len(np.unique(y_true)) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            auc = float(roc_auc_score(y_true, p))
            ap = float(average_precision_score(y_true, p))
        except Exception:
            pass
    return {"f1": f1, "precision": prec, "recall": rec, "auc": auc, "ap": ap,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(len(y_true))}


def _train(kind: str, defense: str, g_np: Dict, y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray,
           cfg: CellConfig, stats):
    from .detectors.gnn import Detector, TrainConfig, to_tensors
    tc = TrainConfig(**cfg.train_cfg)
    model_kind = {"robust": "robust", "age_gate": "provenance", "variance_gcn": "rgcn"}.get(defense, kind)
    if defense == "aggregate_noise":
        tc.gradient_shaping = {"clip": 1.0, "noise": 0.1}
    det = Detector(model_kind, g_np["x"].shape[1], seed=cfg.seed, device=cfg.device, cfg=tc, **cfg.model_cfg)
    g = to_tensors(apply_standardize(g_np["x"], stats), g_np["edge_index"], g_np["first_seen"], g_np["t_ref"], cfg.device)
    det.fit(g, y, train_mask, val_mask)
    return det


def _graph_np(s: GraphSnapshot) -> Dict:
    return {"x": s.x, "edge_index": s.edge_index, "first_seen": s.first_seen, "t_ref": s.t_ref}


def _score(det, s: GraphSnapshot, stats, cfg: CellConfig) -> np.ndarray:
    from .detectors.gnn import to_tensors
    g = to_tensors(apply_standardize(s.x, stats), s.edge_index, s.first_seen, s.t_ref, cfg.device)
    return det.predict_proba(g)


def _threshold(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    """F1-optimal threshold on the validation mask (fixed before test)."""
    m = mask & (y >= 0)
    if m.sum() == 0:
        return 0.5
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f = _metrics(y[m], p[m], t)["f1"]
        if f > best_f:
            best_f, best_t = f, t
    return best_t



class CleanContext:
    """Earlier model plus an equally timed clean retraining counterpart."""
    def __init__(self, ledger, labels, label_time, meta, fee, coalition, cfg):
        if meta.get("representation") == "aggregate_reconstruction" and not cfg.allow_reconstructed:
            raise ValueError("reconstructed data requires explicit allow_reconstructed=True")
        self.ledger,self.labels,self.meta,self.fee,self.coalition,self.cfg=ledger,labels,meta,fee,coalition,cfg
        self.label_time=ensure_label_times(ledger,labels,label_time,cfg.forensic_lag_days)
        self.t_train,self.t_eval=_split_times(ledger,cfg)
        self.age_cap=self.t_eval-self.t_train
        self.labeler=PseudoLabeler(cfg.pseudo_labeler)
        self.A=build_snapshot(ledger,self.t_train,labels,self.label_time)
        self.C=build_snapshot(ledger,self.t_eval,labels,self.label_time)
        self.coalition_ids_A=np.array([self.A.address_index[a] for a in coalition if a in self.A.address_index],dtype=np.int64)
        self.coalition_ids_C=np.array([self.C.address_index[a] for a in coalition if a in self.C.address_index],dtype=np.int64)
        self.test_C=_test_mask(self.C,self.t_train,self.t_eval,self.coalition_ids_C)
        boot=self.labeler if cfg.pseudo_labeler!='self_train' else PseudoLabeler('none')
        self.yA,self.trA,self.vaA=_train_val_masks(self.A,self.t_train,cfg,np.random.default_rng(cfg.seed),boot,np.zeros(self.A.num_nodes,bool))
        _,self.stats=standardize(self.A.x,self.trA)
        self.M0=_train(cfg.detector,'none',_graph_np(self.A),self.yA,self.trA,self.vaA,cfg,self.stats)
        self.thr=_threshold(self.yA,_score(self.M0,self.A,self.stats,cfg),self.vaA)
        self._score_C_M0=_score(self.M0,self.C,self.stats,cfg)
        self.y_public=visible_labels(self.A,self.t_train)
        scores=self._score_C_M0 if cfg.pseudo_labeler=='self_train' else None
        self.yC,self.trC,self.vaC=_train_val_masks(self.C,self.t_eval,cfg,np.random.default_rng(cfg.seed),self.labeler,self.test_C,scores)
        self.trC,self.clean_defense_diag=defend(self.C,self.trC,cfg)
        self.Mc=_train(cfg.detector,cfg.defense,_graph_np(self.C),self.yC,self.trC,self.vaC,cfg,self.stats)
        self.pCC=_score(self.Mc,self.C,self.stats,cfg)
        self.thrC=_threshold(self.yC,self.pCC,self.vaC)

    def make_attack_context(self,budget_usd,stream='plan'):
        import hashlib
        seed=int.from_bytes(hashlib.sha256(f"{self.cfg.seed}:{budget_usd}:{stream}".encode()).digest()[:8],'little')
        return AttackContext(self.A,self.ledger.snapshot(self.t_train),self.y_public,self.coalition,self.fee,
             budget_usd,self.t_train,self.t_eval,np.random.default_rng(seed),
             edges_per_buffer=self.cfg.edges_per_buffer,fund_buffers=self.cfg.fund_buffers,
             surrogate=Surrogate(self.A,self.y_public,self.cfg.seed),
             bankroll_eth=self.cfg.bankroll_eth,coalition_balance_eth=self.cfg.coalition_balance_eth)

def defend(snapshot,mask,cfg):
    if cfg.defense=='temporal':return temporal_consistency_filter(snapshot,mask,snapshot.t_ref)
    return mask,{}

def identity(cfg,meta):
    import hashlib
    payload={"protocol":"ledgerpoison-v2", "config":asdict(cfg),"input_fingerprint":meta.get('input_fingerprint'),
             "representation":meta.get('representation'),"diagnostic_only":cfg.diagnostic_only}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()

def run_cell(ledger,labels,label_time,meta,fee,coalition,cfg,out_dir=None,clean=None):
    if cfg.defense not in ('none','temporal','robust','age_gate','variance_gcn','aggregate_noise'):
        raise ValueError("unsupported defense; use accurate v2 names")
    if clean is None:clean=CleanContext(ledger,labels,label_time,meta,fee,coalition,cfg)
    def base_config(c):
        return {k:v for k,v in asdict(c).items() if k not in ('attack','attack_params','budget_usd')}
    if base_config(clean.cfg)!=base_config(cfg):
        raise ValueError("incompatible cached clean context")
    from .attacks.strategies import matched_random_plan
    reference=cfg.attack_params.get('match_to','camouflage') if cfg.attack=='random' else cfg.attack
    params={k:v for k,v in cfg.attack_params.items() if k!='match_to'}
    ctx=clean.make_attack_context(cfg.budget_usd,f'reference:{reference}:{json.dumps(params,sort_keys=True)}')
    if reference=='none':plan=[]
    else:
        if not meta.get('features_derivable',False):raise ValueError("non-derivable data")
        plan=ATTACKS[reference](**params).plan(ctx)
    if cfg.attack=='random':
        ctx.rng=clean.make_attack_context(cfg.budget_usd,f'control:{reference}:{json.dumps(params,sort_keys=True)}').rng
        plan=matched_random_plan(plan,ctx)
    plan_stats=validate_plan(plan,ctx)
    attacked=Ledger(ledger.transactions);attacked.append(plan)
    P=build_snapshot(attacked,clean.t_eval,labels,clean.label_time)
    # Test identities are fixed from C, independent of adversarial activity.
    test_addresses=sorted(a for a,i in clean.C.address_index.items() if clean.test_C[i])
    testP=np.zeros(P.num_nodes,bool)
    for a in test_addresses:testP[P.address_index[a]]=True
    # Validation eligibility and ground truth are fixed using C, then remapped.
    visible=visible_labels(P,clean.t_eval).copy();visible[testP]=-1
    vaP=np.zeros(P.num_nodes,bool)
    for a,i in clean.C.address_index.items():
        if clean.vaC[i]:vaP[P.address_index[a]]=True
    withheld=visible.copy();withheld[vaP]=-1
    scores=_score(clean.M0,P,clean.stats,cfg) if cfg.pseudo_labeler=='self_train' else None
    yP=clean.labeler(P,withheld,scores);yP[vaP]=visible[vaP]
    trP=(yP>=0)&(_observed_tx_count(P)>=cfg.min_observed_txs)&~testP&~vaP
    trP,diag=defend(P,trP,cfg)
    Mp=_train(cfg.detector,cfg.defense,_graph_np(P),yP,trP,vaP,cfg,clean.stats)
    thrP=_threshold(yP,_score(Mp,P,clean.stats,cfg),vaP)
    iC=np.array([clean.C.address_index[a] for a in test_addresses],dtype=np.int64)
    iP=np.array([P.address_index[a] for a in test_addresses],dtype=np.int64)
    if not len(iC) or len(np.unique(clean.C.y[iC]))<2:raise ValueError("test needs both classes")
    y=clean.C.y[iC]
    probs={'clean_M0':clean._score_C_M0[iC], 'evasion_only':_score(clean.M0,P,clean.stats,cfg)[iP],
           'clean_retrained':clean.pCC[iC], 'clean_retrained_on_attacked':_score(clean.Mc,P,clean.stats,cfg)[iP],
           'poison_only':_score(Mp,clean.C,clean.stats,cfg)[iC], 'combined':_score(Mp,P,clean.stats,cfg)[iP]}
    thresholds={k:clean.thr if k in ('clean_M0','evasion_only') else clean.thrC if k.startswith('clean_retrained') else thrP for k in probs}
    cells={k:_metrics(y,v,thresholds[k]) for k,v in probs.items()}
    public=visible_labels(P,clean.t_eval);pseudo=(public<0)&(yP>=0)
    result={'protocol_version':2,'run_id':identity(cfg,meta),'config':asdict(cfg),'meta':dict(meta,diagnostic_only=bool(cfg.diagnostic_only or meta.get("diagnostic_only"))),
      't_train':clean.t_train,'t_eval':clean.t_eval,'plan':plan_stats,'cells':cells,'defense':diag,
      'control_family':reference,'control_params':params,
      'labels':{'pseudo_labeled':int(pseudo.sum()),'attacker_pseudo_licit':int((pseudo&(yP==0)&P.attacker_mask).sum()),'attacker_nodes':int(P.attacker_mask.sum())},
      'empty_plan':not bool(plan), 'n_test':len(iC),'n_test_illicit':int((y==1).sum()),'test_addresses':test_addresses,'test_y':y.tolist(),
      'predictions':{k:v.tolist() for k,v in probs.items()},'thresholds':thresholds,
      'plan_transactions':[asdict(t) for t in plan], 'timestamp':time.time(),
      'effect_definition':'clean_retrained minus attacked retrained, and paired strategic-minus-random damage',
      'scope':'aggregate-history simulation; no historical EVM execution claim'}
    if out_dir:
        from pathlib import Path
        import torch
        d=Path(out_dir);d.mkdir(parents=True,exist_ok=True);stem=result['run_id']
        result['checkpoint_sha256']={}
        for name,det in [('M0',clean.M0),('Mc',clean.Mc),('Mp',Mp)]:
            checkpoint=d/f'{stem}_{name}.pt'
            torch.save({'state_dict':det.model.state_dict(),'config':asdict(cfg),'stats':clean.stats,'history':det.history},checkpoint)
            import hashlib
            result['checkpoint_sha256'][name]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        tmp=d/f'{stem}.json.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False,default=float));os.replace(tmp,d/f'{stem}.json')
    return result
