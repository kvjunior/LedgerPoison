"""
Detectors in plain PyTorch (no torch_geometric dependency, so the server
environment is easy to reproduce). Message passing uses index_add_ and
scatter_reduce, both in torch >= 1.12.

Models
  GCN, SAGE, GAT              — the three canonical detectors
  TimeSAGE                    — SAGE + sinusoidal encoding of node first-seen time
                                (temporal baseline; a TGN-class model can be added
                                behind the same interface)
  ProvenanceGNN (defense)     — attention GNN with an age gate on incoming messages
  RobustGNN (defense)         — sampled-neighbourhood coordinate-wise trimmed mean
  RGCN (baseline)             — Gaussian-based robust GCN (Zhu et al., KDD 2019), simplified
All expose forward(x, edge_index, node_time=None) -> logits and get_embeddings.

Training uses full-batch AdamW and early stopping. Aggregate gradient clipping
and Gaussian noise is an experimental regularizer, not per-example DP-SGD,
not an implementation of Hong et al., and not a privacy guarantee.
The variance-gated GCN is a custom simplified baseline, not a faithful published
RGCN reproduction. The age gate uses observed/proxy age, not trusted provenance.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------- utils
def add_self_loops(edge_index: Tensor, n: int) -> Tensor:
    loops = torch.arange(n, device=edge_index.device)
    return torch.cat([edge_index, torch.stack([loops, loops])], dim=1)


def scatter_sum(msg: Tensor, index: Tensor, n: int) -> Tensor:
    out = torch.zeros(n, msg.size(1), device=msg.device, dtype=msg.dtype)
    return out.index_add_(0, index, msg)


def scatter_mean(msg: Tensor, index: Tensor, n: int) -> Tensor:
    s = scatter_sum(msg, index, n)
    cnt = torch.zeros(n, device=msg.device, dtype=msg.dtype).index_add_(0, index, torch.ones_like(index, dtype=msg.dtype))
    return s / cnt.clamp(min=1).unsqueeze(1)


def segment_softmax(logits: Tensor, index: Tensor, n: int) -> Tensor:
    mx = torch.full((n,), -1e30, device=logits.device, dtype=logits.dtype)
    mx = mx.scatter_reduce(0, index, logits, reduce="amax", include_self=True)
    ex = torch.exp(logits - mx[index])
    den = torch.zeros(n, device=logits.device, dtype=logits.dtype).index_add_(0, index, ex)
    return ex / den[index].clamp(min=1e-16)


def time_encoding(t: Tensor, dim: int = 16, max_period: float = 365 * 86400.0) -> Tensor:
    """Sinusoidal encoding of a time tensor (seconds)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    ang = t.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


def sample_neighbors(edge_index: Tensor, n: int, k: int, gen: torch.Generator) -> Tuple[Tensor, Tensor]:
    """Fixed-size sampled in-neighbourhood: returns idx [n, k] and mask [n, k]."""
    src, dst = edge_index
    perm = torch.randperm(src.numel(), generator=gen, device="cpu").to(src.device)
    src, dst = src[perm], dst[perm]
    order = torch.argsort(dst, stable=True)
    src, dst = src[order], dst[order]
    counts = torch.bincount(dst, minlength=n)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(src.numel(), device=src.device) - starts[dst]
    keep = pos < k
    idx = torch.zeros(n, k, dtype=torch.long, device=src.device)
    mask = torch.zeros(n, k, dtype=torch.bool, device=src.device)
    idx[dst[keep], pos[keep]] = src[keep]
    mask[dst[keep], pos[keep]] = True
    return idx, mask


# -------------------------------------------------------------------- layers
class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.lin = nn.Linear(i, o)

    def forward(self, x, edge_index):
        # Standard symmetric GCN normalisation D^-1/2 (A+I) D^-1/2 on the graph
        # made undirected (A + A^T); self-loops added once. (Previously the
        # degree double-counted in+out including self-loops; see review S3.)
        n = x.size(0)
        src0, dst0 = edge_index
        src = torch.cat([src0, dst0, torch.arange(n, device=x.device)])
        dst = torch.cat([dst0, src0, torch.arange(n, device=x.device)])
        deg = torch.bincount(dst, minlength=n).float().clamp(min=1)
        norm = (deg[src] * deg[dst]).rsqrt()
        h = self.lin(x)
        return scatter_sum(h[src] * norm.unsqueeze(1), dst, n)


class SAGELayer(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.lin_self = nn.Linear(i, o); self.lin_nb = nn.Linear(i, o)

    def forward(self, x, edge_index):
        src, dst = edge_index
        return self.lin_self(x) + self.lin_nb(scatter_mean(x[src], dst, x.size(0)))


class GATLayer(nn.Module):
    def __init__(self, i, o, heads=4, gate: bool = False):
        super().__init__()
        self.h, self.o = heads, o
        self.lin = nn.Linear(i, o * heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, o) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, o) * 0.1)
        self.gate = gate
        if gate:  # provenance age gate: sigmoid((age - a0)/s), learnable a0, s
            self.a0 = nn.Parameter(torch.tensor(7.0 * 86400.0))
            self.log_s = nn.Parameter(torch.tensor(math.log(3.0 * 86400.0)))
        self.out = nn.Linear(o * heads, o)

    def forward(self, x, edge_index, age=None):
        n = x.size(0)
        ei = add_self_loops(edge_index, n)
        src, dst = ei
        h = self.lin(x).view(n, self.h, self.o)
        e = (h[src] * self.a_src).sum(-1) + (h[dst] * self.a_dst).sum(-1)      # [E, H]
        e = F.leaky_relu(e, 0.2)
        alpha = torch.stack([segment_softmax(e[:, k], dst, n) for k in range(self.h)], 1)
        if self.gate and age is not None:
            g = torch.sigmoid((age[src] - self.a0) / torch.exp(self.log_s))
            g = torch.where(src == dst, torch.ones_like(g), g)                   # self message ungated
            alpha = alpha * g.unsqueeze(1)
        msg = (alpha.unsqueeze(-1) * h[src]).reshape(-1, self.h * self.o)
        return self.out(scatter_sum(msg, dst, n))


class TrimmedMeanLayer(nn.Module):
    def __init__(self, i, o, k=32, trim=0.1, seed=0):
        super().__init__(); self.lin_self = nn.Linear(i, o); self.lin_nb = nn.Linear(i, o)
        self.k, self.trim = k, trim
        self.seed = seed
        self.gen = torch.Generator().manual_seed(seed)

    def forward(self, x, edge_index):
        n = x.size(0)
        idx, mask = sample_neighbors(edge_index, n, self.k, torch.Generator().manual_seed(self.seed))
        nb = x[idx]                                                # [n, k, d]
        big = torch.finfo(nb.dtype).max
        # coordinate-wise trimmed mean over valid neighbours
        cnt = mask.sum(1)                                          # [n]
        t = (cnt.float() * self.trim).floor().long()               # trim each side
        sorted_vals, _ = torch.where(mask.unsqueeze(-1), nb, torch.full_like(nb, big)).sort(dim=1)
        rank = torch.arange(self.k, device=x.device).view(1, -1, 1)
        lo = t.view(-1, 1, 1); hi = (cnt - t).view(-1, 1, 1)
        keep = (rank >= lo) & (rank < hi)
        agg = (sorted_vals * keep).sum(1) / keep.sum(1).clamp(min=1)
        agg = torch.where(cnt.unsqueeze(1) > 0, agg, torch.zeros_like(agg))
        return self.lin_self(x) + self.lin_nb(agg)


# -------------------------------------------------------------------- models
class GNN(nn.Module):
    """Generic stack: kind in {gcn, sage, gat, time_sage, provenance, robust}."""

    def __init__(self, kind: str, in_dim: int, hid: int = 128, out: int = 2, layers: int = 3,
                 dropout: float = 0.3, heads: int = 4, time_dim: int = 16, trim: float = 0.1, seed: int = 0):
        super().__init__()
        self.kind, self.dropout, self.time_dim = kind, dropout, time_dim
        d_in = in_dim + (time_dim if kind == "time_sage" else 0)
        dims = [d_in] + [hid] * (layers - 1) + [hid]
        mk = {"gcn": lambda i, o: GCNLayer(i, o), "sage": lambda i, o: SAGELayer(i, o),
              "time_sage": lambda i, o: SAGELayer(i, o),
              "gat": lambda i, o: GATLayer(i, o, heads), "provenance": lambda i, o: GATLayer(i, o, heads, gate=True),
              "robust": lambda i, o: TrimmedMeanLayer(i, o, trim=trim, seed=seed)}[kind]
        self.layers = nn.ModuleList([mk(dims[i], dims[i + 1]) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.head = nn.Linear(hid, out)

    def get_embeddings(self, x, edge_index, node_time=None, age=None):
        # time_sage adds a time encoding of each node's first-seen time. Because
        # absolute recency reveals cohort membership, which is not label leakage, the provenance DEFENSE
        # uses age only as a message GATE (in GATLayer), never as a node feature;
        # time_sage is kept as an explicit *ablation* showing the leak, and the
        # harness does not use it for headline results. See README.
        if self.kind == "time_sage":
            t = node_time if node_time is not None else torch.zeros(x.size(0), device=x.device)
            x = torch.cat([x, time_encoding(t - t.min(), self.time_dim)], dim=1)
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, age) if self.kind == "provenance" else layer(h, edge_index)
            h = F.dropout(F.gelu(norm(h)), self.dropout, self.training)
        return h

    def forward(self, x, edge_index, node_time=None, age=None):
        return self.head(self.get_embeddings(x, edge_index, node_time, age))


class RGCN(nn.Module):
    """Gaussian-based Robust GCN (Zhu et al. 2019), simplified: hidden means and
    variances are propagated with GCN normalisation and messages are attenuated
    by exp(-gamma * variance)."""

    def __init__(self, in_dim, hid=128, out=2, layers=3, dropout=0.3, gamma=1.0, seed=0):
        super().__init__()
        self.dropout, self.gamma = dropout, gamma
        self.mean0, self.var0 = nn.Linear(in_dim, hid), nn.Linear(in_dim, hid)
        self.mean = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers - 1)])
        self.var = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers - 1)])
        self.head = nn.Linear(hid, out)

    def _prop(self, m, v, edge_index):
        n = m.size(0)
        ei = add_self_loops(edge_index, n); src, dst = ei
        deg = torch.bincount(dst, minlength=n).float().clamp(min=1)
        att = torch.exp(-self.gamma * v[src])
        norm1 = (deg[src] * deg[dst]).rsqrt().unsqueeze(1)
        norm2 = (1.0 / (deg[src] * deg[dst])).unsqueeze(1)
        return scatter_sum(m[src] * att * norm1, dst, n), scatter_sum(v[src] * att * att * norm2, dst, n)

    def get_embeddings(self, x, edge_index, node_time=None, age=None):
        m, v = F.elu(self.mean0(x)), F.relu(self.var0(x))
        m, v = self._prop(m, v, edge_index)
        for lm, lv in zip(self.mean, self.var):
            m, v = F.elu(lm(m)), F.relu(lv(v))
            m, v = self._prop(m, v, edge_index)
            m = F.dropout(m, self.dropout, self.training)
        if self.training:
            m = m + torch.randn_like(m) * torch.sqrt(v + 1e-8)
        return m

    def forward(self, x, edge_index, node_time=None, age=None):
        return self.head(self.get_embeddings(x, edge_index, node_time, age))


def build_model(kind: str, in_dim: int, seed: int = 0, **kw) -> nn.Module:
    torch.manual_seed(seed)
    if kind == "rgcn":
        return RGCN(in_dim, seed=seed, **{k: v for k, v in kw.items() if k in ("hid", "out", "layers", "dropout", "gamma")})
    return GNN(kind, in_dim, seed=seed, **kw)


# ------------------------------------------------------------------- trainer
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 40
    max_illicit_weight: float = 10.0
    gradient_shaping: Optional[Dict[str, float]] = None   # {"clip": 1.0, "noise": 0.1}


@dataclass
class GraphTensors:
    x: Tensor; edge_index: Tensor; node_time: Tensor; age: Tensor


def to_tensors(x: np.ndarray, edge_index: np.ndarray, first_seen: np.ndarray, t_ref: float,
               device: str = "cpu") -> GraphTensors:
    return GraphTensors(
        x=torch.as_tensor(x, dtype=torch.float32, device=device),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long, device=device),
        node_time=torch.as_tensor(first_seen, dtype=torch.float32, device=device),
        age=torch.as_tensor(np.maximum(t_ref - first_seen, 0.0), dtype=torch.float32, device=device),
    )


class Detector:
    def __init__(self, kind: str, in_dim: int, seed: int = 0, device: str = "cpu", cfg: Optional[TrainConfig] = None, **model_kw):
        self.kind, self.seed, self.device = kind, seed, device
        self.cfg = cfg or TrainConfig()
        self.model = build_model(kind, in_dim, seed=seed, **model_kw).to(device)
        self.history = []

    def _forward(self, g: GraphTensors) -> Tensor:
        return self.model(g.x, g.edge_index, g.node_time, g.age)

    def fit(self, g: GraphTensors, y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray) -> Dict[str, float]:
        if not train_mask.any() or not np.isfinite(g.x.cpu().numpy()).all():
            raise ValueError("finite inputs and nonempty training set required")
        torch.manual_seed(self.seed)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        tr = torch.as_tensor(train_mask & (y >= 0), device=self.device)
        va = torch.as_tensor(val_mask & (y >= 0), device=self.device)
        n0, n1 = int((y[train_mask] == 0).sum()), int((y[train_mask] == 1).sum())
        w = torch.tensor([1.0, min(n0 / max(n1, 1), self.cfg.max_illicit_weight)], device=self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        best, best_state, wait = float("inf"), None, 0
        for ep in range(self.cfg.epochs):
            self.model.train(); opt.zero_grad()
            out = self._forward(g)
            loss = F.cross_entropy(out[tr], y_t[tr], weight=w)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
            loss.backward()
            gs = self.cfg.gradient_shaping
            if gs:  # aggregate-gradient regularizer; no per-example clipping or privacy claim
                nn.utils.clip_grad_norm_(self.model.parameters(), gs["clip"])
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * gs["noise"] * gs["clip"]
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(self._forward(g)[va], y_t[va]).item() if va.any() else loss.item()
            self.history.append({"epoch": ep, "train_loss": loss.item(), "val_loss": vl})
            if vl < best - 1e-5:
                best, best_state, wait = vl, copy.deepcopy(self.model.state_dict()), 0
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_val_loss": best, "epochs": len(self.history)}

    @torch.no_grad()
    def predict_proba(self, g: GraphTensors) -> np.ndarray:
        self.model.eval()
        return F.softmax(self._forward(g), dim=1)[:, 1].cpu().numpy()
