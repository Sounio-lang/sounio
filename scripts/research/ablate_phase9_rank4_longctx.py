#!/usr/bin/env python3
"""Phase 9 falsification suite: Rank-4 & long-context (T=128), 6 models x 10 seeds.

Models (d_state=16, shared MLP encoder din->enc_h->tanh->NOUT, enc_h tuned for +-2% trainable-param parity):
 SedenionGatedDeltaSSM, SedenionDeltaSSM, GatedDeltaNet4x4, DeltaProductRank4, GrassmannQR4SSM, GatedDeltaNet16.
No op-specific branches inside any model. Algebra (cd_mul, T16, z_embed, P4) and KVReadout are the Phase-8 definitions.
Raw numbers and paired tests are reported whatever the outcome.
"""
import json, math, os, sys, time
NTHREADS = int(os.environ.get("P9_THREADS", "12"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, str(NTHREADS))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
torch.set_num_threads(NTHREADS)
DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

OUT = os.environ.get("P9_OUT", "/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase9_rank4_longctx_report.json")
SEEDS = [42, 137, 271, 314, 577, 777, 1024, 1618, 2026, 4096]
MODELS = ["SedenionGatedDeltaSSM", "SedenionDeltaSSM", "GatedDeltaNet4x4",
          "DeltaProductRank4", "GrassmannQR4SSM", "GatedDeltaNet16"]
SED = ["SedenionGatedDeltaSSM", "SedenionDeltaSSM"]
DIN = 16
REF_ENC_H = 96
STEPS, BATCH, LR = int(os.environ.get("P9_STEPS", 600)), 256, 3e-3
T_LEN = int(os.environ.get("P9_T", 128))
NTRAIN, NTEST = int(os.environ.get("P9_NTRAIN", 4096)), 1024

# ----------------------------------------------------------------- algebra (verbatim from Phase 8)
def _conj(x):
    y = -x.copy(); y[0] = x[0]; return y

def cd_mul(a, b):
    n = len(a)
    if n == 1:
        return a * b
    h = n // 2
    alo, ahi, blo, bhi = a[:h], a[h:], b[:h], b[h:]
    lo = cd_mul(alo, blo) - cd_mul(_conj(bhi), ahi)
    hi = cd_mul(bhi, alo) + cd_mul(ahi, _conj(blo))
    return np.concatenate([lo, hi])

def structure_tensor(n):
    T = np.zeros((n, n, n)); E = np.eye(n)
    for i in range(n):
        for j in range(n):
            T[i, j] = cd_mul(E[i], E[j])
    return torch.tensor(T, dtype=torch.float32)

T16 = structure_tensor(16)

def left_mat(a, T):
    return torch.einsum("...i,ijk->...kj", a, T.to(a.dtype))

# T16 entries are signed units: for each (k,j) exactly one i has T[i,j,k] != 0 -> gather instead of einsum (0.0 diff)
IDX_KJ = T16.abs().argmax(dim=0).T.contiguous()
SGN_KJ = T16.sum(dim=0).T.contiguous()

def left_mat_fast(z):
    return z[..., IDX_KJ.to(z.device)] * SGN_KJ.to(device=z.device, dtype=z.dtype)

def unit(x, eps=1e-8):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)

def z_embed(u, w):
    u = unit(u)
    w = w - (w * u).sum(-1, keepdim=True) * u
    w = unit(w)
    zeros = torch.zeros_like(u[..., :1])
    return torch.cat([zeros, u, zeros, w], -1) / math.sqrt(2.0)

def P4(z):
    L = left_mat_fast(z)
    L2 = L @ L
    I = torch.eye(16, dtype=z.dtype, device=z.device)
    return I + 0.5 * L2 @ (3 * I + L2)

# ------------------------------------------------------------------ models
NOUT = {"SedenionGatedDeltaSSM": 7 + 7 + 16 + 2, "SedenionDeltaSSM": 7 + 7 + 16 + 1,
        "GatedDeltaNet4x4": 4 * (4 + 1 + 1 + 4), "DeltaProductRank4": 1 + 16 + 4 * (16 + 1),
        "GrassmannQR4SSM": 64 + 16 + 2, "GatedDeltaNet16": 16 + 2 + 16}

def mgs4(K):
    """Modified Gram-Schmidt on the 4 columns of K:(...,16,4) -> Q:(...,16,4), orthonormal, same span as QR."""
    cols = []
    for c in range(4):
        v = K[..., c]
        for q in cols:
            v = v - (q * v).sum(-1, keepdim=True) * q
        cols.append(unit(v))
    return torch.stack(cols, -1)

def recur(n, p):
    """p:(*lead,T,NOUT) -> h_T (*lead,16). Encoder/projector/orthonormalisation precomputed over (*lead,T,...) before the t-loop.
    Identical code is used by the single-model path and the batched ensemble."""
    sg = torch.sigmoid
    T = p.shape[-2]
    lead = p.shape[:-2]
    h = p.new_zeros(*lead, 16)
    ein = lambda P, d: torch.einsum("...ij,...j->...i", P, d)
    SB = math.prod(lead)
    if n == "SedenionGatedDeltaSSM":
        z = z_embed(p[..., 0:7], p[..., 7:14]); v = p[..., 14:30]
        al = sg(p[..., 30:31]); be = sg(p[..., 31:32]); P = P4(z)
        Pt = P.reshape(SB, T, 16, 16).unbind(1); vt = v.reshape(SB, T, 16).unbind(1)
        alt = al.reshape(SB, T, 1).unbind(1); bet = be.reshape(SB, T, 1).unbind(1)
        h = h.reshape(SB, 16)
        for t in range(T):
            ht = alt[t] * h
            h = ht + bet[t] * torch.bmm(Pt[t], (vt[t] - ht).unsqueeze(-1)).squeeze(-1)
        h = h.reshape(*lead, 16)
    elif n == "SedenionDeltaSSM":
        z = z_embed(p[..., 0:7], p[..., 7:14]); v = p[..., 14:30]; be = sg(p[..., 30:31]); P = P4(z)
        Pt = P.reshape(SB, T, 16, 16).unbind(1); vt = v.reshape(SB, T, 16).unbind(1)
        bet = be.reshape(SB, T, 1).unbind(1)
        h = h.reshape(SB, 16)
        for t in range(T):
            h = h + bet[t] * torch.bmm(Pt[t], (vt[t] - h).unsqueeze(-1)).squeeze(-1)
        h = h.reshape(*lead, 16)
    elif n == "GatedDeltaNet4x4":
        q = p.reshape(*lead, T, 4, 10)
        k = unit(q[..., 0:4]); al = sg(q[..., 4:5]); be = sg(q[..., 5:6]); v = q[..., 6:10]
        h = h.reshape(*lead, 4, 4)
        for t in range(T):
            ht = al[..., t, :, :] * h
            kt = k[..., t, :, :]
            h = ht + be[..., t, :, :] * kt * (kt * (v[..., t, :, :] - ht)).sum(-1, keepdim=True)
        h = h.reshape(*lead, 16)
    elif n == "DeltaProductRank4":
        al = sg(p[..., 0:1]); v = p[..., 1:17]
        kb = p[..., 17:].reshape(*lead, T, 4, 17)
        k = unit(kb[..., 0:16]); be = sg(kb[..., 16:17])
        for t in range(T):
            h = al[..., t, :] * h
            for r in range(4):
                kr = k[..., t, r, :]
                h = h + be[..., t, r, :] * kr * (kr * (v[..., t, :] - h)).sum(-1, keepdim=True)
    elif n == "GrassmannQR4SSM":
        Q = mgs4(p[..., 0:64].reshape(*lead, T, 16, 4)); v = p[..., 64:80]
        al = sg(p[..., 80:81]); be = sg(p[..., 81:82])
        Qt = Q.reshape(SB, T, 16, 4).unbind(1); vt = v.reshape(SB, T, 16).unbind(1)
        alt = al.reshape(SB, T, 1).unbind(1); bet = be.reshape(SB, T, 1).unbind(1)
        h = h.reshape(SB, 16)
        for t in range(T):
            ht = alt[t] * h
            d = vt[t] - ht
            h = ht + bet[t] * torch.bmm(Qt[t], torch.bmm(Qt[t].transpose(1, 2), d.unsqueeze(-1))).squeeze(-1)
        h = h.reshape(*lead, 16)
    elif n == "GatedDeltaNet16":
        k = unit(p[..., 0:16]); al = sg(p[..., 16:17]); be = sg(p[..., 17:18]); v = p[..., 18:34]
        for t in range(T):
            ht = al[..., t, :] * h
            kt = k[..., t, :]
            h = ht + be[..., t, :] * kt * (kt * (v[..., t, :] - ht)).sum(-1, keepdim=True)
    return h

# symmetric long-horizon gate bias init (0.5^128 underflows otherwise)
GATE_BIAS = {
    "SedenionGatedDeltaSSM": {30: 4.5, 31: -2.0},
    "SedenionDeltaSSM": {30: -2.0},
    "GatedDeltaNet4x4": {**{i: 4.5 for i in (4, 14, 24, 34)}, **{i: -2.0 for i in (5, 15, 25, 35)}},
    "DeltaProductRank4": {0: 4.5, **{i: -3.4 for i in (33, 50, 67, 84)}},
    "GrassmannQR4SSM": {80: 4.5, 81: -2.0},
    "GatedDeltaNet16": {16: 4.5, 17: -2.0},
}

def init_gate_bias(enc_linear2, name):
    with torch.no_grad():
        for i, v in GATE_BIAS[name].items():
            enc_linear2.bias[i] = v

class SSM(nn.Module):
    def __init__(self, name, enc_h, readout):
        super().__init__()
        self.name = name
        self.enc = nn.Sequential(nn.Linear(DIN, enc_h), nn.Tanh(), nn.Linear(enc_h, NOUT[name]))
        init_gate_bias(self.enc[2], name)
        self.readout = readout

    def states(self, X):
        return recur(self.name, self.enc(X))

class KVReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(8, 4, 16) * (1 / 4.0))
    def forward(self, hT, kq):
        return torch.einsum("bij,bj->bi", self.W[kq], hT)

class ContinuousQueryReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 64))
    def forward(self, hT, kq):
        W = self.net(kq).reshape(-1, 4, 16)
        return torch.einsum("bij,bj->bi", W, hT)

def make_model(name, enc_h, part):
    return SSM(name, enc_h, KVReadout() if part == 2 else ContinuousQueryReadout())

def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def tune_enc_h():
    torch.manual_seed(0)
    res = {"SedenionGatedDeltaSSM": REF_ENC_H}
    for part in (2, 3):
        pass
    tgt = nparams(make_model("SedenionGatedDeltaSSM", REF_ENC_H, 2))
    for n in MODELS:
        if n == "SedenionGatedDeltaSSM": continue
        res[n] = min(range(4, 400), key=lambda h: abs(nparams(make_model(n, h, 2)) - tgt))
    info = {"reference_model": "SedenionGatedDeltaSSM", "reference_enc_h": REF_ENC_H, "enc_h": res, "NOUT": NOUT, "param_count": {}}
    for part in (2, 3):
        c = {n: nparams(make_model(n, res[n], part)) for n in MODELS}
        ref = c["SedenionGatedDeltaSSM"]
        dev = {n: c[n] / ref - 1 for n in MODELS}
        mx = max(abs(v) for v in dev.values())
        info["param_count"]["part%d" % part] = {"counts": c, "rel_dev_vs_SedenionGatedDeltaSSM": dev,
                                                 "max_abs_dev": mx, "within_2pct": bool(mx <= 0.02)}
    return res, info

# ---------------------------------------------------------------- Part 1
def p4_block_check(n=1000):
    g = torch.Generator().manual_seed(1)
    u = torch.randn(n, 7, generator=g, dtype=torch.float64); w = torch.randn(n, 7, generator=g, dtype=torch.float64)
    out = {}
    for tag, dt in (("float64", torch.float64), ("float32", torch.float32)):
        P = P4(z_embed(u.to(dt), w.to(dt)))
        cols = P[:, :, [0, 8]].abs().max().item(); rows = P[:, [0, 8], :].abs().max().item()
        out[tag] = {"max_abs_P4_cols_0_8": cols, "max_abs_P4_rows_0_8": rows}
    out["n_samples"] = n
    out["immortal_lt_1e-6_float64"] = bool(max(out["float64"].values()) < 1e-6)
    out["immortal_lt_1e-6_float32"] = bool(max(out["float32"].values()) < 1e-6)
    return out

def welch_mean_offdiag(K, m, n):
    return max(0.0, m * (K * m - n) / (n * (K - 1)))

def pack(param, K, restarts=3, steps=1500):
    """Minimise mean off-diag tr(PiPj)^2 over K projectors of the given parametrisation."""
    def projectors(par):
        if param == "Sed_V72":
            return P4(z_embed(par[:, :7], par[:, 7:14]))
        if param == "4Head_GDN":
            k = unit(par.reshape(K, 4, 4))
            Ph = k.unsqueeze(-1) * k.unsqueeze(-2)                      # (K,4,4,4)
            P = torch.zeros(K, 16, 16, dtype=par.dtype)
            for hd in range(4):
                P[:, 4 * hd:4 * hd + 4, 4 * hd:4 * hd + 4] = Ph[:, hd]
            return P
        if param == "Full_Gr4_16":
            Q, _ = torch.linalg.qr(par.reshape(K, 16, 4)); return Q @ Q.transpose(-1, -2)
    npar = {"Sed_V72": 14, "4Head_GDN": 16, "Full_Gr4_16": 64}[param]
    iu = torch.triu_indices(K, K, 1)
    runs = []
    for r in range(restarts):
        g = torch.Generator().manual_seed(1000 * r + K)
        par = torch.randn(K, npar, generator=g, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([par], lr=0.05)
        def tr_mat(par):
            P = projectors(par); return torch.einsum("aij,bji->ab", P, P)
        for _ in range(steps):
            M = tr_mat(par); loss = (M[iu[0], iu[1]] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            M = tr_mat(par)[iu[0], iu[1]]
            runs.append({"restart": r, "mean_offdiag_tr": M.mean().item(), "max_offdiag_tr": M.max().item(),
                         "rms_offdiag_tr": M.pow(2).mean().sqrt().item()})
    return runs

def throughput(enc_h):
    prev_t = torch.get_num_threads()
    try:
        return _throughput(enc_h)
    finally:
        torch.set_num_threads(prev_t)

def _throughput(enc_h):
    torch.set_num_threads(1)
    res = {}
    for n in MODELS:
        torch.manual_seed(0)
        m = make_model(n, enc_h[n], 2)
        X = torch.randn(BATCH, T_LEN, DIN); kq = torch.randint(0, 8, (BATCH,)); y = torch.randn(BATCH, 4)
        def it():
            loss = F.mse_loss(m.readout(m.states(X), kq), y); m.zero_grad(); loss.backward()
        it()  # warmup
        t0 = time.time()
        for _ in range(10): it()
        dt = time.time() - t0
        res[n] = {"tokens_per_sec": 10 * BATCH * T_LEN / dt, "sec_per_iter": dt / 10, "NOUT": NOUT[n]}
    return res

def throughput_mps(enc_h):
    res = {}
    S = len(SEEDS)
    for n in MODELS:
        e = Ensemble(n, enc_h[n], 2, SEEDS)
        X = torch.randn(S, BATCH, T_LEN, DIN, device=DEV); kq = torch.randint(0, 8, (S, BATCH), device=DEV); y = torch.randn(S, BATCH, 4, device=DEV)
        def it():
            ((e(X, kq) - y) ** 2).mean(dim=(1, 2)).sum().backward()
            e.zero_grad()
        def sync():
            if DEV.type == "mps": torch.mps.synchronize()
        it(); sync(); t0 = time.time()
        for _ in range(10): it()
        sync(); dt = time.time() - t0
        res[n] = {"tokens_per_sec_10seeds_total": 10 * S * BATCH * T_LEN / dt, "sec_per_iter": dt / 10, "device": str(DEV), "NOUT": NOUT[n]}
    return res

def mgs_vs_qr():
    g = torch.Generator().manual_seed(9)
    K = torch.randn(1000, 16, 4, generator=g, dtype=torch.float64)
    Qm = mgs4(K); Qq, _ = torch.linalg.qr(K)
    return {"mgs4_vs_qr_projector_max_diff": (Qm @ Qm.transpose(-1, -2) - Qq @ Qq.transpose(-1, -2)).abs().max().item(),
            "left_mat_fast_vs_left_mat_max_diff": (lambda z: (left_mat_fast(z) - left_mat(z, T16)).abs().max().item())(torch.randn(100, 16))}

def part1(enc_h):
    r = {"immortality_V72": p4_block_check(), "welch_packing": {}}
    r["exactness_checks"] = mgs_vs_qr()
    for K in (4, 6, 8, 12):
        e = {"welch_bound_mean_offdiag_tr_Gr(4,14)": welch_mean_offdiag(K, 4, 14),
             "welch_bound_mean_offdiag_tr_Gr(4,16)": welch_mean_offdiag(K, 4, 16)}
        for pn in ("Sed_V72", "4Head_GDN", "Full_Gr4_16"):
            runs = pack(pn, K)
            e[pn] = {"restarts": runs, "best_by_mean_offdiag_tr": min(runs, key=lambda x: x["mean_offdiag_tr"])}
        r["welch_packing"]["K=%d" % K] = e
        print("welch K=%d done" % K, flush=True)
    r["throughput_single_thread_B256_T%d" % T_LEN] = throughput(enc_h)
    r["throughput_mps_10seed"] = throughput_mps(enc_h)
    return r

# ---------------------------------------------------------------- data (Parts 2 & 3)
def rs3(rng, n=None):
    v = rng.normal(size=4 if n is None else (n, 4)); return v / np.linalg.norm(v, axis=-1, keepdims=True)

def gen(rng, n, T, continuous):
    """Tokens [op onehot(4), key(8), val(4)]. 3 writes, 2 overwrites, 1 delete, T-6 NOOPs with random distractor key/val.
    Discrete: key one-hot over 8. Continuous: fresh k~Uniform(S^7) per episode key; NOOP keys also random S^7."""
    X = np.zeros((n, T, 16), np.float32)
    kq = np.zeros((n, 8), np.float32) if continuous else np.zeros(n, np.int64)
    y = np.zeros((n, 4), np.float32); isdel = np.zeros(n, bool)
    for i in range(n):
        v0 = rs3(rng, 3); v1 = rs3(rng, 2)
        if continuous:
            keys = rng.normal(size=(3, 8)); keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        else:
            keys = np.eye(8)[rng.permutation(8)[:3]]
        tok = np.zeros((T, 16), np.float32)
        tok[:, 0] = 1
        if continuous:
            nk = rng.normal(size=(T, 8)); tok[:, 4:12] = nk / np.linalg.norm(nk, axis=1, keepdims=True)
        else:
            tok[np.arange(T), 4 + rng.integers(8, size=T)] = 1
        tok[:, 12:] = rs3(rng, T)
        slots = np.sort(rng.permutation(T)[:6]); order = rng.permutation([0, 0, 1, 1, 2, 2])
        seen = [0, 0, 0]; final = {0: v1[0], 1: v1[1], 2: np.zeros(4)}
        for s, c in zip(slots, order):
            tok[s] = 0
            if seen[c] == 0:
                tok[s, 1] = 1; tok[s, 12:] = v0[c]
            elif c == 2:
                tok[s, 3] = 1
            else:
                tok[s, 2] = 1; tok[s, 12:] = v1[c]
            tok[s, 4:12] = keys[c]; seen[c] += 1
        X[i] = tok
        deleted = (i % 2 == 0)
        c = 2 if deleted else int(rng.integers(2))
        kq[i] = keys[c] if continuous else int(np.argmax(keys[c]))
        y[i] = final[c]; isdel[i] = deleted
    perm = rng.permutation(n)
    return X[perm], kq[perm], y[perm], isdel[perm]

class Ensemble(nn.Module):
    """S independent models (each built with torch.manual_seed(seed_s)) whose parameters are stacked on a leading S axis."""
    def __init__(self, name, enc_h, part, seeds):
        super().__init__()
        self.name, self.part, self.S = name, part, len(seeds)
        ms_ = []
        for sd in seeds:
            torch.manual_seed(sd); ms_.append(make_model(name, enc_h, part))
        self.keys = [k for k, _ in ms_[0].named_parameters()]
        self.P = nn.ParameterDict({k.replace(".", "_"): nn.Parameter(torch.stack([dict(m.named_parameters())[k].detach() for m in ms_]).to(DEV))
                                   for k in self.keys})
    def prm(self, k):
        return self.P[k.replace(".", "_")]
    def forward(self, X, kq):
        """X:(S,B,T,16); kq:(S,B) long (part 2) or (S,B,8) float (part 3) -> (S,B,4)."""
        a = torch.tanh(torch.einsum("sbtd,shd->sbth", X, self.prm("enc.0.weight")) + self.prm("enc.0.bias")[:, None, None])
        p = torch.einsum("sbth,soh->sbto", a, self.prm("enc.2.weight")) + self.prm("enc.2.bias")[:, None, None]
        hT = recur(self.name, p)
        if self.part == 2:
            W = self.prm("readout.W")[torch.arange(self.S, device=X.device)[:, None], kq]
        else:
            r = torch.tanh(torch.einsum("sbd,sjd->sbj", kq, self.prm("readout.net.0.weight")) + self.prm("readout.net.0.bias")[:, None])
            W = (torch.einsum("sbj,soj->sbo", r, self.prm("readout.net.2.weight")) + self.prm("readout.net.2.bias")[:, None]).reshape(self.S, -1, 4, 16)
        return torch.einsum("sbij,sbj->sbi", W, hT)

def clip_per_seed(params, maxnorm=1.0):
    S = params[0].shape[0]
    tot = torch.zeros(S, device=params[0].device)
    for q in params:
        tot = tot + (q.grad.reshape(S, -1) ** 2).sum(1)
    scale = torch.clamp(maxnorm / (tot.sqrt() + 1e-6), max=1.0)
    for q in params:
        q.grad.mul_(scale.reshape(S, *([1] * (q.grad.dim() - 1))))

def make_opt(params):
    opt = torch.optim.AdamW(params, lr=LR)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

def ensemble_parity(steps=2, S=2, B=8, T=8):
    """Ensemble vs independent SSM: forward + clipped AdamW steps on identical random data; returns max abs diff."""
    worst = 0.0; per = {}
    seeds = [11, 22][:S]
    for part in (2, 3):
        for n in MODELS:
            enc_h = ENC_H[n]
            g = torch.Generator().manual_seed(5)
            X = torch.randn(S, B, T, DIN, generator=g)
            y = torch.randn(S, B, 4, generator=g)
            kq = torch.randint(0, 8, (S, B), generator=g) if part == 2 else unit(torch.randn(S, B, 8, generator=g))
            # independent
            singles = []
            for s_, sd in enumerate(seeds):
                torch.manual_seed(sd); m = make_model(n, enc_h, part)
                opt, sch = make_opt(list(m.parameters()))
                for _ in range(steps):
                    loss = F.mse_loss(m.readout(m.states(X[s_]), kq[s_]), y[s_])
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    singles.append(m.readout(m.states(X[s_]), kq[s_]))
            # ensemble
            e = Ensemble(n, enc_h, part, seeds)
            ps = list(e.parameters()); opt, sch = make_opt(ps)
            Xd, yd, kd = X.to(DEV), y.to(DEV), kq.to(DEV)
            for _ in range(steps):
                per_seed = ((e(Xd, kd) - yd) ** 2).mean(dim=(1, 2))
                opt.zero_grad(); per_seed.sum().backward(); clip_per_seed(ps); opt.step(); sch.step()
            with torch.no_grad():
                pe = e(Xd, kd).cpu()
            d = max((pe[s_] - singles[s_]).abs().max().item() for s_ in range(S))
            per["part%d/%s" % (part, n)] = d; worst = max(worst, d)
    return worst, per

def train_ensemble(part, name, enc_h):
    S = len(SEEDS)
    continuous = (part == 3)
    D = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd + (0 if part == 2 else 7919))
        D.append((gen(rng, NTRAIN, T_LEN, continuous), gen(rng, NTEST, T_LEN, continuous)))
    Xtr = torch.tensor(np.stack([d[0][0] for d in D])).to(DEV); ktr = torch.tensor(np.stack([d[0][1] for d in D])).to(DEV)
    ytr = torch.tensor(np.stack([d[0][2] for d in D])).to(DEV)
    Xte = torch.tensor(np.stack([d[1][0] for d in D])).to(DEV); kte = torch.tensor(np.stack([d[1][1] for d in D])).to(DEV)
    yte = torch.tensor(np.stack([d[1][2] for d in D])).to(DEV); dte = torch.tensor(np.stack([d[1][3] for d in D])).to(DEV)
    e = Ensemble(name, enc_h, part, SEEDS)
    ps = list(e.parameters()); opt, sch = make_opt(ps)
    gens = [torch.Generator().manual_seed(sd) for sd in SEEDS]
    ar = torch.arange(S, device=DEV)[:, None]
    for step in range(STEPS):
        idx = torch.stack([torch.randint(0, NTRAIN, (BATCH,), generator=g_) for g_ in gens]).to(DEV)
        pred = e(Xtr[ar, idx], ktr[ar, idx])
        per_seed = ((pred - ytr[ar, idx]) ** 2).mean(dim=(1, 2))
        opt.zero_grad(); per_seed.sum().backward(); clip_per_seed(ps); opt.step(); sch.step()
        if step % 100 == 0:
            print("  p%d %s step %d mean train mse %.5f" % (part, name, step, per_seed.mean().item()), flush=True)
    out = []
    with torch.no_grad():
        pred = torch.cat([e(Xte[:, i:i + 256], kte[:, i:i + 256]) for i in range(0, NTEST, 256)], 1)
        for s_, sd in enumerate(SEEDS):
            d = dte[s_]; pr, yy = pred[s_], yte[s_]
            out.append({"part": part, "model": name, "seed": sd,
                        "test_mse": F.mse_loss(pr, yy).item(),
                        "active_mse": F.mse_loss(pr[~d], yy[~d]).item(),
                        "deleted_mse": F.mse_loss(pr[d], yy[d]).item(),
                        "active_cos_sim": F.cosine_similarity(pr[~d], yy[~d], dim=-1).mean().item(),
                        "deleted_norm": pr[d].norm(dim=-1).mean().item(),
                        "last_batch_train_mse": per_seed[s_].item()})
    return out

# ------------------------------------------------------------------ stats
def paired(a, b):
    a = np.asarray(a); b = np.asarray(b)
    r = {"mean_diff_sed_minus_base": float((a - b).mean()), "sed_lower_n": int((a < b).sum()), "n": len(a)}
    try:
        t = stats.ttest_rel(a, b); r["ttest_rel_p"] = float(t.pvalue); r["ttest_rel_t"] = float(t.statistic)
    except Exception as e:
        r["ttest_rel_p"] = None
    try:
        r["wilcoxon_p"] = float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)
    except Exception as e:
        r["wilcoxon_p"] = None; r["wilcoxon_err"] = str(e)
    return r

def ms(x):
    x = np.asarray(x, float); return {"mean": float(x.mean()), "std": float(x.std(ddof=1))}

METRICS = ["test_mse", "active_mse", "deleted_mse", "active_cos_sim", "deleted_norm"]

def summarize(raw, part):
    out = {"models": {}, "paired": {}}
    for n in MODELS:
        rs = sorted([r for r in raw if r["part"] == part and r["model"] == n], key=lambda r: SEEDS.index(r["seed"]))
        out["models"][n] = {met: {"seeds": [r[met] for r in rs], **ms([r[met] for r in rs])} for met in METRICS}
    for met in METRICS:
        out["paired"][met] = {s: {n: paired(out["models"][s][met]["seeds"], out["models"][n][met]["seeds"])
                                  for n in MODELS if n != s} for s in SED}
    return out

def fmt(x):
    return "n/a" if x is None else "%.3g" % x

def print_tables(rep):
    for key in ("part2_discrete_kv4d_T128", "part3_continuous_unseen_kv4d_T128"):
        print("\n== %s ==" % key)
        s = rep[key]
        print("%-22s %s" % ("model", "  ".join("%-17s" % m for m in METRICS)))
        for n in MODELS:
            print("%-22s %s" % (n, "  ".join("%.5f±%.5f" % (s["models"][n][m]["mean"], s["models"][n][m]["std"]) for m in METRICS)))
        for met in METRICS:
            for sd in SED:
                for n, r in s["paired"][met][sd].items():
                    print("%s %s %s vs %-22s t p=%s wilcoxon p=%s meandiff=%.5g sed_lower=%d/10" % (
                        key[:5], met, sd, n, fmt(r["ttest_rel_p"]), fmt(r["wilcoxon_p"]), r["mean_diff_sed_minus_base"], r["sed_lower_n"]))
        print("seed arrays test_mse:")
        for n in MODELS:
            print("  %-22s %s" % (n, ["%.5f" % x for x in s["models"][n]["test_mse"]["seeds"]]))

ENC_H = {}

def main():
    global ENC_H
    print("MPS_AVAILABLE:", torch.backends.mps.is_available(), "DEV:", DEV, "THREADS:", torch.get_num_threads(), flush=True)
    t0 = time.time()
    enc_h, pinfo = tune_enc_h(); ENC_H = enc_h
    print("enc_h:", enc_h, "device:", DEV); print(json.dumps(pinfo["param_count"], indent=1), flush=True)
    assert all(pinfo["param_count"]["part%d" % q]["within_2pct"] for q in (2, 3)), "param parity violated"
    worst, per = ensemble_parity()
    print("ensemble_parity_max_diff", worst, json.dumps(per), flush=True)
    if worst >= 1e-4:
        print("ABORT: ensemble parity >= 1e-4"); sys.exit(2)
    rep = {"config": {"seeds": SEEDS, "steps": STEPS, "batch": BATCH, "lr": LR, "d_state": 16, "T": T_LEN,
                      "optimizer": "AdamW cosine, per-seed clip_grad_norm 1.0 on all models (batched Ensemble10)", "din": DIN,
                      "ntrain": NTRAIN, "ntest": NTEST, "device": str(DEV), "gate_bias_init": GATE_BIAS,
                      "ensemble_parity_max_diff": worst, "ensemble_parity_per_model": per},
           "parity": pinfo}
    rep["part1_welch_immortal_and_speed"] = part1(enc_h)
    print(json.dumps(rep["part1_welch_immortal_and_speed"], indent=1), flush=True)
    done = []
    for part in (2, 3):
        for n in MODELS:
            done += train_ensemble(part, n, enc_h[n])
            print("done p%d %s (%.0fs) mse by seed %s" % (part, n, time.time() - t0, ["%.5f" % r["test_mse"] for r in done[-len(SEEDS):]]), flush=True)
    rep["raw"] = done
    rep["part2_discrete_kv4d_T128"] = summarize(done, 2)
    rep["part3_continuous_unseen_kv4d_T128"] = summarize(done, 3)
    rep["wall_time_s"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1)
    print_tables(rep)
    print("wall %.0fs ; wrote %s" % (rep["wall_time_s"], OUT))

class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.o = sys.__stdout__
    def write(self, x):
        self.o.write(x); self.f.write(x); self.f.flush()
    def flush(self):
        self.o.flush(); self.f.flush()

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sys.stdout = Tee(os.environ.get("P9_LOG", os.path.join(os.path.dirname(OUT), "phase9_rank4_longctx_stdout.log")))
    main()
