#!/usr/bin/env python3
"""Phase 8 falsification suite: SedenionDeltaSSM vs 5 baselines (10 seeds).

Part 1: projector geometry. Part 2: algebra-free KV4D overwrite/delete.
Part 3: real 16-channel EEG (EEGMMIDB S001R01) + CHB-MIT seizure (4 patients).

All 6 models consume the IDENTICAL token vector through an identically shaped
MLP encoder (din -> enc_h -> tanh -> n_params_of_model); enc_h is tuned per
model for trainable-parameter parity (+-2%). No model has any op-specific
branch. Raw numbers and paired tests are reported whatever the outcome.
"""
import json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.environ.get("P8_OUT", "/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase8_sedenion_ablation_report.json")
SEEDS = [42, 137, 271, 314, 577, 777, 1024, 1618, 2026, 4096]
MODELS = ["SedenionDeltaSSM", "SedenionMulSSM", "TwoOctonionSSM",
          "FourQuatBlockSSM", "GatedDeltaNet16", "Random16BasisSSM"]
DIN = 16          # Part 2 token: 4 op one-hot + 8 key one-hot + 4 val ; Part 3: 16 channels
REF_ENC_H = 64
STEPS, BATCH, LR = int(os.environ.get("P8_STEPS", 600)), 256, 3e-3
RANDOM_BASIS_SEED = 20260101

# ----------------------------------------------------------------- algebra
def _conj(x):
    y = -x.copy(); y[0] = x[0]; return y

def cd_mul(a, b):
    """Cayley-Dickson product (sedenion.sio): (alo*blo - conj(bhi)*ahi, bhi*alo + ahi*conj(blo))."""
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

T16, T8, T4 = structure_tensor(16), structure_tensor(8), structure_tensor(4)

def left_mat(a, T):
    """L_a such that (a*x) = L_a @ x.  a: (...,n) -> (...,n,n)."""
    return torch.einsum("...i,ijk->...kj", a, T)

def unit(x, eps=1e-8):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)

def z_embed(u, w):
    """(u,w) orthonormal in R^7 -> unit sedenion z = ((0,u),(0,w))/sqrt2."""
    u = unit(u)
    w = w - (w * u).sum(-1, keepdim=True) * u
    w = unit(w)
    zeros = torch.zeros_like(u[..., :1])
    return torch.cat([zeros, u, zeros, w], -1) / math.sqrt(2.0)

def P4(z):
    L = left_mat(z, T16)
    L2 = L @ L
    I = torch.eye(16, dtype=z.dtype).expand_as(L2)
    return I + 0.5 * L2 @ (3 * I + L2)

def random_basis():
    g = torch.Generator().manual_seed(RANDOM_BASIS_SEED)
    M = torch.randn(16, 256, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M.T)           # orthonormal columns in R^256
    return Q.T.reshape(16, 16, 16).float().contiguous()   # Frobenius-orthonormal M_k

M_BASIS = random_basis()

# ------------------------------------------------------------------ models
NOUT = {"SedenionDeltaSSM": 7 + 7 + 16 + 1, "SedenionMulSSM": 16 + 16 + 1,
        "TwoOctonionSSM": 8 + 8 + 16 + 2, "FourQuatBlockSSM": 16 + 4 + 16,
        "GatedDeltaNet16": 16 + 2 + 16, "Random16BasisSSM": 16 + 16 + 1}

class SSM(nn.Module):
    def __init__(self, name, enc_h, readout):
        super().__init__()
        self.name = name
        self.enc = nn.Sequential(nn.Linear(DIN, enc_h), nn.Tanh(), nn.Linear(enc_h, NOUT[name]))
        self.readout = readout  # module or None (Part-specific)

    def states(self, X):
        """X: (B,T,DIN) -> h_t after every token, (B,T,16). h_0 = 0."""
        p = self.enc(X)
        B, T, _ = p.shape
        n = self.name
        sg = torch.sigmoid
        h = X.new_zeros(B, 16)
        outs = []
        if n == "SedenionDeltaSSM":
            z = z_embed(p[..., 0:7], p[..., 7:14]); v = p[..., 14:30]; beta = sg(p[..., 30:31])
            P = P4(z)
            for t in range(T):
                d = v[:, t] - h
                h = h + beta[:, t] * torch.einsum("bij,bj->bi", P[:, t], d)
                outs.append(h)
        elif n == "SedenionMulSSM":
            a = unit(p[..., 0:16]); v = p[..., 16:32]; dl = sg(p[..., 32:33])
            L = left_mat(a, T16)
            for t in range(T):
                m = torch.einsum("bij,bj->bi", L[:, t], h) + v[:, t]
                h = (1 - dl[:, t]) * h + dl[:, t] * unit(torch.tanh(m))
                outs.append(h)
        elif n == "TwoOctonionSSM":
            a1 = unit(p[..., 0:8]); a2 = unit(p[..., 8:16]); v = p[..., 16:32].reshape(B, T, 2, 8)
            dl = sg(p[..., 32:34]).unsqueeze(-1)
            L = left_mat(torch.stack([a1, a2], -2), T8)   # (B,T,2,8,8)
            h = h.reshape(B, 2, 8)
            for t in range(T):
                m = torch.einsum("bhij,bhj->bhi", L[:, t], h) + v[:, t]
                h = (1 - dl[:, t]) * h + dl[:, t] * unit(torch.tanh(m))
                outs.append(h.reshape(B, 16))
        elif n == "FourQuatBlockSSM":
            q = unit(p[..., 0:16].reshape(B, T, 4, 4)); beta = sg(p[..., 16:20]).unsqueeze(-1)
            v = p[..., 20:36].reshape(B, T, 4, 4)
            L = left_mat(q, T4)                           # (B,T,4,4,4)
            h = h.reshape(B, 4, 4)
            for t in range(T):
                qh = torch.einsum("bkij,bkj->bki", L[:, t], h)
                h = (1 - beta[:, t]) * qh + beta[:, t] * v[:, t]
                outs.append(h.reshape(B, 16))
        elif n == "GatedDeltaNet16":
            k = unit(p[..., 0:16]); al = sg(p[..., 16:17]); be = sg(p[..., 17:18]); v = p[..., 18:34]
            for t in range(T):
                ah = al[:, t] * h
                r = v[:, t] - ah
                h = ah + be[:, t] * k[:, t] * (k[:, t] * r).sum(-1, keepdim=True)
                outs.append(h)
        elif n == "Random16BasisSSM":
            a = unit(p[..., 0:16]); v = p[..., 16:32]; dl = sg(p[..., 32:33])
            A = torch.einsum("btk,kij->btij", a, M_BASIS)
            for t in range(T):
                m = torch.einsum("bij,bj->bi", A[:, t], h) + v[:, t]
                h = (1 - dl[:, t]) * h + dl[:, t] * torch.tanh(m)
                outs.append(h)
        return torch.stack(outs, 1)

class KVReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(8, 4, 16) * (1 / 4.0))
    def forward(self, hT, kq):
        return torch.einsum("bij,bj->bi", self.W[kq], hT)

def make_model(name, enc_h, part):
    ro = KVReadout() if part == 2 else nn.Linear(16, 16, bias=False)
    return SSM(name, enc_h, ro)

def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def tune_enc_h():
    torch.manual_seed(0)
    tgt = nparams(make_model("SedenionDeltaSSM", REF_ENC_H, 2))
    res = {}
    for n in MODELS:
        best = min(range(4, 400), key=lambda h: abs(nparams(make_model(n, h, 2)) - tgt))
        res[n] = best
    counts = {p: {n: nparams(make_model(n, res[n], p)) for n in MODELS} for p in (2, 3)}
    info = {"reference_model": "SedenionDeltaSSM", "reference_enc_h": REF_ENC_H, "enc_h": res, "param_count": {}}
    for p in (2, 3):
        c = counts[p]; ref = c["SedenionDeltaSSM"]
        dev = {n: c[n] / ref - 1 for n in MODELS}
        info["param_count"][f"part{p}"] = {"counts": c, "rel_dev_vs_sedenion": dev,
                                            "max_abs_dev": max(abs(v) for v in dev.values()),
                                            "within_2pct": bool(max(abs(v) for v in dev.values()) <= 0.02)}
    return res, info

# ---------------------------------------------------------------- Part 1
def part1():
    g = torch.Generator().manual_seed(1)
    def sample_z(n):
        return z_embed(torch.randn(n, 7, generator=g), torch.randn(n, 7, generator=g))
    z = sample_z(1000).double()
    z32 = z.float()
    P = P4(z32).double()
    I = torch.eye(16, dtype=torch.float64)
    idem = torch.linalg.matrix_norm(P @ P - P).max().item()
    sym = torch.linalg.matrix_norm(P.transpose(-1, -2) - P).max().item()
    ranks = [int(np.linalg.matrix_rank(P[i].numpy(), tol=1e-4)) for i in range(1000)]
    svals = torch.linalg.svdvals(P)
    # 200 projectors span
    flat = P[:200].reshape(200, 256).numpy()
    sv = np.linalg.svd(flat, compute_uv=False)
    span_p4 = int((sv > 1e-4 * sv[0]).sum())
    # FourQuatBlock: 200 block-diag(L_q1..L_q4) operators with independent unit q_b
    gq = torch.Generator().manual_seed(2)
    q = unit(torch.randn(200, 4, 4, generator=gq))
    Lq = left_mat(q, T4)
    ops = torch.stack([torch.block_diag(*Lq[i]) for i in range(200)]).reshape(200, 256).numpy()
    sq = np.linalg.svd(ops, compute_uv=False)
    span_fq = int((sq > 1e-4 * sq[0]).sum())
    # FourQuat with ONE shared unit quaternion in all four blocks
    q1 = unit(torch.randn(200, 4, generator=gq))
    Lq1 = left_mat(q1, T4)
    ops1 = torch.stack([torch.block_diag(*[Lq1[i]] * 4) for i in range(200)]).reshape(200, 256).numpy()
    s1 = np.linalg.svd(ops1, compute_uv=False)
    span_fq_shared = int((s1 > 1e-4 * s1[0]).sum())
    # Random16 basis: 13th singular value over 5000 unit a
    ga = torch.Generator().manual_seed(3)
    a = unit(torch.randn(5000, 16, generator=ga, dtype=torch.float64))
    A = torch.einsum("nk,kij->nij", a, M_BASIS.double())
    s = torch.linalg.svdvals(A)
    s13 = s[:, 12]
    # P4(u)=0 on 100 unit octonions (u,0): imaginary unit octonions (the sedenion-consistent case) and general
    gu = torch.Generator().manual_seed(4)
    ui = torch.randn(100, 7, generator=gu); ui = ui / ui.norm(dim=-1, keepdim=True)
    zi = torch.cat([torch.zeros(100, 1), ui, torch.zeros(100, 8)], -1)
    Pi = P4(zi).abs().amax().item()
    ug = torch.randn(100, 8, generator=gu); ug = ug / ug.norm(dim=-1, keepdim=True)
    zg = torch.cat([ug, torch.zeros(100, 8)], -1)
    Pg = P4(zg).abs().amax().item()
    return {
        "n_V72_samples": 1000,
        "max_idempotence_err_fro": idem, "max_symmetry_err_fro": sym,
        "rank_P4_all_equal_4": bool(all(r == 4 for r in ranks)), "rank_P4_min": min(ranks), "rank_P4_max": max(ranks),
        "trace_P4_min_max": [P.diagonal(dim1=-2, dim2=-1).sum(-1).min().item(), P.diagonal(dim1=-2, dim2=-1).sum(-1).max().item()],
        "linear_span_dim_200_P4_in_R256": span_p4,
        "linear_span_dim_200_FourQuatBlock_ops_independent_q": span_fq,
        "linear_span_dim_200_FourQuatBlock_ops_shared_q": span_fq_shared,
        "note_span": "FourQuatBlock spans measured on block-diag(L_q1..L_q4) operators (not projectors); the '4' in the brief is reported as measured, not assumed.",
        "random16_min_13th_singular_value_5000_unit_a": s13.min().item(),
        "random16_13th_singular_value_gt_0": bool(s13.min().item() > 0),
        "P4_on_imag_unit_octonions_(u,0)_max_abs_entry_100": Pi,
        "P4_on_general_unit_octonions_(u,0)_max_abs_entry_100": Pg,
        "note_octonion": "P4=0 holds for imaginary unit octonions (L_u^2=-I); a general unit octonion with nonzero real part has L_u^2 != -I so P4 != 0 (reported honestly).",
    }

# ---------------------------------------------------------------- Part 2
OPS = {"NOOP": 0, "WRITE": 1, "OVERWRITE": 2, "DELETE": 3}

def gen_kv(rng, n):
    """Algebra-free KV task. Token = [op onehot(4), key onehot(8), val(4)] (16).
    NOOP steps carry a random distractor key and random unit value (pure noise)."""
    T = 16
    X = np.zeros((n, T, 16), np.float32); kq = np.zeros(n, np.int64); y = np.zeros((n, 4), np.float32)
    isdel = np.zeros(n, bool)
    def rv():
        v = rng.normal(size=4); return v / np.linalg.norm(v)
    for i in range(n):
        keys = rng.permutation(8)[:3]            # A, B overwritten ; C deleted
        v0 = [rv() for _ in range(3)]; v1 = [rv() for _ in range(2)]
        slots = np.sort(rng.permutation(T)[:6])
        order = rng.permutation([0, 0, 1, 1, 2, 2])
        seen = [0, 0, 0]; final = {0: v1[0], 1: v1[1], 2: np.zeros(4)}
        tok = np.zeros((T, 16), np.float32)
        for t in range(T):  # default NOOP with distractor
            tok[t, 0] = 1; tok[t, 4 + rng.integers(8)] = 1; tok[t, 12:] = rv()
        for s, c in zip(slots, order):
            tok[s] = 0
            if seen[c] == 0:
                tok[s, 1] = 1; tok[s, 12:] = v0[c]
            else:
                if c == 2:
                    tok[s, 3] = 1
                else:
                    tok[s, 2] = 1; tok[s, 12:] = v1[c]
            tok[s, 4 + keys[c]] = 1; seen[c] += 1
        X[i] = tok
        deleted = (i % 2 == 0)                     # exact 50/50 deleted vs active queries
        c = 2 if deleted else int(rng.integers(2))
        kq[i] = keys[c]; y[i] = final[c]; isdel[i] = deleted
    perm = rng.permutation(n)
    return X[perm], kq[perm], y[perm], isdel[perm]

def run_part2(args):
    name, seed, enc_h = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    Xtr, ktr, ytr, _ = gen_kv(rng, 4096)
    Xte, kte, yte, dte = gen_kv(rng, 1024)
    Xtr, ktr, ytr, Xte, kte, yte = map(torch.tensor, (Xtr, ktr, ytr, Xte, kte, yte))
    torch.manual_seed(seed)
    m = make_model(name, enc_h, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        idx = torch.randint(0, 4096, (BATCH,), generator=g)
        H = m.states(Xtr[idx])[:, -1]
        loss = F.mse_loss(m.readout(H, ktr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        pred = m.readout(m.states(Xte)[:, -1], kte)
        mse = F.mse_loss(pred, yte).item()
        d = torch.tensor(dte)
        cos = F.cosine_similarity(pred[~d], yte[~d], dim=-1).mean().item()
        dn = pred[d].norm(dim=-1).mean().item()
        trmse = loss.item()
    return {"model": name, "seed": seed, "test_mse": mse, "active_cos_sim": cos, "deleted_norm": dn,
            "last_batch_train_mse": trmse, "finite": bool(np.isfinite(mse))}

# ---------------------------------------------------------------- Part 3
def parse_sio(path, prefix):
    txt = open(path).read()
    arr = {}
    for nm, i, val in re.findall(r"^\s*(%sCH\d+)\[(\d+)\]\s*=\s*([-+0-9.eE]+)\s*$" % prefix, txt, re.M):
        arr.setdefault(nm, {})[int(i)] = float(val)
    X = np.zeros((80, 16))
    for c in range(16):
        d = arr["%sCH%d" % (prefix, c)]
        assert sorted(d) == list(range(80)), (path, prefix, c)
        X[:, c] = [d[i] for i in range(80)]
    return X.astype(np.float32)

def load_part3():
    ex = os.path.join(ROOT, "examples")
    D = {"eegmmidb_S001R01": {"ii": parse_sio(os.path.join(ex, "sedenion_ssm_16ch_eeg.sio"), "")}}
    for p in ("chb01", "chb03", "chb05", "chb11"):
        f = os.path.join(ex, "sedenion_ssm_seizure_%s.sio" % p)
        D[p] = {"ii": parse_sio(f, "II_"), "ic": parse_sio(f, "IC_")}
    return D

def run_part3(args):
    ds, name, seed, enc_h, ii, ic = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    m = make_model(name, enc_h, 3)
    X = torch.tensor(ii)[None]                   # (1,80,16)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for step in range(STEPS):
        H = m.states(X[:, :64])[:, :63]          # train: predict X[1..63] from h_0..h_62
        loss = F.mse_loss(m.readout(H), X[:, 1:64])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        H = m.states(X)                          # state continues through holdout
        pred = m.readout(H)
        mse_train = F.mse_loss(pred[:, :63], X[:, 1:64]).item()
        mse_test = F.mse_loss(pred[:, 63:79], X[:, 64:80]).item()   # 16 holdout samples
        out = {"dataset": ds, "model": name, "seed": seed, "mse_train": mse_train, "mse_test_ii": mse_test}
        if ic is not None:
            Xi = torch.tensor(ic)[None]
            pi = m.readout(m.states(Xi))         # fresh hidden state, 79 next-step predictions
            out["mse_ictal"] = F.mse_loss(pi[:, :79], Xi[:, 1:80]).item()
            out["disruption_ratio"] = out["mse_ictal"] / out["mse_test_ii"]
    return out

# ------------------------------------------------------------------ stats
def paired(a, b):
    a = np.asarray(a); b = np.asarray(b)
    r = {"mean_diff_sed_minus_base": float((a - b).mean()), "sed_wins_n": int((a < b).sum()), "n": len(a)}
    try:
        t = stats.ttest_rel(a, b); r["ttest_rel_p"] = float(t.pvalue); r["ttest_rel_t"] = float(t.statistic)
    except Exception as e:
        r["ttest_rel_p"] = None
    try:
        w = stats.wilcoxon(a, b, alternative="two-sided"); r["wilcoxon_p"] = float(w.pvalue)
    except Exception as e:
        r["wilcoxon_p"] = None; r["wilcoxon_err"] = str(e)
    return r

def ms(x):
    x = np.asarray(x, float); return {"mean": float(x.mean()), "std": float(x.std(ddof=1))}

def vs_all(per_model):  # per_model: name -> list over paired units
    return {n: paired(per_model["SedenionDeltaSSM"], per_model[n]) for n in MODELS if n != "SedenionDeltaSSM"}

# ------------------------------------------------------------------- main
def fmt(x):
    return "n/a" if x is None else ("%.3g" % x)

def main():
    t0 = time.time()
    enc_h, pinfo = tune_enc_h()
    print("enc_h:", enc_h); print(json.dumps(pinfo["param_count"], indent=1))
    rep = {"config": {"seeds": SEEDS, "steps": STEPS, "batch": BATCH, "lr": LR, "d_state": 16,
                      "optimizer": "AdamW cosine", "random_basis_seed": RANDOM_BASIS_SEED, "din": DIN},
           "parity": pinfo}
    rep["part1_projector_geometry"] = part1()
    print(json.dumps(rep["part1_projector_geometry"], indent=1))
    nproc = min(16, os.cpu_count() or 4)
    with Pool(nproc) as pool:
        jobs = [(n, s, enc_h[n]) for s in SEEDS for n in MODELS]
        r2 = pool.map(run_part2, jobs, chunksize=1)
        print("part2 done %.0fs" % (time.time() - t0), flush=True)
        p2 = {"raw": r2, "models": {}}
        for n in MODELS:
            rs = sorted([r for r in r2 if r["model"] == n], key=lambda r: SEEDS.index(r["seed"]))
            p2["models"][n] = {"seed_mses": [r["test_mse"] for r in rs],
                               "test_mse": ms([r["test_mse"] for r in rs]),
                               "active_cos_sim": ms([r["active_cos_sim"] for r in rs]),
                               "deleted_norm": ms([r["deleted_norm"] for r in rs]),
                               "seed_active_cos_sim": [r["active_cos_sim"] for r in rs],
                               "seed_deleted_norm": [r["deleted_norm"] for r in rs]}
        for met in ("test_mse", "active_cos_sim", "deleted_norm"):
            key = "seed_mses" if met == "test_mse" else "seed_" + met
            p2["paired_vs_SedenionDeltaSSM_" + met] = vs_all({n: p2["models"][n][key] for n in MODELS})
        p2["task_note"] = ("Query classes exactly 50% deleted / 50% active (active = one of the 2 overwritten keys uniformly); "
                           "the 3 active keys are 2 overwritten + 1 deleted, so no never-touched survivor exists. "
                           "NOOP tokens carry random distractor key/value. Baseline chance: predict 0 -> mse=0.5*(1/4)=0.125 (E|v|^2/4 on active half).")
        rep["part2_kv4d_overwrite_delete"] = p2

        D = load_part3()
        jobs = []
        for ds, d in D.items():
            for s in SEEDS:
                for n in MODELS:
                    jobs.append((ds, n, s, enc_h[n], d["ii"], d.get("ic")))
        r3 = pool.map(run_part3, jobs, chunksize=1)
        print("part3 done %.0fs" % (time.time() - t0), flush=True)
    p3 = {"raw": r3, "datasets": {}, "protocol": (
        "X_t(16)->X_{t+1}(16), linear 16x16 readout C@h_t (no bias). Train: predict X[1..63] from 64-sample window; "
        "hidden state continues through holdout samples 64..79 (16 predictions of X[64..79] from h_63..h_78). "
        "Ictal: fresh h_0=0, 79 next-step predictions on the 80-sample ictal window. Data used as stored in the .sio files.")}
    def collect(ds, metric):
        out = {}
        for n in MODELS:
            rs = sorted([r for r in r3 if r["dataset"] == ds and r["model"] == n], key=lambda r: SEEDS.index(r["seed"]))
            out[n] = [r[metric] for r in rs]
        return out
    metrics_for = {"eegmmidb_S001R01": ["mse_test_ii"]}
    for ds in D:
        mets = metrics_for.get(ds, ["mse_test_ii", "mse_ictal", "disruption_ratio"])
        p3["datasets"][ds] = {}
        for met in mets:
            c = collect(ds, met)
            p3["datasets"][ds][met] = {"per_model": {n: {"seeds": c[n], **ms(c[n])} for n in MODELS},
                                       "paired_vs_SedenionDeltaSSM": vs_all(c)}
    pooled = {}
    for met in ("mse_test_ii", "mse_ictal", "disruption_ratio"):
        c = {n: [] for n in MODELS}
        for ds in D:
            if ds == "eegmmidb_S001R01": continue
            for n, v in collect(ds, met).items(): c[n] += v
        pooled[met] = {"per_model": {n: ms(c[n]) for n in MODELS}, "paired_vs_SedenionDeltaSSM_40pairs": vs_all(c)}
    p3["chbmit_pooled_4patients_x_10seeds"] = pooled
    rep["part3_real_16ch_eeg_and_seizure"] = p3
    rep["wall_time_s"] = time.time() - t0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1)
    # ---- printed tables
    print("\n== PART 2 ==")
    for n in MODELS:
        m = p2["models"][n]
        print("%-18s mse %.5f±%.5f  cos %.4f±%.4f  delnorm %.4f±%.4f" % (n, m["test_mse"]["mean"], m["test_mse"]["std"],
              m["active_cos_sim"]["mean"], m["active_cos_sim"]["std"], m["deleted_norm"]["mean"], m["deleted_norm"]["std"]))
        print("   seed_mses", ["%.5f" % x for x in m["seed_mses"]])
    for met in ("test_mse", "active_cos_sim", "deleted_norm"):
        for n, r in p2["paired_vs_SedenionDeltaSSM_" + met].items():
            print("P2 %s Sed vs %-18s t p=%s wilcoxon p=%s  meandiff=%.5g wins=%d/10" % (met, n, fmt(r["ttest_rel_p"]), fmt(r["wilcoxon_p"]), r["mean_diff_sed_minus_base"], r["sed_wins_n"]))
    print("\n== PART 3 ==")
    for ds, dd in p3["datasets"].items():
        for met, v in dd.items():
            print("-- %s %s" % (ds, met))
            for n in MODELS:
                print("  %-18s %.5f±%.5f" % (n, v["per_model"][n]["mean"], v["per_model"][n]["std"]))
            for n, r in v["paired_vs_SedenionDeltaSSM"].items():
                print("  Sed vs %-18s t p=%s wilcoxon p=%s meandiff=%.5g wins=%d/10" % (n, fmt(r["ttest_rel_p"]), fmt(r["wilcoxon_p"]), r["mean_diff_sed_minus_base"], r["sed_wins_n"]))
    for met, v in pooled.items():
        print("-- POOLED CHB %s" % met)
        for n in MODELS:
            print("  %-18s %.5f±%.5f" % (n, v["per_model"][n]["mean"], v["per_model"][n]["std"]))
        for n, r in v["paired_vs_SedenionDeltaSSM_40pairs"].items():
            print("  Sed vs %-18s t p=%s wilcoxon p=%s meandiff=%.5g wins=%d/40" % (n, fmt(r["ttest_rel_p"]), fmt(r["wilcoxon_p"]), r["mean_diff_sed_minus_base"], r["sed_wins_n"]))
    print("wall %.0fs ; wrote %s" % (rep["wall_time_s"], OUT))

if __name__ == "__main__":
    main()
