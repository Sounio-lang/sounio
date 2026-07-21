#!/usr/bin/env python3
"""Multi-seed ResMLP clean-target panel: paired Δ and empirical p.

Trains n_seeds independent ResMLPs (same as deep_ffn_train.py), then for each
seed and input draw computes
  Δ_i(k) = align_trained_i(k) − median_b align_scrambled_{i,b}(k)
and reports mean±sd of align curves, Δ distributions, and a one-sided
permutation / sign p for mean Δ at selected k.

Requires torch (CPU ok). Writes docs/research/artifacts/multiseed_resmlp.json.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

# Prefer agent torch if system has none
if "torch" not in sys.modules:
    for p in (
        "/workspace/.home/openvscode-server/.agents/claude-1/.local/lib/python3.12/site-packages",
        "/workspace/.home/openvscode-server/.agents/claude-2/.local/lib/python3.12/site-packages",
    ):
        if Path(p, "torch").is_dir() and p not in sys.path:
            sys.path.insert(0, p)
            break

import torch
import torch.nn as nn

np.seterr(all="ignore")

d, W, L = 64, 96, 8
N_SEEDS = int(os.environ.get("N_SEEDS", "16"))
N_INPUT = int(os.environ.get("N_INPUT", "16"))
N_SCR = int(os.environ.get("N_SCR", "8"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5000"))
ACC_STOP = float(os.environ.get("ACC_STOP", "0.90"))
SHOW = [1, 2, 4, 8, 16, 32, 48]


class ResMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(d, W)
        self.A = nn.ParameterList(
            [nn.Parameter(torch.randn(W, W) / W**0.5) for _ in range(L)]
        )
        self.B = nn.ParameterList(
            [nn.Parameter(torch.randn(W, W) / W**0.5 * 0.5) for _ in range(L)]
        )
        self.rd = nn.Linear(W, 1)

    def forward(self, x):
        z = self.emb(x)
        for A, B in zip(self.A, self.B):
            z = z + torch.tanh(z @ A.t()) @ B.t()
        return self.rd(z).squeeze(-1)


def gen(n, rng):
    X = rng.standard_normal((n, d)).astype("float32")
    s_ = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] * X[:, 5]
    return torch.tensor(X), torch.tensor((s_ > 0).astype("float32"))


def train_one(seed: int):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = ResMLP()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xte, yte = gen(4000, np.random.default_rng(seed + 99991))
    acc = 0.0
    steps_used = MAX_STEPS
    for it in range(MAX_STEPS):
        X, y = gen(256, rng)
        opt.zero_grad()
        lossf(net(X), y).backward()
        opt.step()
        if it % 500 == 0 or it == MAX_STEPS - 1:
            with torch.no_grad():
                acc = ((net(Xte) > 0).float() == yte).float().mean().item()
            if acc >= ACC_STOP and it >= 1000:
                steps_used = it + 1
                break
    return net, acc, steps_used


def branch_jacs(embW, embb, A, B, x):
    z = embW @ x + embb
    Js = []
    for Al, Bl in zip(A, B):
        a = np.tanh(Al @ z)
        Js.append(Bl @ ((1 - a * a)[:, None] * Al))
        z = z + Bl @ a
    return Js


def align_curve(mats, ks):
    Vt = [np.linalg.svd(A, full_matrices=False)[2] for A in mats]
    return np.array(
        [
            np.mean(
                [
                    np.linalg.svd(Vt[l][-k:] @ Vt[l + 1][-k:].T, compute_uv=False).mean()
                    for l in range(len(Vt) - 1)
                ]
            )
            for k in ks
        ]
    )


def scramble(mats, rng):
    Os = [np.linalg.qr(rng.standard_normal((W, W)))[0] for _ in range(len(mats) + 1)]
    return [Os[l + 1] @ mats[l] @ Os[l].T for l in range(len(mats))]


def weights_np(net):
    emb = net.emb.weight.detach().numpy()
    embb = net.emb.bias.detach().numpy()
    At = [p.detach().numpy() for p in net.A]
    Bt = [p.detach().numpy() for p in net.B]
    return emb, embb, At, Bt


def probe_seed(net, seed: int, ks):
    emb, embb, At, Bt = weights_np(net)
    # fresh init twin for architectural control (same architecture, new seed)
    torch.manual_seed(seed + 123456)
    net0 = ResMLP()
    e0, b0, A0, B0 = weights_np(net0)

    rr = np.random.default_rng(7000 + seed)
    tr_rows, in_rows, sc_rows, delta_rows = [], [], [], []
    for i in range(N_INPUT):
        x = rr.standard_normal(d)
        tr = align_curve(branch_jacs(emb, embb, At, Bt, x), ks)
        inn = align_curve(branch_jacs(e0, b0, A0, B0, x), ks)
        scr_list = []
        for b in range(N_SCR):
            mats = branch_jacs(emb, embb, At, Bt, x)
            scr_list.append(align_curve(scramble(mats, rr), ks))
        scr_med = np.median(scr_list, axis=0)
        tr_rows.append(tr)
        in_rows.append(inn)
        sc_rows.append(scr_med)
        delta_rows.append(tr - scr_med)
    return (
        np.mean(tr_rows, 0),
        np.mean(in_rows, 0),
        np.mean(sc_rows, 0),
        np.array(delta_rows),  # (N_INPUT, n_k)
    )


def empirical_p_mean_delta(deltas_flat: np.ndarray) -> float:
    """One-sided sign / permutation p for mean(Δ) > 0: fraction of sign-flips
    with |mean| ≥ observed mean, under H0 symmetric around 0.
    p = (1 + #{|mean_b| ≥ |mean_obs| with sign random}) / (B+1) using B=9999
    random sign flips; equivalent to mean of signed values.
    """
    obs = float(deltas_flat.mean())
    rng = np.random.default_rng(0)
    B = 9999
    signs = rng.choice([-1.0, 1.0], size=(B, deltas_flat.size))
    null_means = (signs * deltas_flat.reshape(1, -1)).mean(axis=1)
    # one-sided: P(null ≥ obs) if obs>0, else P(null ≤ obs)
    if obs >= 0:
        ge = int((null_means >= obs).sum())
    else:
        ge = int((null_means <= obs).sum())
    return (1 + ge) / (B + 1), obs


def main():
    ks = list(range(1, W))
    base = np.sqrt(np.array(ks) / W)
    show_idx = [ks.index(k) for k in SHOW]

    seed_tr, seed_in, seed_sc = [], [], []
    all_delta = []  # list of (N_INPUT, n_k)
    accs, steps = [], []

    print(
        f"ResMLP multi-seed  W={W} L={L}  n_seeds={N_SEEDS} n_input={N_INPUT} "
        f"n_scr={N_SCR} max_steps={MAX_STEPS} acc_stop={ACC_STOP}",
        flush=True,
    )
    print(f"torch {torch.__version__} device=cpu", flush=True)

    for s in range(N_SEEDS):
        net, acc, st = train_one(seed=s)
        accs.append(acc)
        steps.append(st)
        tr, inn, sc, deltas = probe_seed(net, seed=s, ks=ks)
        seed_tr.append(tr)
        seed_in.append(inn)
        seed_sc.append(sc)
        all_delta.append(deltas)
        d48 = deltas[:, show_idx[-1]].mean()
        print(
            f"  seed {s + 1}/{N_SEEDS}  acc={acc:.3f} steps={st}  "
            f"meanΔ@48={d48:+.4f}  tr@4={tr[3]:.3f} sc@4={sc[3]:.3f}",
            flush=True,
        )

    seed_tr = np.array(seed_tr)
    seed_in = np.array(seed_in)
    seed_sc = np.array(seed_sc)
    # stack Δ: (n_seeds * N_INPUT, n_k)
    delta_stack = np.concatenate(all_delta, axis=0)

    def row(a):
        return " ".join(f"{a[ks.index(k)]:7.3f}" for k in SHOW)

    print("\nMean over seeds of per-seed mean-over-inputs align(k):")
    print("  k        " + " ".join(f"{k:>7}" for k in SHOW))
    print("  baseline " + " ".join(f"{np.sqrt(k / W):7.3f}" for k in SHOW))
    print("  TRAINED  " + row(seed_tr.mean(0)) + "  ± " + row(seed_tr.std(0)))
    print("  INIT     " + row(seed_in.mean(0)) + "  ± " + row(seed_in.std(0)))
    print("  SCRAMBLE " + row(seed_sc.mean(0)) + "  ± " + row(seed_sc.std(0)))

    p_table = {}
    print("\nPaired Δ = trained − median_b scramble  (pooled over seeds × inputs):")
    for k in SHOW:
        di = delta_stack[:, ks.index(k)]
        p, mean = empirical_p_mean_delta(di)
        p_table[str(k)] = {
            "mean_delta": float(mean),
            "std_delta": float(di.std()),
            "median_delta": float(np.median(di)),
            "p_one_sided_signflip": float(p),
            "n": int(di.size),
            "frac_positive": float((di > 0).mean()),
        }
        print(
            f"  k={k:>2}  meanΔ={mean:+.4f}  sd={di.std():.4f}  "
            f"med={np.median(di):+.4f}  frac(Δ>0)={(di > 0).mean():.2f}  "
            f"p_signflip={p:.4f}  n={di.size}"
        )

    # global verdict: max |mean Δ| and whether any k has p < 0.05 and meanΔ > 0.05
    any_pos = any(
        p_table[str(k)]["mean_delta"] > 0.05 and p_table[str(k)]["p_one_sided_signflip"] < 0.05
        for k in SHOW
    )
    print(
        f"\n  acc across seeds: mean={np.mean(accs):.3f} ± {np.std(accs):.3f}  "
        f"min={np.min(accs):.3f}"
    )
    print(
        f"  → {'SIGNAL above scramble at some k' if any_pos else 'NEGATIVE — no k with meanΔ>0.05 and p<0.05'}"
    )

    out = {
        "W": W,
        "L": L,
        "d": d,
        "n_seeds": N_SEEDS,
        "n_input": N_INPUT,
        "n_scr_per_input": N_SCR,
        "max_steps": MAX_STEPS,
        "acc_stop": ACC_STOP,
        "show_k": SHOW,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "acc_per_seed": [float(a) for a in accs],
        "steps_per_seed": steps,
        "trained_mean": {str(k): float(seed_tr.mean(0)[ks.index(k)]) for k in SHOW},
        "trained_std": {str(k): float(seed_tr.std(0)[ks.index(k)]) for k in SHOW},
        "init_mean": {str(k): float(seed_in.mean(0)[ks.index(k)]) for k in SHOW},
        "init_std": {str(k): float(seed_in.std(0)[ks.index(k)]) for k in SHOW},
        "scramble_mean": {str(k): float(seed_sc.mean(0)[ks.index(k)]) for k in SHOW},
        "scramble_std": {str(k): float(seed_sc.std(0)[ks.index(k)]) for k in SHOW},
        "delta": p_table,
        "verdict_any_positive": bool(any_pos),
    }
    dest = Path(__file__).resolve().parent / "artifacts" / "multiseed_resmlp.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
