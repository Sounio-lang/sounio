#!/usr/bin/env python3
"""Multi-seed untrained LSTM init panel (H=40, matching the primary run).

Pure numpy analytic h→h / c→c Jacobians — no torch. Answers the §4.2 / §6.1
FILL: is the high init alignment stable across seeds?

Output: docs/research/artifacts/multiseed_lstm_init.json (+ stdout table).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

np.seterr(all="ignore")

sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def lstm_init(H: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    k = 1.0 / np.sqrt(H)
    return dict(
        Wih=rng.uniform(-k, k, (4 * H, 2)),
        Whh=rng.uniform(-k, k, (4 * H, H)),
        bih=rng.uniform(-k, k, 4 * H),
        bhh=rng.uniform(-k, k, 4 * H),
        H=H,
    )


def gen_adding(T: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.random(T).astype("float32")
    mk = np.zeros(T, "float32")
    mk[rng.choice(T, 2, replace=False)] = 1.0
    return np.stack([v, mk], -1)


def Vts(W: dict, X: np.ndarray):
    """Right singular vectors of per-step h→h and c→c Jacobians along X."""
    H = W["H"]
    Whi = W["Whh"][:H]
    Whf = W["Whh"][H : 2 * H]
    Whg = W["Whh"][2 * H : 3 * H]
    Who = W["Whh"][3 * H : 4 * H]
    h = np.zeros(H)
    c = np.zeros(H)
    VH, VC = [], []
    for t in range(X.shape[0]):
        pre = W["Wih"] @ X[t] + W["Whh"] @ h + W["bih"] + W["bhh"]
        i = sig(pre[:H])
        f = sig(pre[H : 2 * H])
        g = np.tanh(pre[2 * H : 3 * H])
        o = sig(pre[3 * H : 4 * H])
        cp = f * c + i * g
        tc = np.tanh(cp)
        dtc = 1 - tc * tc
        di = (i * (1 - i))[:, None] * Whi
        df = (f * (1 - f))[:, None] * Whf
        dg = (1 - g * g)[:, None] * Whg
        do = (o * (1 - o))[:, None] * Who
        dcp_dh = c[:, None] * df + g[:, None] * di + i[:, None] * dg
        dhp_dh = tc[:, None] * do + (o * dtc)[:, None] * dcp_dh
        VH.append(np.linalg.svd(dhp_dh, full_matrices=False)[2])
        VC.append(np.linalg.svd(np.diag(f), full_matrices=False)[2])
        h, c = o * tc, cp
    return VH, VC


def align_from_Vt(Vt_list, ks):
    out = []
    for k in ks:
        cs = [
            np.linalg.svd(Vt_list[l][-k:] @ Vt_list[l + 1][-k:].T, compute_uv=False).mean()
            for l in range(len(Vt_list) - 1)
        ]
        out.append(float(np.mean(cs)))
    return np.array(out)


def main():
    H, T = 40, 30
    n_seeds, n_seq = 16, 16
    show = [1, 2, 3, 4, 6, 8, 12]
    ks = list(range(1, H))
    base = np.sqrt(np.array(ks) / H)

    seed_hh = []  # (n_seeds, len(ks))
    seed_cc = []
    for s in range(n_seeds):
        W = lstm_init(H, seed=1000 + s)
        HH, CC = [], []
        for q in range(n_seq):
            X = gen_adding(T, np.random.default_rng(20000 + 100 * s + q))
            vh, vc = Vts(W, X)
            HH.append(align_from_Vt(vh, ks))
            CC.append(align_from_Vt(vc, ks))
        seed_hh.append(np.mean(HH, axis=0))
        seed_cc.append(np.mean(CC, axis=0))
        print(f"  seed {s + 1}/{n_seeds} done  hh@4={seed_hh[-1][3]:.3f}", flush=True)

    seed_hh = np.array(seed_hh)
    seed_cc = np.array(seed_cc)
    mean_hh, std_hh = seed_hh.mean(0), seed_hh.std(0)
    mean_cc, std_cc = seed_cc.mean(0), seed_cc.std(0)

    print(f"\nUNTRAINED LSTM multi-seed  H={H} T={T}  n_seeds={n_seeds} n_seq={n_seq}/seed")
    print("  k        " + " ".join(f"{k:>7}" for k in show))
    print("  baseline " + " ".join(f"{np.sqrt(k / H):7.3f}" for k in show))
    print(
        "  INIT hh  "
        + " ".join(f"{mean_hh[ks.index(k)]:7.3f}" for k in show)
        + "   mean over seeds"
    )
    print("  ±1 sd    " + " ".join(f"{std_hh[ks.index(k)]:7.3f}" for k in show))
    print("  INIT cc  " + " ".join(f"{mean_cc[ks.index(k)]:7.3f}" for k in show))
    print("  ±1 sd    " + " ".join(f"{std_cc[ks.index(k)]:7.3f}" for k in show))
    # decisive: init@4 across seeds
    a4 = seed_hh[:, 3]
    print(
        f"\n  INIT h→h @k=4 across seeds: mean={a4.mean():.3f}  sd={a4.std():.3f}  "
        f"min={a4.min():.3f}  max={a4.max():.3f}"
    )
    print(
        f"  fraction of seeds with INIT@4 > 0.90: {(a4 > 0.90).mean():.2f}  "
        f"> 0.95: {(a4 > 0.95).mean():.2f}"
    )
    print(
        "  → architectural (high at every seed) if min(INIT@4) ≫ trained reported 0.92 "
        f"and baseline {np.sqrt(4 / H):.2f}"
    )

    out = {
        "H": H,
        "T": T,
        "n_seeds": n_seeds,
        "n_seq_per_seed": n_seq,
        "show_k": show,
        "baseline": {str(k): float(np.sqrt(k / H)) for k in show},
        "init_hh_mean": {str(k): float(mean_hh[ks.index(k)]) for k in show},
        "init_hh_std": {str(k): float(std_hh[ks.index(k)]) for k in show},
        "init_cc_mean": {str(k): float(mean_cc[ks.index(k)]) for k in show},
        "init_cc_std": {str(k): float(std_cc[ks.index(k)]) for k in show},
        "init_hh_k4_per_seed": [float(x) for x in a4],
        "init_hh_k4_summary": {
            "mean": float(a4.mean()),
            "std": float(a4.std()),
            "min": float(a4.min()),
            "max": float(a4.max()),
            "frac_gt_0.90": float((a4 > 0.90).mean()),
            "frac_gt_0.95": float((a4 > 0.95).mean()),
        },
    }
    dest = Path(__file__).resolve().parent / "artifacts" / "multiseed_lstm_init.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
