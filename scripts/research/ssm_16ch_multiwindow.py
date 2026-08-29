#!/usr/bin/env python3
"""Multi-window 16-channel sedenion SSM benchmark across EEGMMIDB files.

Resolves Exp 13 single-window limitation: evaluates S-SSM-16ch vs S-SSM-scalar
vs Real-16ch-diagonal across ~2000+ windows from all available EDF files.

Architecture matches examples/sedenion_ssm_16ch_eeg.sio exactly:
  - S-SSM-16ch:  h_t = normalize(tanh(A(α=0.2) ⊗ h_{t-1} + x_t)),  x_t = 16ch sedenion
  - S-SSM-scalar: same SSM, x_t uses only CH0 (e0 component, rest zero)
  - Real-16ch:   h_t = tanh(diag(a) * h + W*x_t), h in R^16

Key optimization: A ⊗ h = M_A @ h  (precomputed 16×16 left-multiplication matrix).
This makes the inner loop O(16²) numpy instead of recursive Cayley-Dickson.

G₂ regularization hypothesis: spatial inductive bias from sedenion product
reduces train-test generalization gap vs scalar SSM.

Usage:
    python3 scripts/research/ssm_16ch_multiwindow.py [--edf-dir DIR] [--windows N]
"""

import os
import sys
import time
import argparse
import numpy as np

CH_NAMES = ["Fc5","Fc3","Fc1","Fcz","Fc2","Fc4","Fc6","C5",
            "C3","C1","Cz","C2","C4","C6","Cp5","Cp3"]
WIN_LEN   = 81
TRAIN_N   = 64
TEST_N    = 16
ALPHA     = 0.2
TRAIN_EPOCHS = 150
LR        = 0.01
SEED      = 42

# ---------------------------------------------------------------------------
# Sedenion multiplication table: e_i * e_j = sign * e_k
# Cayley-Dickson basis for 16D: precompute once.
# ---------------------------------------------------------------------------

def _cd_mul_vecs(a, b):
    """Cayley-Dickson product, shape (16,)×(16,) → (16,)."""
    n = a.shape[0] // 2
    if n == 1:
        ar, ai, br, bi = a[0], a[1], b[0], b[1]
        return np.array([ar*br - bi*ai, ar*bi + ai*br])
    a0, a1 = a[:n], a[n:]
    b0, b1 = b[:n], b[n:]
    # For Cayley-Dickson: (a0,a1)(b0,b1) = (a0*b0 - b1*conj(a1), a0*b1 + a1*conj(b0))
    # For real inputs: conj = identity → (a0*b0 - b1*a1, a0*b1 + a1*b0)
    c0 = _cd_mul_vecs(a0, b0) - _cd_mul_vecs(b1.copy(), a1)
    c1 = _cd_mul_vecs(a0, b1) + _cd_mul_vecs(a1, b0.copy())
    return np.concatenate([c0, c1])


def build_left_mul_matrix(A):
    """Build 16×16 matrix M such that M @ h = A ⊗ h for any h.
    Column j = A ⊗ e_j  where e_j is the j-th basis vector.
    """
    M = np.zeros((16, 16))
    for j in range(16):
        ej = np.zeros(16); ej[j] = 1.0
        M[:, j] = _cd_mul_vecs(A, ej)
    return M


def build_A(alpha=ALPHA):
    """Build fixed sedenion A(α=0.2) matching Sounio experiment init."""
    A = np.zeros(16)
    A[0] = alpha
    rest = np.ones(15) * 0.1
    A[1:] = rest
    n = float(np.sqrt(np.dot(A, A)))
    if n > 1e-9:
        A = A * (alpha / n)
    return A


# ---------------------------------------------------------------------------
# Model forward passes  (vectorized over time axis)
# ---------------------------------------------------------------------------

def ssm16_forward(M_A, x_seq, C):
    """S-SSM-16ch: h_t = normalize(tanh(M_A @ h_{t-1} + x_t)).
    x_seq: (T, 16);  C: (16,).
    Returns predictions (T,) and hidden states (T, 16).
    """
    T = x_seq.shape[0]
    h = np.ones(16); h /= (np.linalg.norm(h) + 1e-12)
    hs = np.empty((T, 16))
    for t in range(T):
        raw = M_A @ h + x_seq[t]
        act = np.tanh(raw)
        n = float(np.linalg.norm(act))
        h = act / (n + 1e-7) if n > 1e-7 else act
        hs[t] = h
    return hs @ C, hs


def ssm_scalar_forward(M_A, x0_seq, C):
    """S-SSM-scalar: only e0 component filled from CH0."""
    x_sed = np.zeros((len(x0_seq), 16))
    x_sed[:, 0] = x0_seq
    return ssm16_forward(M_A, x_sed, C)


def real_forward(a_diag, W, x_seq, C):
    """Real-16ch diagonal SSM: h_t = tanh(a_diag*h + W*x)."""
    T = x_seq.shape[0]
    h = np.zeros(16)
    hs = np.empty((T, 16))
    for t in range(T):
        h = np.tanh(a_diag * h + W @ x_seq[t])
        hs[t] = h
    return hs @ C, hs


def mse(pred, tgt):
    return float(np.mean((pred - tgt) ** 2))


# ---------------------------------------------------------------------------
# Training: C-only gradient descent (A, W, a_diag frozen for fair comparison)
# ---------------------------------------------------------------------------

def train_C(forward_fn, x_train, y_train, rng, epochs=TRAIN_EPOCHS, lr=LR, **fwd_kw):
    C = rng.normal(0, 0.1, 16)
    for _ in range(epochs):
        preds, hs = forward_fn(**fwd_kw, x_seq=x_train if 'x_seq' in forward_fn.__code__.co_varnames else x_train, C=C)
        err = preds - y_train
        C -= lr * (2.0 / len(y_train)) * (hs.T @ err)
    return C


def train_C_ssm16(M_A, x_train, y_train, rng, epochs=TRAIN_EPOCHS, lr=LR):
    C = rng.normal(0, 0.1, 16)
    for _ in range(epochs):
        preds, hs = ssm16_forward(M_A, x_train, C)
        err = preds - y_train
        C -= lr * (2.0 / TRAIN_N) * (hs.T @ err)
    return C


def train_C_scalar(M_A, x0_train, y_train, rng, epochs=TRAIN_EPOCHS, lr=LR):
    x_sed = np.zeros((TRAIN_N, 16)); x_sed[:, 0] = x0_train
    C = rng.normal(0, 0.1, 16)
    for _ in range(epochs):
        preds, hs = ssm16_forward(M_A, x_sed, C)
        err = preds - y_train
        C -= lr * (2.0 / TRAIN_N) * (hs.T @ err)
    return C


def train_real(x_train, y_train, rng, epochs=TRAIN_EPOCHS, lr=LR):
    a_diag = rng.normal(0, 0.1, 16)
    W = rng.normal(0, 0.1, (16, 16))
    C = rng.normal(0, 0.1, 16)
    for _ in range(epochs):
        preds, hs = real_forward(a_diag, W, x_train, C)
        err = preds - y_train
        C -= lr * (2.0 / TRAIN_N) * (hs.T @ err)
    return a_diag, W, C


# ---------------------------------------------------------------------------
# EDF loading and window extraction
# ---------------------------------------------------------------------------

def load_edf(path):
    try:
        import pyedflib
        f = pyedflib.EdfReader(path)
        n_ch = f.signals_in_file
        sigs = np.stack([f.readSignal(i) for i in range(min(n_ch, 16))])
        f.close()
        return sigs
    except Exception:
        return None


def extract_windows(sigs, stride=40):
    """Extract windows of shape (16, WIN_LEN) from (16, T) signal."""
    _, T = sigs.shape
    wins = []
    start = 0
    while start + WIN_LEN <= T:
        wins.append(sigs[:, start:start+WIN_LEN])
        start += stride
    return wins


def normalize_window(w):
    """Per-channel normalize by train-split max-abs. Returns (16,80) input, (80,) target."""
    data = w[:, :80]
    target_raw = w[0, 1:]         # CH0 one-step ahead
    scales = np.abs(data[:, :TRAIN_N]).max(axis=1) + 1e-6
    normed = data / scales[:, None]
    tscale = np.abs(target_raw[:TRAIN_N]).max() + 1e-6
    return normed, target_raw / tscale


# ---------------------------------------------------------------------------
# Per-window benchmark
# ---------------------------------------------------------------------------

def run_window(normed, target, M_A, rng):
    x_all = normed.T          # (80, 16)
    y_all = target            # (80,)
    x_train, y_train = x_all[:TRAIN_N], y_all[:TRAIN_N]
    x_test,  y_test  = x_all[TRAIN_N:], y_all[TRAIN_N:]

    # S-SSM-16ch
    C16 = train_C_ssm16(M_A, x_train, y_train, rng)
    tr16 = mse(ssm16_forward(M_A, x_train, C16)[0], y_train)
    te16 = mse(ssm16_forward(M_A, x_test,  C16)[0], y_test)

    # S-SSM-scalar
    x0_tr = x_train[:, 0]
    x_sc_tr = np.zeros((TRAIN_N, 16)); x_sc_tr[:, 0] = x0_tr
    x_sc_te = np.zeros((TEST_N,  16)); x_sc_te[:, 0] = x_test[:, 0]
    Csc = train_C_scalar(M_A, x0_tr, y_train, rng)
    trsc = mse(ssm16_forward(M_A, x_sc_tr, Csc)[0], y_train)
    tesc = mse(ssm16_forward(M_A, x_sc_te, Csc)[0], y_test)

    # Real-16ch diagonal
    ad, W, Cr = train_real(x_train, y_train, rng)
    trR = mse(real_forward(ad, W, x_train, Cr)[0], y_train)
    teR = mse(real_forward(ad, W, x_test,  Cr)[0], y_test)

    return {
        "tr16": tr16, "te16": te16, "gap16": te16 - tr16,
        "trsc": trsc, "tesc": tesc, "gapsc": tesc - trsc,
        "trR":  trR,  "teR":  teR,  "gapR":  teR - trR,
        "spatial_gain": tesc - te16,            # positive = 16ch wins
        "reg_ratio": (te16 - tr16) / (teR - trR + 1e-12),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf-dir", default="data/eegmmidb")
    ap.add_argument("--windows", type=int, default=0, help="Max windows (0=all)")
    ap.add_argument("--stride",  type=int, default=40)
    ap.add_argument("--output",  default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    A   = build_A(ALPHA)
    M_A = build_left_mul_matrix(A)
    print(f"A norm: {float(np.linalg.norm(A)):.4f}  |M_A|_F: {float(np.linalg.norm(M_A,'fro')):.4f}", flush=True)

    edf_files = sorted([
        os.path.join(args.edf_dir, f)
        for f in os.listdir(args.edf_dir) if f.endswith(".edf")
    ])
    print(f"Found {len(edf_files)} EDF files in {args.edf_dir}", flush=True)

    results = []
    n_total = 0
    t0 = time.time()

    for path in edf_files:
        sigs = load_edf(path)
        if sigs is None:
            continue
        for w in extract_windows(sigs, stride=args.stride):
            normed, target = normalize_window(w)
            if np.abs(target[:TRAIN_N]).max() < 1e-6:
                continue
            results.append(run_window(normed, target, M_A, rng))
            n_total += 1
            if n_total % 200 == 0:
                print(f"  [{n_total}] {time.time()-t0:.1f}s  "
                      f"sg={np.mean([r['spatial_gain'] for r in results]):.4f}", flush=True)
            if args.windows > 0 and n_total >= args.windows:
                break
        if args.windows > 0 and n_total >= args.windows:
            break

    elapsed = time.time() - t0
    print(f"\nTotal windows: {n_total}  elapsed: {elapsed:.1f}s  "
          f"({elapsed/max(n_total,1)*1000:.0f} ms/window)", flush=True)

    if not results:
        print("ERROR: no valid windows found", flush=True)
        return 1

    def stat(key):
        v = [r[key] for r in results]
        return np.mean(v), np.std(v), np.median(v)

    print("\n=== S-SSM-16ch ===")
    m,s,md = stat("tr16");  print(f"  train MSE: {m:.4f} ± {s:.4f}  (med {md:.4f})")
    m,s,md = stat("te16");  print(f"  test  MSE: {m:.4f} ± {s:.4f}  (med {md:.4f})")
    m,s,md = stat("gap16"); print(f"  gap:       {m:.4f} ± {s:.4f}  (med {md:.4f})")

    print("\n=== S-SSM-scalar ===")
    m,s,md = stat("trsc");  print(f"  train MSE: {m:.4f} ± {s:.4f}  (med {md:.4f})")
    m,s,md = stat("tesc");  print(f"  test  MSE: {m:.4f} ± {s:.4f}  (med {md:.4f})")
    m,s,md = stat("gapsc"); print(f"  gap:       {m:.4f} ± {s:.4f}  (med {md:.4f})")

    print("\n=== Real-16ch diagonal ===")
    m,s,md = stat("trR");   print(f"  train MSE: {m:.4f} ± {s:.4f}  (med {md:.4f})")
    m,s,md = stat("teR");   print(f"  test  MSE: {m:.4f} ± {s:.4f}  (med {md:.4f})")
    m,s,md = stat("gapR");  print(f"  gap:       {m:.4f} ± {s:.4f}  (med {md:.4f})")

    sg_vals = [r["spatial_gain"] for r in results]
    n_pos   = sum(1 for v in sg_vals if v > 0)
    rr_vals = [r["reg_ratio"]    for r in results]

    print("\n=== Key Metrics ===")
    print(f"  spatial_gain mean: {np.mean(sg_vals):.4f} ± {np.std(sg_vals):.4f}  "
          f"(med {np.median(sg_vals):.4f})")
    print(f"  16ch < scalar: {n_pos}/{n_total}  ({100*n_pos/n_total:.1f}%)")
    print(f"  reg_ratio mean: {np.mean(rr_vals):.3f} ± {np.std(rr_vals):.3f}  "
          f"(med {np.median(rr_vals):.3f})")

    gap16_m = np.mean([r["gap16"] for r in results])
    gapR_m  = np.mean([r["gapR"]  for r in results])
    print(f"  mean gap16: {gap16_m:.4f}  mean gapR: {gapR_m:.4f}  "
          f"ratio: {gap16_m/(gapR_m+1e-12):.3f}x")

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump({
                "n_windows": n_total, "elapsed_s": elapsed,
                "alpha": ALPHA, "train_n": TRAIN_N, "test_n": TEST_N,
                "epochs": TRAIN_EPOCHS, "lr": LR,
                "spatial_gain_mean": float(np.mean(sg_vals)),
                "spatial_gain_std":  float(np.std(sg_vals)),
                "n_16ch_better": n_pos,
                "reg_ratio_mean": float(np.mean(rr_vals)),
                "gap16_mean": float(gap16_m),
                "gapR_mean":  float(gapR_m),
                "te16_mean":  float(np.mean([r["te16"] for r in results])),
                "tesc_mean":  float(np.mean([r["tesc"] for r in results])),
                "teR_mean":   float(np.mean([r["teR"]  for r in results])),
            }, f, indent=2)
        print(f"Results → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
