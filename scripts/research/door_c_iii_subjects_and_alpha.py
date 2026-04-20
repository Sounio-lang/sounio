#!/usr/bin/env python3
"""Door C + III: Extend subjects to chb06/chb11, then α sweep.

Door C — Run Door I protocol on all 5 subjects:
  chb01, chb03, chb05 (confirmed from Door I Sounio experiments)
  chb06, chb11 (new — first appearance in this benchmark)

Door III — Sweep α ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80}
  across all subjects. Metric: seizure sensitivity ratio = MSE_ictal / MSE_test_ii.
  Door H showed linear resonance does NOT explain the α=0.2 optimum.
  This sweep reveals the nonlinear sensitivity landscape.

Architecture: identical to Door I Sounio experiments.
  - S-SSM-16ch α: sedenion gate, CH0 next-step prediction
  - S-SSM-scalar: only e0 filled (no spatial coupling)
  - Real-16ch-diagonal: linear diagonal AR baseline
  - Train: 64-sample interictal window (t=10s into file)
  - Test-II: held-out interictal (samples 64-79)
  - Test-IC: ictal onset (64 samples, fresh hidden state)
  - Metric: ratio = MSE_ictal / MSE_test_ii

Usage:
    python3 scripts/research/door_c_iii_subjects_and_alpha.py
"""

import os
import sys
import numpy as np

try:
    import pyedflib
except ImportError:
    print("ERROR: pyedflib not installed.", file=sys.stderr); sys.exit(1)

# ─── Subject registry ────────────────────────────────────────────────────────

SUBJECTS = {
    "chb01": {"edf": "data/chbmit/chb01_03.edf", "seizure_start_s": 2996, "n_ch_edf": 23},
    "chb03": {"edf": "data/chbmit/chb03_01.edf", "seizure_start_s": 362,  "n_ch_edf": 23},
    "chb05": {"edf": "data/chbmit/chb05_06.edf", "seizure_start_s": 417,  "n_ch_edf": 23},
    "chb06": {"edf": "data/chbmit/chb06_04.edf", "seizure_start_s": 327,  "n_ch_edf": 23},
    "chb11": {"edf": "data/chbmit/chb11_82.edf", "seizure_start_s": 298,  "n_ch_edf": 28},
}

INTERICTAL_S = 10.0
N_TRAIN      = 64
N_TEST_II    = 16
N_TEST_IC    = 64
N_CH         = 16
FS           = 256
TRAIN_EPOCHS = 150
LR           = 0.01
SEED         = 42

ALPHA_DEFAULT = 0.2
ALPHA_SWEEP   = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]

# ─── Sedenion arithmetic ────────────────────────────────────────────────────

def _cd_mul(a, b):
    n = a.shape[0] // 2
    if n == 1:
        return np.array([a[0]*b[0] - b[1]*a[1], a[0]*b[1] + a[1]*b[0]])
    a0, a1 = a[:n], a[n:]
    b0, b1 = b[:n], b[n:]
    return np.concatenate([
        _cd_mul(a0, b0) - _cd_mul(b1.copy(), a1),
        _cd_mul(a0, b1) + _cd_mul(a1, b0.copy()),
    ])


def build_left_mul_matrix(A):
    M = np.zeros((16, 16))
    for j in range(16):
        ej = np.zeros(16); ej[j] = 1.0
        M[:, j] = _cd_mul(A, ej)
    return M


def build_A(alpha):
    A = np.zeros(16)
    A[0] = alpha
    A[1:] = 0.1
    n = float(np.sqrt(np.dot(A, A)))
    if n > 1e-9:
        A = A * (alpha / n)
    return A

# ─── Model forward passes ────────────────────────────────────────────────────

def ssm16_forward(M_A, x_seq, C):
    T = x_seq.shape[0]
    h = np.ones(16); h /= (np.linalg.norm(h) + 1e-12)
    hs = np.empty((T, 16))
    for t in range(T):
        raw = M_A @ h + x_seq[t]
        act = np.tanh(raw)
        nn = float(np.linalg.norm(act))
        h = act / (nn + 1e-7) if nn > 1e-7 else act
        hs[t] = h
    return hs @ C, hs


def ssm_scalar_forward(M_A, x0_seq, C):
    x_sed = np.zeros((len(x0_seq), 16)); x_sed[:, 0] = x0_seq
    return ssm16_forward(M_A, x_sed, C)


def real_forward(a_diag, W, x_seq, C):
    T = x_seq.shape[0]
    h = np.zeros(16); hs = np.empty((T, 16))
    for t in range(T):
        h = np.tanh(a_diag * h + W @ x_seq[t]); hs[t] = h
    return hs @ C, hs


def mse(pred, tgt):
    return float(np.mean((pred - tgt) ** 2))

# ─── Training ────────────────────────────────────────────────────────────────

def train_ssm16(M_A, x_tr, y_tr, rng):
    C = rng.normal(0, 0.1, 16)
    n = len(y_tr)
    for _ in range(TRAIN_EPOCHS):
        p, hs = ssm16_forward(M_A, x_tr, C)
        C -= LR * (2.0/n) * (hs.T @ (p - y_tr))
    return C


def train_scalar(M_A, x0_tr, y_tr, rng):
    x_sed = np.zeros((len(x0_tr), 16)); x_sed[:, 0] = x0_tr
    C = rng.normal(0, 0.1, 16)
    n = len(y_tr)
    for _ in range(TRAIN_EPOCHS):
        p, hs = ssm16_forward(M_A, x_sed, C)
        C -= LR * (2.0/n) * (hs.T @ (p - y_tr))
    return C


def train_real(x_tr, y_tr, rng):
    a = rng.normal(0, 0.1, 16)
    W = rng.normal(0, 0.1, (16, 16))
    C = rng.normal(0, 0.1, 16)
    n = len(y_tr)
    for _ in range(TRAIN_EPOCHS):
        p, hs = real_forward(a, W, x_tr, C)
        C -= LR * (2.0/n) * (hs.T @ (p - y_tr))
    return a, W, C

# ─── Data loading ────────────────────────────────────────────────────────────

def load_subject(cfg):
    edf_path = cfg["edf"]
    if not os.path.exists(edf_path):
        return None
    f = pyedflib.EdfReader(edf_path)
    n = min(f.signals_in_file, N_CH)
    sigs = np.zeros((N_CH, f.getNSamples()[0]))
    for i in range(n):
        s = f.readSignal(i)
        sigs[i, :len(s)] = s
    f.close()
    return sigs  # (16, T)


def prepare_windows(sigs, seizure_s):
    """Normalize and build train/test_ii/test_ic windows matching Door I."""
    fs = FS
    ii_start = int(INTERICTAL_S * fs)
    ic_start = int(seizure_s * fs)
    N_TOTAL_II = N_TRAIN + N_TEST_II

    ii_raw = sigs[:, ii_start : ii_start + N_TOTAL_II]   # (16, 80)
    ic_raw = sigs[:, ic_start : ic_start + N_TEST_IC]    # (16, 64)

    means   = ii_raw[:, :N_TRAIN].mean(axis=1, keepdims=True)
    stds    = ii_raw[:, :N_TRAIN].std(axis=1, keepdims=True)
    stds    = np.where(stds < 1e-6, 1.0, stds)
    ii_z    = (ii_raw - means) / stds
    ic_z    = (ic_raw - means) / stds

    max_abs = np.abs(ii_z[:, :N_TRAIN]).max(axis=1, keepdims=True)
    max_abs = np.where(max_abs < 1e-6, 1.0, max_abs)

    ii_norm = ii_z / max_abs
    ic_norm = np.clip(ic_z / max_abs, -5.0, 5.0)

    # Target: CH0 next-step
    def targets(norm, n):
        t = np.zeros(n)
        t[:n-1] = norm[0, 1:n]; t[n-1] = norm[0, n-1]
        return t

    return {
        "x_train":    ii_norm[:, :N_TRAIN].T,            # (64, 16)
        "y_train":    targets(ii_norm, N_TRAIN),
        "x_test_ii":  ii_norm[:, N_TRAIN:].T,            # (16, 16)
        "y_test_ii":  targets(ii_norm[:, N_TRAIN:], N_TEST_II),
        "x_test_ic":  ic_norm.T,                          # (64, 16)
        "y_test_ic":  targets(ic_norm, N_TEST_IC),
        "ictal_std":  float(ic_norm.std()),
        "ii_std":     float(ii_norm.std()),
    }


# ─── Single-α run for one subject ────────────────────────────────────────────

def run_one(windows, alpha, rng_seed=SEED):
    """Run Door I protocol. Returns dict of MSE metrics."""
    rng = np.random.RandomState(rng_seed)
    A   = build_A(alpha)
    M_A = build_left_mul_matrix(A)

    x_tr = windows["x_train"]; y_tr = windows["y_train"]
    x_ii = windows["x_test_ii"]; y_ii = windows["y_test_ii"]
    x_ic = windows["x_test_ic"]; y_ic = windows["y_test_ic"]

    # S-SSM-16ch
    C_ssm = train_ssm16(M_A, x_tr, y_tr, rng)
    m_tr_ssm  = mse(ssm16_forward(M_A, x_tr, C_ssm)[0], y_tr)
    m_ii_ssm  = mse(ssm16_forward(M_A, x_ii, C_ssm)[0], y_ii)
    m_ic_ssm  = mse(ssm16_forward(M_A, x_ic, C_ssm)[0], y_ic)

    # S-SSM-scalar
    C_sc = train_scalar(M_A, x_tr[:, 0], y_tr, rng)
    m_ii_sc   = mse(ssm_scalar_forward(M_A, x_ii[:, 0], C_sc)[0], y_ii)
    m_ic_sc   = mse(ssm_scalar_forward(M_A, x_ic[:, 0], C_sc)[0], y_ic)

    # Real diagonal
    a_d, W_d, C_d = train_real(x_tr, y_tr, rng)
    m_ii_diag = mse(real_forward(a_d, W_d, x_ii, C_d)[0], y_ii)
    m_ic_diag = mse(real_forward(a_d, W_d, x_ic, C_d)[0], y_ic)

    # G₂ amplification: how much MORE does S-SSM jump vs diagonal
    ratio_ssm  = m_ic_ssm  / (m_ii_ssm  + 1e-12)
    ratio_sc   = m_ic_sc   / (m_ii_sc   + 1e-12)
    ratio_diag = m_ic_diag / (m_ii_diag + 1e-12)
    g2_amp     = ratio_ssm - ratio_diag

    return {
        "train_ssm": m_tr_ssm,
        "ii_ssm":    m_ii_ssm,  "ic_ssm":  m_ic_ssm,  "ratio_ssm":  ratio_ssm,
        "ii_sc":     m_ii_sc,   "ic_sc":   m_ic_sc,   "ratio_sc":   ratio_sc,
        "ii_diag":   m_ii_diag, "ic_diag": m_ic_diag, "ratio_diag": ratio_diag,
        "g2_amp":    g2_amp,
    }


# ─── Door C: all subjects at α=0.2 ──────────────────────────────────────────

def door_c(all_windows):
    print(f"\n{'═'*76}")
    print("  DOOR C — All 5 subjects at α=0.2 (Door I protocol in Python)")
    print(f"{'─'*76}")
    print(f"  {'subj':>6}  {'ratio_ssm':>11}  {'ratio_sc':>10}  {'ratio_diag':>11}  "
          f"{'G₂_amp':>8}  {'ictal_std':>10}  verdict")
    print(f"{'─'*76}")

    door_i_expected = {"chb01": 5.90, "chb03": 166.9, "chb05": 3.08}

    for subj, windows in sorted(all_windows.items()):
        r = run_one(windows, ALPHA_DEFAULT)
        verdict = "PASS" if r["ratio_ssm"] > r["ratio_diag"] else "FAIL"
        note = ""
        if subj in door_i_expected:
            note = f" (Sounio ref: {door_i_expected[subj]:.1f}×)"
        print(f"  {subj:>6}  {r['ratio_ssm']:>11.3f}×  {r['ratio_sc']:>10.3f}×  "
              f"{r['ratio_diag']:>11.3f}×  {r['g2_amp']:>+8.3f}  "
              f"{windows['ictal_std']:>10.3f}  {verdict}{note}")


# ─── Door III: α sweep ───────────────────────────────────────────────────────

def door_iii(all_windows):
    print(f"\n{'═'*80}")
    print("  DOOR III — α sweep (nonlinear sensitivity landscape)")
    print("  Metric: MSE_ictal / MSE_test_ii  (Door I protocol)")
    print(f"{'─'*80}")

    # Header
    hdr = f"  {'subj':>6}  " + "  ".join(f"α={a:.2f}  " for a in ALPHA_SWEEP)
    print(hdr)
    print(f"{'─'*80}")

    # Per-subject row: ratio_ssm at each α
    subject_data = {}
    for subj, windows in sorted(all_windows.items()):
        ratios = []
        for alpha in ALPHA_SWEEP:
            r = run_one(windows, alpha)
            ratios.append(r["ratio_ssm"])
        subject_data[subj] = ratios
        best_idx = int(np.argmax(ratios))
        best_a   = ALPHA_SWEEP[best_idx]
        row = f"  {subj:>6}  " + "  ".join(f"{v:7.3f}×  " for v in ratios)
        row += f"  ← best α={best_a:.2f}"
        print(row)

    # Column-wise: mean across subjects at each α
    print(f"{'─'*80}")
    means = []
    for i, alpha in enumerate(ALPHA_SWEEP):
        vals = [subject_data[s][i] for s in sorted(all_windows)]
        means.append(np.mean(vals))
    mean_row = f"  {'mean':>6}  " + "  ".join(f"{v:7.3f}×  " for v in means)
    best_mean_idx = int(np.argmax(means))
    mean_row += f"  ← best mean α={ALPHA_SWEEP[best_mean_idx]:.2f}"
    print(mean_row)
    print(f"{'─'*80}")

    # Detailed breakdown at α=0.2 and best-mean α
    best_alpha = ALPHA_SWEEP[best_mean_idx]
    if best_alpha != ALPHA_DEFAULT:
        compare_alphas = [ALPHA_DEFAULT, best_alpha]
        print(f"\n  Comparison: α={ALPHA_DEFAULT} vs α={best_alpha} (best mean)")
        print(f"  {'subj':>6}  {'ssm α=0.20':>12}  {'diag α=0.20':>12}  "
              f"{'ssm best_α':>12}  {'diag best_α':>12}  {'Δg2_amp':>10}")
        print(f"  {'─'*74}")
        for subj, windows in sorted(all_windows.items()):
            r1 = run_one(windows, ALPHA_DEFAULT)
            r2 = run_one(windows, best_alpha)
            delta = r2["g2_amp"] - r1["g2_amp"]
            print(f"  {subj:>6}  {r1['ratio_ssm']:>12.3f}×  {r1['ratio_diag']:>12.3f}×  "
                  f"{r2['ratio_ssm']:>12.3f}×  {r2['ratio_diag']:>12.3f}×  {delta:>+10.3f}")

    # Monotonicity check: does ratio grow monotonically with α?
    print(f"\n  Monotonicity across α (S-SSM mean ratios):")
    print(f"  " + " → ".join(f"α={a:.2f}:{m:.2f}×" for a, m in zip(ALPHA_SWEEP, means)))
    diffs = np.diff(means)
    if all(d > 0 for d in diffs):
        print("  MONOTONE INCREASING — higher α always better")
    elif all(d < 0 for d in diffs):
        print("  MONOTONE DECREASING — lower α always better")
    else:
        peak_idx = int(np.argmax(means))
        print(f"  NON-MONOTONE — peak at α={ALPHA_SWEEP[peak_idx]:.2f} "
              f"(ratio={means[peak_idx]:.3f}×)")
        if peak_idx > 0 and peak_idx < len(ALPHA_SWEEP) - 1:
            print(f"  *** INTERIOR OPTIMUM — nonlinear mechanism confirmed ***")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading EDF files...")
    all_windows = {}
    for subj, cfg in sorted(SUBJECTS.items()):
        sigs = load_subject(cfg)
        if sigs is None:
            print(f"  [SKIP] {subj}: {cfg['edf']} not found")
            continue
        windows = prepare_windows(sigs, cfg["seizure_start_s"])
        all_windows[subj] = windows
        print(f"  {subj}: loaded  ictal_std={windows['ictal_std']:.3f}  ii_std={windows['ii_std']:.3f}")

    if not all_windows:
        print("No subjects loaded."); sys.exit(1)

    print(f"\n{len(all_windows)} subjects loaded: {sorted(all_windows)}")

    door_c(all_windows)
    door_iii(all_windows)


if __name__ == "__main__":
    main()
