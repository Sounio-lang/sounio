#!/usr/bin/env python3
"""Door II: Pre-ictal ramp detection via S-SSM-16ch on CHB-MIT.

Hypothesis: G₂ sedenion SSM MSE rises *before* seizure onset (pre-ictal ramp)
because the model learns inter-ictal spatial coupling; the ramp tracks the
structural dissolution of that coupling in the minutes/seconds preceding onset.

Architecture: identical to Door I (S-SSM-16ch α=0.2, CH0 next-step prediction).

Protocol:
  1. Train on interictal window (t=10s, 64 samples)
  2. Slide evaluation window (80 samples, stride 256=1s) from t_onset-60s to t_onset+10s
  3. For each eval window: compute MSE vs CH0 next-step, for all 3 models
  4. Report temporal MSE profile binned at 5s resolution

Output columns:
  time_s (relative to onset) | mse_ssm16 | mse_scalar | mse_diag | ratio_ssm | ratio_diag

Null: MSE flat until t=0
Alt:  S-SSM MSE rises 5–30s before onset (pre-ictal ramp)

Usage:
    python3 scripts/research/door_ii_preictal_ramp.py [--subjects chb01,chb03,chb05]
"""

import os
import sys
import argparse
import numpy as np

try:
    import pyedflib
except ImportError:
    print("ERROR: pyedflib not installed. Run: pip install pyedflib", file=sys.stderr)
    sys.exit(1)

# ─── Config ─────────────────────────────────────────────────────────────────

SUBJECTS = {
    "chb01": {"edf": "data/chbmit/chb01_03.edf", "seizure_start_s": 2996},
    "chb03": {"edf": "data/chbmit/chb03_01.edf", "seizure_start_s": 362},
    "chb05": {"edf": "data/chbmit/chb05_06.edf", "seizure_start_s": 417},
}

N_CH             = 16      # channels to use (0–15 from 23-ch EDF)
FS               = 256     # Hz
ALPHA            = 0.2     # sedenion gate parameter
N_TRAIN          = 64      # inter-ictal training samples
N_WINDOW         = 80      # evaluation window width (same as Door I)
STRIDE_S         = 1.0     # evaluation stride in seconds
PRE_SCAN_S       = 60.0    # scan from this many seconds before onset
POST_SCAN_S      = 10.0    # scan to this many seconds after onset
INTERICTAL_S     = 10.0    # start of inter-ictal training window in file
TRAIN_EPOCHS     = 150
LR               = 0.01
SEED             = 42
BIN_S            = 5.0     # reporting bin width in seconds

# ─── Sedenion arithmetic ────────────────────────────────────────────────────

def _cd_mul(a, b):
    """Cayley-Dickson product, recursive. a, b: float64 arrays of power-of-2 length."""
    n = a.shape[0] // 2
    if n == 1:
        ar, ai, br, bi = a[0], a[1], b[0], b[1]
        return np.array([ar*br - bi*ai, ar*bi + ai*br])
    a0, a1 = a[:n], a[n:]
    b0, b1 = b[:n], b[n:]
    c0 = _cd_mul(a0, b0) - _cd_mul(b1.copy(), a1)
    c1 = _cd_mul(a0, b1) + _cd_mul(a1, b0.copy())
    return np.concatenate([c0, c1])


def build_left_mul_matrix(A):
    """16×16 matrix M such that M @ h = A ⊗ h for any h in R^16."""
    M = np.zeros((16, 16))
    for j in range(16):
        ej = np.zeros(16); ej[j] = 1.0
        M[:, j] = _cd_mul(A, ej)
    return M


def build_A(alpha=ALPHA):
    """Build normalized sedenion A matching Door I / Exp 13 init."""
    A = np.zeros(16)
    A[0] = alpha
    A[1:] = 0.1
    n = float(np.sqrt(np.dot(A, A)))
    if n > 1e-9:
        A = A * (alpha / n)
    return A

# ─── Model forward passes ────────────────────────────────────────────────────

def ssm16_forward(M_A, x_seq, C):
    """S-SSM-16ch: h_t = normalize(tanh(M_A @ h_{t-1} + x_t)).
    x_seq: (T, 16); returns predictions (T,), hidden states (T, 16).
    """
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
    """S-SSM-scalar: only e0 filled from CH0, rest zero."""
    x_sed = np.zeros((len(x0_seq), 16))
    x_sed[:, 0] = x0_seq
    return ssm16_forward(M_A, x_sed, C)


def real_forward(a_diag, W, x_seq, C):
    """Real-16ch diagonal SSM: h_t = tanh(a_diag*h + W @ x_t)."""
    T = x_seq.shape[0]
    h = np.zeros(16)
    hs = np.empty((T, 16))
    for t in range(T):
        h = np.tanh(a_diag * h + W @ x_seq[t])
        hs[t] = h
    return hs @ C, hs


def mse(pred, tgt):
    return float(np.mean((pred - tgt) ** 2))


# ─── Training (C-only, A frozen) ────────────────────────────────────────────

def train_ssm16(M_A, x_train, y_train, rng):
    C = rng.normal(0, 0.1, 16)
    n = len(y_train)
    for _ in range(TRAIN_EPOCHS):
        preds, hs = ssm16_forward(M_A, x_train, C)
        err = preds - y_train
        C -= LR * (2.0 / n) * (hs.T @ err)
    return C


def train_scalar(M_A, x0_train, y_train, rng):
    x_sed = np.zeros((len(x0_train), 16)); x_sed[:, 0] = x0_train
    C = rng.normal(0, 0.1, 16)
    n = len(y_train)
    for _ in range(TRAIN_EPOCHS):
        preds, hs = ssm16_forward(M_A, x_sed, C)
        err = preds - y_train
        C -= LR * (2.0 / n) * (hs.T @ err)
    return C


def train_real(x_train, y_train, rng):
    a_diag = rng.normal(0, 0.1, 16)
    W = rng.normal(0, 0.1, (16, 16))
    C = rng.normal(0, 0.1, 16)
    n = len(y_train)
    for _ in range(TRAIN_EPOCHS):
        preds, hs = real_forward(a_diag, W, x_train, C)
        err = preds - y_train
        C -= LR * (2.0 / n) * (hs.T @ err)
    return a_diag, W, C


# ─── Data loading ────────────────────────────────────────────────────────────

def load_edf(edf_path):
    if not os.path.exists(edf_path):
        return None
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    sigs = np.stack([f.readSignal(i) for i in range(n)])
    f.close()
    return sigs  # (n_ch_edf, T)


def normalize_window(raw_window, means, stds, max_abs):
    """Apply the inter-ictal normalization to an arbitrary window."""
    z = (raw_window - means) / stds
    normed = z / max_abs
    return np.clip(normed, -5.0, 5.0)


# ─── Per-subject analysis ────────────────────────────────────────────────────

def analyze_subject(subject, cfg, rng):
    edf_path = cfg["edf"]
    seizure_s = cfg["seizure_start_s"]

    sigs_all = load_edf(edf_path)
    if sigs_all is None:
        print(f"  [SKIP] {subject}: {edf_path} not found", file=sys.stderr)
        return None

    n_ch_edf, T_total = sigs_all.shape
    print(f"\n{'─'*60}", file=sys.stderr)
    print(f"  {subject}: {edf_path} ({n_ch_edf}ch, {T_total/FS:.1f}s)", file=sys.stderr)

    # ── Build inter-ictal training data ──────────────────────────────────────
    ii_start = int(INTERICTAL_S * FS)
    sigs = sigs_all[:N_CH]  # keep first 16 channels

    ii_raw = sigs[:, ii_start : ii_start + N_TRAIN]   # (16, 64)

    means   = ii_raw.mean(axis=1, keepdims=True)
    stds    = ii_raw.std(axis=1, keepdims=True)
    stds    = np.where(stds < 1e-6, 1.0, stds)

    ii_z       = (ii_raw - means) / stds
    max_abs    = np.abs(ii_z).max(axis=1, keepdims=True)
    max_abs    = np.where(max_abs < 1e-6, 1.0, max_abs)

    ii_norm    = ii_z / max_abs   # (16, 64)

    x_train    = ii_norm.T        # (64, 16)
    y_train    = np.zeros(N_TRAIN)
    y_train[:N_TRAIN - 1] = ii_norm[0, 1:]
    y_train[N_TRAIN - 1]  = ii_norm[0, -1]

    # ── Train models ─────────────────────────────────────────────────────────
    A   = build_A(ALPHA)
    M_A = build_left_mul_matrix(A)

    C_ssm16  = train_ssm16(M_A, x_train, y_train, rng)
    C_scalar = train_scalar(M_A, x_train[:, 0], y_train, rng)
    a_diag, W_diag, C_diag = train_real(x_train, y_train, rng)

    train_mse_ssm    = mse(ssm16_forward(M_A, x_train, C_ssm16)[0], y_train)
    train_mse_scalar = mse(ssm_scalar_forward(M_A, x_train[:, 0], C_scalar)[0], y_train)
    train_mse_diag   = mse(real_forward(a_diag, W_diag, x_train, C_diag)[0], y_train)

    print(f"  Train MSE — ssm16={train_mse_ssm:.4f}  scalar={train_mse_scalar:.4f}  diag={train_mse_diag:.4f}",
          file=sys.stderr)

    # ── Sliding evaluation ───────────────────────────────────────────────────
    ic_start   = int(seizure_s * FS)
    scan_start = max(0, ic_start - int(PRE_SCAN_S * FS))
    scan_end   = min(T_total, ic_start + int(POST_SCAN_S * FS) + N_WINDOW)
    stride     = int(STRIDE_S * FS)

    results = []  # (time_rel_s, mse_ssm16, mse_scalar, mse_diag)

    pos = scan_start
    while pos + N_WINDOW <= scan_end:
        raw_w  = sigs[:, pos : pos + N_WINDOW]   # (16, 80)
        w_norm = normalize_window(raw_w, means, stds, max_abs)  # (16, 80)

        x_eval = w_norm.T                         # (80, 16)
        y_eval = np.zeros(N_WINDOW)
        y_eval[:N_WINDOW - 1] = w_norm[0, 1:]
        y_eval[N_WINDOW - 1]  = w_norm[0, -1]

        m_ssm    = mse(ssm16_forward(M_A, x_eval, C_ssm16)[0], y_eval)
        m_scalar = mse(ssm_scalar_forward(M_A, x_eval[:, 0], C_scalar)[0], y_eval)
        m_diag   = mse(real_forward(a_diag, W_diag, x_eval, C_diag)[0], y_eval)

        center   = pos + N_WINDOW // 2
        time_rel = (center - ic_start) / FS      # seconds relative to seizure onset

        results.append((time_rel, m_ssm, m_scalar, m_diag))
        pos += stride

    print(f"  Eval windows: {len(results)}  range: {results[0][0]:.1f}s to {results[-1][0]:.1f}s",
          file=sys.stderr)

    # Inter-ictal baseline MSE (windows with time_rel < -30s)
    ii_windows = [(t, s, sc, d) for t, s, sc, d in results if t < -30.0]
    if not ii_windows:
        ii_windows = [(t, s, sc, d) for t, s, sc, d in results if t < 0.0]
    if not ii_windows:
        print(f"  [WARN] no pre-ictal baseline for {subject}", file=sys.stderr)
        return None

    baseline_ssm    = np.mean([s for _, s, _, _ in ii_windows])
    baseline_scalar = np.mean([sc for _, _, sc, _ in ii_windows])
    baseline_diag   = np.mean([d for _, _, _, d in ii_windows])

    print(f"  Baseline (t<-30s) — ssm16={baseline_ssm:.4f}  scalar={baseline_scalar:.4f}  diag={baseline_diag:.4f}",
          file=sys.stderr)

    return {
        "subject":         subject,
        "results":         results,
        "baseline_ssm":    baseline_ssm,
        "baseline_scalar": baseline_scalar,
        "baseline_diag":   baseline_diag,
        "train_mse_ssm":   train_mse_ssm,
        "train_mse_diag":  train_mse_diag,
    }


# ─── Reporting ───────────────────────────────────────────────────────────────

def bin_results(results, bin_s=BIN_S):
    """Aggregate results into time bins. Returns list of (bin_center, mean_ssm, mean_scalar, mean_diag)."""
    if not results:
        return []
    times = np.array([r[0] for r in results])
    t_min = np.floor(times.min() / bin_s) * bin_s
    t_max = np.ceil(times.max() / bin_s) * bin_s
    bins = np.arange(t_min, t_max + bin_s, bin_s)
    out = []
    for b in bins[:-1]:
        mask = (times >= b) & (times < b + bin_s)
        if mask.sum() == 0:
            continue
        m_ssm    = np.mean([results[i][1] for i in np.where(mask)[0]])
        m_scalar = np.mean([results[i][2] for i in np.where(mask)[0]])
        m_diag   = np.mean([results[i][3] for i in np.where(mask)[0]])
        out.append((b + bin_s / 2, m_ssm, m_scalar, m_diag))
    return out


def print_subject_table(sub_data):
    subject     = sub_data["subject"]
    results     = sub_data["results"]
    b_ssm       = sub_data["baseline_ssm"]
    b_scalar    = sub_data["baseline_scalar"]
    b_diag      = sub_data["baseline_diag"]

    binned = bin_results(results)
    print(f"\n{'═'*72}")
    print(f"  {subject}  |  baseline ssm16={b_ssm:.4f}  scalar={b_scalar:.4f}  diag={b_diag:.4f}")
    print(f"{'─'*72}")
    print(f"  {'time_s':>8}  {'mse_ssm16':>10}  {'mse_scalar':>10}  {'mse_diag':>10}  "
          f"{'ratio_ssm':>10}  {'ratio_diag':>10}  phase")
    print(f"{'─'*72}")

    for (t, m_ssm, m_scalar, m_diag) in binned:
        r_ssm  = m_ssm  / (b_ssm  + 1e-12)
        r_diag = m_diag / (b_diag + 1e-12)
        phase  = "ICTAL" if t >= 0 else ("near_pre" if t >= -10 else "pre")
        marker = " ◀" if (phase == "near_pre" and r_ssm > 1.5) else ""
        print(f"  {t:+8.1f}  {m_ssm:10.4f}  {m_scalar:10.4f}  {m_diag:10.4f}  "
              f"{r_ssm:10.3f}×  {r_diag:10.3f}×  {phase}{marker}")

    # Pre-ictal window summary: mean ratio in (-10, 0)
    pre10 = [(t, s, sc, d) for t, s, sc, d in results if -10.0 <= t < 0.0]
    if pre10:
        r_ssm_pre  = np.mean([s  / (b_ssm  + 1e-12) for _, s, _, _ in pre10])
        r_diag_pre = np.mean([d  / (b_diag + 1e-12) for _, _, _, d in pre10])
        print(f"{'─'*72}")
        print(f"  Pre-ictal window (-10 to 0s):  ssm16 ratio={r_ssm_pre:.3f}×  diag ratio={r_diag_pre:.3f}×")
        if r_ssm_pre > 1.5 and r_ssm_pre > r_diag_pre:
            print(f"  *** PRE-ICTAL RAMP DETECTED — S-SSM leads diagonal ***")
        elif r_ssm_pre > 1.2:
            print(f"  *** WEAK PRE-ICTAL SIGNAL — S-SSM elevated ***")
        else:
            print(f"  (no pre-ictal ramp in this subject)")

    ictal = [(t, s, sc, d) for t, s, sc, d in results if 0.0 <= t < 10.0]
    if ictal:
        r_ssm_ic  = np.mean([s  / (b_ssm  + 1e-12) for _, s, _, _ in ictal])
        r_diag_ic = np.mean([d  / (b_diag + 1e-12) for _, _, _, d in ictal])
        print(f"  Ictal window (0 to +10s):      ssm16 ratio={r_ssm_ic:.3f}×  diag ratio={r_diag_ic:.3f}×")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Door II: S-SSM pre-ictal ramp on CHB-MIT")
    ap.add_argument("--subjects", default="chb01,chb03,chb05",
                    help="Comma-separated subject IDs (default: chb01,chb03,chb05)")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help=f"Sedenion gate parameter (default: {ALPHA})")
    ap.add_argument("--bin-s", type=float, default=BIN_S,
                    help=f"Bin width in seconds for reporting (default: {BIN_S})")
    ap.add_argument("--pre-scan-s", type=float, default=PRE_SCAN_S,
                    help=f"Seconds before onset to start scan (default: {PRE_SCAN_S})")
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",")]
    rng = np.random.RandomState(SEED)

    print(f"Door II: S-SSM Pre-ictal Ramp Detection")
    print(f"Subjects: {subjects}  α={args.alpha}  bin={args.bin_s}s  pre_scan={args.pre_scan_s}s")
    print(f"Architecture: S-SSM-16ch + S-SSM-scalar + Real-16ch-diagonal")

    all_results = []
    for subj in subjects:
        if subj not in SUBJECTS:
            print(f"  [WARN] Unknown subject: {subj} — skipping", file=sys.stderr)
            continue
        cfg = SUBJECTS[subj]
        data = analyze_subject(subj, cfg, rng)
        if data is not None:
            all_results.append(data)

    # ── Per-subject tables ───────────────────────────────────────────────────
    for data in all_results:
        print_subject_table(data)

    # ── Cross-subject summary ────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'═'*72}")
        print("  CROSS-SUBJECT SUMMARY")
        print(f"{'─'*72}")
        print(f"  {'subject':>8}  {'pre10_ssm':>12}  {'pre10_diag':>12}  {'ictal_ssm':>12}  {'ictal_diag':>12}")
        print(f"{'─'*72}")
        for data in all_results:
            results  = data["results"]
            b_ssm    = data["baseline_ssm"]
            b_diag   = data["baseline_diag"]

            pre10 = [(t, s, sc, d) for t, s, sc, d in results if -10.0 <= t < 0.0]
            ictal  = [(t, s, sc, d) for t, s, sc, d in results if  0.0 <= t < 10.0]

            r_pre_ssm  = np.mean([s / (b_ssm  + 1e-12) for _, s, _, _ in pre10]) if pre10 else float("nan")
            r_pre_diag = np.mean([d / (b_diag + 1e-12) for _, _, _, d in pre10]) if pre10 else float("nan")
            r_ic_ssm   = np.mean([s / (b_ssm  + 1e-12) for _, s, _, _ in ictal]) if ictal else float("nan")
            r_ic_diag  = np.mean([d / (b_diag + 1e-12) for _, _, _, d in ictal]) if ictal else float("nan")

            print(f"  {data['subject']:>8}  {r_pre_ssm:>12.3f}×  {r_pre_diag:>12.3f}×  "
                  f"{r_ic_ssm:>12.3f}×  {r_ic_diag:>12.3f}×")

        print(f"{'─'*72}")
        # Aggregate
        pre_ssm_all  = []
        pre_diag_all = []
        for data in all_results:
            results = data["results"]
            b_ssm   = data["baseline_ssm"]
            b_diag  = data["baseline_diag"]
            pre10   = [(t, s, sc, d) for t, s, sc, d in results if -10.0 <= t < 0.0]
            pre_ssm_all.extend( [s / (b_ssm  + 1e-12) for _, s, _, _ in pre10])
            pre_diag_all.extend([d / (b_diag + 1e-12) for _, _, _, d in pre10])
        if pre_ssm_all:
            print(f"  Pooled pre-ictal (-10 to 0s):  ssm16={np.mean(pre_ssm_all):.3f}×  diag={np.mean(pre_diag_all):.3f}×")


if __name__ == "__main__":
    main()
