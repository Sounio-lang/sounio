#!/usr/bin/env python3
"""Exp 2: Quaternion null model — matched null for S-SSM seizure biomarker.

Quaternions are ASSOCIATIVE: (q1⊗q2)⊗q3 = q1⊗(q2⊗q3) always.
The associator observable therefore identically vanishes.

We build a Q-SSM with the same single-gate architecture as the S-SSM
(one algebra element A as gate, same α=0.20 parameterisation, same
training regime, same readout), using only 4 channels to match the
quaternion dimension.

The NATURAL quaternion analog of the associator is the commutator:
   comm(A_q, h_q) = A_q⊗h_q − h_q⊗A_q   (non-commutativity)

For A_q = (a0, a1, 0, 0) and generic h_q = (h0, h1, h2, h3):
   comm = (0, 0, −2·a1·h3, 2·a1·h2)
   ||comm|| = 2·a1·√(h2² + h3²)

This is purely algebraic — it does not depend on the EEG content, only
on how much h lives in the e2/e3 directions.  Crucially, it can NEVER
show a "dip" from seizure dynamics unless the EEG content happens to
align h with e2/e3 during pre-ictal.

We also track:
  - Q-SSM h_sep = ||h_II − h_IC||   (classification sensitivity)
  - S-SSM assoc_norm (reproduced from Exp 1)
  - S-SSM h_sep

Two-outcome table (from experimental roadmap):
  A) Q-SSM h_sep ≈ S-SSM h_sep (within 0.05):
       Classification frame is dead. Paper pivots to biomarker frame.
  B) Commutator shows NO pre-ictal dip while assoc DOES:
       Dip is sedenion-specific. Associator is a NOVEL observable.

Outputs: artifacts/research/eeg_hessian/quaternion_null_chb03.tsv
"""

import pyedflib
import numpy as np
import os

EDF_PATH = "data/chbmit/chb03_01.edf"
OUT_PATH = "artifacts/research/eeg_hessian/quaternion_null_chb03.tsv"
ALPHA    = 0.20
FS       = 256
N_CH_S   = 16   # sedenion channels
N_CH_Q   = 4    # quaternion channels (first 4 of the 16)
N_WIN    = 80
N_TRAIN  = 64
INTERICTAL_S = 10.0

WINDOWS = [
    ("FAR",   302, -60),
    ("PRE30", 332, -30),
    ("PRE10", 352, -10),
    ("PRE5",  357,  -5),
    ("IC",    362,   0),
    ("POST",  367,  +5),
]

# Interictal window (for h_sep)
II_START_S = INTERICTAL_S


# ─── Quaternion arithmetic ─────────────────────────────────────────────────

def quat_mul(q, p):
    """Hamilton product: (w,x,y,z) ⊗ (a,b,c,d)."""
    w,x,y,z = q
    a,b,c,d = p
    return np.array([
        w*a - x*b - y*c - z*d,
        w*b + x*a + y*d - z*c,
        w*c - x*d + y*a + z*b,
        w*d + x*c - y*b + z*a,
    ])

def quat_norm(q):
    return float(np.sqrt(np.dot(q, q)))

def quat_tanh(q):
    return np.tanh(q)

def quat_normalize(q):
    n = quat_norm(q)
    return q / n if n > 1e-7 else q

def quat_commutator_norm(a, h):
    """||A⊗h − h⊗A||."""
    diff = quat_mul(a, h) - quat_mul(h, a)
    return float(np.sqrt(np.dot(diff, diff)))

def quat_imag_norm(q):
    """||Im(q)|| = sqrt(x²+y²+z²)."""
    return float(np.sqrt(q[1]**2 + q[2]**2 + q[3]**2))


# ─── Sedenion arithmetic (from Exp 1 / generate_g2_projection_chb03.py) ───

def oct_mul(a, b):
    a0,a1,a2,a3,a4,a5,a6,a7 = a
    b0,b1,b2,b3,b4,b5,b6,b7 = b
    return np.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7,
        a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6,
        a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5,
        a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4,
        a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3,
        a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2,
        a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1,
        a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0,
    ])

def oct_conj(o):
    c = o.copy(); c[1:] = -c[1:]; return c

def sed_mul(a, b):
    o1a, o2a = a[:8], a[8:]
    o1b, o2b = b[:8], b[8:]
    return np.concatenate([
        oct_mul(o1a, o1b) - oct_mul(oct_conj(o2b), o2a),
        oct_mul(o2b, o1a) + oct_mul(o2a, oct_conj(o1b)),
    ])

def sed_norm(s):
    return float(np.sqrt(np.dot(s, s)))


# ─── Gates ────────────────────────────────────────────────────────────────

def make_quat_gate(alpha):
    q = np.zeros(4)
    q[0] = np.sqrt(1.0 - alpha**2)
    q[1] = alpha
    return q

def make_sed_gate(alpha):
    a = np.zeros(16)
    a[0] = np.sqrt(1.0 - alpha**2)
    a[1] = alpha
    return a


# ─── Data loading ─────────────────────────────────────────────────────────

def load_edf(path):
    f = pyedflib.EdfReader(path)
    n = min(f.signals_in_file, N_CH_S)
    sigs = np.stack([f.readSignal(i) for i in range(n)])
    f.close()
    return sigs

def extract_normalize(sigs, start_s, n_channels, n_samples):
    ii_start = int(INTERICTAL_S * FS)
    ii_ref   = sigs[:n_channels, ii_start : ii_start + N_TRAIN]
    means    = ii_ref.mean(axis=1, keepdims=True)
    stds     = ii_ref.std(axis=1, keepdims=True)
    stds     = np.where(stds < 1e-6, 1.0, stds)
    max_abs  = np.abs((ii_ref - means) / stds).max(axis=1, keepdims=True)
    max_abs  = np.where(max_abs < 1e-6, 1.0, max_abs)
    start    = int(start_s * FS)
    raw      = sigs[:n_channels, start : start + n_samples]
    if raw.shape[1] < n_samples:
        raw  = np.pad(raw, ((0,0),(0, n_samples - raw.shape[1])))
    norm = (raw - means) / stds / max_abs
    return np.clip(norm, -5.0, 5.0)


# ─── SSM forward passes ───────────────────────────────────────────────────

def quat_ssm_forward(a_gate, eeg_4ch):
    """Run Q-SSM for N_WIN steps; return final h and per-step observables."""
    h = np.array([1.0, 0.0, 0.0, 0.0])
    commutators = []
    imag_norms  = []
    for t in range(eeg_4ch.shape[1]):
        x  = eeg_4ch[:, t]
        ah = quat_mul(a_gate, h)
        commutators.append(quat_commutator_norm(a_gate, h))
        imag_norms.append(quat_imag_norm(h))
        raw = ah + x
        act = quat_tanh(raw)
        h   = quat_normalize(act)
    return h, float(np.mean(commutators)), float(np.mean(imag_norms))

def sed_ssm_forward(a_gate, eeg_16ch):
    """Run S-SSM for N_WIN steps; return final h and mean assoc_norm."""
    h = np.zeros(16); h[0] = 1.0
    assocs = []
    for t in range(eeg_16ch.shape[1]):
        x    = eeg_16ch[:, t]
        ah   = sed_mul(a_gate, h)
        ahx  = sed_mul(ah, x)
        hx   = sed_mul(h, x)
        a_hx = sed_mul(a_gate, hx)
        assocs.append(sed_norm(ahx - a_hx))
        raw  = ah + x
        act  = np.tanh(raw)
        n    = sed_norm(act)
        h    = act / n if n > 1e-7 else act
    return h, float(np.mean(assocs))


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(EDF_PATH):
        print(f"ERROR: {EDF_PATH} not found"); return 1

    sigs    = load_edf(EDF_PATH)
    a_quat  = make_quat_gate(ALPHA)
    a_sed   = make_sed_gate(ALPHA)

    print(f"Loaded {EDF_PATH}: {sigs.shape[1]/FS:.1f}s, {sigs.shape[0]} channels")
    print(f"Q-SSM gate: α={ALPHA}, q=({a_quat[0]:.4f},{a_quat[1]:.4f},0,0)")
    print(f"S-SSM gate: α={ALPHA}, e0={a_sed[0]:.4f}, e1={a_sed[1]:.4f}")
    print()

    # ── Compute h_sep on II vs IC windows ─────────────────────────────────
    win_ii = extract_normalize(sigs,   10.0, N_CH_S, N_WIN)  # interictal
    win_ic = extract_normalize(sigs, 362.0,  N_CH_S, N_WIN)  # ictal onset

    h_ii_q, _, _ = quat_ssm_forward(a_quat, win_ii[:N_CH_Q])
    h_ic_q, _, _ = quat_ssm_forward(a_quat, win_ic[:N_CH_Q])
    h_sep_q = quat_norm(h_ii_q - h_ic_q)

    h_ii_s, _  = sed_ssm_forward(a_sed, win_ii)
    h_ic_s, _  = sed_ssm_forward(a_sed, win_ic)
    h_sep_s = sed_norm(h_ii_s - h_ic_s)

    print(f"h_sep (quaternion, 4-dim): {h_sep_q:.6f}")
    print(f"h_sep (sedenion, 16-dim):  {h_sep_s:.6f}")
    hsep_diff = abs(h_sep_q - h_sep_s)
    # Note: direct comparison is scale-dependent (4 vs 16 dims)
    # Normalise by dimension: h_sep / sqrt(dim)
    h_sep_q_norm = h_sep_q / 2.0          # sqrt(4)=2
    h_sep_s_norm = h_sep_s / 4.0          # sqrt(16)=4
    print(f"h_sep/√dim  quaternion: {h_sep_q_norm:.6f}  sedenion: {h_sep_s_norm:.6f}")
    print()

    # ── Pre-ictal sweep ────────────────────────────────────────────────────
    header = f"{'Window':<8}  {'t_rel':>6}  {'q_comm':>10}  {'q_imag':>10}  {'s_assoc':>10}"
    print(header)
    print("-" * 60)

    rows = []
    for label, start_s, time_rel in WINDOWS:
        win_q  = extract_normalize(sigs, start_s, N_CH_Q, N_WIN)
        win_s  = extract_normalize(sigs, start_s, N_CH_S, N_WIN)
        _, q_comm, q_imag = quat_ssm_forward(a_quat, win_q)
        _, s_assoc        = sed_ssm_forward(a_sed, win_s)
        rows.append((label, time_rel, q_comm, q_imag, s_assoc))
        print(f"{label:<8}  {time_rel:>6}  {q_comm:>10.6f}  {q_imag:>10.6f}  {s_assoc:>10.6f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fp:
        fp.write("# schema=quaternion_null.v1\n")
        fp.write(f"# subject=chb03  alpha={ALPHA}  q_dim=4  s_dim=16\n")
        fp.write(f"# h_sep_q_normd={h_sep_q_norm:.6f}  h_sep_s_normd={h_sep_s_norm:.6f}\n")
        fp.write("window\ttime_rel\tq_comm\tq_imag\ts_assoc\n")
        for row in rows:
            fp.write("\t".join(str(v) for v in row) + "\n")
    print(f"\nWrote: {OUT_PATH}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    pre      = [r for r in rows if r[1] <= 0]  # FAR..IC
    s_dip    = any(r[4] < rows[0][4] * 0.70 for r in pre[1:])  # assoc dips >30%
    q_dip    = any(r[2] < rows[0][2] * 0.70 for r in pre[1:])  # commutator dips?

    # Dimension-normalised h_sep difference
    hsep_diff_normd = abs(h_sep_q_norm - h_sep_s_norm)

    print("\n=== Quaternion null verdict ===")
    print(f"  S-SSM assoc shows pre-ictal dip: {s_dip}")
    print(f"  Q-SSM commutator shows dip:      {q_dip}")
    if not q_dip and s_dip:
        print("  → Associator dip is SEDENION-SPECIFIC. Observable is novel.")
    elif q_dip:
        print("  → Commutator also dips. Dip may not be algebra-specific.")

    print(f"  h_sep/√dim: Q={h_sep_q_norm:.4f}  S={h_sep_s_norm:.4f}  diff={hsep_diff_normd:.4f}")
    if hsep_diff_normd < 0.05:
        print("  → h_sep parity (<0.05 diff): classification frame DEAD")
        print("    Paper must pivot to biomarker frame (Exp 2 outcome A)")
    else:
        print("  → h_sep differs by >{:.4f}: sedenion shows additional separation".format(hsep_diff_normd))
        print("    (Note: dimensional mismatch limits direct comparison)")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
