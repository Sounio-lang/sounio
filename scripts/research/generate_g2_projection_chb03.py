#!/usr/bin/env python3
"""Exp 1: G₂-invariant subspace projection across the pre-ictal window.

Falsifier for the Door F mechanism.

Computes, at each probe window relative to seizure onset in chb03:
  - assoc_norm = mean_t ||(A⊗h_t)⊗x_t − A⊗(h_t⊗x_t)||
  - g2_proj    = mean_t ||π(h_t)||   [projection onto e1..e7 = G₂-invariant Im(𝕆)]
  - g2_comp    = mean_t ||π⊥(h_t)||  [sedenion extension e8..e15]

Predicted result (Door F mechanism):
  g2_proj increases monotonically FAR→IC while assoc_norm decreases.
  If assoc contracts WITHOUT g2_proj increasing, the mechanism is a pure
  norm-contraction effect (not G₂-drift) — that would falsify Door F.

Outputs: artifacts/research/eeg_hessian/g2_projection_chb03.tsv
"""

import pyedflib
import numpy as np
import os

EDF_PATH  = "data/chbmit/chb03_01.edf"
OUT_PATH  = "artifacts/research/eeg_hessian/g2_projection_chb03.tsv"
ALPHA     = 0.20
FS        = 256
N_CH      = 16
N_WIN     = 80
N_TRAIN   = 64
INTERICTAL_S = 10.0

WINDOWS = [
    ("FAR",   302, -60),
    ("PRE30", 332, -30),
    ("PRE10", 352, -10),
    ("PRE5",  357,  -5),
    ("IC",    362,   0),
    ("POST",  367,  +5),
]


# ─── Sedenion arithmetic (matches Sounio stdlib sedenion.sio exactly) ────────

def oct_mul(a, b):
    """Octonion multiplication using the Baez/Fano table in sedenion.sio."""
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
    """Sedenion product: (o1a,o2a)*(o1b,o2b) = (o1a*o1b - conj(o2b)*o2a,
                                                  o2b*o1a + o2a*conj(o1b))"""
    o1a, o2a = a[:8], a[8:]
    o1b, o2b = b[:8], b[8:]
    r_lo = oct_mul(o1a, o1b) - oct_mul(oct_conj(o2b), o2a)
    r_hi = oct_mul(o2b, o1a) + oct_mul(o2a, oct_conj(o1b))
    return np.concatenate([r_lo, r_hi])

def sed_norm(s):
    return float(np.sqrt(np.dot(s, s)))

def sed_tanh(s):
    return np.tanh(s)

def sed_add(a, b):
    return a + b

def sed_scale(s, k):
    return s * k


# ─── Gate A (α=0.20, same as Door I/E/F) ─────────────────────────────────────

def make_gate_a(alpha):
    """Gate A: e0=sqrt(1-α²), e1=α, rest=0."""
    a = np.zeros(16)
    a[0] = np.sqrt(1.0 - alpha**2)
    a[1] = alpha
    return a


# ─── G₂ projection metrics ────────────────────────────────────────────────────

def g2_proj_norm(h):
    """Projection onto G₂-invariant Im(𝕆) = span{e1..e7}."""
    return float(np.sqrt(np.dot(h[1:8], h[1:8])))

def g2_comp_norm(h):
    """Sedenion extension complement = span{e8..e15}."""
    return float(np.sqrt(np.dot(h[8:], h[8:])))


# ─── SSM forward pass ─────────────────────────────────────────────────────────

def ssm_forward(a_gate, eeg_window):
    """Run S-SSM forward for N_WIN steps; return per-step assoc/g2 metrics.

    eeg_window: (N_CH, N_WIN) array of normalised EEG
    a_gate:     16-dim sedenion gate
    """
    h = np.zeros(16); h[0] = 1.0  # h = e0 (identity = 'one' in Sounio)
    assocs    = []
    g2_projs  = []
    g2_comps  = []

    for t in range(eeg_window.shape[1]):
        # Input: map 16 EEG channels onto sedenion components
        x = eeg_window[:, t].copy()  # 16 floats → sedenion

        # Associator: ||(A⊗h)⊗x − A⊗(h⊗x)||
        ah   = sed_mul(a_gate, h)
        ahx  = sed_mul(ah, x)
        hx   = sed_mul(h, x)
        a_hx = sed_mul(a_gate, hx)
        diff = ahx - a_hx
        assocs.append(sed_norm(diff))

        # G₂ projection of current hidden state
        g2_projs.append(g2_proj_norm(h))
        g2_comps.append(g2_comp_norm(h))

        # Advance hidden state: h = normalize(tanh(A⊗h + x))
        raw = sed_add(ah, x)
        act = sed_tanh(raw)
        n   = sed_norm(act)
        h   = sed_scale(act, 1.0 / n) if n > 1e-7 else act

    return (
        float(np.mean(assocs)),
        float(np.mean(g2_projs)),
        float(np.mean(g2_comps)),
    )


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_edf(path):
    f = pyedflib.EdfReader(path)
    n = min(f.signals_in_file, N_CH)
    sigs = np.stack([f.readSignal(i) for i in range(n)])
    f.close()
    # Pad to N_CH if fewer channels
    if sigs.shape[0] < N_CH:
        pad = np.zeros((N_CH - sigs.shape[0], sigs.shape[1]))
        sigs = np.vstack([sigs, pad])
    return sigs  # (N_CH, T)

def extract_normalize(sigs, start_s, n_samples):
    """Extract and z-score normalise a window using interictal statistics."""
    ii_start = int(INTERICTAL_S * FS)
    ii_ref   = sigs[:, ii_start : ii_start + N_TRAIN]
    means    = ii_ref.mean(axis=1, keepdims=True)
    stds     = ii_ref.std(axis=1, keepdims=True)
    stds     = np.where(stds < 1e-6, 1.0, stds)
    max_abs  = np.abs((ii_ref - means) / stds).max(axis=1, keepdims=True)
    max_abs  = np.where(max_abs < 1e-6, 1.0, max_abs)

    start  = int(start_s * FS)
    raw    = sigs[:, start : start + n_samples]
    if raw.shape[1] < n_samples:
        raw = np.pad(raw, ((0,0),(0, n_samples - raw.shape[1])))
    norm   = (raw - means) / stds / max_abs
    return np.clip(norm, -5.0, 5.0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(EDF_PATH):
        print(f"ERROR: EDF file not found: {EDF_PATH}")
        return 1

    sigs   = load_edf(EDF_PATH)
    a_gate = make_gate_a(ALPHA)

    print(f"Loaded {EDF_PATH}: {sigs.shape[1]/FS:.1f}s, {sigs.shape[0]} channels")
    print(f"Gate A: α={ALPHA}, e0={a_gate[0]:.4f}, e1={a_gate[1]:.4f}")
    print()
    print(f"{'Window':<8}  {'time_rel':>8}  {'assoc':>10}  {'g2_proj':>10}  {'g2_comp':>10}  {'g2_ratio':>10}")
    print("-" * 70)

    rows = []
    for label, start_s, time_rel in WINDOWS:
        win = extract_normalize(sigs, start_s, N_WIN)
        mean_assoc, mean_g2_proj, mean_g2_comp = ssm_forward(a_gate, win)
        denom = mean_g2_proj + mean_g2_comp
        g2_ratio = mean_g2_proj / denom if denom > 1e-10 else 0.0
        rows.append((label, time_rel, mean_assoc, mean_g2_proj, mean_g2_comp, g2_ratio))
        print(f"{label:<8}  {time_rel:>8}  {mean_assoc:>10.6f}  {mean_g2_proj:>10.6f}  "
              f"{mean_g2_comp:>10.6f}  {g2_ratio:>10.6f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fp:
        fp.write("# schema=g2_projection.v1\n")
        fp.write("# subject=chb03  seizure_onset=362s  alpha=0.20\n")
        fp.write("# g2_proj=||h[1..7]||  g2_comp=||h[8..15]||  (h normalised)\n")
        fp.write("window\ttime_rel\tmean_assoc\tmean_g2_proj\tmean_g2_comp\tg2_ratio\n")
        for row in rows:
            fp.write("\t".join(str(v) for v in row) + "\n")

    print(f"\nWrote: {OUT_PATH}")

    # Quick monotonicity check (FAR→IC only, 5 windows)
    pre = [r for r in rows if r[1] <= 0]  # FAR, PRE30, PRE10, PRE5, IC
    assoc_mono  = all(pre[i][2] >= pre[i+1][2] for i in range(len(pre)-1))
    g2proj_mono = all(pre[i][3] <= pre[i+1][3] for i in range(len(pre)-1))
    print()
    print("=== G₂-drift hypothesis check ===")
    print(f"  assoc_norm  monotone decreasing FAR→IC: {assoc_mono}")
    print(f"  g2_proj     monotone increasing  FAR→IC: {g2proj_mono}")
    if assoc_mono and g2proj_mono:
        print("  RESULT: G₂-drift hypothesis SUPPORTED")
    elif assoc_mono and not g2proj_mono:
        print("  RESULT: PARTIAL — assoc contracts but g2_proj does not increase")
        print("          → mechanism is norm-contraction, NOT G₂-drift")
        print("          → Door F interpretation requires revision")
    else:
        print("  RESULT: hypothesis NOT supported — see TSV for details")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
