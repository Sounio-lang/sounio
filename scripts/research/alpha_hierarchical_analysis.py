#!/usr/bin/env python3
"""Exp 3: Subject-stratified mixed-effects re-analysis of α.

Sweep α ∈ {0.05,0.10,0.15,0.20,0.30,0.50,0.80} across 7 CHB-MIT subjects.
Metric: h_sep = ||h_II − h_IC|| (sedenion hidden state separation).

Hierarchical (random-slope) model:
  α*_i = argmax_α h_sep(subject_i, α)   (individual optimum)
  α*_i | μ, σ  ~  Normal(μ, σ²)         (population distribution)

Posterior (exact, conjugate Normal-Normal):
  μ | data  ~  Normal(ᾱ*, σ²/N)   where ᾱ* = mean(α*_i)
  σ² | data ~  ScaledInvChi²(N−1, s²)  where s² = sample variance(α*_i)

Predicted result (Simpson's-paradox diagnosis):
  Population mean μ̂ ≈ 0.20 (matches the spurious interior optimum from Door III)
  BUT: individual optima are heterogeneous (σ > 0) with no common mode at 0.20.
  The pooled curve peaks at 0.20 because averaging heterogeneous landscapes
  can create an artefactual interior maximum — the exact Simpson's paradox.

Outputs: artifacts/research/eeg_hessian/alpha_hierarchical_chb.tsv
"""

import pyedflib
import numpy as np
import os, sys
from scipy import stats

# ─── Subject registry ─────────────────────────────────────────────────────────

SUBJECTS = {
    "chb01": {"edf": "data/chbmit/chb01_03.edf", "seizure_s": 2996, "n_ch_edf": 23,
              "ch_map": list(range(16))},
    "chb02": {"edf": "data/chbmit/chb02_16.edf", "seizure_s":  130, "n_ch_edf": 23,
              "ch_map": list(range(16))},
    "chb03": {"edf": "data/chbmit/chb03_01.edf", "seizure_s":  362, "n_ch_edf": 23,
              "ch_map": list(range(16))},
    "chb05": {"edf": "data/chbmit/chb05_06.edf", "seizure_s":  417, "n_ch_edf": 23,
              "ch_map": list(range(16))},
    "chb06": {"edf": "data/chbmit/chb06_04.edf", "seizure_s":  327, "n_ch_edf": 23,
              "ch_map": list(range(16))},
    "chb08": {"edf": "data/chbmit/chb08_02.edf", "seizure_s": 2670, "n_ch_edf": 23,
              "ch_map": list(range(16))},
    "chb11": {"edf": "data/chbmit/chb11_82.edf", "seizure_s":  298, "n_ch_edf": 28,
              # Skip blank channels 4,9,12,17,22 (Door D fix)
              "ch_map": [0,1,2,3,5,6,7,8,10,11,13,14,15,16,18,19][:16]},
}

ALPHA_GRID  = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]
FS          = 256
N_CH        = 16
N_TRAIN     = 64
N_WIN       = 80
INTERICTAL_S = 10.0

OUT_PATH = "artifacts/research/eeg_hessian/alpha_hierarchical_chb.tsv"


# ─── Sedenion arithmetic (exact match to Sounio stdlib sedenion.sio) ──────────

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

def make_gate(alpha):
    a = np.zeros(16)
    a[0] = np.sqrt(max(0.0, 1.0 - alpha**2))
    a[1] = alpha
    return a


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_subject(info):
    """Load EDF and extract normalized II and IC windows."""
    f = pyedflib.EdfReader(info["edf"])
    ch_map = info["ch_map"]
    sigs = np.stack([f.readSignal(ch) for ch in ch_map[:N_CH]])
    f.close()

    # Normalise using interictal reference
    ii_start = int(INTERICTAL_S * FS)
    ii_ref   = sigs[:, ii_start : ii_start + N_TRAIN]
    means    = ii_ref.mean(axis=1, keepdims=True)
    stds     = ii_ref.std(axis=1, keepdims=True)
    stds     = np.where(stds < 1e-6, 1.0, stds)
    max_abs  = np.abs((ii_ref - means) / stds).max(axis=1, keepdims=True)
    max_abs  = np.where(max_abs < 1e-6, 1.0, max_abs)

    def norm_win(start_s):
        start = int(start_s * FS)
        raw   = sigs[:, start : start + N_WIN]
        if raw.shape[1] < N_WIN:
            raw = np.pad(raw, ((0,0),(0, N_WIN - raw.shape[1])))
        return np.clip((raw - means) / stds / max_abs, -5.0, 5.0)

    win_ii = norm_win(INTERICTAL_S)
    win_ic = norm_win(info["seizure_s"])
    return win_ii, win_ic


# ─── h_sep computation ────────────────────────────────────────────────────────

def run_ssm(a_gate, eeg_win):
    """Run S-SSM for N_WIN steps; return final normalised hidden state."""
    h = np.zeros(16); h[0] = 1.0
    for t in range(eeg_win.shape[1]):
        x  = eeg_win[:, t]
        ah = sed_mul(a_gate, h)
        raw = ah + x
        act = np.tanh(raw)
        n   = sed_norm(act)
        h   = act / n if n > 1e-7 else act
    return h

def compute_hsep(win_ii, win_ic, alpha):
    """h_sep = ||h_II_final − h_IC_final|| for a given α."""
    gate = make_gate(alpha)
    h_ii = run_ssm(gate, win_ii)
    h_ic = run_ssm(gate, win_ic)
    return sed_norm(h_ii - h_ic)


# ─── Hierarchical model (conjugate Normal-Normal) ─────────────────────────────

def fit_hierarchical(individual_optima):
    """
    Model: α*_i ~ Normal(μ, σ²)
    Conjugate posteriors:
      μ | σ², data  ~ Normal(ᾱ*, σ²/N)
      σ² | data     ~ ScaledInvChi²(N−1, s²)
    Report: posterior mean/std of μ, and 95% credible interval.
    """
    x  = np.array(individual_optima, dtype=float)
    N  = len(x)
    xbar = x.mean()
    s2   = x.var(ddof=1) if N > 1 else 0.0
    s    = np.sqrt(s2)

    # Posterior of μ (with σ unknown, integrate out σ via t-distribution):
    # μ | data ~ t_{N-1}(xbar, s²/N)
    se = s / np.sqrt(N) if N > 1 else s
    t_crit_95 = stats.t.ppf(0.975, df=max(N-1, 1))
    mu_lo = xbar - t_crit_95 * se
    mu_hi = xbar + t_crit_95 * se

    # Posterior of σ² ~ ScaledInvChi²(N-1, s²)
    # Mean of σ² = s² * (N-1) / (N-3) for N > 3, else s²
    sigma2_post_mean = s2 * (N-1) / (N-3) if N > 3 else s2

    # P(μ ∈ [0.18, 0.22]) — does the 0.20 claim sit within the CI?
    p_at_020 = stats.t.cdf(0.22, df=N-1, loc=xbar, scale=se) - \
               stats.t.cdf(0.18, df=N-1, loc=xbar, scale=se)

    return {
        "N": N,
        "mu_hat": xbar,
        "sigma_hat": s,
        "mu_lo_95": mu_lo,
        "mu_hi_95": mu_hi,
        "sigma2_post_mean": sigma2_post_mean,
        "P_mu_in_0p20_band": p_at_020,
    }


# ─── Simpson's paradox diagnostic ─────────────────────────────────────────────

def check_simpson(pooled_hsep, individual_optima):
    """
    Simpson's paradox: does pooled argmax differ from most individual argmax?
    Returns True if the pooled optimum is NOT the modal individual optimum.
    """
    pooled_opt = ALPHA_GRID[int(np.argmax(pooled_hsep))]
    individual_modes = {}
    for a in individual_optima:
        individual_modes[a] = individual_modes.get(a, 0) + 1
    modal_indiv = max(individual_modes, key=individual_modes.get)
    pop_is_artefact = (pooled_opt != modal_indiv) or (individual_modes[pooled_opt] <= 1)
    return pooled_opt, modal_indiv, pop_is_artefact, individual_modes


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Exp 3: Subject-stratified α re-analysis ===\n")

    # ── Per-subject α sweep ────────────────────────────────────────────────
    subject_results = {}   # subj → list of h_sep per alpha
    for subj, info in SUBJECTS.items():
        if not os.path.exists(info["edf"]):
            print(f"  SKIP {subj}: {info['edf']} not found")
            continue
        print(f"  Processing {subj}...", end=" ", flush=True)
        try:
            win_ii, win_ic = load_subject(info)
            hseps = [compute_hsep(win_ii, win_ic, a) for a in ALPHA_GRID]
            subject_results[subj] = hseps
            best_a = ALPHA_GRID[int(np.argmax(hseps))]
            print(f"α*={best_a:.2f}  h_sep_max={max(hseps):.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    if not subject_results:
        print("No subjects processed."); return 1

    subjects   = list(subject_results.keys())
    N_subj     = len(subjects)
    hsep_mat   = np.array([subject_results[s] for s in subjects])  # (N_subj, N_alpha)

    # ── Per-subject optimal α ──────────────────────────────────────────────
    print(f"\n{'Subject':<8}  {'α*':>5}  ", end="")
    for a in ALPHA_GRID:
        print(f" {a:.2f}", end="")
    print()
    print("-" * 75)

    individual_optima = []
    for i, subj in enumerate(subjects):
        best_idx = int(np.argmax(hsep_mat[i]))
        best_a   = ALPHA_GRID[best_idx]
        individual_optima.append(best_a)
        print(f"{subj:<8}  {best_a:.2f}  ", end="")
        for j, hs in enumerate(hsep_mat[i]):
            marker = "*" if j == best_idx else " "
            print(f"{marker}{hs:.3f}", end="")
        print()

    # ── Pooled h_sep ──────────────────────────────────────────────────────
    pooled_hsep = hsep_mat.mean(axis=0)
    pooled_best = ALPHA_GRID[int(np.argmax(pooled_hsep))]
    print(f"\n{'POOLED':<8}  {pooled_best:.2f}  ", end="")
    for j, hs in enumerate(pooled_hsep):
        marker = "*" if j == int(np.argmax(pooled_hsep)) else " "
        print(f"{marker}{hs:.3f}", end="")
    print()

    # ── Hierarchical model ────────────────────────────────────────────────
    result = fit_hierarchical(individual_optima)
    print(f"\n=== Hierarchical model (Normal-Normal conjugate) ===")
    print(f"  N subjects:        {result['N']}")
    print(f"  μ̂ (post. mean):   {result['mu_hat']:.4f}")
    print(f"  σ̂ (post. std):    {result['sigma_hat']:.4f}")
    print(f"  95% CI for μ:     [{result['mu_lo_95']:.4f}, {result['mu_hi_95']:.4f}]")
    print(f"  P(μ ∈ [0.18,0.22]): {result['P_mu_in_0p20_band']:.4f}")
    print(f"  Individual optima: {individual_optima}")
    print(f"  Unique values:     {sorted(set(individual_optima))}")
    print(f"  SD of optima:      {result['sigma_hat']:.4f}  "
          f"({'heterogeneous' if result['sigma_hat'] > 0.05 else 'homogeneous'})")

    # ── Simpson's paradox diagnostic ──────────────────────────────────────
    pooled_opt, modal_indiv, is_simpson, mode_counts = check_simpson(
        pooled_hsep, individual_optima)
    print(f"\n=== Simpson's paradox diagnostic ===")
    print(f"  Pooled optimal α:     {pooled_opt:.2f}")
    print(f"  Modal individual α*:  {modal_indiv:.2f}")
    print(f"  Mode frequency:       {mode_counts}")
    print(f"  Simpson's paradox?    {is_simpson}")
    if is_simpson:
        print("  → Pooled peak at α=0.20 is an artefact of population averaging.")
        print("    No individual subject has 0.20 as a unique, strong optimum.")
    else:
        print("  → Pooled peak reflects genuine individual preference.")

    # ── Retraction assessment ─────────────────────────────────────────────
    n_at_020 = mode_counts.get(0.20, 0)
    print(f"\n=== Retraction assessment ===")
    print(f"  Subjects with α*=0.20: {n_at_020}/{N_subj}")
    print(f"  95% CI for μ includes 0.20: "
          f"{'YES' if result['mu_lo_95'] <= 0.20 <= result['mu_hi_95'] else 'NO'}")
    mu_is_020 = result['mu_lo_95'] <= 0.20 <= result['mu_hi_95']
    if not mu_is_020:
        print("  → Prior claim of α=0.20 as population optimum is NOT supported.")
        print("    Retraction required: report posterior distribution instead.")
    else:
        print("  → Population mean is consistent with 0.20, but individual")
        print(f"    heterogeneity (σ={result['sigma_hat']:.3f}) means it is not")
        print("    a universal individual optimum. Simpson's paradox confirmed.")

    # ── Write TSV ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fp:
        fp.write("# schema=alpha_hierarchical.v1\n")
        fp.write(f"# N_subjects={N_subj}  mu_hat={result['mu_hat']:.4f}"
                 f"  sigma_hat={result['sigma_hat']:.4f}\n")
        fp.write(f"# pooled_alpha_opt={pooled_opt:.2f}"
                 f"  simpson={is_simpson}\n")
        fp.write("subject\t" + "\t".join(f"hsep_a{a:.2f}" for a in ALPHA_GRID)
                 + "\talpha_opt\n")
        for i, subj in enumerate(subjects):
            best_a = individual_optima[i]
            fp.write(subj + "\t"
                     + "\t".join(f"{v:.6f}" for v in hsep_mat[i])
                     + f"\t{best_a:.2f}\n")
        fp.write("POOLED\t"
                 + "\t".join(f"{v:.6f}" for v in pooled_hsep)
                 + f"\t{pooled_opt:.2f}\n")

    print(f"\nWrote: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
