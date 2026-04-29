#!/usr/bin/env python3
# P5.4 Pre-registered permutation test for the Door F sedenion associator.
#
# PROTOCOL artifacts/research/door_f_cohort_N24/PROTOCOL.md sections 1 & 4:
#   Per-patient statistics (one-sided):
#     T1 (dip)   = (FAR - min(PRE30, PRE10, PRE5)) / FAR
#     T2 (spike) = (IC - PRE5) / PRE5
#
#   Exchangeable nulls (primary per PROTOCOL: iid label permutation of the
#   epoch -> window-label mapping; two extra nulls reported as sensitivity):
#     * iid       primary (PROTOCOL §4: "within-patient epoch labels are
#                 permuted"; each draw picks 6 distinct random epochs)
#     * circular  sensitivity (shift preserves 1/f-style autocorrelation)
#     * block     sensitivity (shuffle 10-epoch blocks)
#
#   Per-patient p_i = (1 + sum(null_stat >= observed)) / (1 + n_iter);
#   cohort combination via Fisher's method (-2 sum ln p_i, chi^2, df=2N);
#   BH-FDR q=0.05 across the PRIMARY family {T1, T2, T4} — applied to the
#   three combined p's from the (null=iid) rows only. Sensitivity nulls are
#   reported uncorrected.
#
# Usage:
#   python3 permutation_test.py \
#       --trajectory artifacts/research/door_f_cohort_N24/trajectories/cohort_trajectory.tsv \
#       --cohort-tsv artifacts/research/door_f_cohort_N24/door_f_cohort.tsv \
#       --out-dir    artifacts/research/door_f_cohort_N24/p5_permutation \
#       [--n-iter 10000] [--radius-s 90] [--window-len 80] [--fs 256] \
#       [--seed 20260421]
import argparse
import collections
import math
import os

import numpy as np

PROBE_OFFSETS_S = collections.OrderedDict([
    ("FAR",   -60),
    ("PRE30", -30),
    ("PRE10", -10),
    ("PRE5",   -5),
    ("IC",      0),
    ("POST",   +5),
])


def load_trajectory(path):
    by_patient = {}
    with open(path) as f:
        header = f.readline()
        for ln in f:
            p, ep, t, v = ln.rstrip("\n").split("\t")
            by_patient.setdefault(p, []).append((int(ep), float(t), float(v)))
    out = {}
    for p, rows in by_patient.items():
        rows.sort(key=lambda r: r[0])
        epochs = np.array([r[0] for r in rows])
        t_rel  = np.array([r[1] for r in rows])
        assoc  = np.array([r[2] for r in rows])
        out[p] = {"epoch": epochs, "t_rel_s": t_rel, "assoc": assoc}
    return out


def probe_epoch(t_rel_s, t_target_s):
    # nearest epoch by centre time
    return int(np.argmin(np.abs(t_rel_s - t_target_s)))


def _dip_spike_from_labels(assoc, labels):
    """labels is a dict {FAR, PRE30, PRE10, PRE5, IC, POST} -> epoch index.
    Returns (dip, spike) under the PROTOCOL definitions (§1).
    """
    far   = assoc[labels["FAR"]]
    pre30 = assoc[labels["PRE30"]]
    pre10 = assoc[labels["PRE10"]]
    pre5  = assoc[labels["PRE5"]]
    ic    = assoc[labels["IC"]]
    dip_denom = far   if far  > 1e-12 else 1e-12
    sp_denom  = pre5  if pre5 > 1e-12 else 1e-12
    dip   = (far - min(pre30, pre10, pre5)) / dip_denom
    spike = (ic  - pre5)                    / sp_denom
    return dip, spike


def null_statistics(assoc, probe_idx, null_kind, n_iter, rng):
    """Return arrays (dip_null, spike_null) of length n_iter.

    Nulls are constructed over the epoch -> window-label mapping:
      - iid: pick 6 distinct random epochs and assign them to the 6 labels
             in the PROTOCOL order (FAR, PRE30, PRE10, PRE5, IC, POST).
      - circular: shift all 6 reserved epoch indices by the same random r.
      - block: shuffle contiguous 10-epoch blocks of the trajectory, then
             apply the original reserved epoch indices to the shuffled
             series (destroys local autocorrelation above the block size).
    """
    n = assoc.shape[0]
    label_order = ["FAR", "PRE30", "PRE10", "PRE5", "IC", "POST"]
    dip   = np.empty(n_iter)
    spike = np.empty(n_iter)
    if null_kind == "iid":
        for i in range(n_iter):
            draws = rng.choice(n, size=6, replace=False)
            labels = {name: int(draws[j]) for j, name in enumerate(label_order)}
            dip[i], spike[i] = _dip_spike_from_labels(assoc, labels)
    elif null_kind == "circular":
        shifts = rng.integers(1, n, size=n_iter)
        base   = {name: probe_idx[name] for name in label_order}
        for i, r in enumerate(shifts):
            labels = {name: (base[name] + int(r)) % n for name in label_order}
            dip[i], spike[i] = _dip_spike_from_labels(assoc, labels)
    elif null_kind == "block":
        block = 10
        nblocks = n // block
        for i in range(n_iter):
            order = rng.permutation(nblocks)
            shuffled = np.concatenate([assoc[k * block:(k + 1) * block]
                                       for k in order])
            if shuffled.shape[0] < n:
                pad = n - shuffled.shape[0]
                shuffled = np.concatenate([shuffled, assoc[-pad:]])
            dip[i], spike[i] = _dip_spike_from_labels(shuffled, probe_idx)
    else:
        raise ValueError(f"unknown null_kind: {null_kind}")
    return dip, spike


def fishers_combined(p_values):
    p = np.asarray(p_values, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    chi2 = -2.0 * float(np.sum(np.log(p)))
    df   = 2 * len(p)
    # survival function of chi2 w/o scipy
    from math import lgamma, log, exp
    k = df // 2
    # for even df, P(X>=x) = sum_{i=0}^{k-1} (x/2)^i * exp(-x/2) / i!
    x2 = chi2 / 2.0
    terms = []
    acc = 0.0
    log_term = -x2
    for i in range(k):
        terms.append(log_term - lgamma(i + 1) + i * log(x2))
        if i < k - 1:
            pass
    m = max(terms)
    s = sum(exp(t - m) for t in terms)
    p_combined = exp(m) * s
    p_combined = min(1.0, max(p_combined, 0.0))
    return chi2, df, p_combined


def bh_fdr(p_values, q=0.05):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    thresh = (np.arange(1, m + 1) / m) * q
    sig = ranked <= thresh
    k_max = int(np.max(np.where(sig)[0])) + 1 if sig.any() else 0
    reject = np.zeros(m, dtype=bool)
    if k_max:
        reject[order[:k_max]] = True
    return reject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--n-iter",     type=int, default=10000)
    ap.add_argument("--radius-s",   type=int, default=90)
    ap.add_argument("--window-len", type=int, default=80)
    ap.add_argument("--fs",         type=int, default=256)
    ap.add_argument("--seed",       type=int, default=20260421)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    traj = load_trajectory(args.trajectory)

    null_kinds = ["iid", "circular", "block"]  # PROTOCOL §4 primary = iid

    # Per-patient per-cell p-values
    p_rows = []
    rng_master = np.random.default_rng(args.seed)

    for patient in sorted(traj):
        tr = traj[patient]
        assoc = tr["assoc"]
        t_rel = tr["t_rel_s"]
        probe_idx = {n: probe_epoch(t_rel, off)
                     for n, off in PROBE_OFFSETS_S.items()}
        dip_obs, spike_obs = _dip_spike_from_labels(assoc, probe_idx)
        for null_kind in null_kinds:
            rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
            dip_null, spike_null = null_statistics(
                assoc, probe_idx, null_kind, args.n_iter, rng)
            p_dip   = float((np.sum(dip_null   >= dip_obs  ) + 1)
                            / (args.n_iter + 1))
            p_spike = float((np.sum(spike_null >= spike_obs) + 1)
                            / (args.n_iter + 1))
            p_rows.append((patient, null_kind,
                           p_dip, p_spike, dip_obs, spike_obs,
                           float(assoc[probe_idx["FAR"]]),
                           float(assoc[probe_idx["PRE5"]])))

    per_patient_path = os.path.join(args.out_dir, "per_patient.tsv")
    with open(per_patient_path, "w") as f:
        f.write("patient\tnull\tp_dip\tp_spike\t"
                "dip_obs\tspike_obs\tFAR_assoc\tPRE5_assoc\n")
        for r in p_rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    # Cohort combination via Fisher's method
    cohort_path = os.path.join(args.out_dir, "cohort.tsv")
    cohort_rows = []
    for null_kind in null_kinds:
        sel = [r for r in p_rows if r[1] == null_kind]
        p_dips   = [r[2] for r in sel]
        p_spikes = [r[3] for r in sel]
        chi2_d, df, p_comb_d = fishers_combined(p_dips)
        chi2_s, _,  p_comb_s = fishers_combined(p_spikes)
        n_dip_sig   = sum(1 for r in sel if r[2] < 0.05)
        n_spike_sig = sum(1 for r in sel if r[3] < 0.05)
        cohort_rows.append((null_kind, len(sel),
                            n_dip_sig, p_comb_d,
                            n_spike_sig, p_comb_s,
                            chi2_d, chi2_s, df))

    with open(cohort_path, "w") as f:
        f.write("null\tN\tn_dip_sig_p05\tp_combined_dip\t"
                "n_spike_sig_p05\tp_combined_spike\t"
                "chi2_dip\tchi2_spike\tdf\n")
        for r in cohort_rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    # BH-FDR across the PRIMARY family {T1 dip iid, T2 spike iid} only.
    # T4 LOO classification is a separate test; its p enters the BH family
    # when computed (not in this script). Sensitivity nulls are uncorrected.
    primary = [r for r in cohort_rows if r[0] == "iid"]
    primary_ps   = [primary[0][3], primary[0][5]]
    primary_lbls = ["T1_dip_iid", "T2_spike_iid"]
    reject = bh_fdr(primary_ps, q=0.05)
    bh_path = os.path.join(args.out_dir, "bh_fdr.tsv")
    with open(bh_path, "w") as f:
        f.write("test\tp_combined\treject_BH_q05\n")
        for lbl, p, rj in zip(primary_lbls, primary_ps, reject):
            f.write(f"{lbl}\t{p:.6g}\t{bool(rj)}\n")

    print(f"per-patient: {per_patient_path}")
    print(f"cohort:      {cohort_path}")
    print(f"BH-FDR:      {bh_path}")

    print("\n=== HEADLINE (combined p via Fisher) ===")
    print(f"{'null':<10}{'p_dip':>12}{'p_spike':>12}"
          f"{'n_dip_p<.05':>14}{'n_spike_p<.05':>14}")
    for r in cohort_rows:
        print(f"{r[0]:<10}{r[3]:>12.3e}{r[5]:>12.3e}"
              f"{r[2]:>14d}{r[4]:>14d}")
    print()
    print("=== PRIMARY FAMILY (BH-FDR q=0.05; T4 to be added by LOO script) ===")
    for lbl, p, rj in zip(primary_lbls, primary_ps, reject):
        print(f"  {lbl:<15} p={p:.4g}  reject_BH_q05={bool(rj)}")


if __name__ == "__main__":
    main()
