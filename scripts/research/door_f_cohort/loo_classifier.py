#!/usr/bin/env python3
# P5.4 T4 (PROTOCOL §2): Leave-One-Patient-Out logistic regression on
# z-scored per-window (dip, spike) features, pooled across folds, with
# DeLong-equivalent AUC 95% bootstrap CI.
#
# Features per (patient, window):
#   dip   = (FAR_assoc - window_assoc) / FAR_assoc
#   spike = (window_assoc - PRE5_assoc) / PRE5_assoc
# Label: 1 if window == IC, else 0.
# Windows considered (PROTOCOL §4): {FAR, PRE30, PRE10, PRE5, IC, POST}.
# The II window is not included in classification (it is the training
# baseline for the state-space model, not a probed window).
#
# Decision rule (H4): reject H0 (AUC = 0.5) iff lower bound of the 95%
# bootstrap CI on pooled AUC is strictly above 0.5. One-sided.
#
# Usage:
#   python3 scripts/research/door_f_cohort/loo_classifier.py \
#       --cohort-tsv artifacts/research/door_f_cohort_N24/door_f_cohort.tsv \
#       --out-dir    artifacts/research/door_f_cohort_N24/p5_permutation \
#       [--n-boot 2000] [--seed 20260420]

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

WINDOWS = ["FAR", "PRE30", "PRE10", "PRE5", "IC", "POST"]
IC_LABEL = "IC"


def load_cohort(path):
    out = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            p = c[idx["patient"]]
            out[p] = {w: float(c[idx[f"assoc_{w}"]]) for w in WINDOWS}
    return out


def build_examples(cohort):
    """Return (patients, X, y) with X.shape (N*W, 2), y.shape (N*W,)."""
    patients = sorted(cohort)
    X = np.zeros((len(patients) * len(WINDOWS), 2))
    y = np.zeros((len(patients) * len(WINDOWS),), dtype=int)
    pid_col = []
    win_col = []
    for pi, p in enumerate(patients):
        row = cohort[p]
        far  = row["FAR"]
        pre5 = row["PRE5"]
        for wi, w in enumerate(WINDOWS):
            a = row[w]
            dip   = (far  - a) / (far  if abs(far)  > 1e-12 else 1e-12)
            spike = (a - pre5) / (pre5 if abs(pre5) > 1e-12 else 1e-12)
            idx = pi * len(WINDOWS) + wi
            X[idx] = [dip, spike]
            y[idx] = 1 if w == IC_LABEL else 0
            pid_col.append(p)
            win_col.append(w)
    return patients, X, y, pid_col, win_col


def loo_pooled_probs(patients, X, y, pid_col):
    """Leave-One-Patient-Out logistic regression; return (probs, labels).

    Each fold trains on 23 patients (138 windows) and predicts the 6
    windows of the held-out patient. Pooled predictions are returned in
    the original example order.
    """
    probs = np.zeros(len(y))
    for held in patients:
        train_mask = np.array([p != held for p in pid_col])
        test_mask  = ~train_mask
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_mask])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 max_iter=1000)
        clf.fit(Xtr, y[train_mask])
        probs[test_mask] = clf.predict_proba(Xte)[:, 1]
    return probs


def bootstrap_auc_ci(y, probs, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        samp_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        samp_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        samp = np.concatenate([samp_pos, samp_neg])
        try:
            aucs[b] = roc_auc_score(y[samp], probs[samp])
        except ValueError:
            aucs[b] = 0.5
    low, high = np.percentile(aucs, [2.5, 97.5])
    return aucs, float(low), float(high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-tsv", required=True)
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--n-boot",     type=int, default=2000)
    ap.add_argument("--seed",       type=int, default=20260420)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cohort = load_cohort(args.cohort_tsv)
    patients, X, y, pid_col, win_col = build_examples(cohort)
    probs = loo_pooled_probs(patients, X, y, pid_col)
    auc_pooled = float(roc_auc_score(y, probs))
    aucs, lo, hi = bootstrap_auc_ci(y, probs, args.n_boot, args.seed)
    reject_H0 = bool(lo > 0.5)

    p_from_boot = float((np.sum(aucs <= 0.5) + 1) / (args.n_boot + 1))

    # Per-example predictions TSV
    pred_path = os.path.join(args.out_dir, "t4_loo_predictions.tsv")
    with open(pred_path, "w") as f:
        f.write("patient\twindow\tdip\tspike\tlabel\tp_IC\n")
        for i in range(len(y)):
            f.write(f"{pid_col[i]}\t{win_col[i]}\t{X[i,0]:.6f}\t{X[i,1]:.6f}"
                    f"\t{y[i]}\t{probs[i]:.6f}\n")

    summary = {
        "N_patients": len(patients),
        "windows": WINDOWS,
        "auc_pooled": auc_pooled,
        "auc_ci95_low": lo,
        "auc_ci95_high": hi,
        "reject_H0_AUC_eq_0p5": reject_H0,
        "p_one_sided_boot": p_from_boot,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    summary_path = os.path.join(args.out_dir, "t4_loo.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # PROTOCOL §3 "Primary family" BH-FDR across T1, T2, T4
    cohort_p_tsv = os.path.join(args.out_dir, "cohort.tsv")
    combined = {"T4_loo_auc": p_from_boot}
    if os.path.exists(cohort_p_tsv):
        with open(cohort_p_tsv) as f:
            header = f.readline().rstrip("\n").split("\t")
            for ln in f:
                c = ln.rstrip("\n").split("\t")
                row = dict(zip(header, c))
                if row.get("null") == "iid":
                    combined["T1_dip_iid"]   = float(row["p_combined_dip"])
                    combined["T2_spike_iid"] = float(row["p_combined_spike"])

        tests = sorted(combined.items(), key=lambda kv: kv[1])
        m = len(tests)
        bh_cut = [(i + 1) / m * 0.05 for i in range(m)]
        reject = [False] * m
        max_k = -1
        for k, ((name, p), cut) in enumerate(zip(tests, bh_cut)):
            if p <= cut:
                max_k = k
        for k in range(max_k + 1):
            reject[k] = True
        bh_family = [{"test": n, "p": p, "bh_cut": c, "reject": r}
                     for ((n, p), c, r) in zip(tests, bh_cut, reject)]
        summary["primary_family_BH"] = bh_family
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
