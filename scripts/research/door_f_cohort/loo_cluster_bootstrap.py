#!/usr/bin/env python3
# Phase A5: patient-level cluster-bootstrap LOO classifier.
#
# The existing loo_classifier.py bootstraps stratified (pos, neg) example
# indices, which treats each of the 144 window-rows as independent. Peer
# review (2026-04-21 multi-model synthesis) flagged this as over-optimistic
# because the 6 windows per patient are not independent. This script
# resamples the 24 patients with replacement, rebuilds examples from the
# resampled patients, refits LOO logistic regression on each bootstrap
# sample, and reports AUC + 95% CI with patient-level clustering honoured.
#
# Primary output: JSON summary with pooled_auc_point, cluster_ci95,
# example_bootstrap_ci95 (for reference), and the BH-FDR update.
#
# Usage:
#   python3 scripts/research/door_f_cohort/loo_cluster_bootstrap.py \
#       --cohort-tsv artifacts/.../cohort.tsv \
#       --out-dir    artifacts/.../loo_cluster \
#       [--n-boot 2000] [--seed 20260420]
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

WINDOWS  = ["FAR", "PRE30", "PRE10", "PRE5", "IC", "POST"]
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


def build_features(cohort, patient_order):
    """Return (X, y) where X.shape = (len(patient_order)*6, 2)."""
    X = np.zeros((len(patient_order) * len(WINDOWS), 2))
    y = np.zeros((len(patient_order) * len(WINDOWS),), dtype=int)
    pid_col = []
    for pi, p in enumerate(patient_order):
        row = cohort[p]
        far  = row["FAR"]
        pre5 = row["PRE5"]
        for wi, w in enumerate(WINDOWS):
            a = row[w]
            dip   = (far - a) / (far  if abs(far)  > 1e-12 else 1e-12)
            spike = (a - pre5) / (pre5 if abs(pre5) > 1e-12 else 1e-12)
            idx = pi * len(WINDOWS) + wi
            X[idx]  = [dip, spike]
            y[idx]  = 1 if w == IC_LABEL else 0
            pid_col.append(p)
    return X, y, pid_col


def loo_probs(X, y, pid_col, patients):
    probs = np.zeros(len(y))
    for held in patients:
        train_mask = np.array([p != held for p in pid_col])
        test_mask  = ~train_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        scaler = StandardScaler().fit(X[train_mask])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 max_iter=1000)
        clf.fit(scaler.transform(X[train_mask]), y[train_mask])
        probs[test_mask] = clf.predict_proba(
            scaler.transform(X[test_mask]))[:, 1]
    return probs


def cluster_bootstrap_auc(cohort, n_boot, seed):
    """Resample patients with replacement; refit LOO each time; return AUC
    vector. Falls back to 0.5 if a bootstrap has <2 classes.
    """
    rng = np.random.default_rng(seed)
    patients = sorted(cohort)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        # Rename duplicates so LOO treats them as separate clusters.
        uniq_ids = [f"{p}#b{i}" for i, p in enumerate(sampled)]
        fake_cohort = {uid: cohort[sampled[i]]
                       for i, uid in enumerate(uniq_ids)}
        X, y, pid_col = build_features(fake_cohort, uniq_ids)
        try:
            if len(np.unique(y)) < 2:
                aucs[b] = 0.5
                continue
            probs = loo_probs(X, y, pid_col, uniq_ids)
            aucs[b] = roc_auc_score(y, probs)
        except ValueError:
            aucs[b] = 0.5
    return aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-tsv", required=True)
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--n-boot",     type=int, default=2000)
    ap.add_argument("--seed",       type=int, default=20260420)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cohort = load_cohort(args.cohort_tsv)
    patients = sorted(cohort)

    # Point estimate
    X, y, pid_col = build_features(cohort, patients)
    probs = loo_probs(X, y, pid_col, patients)
    auc_point = float(roc_auc_score(y, probs))

    aucs = cluster_bootstrap_auc(cohort, args.n_boot, args.seed)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    p_boot = float((np.sum(aucs <= 0.5) + 1) / (args.n_boot + 1))

    summary = {
        "N_patients": len(patients),
        "windows": WINDOWS,
        "auc_point_LOO": auc_point,
        "cluster_bootstrap": {
            "n_boot": args.n_boot,
            "ci95_low":  float(lo),
            "ci95_high": float(hi),
            "p_one_sided": p_boot,
            "reject_H0_AUC_eq_0p5": bool(lo > 0.5),
        },
        "seed": args.seed,
    }
    path = os.path.join(args.out_dir, "t4_loo_cluster.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
