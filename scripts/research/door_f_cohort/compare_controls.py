#!/usr/bin/env python3
# Aggregate the Phase A control-biomarker results into a single comparison
# table and effect-size summary for the manuscript revision.
#
# Inputs expected under <root>/<biomarker>/ for each biomarker:
#   permutation/cohort.tsv
#   permutation/bh_fdr.tsv
#   loo_cluster/t4_loo_cluster.json
#   cohort.tsv   (6-window per-patient assoc values)
#
# Emits:
#   <root>/phase_a_summary.tsv
#   <root>/phase_a_summary.json
#   <root>/phase_a_effect_sizes.tsv   (Cohen's d IC vs FAR per biomarker)
#
# Usage:
#   python3 scripts/research/door_f_cohort/compare_controls.py \
#       --root artifacts/research/door_f_cohort_N24/phase_a_controls
import argparse
import csv
import json
import os

import numpy as np

BIOMARKERS = ["sedenion", "permuted", "rand_tri", "bispec"]


def load_per_patient(tsv):
    out = {}
    with open(tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            p = c[idx["patient"]]
            out[p] = {w: float(c[idx[f"assoc_{w}"]])
                      for w in ["FAR", "PRE30", "PRE10", "PRE5", "IC", "POST"]}
    return out


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sp2 = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    return float((np.mean(x) - np.mean(y)) / np.sqrt(sp2 + 1e-12))


def load_perm_cohort(tsv):
    out = {}
    with open(tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            null = c[idx["null"]]
            out[null] = {
                "chi2_dip":   float(c[idx["chi2_dip"]]),
                "p_dip":      float(c[idx["p_combined_dip"]]),
                "chi2_spike": float(c[idx["chi2_spike"]]),
                "p_spike":    float(c[idx["p_combined_spike"]]),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    summary_rows = []
    effect_rows = []
    json_summary = {}

    for b in BIOMARKERS:
        bdir = os.path.join(args.root, b)
        perm = load_perm_cohort(os.path.join(bdir, "permutation", "cohort.tsv"))
        with open(os.path.join(bdir, "loo_cluster",
                               "t4_loo_cluster.json")) as f:
            loo = json.load(f)
        cohort = load_per_patient(os.path.join(bdir, "cohort.tsv"))
        ic   = np.array([cohort[p]["IC"]   for p in cohort])
        far  = np.array([cohort[p]["FAR"]  for p in cohort])
        pre5 = np.array([cohort[p]["PRE5"] for p in cohort])
        d_ic_far  = cohen_d(ic, far)
        d_ic_pre5 = cohen_d(ic, pre5)
        summary_rows.append({
            "biomarker":      b,
            "T1_dip_iid_p":   perm["iid"]["p_dip"],
            "T2_spike_iid_p": perm["iid"]["p_spike"],
            "T2_spike_circ_p": perm.get("circular", {}).get("p_spike", ""),
            "T2_spike_block_p": perm.get("block", {}).get("p_spike", ""),
            "auc_LOO":        loo["auc_point_LOO"],
            "cluster_ci_low": loo["cluster_bootstrap"]["ci95_low"],
            "cluster_ci_hi":  loo["cluster_bootstrap"]["ci95_high"],
            "cluster_p":      loo["cluster_bootstrap"]["p_one_sided"],
            "rejects_AUC_05": loo["cluster_bootstrap"]["reject_H0_AUC_eq_0p5"],
        })
        effect_rows.append({
            "biomarker":        b,
            "cohens_d_IC_FAR":  d_ic_far,
            "cohens_d_IC_PRE5": d_ic_pre5,
            "mean_IC":          float(ic.mean()),
            "mean_FAR":         float(far.mean()),
            "mean_PRE5":        float(pre5.mean()),
        })
        json_summary[b] = {"perm": perm, "loo": loo,
                           "effect_size": effect_rows[-1]}

    out_tsv = os.path.join(args.root, "phase_a_summary.tsv")
    with open(out_tsv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]),
                           delimiter="\t")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"-> {out_tsv}")

    out_es = os.path.join(args.root, "phase_a_effect_sizes.tsv")
    with open(out_es, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(effect_rows[0]),
                           delimiter="\t")
        w.writeheader()
        for r in effect_rows:
            w.writerow(r)
    print(f"-> {out_es}")

    out_json = os.path.join(args.root, "phase_a_summary.json")
    with open(out_json, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"-> {out_json}")

    print()
    print("PHASE A HEAD-TO-HEAD")
    print(f"{'biomarker':<10}{'T2_iid_p':>12}{'AUC':>8}"
          f"{'CI_low':>9}{'CI_hi':>9}{'rejects':>10}{'d_IC_FAR':>11}")
    for s, e in zip(summary_rows, effect_rows):
        rej = "yes" if s["rejects_AUC_05"] else "no"
        print(f"{s['biomarker']:<10}{s['T2_spike_iid_p']:>12.5f}"
              f"{s['auc_LOO']:>8.3f}{s['cluster_ci_low']:>9.3f}"
              f"{s['cluster_ci_hi']:>9.3f}{rej:>10}"
              f"{e['cohens_d_IC_FAR']:>11.3f}")


if __name__ == "__main__":
    main()
