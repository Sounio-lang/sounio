#!/usr/bin/env python3
# Generate publication figures for the sedenion-associator seizure preprint
# from committed TSV/JSON artifacts under artifacts/research/door_f_cohort_N24/.
#
# Outputs (PDF + PNG) under artifacts/paper-drafts/figures/:
#   fig_window_trajectory.*   per-window associator norm, cohort strip + median
#   fig_permutation_summary.* Fisher-combined p across iid/circular/block
#   fig_loo_roc.*             LOO ROC with AUC 95% CI
#   fig_t3_robust.*           T3 cohort-median scatter across 100 draws
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path("artifacts/research/door_f_cohort_N24")
OUT = Path("artifacts/paper-drafts/figures")
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = ["FAR", "PRE30", "PRE10", "PRE5", "IC", "POST"]


def load_cohort():
    rows = list(csv.DictReader(open(ART / "door_f_cohort.tsv"), delimiter="\t"))
    mat = np.array([[float(r[f"assoc_{w}"]) for w in WINDOWS] for r in rows])
    return [r["patient"] for r in rows], mat


def load_cohort_pvalues():
    p = {}
    for r in csv.DictReader(open(ART / "p5_permutation_100k/cohort.tsv"),
                            delimiter="\t"):
        p[r["null"]] = {"dip": float(r["p_combined_dip"]),
                        "spike": float(r["p_combined_spike"])}
    return p


def fig_window_trajectory():
    pats, mat = load_cohort()
    meds = np.median(mat, axis=0)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(WINDOWS))
    for i in range(mat.shape[0]):
        ax.plot(x, mat[i], color="#b0b0b0", lw=0.6, alpha=0.6)
    ax.plot(x, meds, color="#1f77b4", lw=2.2, marker="o",
            markersize=5, label="cohort median (N=24)")
    ax.set_xticks(x)
    ax.set_xticklabels(WINDOWS)
    ax.set_xlabel("window")
    ax.set_ylabel(r"associator norm $\|[a,b,c]\|_2$")
    ax.axvline(4, color="k", lw=0.4, alpha=0.3)
    ax.text(4.02, ax.get_ylim()[1] * 0.95, "seizure onset",
            fontsize=7, alpha=0.6)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_yscale("log")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_window_trajectory.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_permutation_summary():
    pvals = load_cohort_pvalues()
    nulls = ["iid", "circular", "block"]
    labels = ["iid", "circular shift", "10-epoch block"]
    spike = [pvals[n]["spike"] for n in nulls]
    dip = [pvals[n]["dip"] for n in nulls]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(len(nulls))
    w = 0.36
    ax.bar(x - w/2, spike, w, color="#d62728",
           label=r"T2 spike (IC$>$PRE5)")
    ax.bar(x + w/2, dip, w, color="#7f7f7f",
           label=r"T1 dip (pre-ictal)")
    ax.axhline(0.05, color="k", ls=":", lw=0.8)
    ax.text(2.35, 0.06, "p = 0.05", fontsize=7, color="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fisher combined p (100k permutations)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.1)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_permutation_summary.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def _roc(labels, scores):
    order = np.argsort(-scores)
    y = labels[order]
    P = y.sum()
    N = len(y) - P
    tpr = np.cumsum(y) / max(P, 1)
    fpr = np.cumsum(1 - y) / max(N, 1)
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = np.concatenate([[0], tpr, [1]])
    return fpr, tpr


def fig_loo_roc():
    rows = list(csv.DictReader(
        open(ART / "p5_permutation_100k/t4_loo_predictions.tsv"),
        delimiter="\t"))
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([float(r["p_IC"]) for r in rows])
    fpr, tpr = _roc(y, s)
    j = json.load(open(ART / "p5_permutation_100k/t4_loo.json"))
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
    ax.plot(fpr, tpr, color="#1f77b4", lw=2,
            label=f"AUC = {j['auc_pooled']:.3f} "
                  f"[{j['auc_ci95_low']:.3f}, {j['auc_ci95_high']:.3f}]")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title(f"T4 LOO classifier  (N=24, p={j['p_one_sided_boot']:.4f})",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_loo_roc.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_t3_robust():
    rows = list(csv.DictReader(
        open(ART / "p5_t3_channel_robust/t3_cohort_by_draw.tsv"),
        delimiter="\t"))
    d = np.array([float(r["dip_median"]) for r in rows])
    s = np.array([float(r["spike_median"]) for r in rows])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.scatter(d, s, s=22, alpha=0.75, color="#1f77b4",
               edgecolor="k", lw=0.3)
    ax.set_xlabel("dip median  (cohort, N=24)")
    ax.set_ylabel("spike median  (cohort, N=24)")
    ax.set_title("T3 channel-subset robustness  (100 draws, 16/23 ch)",
                 fontsize=9)
    ax.text(0.05, 0.95,
            f"sign preserved:\n  dip   {int((d>0).sum())}/100\n"
            f"  spike {int((s>0).sum())}/100",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="0.7", pad=3))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_t3_robust.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_window_trajectory()
    fig_permutation_summary()
    fig_loo_roc()
    fig_t3_robust()
    print("wrote:", sorted(p.name for p in OUT.glob("fig_*.pdf")))
