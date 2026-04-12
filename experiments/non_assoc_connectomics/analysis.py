#!/usr/bin/env python3
"""
Phase 1 — Non-Associative Epistemic Connectomics
analysis.py — Cohen's d + 10k-bootstrap 95% CI + KS distribution test.

Reads the CSV output of associator_field.sio (either from a file or stdin):
  group,mean,p95,n_triples,subject_id
  ASD,8.93,30.39,4060,0
  ...

Lines beginning with '#' are treated as diagnostics and skipped. Lines that do
not parse as data are silently ignored (a stdlib quirk emits an integer line
at end-of-stream in some builds).

Emits:
  - stdout: effect-size summary
  - artifacts/research/non_assoc_connectomics_pilot.png (matplotlib)
  - artifacts/research/non_assoc_connectomics_pilot.json (point estimate + CI)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PNG = os.path.join(ROOT, "artifacts", "research", "non_assoc_connectomics_pilot.png")
OUT_JSON = os.path.join(ROOT, "artifacts", "research", "non_assoc_connectomics_pilot.json")


def parse_records(handle) -> List[dict]:
    records = []
    for raw in handle:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("group,"):
            continue
        parts = line.split(",")
        if len(parts) != 5:
            continue
        try:
            records.append({
                "group": parts[0],
                "mean": float(parts[1]),
                "p95": float(parts[2]),
                "n_triples": int(parts[3]),
                "subject_id": int(parts[4]),
            })
        except ValueError:
            continue
    return records


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    # Two-sample, pooled SD, corrected (unbiased pooled variance)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(pooled)


def bootstrap_d_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    ds = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ai = rng.integers(0, na, size=na)
        bi = rng.integers(0, nb, size=nb)
        ds[i] = cohens_d(a[ai], b[bi])
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    # Two-sample KS. Return (D, p). Use scipy if available; else hand-rolled D only.
    try:
        from scipy.stats import ks_2samp  # type: ignore
        res = ks_2samp(a, b)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        merged = np.sort(np.concatenate([a, b]))
        cdf_a = np.searchsorted(np.sort(a), merged, side="right") / len(a)
        cdf_b = np.searchsorted(np.sort(b), merged, side="right") / len(b)
        D = float(np.max(np.abs(cdf_a - cdf_b)))
        return D, float("nan")


def make_plot(asd_p95: np.ndarray, td_p95: np.ndarray, d_hat: float,
              ci_lo: float, ci_hi: float, ks_D: float, ks_p: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analysis] matplotlib unavailable; skipping PNG")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    positions = [1, 2]
    data = [asd_p95, td_p95]
    bp = ax.boxplot(data, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], ["#c74449", "#3a7ca5"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    # Scatter
    rng = np.random.default_rng(0)
    for pos, d in zip(positions, data):
        jitter = rng.normal(0, 0.05, size=len(d))
        ax.scatter(pos + jitter, d, color="black", s=25, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"ASD (n={len(asd_p95)})", f"TD (n={len(td_p95)})"])
    ax.set_ylabel("p95 of ‖[L_i, L_j, L_k]‖² (octonion associator)")
    title = (f"Phase 1 pilot: Cohen's d = {d_hat:+.3f}  "
             f"[{ci_lo:+.3f}, {ci_hi:+.3f}]   "
             f"KS D = {ks_D:.3f}, p = {ks_p:.3g}")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"[analysis] wrote {OUT_PNG}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default="-",
                   help="CSV file (or '-' for stdin)")
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()

    if args.input == "-":
        handle = sys.stdin
    else:
        handle = open(args.input, "r")

    records = parse_records(handle)
    if args.input != "-":
        handle.close()

    asd = np.array([r["p95"] for r in records if r["group"] == "ASD"])
    td = np.array([r["p95"] for r in records if r["group"] == "TD"])
    print(f"[analysis] parsed {len(asd)} ASD + {len(td)} TD records")

    if len(asd) < 2 or len(td) < 2:
        print("[analysis] ERROR: need ≥2 per group")
        return 1

    d_hat = cohens_d(asd, td)
    ci_lo, ci_hi = bootstrap_d_ci(asd, td, n_boot=args.n_boot)
    ks_D, ks_p = ks_two_sample(asd, td)
    ci_crosses_zero = (ci_lo <= 0 <= ci_hi)

    print(f"[analysis] Cohen's d (ASD − TD on p95) = {d_hat:+.4f}")
    print(f"[analysis] 95% bootstrap CI              = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"[analysis] CI crosses zero?              = {ci_crosses_zero}")
    print(f"[analysis] KS two-sample D               = {ks_D:.4f}")
    print(f"[analysis] KS two-sample p               = {ks_p:.4g}")

    summary = {
        "n_asd": int(len(asd)),
        "n_td": int(len(td)),
        "mean_asd_p95": float(asd.mean()),
        "mean_td_p95": float(td.mean()),
        "cohens_d": float(d_hat),
        "ci95_lower": float(ci_lo),
        "ci95_upper": float(ci_hi),
        "ci_crosses_zero": bool(ci_crosses_zero),
        "ks_D": float(ks_D),
        "ks_p": float(ks_p) if not np.isnan(ks_p) else None,
        "n_boot": int(args.n_boot),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analysis] wrote {OUT_JSON}")

    make_plot(asd, td, d_hat, ci_lo, ci_hi, ks_D, ks_p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
