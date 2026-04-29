#!/usr/bin/env python3
"""
Phase 2 analysis — full-cohort ABIDE non-associative connectomics test.

Inputs:
- obs_dir/<SUBJECT_ID>.csv   : one CSV per subject, single row
                                group,subject_id,mean,p95,n_triples
                                (produced by associator_field_full.sio)
- null_dir/<SUBJECT_ID>.csv  : 1000 rows per subject: perm_id,p95
                                (produced by null_permutations.sio)
- zd_dir/<SUBJECT_ID>.csv    : one row per subject: group,subject_id,d_ZD
                                (produced by zero_divisor_full.sio)
- phenotypic.csv             : ABIDE phenotypic table with columns
                                FILE_ID, DX_GROUP, SITE_ID (or derivable),
                                AGE_AT_SCAN, SEX, func_mean_fd

Pipeline per protocol PROTOCOL_PHASE2.md:
1. Compute per-subject z-score:
     z^s_B = (p95^s − mean_m(p95^{s,m})) / std_m(p95^{s,m})
2. Join with phenotypic on FILE_ID.
3. Residualize z^s_B against {age, sex, mean_FD, site (dummy-coded)} via OLS.
4. Group test on residuals: Cohen's d, 10k-bootstrap 95% CI, KS two-sample.
5. Same for Experiment C (d_ZD).
6. Holm-Bonferroni correct {p_H1, p_H2}.
7. Emit publication figure + JSON summary.

Usage:
    python3 analysis_phase2.py --obs obs/ --null null/ --zd zd/ --pheno phenotypic.csv \\
                               --out artifacts/research/phase2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.stats import ks_2samp, norm as scipy_norm  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def parse_obs_csv(path: str) -> Dict[int, Tuple[str, float, float]]:
    """Parse all obs/*.csv into {subject_id: (group, mean, p95)}."""
    out: Dict[int, Tuple[str, float, float]] = {}
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if not fn.endswith(".csv"):
                continue
            for row in _iter_csv_rows(os.path.join(path, fn)):
                if len(row) != 5 or row[0] not in ("ASD", "TD"):
                    continue
                try:
                    sid = int(row[1])
                    mean, p95 = float(row[2]), float(row[3])
                    out[sid] = (row[0], mean, p95)
                except ValueError:
                    continue
    else:
        for row in _iter_csv_rows(path):
            if len(row) != 5 or row[0] not in ("ASD", "TD"):
                continue
            try:
                sid = int(row[1])
                out[sid] = (row[0], float(row[2]), float(row[3]))
            except ValueError:
                continue
    return out


def parse_null_csv(path: str) -> Dict[int, np.ndarray]:
    """Parse all null/*.csv into {subject_id: array of p95 values}."""
    out: Dict[int, List[float]] = {}
    if not os.path.isdir(path):
        print(f"[analysis] null dir not found: {path}", file=sys.stderr)
        return {}
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".csv"):
            continue
        try:
            sid = int(os.path.splitext(fn)[0])
        except ValueError:
            continue
        arr: List[float] = []
        for row in _iter_csv_rows(os.path.join(path, fn)):
            if len(row) != 2:
                continue
            try:
                arr.append(float(row[0]))
            except ValueError:
                continue
        if arr:
            out[sid] = np.asarray(arr, dtype=np.float64)
    return {k: v for k, v in out.items() if isinstance(v, np.ndarray)}


def parse_zd_csv(path: str) -> Dict[int, Tuple[str, float]]:
    out: Dict[int, Tuple[str, float]] = {}
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if not fn.endswith(".csv"):
                continue
            for row in _iter_csv_rows(os.path.join(path, fn)):
                if len(row) != 3:
                    continue
                try:
                    sid = int(row[1])
                    out[sid] = (row[0], float(row[2]))
                except ValueError:
                    continue
    return out


def _iter_csv_rows(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("group,") or line.startswith("perm_id"):
                continue
            yield line.split(",")


def parse_phenotypic(path: str) -> Dict[int, Dict]:
    """Return {subject_id (numeric FILE_ID suffix): {DX_GROUP, AGE_AT_SCAN, SEX, func_mean_fd, site}}."""
    out: Dict[int, Dict] = {}
    if not os.path.exists(path):
        print(f"[analysis] WARNING: phenotypic csv missing: {path}", file=sys.stderr)
        return out
    with open(path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fid = row.get("FILE_ID", "")
            # Extract numeric suffix — ABIDE FILE_IDs look like "NYU_0050952"
            digits = "".join(ch for ch in fid if ch.isdigit())
            if not digits:
                continue
            sid = int(digits)
            site = fid.split("_")[0] if "_" in fid else "UNKNOWN"
            try:
                out[sid] = {
                    "dx": int(row.get("DX_GROUP", 0)),
                    "age": float(row.get("AGE_AT_SCAN", 0.0)),
                    "sex": int(row.get("SEX", 0)),
                    "mean_fd": float(row.get("func_mean_fd", 0.0)),
                    "site": site,
                }
            except ValueError:
                continue
    return out


def z_score_subject(obs_p95: float, null_p95: np.ndarray) -> float:
    """z^s = (obs − mean_null) / std_null. If null is degenerate, returns 0."""
    if len(null_p95) < 2:
        return 0.0
    mu = null_p95.mean()
    sd = null_p95.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return (obs_p95 - mu) / sd


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(pooled)


def bootstrap_d_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    ds = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ai = rng.integers(0, na, size=na)
        bi = rng.integers(0, nb, size=nb)
        ds[i] = cohens_d(a[ai], b[bi])
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if HAS_SCIPY:
        r = ks_2samp(a, b)
        return float(r.statistic), float(r.pvalue)
    # Fallback: D statistic only, p via Kolmogorov approximation
    a_sort = np.sort(a)
    b_sort = np.sort(b)
    merged = np.sort(np.concatenate([a_sort, b_sort]))
    cdf_a = np.searchsorted(a_sort, merged, side="right") / len(a)
    cdf_b = np.searchsorted(b_sort, merged, side="right") / len(b)
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    # Asymptotic p: 2 · Σ (-1)^(k-1) · exp(-2·k²·(n*D²)) for n = (na·nb)/(na+nb)
    na, nb = len(a), len(b)
    n_eff = na * nb / (na + nb)
    p = 0.0
    for k in range(1, 100):
        p += (-1) ** (k - 1) * np.exp(-2.0 * k * k * n_eff * D * D)
    p = min(max(2.0 * p, 0.0), 1.0)
    return D, float(p)


def p_from_d_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 43) -> float:
    """Bootstrap p-value: fraction of bootstrap d's with opposite sign of observed."""
    obs_d = cohens_d(a, b)
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    ds = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ai = rng.integers(0, na, size=na)
        bi = rng.integers(0, nb, size=nb)
        ds[i] = cohens_d(a[ai], b[bi])
    if obs_d > 0:
        frac_wrong = float(np.mean(ds <= 0))
    else:
        frac_wrong = float(np.mean(ds >= 0))
    # Two-sided
    return min(2.0 * frac_wrong, 1.0)


def residualize(z: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Remove variance explained by design matrix via OLS. Returns residuals."""
    if design.shape[1] == 0:
        return z.copy()
    beta, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    return z - design @ beta


def build_design_matrix(pheno_rows: List[Dict]) -> np.ndarray:
    """Design matrix: intercept, age, sex, mean_fd, site dummies."""
    n = len(pheno_rows)
    ages = np.array([r["age"] for r in pheno_rows])
    sexes = np.array([r["sex"] for r in pheno_rows], dtype=float)
    fds = np.array([r["mean_fd"] for r in pheno_rows])
    sites = sorted({r["site"] for r in pheno_rows})
    if len(sites) > 1:
        sites = sites[:-1]  # drop one to avoid singular design
    site_dummies = np.zeros((n, len(sites)))
    for i, r in enumerate(pheno_rows):
        for j, site in enumerate(sites):
            if r["site"] == site:
                site_dummies[i, j] = 1.0
    cols = [np.ones(n), ages, sexes, fds] + [site_dummies[:, j] for j in range(len(sites))]
    return np.column_stack(cols)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Returns list of rejection decisions, same order as input."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n
    for k, (idx, p) in enumerate(indexed):
        threshold = alpha / (n - k)
        if p <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


def run_group_test(labels: List[str], values: np.ndarray, covariate_design: np.ndarray,
                    n_boot: int = 10_000) -> Dict:
    """Returns {cohens_d, ci_low, ci_high, ks_D, ks_p, bootstrap_p, residualized}."""
    residuals = residualize(values, covariate_design)
    asd = residuals[np.array([l == "ASD" for l in labels])]
    td = residuals[np.array([l == "TD" for l in labels])]
    d = cohens_d(asd, td)
    ci_lo, ci_hi = bootstrap_d_ci(asd, td, n_boot=n_boot)
    ks_D, ks_p = ks_two_sample(asd, td)
    p_boot = p_from_d_bootstrap(asd, td, n_boot=n_boot)
    return {
        "n_asd": int(len(asd)),
        "n_td": int(len(td)),
        "cohens_d": float(d),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "ks_D": float(ks_D),
        "ks_p": float(ks_p),
        "bootstrap_p": float(p_boot),
    }


def make_figure(h1_res: Dict, h2_res: Dict, out_png: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analysis] matplotlib unavailable — skipping figure", file=sys.stderr)
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, title, res in [(axes[0], "H1: associator p95 (z, residualized)", h1_res),
                            (axes[1], "H2: sedenion ZD distance (residualized)", h2_res)]:
        d = res["cohens_d"]
        lo = res["ci95_low"]
        hi = res["ci95_high"]
        ax.errorbar([0.5], [d], yerr=[[d - lo], [hi - d]], fmt="o", capsize=8)
        ax.axhline(0, linestyle=":", alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(min(-0.6, lo * 1.2), max(0.6, hi * 1.2))
        ax.set_title(title + f"\nd = {d:+.3f}, CI [{lo:+.3f}, {hi:+.3f}]",
                      fontsize=10)
        ax.set_xticks([])
        ax.set_ylabel("Cohen's d")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[analysis] wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="obs CSV dir or single file")
    ap.add_argument("--null", required=True, help="null CSV dir")
    ap.add_argument("--zd", required=False, default="", help="zd CSV dir (optional)")
    ap.add_argument("--pheno", required=True, help="phenotypic CSV")
    ap.add_argument("--out", required=True, help="output base path (e.g. artifacts/research/phase2)")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    obs = parse_obs_csv(args.obs)
    null_d = parse_null_csv(args.null)
    zd = parse_zd_csv(args.zd) if args.zd else {}
    pheno = parse_phenotypic(args.pheno)

    print(f"[analysis] {len(obs)} obs, {len(null_d)} null, {len(zd)} zd, {len(pheno)} pheno")

    # Cohort join on subject_id
    common = sorted(set(obs) & set(null_d) & set(pheno))
    if zd:
        common = sorted(set(common) & set(zd))
    print(f"[analysis] {len(common)} subjects after join")

    if len(common) < 20:
        print("[analysis] ERROR: cohort too small for residualization", file=sys.stderr)
        return 1

    labels = [obs[sid][0] for sid in common]
    obs_p95 = np.array([obs[sid][2] for sid in common])
    z_values = np.array([z_score_subject(obs[sid][2], null_d[sid]) for sid in common])
    pheno_rows = [pheno[sid] for sid in common]
    design = build_design_matrix(pheno_rows)

    # H1 test
    h1 = run_group_test(labels, z_values, design, n_boot=args.n_boot)
    print(f"[analysis] H1 (associator p95 z-score): d={h1['cohens_d']:+.4f}, "
          f"CI=[{h1['ci95_low']:+.4f}, {h1['ci95_high']:+.4f}], ks_p={h1['ks_p']:.4g}, boot_p={h1['bootstrap_p']:.4g}")

    h2: Dict = {}
    if zd:
        zd_values = np.array([zd[sid][1] for sid in common])
        h2 = run_group_test(labels, zd_values, design, n_boot=args.n_boot)
        print(f"[analysis] H2 (ZD distance): d={h2['cohens_d']:+.4f}, "
              f"CI=[{h2['ci95_low']:+.4f}, {h2['ci95_high']:+.4f}], ks_p={h2['ks_p']:.4g}, boot_p={h2['bootstrap_p']:.4g}")

    # Holm-Bonferroni correction
    p_values = [h1["bootstrap_p"]]
    if h2:
        p_values.append(h2["bootstrap_p"])
    rejections = holm_bonferroni(p_values, alpha=0.05)
    print(f"[analysis] Holm-Bonferroni @ α=0.05: H1 {'REJECT' if rejections[0] else 'retain'}; "
          + (f"H2 {'REJECT' if rejections[1] else 'retain'}" if len(rejections) > 1 else ""))

    # Emit JSON summary + figure
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    summary = {
        "n_subjects": len(common),
        "n_asd": int(sum(1 for l in labels if l == "ASD")),
        "n_td": int(sum(1 for l in labels if l == "TD")),
        "H1": h1,
        "H2": h2 if h2 else None,
        "holm_rejections": {"H1": bool(rejections[0]),
                            "H2": bool(rejections[1]) if len(rejections) > 1 else None},
    }
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analysis] wrote {args.out}.json")

    if h2:
        make_figure(h1, h2, args.out + ".png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
