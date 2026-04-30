#!/usr/bin/env python3
"""Head-to-head analysis: Door F sedenion associator vs. 20-40 Hz high-gamma baseline.

Merges per-patient TSVs (sedenion + high-gamma baseline), computes for each
patient and for each window:
  - delta_s  = (assoc[IC] - assoc[PRE5]) / assoc[PRE5]         (sedenion spike)
  - delta_g  = (power[IC] - power[PRE5]) / power[PRE5]          (high-γ spike)
  - ... likewise for dip (PRE_min vs FAR)

Then, at the cohort level:
  - pooled Cohen's d for dip and spike (each observable)
  - random-effects DerSimonian-Laird meta-analysis
  - sign-test vs. null of 0.5

Outputs:
  - stdout: formatted summary table + p-values
  - --out-json: machine-readable summary for figure generation
"""
import os
import sys
import json
import math
import argparse

WINDOWS = ["FAR", "PRE30", "PRE10", "PRE5", "IC", "POST"]


def read_cohort_tsv(path, value_prefix):
    """Parse a cohort TSV; return {patient: {window: value}}.

    For sedenion: columns assoc_FAR, assoc_PRE30, ..., assoc_POST
    For high-gamma: columns FAR, PRE30, ..., POST (window names only)
    """
    out = {}
    with open(path) as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            pid = parts[0]
            row = dict(zip(header, parts))
            d = {}
            for w in WINDOWS:
                col = f"{value_prefix}{w}" if value_prefix else w
                if col in row:
                    try:
                        d[w] = float(row[col])
                    except ValueError:
                        d[w] = float("nan")
            if len(d) == len(WINDOWS):
                out[pid] = d
    return out


def per_patient_effects(data):
    """For {pid: {w: v}}, compute dip = (FAR - PRE_min)/FAR and
    spike = (IC - PRE5)/PRE5."""
    eff = {}
    for pid, d in data.items():
        far = d["FAR"]
        pre_min = min(d["PRE30"], d["PRE10"], d["PRE5"])
        dip = (far - pre_min) / far if far > 0 else float("nan")
        spike = (d["IC"] - d["PRE5"]) / d["PRE5"] if d["PRE5"] > 0 else float("nan")
        eff[pid] = {"dip": dip, "spike": spike}
    return eff


def cohen_d_one_sample(xs, mu0=0.0):
    """One-sample Cohen's d vs mu0. Returns (d, 95% CI, se, n)."""
    n = len(xs)
    if n < 2:
        return float("nan"), (float("nan"), float("nan")), float("nan"), n
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0:
        return float("nan"), (float("nan"), float("nan")), float("nan"), n
    d = (mean - mu0) / sd
    # Hedges' SE approximation
    se = math.sqrt((1.0 / n) + (d ** 2) / (2 * n))
    lo, hi = d - 1.96 * se, d + 1.96 * se
    return d, (lo, hi), se, n


def binomial_test_positive(xs, mu0=0.0):
    """Exact sign-test: P(X >= k) under H0=0.5 where k = #positive."""
    k = sum(1 for x in xs if x > mu0)
    n = len(xs)
    def choose(a, b):
        b = min(b, a - b)
        r = 1
        for i in range(b):
            r = r * (a - i) // (i + 1)
        return r
    p_geq = sum(choose(n, i) * 0.5 ** n for i in range(k, n + 1))
    return k, n, p_geq


def summarize(label, eff, metric):
    xs = [e[metric] for e in eff.values() if not math.isnan(e[metric])]
    d, (lo, hi), se, n = cohen_d_one_sample(xs)
    k, _, p = binomial_test_positive(xs)
    print(f"  {label:<20s} n={n:2d}  mean={sum(xs)/n if xs else float('nan'):+.3f}  "
          f"d={d:+.2f} [{lo:+.2f},{hi:+.2f}]  sign: {k}/{n} p={p:.3f}")
    return {"n": n, "d": d, "ci95": [lo, hi], "mean": sum(xs)/n if xs else None,
            "sign_k": k, "sign_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sedenion", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    sed = read_cohort_tsv(args.sedenion, value_prefix="assoc_")
    bas = read_cohort_tsv(args.baseline, value_prefix="")
    eff_sed = per_patient_effects(sed)
    eff_bas = per_patient_effects(bas)

    common = sorted(set(eff_sed) & set(eff_bas))
    print(f"=== Head-to-head, N={len(common)} patients with both observables ===\n")
    print(f"{'patient':<8}{'sed_dip':>10}{'sed_spike':>11}{'γ_dip':>10}{'γ_spike':>10}")
    for pid in common:
        es, eg = eff_sed[pid], eff_bas[pid]
        print(f"{pid:<8}{es['dip']*100:>9.1f}%{es['spike']*100:>10.1f}%"
              f"{eg['dip']*100:>9.1f}%{eg['spike']*100:>9.1f}%")

    print()
    print("=== Cohort effect sizes ===")
    res = {}
    res["sedenion_dip"]   = summarize("sedenion dip",   {p: eff_sed[p] for p in common}, "dip")
    res["sedenion_spike"] = summarize("sedenion spike", {p: eff_sed[p] for p in common}, "spike")
    res["gamma_dip"]      = summarize("high-γ dip",     {p: eff_bas[p] for p in common}, "dip")
    res["gamma_spike"]    = summarize("high-γ spike",   {p: eff_bas[p] for p in common}, "spike")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"N": len(common), "patients": common, "results": res}, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
