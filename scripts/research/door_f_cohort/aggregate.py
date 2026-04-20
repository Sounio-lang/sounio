#!/usr/bin/env python3
"""Door F cohort — parse per-patient stdouts and produce cohort stats + TSV.

Inputs :
  - Legacy: artifacts/research/door_f_{patient}.stdout for each in COHORT (5)
  - Flag mode: --results-dir <dir>/door_f_<pat>.stdout + --manifest <tsv>
               supports arbitrary N (e.g. full CHB-MIT N=24).
Outputs:
  - <TSV_OUT> (per-patient assoc/mse by window)
  - stdout summary with per-patient dip/spike, sign-tests, binomial tests.
"""
import re
import os
import sys
import math
import argparse

COHORT    = ["chb02", "chb03", "chb05", "chb06", "chb10"]
WINDOWS   = ["FAR", "PRE30", "PRE10", "PRE5", "IC", "POST"]
STDOUT_FMT = "artifacts/research/door_f_{p}.stdout"
TSV_OUT   = "artifacts/research/door_f_cohort.tsv"

LINE_RE = re.compile(
    r"^\s+(FAR|PRE30|PRE10|PRE5|IC|POST)\s+[+-]?\d+s\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
BASELINE_RE = re.compile(r"^\s+baseline_(mse|assoc)=([\d.]+)\s*$")


def parse(stdout_path):
    data = {"baseline_mse": None, "baseline_assoc": None,
            "mse": {}, "assoc": {}, "mse_ratio": {}, "assoc_ratio": {}}
    with open(stdout_path) as f:
        for line in f:
            m = BASELINE_RE.match(line)
            if m:
                data[f"baseline_{m.group(1)}"] = float(m.group(2))
                continue
            m = LINE_RE.match(line)
            if m:
                w = m.group(1)
                data["mse"][w]          = float(m.group(2))
                data["mse_ratio"][w]    = float(m.group(3))
                data["assoc"][w]        = float(m.group(4))
                data["assoc_ratio"][w]  = float(m.group(5))
    if len(data["assoc"]) != 6:
        raise ValueError(f"{stdout_path}: parsed {len(data['assoc'])}/6 windows")
    return data


def wilcoxon_signed_rank(diffs):
    """Simple paired Wilcoxon signed-rank; returns (W+, n_nonzero, approx_p_two_sided).
    Uses normal approximation (N=5 is tiny, exact would be better, but we
    report this alongside exact sign-test p-values below)."""
    xs = [x for x in diffs if x != 0.0]
    n = len(xs)
    if n == 0:
        return 0.0, 0, 1.0
    ranks_ordered = sorted(range(n), key=lambda i: abs(xs[i]))
    ranks = [0] * n
    for rank_idx, orig_idx in enumerate(ranks_ordered):
        ranks[orig_idx] = rank_idx + 1
    w_plus = sum(r for r, x in zip(ranks, xs) if x > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return w_plus, n, 1.0
    z = (w_plus - mu) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    return w_plus, n, p


def binom_cdf_geq(k, n, p):
    """P(X >= k) where X ~ Binomial(n, p)."""
    def choose(a, b):
        if b < 0 or b > a:
            return 0
        b = min(b, a - b)
        num = 1
        for i in range(b):
            num = num * (a - i) // (i + 1)
        return num
    total = 0.0
    for i in range(k, n + 1):
        total += choose(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def read_manifest_cohort(manifest_path):
    cohort = []
    with open(manifest_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "patient":
                continue
            cohort.append(parts[0])
    return cohort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    help="dir containing door_f_<patient>.stdout files")
    ap.add_argument("--manifest",
                    help="cohort manifest TSV (column 1 = patient id)")
    ap.add_argument("--out-tsv", help="output TSV path")
    args = ap.parse_args()

    if args.manifest:
        cohort = read_manifest_cohort(args.manifest)
    else:
        cohort = COHORT
    results_dir = args.results_dir  # None => legacy path
    tsv_out     = args.out_tsv or TSV_OUT

    def stdout_path(p):
        if results_dir:
            return os.path.join(results_dir, f"door_f_{p}.stdout")
        return STDOUT_FMT.format(p=p)

    parsed = {}
    for p in cohort:
        path = stdout_path(p)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"  SKIP {p}: missing/empty {path}", file=sys.stderr)
            continue
        try:
            parsed[p] = parse(path)
        except ValueError as e:
            print(f"  SKIP {p}: parse error ({e})", file=sys.stderr)

    effective = [p for p in cohort if p in parsed]
    N = len(effective)
    if N == 0:
        sys.exit("no parseable results")

    os.makedirs(os.path.dirname(tsv_out) or ".", exist_ok=True)
    with open(tsv_out, "w") as f:
        hdr = ["patient", "baseline_mse", "baseline_assoc"]
        for w in WINDOWS:
            hdr += [f"mse_{w}", f"assoc_{w}"]
        f.write("\t".join(hdr) + "\n")
        for p in effective:
            d = parsed[p]
            row = [p, f"{d['baseline_mse']:.6f}", f"{d['baseline_assoc']:.6f}"]
            for w in WINDOWS:
                row += [f"{d['mse'][w]:.6f}", f"{d['assoc'][w]:.6f}"]
            f.write("\t".join(row) + "\n")
    print(f"wrote {tsv_out}  (N={N})")
    print()

    print("=" * 78)
    print(f"  Per-patient pattern metrics  (N={N})")
    print("=" * 78)
    print(f"{'patient':<8}{'FAR':>8}{'PRE_min':>9}{'PRE5':>8}{'IC':>8}{'POST':>8}"
          f"{'dip%':>8}{'spike%':>8}  flags")
    dips, spikes, both = [], [], []
    for p in effective:
        d         = parsed[p]["assoc"]
        far       = d["FAR"]
        pre_min   = min(d["PRE30"], d["PRE10"], d["PRE5"])
        dip       = (far - pre_min) / far if far > 0 else 0.0
        spike     = (d["IC"] - d["PRE5"]) / d["PRE5"] if d["PRE5"] > 0 else 0.0
        dip_flag  = "D" if dip   > 0 else "-"
        spike_flag= "S" if spike > 0 else "-"
        print(f"{p:<8}{far:>8.2f}{pre_min:>9.2f}{d['PRE5']:>8.2f}"
              f"{d['IC']:>8.2f}{d['POST']:>8.2f}{dip*100:>7.1f}%{spike*100:>7.1f}%  "
              f"{dip_flag}{spike_flag}")
        dips.append(dip); spikes.append(spike)
        both.append(1 if (dip > 0 and spike > 0) else 0)

    print()
    print("=" * 78)
    print(f"  Cohort-level tests  (N={N} patients)")
    print("=" * 78)
    w1, n1, p1 = wilcoxon_signed_rank([d for d in dips])
    n_dip   = sum(1 for d in dips   if d > 0)
    n_spike = sum(1 for s in spikes if s > 0)
    print(f"Dip test   (H1: assoc[PRE_min] < assoc[FAR]):")
    print(f"  W+={w1:.1f}  n={n1}  Wilcoxon approx two-sided p={p1:.3f}")
    print(f"  sign-test: {n_dip}/{N} positive  "
          f"p(>=k|H0=0.5) = {binom_cdf_geq(n_dip, N, 0.5):.3f}")
    w2, n2, p2 = wilcoxon_signed_rank([s for s in spikes])
    print(f"Spike test (H1: assoc[IC] > assoc[PRE5]):")
    print(f"  W+={w2:.1f}  n={n2}  Wilcoxon approx two-sided p={p2:.3f}")
    print(f"  sign-test: {n_spike}/{N} positive  "
          f"p(>=k|H0=0.5) = {binom_cdf_geq(n_spike, N, 0.5):.3f}")
    k = sum(both)
    p_chance = 0.25
    print(f"Dip-AND-Spike co-occurrence: {k}/{N} patients")
    print(f"  Binomial test P(X>={k}|H0 p={p_chance}) = "
          f"{binom_cdf_geq(k, N, p_chance):.3f}")
    print()
    print("Interpretation:")
    print("  - sign-test p <= 0.05 for both dip and spike ⇒ cohort-universal pattern")
    print("  - else: report as observable with per-patient-replicability, not biomarker")


if __name__ == "__main__":
    main()
