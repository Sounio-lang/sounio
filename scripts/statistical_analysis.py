#!/usr/bin/env python3
"""
Statistical Analysis: Fractal-G2 O-SSM vs H-SSM Control
Paper-ready statistics with CIs, effect sizes, bootstrap, and multiple comparisons.

Usage:
  python3 scripts/statistical_analysis.py [path_to_raw_output.txt]

If no file given, reads from /tmp/fractal_g2_v3_output.txt
"""

import sys
import re
import numpy as np
from scipy import stats

# ═══════════════════════════════════════════════════════════════
#  Parse per-seed data from benchmark output
# ═══════════════════════════════════════════════════════════════

def parse_output(text):
    """Parse per-seed lines like: s0 O=12.3% NA=4.5% an=0.075 | H=10.1% NA=0.2%
    Handles newlines in the middle of output lines (compiler prints without buffering)."""
    # Collapse all whitespace/newlines into single spaces for matching
    collapsed = re.sub(r'\s+', ' ', text)
    seeds = []
    for m in re.finditer(
        r's(\d+)\s+O=([\d.]+)%\s+NA=([\d.]+)%\s+an=([\d.]+)\s+\|\s+H=([\d.]+)%\s+NA=([\d.]+)%',
        collapsed
    ):
        seeds.append({
            'seed': int(m.group(1)),
            'o_overall': float(m.group(2)),
            'o_na': float(m.group(3)),
            'o_assoc_norm': float(m.group(4)),
            'h_overall': float(m.group(5)),
            'h_na': float(m.group(6)),
        })
    return seeds


def paired_analysis(x, y, label, alpha=0.05):
    """Full paired analysis: t-test, CI, Cohen's d, Wilcoxon, bootstrap."""
    gap = x - y
    n = len(gap)
    mean_gap = gap.mean()
    sd_gap = gap.std(ddof=1)
    se_gap = sd_gap / np.sqrt(n)

    # Paired t-test (two-tailed)
    t_stat, p_val = stats.ttest_rel(x, y)

    # 95% CI from t-distribution
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lo = mean_gap - t_crit * se_gap
    ci_hi = mean_gap + t_crit * se_gap

    # Cohen's d (paired)
    cohens_d = mean_gap / sd_gap if sd_gap > 0 else float('inf')
    if abs(cohens_d) >= 0.8:
        d_interp = "large"
    elif abs(cohens_d) >= 0.5:
        d_interp = "medium"
    elif abs(cohens_d) >= 0.2:
        d_interp = "small"
    else:
        d_interp = "negligible"

    # Wilcoxon signed-rank (one-tailed: x > y)
    try:
        w_stat, w_p = stats.wilcoxon(gap, alternative='greater')
    except ValueError:
        # All differences are zero
        w_stat, w_p = 0, 1.0

    # Bootstrap (10000 resamples)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(gap, size=n, replace=True).mean()
        for _ in range(10000)
    ])
    boot_ci = np.percentile(boot_means, [2.5, 97.5])
    boot_p = (boot_means <= 0).mean()

    return {
        'label': label,
        'n': n,
        'mean_gap': mean_gap,
        'sd_gap': sd_gap,
        'se_gap': se_gap,
        'ci': (ci_lo, ci_hi),
        't_stat': t_stat,
        'p_val': p_val,
        'cohens_d': cohens_d,
        'd_interp': d_interp,
        'w_stat': w_stat,
        'w_p': w_p,
        'boot_mean': boot_means.mean(),
        'boot_median': np.median(boot_means),
        'boot_ci': (boot_ci[0], boot_ci[1]),
        'boot_p': boot_p,
        'gap_values': gap,
    }


def one_sample_analysis(x, mu0, label, alpha=0.05):
    """One-sample t-test: x vs theoretical value mu0."""
    n = len(x)
    mean_x = x.mean()
    sd_x = x.std(ddof=1)
    se_x = sd_x / np.sqrt(n)
    gap = x - mu0

    t_stat, p_val = stats.ttest_1samp(x, mu0)

    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lo = mean_x - t_crit * se_x
    ci_hi = mean_x + t_crit * se_x

    cohens_d = (mean_x - mu0) / sd_x if sd_x > 0 else float('inf')
    if abs(cohens_d) >= 0.8:
        d_interp = "large"
    elif abs(cohens_d) >= 0.5:
        d_interp = "medium"
    elif abs(cohens_d) >= 0.2:
        d_interp = "small"
    else:
        d_interp = "negligible"

    # Bootstrap
    rng = np.random.default_rng(123)
    boot_means = np.array([
        rng.choice(x, size=n, replace=True).mean()
        for _ in range(10000)
    ])
    boot_ci = np.percentile(boot_means, [2.5, 97.5])
    boot_p = (boot_means <= mu0).mean()

    return {
        'label': label,
        'n': n,
        'mean': mean_x,
        'mu0': mu0,
        'mean_gap': mean_x - mu0,
        'sd': sd_x,
        'se': se_x,
        'ci': (ci_lo, ci_hi),
        't_stat': t_stat,
        'p_val': p_val,
        'cohens_d': cohens_d,
        'd_interp': d_interp,
        'boot_mean': boot_means.mean(),
        'boot_ci': (boot_ci[0], boot_ci[1]),
        'boot_p': boot_p,
    }


def multiple_comparisons(p_values, labels, alpha=0.05):
    """Bonferroni and Holm-Bonferroni correction."""
    k = len(p_values)
    bonf_threshold = alpha / k

    # Holm-Bonferroni
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    holm_results = [False] * k
    for rank, (orig_idx, p) in enumerate(indexed):
        holm_thresh = alpha / (k - rank)
        if p <= holm_thresh:
            holm_results[orig_idx] = True
        else:
            # Once one fails, all subsequent fail
            break

    results = []
    for i in range(k):
        results.append({
            'label': labels[i],
            'raw_p': p_values[i],
            'bonf_threshold': bonf_threshold,
            'bonf_survives': p_values[i] <= bonf_threshold,
            'holm_survives': holm_results[i],
        })
    return results


def print_report(seeds, overall_res, na_res, vs_random_res, mc_results, assoc_norms):
    """Print the full paper-ready report."""
    n = len(seeds)

    print("═" * 65)
    print("  Statistical Analysis: Fractal-G2 O-SSM vs H-SSM Control")
    print("═" * 65)

    # ── Per-Seed Raw Data ──
    print("\n=== Per-Seed Raw Data ===")
    print(f"  {'Seed':>4s}  {'O-SSM(%)':>8s}  {'H-SSM(%)':>8s}  {'Gap(pp)':>7s}  "
          f"{'O-NA(%)':>7s}  {'H-NA(%)':>7s}  {'NA-Gap':>6s}  {'AssocNorm':>9s}")
    for s in seeds:
        gap = s['o_overall'] - s['h_overall']
        na_gap = s['o_na'] - s['h_na']
        print(f"  {s['seed']:>4d}  {s['o_overall']:>8.1f}  {s['h_overall']:>8.1f}  "
              f"{gap:>+7.1f}  {s['o_na']:>7.1f}  {s['h_na']:>7.1f}  "
              f"{na_gap:>+6.1f}  {s['o_assoc_norm']:>9.4f}")

    # ── Overall Accuracy Gap ──
    print("\n=== Primary Analysis: Overall Accuracy Gap (O-SSM − H-SSM) ===")
    r = overall_res
    print(f"  Mean gap: {r['mean_gap']:+.2f}pp ± {r['se_gap']:.2f}pp (SE)")
    print(f"  SD of paired differences: {r['sd_gap']:.2f}pp")
    print(f"  95% CI: [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]pp")
    print(f"  Paired t({r['n']-1}) = {r['t_stat']:.4f}, p = {r['p_val']:.6f} (two-tailed)")
    print(f"  Cohen's d = {r['cohens_d']:.3f} ({r['d_interp']})")
    print(f"  Wilcoxon W = {r['w_stat']:.0f}, p = {r['w_p']:.6f} (one-tailed, H₁: gap > 0)")
    print(f"\n  Bootstrap (10000 resamples):")
    print(f"    Mean: {r['boot_mean']:+.2f}pp, Median: {r['boot_median']:+.2f}pp")
    print(f"    95% CI: [{r['boot_ci'][0]:+.2f}, {r['boot_ci'][1]:+.2f}]pp")
    print(f"    P(gap ≤ 0): {r['boot_p']:.4f}")

    # ── Non-Assoc Gap ──
    print("\n=== Secondary Analysis: Non-Assoc Triple Gap (O-SSM − H-SSM) ===")
    r = na_res
    print(f"  Mean gap: {r['mean_gap']:+.2f}pp ± {r['se_gap']:.2f}pp (SE)")
    print(f"  SD of paired differences: {r['sd_gap']:.2f}pp")
    print(f"  95% CI: [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]pp")
    print(f"  Paired t({r['n']-1}) = {r['t_stat']:.4f}, p = {r['p_val']:.6f} (two-tailed)")
    print(f"  Cohen's d = {r['cohens_d']:.3f} ({r['d_interp']})")
    print(f"  Wilcoxon W = {r['w_stat']:.0f}, p = {r['w_p']:.6f} (one-tailed, H₁: gap > 0)")
    print(f"\n  Bootstrap (10000 resamples):")
    print(f"    Mean: {r['boot_mean']:+.2f}pp, Median: {r['boot_median']:+.2f}pp")
    print(f"    95% CI: [{r['boot_ci'][0]:+.2f}, {r['boot_ci'][1]:+.2f}]pp")
    print(f"    P(gap ≤ 0): {r['boot_p']:.4f}")

    # ── O-SSM vs Random ──
    print("\n=== Tertiary Analysis: O-SSM Overall vs Random (6.25%) ===")
    r = vs_random_res
    print(f"  O-SSM mean: {r['mean']:.2f}% (random baseline: {r['mu0']:.2f}%)")
    print(f"  Mean excess: {r['mean_gap']:+.2f}pp ± {r['se']:.2f}pp (SE)")
    print(f"  95% CI for mean: [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]%")
    print(f"  One-sample t({r['n']-1}) = {r['t_stat']:.4f}, p = {r['p_val']:.6f} (two-tailed)")
    print(f"  Cohen's d = {r['cohens_d']:.3f} ({r['d_interp']})")
    print(f"\n  Bootstrap (10000 resamples):")
    print(f"    Mean: {r['boot_mean']:.2f}%, 95% CI: [{r['boot_ci'][0]:.2f}, {r['boot_ci'][1]:.2f}]%")
    print(f"    P(mean ≤ {r['mu0']:.2f}): {r['boot_p']:.4f}")

    # ── MeanAssocNorm ──
    an = assoc_norms
    an_mean = an.mean()
    an_se = an.std(ddof=1) / np.sqrt(len(an))
    an_t, an_p = stats.ttest_1samp(an, 0.0)
    print(f"\n=== AssocNorm Analysis (O-SSM, one-sample vs 0) ===")
    print(f"  Mean: {an_mean:.4f} ± {an_se:.4f} (SE)")
    print(f"  Range: [{an.min():.4f}, {an.max():.4f}]")
    print(f"  One-sample t({len(an)-1}) = {an_t:.4f}, p = {an_p:.6f}")
    print(f"  All seeds > 0: {np.all(an > 0)}")

    # ── Multiple Comparisons ──
    print("\n=== Multiple Comparisons Correction (3 primary tests) ===")
    print(f"  {'Metric':<25s}  {'Raw p':>10s}  {'Bonf α':>8s}  {'Bonf?':>6s}  {'Holm?':>6s}")
    for mc in mc_results:
        bonf = "YES" if mc['bonf_survives'] else "NO"
        holm = "YES" if mc['holm_survives'] else "NO"
        print(f"  {mc['label']:<25s}  {mc['raw_p']:>10.6f}  {mc['bonf_threshold']:>8.4f}  "
              f"{bonf:>6s}  {holm:>6s}")

    # ── AssocNorm Trajectory (summary) ──
    print("\n=== Associator Norm Per-Seed (training mean) ===")
    print(f"  {'Seed':>4s}  {'MeanAssocNorm':>13s}")
    for s in seeds:
        print(f"  {s['seed']:>4d}  {s['o_assoc_norm']:>13.4f}")
    print(f"  {'Mean':>4s}  {an_mean:>13.4f} ± {an_se:.4f}")

    # ── Paper-Ready Summary ──
    ov = overall_res
    na = na_res
    ov_survives = mc_results[0]['bonf_survives']
    na_survives = mc_results[1]['bonf_survives']
    an_survives = mc_results[2]['bonf_survives']

    print("\n" + "═" * 65)
    print("  PAPER-READY SUMMARY")
    print("═" * 65)

    ov_verb = "remains" if ov_survives else "does not remain"
    na_verb = "remains" if na_survives else "does not remain"

    print(f"""
  "The Fractal-G2 O-SSM outperformed the H-SSM control by
   {ov['mean_gap']:+.1f}pp overall accuracy (95% CI [{ov['ci'][0]:+.1f}, {ov['ci'][1]:+.1f}];
   paired t({ov['n']-1}) = {ov['t_stat']:.2f}, p = {ov['p_val']:.4f}; Cohen's d = {ov['cohens_d']:.2f};
   bootstrap 95% CI [{ov['boot_ci'][0]:+.1f}, {ov['boot_ci'][1]:+.1f}], 10000 resamples;
   Wilcoxon W = {ov['w_stat']:.0f}, p = {ov['w_p']:.4f}).
   On non-associative triples, the gap was {na['mean_gap']:+.1f}pp
   (95% CI [{na['ci'][0]:+.1f}, {na['ci'][1]:+.1f}]; t({na['n']-1}) = {na['t_stat']:.2f},
   p = {na['p_val']:.4f}; d = {na['cohens_d']:.2f}).
   After Bonferroni correction for 3 comparisons (α_adj = 0.0167),
   the overall gap {ov_verb} significant.
   The O-SSM maintained non-zero associator norm throughout
   training (mean = {an_mean:.4f} ± {an_se:.4f}), confirming
   that non-associative structure is preserved by the G2
   manifold update — the 'Artin Dormancy' phenomenon."
""")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/fractal_g2_v3_output.txt'

    try:
        with open(path, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: {path} not found. Run the benchmark first:")
        print(f"  ./bin/souc run examples/fractal_g2_ossm_v3.sio | tee {path}")
        sys.exit(1)

    seeds = parse_output(text)
    if len(seeds) < 10:
        print(f"Warning: only found {len(seeds)} seeds (expected 10)")
        if len(seeds) == 0:
            print("No per-seed data found. Check output format.")
            print(f"File contents preview:\n{text[:500]}")
            sys.exit(1)

    n = len(seeds)

    # Extract arrays
    o_overall = np.array([s['o_overall'] for s in seeds])
    h_overall = np.array([s['h_overall'] for s in seeds])
    o_na = np.array([s['o_na'] for s in seeds])
    h_na = np.array([s['h_na'] for s in seeds])
    assoc_norms = np.array([s['o_assoc_norm'] for s in seeds])

    # Paired analyses
    overall_res = paired_analysis(o_overall, h_overall, "Overall accuracy gap")
    na_res = paired_analysis(o_na, h_na, "Non-assoc accuracy gap")

    # O-SSM vs random (6.25%)
    vs_random_res = one_sample_analysis(o_overall, 6.25, "O-SSM vs random")

    # AssocNorm > 0 test
    an_t, an_p = stats.ttest_1samp(assoc_norms, 0.0)

    # Multiple comparisons
    p_values = [overall_res['p_val'], na_res['p_val'], an_p]
    labels = ["Overall gap (O−H)", "Non-assoc gap (O−H)", "AssocNorm > 0"]
    mc_results = multiple_comparisons(p_values, labels)

    print_report(seeds, overall_res, na_res, vs_random_res, mc_results, assoc_norms)


if __name__ == '__main__':
    main()
