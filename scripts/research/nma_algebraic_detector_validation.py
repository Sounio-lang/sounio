#!/usr/bin/env python3
"""
NMA Algebraic Inconsistency Detector Validation
================================================
Constellation 3 — Novelty Weather Map garden seed.
Validation gate for the NMA non-associative algebra spin-out.

QUESTION
  Can any non-standard algebraic detector flag NMA inconsistency that the
  standard Bucher z-test misses?

FOUR DETECTORS
  D0 — Bucher z-score          (scalar additive cocycle, design-agnostic)
  D1 — Multivariate Hotelling T² (vector cocycle, two correlated outcomes)
  D2 — Design-weighted cocycle  (regression-adjusted, uses design features)
  D3 — Octonion associator     (embeds effect + design into O, measures [a,b,c])

FIVE SCENARIOS (2000 networks each)
  S0 — Fully consistent               (negative control)
  S1 — Simple scalar inconsistency     (positive control, D0 should win)
  S2 — Design-confounded inconsistency (design features bias effects)
  S3 — Design-masked inconsistency     (biases cancel on the scalar loop)
  S4 — Multivariate inconsistency      (scalar projection cancels, joint doesn't)

FALSIFICATION
  If D3 never beats D0 on any scenario, Route C (octonion) is refuted.
  If D1 beats D0 on S4, Route A (multivariate) is supported.
  If D2 beats D0 on S2/S3, design information adds value.

ALGEBRA NOTE
  The additive NMA consistency condition is a 1-cocycle on the treatment
  graph with values in R.  Inconsistency is a nonzero H^1 class:

    Δ(A,B,C) = c_AB + c_BC + c_CA

  This is abelian and associative — no natural associator.  Non-associativity
  can only enter through Route A (multivariate non-commutative composition),
  Route B (sequential/path-dependent effects), or Route C (imposed O embedding).
  This simulation lets the data decide which route, if any, has practical value.

  See: docs/proposals/nma_nonassociative_algebra_note.md
"""

import numpy as np
import json
import sys
from itertools import combinations

# ============================================================
# OCTONION ARITHMETIC (Cayley–Dickson / Fano plane)
# ============================================================

_FANO = [(1, 2, 4), (2, 3, 5), (3, 4, 6),
         (4, 5, 7), (5, 6, 1), (6, 7, 2), (7, 1, 3)]


def _build_mul_tensor():
    """8×8×8 octonion multiplication tensor T[i,j,k] such that
    (a*b)_k = sum_{i,j} a_i b_j T[i,j,k]."""
    sign = np.zeros((8, 8))
    idx = np.zeros((8, 8), dtype=int)
    # Real unit e_0 is identity
    for i in range(8):
        sign[i, 0] = 1; idx[i, 0] = i
        sign[0, i] = 1; idx[0, i] = i
    # e_i^2 = -e_0
    for i in range(1, 8):
        sign[i, i] = -1; idx[i, i] = 0
    # Fano plane triples
    for a, b, c in _FANO:
        for p, q, r in [(a, b, c), (b, c, a), (c, a, b)]:
            sign[p, q] = 1; idx[p, q] = r
        for p, q, r in [(b, a, c), (c, b, a), (a, c, b)]:
            sign[p, q] = -1; idx[p, q] = r
    T = np.zeros((8, 8, 8))
    for i in range(8):
        for j in range(8):
            T[i, j, idx[i, j]] = sign[i, j]
    return T


_MUL = _build_mul_tensor()


def oct_mul(a, b):
    """Multiply two octonions (8-vectors)."""
    return np.einsum('i,j,ijk->k', a, b, _MUL, optimize=True)


def associator(a, b, c):
    """[a, b, c] = (a*b)*c - a*(b*c)."""
    return oct_mul(oct_mul(a, b), c) - oct_mul(a, oct_mul(b, c))


# ============================================================
# SYNTHETIC NETWORK GENERATION
# ============================================================

N_DESIGN = 7  # 7 design features → 7 imaginary octonion components


def generate_network(scenario, rng, n_outcomes=2):
    """Generate a 3-treatment loop A-B-C.

    Returns a dict with per-edge observed effects, SEs, design features,
    and a boolean `inconsistent` ground-truth label.
    """
    edges = ['AB', 'BC', 'AC']

    # True treatment effects (A = 0 reference)
    theta = np.concatenate([[0.0], rng.normal(0, 0.5, size=2)])

    # For multivariate: each edge has n_outcomes effect components
    if n_outcomes > 1:
        theta_multi = np.zeros((3, n_outcomes))
        for k in range(n_outcomes):
            t = np.concatenate([[0.0], rng.normal(0, 0.5, size=2)])
            theta_multi[:, k] = t

    true_scalar = {
        'AB': theta[1] - theta[0],
        'BC': theta[2] - theta[1],
        'AC': theta[2] - theta[0],
    }

    # Design features: 7 per edge, uniform [-1, 1]
    design = {e: rng.uniform(-1, 1, size=N_DESIGN) for e in edges}

    # Standard errors
    se = {e: rng.uniform(0.10, 0.25) for e in edges}

    # Observed effects start from truth
    obs = dict(true_scalar)
    obs_multi = {}
    if n_outcomes > 1:
        for e in edges:
            i, j = {'AB': (0, 1), 'BC': (1, 2), 'AC': (0, 2)}[e]
            obs_multi[e] = theta_multi[j] - theta_multi[i]

    inconsistent = False

    if scenario == 'S0':
        # Fully consistent — only sampling error
        pass

    elif scenario == 'S1':
        # Simple scalar inconsistency: perturb one edge
        edge = rng.choice(edges)
        shift = rng.normal(0.4, 0.1)
        obs[edge] += shift
        if n_outcomes > 1:
            obs_multi[edge] += shift
        inconsistent = True

    elif scenario == 'S2':
        # Design-confounded: design features bias observed effects
        # The bias is a linear function of design features
        beta = rng.uniform(0.15, 0.35)
        # Random design-effect direction
        w = rng.normal(0, 1, size=N_DESIGN)
        w /= np.linalg.norm(w)
        for e in edges:
            bias = beta * np.dot(design[e], w)
            obs[e] += bias
            if n_outcomes > 1:
                obs_multi[e] += bias
        # Loop doesn't close because designs differ across edges
        loop_bias = sum(obs[e] * s for e, s in
                        [('AB', 1), ('BC', 1), ('AC', -1)])
        # But sampling might mask it — flag as inconsistent if design truly differs
        design_spread = np.max([np.linalg.norm(design['AB'] - design['BC']),
                                np.linalg.norm(design['BC'] - design['AC']),
                                np.linalg.norm(design['AB'] - design['AC'])])
        inconsistent = design_spread > 0.5  # not always, but usually

    elif scenario == 'S3':
        # Design-masked: biases cancel on the scalar loop
        # (AB + BC - AC biases = 0) but individual edges ARE biased
        biases = rng.normal(0, 0.2, size=2)  # bias_AB, bias_BC
        bias_ac = biases[0] + biases[1]      # forces loop cancellation
        edge_biases = {'AB': biases[0], 'BC': biases[1], 'AC': bias_ac}
        for e in edges:
            obs[e] += edge_biases[e]
            if n_outcomes > 1:
                obs_multi[e] += edge_biases[e]
        # The network IS inconsistent (edges are biased), but the scalar
        # loop test sees zero net inconsistency
        inconsistent = np.max(np.abs(list(edge_biases.values()))) > 0.15

    elif scenario == 'S4':
        # Multivariate: two outcome dimensions, scalar projection cancels
        if n_outcomes >= 2:
            # Inject inconsistency in outcome 2 that cancels outcome 1
            edge = rng.choice(edges)
            obs_multi[edge][0] += 0.2   # efficacy shift
            obs_multi[edge][1] -= 0.2   # safety shift (opposite sign)
            # Scalar (outcome 1 only): small inconsistency
            obs[edge] = obs_multi[edge][0]
            inconsistent = True
        else:
            # Fallback: just do scalar
            edge = rng.choice(edges)
            obs[edge] += rng.normal(0.3, 0.1)
            inconsistent = True

    # Add sampling error
    for e in edges:
        noise = rng.normal(0, se[e])
        obs[e] += noise
        if n_outcomes > 1:
            obs_multi[e] += rng.normal(0, se[e], size=n_outcomes)

    return {
        'effect': obs,
        'effect_multi': obs_multi if n_outcomes > 1 else None,
        'se': se,
        'design': design,
        'true_effect': true_scalar,
        'inconsistent': inconsistent,
        'scenario': scenario,
        'n_outcomes': n_outcomes,
    }


# ============================================================
# DETECTORS
# ============================================================

def detect_bucher(net):
    """D0: Standard Bucher z-score on loop AB + BC - AC.

    z = Δ / SE(Δ), where Δ = δ_AB + δ_BC - δ_AC.
    Returns |z| as the detection statistic (higher = more inconsistent).
    """
    delta = net['effect']['AB'] + net['effect']['BC'] - net['effect']['AC']
    se_delta = np.sqrt(net['se']['AB']**2 + net['se']['BC']**2 + net['se']['AC']**2)
    return abs(delta) / se_delta


def detect_hotelling(net):
    """D1: Multivariate Hotelling T² on the cocycle defect.

    Requires n_outcomes ≥ 2.  The cocycle defect is a vector Δ ∈ R^k.
    T² = Δᵀ Σ⁻¹ Δ where Σ is the diagonal SE matrix (no cross-study cov).
    """
    if net['effect_multi'] is None:
        return detect_bucher(net)
    edges = ['AB', 'BC', 'AC']
    signs = [1, 1, -1]
    k = net['n_outcomes']
    delta = np.zeros(k)
    for e, s in zip(edges, signs):
        delta += s * net['effect_multi'][e]
    # Covariance of delta = sum of per-edge covariances (independent studies)
    sigma = np.zeros((k, k))
    for e, s in zip(edges, signs):
        sigma += (s * net['se'][e])**2 * np.eye(k)
    # Add small correlation between outcomes (as in real meta-analysis)
    rho = 0.3
    for i in range(k):
        for j in range(k):
            if i != j:
                sigma[i, j] += rho * net['se']['AB'] * net['se']['BC']
    try:
        t2 = delta @ np.linalg.solve(sigma, delta)
    except np.linalg.LinAlgError:
        t2 = 0.0
    return np.sqrt(max(t2, 0))


def detect_design_weighted(net):
    """D2: Design-regressed cocycle.

    Regress the loop defect on design feature differences between edges.
    If design features explain the inconsistency, the regression R² is high
    and the design-adjusted defect drops.

    Returns the design interaction score: how much of the loop defect is
    explained by design feature interactions.
    """
    edges = ['AB', 'BC', 'AC']
    signs = np.array([1, 1, -1])

    # Loop defect
    delta = sum(s * net['effect'][e] for e, s in zip(edges, signs))

    # Design feature differences: pairwise products across edges
    # (captures which design feature interactions correlate with the defect)
    d = net['design']
    features = []
    for i in range(N_DESIGN):
        # Design feature of the edge with largest weight in the loop
        features.append(
            signs[0] * d['AB'][i] + signs[1] * d['BC'][i] + signs[2] * d['AC'][i]
        )
    feat_vec = np.array(features)
    feat_norm = np.linalg.norm(feat_vec)

    # The design interaction score: |defect| weighted by how aligned it is
    # with the design feature direction
    if feat_norm < 1e-10:
        return abs(delta) / np.sqrt(sum(s**2 * se**2 for s, se in
                zip(signs, [net['se'][e] for e in edges])))
    # Design-weighted detection: the defect is more suspicious if the design
    # features are spread (different designs across edges)
    design_spread = np.mean([
        np.linalg.norm(d['AB'] - d['BC']),
        np.linalg.norm(d['BC'] - d['AC']),
        np.linalg.norm(d['AB'] - d['AC']),
    ])
    se_delta = np.sqrt(sum(s**2 * se**2 for s, se in
                           zip(signs, [net['se'][e] for e in edges])))
    return abs(delta) / se_delta * (0.5 + 0.5 * design_spread)


def detect_octonion_associator(net):
    """D3: Octonion associator norm.

    Embed each edge as an octonion:
      o_e = δ_e * e_0 + σ * D_e   (effect in real part, design in imaginary)

    Compute [o_AB, o_BC, o_AC] = (AB·BC)·AC - AB·(BC·AC).
    Return the Euclidean norm of the associator.
    """
    sigma = 0.5  # design scaling — how much design features contribute

    def embed(edge):
        o = np.zeros(8)
        o[0] = net['effect'][edge]
        o[1:] = sigma * net['design'][edge]
        return o

    o_ab = embed('AB')
    o_bc = embed('BC')
    o_ac = embed('AC')

    assoc = associator(o_ab, o_bc, o_ac)
    return np.linalg.norm(assoc)


# ============================================================
# EVALUATION
# ============================================================

def auroc(scores, labels):
    """Area under the ROC curve.  Higher score = more inconsistent."""
    scores = np.array(scores)
    labels = np.array(labels, dtype=bool)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores)
    ranks = np.zeros(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Handle ties via average ranks
    unique_scores = np.unique(scores)
    for s in unique_scores:
        mask = scores == s
        ranks[mask] = ranks[mask].mean()
    sum_ranks_pos = ranks[labels].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def run_experiment(n_networks=2000, n_outcomes=2, seed=20260806):
    """Run the full experiment: generate, detect, evaluate."""
    rng = np.random.default_rng(seed)

    scenarios = ['S0', 'S1', 'S2', 'S3', 'S4']
    detectors = {
        'D0_bucher': detect_bucher,
        'D1_hotelling': detect_hotelling,
        'D2_design': detect_design_weighted,
        'D3_octonion': detect_octonion_associator,
    }

    # Generate all networks
    nets_by_scenario = {}
    for sc in scenarios:
        nets = []
        for _ in range(n_networks):
            nets.append(generate_network(sc, rng, n_outcomes=n_outcomes))
        nets_by_scenario[sc] = nets

    # For each detector, compute scores on each scenario
    results = {}
    for dname, dfn in detectors.items():
        results[dname] = {}
        for sc in scenarios:
            scores = [dfn(net) for net in nets_by_scenario[sc]]
            results[dname][sc] = scores

    # AUROC for each detector on each binary task:
    # Task 1: S0 vs S1 (can it detect simple inconsistency?)
    # Task 2: S0 vs S2 (design-confounded)
    # Task 3: S0 vs S3 (design-masked)
    # Task 4: S0 vs S4 (multivariate)
    # Task 5: S0 vs (S1+S2+S3+S4) (all inconsistency types)

    tasks = {
        'S0_vs_S1': ('S0', 'S1'),
        'S0_vs_S2': ('S0', 'S2'),
        'S0_vs_S3': ('S0', 'S3'),
        'S0_vs_S4': ('S0', 'S4'),
        'S0_vs_ALL': ('S0', None),  # None = all non-S0
    }

    print("\n" + "=" * 72)
    print("NMA ALGEBRAIC INCONSISTENCY DETECTOR VALIDATION")
    print("=" * 72)
    print(f"Networks per scenario: {n_networks}")
    print(f"Outcomes per edge:     {n_outcomes}")
    print(f"Seed:                  {seed}")
    print()

    # Header
    dnames = list(detectors.keys())
    header = f"{'Task':<16}" + "".join(f"{d:>16}" for d in dnames)
    print(header)
    print("-" * len(header))

    auroc_table = {}
    for tname, (neg_sc, pos_sc) in tasks.items():
        row = {}
        cells = [f"{tname:<16}"]
        for dname in dnames:
            neg_scores = results[dname][neg_sc]
            neg_labels = [False] * len(neg_scores)

            if pos_sc is None:
                pos_scores = []
                pos_labels = []
                for sc in scenarios:
                    if sc == neg_sc:
                        continue
                    pos_scores.extend(results[dname][sc])
                    pos_labels.extend([nets_by_scenario[sc][i]['inconsistent']
                                      for i in range(len(nets_by_scenario[sc]))])
            else:
                pos_scores = results[dname][pos_sc]
                pos_labels = [net['inconsistent'] for net in nets_by_scenario[pos_sc]]

            # Use ground-truth labels for AUROC
            all_scores = np.array(neg_scores + pos_scores)
            all_labels = np.array([False] * len(neg_scores) + pos_labels)

            # If no positive labels, use scenario as label
            if all_labels.sum() == 0:
                all_labels = np.array([False] * len(neg_scores) + [True] * len(pos_scores))

            au = auroc(all_scores.tolist(), all_labels.tolist())
            row[dname] = au
            cells.append(f"{au:>16.3f}")
        auroc_table[tname] = row
        print("".join(cells))

    print()

    # Score distributions
    print("Score distributions (median [IQR] per scenario):")
    print()
    dist_header = f"{'Detector':<16}" + "".join(f"{sc:>14}" for sc in scenarios)
    print(dist_header)
    print("-" * len(dist_header))
    for dname in dnames:
        cells = [f"{dname:<16}"]
        for sc in scenarios:
            s = np.array(results[dname][sc])
            cells.append(f"{np.median(s):>8.2f} [{np.percentile(s,25):.2f}-{np.percentile(s,75):.2f}]")
        print("".join(cells))

    # False positive rate at z>1.96 threshold (per scenario S0)
    print()
    print("False positive rate at |score| > median(S0) + 1.5*MAD(S0):")
    fpr_header = f"{'Detector':<16}" + "".join(f"{sc:>14}" for sc in scenarios)
    print(fpr_header)
    print("-" * len(fpr_header))
    for dname in dnames:
        s0 = np.array(results[dname]['S0'])
        med = np.median(s0)
        mad = np.median(np.abs(s0 - med))
        thresh = med + 1.5 * mad
        cells = [f"{dname:<16}"]
        for sc in scenarios:
            s = np.array(results[dname][sc])
            fpr = np.mean(s > thresh)
            cells.append(f"{fpr:>14.3f}")
        print("".join(cells))

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    # Check key claims
    for task_name in ['S0_vs_S2', 'S0_vs_S3', 'S0_vs_S4']:
        d0 = auroc_table[task_name].get('D0_bucher', 0)
        d3 = auroc_table[task_name].get('D3_octonion', 0)
        d2 = auroc_table[task_name].get('D2_design', 0)
        d1 = auroc_table[task_name].get('D1_hotelling', 0)
        best = max(d0, d1, d2, d3)
        best_name = [k for k, v in [('D0', d0), ('D1', d1), ('D2', d2), ('D3', d3)]
                     if v == best][0]
        print(f"  {task_name}: best = {best_name} ({best:.3f}), "
              f"D0={d0:.3f}, D3={d3:.3f}")

    # Verdict
    d3_beats_d0 = any(
        auroc_table[t].get('D3_octonion', 0) > auroc_table[t].get('D0_bucher', 0) + 0.02
        for t in auroc_table
    )
    print()
    if d3_beats_d0:
        print("  D3 (octonion) BEATS D0 (Bucher) on at least one task → Route C not refuted.")
    else:
        print("  D3 (octonion) does NOT beat D0 (Bucher) on any task → Route C REFUTED for this embedding.")

    return {
        'auroc': auroc_table,
        'n_networks': n_networks,
        'n_outcomes': n_outcomes,
        'seed': seed,
    }


def sensitivity_sigma_sweep(n_networks=1000, seed=42):
    """Test how the octonion design scaling σ affects D3 performance."""
    print("\n" + "=" * 72)
    print("SENSITIVITY: Octonion design scaling σ sweep")
    print("=" * 72)
    print(f"{'σ':<10}{'S0_vs_S2 AUROC':>20}{'S0_vs_S3 AUROC':>20}")
    print("-" * 50)

    sigmas = [0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    for sigma in sigmas:
        rng = np.random.default_rng(seed)
        nets = {}
        for sc in ['S0', 'S2', 'S3']:
            nets[sc] = [generate_network(sc, rng) for _ in range(n_networks)]

        # Patch the sigma
        import types
        orig = sys.modules['__main__'].__dict__.get('detect_octonion_associator')

        def patched_detect(net, _sigma=sigma):
            def embed(edge):
                o = np.zeros(8)
                o[0] = net['effect'][edge]
                o[1:] = _sigma * net['design'][edge]
                return o
            assoc = associator(embed('AB'), embed('BC'), embed('AC'))
            return np.linalg.norm(assoc)

        for task, (neg, pos) in [('S0_vs_S2', ('S0', 'S2')), ('S0_vs_S3', ('S0', 'S3'))]:
            scores = ([patched_detect(n) for n in nets[neg]] +
                      [patched_detect(n) for n in nets[pos]])
            labels = [False] * len(nets[neg]) + [True] * len(nets[pos])
            au = auroc(scores, labels)
            if task == 'S0_vs_S2':
                s2_au = au
            else:
                s3_au = au

        print(f"{sigma:<10.2f}{s2_au:>20.3f}{s3_au:>20.3f}")


if __name__ == '__main__':
    results = run_experiment(n_networks=2000, n_outcomes=2, seed=20260806)
    sensitivity_sigma_sweep(n_networks=1000, seed=42)

    # Save results
    outpath = "scripts/research/nma_detector_results.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")
