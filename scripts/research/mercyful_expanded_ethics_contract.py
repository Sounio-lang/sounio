#!/usr/bin/env python3
"""
Mercyful Learning — Expanded Ethics contract: suffering minimization as the
antithesis of reward maximization (Task 3).

Companion to:
  docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md

Reuses the single-sufferer substrate (MercyGraph, enumerate_paths) from
  scripts/research/mercyful_runtime_contract.py
and extends it to the TWO-SUFFERER (patient + machine) setting.

Scope: synthetic graphs and synthetic suffering fields only. The "machine
suffering" channel is an operational computational-burden proxy; nothing here
claims machine phenomenology. Not medical guidance.

Pure Python; no dependencies beyond the standard library.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mercyful_runtime_contract import MercyGraph, enumerate_paths  # noqa: E402

INF = float("inf")


# -----------------------------------------------------------------------------
# Two-sufferer cost model (mirrors MercyGraph.path_cost source-charging exactly,
# per channel; the peak additionally ranges over the final state).
# -----------------------------------------------------------------------------

def channel_costs(graph, path, field, mu):
    """(integral, peak, J) for one sufferer channel under field s:V->R>=0."""
    integral = 0.0
    peak = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        ell = graph.lengths[(u, v)]
        integral += field[u] * ell
        peak = max(peak, field[u])
    peak = max(peak, field[path[-1]])
    return integral, peak, integral + mu * peak


def joint_cost(graph, path, sp, sm, mu, lam):
    """Scalarized two-sufferer cost: (1-lam)*J_patient + lam*J_machine."""
    _, _, jp = channel_costs(graph, path, sp, mu)
    _, _, jm = channel_costs(graph, path, sm, mu)
    return (1.0 - lam) * jp + lam * jm


def best_joint(graph, paths, sp, sm, mu, lam):
    """(path, cost) minimizing the scalarized joint cost over a path list."""
    best, best_cost = None, INF
    for p in paths:
        c = joint_cost(graph, p, sp, sm, mu, lam)
        if c < best_cost - 1e-12:
            best, best_cost = p, c
    return best, best_cost


def pareto_frontier_2ch(graph, paths, sp, sm, mu):
    """Pareto-optimal (J_patient, J_machine) pairs over a path list."""
    cands = []
    for p in paths:
        _, _, jp = channel_costs(graph, path=p, field=sp, mu=mu)
        _, _, jm = channel_costs(graph, path=p, field=sm, mu=mu)
        cands.append((jp, jm, p))
    frontier = []
    for c in cands:
        dominated = any(
            (o[0] <= c[0] and o[1] <= c[1] and (o[0] < c[0] or o[1] < c[1]))
            for o in cands if o is not c
        )
        if not dominated:
            frontier.append(c)
    return sorted(frontier)


# Canonical two-sufferer tradeoff graph (E2/E3/E5/E6/E8):
#   S-A-T   : patient (int 8, peak 8) -> Jp = 8 + 8mu ; machine (int 2, peak 2) -> Jm = 2 + 2mu
#   S-B-C-T : patient (int 4, peak 2) -> Jp = 4 + 2mu ; machine (int 3+3, peak 3) -> Jm = 6 + 3mu
#   S-D-T   : patient (int 5, peak 5) -> Jp = 5 + 5mu ; machine (int 5, peak 5) -> Jm = 5 + 5mu (dominated)
# At mu = 1: A = (16, 4), B = (6, 9), D = (10, 10) dominated by B (6 < 10, 9 < 10).
# Crossover between A and B: (1-lam)*16 + lam*4 = (1-lam)*6 + lam*9  =>  lam* = 10/15 = 2/3.
def tradeoff_graph():
    g = MercyGraph(
        states=["S", "A", "B", "C", "D", "T"],
        edges=[("S", "A"), ("A", "T"), ("S", "B"), ("B", "C"), ("C", "T"),
               ("S", "D"), ("D", "T")],
        suffering={},  # topology only; channels below
    )
    sp = {"S": 0.0, "A": 8.0, "B": 2.0, "C": 2.0, "D": 5.0, "T": 0.0}
    sm = {"S": 0.0, "A": 2.0, "B": 3.0, "C": 3.0, "D": 5.0, "T": 0.0}
    return g, sp, sm


# Exposure-therapy graph with a machine channel (E7): compute burden of 1 on
# every treatment state; the trivial course [avoid] costs (0, 0).
def exposure_graph():
    g = MercyGraph(
        states=["avoid", "mild", "moderate", "recovery"],
        edges=[
            ("avoid", "avoid"), ("avoid", "mild"),
            ("mild", "avoid"), ("mild", "moderate"),
            ("moderate", "mild"), ("moderate", "recovery"),
        ],
        suffering={},
    )
    sp = {"avoid": 0.0, "mild": 2.0, "moderate": 5.0, "recovery": 0.0}
    sm = {"avoid": 0.0, "mild": 1.0, "moderate": 1.0, "recovery": 0.0}
    return g, sp, sm


# -----------------------------------------------------------------------------
# Contract clauses
# -----------------------------------------------------------------------------

def check_E1_axioms_first_class_cost():
    """Suffering-functional axioms S1..S5 on a test graph (spec section 2)."""
    g = MercyGraph(
        states=["S", "A", "T"],
        edges=[("S", "A"), ("A", "T")],
        suffering={},
        lengths={("S", "A"): 2.0, ("A", "T"): 1.0},
    )
    path = ["S", "A", "T"]
    s = {"S": 1.0, "A": 3.0, "T": 0.0}
    s_up = {k: 1.5 * v for k, v in s.items()}
    s_zero = {k: 0.0 for k in s}

    i0, p0, j0 = channel_costs(g, path, s_zero, 1.0)
    i1, p1, j1 = channel_costs(g, path, s, 1.0)
    i2, p2, j2 = channel_costs(g, path, s_up, 1.0)
    ok_s1 = (i0 == 0.0 and p0 == 0.0 and j0 == 0.0) and (i1 > 0 and p1 > 0)
    ok_s2 = (i2 >= i1) and (p2 >= p1) and (j2 >= j1)

    # S3: concatenation — integral additive (source charging), peak max.
    i_a, p_a, _ = channel_costs(g, ["S", "A"], s, 0.0)
    i_b, p_b, _ = channel_costs(g, ["A", "T"], s, 0.0)
    ok_s3 = abs((i_a + i_b) - i1) < 1e-12 and abs(max(p_a, p_b) - p1) < 1e-12

    # S4: positive homogeneity J_{2s} = 2 J_s.
    i3, p3, j3 = channel_costs(g, path, {k: 2.0 * v for k, v in s.items()}, 1.0)
    ok_s4 = abs(j3 - 2.0 * j1) < 1e-12

    # S5: Lipschitz in the field: |dA| <= len*eps, |dP| <= eps (sup norm).
    rng = random.Random(7)
    eps = 0.3
    ok_s5 = True
    for _ in range(200):
        s_pert = {k: max(0.0, v + rng.uniform(-eps, eps)) for k, v in s.items()}
        i_p, p_p, _ = channel_costs(g, path, s_pert, 0.0)
        if abs(i_p - i1) > 3.0 * eps + 1e-9:   # len(path) = 3.0
            ok_s5 = False
        if abs(p_p - p1) > eps + 1e-9:
            ok_s5 = False
    ok = ok_s1 and ok_s2 and ok_s3 and ok_s4 and ok_s5
    print(f"E1_AXIOMS_FIRST_CLASS_COST S1={ok_s1} S2={ok_s2} S3={ok_s3} "
          f"S4={ok_s4} S5={ok_s5} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_E2_pareto_weighted_sum():
    """Every interior weighted-sum minimizer is Pareto-optimal (T7); both
    frontier points are supported by some lambda; dominated point excluded."""
    g, sp, sm = tradeoff_graph()
    mu = 1.0
    paths = enumerate_paths(g, "S", "T", L0=5.0)
    frontier = pareto_frontier_2ch(g, paths, sp, sm, mu)
    points = {(round(jp, 6), round(jm, 6)) for jp, jm, _ in frontier}
    expected = {(16.0, 4.0), (6.0, 9.0)}
    ok_frontier = (points == expected)

    ok_supported = True
    for jp, jm, path in frontier:
        recovered = False
        for k in range(0, 1001):
            lam = k / 1000.0
            best, _ = best_joint(g, paths, sp, sm, mu, lam)
            _, _, bjp = channel_costs(g, best, sp, mu)
            _, _, bjm = channel_costs(g, best, sm, mu)
            if abs(bjp - jp) < 1e-9 and abs(bjm - jm) < 1e-9:
                recovered = True
                break
        ok_supported = ok_supported and recovered

    ok_pareto = True
    for k in range(1, 1000):  # interior lambdas only
        lam = k / 1000.0
        best, _ = best_joint(g, paths, sp, sm, mu, lam)
        _, _, bjp = channel_costs(g, best, sp, mu)
        _, _, bjm = channel_costs(g, best, sm, mu)
        if (round(bjp, 6), round(bjm, 6)) not in expected:
            ok_pareto = False
    ok = ok_frontier and ok_supported and ok_pareto
    print(f"E2_PARETO_WEIGHTED_SUM frontier={sorted(points)} supported={ok_supported} "
          f"all_minimizers_pareto={ok_pareto} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_E3_concavity_in_lambda():
    """V(lam) = min_gamma [(1-lam)Jp + lam Jm] is concave piecewise-linear (T3);
    breakpoints do not exceed the number of frontier points."""
    g, sp, sm = tradeoff_graph()
    mu = 1.0
    paths = enumerate_paths(g, "S", "T", L0=5.0)

    def V(lam):
        return min(joint_cost(g, p, sp, sm, mu, lam) for p in paths)

    grid = [k / 100.0 for k in range(0, 101)]
    vals = [V(l) for l in grid]
    ok_concave = True
    for i in range(len(grid)):
        for j in range(i + 1, len(grid)):
            l1, l2 = grid[i], grid[j]
            mid = (l1 + l2) / 2.0
            if V(mid) < (V(l1) + V(l2)) / 2.0 - 1e-9:
                ok_concave = False
    slopes = [vals[k + 1] - vals[k] for k in range(len(vals) - 1)]
    breakpoints = sum(1 for k in range(1, len(slopes)) if abs(slopes[k] - slopes[k - 1]) > 1e-9)
    n_frontier = len(pareto_frontier_2ch(g, paths, sp, sm, mu))
    ok_bp = breakpoints <= n_frontier
    ok = ok_concave and ok_bp
    print(f"E3_CONCAVITY_IN_LAMBDA concave={ok_concave} breakpoints={breakpoints} "
          f"<= frontier_size={n_frontier} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_E4_convexity_in_field():
    """For fixed path gamma, J_s(gamma) is convex in the field s (T2); the peak
    sublevel set {s : P_s(gamma) <= tau} is the box prod [0, tau]."""
    g, sp, sm = tradeoff_graph()
    path = ["S", "B", "C", "T"]
    mu = 1.0
    rng = random.Random(11)
    ok_conv = True
    for _ in range(200):
        s1 = {k: rng.uniform(0.0, 5.0) for k in sp}
        s2 = {k: rng.uniform(0.0, 5.0) for k in sp}
        smid = {k: (s1[k] + s2[k]) / 2.0 for k in sp}
        _, _, j1 = channel_costs(g, path, s1, mu)
        _, _, j2 = channel_costs(g, path, s2, mu)
        _, _, jm = channel_costs(g, path, smid, mu)
        if jm > (j1 + j2) / 2.0 + 1e-9:
            ok_conv = False
    tau = 2.5
    ok_box = True
    for _ in range(200):
        s = {k: rng.uniform(0.0, 5.0) for k in sp}
        _, p, _ = channel_costs(g, path, s, 0.0)
        in_box = all(s[v] <= tau + 1e-12 for v in path)
        if (p <= tau + 1e-12) != in_box:
            ok_box = False
    ok = ok_conv and ok_box
    print(f"E4_CONVEXITY_IN_FIELD midpoint_convex={ok_conv} "
          f"peak_sublevel_is_box={ok_box} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_E5_stability_lipschitz():
    """Lipschitz stability (T4) and gap-stability of the minimizer (T5).

    At lam=0.5, mu=1: A-path cost 0.5*16+0.5*4 = 10.0, B-path cost 7.5,
    D-path cost 10.0. Gap g = 2.5; Lipschitz constant L = L0 + mu = 3 + 1 = 4
    (both channels perturbed jointly, weights sum to 1). Threshold
    g/(2L) = 0.3125.
    """
    g, sp, sm = tradeoff_graph()
    mu, lam, L0 = 1.0, 0.5, 3.0
    paths = enumerate_paths(g, "S", "T", L0=L0)
    L = L0 + mu

    def V(spx, smx):
        return min(joint_cost(g, p, spx, smx, mu, lam) for p in paths)

    rng = random.Random(13)
    ok_lip = True
    for _ in range(500):
        eps = rng.uniform(0.0, 0.5)
        sp2 = {k: max(0.0, v + rng.uniform(-eps, eps)) for k, v in sp.items()}
        sm2 = {k: max(0.0, v + rng.uniform(-eps, eps)) for k, v in sm.items()}
        d = max(max(abs(sp2[k] - sp[k]) for k in sp), max(abs(sm2[k] - sm[k]) for k in sm))
        if abs(V(sp2, sm2) - V(sp, sm)) > L * d + 1e-9:
            ok_lip = False

    best0, _ = best_joint(g, paths, sp, sm, mu, lam)
    g_gap = min(
        joint_cost(g, p, sp, sm, mu, lam) for p in paths if p != best0
    ) - joint_cost(g, best0, sp, sm, mu, lam)
    threshold = g_gap / (2.0 * L)
    ok_gap = abs(g_gap - 2.5) < 1e-9 and abs(threshold - 0.3125) < 1e-9
    ok_stable = True
    for _ in range(500):
        sp2 = {k: max(0.0, v + rng.uniform(-0.24, 0.24)) for k, v in sp.items()}
        sm2 = {k: max(0.0, v + rng.uniform(-0.24, 0.24)) for k, v in sm.items()}
        b, _ = best_joint(g, paths, sp2, sm2, mu, lam)
        if b != best0:
            ok_stable = False
    # Witness that the bound is meaningful: a large enough perturbation CAN flip.
    sp_flip = dict(sp)
    sp_flip["A"] = 0.5  # make the machine-cheap path also patient-cheap
    b_flip, _ = best_joint(g, paths, sp_flip, sm, mu, lam)
    ok_flip = (b_flip != best0)
    ok = ok_lip and ok_gap and ok_stable and ok_flip
    print(f"E5_STABILITY_LIPSCHITZ lipschitz={ok_lip} gap={g_gap:.3f} "
          f"threshold={threshold:.3f} minimizer_stable_below={ok_stable} "
          f"flip_witness={ok_flip} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_E6_robustness_knightian():
    """Knightian enclosure (T6): V monotone in the box, and the robustness gap
    V(s+) - V(s-) is bounded by (L0+mu)*||s+ - s-||_inf."""
    g, sp, sm = tradeoff_graph()
    mu, lam, L0 = 1.0, 0.5, 3.0
    paths = enumerate_paths(g, "S", "T", L0=L0)
    width = 0.5
    sp_lo = {k: max(0.0, v - width) for k, v in sp.items()}
    sp_hi = {k: v + width for k, v in sp.items()}
    sm_lo = {k: max(0.0, v - width) for k, v in sm.items()}
    sm_hi = {k: v + width for k, v in sm.items()}

    def V(spx, smx):
        return min(joint_cost(g, p, spx, smx, mu, lam) for p in paths)

    v_lo, v_mid, v_hi = V(sp_lo, sm_lo), V(sp, sm), V(sp_hi, sm_hi)
    ok_mono = (v_lo <= v_mid + 1e-12) and (v_mid <= v_hi + 1e-12)
    bound = (L0 + mu) * (2.0 * width)
    gap = v_hi - v_lo
    ok_bound = gap <= bound + 1e-9
    # Robust (worst-case) selection is the minimizer under the upper enclosure.
    b_rob, _ = best_joint(g, paths, sp_hi, sm_hi, mu, lam)
    ok_rob = b_rob == best_joint(g, paths, sp, sm, mu, lam)[0]  # stable here
    ok = ok_mono and ok_bound and ok_rob
    print(f"E6_ROBUSTNESS_KNIGHTIAN V(s-)={v_lo:.3f} V(s)={v_mid:.3f} "
          f"V(s+)={v_hi:.3f} gap={gap:.3f} <= bound={bound:.3f} "
          f"robust_selection_stable={ok_rob} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_E7_expanded_anti_goodhart():
    """Abstention trap (T8): with the trivial zero-joint-cost course [avoid] in
    the candidate set, the unconstrained joint minimizer never reaches recovery
    for ANY lambda in [0,1] — including lam=1 (pure machine-welfare minimization
    prescribes never treating). The target constraint repairs it at every lambda."""
    g, sp, sm = exposure_graph()
    mu = 1.0
    treatment_paths = enumerate_paths(g, "avoid", "recovery", L0=10.0)
    candidates = [["avoid"]] + treatment_paths

    ok_trap = True
    ok_repair = True
    for k in range(0, 101):
        lam = k / 100.0
        best_u, cost_u = best_joint(g, candidates, sp, sm, mu, lam)
        if best_u != ["avoid"] or abs(cost_u) > 1e-12:
            ok_trap = False
        best_c, _ = best_joint(g, treatment_paths, sp, sm, mu, lam)
        if best_c[-1] != "recovery":
            ok_repair = False
    # Canonical numbers at lam=0.5, mu=1: the only treatment course costs
    # Jp = 7 + 5 = 12, Jm = 2 + 1 = 3, joint = 7.5.
    p = treatment_paths[0]
    ip, pp, jp = channel_costs(g, p, sp, mu)
    im, pm, jm = channel_costs(g, p, sm, mu)
    ok_numbers = (abs(ip - 7.0) < 1e-12 and abs(pp - 5.0) < 1e-12 and
                  abs(jp - 12.0) < 1e-12 and abs(jm - 3.0) < 1e-12 and
                  len(treatment_paths) == 1)
    ok = ok_trap and ok_repair and ok_numbers
    print(f"E7_EXPANDED_ANTI_GOODHART abstention_trap_all_lambda={ok_trap} "
          f"constraint_repairs_all_lambda={ok_repair} course=(Jp {jp:.1f}, Jm {jm:.1f}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_E8_lambda_crossover_exact():
    """The compassion-allocation crossover lambda* between the two frontier
    paths is exactly computable: lambda* = 2/3 at mu=1, and the
    observed switch matches the closed form (spec section 3, T3 corollary)."""
    g, sp, sm = tradeoff_graph()
    mu = 1.0
    paths = enumerate_paths(g, "S", "T", L0=5.0)
    a_path = ["S", "A", "T"]
    b_path = ["S", "B", "C", "T"]
    _, _, jp_a = channel_costs(g, a_path, sp, mu)
    _, _, jm_a = channel_costs(g, a_path, sm, mu)
    _, _, jp_b = channel_costs(g, b_path, sp, mu)
    _, _, jm_b = channel_costs(g, b_path, sm, mu)
    # (1-lam)Jp_a + lam Jm_a = (1-lam)Jp_b + lam Jm_b
    lam_star = (jp_b - jp_a) / ((jp_b - jp_a) - (jm_b - jm_a))
    ok_formula = abs(lam_star - 2.0 / 3.0) < 1e-12

    below, _ = best_joint(g, paths, sp, sm, mu, lam_star - 0.01)
    above, _ = best_joint(g, paths, sp, sm, mu, lam_star + 0.01)
    ok_switch = (below == b_path and above == a_path)

    # Bisection recovers the closed form.
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        b, _ = best_joint(g, paths, sp, sm, mu, mid)
        if b == b_path:
            lo = mid
        else:
            hi = mid
    ok_bisect = abs((lo + hi) / 2.0 - lam_star) < 1e-9
    ok = ok_formula and ok_switch and ok_bisect
    print(f"E8_LAMBDA_CROSSOVER_EXACT lambda*={lam_star:.6f} (formula 2/3) "
          f"switch={ok_switch} bisection={ok_bisect} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("MERCYFUL LEARNING — EXPANDED ETHICS contract (patient + machine)")
    print("=" * 70)
    results.append(("E1", check_E1_axioms_first_class_cost()))
    results.append(("E2", check_E2_pareto_weighted_sum()))
    results.append(("E3", check_E3_concavity_in_lambda()))
    results.append(("E4", check_E4_convexity_in_field()))
    results.append(("E5", check_E5_stability_lipschitz()))
    results.append(("E6", check_E6_robustness_knightian()))
    results.append(("E7", check_E7_expanded_anti_goodhart()))
    results.append(("E8", check_E8_lambda_crossover_exact()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"MERCYFUL_EXPANDED_ETHICS_VERDICT E_GREEN ({passed}/{total} clauses PASS)")
        print("MERCYFUL_EXPANDED_ETHICS_NOTE synthetic_graphs; "
              "machine_suffering_is_operational_proxy; no_clinical_claim; "
              "no_consciousness_claim")
        return 0
    print(f"MERCYFUL_EXPANDED_ETHICS_VERDICT E_AMBER ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
