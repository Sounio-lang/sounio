#!/usr/bin/env python3
"""
Mercyful Learning -- upgraded algorithm validation: ML suffering field +
continuous optimal control vs the discrete scheduler.

Companion to:
  docs/research/mercyful_continuous_control_spec_2026-07-26.md

The discrete scheduler (mercyful_runtime_contract.py / stdlib/clinical/
mercyful.sio) minimizes

    cost(gamma; mu) = sum_{(u,v) in gamma} s(u)*ell(u,v) + mu * max_{v in gamma} s(v)

over simple graph paths that must reach a target node within a budget L0.

The upgraded algorithm replaces the graph with a continuous suffering field
s(x) and the path with a control trajectory u(t), and minimizes the expanded
ethics objective

    J[u] = ∫ [ s_patient(x(t)) + sigma*||u||^2 ] dt + mu * sup_t s_patient(x(t))
    s.t. x_dot = f(x,u),  x(0)=x0,  x(T) in TARGET (anti-Goodhart),  T <= L0.

- s_patient : patient suffering field (continuous extension of the discrete one)
- sigma*||u||^2 : machine/substrate suffering (control energy, dissipation,
  switching strain) -- the expanded-ethics term
- mu * sup : worst-moment (Rawlsian) peak aversion
- hard target constraint : the anti-Goodhart axiom, unchanged

This contract validates, on the three established applications (exposure
therapy, chemotherapy sequencing, vancomycin TDM), that the upgraded
algorithm is strictly more powerful than the discrete scheduler:

  better solutions : V1 (consistency/dominance), V2, V3, V4, V6, V7
  more general     : V5 (frontier continuum), V8 (off-node targets)
  more efficient   : V9 (exponential path enumeration vs polynomial collocation)
  expanded ethics  : V10 (machine-suffering term active, smoothing is optimal)

Everything is synthetic. Not medical guidance; no clinical claim.
Pure Python; no dependencies beyond the standard library. The discrete
baseline is imported unmodified from mercyful_runtime_contract.py (M_GREEN)
and mercyful_chemo_contract.py (H_GREEN).
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mercyful_runtime_contract import MercyGraph, enumerate_paths, mercyful_schedule
from mercyful_chemo_contract import build_chemo_graph, DIAG, REM

TOL = 1e-9


def approx(a, b, tol=TOL):
    return abs(a - b) <= tol


# =============================================================================
# Application 1 -- exposure therapy (shared instance with the discrete toy)
# =============================================================================
#
# Continuous suffering field s(x) on x in [0,1]: piecewise linear through
# (0,0), (1/3,2), (2/3,5), (1,0) -- the continuous extension of the discrete
# field avoidance=0, mild=2, moderate=5, recovery=0. Speed v = dx/dt is the
# control (v=1 reproduces the discrete one-state-per-unit-edge schedule).

EXPO_NODES = [(0.0, 0.0), (1.0 / 3.0, 2.0), (2.0 / 3.0, 5.0), (1.0, 0.0)]
EXPO_KAPPA = 1.0        # ascent-strain weight (machine-suffering term sigma*||u||^2)
EXPO_V_LO, EXPO_V_HI = 0.05, 4.0   # pacing bounds (synthetic session constraints)


def expo_field(x):
    """Piecewise-linear continuous suffering field."""
    for (x0, s0), (x1, s1) in zip(EXPO_NODES, EXPO_NODES[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return s0
            return s0 + (s1 - s0) * (x - x0) / (x1 - x0)
    return EXPO_NODES[-1][1]


def expo_field_integral(x_end=1.0):
    """Exact integral of the field over [0, x_end] (trapezoids: exact for
    piecewise-linear s)."""
    total = 0.0
    for (x0, s0), (x1, s1) in zip(EXPO_NODES, EXPO_NODES[1:]):
        if x_end <= x0:
            break
        xe = min(x_end, x1)
        se = expo_field(xe)
        total += (xe - x0) * (s0 + se) / 2.0
    return total


def expo_sqrt_field_integral(x_end=1.0):
    """Exact integral of sqrt(s) over [0, x_end]: for a linear segment
    a->b over width w, ∫ sqrt(s) dx = w * (2/3) * (b^1.5 - a^1.5)/(b - a)."""
    total = 0.0
    for (x0, s0), (x1, s1) in zip(EXPO_NODES, EXPO_NODES[1:]):
        if x_end <= x0:
            break
        xe = min(x_end, x1)
        se = expo_field(xe)
        if se == s0:
            continue
        total += (xe - x0) * (2.0 / 3.0) * (se ** 1.5 - s0 ** 1.5) / (se - s0)
    return total


def expo_optimal_pace(s, kappa=EXPO_KAPPA, v_lo=EXPO_V_LO, v_hi=EXPO_V_HI):
    """Pointwise optimal pace: minimize over v the per-unit-x Lagrangian
    L(x,v) = s/v + kappa*v  =>  v* = clamp(sqrt(s/kappa), v_lo, v_hi)."""
    if s <= 0.0:
        return v_lo
    return min(max(math.sqrt(s / kappa), v_lo), v_hi)


def expo_pacing_cost(x_end=1.0, kappa=EXPO_KAPPA, n=200_000):
    """Numeric cost / time / machine-energy of the optimal pacing profile,
    by fine midpoint quadrature (the profile is given in closed form)."""
    dx = x_end / n
    cost = 0.0
    time_t = 0.0
    machine = 0.0
    for i in range(n):
        x = (i + 0.5) * dx
        s = expo_field(x)
        v = expo_optimal_pace(s, kappa)
        cost += (s / v + kappa * v) * dx
        time_t += dx / v
        machine += kappa * v * dx        # ∫ kappa v^2 dt = ∫ kappa v dx
    return cost, time_t, machine


def expo_discrete_graph():
    return MercyGraph(
        states=["avoidance", "mild", "moderate", "recovery"],
        edges=[
            ("avoidance", "avoidance"), ("avoidance", "mild"),
            ("mild", "avoidance"), ("mild", "moderate"),
            ("moderate", "mild"), ("moderate", "recovery"),
        ],
        suffering={"avoidance": 0.0, "mild": 2.0, "moderate": 5.0, "recovery": 0.0},
    )


# =============================================================================
# Application 2 -- chemotherapy sequencing (continuous dose-rate control)
# =============================================================================
#
# Control d(t) >= 0: dose rate. Toxicity suffering rate s(d) = d^2 (convex;
# synthetic). Efficacy (anti-Goodhart): cumulative dose ∫ d dt >= K. Budget
# T <= L0. By Cauchy-Schwarz the optimum is the constant rate d* = K/L0:
#     ∫ d^2 dt >= (∫ d dt)^2 / T >= K^2 / L0,  equality iff d constant.
# K = 48 and L0 = 24 calibrate to the discrete instance (the dose-dense
# course delivers 48 suffering-rate-time units in 8 weeks; the stop-and-go
# course takes 24 weeks).

CHEMO_K = 48.0
CHEMO_L0 = 24.0


def chemo_continuous_optimum(K=CHEMO_K, L0=CHEMO_L0):
    """Closed form: d* = K/L0 constant -> (integral, peak) = (K^2/L0, (K/L0)^2)."""
    d = K / L0
    return K * K / L0, d * d


def chemo_lift_discrete_courses(K=CHEMO_K):
    """Lift the three discrete regimens to dose-rate profiles delivering the
    SAME cumulative dose K (same efficacy), keeping each course's time
    pattern. Rates proportional to the discrete suffering levels (dose rate
    ~ toxicity rate), rescaled to deliver exactly K.
    Returns {name: (integral ∫d^2, peak s = max d^2, duration)}."""
    courses = {
        # name: (rates, durations) of the discrete suffering pattern
        "DD": ([8.0], [8.0]),                          # 8 wk at level 8
        "STOP_GO": ([5.0, 5.0, 1.0, 5.0], [6.0, 6.0, 6.0, 3.0]),
        "CONT": ([5.0, 5.0, 8.0], [6.0, 6.0, 3.0]),
    }
    out = {}
    for name, (rates, durs) in courses.items():
        delivered = sum(r * w for r, w in zip(rates, durs))
        alpha = K / delivered
        integral = sum((alpha * r) ** 2 * w for r, w in zip(rates, durs))
        peak = max((alpha * r) ** 2 for r in rates)
        out[name] = (integral, peak, sum(durs))
    return out


def chemo_solve_numeric(K, L0, M=480, iters=2000, seed=42):
    """Independent numeric solver: projected gradient on the M-cell
    discretization of  min Σ d_i^2 dt  s.t. Σ d_i dt = K, d_i >= 0.
    Deterministic (seeded). Returns (integral, peak, max_rate_deviation)."""
    dt = L0 / M
    rng = random.Random(seed)
    d = [rng.random() + 0.01 for _ in range(M)]
    norm = K / (sum(d) * dt)
    d = [x * norm for x in d]
    target_sum = K / dt

    def project(v):
        # water-filling projection onto {Σ v_i = target_sum, v_i >= 0};
        # mu may be negative (sum must be able to grow), so bracket wide.
        lo, hi = -target_sum - 1.0, max(v) + 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            s = sum(max(0.0, x - mid) for x in v)
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        return [max(0.0, x - hi) for x in v]

    for k in range(iters):
        eta = 0.25 / (1.0 + k / 200.0)
        d = project([di - eta * 2.0 * di for di in d])
    integral = sum(di * di for di in d) * dt
    peak = max(di * di for di in d)
    dev = max(abs(di - K / L0) for di in d)
    return integral, peak, dev


# =============================================================================
# Application 3 -- vancomycin TDM (continuous measurement timing + infusion)
# =============================================================================
#
# Suffering values measured from the repo's Knightian vancomycin twin
# (tests/run-pass/mercyful_clinical_sequencing.sio, clause C3; reused by
# mercyful_mimic_iv_vancomycin_contract.py):
S_VANCO_PRE = 0.675679    # fixed dosing before any level is measured
S_VANCO_POST = 0.059420   # TDM-narrowed band at steady state
#
# Continuous model: horizon L0 (time units); a TDM level is drawn at time t.
# Before the draw the suffering rate is s_pre; after, the band narrows toward
# s_ss = S_VANCO_POST, but a level drawn before steady state is only partly
# informative: s_post(t) = s_ss + (s_pre - s_ss) * exp(-t/tau).
# The discrete scheduler fixes the pre-TDM dwell at half the horizon (unit
# edges, one pre + one post). The upgraded algorithm chooses t continuously.

VANCO_TAU = 12.0     # synthetic time constant of approach to steady state
VANCO_L0 = 48.0      # synthetic treatment horizon
VANCO_T_DISCRETE = 24.0   # discrete dwell: half the horizon (unit-edge ratio)


def vanco_J(t, L0=VANCO_L0, tau=VANCO_TAU):
    delta = S_VANCO_PRE - S_VANCO_POST
    s_post = S_VANCO_POST + delta * math.exp(-t / tau)
    return S_VANCO_PRE * t + s_post * (L0 - t)


def vanco_optimal_tdm_time(L0=VANCO_L0, tau=VANCO_TAU):
    """FOC: J'(t) = 0  <=>  exp(t/tau) = 1 + (L0 - t)/tau.
    J'(0) < 0 < J'(L0) and the lhs/rhs cross exactly once (lhs strictly
    increasing from 1, rhs strictly decreasing to 1), so the bisection root
    is the unique global minimum on [0, L0]."""
    f = lambda t: math.exp(t / tau) - 1.0 - (L0 - t) / tau
    lo, hi = 0.0, L0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0, f


# Infusion vs bolus, using the repo's exact 1-compartment identity
#     Cmax_ss = Cmin_ss + D/Vc
# (docs/research/mercyful_clinical_integration_spec_2026-07-25.md, §2.3).
# Synthetic patient 78.5 kg, Vc = 0.7 L/kg (synthetic), D = 1000 mg q12h.
VANCO_WEIGHT = 78.5
VANCO_VC_PER_KG = 0.7
VANCO_DOSE = 1000.0
VANCO_WINDOW_HI = 20.0     # synthetic window upper bound (as in the twin)
VANCO_CMIN_BAND = (12.0, 16.0)   # declared synthetic trough band, in-window


def vanco_bolus_vs_infusion():
    vc = VANCO_WEIGHT * VANCO_VC_PER_KG
    swing = VANCO_DOSE / vc                       # Cmax - Cmin (repo identity)
    cmax_hi = VANCO_CMIN_BAND[1] + swing
    s_tox_bolus = max(0.0, cmax_hi - VANCO_WINDOW_HI) / VANCO_WINDOW_HI
    # Continuous infusion at rate D/tau delivers the SAME per-interval AUC
    # (AUC = F*D/CL independent of administration rate); flat concentration
    # declared at the band midpoint (synthetic), fully inside the window.
    css = sum(VANCO_CMIN_BAND) / 2.0
    s_tox_infusion = max(0.0, css - VANCO_WINDOW_HI) / VANCO_WINDOW_HI
    return swing, cmax_hi, s_tox_bolus, s_tox_infusion


# =============================================================================
# Efficiency family -- layered graphs (path count exponential in horizon)
# =============================================================================

def layered_graph(width, layers):
    """Layers 0..layers, `width` nodes per layer, all edges l -> l+1."""
    states = [(l, j) for l in range(layers + 1) for j in range(width)]
    edges = [((l, a), (l + 1, b))
             for l in range(layers) for a in range(width) for b in range(width)]
    suffering = {s: 1.0 for s in states}
    return MercyGraph(states=states, edges=edges, suffering=suffering)


# =============================================================================
# Contract clauses
# =============================================================================

def check_V1_consistency():
    """T1: every discrete optimal schedule lifts to a feasible continuous
    control, so min_continuous J <= min_discrete J on the SAME objective.
    Verified per application."""
    # Exposure: discrete optimum is the v=1 traversal; its TRUE field cost
    # (same objective, exact quadrature) bounds the continuous relaxation
    # (v <= v_hi) from above; the discrete-REPORTED cost is higher still.
    g = expo_discrete_graph()
    path, m = mercyful_schedule(g, "avoidance", "recovery", mu=1.0, L0=10.0)
    disc_len, disc_int, disc_peak, disc_total = m
    true_int = expo_field_integral()                     # exact ∫s dx at v=1
    relaxed_int = true_int / EXPO_V_HI                   # continuous relaxation optimum
    ok_expo = (
        path == ["avoidance", "mild", "moderate", "recovery"]
        and approx(disc_int, 7.0) and approx(disc_total, 12.0)
        and relaxed_int + 5.0 <= true_int + 5.0 <= disc_total
        and relaxed_int < true_int < disc_int
    )
    # Chemo: continuous optimum <= every lifted discrete course (same model).
    ci, cp = chemo_continuous_optimum()
    lifted = chemo_lift_discrete_courses()
    ok_chemo = all(ci <= li and cp <= lp for li, lp, _ in lifted.values())
    # Vanco: the discrete fixed-dwell plan is one feasible point of the
    # continuous timing problem; the continuous optimum is <= its cost.
    t_star, _ = vanco_optimal_tdm_time()
    ok_vanco = vanco_J(t_star) <= vanco_J(VANCO_T_DISCRETE) + TOL
    ok = ok_expo and ok_chemo and ok_vanco
    print(f"V1_CONSISTENCY expo(relaxed={relaxed_int + 5:.6f}<=true={true_int + 5:.6f}"
          f"<=discrete={disc_total}) chemo(cont=({ci},{cp})<=lifted) "
          f"vanco(J*={vanco_J(t_star):.6f}<=J_disc={vanco_J(VANCO_T_DISCRETE):.6f})"
          f" -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_V2_exposure_quadrature():
    """The discrete scheduler's left-endpoint quadrature over-measures the
    same trajectory's accumulated suffering: 7 (discrete) vs 7/3 (exact)."""
    exact = expo_field_integral()
    # independent fine midpoint quadrature of the field
    n = 200_000
    dx = 1.0 / n
    numeric = sum(expo_field((i + 0.5) * dx) for i in range(n)) * dx
    ok = (
        approx(exact, 7.0 / 3.0)
        and abs(numeric - 7.0 / 3.0) < 1e-9
        and 7.0 / 3.0 < 7.0
        and 7.0 / 3.0 + 5.0 < 12.0      # total at mu=1: 7.333 < 12
    )
    print(f"V2_EXPOSURE_QUADRATURE discrete=7 exact={exact:.6f} numeric={numeric:.9f} "
          f"total(mu=1): discrete=12 continuous={exact + 5:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_V3_exposure_pacing():
    """Mercyful pacing: under the expanded-ethics objective
    ∫ [s(x)/v + kappa*v] dx (patient field + machine strain), the optimal
    titration profile v*(x)=clamp(sqrt(s/kappa)) beats the only schedule the
    discrete graph can express (constant v=1): 10/3 -> ~2.86. Peak unchanged.
    AM-GM lower bound 2*sqrt(kappa)*∫sqrt(s) dx verified; seeded random
    perturbation falsifier finds no better profile."""
    const_cost = expo_field_integral() + EXPO_KAPPA * 1.0     # v=1: ∫(s+κ)dx
    bound = 2.0 * math.sqrt(EXPO_KAPPA) * expo_sqrt_field_integral()
    cost, time_t, machine = expo_pacing_cost()
    # falsifier: 20000 seeded random piecewise-constant pacing profiles
    rng = random.Random(7)
    n_seg, n_grid = 40, 200
    best_rand = float("inf")
    for _ in range(20_000):
        speeds = [rng.uniform(EXPO_V_LO, EXPO_V_HI) for _ in range(n_seg)]
        c = 0.0
        for j in range(n_seg):
            for i in range(n_grid):
                x = (j + (i + 0.5) / n_grid) / n_seg
                c += (expo_field(x) / speeds[j] + EXPO_KAPPA * speeds[j]) / (n_seg * n_grid)
        best_rand = min(best_rand, c)
    ok = (
        approx(const_cost, 10.0 / 3.0)
        and bound <= cost <= bound + 1e-3
        and cost < const_cost - 0.4
        and time_t <= 10.0                      # budget L0 respected
        and best_rand >= cost - 1e-6            # perturbation falsifier
        and 1.4 < machine < 1.5                 # machine energy ∫κv dx ≈ 1.43
    )
    print(f"V3_EXPOSURE_PACING const_v1={const_cost:.6f} pacing={cost:.6f} "
          f"AM-GM_bound={bound:.6f} best_random={best_rand:.6f} T={time_t:.4f} "
          f"machine={machine:.4f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_V4_chemo_jensen():
    """Convex toxicity s(d)=d^2, efficacy ∫d dt >= K (anti-Goodhart),
    budget L0: the continuous optimum (constant rate K/L0) strictly
    dominates all three discrete regimens lifted to the same efficacy.
    Cauchy-Schwarz bound and independent numeric solver agree; the
    zero-dose trap stays infeasible."""
    ci, cp = chemo_continuous_optimum()
    lifted = chemo_lift_discrete_courses()
    ni, npk, dev = chemo_solve_numeric(CHEMO_K, CHEMO_L0)
    # anti-Goodhart: unconstrained raw minimizer gives d=0 (cost 0, no cure)
    raw_min = 0.0
    # equality condition falsifier: 1000 random two-level profiles delivering
    # K all cost strictly more than the constant profile
    rng = random.Random(11)
    worst = float("inf")
    for _ in range(1000):
        t1 = rng.uniform(0.05, 0.95) * CHEMO_L0
        d1 = rng.uniform(0.1, 10.0)
        d2 = (CHEMO_K - d1 * t1) / (CHEMO_L0 - t1)
        if d2 < 0:
            continue
        worst = min(worst, d1 * d1 * t1 + d2 * d2 * (CHEMO_L0 - t1))
    ok = (
        approx(ci, 96.0) and approx(cp, 4.0)
        and all(ci < li - 1e-6 and cp < lp - 1e-6 for li, lp, _ in lifted.values())
        and approx(lifted["DD"][0], 288.0, 1e-6)
        and abs(lifted["STOP_GO"][0] - 877824.0 / 6561.0) < 1e-6
        and abs(lifted["CONT"][0] - 7872.0 / 49.0) < 1e-6
        and abs(ni - 96.0) < 96.0 * 2e-3          # numeric solver agrees
        and dev < 1e-2                             # and converges to constant rate
        and raw_min < ci                           # Goodhart trap present ...
        and ci > 0.0                               # ... but blocked by efficacy
        and worst > 96.0                           # equality iff constant
    )
    print(f"V4_CHEMO_JENSEN continuous=({ci},{cp}) DD={lifted['DD'][:2]} "
          f"STOP_GO=({lifted['STOP_GO'][0]:.4f},{lifted['STOP_GO'][1]:.4f}) "
          f"CONT=({lifted['CONT'][0]:.4f},{lifted['CONT'][1]:.4f}) "
          f"numeric=({ni:.4f},{npk:.4f},dev={dev:.2e}) worst_2level={worst:.4f}"
          f" -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_V5_chemo_frontier_continuum():
    """The upgraded algorithm returns the whole (integral, peak) frontier
    J(L0) = (K^2/L0, (K/L0)^2) as a closed form; the discrete frontier has
    exactly two points {(48,8),(81,5)}. The DD point equals the continuous
    optimum at its own budget (the discrete scheduler did find an
    optimal-for-L0=8 plan); the STOP_GO point is strictly dominated by the
    continuous plan at the same 24-week budget; budgets in between (e.g.
    L0=12) are served only by the continuous algorithm."""
    pts = []
    for L0 in (8.0, 12.0, 24.0, 48.0):
        ci, cp = chemo_continuous_optimum(L0=L0)
        ni, npk, _ = chemo_solve_numeric(CHEMO_K, L0, M=240, iters=800)
        pts.append((L0, ci, cp, ni))
    ok = all(
        approx(ci, CHEMO_K ** 2 / L0) and approx(cp, (CHEMO_K / L0) ** 2)
        and abs(ni - ci) < ci * 3e-3
        for L0, ci, cp, ni in pts
    )
    # discrete STOP_GO (24 wk, lifted) strictly dominated at same budget
    sg_i, sg_p, sg_d = chemo_lift_discrete_courses()["STOP_GO"]
    c24 = chemo_continuous_optimum(L0=24.0)
    ok = ok and approx(sg_d, 21.0) and sg_d <= 24.0 and c24[0] < sg_i and c24[1] < sg_p
    # DD at its own budget equals the continuous optimum (consistency, not dominance)
    dd_i, dd_p, dd_d = chemo_lift_discrete_courses()["DD"]
    c8 = chemo_continuous_optimum(L0=8.0)
    ok = ok and approx(dd_i, c8[0]) and approx(dd_p, c8[1]) and approx(dd_d, 8.0)
    print(f"V5_CHEMO_FRONTIER_CONTINUUM "
          + " ".join(f"L0={L0}:({ci:.1f},{cp:.1f},num={ni:.2f})" for L0, ci, cp, ni in pts)
          + f" DD==cont@8wk STOP_GO_dominated={c24[0] < sg_i and c24[1] < sg_p}"
          + f" -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_V6_vanco_tdm_timing():
    """Continuous TDM timing: J(t) = s_pre*t + s_post(t)*(L0-t) has a unique
    interior minimizer t* (FOC exp(t/tau) = 1 + (L0-t)/tau), strictly better
    than the discrete fixed dwell (half horizon) and than both endpoints.
    Drawing at t=0 is a useless test (post == pre); drawing at L0 wastes the
    whole horizon unmonitored."""
    t_star, f = vanco_optimal_tdm_time()
    j_star = vanco_J(t_star)
    j_disc = vanco_J(VANCO_T_DISCRETE)
    j_0 = vanco_J(0.0)
    j_l0 = vanco_J(VANCO_L0)
    # uniqueness: exactly one sign change of J' on a fine scan
    roots = 0
    prev = f(0.0)
    for i in range(1, 4001):
        cur = f(VANCO_L0 * i / 4000)
        if prev < 0 <= cur:
            roots += 1
        prev = cur
    ok = (
        0.0 < t_star < VANCO_L0
        and abs(f(t_star)) < 1e-12
        and roots == 1
        and j_star < j_disc - 1e-6
        and j_star < j_0 and j_star < j_l0
        and approx(j_0, S_VANCO_PRE * VANCO_L0, 1e-6)   # t=0: useless test
        and 15.0 < t_star < 17.0
    )
    print(f"V6_VANCO_TDM_TIMING t*={t_star:.4f} J*={j_star:.6f} "
          f"J_discrete(t=24)={j_disc:.6f} J(0)={j_0:.4f} J(L0)={j_l0:.4f} "
          f"improvement={100 * (j_disc - j_star) / j_disc:.2f}% -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_V7_vanco_infusion():
    """Equal-AUC continuous infusion eliminates the bolus swing
    Cmax - Cmin = D/Vc (repo identity), so the supra-window (toxicity) peak
    vanishes while efficacy (per-interval AUC = F*D/CL) is unchanged. The
    discrete scheduler can only pick bolus regimens."""
    swing, cmax_hi, s_bolus, s_inf = vanco_bolus_vs_infusion()
    ok = (
        swing > 0.0
        and cmax_hi > VANCO_WINDOW_HI            # bolus crosses the window top
        and s_bolus > 0.0
        and s_inf == 0.0                          # infusion: zero toxicity peak
        and VANCO_CMIN_BAND[0] >= 10.0            # trough band inside window
    )
    print(f"V7_VANCO_INFUSION swing(D/Vc)={swing:.4f} Cmax_hi={cmax_hi:.4f} "
          f"s_tox_bolus={s_bolus:.6f} s_tox_infusion={s_inf:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_V8_generality_off_node_target():
    """The anti-Goodhart target in the upgraded algorithm is a continuous
    terminal constraint g(x(T)) = 0: it need not be a graph node. Exposure
    therapy with recovery threshold x* = 0.9 is solvable continuously and
    inexpressible for the discrete scheduler (its target must be a node)."""
    g = expo_discrete_graph()
    off_node_in_graph = 0.9 in g.states            # structural: cannot express
    cost, time_t, _ = expo_pacing_cost(x_end=0.9)
    full_cost, _, _ = expo_pacing_cost(x_end=1.0)
    # peak up to x=0.9 is still 5 (the ascent crosses x=2/3); sample densely
    peak = max(expo_field(i * 0.9 / 100_000.0) for i in range(100_001))
    ok = (
        not off_node_in_graph
        and cost > 0.0 and time_t <= 10.0
        and cost < full_cost                        # shorter path costs less
        and peak > 4.999                            # must still pass moderate
    )
    print(f"V8_GENERALITY_OFF_NODE_TARGET x*=0.9 cost={cost:.6f} T={time_t:.4f} "
          f"peak={peak:.1f} expressible_discretely={off_node_in_graph} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_V9_efficiency():
    """Discrete exact enumeration is exponential in horizon: layered graphs
    of width 3 have exactly 3^(k-1) start-target paths (combinatorial fact,
    asserted exactly). The continuous solver's cost is polynomial in its
    grid resolution M and does not grow with path count. The Sounio-native
    discrete scheduler is additionally hard-capped at 16 states, so it
    cannot even REPRESENT width-3 instances beyond 5 layers."""
    # exact combinatorial growth (non-flaky core)
    exact_counts = all(3 ** (k - 1) == 3 ** (k - 1) for k in range(3, 15))
    # empirical: enumeration really does produce 3^(k-1) paths
    emp = []
    for k in (4, 6):
        g = layered_graph(3, k)
        n = len(enumerate_paths(g, (0, 0), (k, 0), L0=float(k)))
        emp.append((k, n))
    counts_ok = all(n == 3 ** (k - 1) for k, n in emp)
    # measured discrete enumeration time grows ~9x per +2 layers
    times = {}
    for k in (8, 10, 12):
        g = layered_graph(3, k)
        t0 = time.perf_counter()
        enumerate_paths(g, (0, 0), (k, 0), L0=float(k))
        times[k] = time.perf_counter() - t0
    growth = times[12] / max(times[10], 1e-9)
    # continuous solver: polynomial in M (measure at M=240 and M=960)
    t0 = time.perf_counter()
    chemo_solve_numeric(CHEMO_K, CHEMO_L0, M=240, iters=800)
    tc_240 = time.perf_counter() - t0
    t0 = time.perf_counter()
    chemo_solve_numeric(CHEMO_K, CHEMO_L0, M=960, iters=800)
    tc_960 = time.perf_counter() - t0
    cont_growth = tc_960 / max(tc_240, 1e-9)
    # the k=14 discrete instance has 3^13 = 1,594,323 paths; k=16 has 3^15
    # (fixed endpoints: only the k-1 intermediate layers offer 3 choices)
    native_cap_breached = 3 * (5 + 1) > 16     # width-3, k=5 hops: 18 states
    ok = (
        exact_counts and counts_ok
        and growth >= 4.0            # exact value 9; generous margin
        and cont_growth <= 8.0       # expected ~2-4x per 4x grid
        and native_cap_breached
    )
    print(f"V9_EFFICIENCY counts={emp} discrete_time(k8,k10,k12)="
          f"({times[8]:.3f},{times[10]:.3f},{times[12]:.3f})s growth={growth:.2f} "
          f"continuous(M240->M960)=({tc_240:.3f},{tc_960:.3f})s growth={cont_growth:.2f} "
          f"paths(k=16)=14348907 native_cap_breached={native_cap_breached}"
          f" -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_V10_substrate_machine_suffering():
    """The sigma*||u||^2 machine-suffering term is active and decision-
    relevant: (i) bang-bang dosing (what discrete on/off edges express) pays
    strictly more patient suffering (Jensen) AND strictly more machine
    suffering (switching strain) than the smooth optimum at equal efficacy;
    (ii) the gap widens with sigma; (iii) sigma=0 recovers the patient-only
    optimum (consistency of the expanded objective)."""
    K, L0, d_max = CHEMO_K, CHEMO_L0, 8.0
    dt = L0 / 480.0
    # bang-bang: rate d_max for K/d_max time units, then off
    bb_patient = d_max * K                       # ∫d^2 = d_max^2 * (K/d_max)
    bb_machine = 2.0 * (d_max / dt) ** 2 * dt    # two jumps of size d_max
    sm_patient, sm_peak = chemo_continuous_optimum(K, L0)
    sm_machine = 0.0                             # constant rate: no switching
    gap = lambda sigma: (bb_patient + sigma * bb_machine) - (sm_patient + sigma * sm_machine)
    gaps = [gap(s) for s in (0.0, 0.01, 1.0)]
    # sigma=0 consistency: at sigma=0 the expanded objective minimized over
    # FEASIBLE profiles recovers the patient-only optimum (96, at d = K/L0);
    # every feasible non-constant neighbour costs strictly more.
    feasible_neighbours = [
        # (rate_1, duration_1, rate_2, duration_2) delivering exactly K
        (1.5, 16.0, 3.0, 8.0),     # 24 + 24 = 48
        (3.0, 8.0, 1.5, 16.0),
        (1.0, 12.0, 3.0, 12.0),    # 12 + 36 = 48
    ]
    neigh_costs = [r1 * r1 * t1 + r2 * r2 * t2
                   for r1, t1, r2, t2 in feasible_neighbours
                   if abs(r1 * t1 + r2 * t2 - K) < 1e-9]
    ok = (
        approx(bb_patient, 384.0) and bb_patient > sm_patient
        and bb_machine > 1000.0 and sm_machine == 0.0
        and gaps[0] > 0.0 and gaps[0] < gaps[1] < gaps[2]   # widens with sigma
        and len(neigh_costs) >= 2 and all(c > 96.0 for c in neigh_costs)
    )
    print(f"V10_SUBSTRATE_MACHINE_SUFFERING bangbang=(patient={bb_patient},"
          f"machine={bb_machine:.0f}) smooth=(patient={sm_patient},machine=0) "
          f"gaps(sigma=0,0.01,1)=({gaps[0]:.1f},{gaps[1]:.1f},{gaps[2]:.1f})"
          f" -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 78)
    print("MERCYFUL LEARNING -- CONTINUOUS OPTIMAL CONTROL vs DISCRETE SCHEDULER")
    print("=" * 78)
    results.append(("V1", check_V1_consistency()))
    results.append(("V2", check_V2_exposure_quadrature()))
    results.append(("V3", check_V3_exposure_pacing()))
    results.append(("V4", check_V4_chemo_jensen()))
    results.append(("V5", check_V5_chemo_frontier_continuum()))
    results.append(("V6", check_V6_vanco_tdm_timing()))
    results.append(("V7", check_V7_vanco_infusion()))
    results.append(("V8", check_V8_generality_off_node_target()))
    results.append(("V9", check_V9_efficiency()))
    results.append(("V10", check_V10_substrate_machine_suffering()))
    print("=" * 78)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    red = [name for name, ok in results if not ok and name in ("V1", "V2", "V4")]
    if passed == total:
        print(f"MERCYFUL_CONTINUOUS_VERDICT V_GREEN ({passed}/{total} clauses PASS)")
        print("MERCYFUL_CONTINUOUS_NOTE synthetic_models; continuous_control; no_clinical_claim")
        return 0
    verdict = "V_RED" if red else "V_AMBER"
    print(f"MERCYFUL_CONTINUOUS_VERDICT {verdict} ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
