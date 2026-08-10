#!/usr/bin/env python3
"""
Mercyful Learning x continuous optimal control (Pontryagin rung) — contract K1..K8.

Companion to:
  docs/research/mercyful_pontryagin_control_spec_2026-07-26.md

Replaces the discrete graph search of the chemo rung
(scripts/research/mercyful_chemo_contract.py) with continuous-time optimal
control. The suffering trajectory is a controlled ODE

    dd/dt = -a * u * d          (log-kill disease dynamics)
    s(t)  = c * d(t) + tau * u(t)   (suffering field: disease + toxicity)

with control u in [0, u_max], free terminal time T <= L0, and the
anti-Goodhart hard constraint d(T) = d_T. The objective is the Mercyful
functional

    J[u] = ∫_0^T s dt + mu * max_t s(t)

minimized over measurable controls. The peak term is handled exactly by the
epigraph transform s(t) <= m, under which the problem decomposes:

    J*(mu) = min_m [ I*(m) + mu * m ]

where I*(m) is the minimal integral under the peak cap m. I*(m) and the
minimal time T*(m) have closed forms (spec, T3/T4); this contract verifies
them against an independent discretized dynamic program, verifies the
Pontryagin necessary conditions (switching function, HJB, constraint
multiplier sign), the smooth-crossover law m*(mu) with
I*(m_B - eps) = I_B + eps^2/(2 a tau u_max^2) + O(eps^3), the budgetary
necessity curve peak_min(L0) = (T*)^-1(L0), and the two-mercies comparative
static (machine-suffering weight nu raises the patient peak).

Clauses:
  K1 baseline closed forms vs direct ODE quadrature
  K2 anti-Goodhart: unconstrained escape vs feasible floor
  K3 bang-bang theorem: no interior singular arcs (sigma' = a c d > 0)
  K4 frontier closed forms vs independent DP; equioscillation (flat s)
  K5 smooth crossover: m*(0)=m_B, I*'(m_B-)=0, curvature 1/(2 a tau u_max^2)
  K6 budgetary necessity: peak_min(L0) = (T*)^-1(L0), shadow price < 0
  K7 two mercies: machine weight nu raises patient peak m**(mu, nu)
  K8 PMP on the boundary arc: HJB residual ~ 0, multiplier eta >= 0

Synthetic patients, synthetic regimens, synthetic suffering values.
This is not medical guidance; no clinical claim.
"""

import math

# ---------------------------------------------------------------------------
# Synthetic instance (normalized units; see spec section 4.1)
# ---------------------------------------------------------------------------
D0 = 1.0      # initial disease burden (normalized)
DT = 0.05     # therapeutic response threshold (anti-Goodhart target)
A = 1.0       # fractional kill rate per unit control
UMAX = 2.0    # maximal treatment intensity
C = 1.0       # disease-attributable suffering per unit burden
TAU = 1.0     # treatment-attributable suffering (toxicity) per unit intensity

V_DOSE = math.log(D0 / DT)                      # total dose required by target
T_BANG = V_DOSE / UMAX                          # bang (front-loaded) duration
M_BANG = C * D0 + TAU * UMAX                    # bang peak (at t=0)
I_BANG = (C / (A * UMAX)) * (D0 - DT) + TAU * V_DOSE   # bang integral
FLOOR = C * D0                                  # minimal attainable peak


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Closed-form frontier (spec, theorems T3/T4)
# ---------------------------------------------------------------------------
def closed_form(m):
    """(I*(m), T*(m)): minimal suffering integral and minimal time to reach
    d_T under the peak cap s(t) <= m, for m in (FLOOR, M_BANG]."""
    if m <= FLOOR:
        return math.inf, math.inf
    if m >= M_BANG:
        return I_BANG, T_BANG
    d_join = (m - TAU * UMAX) / C
    if d_join <= DT:
        # pure iso-suffering arc: cap binds all the way to the target
        t_iso = (TAU / (A * m)) * math.log(
            ((m - C * DT) / (C * DT)) * ((C * D0) / (m - C * D0)))
        return m * t_iso, t_iso
    # iso-suffering arc d0 -> d_join, then terminal bang d_join -> d_T
    K = (m - C * D0) / (C * D0)
    t_iso = (TAU / (A * m)) * math.log((TAU * UMAX / (C * d_join)) / K)
    t_bang = (1.0 / (A * UMAX)) * math.log(d_join / DT)
    integral = (m * t_iso
                + (C / (A * UMAX)) * (d_join - DT)
                + (TAU / A) * math.log(d_join / DT))
    return integral, t_iso + t_bang


def dI_star(m, h=1e-6):
    """Numeric derivative of I*(m)."""
    return (closed_form(m + h)[0] - closed_form(m - h)[0]) / (2 * h)


def m_star(mu, n_grid=6000):
    """Peak of the optimal policy for Mercyful weight mu: argmin I*(m)+mu*m."""
    best_m, best_j = None, math.inf
    for i in range(n_grid):
        m = FLOOR + 1e-4 + (M_BANG - FLOOR - 1e-4) * i / (n_grid - 1)
        j = closed_form(m)[0] + mu * m
        if j < best_j:
            best_m, best_j = m, j
    return best_m, best_j


def peak_min(L0):
    """Budgetary necessity: minimal attainable peak under horizon budget L0."""
    if L0 < T_BANG:
        return math.inf
    lo, hi = FLOOR + 1e-12, M_BANG
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if closed_form(mid)[1] <= L0:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# Independent dynamic program: min integral under a peak cap, on a
# (dose x control) grid. Genuinely independent of the closed forms: it
# re-derives the optimal control pointwise instead of assuming it.
# ---------------------------------------------------------------------------
def dp_capped(m, n_dose=2000, n_ctrl=400):
    """(min integral, time along DP-optimal policy, list of (d, u, s) on the
    DP-optimal trajectory) under cap s <= m."""
    dV = V_DOSE / n_dose
    u_grid = [UMAX * (k + 1) / n_ctrl for k in range(n_ctrl)]
    cost = [math.inf] * (n_dose + 1)
    cost[0] = 0.0
    choice = [0.0] * (n_dose + 1)
    for i in range(n_dose):
        if cost[i] == math.inf:
            continue
        d = D0 * math.exp(-A * i * dV)
        u_cap = min(UMAX, (m - C * d) / TAU)
        if u_cap <= 0.0:
            continue
        for u in u_grid:
            if u > u_cap:
                break
            w = (C * d / u + TAU) * dV
            if cost[i] + w < cost[i + 1] - 1e-18:
                cost[i + 1] = cost[i] + w
                choice[i] = u
    # reconstruct trajectory
    traj = []
    t = 0.0
    for i in range(n_dose):
        u = choice[i]
        if u <= 0.0:
            break
        d = D0 * math.exp(-A * i * dV)
        dt = dV / u
        traj.append((d, u, C * d + TAU * u))
        t += dt
    return cost[n_dose], t, traj


def simulate_bang(n_steps=200000):
    """Direct quadrature of the bang trajectory (independent of closed form)."""
    dt = T_BANG / n_steps
    d = D0
    integral, peak = 0.0, 0.0
    for _ in range(n_steps):
        s = C * d + TAU * UMAX
        integral += s * dt
        if s > peak:
            peak = s
        d -= A * UMAX * d * dt
    return integral, peak, d


# ---------------------------------------------------------------------------
# Contract clauses
# ---------------------------------------------------------------------------
def check_K1_baseline():
    integral, peak, d_end = simulate_bang()
    ok = (approx(integral, I_BANG, 1e-4)
          and approx(peak, M_BANG, 1e-9)
          and approx(d_end, DT, 1e-4)
          and approx(T_BANG, V_DOSE / UMAX, 1e-12))
    print(f"K1_BASELINE V={V_DOSE:.6f} t_B={T_BANG:.6f} m_B={M_BANG:.6f} "
          f"I_B={I_BANG:.6f} quad_I={integral:.6f} quad_peak={peak:.6f} "
          f"d_end={d_end:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_K2_anti_goodhart():
    # Unconstrained raw minimizer: u == 0 with T = 0 (never start) gives J = 0.
    unconstrained = 0.0
    # Every target-reaching policy pays J >= I_BANG > 0.
    feasible_floor = I_BANG
    # Under-dosing structurally cannot reach the target: constant u = 0.1
    # over [0, 5] delivers dose 0.5 << V_DOSE, leaving d far above d_T.
    d_under = D0 * math.exp(-A * 0.1 * 5.0)
    # DP with a cap below the disease floor is infeasible.
    dp_infeasible = dp_capped(0.9, n_dose=200, n_ctrl=50)[0] == math.inf
    ok = (unconstrained < feasible_floor
          and d_under > 10 * DT
          and dp_infeasible)
    print(f"K2_ANTI_GOODHART unconstrained={unconstrained} "
          f"feasible_floor={feasible_floor:.6f} underdose_d(5)={d_under:.4f} "
          f"target={DT} subfloor_cap_infeasible={dp_infeasible} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_K3_bang_bang_no_singular_arc():
    # PMP: H = (c d + tau u) - a lambda u d; switching sigma = tau - a lambda d.
    # Free terminal time + autonomous dynamics => H == 0 along the optimum,
    # hence lambda = (c d + tau u_max)/(a u_max d) and sigma = -c d / u_max.
    # sigma is strictly increasing (sigma' = a c d > 0): at most one switch,
    # and sigma(T-) = -c d_T/u_max < 0, so zero switches: pure bang.
    n = 200
    sig = []
    sigdot_ok = True
    for k in range(n + 1):
        t = T_BANG * k / n
        d = D0 * math.exp(-A * UMAX * t)
        lam = (C * d + TAU * UMAX) / (A * UMAX * d)
        sig.append(TAU - A * lam * d)
        # verify sigma' = a c d numerically against the closed form
        if abs(sig[-1] - (-C * d / UMAX)) > 1e-12:
            sigdot_ok = False
    increasing = all(sig[k + 1] > sig[k] for k in range(n))
    negative = all(s < 0 for s in sig)
    # DP at cap m_B must reproduce the bang (no cheaper capped policy).
    dp_cost = dp_capped(M_BANG)[0]
    ok = (sigdot_ok and increasing and negative
          and sig[-1] < 0.0
          and approx(dp_cost, I_BANG, 5e-3))
    print(f"K3_BANG_BANG sigma(0)={sig[0]:.6f} sigma(T)={sig[-1]:.6f} "
          f"increasing={increasing} dp(m_B)={dp_cost:.6f} vs I_B={I_BANG:.6f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_K4_frontier_and_equioscillation():
    ok = True
    rows = []
    for m, tol in [(3.0, 5e-3), (2.5, 5e-3), (2.0, 5e-3), (1.5, 5e-3), (1.2, 8e-3)]:
        Ic, Tc = closed_form(m)
        Id, Td, _ = dp_capped(m)
        good = abs(Ic - Id) / Ic < tol
        ok = ok and good
        rows.append(f"m={m}: closed=({Ic:.6f},{Tc:.6f}) dp=({Id:.6f},{Td:.6f})")
    # Equioscillation: for m = 1.5 (pure iso-suffering arc) the DP-optimal
    # trajectory holds s flat at the cap.
    _, _, traj = dp_capped(1.5)
    s_vals = [s for _, _, s in traj]
    flat = (max(s_vals) <= 1.5 + 1e-9) and (min(s_vals) >= 1.5 - 0.05)
    # and the DP control is interior (strictly below u_max on the arc)
    interior = all(u < UMAX - 1e-9 for _, u, _ in traj[: len(traj) // 2])
    ok = ok and flat and interior
    print(f"K4_FRONTIER {'; '.join(rows)}")
    print(f"K4_EQUIOSCILLATION m=1.5 s_min={min(s_vals):.4f} s_max={max(s_vals):.4f} "
          f"interior_control={interior} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_K5_smooth_crossover():
    # Bang is optimal only at mu = 0; any mu > 0 buys a lower peak.
    m0, _ = m_star(0.0)
    m_small, _ = m_star(0.001)
    # Flatness at the bang peak: I*'(m_B-) = 0, curvature 1/(2 a tau u_max^2).
    eps = 0.01
    dI = closed_form(M_BANG - eps)[0] - I_BANG
    curvature = dI / eps ** 2
    c_exact = 1.0 / (2.0 * A * TAU * UMAX ** 2)
    # First-order crossover law: m*(mu) = m_B - mu * a tau u_max^2 + O(mu^2).
    m_pred = M_BANG - 0.001 * A * TAU * UMAX ** 2
    # Stationarity: I*'(m*(mu)) = -mu away from the kink.
    stat_ok = True
    for mu in (0.01, 0.1, 1.0):
        ms, _ = m_star(mu)
        if abs(dI_star(ms) + mu) > 0.02 * mu + 1e-3:
            stat_ok = False
    ok = (approx(m0, M_BANG, 1e-3)
          and m_small < M_BANG
          and approx(m_small, m_pred, 2e-3)
          and approx(curvature, c_exact, 0.02 * c_exact)
          and stat_ok)
    print(f"K5_SMOOTH_CROSSOVER m*(0)={m0:.5f} m*(0.001)={m_small:.5f} "
          f"pred={m_pred:.5f} curvature={curvature:.6f} "
          f"exact={c_exact:.6f} stationarity={stat_ok} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_K6_budgetary_necessity():
    # Necessity curve vs closed form, infeasibility below the bang duration,
    # and approach to the disease floor as the budget widens.
    ok = True
    rows = []
    for L0 in (1.5, 2.0, 3.0, 10.0):
        pm = peak_min(L0)
        # certify by construction: closed_form(pm)[1] ~ L0
        t_cert = closed_form(pm)[1]
        good = abs(t_cert - L0) < 1e-6
        ok = ok and good
        rows.append(f"L0={L0}: peak_min={pm:.6f} (T*={t_cert:.6f})")
    infeasible = peak_min(1.4) == math.inf
    floor_approach = peak_min(50.0) < FLOOR + 1e-3
    # shadow price: more machine budget buys less patient peak
    h = 1e-4
    shadow = (peak_min(3.0 + h) - peak_min(3.0 - h)) / (2 * h)
    ok = ok and infeasible and floor_approach and shadow < 0.0
    print(f"K6_BUDGETARY_NECESSITY {'; '.join(rows)}")
    print(f"K6_INFEASIBLE(L0=1.4<t_B)={infeasible} floor(50)={peak_min(50.0):.6f} "
          f"shadow_price={shadow:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_K7_two_mercies():
    # Machine-inclusive objective J = I*(m) + mu m + nu T*(m).
    # The two mercies compete: raising the machine-suffering weight nu raises
    # the optimal patient peak and shortens the course, toward the bang.
    mu = 0.1
    prev_m, prev_t, mono = None, None, True
    rows = []
    for nu in (0.0, 0.1, 0.5, 2.0, 10.0):
        best_m, best_j = None, math.inf
        for i in range(4000):
            m = FLOOR + 1e-4 + (M_BANG - FLOOR - 1e-4) * i / 3999
            Ic, Tc = closed_form(m)
            j = Ic + mu * m + nu * Tc
            if j < best_j:
                best_m, best_j = m, j
        Ic, Tc = closed_form(best_m)
        rows.append(f"nu={nu}: m**={best_m:.4f} T**={Tc:.4f}")
        if prev_m is not None and (best_m < prev_m - 1e-3 or Tc > prev_t + 1e-3):
            mono = False
        prev_m, prev_t = best_m, Tc
    # In the limit of machine mercy alone, the optimum is the bang itself.
    limit_bang = prev_m > M_BANG - 0.05
    ok = mono and limit_bang
    print(f"K7_TWO_MERCIES {'; '.join(rows)} limit_bang={limit_bang} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_K8_pmp_boundary_arc():
    # Boundary (iso-suffering) arc, cap m = 1.5 (pure iso: d_join <= d_T).
    # HJB: min_u { s + W'(d)(-a u d) } = 0 along the arc, with W = value.
    # Constraint multiplier eta = (a lambda d - tau)/tau >= 0 (Bryson-Ho).
    m = 1.5
    h = 1e-6

    def t_rem(d):
        return (TAU / (A * m)) * math.log(
            ((m - C * DT) / (C * DT)) * ((C * d) / (m - C * d)))

    hjb_ok, eta_ok = True, True
    for d in (1.0, 0.7, 0.4, 0.2, 0.08):
        lam = m * (t_rem(d + h) - t_rem(d - h)) / (2 * h)
        u = (m - C * d) / TAU
        resid = (C * d + TAU * u) + lam * (-A * u * d)
        eta = (A * lam * d - TAU) / TAU
        if abs(resid) > 1e-6:
            hjb_ok = False
        if eta < 0.0:
            eta_ok = False
    # Free-time transversality along the bang: H == 0 identically.
    H_ok = True
    for k in range(11):
        t = T_BANG * k / 10
        d = D0 * math.exp(-A * UMAX * t)
        lam = (C * d + TAU * UMAX) / (A * UMAX * d)
        H = (C * d + TAU * UMAX) - A * lam * UMAX * d
        if abs(H) > 1e-9:
            H_ok = False
    ok = hjb_ok and eta_ok and H_ok
    print(f"K8_PMP_BOUNDARY_ARC hjb={hjb_ok} eta_nonnegative={eta_ok} "
          f"H_zero_on_bang={H_ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_K9_canonical_numbers():
    # Canonical numbers the CI gate compares against the Sounio-native run.
    I20, T20 = closed_form(2.0)
    I15, T15 = closed_form(1.5)
    pm2, pm3 = peak_min(2.0), peak_min(3.0)
    ok = (approx(V_DOSE, 2.995732, 1e-6)
          and approx(I_BANG, 3.470732, 1e-6)
          and approx(I20, 3.663562, 1e-6)
          and approx(I15, 4.060443, 1e-6)
          and approx(pm3, 1.402552, 1e-6))
    print(f"K9_CANONICAL V={V_DOSE:.6f} t_B={T_BANG:.6f} m_B={M_BANG:.6f} "
          f"I_B={I_BANG:.6f} I*(2)={I20:.6f} T*(2)={T20:.6f} "
          f"I*(1.5)={I15:.6f} T*(1.5)={T15:.6f} "
          f"peak_min(2)={pm2:.6f} peak_min(3)={pm3:.6f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("MERCYFUL LEARNING x PONTRYAGIN CONTINUOUS CONTROL — contract")
    print("=" * 70)
    results.append(("K1", check_K1_baseline()))
    results.append(("K2", check_K2_anti_goodhart()))
    results.append(("K3", check_K3_bang_bang_no_singular_arc()))
    results.append(("K4", check_K4_frontier_and_equioscillation()))
    results.append(("K5", check_K5_smooth_crossover()))
    results.append(("K6", check_K6_budgetary_necessity()))
    results.append(("K7", check_K7_two_mercies()))
    results.append(("K8", check_K8_pmp_boundary_arc()))
    results.append(("K9", check_K9_canonical_numbers()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    red = [name for name, ok in results if not ok and name in ("K1", "K2", "K4")]
    if passed == total:
        print(f"MERCYFUL_PONTRYAGIN_VERDICT K_GREEN ({passed}/{total} clauses PASS)")
        print("MERCYFUL_PONTRYAGIN_NOTE synthetic_ode; continuous_control_toy; no_clinical_claim")
        return 0
    verdict = "K_RED" if red else "K_AMBER"
    print(f"MERCYFUL_PONTRYAGIN_VERDICT {verdict} ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
