#!/usr/bin/env python3
"""
Mercyful Learning x cancer chemotherapy sequencing — contract H1..H8.

Companion to:
  docs/research/mercyful_chemo_sequencing_spec_2026-07-26.md

Reuses the exact scheduler from mercyful_runtime_contract.py (same directory)
so the Python and Sounio implementations can be compared number-for-number
(clause H8 is enforced in scripts/ci/mercyful_chemo_sequencing_gate.sh).

Synthetic patients, synthetic regimens, synthetic suffering values.
Not medical guidance; no clinical claim.
"""

from mercyful_runtime_contract import (
    MercyGraph,
    enumerate_paths,
    mercyful_schedule,
    pareto_frontier,
)

# States
DIAG, REDUCED, DD_A, DD_B, STD_A, STD_B, CFI, RECH, CONT_C, REM = (
    "DIAG", "REDUCED", "DD_A", "DD_B", "STD_A", "STD_B", "CFI", "RECH", "CONT_C", "REM",
)

SUFFERING = {
    DIAG: 0.0,      # untreated at diagnosis -- Goodhart trap A
    REDUCED: 1.5,   # dose-reduced chemo (RDI ~60%) -- Goodhart trap B
    DD_A: 8.0,      # dose-dense block 1 (G-CSF-supported)
    DD_B: 8.0,      # dose-dense block 2
    STD_A: 5.0,     # standard q3w block 1
    STD_B: 5.0,     # standard q3w block 2
    CFI: 1.0,       # chemo-free interval (stop-and-go break)
    RECH: 5.0,      # rechallenge after break
    CONT_C: 8.0,    # continuous block 3 (cumulative neuropathy)
    REM: 0.0,       # target: remission
}

# (u, v, weeks)
BASE_EDGES = [
    (DIAG, DIAG, 1.0),     # watch-and-wait trap
    (DIAG, REDUCED, 2.0),  # start dose-reduced
    (REDUCED, REDUCED, 2.0),  # continue reduced: dead end, no edge to REM
    (DIAG, STD_A, 3.0),
    (STD_A, STD_B, 6.0),
    (STD_B, CFI, 6.0),
    (CFI, RECH, 6.0),
    (RECH, REM, 3.0),      # stop-and-go completes
    (STD_B, CONT_C, 6.0),
    (CONT_C, REM, 3.0),    # continuous course completes
]

# Dose-dense edges, gated by G_GCSF (high-FN-risk schedule requires support).
DD_EDGES = [
    (DIAG, DD_A, 2.0),
    (DD_A, DD_B, 4.0),
    (DD_B, REM, 2.0),
]


def build_chemo_graph(gcsf: bool) -> MercyGraph:
    edges = list(BASE_EDGES) + (list(DD_EDGES) if gcsf else [])
    return MercyGraph(
        states=list(SUFFERING.keys()),
        edges=[(u, v) for u, v, _ in edges],
        suffering=SUFFERING,
        lengths={(u, v): w for u, v, w in edges},
    )


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def check_H1_baseline():
    g = build_chemo_graph(gcsf=True)
    path, metrics = mercyful_schedule(g, DIAG, REM, mu=1.0, L0=30.0)
    length, integral, peak, total = metrics
    ok = (
        path == [DIAG, DD_A, DD_B, REM]
        and approx(length, 8.0)
        and approx(integral, 48.0)
        and approx(peak, 8.0)
        and approx(total, 56.0)
    )
    print(f"H1_BASELINE path={path} length={length} integral={integral} peak={peak} total={total} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_H2_anti_goodhart():
    g = build_chemo_graph(gcsf=True)
    # Raw unconstrained minimizer: DIAG self-loop, cost 0, never reaches REM.
    unconstrained_best = 0.0
    # Every feasible (target-reaching) path has integral >= 48.
    paths = enumerate_paths(g, DIAG, REM, L0=1e9)
    min_integral = min(g.path_cost(p, 0.0)[1] for p in paths)
    # The reduced route (Goodhart trap B) is present in the graph but no
    # feasible path uses it: under-dosing structurally cannot reach remission.
    reduced_reaches = any(REDUCED in p for p in paths)
    ok = (unconstrained_best < min_integral) and approx(min_integral, 48.0) and not reduced_reaches
    print(f"H2_ANTI_GOODHART unconstrained={unconstrained_best} min_feasible_integral={min_integral} "
          f"reduced_reaches_rem={reduced_reaches} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_H3_mu_crossover():
    g = build_chemo_graph(gcsf=True)
    path_low, _ = mercyful_schedule(g, DIAG, REM, mu=0.0, L0=30.0)
    path_high, _ = mercyful_schedule(g, DIAG, REM, mu=20.0, L0=30.0)
    # Exact crossover: (81 - 48) / (8 - 5) = 11
    mu_star = (81.0 - 48.0) / (8.0 - 5.0)
    ok = (
        path_low == [DIAG, DD_A, DD_B, REM]
        and path_high == [DIAG, STD_A, STD_B, CFI, RECH, REM]
        and approx(mu_star, 11.0)
    )
    print(f"H3_MU_CROSSOVER mu_star={mu_star} mu0={path_low} mu20={path_high} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_H4_frontier():
    g = build_chemo_graph(gcsf=True)
    pf = pareto_frontier(g, DIAG, REM, L0=30.0)
    points = {(round(i, 6), round(p, 6)) for i, p, _, _ in pf}
    expected = {(48.0, 8.0), (81.0, 5.0)}
    # CONT (84, 8) must be present among candidates but dominated off the frontier.
    all_points = {(round(g.path_cost(p, 0.0)[1], 6), round(g.path_cost(p, 0.0)[2], 6))
                  for p in enumerate_paths(g, DIAG, REM, L0=30.0)}
    ok = points == expected and (84.0, 8.0) in all_points
    print(f"H4_FRONTIER frontier={sorted(points)} all={sorted(all_points)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_H5_gcsf_gate_causal():
    g_on = build_chemo_graph(gcsf=True)
    g_off = build_chemo_graph(gcsf=False)
    path_on, _ = mercyful_schedule(g_on, DIAG, REM, mu=1.0, L0=30.0)
    path_off, metrics_off = mercyful_schedule(g_off, DIAG, REM, mu=1.0, L0=30.0)
    ok = (
        path_on == [DIAG, DD_A, DD_B, REM]
        and path_off == [DIAG, STD_A, STD_B, CFI, RECH, REM]
        and approx(metrics_off[2], 5.0)  # peak 5: gate blocks DD only, not everything
    )
    print(f"H5_GCSF_GATE on={path_on} off={path_off} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_H6_budget_hardness():
    g_on = build_chemo_graph(gcsf=True)
    g_off = build_chemo_graph(gcsf=False)
    p1, _ = mercyful_schedule(g_on, DIAG, REM, mu=1.0, L0=7.0)    # DD needs 8 weeks
    p2, _ = mercyful_schedule(g_off, DIAG, REM, mu=1.0, L0=12.0)  # only STOP_GO (24) left
    ok = (p1 == 'INFEASIBLE') and (p2 == 'INFEASIBLE')
    print(f"H6_BUDGET_HARDNESS L0=7/gcsf_on={p1} L0=12/gcsf_off={p2} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_H7_budgetary_necessity():
    g = build_chemo_graph(gcsf=True)
    _, m_tight = mercyful_schedule(g, DIAG, REM, mu=100.0, L0=10.0)
    _, m_wide = mercyful_schedule(g, DIAG, REM, mu=100.0, L0=30.0)
    ok = approx(m_tight[2], 8.0) and approx(m_wide[2], 5.0)
    print(f"H7_BUDGETARY_NECESSITY peak(L0=10)={m_tight[2]} peak(L0=30)={m_wide[2]} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_H8_canonical_numbers():
    # Canonical numbers the CI gate compares against the Sounio-native run.
    g = build_chemo_graph(gcsf=True)
    _, m1 = mercyful_schedule(g, DIAG, REM, mu=1.0, L0=30.0)
    _, m2 = mercyful_schedule(build_chemo_graph(gcsf=False), DIAG, REM, mu=1.0, L0=30.0)
    ok = (
        approx(m1[0], 8.0) and approx(m1[1], 48.0) and approx(m1[2], 8.0) and approx(m1[3], 56.0)
        and approx(m2[0], 24.0) and approx(m2[1], 81.0) and approx(m2[2], 5.0) and approx(m2[3], 86.0)
    )
    print(f"H8_CANONICAL gcsf_on={m1} gcsf_off={m2} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("MERCYFUL LEARNING x CHEMO SEQUENCING — contract")
    print("=" * 70)
    results.append(("H1", check_H1_baseline()))
    results.append(("H2", check_H2_anti_goodhart()))
    results.append(("H3", check_H3_mu_crossover()))
    results.append(("H4", check_H4_frontier()))
    results.append(("H5", check_H5_gcsf_gate_causal()))
    results.append(("H6", check_H6_budget_hardness()))
    results.append(("H7", check_H7_budgetary_necessity()))
    results.append(("H8", check_H8_canonical_numbers()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    red = [name for name, ok in results if not ok and name in ("H1", "H2", "H6")]
    if passed == total:
        print(f"MERCYFUL_CHEMO_VERDICT H_GREEN ({passed}/{total} clauses PASS)")
        print("MERCYFUL_CHEMO_NOTE synthetic_graph; chemo_sequencing_toy; no_clinical_claim")
        return 0
    verdict = "H_RED" if red else "H_AMBER"
    print(f"MERCYFUL_CHEMO_VERDICT {verdict} ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
