#!/usr/bin/env python3
"""
Mercyful Learning x MIMIC-IV vancomycin TDM — sensitivity analysis contract.

Companion to:
  docs/research/mimic_iv_sensitivity_analysis_2026-07-26.md (spec + landscape)
  docs/research/mimic_iv_mercyful_validation_2026-07-26.md (parent validation)
  scripts/research/mercyful_mimic_iv_vancomycin_contract.py (parent contract V1..V7)

Question: does the POSITIVE structural-correspondence verdict survive variation
of the modeler's declared knobs? Axes varied:

  therapeutic window  [a,b] in {[10,15], [15,20], [10,20], [15,25]} mg/L
  toxicity level      tau in {0.1, 0.2, 0.3, 0.6}  (FIXED_STD supra-window
                      violation; tau=0.3 reproduces the parent's (6,26) band
                      on the reference window [10,20])
  TDM residual        rho in {0.0, 0.03, 0.059420, 0.1}  (residual suffering
                      of the TDM-narrowed band; 0.059420 is the measured
                      twin value, clause C3)
  peak weight         mu in {0, 0.5, 1, 2, 5, 10, 20}
  field shape         f in {linear, quadratic, softplus, sqrt-hinge} applied
                      to the window-violation components

The graph below is SYNTHETIC, as in the parent contract. No patient data were
used; this is not medical guidance and carries no clinical claim. The only
measured (non-declared) numbers are the twin's clause-C3 suffering values
0.675679 / 0.059420 (1000 mg q12h, window [10,20] mg/L), used in the
twin-anchored reference cell (S3), and the real MIMIC-IV cohort statistics
re-asserted in S6 (Wang J et al., Sci Rep 2026, doi:10.1038/s41598-026-42395-1).

Headline findings (frozen as regression expectations, see spec section 4):
  * GATED (the model's actual semantics: only window-verified courses reach
    TARGET): the TDM-guided route is the unique feasible path and is selected
    in 1792/1792 varied cells. The POSITIVE verdict is invariant to every
    axis varied here.
  * OPEN (verification-blind counterfactual, the parent V4 probe, swept):
    the conclusion DOES change. Closed-form classification (route costs are
    linear in mu): FIXED_STD wins strictly at every tested mu in 110/256
    cells (concentrated at low toxicity tau in {0.1, 0.2} and wide windows),
    TDM wins strictly everywhere in 110/256, 20/256 cells are exact ties at
    every mu, and 16/256 show a single STD->TDM crossover. The gate, not the
    field shape, is what makes TDM the optimum; at the twin-anchored
    reference point the crossover is mu* = 1.443156.

Pure Python; no dependencies beyond the standard library. The scheduler is
imported unchanged from mercyful_runtime_contract.py (M1..M6, M_GREEN).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mercyful_runtime_contract import MercyGraph, mercyful_schedule, enumerate_paths

# -----------------------------------------------------------------------------
# Variation axes
# -----------------------------------------------------------------------------

WINDOWS = [(10.0, 15.0), (15.0, 20.0), (10.0, 20.0), (15.0, 25.0)]
TAUS = [0.1, 0.2, 0.3, 0.6]
RHOS = [0.0, 0.03, 0.059420, 0.1]
MUS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SHAPES = ["linear", "quadratic", "softplus", "sqrt"]

# Declared synthetic Cmin p-boxes (mg/L), absolute, inherited from the parent
# contract. FIXED_LOW is fully below every window (hi=9 < 10 <= a).
FIXED_LOW_BAND = (4.0, 9.0)
VANCO_PRE_BAND = (6.0, 24.0)   # declared pre-TDM band (functional cells)

# Twin-anchored reference cell (window [10,20], linear, tau=0.3,
# rho=0.059420): exact clause-C3 measured values, matching the parent V5.
S_VANCO_PRE_MEAS = 0.675679
S_VANCO_POST_MEAS = 0.059420
S_FIXED_LOW_REF = 0.6
S_FIXED_STD_REF = 0.7

START, FIXED_LOW, FIXED_STD = 'START', 'FIXED_LOW', 'FIXED_STD'
VANCO_PRE, TDM_GUIDED, TARGET = 'VANCO_PRE', 'TDM_GUIDED', 'TARGET'

TDM_PATH = [START, VANCO_PRE, TDM_GUIDED, TARGET]
STD_PATH = [START, FIXED_STD, TARGET]

# Real MIMIC-IV statistics (Wang et al. 2026, doi:10.1038/s41598-026-42395-1);
# identical block to the parent contract, re-asserted in S6.
MIMIC = {
    "n_total": 28451,
    "n_tdm": 10758,
    "n_non_tdm": 17693,
    "psm_pairs": 9785,
    "icu_mortality":      (0.691, 0.580, 0.820),
    "hospital_mortality": (0.672, 0.570, 0.790),
    "aki":                (0.580, 0.540, 0.610),
}


def shape_fn(name, v):
    """Suffering-field shape applied to a violation fraction v >= 0."""
    if name == "linear":
        return v
    if name == "quadratic":
        return v * v
    if name == "softplus":
        return math.log1p(v)
    if name == "sqrt":
        return math.sqrt(v)
    raise ValueError(name)


def s_window(band, a, b, shape):
    """Window-violation suffering of a Cmin p-box (paper section 7.2), with
    the shape functional applied componentwise."""
    lo, hi = band
    v_lo = max(0.0, a - lo) / a
    v_hi = max(0.0, hi - b) / b
    return shape_fn(shape, v_lo) + shape_fn(shape, v_hi)


def tdm_band(a, b):
    """TDM-narrowed band: middle 40% of the window (dose adjusted to the
    measured level; synthetic declaration)."""
    w = b - a
    return (a + 0.3 * w, b - 0.3 * w)


def fixed_std_band(tau):
    """Unmonitored fixed dose: low edge 6.0 mg/L (below every window), high
    edge tau above 20 mg/L. tau=0.3 reproduces the parent's (6, 26)."""
    return (6.0, 20.0 * (1.0 + tau))


def build_graph(a, b, tau, rho, shape, gated, twin_anchored=False):
    """Sensitivity-cell dosing graph. Same structure as the parent contract;
    G_VERIFY admits an edge to TARGET only for window-contained bands."""
    tb = tdm_band(a, b)
    assert tb[0] >= a and tb[1] <= b, "TDM band must be window-contained"
    assert FIXED_LOW_BAND[1] < a, "FIXED_LOW must be sub-window"
    fsb = fixed_std_band(tau)
    assert fsb[0] < a, "FIXED_STD must straddle on the low side"
    if twin_anchored:
        s_fl, s_fs = S_FIXED_LOW_REF, S_FIXED_STD_REF
        s_vp, s_tg = S_VANCO_PRE_MEAS, S_VANCO_POST_MEAS
    else:
        s_fl = s_window(FIXED_LOW_BAND, a, b, shape)
        s_fs = s_window(fsb, a, b, shape)
        s_vp = s_window(VANCO_PRE_BAND, a, b, shape)
        s_tg = s_window(tb, a, b, shape) + rho
    edges = [
        (START, START),
        (START, FIXED_LOW),
        (START, FIXED_STD),
        (START, VANCO_PRE),
        (VANCO_PRE, TDM_GUIDED),
    ]
    if not gated:
        edges.append((FIXED_STD, TARGET))   # counterfactual: unverified admitted
    edges.append((TDM_GUIDED, TARGET))      # admitted: band inside window
    return MercyGraph(
        states=[START, FIXED_LOW, FIXED_STD, VANCO_PRE, TDM_GUIDED, TARGET],
        edges=edges,
        suffering={
            START: 0.0,
            FIXED_LOW: s_fl,
            FIXED_STD: s_fs,
            VANCO_PRE: s_vp,
            TDM_GUIDED: s_tg,
            TARGET: 0.0,
        },
    )


def winner(g, mu):
    path, _ = mercyful_schedule(g, START, TARGET, mu=mu, L0=10.0)
    if path == TDM_PATH:
        return 'TDM'
    if path == STD_PATH:
        return 'FIXED_STD'
    raise AssertionError(f"unexpected path {path}")


# Frozen landscape expectations (computed 2026-07-26, this exact grid; see
# spec section 4). Winners are classified by CLOSED-FORM route costs with an
# explicit TIE class (both route costs are linear in mu, so equality is
# detectable exactly); the scheduler is then cross-checked against the strict
# classification on every non-tie point. 20 cells are exact ties at every mu
# (FIXED_STD(0.2) == VANCO_PRE band, and [15,25] with tau=0.1 where both
# bands violate only the shared low edge); there the scheduler's pick is an
# enumeration-order artifact and is not counted as a win for either route.
FROZEN_TDM_WINS_OPEN = {0.0: 110, 0.5: 120, 1.0: 121, 2.0: 125,
                        5.0: 125, 10.0: 125, 20.0: 126}
FROZEN_STD_WINS_OPEN = {0.0: 124, 0.5: 116, 1.0: 113, 2.0: 111,
                        5.0: 111, 10.0: 111, 20.0: 110}
FROZEN_TIES_OPEN = {0.0: 22, 0.5: 20, 1.0: 22, 2.0: 20,
                    5.0: 20, 10.0: 20, 20.0: 20}
FROZEN_N_CELLS = 4 * 4 * 4 * 4          # windows x taus x rhos x shapes = 256
FROZEN_CONST_STD = 110                  # FIXED_STD wins at every mu (strict)
FROZEN_CONST_TDM = 110                  # TDM wins at every mu (strict)
FROZEN_CONST_TIE = 20                   # exact tie at every mu
FROZEN_FLIP = 16                        # single STD->TDM crossover (2 with an
                                        # exact tie at mu=0, 2 at mu=1)
FROZEN_MU_STAR_REF = 1.443156           # twin-anchored reference crossover
COST_EPS = 1e-12


# -----------------------------------------------------------------------------
# Contract clauses S1..S7
# -----------------------------------------------------------------------------

def check_S1_declaration_consistency():
    """Every varied cell keeps the structural declarations: FIXED_LOW
    sub-window (no path to TARGET), FIXED_STD straddling (G_VERIFY refuses),
    TDM band window-contained (G_VERIFY admits). Also the V1 hazard analog:
    the naive toxicity-only minimizer picks FIXED_LOW in every cell."""
    n = 0
    for (a, b) in WINDOWS:
        for tau in TAUS:
            for shape in SHAPES:
                n += 1
                tb = tdm_band(a, b)
                fsb = fixed_std_band(tau)
                assert tb[0] >= a and tb[1] <= b
                assert FIXED_LOW_BAND[1] < a
                assert fsb[0] < a
                # naive toxicity-only minimizer (linear supra-window component);
                # at [15,25] with tau=0.1 both arms score 0 (tie broken by
                # declaration order) -- either pick is target-unreachable here,
                # so the hazard statement is unaffected.
                tox = {
                    FIXED_LOW: max(0.0, FIXED_LOW_BAND[1] - b) / b,
                    FIXED_STD: max(0.0, fsb[1] - b) / b,
                }
                pick = min(tox, key=tox.get)
                g = build_graph(a, b, tau, 0.0, shape, gated=True)
                assert pick == FIXED_LOW and TARGET not in g.adj[pick]
    print(f"S1_DECLARATION_CONSISTENCY cells={n} fixed_low_subwindow=True "
          f"fixed_std_straddles=True tdm_in_window=True "
          f"naive_tox_pick=FIXED_LOW(unreachable) -> PASS")
    return True


def check_S2_gated_selects_tdm_all_cells():
    """MAIN CLAIM. Under G_VERIFY, the TDM-guided route is the unique feasible
    path to TARGET and is selected in every varied cell: 4 windows x 4 tau x
    4 rho x 4 shapes x 7 mu = 1792 scheduler runs."""
    n_cells, n_runs, n_tdm = 0, 0, 0
    for (a, b) in WINDOWS:
        for tau in TAUS:
            for rho in RHOS:
                for shape in SHAPES:
                    n_cells += 1
                    g = build_graph(a, b, tau, rho, shape, gated=True)
                    paths = enumerate_paths(g, START, TARGET, L0=10.0)
                    assert paths == [TDM_PATH], f"feasible paths changed: {paths}"
                    for mu in MUS:
                        n_runs += 1
                        if winner(g, mu) == 'TDM':
                            n_tdm += 1
    ok = (n_tdm == n_runs == 1792)
    print(f"S2_GATED_SELECTS_TDM_ALL_CELLS cells={n_cells} runs={n_runs} "
          f"tdm_selected={n_tdm}/1792 -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_S3_twin_anchored_reference():
    """Reference cell (window [10,20], linear, tau=0.3, rho=0.059420,
    measured twin suffering): reproduces the parent contract exactly --
    gated V5 canonical numbers at mu=1 (integral 0.735099, peak 0.675679,
    total 1.410778) and the V4 open-gate counterfactual. Extends V4 over mu:
    open-gate selection is FIXED_STD for mu in {0, 0.5, 1} and TDM for
    mu >= 2, with analytic crossover mu* = (0.735099-0.7)/(0.7-0.675679)
    = 1.443156."""
    g_gated = build_graph(10.0, 20.0, 0.3, 0.059420, "linear",
                          gated=True, twin_anchored=True)
    path, m = mercyful_schedule(g_gated, START, TARGET, mu=1.0, L0=10.0)
    length, integral, peak, total = m
    ok_gated = (path == TDM_PATH
                and abs(integral - 0.735099) < 5e-7
                and abs(peak - 0.675679) < 5e-7
                and abs(total - 1.410778) < 5e-7)
    g_open = build_graph(10.0, 20.0, 0.3, 0.059420, "linear",
                         gated=False, twin_anchored=True)
    seq = [winner(g_open, mu) for mu in MUS]
    expected_seq = ['FIXED_STD', 'FIXED_STD', 'FIXED_STD', 'TDM', 'TDM', 'TDM', 'TDM']
    mu_star = (0.735099 - 0.7) / (0.7 - 0.675679)
    ok_open = (seq == expected_seq
               and abs(mu_star - FROZEN_MU_STAR_REF) < 5e-7)
    ok = ok_gated and ok_open
    print(f"S3_TWIN_ANCHORED_REFERENCE gated_integral={integral:.6f} "
          f"peak={peak:.6f} total={total:.6f} open_seq={seq} "
          f"mu_star={mu_star:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def route_costs(a, b, tau, rho, shape, mu):
    """Closed-form costs of the two open-gate routes in a functional cell.
    Both are linear in mu; equality is detected exactly (COST_EPS)."""
    s_vp = s_window(VANCO_PRE_BAND, a, b, shape)
    s_tg = s_window(tdm_band(a, b), a, b, shape) + rho
    s_fs = s_window(fixed_std_band(tau), a, b, shape)
    cost_tdm = (s_vp + s_tg) + mu * max(s_vp, s_tg)
    cost_std = s_fs * (1.0 + mu)
    return cost_tdm, cost_std


def route_coefficients(a, b, tau, rho, shape):
    """(intercept, slope) of each route cost as a linear function of mu.
    An all-mu tie holds iff BOTH coefficients agree (Grok round-2
    TIGHTENABLE: identity of linear functions, not sampled closeness)."""
    s_vp = s_window(VANCO_PRE_BAND, a, b, shape)
    s_tg = s_window(tdm_band(a, b), a, b, shape) + rho
    s_fs = s_window(fixed_std_band(tau), a, b, shape)
    return (s_vp + s_tg), max(s_vp, s_tg), s_fs, s_fs


def check_S4_open_gate_landscape():
    """HONEST NEGATIVE CONTROL. Under the verification-blind counterfactual
    (FIXED_STD admitted to TARGET), the TDM conclusion does NOT hold
    universally. Closed-form classification (both route costs linear in mu):
    FIXED_STD wins strictly at every tested mu in 110/256 cells (concentrated
    at tau in {0.1, 0.2} and the wide window [15,25]), TDM wins strictly
    everywhere in 110/256, 20/256 cells are EXACT TIES at every mu (tau=0.2,
    rho=0 makes FIXED_STD's band identical to VANCO_PRE's; [15,25], tau=0.1,
    rho=0 leaves both bands violating only the shared low edge), and 16/256
    cells show a single STD->TDM crossover (2 of them tied exactly at mu=0,
    2 at mu=1). The scheduler is cross-checked against this classification on
    every strict (non-tie) point: 1648 comparisons. Frozen counts are
    regression expectations."""
    tdm_wins = {mu: 0 for mu in MUS}
    std_wins = {mu: 0 for mu in MUS}
    ties = {mu: 0 for mu in MUS}
    n_const_std = n_const_tdm = n_const_tie = n_flip = 0
    monotone = True
    scheduler_agree = scheduler_checked = 0
    for (a, b) in WINDOWS:
        for tau in TAUS:
            for rho in RHOS:
                for shape in SHAPES:
                    g = build_graph(a, b, tau, rho, shape, gated=False)
                    seq = []
                    for mu in MUS:
                        ct, cs = route_costs(a, b, tau, rho, shape, mu)
                        w = ('TDM' if ct < cs - COST_EPS
                             else 'FIXED_STD' if cs < ct - COST_EPS
                             else 'TIE')
                        seq.append(w)
                        if w == 'TDM':
                            tdm_wins[mu] += 1
                        elif w == 'FIXED_STD':
                            std_wins[mu] += 1
                        else:
                            ties[mu] += 1
                        # cross-check the actual scheduler on strict points
                        if w != 'TIE':
                            scheduler_checked += 1
                            if winner(g, mu) == w:
                                scheduler_agree += 1
                    s = set(seq)
                    if s == {'FIXED_STD'}:
                        n_const_std += 1
                    elif s == {'TDM'}:
                        n_const_tdm += 1
                    elif s == {'TIE'}:
                        # strengthen: the two linear cost functions must be
                        # IDENTICAL (both intercept and slope), not merely
                        # close on the sampled mu grid.
                        a0, b0, c0, d0 = route_coefficients(a, b, tau, rho, shape)
                        assert (abs(a0 - c0) < COST_EPS
                                and abs(b0 - d0) < COST_EPS), \
                            f"sampled tie without coefficient identity {(a, b, tau, rho, shape)}"
                        n_const_tie += 1
                    else:
                        # non-constant cell: a crossover (possibly with an
                        # exact tie at the boundary mu=0 or at the crossover
                        # grid point); TDM must be present and no FIXED_STD
                        # may appear after the first TDM (single crossing).
                        assert 'TDM' in s, f"reverse-crossover cell {seq}"
                        n_flip += 1
                        # single STD->TDM switch: no FIXED_STD after first TDM
                        seen_tdm = False
                        for w in seq:
                            if w == 'TDM':
                                seen_tdm = True
                            elif w == 'FIXED_STD' and seen_tdm:
                                monotone = False
    ok = (tdm_wins == FROZEN_TDM_WINS_OPEN
          and std_wins == FROZEN_STD_WINS_OPEN
          and ties == FROZEN_TIES_OPEN
          and n_const_std == FROZEN_CONST_STD
          and n_const_tdm == FROZEN_CONST_TDM
          and n_const_tie == FROZEN_CONST_TIE
          and n_flip == FROZEN_FLIP
          and n_const_std + n_const_tdm + n_const_tie + n_flip == FROZEN_N_CELLS
          and monotone
          and scheduler_agree == scheduler_checked == 1648)
    print(f"S4_OPEN_GATE_LANDSCAPE tdm_wins_per_mu={tdm_wins} "
          f"ties_per_mu={ties} const_std={n_const_std} const_tdm={n_const_tdm} "
          f"const_tie={n_const_tie} flips={n_flip} single_crossover={monotone} "
          f"scheduler_xcheck={scheduler_agree}/{scheduler_checked} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_S5_gate_is_causal_robust():
    """The gate's causality (parent V4) is not an artifact of the reference
    point: gated selection is TDM in 256/256 (window, tau, rho, shape) cells
    while the open-gate strict classification at mu=1 has FIXED_STD winning
    113/256 (plus 22 exact ties). The correspondence claim 'verified dosing
    is the optimum' is therefore a statement about the verified feasible
    set, robust across every axis varied."""
    n_gated_tdm = 0
    for (a, b) in WINDOWS:
        for tau in TAUS:
            for rho in RHOS:
                for shape in SHAPES:
                    g = build_graph(a, b, tau, rho, shape, gated=True)
                    if winner(g, 1.0) == 'TDM':
                        n_gated_tdm += 1
    open_std_at_mu1 = FROZEN_STD_WINS_OPEN[1.0]
    ok = (n_gated_tdm == FROZEN_N_CELLS and open_std_at_mu1 == 113)
    print(f"S5_GATE_IS_CAUSAL_ROBUST gated_tdm_at_mu1={n_gated_tdm}/256 "
          f"open_fixed_std_at_mu1={open_std_at_mu1}/256 "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_S6_mimic_iv_direction_unchanged():
    """The real-cohort side of the correspondence is untouched by the model
    variations; re-assert the parent V6 checks (CIs exclude 1.0; cohort
    arithmetic) so this contract stands alone."""
    for name in ("icu_mortality", "hospital_mortality", "aki"):
        est, lo, hi = MIMIC[name]
        assert lo <= est <= hi, f"{name}: CI does not contain point estimate"
    ci_exclude_null = all(MIMIC[k][2] < 1.0
                          for k in ("icu_mortality", "hospital_mortality", "aki"))
    counts_ok = (MIMIC["n_tdm"] + MIMIC["n_non_tdm"] == MIMIC["n_total"]
                 and abs(MIMIC["n_tdm"] / MIMIC["n_total"] - 0.378) < 0.001)
    ok = ci_exclude_null and counts_ok
    print(f"S6_MIMIC_IV_DIRECTION_UNCHANGED "
          f"icu_mort={MIMIC['icu_mortality']} hosp_mort={MIMIC['hospital_mortality']} "
          f"aki={MIMIC['aki']} n={MIMIC['n_total']} ci_exclude_null={ci_exclude_null} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_S7_no_overreach():
    """The contract itself must carry the scope guards."""
    with open(os.path.abspath(__file__)) as f:
        src = f.read()
    ok = ('not medical guidance' in src) and ('SYNTHETIC' in src)
    print(f"S7_NO_OVERREACH scope_guards_present={ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("MERCYFUL LEARNING x MIMIC-IV VANCOMYCIN TDM — sensitivity contract")
    print("=" * 70)
    results.append(("S1", check_S1_declaration_consistency()))
    results.append(("S2", check_S2_gated_selects_tdm_all_cells()))
    results.append(("S3", check_S3_twin_anchored_reference()))
    results.append(("S4", check_S4_open_gate_landscape()))
    results.append(("S5", check_S5_gate_is_causal_robust()))
    results.append(("S6", check_S6_mimic_iv_direction_unchanged()))
    results.append(("S7", check_S7_no_overreach()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"MERCYFUL_MIMIC_IV_SENSITIVITY_VERDICT S_GREEN ({passed}/{total} clauses PASS)")
        print("MERCYFUL_MIMIC_IV_SENSITIVITY_NOTE gated_verdict_invariant=1792/1792; "
              "open_gate_conclusion_changes=(const_std=110 const_tdm=110 const_tie=20 flips=16); "
              "synthetic_graph; no_patient_data; not_medical_guidance")
        return 0
    print(f"MERCYFUL_MIMIC_IV_SENSITIVITY_VERDICT S_AMBER ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
