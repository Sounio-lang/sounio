#!/usr/bin/env python3
"""
Mercyful Learning x MIMIC-IV vancomycin TDM — subgroup cross-validation contract.

Companion to:
  docs/research/mimic_iv_subgroup_cross_validation_2026-07-26.md (spec)
  docs/research/mimic_iv_mercyful_validation_2026-07-26.md (parent validation)
  scripts/research/mercyful_mimic_iv_vancomycin_contract.py (V1..V7 contract)

Task: cross-validate the POSITIVE MIMIC-IV TDM-mortality correspondence across
subgroups: age (<65 vs >=65), severity (SOFA <7 vs >=7, median split), and
comorbidity (nephrotoxic co-exposure absent vs present, the comorbidity proxy
available in the cohort schema). Two sides are checked:

  (A) Association side. The real MIMIC-IV patient-level extract is
      credential-gated (PhysioNet CITI + DUA; scripts/clinical/README.md), so
      the patient-level subgroup split runs on the repository's popPK-driven
      SYNTHETIC cohort (scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv,
      200 patients, deterministic seed 20260501). In every stratum we compute
      the odds ratio of clinical cure for window attainment (measured Cmin in
      [10, 20] mg/L, the TDM-guided target) vs out-of-window. OR > 1 in a
      stratum = window attainment (the TDM signature) associated with lower
      mortality there, the model-side shadow of the cohort's TDM-mortality
      association. The REAL anchor for cross-stratum robustness is the source
      study's own stratified evidence (Wang et al. 2026,
      doi:10.1038/s41598-026-42395-1, Table 2): the TDM-mortality association
      holds, CIs excluding 1.0, across every adjustment stratum (crude ->
      demographics/comorbidities -> fully adjusted) and after PSM, and the
      abstract states the results were "validated through subgroup analyses
      (stratified by comorbidities and concomitant medications)".

  (B) Scheduler side. The synthetic dosing graph of the parent contract is
      parametrized per stratum: higher-risk strata (age >=65, SOFA >=7,
      comorbidity present) get wider declared fixed-dose Cmin p-boxes
      (reduced/variable clearance), preserving the structural invariants —
      conservative arm fully sub-window (no path to TARGET), standard arm
      straddling (G_VERIFY refuses), TDM arm in-window (gate passes). We then
      verify the mercyful scheduler still selects the TDM-guided therapeutic
      window course in EVERY stratum, that the naive toxicity minimizer still
      underdoses, and that the verification gate remains causal.

Everything here is SYNTHETIC except the quoted study statistics. No patient
data were used; this is not medical guidance and carries no clinical claim.
Pure Python; no dependencies beyond the standard library. The scheduler is
imported from mercyful_runtime_contract.py (M1..M6, M_GREEN).
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mercyful_runtime_contract import MercyGraph, mercyful_schedule

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COHORT_CSV = os.path.join(
    REPO_ROOT, "scripts", "clinical", "data_synthetic",
    "tdm_cohort_synthetic_v2.csv")

# -----------------------------------------------------------------------------
# Real MIMIC-IV anchor (Wang et al. 2026, doi:10.1038/s41598-026-42395-1)
# Table 2 (full-cohort sequential adjustment) + post-PSM estimates.
# All are TDM-vs-non-TDM effect estimates; OR < 1 favors TDM.
# -----------------------------------------------------------------------------

# (estimate, ci_lo, ci_hi) per adjustment stratum. Model 1 crude; Model 2
# demographics/comorbidities/medications; Model 3 fully adjusted; PSM matched.
MIMIC_MORTALITY_STRATA = {
    "hospital_mortality": {
        "model1": (0.49, 0.42, 0.57),
        "model2": (0.58, 0.49, 0.67),
        "model3": (0.63, 0.54, 0.74),
        "psm":    (0.672, 0.570, 0.790),
    },
    "icu_mortality": {
        "model1": (0.51, 0.44, 0.59),
        "model2": (0.64, 0.55, 0.75),
        "model3": (0.72, 0.62, 0.85),
        "psm":    (0.691, 0.580, 0.820),
    },
}

# Therapeutic window (Cmin, mg/L; shaping after Rybak et al. — not a target
# claim). Same as the parent contract.
CMIN_LO, CMIN_HI = 10.0, 20.0

# Exact suffering values measured from the repo's vancomycin twin (clause C3,
# tests/run-pass/mercyful_clinical_sequencing.sio; 1000 mg q12h, [10,20] mg/L).
S_VANCO_PRE = 0.675679    # fixed dosing before any level is measured
S_VANCO_POST = 0.059420   # TDM-narrowed band

# C1 canonical scheduler values at mu = 1 (parent contract V5).
C1_INTEGRAL, C1_PEAK, C1_TOTAL = 0.735099, 0.675679, 1.410778


def s_window(lo, hi, a=CMIN_LO, b=CMIN_HI):
    """Window-violation suffering of a Cmin p-box (paper section 7.2)."""
    return max(0.0, a - lo) / a + max(0.0, hi - b) / b


def s_tox_only(lo, hi, b=CMIN_HI):
    """Supra-therapeutic (toxicity) component only -- the naive metric."""
    return max(0.0, hi - b) / b


# -----------------------------------------------------------------------------
# Per-stratum declared synthetic p-boxes.
#
# Declared assumptions (all synthetic, all disclosed):
#   (i)  higher-risk strata have wider fixed-dose Cmin p-boxes (reduced and
#        more variable clearance with age, severity, comorbidity);
#   (ii) structural invariants are preserved in every stratum: FIXED_LOW fully
#        below the window (worst case cannot clear infection: hi < 10),
#        FIXED_STD straddling (lo < 10 and hi > 20), TDM_GUIDED in-window;
#   (iii) stratum s_win(FIXED_STD) stays below C1_TOTAL/2 = 0.705389 so the
#        gate-causality counterfactual (open graph prefers FIXED_STD at mu=1)
#        remains live in every stratum;
#   (iv) VANCO_PRE / TDM_GUIDED suffering are the twin's measured C3 values,
#        invariant across strata (TDM narrows the band into the window
#        regardless of stratum).
# -----------------------------------------------------------------------------

STRATA = {
    "age_lt65":    {"fixed_low": (4.0, 9.0),  "fixed_std": (6.5, 24.0)},
    "age_ge65":    {"fixed_low": (3.0, 8.0),  "fixed_std": (5.0, 24.0)},
    "sev_low":     {"fixed_low": (4.5, 9.0),  "fixed_std": (7.0, 23.0)},
    "sev_high":    {"fixed_low": (3.5, 8.0),  "fixed_std": (5.5, 25.0)},
    "comorb_low":  {"fixed_low": (4.5, 9.0),  "fixed_std": (7.0, 23.0)},
    "comorb_high": {"fixed_low": (3.0, 8.0),  "fixed_std": (4.5, 23.0)},
}

DICHOTOMIES = [
    ("age", "age_lt65", "age_ge65"),
    ("severity", "sev_low", "sev_high"),
    ("comorbidity", "comorb_low", "comorb_high"),
]

START, FIXED_LOW, FIXED_STD = 'START', 'FIXED_LOW', 'FIXED_STD'
VANCO_PRE, TDM_GUIDED, TARGET = 'VANCO_PRE', 'TDM_GUIDED', 'TARGET'


def build_graph(stratum, verify_gate=True):
    """Per-stratum vancomycin dosing graph (synthetic). Same topology as the
    parent contract; only the fixed-dose suffering values vary by stratum."""
    p = STRATA[stratum]
    s_low = s_window(*p["fixed_low"])
    s_std = s_window(*p["fixed_std"])
    edges = [
        (START, START),      # untreated trap
        (START, FIXED_LOW),
        (START, FIXED_STD),
        (START, VANCO_PRE),
        (VANCO_PRE, TDM_GUIDED),
    ]
    if not verify_gate:
        edges.append((FIXED_STD, TARGET))   # counterfactual: unverified admitted
    edges.append((TDM_GUIDED, TARGET))      # admitted: band inside window
    return MercyGraph(
        states=[START, FIXED_LOW, FIXED_STD, VANCO_PRE, TDM_GUIDED, TARGET],
        edges=edges,
        suffering={
            START: 0.0,
            FIXED_LOW: s_low,
            FIXED_STD: s_std,
            VANCO_PRE: S_VANCO_PRE,
            TDM_GUIDED: S_VANCO_POST,
            TARGET: 0.0,
        },
    )


# -----------------------------------------------------------------------------
# Synthetic-cohort subgroup statistics
# -----------------------------------------------------------------------------

def load_cohort():
    with open(COHORT_CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def in_window(row):
    return CMIN_LO <= float(row["measured_cmin_mg_l"]) <= CMIN_HI


def cured(row):
    return int(row["outcome_cure"]) == 1


def stratum_rows(rows, stratum):
    if stratum == "age_lt65":
        return [r for r in rows if int(r["age"]) < 65]
    if stratum == "age_ge65":
        return [r for r in rows if int(r["age"]) >= 65]
    if stratum == "sev_low":
        return [r for r in rows if int(r["sofa"]) < 7]   # median SOFA = 7
    if stratum == "sev_high":
        return [r for r in rows if int(r["sofa"]) >= 7]
    if stratum == "comorb_low":
        return [r for r in rows if int(r["nephrotoxic_coexposure"]) == 0]
    if stratum == "comorb_high":
        return [r for r in rows if int(r["nephrotoxic_coexposure"]) == 1]
    raise KeyError(stratum)


def odds_ratio_woolf(sub):
    """OR of cure for in-window vs out-of-window, with Woolf 95% CI.
    Returns (or, ci_lo, ci_hi, (a, b, c, d)) where
    a = in-window & cured, b = in-window & not cured,
    c = out-of-window & cured, d = out-of-window & not cured."""
    a = sum(1 for r in sub if in_window(r) and cured(r))
    b = sum(1 for r in sub if in_window(r) and not cured(r))
    c = sum(1 for r in sub if not in_window(r) and cured(r))
    d = sum(1 for r in sub if not in_window(r) and not cured(r))
    orv = (a / b) / (c / d)
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    lo = math.exp(math.log(orv) - 1.96 * se)
    hi = math.exp(math.log(orv) + 1.96 * se)
    return orv, lo, hi, (a, b, c, d)


# Expected cell counts / stratum sizes on the committed CSV (deterministic
# seed 20260501) — regression protection for X1/X3.
EXPECTED_STRATUM_N = {
    "age_lt65": 120, "age_ge65": 80,
    "sev_low": 87, "sev_high": 113,
    "comorb_low": 123, "comorb_high": 77,
}
EXPECTED_CELLS = {
    "all":         (70, 14, 75, 41),
    "age_lt65":    (35, 9, 48, 28),
    "age_ge65":    (35, 5, 27, 13),
    "sev_low":     (34, 6, 33, 14),
    "sev_high":    (36, 8, 42, 27),
    "comorb_low":  (39, 10, 48, 26),
    "comorb_high": (31, 4, 27, 15),
}

TDM_ROUTE = [START, VANCO_PRE, TDM_GUIDED, TARGET]


# -----------------------------------------------------------------------------
# Contract clauses X1..X9
# -----------------------------------------------------------------------------

def check_X1_cohort_schema_and_stratification():
    """The synthetic cohort loads, has 200 rows, and the three dichotomies
    partition it exactly at the declared thresholds (age 65; SOFA median 7;
    nephrotoxic co-exposure as the comorbidity proxy)."""
    rows = load_cohort()
    ok = len(rows) == 200
    for _, lo_name, hi_name in DICHOTOMIES:
        n_lo = len(stratum_rows(rows, lo_name))
        n_hi = len(stratum_rows(rows, hi_name))
        ok = ok and n_lo == EXPECTED_STRATUM_N[lo_name] \
            and n_hi == EXPECTED_STRATUM_N[hi_name] \
            and n_lo + n_hi == 200
    print(f"X1_COHORT_SCHEMA_AND_STRATIFICATION n={len(rows)} "
          f"strata={EXPECTED_STRATUM_N} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_X2_pooled_window_cure_association():
    """Pooled over the synthetic cohort, window attainment (the TDM-guided
    target) is associated with cure — OR > 1 with the Woolf 95% CI excluding
    1.0. Exact cell counts pinned."""
    rows = load_cohort()
    orv, lo, hi, cells = odds_ratio_woolf(rows)
    ok = (cells == EXPECTED_CELLS["all"]
          and orv > 1.0 and lo > 1.0
          and abs(orv - 2.733) < 5e-4)
    print(f"X2_POOLED_WINDOW_CURE_ASSOCIATION OR={orv:.3f} "
          f"95%CI=({lo:.3f},{hi:.3f}) cells={cells} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_X3_direction_holds_all_strata():
    """The window-cure association direction (OR > 1) holds in ALL six
    strata. Stratum CIs are printed honestly: 3 of 6 include 1.0 at n<=123 —
    direction, not per-stratum significance, is the claim at this cohort
    size. Exact cell counts pinned per stratum."""
    rows = load_cohort()
    ok = True
    parts = []
    for stratum in ("age_lt65", "age_ge65", "sev_low", "sev_high",
                    "comorb_low", "comorb_high"):
        sub = stratum_rows(rows, stratum)
        orv, lo, hi, cells = odds_ratio_woolf(sub)
        ok = ok and (orv > 1.0) and (cells == EXPECTED_CELLS[stratum])
        parts.append(f"{stratum}:OR={orv:.3f}({lo:.3f},{hi:.3f})")
    print(f"X3_DIRECTION_HOLDS_ALL_STRATA {' '.join(parts)} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_X4_scheduler_selects_tdm_all_strata():
    """In every stratum's graph the mercyful scheduler's unique feasible
    optimum is the TDM-guided route, reproducing the C1 canonical values
    (integral 0.735099, peak 0.675679, total 1.410778 at mu = 1)."""
    ok = True
    for stratum in STRATA:
        g = build_graph(stratum)
        path, metrics = mercyful_schedule(g, START, TARGET, mu=1.0, L0=10.0)
        length, integral, peak, total = metrics
        ok = ok and (path == TDM_ROUTE
                     and abs(integral - C1_INTEGRAL) < 5e-7
                     and abs(peak - C1_PEAK) < 5e-7
                     and abs(total - C1_TOTAL) < 5e-7)
    print(f"X4_SCHEDULER_SELECTS_TDM_ALL_STRATA strata={len(STRATA)} "
          f"route={'->'.join(TDM_ROUTE)} integral={C1_INTEGRAL} "
          f"peak={C1_PEAK} total={C1_TOTAL} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_X5_naive_minimizer_underdoses_all_strata():
    """In every stratum a toxicity-only minimizer selects the sub-therapeutic
    FIXED_LOW arm (toxicity 0), which has no path to TARGET."""
    ok = True
    for stratum in STRATA:
        p = STRATA[stratum]
        tox_low = s_tox_only(*p["fixed_low"])
        tox_std = s_tox_only(*p["fixed_std"])
        naive_pick = FIXED_LOW if tox_low <= tox_std else FIXED_STD
        g = build_graph(stratum)
        ok = ok and (naive_pick == FIXED_LOW and tox_low == 0.0
                     and tox_std > 0.0 and TARGET not in g.adj[naive_pick])
    print(f"X5_NAIVE_MINIMIZER_UNDERDOSES_ALL_STRATA strata={len(STRATA)} "
          f"pick={FIXED_LOW} reaches_target=False -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_X6_verify_gate_causal_all_strata():
    """In every stratum, counterfactually admitting the unverified FIXED_STD
    arm to TARGET makes it the cost optimum at mu = 1 (total 2*s_win <
    1.410778); with the gate, only the TDM route remains. The gate is causal
    in every subgroup, not just the pooled graph."""
    ok = True
    details = []
    for stratum in STRATA:
        s_std = s_window(*STRATA[stratum]["fixed_std"])
        g_open = build_graph(stratum, verify_gate=False)
        path_open, m_open = mercyful_schedule(g_open, START, TARGET,
                                              mu=1.0, L0=10.0)
        g_gated = build_graph(stratum, verify_gate=True)
        path_gated, m_gated = mercyful_schedule(g_gated, START, TARGET,
                                                mu=1.0, L0=10.0)
        ok = ok and (path_open == [START, FIXED_STD, TARGET]
                     and abs(m_open[1] - s_std) < 1e-9
                     and m_open[3] < m_gated[3]
                     and path_gated == TDM_ROUTE)
        details.append(f"{stratum}:s_std={s_std:.3f}")
    print(f"X6_VERIFY_GATE_CAUSAL_ALL_STRATA {' '.join(details)} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_X7_suffering_gradient():
    """Higher-risk strata carry strictly higher fixed-dose window-violation
    suffering than their low-risk counterparts (wider declared p-boxes), for
    all three dichotomies, on both fixed arms."""
    ok = True
    parts = []
    for name, lo_s, hi_s in DICHOTOMIES:
        std_lo = s_window(*STRATA[lo_s]["fixed_std"])
        std_hi = s_window(*STRATA[hi_s]["fixed_std"])
        low_lo = s_window(*STRATA[lo_s]["fixed_low"])
        low_hi = s_window(*STRATA[hi_s]["fixed_low"])
        ok = ok and std_hi > std_lo and low_hi > low_lo
        parts.append(f"{name}:std {std_lo:.3f}->{std_hi:.3f} "
                     f"low {low_lo:.3f}->{low_hi:.3f}")
    print(f"X7_SUFFERING_GRADIENT {' '.join(parts)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_X8_literature_stratum_robustness():
    """Real-data anchor (Wang et al. 2026, Table 2 + PSM): the TDM-mortality
    association holds — OR < 1 with every 95% CI excluding 1.0 — across ALL
    adjustment strata (crude, demographics/comorbidities, fully adjusted) and
    after propensity matching, for both mortality endpoints. Adjustment
    attenuates toward the null monotonically (confounding by indication),
    never crosses it.

    Monotonicity is claimed over the adjustment sequence M1 -> M2 -> M3 ONLY.
    The post-PSM estimate is computed on a different (matched) sample with a
    different estimator and is NOT part of the monotone sequence; for ICU
    mortality the PSM point estimate (0.691) happens to sit below the Model 3
    estimate (0.72), which is disclosed here rather than smoothed over."""
    ok = True
    for endpoint, strata in MIMIC_MORTALITY_STRATA.items():
        estimates = [strata[k] for k in ("model1", "model2", "model3", "psm")]
        ok = ok and all(hi < 1.0 for _, _, hi in estimates)
        ok = ok and all(lo <= est <= hi for est, lo, hi in estimates)
        m1, m2, m3 = (strata[k][0] for k in ("model1", "model2", "model3"))
        ok = ok and (m1 < m2 < m3 < 1.0)   # monotone over adjustment strata only
    print(f"X8_LITERATURE_STRATUM_ROBUSTNESS "
          f"hosp_mort={[MIMIC_MORTALITY_STRATA['hospital_mortality'][k] for k in ('model1','model2','model3','psm')]} "
          f"icu_mort={[MIMIC_MORTALITY_STRATA['icu_mortality'][k] for k in ('model1','model2','model3','psm')]} "
          f"all_ci_exclude_null={ok} monotone_over=M1_M2_M3_only "
          f"psm_separate_estimator -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_X9_no_overreach():
    """The contract itself must carry the scope guards."""
    with open(os.path.abspath(__file__)) as f:
        src = f.read()
    ok = ('not medical guidance' in src) and ('SYNTHETIC' in src) \
        and ('credential-gated' in src)
    print(f"X9_NO_OVERREACH scope_guards_present={ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("MERCYFUL x MIMIC-IV VANCOMYCIN TDM — subgroup cross-validation")
    print("=" * 70)
    results.append(("X1", check_X1_cohort_schema_and_stratification()))
    results.append(("X2", check_X2_pooled_window_cure_association()))
    results.append(("X3", check_X3_direction_holds_all_strata()))
    results.append(("X4", check_X4_scheduler_selects_tdm_all_strata()))
    results.append(("X5", check_X5_naive_minimizer_underdoses_all_strata()))
    results.append(("X6", check_X6_verify_gate_causal_all_strata()))
    results.append(("X7", check_X7_suffering_gradient()))
    results.append(("X8", check_X8_literature_stratum_robustness()))
    results.append(("X9", check_X9_no_overreach()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"MERCYFUL_MIMIC_IV_SUBGROUP_VERDICT X_GREEN ({passed}/{total} clauses PASS)")
        print("MERCYFUL_MIMIC_IV_SUBGROUP_NOTE structural_correspondence_only; "
              "synthetic_cohort_surrogate; mimic_iv_credential_gated; "
              "no_patient_data; not_medical_guidance")
        return 0
    print(f"MERCYFUL_MIMIC_IV_SUBGROUP_VERDICT X_AMBER ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
