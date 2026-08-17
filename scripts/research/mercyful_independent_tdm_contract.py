#!/usr/bin/env python3
"""
Mercyful Learning x INDEPENDENT vancomycin TDM datasets — structural
correspondence contract (cross-dataset validation).

Companion to:
  docs/research/independent_dataset_vancomycin_tdm_validation_2026-07-26.md
  docs/research/mimic_iv_mercyful_validation_2026-07-26.md (anchor)

Independence criterion: a source counts iff its patient data are disjoint from
MIMIC-IV (Beth Israel Deaconess ICU/ED, 2008-2022).

Counted independent sources (verified 2026-07-26):
  * Ye, Tang & Zhai 2013, PLoS One 8(10):e77169
    (doi:10.1371/journal.pone.0077169; PMCID PMC3799644) — meta-analysis of
    1 RCT + 5 cohort studies (USA, Spain, Japan x3, China), 521 patients
    (249 TDM / 272 non-TDM), era 1990-2010. TDM vs non-TDM:
      clinical efficacy  OR 2.62 (95% CI 1.34-5.11), P = 0.005,  I^2 = 0%
      nephrotoxicity     OR 0.25 (95% CI 0.13-0.48), P < 0.0001, I^2 = 0%
  * Yang et al. 2024, J Clin Pharmacol 64(1):19-29 (doi:10.1002/jcph.2363;
    PMID 37779493) — Australian hospital AUC-TDM advisory service, 971 courses;
    nephrotoxicity 15% pre vs 10% post (P = 0.075; direction-concordant, NS).
  * Hou et al. 2021, Front Pharmacol 12:690157 (doi:10.3389/fphar.2021.690157) —
    eICU-CRD v2.0, 3,603 monitored patients, 335 ICUs / 208 hospitals. BOUNDARY
    CONDITION: no non-TDM arm; trough targeting not associated with reduced
    mortality; supratherapeutic mean VTC (>20 mg/L) associated with HIGHER
    mortality (ICU OR 2.428 [1.385-4.258]; hospital OR 1.585 [1.053-2.387]).

Excluded as NON-independent (MIMIC-IV replications; concordant but same
sampling frame, so they must never be cited as independent):
  * Peng et al. 2024 (PMID 39726684) — MIMIC-IV sepsis, HR 0.66 (0.61-0.71).
  * Peng et al. 2026 (PMCID PMC12819319) — MIMIC-IV RRT, HR 0.457-0.478.

NO-REFIT GUARANTEE: the dosing graph is IMPORTED unchanged from
mercyful_mimic_iv_vancomycin_contract.py (frozen before the MIMIC-IV
comparison). Nothing below re-derives, re-fits, or re-declares a single
parameter. The graph, p-boxes, and suffering values are SYNTHETIC; no patient
data were used; this is not medical guidance and carries no clinical claim.

Pure Python; no dependencies beyond the standard library.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The graph and all model constants come from the frozen MIMIC-IV contract.
# Importing (rather than copying) is what makes "no refit" enforceable: any
# change to the anchor graph propagates here and is caught by I4/I5.
from mercyful_mimic_iv_vancomycin_contract import (
    CMIN_LO,
    FIXED_LOW,
    FIXED_LOW_BAND,
    FIXED_STD,
    FIXED_STD_BAND,
    START,
    TARGET,
    TDM_GUIDED,
    VANCO_PRE,
    build_graph,
    s_tox_only,
    s_window,
)
from mercyful_runtime_contract import mercyful_schedule

# -----------------------------------------------------------------------------
# Independent-source statistics (verbatim from the published texts, 2026-07-26)
# -----------------------------------------------------------------------------

YE2013 = {
    "cite": "Ye, Tang & Zhai 2013, PLoS One 8(10):e77169",
    "doi": "10.1371/journal.pone.0077169",
    "dataset": "6 studies (1 RCT + 5 cohorts), USA/Spain/Japan/China",
    "era": "1990-2010",
    "is_mimic_iv": False,
    "n_total": 521,
    "n_tdm": 249,
    "n_non_tdm": 272,
    # Pooled Mantel-Haenszel fixed-effect estimates, TDM vs non-TDM.
    # OR > 1 favors TDM for efficacy; OR < 1 favors TDM for nephrotoxicity.
    "clinical_efficacy": (2.62, 1.34, 5.11),
    "nephrotoxicity": (0.25, 0.13, 0.48),
    "i_squared": 0.0,
}

YANG2024 = {
    "cite": "Yang et al. 2024, J Clin Pharmacol 64(1):19-29",
    "doi": "10.1002/jcph.2363",
    "dataset": "single Australian tertiary hospital, before/after AUC-TDM service",
    "is_mimic_iv": False,
    "n_courses": 971,
    "n_pre": 764,
    "n_post": 207,
    "nephrotox_pre_pct": 15.0,
    "nephrotox_post_pct": 10.0,
    "p_value": 0.075,  # direction-concordant, NOT significant: no weight carried
}

HOU2021_EICU = {
    "cite": "Hou et al. 2021, Front Pharmacol 12:690157",
    "doi": "10.3389/fphar.2021.690157",
    "dataset": "eICU-CRD v2.0, 335 ICUs at 208 US hospitals",
    "is_mimic_iv": False,
    "n_total": 3603,
    # Boundary condition: all patients monitored (>=2 VTC records); contrast is
    # mean-VTC band vs <10 mg/L, NOT TDM vs non-TDM.
    "icu_mortality_vtc_gt20": (2.428, 1.385, 4.258),   # supratherapeutic harm
    "hosp_mortality_vtc_gt20": (1.585, 1.053, 2.387),  # supratherapeutic harm
    "icu_mortality_vtc_10_15": (1.705, 0.975, 2.981),  # NS: no benefit shown
    "hosp_mortality_vtc_15_20": (1.370, 0.924, 2.029),  # NS: no benefit shown
    # Median creatinine clearance falls monotonically across VTC bands
    # (129.7 / 109.8 / 91.9 / 75.2 ml/min): severe residual confounding by
    # renal function, registered in the spec doc section 5.
}

EXCLUDED_NON_INDEPENDENT = [
    {"cite": "Peng et al. 2024, Front Med (PMID 39726684)",
     "dataset": "MIMIC-IV sepsis subset", "is_mimic_iv": True,
     "mortality_hr": (0.66, 0.61, 0.71)},
    {"cite": "Peng et al. 2026 (PMCID PMC12819319)",
     "dataset": "MIMIC-IV v3.1 RRT subset", "is_mimic_iv": True,
     "mortality_hr": (0.457, 0.385, 0.544)},
]

INDEPENDENT_SOURCES = [YE2013, YANG2024, HOU2021_EICU]

# BOUNDARY REGISTRATION (clause I6 checks this string survives; do not remove):
# The eICU-CRD study does NOT replicate a mortality benefit of vancomycin
# trough targeting. No mean-VTC band showed significantly reduced mortality vs
# <10 mg/L. The independent support for the framework covers the TDM-vs-non-TDM
# decision structure (efficacy and nephrotoxicity), NOT trough-level escalation.
EICU_NON_REPLICATION_REGISTERED = True


# -----------------------------------------------------------------------------
# Contract clauses I1..I7
# -----------------------------------------------------------------------------

def check_I1_independence_from_mimic():
    """Every counted source is disjoint from MIMIC-IV, and every known
    MIMIC-IV replication is on the exclusion list (so it cannot be silently
    upgraded to 'independent' later)."""
    counted_disjoint = all(not s["is_mimic_iv"] for s in INDEPENDENT_SOURCES)
    excluded_flagged = all(s["is_mimic_iv"] for s in EXCLUDED_NON_INDEPENDENT)
    counts_ok = (YE2013["n_tdm"] + YE2013["n_non_tdm"] == YE2013["n_total"]
                 and YANG2024["n_pre"] + YANG2024["n_post"] == YANG2024["n_courses"])
    ok = counted_disjoint and excluded_flagged and counts_ok \
        and len(EXCLUDED_NON_INDEPENDENT) == 2
    print(f"I1_INDEPENDENCE_FROM_MIMIC sources={len(INDEPENDENT_SOURCES)} "
          f"disjoint={counted_disjoint} "
          f"excluded_mimic_replications={len(EXCLUDED_NON_INDEPENDENT)} "
          f"ye2013_n={YE2013['n_total']} (tdm={YE2013['n_tdm']}) "
          f"yang2024_courses={YANG2024['n_courses']} "
          f"hou2021_eicu_n={HOU2021_EICU['n_total']} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_I2_efficacy_direction_match():
    """Efficacy (lower window bound). Independent pooled evidence: TDM raises
    clinical efficacy, OR 2.62 with CI excluding 1.0. Model side: the
    sub-therapeutic FIXED_LOW arm (worst-case band below the window) has no
    edge to TARGET -- unchecked efficacy shortfall is priced as failure."""
    est, lo, hi = YE2013["clinical_efficacy"]
    assert lo <= est <= hi, "efficacy CI does not contain point estimate"
    evidence_favors_tdm = lo > 1.0
    g = build_graph()
    model_prices_shortfall = (FIXED_LOW_BAND[1] < CMIN_LO) and (TARGET not in g.adj[FIXED_LOW])
    ok = evidence_favors_tdm and model_prices_shortfall
    print(f"I2_EFFICACY_DIRECTION_MATCH ye_efficacy={YE2013['clinical_efficacy']} "
          f"model_subtherapeutic_unreachable={model_prices_shortfall} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_I3_toxicity_direction_match():
    """Toxicity (upper window bound). Independent pooled evidence: TDM cuts
    nephrotoxicity, OR 0.25 with CI excluding 1.0; Yang 2024 direction
    concordant (15% -> 10%, NS, carries no significance weight). Model side:
    the straddling FIXED_STD band pays a positive supratherapeutic component."""
    est, lo, hi = YE2013["nephrotoxicity"]
    assert lo <= est <= hi, "nephrotoxicity CI does not contain point estimate"
    evidence_favors_tdm = hi < 1.0
    yang_concordant = YANG2024["nephrotox_post_pct"] < YANG2024["nephrotox_pre_pct"] \
        and YANG2024["p_value"] > 0.05  # registered as underpowered, not hidden
    model_prices_exceedance = s_tox_only(*FIXED_STD_BAND) > 0.0
    ok = evidence_favors_tdm and model_prices_exceedance and yang_concordant
    print(f"I3_TOXICITY_DIRECTION_MATCH ye_nephrotox={YE2013['nephrotoxicity']} "
          f"model_prices_supratherapeutic={s_tox_only(*FIXED_STD_BAND)} "
          f"yang2024_concordant_ns={yang_concordant} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_I4_scheduler_unchanged_still_selects_tdm():
    """NO-REFIT: the imported, frozen graph still yields the TDM-guided route
    as the unique feasible optimum with the exact canonical C1 values
    (integral 0.735099, peak 0.675679, total 1.410778 at mu = 1)."""
    g = build_graph()
    path, metrics = mercyful_schedule(g, START, TARGET, mu=1.0, L0=10.0)
    length, integral, peak, total = metrics
    ok = (path == [START, VANCO_PRE, TDM_GUIDED, TARGET]
          and abs(integral - 0.735099) < 5e-7
          and abs(peak - 0.675679) < 5e-7
          and abs(total - 1.410778) < 5e-7)
    print(f"I4_SCHEDULER_UNCHANGED_STILL_SELECTS_TDM path={path} "
          f"integral={integral:.6f} peak={peak:.6f} total={total:.6f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_I5_verify_gate_still_causal():
    """The gate causality statement must survive unchanged on the frozen
    graph: with unverified fixed dosing admitted, the scheduler rationally
    picks the non-TDM arm (integral 0.7 < 0.735099)."""
    g_open = build_graph(verify_gate=False)
    path_open, m_open = mercyful_schedule(g_open, START, TARGET, mu=1.0, L0=10.0)
    g_gated = build_graph(verify_gate=True)
    path_gated, m_gated = mercyful_schedule(g_gated, START, TARGET, mu=1.0, L0=10.0)
    ok = (path_open == [START, FIXED_STD, TARGET]
          and path_gated == [START, VANCO_PRE, TDM_GUIDED, TARGET]
          and m_open[1] < m_gated[1])
    print(f"I5_VERIFY_GATE_STILL_CAUSAL open_integral={m_open[1]:.6f} "
          f"gated_integral={m_gated[1]:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_I6_eicu_boundary_condition():
    """eICU-CRD (Hou 2021) is a boundary condition, not supporting evidence.
    (a) Its supratherapeutic-harm direction matches the model's upper window
    bound: mean VTC >20 mg/L associated with higher mortality, CIs excluding
    1.0. (b) Its non-replication of trough-targeting benefit (no band
    significantly BELOW 1.0 vs <10 mg/L) must remain explicitly registered
    in this contract; stripping the registration fails this clause."""
    icu_est, icu_lo, icu_hi = HOU2021_EICU["icu_mortality_vtc_gt20"]
    hosp_est, hosp_lo, hosp_hi = HOU2021_EICU["hosp_mortality_vtc_gt20"]
    harm_direction_matches = icu_lo > 1.0 and hosp_lo > 1.0
    no_band_shows_benefit = all(
        HOU2021_EICU[k][0] >= 1.0 or HOU2021_EICU[k][1] < 1.0 < HOU2021_EICU[k][2]
        for k in ("icu_mortality_vtc_10_15", "hosp_mortality_vtc_15_20")
    )
    with open(os.path.abspath(__file__)) as f:
        src = f.read()
    registration_present = ("EICU_NON_REPLICATION_REGISTERED = True" in src
                            and "does NOT replicate a mortality benefit" in src)
    ok = (harm_direction_matches and no_band_shows_benefit
          and registration_present and EICU_NON_REPLICATION_REGISTERED)
    print(f"I6_EICU_BOUNDARY_CONDITION supratherapeutic_harm={HOU2021_EICU['icu_mortality_vtc_gt20']} "
          f"matches={harm_direction_matches} "
          f"trough_targeting_benefit_replicated={not no_band_shows_benefit} "
          f"registered={registration_present} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_I7_no_overreach():
    """The contract itself must carry the scope guards."""
    with open(os.path.abspath(__file__)) as f:
        src = f.read()
    ok = ('not medical guidance' in src) and ('SYNTHETIC' in src or 'synthetic' in src)
    print(f"I7_NO_OVERREACH scope_guards_present={ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("MERCYFUL LEARNING x INDEPENDENT VANCOMYCIN TDM DATASETS — contract")
    print("=" * 70)
    results.append(("I1", check_I1_independence_from_mimic()))
    results.append(("I2", check_I2_efficacy_direction_match()))
    results.append(("I3", check_I3_toxicity_direction_match()))
    results.append(("I4", check_I4_scheduler_unchanged_still_selects_tdm()))
    results.append(("I5", check_I5_verify_gate_still_causal()))
    results.append(("I6", check_I6_eicu_boundary_condition()))
    results.append(("I7", check_I7_no_overreach()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"MERCYFUL_INDEPENDENT_TDM_VERDICT I_GREEN ({passed}/{total} clauses PASS)")
        print("MERCYFUL_INDEPENDENT_TDM_NOTE structural_correspondence_only; "
              "no_refit; synthetic_graph; no_patient_data; "
              "mortality_not_independently_replicated; not_medical_guidance")
        return 0
    print(f"MERCYFUL_INDEPENDENT_TDM_VERDICT I_AMBER ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
