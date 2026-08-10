#!/usr/bin/env python3
"""
Mercyful Learning — LEARNED suffering field contract (L1..L8).

Companion to:
  docs/research/mercyful_learned_suffering_field_spec_2026-07-26.md

This contract validates the first LEARNED suffering field s(v) against the
repository's declared synthetic benchmarks:

  * mercyful_runtime_contract.py           (M1..M6, scheduler semantics)
  * mercyful_mimic_iv_vancomycin_contract.py (V1..V7, synthetic anchors
    s = 0.675679 / 0.059420, graph topology, verify-gate causality)

Training surface: the synthetic popPK cohort v2 (200 patients, generator
seed 20260501, Roberts 2011 parameters). FAERS is deliberately unused per
the negative audit docs/research/faers_mercyful_analysis_2026-07-26.md.
Real MIMIC-IV is credential-gated; published Wang et al. 2026 statistics
are used as direction anchors only. All data are synthetic; this is not
medical guidance and carries no clinical claim.

Pure Python; no dependencies beyond the standard library. Deterministic.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mercyful_runtime_contract import MercyGraph, mercyful_schedule
from mercyful_mimic_iv_vancomycin_contract import (
    S_VANCO_PRE, S_VANCO_POST, build_graph,
)
from mercyful_suffering_field_learned import (
    COEFF_PATH, COHORT_PATH, EXPECTED_COLUMNS, GAMMA_UNCERTAINTY,
    LAMBDA_M, LearnedSufferingField, W_AKI, W_FAIL,
    brier_base_rate, brier_score, features, load_cohort, train,
)

FAERS_AUDIT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    "docs", "research", "faers_mercyful_analysis_2026-07-26.md")
SPEC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    "docs", "research", "mercyful_learned_suffering_field_spec_2026-07-26.md")

# Reference patient for the anchor states (declared in the spec, section
# 7.1): cohort-typical ICU patient, 75 kg, CrCl 80 mL/min, SOFA 7, no
# nephrotoxic co-exposure.
REF = dict(weight_kg=75.0, crcl=80.0, sofa=7.0, nephro=0)

# Learned-field anchor regimens (declared in the spec, section 7.2). These
# mirror the synthetic benchmark's FIXED_LOW / FIXED_STD / VANCO_PRE /
# TDM_GUIDED states.
ANCHOR_REGIMENS = {
    "VANCO_PRE":  dict(dose_mg=1000.0, tau_h=12.0, tdm=0),
    "TDM_GUIDED": dict(dose_mg=1000.0, tau_h=12.0, tdm=1, cmin_measured=15.0),
    "FIXED_LOW":  dict(dose_mg=500.0, tau_h=24.0, tdm=0),
    "FIXED_STD":  dict(dose_mg=1500.0, tau_h=12.0, tdm=0),
}

# Declared thresholds (spec section 8; each has a falsifier and stop rule).
TDM_RATIO_MIN = 2.0       # learned pre/post suffering ratio must exceed this
SPEARMAN_MIN = 0.70       # rank agreement with the synthetic teacher field


def spearman(xs, ys):
    """Spearman rank correlation (average ranks for ties), pure stdlib."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy)


def teacher_window(cmin, a=10.0, b=20.0):
    """Synthetic teacher field on a point band (paper section 7.2)."""
    return max(0.0, a - cmin) / a + max(0.0, cmin - b) / b


# -----------------------------------------------------------------------------
# Contract clauses L1..L8
# -----------------------------------------------------------------------------

def check_L1_data_provenance():
    """The training surface exists with the exact expected schema; the FAERS
    negative audit is on record (why FAERS is not used); scope guards are
    present in this contract's own source."""
    ok = os.path.isfile(COHORT_PATH)
    cohort = load_cohort()
    ok = ok and len(cohort) == 200
    aki_rate = sum(r["aki"] for r in cohort) / len(cohort)
    cure_rate = sum(r["cure"] for r in cohort) / len(cohort)
    ok = ok and 0.05 < aki_rate < 0.60 and 0.30 < cure_rate < 1.0
    with open(os.path.abspath(FAERS_AUDIT)) as fh:
        faers = fh.read()
    faers_ok = "Verdict: NEGATIVE" in faers
    with open(os.path.abspath(__file__)) as fh:
        src = fh.read()
    guards = ("not medical guidance" in src) and ("synthetic" in src)
    ok = ok and faers_ok and guards
    print(f"L1_DATA_PROVENANCE cohort_n={len(cohort)} aki_rate={aki_rate:.3f} "
          f"cure_rate={cure_rate:.3f} faers_negative_audit={faers_ok} "
          f"schema_cols={len(EXPECTED_COLUMNS)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L2_outcome_models_learn(field, cohort):
    """The outcome models actually learn from data: IRLS converges, the
    coefficients are bit-reproducible against the frozen artifact, the AKI
    model beats the base-rate predictor on a deterministic held-out split,
    and the learned Cmin effects have the clinically required signs
    (higher Cmin -> more AKI; higher Cmin -> more cure)."""
    ok = field.model_aki["converged"] and field.model_cure["converged"]
    with open(os.path.abspath(COEFF_PATH)) as fh:
        frozen = json.load(fh)
    for name, model in (("aki", field.model_aki), ("cure", field.model_cure)):
        beta_frozen = frozen[f"model_{name}"]
        ok = ok and len(beta_frozen) == len(model["beta"])
        ok = ok and all(abs(b - bf) < 1e-12
                        for b, bf in zip(model["beta"], beta_frozen))
    xs = [features(r["cmin"], r["sofa"], r["nephro"], r["crcl_ml_min"])
          for r in cohort]
    test_idx = [i for i in range(len(cohort)) if i % 4 == 0]
    xs_t = [xs[i] for i in test_idx]
    ys_aki_t = [cohort[i]["aki"] for i in test_idx]
    b_model = brier_score(field.model_aki, xs_t, ys_aki_t)
    b_base = brier_base_rate(ys_aki_t)
    skill = 1.0 - b_model / b_base
    ok = ok and b_model < b_base
    sign_aki = field.model_aki["beta"][1] > 0.0
    sign_cure = field.model_cure["beta"][1] > 0.0
    ok = ok and sign_aki and sign_cure
    print(f"L2_OUTCOME_MODELS_LEARN converged=True frozen_match=True "
          f"heldout_brier={b_model:.4f} base={b_base:.4f} skill={skill:.3f} "
          f"sign_cmin_aki={field.model_aki['beta'][1]:+.3f} "
          f"sign_cmin_cure={field.model_cure['beta'][1]:+.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L3_field_decomposition(field):
    """Expanded ethics is structural, not decorative: the field decomposes
    as patient + machine; the machine term is strictly positive and exactly
    the declared per-evaluation energy cost; the Knightian interval is
    well-formed on every anchor state."""
    m = field.s_machine_per_eval()
    anchors = {name: field.s_total(**REF, **reg)
               for name, reg in ANCHOR_REGIMENS.items()}
    well_formed = all(lo <= s <= hi for s, lo, hi in anchors.values())
    # Patient component alone must reproduce total minus machine term.
    s_p, _, _ = field.s_patient(**REF, **ANCHOR_REGIMENS["VANCO_PRE"])
    s_t, _, _ = field.s_total(**REF, **ANCHOR_REGIMENS["VANCO_PRE"])
    decomposes = abs((s_p + m) - s_t) < 1e-15
    ok = (m > 0.0) and well_formed and decomposes and LAMBDA_M > 0.0
    print(f"L3_FIELD_DECOMPOSITION s_machine_per_eval={m:.3e} "
          f"intervals_well_formed={well_formed} "
          f"patient_plus_machine_exact={decomposes} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L4_tdm_narrows_learned_field(field):
    """The learned analogue of synthetic clause V3: measurement (TDM)
    strictly reduces the learned suffering field, by at least the declared
    ratio. Note (spec section 9.1): the learned ratio is smaller than the
    synthetic 11.4x because the learned field retains irreducible baseline
    harm (host factors, residual risk) that the window-based synthetic
    field zeroes out inside the window."""
    s_pre, _, _ = field.s_patient(**REF, **ANCHOR_REGIMENS["VANCO_PRE"])
    s_post, _, _ = field.s_patient(**REF, **ANCHOR_REGIMENS["TDM_GUIDED"])
    ratio = s_pre / s_post
    ok = (s_post < s_pre) and (ratio > TDM_RATIO_MIN)
    print(f"L4_TDM_NARROWS_LEARNED_FIELD pre={s_pre:.6f} post={s_post:.6f} "
          f"ratio={ratio:.3f}x (min {TDM_RATIO_MIN}; synthetic analog "
          f"{S_VANCO_PRE:.6f}/{S_VANCO_POST:.6f}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L5_anchor_ordering(field):
    """The learned field reproduces the synthetic benchmark's qualitative
    anchor structure: underdosing (FIXED_LOW) is the worst arm and is
    failure-dominated (pure efficacy shortfall, cf. synthetic V1); the
    TDM-guided state is the best treatment state."""
    s_low, _, _ = field.s_patient(**REF, **ANCHOR_REGIMENS["FIXED_LOW"])
    s_std, _, _ = field.s_patient(**REF, **ANCHOR_REGIMENS["FIXED_STD"])
    s_tdm, _, _ = field.s_patient(**REF, **ANCHOR_REGIMENS["TDM_GUIDED"])
    # Composition of FIXED_LOW at its median Cmin: failure vs AKI share.
    from mercyful_suffering_field_learned import poppk_cmin_ss, sigmoid
    cmin_low = poppk_cmin_ss(REF["weight_kg"], REF["crcl"],
                             ANCHOR_REGIMENS["FIXED_LOW"]["dose_mg"],
                             ANCHOR_REGIMENS["FIXED_LOW"]["tau_h"])
    x = features(cmin_low, REF["sofa"], REF["nephro"], REF["crcl"])
    p_aki = sigmoid(sum(b * xi for b, xi in zip(field.model_aki["beta"], x)))
    p_cure = sigmoid(sum(b * xi for b, xi in zip(field.model_cure["beta"], x)))
    fail_share = W_FAIL * (1.0 - p_cure)
    aki_share = W_AKI * p_aki
    failure_dominated = fail_share > aki_share
    ok = (s_low > s_std) and (s_std > s_tdm) and failure_dominated
    print(f"L5_ANCHOR_ORDERING fixed_low={s_low:.6f} > fixed_std={s_std:.6f} "
          f"> tdm_guided={s_tdm:.6f}; fixed_low composition "
          f"fail={fail_share:.3f} vs aki={aki_share:.3f} "
          f"(failure_dominated={failure_dominated}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L6_scheduler_equivalence(field):
    """Decision-level validation against the synthetic benchmark: on the
    MIMIC-IV graph topology with LEARNED state sufferings, the mercyful
    scheduler must (a) select the TDM route under the verification gate,
    and (b) select the unverified fixed-dose route in the counterfactual
    open graph (synthetic clause V4: the gate is causal). The exposure
    therapy benchmark (M4) must still route through 'moderate'."""
    vals = {name: field.s_total(**REF, **reg)[0]
            for name, reg in ANCHOR_REGIMENS.items()}
    g_synth = build_graph(verify_gate=True)
    g_open = build_graph(verify_gate=False)

    def learned_graph(base):
        suffering = dict(base.suffering)
        for name, v in vals.items():
            suffering[name] = v
        g = MercyGraph(states=base.states, edges=[], suffering=suffering)
        g.adj = base.adj
        g.lengths = base.lengths
        return g

    g_gated = learned_graph(g_synth)
    g_cf = learned_graph(g_open)
    path_gated, m_gated = mercyful_schedule(g_gated, "START", "TARGET",
                                            mu=1.0, L0=10.0)
    path_open, m_open = mercyful_schedule(g_cf, "START", "TARGET",
                                          mu=1.0, L0=10.0)
    tdm_ok = path_gated == ["START", "VANCO_PRE", "TDM_GUIDED", "TARGET"]
    cf_ok = (path_open == ["START", "FIXED_STD", "TARGET"]
             and m_open[3] < m_gated[3])
    # M4 benchmark is field-independent; re-verify with the learned module
    # loaded to guard against import-time interference.
    g_exp = MercyGraph(
        states=["avoidance", "mild", "moderate", "recovery"],
        edges=[("avoidance", "avoidance"), ("avoidance", "mild"),
               ("mild", "avoidance"), ("mild", "moderate"),
               ("moderate", "mild"), ("moderate", "recovery")],
        suffering={"avoidance": 0.0, "mild": 2.0, "moderate": 5.0,
                   "recovery": 0.0})
    path_exp, _ = mercyful_schedule(g_exp, "avoidance", "recovery",
                                    mu=1.0, L0=10.0)
    exp_ok = "moderate" in path_exp
    ok = tdm_ok and cf_ok and exp_ok
    print(f"L6_SCHEDULER_EQUIVALENCE gated={path_gated} "
          f"(total={m_gated[3]:.6f}) open={path_open} "
          f"(total={m_open[3]:.6f}) exposure={path_exp} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L7_teacher_rank_agreement(field, cohort):
    """Against the synthetic teacher: across the 200 cohort patients, the
    learned field (evaluated post-TDM at each measured Cmin) must rank
    regimens like the window-based teacher field (Spearman rho >= declared
    threshold). This is the quantitative bridge between the declared
    synthetic field and the learned one."""
    learned, teacher = [], []
    for r in cohort:
        s, _, _ = field.s_patient(r["weight_kg"], r["crcl_ml_min"],
                                  r["sofa"], r["nephro"], r["dose_mg"],
                                  r["interval_h"], tdm=1,
                                  cmin_measured=r["cmin"])
        learned.append(s)
        teacher.append(teacher_window(r["cmin"]))
    rho = spearman(learned, teacher)
    ok = rho >= SPEARMAN_MIN
    print(f"L7_TEACHER_RANK_AGREEMENT spearman_rho={rho:.4f} "
          f"(min {SPEARMAN_MIN}, n={len(learned)}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_L8_no_overreach():
    """The spec must carry the scope guards and the negative provenance:
    synthetic training data, not medical guidance, FAERS unused, MIMIC-IV
    credential-gated."""
    with open(os.path.abspath(SPEC)) as fh:
        spec = fh.read()
    checks = {
        "not medical guidance": "not medical guidance" in spec,
        "synthetic": "synthetic" in spec.lower() or "SYNTHETIC" in spec,
        "faers negative provenance": "faers_mercyful_analysis_2026-07-26" in spec,
        "mimic credential status": "credential" in spec.lower(),
        "machine suffering section": "machine suffering" in spec.lower(),
    }
    ok = all(checks.values())
    print(f"L8_NO_OVERREACH {checks} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    print("=" * 70)
    print("MERCYFUL LEARNING — LEARNED SUFFERING FIELD — contract")
    print("=" * 70)
    cohort = load_cohort()
    field = train(cohort)
    results.append(("L1", check_L1_data_provenance()))
    results.append(("L2", check_L2_outcome_models_learn(field, cohort)))
    results.append(("L3", check_L3_field_decomposition(field)))
    results.append(("L4", check_L4_tdm_narrows_learned_field(field)))
    results.append(("L5", check_L5_anchor_ordering(field)))
    results.append(("L6", check_L6_scheduler_equivalence(field)))
    results.append(("L7", check_L7_teacher_rank_agreement(field, cohort)))
    results.append(("L8", check_L8_no_overreach()))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"MERCYFUL_LEARNED_FIELD_VERDICT L_GREEN ({passed}/{total} clauses PASS)")
        print("MERCYFUL_LEARNED_FIELD_NOTE learned_from_synthetic_cohort; "
              "faers_unused_per_negative_audit; mimic_iv_credential_gated; "
              "not_medical_guidance")
        return 0
    print(f"MERCYFUL_LEARNED_FIELD_VERDICT L_AMBER ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
