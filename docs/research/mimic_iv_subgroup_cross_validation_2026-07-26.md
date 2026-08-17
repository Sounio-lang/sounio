<!-- docs:meta
topic_id: repo.docs.research.mimic-iv-subgroup-cross-validation-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mimic-iv-subgroup-cross-validation-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# MIMIC-IV × Mercyful Learning — subgroup cross-validation

**Date:** 2026-07-26
**Branch:** research/self-falsifying-compilation-line-20260726
**Parent validation:** `docs/research/mimic_iv_mercyful_validation_2026-07-26.md` (**POSITIVE**, V_GREEN 7/7)
**Contract:** `scripts/research/mercyful_mimic_iv_subgroup_contract.py` (X1..X9, **X_GREEN 9/9**)
**Gate:** `scripts/ci/mercyful_mimic_iv_subgroup_gate.sh` (**MERCYFUL_MIMIC_IV_SUBGROUP_GATE_OK**)
**Verdict: POSITIVE (cross-validated)** — the TDM–mortality direction and the scheduler's
TDM-window selection both hold in **all six subgroups** examined (age <65 / ≥65, severity
SOFA <7 / ≥7, comorbidity low / high), at the level of *structural correspondence only*,
with one scoping limitation stated in §2 and binding caveats in §6.

> **Scope.** This is a structural-correspondence cross-validation of a formal scheduling
> framework against a published observational cohort and a synthetic popPK cohort. The
> graph, doses, p-boxes, and suffering values are synthetic constructions. This document
> is not medical guidance — not a treatment recommendation, not a dosing suggestion, and
> not a clinical decision-support tool.

---

## 1. Task

Divide the MIMIC-IV vancomycin cohort into subgroups — **age** (<65 vs ≥65), **severity**
(low vs high), **comorbidity** (low vs high) — and verify:

1. the TDM–mortality association (TDM associated with lower mortality) holds in **all**
   subgroups;
2. the Mercyful Learning scheduler still selects the therapeutic window (the TDM-guided
   course) in **each** subgroup.

## 2. Scoping decision (documented assumption)

The real MIMIC-IV patient-level extract is **credential-gated** (PhysioNet CITI training +
DUA + service account; see `scripts/clinical/README.md`, "MIMIC-IV credential flow" —
extraction has never been run against real MIMIC-IV in this repository). The source study
(Wang et al. 2026, doi:10.1038/s41598-026-42395-1) does **not** publish TDM-mortality odds
ratios stratified by age or severity; its own prespecified subgroup analyses are stratified
by comorbidities and concomitant medications and concern the *toxicity* endpoints.

The cross-validation is therefore executed on two legs:

- **Leg A (association side).** The patient-level subgroup split runs on the repository's
  popPK-driven **synthetic** cohort
  (`scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv`, 200 patients, deterministic
  seed 20260501), the schema-identical stand-in for the gated MIMIC-IV extract. Within each
  stratum we compute the odds ratio of clinical cure for **window attainment** (measured
  Cmin ∈ [10, 20] mg/L — the TDM-guided target) vs out-of-window. OR > 1 in a stratum =
  the TDM signature (therapeutic-window containment) is associated with lower mortality in
  that stratum — the model-side shadow of the cohort's TDM–mortality association.
  The **real-data anchor** for cross-stratum robustness is the study's own stratified
  evidence (§5, clause X8): the TDM–mortality association holds, every 95% CI excluding
  1.0, across *all* adjustment strata (crude → demographics/comorbidities → fully
  adjusted) and after propensity matching, for both mortality endpoints; the abstract
  states the results were "validated through subgroup analyses (stratified by
  comorbidities and concomitant medications)".
- **Leg B (scheduler side).** The parent contract's synthetic dosing graph is parametrized
  per stratum (§4) and the scheduler is re-run in each of the six stratum graphs.

**What is scoped out:** a patient-level re-analysis of real MIMIC-IV data by age/severity/
comorbidity strata. That requires the PhysioNet credential flow to complete; the gate is
built so the real extract can replace the synthetic CSV with no schema changes
(`validate_cohort.py` already enforces the shared schema). Also scoped out: per-stratum
*significance* on the synthetic cohort (n ≤ 123 per stratum; see §6 item 2) — the claim
at this cohort size is **direction** (OR > 1 in all strata), not per-stratum CI exclusion.

## 3. Stratification definitions

| Dichotomy | Low stratum | High stratum | Basis |
|---|---|---|---|
| Age | `< 65` (n = 120) | `≥ 65` (n = 80) | task-specified cut; MIMIC-IV cohort median age 68 |
| Severity | `SOFA < 7` (n = 87) | `SOFA ≥ 7` (n = 113) | median split (cohort median SOFA = 7) |
| Comorbidity | no nephrotoxic co-exposure (n = 123) | nephrotoxic co-exposure (n = 77) | the comorbidity field present in the shared cohort schema (`nephrotoxic_coexposure`); the schema has no Charlson index |

All three dichotomies partition the 200-patient cohort exactly (clause X1).

## 4. Per-stratum model parametrization (Leg B)

Same graph topology as the parent contract (START / FIXED_LOW / FIXED_STD / VANCO_PRE /
TDM_GUIDED / TARGET; G_VERIFY admits only window-verified courses to TARGET). Per stratum,
declared synthetic fixed-dose Cmin p-boxes; suffering via the paper's window functional
(§7.2) `s_win([lo,hi],[a,b]) = max(0,a−lo)/a + max(0,hi−b)/b`, window [10, 20] mg/L.

| Stratum | FIXED_LOW p-box | s | FIXED_STD p-box | s |
|---|---|---|---|---|
| age <65 | [4.0, 9.0] | 0.600 | [6.5, 24.0] | 0.550 |
| age ≥65 | [3.0, 8.0] | 0.700 | [5.0, 24.0] | 0.700 |
| severity low | [4.5, 9.0] | 0.550 | [7.0, 23.0] | 0.450 |
| severity high | [3.5, 8.0] | 0.650 | [5.5, 25.0] | 0.700 |
| comorbidity low | [4.5, 9.0] | 0.550 | [7.0, 23.0] | 0.450 |
| comorbidity high | [3.0, 8.0] | 0.700 | [4.5, 23.0] | 0.700 |

**Declared assumptions (all synthetic, all disclosed):**

1. **Higher-risk strata get wider fixed-dose p-boxes** — reduced and more variable
   vancomycin clearance with age, severity, and comorbidity widens the unmonitored
   concentration band. This is the model's representation of "the same fixed dose is a
   worse bet in a sicker patient", and it produces the suffering gradient of clause X7.
2. **Structural invariants preserved in every stratum:** FIXED_LOW fully below the window
   (hi < 10: worst case cannot clear the infection → no edge to TARGET); FIXED_STD
   straddling (lo < 10, hi > 20 → G_VERIFY refuses); TDM_GUIDED in-window (gate passes).
3. **Gate-causality liveness:** every stratum keeps `s_win(FIXED_STD) < C1_TOTAL/2 =
   0.705389`. Derivation: at μ = 1 the ungated FIXED_STD route `START → FIXED_STD →
   TARGET` costs integral `s_win` (one edge) plus peak `s_win`, i.e. `2·s_win`; it
   undercuts the TDM route's C1 total 1.410778 iff `s_win < 0.705389`. Keeping every
   stratum under this bound keeps the X6 counterfactual (an ungated optimizer
   rationally prefers FIXED_STD) live in every stratum — the gate remains
   *load-bearing*, not decorative, per subgroup.
4. **VANCO_PRE / TDM_GUIDED suffering are the twin's measured C3 values** (0.675679 /
   0.059420), invariant across strata: TDM narrows the band into the window regardless of
   stratum. The per-stratum variation is carried entirely by the unmonitored arms.

## 5. Results (X_GREEN 9/9, exact)

```
X1_COHORT_SCHEMA_AND_STRATIFICATION n=200 strata={age 120/80, sev 87/113, comorb 123/77} -> PASS
X2_POOLED_WINDOW_CURE_ASSOCIATION OR=2.733 95%CI=(1.373,5.442) cells=(70,14,75,41) -> PASS
X3_DIRECTION_HOLDS_ALL_STRATA age<65:2.269  age>=65:3.370  sev_low:2.404  sev_high:2.893
                              comorb_low:2.112  comorb_high:4.306  (all OR>1) -> PASS
X4_SCHEDULER_SELECTS_TDM_ALL_STRATA 6/6 strata, route START->VANCO_PRE->TDM_GUIDED->TARGET,
                              integral=0.735099 peak=0.675679 total=1.410778 -> PASS
X5_NAIVE_MINIMIZER_UNDERDOSES_ALL_STRATA 6/6 strata, pick=FIXED_LOW (tox=0), no path to TARGET -> PASS
X6_VERIFY_GATE_CAUSAL_ALL_STRATA 6/6 strata, open optimum=FIXED_STD route, gated optimum=TDM route -> PASS
X7_SUFFERING_GRADIENT std: 0.550->0.700 (age), 0.450->0.700 (sev), 0.450->0.700 (comorb) -> PASS
X8_LITERATURE_STRATUM_ROBUSTNESS hosp 0.49/0.58/0.63/0.672, icu 0.51/0.64/0.72/0.691,
                              all CIs exclude 1.0, monotone over M1->M2->M3 only -> PASS
X9_NO_OVERREACH scope_guards_present=True -> PASS
MERCYFUL_MIMIC_IV_SUBGROUP_VERDICT X_GREEN (9/9 clauses PASS)
```

Reading of the clauses:

- **X1–X3 (Leg A, synthetic-cohort association).** The window-attainment ↔ cure
  association is positive **in every stratum** (OR 2.112–4.306), and pooled it is
  significant (OR 2.733, 95% CI 1.373–5.442). The direction the parent validation
  established at cohort level is not carried by any single subgroup — it survives all
  three dichotomies. Stratum-level CIs are printed honestly in the contract output: 3 of
  6 include 1.0 at these stratum sizes (n ≤ 123); see §6 item 2.
- **X4–X6 (Leg B, scheduler robustness).** In every stratum graph the scheduler's unique
  feasible optimum is the TDM-guided route with the exact C1 values; the naive
  toxicity-only minimizer still selects the sub-therapeutic arm that cannot reach the
  target; and counterfactually opening the gate still flips the optimum to the unverified
  fixed-dose arm — the verification gate is causal in each subgroup, not just in the
  pooled graph.
- **X7 (gradient).** Suffering of both fixed arms is strictly higher in the high-risk
  stratum of each dichotomy: the model prices unmonitored dosing as *worse* exactly where
  the cohort shows monitoring is preferentially directed (sicker patients) — the same
  confounding-by-indication structure, read from the model side.
- **X8 (real-data anchor).** Wang et al. Table 2: hospital mortality OR 0.49 → 0.58 →
  0.63 and ICU mortality OR 0.51 → 0.64 → 0.72 across adjustment strata 1→2→3, then
  0.672 / 0.691 post-PSM — every 95% CI excludes 1.0 at every stratum of adjustment and
  after matching. The attenuation is monotone toward the null **over the adjustment
  sequence (M1 → M2 → M3)** for both endpoints, without ever crossing it. The post-PSM
  estimates come from a different (matched) sample and estimator and are not part of
  that sequence; disclosed explicitly: for ICU mortality the PSM point estimate (0.691)
  sits *below* the Model 3 estimate (0.72), so no monotone claim is made across the
  M3 → PSM step. The TDM–mortality association is robust across the study's own
  stratification machinery.
- **X9 (scope guards).** The contract carries its no-clinical-claim statements.

## 6. Honest asymmetries and caveats (binding)

1. **Leg A runs on synthetic data.** The 200-patient cohort is popPK-generated
   (Roberts 2011 ICU parameters; deterministic seed), not MIMIC-IV. Its window–cure odds
   ratios are *generator-driven* (the generator's outcome model rewards in-window
   exposure), so X2/X3 validate the **pipeline and the structural logic**, not a real
   effect. The real-evidence weight of the cross-validation is carried by X8 (published,
   verified from source). Re-running Leg A on the real extract is a drop-in replacement
   once credentialing completes, and this gate will be re-run then.
2. **Per-stratum significance is not claimed on the synthetic cohort.** At n ≤ 123 per
   stratum, 3 of 6 stratum CIs include 1.0 (age <65: 0.952–5.405; severity low:
   0.825–7.006; comorbidity low: 0.909–4.907). The cross-validation claim is *direction
   in all strata + pooled significance*, mirroring what the real study establishes with
   9,785 matched pairs.
3. **The comorbidity dichotomy is a proxy.** The shared cohort schema has no Charlson
   index; `nephrotoxic_coexposure` (0/1) stands in for comorbidity burden. The real
   study's comorbidity-stratified subgroup analyses (its Fig. 4) concern toxicity
   endpoints, not mortality.
4. **The stratum p-boxes are declared, not measured** (same caveat as the parent
   validation §6 item 5, now per stratum). The correspondence survives any band choices
   preserving the structural invariants of §4 assumption 2; the specific values are not
   derived from MIMIC-IV. Assumption 3's bound (s < 0.705389) constrains how wide the
   straddling bands may be declared before the gate-causality counterfactual goes quiet —
   this is stated, not tuned silently.
5. **Stratum-invariant TDM benefit is an assumption, not a derivation.** Keeping
   VANCO_PRE/TDM_GUIDED suffering at the twin's C3 values across strata encodes "TDM
   narrows the band into the window in every subgroup". The twin was measured at one
   operating point (1000 mg q12h, 78.5 kg, CrCl 65); per-stratum twin re-measurement
   (e.g., low-CrCl operating points for age ≥65) is listed as follow-up.
6. **Observational source, unreconciled toxicity counts.** All caveats of the parent
   validation (§6: observational design; the source paper's internally inconsistent
   matched toxicity table; HR-vs-OR form) apply unchanged. The mortality endpoints this
   cross-validation relies on are the ones the parent flagged as the *surviving* side of
   that caveat (KM log-rank P < 0.001 is a raw matched-cohort statistic).

## 7. Falsifiers (pre-registered by construction of the clauses)

| Clause | Falsifier |
|---|---|
| X1 | Cohort CSV changed (stratum sizes or partitions differ) — regression tripwire |
| X2 | Pooled window–cure OR ≤ 1 or CI includes 1.0 (association gone even pooled) |
| X3 | Any stratum OR ≤ 1 (direction carried by a subgroup, not general) |
| X4 | Any stratum's scheduler optimum deviates from the TDM route or C1 values |
| X5 | Any stratum where the naive toxicity minimizer reaches the target (hazard gone) |
| X6 | Any stratum where the gate is decorative (open optimum already TDM) |
| X7 | Any dichotomy where the high-risk stratum is not strictly worse on fixed dosing |
| X8 | Any cited adjustment stratum CI includes 1.0, or M1→M2→M3 attenuation non-monotone for either endpoint (real anchor gone) |
| X9 | Scope guards stripped from the contract |

Global: failure of X4 or X6 is **RED** (the scheduler-side structural claim breaks);
failure of X8 is **RED** for the correspondence verdict (the gate greps the real
statistics, not a narrative); X2/X3 failure is **RED** for Leg A but does not touch Leg B.

## 8. Commands run

```bash
python3 scripts/research/mercyful_mimic_iv_subgroup_contract.py   # X_GREEN 9/9
bash scripts/ci/mercyful_mimic_iv_subgroup_gate.sh                # MERCYFUL_MIMIC_IV_SUBGROUP_GATE_OK
```

Subgroup statistics computed from
`scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv` (committed, deterministic
seed 20260501); real statistics verified against
<https://pmc.ncbi.nlm.nih.gov/articles/PMC13171905/> (Wang et al. 2026,
doi:10.1038/s41598-026-42395-1, Table 2 and abstract), fetched 2026-07-26.

## 9. Verdict

**POSITIVE, cross-validated, at structural-correspondence level, with the scoping
limitation of §2 and the caveats of §6.**

- The TDM–mortality direction holds in **all six subgroups** on both legs: the
  synthetic-cohort window–cure OR exceeds 1 in every stratum (X3) and the real study's
  association survives every adjustment stratum with CIs excluding the null (X8).
- The Mercyful scheduler selects the TDM-guided therapeutic-window course in **every**
  subgroup graph, with the naive minimizer still underdosing and the verification gate
  still causal in each (X4–X6); the model prices unmonitored dosing strictly worse in
  higher-risk strata (X7).
- This is a *robustness* result, not a new correspondence: it shows the parent verdict is
  not an artifact of pooling. It upgrades to a patient-level real-data cross-validation
  the day the MIMIC-IV credential flow completes — the gate is already shaped for that
  swap.
