<!-- docs:meta
topic_id: repo.docs.research.independent-dataset-vancomycin-tdm-validation-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.independent-dataset-vancomycin-tdm-validation-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Independent-dataset validation — vancomycin TDM × Mercyful Learning

**Date:** 2026-07-26
**Branch:** research/self-falsifying-compilation-line-20260726
**Predecessor:** `docs/research/mimic_iv_mercyful_validation_2026-07-26.md` (V_GREEN 7/7)
**Contract:** `scripts/research/mercyful_independent_tdm_contract.py` (I1..I7, **I_GREEN 7/7**)
**Gate:** `scripts/ci/mercyful_independent_tdm_gate.sh` (**MERCYFUL_INDEPENDENT_TDM_GATE_OK**)
**Verdict: POSITIVE, with one registered boundary condition.** Independent evidence
disjoint from MIMIC-IV — a six-study meta-analysis (521 patients, four countries,
1990–2010, including one RCT) and an Australian AUC-TDM implementation study —
associates TDM with higher clinical efficacy (OR 2.62, 95% CI 1.34–5.11) and lower
nephrotoxicity (OR 0.25, 95% CI 0.13–0.48). The Mercyful scheduler, on the **same
fixed synthetic graph** used for the MIMIC-IV comparison (no parameter refitted),
still makes the TDM-guided window-verified course its unique feasible optimum.
The eICU-CRD trough-level study (3,603 patients, 208 hospitals) does **not**
replicate a mortality benefit of trough targeting; it is registered as a binding
boundary condition (§5, clause I6), not rounded off.

> **Scope.** Structural-correspondence analysis of a formal scheduling framework
> against published cohort statistics. The graph, doses, p-boxes, and suffering
> values are synthetic constructions, identical to the MIMIC-IV contract. This is
> not medical guidance, not a treatment recommendation, not a dosing suggestion,
> and not a clinical decision-support tool.

---

## 1. Independence criterion

A source counts as **independent** of the MIMIC-IV anchor (Wang et al. 2026,
doi:10.1038/s41598-026-42395-1; Beth Israel Deaconess ICU/ED, 2008–2022) iff its
underlying patient data come from a disjoint set of hospitals or a disjoint
sampling frame. Published replications drawn from MIMIC-IV itself strengthen the
anchor result but do not count. Clause I1 enforces this metadata check.

**Counted as independent:**

| Source | Data | Patients | Era | Contrast |
|---|---|---|---|---|
| Ye, Tang & Zhai 2013, *PLoS One* 8(10):e77169, doi:10.1371/journal.pone.0077169 (PMID 24204764; PMCID PMC3799644) | 6 studies (1 RCT + 5 cohorts): USA, Spain, Japan ×3, China | 521 (249 TDM / 272 non-TDM) | 1990–2010 | TDM vs non-TDM |
| Yang et al. 2024, *J Clin Pharmacol* 64(1):19–29, doi:10.1002/jcph.2363 (PMID 37779493) | Single Australian tertiary hospital, before/after AUC-TDM advisory service | 971 courses / 781 patients (764 pre / 207 post) | ~2015–2019 | TDM service vs none |
| Hou et al. 2021, *Front Pharmacol* 12:690157, doi:10.3389/fphar.2021.690157 | eICU-CRD v2.0, 335 ICUs at 208 US hospitals | 3,603 (all monitored) | 2014–2015 | mean trough level vs mortality (boundary condition, §5) |

**Excluded as non-independent (MIMIC-IV replications, same sampling frame):**

- Peng et al. 2024, *Front Med (Lausanne)* (PMID 39726684; PMCID PMC11669523):
  MIMIC-IV sepsis subset, 14,053 patients; TDM associated with reduced 30-day
  mortality (adjusted HR 0.66, 95% CI 0.61–0.71; post-PSM HR 0.71, 0.66–0.77).
- Peng et al. 2026 (PMCID PMC12819319): MIMIC-IV v3.1 RRT subset, 2,085 patients;
  TDM associated with reduced 30-day mortality (HR 0.457–0.478 across adjusted
  models, all P < 0.001).

Both are direction-concordant with the anchor but reuse MIMIC-IV; they are
recorded in the contract's `EXCLUDED_NON_INDEPENDENT` list so the gate can detect
any future attempt to cite them as independent.

## 2. Independent evidence (verified from source, 2026-07-26)

### 2.1 Ye, Tang & Zhai 2013 — meta-analysis of six independent hospital studies

Fetched from <https://pmc.ncbi.nlm.nih.gov/articles/PMC3799644/>.

- **Design.** Systematic review (Medline, Embase, Web of Science, Cochrane, CNKI,
  CBM to 2013-03-29). 1 RCT (Fernandez de Gatta 1996, Spain) + 5 cohort studies
  (Welty 1994 USA; Iwamoto 2003, Sato 2007, Mochizuki 2010 Japan; Huang 2011
  China). Total 521 patients: 249 TDM, 272 non-TDM. Mantel–Haenszel fixed effect;
  Newcastle–Ottawa appraisal; no detected publication bias (Begg/Egger NS).
- **Clinical efficacy (primary).** TDM vs non-TDM: **OR 2.62 (95% CI 1.34–5.11),
  P = 0.005**. Cohort subgroup OR 3.04 (1.34–6.90); RCT alone 1.94 (0.61–6.20,
  NS, n = 70). I² = 0%.
- **Nephrotoxicity (secondary).** **OR 0.25 (95% CI 0.13–0.48), P < 0.0001**.
  Consistent in cohort (0.27, 0.12–0.58) and RCT (0.21, 0.07–0.68) subgroups and
  in Asian (0.30, 0.11–0.84) and non-Asian (0.22, 0.10–0.51) subgroups. I² = 0%.
- **Mortality.** Reported by one study only (Fernandez de Gatta 1996): 2/37 TDM
  vs 6/33 non-TDM deaths — direction favors TDM, underpowered; the meta-analysis
  makes no mortality claim.
- **Duration of therapy / length of stay.** No significant difference
  (WMD −0.40, −2.83–2.02; WMD −1.01, −7.51–5.49).

This is the strongest independent evidence: the TDM-vs-non-TDM contrast — the
exact decision structure the scheduler prices — tested on patients drawn from
hospitals on three continents, none of them Beth Israel, in an era disjoint from
MIMIC-IV, with one randomized trial included and zero measured heterogeneity.

### 2.2 Yang et al. 2024 — AUC-TDM advisory service, Australian hospital

Fetched from <https://pubmed.ncbi.nlm.nih.gov/37779493/>.

- 4-year retrospective before/after study of IV vancomycin > 48 h in adults:
  971 courses (764 pre-service / 207 post-service), 781 patients.
- Vancomycin-associated nephrotoxicity (KDIGO-like creatinine definition):
  **15% pre vs 10% post, P = 0.075** — a 5% absolute reduction in the direction
  the framework predicts, not statistically significant at this sample size.
- Independent VAN risk factors: sepsis, heart failure, solid-organ transplant,
  concomitant piperacillin–tazobactam, and average vancomycin AUC during therapy
  (the exposure variable the window functional prices).

Registered as **direction-concordant, underpowered**: it supports the toxicity
side of the correspondence but cannot carry significance weight.

### 2.3 Hou et al. 2021 — eICU-CRD trough levels (boundary condition, §5)

Fetched from <https://pmc.ncbi.nlm.nih.gov/articles/PMC8326564/>. Full analysis in
§5: this study does not test TDM-vs-no-TDM and does not replicate a mortality
benefit of trough targeting. It is included in the contract as a *boundary
condition* clause (I6), not as supporting evidence.

## 3. Scheduler validation on the independent evidence

The validation logic: the scheduler and its graph are **fixed** — literally the
same Python object, imported unchanged from
`scripts/research/mercyful_mimic_iv_vancomycin_contract.py` (`build_graph`),
which was frozen before the MIMIC-IV comparison. No parameter is refitted to any
of the independent sources. The question asked is whether the *same* decision
structure corresponds to data the model has never been near.

Contract clauses (`scripts/research/mercyful_independent_tdm_contract.py`):

```
I1_INDEPENDENCE_FROM_MIMIC sources=3 disjoint=True excluded_mimic_replications=2 ye2013_n=521 (tdm=249) yang2024_courses=971 hou2021_eicu_n=3603 -> PASS
I2_EFFICACY_DIRECTION_MATCH ye_efficacy=(2.62,1.34,5.11) model_subtherapeutic_unreachable=True -> PASS
I3_TOXICITY_DIRECTION_MATCH ye_nephrotox=(0.25,0.13,0.48) model_prices_supratherapeutic=0.3 -> PASS
I4_SCHEDULER_UNCHANGED_STILL_SELECTS_TDM integral=0.735099 peak=0.675679 total=1.410778 -> PASS
I5_VERIFY_GATE_STILL_CAUSAL open_integral=0.700000 gated_integral=0.735099 -> PASS
I6_EICU_BOUNDARY_CONDITION supratherapeutic_harm=(2.428,1.385,4.258) matches=True
                          trough_targeting_benefit_replicated=False registered=True -> PASS
I7_NO_OVERREACH scope_guards_present=True -> PASS
MERCYFUL_INDEPENDENT_TDM_VERDICT I_GREEN (7/7 clauses PASS)
```

- **I1 (independence).** Every counted source's dataset metadata is disjoint from
  MIMIC-IV; both MIMIC-IV replications (Peng 2024, Peng 2026) are on the
  exclusion list.
- **I2 (efficacy side).** Ye OR 2.62 with CI excluding 1.0: monitored dosing
  achieves clinical efficacy more often — the real-world counterpart of the
  model's lower window bound (the sub-therapeutic `FIXED_LOW` arm, worst-case
  band below 10 mg/L, has no edge to `TARGET`: unchecked treatment failure).
- **I3 (toxicity side).** Ye OR 0.25 with CI excluding 1.0: monitored dosing
  cuts nephrotoxicity fourfold — the counterpart of the model's upper window
  bound (the supratherapeutic violation component of `FIXED_STD` = 0.3).
  Vancomycin remains the framework's cleanest domain precisely because the
  independent evidence now matches **both** sides of the suffering field, not
  just the mortality direction matched by MIMIC-IV.
- **I4 (no refit).** The unchanged scheduler reproduces the exact canonical
  values (∫s = 0.735099, peak 0.675679, total 1.410778 at μ = 1) and selects the
  TDM-guided route. The graph's identity with the MIMIC-IV contract is enforced
  by import, not by copied constants.
- **I5 (gate causality).** The counterfactual still holds: admit unverified fixed
  dosing and the scheduler rationally chooses the non-TDM arm (∫ = 0.700 <
  0.735099). The benefit of TDM remains invisible to any metric that does not
  constrain courses to be verified — now supported by two disjoint evidence bases.
- **I6 (boundary condition).** See §5.
- **I7 (scope guards).** Contract and this report carry the no-clinical-claim
  statements.

## 4. Structural mapping (independent evidence)

| Model object (synthetic, fixed) | Independent counterpart |
|---|---|
| `FIXED_LOW`: sub-therapeutic arm, no path to `TARGET` | Ye 2013: non-TDM dosing achieves clinical efficacy less often; TDM OR 2.62 (1.34–5.11) for efficacy |
| `FIXED_STD`: band straddles window, supratherapeutic component 0.3 | Ye 2013: non-TDM dosing nephrotoxic 4× as often; TDM OR 0.25 (0.13–0.48) against nephrotoxicity; Yang 2024: VAN 15%→10% after AUC-TDM service (P = 0.075) |
| G_VERIFY: only window-verified courses reach `TARGET` | The TDM arm in every counted study: dose adjusted to measured concentration / AUC advisory |
| Mercyful scheduler's unique feasible optimum is the TDM route (I4) | Direction of the pooled independent evidence favors monitored, window-guided dosing on both window bounds |
| (boundary) supratherapeutic exposure priced as suffering | Hou 2021 (eICU): mean VTC > 20 mg/L associated with higher ICU mortality OR 2.428 (1.385–4.258) and hospital mortality OR 1.585 (1.053–2.387) vs < 10 mg/L |

## 5. The eICU boundary condition (binding)

Hou et al. 2021 (eICU-CRD v2.0; 3,603 adults with ≥2 trough records at 335 ICUs /
208 hospitals) is the one large independent dataset that speaks to vancomycin
monitoring, and it is **not** a clean replication:

1. **It does not test the framework's contrast.** All 3,603 patients were
   monitored (inclusion required ≥2 VTC records); there is no non-TDM arm. It
   compares *mean trough level bands* (<10, 10–15, 15–20, >20 mg/L), not
   monitored vs unmonitored dosing. The framework's claim is about the decision
   structure of verifying against a window, not about which trough level is best.
2. **It does not replicate a benefit of trough targeting.** Adjusted for
   covariates, no band showed significantly *reduced* mortality vs < 10 mg/L
   (10–15: ICU OR 1.705, 0.975–2.981; hospital 1.235, 0.829–1.841; 15–20
   hospital: 1.370, 0.924–2.029). The authors conclude "VTC monitoring might not
   guarantee vancomycin efficacy." If the framework were read as "higher trough
   into the 15–20 band is better," eICU would falsify that reading. The framework
   does not make that reading — its window upper bound prices exactly the harm
   eICU measures — but the non-replication is registered, not explained away.
3. **What it does support is the model's toxicity side.** Mean VTC > 20 mg/L was
   associated with significantly *higher* ICU (OR 2.428, 1.385–4.258) and
   hospital (OR 1.585, 1.053–2.387) mortality vs < 10 mg/L, monotonically
   increasing across bands (raw ICU mortality 4.6 / 8.2 / 11.6 / 15.4%).
   Supratherapeutic exposure is harmful in the independent data, in the same
   direction the suffering functional prices it.
4. **Residual confounding is severe and documented here.** Mean VTC is
   mechanistically downstream of renal clearance: median creatinine clearance
   falls monotonically across the bands (129.7 → 109.8 → 91.9 → 75.2 ml/min), and
   renal failure drives ICU mortality. The level–mortality gradient is therefore
   not evidence that high troughs *cause* the excess deaths; equally, adjustment
   cannot rescue a benefit that is not there. We record both directions of that
   uncertainty.

Clause I6 passes only if (a) the supratherapeutic-harm direction matches the
model, and (b) the non-replication of trough-targeting benefit is explicitly
registered in the contract source. Deleting the registration fails the gate.

## 6. Honest asymmetries and caveats (binding)

1. **The independent evidence is older and smaller than the anchor.** Ye 2013
   pools 521 patients from 1990–2010-era practice (trough-era, pre-AUC-consensus
   dosing); MIMIC-IV is 28,451 patients from 2008–2022. The correspondence is
   directional on both sides; era effects (changing targets, assays, supportive
   care) are uncontrolled.
2. **Mortality is not independently replicated.** The Ye meta-analysis had one
   mortality report (2/37 vs 6/33, NS); Yang 2024 did not power mortality; Hou
   2021 has no unmonitored arm. The independent support covers *clinical
   efficacy* and *nephrotoxicity* — the two window bounds — while the mortality
   direction rests on MIMIC-IV alone (three concordant MIMIC-IV analyses:
   Wang 2026, Peng 2024, Peng 2026).
3. **Observational dominance.** Five of six Ye studies are cohort studies; the
   single RCT (n = 70) is underpowered for efficacy alone (OR 1.94, 0.61–6.20)
   though concordant, and its nephrotoxicity subgroup (OR 0.21, 0.07–0.68) is
   significant. Residual confounding by indication applies to the cohort
   evidence exactly as it did to MIMIC-IV.
4. **Direction, not magnitude, unchanged.** The model predicts nothing about OR
   2.62 or 0.25, absolute risks, or NNT. The synthetic graph's numbers
   (0.735099 etc.) are statements about the graph.
5. **Trough vs AUC.** The independent evidence spans trough-era TDM (Ye studies)
   and AUC-guided TDM (Yang 2024). The model's window functional is agnostic to
   the monitoring statistic; that agnosticism is a modeling simplification, and
   the Rybak 2020 consensus preference for AUC monitoring is noted without being
   claimed as model evidence.

## 7. Falsifiers (pre-registered by construction of the clauses)

| Clause | Falsifier |
|---|---|
| I1 | Any counted source shown to draw from MIMIC-IV, or an excluded MIMIC replication cited as independent |
| I2 | Ye efficacy CI corrected/retracted to include 1.0, or pooled direction flips |
| I3 | Ye nephrotoxicity CI corrected/retracted to include 1.0, or pooled direction flips |
| I4 | Selected path or any of ∫s/peak/total deviates from the C1 canonical values (someone refit the graph) |
| I5 | Gate changes nothing (verification decorative) |
| I6 | eICU supratherapeutic-harm direction flips, or the non-replication registration is stripped |
| I7 | Scope guards removed |

Global: failure of I4 or I5 is **RED** (structural claim breaks); failure of I2 or
I3 demotes the verdict to **NEGATIVE for independent validation** (the whole point
of this report); I6 failure is **RED** for honesty (boundary condition
suppressed). The gate greps the real statistics, not a narrative.

## 8. Commands run

```bash
python3 scripts/research/mercyful_independent_tdm_contract.py   # I_GREEN 7/7
bash scripts/ci/mercyful_independent_tdm_gate.sh                # MERCYFUL_INDEPENDENT_TDM_GATE_OK
python3 scripts/research/mercyful_mimic_iv_vancomycin_contract.py  # anchor still V_GREEN 7/7
```

Sources fetched and read on 2026-07-26:

- <https://pmc.ncbi.nlm.nih.gov/articles/PMC3799644/> (Ye 2013, full text)
- <https://pubmed.ncbi.nlm.nih.gov/37779493/> (Yang 2024, abstract)
- <https://pmc.ncbi.nlm.nih.gov/articles/PMC8326564/> (Hou 2021, full text)
- <https://pubmed.ncbi.nlm.nih.gov/39726684/> (Peng 2024, abstract — excluded, MIMIC-IV)
- <https://pmc.ncbi.nlm.nih.gov/articles/PMC12819319/> (Peng 2026, full text — excluded, MIMIC-IV)

Math-review offload: `bin/llm-offload -t math-review -i` on this spec (logged per
`.claude/AGENT_OFFLOAD_POLICY.md`).

## 9. Verdict

**POSITIVE for independent validation, at structural-correspondence level, with
one registered boundary condition.**

- An independent multi-hospital evidence base (Ye 2013: six studies, four
  countries, one RCT, 521 patients, era and sampling frame disjoint from
  MIMIC-IV) associates TDM with both window bounds the suffering functional
  prices: clinical efficacy up (OR 2.62, CI excluding 1) and nephrotoxicity down
  (OR 0.25, CI excluding 1). The unchanged scheduler — same imported graph, no
  refit — still makes the TDM-guided verified course its unique feasible optimum.
- The mortality direction is not independently replicated (MIMIC-IV-only, though
  three concordant MIMIC-IV analyses), and the eICU trough-level study does not
  show a benefit of trough targeting while confirming supratherapeutic harm.
  Both facts are encoded in the contract (I2/I3 scope note, I6) and enforced by
  the gate.
- This is the fourth structural correspondence for the framework and the first
  *cross-dataset* one in pharmacokinetic dosing: the same fixed decision
  structure matched two disjoint evidence bases without refitting.
