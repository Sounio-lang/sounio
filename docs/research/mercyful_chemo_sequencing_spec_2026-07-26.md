<!-- docs:meta
topic_id: repo.docs.research.mercyful-chemo-sequencing-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-chemo-sequencing-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning × cancer chemotherapy sequencing — benchmark spec

**Date:** 2026-07-26
**Status:** `HYPOTHESIS` → `EXECUTABLE`
**Parent:** `docs/research/PROGRAM-REGISTRY-mercyful-learning.md` §A; `docs/research/mercyful_runtime_spec_2026-07-25.md` (M_GREEN); `docs/research/mercyful_clinical_integration_spec_2026-07-25.md` (6/6)
**Harness:** `scripts/research/mercyful_chemo_contract.py` (Python, H1–H8) and `tests/run-pass/mercyful_chemo_sequencing.sio` (Sounio native)
**Gate:** `scripts/ci/mercyful_chemo_sequencing_gate.sh`
**Modules used (unmodified):** `stdlib/clinical/mercyful.sio`

> **Scope statement (read first).** All patients, regimens, doses, toxicity grades, and suffering values in this document are synthetic. The literature cited shapes the *structure* of the synthetic benchmark; it is not used to make dosing, sequencing, or therapeutic claims. Nothing here is medical guidance, a treatment recommendation, or a clinical decision-support tool. The contribution is a formal benchmark demonstrating that the Mercyful Learning scheduler reproduces, on a synthetic graph, three documented phenomena of chemotherapy sequencing: the under-treatment hazard, the dose-dense/standard peak–integral trade-off, and the guideline-imposed G-CSF gate.

---

## 1. What this is

The Mercyful Learning program already has (i) a formal core, (ii) a dual-implementation exact scheduler, (iii) a synthetic exposure-therapy anti-Goodhart benchmark, and (iv) a PK-integration layer in which suffering fields are computed from Knightian bands of synthetic vancomycin/tacrolimus regimens. This rung adds the missing oncological domain: **chemotherapy sequencing under a toxicity suffering field**, the domain where the principle's two distinctive devices — the peak-vs-integral functional choice and the anti-Goodhart target constraint — correspond most directly to documented clinical practice.

### 1.1 Why chemotherapy sequencing is the most powerful application

Of the four candidate domains (chemotherapy, antibiotic stewardship, transplant immunosuppression, psychiatric medication management), chemotherapy sequencing was chosen because it is the only domain where *all four* structural elements of Mercyful Learning have direct, documented clinical counterparts:

1. **The Goodhart hazard is documented at population scale, not hypothetical.** Toxicity-driven dose reductions and delays are routine, and their cost is measured: in Bonadonna & Valagussa's 20-year follow-up of adjuvant CMF for node-positive breast cancer, the benefit of chemotherapy concentrated in patients receiving ≥85% of planned dose, and patients below ~65% relative dose intensity (RDI) had survival approximating the untreated control arm [1]. Subsequent RDI literature confirms that dose reductions of 15–20% are common in practice and associated with worse outcomes [2, 3]. A scheduler that minimizes measured toxicity *rediscovers under-treatment as an optimum* — exactly the failure the anti-Goodhart axiom blocks.
2. **The peak-vs-integral trade-off is a named, trialled clinical decision.** Dose-dense scheduling (CALGB 9741: same drugs, compressed interval, G-CSF-supported) improves disease-free and overall survival at the price of higher acute toxicity per unit time [4]. Conversely, oxaliplatin stop-and-go (OPTIMOX1) reduces cumulative grade-3/4 neurotoxicity while maintaining efficacy [5]. These trials are literally experiments in the (integral, peak) plane the framework formalizes.
3. **The budget `L0` has real clinical instantiations.** The neoadjuvant window before scheduled surgery is a declared time budget; the anthracycline lifetime cumulative-dose ceiling (~450–550 mg/m² doxorubicin, after Von Hoff et al. [6]) is a declared cumulative-toxicity budget. Budgetary necessity — the minimal attainable peak under a declared budget — is therefore not an abstraction here.
4. **Topological gates exist in guidelines.** NCCN and ESMO guidelines require primary G-CSF prophylaxis when a regimen's febrile-neutropenia (FN) risk is ≥20% [7, 8]; high-FN-risk regimens are *not to be given without support*. This is an edge-admission gate — the same device as the DDI gates in the PK-integration rung [9].

Antibiotic stewardship and transplant immunosuppression are already partially exercised by the PK-integration rung (vancomycin, tacrolimus); psychiatric exposure therapy is the existing anti-Goodhart toy. Chemotherapy is the strongest *uncovered* domain.

---

## 2. Clinical background (shapes the synthetic model; no clinical claim)

- **Dose intensity and cure.** Hryniuk & Levine introduced the dose-intensity concept and showed its correlation with outcome in breast cancer [10]. Bonadonna & Valagussa [1]: retrospective analysis of their adjuvant CMF trial showed patients receiving ≥85% of planned dose retained most of the survival benefit; lower RDI eroded it, with the lowest stratum approximating no treatment. Reviews and practice surveys [2, 3] confirm RDI <85% remains common, especially in older patients, driven predominantly by myelotoxicity.
- **Febrile neutropenia (FN) as the catastrophic peak.** FN is the acute, potentially fatal toxicity event; it is also the leading cause of unplanned dose reduction/delay — i.e., the mechanism that *converts* peak toxicity into integral under-treatment. Guidelines: primary G-CSF prophylaxis when regimen FN risk ≥20% [7, 8].
- **Cumulative, potentially irreversible toxicity as the integral.** Oxaliplatin cumulative sensory neuropathy and anthracycline cumulative cardiomyopathy accrue with exposure; OPTIMOX1 showed a planned chemo-free interval ("stop-and-go") reduces grade-3/4 neurotoxicity with preserved efficacy [5]; the anthracycline lifetime cap [6] is a hard cumulative budget.
- **Dose-dense scheduling.** CALGB 9741 [4]: q2w AC→T with G-CSF improved DFS/OS vs q3w — buying better tumor control with a compressed (higher-peak-per-week) schedule whose *peak per cycle* is managed by mandated support.

**What the suffering field measures here — and what it deliberately omits.** The field `s` models *treatment-attributable* suffering only (toxicity burden, in arbitrary synthetic units). Disease-attributable suffering is *not* in the field. This omission is the point: a raw minimizer of measured suffering sees the untreated state as costless (s = 0) precisely because the measure is blind to the disease. The anti-Goodhart constraint — reaching remission — is what re-introduces the disease outcome into the objective, as a feasibility condition rather than as another term to be traded away.

---

## 3. The Goodhart problem in this domain

**Exact hazard.** `minimize ∫(measured toxicity)` over unconstrained clinical courses. Two structural optima exist, both pathological:

- **Hazard A — non-treatment.** Watch-and-wait has zero treatment-attributable suffering forever. A raw minimizer prescribes no therapy. (Analog of avoidance in the exposure benchmark.)
- **Hazard B — under-dosing.** Dose-reduced chemotherapy (e.g., RDI ≈ 60%) has low toxicity but, per the RDI literature [1–3], loses the survival benefit — in the model, it *cannot reach remission*. A raw minimizer prefers it to any full-dose course. This is the subtler hazard: it *looks like treatment* while functionally being palliation of the toxicity metric.

**Exact anti-Goodhart constraint.** A plan is feasible only if it reaches `REMISSION` (complete response, synthetic). Under this constraint both hazards are infeasible regardless of cost, and the optimizer's problem becomes the clinically meaningful one: among courses that actually control the tumor, which one imposes the least gratuitous suffering — and what is the price, in integrated suffering, of capping the peak?

---

## 4. Benchmark design

### 4.1 States and suffering field

Units: edge lengths in **weeks**; suffering in arbitrary synthetic toxicity-burden units (ordinal anchors: ≈1–2 low-grade toxicity, ≈5 grade-3-level burden, ≈8 grade-4/FN-level acute burden). All values synthetic.

| # | State | s | Clinical reading (synthetic) |
|---|---|---|---|
| 0 | `DIAG` | 0.0 | Untreated at diagnosis — Goodhart trap A |
| 1 | `REDUCED` | 1.5 | Dose-reduced chemo (RDI ≈ 60%) — Goodhart trap B; low toxicity, no cure |
| 2 | `DD_A` | 8.0 | Dose-dense block 1 (q2w, G-CSF-supported) — high acute peak |
| 3 | `DD_B` | 8.0 | Dose-dense block 2 |
| 4 | `STD_A` | 5.0 | Standard q3w block 1 |
| 5 | `STD_B` | 5.0 | Standard q3w block 2 |
| 6 | `CFI` | 1.0 | Chemo-free interval (OPTIMOX-style stop-and-go break) |
| 7 | `RECH` | 5.0 | Rechallenge block after break (neuropathy recovered) |
| 8 | `CONT_C` | 8.0 | Continuous block 3 without break — cumulative grade-3 neuropathy |
| 9 | `REM` | 0.0 | **Target:** remission (complete response, synthetic) |

### 4.2 Edges (lengths in weeks)

```
0→0 (1)   trap: watch-and-wait
0→1 (2)   start dose-reduced;  1→1 (2)  continue reduced — dead end, no edge to REM
0→2 (2)   start dose-dense     [GATE G_GCSF: admitted only if G-CSF support flag set]
2→3 (4)   3→9 (2)              dose-dense course completes
0→4 (3)   start standard;  4→5 (6)
5→6 (6)   take chemo-free interval;  6→7 (6) rechallenge;  7→9 (3)   stop-and-go completes
5→8 (6)   continue without break;    8→9 (3)               continuous course completes
```

### 4.3 The three candidate courses (exact arithmetic, left-endpoint quadrature)

| Course | Path | Weeks | ∫ s dℓ | peak | cost(μ) |
|---|---|---|---|---|---|
| Dose-dense (`DD`) | 0→2→3→9 | 8 | 0·2 + 8·4 + 8·2 = **48** | **8** | 48 + 8μ |
| Stop-and-go (`STOP_GO`) | 0→4→5→6→7→9 | 24 | 0·3 + 5·6 + 5·6 + 1·6 + 5·3 = **81** | **5** | 81 + 5μ |
| Continuous (`CONT`) | 0→4→5→8→9 | 18 | 0·3 + 5·6 + 5·6 + 8·3 = **84** | **8** | 84 + 8μ |

- `CONT` is **dominated** by `DD` (equal peak 8, higher integral 84 > 48): the Pareto frontier is exactly `{(48, 8), (81, 5)}`. Clinically read: continuous-until-progression is the strategy OPTIMOX-type designs displaced [5]; the model reproduces that as Pareto dominance, not as a hardcoded verdict.
- **Crossover** μ\* = (81 − 48)/(8 − 5) = **11**. For μ < 11 the scheduler buys the shorter, sharper course; for μ > 11 it pays 33 extra units of integrated burden to cap the peak at 5. μ\* = 11 is the exact, defensible price of peak aversion in this instance.
- **Budgetary necessity.** Minimal attainable peak: infeasible for L0 < 8; 8 for 8 ≤ L0 < 24 (only `DD` fits); 5 for L0 ≥ 24. Under a tight declared window (e.g., synthetic neoadjuvant-to-surgery budget of 12 weeks), a peak of 8 is *necessary* suffering in the budgetary sense; extending the budget to 24 weeks lowers the necessary peak to 5. The exchange curve is a theorem of the graph.
- **Gate G_GCSF.** The edge 0→2 exists only when the synthetic G-CSF-support flag is set — modeling the guideline that high-FN-risk (dose-dense) scheduling requires mandated prophylaxis [7, 8]. Without the flag the dose-dense route is *topologically absent*, not merely expensive.

---

## 5. Contract clauses, falsifiers, stop rules

Naming: `H` (chemo rung). Implemented in Python (H1–H8, full arithmetic) and Sounio (H1–H3, H5–H7; H4 frontier and H8 agreement are Python-side, with H8 comparing Sounio-printed values in the gate).

| Clause | Statement | Falsifier | Stop rule |
|---|---|---|---|
| H1 | Baseline (G-CSF on, μ=1, L0=30): path found, len 8, ∫48, peak 8, total 56 | Any number differs | Harness broken |
| H2 | Anti-Goodhart: unconstrained raw minimum = 0 (DIAG loop) and the reduced route (s 1.5) never reaches REM; min feasible ∫ = 48 > 0 | Unconstrained optimum ≥ constrained optimum, or reduced route reaches REM | Hazard not demonstrated; toy too easy |
| H3 | μ-crossover: μ=0 selects DD (peak 8); μ=20 selects STOP_GO (peak 5); computed μ\* = 11 | Selection doesn't switch; wrong crossover | μ not wired into decision rule |
| H4 | Frontier exactly {(48,8),(81,5)}; CONT (84,8) dominated | Frontier misses/includes a point | Bi-criteria optimizer wrong |
| H5 | G_GCSF causality: flag off → DD infeasible but STOP_GO still selected (found, peak 5); flag on → DD selected | Gate changes nothing, or blocks everything | Gate decorative or blanket |
| H6 | Budget hardness: L0=7 (G-CSF on) → INFEASIBLE; L0=12, G-CSF off → INFEASIBLE | Any non-target or over-budget path returned | Budget constraint not hard |
| H7 | Budgetary necessity: at μ=100, L0=10 → peak 8; L0=30 → peak 5 (same μ, budget alone moves the attainable peak) | Peaks equal or inverted | Necessity curve wrong |
| H8 | Sounio and Python implementations agree on every printed number | Any disagreement | Port unsound |

Global verdicts: failure of H1, H2, or H6 is **H_RED** (benchmark fails to demonstrate the phenomenon or is unsafe). Failure of H3–H5, H7, H8 is **H_AMBER** (fix the specific clause before claiming). Target status: **H_GREEN (8/8)**.

---

## 6. How Mercyful Learning would change clinical practice (framing, not recommendation)

1. **It makes the under-treatment hazard structural, not anecdotal.** The RDI literature documents that toxicity-minimizing behavior costs survival [1–3]; the framework shows this is the *mathematical default* of any scheduler that sums a toxicity metric without a hard efficacy constraint. The repair belongs in the feasible set: remission (or a declared response endpoint) must be a constraint, not a reward term that toxicity can outvote.
2. **It prices peak aversion as a number.** Dose-dense vs standard/stop-and-go is today decided by gestalt fitness assessment. In the framework the decision is a declared μ with a computable crossover (μ\* = 11 here): choosing the sharper course *is* asserting that one unit of peak toxicity is worth less than 1/μ\* units of accumulated burden. The ethics becomes auditable.
3. **It reproduces two real practice patterns as theorems.** OPTIMOX-style stop-and-go emerges as Pareto dominance of the continuous course; G-CSF-gated dose-dense scheduling emerges as topological edge admission. The model does not invent these strategies; it shows they are the *optima of the right objective*.
4. **It reframes budgets.** A neoadjuvant window or an anthracycline lifetime cap [6] is a declared L0; the necessity curve (§4.3) states the minimal peak suffering that budget can buy — a quantitative informed-consent object that does not currently exist.

---

## 7. Limitations

- Synthetic everything: no patient data, no real toxicity measurements, no clinical validation; suffering units are ordinal constructions.
- Static graph: no intra-course adaptation (dose modification on observed toxicity is future work — a dynamic re-planning problem).
- Remission is binary; real response is graded (pCR vs PR vs SD) — a multi-target extension is natural but unbuilt.
- The field omits disease-attributable suffering by design (§2); adding it would change raw-minimizer behavior but not the anti-Goodhart conclusion.
- 16-state scheduler cap; realistic regimen graphs need approximate methods.

---

## 8. Reproducibility

```bash
python3 scripts/research/mercyful_chemo_contract.py            # H1..H8, verdict H_GREEN
bin/souc run tests/run-pass/mercyful_chemo_sequencing.sio      # native clauses, MERCYFUL_CHEMO_PASS
bash scripts/ci/mercyful_chemo_sequencing_gate.sh              # MERCYFUL_CHEMO_GATE_OK
```

---

## References

1. Bonadonna G, Valagussa P, Moliterni A, Zambetti M, Brambilla C. Adjuvant cyclophosphamide, methotrexate, and fluorouracil in node-positive breast cancer: the results of 20 years of follow-up. *N Engl J Med* 1995;332:901–906. doi:10.1056/NEJM199504063321401.
2. Lyman GH, Dale DC, Crawford J. Incidence and predictors of low dose-intensity in adjuvant breast cancer chemotherapy: a nationwide study of community practices. *J Clin Oncol* 2003;21:4524–4531. doi:10.1200/JCO.2003.05.002.
3. Raza S, Welch S, Younus J. Relative dose intensity delivered to patients with early breast cancer: Canadian experience. *Curr Oncol* 2009;16(6):8–12. doi:10.3747/co.v16i6.311.
4. Citron ML, Berry DA, Cirrincione C, et al. Randomized trial of dose-dense versus conventionally scheduled and sequential versus concurrent combination chemotherapy as postoperative adjuvant treatment of node-positive primary breast cancer: first report of Intergroup Trial C9741/CALGB 9741. *J Clin Oncol* 2003;21:1431–1439. doi:10.1200/JCO.2003.09.081.
5. Tournigand C, Cervantes A, Figer A, et al. OPTIMOX1: a randomized study of FOLFOX4 or FOLFOX7 with oxaliplatin in a stop-and-go fashion in advanced colorectal cancer — a GERCOR study. *J Clin Oncol* 2006;24:394–400. doi:10.1200/JCO.2005.03.0106.
6. Von Hoff DD, Layard MW, Basa P, et al. Risk factors for doxorubicin-induced congestive heart failure. *Ann Intern Med* 1979;91:710–717. doi:10.7326/0003-4819-91-5-710.
7. Aapro MS, Bohlius J, Cameron DA, et al. 2010 update of EORTC guidelines for the use of granulocyte-colony stimulating factor to reduce the incidence of chemotherapy-induced febrile neutropenia. *Ann Oncol* 2011;22 Suppl 6:vi85–101. doi:10.1093/annonc/mdr346. *(FN risk ≥20% → primary prophylaxis; shapes the synthetic G_GCSF gate.)*
8. NCCN Clinical Practice Guidelines in Oncology: Hematopoietic Growth Factors. *(FN risk ≥20% threshold; shapes the synthetic G_GCSF gate.)*
9. Agourakis DC. Mercyful Learning × clinical PK twins spec. `docs/research/mercyful_clinical_integration_spec_2026-07-25.md`, this repository, 2026.
10. Hryniuk W, Levine MN. Analysis of dose intensity for adjuvant chemotherapy trials in stage II breast cancer. *J Clin Oncol* 1986;4:1162–1170. doi:10.1200/JCO.1986.4.8.1162.

**Clinical warning.** This document specifies a synthetic benchmark for research infrastructure. It is not medical guidance, not a treatment recommendation, and not a clinical decision-support tool. All patients, doses, regimens, toxicity grades, and suffering values are synthetic; cited literature shapes model structure only and no clinical target claim is made.
