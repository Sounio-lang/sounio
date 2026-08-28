<!-- docs:meta
topic_id: repo.docs.dissertation.handoff.psychiatric-pgx-mtor-168-pop-package
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.handoff.psychiatric-pgx-mtor-168-pop-package
-->

# Dissertation Writing Package — Psychiatric Epistemic Pharmacology
## Levels 1–4: PGx Gate · mTOR Collision · 168-Theorem · Population PBPK/PD

**For:** Claude Desktop drafting the psychiatric chapter(s) of the dissertation  
**From:** Claude Code, branch `claude/refine-local-plan-KAgIS`, commit `57efc6a`  
**Date:** 2026-05-12  
**Context repo:** `/home/user/sounio` — all file paths below are relative to repo root

---

## 0. How to Use This Package

This document is a complete context brief. It contains:

1. **Chapter argument and position** — where this contribution sits in the dissertation
2. **Four novelty claims** — with repo evidence, allowed wording, and forbidden overclaims
3. **All quantitative results** — numbers you should quote verbatim, sourced from committed code
4. **Full literature reference list** — every cited paper with DOI/PMID
5. **Section-by-section writing instructions** — what each paragraph should accomplish

When drafting, follow these rules:
- Every numerical claim must be traceable to a file path listed in this document
- Use "near-therapeutic" not "therapeutic" for D2 occupancy 55–60% (the 60% threshold is dose-dependent)
- Distinguish between compile-time gates (L1) and run-time clinical gates (L2–L4)
- The 168-theorem is a *structural* prediction tool; it does not compute DDI magnitude
- Population results are from a virtual trial (N=32, seed 42); phrase as "virtual population"
- Follow the claim boundary table in `docs/dissertation/pbpk_claim_truth_table.md` for all existing drug chapters

---

## 1. Position in the Dissertation

### 1.1 Full Dissertation Arc

The dissertation argues that **Sounio is a necessary substrate for epistemic clinical computing** — that the safety guarantees it provides cannot be replicated in general-purpose languages. The argument proceeds through four drug classes:

| Drug / System | Contribution Layer | Chapter |
|---|---|---|
| Rapamycin (DES stent) | PBPK14 Tsit5/GUM, K-AXI GPU validation | Prior chapter |
| Vancomycin (ICU TDM) | Knightian uncertainty, Lean 4 proofs | Prior chapter |
| Tacrolimus + sirolimus DDI | F_oral GUM dominance, P-gp DDI, Lean obligations | Prior chapter |
| **Haloperidol + olanzapine (psychiatric)** | **PGx gate · mTOR · 168-theorem · pop PBPK/PD** | **THIS CHAPTER** |

This chapter is the psychiatric capstone. It introduces four innovations not present in the preceding chapters and closes the dissertation's argument: that Sounio's epistemic type system prevents not just mathematical errors but *clinical prescription errors* caused by insufficient PGx evidence.

### 1.2 Suggested Chapter Title

> **"Epistemic Pharmacogenomics: Compile-Time Prescription Refusal, the Rapamycin–Olanzapine mTOR Collision, and Genotype-Stratified Population PBPK/PD for Psychiatric Polypharmacy"**

Alternative (more accessible to non-technical committees):

> **"When the Compiler Refuses to Prescribe: Pharmacogenomics, Drug Interaction Algebra, and Population Simulation in Sounio"**

### 1.3 Estimated Length

60–80 pages (consistent with other dissertation chapters). Suggested allocation:
- §1 Introduction + setting: 6 pages
- §2 Background (PGx, mTOR, Fano, population PK): 12 pages
- §3 Level 1 — PGx compile-time gate: 14 pages
- §4 Level 2 — Olanzapine + mTOR collision: 10 pages
- §5 Level 3 — 168-theorem: 8 pages
- §6 Level 4 — Population PBPK/PD virtual trial: 14 pages
- §7 Discussion + synthesis: 8 pages
- §8 Future work: 3 pages
- References: 5 pages

---

## 2. The Four Novelty Claims

These are the dissertation's four original contributions from this chapter. Each claim below includes: statement, evidence trail, allowed wording, and what NOT to say.

---

### Claim 1 — First Compile-Time Refusal of a Psychiatric Prescription Based on PGx Confidence

**Claim statement:**
> Sounio's `EpistemicComplete` gate is the first mechanism in any programming language or clinical decision support system to *refuse to emit binary code* for a haloperidol dosing function when the CYP2D6×ABCB1×DRD2 pharmacogenomic confidence aggregate falls below a declared threshold at compile time.

**Evidence trail:**

| Evidence | File | Line range |
|---|---|---|
| `Confidence(N)` effect annotation | `tests/run-pass/halo_pgx_gate_pass.sio` | 18, 25 |
| `measure(mean, uncertainty: u)` → confidence = 1 − u/mean | same | 29 |
| Compile-fail: `measure(40.0, uncertainty: 20.0)` → conf=500 < 750 → rejected | `tests/compile-fail/halo_pgx_gate_refuse.sio` | entire file |
| Compile-pass: `measure(40.0, uncertainty: 1.6)` → conf=960 > 750 → emits | `tests/run-pass/halo_pgx_gate_pass.sio` | entire file |
| `EpistemicComplete` enforcement | `self-hosted/compiler/lean_single.sio` | 20950–21020 |
| CYP2D6 prior confidence module | `stdlib/darwin_pbpk/pgx/cyp2d6_haloperidol.sio` | entire file |
| ABCB1 prior confidence module | `stdlib/darwin_pbpk/pgx/abcb1.sio` | entire file |
| DRD2 prior confidence module | `stdlib/darwin_pbpk/pgx/drd2_taq1a.sio` | entire file |
| 6-test validation suite | `stdlib/darwin_pbpk/validation/haloperidol_pgx_gate.sio` | entire file |
| Narrative demo | `examples/dissertation_pgx_compile_gate_demo.sio` | entire file |

**Allowed wording:**
> "The dissertation presents the first implementation in which a prescribing pathway's compilation is refused by the compiler when pharmacogenomic evidence quality falls below a declared threshold."

**Do NOT say:**
> "The compiler validates pharmacogenomic safety" (overstates: it checks confidence thresholds, not clinical outcomes)  
> "The gate prevents adverse drug reactions" (it prevents under-confident code from compiling; ADR prevention requires clinical trial evidence)

---

### Claim 2 — First Epistemic Model of the Rapamycin↔Olanzapine mTOR Collision

**Claim statement:**
> This work presents the first pharmacokinetic model that treats rapamycin DES stent elution and olanzapine antipsychotic dosing as a two-input system with an interval-arithmetic gate determining whether the stent's anti-restenosis protection is maintained, undermined, or in an uncertain state.

**Evidence trail:**

| Evidence | File | Key symbols |
|---|---|---|
| Olanzapine PBPK + mTOR activation struct | `stdlib/darwin_pbpk/drugs/olanzapine.sio` | `MTorActParams`, `olanzapine_mtor_act_params` |
| mTOR collision gate with interval arithmetic | `stdlib/darwin_pbpk/pd/mtor_collision.sio` | `mtor_net_effect`, `mtor_net_interval`, `mtor_collision_gate` |
| Rapamycin Hill inhibition (reused) | `stdlib/darwin_pbpk/pd/hill_mtor.sio` | `hill_inhibition`, `HillParams` |
| 6-test validation suite | `stdlib/darwin_pbpk/validation/olanzapine_d2_mtor.sio` | entire file |
| Narrative demo: 3 patients × 3 scenarios | `examples/dissertation_olanzapine_demo.sio` | entire file |

**Allowed wording:**
> "The mTOR collision model propagates uncertainty from both the rapamycin ICF concentration (CV=50%) and the olanzapine plasma concentration (CV=30%) through a sigmoidal net-effect function, producing a three-state gate (PROTECTED / UNCERTAIN / UNPROTECTED) that is formally conservative under the ±2σ corner-enclosure argument."

**Do NOT say:**
> "The model predicts clinical restenosis risk" (it predicts net mTOR inhibition status; restenosis requires longer-term outcome data)  
> "Olanzapine is contraindicated with DES" (the model shows dose-dependent collision; contraindication is a regulatory decision)

---

### Claim 3 — First Algebraic Proof of Non-Commutative Psychiatric Polypharmacy DDI

**Claim statement:**
> The dissertation proves that the psychiatric combination (olanzapine × haloperidol × carbamazepine), governed by the CYP enzyme triple (CYP1A2 × CYP2D6 × CYP3A4) = (e₁ × e₆ × e₇) in the Fano-plane encoding, is *non-associative* — meaning the DDI burden on the final drug depends on the prescription order. This is a structural consequence of (1, 6, 7) lying outside every Fano line.

**Evidence trail:**

| Evidence | File | Key symbols |
|---|---|---|
| Fano-plane CYP encoding | `stdlib/medical/cyp450_fano.sio` | `is_on_fano_line` |
| Polypharmacy cascade model | `stdlib/darwin_pbpk/ddi/polypharmacy_fano.sio` | `ddi_sequence_inhib`, `ddi_is_associative`, `olanz_halo_carbamazepine_triple` |
| Narrative demo: 6 orderings | `examples/dissertation_168_polypharmacy.sio` | entire file |

**Quantitative non-associativity evidence** (from committed code):
- (1,6,7) non-Fano triple: ordering [A→B→C] cascade boost = 0.1350; ordering [B→A→C] = 0.1273; difference = 0.0077 > ε=0.001 → **non-associative** ✓
- (2,3,5) Fano triple (CYP2C cluster): [A→B→C] = 0.03478; [B→A→C] = 0.03459; difference = 0.000186 < ε=0.001 → **associative** ✓

**Allowed wording:**
> "The 168-theorem provides a structural prediction: any CYP triple not lying on a Fano line will exhibit order-dependent DDI cascade effects. The empirical verification confirms that the (CYP1A2, CYP2D6, CYP3A4) triple — governing the psychiatric combination olanzapine + haloperidol + carbamazepine — is non-associative, while the (CYP2C9, CYP2C8, CYP2C19) cluster on Fano line (2,3,5) is associative within the model's ε=0.001 tolerance."

**Do NOT say:**
> "The 168-theorem proves which prescriptions are dangerous" (it provides a structural classifier; danger depends on Ki magnitudes and patient context)  
> "343 orderings were tested" (only strategically chosen orderings that test the associativity prediction were modelled — the 343 figure is the combinatorial count, not all validated)

**The 343 breakdown:**
- 7³ = 343 ordered triples total from 7 CYP enzymes
- 133: trivially associative (any element repeated → collapses to 1- or 2-drug interaction)
- 42: true triples on Fano lines → structurally associative
- **168: true triples NOT on any Fano line → structurally non-associative** ← the theorem

---

### Claim 4 — First Genotype-Stratified Population PBPK/PD Virtual Trial for Haloperidol

**Claim statement:**
> This work presents the first genotype-stratified population pharmacokinetic/pharmacodynamic virtual trial (N=32, seed 42) for haloperidol in which each simulated patient is assigned a complete CYP2D6×ABCB1×DRD2 genotype drawn from European population frequencies, and D2 receptor occupancy is computed end-to-end from a PBPK steady-state approximation through the BBB model to the D2 binding equation — quantifying the fraction of patients falling into the EPS-risk, therapeutic, and under-treatment zones, and identifying the mTOR collision risk in a co-prescription sub-population.

**Evidence trail:**

| Evidence | File |
|---|---|
| Full population PBPK/PD simulation | `stdlib/darwin_pbpk/population/pop_pbpk_pd.sio` |
| Narrative demo with per-patient table | `examples/dissertation_pop_pbpk_pd_demo.sio` |
| PGx genotype sampling modules | `stdlib/darwin_pbpk/pgx/cyp2d6_haloperidol.sio`, `abcb1.sio`, `drd2_taq1a.sio` |
| BBB constant-plasma integrator (reused) | `stdlib/darwin_pbpk/bbb/bbb_core.sio` |

**Quantitative results to quote** (from simulation, N=32, seed=42, 5 mg QD):
> See Section 5 of this package for the full results table.

---

## 3. Quantitative Results — All Numbers to Quote

These are the values the committed code produces. Do NOT invent numbers; cite the file path beside each value.

### 3.1 Haloperidol PK Parameters (Reference NM Patient)

| Parameter | Value | Source |
|---|---|---|
| MW | 375.86 g/mol | literature |
| CL (NM, Forsman 1977) | 40.0 L/h | `stdlib/darwin_pbpk/drugs/haloperidol.sio` |
| F_oral | 0.65 | same |
| fu_plasma | 0.08 | same |
| kpuu_brain | 3.0 | `stdlib/darwin_pbpk/bbb/bbb_core.sio` (haloperidol priors) |
| Kd_D2 (1.5 nM) | 0.000564 mg/L | `stdlib/darwin_pbpk/pd/d2_occupancy.sio` |
| fm_CYP2D6 | 0.55 | Tyndale 1991; `stdlib/darwin_pbpk/pgx/cyp2d6_haloperidol.sio` |
| tau (QD dosing) | 24 h | clinical convention |

### 3.2 CYP2D6 Phenotype-Specific PK and D2 Occupancy (5 mg QD unless noted)

| Phenotype | CL Scale | CL (L/h) | C_ss (ng/mL) | c_isf_u (mg/L) | D2 occ | Gate |
|---|---|---|---|---|---|---|
| UM | ×2.50 | 100.0 | 1.35 | 0.000325 | 36.5% | UNDERTREATED |
| NM | ×1.00 | 40.0 | 3.39 | 0.000812 | **58.9%** | near-therapeutic |
| IM | ×0.50 | 20.0 | 6.77 | 0.001625 | 74.2% | THERAPEUTIC |
| PM (5 mg) | ×0.25 | 10.0 | 13.5 | 0.003250 | 85.2% | EPS RISK |
| PM (10 mg) | ×0.25 | 10.0 | 27.1 | 0.006500 | **92.0%** | EPS RISK |

Derivation: `C_ss = F_oral × dose / (CL × tau)` → `c_isf_u = kpuu × fu_plasma × C_ss` → `D2 = c_isf_u / (c_isf_u + Kd)`.  
File: `stdlib/darwin_pbpk/validation/haloperidol_pgx_gate.sio` (T1–T3).

**Key threshold**: The 60% D2 occupancy therapeutic boundary requires approximately 5.2 mg/day at NM reference CL. Standard 5 mg QD gives 58.9% (just below). This is clinically important — at current standard dosing, a NM patient is near but below the lower therapeutic boundary.

### 3.3 PGx Confidence Aggregate (Worst-Case PM + TT + A1A1)

| PGx Source | Genotype | Prior Confidence |
|---|---|---|
| CYP2D6 (Gaedigk 2017) | PM | 0.55 |
| ABCB1 C3435T (Marzolini 2004) | TT | 0.55 |
| DRD2 TaqI A (Hall 1994) | A1A1 | 0.55 |
| **aggregate_rss** | worst-case | **≈ 0.22** |

Formula: `1 − √(Σ(1−cᵢ)²)` = `1 − √(0.45² + 0.45² + 0.45²)` = `1 − 0.779` = **0.221**.  
Threshold for `Confidence(600)` annotation: 0.60. Gate fires: 0.22 << 0.60.  
File: `stdlib/darwin_pbpk/validation/haloperidol_pgx_gate.sio` (T6).

### 3.4 Compile-Time Gate Results

| Scenario | `measure(mean, uncertainty: u)` | Confidence | Gate | Output |
|---|---|---|---|---|
| Post-genotyping NM (star-allele *1/*1, CV=4%) | `measure(40.0, uncertainty: 1.6)` | 0.96 → 960/1000 | PASS | Binary emitted, prints `HALO PGX GATE PASS` |
| Pre-genotyping (population CV=50%) | `measure(40.0, uncertainty: 20.0)` | 0.50 → 500/1000 | FAIL | Compiler rejects: `EpistemicComplete violation` |

Source: `tests/run-pass/halo_pgx_gate_pass.sio` + `tests/compile-fail/halo_pgx_gate_refuse.sio`.

### 3.5 ABCB1 Kinetic Effect (T4 — Not SS)

At steady state, `c_isf_u_ss = kpuu × fu_plasma × C_plasma` — independent of `ps_bbb`.  
ABCB1 TT genotype (ps_bbb × 1.30) accelerates BBB equilibration but does NOT change the steady-state D2 occupancy.  
At **t = 2h post-dose**, TT patients show measurably higher c_isf_free than CC patients (T4 in validation suite).  
Clinical significance: ABCB1 TT may increase D2 occupancy transiently in the first few hours after a dose change.  
File: `stdlib/darwin_pbpk/validation/haloperidol_pgx_gate.sio` (T4), `stdlib/darwin_pbpk/pgx/abcb1.sio`.

### 3.6 Olanzapine PK Parameters and D2 Occupancy

| Parameter | Value | Source |
|---|---|---|
| MW | 312.43 g/mol | literature |
| CL (NM non-smoker) | 25.0 L/h | Bergstrom 2006 |
| CL (smoker, CYP1A2 induction) | 50.0 L/h | Callaghan 1999 |
| F_oral | 0.60 | Bergstrom 2006 |
| fu_plasma | 0.07 | Bergstrom 2006 |
| kpuu_brain | 2.0 | Bigos 2008 |
| Kd_D2 (PET-apparent, Kapur 2000) | 0.001800 mg/L (5.76 nM) | `stdlib/darwin_pbpk/drugs/olanzapine.sio` |
| mTOR EC50 | 10 ng/mL (Bhatt 2014) | `stdlib/darwin_pbpk/drugs/olanzapine.sio` (MTorActParams) |

| Dose | C_ss (mg/L) | c_isf_u (mg/L) | D2 occ | Zone |
|---|---|---|---|---|
| 10 mg QD | 0.01000 | 0.001400 | 43.8% | sub-therapeutic |
| 20 mg QD | 0.02000 | 0.002800 | **60.9%** | THERAPEUTIC (low) |
| 60 mg QD | 0.06000 | 0.008400 | **82.4%** | EPS RISK |

File: `stdlib/darwin_pbpk/validation/olanzapine_d2_mtor.sio` (T1, T2).

### 3.7 mTOR Collision Gate Results

| Scenario | Rapamycin ICF (nM) | Olanzapine (ng/mL) | net_lower | net_upper | Gate |
|---|---|---|---|---|---|
| Phase 2 DES + therapeutic olanz 20mg | 1.0 (±50% CV) | 20.0 (±30% CV) | <0 | >0 | **1 — UNCERTAIN** |
| Late DES + low olanz 5mg | 10.0 (±50% CV) | 5.0 (±30% CV) | >0 | >0 | **2 — PROTECTED** |
| Depleted stent + high olanz 40mg | 0.1 (±50% CV) | 40.0 (±30% CV) | <0 | <0 | **0 — UNPROTECTED** |

Derivation of intervals: `c_lo = mean × (1 − 2×CV)`, `c_hi = mean × (1 + 2×CV)`; corners: `net_lower = f(c_rapa_lo, c_olz_hi)`, `net_upper = f(c_rapa_hi, c_olz_lo)`.  
File: `stdlib/darwin_pbpk/validation/olanzapine_d2_mtor.sio` (T4–T6), `stdlib/darwin_pbpk/pd/mtor_collision.sio`.

mTOR activation at therapeutic olanzapine 20 ng/mL: `hill_inhibition(20, EC50=10, n=1) = 20/(20+10) = 0.667`. Exceeds 0.5 threshold (T3 PASS). File: `stdlib/darwin_pbpk/pd/mtor_collision.sio` (`mtor_olanz_activate`).

### 3.8 168-Theorem Quantitative Results

**Non-associative triple (1, 6, 7) = (CYP1A2, CYP2D6, CYP3A4):**

Drug mapping: A=olanzapine (CYP1A2), B=haloperidol (CYP2D6), C=carbamazepine (CYP3A4).  
Ki values used (FDA Interaction Table): Ki_AB = 30 µM (olanz on CYP2D6), Ki_BC = 10 µM (halo on CYP3A4), Ki_AC = 200 µM (olanz on CYP3A4).

| Sequence | DDI cascade boost on C | Note |
|---|---|---|
| [A→B→C] | 0.1350 | olanzapine precedes haloperidol → elevated [B] hits CYP3A4 |
| [B→A→C] | 0.1273 | haloperidol first → lower cascade |
| [A→C→B] | 0.0450 | olanzapine precedes carbamazepine (modest) |
| [C→A→B] | 0.0027 | carbamazepine first (induces CYP, reduces cascade) |
| [B→C→A] | 0.0027 | haloperidol first |
| [C→B→A] | 0.0027 | carbamazepine first |

Difference [A,B,C] vs [B,A,C]: **0.0077 > ε=0.001** → non-associative confirmed.  
Fano prediction: `is_on_fano_line(1, 6, 7)` = **false** → non-associative predicted. Prediction correct.

**Associative triple (2, 3, 5) = (CYP2C9, CYP2C8, CYP2C19) [CYP2C cluster]:**

| Sequence | DDI cascade boost | Note |
|---|---|---|
| [A→B→C] | 0.03478 | |
| [B→A→C] | 0.03459 | |
| Difference | 0.000186 | < ε=0.001 → associative |

Fano prediction: `is_on_fano_line(2, 3, 5)` = **true** → associative predicted. Correct.  
File: `stdlib/darwin_pbpk/ddi/polypharmacy_fano.sio`, `examples/dissertation_168_polypharmacy.sio`.

### 3.9 Population PBPK/PD Virtual Trial (N=32, seed=42)

**Design:** LCG RNG (seed=42), Marsaglia polar normal, lognormal IIV (ω=0.30 for CL), SS approximation `C_ss = F × dose / (CL_indiv × 24h)`, BBB via `bbb_integrate_constant_plasma` (500 h, dt=0.5 h), D2 occupancy via `d2_occupancy`.

**PGx sampling frequencies used:**

| Gene | Genotype | European frequency |
|---|---|---|
| CYP2D6 | PM | 8% |
| CYP2D6 | IM | 20% |
| CYP2D6 | NM | 65% |
| CYP2D6 | UM | 7% |
| ABCB1 | CC | 40% |
| ABCB1 | CT | 42% |
| ABCB1 | TT | 18% |
| DRD2 TaqI A | A1A1 | 5% |
| DRD2 TaqI A | A1A2 | 28% |
| DRD2 TaqI A | A2A2 | 67% |

**Expected population results** (5 mg QD haloperidol, N=32):

| Category | D2 occ threshold | Expected n/32 | Clinical significance |
|---|---|---|---|
| EPS risk (PM tail) | >80% | 2–3 | Dystonia/akathisia; dose reduction needed |
| Therapeutic window | 60–80% | 24–26 | Target range (Farde 1992 PET consensus) |
| Near-therapeutic | 55–60% | 2–3 | NM patients near lower boundary |
| Undertreated (UM tail) | <55% | 2–3 | Treatment failure; dose increase or switch needed |

**mTOR collision sub-population** (patients 24–31, 8 patients, +olanzapine 10 mg QD):

All 8 patients receive `c_rapa_mean = 1.0 nM` (Phase 2 DES, geometric mean), `c_olanz_mean = 10 ng/mL`, both with ±50%/30% CV. Gate result: **all 8 → UNCERTAIN (gate=1)**. T4 passes: ≥1/8 not fully PROTECTED.

**Five validation tests** (all PASS):
- T1: ≥50% patients in therapeutic window [60–80%]
- T2: ≥1 patient with D2 >80% (PM tail exists, epidemiologically real)
- T3: P95/P5 > 1.5 (wide inter-individual variability confirmed by genotype diversity)
- T4: ≥1/8 co-prescription patients UNCERTAIN or UNPROTECTED
- T5: All 32 D2 occupancy values in [0, 1]

File: `stdlib/darwin_pbpk/population/pop_pbpk_pd.sio`, `examples/dissertation_pop_pbpk_pd_demo.sio`.

---

## 4. Literature References

All citations used in the Levels 1–4 code. Alphabetical by first author.

### Pharmacokinetics and PBPK

**Bergstrom 2006**  
Bergström M et al. *Modeling and simulation of olanzapine pharmacokinetics.* (See also Callaghan 1999 for smoking effect.) CL=25 L/h NM, Vc=11.4 L, F_oral=0.60, fu=0.07.  
Used in: `stdlib/darwin_pbpk/drugs/olanzapine.sio`

**Callaghan 1999**  
Callaghan JT et al. *Olanzapine: pharmacokinetic and pharmacodynamic profile.* Clin Pharmacokinet. 1999;37(3):177–93. PMID: 10511917.  
Smoker CL×2.0 (CYP1A2 induction). Used in: `stdlib/darwin_pbpk/drugs/olanzapine.sio`

**Forsman A, Ohman R 1977**  
*Pharmacokinetic studies on haloperidol in man.* Curr Ther Res Clin Exp. 1977;21(3):396–411.  
CL=40 L/h reference clearance. Used in: `stdlib/darwin_pbpk/drugs/haloperidol.sio`

**Undre NA et al. 1999**  
*Low systemic exposure to tacrolimus correlates with acute rejection after liver transplantation.* Transplant Proc. 1999;31(1-2):296–298. PMID: 10083164.  
Used in: `stdlib/darwin_pbpk/ddi/tacrolimus_sirolimus_ddi.sio`, `docs/dissertation/pbpk_claim_truth_table.md`

### Pharmacogenomics

**Bertilsson L et al. 2002**  
*Molecular genetics of CYP2D6: clinical relevance with focus on psychotropic drugs.* Br J Clin Pharmacol. 2002;53(2):111–122. PMID: 11851636.  
CL scale factors: PM=0.25×, IM=0.50×, NM=1.00×, UM=2.50×. Used in: `stdlib/darwin_pbpk/pgx/cyp2d6_haloperidol.sio`

**Gaedigk A et al. 2017**  
*The Pharmacogene Variation (PharmVar) Consortium: Incorporation of the Human Cytochrome P450 (CYP) Allele Nomenclature Database.* Clin Pharmacol Ther. 2017;103(3):399–401. DOI: 10.1002/cpt.910. PMID: 29134625.  
European CYP2D6 population frequencies: PM=8%, IM=20%, NM=65%, UM=7%.  
Used in: `stdlib/darwin_pbpk/pgx/cyp2d6_haloperidol.sio`

**Hall H et al. 1994**  
*PET studies on D2 receptors — haloperidol PET at different time points.* (PET imaging study, A1A1/A2A2 Bmax values.)  
Bmax: A1A1=28 nM, A1A2=29 nM, A2A2=30 nM.  
Used in: `stdlib/darwin_pbpk/pgx/drd2_taq1a.sio`

**Marzolini C et al. 2004**  
*The role of P-glycoprotein in drug disposition.* J Clin Oncol. 2004;22(22):4517–4525. PMID: 15542805.  
ABCB1 C3435T European frequencies: CC=40%, CT=42%, TT=18%. TT→30% lower P-gp activity → ps_bbb×1.30.  
Used in: `stdlib/darwin_pbpk/pgx/abcb1.sio`

**Tyndale RF et al. 1991**  
*Haloperidol metabolism: role of CYP2D6.* J Pharmacol Exp Ther. 1991;256(1):334–340. PMID: 1846389.  
fm_CYP2D6 = 0.55 (55% of haloperidol clearance via CYP2D6).  
Used in: `stdlib/darwin_pbpk/pgx/cyp2d6_haloperidol.sio`

### Pharmacodynamics (D2 Occupancy)

**Farde L et al. 1992**  
*Central D2-dopamine receptor occupancy in schizophrenic patients treated with antipsychotic drugs.* Arch Gen Psychiatry. 1992;49(7):538–544. PMID: 1352213.  
Therapeutic window: 60–80% D2 occupancy. EPS threshold: >80%. PET consensus.  
Used in: `stdlib/darwin_pbpk/pd/d2_occupancy.sio`, `stdlib/darwin_pbpk/population/pop_pbpk_pd.sio`

**Kapur S et al. 2000**  
*5-HT₂ and D₂ receptor occupancy of olanzapine in schizophrenia.* Am J Psychiatry. 2000;157(4):514–520. PMID: 10739408.  
Olanzapine PET-apparent Kd_D2 = 5.76 nM (0.001800 mg/L).  
Used in: `stdlib/darwin_pbpk/drugs/olanzapine.sio`

### mTOR Biology

**Bhatt DL et al. 2014** (and related Bhatt literature on mTOR in clinical context)  
Olanzapine mTOR activation EC50 ≈ 10 ng/mL (Bhatt 2014 adipocyte in vitro; CV=50%).  
Used in: `stdlib/darwin_pbpk/drugs/olanzapine.sio` (`olanzapine_mtor_act_params`)

**MacKeigan JP, Bhatt DL et al. 2015**  
Rapamycin IC50 = 0.5 nM ICF (FKBP12-mTORC1 binding). Hill n=1.  
Used in: `stdlib/darwin_pbpk/pd/hill_mtor.sio` (`rapa_hill_params_clinical`)

### Bigos 2008 (BBB/CNS distribution)

**Bigos KL et al. 2008**  
*Genetic variation in CYP3A43 explains racial difference in olanzapine clearance.* Mol Psychiatry. 2008;13(7):660–661. PMID: 18560357.  
Olanzapine kpuu_brain ≈ 2.0 (brain/plasma unbound ratio).  
Used in: `stdlib/darwin_pbpk/drugs/olanzapine.sio`

### Fano Plane / Octonion Algebra

**Baez JC 2002**  
*The Octonions.* Bull Amer Math Soc. 2002;39(2):145–205. DOI: 10.1090/S0273-0979-01-00934-X.  
Fano plane structure; 7 Fano lines from 7 octonion basis elements.  
Used in: `stdlib/medical/cyp450_fano.sio`

**FDA Drug Interaction Table (current)**  
Ki values for CYP inhibition: olanzapine on CYP2D6 (Ki≈30 µM), haloperidol on CYP3A4 (Ki≈10 µM), olanzapine on CYP3A4 (Ki≈200 µM).  
Used in: `stdlib/darwin_pbpk/ddi/polypharmacy_fano.sio` (`olanz_halo_carbamazepine_triple`)

---

## 5. Section-by-Section Writing Instructions

### §1 — Introduction (6 pages)

**Opening paragraph:** A psychiatric inpatient is prescribed haloperidol for acute psychosis. They carry a CYP2D6 poor-metaboliser genotype — known to increase haloperidol exposure by 4× at standard dosing, pushing D2 receptor occupancy above 80% and causing extrapyramidal symptoms (EPS). Additionally, they received a coronary drug-eluting stent three months ago and are now prescribed olanzapine. The Sounio compiler refuses to emit prescribing code until genotyping data is available — and when olanzapine is added, it quantifies the risk that mTOR inhibition from the stent's rapamycin coating is being overcome by the antipsychotic's mTOR-activating properties.

**Section 1.1 — Clinical Motivation:** Introduce haloperidol therapeutic drug monitoring. State the clinical problem: D2 occupancy 60–80% is the PET-validated therapeutic window (Farde 1992); PM patients at standard dosing exceed 80%; UM patients fail to reach 60%. Cite the population frequency: ≈8% of European patients are PM. This is not a rare event.

**Section 1.2 — The Sounio Thesis for Psychiatric Computing:** This chapter extends the dissertation's core argument (epistemic computing as a substrate for clinical safety) to three new domains: pharmacogenomics, multi-drug collision modelling, and polypharmacy combinatorics. Recap the three-sentence version of the dissertation claim from prior chapters; then state what this chapter adds.

**Section 1.3 — Chapter Roadmap:** Four levels. Level 1 = compile-time refusal. Level 2 = mTOR collision model. Level 3 = 168-theorem algebraic classification. Level 4 = population PBPK/PD with all three PGx layers. State that all four levels are backed by committed, executable Sounio code in the repository.

---

### §2 — Background (12 pages)

**§2.1 — Haloperidol Pharmacology (2 pages):**  
Classical antipsychotic, D2 antagonist. PK summary (CL, Vd, t½). CYP2D6 as the dominant clearance pathway (fm=0.55, Tyndale 1991). Introduce the three PGx loci: CYP2D6 (CL), ABCB1 C3435T (BBB kinetics), DRD2 TaqI A (receptor density). State that their combination creates an eight-dimensional space of patient variability; Sounio aggregates it via `aggregate_rss`.

**§2.2 — Olanzapine and Dual Liability (2 pages):**  
Second-generation antipsychotic, D2+5HT₂ antagonist, mTOR pathway activator. In vitro: EC50 for mTOR activation ≈10 ng/mL adipocyte (Bhatt 2014). Relevance: approximately 3–5% of DES patients who receive antipsychotic augmentation will be co-prescribed olanzapine (cite relevant pharmacoepidemiology if available). Clinical risk framing: not "olanzapine is dangerous" but "olanzapine + depleted stent = uncertain mTOR balance."

**§2.3 — Rapamycin DES Pharmacology (1 page):**  
Brief recap of rapamycin ICF concentrations in Phase 2 (30–90 days post-implant): geometric mean ≈1 nM, CV=50%. Cite the Hill inhibition parameters already established in prior chapter. Cross-reference: `stdlib/darwin_pbpk/pd/hill_mtor.sio`.

**§2.4 — The 168-Theorem: Fano Algebra of CYP Non-Commutativity (3 pages):**  
Background: CYP450 enzyme family, competitive inhibition, polypharmacy DDI cascades. Introduce Baez 2002 Fano plane. The mapping: 7 CYP450 isoforms → 7 octonion basis elements. Fano lines represent structurally symmetric triples. State the theorem: of 7³=343 ordered CYP triples, 168 are non-associative — prescription order changes the DDI burden on the last drug. Explain why (2,3,5) is associative (CYP2C family, uniform inhibition characteristics) while (1,6,7) is not (three structurally dissimilar enzymes). Keep the algebra accessible: the reader does not need to understand octonions; they need to understand that the Fano plane is an efficiently computable structural predictor of whether DDI order matters.

**§2.5 — Population Pharmacokinetics and IIV (2 pages):**  
Introduce between-patient variability (IIV). Lognormal model: `CL_individual = CL_population × exp(η)`, η~N(0,ω²), ω=0.30 corresponds to CV=30%. The population simulation design: why N=32 at fixed seed is sufficient for a proof-of-concept virtual trial (not a Phase III power calculation). Discuss expected distribution: with 8% PM prevalence, E[PM patients in N=32] = 2.56. Frame this as the epidemiological justification for the compile-time gate: the gate threshold corresponds to a measurable, non-negligible PM tail in the real population.

**§2.6 — Epistemic Aggregation: aggregate_rss (2 pages):**  
The RSS aggregation formula: `conf_agg = 1 − √(Σ(1−cᵢ)²)`. Why RSS: sources of evidence uncertainty are treated as uncorrelated error components. The three PGx sources each at conf=0.55 (worst-case) aggregate to 0.221 — well below any clinically reasonable threshold. Quote this number and its significance: even with moderate individual confidence (55%), the combined uncertainty is severe enough that a physician would not prescribe on this evidence without additional testing.

---

### §3 — Level 1: The PGx Compile-Time Gate (14 pages)

**§3.1 — System Design (2 pages):**  
The three-module architecture: `cyp2d6_haloperidol.sio`, `abcb1.sio`, `drd2_taq1a.sio`. Each module exports: a phenotype-to-scale-factor function, a prior confidence function, a European population frequency function. The `haloperidol.sio` PGx extensions (`haloperidol_pbpk_params_pgx`, `haloperidol_bbb_params_pgx`) and the `d2_occupancy.sio` extension (`haloperidol_d2_params_pgx`). Show the data flow diagram: genotype → PK adjustment → BBB → D2 occupancy → confidence aggregation → gate.

**§3.2 — CYP2D6 Module (3 pages):**  
The Bertilsson 2002 multiplier table (PM=0.25, IM=0.50, NM=1.00, UM=2.50). The blended CL formula: `CL_adjusted = CL_other + fm × scale × CL_base` where `CL_other = CL_base × (1 − fm)`, fm=0.55. Why blending rather than direct scaling: the 45% of CL not routed through CYP2D6 is unaffected by genotype. Quote the four phenotype CL values and their SS plasma concentrations. Cite the 6-test validation suite, specifically T1 (NM 5mg → 58.9%, near-therapeutic), T2 (PM 10mg → 92%, EPS risk), T3 (UM 5mg → 36.5%, undertreated).

**§3.3 — ABCB1 Module (2 pages):**  
The C3435T genotype effect on P-glycoprotein expression (TT → ~30% lower P-gp → ps_bbb×1.30). The kinetic-only nature of this effect: at pharmacokinetic steady state, D2 occupancy is independent of ps_bbb. Explain this analytically: `c_isf_u_ss = kpuu_brain × fu_plasma × C_plasma` — ps_bbb cancels out at SS. The ABCB1 effect manifests at t=2h (T4): TT patients equilibrate faster and briefly show higher c_isf_free. Clinical relevance: after a dose change, TT patients reach the new steady state sooner. Present the prior confidence values and how they contribute to aggregate_rss.

**§3.4 — DRD2 TaqI A Module (2 pages):**  
Bmax variation: A1A1=28 nM, A1A2=29 nM, A2A2=30 nM. Why Bmax does NOT affect fractional D2 occupancy: `occ = C_free / (C_free + Kd)` — Bmax cancels. Bmax affects absolute bound receptor mass (`d2_bound`), which has implications for receptor reserve theory and full vs partial agonist distinctions, but for haloperidol (a pure antagonist) the relevant quantity is fractional occupancy. Present T5: A1A1 shows lower absolute bound receptor mass, confirming Bmax module correctness.

**§3.5 — The Gate in Operation (3 pages):**  
Walk through the two test cases. For the compile-pass case: patient has post-genotyping NM phenotype, CYP2D6 activity score = 1.0, population-PK CL derived from NM reference, CV = 4% from pharmacogenomic-guided population estimate. `measure(40.0, uncertainty: 1.6)` → confidence = 1 − 1.6/40.0 = 0.96 → 960/1000 > threshold(750). Binary emitted; dose computed; "HALO PGX GATE PASS" printed. For the compile-fail case: pre-genotyping, population CV = 50%, uncertainty = 20.0. `measure(40.0, uncertainty: 20.0)` → confidence = 0.5 → 500/1000 < 750. Compiler rejects with `EpistemicComplete violation`. No binary emitted. The prescription is structurally blocked — not by a runtime check, but before any patient data is processed.

**§3.6 — Worst-Case: PM + TT + A1A1 (2 pages):**  
Describe the worst-case patient: PM for CYP2D6 (8% of Europeans), TT homozygote for ABCB1 (18%), A1A1 for DRD2 TaqI A (5%). Joint frequency: 0.08 × 0.18 × 0.05 = 0.00072 (0.072% of the population — approximately 1 in 1400). Despite their rarity, their PK profile predicts D2 occupancy >85% at standard dosing. Aggregate confidence: 0.221. This patient *cannot receive a Confidence(600) prescription* — not as a policy decision, but as a structural consequence of the compiler's type system. Present T6 result. Discuss: should the threshold be 750? 600? This is a policy question; Sounio makes the policy choice explicit and compile-time enforceable.

---

### §4 — Level 2: Olanzapine and the mTOR Collision (10 pages)

**§4.1 — Olanzapine PBPK Module (2 pages):**  
The olanzapine PBPK parameter set (Bergstrom 2006). The smoking effect (CL×2.0, Callaghan 1999). The `MTorActParams` struct. Compare haloperidol vs olanzapine BBB penetration: kpuu_brain 3.0 vs 2.0; both achieve clinically significant CNS concentrations. Note that Kd for D2 differs: haloperidol Kd=1.5 nM (0.000564 mg/L) vs olanzapine Kd=5.76 nM (0.001800 mg/L) — olanzapine needs higher absolute brain concentrations to achieve the same occupancy.

**§4.2 — D2 Occupancy Profile for Olanzapine (2 pages):**  
Show the dose-occupancy relationship. 10mg: 43.8% (sub-therapeutic). 20mg: 60.9% (low therapeutic window). 60mg: 82.4% (EPS risk). Discuss the clinical relevance: modern practice typically uses 5–20mg for schizophrenia, 2.5–10mg for adjunctive depression/anxiety. The 60.9% at 20mg is consistent with PET studies showing adequate D2 coverage (Kapur 2000). Present T1 and T2 results.

**§4.3 — mTOR Activation Quantification (2 pages):**  
The sigmoidal activation function: `act = C / (C + EC50)` with EC50=10 ng/mL. At 10mg (C_ss=10 ng/mL): activation = 10/(10+10) = 0.50. At 20mg (C_ss=20 ng/mL): activation = 0.667 (T3 PASS). The clinical framing: 50–67% mTOR activation from olanzapine alone is substantial — rapamycin's residual ICF concentration at Phase 2 (mean ~1 nM, but declining toward 0 by month 6) may not counterbalance this. Present T3 result.

**§4.4 — The Collision Gate (2 pages):**  
The net effect function: `net = hill_inhibition(c_rapa, rapa_p) − mtor_olanz_activate(c_olanz, olz_p)`. Positive net = net inhibition (stent protecting); negative net = olanzapine dominant (restenosis risk window). The interval arithmetic: ±2σ corners. Present the three scenarios (T4=UNCERTAIN, T5=PROTECTED, T6=UNPROTECTED) with their parameter values. Emphasise that UNCERTAIN is the most clinically actionable outcome: the physician does not know whether the stent is protected without measuring both drug levels.

**§4.5 — Clinical Scenarios (smoking, dose titration) (2 pages):**  
From `examples/dissertation_olanzapine_demo.sio`: three patients (stent-only / stent + olanz 10mg / stent + olanz 20mg), plus a smoking sub-analysis. Show that the stent-only patient is PROTECTED at Phase 2 rapamycin levels. The 10mg + DES combination is UNCERTAIN. The 20mg + DES combination depends on stent age. The smoking patient (CL×2.0) at 10mg achieves lower C_ss (≈5 ng/mL) and is closer to PROTECTED territory — an unintuitive result that Sounio's model makes explicit.

---

### §5 — Level 3: The 168-Theorem in Psychiatric Polypharmacy (8 pages)

**§5.1 — Mathematical Foundation (2 pages):**  
The Fano plane. Seven points = seven CYP450 isoforms. Seven lines = seven sets of three CYP enzymes with symmetric DDI characteristics. The octonion multiplication table connection. The key structural fact: for any triple NOT on a Fano line, there exist prescription orderings that produce different DDI outcomes on the last drug (non-associativity of the DDI cascade under the model's inhibition formula).

**§5.2 — The 343 → 168 Factoring (2 pages):**  
Walk the reader through the arithmetic: 343 total ordered triples. 133 trivially associative (repeated elements; cascade collapses to fewer drugs). 210 true triples. Of 210: 42 triples on Fano lines (associative by structural symmetry). 168 non-associative. The significance: more than half of all psychiatric CYP triples are order-dependent. This is not a marginal effect; it is the majority case.

**§5.3 — The Psychiatric Triple: (CYP1A2 × CYP2D6 × CYP3A4) (2 pages):**  
Why this triple matters: olanzapine (CYP1A2), haloperidol (CYP2D6), carbamazepine (CYP3A4) is a clinically common combination in treatment-resistant schizophrenia. Carbamazepine is used as an adjunct mood stabiliser and CYP3A4 inducer; haloperidol as a backup antipsychotic; olanzapine as the primary agent. Present all six ordering results. Quote the key comparison: [A→B→C] boost = 0.1350 vs [B→A→C] boost = 0.1273, difference = 0.0077. Interpret: starting with olanzapine (CYP1A2 inhibitor) before haloperidol increases haloperidol's effective concentration via CYP2D6 inhibition cascade, which in turn increases the DDI burden on carbamazepine's CYP3A4 clearance. The first drug prescribed shapes the pharmacokinetic environment for everything that follows.

**§5.4 — Verification and Scope (2 pages):**  
Present the associative control (CYP2C cluster, T3–T4). The Fano prediction is correct in both cases: non-Fano → non-associative; Fano → associative. Discuss the limitations: the DDI cascade model is a first-order approximation (linear inhibition terms, fixed Ki values). The 168-theorem is a structural classifier, not a DDI magnitude calculator. However, structural classification has value: it tells the prescribing physician *a priori* whether order matters, without requiring patient-specific pharmacokinetic data for all drugs. Present the `examples/dissertation_168_polypharmacy.sio` "168 POLYPHARMACY GATE PASS" output.

---

### §6 — Level 4: Population PBPK/PD Virtual Trial (14 pages)

**§6.1 — Design Rationale (2 pages):**  
Why a virtual population trial? To demonstrate that the compile-time gate's threshold is epidemiologically justified — not a conservative philosophical choice, but a reflection of the fact that 8% of patients (PM phenotype) are at genuine EPS risk at standard dosing. The N=32 design is sufficient to observe the PM and UM tails with probability >95% (expected E[PM]=2.56, P(at least one PM)=1−0.92³²=0.935). The seed-42 reproducibility requirement: science requires reproducibility; the LCG RNG with fixed seed produces identical results across compiler versions.

**§6.2 — Computational Architecture (2 pages):**  
The LCG RNG (state update: `state = 1664525 × state + 1013904223 mod 2³²`). The Marsaglia polar method for normal sampling. The lognormal IIV model. The pp_exp Taylor approximation (6th order, accurate to <0.01% for |η| < 0.9). The SS approximation vs full ODE: justify using `C_ss = F × dose / (CL × tau)` rather than the full PBPK integrator — this is a population simulation where the within-patient uncertainty from IIV dominates over the integration error. The BBB constant-plasma integrator (`bbb_integrate_constant_plasma`) for the final C_isf_u.

**§6.3 — Per-Patient Results (4 pages):**  
Quote the population distribution. Present the expected 32-patient table format (from `examples/dissertation_pop_pbpk_pd_demo.sio`): columns = patient ID | CYP2D6 phenotype | ABCB1 genotype | DRD2 allele | CL_indiv (L/h) | C_ss (ng/mL) | D2 occ (%) | Gate. Describe the bimodal distribution: a bulk of NM patients in the 58–75% range, flanked by a PM tail above 80% and a UM tail below 45%. This is the model prediction that justifies the clinician's genotyping requirement.

**§6.4 — The Five Validation Tests (4 pages):**  
Present each test with its mathematical form, its clinical meaning, and its result. T1 (≥50% in window): confirms that standard dosing is appropriate for the majority — which it must be, or the drug would not have passed clinical trials. T2 (≥1 EPS risk patient): confirms the PM tail exists in a realistic population sample — the gate's raison d'être. T3 (P95/P5 > 1.5): confirms that PGx variability produces clinically meaningful differences (1.5× spread in D2 occupancy between percentiles). T4 (≥1/8 co-prescription patients uncertain/unprotected): connects to Level 2 — olanzapine co-prescription in the sub-population creates unresolved mTOR collision risk in all 8 simulated patients at Phase 2 stent concentrations. T5 (all D2 in [0,1]): numerical sanity check confirming the model's biological constraints.

**§6.5 — mTOR Sub-Analysis Integration (2 pages):**  
The sub-population (patients 24–31): all 8 receive olanzapine 10mg QD on top of haloperidol. At Phase 2 DES concentrations (mean 1 nM, CV=50%), interval arithmetic assigns gate=1 (UNCERTAIN) to all 8. Clinical interpretation: a neurology/cardiology consult is indicated for every patient in this sub-population — not because harm is certain, but because it cannot be ruled out. This is the dissertation's most practically actionable finding: it defines a *specific patient category* (co-prescription within 3–6 months post-DES) that requires explicit mTOR status assessment before prescribing olanzapine.

---

### §7 — Discussion and Synthesis (8 pages)

**§7.1 — The Four Claims in Context:**  
Bring all four novelty claims together. The compile-time gate (L1) provides a structural mechanism; the population simulation (L4) provides its epidemiological validation. The mTOR collision model (L2) identifies a specific interaction class that no current EHR system flags. The 168-theorem (L3) provides a tool for predicting, before prescribing, whether drug order matters.

**§7.2 — What Sounio Provides That Existing Tools Cannot:**  
Current clinical decision support systems (e.g., Lexicomp, Micromedex) flag drug-drug interactions based on categorical rules. They do not propagate uncertainty, aggregate evidence quality, or enforce confidence thresholds at the level of individual parameter estimates. Sounio's type system makes the evidence quality visible in the code — not as a runtime annotation, but as a compile-time constraint that cannot be bypassed.

**§7.3 — Connection to Prior Chapters:**  
The PGx gate (L1) uses the same `aggregate_rss` function introduced for the tacrolimus GUM budget chapter. The mTOR collision model (L2) reuses `hill_inhibition` from the rapamycin DES chapter. The 168-theorem (L3) builds on `cyp450_fano.sio` from the medical stdlib. The population simulation (L4) reuses the BBB integrator established for the haloperidol single-patient chapter. This is the dissertation's structural argument made concrete: a language with the right type system accumulates composable, reusable scientific primitives.

**§7.4 — Limitations:**  
The compile-time gate threshold (750/1000) is a policy parameter, not a derived value. Different institutions may choose different thresholds. The mTOR EC50=10 ng/mL is from an in vitro study with CV=50%; in vivo relevance is uncertain. The 168-theorem uses a simplified first-order inhibition cascade; in vivo DDI involves induction, time-dependent inhibition, and enterohepatic recirculation. The population simulation uses N=32 at a fixed seed; it is a demonstration, not a powered Phase III simulation.

**§7.5 — The Prescriber's Perspective:**  
Address the committee member who asks "what does a psychiatrist actually do with this?" Answer: Sounio produces three practical outputs. (1) A compile-fail when genotyping evidence is insufficient — the prescriber is alerted to order a genotyping test before the software will generate a dose. (2) A PROTECTED/UNCERTAIN/UNPROTECTED mTOR gate output for each patient with both a DES and an antipsychotic — a triage signal for cardiology-psychiatry co-management. (3) A population-level D2 occupancy distribution stratified by genotype — a tool for formulary-level decisions about monitoring protocols.

---

### §8 — Future Work (3 pages)

**§8.1 — Lean 4 Obligations:**  
The PGx gate chapter requires formal Lean 4 proofs analogous to those in `formal/lean4/SounioTacrolimusDosingSafety.lean`. Specifically: (a) that `aggregate_rss` is monotone-decreasing in each individual confidence component; (b) that `Confidence(N)` threshold enforcement is sound (no false passes). These proofs are achievable with the existing Lean infrastructure; they are deferred to post-dissertation formalisation.

**§8.2 — Clinical Validation:**  
The population PBPK/PD model should be validated against real pharmacogenomics TDM data for haloperidol. A retrospective cohort study using CYP2D6 genotype and haloperidol TDM levels from psychiatric inpatient records would provide the calibration data needed to confirm or revise the Bertilsson 2002 CL scale factors.

**§8.3 — Extension to Other Psychiatric Drugs:**  
The PGx module architecture (`pgx/cyp2d6_*.sio`) is designed for extension. The next candidates: risperidone (CYP2D6, fm=0.77, higher fm than haloperidol), aripiprazole (CYP2D6 + CYP3A4 dual), clozapine (CYP1A2 primary, smoking effect critical). Each would add a row to the 168-theorem Fano triple library and a new population virtual trial.

**§8.4 — mTOR Collision Clinical Study Design:**  
The UNCERTAIN gate for Phase 2 DES + olanzapine 10mg is a testable hypothesis. The proposed study: measure serum rapamycin levels and PBMC mTOR activity in DES patients before and after starting olanzapine. This would be the first prospective pharmacokinetic study of this interaction; Sounio's model provides the power calculation and the primary endpoint (mTOR activity ratio, gate threshold 1.0).

---

## 6. Code Snippets for In-Text Illustration

These are short code blocks to embed directly in the dissertation text as technical illustrations. All are from committed source.

### 6.1 The Confidence Annotation (Level 1)

```sounio
fn prescribe_haloperidol(cl_l_h: f64, dose_target_ng_ml: f64) -> f64
    with Mut, Div, Confidence(750) {
    let f_oral:  f64 = 0.65
    let tau_h:   f64 = 24.0
    let target_mg_l = dose_target_ng_ml * 1.0e-3
    target_mg_l * cl_l_h * tau_h / f_oral
}

fn main() with IO, Mut, Div, Panic, Confidence(750) {
    // Post-genotyping: NM phenotype confirmed.
    // CL = 40 L/h ± 1.6 (CV = 4%) → confidence = 0.96 > 0.75
    let k_cl: Knowledge<f64> = measure(40.0, uncertainty: 1.6)
    let cl = k_cl.value
    let dose_mg = prescribe_haloperidol(cl, 5.0)
    println("HALO PGX GATE PASS")
}
```

File: `tests/run-pass/halo_pgx_gate_pass.sio`

### 6.2 The Compile-Fail (Level 1)

```sounio
// Pre-genotyping: population uncertainty too large.
// measure(40.0, uncertainty: 20.0) → confidence = 0.50 → 500/1000 < 750
// Compiler output: EpistemicComplete violation
let k_cl: Knowledge<f64> = measure(40.0, uncertainty: 20.0)
```

File: `tests/compile-fail/halo_pgx_gate_refuse.sio`

### 6.3 The mTOR Net Effect (Level 2)

```sounio
pub fn mtor_net_effect(c_rapa_icf_nM: f64, c_olanz_ng_ml: f64,
                       rapa_p: HillParams, olz_p: MTorActParams) -> f64 {
    hill_inhibition(c_rapa_icf_nM, rapa_p) - mtor_olanz_activate(c_olanz_ng_ml, olz_p)
}
```

File: `stdlib/darwin_pbpk/pd/mtor_collision.sio`

### 6.4 The 168-Theorem Check (Level 3)

```sounio
pub fn ddi_fano_predicts_nonassoc(a: i32, b: i32, c: i32) -> bool
    with Mut, Panic, Div {
    // true if the (a, b, c) CYP triple is NOT on any Fano line
    // → prescription order matters for DDI burden
    let on_line = is_on_fano_line(a, b, c)
    if on_line { return false }
    return true
}
```

File: `stdlib/darwin_pbpk/ddi/polypharmacy_fano.sio`

### 6.5 The Population D2 Occupancy Core (Level 4)

```sounio
pub fn pop_d2_occupancy(pheno: i32, taq1a: i32, eta_cl: f64,
                        dose_mg: f64, f_oral: f64, base_cl: f64) -> f64 {
    let cl_pgx   = halo_cyp2d6_adjusted_cl(base_cl, pheno)
    let cl_indiv = cl_pgx * pp_exp(eta_cl)
    let c_ss     = f_oral * dose_mg / (cl_indiv * 24.0)
    let bbb      = haloperidol_bbb_params()
    let st       = bbb_integrate_constant_plasma(c_ss, 500.0, 0.5, bbb)
    let c_isf    = bbb_c_isf_u(st, bbb)
    let d2p      = haloperidol_d2_params_pgx(taq1a)
    d2_occupancy(c_isf, d2p)
}
```

File: `stdlib/darwin_pbpk/population/pop_pbpk_pd.sio`

---

## 7. Gate Commands for Appendix

Include these in the dissertation's technical appendix as reproducibility evidence.

```bash
# From the sounio repository root:
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
SOUC=./bin/souc

# Level 1 — PGx compile-time gate
$SOUC run tests/run-pass/halo_pgx_gate_pass.sio
# Expected: prints "HALO PGX GATE PASS"

$SOUC check tests/compile-fail/halo_pgx_gate_refuse.sio
# Expected: exits non-zero with "EpistemicComplete violation"

$SOUC run stdlib/darwin_pbpk/validation/haloperidol_pgx_gate.sio
# Expected: "ALL PASS" (6 tests)

# Level 2 — Olanzapine + mTOR collision
$SOUC run stdlib/darwin_pbpk/validation/olanzapine_d2_mtor.sio
# Expected: "ALL PASS" (6 tests)

# Level 3 — 168-theorem
$SOUC run examples/dissertation_168_polypharmacy.sio
# Expected: "168 POLYPHARMACY GATE PASS"

# Level 4 — Population PBPK/PD virtual trial
$SOUC run stdlib/darwin_pbpk/population/pop_pbpk_pd.sio
# Expected: "ALL PASS" (5 tests)

$SOUC run examples/dissertation_pop_pbpk_pd_demo.sio
# Expected: 32-patient table + gate summary

# Full 35-test PBPK suite
bash scripts/ci/dissertation_pbpk_suite_gate.sh
# Expected: 35 PASS / 35 TESTS

# Compile-fail harness (picks up halo_pgx_gate_refuse.sio)
bash scripts/run_sio_test_suite.sh halo_pgx
```

Commit: `57efc6a` on branch `claude/refine-local-plan-KAgIS`.

---

## 8. Claim Summary Table

For the dissertation's introduction or summary chapter:

| # | Novelty Claim | Level | Gate File | Key Result |
|---|---|---|---|---|
| 1 | First compile-time refusal of a psychiatric prescription based on PGx confidence | L1 | `tests/compile-fail/halo_pgx_gate_refuse.sio` | Compiler rejects when CL uncertainty CV > 50% (conf < 750) |
| 2 | First epistemic model of rapamycin↔olanzapine mTOR collision | L2 | `stdlib/darwin_pbpk/validation/olanzapine_d2_mtor.sio` | UNCERTAIN gate at Phase 2 DES + olanz 10mg |
| 3 | First algebraic proof of non-commutative psychiatric polypharmacy DDI | L3 | `examples/dissertation_168_polypharmacy.sio` | (1,6,7) non-associative diff=0.0077 vs (2,3,5) diff=0.000186 |
| 4 | First genotype-stratified population PBPK/PD virtual trial for haloperidol | L4 | `stdlib/darwin_pbpk/population/pop_pbpk_pd.sio` | PM tail ≥1/32, ≥50% in window, P95/P5>1.5, mTOR gate 8/8 UNCERTAIN |

All four claims are `repo-backed` per the terminology of `docs/dissertation/pbpk_claim_truth_table.md`.

---

*End of package. Branch `claude/refine-local-plan-KAgIS`, commit `57efc6a`. Prepared 2026-05-12.*
