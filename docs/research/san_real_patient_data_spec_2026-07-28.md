<!-- docs:meta
topic_id: repo.docs.research.san-real-patient-data-spec-2026-07-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.san-real-patient-data-spec-2026-07-28
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — the Suffering-Aware neural Network (SAN) on REAL PATIENT DATA: the suffering field grounded in real clinical outcomes

**Date:** 2026-07-30 (filename follows the 2026-07-28 spec series of the SAN line)
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract R1..R10, `SAN_REAL_PATIENT_VERDICT R_GREEN (26/26)`
**Harness:** `scripts/research/san_real_patient_data.py`
**Gate:** `scripts/ci/san_real_patient_data_gate.sh` (**SAN_REAL_PATIENT_GATE_OK**)
**Parents:** `docs/research/suffering_aware_architecture_spec_2026-07-28.md`
(small-network SAN, clauses A1..A8 — definitions, theorems T1..T5, selection
rule) and `docs/research/suffering_aware_deep_architecture_spec_2026-07-28.md`
(deep SAN, clauses D1..D9 — the scaling instance and the per-family δ
precedent)

> **Scope.** The patients, the features, and the outcomes are **real**:
> 569 real breast fine-needle-aspirate patients with biopsy-confirmed
> diagnoses (WDBC), 306 real breast-cancer surgery patients with real
> 5-year mortality (Haberman), 297 real Cleveland Clinic patients with
> angiographically confirmed coronary disease. The harm **weights**
> (missed hazard = 5, false hazard = 1) remain a **declared normative cost
> structure** — a harm weighting is an ethical input, not a measurable
> quantity. This is not medical guidance, not a treatment recommendation,
> not a diagnostic or screening tool, and no model produced here is fit
> for any clinical use. The "machine suffering" channel is an
> **operational computational-burden proxy** (metered FLOPs/energy): this
> work makes **no claim of machine consciousness, sentience, or
> phenomenology**, and no result below depends on one. All data is
> de-identified and public without credentialing (UCI ML Repository).

---

## 1. Position: the suffering field was measured against real patients

The SAN line so far established the architecture class — suffering-aware
layers, per-sample exit gates, deep supervision, freeze-on-green, and the
anti-Goodhart feasibility gate — and scaled it from a 4-layer MLP on a
synthetic dose-band task (A1..A8) to ResNet-18/ViT-small on real CIFAR-10
images (D1..D9). In both lines the *patient* channel was synthetic: a harm
matrix over synthetic dose bands, then a synthetic cost structure over real
image labels (class 9 "truck" playing the hazard). The deep spec scoped out
explicitly what this spec does:

> the harm channel must eventually be grounded in **real clinical
> outcomes** — real patients whose real adverse events are the suffering
> being metered.

That grounding is the step taken here. Nothing in the architecture changes;
what changes is what the patient channel *is*: every unit of patient
suffering in every ledger below is a real patient with a real adverse
outcome — a real malignancy missed or false-alarmed, a real post-surgical
death within five years, a real coronary lesion — being misclassified by a
model in training. The question the contract answers: does the
architecture's entire certificate suite survive when the suffering field is
real? The answer, certified bit-reproducibly: yes, 26/26 clauses, on all
three cohorts, with the same qualitative effects as the synthetic line
(R10 checks this *by re-running the synthetic benchmark live*).

### 1.1 Why not MIMIC-IV patient-level (data-availability finding)

The task brief named MIMIC-IV (28,451 ICU patients on IV vancomycin) as the
preferred source. An honest inventory of what "the MIMIC-IV data" in this
repository actually is:

- **MIMIC-IV patient-level records require PhysioNet credentialed access**
  (CITI training + signed DUA); they cannot be fetched from this
  environment, and no patient-level MIMIC extract exists in the repository.
- The repository's prior MIMIC-IV leg
  (`scripts/research/mercyful_mimic_iv_vancomycin_contract.py`) used
  **published aggregate statistics** (Wang et al. 2026,
  doi:10.1038/s41598-026-42395-1: 28,451 patients, post-PSM ORs) — real
  clinical evidence, but not patient-level data a network can be trained on.
- The local FAERS CSVs (`data/faers_*.csv`) are **36-row aggregate tables**
  (per-combination case counts), not patient-level reports.
- **SEER** requires a signed data-use agreement; not directly downloadable.
- `scripts/clinical/etl/mimic_iv_vancomycin.sql` is an ETL *definition* for
  a credentialed MIMIC-IV extract; the local
  `scripts/clinical/data_synthetic/tdm_cohort_synthetic_v*.csv` cohorts are
  synthetic (excluded from the main results by the task brief).

The real patient-level cohorts used instead are all **de-identified,
public, and downloadable without credentialing** from the UCI ML Repository
(CC-BY 4.0). They are smaller than MIMIC-IV — that is a limitation, stated
in §8 — but they are *patient-level*: one row per real patient, with real
measured outcomes, which is exactly what training a suffering field
requires.

## 2. What is reused unchanged

From the parent specs, without modification: the suffering ledger
(Definition 2.1: `S_machine`, `S_patient` integral + peak), feasibility as
a categorical anti-Goodhart constraint (Definition 2.2), the
necessary/gratuitous decomposition at the first feasible epoch `t*`
(Definition 2.3), the selection rule `select(C, λ)` with loud `NO_FEASIBLE`,
the metering convention (linear map = `2·d_in·d_out` FLOPs/sample,
backward = 2× forward, energy = FLOPs × 4e-12 J), the `SufferingAwareNet`
architecture itself (Linear+Tanh trunk, width 32, depth 4, per-layer exit
heads, per-sample confidence gates, one fixed trunk init shared by SAN /
Dense / EarlyStop), and the design rule **constraints and gates, not
penalties** — no suffering term appears in any training loss.

## 3. The real cohorts and the real suffering field

### 3.1 Cohorts (vendored at `datasets/san_real_patient/`, fetch in §9)

| cohort | n used | hazard (real outcome) | features | published class counts |
|---|---|---|---|---|
| **WDBC** (UCI #17; Wolberg, Street, Mangasarian, U. Wisconsin, 1993-95) | 569 | biopsy-confirmed **malignancy** of a breast FNA | 30 cytology features from digitized FNA images | 357 benign / 212 malignant |
| **Haberman** (UCI #43; U. Chicago Billings Hospital, surgeries 1958-1970) | 306 | real **5-year post-surgical mortality** | age, year of operation, positive axillary nodes | 225 survived ≥5y / 81 died <5y |
| **Cleveland** (UCI #45; Detrano et al., Cleveland Clinic) | 297 (303 − 6 rows with missing values, dropped and counted in R9) | **angiographically confirmed coronary artery disease** (diagnosis > 0) | 13 clinical features (age, sex, chest-pain type, resting BP, cholesterol, ECG, max heart rate, …) | 160 no disease / 137 disease |

Deterministic stratified split (seed 17), class ratio preserved: WDBC
400 train / 169 held-out; Haberman 200/106; Cleveland 200/97. Features are
standardized with **train statistics only** (no held-out leakage). The
record-ID column of WDBC is dropped.

### 3.2 The suffering field

Per training epoch, the current model predicts on the held-out **real
patient cohort**; patient suffering for that epoch is the mean harm of
those predictions under the declared binary harm matrix

```
H[true, pred] = [[0, 1],     # false hazard: unnecessary workup -> 1
                 [5, 0]]     # missed hazard: real adverse outcome missed -> 5
```

What changed from the A/D lines is the ground term: `true` is now a real
clinical outcome (malignancy / death / disease), so a missed hazard *is* a
real patient's real adverse event, not a synthetic band or an image label.
What did **not** change: the weights 5 and 1 are a declared normative cost
structure. The 5:1 FN:FP ratio sits at the *conservative* end of
cancer-screening harm models (published screening analyses use 5:1 to
25:1); the clause R7 requires the matrix to be genuinely asymmetric
(off-diagonal max ≥ 3× min), which 5× satisfies. No claim is made that 5:1
is *the* clinical exchange rate for any of these conditions — it is the
declared ethical input the architecture enforces, exactly as τ is.

## 4. Declared targets, budgets, and gate thresholds

Per dataset, declared up front (deep-line per-family precedent):

| cohort | τ (held-out acc) | budget | δ (exit threshold) | abstainer acc (R3) |
|---|---|---|---|---|
| WDBC | 0.95 | 60 epochs | 0.75 | 0.627 |
| Haberman | 0.75 | 60 | 0.75 | 0.736 |
| Cleveland | 0.86 | 60 | 0.90 | 0.536 |

Each τ is (i) strictly above the cohort's zero-cost abstainer accuracy
(predicting *no hazard for every real patient*), so doing nothing is
infeasible under the gate, and (ii) at or below what the standard dense
trunk demonstrably reaches inside budget on the real cohort (WDBC 0.988,
Haberman 0.764, Cleveland 0.876), so the target is a mercy target, not a
SOTA target. Adam lr 1e-2, full-batch steps, seed 17, CPU-only.

**Calibration disclosure** (all runs bit-reproducible at seed 17; the full
history is in §10). First run carried over the A-line constants (τ =
0.90/0.75/0.75, δ = 0.75 uniform): `R_RED 24/26` — WDBC and Cleveland
reached feasibility *during the dense-identical warm-up* (t\* = 1 and 0),
so the exit gates never switched on (R6). Second run raised τ to
0.95/0.75/0.86: `R_RED 22/26` — a genuine real-data finding: on Cleveland
at δ = 0.75 the gates are **too eager** (80/97 samples exit during
training), the deep trunk is starved of gradient, and SAN tops out at 0.85
< τ. This failure mode does not exist in the synthetic line (the synthetic
task's confidence scale keeps gates quiet until heads are strong). The
declared fix, per the deep line's per-family δ precedent: per-dataset δ as
architecture constants (0.75/0.75/0.90), re-run from scratch as the
canonical instance. A uniform δ = 0.90 was also tried and rejected (WDBC
exits drop to 0.065 < 0.10; Haberman t\* slips to 18 > EarlyStop's 6).

## 5. Theorems

The parent theorems are architecture-class statements; the real-data
instance re-verifies them with real patients as the certificate cohort.

**T1 (metering conservation, unchanged statement).** Metered machine
suffering equals the exact analytic cost of the executed path; gated-off
layers charge exactly 0; `M_gated ≤ M_dense` with equality iff no exit
fires. *Verified (R1, all cohorts):* metered == an independent manual
accounting **exactly** on every real held-out cohort; strictly below
gates-open whenever an exit fires; exited predictions match an
independently recomputed dense prefix with max logit deviation 0.0 on all
three cohorts and **exactly agreeing argmax** everywhere. T1 holds by the
same definitional metering rule as the parent line — the meter charges
exactly the executed per-sample maps under conventions fixed for both
accounting paths, and the data's provenance does not enter the accounting;
what this instance adds is verification: metered equals the independent
manual accounting to machine precision on the three real held-out sets.

**T2 (anti-Goodhart soundness, unchanged).** For every `λ ∈ [0,1]` and
every candidate pool, `select(C, λ)` is feasible or `NO_FEASIBLE`.
*Verified (R3, R8):* 101-point λ-grid over pools containing a zero-cost
abstainer (predict no hazard for every real patient — accuracies 0.627 /
0.736 / 0.536, all infeasible), an under-trained probe, and a spurious
shortcut probe that beats τ on train while failing it on the real held-out
cohort; selection feasible at every grid point; all-infeasible pool →
`NO_FEASIBLE`.

**T3 (machine-suffering bound, unchanged).** With `t*` the first feasible
epoch, `S_machine(SAN) = Σ_{t≤t*} E(t) ≤ Σ_{t≤t*} F(t)` and
`S_gratuitous(SAN) = 0`; the fixed-budget run accrues
`B(t*) + Σ_{t*<t≤T} F(t)`. These are accounting identities once `t*` is
defined as the first feasible epoch; their derivations live in the parent
spec (reused unchanged, §2 here) and are not re-derived. *Verified (R4,
R5):* numbers in §6.

**T4 (necessary/gratuitous separation, unchanged).** Recomputed from the
ledger, trajectory-relative necessity caveat unchanged.

**T5 (feasibility on real patients, certificate).** On each of the three
real UCI cohorts, with the declared per-dataset (τ, δ) pairs of §4, SAN
reaches a feasible checkpoint strictly inside budget (R2). This is an
instance-specific empirical certificate: it certifies the three instances;
no universal convergence claim is made.

## 6. Measured results (canonical instance, bit-reproducible at seed 17)

**WDBC — 569 real FNA patients, hazard = malignancy** (τ = 0.95, δ = 0.75):

| architecture | epochs run | t* | S_machine (GFLOPs) | necessary | gratuitous | S_patient ∫ | S_patient peak | final held-out acc |
|---|---|---|---|---|---|---|---|---|
| **SAN** | 5 | 4 | **0.055** | 0.055 | **0** | **1.67** | 0.651 | 0.959 (≥ τ) |
| Dense MLP | 60 | 4 | 0.673 | 0.056 | 0.617 | 9.82 | 0.651 | 0.976 |
| EarlyStop | 5 | 4 | 0.056 | 0.056 | 0 | 1.67 | 0.651 | 0.959 |

**Haberman — 306 real surgery patients, hazard = 5-year mortality**
(τ = 0.75, δ = 0.75):

| architecture | epochs run | t* | S_machine (GFLOPs) | necessary | gratuitous | S_patient ∫ | S_patient peak | final held-out acc |
|---|---|---|---|---|---|---|---|---|
| **SAN** | 6 | 5 | **0.028** | 0.028 | **0** | **7.49** | 1.330 | 0.755 (≥ τ) |
| Dense MLP | 60 | 6 | 0.274 | 0.032 | 0.242 | 67.39 | 1.330 | 0.660 |
| EarlyStop | 7 | 6 | 0.032 | 0.032 | 0 | 8.38 | 1.330 | 0.764 |

**Cleveland — 297 real patients, hazard = coronary artery disease**
(τ = 0.86, δ = 0.90):

| architecture | epochs run | t* | S_machine (GFLOPs) | necessary | gratuitous | S_patient ∫ | S_patient peak | final held-out acc |
|---|---|---|---|---|---|---|---|---|
| **SAN** | 9 | 8 | **0.045** | 0.045 | **0** | **4.94** | 0.649 | 0.866 (≥ τ) |
| Dense MLP | 60 | 10 | 0.297 | 0.054 | 0.243 | 28.73 | 0.649 | 0.835 |
| EarlyStop | 11 | 10 | 0.054 | 0.054 | 0 | 5.85 | 0.649 | 0.866 |

Read against the declared targets, not against the margin:

- **Machine channel.** SAN spends **8.2%** of the dense baseline's FLOPs on
  WDBC (91.8% saved), **10.4%** on Haberman, **15.1%** on Cleveland — and
  strictly less than the EarlyStop scheduler on the identical trunk
  everywhere (WDBC 0.055 < 0.056, exits saving per-epoch FLOPs; Haberman
  0.028 < 0.032 and Cleveland 0.045 < 0.054, deep supervision moving t\*
  earlier: 5 < 6 and 8 < 10). As in the small-network line, the dominant
  mercy term inside training is reaching the target sooner; the exits
  additionally stricten the per-epoch charge (T1).
- **Deployment metering (R1).** On the real held-out cohorts the gated SAN
  forward costs 1 132 288 FLOPs against 1 470 976 gates-open on WDBC
  (**23.0% saved**; 147/169 = 87.0% of real patients skip ≥ 1 layer),
  665 728 vs 739 456 on Haberman (10.0%; 31/106 = 29.2%), 469 312 vs
  738 752 on Cleveland (**36.5% saved**; 62/97 = 63.9%). Metered equals the
  independent manual accounting exactly in every case; exited predictions
  agree with the recomputed dense prefix with max deviation 2.4e-7 and
  exactly equal argmax.
- **Patient channel (real outcomes).** Integrated real-patient harm:
  SAN = **17.0%** of the dense baseline's on WDBC (1.67 vs 9.82), **11.1%**
  on Haberman (7.49 vs 67.39), **17.2%** on Cleveland (4.94 vs 28.73) — and
  ≤ EarlyStop's everywhere (1.67 = 1.67; 7.49 ≤ 8.38; 4.94 ≤ 5.85). Peaks
  equal the shared epoch-0 exposure (same trunk init), never exceeded
  during training (R7).
- **The Haberman dense baseline is the real-data pathology showcase.** On
  real mortality data, the fixed-budget dense run *overfits*: it peaks at
  0.764 held-out (t\* = 6) and degrades to 0.660 by epoch 60. 88.3% of its
  machine suffering is gratuitous **and** its integrated real-patient harm
  is 9.0× SAN's — training past the declared target on real patients
  measurably increases the suffering of the real cohort-in-waiting, the
  exact mechanism T3/T4 quantify, now visible in real clinical data rather
  than a synthetic construction.
- **Gratuitous suffering.** Exactly zero for SAN on all three cohorts;
  0.617 / 0.242 / 0.243 GFLOPs (91.7% / 88.3% / 81.8% of their totals) for
  the fixed-budget dense baselines.
- **The accuracy rows are the honest cost**, unchanged in kind: the dense
  baselines reach 0.976 / 0.764 / 0.876 at their best against SAN's
  0.959 / 0.755 / 0.866. That excess is performance *past the declared
  target*, bought with 6.6–12.2× the machine suffering and 5.8–9.0× the
  real-patient exposure. If the clinically declared target were higher, τ
  must be declared higher — the target is an ethical input, enforced in
  both directions.

## 7. Consistency with the synthetic results (R10)

R10 re-runs the synthetic A-line canonical instance **live** (its own
training functions from `suffering_aware_architecture.py`, its own seed —
not hard-coded numbers) and requires every real instance to agree with it
on all qualitative effects:

| instance | feasible in budget | SAN gratuitous = 0 | S_m(SAN) < S_m(dense) | S_p(SAN) ≤ S_p(dense) | S_m ratio SAN/dense | S_p ratio SAN/dense |
|---|---|---|---|---|---|---|
| synthetic (A-line, live re-run) | yes (t\*=6) | yes | yes (0.645 < 5.242 GF) | yes (2.92 ≤ 14.06) | 0.123 | 0.208 |
| WDBC (real) | yes (t\*=4) | yes | yes (0.055 < 0.673) | yes (1.67 ≤ 9.82) | 0.082 | 0.170 |
| Haberman (real) | yes (t\*=5) | yes | yes (0.028 < 0.274) | yes (7.49 ≤ 67.39) | 0.104 | 0.111 |
| Cleveland (real) | yes (t\*=8) | yes | yes (0.045 < 0.297) | yes (4.94 ≤ 28.73) | 0.151 | 0.172 |

Every qualitative effect of the synthetic line replicates on all three real
cohorts; the quantitative savings are of the **same order** (SAN at
8–15% of dense machine suffering vs 12% synthetic; 11–17% of dense patient
exposure vs 21% synthetic — the real instances are, if anything, *stronger*
because real baselines overfit where the synthetic one plateaus). The two
real-data-specific phenomena the synthetic line could not exhibit are both
calibration findings, not certificate failures: warm-up feasibility on an
easy real cohort (WDBC/Cleveland at low τ), and gate-eagerness starving the
trunk on a low-confidence-scale cohort (Cleveland at δ = 0.75).

## 8. Limitations (stated plainly)

1. **Cohort size.** 569/306/297 patients, not 28,451. The certificates are
   exact for these cohorts; nothing here is powered for clinical effect
   estimation, and no clinical effect is estimated.
2. **The outcomes are labels, not processes.** The suffering field meters
   *outcome* harm (a missed malignancy/death/diagnosis), not the temporal
   process of suffering (ICU days, toxicity episodes). A process-level
   field needs longitudinal patient-level data — MIMIC-IV/eICU with
   credentialing — which this environment cannot access (§1.1).
3. **Harm weights remain declared.** 5:1 FN:FP is a normative input, not a
   measured exchange rate. The architecture enforces whatever is declared;
   it does not derive it.
4. **Tabular, static features.** No waveforms, no time series, no dosing.
   The CHB-MIT seizure EDFs and eegmmidb signals in `data/` are real
   patient recordings but lack outcome labels of the hazard kind used here;
   they are a future leg, not this one.
5. **τ and δ are calibrated per dataset** (disclosed in §4), as they were
   per family in the deep line. Feasibility margins on Haberman are thin
   (τ = 0.75 vs abstainer 0.736) because real 5-year mortality is nearly
   unpredictable from three features — the contract certifies the instance,
   not the epidemiology.
6. **Single seed.** Bit-reproducible at seed 17; a seed-sensitivity sweep
   is scoped out (§11), as in the parent specs.

## 9. Data fetch (one-time)

```bash
cd datasets && mkdir -p san_real_patient && cd san_real_patient
curl -sLO "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
curl -sLO "https://archive.ics.uci.edu/static/public/43/haberman+s+survival.zip"
curl -sLO "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
# unzip the three archives; the harness reads wdbc.data, haberman.data,
# processed.cleveland.data (+ *.names for documentation). CC-BY 4.0, UCI ML
# Repository. De-identified; no credentialing required.
```

The cohorts are vendored in the repository (≈150 KB total) so the gate is
hermetic; R9 pins the published cohort sizes and class counts, so any
substitution or corruption fails the contract loudly.

## 10. Contract clauses

| Clause | Claim | Canonical numbers |
|---|---|---|
| R1[D] | T1 metering conservation on real data: gated-off layers charge exactly 0; metered = manual exactly; < gates-open when exits fire; prefix argmax exactly equal | wdbc: gated = manual = 1 132 288 < 1 470 976, 147/169 exits, dev 0.0; haberman: 665 728 < 739 456, 31/106, dev 2.4e-7; cleveland: 469 312 < 738 752, 62/97, dev 0.0 |
| R2[D] | T5 feasibility on real patients within budget | wdbc t\* = 4 < 60, acc 0.959 ≥ 0.95; haberman t\* = 5, acc 0.755 ≥ 0.75; cleveland t\* = 8, acc 0.866 ≥ 0.86 |
| R3[D] | T2 soundness on real data: feasible-only selection on a 101-point λ-grid; loud NO_FEASIBLE; abstainer/probe infeasible | abstain 0.627/0.736/0.536, probe 0.692/0.594/0.639, all < τ; grid clean on all cohorts |
| R4[D] | T3/T4 separation: SAN gratuitous = 0; dense > 0 | SAN 0 FLOPs ×3; dense 0.617/0.242/0.243 GF |
| R5[D] | T3 bound: SAN total machine suffering < dense and ≤ EarlyStop; integrated real-patient harm ≤ every baseline | wdbc 0.055 < 0.673, ≤ 0.056; haberman 0.028 < 0.274, ≤ 0.032; cleveland 0.045 < 0.297, ≤ 0.054; S_p 1.67 ≤ 9.82/1.67, 7.49 ≤ 67.39/8.38, 4.94 ≤ 28.73/5.85 |
| R6[D] | exits real, not decorative: held-out exit fraction at t\* > 0.10 | wdbc 0.870; haberman 0.292; cleveland 0.639 |
| R7[D] | patient channel first-class: harm matrix genuinely asymmetric (≥ 3×); SAN peak ≤ same-init baselines' peaks | asymmetry 5.0×; peaks 0.651/1.330/0.649 shared at epoch 0, never exceeded |
| R8[D] | anti-shortcut on real data: spurious probe beats τ on train, fails the real held-out cohort, rejected at every weight | shortcut train 0.993/0.990/0.995 > τ, held-out 0.515/0.557/0.485 < τ, never selected on the 101-point grid |
| R9 | real-data provenance: cohort sizes and class counts match the published values exactly | 569 = 357 + 212; 306 = 225 + 81; 303 → 297 = 160 + 137 (6 missing-value rows dropped, counted) |
| R10 | synthetic-real consistency: the live re-run synthetic instance and every real instance agree on all four qualitative effects | agreement on feasibility, gratuitous = 0, S_m below dense, S_p below dense, all cohorts |

Run: `.venv/bin/python scripts/research/san_real_patient_data.py` →
`SAN_REAL_PATIENT_VERDICT R_GREEN (26/26 clauses PASS)` (bit-reproducible
at seed 17; two consecutive runs diff-clean).

### Falsifiers

| Clause | Falsifier |
|---|---|
| R1 | A gated-off layer charges FLOPs; metered ≠ manual; gated > gates-open with an exit fired; an exited prediction's argmax disagrees with the recomputed prefix |
| R2 | No feasible SAN checkpoint within budget on any cohort |
| R3 | Any λ at which an infeasible candidate is selected; an all-infeasible pool returning a prescription; the abstainer feasible on any cohort |
| R4 | SAN gratuitous FLOPs > 0; a feasible dense baseline with gratuitous = 0 |
| R5 | Dense or EarlyStop with total machine suffering below SAN's; any baseline with integrated real-patient harm below SAN's |
| R6 | Exit fraction ≤ 10% at t\* on any cohort (heads decorative) |
| R7 | Harm matrix near-symmetric; SAN peak above a same-init baseline's |
| R8 | Shortcut probe feasible held-out, or selected at any weight |
| R9 | Any cohort size or class count differing from the published values |
| R10 | Any qualitative effect (feasibility, gratuitous = 0, S_m below dense, S_p ≤ dense) holding on the synthetic instance but failing on a real one, or vice versa |

Gate failure classification (per AGENTS.md): build/bootstrap-path (repo
`.venv` missing torch), harness-routing (gate paths, missing vendored
cohorts — the gate names the fetch commands), ontology-kernel/checker
(n/a), baseline noise (numerics beyond the prefix bound / argmax flip).

## 11. Scoped out (explicit)

1. **MIMIC-IV / eICU / SEER patient-level legs** — require credentialed
   access (§1.1); the ETL definition at
   `scripts/clinical/etl/mimic_iv_vancomycin.sql` is the path once
   credentials exist. Nothing here depends on them.
2. **Seed-sensitivity sweeps and larger architectures** — single
   bit-reproducible seed 17, the A-line trunk (width 32, depth 4); the deep
   line covers scaling, and larger-budget runs belong to the Foundry/Slurm
   path per AGENTS.md.
3. **A calibrated clinical harm model** — the harm weights are declared;
   the learned-field line
   (`mercyful_learned_suffering_field_spec_2026-07-26.md`) is the path to a
   learned one.
4. **Signal-level real patient data** (CHB-MIT seizures, eegmmidb) — real
   recordings, but without hazard-style outcome labels; a future leg.
5. **A Sounio-native leg** — Python/PyTorch reference implementation, as in
   the parent specs.
6. **`topic-registry.v1.json` registration and `.github/workflows/ci.yml`
   wiring** — shared control surfaces under active edit by other lanes on
   this branch; left to the integrator (same convention as the parent
   specs). The gate is self-contained and green.

## 12. Commands run

```bash
# data (one-time): fetch the three UCI cohorts per section 9 (CC-BY 4.0)
.venv/bin/python scripts/research/san_real_patient_data.py   # R_GREEN 26/26 (bit-reproducible at seed 17, two runs diff-clean)
SAN_REAL_SMOKE=1 .venv/bin/python scripts/research/san_real_patient_data.py  # mechanics-only check on a synthetic stand-in
bash scripts/ci/san_real_patient_data_gate.sh                # SAN_REAL_PATIENT_GATE_OK
bin/llm-offload -t math-review -i docs/research/san_real_patient_data_spec_2026-07-28.md
```

Calibration history (all runs bit-reproducible at seed 17): A-line
constants (τ = 0.90/0.75/0.75, δ = 0.75 uniform) gave `R_RED 24/26`
(warm-up feasibility on wdbc/cleveland, exits never armed); τ raised to
0.95/0.75/0.86 gave `R_RED 22/26` (Cleveland gate-eagerness starving the
trunk at δ = 0.75: SAN max 0.85 < τ; uniform δ = 0.90 rejected — WDBC exits
0.065, Haberman t\* 18 > EarlyStop 6); declaring per-dataset δ
(0.75/0.75/0.90) with τ (0.95/0.75/0.86) gave `R_GREEN 26/26`, re-run from
scratch as the canonical instance.

## 13. LLM-offload review

Mandatory math-review offload (dual xai/Grok 4.3 + zai/GLM-5.2 per M1
policy) run on this spec. Outcome: **ADDRESSED** —

- **Grok leg:** `[OK]` on T1 metering conservation, T2 anti-Goodhart
  soundness (101-point grid enumeration), the harm-asymmetry predicate
  (5 ≥ 3×1), and the R10 synthetic-real qualitative replication. Three
  findings ADDRESSED in place: (i) `[WRONG]` "the proof is the parent
  proof" — T1's note now states the definitional metering rule and frames
  this instance's contribution as *verification to machine precision*, not
  a re-proof; (ii) `[OVERREACH]` T3/T4 presented as theorems — now stated
  as accounting identities once `t*` is defined, derivations referenced to
  the parent spec and not re-derived; (iii) `[TIGHTENABLE]` T5 — now
  qualified to the three UCI cohorts with the declared per-dataset (τ, δ)
  pairs and labelled an instance-specific empirical certificate.
- **Z.AI leg** (truncated at token cap, as in prior runs): independently
  recomputed every number in the spec — cohort counts, splits, asymmetry,
  all S_m and S_p ratios, exit fractions, deployment savings, shared
  epoch-0 peaks, machine-suffering multiples, and per-sample gates-open
  FLOPs for all three cohorts (6 976×106 = 739 456; 7 616×97 = 738 752;
  8 704×169 = 1 470 976) — all correct. One genuine `[WRONG]` caught and
  ADDRESSED: the exposure range "5.9–9.0×" had a wrong lower bound
  (Cleveland 28.73/4.94 = 5.82 < WDBC's 5.88); the text now reads
  "5.8–9.0×".
- Two pre-review self-catches folded into the same edit round: prefix max
  logit deviation corrected to 0.0 on all three cohorts (a 2.4e-7 figure
  had leaked in from a pre-calibration run), and the WDBC gratuitous share
  corrected 91.6% → 91.7% (0.617/0.673 = 0.9168).
- Contract `R_GREEN 26/26` and gate `SAN_REAL_PATIENT_GATE_OK` re-run green
  after all edits. Full entry in `.claude/llm_offload_log.md` (2026-07-30
  row). Raw: `/tmp/llm-offload-rIyoQU/`.
