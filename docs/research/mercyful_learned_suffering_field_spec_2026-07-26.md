<!-- docs:meta
topic_id: repo.docs.research.mercyful-learned-suffering-field-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-learned-suffering-field-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — Learned Suffering Field s(v) (patient + machine)

**Date:** 2026-07-26
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` (contract L1..L8, gate green)
**Parent:** `docs/research/PROGRAM-REGISTRY-mercyful-learning.md`
**Harness:** `scripts/research/mercyful_learned_field_contract.py`
**Model:** `scripts/research/mercyful_suffering_field_learned.py`
**Frozen coefficients:** `scripts/research/mercyful_learned_field_coefficients_v1.json`
**Gate:** `scripts/ci/mercyful_learned_field_gate.sh`

> **Scope.** The suffering field in this document is **learned from synthetic
> training data** (the repository's popPK-driven synthetic cohort v2) anchored
> to **published** population-pharmacokinetic parameters and **published**
> MIMIC-IV cohort statistics. No real patient records were used. This is a
> research prototype, **not medical guidance**, and it carries no clinical
> claim. The machine-suffering component is a real measurement of this
> repository's own scheduler.

---

## 1. What this is

Every suffering field in the Mercyful Learning program so far has been
**declared**: algebra-derived (`geodesic_mercy.py`), PK-band-derived
(`vanco_suffering` in `tests/run-pass/mercyful_clinical_sequencing.sio`), or
hand-written (chemo `SUFFERING` dict, exposure-therapy ordinals). This
document specifies the first **learned** field: a model `s(v)` estimated from
data — patient state `v` plus treatment — and validated against the synthetic
benchmarks, while keeping the program's expanded ethics explicit:

```
S(v) = s_patient(v) + λ_m · s_machine(v)
```

The patient component is learned (logistic outcome models + Knightian
uncertainty). The machine component is measured (energy of the scheduler's
own deliberation). Minimizing suffering means minimizing **both**.

## 2. Data provenance (honest inventory)

| Source | Status | Use |
|---|---|---|
| `scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv` | present (200 patients, seed 20260501) | **training surface** |
| Roberts et al. 2011 ICU vancomycin popPK parameters | published | forward model for counterfactual regimens |
| Wang et al. 2026, MIMIC-IV v3.1 (doi:10.1038/s41598-026-42395-1) | published statistics | direction anchors only (TDM ⇓ mortality/AKI) |
| Real MIMIC-IV extract | **credential-gated** (`scripts/clinical/etl/`, CITI + DUA pending) | none yet; retrain through this pipeline when it lands |
| FAERS (in-repo files) | **audited NEGATIVE** (`docs/research/faers_mercyful_analysis_2026-07-26.md`): no reaction terms, no seriousness flags, no doses, no vancomycin rows | deliberately unused; the audit is the negative-provenance record |
| Scheduler wall-clock / evaluation counts | measured live in this repo | machine suffering (§6) |

Claiming FAERS or MIMIC-IV support beyond this table is a falsification of
this spec (clause L8).

## 3. The mercyful functional (recap)

Paths `γ` through a state graph reach a target under budget `len(γ) ≤ L0`;
cost is integrated plus peak suffering:

```
cost(γ; μ) = ∫_γ S dℓ + μ · max_{v∈γ} S(v)
γ* = argmin_{γ : start→target, len(γ)≤L0} cost(γ; μ)
```

A path not reaching the target is infeasible regardless of cost (anti-Goodhart).
Budgetary necessity `c*(L0) = inf_{len(γ)≤L0} max_t S(γ(t))` separates
necessary from gratuitous suffering. This document only changes how `S` is
**obtained**; the functional and scheduler are untouched and re-verified
(clause L6).

## 4. The learned patient field

### 4.1 Outcome models (learned)

From the cohort we learn two logistic regressions by IRLS (deterministic,
pure stdlib, ridge jitter 1e-8, tol 1e-12), on features
`x = (1, Cmin/20, SOFA/10, nephro, CrCl/100)` with fixed scalings:

```
P_aki(x)  = σ(β_a·x)      P_cure(x) = σ(β_c·x)
```

Learned coefficients (frozen in the coefficient artifact, bit-reproducible
per clause L2; signs as clinically required: more Cmin → more AKI **and**
more cure — the suffering-relevant trade-off is learned, not asserted):

```
β_a = (−4.890758, +3.602368, −0.483043, +0.683687, +0.499115)
β_c = (−2.669133, +6.084067, −0.375712, +0.916181, −0.135567)
```

Held-out Brier skill vs base rate (deterministic split, every 4th row):
0.463 for the AKI model.

### 4.2 The harm functional (declared ethical weights)

```
h(c, u) = W_AKI · P_aki(c, u) + W_FAIL · (1 − P_cure(c, u))
```

`W_AKI = W_FAIL = 1`: one unit of averted organ injury counts as one unit of
averted treatment failure. These are **declared ethical-priority constants**,
not fitted; changing them is a semantic change requiring spec revision.

### 4.3 Aleatoric expectation + epistemic penalty

Pre-TDM, the trough is not measured; its distribution follows the published
popPK variability, `Ω² = ω_V² + ω_CL² + σ_prop² = 0.09 + 0.09 + 0.04 = 0.22`
(log-normal around the θ-only point prediction `Ĉmin_ss` from the Roberts
2011 forward model). Post-TDM the measured level collapses Ω to zero.

```
s_patient(r, u) = E_Ω[h] + γ · SD_Ω[h],     γ = 1
```

`E_Ω` and `SD_Ω` are computed by exact 5-point Gauss–Hermite quadrature
(fixed nodes/weights, deterministic). Suffering is thus **expected harm plus
the price of not knowing**: measurement (TDM) removes the spread term. This
is the learned analogue of the synthetic band-width field `s_window`, which
encodes the same idea geometrically.

## 5. Uncertainty: a p-box on the field itself

### 5.1 Epistemic interval

IRLS yields the Fisher covariance `Σ = (XᵀWX)⁻¹` at convergence. By the
delta method each logit carries standard error `SE(x) = √(xᵀΣx)`, and
monotonicity of `σ` turns the ±z interval on the logit into an interval on
each probability (z = 1.959964). Combining pessimistic/optimistic ends:

```
s_lo = W_AKI·σ(η_a − z·SE_a) + W_FAIL·(1 − σ(η_c + z·SE_c))
s_hi = W_AKI·σ(η_a + z·SE_a) + W_FAIL·(1 − σ(η_c − z·SE_c))
```

### 5.2 Knightian reading

`[s_lo, s_hi]` is an interval-valued (p-box) suffering field: the scheduler
may plan on `s_hi` (caution) or inspect the width (epistemic suffering).
Pre-TDM states have wide intervals **and** pay the Ω-spread; TDM collapses
both. This is the same "measurement reduces suffering" theorem as synthetic
clause V3, now derived from data.

### 5.3 Declared approximation

For pre-TDM states the epistemic interval is evaluated at the median
`Ĉmin_ss` and conservatively widened to contain the aleatoric expectation
`E_Ω[h]`, then shifted by `γ·SD_Ω[h]`. This keeps `s_lo ≤ s ≤ s_hi` by
construction (checked on all anchors by clause L3).

## 6. Machine suffering (expanded ethics, measured)

The machine suffers too: deliberation costs computation and energy. We do
not poeticize this; we meter it.

```
energy(v)  = n_evals · τ_ref · P        (deterministic proxy)
s_machine(v) = energy(v) / E_ref         per-evaluation: λ_m·τ_ref·P/E_ref
S(v) = s_patient(v) + λ_m · s_machine(v)
```

Declared calibration: `P = 15 W` (laptop-class package), `τ_ref = 50 µs`
per field evaluation (conservative; measured ≈ 50 µs on the development
host), `E_ref = 1 J`, exchange rate `λ_m = 0.01` — patient suffering
dominates, machine suffering is **counted, honestly small, and never
hidden**. The contract asserts the proxy (evaluation counts are
deterministic), not the wall clock, so the gate is machine-independent.
Wall-clock energy is printed for reporting only. Falsifier for the whole
component: if `λ_m = 0`, the expanded ethics is decorative; clause L3
requires the decomposition to hold exactly.

## 7. Benchmark anchors

### 7.1 Reference patient

Cohort-typical ICU patient: 75 kg, CrCl 80 mL/min, SOFA 7, no nephrotoxic
co-exposure. Declared; anchor regimens are evaluated on this patient.

### 7.2 Anchor regimens and learned values

| State | Regimen | Learned s | Synthetic analog |
|---|---|---|---|
| VANCO_PRE | 1000 mg q12h, no TDM | 0.637356 | 0.675679 |
| TDM_GUIDED | measured Cmin = 15 mg/L | 0.285521 | 0.059420 |
| FIXED_LOW | 500 mg q24h, no TDM | 0.954318 | 0.600000 |
| FIXED_STD | 1500 mg q12h, no TDM | 0.724605 | 0.700000 |

### 7.3 Findings (where learned and synthetic agree — and where not)

- **Agreement (decision level):** identical scheduler decisions on the
  MIMIC-IV graph topology, both gated and counterfactual-open (clause L6);
  Spearman ρ = 0.879 against the window-based teacher across the cohort
  (clause L7); TDM strictly reduces suffering (clause L4).
- **Agreement (qualitative):** FIXED_LOW is the worst arm and is
  **failure-dominated** (fail share 0.921 vs AKI share 0.011): the learned
  field independently reproduces the program's central anti-Goodhart
  finding — underdosing is the worst suffering, invisible to a
  toxicity-only metric (clause L5).
- **Honest divergence:** the learned pre/post TDM ratio is **2.23×**, not
  the synthetic **11.37×**. The synthetic field is exactly zero inside the
  window; the learned field retains irreducible baseline harm at Cmin = 15
  (host factors, residual AKI/cure risk learned from the cohort). The
  learned field is *less* optimistic about what measurement buys. Both
  agree on direction and on every decision; they disagree on the magnitude
  of the residue. We report this rather than tune it away.

## 8. Contract clauses, falsifiers, stop rules

Harness: `scripts/research/mercyful_learned_field_contract.py`
(verdict `MERCYFUL_LEARNED_FIELD_VERDICT L_GREEN`).

- **L1_DATA_PROVENANCE** — cohort present, 200 rows, exact 14-column schema,
  plausible outcome rates; FAERS negative audit on file; scope guards in the
  contract source. *Falsifier:* any missing/mutated input or missing guard.
  *Stop rule:* do not retrain on unvetted data to make L1 pass.
- **L2_OUTCOME_MODELS_LEARN** — IRLS converges; coefficients bit-match the
  frozen artifact (1e-12); held-out Brier beats base rate; `β_Cmin > 0` in
  both models. *Falsifier:* non-determinism, no skill, or a sign flip.
  *Stop rule:* if the frozen artifact and source disagree, believe the
  failure, not either artifact; investigate before re-freezing.
- **L3_FIELD_DECOMPOSITION** — `S = s_patient + λ_m·s_machine` holds exactly;
  machine term > 0; `s_lo ≤ s ≤ s_hi` on every anchor. *Falsifier:* machine
  suffering zeroed, or an ill-formed p-box. *Stop rule:* never "fix" L3 by
  setting `λ_m = 0`.
- **L4_TDM_NARROWS_LEARNED_FIELD** — `s_post < s_pre`, ratio > 2.0 (declared;
  synthetic analog 11.37×, divergence documented in §7.3). *Falsifier:*
  measurement fails to reduce learned suffering. *Stop rule:* do not raise
  `γ` to inflate the ratio.
- **L5_ANCHOR_ORDERING** — `s(FIXED_LOW) > s(FIXED_STD) > s(TDM_GUIDED)` and
  FIXED_LOW is failure-dominated. *Falsifier:* underdosing stops being the
  worst arm. *Stop rule:* a change here is a semantic change; revisit §4.2
  weights only via spec revision.
- **L6_SCHEDULER_EQUIVALENCE** — with learned state sufferings on the
  synthetic MIMIC-IV graph topology: gated graph selects
  `START→VANCO_PRE→TDM_GUIDED→TARGET`; counterfactual open graph selects
  `START→FIXED_STD→TARGET` (the verify gate remains causal, cf. V4);
  exposure-therapy benchmark still routes through `moderate`. *Falsifier:*
  any decision change. *Stop rule:* do not edit graph topology to restore a
  decision.
- **L7_TEACHER_RANK_AGREEMENT** — Spearman ρ ≥ 0.70 between learned and
  teacher fields across all 200 cohort patients. *Falsifier:* rank divorce
  from the synthetic benchmark. *Stop rule:* report low ρ as a finding (as
  §7.3 does for the ratio); do not add teacher-fitted calibration terms.
- **L8_NO_OVERREACH** — spec carries `synthetic`, `not medical guidance`,
  the FAERS negative-provenance citation, the MIMIC-IV credential status,
  and a machine-suffering section. *Falsifier:* any missing guard.
  *Stop rule:* none — guards are unconditional.

## 9. Limitations and retraining path

1. Training data are **synthetic**; the learned field inherits the cohort
   generator's outcome models. What is genuinely learned (not declared) is
   the *mapping* from patient state and regimen to suffering: coefficients,
   intervals, and all decisions above.
2. Real MIMIC-IV: when the credential-gated extract
   (`scripts/clinical/etl/mimic_iv_vancomycin.sql`) lands, retrain through
   this exact pipeline, re-freeze coefficients, and expect L4/L5/L7 numbers
   to move; the spec must then be revised with real-data provenance.
3. FAERS remains unusable until severity-graded extracts are imported
   (see the negative audit's path-to-positive section).
4. A Sounio twin of the learned field (port of the logistic models and
   quadrature to `stdlib/clinical/`) is **out of scope** here and is the
   natural follow-up; until then the learned field lives in the Python
   harness only.
5. The machine-suffering calibration (τ_ref, P) is a declared conservative
   bound, not a precision measurement; an energy-metered run on calibrated
   hardware is future work.

---

*All values synthetic unless marked measured; not medical guidance.*
