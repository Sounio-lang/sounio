<!-- docs:meta
topic_id: repo.docs.ppcr.claims-ledger
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ppcr.claims-ledger
-->

# Claims Ledger — Sounio × PPCR for Prof. Felipe Fregni

**Rule:** if a claim is not in this ledger as `VERIFIED`, do not send it or say it in the meeting.

---

## Legend

- `VERIFIED` — compiles/runs/output matches reference; evidence path provided.
- `PARTIAL` — something real exists, but it is incomplete, unvalidated, or needs a non-default engine.
- `DO-NOT-CLAIM` — aspirational, documented only, or factually unsupported.

---

## 1. Epistemic confidence gate and provenance

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "Sounio has a `confidence_gate` module in the stdlib." | VERIFIED | `stdlib/epistemic/confidence_gate.sio:43-48` defines `GatePolicy`; `:112-167` defines `gate_check_value`; `:200-213` defines `confidence_gate`. | Passes `bin/souc check stdlib/epistemic/confidence_gate.sio`. |
| "The confidence gate rejects inputs below a configurable confidence threshold." | VERIFIED | `demo/fregni/fregni_demo.sio` scenario 2 rejects `simulation(7500, 700)` against `min_conf = 950`. | Output in `demo/fregni/OUTPUT.md`. |
| "Sounio has path/fermentation wrappers that distinguish measured, imputed, and simulated values." | VERIFIED | `stdlib/epistemic/path.sio:32-57` defines `MeasuredGUMI64`, `ImputedModelI64`, `SimulationI64`; `:136-152` defines constructors. | Passes `bin/souc check stdlib/epistemic/path.sio`. |
| "The path wrappers give a compile-time provenance guard." | VERIFIED | `demo/fregni/bad_path.sio` fails `bin/souc check` with `expected MeasuredGUMI64, found SimulationI64`. | Output in `demo/fregni/OUTPUT.md`. |
| "The stdlib epistemic modules can be imported and run directly via `bin/souc run`." | DO-NOT-CLAIM | `tests/run-pass/epistemic_fermentation.sio` passes `check` but `bin/souc run` segfaults in Madaros v0.80.0. | The demo therefore inlines the relevant pieces. |
| "Sounio enforces confidence thresholds at compile time." | DO-NOT-CLAIM | `confidence_gate.sio:14-15` explicitly says compile-time enforcement is future work. | Current gate is runtime. |
| "The epistemic path wrappers are cryptographically tamper-proof." | DO-NOT-CLAIM | `stdlib/epistemic/path.sio:84-85` and `:155-157` state labels are audit-only, not anti-forgery tokens. | |

---

## 2. Clinical dosing demo

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "`demo/fregni/fregni_demo.sio` runs end-to-end with one command." | VERIFIED | `bash demo/fregni/run.sh` produces output captured in `demo/fregni/OUTPUT.md`. | |
| "The demo computes a vancomycin-style AUC-based daily dose of 32000 mg for CL = 80 L/h and target AUC = 400 mg·h/L." | VERIFIED | `demo/fregni/reference.py` computes `400 * 80 / 1 = 32000.00`; `demo/fregni/fregni_demo.sio` prints `32000.0`. | Dose is integer-scaled (x100) inside Sounio. |
| "The demo rejects a simulated clearance with 700 permille confidence against a 950 permille threshold." | VERIFIED | `demo/fregni/OUTPUT.md` shows `Scenario 2 ... FAIL`. | |
| "The demo rejects an imputed clearance whose CV (30%) exceeds the 25% threshold even though its confidence (960‰) passes." | VERIFIED | `demo/fregni/reference.py` computes CV = 30.00%; `demo/fregni/OUTPUT.md` shows `Scenario 3 ... FAIL`. | |
| "The demo is a validated clinical dosing tool." | DO-NOT-CLAIM | It is a teaching illustration. It uses scaled integer arithmetic and does not implement patient-specific factors, renal adjustment nomograms, or therapeutic drug monitoring. | |

---

## 3. Study design and estimands (ICH E9(R1))

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "`examples/clinical_trial_epistemic.sio` mentions ICH E9(R1)." | VERIFIED | `examples/clinical_trial_epistemic.sio:13` contains the text "ICH E9(R1) Statistical Principles for Clinical Trials". | |
| "Sounio implements the five ICH E9(R1) estimand strategies." | DO-NOT-CLAIM | No `Estimand` type, no population/treatment/variable/intercurrent-event/summary-measure decomposition, and no strategy implementation found. | |
| "The clinical-trial example runs and produces validated output." | DO-NOT-CLAIM | `bin/souc check examples/clinical_trial_epistemic.sio` passes, but `bin/souc run` segfaults. | |

---

## 4. Sample size and power

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "`stdlib/stats/clinical/power_analysis.sio` contains formulas for two-sample t, paired t, two proportions, and log-rank sample size." | VERIFIED | `stdlib/stats/clinical/power_analysis.sio:126-152` (two-sample t), `:158-183` (paired t), `:189-234` (two proportions), `:240-283` (log-rank). | |
| "The power module compiles on the default `bin/souc` engine." | DO-NOT-CLAIM | `bin/souc check stdlib/stats/clinical/power_analysis.sio` fails with `String`/`string` and `i32`/`i64` errors. | Compiles only with `SOUNIO_SOUC_ENGINE=lean_single`. |
| "The power module is validated against R or Python reference output." | DO-NOT-CLAIM | No reference test found in repo. | |
| "Unequal allocation ratios are supported." | DO-NOT-CLAIM | All sample-size functions hard-code equal group sizes (`n_per_group = total/2`). | |

---

## 5. Randomization

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "Sounio implements simple, permuted-block, stratified, or minimisation randomization." | DO-NOT-CLAIM | Exact searches for these terms returned zero matches in `stdlib/`, `examples/`, `formal/`, and `docs/`. | |
| "Sounio has generic sampling primitives." | VERIFIED | `stdlib/random/sampling.sio:150-189` implements Fisher–Yates shuffle; `:117-144` implements weighted sampling. | Not clinical randomization. |

---

## 6. Blinding indices

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "Sounio implements Bang's Blinding Index or James' index." | DO-NOT-CLAIM | Exact searches for `Bang's`, `James'`, `blinding index`, `Hróbjartsson`, and `Kolahi` returned zero matches. | |

---

## 7. Survival analysis

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "`stdlib/medical/survival.sio` contains KM, Nelson–Aalen, Cox PH, and log-rank code." | VERIFIED | `stdlib/medical/survival.sio:186-247` (KM), `:296-341` (Nelson–Aalen), `:353-451` (Cox), `:477-546` (log-rank). | |
| "The survival module compiles on the default engine." | DO-NOT-CLAIM | `bin/souc check stdlib/medical/survival.sio` fails with field/type errors. | Compiles only with `SOUNIO_SOUC_ENGINE=lean_single`. |
| "The survival module is validated against R `survival` or Python `lifelines`." | DO-NOT-CLAIM | No reference validation found. | |
| "The log-rank implementation is the standard Mantel–Haenszel procedure." | DO-NOT-CLAIM | `stdlib/medical/survival.sio:477-546` uses pooled group proportions, not per-time risk sets. | |
| "The Cox PH standard errors are exact inverse-Hessian." | DO-NOT-CLAIM | `:444-447` uses an approximation. | |

---

## 8. Causal inference

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "Sounio has a verified Lean 4 causal module covering do-calculus, d-separation, confounders, and identifiability." | VERIFIED | `formal/lean4/SounioCausality.lean:106-157` (do-operator), `:167-184` (d-separation), `:410-428` (confounders), `:454-460` (identifiability). | `lake build SounioCausality` succeeds with no `sorry`. |
| "The native Sounio causal code runs under `bin/souc`." | DO-NOT-CLAIM | `stdlib/causal/do_calculus.sio` fails `bin/souc check` with `parse_failed`; `stdlib/epistemic/causal.sio` has type errors. | |
| "The causal module estimates effects from data." | DO-NOT-CLAIM | The Lean module is graph-theoretic/semantic; it does not ingest data or estimate ATEs. | |

---

## 9. Bayesian / adaptive designs

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "`stdlib/medical/trial.sio` contains interim analysis, efficacy/futility stopping, predictive success, and operating-characteristics code." | VERIFIED | `stdlib/medical/trial.sio:393-451` (interim), `:422-429` (efficacy stop), `:431-448` (futility stop), `:599-681` (predictive success), `:731-781` (operating characteristics). | |
| "The adaptive-trial module compiles and runs." | DO-NOT-CLAIM | `bin/souc check stdlib/medical/trial.sio` fails with `i64`/`i32` type errors. | |

---

## 10. Regulatory hooks

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "Sounio has an ISO 17025-style calibration module that type-checks." | VERIFIED | `stdlib/metrology/calibration.sio:1-353`; `bin/souc check stdlib/metrology/calibration.sio` passes. | |
| "Sounio has a `RegulatoryContext` struct with a `cfr_part11` field." | VERIFIED | `stdlib/epistemic/regulatory.sio:1-26`. | Passes `bin/souc check`. |
| "21 CFR Part 11 is enforced by the Sounio type system or compiler." | DO-NOT-CLAIM | `stdlib/epistemic/regulatory.sio` is a struct with boolean flags; no electronic signatures, user identity, record locking, or ALCOA enforcement exists. | |
| "Sounio implements FAIR principles in code." | DO-NOT-CLAIM | No code module for Findability, Accessibility, Interoperability, or Reusability found; FAIR appears only in `docs/MANIFESTO.md` and website i18n prose. | |
| "The audit trail is tamper-evident." | DO-NOT-CLAIM | `stdlib/epistemic/audit_runtime.sio` has a circular log with a `regulatory_layer` slot, but no tamper-evidence binding. | Currently type-broken under default `bin/souc`. |

---

## 11. Meta-claims about the repository

| Claim | Tag | Evidence | Notes |
|---|---|---|---|
| "Sounio is a self-hosted compiler." | VERIFIED | `./bin/souc --version` prints `Madares v0.80.0 -- the Sounio self-hosted compiler`. | |
| "The PPCR demo uses only code that compiles and runs on the current `bin/souc`." | VERIFIED | `bash demo/fregni/run.sh` succeeds; output in `demo/fregni/OUTPUT.md`. | The demo inlines small pieces of `confidence_gate.sio` and `path.sio` because the stdlib import path segfaults at runtime. |
| "Sounio has an 89% stdlib completeness figure." | DO-NOT-CLAIM | The 814/910 figure is harness inventory, not stdlib completeness. Do not repeat it. | Per repository caveat and AGENTS.md. |

---

## Allowed talking points

- "We can demonstrate a confidence-gated clinical dosing pipeline today; the compiler rejects the wrong provenance at type-check time."
- "We have a machine-checked causal inference formalisation in Lean 4."
- "We have ISO 17025-style calibration code that runs."
- "Randomisation, blinding indices, validated survival analysis, and enforced 21 CFR Part 11 are not yet implemented — those are exactly what a PPCR collaboration would build on top of the epistemic foundation."
