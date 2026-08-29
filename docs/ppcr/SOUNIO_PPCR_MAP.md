<!-- docs:meta
topic_id: repo.docs.ppcr.sounio-ppcr-map
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ppcr.sounio-ppcr-map
-->

# Sounio × PPCR Capability Map

**Scope:** map each PPCR competency to the actual Sounio artifact that implements it, with an honest maturity tag, a runnable entrypoint, and a confidence level. This map is the ground truth for everything that may be said to Prof. Felipe Fregni.

**Maturity tags**

- `WORKS` — compiles and runs under `bin/souc` (or `lake build` for Lean) with validated output.
- `PARTIAL` — compiles or type-checks, but is incomplete, unvalidated, or needs a non-default engine.
- `STUB` — placeholder / hard-coded smoke test / demo-only.
- `ASPIRATIONAL` — spec, comment, or documented intent; no runnable code.

**Confidence levels**

- `High` — direct, inspectable match between the competency and the code.
- `Medium` — the code addresses the topic, but gaps (compiler, validation, scope) remain.
- `Low` — mostly naming or commentary; little or no implementation.

---

| PPCR competency | Sounio artifact (path:lines) | Maturity | Confidence | What actually runs |
|---|---|---|---|---|
| 1. Study design & estimands (ICH E9(R1)) | `examples/clinical_trial_epistemic.sio:13` names ICH E9(R1); `stdlib/medical/trial.sio:1-781` has adaptive-trial structs; `tests/frontend/potential_outcomes_basic.sio:12` has `ATE<T>` | STUB | Low | `bin/souc check examples/clinical_trial_epistemic.sio` passes, but `bin/souc run` segfaults; no ICH E9(R1) estimand framework is implemented. |
| 2. Sample size & power (means, proportions, log-rank) | `stdlib/stats/clinical/power_analysis.sio:126-152` (two-sample t), `:158-183` (paired t), `:189-234` (two proportions), `:240-283` (log-rank); `stdlib/medical/survival.sio:186-247` (KM), `:353-451` (Cox), `:477-546` (log-rank test) | PARTIAL | Medium | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc stdlib/stats/clinical/power_analysis.sio /tmp/out` compiles; default `bin/souc check` fails with `String`/`string` and `i32`/`i64` mismatches. No external validation found. Unequal allocation is not implemented. |
| 3. Randomization (simple, block, stratified, minimisation) | No module implements these four schemes. Closest: `stdlib/medical/trial.sio:684-716` (response-adaptive weights), `stdlib/random/sampling.sio:150-189` (Fisher–Yates shuffle) | ASPIRATIONAL | Low | Nothing runnable for the requested competency. `stdlib/medical/trial.sio` does not compile on the default engine. |
| 4. Blinding & blinding-success indices (Bang, James, Hróbjartsson, Kolahi) | No artifact found | ASPIRATIONAL | High | No references to Bang's Blinding Index, James' index, or the cited literature anywhere in `stdlib/`, `examples/`, `formal/`, or `docs/`. |
| 5. Survival analysis (Kaplan–Meier, Cox PH, log-rank) | `stdlib/medical/survival.sio:186-247` (KM), `:254-288` (CI band), `:296-341` (Nelson–Aalen), `:353-451` (Cox), `:477-546` (log-rank), `:568-611` (RMST); `stdlib/viz/epiviz.sio:728-835` (KM curve) | PARTIAL | Medium | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc stdlib/medical/survival.sio /tmp/out` compiles; default `bin/souc check` fails. No reference validation; Cox SE is approximate; log-rank is simplified, not Mantel–Haenszel. |
| 6. Causal inference (do-calculus, confounding) | **Formal:** `formal/lean4/SounioCausality.lean:106-157` (do-operator), `:167-184` (d-separation), `:410-428` (confounders), `:454-460` (identifiability). **Native:** `stdlib/causal/do_calculus.sio:273-368` (backdoor), `:379-552` (ATE), `:554-644` (IV) | WORKS (Lean) / STUB (native) | High (Lean), Low (native) | `cd formal/lean4 && lake build SounioCausality` — verified, no `sorry`. Native `.sio` files currently `parse_failed` or type-fail under `bin/souc`. |
| 7. Bayesian / adaptive designs (interim, futility) | `stdlib/medical/trial.sio:393-451` (interim analysis), `:422-429` (efficacy stop), `:431-448` (futility stop), `:599-681` (predictive success), `:731-781` (operating characteristics) | STUB | Medium | `bin/souc check stdlib/medical/trial.sio` fails with `i64`/`i32` type errors. The algorithmic intent is real, but there is no runnable entrypoint today. |
| 8. Epistemics: `Knowledge[T]`, GUM propagation, confidence-gate, pre-specified analysis-plan analogue | **Runtime:** `stdlib/epistemic/confidence_gate.sio:43-48` (`GatePolicy`), `:112-167` (`gate_check_value`), `:200-213` (`confidence_gate`); `stdlib/epistemic/path.sio:32-57` (path wrappers), `:136-152` (constructors), `:260-270` (`ep_discharge_path`). **Formal:** `formal/Epistemic.lean:77-81`, `formal/GUM.lean:28-90`, `formal/KnowledgeArithmeticSoundness.lean:111-169` | PARTIAL (runtime) / WORKS (Lean) | High | `bin/souc check stdlib/epistemic/confidence_gate.sio` and `stdlib/epistemic/path.sio` pass. Runtime `run` on epistemic code currently segfaults in Madaros, so the runnable demo inlines the relevant pieces (`demo/fregni/fregni_demo.sio`). The path wrappers give a compile-time provenance analogue; the confidence gate is a runtime pre-specified threshold. |
| 9. Regulatory hooks (21 CFR Part 11, ISO 17025, FAIR) | `stdlib/metrology/calibration.sio:1-353` (ISO 17025 calibration); `stdlib/epistemic/regulatory.sio:1-26` (`RegulatoryContext` with `cfr_part11` bool); `stdlib/epistemic/audit_runtime.sio:1-171` (audit log with regulatory layer); `formal/lean4/SounioRegulatory.lean:1-164` (GDPR/HIPAA/EU AI Act, **not** 21 CFR/ISO/FAIR) | WORKS (ISO 17025 calibration) / STUB (21 CFR Part 11) / ASPIRATIONAL (FAIR) | High for ISO 17025 metrology; Low for 21 CFR Part 11 and FAIR | `bin/souc check stdlib/metrology/calibration.sio` passes. `bin/souc check stdlib/epistemic/regulatory.sio` passes, but it is only a struct with boolean flags — no enforcement. No FAIR implementation exists in code; only website/manifesto prose. |

---

## Summary for Fregni

The only PPCR competencies that are **both conceptually aligned and runnable today** are:

1. **Epistemic confidence-gating and provenance tracking** (`confidence_gate.sio`, `path.sio`), demonstrated in `demo/fregni/`.
2. **Causal inference semantics**, but only in the Lean formal layer (`SounioCausality.lean`), not in the native Sounio compiler.
3. **ISO 17025-style calibration/metrology** (`stdlib/metrology/calibration.sio`).

Everything else is either **partial** (survival, sample-size code exists but is not green on the default compiler and lacks reference validation) or **aspirational** (randomization, blinding indices, 21 CFR Part 11 enforcement, FAIR).
