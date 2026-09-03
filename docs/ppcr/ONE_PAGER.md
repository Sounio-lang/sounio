<!-- docs:meta
topic_id: repo.docs.ppcr.one-pager
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ppcr.one-pager
-->

# Sounio × PPCR: One-Page Partnership Brief

**For:** Prof. Felipe Fregni, Harvard PPCR  
**Date:** 17 June 2026  
**Prepared by:** Sounio team, with AI-assisted drafting (see disclosure below).

## The problem

Clinical trials are drowning in data, but the languages we use to specify their analyses treat uncertainty and provenance as afterthoughts. A p-value or a dose recommendation is produced without a machine-checkable record of whether the input was measured, imputed, or simulated, or whether its confidence fell below the pre-specified threshold. Reproducibility suffers, and regulatory pre-specification becomes a human process rather than a computational one.

## The Sounio mechanism

Sounio is a self-hosted systems language with an **epistemic type system**. It can carry three things that ordinary languages cannot enforce:

1. **Provenance as a type.** A value measured in the lab (`MeasuredGUMI64`), imputed by a model (`ImputedModelI64`), or produced by simulation (`SimulationI64`) has a distinct compile-time type. A function that requires a measured value cannot accidentally accept a simulated one.
2. **Confidence as a runtime gate.** The `confidence_gate` pattern rejects inputs whose confidence is below a pre-specified threshold, or whose coefficient of variation exceeds the protocol limit.
3. **GUM-aware propagation.** The formal layer (`formal/GUM.lean`, `formal/Epistemic.lean`) contains machine-checked proofs that knowledge-value arithmetic preserves variance bounds and confidence invariants.

## PPCR-specific hook

For PPCR, the most immediate fit is **pre-specified analysis plans + data quality gates**. The confidence threshold and the provenance wrapper are the computational analogue of saying: *"Only measured outcomes with ≥95% confidence may enter the primary analysis; imputed values require a documented discharge reason."* The compiler and runtime can enforce this visibly.

## What is real today

- `demo/fregni/fregni_demo.sio` runs end-to-end and shows a dosing pipeline accepting a high-confidence measured clearance while rejecting a low-confidence simulated clearance and a high-CV imputed clearance.
- `demo/fregni/bad_path.sio` fails at compile time when a simulated value is passed to a measured-only extractor.
- `formal/lean4/SounioCausality.lean` is a verified Pearl-style causal module (do-operator, d-separation, confounders, identifiability).
- `stdlib/metrology/calibration.sio` implements ISO 17025-style calibration logic.

## What a collaboration would build

- A PPCR-native `study { ... }` block that declares estimands per ICH E9(R1), randomisation schemes, blinding indices, and interim/futility rules, all compiled into runnable, auditable code.
- Reference-validated sample-size and survival modules (currently present but not green on the default compiler and not validated against R/Python).
- A 21 CFR Part 11 / ALCOA audit trail that binds electronic records to the epistemic path wrappers, not just to boolean flags.

## Runnable snippet

```sio
// demo/fregni/fregni_demo.sio (excerpt)
struct MeasuredGUMI64 { value_i64: i64, confidence_permille: i64 }
struct SimulationI64 { value_i64: i64, confidence_permille: i64 }

fn measured_gum(v: i64, c: i64) -> MeasuredGUMI64 { ... }
fn simulation(v: i64, c: i64) -> SimulationI64 { ... }
fn payload_from_measured(k: MeasuredGUMI64) -> i64 { k.value_i64 }
fn check_confidence(c: i64, min_c: i64) -> bool { c >= min_c }

fn main() -> i32 with IO {
    let min_conf = 950  // 95.0% as permille
    let measured_cl = measured_gum(8000, 980)
    if check_confidence(measured_cl.confidence_permille, min_conf) {
        // dose calculation proceeds
    }
    let simulated_cl = simulation(7500, 700)
    // rejected: 700 permille < 950 permille
}
```

Real output from `bash demo/fregni/run.sh`:

```text
Scenario 1: measured clearance (980 permille confidence)
  CONFIDENCE GATE: PASSED
  Recommended daily dose = 32000.0 mg
Scenario 2: simulated clearance (700 permille confidence)
  CONFIDENCE GATE: FAILED
...
```

Full output is captured in `demo/fregni/OUTPUT.md`.

## Honest boundary

Sounio does **not** yet have production-grade randomisation, blinding indices, validated survival analysis, or enforced 21 CFR Part 11 signatures. The partnership value is to co-design these *inside* a language that already has the epistemic foundation that no mainstream language has.

---

**GAIDeT-style AI disclosure:** This brief was drafted with assistance from Kimi Code CLI, an AI coding assistant. Every factual claim has been checked against the repository, assigned a maturity tag, and recorded with evidence paths in `docs/ppcr/CLAIMS_LEDGER.md`.
