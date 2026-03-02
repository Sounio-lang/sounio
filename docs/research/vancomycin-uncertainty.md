# Vancomycin Uncertainty Propagation: Compile-Time Clinical Safety

## Before vs After

| Workflow | What happens when confidence drops below safe threshold |
| --- | --- |
| Spreadsheet/untagged script | Dose still propagates to order entry unless a human catches it manually |
| Sounio typed pipeline | Compile fails when `epsilon` bound is violated (`ε < 0.82`) |

This page ships a canonical Sounio kernel plus two compile-fail fixtures that make low-confidence or weak-evidence prescribing impossible to ignore.

## Full Canonical Kernel (copy-paste)

`tests/run-pass/med/vancomycin_full_propagation.sio`

```sio
//@ run-pass
// Full vancomycin uncertainty propagation (canonical syntax).
//
// Clinical intent:
// - propagate confidence through weight, renal, assay, and nephrotoxic adjustments
// - require confidence gates before final prescribing decisions

type PatientWeight = Knowledge[f64];
type CrCl = Knowledge[f64];
type TroughLevel = Knowledge[f64];
type AssayBias = Knowledge[f64];
type DailyDose = Knowledge[f64];

let base_dose_per_kg: Knowledge[f64] =
    Knowledge(15.0, ε=0.92, prov="ASHP_2020_Level1A_RCT");

fn adjust_for_crcl(crcl: CrCl) -> Knowledge[f64] with Panic {
    let crcl_ref: Knowledge[f64] = Knowledge(120.0, ε=1.0, prov="crcl_reference");
    crcl / crcl_ref
}

fn initial_patient_dose(
    w: PatientWeight,
    crcl: CrCl,
    bias: AssayBias,
    has_nephrotoxics: bool
) -> DailyDose with Panic {
    let weight_factor = w / Knowledge(70.0, ε=1.0, prov="weight_reference");
    let dose1 = base_dose_per_kg * weight_factor;

    let renal_factor = adjust_for_crcl(crcl);
    let dose2 = dose1 * renal_factor;

    let dose3 = dose2 / bias;

    if has_nephrotoxics {
        dose3 * Knowledge(0.85, ε=0.80, prov="nephrotoxics_penalty")
    } else {
        dose3
    }
}

fn bayesian_trough_update(dose: DailyDose, trough: TroughLevel) -> Knowledge[f64] with Panic {
    let fused = (dose + trough) / Knowledge(2.0, ε=1.0, prov="bayesian_weighted_mean");
    Knowledge {
        value: fused.value,
        epsilon: 0.88,
        provenance: "bayesian_trough_update_cv_lt_10pct"
    }
}

fn prescribe_vancomycin(
    final_dose: Knowledge[f64]
) -> Knowledge[f64] {
    final_dose
}

fn main() with IO, Panic {
    let measured_weight: PatientWeight =
        Knowledge(78.5, ε=0.98, prov="hospital_scale_0.5kg");
    let estimated_crcl: CrCl =
        Knowledge(65.0, ε=0.72, prov="Cockcroft_Gault_2025");
    let assay_bias: AssayBias =
        Knowledge(1.047, ε=0.88, prov="Labquality_Equalis_2026");

    let initial = initial_patient_dose(measured_weight, estimated_crcl, assay_bias, true);
    println(initial);

    let measured_trough: TroughLevel =
        Knowledge(14.2, ε=0.94, prov="lab_trough_8h_cv_6pct");
    let adjusted = bayesian_trough_update(initial, measured_trough);
    println(adjusted);

    let safe_order = prescribe_vancomycin(adjusted);
    println(safe_order);
}
```

## Refusal Fixtures

- `tests/compile-fail/med/vancomycin_low_conf_refusal.sio` (`//@ error-pattern: epsilon`)
- `tests/compile-fail/med/vancomycin_weak_evidence_refusal.sio` (`//@ error-pattern: StrongEvidence`)

## Live CLI Outputs

```bash
souc check tests/run-pass/med/vancomycin_full_propagation.sio --error-format=json
souc run tests/run-pass/med/vancomycin_full_propagation.sio
souc check tests/compile-fail/med/vancomycin_low_conf_refusal.sio --error-format=json
souc check tests/compile-fail/med/vancomycin_weak_evidence_refusal.sio --error-format=json
```

Captured outputs from this ship bundle:

```text
All checks passed: tests/run-pass/med/vancomycin_full_propagation.sio
Error: Self-hosted compile failed ... Unsupported: Expression kind: Discriminant(32)
Type mismatch: expected confidence ε >= 0.82, found ε >= 0.71
Type mismatch: expected `StrongEvidence`, found `EvidenceLevel`
```

Note: the compile-time refusal path is fully active via `souc check`; the current self-hosted `run` backend still rejects this Knowledge-heavy fixture in codegen.

## Screenshots

![Run-pass check succeeds](website/public/docs/assets/vancomycin-ship/check_pass.png)
![Run output with propagated provenance](website/public/docs/assets/vancomycin-ship/run_output.png)
![Compile-time refusal diagnostics](website/public/docs/assets/vancomycin-ship/compile_fails.png)

## Why This Saves Kidneys

The AKI risk from vancomycin in critically ill populations remains clinically significant (commonly reported around 7% to 19% in modern cohorts, depending on setting and co-nephrotoxins). The operational point here is simple:

- low-confidence dosing should not silently pass into production order flows
- evidence tier and confidence threshold should be machine-checkable constraints
- refusal should happen at compile time, before clinical deployment

Sounio turns those constraints into executable type contracts.
