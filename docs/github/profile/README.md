<!-- docs:meta
topic_id: repo.docs.github.profile.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.github.profile.readme
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Sounio-lang/sounio/main/docs/assets/lockups/lockup_horizontal_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Sounio-lang/sounio/main/docs/assets/lockups/lockup_horizontal_light.svg">
    <img alt="Sounio Logo" src="https://raw.githubusercontent.com/Sounio-lang/sounio/main/docs/assets/lockups/lockup_horizontal_dark.svg" width="480">
  </picture>
</p>

<p align="center">
  <strong>The self-hosted systems programming language for safety-critical scientific computing and epistemic integrity.</strong>
</p>

<p align="center">
  <a href="https://www.souniolang.org"><img src="https://img.shields.io/badge/website-souniolang.org-blue.svg" alt="Sounio Website"/></a>
  <a href="https://www.souniolang.org/playground"><img src="https://img.shields.io/badge/playground-wasm-purple.svg" alt="Playground"/></a>
  <a href="https://github.com/Sounio-lang/sounio/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-gold.svg" alt="Apache-2.0 License"/></a>
  <a href="https://github.com/Sounio-lang/sounio/blob/main/SCALE.md"><img src="https://img.shields.io/badge/scale-4.2k%20.sio%20files-informational.svg" alt="~4.2k tracked .sio files"/></a>
</p>

---

## 🏛️ Sounio Ecosystem Overview

Modern scientific computing demands more than correct arithmetic—it demands **epistemic integrity**. 

Most traditional programming languages treat measurement uncertainty, physical units, and side-effects as external, non-enforced library concerns. Sounio makes them foundational. Sounio is not a Rust or Julia dialect; it is a compiled, safety-critical systems language designed from first principles for engineers and scientists who need both raw computational performance and defensible confidence handling.

### ⚡ Technical Pillars

*   **Epistemic gradual compilation (`Knowledge[T]`)**: Every scientific measurement carries a level of confidence $\epsilon \in [0, 1]$. Sounio integrates GUM-compliant uncertainty propagation directly into the type checker.
*   **Compile-Time Confidence Gates**: Functions can mandate minimum confidence limits (e.g. `where c.ε >= 0.82` for AUC-guided vancomycin dosing). Under-confident data causes compile-time rejection—not a runtime warning, not a log, but a compilation error.
*   **Dimensional Safety & Unit Check**: Sounio supports compile-time units of measure (`let dose: mg = 500.0`). The compiler enforces dimensional analysis via `VAR_UNIT_DIM` checks, completely eliminating unit mismatch bugs in safety-critical calculations.
*   **Mandatory Algebraic Effects**: All side effects are explicitly declared and tracked by the compiler (`with IO, Mut, Div, Panic`). Missing side-effect declarations are compile errors.
*   **Self-Hosted Compiler & Native Codegen**: A native, multi-stage, zero-dependency bootstrap chain producing optimized x86_64 ELF binaries from Sounio sources.

---

## 🎮 Code at the Horizon of Certainty

Here is a quick look at Sounio's uncertainty propagation and compile-time validation in action:

```sounio
// ASHP 2020 guidelines mandate ε >= 0.82 before AUC-guided dosing is permitted.
fn prescribe_vancomycin(dose: Knowledge[f64, ε >= 0.82]) with IO {
    println("Safe dose prescription finalized.")
}

fn main() with IO, Div, Panic {
    // A drug dose with tracked confidence and evidence source
    let base_dose = Knowledge(15.0, ε=0.92, prov="ASHP_2020_Level1A_RCT")

    // Hospital scale measurement: high-confidence calibrated device
    let weight = Knowledge(78.5, ε=0.98, prov="hospital_scale_calibrated")
    let ref_wt = Knowledge(70.0, ε=1.0)

    // ISO GUM propagation is automatic: ε(a*b) = ε(a) * ε(b)
    let adjusted_dose = base_dose * (weight / ref_wt)
    
    // The adjusted dose propagates to ~0.90 confidence (safe!)
    prescribe_vancomycin(adjusted_dose)

    // Under-confident estimate (CG formula CV >> 28%)
    let risky_dose = Knowledge(500.0, ε=0.40, prov="uncalibrated_historical_cg")

    // COMPILE-TIME REJECTION: risky_dose (ε=0.40) breaches prescribe_vancomycin's threshold (ε>=0.82)
    prescribe_vancomycin(risky_dose) 
}
```

---

## 📂 Core Repositories

-   🏛️ **[Sounio-lang/sounio](https://github.com/Sounio-lang/sounio)**: The core self-hosted compiler (`souc`), runtime engine, full test suite (~4.2k files), and scientific standard library (`stdlib/`).
-   🎮 **[Sounio-lang/playground](https://www.souniolang.org/playground)**: Try Sounio directly in your browser with our zero-install WebAssembly playground.

---

## 🤝 Join the Epistemic Era

We are building a software ecosystem that communicates the quality of its own knowledge. If you are interested in compiler design, scientific software, formal verification, or clinical modeling:

-   Read our **[Language Tour](https://www.souniolang.org/language)** and **[Evidence Logs](https://www.souniolang.org/proof)**.
-   Review our **[Contributing Guidelines](https://github.com/Sounio-lang/sounio/blob/main/CONTRIBUTING.md)** to see how you can help.
-   Check our **[Security Policy](https://github.com/Sounio-lang/sounio/blob/main/SECURITY.md)** if you are auditing safety-critical components.

---

<p align="center">
  <i>"Uncertainty is not a bug; it is a fundamental property of data. Sounio makes it a type."</i>
</p>
