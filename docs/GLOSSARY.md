<!-- docs:meta
topic_id: repo.docs.glossary
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.glossary
-->

# Sounio Glossary

Definitions of key terms in epistemic computing and the Sounio language.

> **Canonical epistemic API.** The checked public surface for epistemic values
> is `epistemic::knowledge` (free-fn form: `ep_measured`, `ep_val`, `ep_std`,
> `ep_add`, `ep_mul`, `ep_div`, `ep_merge`, `ep_is_credible`, `ep_*_cov` for
> covariance-aware GUM), anchored by
> `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio` and
> `tests/run-pass/ep_gum_covariance.sio`. The legacy
> `epistemic::lib` surface (`epistemic_std`, `add_epistemic`), imported in
> `tests/stdlib/epistemic/test_core_e2e.sio` (a `//@ run-pass` test collected by
> `scripts/dev/run_sio_test_suite.sh`), is a legacy surface separate from the
> canonical `epistemic::knowledge` API. `mul_epistemic` and `fuse_measurements`
> are part of `epistemic::lib` but are not exercised by that test
> (`fuse_measurements` is private; `mul_epistemic` is public but untested).
> `with_confidence` operators and
> units-as-type-parameters are aspirational in source and absent from the
> checked public surface — see `docs/compiler/KNOWN_LIMITATIONS.md`.

## A

**Affine Type**
A type that can be used at most once. Less restrictive than linear types (which must be used exactly once). Used for resources that can be dropped but not duplicated.

**Algebraic Effect**
A programming language feature that allows modeling side effects (I/O, state, exceptions) as first-class values with handlers. Sounio's effect system is based on algebraic effects.

**AUC (Area Under the Curve)**
In pharmacokinetics, the integral of drug concentration over time. A key measure of drug exposure.

> Dimensions: in the checked surface, AUC is `Quantity { value, uncertainty, dim: dim_mass() × dim_length()^(-3) × dim_time()^(1) }` (concentration × time). Unit spellings such as `mg` are implemented and tested (`tests/run-pass/unit_same_add.sio`, which declares and uses only `mg`); `mL` is declared in `stdlib/units/pharmacological.sio`. The `unit Clearance = L/h` declaration syntax exists but is not yet covered by a run-pass fixture. The runtime `Quantity` form is the complementary representation.

---

## B

**Bayesian Inference**
Statistical method for updating probability estimates based on new evidence. Supported in `stdlib.bayes`.

**Bidirectional Type Inference**
Type checking algorithm that combines type synthesis (bottom-up) and type checking (top-down). Used in Sounio's type checker.

---

## C

**Clearance (CL)**
Rate at which a drug is removed from the body. Typical unit: `L/h`.

**Confidence Interval**
Range within which the true value lies with a stated probability (e.g., 95%). In Sounio, `Epistemic.confidence` is an integer *credibility score* (0..1000; 950 ≈ 0.95) carried alongside the value — it is a point rating, not a stored interval bound or coverage probability. Gate on it with `ep_is_credible(&e, 950)`, and build interval bounds yourself from `ep_val(&e)` and `ep_std(&e)`.

**Confidence Gate**
Conditional execution based on epistemic confidence:
```sio
use epistemic::knowledge::{ep_is_credible}

if ep_is_credible(&measurement, 950) {
    proceed()
} else {
    require_review()
}
```

---

## D

**Dimensional Analysis**
Checking physical dimensions (length, mass, time) for consistency. Sounio's unit system provides compile-time dimensional analysis.

**DSL (Domain-Specific Language)**
A programming language specialized for a particular domain. MedLang is Sounio's DSL for pharmacokinetics/pharmacodynamics.

---

## E

**Effect**
A computational side effect (I/O, mutation, errors, etc.) tracked in the type system:
```sio
fn read_file() -> string with IO { }
fn increment(x: &! i32) with Mut { }
```

**Effect Handler**
Code that defines how to interpret an effect. Allows custom behavior for effects like logging, state, or async operations.

**Epistemic**
Relating to knowledge or the process of knowing. In computing: explicitly representing what we know and our confidence in it.

**Epistemic Computing**
Computation that tracks uncertainty, confidence, and provenance alongside values. The foundational paradigm of Sounio.

**Epistemic Integrity**
The property that computational results accurately reflect the uncertainty and limitations of their inputs.

---

## F

**fMRI (Functional Magnetic Resonance Imaging)**
Brain imaging technique measuring neural activity. Sounio provides specialized support in `stdlib.fmri`.

---

## G

**GUM (Guide to the Expression of Uncertainty in Measurement)**
ISO standard for calculating and expressing measurement uncertainty. Sounio implements GUM-compliant propagation.

---

## H

**HIR (High-Level Intermediate Representation)**
Early compiler IR close to source code. First step after type checking.

**HLIR (Higher-Level Intermediate Representation)**
Mid-level IR with polyhedral analysis. Used for loop optimization.

---

## K

**Epistemic Type**
Sounio's canonical epistemic value type, defined in `stdlib/epistemic/knowledge.sio`:
```sio
pub struct Epistemic {
    pub val: f64,         // point estimate
    pub variance: f64,    // sigma^2 (variance, NOT std-dev)
    pub confidence: i64,  // 0..1000; 1000 == full knowledge
}
```

Construction: `ep_measured(val, std_dev)` stores `variance = std_dev^2` and `confidence = 900`; `ep_certain(val)` stores `confidence = 1000`. Struct-literal form (e.g. `Epistemic { val: 3.0, variance: 0.25, confidence: 900 }`) is used in `tests/run-pass/ep_gum_covariance.sio` and works under both Madaros and `lean_single`.

Arithmetic (free-fn form, portable across engines; see `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio`): `ep_add` / `ep_sub` / `ep_mul` / `ep_div` / `ep_scale` / `ep_shift` apply GUM δ-method, **uncorrelated** (covariance-aware variants `ep_add_cov` / `ep_sub_cov` / `ep_mul_cov` / `ep_div_cov` take a numeric covariance; see `tests/run-pass/ep_gum_covariance.sio`). Merge via `ep_merge` (inverse-variance weighted).

`provenance` is **not** a field of `Epistemic`, and there is **no public provenance API** in the checked surface — `stdlib/epistemic/prov.sio` is a private/internal PROV model (it declares no `pub` symbols), so it cannot be imported as a user-facing module.

---

## L

**Linear Type**
A type that must be used exactly once. Useful for resources like file handles that must be properly closed. Sounio enforces linear use at check time: a `linear struct` must be consumed on every path (the checker emits `E039` for use-after-consumption and `E040` for an unconsumed linear value). The source-level `linear` keyword on a struct is part of the checked surface — see `tests/run-pass/linear_balanced_branches.sio`.

---

## M

**MedLang**
Domain-specific language for pharmacokinetic/pharmacodynamic modeling, embedded in Sounio. Part of `stdlib.medlang`.

**MIR (Mid-Level Intermediate Representation)**
SSA-form IR for optimization passes. Similar to LLVM IR.

**Monte Carlo Simulation**
Statistical technique using repeated random sampling. Supported in `stdlib.monte_carlo`.

---

## O

**ODE (Ordinary Differential Equation)**
Equation involving derivatives of a function. Common in scientific modeling. Sounio ships `stdlib::ode` as a set of submodules — `rk4.sio`, `tsit5.sio`, `bdf.sio`, `epistemic.sio`, `solver.sio`, `solvers.sio`, and `pbpk3_stable.sio` — providing RK4, RK45, Tsit5, BDF, epistemic integration, and PBPK sources. Note: `stdlib/ode/lib.sio` itself only re-exports the epistemic PK-fit functions (`epistemic_pk_fit`, `epistemic_pkpd_fit`); import the solvers from their defining submodules, not from `lib.sio` as a solver entry point: the RK4 solver lives in `ode::rk4` (`rk4_step`, `rk4_integrate`), and the adaptive (RK45) and BDF wrappers live in `ode::solver` (`solve_rk45_exp_decay`, `solve_bdf1_exp_decay`). Note that `ode::tsit5` only exports its test `main` and `ode::bdf` exports no public functions, so do not import solvers from `ode::tsit5` or `ode::bdf`. For source-tracked uncertainty propagation see `stdlib/epistemic/affine` (anchor: `tests/run-pass/affine_shared_source_add.sio`).

**Ownership**
System ensuring memory safety by tracking which part of code "owns" each value. Sounio uses affine/linear types instead of Rust's borrow checker.

---

## P

**PBPK (Physiologically-Based Pharmacokinetic Model)**
Detailed pharmacokinetic model incorporating anatomical and physiological information. Supported in `stdlib.pbpk`.

**PK (Pharmacokinetics)**
Study of how the body processes drugs (absorption, distribution, metabolism, excretion).

**PD (Pharmacodynamics)**
Study of drug effects on the body.

**Provenance**
Information about where data came from: instrument, operator, timestamp, processing steps.

**Propagation (Uncertainty)**
Calculation of output uncertainty from input uncertainties. Automatic in Sounio via GUM formulas.

---

## R

**Refinement Type**
Type with logical predicates restricting values:
```sio
type Positive = { x: i32 | x > 0 }
type Even = { x: i32 | x % 2 == 0 }
```

**REPL (Read-Eval-Print Loop)**
Interactive programming environment. Sounio provides a REPL via `souc repl`.

---

## S

**SIR (Domain-Specific Intermediate Representation)**
Specialized IR for scientific computing operations (ODEs, tensors, autodiff, GPU kernels).

**SMT Solver (Satisfiability Modulo Theories)**
Tool for checking logical formulas. Used in Sounio for verifying refinement types (via Z3).

**SSA (Static Single Assignment)**
IR form where each variable is assigned exactly once. Used in MIR and optimization passes.

**Standard Uncertainty**
One standard deviation of measurement uncertainty. In the canonical `Epistemic` struct, store as `variance = sigma^2`; access sigma via `ep_std(&e)` (which returns `sqrt(variance)`).

---

## T

**Type Inference**
Automatic deduction of types from context. Sounio uses bidirectional type inference.

---

## U

**Uncertainty**
Quantitative measure of doubt about a measurement. In Sounio, typically represented as standard uncertainty (one standard deviation).

**Uncertainty Budget**
Breakdown of contributors to total uncertainty. Can be tracked via provenance metadata.

**Uncertainty Propagation**
See [Propagation](#p).

**Units of Measure**
Physical dimensions (meters, kilograms, seconds) carried at runtime by `Quantity` values and built with `dim_*()` constructors from `stdlib/units/lib.sio` (compatibility is checked by `dim_eq`/`quantity_is_compatible`, asserted inside `quantity_add`):
```sio
use units::lib::*   // dim_mass, dim_length, dim_time, quantity_new, quantity_div, ...

let distance = quantity_new(100.0, 0.0, dim_length())
let time     = quantity_new(10.0,  0.0, dim_time())
let velocity = quantity_div(distance, time)
// Convert: convert_m_to_cm(value), convert_celsius_to_kelvin(value), etc.
```

> Units-as-type spellings (`let distance: m = 100.0`, `unit N; fn f(x: kg) -> N`) and
> derived unit spellings (`f64<m/s>`, `mg`) ARE part of the checked surface
> and `mg` is tested in `tests/run-pass/unit_same_add.sio`; `mL` is declared in
> `stdlib/units/pharmacological.sio`. The `unit Clearance = L/h`
> and `mg*h/L` derived-declaration syntax exists but is not yet covered by a
> run-pass fixture. The runtime `Quantity` + `quantity_new(val, unc, dim_*())`
> form is the complementary representation for dimensions computed at runtime.

---

## V

**Volume of Distribution (V)**
Apparent volume in which a drug distributes. Typical unit: `L` or `L/kg`.

---

## References

- **GUM**: JCGM 100:2008 - Evaluation of measurement data
- **ISO 17025**: General requirements for testing and calibration laboratories
- **FDA Guidance**: Population pharmacokinetics
- **Algebraic Effects**: Plotkin & Power (2003)

---

*For implementation details, see the [Language Guide](LLM_PROGRAMMING_GUIDE.md).*
