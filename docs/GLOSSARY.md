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

## A

**Affine Type**
A type that can be used at most once. Less restrictive than linear types (which must be used exactly once). Used for resources that can be dropped but not duplicated.

**Algebraic Effect**
A programming language feature that allows modeling side effects (I/O, state, exceptions) as first-class values with handlers. Sounio's effect system is based on algebraic effects.

**AUC (Area Under the Curve)**
In pharmacokinetics, the integral of drug concentration over time. A key measure of drug exposure. Type: `mg*h/L` in Sounio.

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
Range within which the true value lies with a stated probability (e.g., 95%). Tracked in `Knowledge<T>.confidence`.

**Confidence Gate**
Conditional execution based on epistemic confidence:
```sio
if measurement.confidence > 0.95 {
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

**Knowledge Type**
Sounio's fundamental epistemic type:
```sio
struct Knowledge<T> {
    value: T,
    uncertainty: f64,
    confidence: f64,
    provenance: Source,
}
```

---

## L

**Linear Type**
A type that must be used exactly once. Useful for resources like file handles that must be properly closed. Sounio supports linear types via the `linear` keyword.

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
Equation involving derivatives of a function. Common in scientific modeling. Sounio provides solvers in `stdlib.ode`.

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
One standard deviation of measurement uncertainty. The `uncertainty` field in `Knowledge<T>`.

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
Physical dimensions (meters, kilograms, seconds) tracked in the type system:
```sio
let distance: m = 100.0
let time: s = 10.0
let velocity: m/s = distance / time
```

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
