<!-- docs:meta
topic_id: repo.docs.competitive-position-2026
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.competitive-position-2026
-->

# Sounio Competitive Position — November 2026

> *Evidence-based analysis of where Sounio stands. The November 2026 refresh compares
> Sounio against its four real neighbours — GTC (MSL New Zealand), METAS UncLib,
> Uncertain\<T> (ASPLOS 2014) and Signaloid — with links opened on 2026-11-10.
> The April 2026 sections on Julia, Python/SciPy, Mojo and Numbat follow unchanged.*

---

## Executive Summary

Sounio occupies a **distinctive but contested niche** in 2026: it is, to our knowledge, the only
systems programming language whose type system natively encodes epistemic trust. No single
competitor provides compile-time confidence gates, GUM-compliant uncertainty propagation,
*and* ISO 1788 interval arithmetic *in one standard library* — but each axis has a real
neighbour that beats Sounio on its home ground (see "The real neighbours" below). The older
challengers each miss at least two of these axes:

| Feature | Sounio | Julia 1.12 | Python/SciPy | Mojo v26.2 | Numbat v1.23 |
|---|:---:|:---:|:---:|:---:|:---:|
| Compile-time confidence gate (`ε >= k`) | **✓** | ✗ | ✗ | ✗ | ✗ |
| GUM §5 variance propagation | **✓** (stdlib) | package only¹ | package only² | ✗ | ✗ |
| ISO 1788 interval arithmetic | **✓** (stdlib) | package only³ | package only⁴ | ✗ | ✗ |
| Physical units as types | **✓** | package only⁵ | package only⁶ | ✗ | **✓** (calc only) |
| Explicit effects system | **✓** | ✗ | ✗ | ✗ | ✗ |
| Property-based testing (stdlib) | **✓** | ✗⁷ | package only⁸ | partial | ✗ |
| ISO 17025 metrology (stdlib) | **✓** | ✗ | ✗ | ✗ | ✗ |
| Self-hosted compiler | **✓** | ✗ | ✗ | ✗ | ✗ |
| Non-commutative algebras (stdlib) | **✓** | ✗ | ✗ | ✗ | ✗ |
| Linear types | **✓** | ✗ | ✗ | partial⁹ | ✗ |

**Sources:**
1. `Measurements.jl` — external package, not in Julia stdlib
2. `uncertainties` Python library — PyPI, not in stdlib
3. `IntervalArithmetic.jl` — JuliaIntervals, not in Julia stdlib (confirmed Apr 2026 issue #747)
4. `pyinterval` / `mpmath` — PyPI only
5. `Unitful.jl` — external package
6. `pint` — PyPI only
7. Julia has no property-based testing in stdlib; `PropCheck.jl` is unmaintained as of 2026
8. Python has `Hypothesis` (PyPI), `CrossHair`, `Deal` — none in stdlib
9. Mojo has borrow checker but no true linear types

---

## The real neighbours (November 2026)

Links below were opened on 2026-11-10. For each neighbour: what it is, and what it does
better than Sounio today.

### GTC — GUM Tree Calculator (MSL New Zealand)
https://github.com/MSLNZ/GTC
Python package (Hall; Hall and Borbely) implementing *uncertain numbers*: automatic GUM
propagation with correlations, real and complex quantities, and persistence of full
measurement models. Used for over a decade inside MSL, including international key
comparisons (e.g. CCT-K7.2021 thermal metrology analysis accounting for correlations),
with a tutorial paper and a Metrology journal article, and versioned releases with DOI.
**Better than Sounio today**: institutional metrological scrutiny no language project can
claim — real key-comparison pedigree, a decade of scrutiny by metrologists, and proven
correlation bookkeeping in production analyses. Sounio's propagation has no such pedigree.

### METAS UncLib (Swiss Federal Institute of Metrology)
https://www.callabmag.com/metas-unclib/ (the official METAS UncLib page returns 404 as of 2026-11-10)
C#/.NET library (Wollensack, VNA Tools II programme): multivariate uncertainties, complex
values, correlations, vector and matrix algebra, MATLAB wrapper, storage of results.
Propagation is GUM-Tree-based (Hall's concept) with linear *and higher-order* terms.
**Better than Sounio today**: higher-order (nonlinear) propagation, full-covariance
multivariate algebra, and RF/VNA domain pedigree. Sounio propagates first-order variance
only and has no complex-quantity story at this level.

### Uncertain\<T> (ASPLOS 2014, Bornholt / Mytkowicz / McKinley)
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos077-bornholtA.pdf
A first-order type for uncertain data with *sampling-based* runtime semantics:
distributions as values, conditional execution over uncertain predicates, hypothesis
tests, and an explicit static-vs-dynamic cost model, evaluated on real sensors and GPS
traces. The closest intellectual ancestor of `Knowledge<T>`.
**Better than Sounio today**: a published semantics for conditionals over uncertain
values (Sounio has none), dynamic sampling with accuracy budgets, and a peer-reviewed
evaluation. Sounio's static GUM typing is stricter at compile time but says nothing
about branching on uncertain data.

### Signaloid (C0 processor)
https://signaloid.com/
Hardware and cloud platform that tracks full probability distributions through
computation: a RISC-V-based C0 core (FPGA C0-microSD modules, Cloud Compute Engine)
executing deterministic operations on distributions, with reference-quality
Monte-Carlo-parity cores and three-orders-of-magnitude speedups on stochastic workloads
(finance VaR, particle filters, Bayesian inference).
**Better than Sounio today**: distribution tracking at hardware speed with real-time
edge deployment. Sounio is software-only on CPU; it cannot touch C0's throughput or
its Monte-Carlo-parity execution model. Sounio's answer is static guarantees, not speed.

## Gap Analysis vs. Competitors (SOTA April 2026)

### Julia 1.12 / 1.13-rc

**Julia's real strength**: `DifferentialEquations.jl` (SciML ecosystem), `LinearAlgebra.jl` stdlib,
`FFTW.jl`, `MixedModels.jl`. These have no equal in any other language.

**Julia's weaknesses (identified from Julia community data, 2026)**:
- No stdlib property-based testing (arxiv:2410.10908 §4: "Julia lacks property-based testing,
  symbolic execution, contract-based testing")
- No static type checker for Julia code
- Startup / TTFX (time-to-first-execution) penalty remains ~2–10s for most packages
- Mutually recursive types only landed in 1.13-rc (Jan 2026)
- `DataFrames.jl` performance still trails Pandas on groupby workloads

**Sounio advantage**: Effects system + `Knowledge<T>` makes correctness *structural*, not
optional. Julia has no mechanism to enforce epistemic constraints at compile time.

### Python / NumPy / SciPy

**SciPy's real strength**: 30 years of tested numerical algorithms, 21,000+ CRAN-equivalent
packages (R ecosystem), massive community, `Hypothesis` for property-based testing.

**Python's weaknesses**:
- No compile-time anything — all guarantees are runtime
- No dimensional analysis in stdlib (only `pint`, a 3rd-party library)
- No uncertainty propagation in stdlib
- No effects system — side effects are invisible to callers
- GIL (still partially present in CPython 3.13's free-threaded mode) limits true parallelism

**Sounio advantage**: Sounio provides the guarantees Python cannot: `fn prescribe(dose: Knowledge[f64, ε >= 0.82])` literally *cannot compile* with an under-confident value. Python's equivalent would be a runtime assertion at best.

### Mojo v26.2 (Modular)

**Mojo's real strength**: MLIR backend → hardware portability (CPU/GPU/accelerator), Python
interop, SIMD vectorization, `MAX AI Kernels` library for LLM inference.

**Mojo's weaknesses**:
- No uncertainty quantification of any kind
- No units/dimensional analysis
- Closed-source compiler (open-sourcing planned by end-2026)
- Standard library focused on AI/ML, not general scientific computing
- No effects system; no provenance tracking

**Sounio advantage**: Mojo is a better Python replacement for AI/ML. Sounio is a better
Julia/MATLAB replacement for epistemic scientific computing. Different targets.

### Numbat v1.23

**Numbat's real strength**: Physical dimensions as types (beautiful, clean design), first-class
units with automatic conversion, rich SI stdlib.

**Numbat's weaknesses**:
- Calculator-only (no general-purpose programming)
- No uncertainty propagation (unit correctness ≠ measurement uncertainty)
- No systems-level programming (no `malloc`, no FFI, no effects)
- No compiled output — interpreted only
- No ODE solvers, neural networks, or scientific algorithms

**Sounio advantage**: Sounio has everything Numbat has (units + type safety) *plus*
uncertainty propagation, effects, systems programming, ODE solvers, neural networks,
non-commutative algebras, and a self-hosted native compiler.

---

## What Sounio Has That Nobody Else Has

### 1. Compile-Time Confidence Gate

```sio
// ASHP 2020 §8.3: AUC-guided dosing requires ε >= 0.82
fn prescribe_vancomycin(dose: Knowledge[f64, ε >= 0.82]) with IO {
    println("Vancomycin prescribed")
}

fn main() with IO {
    let risky: Knowledge[f64, ε=0.40] = Knowledge { value: 500.0, epsilon: 0.40 }
    prescribe_vancomycin(risky)  // COMPILE ERROR — patient safety enforced
}
```

**Equivalent in Julia**: impossible — would require a runtime check.  
**Equivalent in Python**: `assert dose.epsilon >= 0.82` — runtime only, bypassable.  
**Equivalent in Mojo**: no concept of epistemic confidence.

### 2. ISO 1788 Interval Arithmetic in stdlib/verify/

```sio
let a = interval(1.0, 3.0)       // [1, 3]
let b = interval(2.0, 5.0)       // [2, 5]
let s = iv_add(a, b)             // [3, 8] — ISO 1788 §9 containment
let d = iv_div(a, b)             // [0.2, 1.5]
let b_bad = interval(-1.0, 1.0)
let d_ill = iv_div(a, b_bad)     // ILL-decorated — 0 ∈ denominator
```

**Julia**: `IntervalArithmetic.jl` is an external package (324 stars, Apr 2026 still in dev)  
**Python**: `pyinterval` unmaintained; `mpmath` does intervals but not ISO 1788  
**All others**: no interval arithmetic at all

### 3. GUM §5 Uncertainty Budget (Knowledge<T> propagation)

```sio
let v: Knowledge[f64] = measure(10.5, uncertainty: 0.03)  // u = 0.03
let w: Knowledge[f64] = measure(2.0,  uncertainty: 0.01)
let result = v * w   // GUM §5.1.2: u_c² = u_v² * w² + u_w² * v²  (automatic)
```

GUM variance propagation is computed at runtime by `stdlib/epistemic/` (e.g. the delta-method
`mul` in `composed_effects.sio`), operating on the `Knowledge<T>` type. The compiler's role is
type- and effect-checking: `Observe` is a registered algebraic effect (`self-hosted/check/effects.sio`)
that a function must declare to read an unobserved epistemic value. The compiler does not itself
evaluate the GUM formulas.

### 4. ISO 17025 Metrology in stdlib/metrology/

```sio
var cert = cal_cert_new(CAL_LEVEL_SECONDARY, 2.0, 365)
let p = cal_point_new(100.0, 100.05, 0.01, 0.005, 0.003, 20.0)
cal_cert_add_point(&!cert, p)

let u_c = cal_point_combined_u(p)   // GUM §5.1.1 RSS combination
let u_A = type_a_eval(&readings, 5) // GUM §4.2 Type A evaluation
let u_B = type_b_rectangular(0.5)   // GUM §4.4 Type B rectangular
```

No competitor has this in their standard library. This is the first
ISO 17025-aligned calibration module in any open-source PL stdlib.

### 5. Explicit Effects System (unique in systems languages)

```sio
fn pure_fn(a: f64, b: f64) -> f64 { a + b }           // no effects
fn io_fn(x: f64) with IO { println(x) }               // must declare IO
fn mut_fn(r: &!f64) with Mut { *r = 42.0 }            // must declare Mut
fn div_fn(a: f64, b: f64) -> f64 with Div { a / b }   // must declare Div
fn safe(a: f64) -> f64 with Panic { assert(a > 0.0); a }  // must declare Panic
```

Julia, Python, Mojo, Numbat: effects are invisible. In Sounio, every side effect
is visible at the call site. This enables:
- Static proof that a function is pure
- Rejection of IO in proofs/kernels
- Provable absence of division errors in certified code paths

### 6. Non-Commutative Algebra System

```sio
algebra Sedenion over f64 {
    add: commutative, associative
    mul: non_associative, non_commutative, alternative: false
}
```

Julia has `Grassmann.jl` (external). Python has `sympy` (slow, symbolic). Mojo has nothing.
Sounio has sedenion, octonion, and quaternion algebras as first-class stdlib modules,
verified by the e-graph optimizer that respects algebraic axioms.

### 7. Property-Based Testing in stdlib/test/

```sio
// stdlib/test/prop.sio — now active
let failures = prop_check_i64(42, 100, -1000, 1000, 1)  // 100 random cases
```

Julia's state (arxiv:2410.10908): "Julia has virtually no support for property-based
testing, symbolic execution, and contract-based testing."

Python has `Hypothesis` — but it's an external PyPI package, not stdlib.

---

## Remaining Gaps (Honest Assessment)

| Area | Gap | Priority | Notes |
|---|---|:---:|---|
| HDF5 / NetCDF I/O | No stdlib support | HIGH | FAIR data formats for science |
| Parquet / Arrow I/O | No stdlib support | HIGH | Data science interop |
| DataFrames | No equivalent | HIGH | Most scientific workloads need tabular data |
| Symbolic math | No CAS | MEDIUM | Julia has `Symbolics.jl`, Python has `SymPy` |
| Distributed computing | No MPI/actors | MEDIUM | Julia has `Distributed`, Python `mpi4py` |
| Package count | ~0 community packages | HIGH | Julia: 10k+, Python: 500k+ |
| IDE / LSP | Partial (tree-sitter) | MEDIUM | Julia has JETLS (new Jan 2026) |
| REPL | Not yet in native mode | MEDIUM | Julia REPL is excellent |
| Time (`instant_now`) | FFI stub only | LOW | `libc` linking not yet wired |
| Windows support | Unknown | LOW | Linux x86-64 only confirmed |

---

## Strategic Positioning

**Sounio is not competing with Julia for the SciML ecosystem.**  
**Sounio is not competing with Python for data science tooling.**  
**Sounio is not competing with Mojo for AI/ML inference.**

Sounio's unique market:

> **Regulated scientific computing**: pharmacokinetics, metrology, medical devices,
> environmental monitoring, experimental physics — anywhere the *trustworthiness* of
> a computed result must be *provable*, not just *likely*.

In these domains, the ASHP 2020 §8.3 example above is not academic. It is the
difference between a compiler catching a fatal drug dosing error and a runtime crash
(or worse: a silent wrong answer).

No other language in 2026 provides this guarantee at compile time.

---

## Appendix: SOTA Sources

| Topic | Source | Date |
|---|---|---|
| Julia 1.12–1.13 features | julialang.org/blog (Jan–Mar 2026) | Mar 2026 |
| Mojo v26.2 stdlib | docs.modular.com/stable/mojo/lib | Mar 2026 |
| Numbat v1.23 | github.com/sharkdp/numbat | Feb 2026 |
| Julia vs Python/R gaps | arxiv:2410.10908 "State of Julia for Scientific ML" | Oct 2024 |
| Julia ecosystem gaps 2026 | r-statistics.co/R-vs-Julia.html | 2026 |
| IntervalArithmetic.jl / validated numerics survey (Baratchart et al., SN Computer Science 2023) | https://link.springer.com/article/10.1007/s42979-023-01776-8 | April 2026 |
| GTC (MSL NZ) | https://github.com/MSLNZ/GTC | Nov 2026 (opened 2026-11-10) |
| METAS UncLib | https://www.callabmag.com/metas-unclib/ (official METAS page 404 on 2026-11-10) | Nov 2026 (opened 2026-11-10) |
| Uncertain\<T> ASPLOS 2014 | https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos077-bornholtA.pdf | Nov 2026 (opened 2026-11-10) |
| Signaloid C0 | https://signaloid.com/ | Nov 2026 (opened 2026-11-10) |
| Property-based testing gaps | arxiv:2410.10908 §4 | Oct 2024 |
| IEEE 1788-2015 Java request | bugs.openjdk.org JDK-8377622 | 2026 |
| Sounio GitHub | github.com/Sounio-lang/sounio | Apr 2026 |
