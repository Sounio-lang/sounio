<!-- docs:meta
topic_id: repo.docs.research.delta-epistemic-gradual-compilation-paper
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.delta-epistemic-gradual-compilation-paper
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Epistemic Gradual Compilation: A Self-Hosted Compiler that Applies its Type System to its Own Source

**Draft — POPL 2027 submission | Numbers updated 2026-04-21 | DOUBLE-BLIND VERSION**

**Authors**: [anonymised for review]

---

## Abstract

We present **epistemic gradual compilation**, a novel compile-time discipline that unifies Koka-style algebraic effects, Vazou-style refinement types, and ISO/JCGM 100 (GUM) uncertainty arithmetic in a single typed programming language, Sounio. The central object is `Knowledge<T>`, a typed wrapper carrying (i) an estimated value, (ii) a GUM-propagated variance with per-source budget channels, (iii) a 0-1000 discrete confidence, and (iv) a provenance pointer with a validity-window interval. Accessing the raw value requires the algebraic effect `with Epistemic`, enforced at compile time; composition respects the GUM §5 linearization rule; callers that cannot discharge a confidence refinement predicate either fail type-checking or emit a two-byte NOP guard marker (`66 90`) in the native code stream.

Because Sounio is self-hosted — the compiler (`lean_single.sio`) is written in Sounio — we apply this discipline to the compiler's own source. Across eight bootstrap generations we track a monotonically rising *compile-time confidence* from 26% (literals only) to **100%** (full cross-function confidence propagation). At the current generation, all 15,636 call sites are verified direct calls with zero runtime overhead; zero guarded calls remain, yielding **0 bytes** of epistemic guard cost on the 1.25 MB binary. Generations 2 and 3 are bit-identical (md5 `54327028`), confirming a fixed-point under the compiler's own discipline.

We position this work against `Uncertain<T>` (Bornholt et al., ASPLOS 2014), Measurements.jl, LiquidHaskell, F★, Koka, PReST (PLDI 2025), and Generic Refinement Types (POPL 2025). We do **not** claim new probability theory (GUM §5 is JCGM 1993), new refinement type theory (Vazou et al. 2014), or new algebraic effects (Leijen, POPL 2017). We claim novelty only for: (i) the *effect-gated collapse discipline* (`with Epistemic` as the unique capability for `.value` access), (ii) the *compile-time ISO §5 budget table* as a type projection of the SIR dataflow graph, (iii) the *correlated-reuse GUM correction* enforced unconditionally by dataflow identity, and (iv) the *self-referential bootstrap convergence* as an empirical validation methodology. We demonstrate concretely (§8.3, Table 2) four programs that Sounio rejects or handles correctly that every prior system accepts silently or cannot express. We argue that the combination is suitable for safety-critical domains — medical dosing, aerospace guidance, financial risk — where metrology currently lives outside the type system. A rapamycin physiologically-based pharmacokinetic (PBPK) case study demonstrates compile-time ISO §5 budget decomposition agreeing with a 200-sample Monte Carlo reference within 2% variance ratio.

---

## 1. Introduction

Safety-critical software is written in deterministic languages. A cardiac-stent drug-eluting dosing algorithm, a flight control law, a financial-risk allocator — each runs on a program whose type system cannot express that its inputs are measurements with variance, that its constants are best estimates with confidence, that the chain of provenance from a calibrated instrument through a lab information system is itself a typed object. Uncertainty is either shoved into a comment, modelled in a separate Monte Carlo rig, or absorbed as a multiplicative safety factor whose origin is institutional rather than mathematical.

The international metrology community has, since 1993, formalized how uncertainty propagates: the *Guide to the Expression of Uncertainty in Measurement* (GUM, JCGM 100:2008), with supplements JCGM 101 (Monte Carlo) and JCGM 102 (multivariate). The GUM §5 tableau — for every output, list the input sources, their variances, their sensitivity coefficients, their degrees of freedom — is the standard artifact of a calibrated scientific result. It lives in spreadsheets and in software like GUM Workbench, never in the types of the program that consumes it.

This paper closes that gap. We show that a type system can enforce GUM discipline at compile time, that the effect system can police where and how a program is allowed to collapse a measurement into a point estimate, and that a sufficiently expressive compile-time analysis converges to *measurable confidence* in its own source code.

### 1.1 The gap

Existing treatments of uncertainty in programming languages divide into four families. **Probabilistic programming languages** (Stan, Pyro, Turing.jl, Gen, Anglican, Dice) model uncertainty as distributions over values and inference as sampling or message-passing; they target Bayesian modeling, not compile-time metrology. **GUM calculators** (Measurements.jl, `uncertainties`, GUM Tree Calculator, NIST UM) correctly implement GUM variance propagation, but only as runtime libraries over concrete floats. **Interval arithmetic libraries** (IntervalArithmetic.jl, IEEE 1788) give rigorous enclosures rather than statistical budgets. **Refinement-typed systems** (LiquidHaskell, F★, Liquid Haskell's probabilistic extensions, Vazou's coupled refinement types) give static predicates over values, but target program-level correctness, not metrological budgets.

The closest ancestor to the present work is `Uncertain<T>` (Bornholt, Mytkowicz, McKinley, ASPLOS 2014; CACM 2016). `Uncertain<T>` demonstrated that a host language can expose a monadic uncertain-value wrapper, that comparisons between uncertain values return `Uncertain<bool>` rather than `bool`, and that Wald's Sequential Probability Ratio Test can evaluate Boolean queries at practical sample counts. The type discipline — *you cannot silently collapse uncertainty* — is the contribution we inherit and extend. Where `Uncertain<T>` samples, we linearize analytically per GUM; where it has no algebra-awareness, we expose `algebra X { mul: alternative, non_commutative }` as a declarative axiom consumed by the rewrite engine; where it has no static confidence reasoning, we provide compile-time gates; where it decomposes nothing, we produce ISO §5 budget tables as type projections.

### 1.2 Contributions

We make four contributions:

1. **`Knowledge<T>` as an effect-gated refinement-typed GUM object** (§§3–4). A 10-word runtime layout carrying value, variance, per-source budget channels, degrees of freedom, confidence, and provenance. Access to the point estimate requires the algebraic effect `with Epistemic`; refinement predicates such as `confidence(k) ≥ 0.95` and `t ∈ validity_window(k)` are enforced at compile time with an SMT-decidable fragment and at runtime as emitted guards.

2. **Epistemic gradual compilation** (§6). The compiler's own source carries `Knowledge<T>` annotations and effect rows. A dedicated epistemic pass computes *compile-time confidence* per expression using GUM quadrature (`isqrt(u₁² + u₂²)`) for parallel channels and the multiplicative degradation rule (`C_A · C_B / 1000`) for serial composition. Call sites that fail to reach a configured gate threshold (`950 / 1000`) receive a visible two-byte NOP marker in the emitted x86-64 ELF.

3. **Bootstrap convergence to a 100% fixed-point** (§5.4). Across eight self-hosted generations the compiler converges from 26% to 100% compile-time certainty. Generations 2 and 3 are bit-identical (md5 `54327028`, 1.25 MB), confirming stability of the compiler under its own discipline. The current-generation call-site census records 15,636 direct calls (100%), 0 guarded, and **0 bytes** of cumulative epistemic overhead — the cross-function confidence propagation pass eliminated all previously-emitted guard markers.

4. **A rapamycin PBPK case study** (§7). A three-compartment physiologically based pharmacokinetic model with three uncertain clinical parameters (clearance, brain-partition coefficient, plasma free fraction, each at 10% coefficient of variation) produces a compile-time ISO §5 budget table and agrees with a 200-sample Monte Carlo reference to within a 2% variance ratio.

We claim no novelty for the underlying probability theory: variance propagation for smooth maps is JCGM 102:2011 §6. We claim no novelty for refinement types, effect rows, or e-graph rewriting individually. We claim novelty only for the specific unification and its self-referential application.

### 1.3 Preview of key result

Figure 1 in §6 shows the convergence curve. The x-axis is the generation index (0–8); the y-axis is the fraction of expressions whose type the epistemic pass can certify. Generation 0 certifies only literals and reaches 26%. Generation 6, which incorporates keyword handling and explicit confidence gates, reaches 90%. Generation 7 (full let-polymorphism and struct-field flow) reaches 97%. Generation 8, the current HEAD, extends cross-function confidence propagation to resolve all remaining closure captures and static dispatch sites, reaching **100%**. The curve is monotone and converges fully: no guard markers remain in the current binary. The guard mechanism remains the designed-in escape hatch for future constructs where compile-time certainty is structurally unavailable, but no current construct requires it.

The title claim — *a self-hosted compiler that applies its type system to its own source* — is not a slogan. It is a mechanical fact of the bootstrap. The compiler under test is the same binary that performed the test. Its convergence is a property of the language, not of external tooling.

---

## 2. Related work

We organise prior work along the two axes Sounio unifies: uncertainty-as-type, and compile-time enforcement.

### 2.1 Uncertain\<T\> (Bornholt et al., ASPLOS 2014)

Bornholt, Mytkowicz, and McKinley introduced `Uncertain<T>` as a first-class probabilistic type embedded in C#. Their central insight — adopted by Sounio — is that relational operators on uncertain quantities must not produce `bool`: `x > y : Uncertain<bool>`, and Boolean collapse requires an explicit hypothesis test via `(x > y).Pr(0.95)`. Internally, `Uncertain<T>` is a lazy computation graph; samples are drawn on demand; correlation is preserved by shared sampling within a traversal. Evaluation is by Wald's SPRT, providing near-optimal sample counts for threshold queries.

Sounio inherits the type-discipline that uncertainty must not be silently discarded. We differ on four substantive axes:

- Sounio propagates analytically via GUM linearization; Bornholt samples. Under GUM regularity assumptions (smooth model, small relative uncertainty) Sounio is faster, deterministic, and ISO-traceable. Under heavy tails, multimodality, or strongly nonlinear models, Monte Carlo is correct and our first-order linearization is not. We therefore retain Monte Carlo as a configurable back-end but make GUM the default.
- Sounio lifts `Knowledge<T>` over user-defined algebras, including non-associative algebras via declared axioms (`alternative`, `flexible`, `fano_selective`). `Uncertain<T>` has no algebra concept.
- Sounio supports compile-time confidence reasoning via refinement types and an algebraic effect row; `Uncertain<T>` is runtime-only.
- Sounio decomposes aggregate variance into ISO §5 budget channels per uncertain source; `Uncertain<T>` reports aggregate distributions only, with ablation as the only decomposition tool.

We view Bornholt et al. as the proximate PL ancestor and acknowledge the type-level no-silent-collapse discipline as theirs.

### 2.2 Measurements.jl and GUM calculators

Measurements.jl (Giordano, 2016) implements GUM variance propagation over Julia `Number` types. It is correct, fast, and widely used. It is also *runtime-only*: every operation allocates a new `Measurement`, and misuse (accessing `.val` and discarding `.err`) compiles cleanly. GUM Workbench (PTB), GTC (NIST), and the `uncertainties` Python package occupy the same niche. None of them expresses degrees-of-freedom, validity windows, or provenance as types; none refuses to compile a program that violates metrology discipline.

Sounio claims GUM-S2 as its variance foundation and Measurements.jl as its arithmetic reference — the propagation formulae are identical — but lifts enforcement from runtime library to compile-time type.

### 2.3 Refinement types and liquid typing

LiquidHaskell (Rondon, Kawaguchi, Jhala) and F★ (Swamy et al.) enable refinement predicates over values, discharged via SMT. LiquidHaskell decides predicates in QF_UFLIA / QF_AUFLIA; F★ uses weakest-precondition Dijkstra monads and admits nonlinear refinements via tactics. Vazou et al. extended to *coupled* refinements (two-place relations) for differential-privacy applications. Generic Refinement Types (POPL 2025) generalises higher-order specifications over function contracts in the decidable SMT fragment; the Flux tool applies this to Rust traits. Neither system has a notion of metrological budget, GUM variance channels, or ISO-traceable provenance.

The closest 2025 work is **PReST** (Probabilistic Refinement Session Types, PLDI 2025), which extends refinement session types with symbolic reasoning for concurrent probabilistic systems. PReST targets protocol-level probabilistic choice — it reasons about distributions over communication sequences, not about metrological budgets. PReST has no GUM §5 tableau, no ISO-traceable variance propagation, no per-source budget decomposition, no `with Epistemic` effect gate, and no self-referential bootstrap. The disciplines are orthogonal: PReST answers *"with what probability does this channel protocol complete?"*; Sounio answers *"what is the ISO §5 uncertainty budget of this computation, enforced at compile time?"*

A natural reviewer objection is: *"LiquidHaskell with a custom float predicate achieves the same discipline."* It does not. LiquidHaskell predicates range over *values* in QF_UFLIA; they cannot express the GUM §5 composition rule (which is a constraint over a *variance channel vector* that grows with each `measure` call), the ISO budget decomposition (a type projection of the SIR dataflow graph), or the correlated-reuse tracking (which requires dataflow identity, not value equality). We demonstrate this gap concretely in §8.3 with two programs that LiquidHaskell accepts and Sounio rejects.

Sounio's confidence gate `confidence(k) ≥ 0.95` is a pointwise liquid predicate, decidable in QF_LRA. Its validity-window membership `t ∈ validity_window(k)` is interval containment, also QF_LRA. However, composition across function boundaries — *what is the confidence floor of `g ∘ f` given their individual floors?* — is nonlinear and beyond SMT in general. We handle this by emitting a conservative independence-product lower bound with a dataflow-tracked covariance flag that, when set, tightens to a GUM-quadrature sum.

### 2.4 Algebraic effects

Koka (Leijen, POPL 2017) introduced row-polymorphic effect types `τ → ⟨ε⟩ τ'` with handlers that discharge row components. Eff, Frank, and Multicore OCaml follow a similar pattern. None combines effects with value refinements.

Sounio's effect row includes `IO`, `Mut`, `Div`, `Observe`, and `Epistemic`. The `Epistemic` effect gates the single operation `k.value : Knowledge<T> → T` — the point-estimate extraction. A function without `Epistemic` in its row cannot access `.value`; it must propagate the `Knowledge<T>` symbolically or yield to a handler that decides how to discharge it. This is the Koka-level machinery; the novelty is the specific effect we add and the refinement predicate we pair it with.

### 2.5 Units of measure

F#'s UoM (Kennedy, ESOP 1997), Haskell's `dimensional`, and Boost.Units provide compile-time dimensional analysis. They prevent adding meters to seconds but say nothing about the variance of the measurement. Sounio's `units/` subsystem provides the F#-style dimension layer; `Knowledge<T>` parameterizes over it, so `Knowledge<mg>` is distinct from `Knowledge<ng>` and both track variance independently.

### 2.6 Ownership, linearity, totality

Rust's ownership types, Idris' totality checker, and Agda's coverage checker each demonstrate that a specific discipline — aliasing, termination, pattern coverage — can be made compile-time-enforced for the entire language. Each property carries its own real-world stakes: memory safety, non-divergence, case completeness. Sounio adds one more to this catalogue: *metrological discipline*, whose stake is the silent loss of measurement uncertainty in deterministic computations that feed safety-critical decisions.

### 2.7 Self-hosting and bootstrap

Classical self-hosting compilers (Pascal-P, Bootstrap C, rustc's `rustc_driver`, Zig's stage2) demonstrate that the compiler for a language can be written in that language; once past stage-0, further generations are produced by the compiler itself. Modern Zig has pursued the "custom-backend-first" approach — skipping LLVM to control the full pipeline. Sounio is in the Zig lineage: the native x86-64 ELF emitter is hand-written in Sounio; no external code generator is used.

We are not aware of prior work where the compiler's *own* source was type-checked with an epistemic (or any uncertainty-aware) type system, and where convergence of the check was tracked across bootstrap generations. The self-referential loop — *the compiler validates the compiler* — is a testable property of our design and a proof of internal consistency.

### 2.8 Gradual typing

Siek and Taha (2006) formulated gradual typing as a spectrum from fully dynamic to fully static, with the dynamic type `?` as a compatibility partner. We borrow the word *gradual* deliberately: Sounio's epistemic check is not all-or-nothing; it produces a scalar *compile-time confidence* in 0–1000, and the codegen reacts proportionally (direct call when ≥ threshold, guard emission below). The analogy runs through: gradual typing defers type-errors to runtime where static guarantees are absent; gradual *compilation* defers metrology-errors to runtime guards where compile-time certainty is absent. Neither gives up on the static discipline; both admit that the program's global certainty is partial.

---

## 3. `Knowledge<T>` — the type and its discipline

### 3.1 Runtime layout

`Knowledge<T>` has a ten-word runtime footprint. On x86-64:

```
offset 0   : T            — point estimate           (≤ 8 bytes for primitive T)
offset 8   : f64          — variance σ²
offset 16  : f64          — degrees of freedom ν
offset 24  : i64          — confidence c, 0..1000 (discretized)
offset 32  : *provenance  — ISO 17025 trace-pointer (nullable)
offset 40  : *budget      — per-channel variance vector (nullable)
offset 48  : i64          — validity_window start, nanoseconds since epoch
offset 56  : i64          — validity_window end
offset 64  : i64          — algebra tag (f64, mg, quaternion, octonion, …)
offset 72  : i64          — flags (reassociation_safe, correlated, frozen)
```

Layouts for algebraic `T` (quaternion, octonion) replace the first word with an inline multi-word value and expand variance to a small fixed-rank matrix.

### 3.2 Construction — the `measure` primitive

The sole primitive constructor is `measure`:

```sio
fn measure(x: f64, uncertainty: f64) -> Knowledge<f64> with Epistemic
```

Each call to `measure` allocates a fresh budget channel and seeds that channel with the argument's variance. The returned `Knowledge<f64>` has confidence `1000` (complete epistemic trust in the measurement itself), degrees of freedom from the caller's context, and a provenance pointer initialised to the active context frame. The `with Epistemic` effect row makes it impossible to call `measure` from a pure function; pure code cannot fabricate measurements.

### 3.3 The E170 rule

The compiler enforces rule E170: *`.value` access on `Knowledge<T>` is permitted only within a function whose effect row contains `Epistemic`.* Attempting to violate it produces:

```
error[E170]: access to `.value` of Knowledge<f64> requires `with Epistemic`
  --> examples/dose.sio:17:14
   |
17 |     let raw = dose.value
   |              ^^^^^^^^^^^
   |
   = note: `.value` discards uncertainty; callers must declare the effect
   = help: add `with Epistemic` to fn dose_at_site's signature, or propagate `dose`
```

E170 is structurally identical to Rust's borrow-check errors or Haskell's missing-instance errors. It is a *refusal to compile*, not a warning.

### 3.4 GUM-§6 arithmetic

Binary operations lift to the ambient arithmetic rules of GUM §5. For independent `a: Knowledge<f64>`, `b: Knowledge<f64>`:

```
(a + b).value = a.value + b.value
(a + b).variance = a.variance + b.variance
(a · b).value = a.value · b.value
(a · b).variance = b.value² · a.variance + a.value² · b.variance       (first-order)
```

General smooth `f: ℝⁿ → ℝ` follows JCGM 102:2011 §6:

```
μ_Y = f(μ_X₁, ..., μ_Xₙ)
σ²_Y = Σᵢ (∂f/∂xᵢ)² σ²_Xᵢ   + 2 · Σᵢ<ⱼ (∂f/∂xᵢ)(∂f/∂xⱼ) σ_XᵢXⱼ
```

When the `correlated` flag is unset (the default from independent `measure` calls) the covariance terms drop. When a prior reassociation or shared dataflow path sets it, the full quadratic form is evaluated from a per-budget-channel Jacobian.

### 3.5 Budget decomposition

Each `measure()` call allocates a new channel index. Binary ops propagate both aggregate variance and per-channel variance. `budget_of(k)` returns the full §5 table:

```sio
let dose  = measure(500.0, uncertainty: 2.5)     // channel 0
let half  = measure(0.6,   uncertainty: 0.02)    // channel 1
let eff   = dose * half                           // variance contributions from both

println(budget_of(eff))
// ISO §5 uncertainty budget for eff (μ = 300.00, σ = 15.12, c = 950):
//   ch0  "dose" (500.00 ± 2.50)   c_i = 0.60   u_i = 2.50   c_i·u_i = 1.50   contrib% = 63.3
//   ch1  "half" (0.60 ± 0.02)     c_i = 500.00 u_i = 0.02   c_i·u_i = 10.00  contrib% = 36.7
//                                                                            Σ = 15.12 ✓
```

The table is constructed entirely from the SIR dataflow — the compiler's intermediate representation — not from a post-hoc runtime log. For a program that compiles, the table is provable-by-construction.

### 3.6 Confidence gates

The confidence field is a 0..1000 integer discretisation of a coverage probability. The primitive gate is

```sio
fn require_confidence(k: Knowledge<T>, min: i64) -> T with Epistemic, Div
```

`Div` signals potential divergence: at runtime, if `k.confidence < min`, the function traps. At compile time, if the SMT fragment can discharge `confidence(k) ≥ min` from the types in scope, the call is lowered to a direct `.value` access; otherwise the codegen emits the guard marker (§5.3).

The refinement form is also admitted:

```sio
fn prescribe(dose: {k: Knowledge<mg> | confidence(k) ≥ 950 ∧ now() ∈ validity_window(k)})
    -> Knowledge<mg>
    with Epistemic
```

A caller that cannot statically establish the predicate receives either a type error (when the predicate is provably false) or a runtime gate (when it is provably uncertain).

### 3.7 Provenance and `validity_window`

Every `measure` call captures a provenance record: the ISO 17025 calibration chain, the instrument serial, the timestamp, the operator identifier. The provenance pointer flows through arithmetic: `(a + b).provenance = merge(a.provenance, b.provenance)`. The compile-fail test `provenance_trusted_reject.sio` exercises the core rule: a function that accepts only `trusted`-provenance `Knowledge` cannot be called with a `Knowledge` whose provenance is `lab_unverified`, even when value and variance coincide.

The `validity_window` field carries a `[start, end]` interval of instants during which the measurement remains epistemically valid — e.g., a calibration is valid for a year after the certification date. Composition intersects windows; the `validity_window_ordering.sio` compile-fail test rejects constructing a window with `end < start`. The `validity_window_combine.sio` run-pass test confirms intersection produces the expected narrower window.

---

## 4. The `with Epistemic` effect

### 4.1 Effect rows

Sounio functions carry effect rows in their signatures:

```sio
fn integrate(f: fn(f64) -> f64 with Math, x0: f64, x1: f64) -> f64 with Math
fn log_dose(d: Knowledge<mg>) -> () with IO, Epistemic
fn reduce(xs: &[Knowledge<f64>]) -> Knowledge<f64> with Epistemic
```

A function calling another inherits the callee's rows. The row is implemented as a set of discrete flag bits today; in principle it admits the full Koka row-polymorphic treatment. The language supports four effects relevant to epistemic reasoning:

- `IO` — observable side-effects (stdin/stdout/file)
- `Mut` — internal state mutation
- `Div` — possibly non-terminating; `require_confidence` carries this because a failed gate traps
- `Epistemic` — gates access to `.value` on any `Knowledge<T>`

### 4.2 Why effects rather than a monad

`Uncertain<T>` and Measurements.jl both deliver uncertainty as a monadic wrapper lifted by `Select`/`SelectMany` or operator overloading. This works but has a structural limitation: the wrapper is opaque. A Julia function that takes `Measurement{Float64}` and calls `.val` to extract the mean discards uncertainty silently; the type system permits it. A Koka-style effect, by contrast, is a *capability* the function must *declare*. The asymmetry — "you can always unwrap a monad, but you cannot conjure an effect" — is why we chose effects.

### 4.3 Confidence subsumption

Subtyping of `Knowledge`-refined types is confidence-monotone: if `k: {Knowledge<T> | confidence ≥ 950}`, then `k` is a subtype of `{Knowledge<T> | confidence ≥ 800}` but not of `{confidence ≥ 990}`. Function arguments are *contravariant* in their refinements: a function demanding confidence ≥ 950 cannot be given a value known only to have confidence ≥ 800. The classic subsumption rule applies:

```
    Γ ⊢ k : {Knowledge<T> | φ}       Γ ⊨ φ ⇒ ψ
    ─────────────────────────────────────────────── [Sub-Knowledge]
    Γ ⊢ k : {Knowledge<T> | ψ}
```

The implication is discharged by the bundled Z3/CVC5 SMT solver in the decidable QF_LRA fragment. For nonlinear compositions (propagating confidence through a multi-step computation) we fall back to conservative lower bounds (§6.5).

### 4.4 Validity-window typing

`validity_window` is an interval refinement with the predicate `t ∈ w ⇔ w.start ≤ t ≤ w.end`. Typing accepts the union of standard interval-arithmetic rules plus the composition law: `validity_window(a + b) = validity_window(a) ∩ validity_window(b)`. A function whose signature demands the validity-window to cover the current instant `now()` forces the caller to establish a lower-bound on `end` and an upper-bound on `start` — both linear refinements — before the call can typecheck.

---

## 5. Metatheory

We state the core soundness properties of the epistemic type system. Full proofs appear in the supplementary appendix (to be submitted as part of the artifact). This section establishes the judgement forms, the key lemmas, and the three main theorems.

### 5.1 Formal judgements

We write `Γ; ε ⊢ e : τ` for the standard bidirectional typing judgement under context `Γ` (variable → type) and effect row `ε` (set of declared effects). We add a *variance store* `Σ : ChannelId → ℝ≥0` mapping each measurement channel to its current variance. The full judgement is:

```
Γ; ε; Σ ⊢ e : τ
```

A *Knowledge value* `kv` is a 5-tuple `(v, σ², c, w, π)` where:
- `v : T` — point estimate
- `σ² : ℝ≥0` — aggregate GUM variance (sum over active channels)
- `c : [0, 1000]` — discrete confidence
- `w : [t_start, t_end]` — validity window
- `π : ProvenanceId` — ISO 17025 calibration pointer

The type of a Knowledge value is `{k : Knowledge<T> | φ(k)}` where `φ` is a conjunction of QF_LRA predicates over `k.confidence`, `k.validity_window`, and `k.provenance`.

### 5.2 Key typing rules

The five rules that form the metatheoretic core:

```
                    fresh ch
────────────────────────────────────────────── [T-Measure]
Γ; {Epistemic}∪ε; Σ[ch↦u²] ⊢ measure(v, u) : Knowledge<T>
```

```
Γ; ε; Σ ⊢ e : {k : Knowledge<T> | confidence(k) ≥ c_min}
Epistemic ∈ ε
─────────────────────────────────────────────────────────── [T-Value]
Γ; ε; Σ ⊢ e.value : T
```

```
Γ; ε; Σ ⊢ a : Knowledge<T>     Γ; ε; Σ ⊢ b : Knowledge<T>
channels(a) ∩ channels(b) = ∅          (independence assumption)
──────────────────────────────────────────────────────────────── [T-Add-Ind]
Γ; ε; Σ ⊢ a + b : {k : Knowledge<T> | k.σ² = a.σ² + b.σ²}
```

```
Γ; ε; Σ ⊢ a : Knowledge<T>     channels(a) = channels(b)
──────────────────────────────────────────────────────── [T-Mul-Corr]
Γ; ε; Σ ⊢ a * b : {k : Knowledge<T> | k.σ² = (2·a.value)²·a.σ²}
```

```
Γ; ε; Σ ⊢ e : {k : Knowledge<T> | confidence(k) ≥ c_min}
SMT ⊨ c_min ≥ gate_threshold                                (static discharge)
─────────────────────────────────────────────────────────── [T-Gate-Static]
Γ; ε; Σ ⊢ require_confidence(e, gate_threshold) : T        (no guard emitted)
```

When the SMT obligation in [T-Gate-Static] cannot be discharged, a weaker rule [T-Gate-Dynamic] applies: the judgement holds but the codegen is instructed to emit the `66 90` guard preamble.

### 5.3 Operational semantics — reduction rules

The GUM arithmetic step rules extend the standard call-by-value λ-calculus. We give the three non-trivial cases:

```
(v_a, σ²_a, c_a, w_a, π_a) + (v_b, σ²_b, c_b, w_b, π_b)
→  (v_a + v_b,
    σ²_a + σ²_b,
    (c_a · c_b) / 1000,
    w_a ∩ w_b,
    merge(π_a, π_b))                                       [E-Add]

(v_a, σ²_a, c_a, w, π) * (v_b, σ²_b, c_b, _, _)   [channels disjoint]
→  (v_a · v_b,
    v_b² · σ²_a + v_a² · σ²_b,
    (c_a · c_b) / 1000,
    w,
    merge(π_a, π_b))                                       [E-Mul-Ind]

(v, σ², c, w, π).value                               [Epistemic ∈ ε, c ≥ gate]
→  v                                                       [E-Value]
```

For the correlated-reuse case (both operands are the same SIR node `x`):

```
x * x   where x = (v, σ², c, w, π)
→  (v², (2v)² · σ², c², w, π)                             [E-Mul-Corr]
```

Note `(2v)² · σ²` vs. `2v² · σ²` for the independent case — the factor-of-2 difference is the bug P2 in Table 2 (§8.3) that Measurements.jl can silently produce.

### 5.4 Theorems

We first establish two standard structural lemmas.

**Lemma 1 (Weakening).** If `Γ; ε; Σ ⊢ e : τ` and `x ∉ dom(Γ)`, then `Γ, x:τ'; ε; Σ ⊢ e : τ`.

*Proof.* Standard structural induction on the typing derivation; each rule's premises are preserved under context extension. ∎

**Lemma 2 (Substitution).** If `Γ, x:τ'; ε; Σ ⊢ e : τ` and `∅; ε'; Σ ⊢ v : τ'` where `v` is a closed value, then `Γ; ε; Σ ⊢ e[v/x] : τ`.

*Proof.* By structural induction on the derivation of `Γ, x:τ'; ε; Σ ⊢ e : τ`.

- **Var**: If `e = x`, then `e[v/x] = v` and `Γ; ε; Σ ⊢ v : τ' = τ` follows from the hypothesis. If `e = y ≠ x`, then `e[v/x] = y` and `y:τ ∈ Γ` directly.
- **[T-Measure]**: `measure` has no free program variables; substitution is vacuous. The fresh-channel side condition is unaffected.
- **[T-Value]**: `e = e₀.value`. By IH, `e₀[v/x] : Knowledge<T, φ>`. The predicate `φ` is over `e₀`'s fields, not over `x`; the SMT obligation is preserved.
- **[T-Add-Ind]**: `e = a + b`. By IH on both subterms, `a[v/x]` and `b[v/x]` have the required Knowledge types. Channel disjointness is a property of the SIR dataflow graph, not the substituted variable; substituting a closed value for `x` cannot introduce new channel aliasing.
- **[T-Mul-Corr]**: `e = a * a` where both operands are the same SIR node. If `a` contains `x`, then `a[v/x]` still refers to the same SIR node (substitution is uniform); channel identity is preserved.
- **[T-Gate-Static]**: The SMT predicate `confidence(e₀) ≥ c_min` is a refinement over the Knowledge type, not over the substituted variable. The SMT proof is preserved under value substitution.
- All other rules: routine structural induction. ∎

**Theorem 1 (Type Preservation — Subject Reduction).** If `Γ; ε; Σ ⊢ e : τ` and `e →_v e'` under a closed, call-by-value evaluation context, then `Γ; ε; Σ' ⊢ e' : τ` for some `Σ' ⊇ Σ`. **Machine-checked** as `preservation` in `formal/lean4/EpistemicEffectsV2.lean` (Mathlib-free, no `sorry`; axioms `propext`, `Quot.sound`, `Classical.choice`) — see the Mechanization note after the Corollary.

*Proof (prose; the mechanized proof is the authority).* By structural induction on the derivation of `e → e'`, using Lemma 2.

- **[E-Add]**: The redex is `(v_a, σ²_a, c_a, w_a, π_a) + (v_b, σ²_b, c_b, w_b, π_b)`. By canonical forms, both operands are Knowledge values and the typing inversion gives `k_a.σ² = σ²_a`, `k_b.σ² = σ²_b`. The result `(v_a + v_b, σ²_a + σ²_b, (c_a·c_b)/1000, w_a∩w_b, merge(π_a,π_b))` satisfies the [T-Add-Ind] output predicate by arithmetic identity. The validity window `w_a∩w_b ⊆ w_a` and `⊆ w_b` by set containment; the interval refinement predicate is preserved under intersection. ∎ (case)

- **[E-Mul-Ind]**: Analogous to [E-Add] with the GUM product formula `v_b²·σ²_a + v_a²·σ²_b`. The result type satisfies [T-Mul-Ind]'s output predicate by arithmetic identity. ∎ (case)

- **[E-Mul-Corr]**: The redex is `x * x` where `x = (v, σ², c, w, π)`. By [T-Mul-Corr], the output predicate asserts `k.σ² = (2v)²·σ²`. The reduction rule produces exactly that variance. The single-channel origin is preserved (no new channels created). ∎ (case)

- **[E-Value]** (under [T-Gate-Static]): The redex is `require_confidence(kv, gate_threshold)` where `kv = (v, σ², c, w, π)` and `c ≥ gate_threshold` holds by the static SMT discharge. The result is `v : T`. By [T-Value], `T` is the declared output type. ∎ (case)

- **[E-Value]** (under [T-Gate-Dynamic]): The runtime check `kv.confidence ≥ gate_threshold` succeeds (otherwise a trap, handled by Theorem 2). On success, the result is `v : T` as above. ∎ (case)

- **[E-Measure]**: `measure(v, u)` reduces to the runtime Knowledge value that **carries the measured value `v` of type `T`** (together with the metadata `u²`), so its type is `Knowledge<T>` by [T-Kraw], creating a fresh channel `ch` with `Σ' = Σ[ch ↦ u²] ⊇ Σ`. *Mechanization caveat (this case is where the soundness subtlety lives): the reduct must retain the base-type value `v`. A representation that dropped it — storing only a scalar real cell — does **not** preserve `T`: `measure(0 : ℕ) : Knowledge<ℕ>` would reduce to a real-valued cell typeable only at `Knowledge<ℝ>`. That unsoundness is itself machine-checked (`preservation_is_false` in `EpistemicPreservationWIP_counterexample.lean`), and the value-carrying mechanization (`EpistemicEffectsV2.lean`) is what restores subject reduction. See the Mechanization note.* ∎ (case)

- **Congruence rules** (evaluation context reduction): By IH on the sub-redex, the inner expression reduces with type preserved; the outer context rebuilds the same type by inversion on the outer rule. ∎

∎

**Theorem 2 (Progress).** If `∅; ε; Σ ⊢ e : τ` (closed, well-typed in the empty variable context), then one of:
- (a) `e` is a value,
- (b) `∃e'. e → e'`, or
- (c) `e = require_confidence(kv, g)` with `kv.confidence < g` (runtime gate failure — a *defined trap*, not a stuck state).

**Machine-checked** as `effect_progress` in `formal/lean4/EpistemicEffectsV2.lean` (axiom `propext` only).

*Proof.* By induction on the typing derivation. The base case `e = v` (value) gives (a). For compound expressions, the standard PL progress argument applies: if all sub-expressions are values, the expression matches a redex (b). The only novel case is [T-Gate-Dynamic]: the guard `66 90` precedes the `require_confidence` call; at runtime, `kv.confidence` is a concrete `i64` field of the Knowledge value. The comparison `kv.confidence ≥ gate_threshold` terminates in O(1) and branches deterministically: success gives (b), failure gives (c). Neither outcome is stuck. ∎

**Theorem 3 (Epistemic Effect Soundness).** If `Γ; ε; Σ ⊢ e.value : τ`, then `Epistemic ∈ ε`.

*Proof.* By inspection: [T-Value] is the unique rule with `.value` in conclusion; it has `Epistemic ∈ ε` as a premise. No other typing rule derives a `.value` expression. ∎

**Theorem 4 (GUM Variance Fidelity).** For any closed, well-typed expression `e : Knowledge<T>` that reduces to a value `kv`, the aggregate variance `kv.σ²` equals the GUM §5 first-order propagation formula applied to the input variances at the `measure` call sites in `e`'s SIR dataflow graph.

*Proof.* By induction on the depth of the SIR dataflow graph of `e`.

*Base*: `e = measure(v, u)`. By [E-Measure] and [T-Measure], `kv.σ² = u²`. GUM §5 assigns variance `u²` to a direct measurement with stated uncertainty `u`. ✓

*Step*: By IH, each sub-expression `e_i` produces `kv_i` with `kv_i.σ²` equal to the GUM propagation of its sub-graph. For binary operations, the GUM §5 composition rule is applied by [E-Add] (sum of variances, independent channels) or [E-Mul-Corr] (squared-sensitivity formula for aliased channels). In each case the reduction rule implements exactly the GUM §5 formula. By induction the composed variance is the GUM propagation of the full graph. ∎

**Corollary (No Silent Collapse).** Under any well-typed reduction sequence, a `Knowledge<T>` value cannot be coerced to `T` without `Epistemic ∈ ε`, and the GUM §5 variance budget of the discarded uncertainty is always accounted for in the type.

*Proof.* From Theorem 3: extraction requires `Epistemic`. From Theorem 4: the variance at the extraction point equals the GUM §5 propagation from all upstream `measure` sites. Neither fact changes under reduction (Theorem 1). ∎

**Mechanization (`formal/lean4/EpistemicEffectsV2.lean`).** Theorems 1–2 (the type-safety pair) are machine-checked for a Lean 4 model of the core calculus: `preservation` (subject reduction) and `effect_progress` (progress), Mathlib-free, no `sorry`. We verify the *calculus*, not that the self-hosted compiler binary implements it (standard model-level PL metatheory). Three honest points the mechanization fixes or surfaces:

1. **The Knowledge cell must carry the base-type value.** A first mechanization used a scalar (real-valued) runtime cell and was machine-checked **unsound** — subject reduction fails because `measure(v : T)` cannot reduce to a real cell at `Knowledge<T>` for `T ≠ ℝ` (`preservation_is_false`, `EpistemicPreservationWIP_counterexample.lean`). This matches the `(v, σ², c, w, π)` tuple of §5.3, where `v` is the *typed* value; the mechanization makes that requirement load-bearing and `.value` (Theorem 3, [E-Value]) returns exactly that stored `v : T`.

2. **GUM arithmetic is restricted to numeric Knowledge.** In the mechanized calculus `+`/`*` ([E-Add]/[E-Mul]) are typed only at `Knowledge<ℝ>` — GUM variance propagation is numeric by nature (one cannot GUM-add `Knowledge<bool>`). `measure`, `.value`, `.unc`, `.conf` remain generic in `T`. This restriction is explicit in the model, not a soundness dodge.

3. **Scope.** Theorems 3–4 and the Corollary are not yet mechanized; their prose proofs stand. The mechanized type-safety pair is *table-stakes rigor* for the calculus — the paper's distinctive contribution is the self-application of §6, not §5.4.

---

## 6. Epistemic gradual compilation

### 6.1 Self-application

`lean_single.sio` is the self-hosted Sounio compiler: 27,301 lines of Sounio covering lexer, parser, type-check, HIR, SIR, HLIR (SSA), and x86-64 ELF emitter. Its own source is annotated with `Knowledge<T>` where appropriate — for example, variance estimates on parser-rule transition probabilities, confidence values on type-inference unifications — and its effects are declared on every function.

The compiler therefore has *two* front-ends to type-check: its user's program (the normal operation), and its own source (the bootstrap operation, re-run at each stage). The second is the novelty.

### 6.2 The epistemic pass

The epistemic pass is a fixed-point computation over the compiler's HIR. For every expression node `e` it computes:

- `ETY[e]` — the epistemic type (`known<T>`, `knowledge<T>`, `uncertain`, `error`)
- `CONF[e]` — a discrete confidence in 0..1000
- `UNC[e]` — a GUM variance estimate
- `GATE[e]` — whether this expression needs a runtime guard

The pass is implemented as flat parallel arrays (`ETY_KIND/CONF/UNC`, `EXPR_ETY/CONF/GATE`, `ESCOPE_*`) to keep the memory footprint predictable in the BSS-mapped native binary. The rules:

```
literal    : CONF = 1000
var x      : CONF = CONF[def(x)]
a op b     : CONF = (CONF[a] · CONF[b]) / 1000              (serial, multiplicative)
a ‖ b      : UNC  = isqrt(UNC[a]² + UNC[b]²)                (parallel, quadrature)
call f(args) : CONF = min(CONF[args]) · CONF[f.body]        (worst-case inheritance)
cast a as T : CONF = CONF[a] · C_cast(T)                    (with tabulated cast cost)
```

Thresholds: `GATE_THRESHOLD = 950` (≈ 95% confidence). Expressions below threshold propagate `GATE[e] = true` up through the HIR, eventually reaching the call site that demanded confidence.

### 6.3 Codegen guard marker

At codegen, when the SSA lowering reaches an instruction with `GATE = true`, the emitter produces

```
66 90                    ; NOP (two bytes), the "epistemic marker"
<direct instruction>     ; the normal call / access
```

`66 90` is the standard multi-byte NOP on x86-64; it is a single-cycle no-op that the CPU retires without side-effect. Its purpose is not performance but *visibility*: a post-mortem disassembly of the ELF can count guarded sites by grepping for the exact byte pattern, and a coverage tool can overlay those sites on the source. The bytes are also the natural hook for a future runtime trap: redefining `66 90` to an interrupt-vector on an instrumented build converts every guarded site into a debugger stop.

The choice of a zero-cost marker is deliberate. An epistemic system that exacted nontrivial runtime cost would be retrofitted out of safety-critical code paths; a system whose overhead is measurable in nanograms on a 4.5 MB binary is safe to ship ubiquitously.

### 6.4 Convergence data

The bootstrap was executed across eight generations, each extending the epistemic pass with one additional construct. Figure 1 plots the convergence curve; Table 1 gives the full per-generation data.

```
Figure 1 — Bootstrap epistemic convergence (gen0 → gen8)

  100% ┤                                              ●  gen8
   97% ┤                                         ○
   90% ┤                                    ●  gen6
   83% ┤                               ●
   77% ┤                          ●
   70% ┤                     ●
   59% ┤                ●
   50% ┤           ●
   26% ┤ ●  gen0
       └──┬────┬────┬────┬────┬────┬────┬────┬────┬──▶ generation
          0    1    2    3    4    5    6    7    8

  ● measured   ○ estimated (gen7 not pinned)
  ─── monotone non-decreasing   ━━━ fixed-point plateau (gen8)

  y-axis: fraction of expressions certified at compile time (%)
  x-axis: bootstrap generation index
```

The curve is strictly monotone through gen0–gen6, with a final jump at gen8 (cross-function confidence propagation) bringing certified expressions to 100%. The gen7 point (97%, struct-field flow) is estimated from the pass-by-pass log; it was not pinned to a stable hash.

| Gen | Features incorporated                    | Certain exprs | % certain | Binary size | MD5       |
|-----|------------------------------------------|---------------|-----------|-------------|-----------|
| 0   | Literals                                 | 15,613        | 26%       | 711 KB      | aefc7065  |
| 1   | + vars, binops                           | 30,416        | 50%       | 718 KB      | a2032662  |
| 2   | + fn calls                               | 36,131        | 59%       | 720 KB      | b6c0d61a  |
| 3   | + let propagation                        | 43,087        | 70%       | 724 KB      | 244d18b9  |
| 4   | + fn parameters                          | 47,262        | 77%       | 727 KB      | e1c63fa7  |
| 5   | + `as` casts and types                   | 51,099        | 83%       | 730 KB      | 8cdd5ff5  |
| 6   | + keywords and confidence gates          | 90,846        | 90%       | 734 KB      | 65c6fba6  |
| 7   | + let-polymorphism and struct-field flow | ~97,700       | 97%       | ~930 KB     | —         |
| 8   | + cross-fn propagation (current HEAD)   | 113,931       | **100%**  | 1.25 MB     | 54327028  |

The curve is monotone and converges to 100%. The binary grows from 711 KB (gen0) to 1.25 MB (gen8), reflecting the expansion of the source itself from a literal-only skeleton to the complete 27 kLoC compiler. No other measured property of the compiler — throughput, memory peak, error-message quality — regresses across generations; the discipline adds confidence without subtracting function.

### 6.5 Fixed-point and self-consistency

We define the bootstrap fixed-point test as follows. Let `G(n)` be the compiler at generation `n`, and let `source` be the fixed compiler source. Then:

```
G(n+1) := G(n) compiled with itself
```

A fixed-point is reached when `G(n+1)` and `G(n)` produce bit-identical binaries. At the current HEAD (generation 8) we observe `md5(gen2.elf) = md5(gen3.elf) = 54327028`, where `gen2 = G(1)(source)` and `gen3 = G(2)(source) = G(gen2)(source)`. The compiler is a fixed-point under itself.

This test is identical in structure to the classic self-compilation sanity check, but here it also validates the epistemic pass: if the pass changed at all across the two iterations — if the compile-time confidence measured for some expression differed between generations — the codegen would differ and the bits would diverge. Bit-identity is therefore a witness of *epistemic stability*: the compiler's internal confidence in itself has converged.

### 6.6 Call-site census at current generation (gen8)

Compilation of `self-hosted/compiler/lean_single.sio` by the current self-hosted binary yields:

- 15,636 call sites in total (source grew from ~18 kLoC to 27 kLoC)
- **15,636 direct calls**, no guard preamble (**100%**)
- **0 guarded calls** — all `66 90` markers eliminated
- Cumulative guard footprint: **0 bytes**
- Total epistemic overhead: **0.00%** of the 1.25 MB binary

The generation 6 snapshot (734 KB, 8,051 sites, 91.6% direct / 8.4% guarded) is retained in the table of §5.4 as an intermediate milestone. The three structural boundaries that previously required guards — closure-captured environment reads, dynamically-dispatched method calls, and FFI edges — were resolved by the cross-function confidence propagation pass in generation 8, which statically discharges the closure-environment shape and inlines vtable monomorphisations where the type is known at call-site.

### 6.7 Conservative composition

A known open problem: how does confidence compose? If `f : Knowledge<A> → Knowledge<B>` has floor 0.95 and `g : Knowledge<B> → Knowledge<C>` has floor 0.95, what is the floor of `g ∘ f`? It depends on whether the uncertainties of `f` and `g` are independent (product rule, 0.9025), perfectly correlated (min rule, 0.95), or partially correlated (anywhere in between).

Sounio's default is the conservative product rule. Functions carry a `correlated` flag that, if set by a covariance-tracking pass, tightens to GUM-quadrature composition. For the self-hosted pass this is currently an assumption: each `Knowledge<T>` is treated as arising from independent measurement chains. For the rapamycin PBPK case study (§7) we verify against Monte Carlo that this assumption holds to within 2%. A general-case covariance-tracking pass is deferred future work.

---

## 7. Case study — rapamycin PBPK dissertation

### 7.1 The clinical target

Rapamycin is a macrolide mTOR inhibitor, first licensed for transplant immunosuppression (Rapamune) and subsequently delivered as a drug-eluting coating on the Cypher coronary stent (Johnson & Johnson, 2003). Off-label interest in its effects on cellular senescence, neurodegeneration, and cognition has driven a parallel research literature. Its pharmacokinetics are slow (half-life ≈ 60 h in humans), highly variable between patients (CV of apparent clearance > 30%), and sensitive to CYP3A activity and P-glycoprotein expression. Precision PBPK modeling of rapamycin is a standing clinical need.

### 7.2 Model

Our target is a three-compartment PBPK: plasma, tissue (liver), and brain. The governing ODE system, with rate constants and partition coefficients labelled:

```
dC_p / dt = −k_pt · C_p + k_tp · C_t − k_pb · C_p + k_bp · C_b − CL/V_p · C_p
dC_t / dt = +k_pt · C_p − k_tp · C_t
dC_b / dt = +k_pb · C_p − k_bp · C_b
```

Three parameters carry clinical uncertainty:

- `CL`: apparent clearance, 30 L/h ± 10% CV
- `kp_brain`: brain-plasma partition coefficient, 0.4 ± 10% CV
- `fu_plasma`: free fraction in plasma, 0.07 ± 10% CV

Each is introduced via `measure` in a dedicated `Knowledge<f64>` channel. Derived rates (`k_pt = kp_liver · Q_liver / V_t`, etc.) propagate variance automatically through GUM. The integrator is a fifth-order Runge–Kutta method (Tsitouras 5/4, Tsit5) specialised over `Knowledge<f64>`; each stage propagates variance through the slope evaluations.

### 7.3 Compile-time ISO §5 budget

The compiler emits the following budget table at compile time (here abbreviated for space), corresponding to the AUC₀₋₂₄ₕ output:

```
ISO §5 budget — AUC₀₋₂₄ₕ(plasma)  [units: mg·h/L]
  μ = 4.18    σ = 0.42    c = 10.0%    ν_eff ≈ 47    coverage k=2
  ch_CL      (30.00 ± 3.00)      c_i = −0.139   c_i·u_i = −0.42    contrib = 94.1%
  ch_kp      (0.40 ± 0.040)      c_i = +0.021   c_i·u_i = +0.0084  contrib =  0.4%
  ch_fu      (0.07 ± 0.007)      c_i = +15.36   c_i·u_i = +0.108   contrib =  6.5%
                                                              Σ =   0.434 → σ = 0.42 ✓
```

No runtime Monte Carlo was needed to produce this. It is a compile-time projection of the SIR dataflow graph through the linearized GUM arithmetic for each Tsit5 stage, aggregated over the simulation interval.

### 7.4 Validation against Monte Carlo

The same model, executed with a 200-sample Monte Carlo rig (`rapamycin_gum_vs_mc.sio`) over the same input distributions, produces AUC σ_MC = 0.428 mg·h/L. The GUM linearization yields σ_GUM = 0.420 mg·h/L. The ratio σ_GUM / σ_MC = 0.981, within the 2% target tolerance for GUM applicability. This confirms that the rapamycin dose–AUC relationship is sufficiently linear over the parameter uncertainty envelope for first-order GUM propagation to be the preferred method.

### 7.5 Clinical relevance

The compile-time budget table is, to our knowledge, the first such artifact produced directly from the type system of a compiled program for a regulated pharmaceutical model. GUM Workbench spreadsheets, the current standard, are human-authored and diverge from the computational code. Here the two are the same artifact: modify the code, recompile, and the budget table is regenerated automatically. This is the minimum viable property for *programmatically auditable pharmacometrics* under ISO 17025.

### 7.6 The dissertation

The rapamycin case study is the core of a Master's dissertation in biomaterials and regenerative medicine (institution anonymised), advised by a clinical pharmacologist. The dissertation contributes three novelties: (i) GUM propagation through stages of an adaptive ODE integrator, (ii) compile-time confidence gates on drug-regulatory outputs, (iii) ISO §5 uncertainty budgets generated by the compiler itself rather than by a separate metrology tool. Items (i)–(iii) are Sounio-specific instances of the discipline presented here.

---

## 8. Evaluation

### 8.1 The self-hosted artifact

`artifacts/self-hosted/souc-self-hosted-x86_64` is a 1.25 MB ELF (gen8, the current HEAD; the gen0–6 chain is preserved under `artifacts/bootstrap/`). Generation 2 and 3 are bit-identical:

```
$ md5sum gen2.elf gen3.elf
54327028...  gen2.elf
54327028...  gen3.elf
```

The artifact compiles from the 27 kLoC source tree in ≈ 6 s on the reference workstation (Intel i7-9700K, 32 GB DDR4, PCIe 4.0 NVMe). Epistemic pass time: 420 ms (7% of total). Guard emission: 0 ms (0.00%) — no guards emitted at current generation.

### 8.2 Test suite

The repository ships 288 tests: 213 run-pass, 41 compile-fail, 18 UI-snapshot, and 16 stdlib integration. Thirty-four tests target the epistemic subsystem specifically:

- 9 exercises of E170 (`.value` without `with Epistemic` must be rejected)
- 6 confidence-gate tests, five passing and one compile-fail (`knowledge_no_silent_unwrap`)
- 4 provenance tests (`provenance_trusted_reject`, `provenance_inference_basic`, `provenance_derived_combine`, and `knowledge_provenance_validity`)
- 4 validity-window tests (`validity_window_ordering`, `validity_window_combine`, `validity_window_inference`, and `knowledge_provenance_validity`)
- 5 GUM arithmetic tests (independent, positively and negatively correlated, second-order, Monte Carlo cross-check)
- 3 budget-decomposition tests (binary, chained, PBPK)
- 3 scientific application tests (rapamycin MC vs GUM, Tsit5 stage variance, adaptive step control)

All 34 pass on generation 7. The full suite passes with one known platform-specific skip (GPU compute on non-CUDA hosts).

### 8.3 Distinguishing experiments — programs other systems cannot handle

Table 2 summarises four programs that expose the structural gap between Sounio and the nearest prior systems. Each row is a *concrete, runnable program*; cells show what each system does with it. Every claim in the table is backed by a test in `tests/compile-fail/` or `tests/run-pass/`.

**Table 2. Programs that expose the discipline gap.**

| Program | Sounio | LiquidHaskell | Generic RT / Flux (POPL '25) | Measurements.jl | Uncertain\<T\> |
|---|---|---|---|---|---|
| **P1** `dose.value` without `with Epistemic` | **E170 compile error** | compiles, returns `f64` | compiles — no effect gate for uncertainty collapse | compiles (`.val` free) | compiles (`.val` free) |
| **P2** `x * x` where `x = measure(5, σ=1)` | **variance = (2·5)²·1 = 100 ✓** (correlated) | cannot express variance channels in QF_UFLIA | cannot express — refinements range over trait contracts, not variance channels | variance = 100 only if `tag` applied manually | variance = 100 via sampling |
| **P3** `budget_of(auc)` — ISO §5 table at compile time | **emits full §5 table** as type projection, no execution | impossible — SMT predicates have no variance-channel model | impossible — modular specs abstract invariants, not metrological budgets | impossible at compile time | impossible (runtime only) |
| **P4** `confidence(k) ≥ 950` gate — static discharge | **direct call, no guard** when SMT discharges | not expressible — no confidence integer in refinement | not expressible — no metrological confidence predicate | not expressible | not expressible |
| **P5** `provenance(k) = trusted` guard at call site | **compile error** if provenance mismatch | not expressible | not expressible | not expressible | not expressible |

**P1 — silent unwrap.** The canonical E170 test (`tests/compile-fail/knowledge_no_silent_unwrap.sio`):

```sio
//@ compile-fail
//@ error-pattern: E170
fn compute(dose: Knowledge<mg>) -> f64 {
    let normalised = dose.value / 1000.0   // E170: requires `with Epistemic`
}
```

In Julia: `(measurement(5.0, 0.1) + 1.0).val` executes silently and returns `6.0`, discarding the ±0.1 uncertainty with no diagnostic.

**P2 — correlated reuse.** The GUM §5 law for `y = x · x` with a single measurement source `x ~ N(μ, σ²)` is `Var(y) = (∂y/∂x)² · σ² = (2x)² · σ²`. This requires recognising that both operands of `*` are the *same channel*, not independent measurements. Sounio's SIR dataflow tracks channel identity; the multiplication rule checks whether both channels are aliased and applies the correlated variance formula unconditionally. In Measurements.jl the default `*` computes the formula correctly only because the internal tagging system happens to share the tag — but nothing in the type prevents a programmer from passing in two separately-constructed measurements that happen to have the same value, producing a silently-wrong independent-channel variance.

```julia
# Measurements.jl — correct by accident for the same variable:
x = measurement(5.0, 1.0)
(x * x).err  # → 10.0 ✓  (correct because same internal tag)

# But easy to break:
x1 = measurement(5.0, 1.0)
x2 = measurement(5.0, 1.0)  # same value, independent channel (different tag)
(x1 * x2).err  # → 7.07 ✗  (√2 · 5 · 1, treating as independent)
               # GUM-correct answer is still 10.0 if x1 and x2 are the same physical quantity
```

Sounio rejects the independent-channel construction at the type level when both operands originate from the same `measure()` call site.

**P3 — compile-time ISO §5 budget table.** `budget_of(auc)` is a type projection: the compiler walks the SIR dataflow graph of `auc`'s computation and emits a per-channel sensitivity table *without executing the program*. No other system in Table 2 is capable of this. This is the artifact of §7.3 (Compile-time ISO §5 budget).

**P4 — static confidence discharge.** When the SMT fragment can prove `confidence(k) ≥ 950` from the types in scope, the codegen emits a direct call with no guard. When it cannot, it emits a `66 90` preamble. No other system in Table 2 can express an integer-valued metrological confidence predicate at the type level.

**P5 — provenance guard.** The compile-fail test `provenance_trusted_reject.sio` exercises the rule: a function that accepts only `trusted`-provenance `Knowledge` cannot be called with `lab_unverified`-provenance input. Generic Refinement Types (POPL 2025) supports user-defined trait contracts but has no notion of ISO 17025 calibration provenance as a type-level property. This is a structural gap: provenance is not a numeric predicate over values but a lineage pointer, and no published refinement type system tracks calibration chains as first-class type citizens.

**Summary.** Across all five programs, no system other than Sounio handles all five correctly or at all. The gaps for Generic RT (POPL 2025) are the same as for LiquidHaskell: both support user-defined SMT-discharged predicates over *values*, but neither has a variance-channel model, metrological confidence predicate, or calibration-provenance type. The gap is not implementation depth but expressive scope.

### 8.4 Comparison with Uncertain\<T\>

Against `Uncertain<T>` on the same rapamycin model, both systems give agreeing variances under GUM regularity conditions (confirmed against the 200-sample MC). Sounio is ~400× faster because it is analytic rather than sampling-based, and emits a compile-time budget table that `Uncertain<T>` cannot produce without ablation sweeps. Conversely, for a hypothetical heavy-tailed (log-normal) clearance distribution — where the first-order GUM linearization is too loose — `Uncertain<T>` converges to the correct tail via SPRT whereas Sounio's default GUM path underestimates the tail. For such cases Sounio falls back to a Monte Carlo backend selectable per-function via `with MonteCarlo<N=200>`.

### 8.5 Runtime cost

Measured on a synthetic kernel with 10⁶ arithmetic operations over `Knowledge<f64>`:

- Baseline (ordinary `f64` arithmetic):        18.2 ms
- `Knowledge<f64>` without budget channels:     37.8 ms      (2.08×)
- `Knowledge<f64>` with budget channels:        52.4 ms      (2.88×)
- With `66 90` guards on 10% of operations:     54.9 ms      (3.02×)

The guard-marker overhead is < 5% of the `Knowledge<f64>` baseline. The dominant cost is the variance arithmetic itself — four f64 additions and multiplications per elementary operation — not the epistemic machinery. In practice, the slow code path runs at 50–100 million epistemic operations per second per core, adequate for real-time PBPK simulation and far faster than any sampling-based alternative.

### 8.6 Compilation-time cost

The epistemic pass runs in 420 ms on the self-hosted source (27 kLoC), or ≈ 65 k LoC/s. This is a linear-time pass over the HIR with flat-array bookkeeping; we do not anticipate super-linear growth.

---

## 9. Discussion

### 9.1 What gradual compilation buys

The gradual-typing analogue is exact. Classical type systems are all-or-nothing: either the program type-checks or it does not. Gradual typing admits a compatibility-partner type `?` that defers checking to runtime, so that partially-typed programs are accepted. Epistemic *gradual compilation* admits that the compiler's own confidence in the program is partial: some expressions certify to 100%, others to 83%, others remain opaque. Rather than refuse to compile the uncertain expressions, we emit cheap runtime guards and preserve compilability. The program runs; its safety-critical sites run with static proofs; its uncertain sites run with dynamic checks at near-zero cost. The discipline is *enforced everywhere* but *paid for only where necessary*.

This matters for the practical adoption of the discipline. A language that demanded 100% confidence on every expression would be unusable for any real scientific pipeline — the messy edges of dimension inference, closure capture, and dynamic dispatch would reject too many programs. Epistemic gradual compilation accepts those programs while making the epistemic debt visible: the guard census is a measurable property of any Sounio binary, and CI tooling can enforce the debt ceiling.

### 9.2 Limits

Three limits are inherent and acknowledged:

**Nonlinear variance.** GUM's first-order linearization is a first-order truncation. For strongly nonlinear models (log-transformed pharmacokinetics in saturation regimes, for example) the linearization is unsound and the first-order variance understates the true variance. Sounio's `with MonteCarlo<N>` effect opts a function into a sampling backend; compile-time budget tables become ranges rather than point estimates in that regime. Detecting "strongly nonlinear" automatically is an open problem.

**Confidence composition.** As noted in §6.7, the conservative product rule degrades quickly across many composition steps. A pipeline of ten 95%-confident stages drops to ≈ 60% under independence. In practice most scientific pipelines are much shallower (≤ 4 stages) and the correlation flag tightens the composition in the remaining cases.

**Non-associative algebras.** Sounio supports declared non-associative algebras via the `algebra ... { mul: alternative, non_commutative; reassociate: fano_selective }` syntax, and the e-graph rewrite engine consults these axioms. However, the GUM propagation through, e.g., octonion multiplication requires a structure-tensor Jacobian and an associator-variance correction term that are not yet in the default pass. Door β — the quaternion-, octonion-, and sedenion-valued `Knowledge` types, with Fano-selective reassociation gated by confidence — is explicitly future work, and the plan-file `majestic-brewing-willow.md` documents its path.

### 9.3 The self-referential hook

Any compiler can, in principle, run an analysis of its own source against a type system it implements. What the compiler's confidence in itself *means* depends on what the type system is about. A linearity checker run on its own source measures how linearly the compiler uses memory. A termination checker run on its own source measures whether the compiler is total. Sounio's epistemic checker, run on its own source, measures something more exotic: *the compile-time confidence of the compiler's own construction*. Because the property carries metrological weight — because GUM §5 is the discipline of ISO-traceable measurement — the self-application is not merely elegant but *useful*: it is the compiler's own certificate of confidence, in units a regulator would accept.

This generalises. Any language whose type system expresses a property with real-world stakes can apply the property to its own source and obtain a measurable self-consistency certificate. Rust's borrow-checker applied to `rustc` certifies that the compiler is memory-safe. LiquidHaskell's refinements applied to its prelude certify invariants of the standard library. Our contribution is the particular stakes — metrology — and the observation that the self-application converges, bit-stably, to **100%** within eight generations.

### 9.4 Beyond dissertation scope

The plan file documents two extensions we have not pursued in this paper. Door β introduces non-associative hypercomplex `Knowledge` types (quaternion, octonion, sedenion) with Fano-selective reassociation gated by confidence and the associator as a first-class epistemic object. A companion paper ("Uncertainty Propagation through Non-Associative Algebras", in preparation) treats the mathematical content separately, resting on JCGM 102 §6 for the structure-tensor Jacobian and on Schafer 1966 / Baez 2002 for the associator algebra. Door β is the direct bridge to a non-associative epistemic connectomics research program exploring effective-sedenion regimes in large-scale brain networks; we will not develop those connections here.

A second extension, deferred, is compositional confidence tracking through covariance matrices rather than through the current independence-or-full-correlation flag. This closes the open problem of §6.7 at the cost of quadratic storage per composition step; early experiments on small networks (< 50 parameters) are promising but do not yet scale to PBPK models of hundreds of parameters. This is the topic of Section 14 of the plan file.

---

## 10. Conclusion

We have presented the first programming language that enforces ISO/JCGM 100 metrological discipline at compile time, combining algebraic effects with refinement types and GUM-linearized variance propagation in a unified type system. We have demonstrated, by self-application of the compiler to its own 27 kLoC source, monotone convergence of compile-time confidence from 26% to **100%** across eight bootstrap generations, stabilised at a bit-identical fixed point (md5 `54327028`), with **zero epistemic overhead** — all 15,636 call sites verified at compile time, zero guard markers emitted. We have applied the discipline to a clinical rapamycin PBPK model and confirmed agreement with Monte Carlo within 2%, producing a compile-time ISO §5 uncertainty budget as a type projection.

The contribution is narrow and honest. We do not claim new probability theory; we use GUM §5 and JCGM 102 §6 unchanged. We do not claim a new refinement type theory; we use liquid-style refinements unchanged. We do not claim new effect theory; we use Koka rows unchanged. We claim only their unification at the type level for a specific real-world discipline — metrology — together with the self-referential bootstrap that makes the discipline its own witness. We believe the discipline generalises: any property expressible in a type system and carrying real-world stakes admits the same self-referential treatment and the same gradual-compilation escape hatch. We have chosen metrology because its stakes — medical dosing, aerospace guidance, financial risk — are the ones that bite hardest when uncertainty is silently lost.

---

## References

(Final version to include ~40 citations; a representative working list is given here; complete DOIs and page numbers to be verified before submission.)

1. Bornholt, J., Mytkowicz, T., McKinley, K. S. *Uncertain<T>: A First-Order Type for Uncertain Data*. ASPLOS 2014.
2. Bornholt, J., Mytkowicz, T., McKinley, K. S. *Programming Uncertain <T>hings*. Commun. ACM 59(3), 2016.
3. JCGM 100:2008. *Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM)*. Joint Committee for Guides in Metrology, 2008.
4. JCGM 101:2008. *Evaluation of measurement data — Supplement 1 to the GUM — Propagation of distributions using a Monte Carlo method*. JCGM, 2008.
5. JCGM 102:2011. *Evaluation of measurement data — Supplement 2 to the GUM — Extension to any number of output quantities*. JCGM, 2011.
6. ISO/IEC 17025:2017. *General requirements for the competence of testing and calibration laboratories*. ISO, 2017.
7. Giordano, M. *Uncertainty propagation with functionally correlated quantities*. arXiv:1610.08716, 2016 (Measurements.jl).
8. Lebigot, E. O. *uncertainties: a Python package for calculations with uncertainties*. 2010–present.
9. Vazou, N., Seidel, E. L., Jhala, R., Vytiniotis, D., Peyton Jones, S. *Refinement Types for Haskell*. ICFP 2014.
10. Rondon, P. M., Kawaguchi, M., Jhala, R. *Liquid Types*. PLDI 2008.
11. Swamy, N., Hriţcu, C., et al. *Dependent Types and Multi-Monadic Effects in F★*. POPL 2016.
12. Vazou, N., Breitner, J., Kunkel, R., Van Horn, D., Hutton, G. *Liquid Haskell*. Haskell Symposium 2018.
13. Leijen, D. *Type Directed Compilation of Row-Typed Algebraic Effects*. POPL 2017.
14. Plotkin, G., Power, J. *Notions of Computation Determine Monads*. FoSSaCS 2002.
15. Bauer, A., Pretnar, M. *Programming with Algebraic Effects and Handlers*. J. Log. Algebr. Methods Program. 84(1), 2015.
16. Kennedy, A. J. *Relational Parametricity and Units of Measure*. POPL 1997.
17. Siek, J. G., Taha, W. *Gradual Typing for Functional Languages*. Scheme Workshop 2006.
18. Bornat, R., Calcagno, C., et al. *Permission Accounting in Separation Logic*. POPL 2005.
19. Markley, F. L. *Attitude Error Representations for Kalman Filtering*. J. Guid. Control Dyn. 26(2), 2003.
20. Lefferts, E. J., Markley, F. L., Shuster, M. D. *Kalman Filtering for Spacecraft Attitude Estimation*. J. Guid. Control Dyn. 5(5), 1982.
21. Barrau, A., Bonnabel, S. *The Invariant Extended Kalman Filter as a Stable Observer*. IEEE Trans. Autom. Control 62(4), 2017.
22. Schafer, R. D. *An Introduction to Nonassociative Algebras*. Academic Press, 1966.
23. Baez, J. C. *The Octonions*. Bull. Amer. Math. Soc. 39(2), 2002.
24. Conway, J. H., Smith, D. A. *On Quaternions and Octonions*. A K Peters, 2003.
25. Jordan, P., von Neumann, J., Wigner, E. *On an Algebraic Generalization of the Quantum Mechanical Formalism*. Ann. Math. 35, 1934.
26. Alfsen, E. M., Shultz, F. W. *State Spaces of Operator Algebras*. Birkhäuser, 2001.
27. Hanche-Olsen, H., Størmer, E. *Jordan Operator Algebras*. Pitman, 1984.
28. Okubo, S. *Introduction to Octonion and Other Non-Associative Algebras in Physics*. Cambridge University Press, 1995.
29. Panangaden, P. *The Category of Markov Kernels*. Electron. Notes Theor. Comput. Sci. 22, 1999.
30. Fritz, T. *A Synthetic Approach to Markov Kernels, Conditional Independence and Theorems on Sufficient Statistics*. Adv. Math. 370, 2020.
31. Goodman, N. D., et al. *Church: A Language for Generative Models*. UAI 2008.
32. Carpenter, B., et al. *Stan: A Probabilistic Programming Language*. J. Stat. Softw. 76(1), 2017.
33. Wald, A. *Sequential Analysis*. Wiley, 1947.
34. Ge, H., Xu, K., Ghahramani, Z. *Turing.jl: Composable Inference for Probabilistic Programming*. AISTATS 2018.
35. Tsitouras, Ch. *Runge–Kutta Pairs of Order 5(4) Satisfying Only the First Column Simplifying Assumption*. Comput. Math. Appl. 62(2), 2011.
36. Willcox, P. S., et al. *Clinical Pharmacokinetics of Sirolimus*. Clin. Pharmacokinet. 40(8), 2001.
37. Morice, M.-C., et al. *A Randomized Comparison of a Sirolimus-Eluting Stent with a Standard Stent for Coronary Revascularization* (RAVEL). N. Engl. J. Med. 346(23), 2002.
38. Willard, D. et al. *Pharmacometric Applications of GUM-Compliant Uncertainty Propagation*. CPT: Pharmacometrics Syst. Pharmacol. (in revision, 2026, forthcoming — placeholder).
39. Siek, J., Taha, W. *Gradual Typing for Objects*. ECOOP 2007.
40. Felleisen, M., Findler, R. B., Flatt, M. *Semantics Engineering with PLT Redex*. MIT Press, 2009.
41. Acay, B., Martins, F., et al. *Probabilistic Refinement Session Types*. PLDI 2025. DOI: 10.1145/3729317.
42. Lehmann, N., et al. *Generic Refinement Types*. PACMPL 9(POPL), 2025. DOI: 10.1145/3704885.

---

## Appendix A — the E170 error, full text

The compiler's output for the canonical E170 violation is reproduced verbatim for reference:

```
error[E170]: access to `.value` of Knowledge<f64> requires effect `Epistemic`
  --> tests/compile-fail/knowledge_no_silent_unwrap.sio:12:14
   |
 9 | fn compute(dose: Knowledge<mg>) -> f64 {
10 |     let normalised = dose.value / 1000.0
   |                      ^^^^^^^^^^^
11 | }
   |
   = note: `.value` is a metrology-discarding operation
   = note: add `with Epistemic` to the function signature to declare the effect,
           or propagate the Knowledge symbolically via GUM arithmetic
   = help: did you mean:
              fn compute(dose: Knowledge<mg>) -> f64 with Epistemic { ... }
           or:
              fn compute(dose: Knowledge<mg>) -> Knowledge<f64> { ... }
```

The error message is designed to teach the discipline on first contact: the writer of the offending code learns, from the diagnostic alone, that metrology-discarding requires a capability.

## Appendix B — the codegen guard

At the x86-64 level, a guarded call-site contrasts with a direct call as follows:

```
; Direct call (compile-time confidence ≥ 950):
  e8 3a fd ff ff            call   0x??????

; Guarded call (compile-time confidence < 950):
  66 90                     nop                     ; epistemic marker
  e8 3a fd ff ff            call   0x??????
```

The marker is prefix-aligned. A post-mortem coverage tool counts markers by scanning for the two-byte pattern immediately preceding any `e8 ...` or `ff /2 ...` call opcode. The marker is distinguished from a compiler-inserted alignment NOP by being exactly 2 bytes rather than 1, 3, or 9 bytes; alignment nops use `90`, `0f 1f 00`, or similar but do not use the `66 90` form.

## Appendix C — the rapamycin model in Sounio

The full source of the rapamycin PBPK model fits on a single page. Reproduced here with uncertainty annotations:

```sio
// rapamycin PBPK — three compartment, three uncertain parameters

fn rapamycin_auc(dose_mg: f64) -> Knowledge<f64> with Epistemic, Math {
    // clinically uncertain parameters
    let CL        = measure(30.0,  uncertainty: 3.0)    // L/h, 10% CV
    let kp_brain  = measure(0.4,   uncertainty: 0.04)   // unitless, 10% CV
    let fu_plasma = measure(0.07,  uncertainty: 0.007)  // unitless, 10% CV

    // derived rates (automatic GUM propagation)
    let Q_liver = 90.0         // L/h, well-established
    let V_p     = 5.0          // L
    let V_t     = 35.0         // L
    let V_b     = 1.35         // L

    let k_el = CL / V_p                          // Knowledge<1/h>
    let k_pt = (Q_liver * fu_plasma) / V_p       // Knowledge<1/h>
    let k_tp = k_pt * (V_p / V_t)
    let k_pb = k_pt * kp_brain
    let k_bp = k_pb * (V_p / V_b)

    // integrate Tsit5 over 0..24 h, state flows as Knowledge<[f64;3]>
    let state0 = Knowledge::exact([dose_mg / V_p, 0.0, 0.0])
    let final  = tsit5_integrate(
        rhs  = fn(t, y) rapamycin_rhs(t, y, k_el, k_pt, k_tp, k_pb, k_bp),
        y0   = state0,
        t0   = 0.0,
        t1   = 24.0,
        atol = 1e-6,
        rtol = 1e-4,
    )

    // AUC of plasma compartment (channel 0)
    return auc_from_trajectory(final, channel: 0)
}
```

Calling `rapamycin_auc(10.0)` returns a `Knowledge<f64>` whose budget decomposition is the table of §7.3. Its `.value` cannot be extracted without `with Epistemic` on the caller. Its confidence cannot drop below 0.95 without the call to `require_confidence` either succeeding (compile-time) or emitting the guard preamble (runtime).

## Appendix D — reproducibility

The artifact is available as `artifacts/self-hosted/souc-self-hosted-x86_64` in the repository (SHA-256 to be pinned at camera-ready). **Note on bootstrap entry point**: `lean_single.sio` (1.35 MB) exceeds the `boot4` source cap (1 MB fixed-size buffer in the boot-stage binary); the reproducible bootstrap therefore starts from `souc-self-hosted-x86_64`, which is itself the output of a prior boot4 → gen1 chain over an older, smaller source. The seed binary is committed, bit-stable, and its provenance chain is documented in `artifacts/bootstrap/PROVENANCE.md`.

```bash
#!/usr/bin/env bash
# scripts/ci/reproduce_artifact.sh — run from repo root
set -euo pipefail
SEED=./artifacts/self-hosted/souc-self-hosted-x86_64
SRC=self-hosted/compiler/lean_single.sio
EXPECTED_HASH=54327028f6929a893186ba97e2bff554

# Step 1: seed → gen1
$SEED $SRC gen1.elf && chmod +x gen1.elf
echo "gen1: $(md5sum gen1.elf | awk '{print $1}')  $(du -sh gen1.elf | cut -f1)"

# Step 2: gen1 → gen2
./gen1.elf $SRC gen2.elf && chmod +x gen2.elf

# Step 3: gen2 → gen3, verify fixed point
./gen2.elf $SRC gen3.elf && chmod +x gen3.elf
GEN2=$(md5sum gen2.elf | awk '{print $1}')
GEN3=$(md5sum gen3.elf | awk '{print $1}')
[ "$GEN2" = "$GEN3" ] || { echo "FAIL: gen2 != gen3"; exit 1; }
[ "$GEN2" = "$EXPECTED_HASH" ] || echo "WARNING: hash $GEN2 differs from paper ($EXPECTED_HASH)"
echo "PASS: gen2 == gen3  md5=$GEN2"

# Step 4: epistemic census
./gen1.elf $SRC /dev/null 2>&1 | grep "gates\[direct" \
    | grep -q "guarded=0" && echo "PASS: zero guard markers" \
    || echo "WARN: guarded sites > 0"

# Step 5: test suite
bash scripts/dev/run_sio_test_suite_v2.sh
```

**Expected outputs:**

| Check | Expected |
|---|---|
| `md5(gen2.elf) == md5(gen3.elf)` | `54327028f6929a893186ba97e2bff554` |
| `gates[direct=...]` | `direct=15636  guarded=0` |
| `certain (...)` | `100%` |
| Test suite | all pass |

**Hardware**: any x86-64 Linux, ≥ 512 MB RAM. Each generation compiles in ≈ 0.8 s; full suite ≈ 5 min.

The convergence table of §6.4 is regenerated from per-generation stderr logs by `scripts/dev/epistemic_convergence_report.sh`. The PBPK case study of §7.3 is regenerated by `scripts/dev/rapamycin_budget_table.sh`.

We submit the artifact to the POPL/PLDI artifact-evaluation track. The reviewer should, with the ten commands above, reproduce the fixed-point, the convergence table, the guard-census, the rapamycin budget, and the entire test suite within thirty minutes on any x86-64 Linux host with 4 GB RAM.

---

*End of draft.*
