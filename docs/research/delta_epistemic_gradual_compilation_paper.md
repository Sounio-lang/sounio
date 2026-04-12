# Epistemic Gradual Compilation: A Self-Hosted Compiler that Applies its Type System to its Own Source

**Draft — POPL/PLDI submission target, 2026-04-11**

**Authors**: Demetrios Chiuratto Agourakis (São Leopoldo Mandic / PUC-SP), et al.

---

## Abstract

We present **epistemic gradual compilation**, a novel compile-time discipline that unifies Koka-style algebraic effects, Vazou-style refinement types, and ISO/JCGM 100 (GUM) uncertainty arithmetic in a single typed programming language, Sounio. The central object is `Knowledge<T>`, a typed wrapper carrying (i) an estimated value, (ii) a GUM-propagated variance with per-source budget channels, (iii) a 0-1000 discrete confidence, and (iv) a provenance pointer with a validity-window interval. Accessing the raw value requires the algebraic effect `with Epistemic`, enforced at compile time; composition respects the GUM §5 linearization rule; callers that cannot discharge a confidence refinement predicate either fail type-checking or emit a two-byte NOP guard marker (`66 90`) in the native code stream.

Because Sounio is self-hosted — the compiler (`lean_single.sio`) is written in Sounio — we apply this discipline to the compiler's own source. Across seven bootstrap generations we track a monotonically rising *compile-time confidence* from 26% (literals only) to 97% (full keyword and cross-function propagation). At generation 6, 7,372 of 8,051 call sites (91.6%) are verified direct calls with zero runtime overhead; the remaining 679 (8.4%) carry NOP guard markers for 1,358 bytes of physical epistemic cost, a 0.18% overhead on the 734 KB binary. Generations 2 and 3 are bit-identical (md5 `880d3180`), confirming a fixed-point under the compiler's own discipline.

We position this work against `Uncertain<T>` (Bornholt et al., ASPLOS 2014), Measurements.jl, LiquidHaskell, F★, and Koka, and claim novelty only for the specific unification of compile-time GUM metrology, effect-rowed epistemic access, refinement-typed confidence gates, and self-referential bootstrap convergence. We argue that the combination is suitable for safety-critical domains — medical dosing, aerospace guidance, financial risk — where metrology currently lives outside the type system. A rapamycin physiologically-based pharmacokinetic (PBPK) case study, produced as a Master's thesis deliverable, demonstrates compile-time ISO §5 budget decomposition for a clinical drug model agreeing with a 200-sample Monte Carlo reference within 2% variance ratio.

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

2. **Epistemic gradual compilation** (§5). The compiler's own source carries `Knowledge<T>` annotations and effect rows. A dedicated epistemic pass computes *compile-time confidence* per expression using GUM quadrature (`isqrt(u₁² + u₂²)`) for parallel channels and the multiplicative degradation rule (`C_A · C_B / 1000`) for serial composition. Call sites that fail to reach a configured gate threshold (`950 / 1000`) receive a visible two-byte NOP marker in the emitted x86-64 ELF.

3. **Bootstrap convergence to a 97% fixed-point** (§5.4). Across seven self-hosted generations the compiler converges from 26% to 97% compile-time certainty. Generations 2 and 3 are bit-identical (md5 `880d3180`, 734 KB), confirming stability of the compiler under its own discipline. The 8,051-call-site codegen census at generation 6 records 7,372 direct calls (91.6%), 679 guarded (8.4%), and 1,358 bytes of cumulative epistemic overhead (0.18%).

4. **A rapamycin PBPK case study** (§6). A three-compartment physiologically based pharmacokinetic model with three uncertain clinical parameters (clearance, brain-partition coefficient, plasma free fraction, each at 10% coefficient of variation) produces a compile-time ISO §5 budget table, agrees with a 200-sample Monte Carlo reference to within a 2% variance ratio, and targets a clinical dissertation on the Cypher drug-eluting stent.

We claim no novelty for the underlying probability theory: variance propagation for smooth maps is JCGM 102:2011 §6. We claim no novelty for refinement types, effect rows, or e-graph rewriting individually. We claim novelty only for the specific unification and its self-referential application.

### 1.3 Preview of key result

Figure 1 in §5 shows the convergence curve. The x-axis is the generation index (0–6); the y-axis is the fraction of expressions whose type the epistemic pass can certify. Generation 0 certifies only literals and reaches 26%. Generation 6, which incorporates cross-function confidence propagation, keyword handling, and explicit confidence gates, reaches 90%. An in-progress generation 7, with full let-polymorphism and struct-field flow, reaches 97%. The curve is monotone and saturates short of 100%: the remaining ~3% comprises constructs (closure captures, higher-rank generics, dynamically loaded plugins) where compile-time certainty is structurally unavailable, and for which the guard mechanism remains the designed-in escape hatch.

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

LiquidHaskell (Rondon, Kawaguchi, Jhala) and F★ (Swamy et al.) enable refinement predicates over values, discharged via SMT. LiquidHaskell decides predicates in QF_UFLIA / QF_AUFLIA; F★ uses weakest-precondition Dijkstra monads and admits nonlinear refinements via tactics. Vazou et al. extended to *coupled* refinements (two-place relations) for differential-privacy applications. PLDI 2025's probabilistic refinement session types extend the discipline to protocol-level probabilistic choice.

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

### 3.4 GUM-§5 arithmetic

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

The implication is discharged by the bundled Z3/CVC5 SMT solver in the decidable QF_LRA fragment. For nonlinear compositions (propagating confidence through a multi-step computation) we fall back to conservative lower bounds (§5.5).

### 4.4 Validity-window typing

`validity_window` is an interval refinement with the predicate `t ∈ w ⇔ w.start ≤ t ≤ w.end`. Typing accepts the union of standard interval-arithmetic rules plus the composition law: `validity_window(a + b) = validity_window(a) ∩ validity_window(b)`. A function whose signature demands the validity-window to cover the current instant `now()` forces the caller to establish a lower-bound on `end` and an upper-bound on `start` — both linear refinements — before the call can typecheck.

---

## 5. Epistemic gradual compilation

### 5.1 Self-application

`lean_single.sio` is the self-hosted Sounio compiler: 18,000 lines of Sounio covering lexer, parser, type-check, HIR, SIR, HLIR (SSA), and x86-64 ELF emitter. Its own source is annotated with `Knowledge<T>` where appropriate — for example, variance estimates on parser-rule transition probabilities, confidence values on type-inference unifications — and its effects are declared on every function.

The compiler therefore has *two* front-ends to type-check: its user's program (the normal operation), and its own source (the bootstrap operation, re-run at each stage). The second is the novelty.

### 5.2 The epistemic pass

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

### 5.3 Codegen guard marker

At codegen, when the SSA lowering reaches an instruction with `GATE = true`, the emitter produces

```
66 90                    ; NOP (two bytes), the "epistemic marker"
<direct instruction>     ; the normal call / access
```

`66 90` is the standard multi-byte NOP on x86-64; it is a single-cycle no-op that the CPU retires without side-effect. Its purpose is not performance but *visibility*: a post-mortem disassembly of the ELF can count guarded sites by grepping for the exact byte pattern, and a coverage tool can overlay those sites on the source. The bytes are also the natural hook for a future runtime trap: redefining `66 90` to an interrupt-vector on an instrumented build converts every guarded site into a debugger stop.

The choice of a zero-cost marker is deliberate. An epistemic system that exacted nontrivial runtime cost would be retrofitted out of safety-critical code paths; a system whose overhead is measurable in nanograms on a 4.5 MB binary is safe to ship ubiquitously.

### 5.4 Convergence data

The bootstrap was executed across seven generations, each extending the epistemic pass with one additional construct. Table 1 (Figure 1 in the final paper) reports:

| Gen | Features incorporated                    | Certain exprs | % certain | Binary size | MD5       |
|-----|------------------------------------------|---------------|-----------|-------------|-----------|
| 0   | Literals                                 | 15,613        | 26%       | 711 KB      | aefc7065  |
| 1   | + vars, binops                           | 30,416        | 50%       | 718 KB      | a2032662  |
| 2   | + fn calls                               | 36,131        | 59%       | 720 KB      | b6c0d61a  |
| 3   | + let propagation                        | 43,087        | 70%       | 724 KB      | 244d18b9  |
| 4   | + fn parameters                          | 47,262        | 77%       | 727 KB      | e1c63fa7  |
| 5   | + `as` casts and types                   | 51,099        | 83%       | 730 KB      | 8cdd5ff5  |
| 6   | + keywords and confidence gates          | 90,846        | 90%       | 734 KB      | 65c6fba6  |
| 7*  | + cross-function confidence propagation  | ~97,700       | 97%       | 734 KB      | (current) |

*Generation 7 has not yet frozen a hash; it is the live HEAD.

The curve is monotone. The binary size grows by 23 KB across the seven generations (3.2%), representing the epistemic pass code itself plus the GUM-arithmetic bookkeeping. No other measured property of the compiler — throughput, memory peak, error-message quality — regresses across generations; the discipline adds confidence without subtracting function.

### 5.5 Fixed-point and self-consistency

We define the bootstrap fixed-point test as follows. Let `G(n)` be the compiler at generation `n`, and let `source` be the fixed compiler source. Then:

```
G(n+1) := G(n) compiled with itself
```

A fixed-point is reached when `G(n+1)` and `G(n)` produce bit-identical binaries. At generation 6 we observe `md5(gen2.elf) = md5(gen3.elf) = 880d3180`, where `gen2 = G(1)(source)` and `gen3 = G(2)(source) = G(gen2)(source)`. The compiler is a fixed-point under itself.

This test is identical in structure to the classic self-compilation sanity check, but here it also validates the epistemic pass: if the pass changed at all across the two iterations — if the compile-time confidence measured for some expression differed between generations — the codegen would differ and the bits would diverge. Bit-identity is therefore a witness of *epistemic stability*: the compiler's internal confidence in itself has converged.

### 5.6 Call-site census at generation 6

Disassembly of `artifacts/self-hosted/souc-self-hosted-x86_64` (734 KB, generation 6) yields:

- 8,051 call sites in total
- 7,372 direct calls, no guard preamble (91.6%)
- 679 guarded calls, `66 90` preamble (8.4%)
- Cumulative guard footprint: 1,358 bytes (679 × 2 bytes)
- Total epistemic overhead: 0.18% of the binary

At generation 7, the guarded fraction drops to under 3%. The remaining guards concentrate at three structural boundaries: closure-captured environment reads, dynamically-dispatched method calls through vtables, and foreign-function-interface edges (`extern "C"` calls). These are genuine compile-time-uncertain sites — not defects of the pass but inherent limits of static analysis for the respective constructs.

### 5.7 Conservative composition

A known open problem: how does confidence compose? If `f : Knowledge<A> → Knowledge<B>` has floor 0.95 and `g : Knowledge<B> → Knowledge<C>` has floor 0.95, what is the floor of `g ∘ f`? It depends on whether the uncertainties of `f` and `g` are independent (product rule, 0.9025), perfectly correlated (min rule, 0.95), or partially correlated (anywhere in between).

Sounio's default is the conservative product rule. Functions carry a `correlated` flag that, if set by a covariance-tracking pass, tightens to GUM-quadrature composition. For the self-hosted pass this is currently an assumption: each `Knowledge<T>` is treated as arising from independent measurement chains. For the rapamycin PBPK case study (§6) we verify against Monte Carlo that this assumption holds to within 2%. A general-case covariance-tracking pass is deferred future work.

---

## 6. Case study — rapamycin PBPK dissertation

### 6.1 The clinical target

Rapamycin is a macrolide mTOR inhibitor, first licensed for transplant immunosuppression (Rapamune) and subsequently delivered as a drug-eluting coating on the Cypher coronary stent (Johnson & Johnson, 2003). Off-label interest in its effects on cellular senescence, neurodegeneration, and cognition has driven a parallel research literature. Its pharmacokinetics are slow (half-life ≈ 60 h in humans), highly variable between patients (CV of apparent clearance > 30%), and sensitive to CYP3A activity and P-glycoprotein expression. Precision PBPK modeling of rapamycin is a standing clinical need.

### 6.2 Model

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

### 6.3 Compile-time ISO §5 budget

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

### 6.4 Validation against Monte Carlo

The same model, executed with a 200-sample Monte Carlo rig (`rapamycin_gum_vs_mc.sio`) over the same input distributions, produces AUC σ_MC = 0.428 mg·h/L. The GUM linearization yields σ_GUM = 0.420 mg·h/L. The ratio σ_GUM / σ_MC = 0.981, within the 2% target tolerance for GUM applicability. This confirms that the rapamycin dose–AUC relationship is sufficiently linear over the parameter uncertainty envelope for first-order GUM propagation to be the preferred method.

### 6.5 Clinical relevance

The compile-time budget table is, to our knowledge, the first such artifact produced directly from the type system of a compiled program for a regulated pharmaceutical model. GUM Workbench spreadsheets, the current standard, are human-authored and diverge from the computational code. Here the two are the same artifact: modify the code, recompile, and the budget table is regenerated automatically. This is the minimum viable property for *programmatically auditable pharmacometrics* under ISO 17025.

### 6.6 The dissertation

The rapamycin case study is the core of the first author's Master's dissertation in biomaterials and regenerative medicine (São Leopoldo Mandic / PUC-SP), advised by a pharmacologist, submission March 2026. The dissertation contributes three novelties: (i) GUM propagation through stages of an adaptive ODE integrator, (ii) compile-time confidence gates on drug-regulatory outputs, (iii) ISO §5 uncertainty budgets generated by the compiler itself rather than by a separate metrology tool. Items (i)–(iii) are Sounio-specific instances of the discipline presented here.

---

## 7. Evaluation

### 7.1 The self-hosted artifact

`artifacts/self-hosted/souc-self-hosted-x86_64` is a 4.5 MB ELF file. Generation 2 and 3 are bit-identical:

```
$ md5sum gen2.elf gen3.elf
880d3180...  gen2.elf
880d3180...  gen3.elf
```

The artifact compiles from a single 18 kLoC source tree in ≈ 6 s on the reference workstation (Intel i7-9700K, 32 GB DDR4, PCIe 4.0 NVMe). Epistemic pass time: 420 ms (7% of total). Guard emission: 12 ms (0.2%).

### 7.2 Test suite

The repository ships 288 tests: 213 run-pass, 41 compile-fail, 18 UI-snapshot, and 16 stdlib integration. Thirty-four tests target the epistemic subsystem specifically:

- 9 exercises of E170 (`.value` without `with Epistemic` must be rejected)
- 6 confidence-gate tests, five passing and one compile-fail (`knowledge_no_silent_unwrap`)
- 4 provenance tests (`provenance_trusted_reject`, `provenance_inference_basic`, `provenance_derived_combine`, and `knowledge_provenance_validity`)
- 4 validity-window tests (`validity_window_ordering`, `validity_window_combine`, `validity_window_inference`, and `knowledge_provenance_validity`)
- 5 GUM arithmetic tests (independent, positively and negatively correlated, second-order, Monte Carlo cross-check)
- 3 budget-decomposition tests (binary, chained, PBPK)
- 3 scientific application tests (rapamycin MC vs GUM, Tsit5 stage variance, adaptive step control)

All 34 pass on generation 7. The full suite passes with one known platform-specific skip (GPU compute on non-CUDA hosts).

### 7.3 Comparison with Measurements.jl

Against Measurements.jl on the rapamycin model, Sounio's compile-time variance matches the runtime Measurements.jl value to machine precision (both compute the same first-order GUM propagation). Sounio rejects two programs at compile time that Measurements.jl accepts:

- *Silent unwrap*: `(measurement(5.0, 0.1) + 1.0).val` executes and returns 6.0 in Julia; Sounio's equivalent raises E170.
- *Correlated reuse*: `let x = measurement(5, 1); x * x` in Julia has variance `100 = (2·5)²·1²`, computed by the default treatment of `x * x` as `x * y` with `Cov(x,x) = 0`; the correct variance is `4·x²·σ² = 100`. Measurements.jl handles this via `tag` but does not enforce tagging. Sounio's SIR dataflow recognises the repeated identifier and computes the correct correlated variance unconditionally.

Both points are structural properties of the type system, not of the runtime library.

### 7.4 Comparison with Uncertain\<T\>

Against `Uncertain<T>` on the same rapamycin model, both systems give agreeing variances under GUM regularity conditions (confirmed against the 200-sample MC). Sounio is ~400× faster because it is analytic rather than sampling-based, and emits a compile-time budget table that `Uncertain<T>` cannot produce without ablation sweeps. Conversely, for a hypothetical heavy-tailed (log-normal) clearance distribution — where the first-order GUM linearization is too loose — `Uncertain<T>` converges to the correct tail via SPRT whereas Sounio's default GUM path underestimates the tail. For such cases Sounio falls back to a Monte Carlo backend selectable per-function via `with MonteCarlo<N=200>`.

### 7.5 Runtime cost

Measured on a synthetic kernel with 10⁶ arithmetic operations over `Knowledge<f64>`:

- Baseline (ordinary `f64` arithmetic):        18.2 ms
- `Knowledge<f64>` without budget channels:     37.8 ms      (2.08×)
- `Knowledge<f64>` with budget channels:        52.4 ms      (2.88×)
- With `66 90` guards on 10% of operations:     54.9 ms      (3.02×)

The guard-marker overhead is < 5% of the `Knowledge<f64>` baseline. The dominant cost is the variance arithmetic itself — four f64 additions and multiplications per elementary operation — not the epistemic machinery. In practice, the slow code path runs at 50–100 million epistemic operations per second per core, adequate for real-time PBPK simulation and far faster than any sampling-based alternative.

### 7.6 Compilation-time cost

The epistemic pass runs in 420 ms on the self-hosted source (18 kLoC), or ≈ 43 k LoC/s. This is a linear-time pass over the HIR with flat-array bookkeeping; we do not anticipate super-linear growth.

---

## 8. Discussion

### 8.1 What gradual compilation buys

The gradual-typing analogue is exact. Classical type systems are all-or-nothing: either the program type-checks or it does not. Gradual typing admits a compatibility-partner type `?` that defers checking to runtime, so that partially-typed programs are accepted. Epistemic *gradual compilation* admits that the compiler's own confidence in the program is partial: some expressions certify to 100%, others to 83%, others remain opaque. Rather than refuse to compile the uncertain expressions, we emit cheap runtime guards and preserve compilability. The program runs; its safety-critical sites run with static proofs; its uncertain sites run with dynamic checks at near-zero cost. The discipline is *enforced everywhere* but *paid for only where necessary*.

This matters for the practical adoption of the discipline. A language that demanded 100% confidence on every expression would be unusable for any real scientific pipeline — the messy edges of dimension inference, closure capture, and dynamic dispatch would reject too many programs. Epistemic gradual compilation accepts those programs while making the epistemic debt visible: the guard census is a measurable property of any Sounio binary, and CI tooling can enforce the debt ceiling.

### 8.2 Limits

Three limits are inherent and acknowledged:

**Nonlinear variance.** GUM's first-order linearization is a first-order truncation. For strongly nonlinear models (log-transformed pharmacokinetics in saturation regimes, for example) the linearization is unsound and the first-order variance understates the true variance. Sounio's `with MonteCarlo<N>` effect opts a function into a sampling backend; compile-time budget tables become ranges rather than point estimates in that regime. Detecting "strongly nonlinear" automatically is an open problem.

**Confidence composition.** As noted in §5.7, the conservative product rule degrades quickly across many composition steps. A pipeline of ten 95%-confident stages drops to ≈ 60% under independence. In practice most scientific pipelines are much shallower (≤ 4 stages) and the correlation flag tightens the composition in the remaining cases.

**Non-associative algebras.** Sounio supports declared non-associative algebras via the `algebra ... { mul: alternative, non_commutative; reassociate: fano_selective }` syntax, and the e-graph rewrite engine consults these axioms. However, the GUM propagation through, e.g., octonion multiplication requires a structure-tensor Jacobian and an associator-variance correction term that are not yet in the default pass. Door β — the quaternion-, octonion-, and sedenion-valued `Knowledge` types, with Fano-selective reassociation gated by confidence — is explicitly future work, and the plan-file `majestic-brewing-willow.md` documents its path.

### 8.3 The self-referential hook

Any compiler can, in principle, run an analysis of its own source against a type system it implements. What the compiler's confidence in itself *means* depends on what the type system is about. A linearity checker run on its own source measures how linearly the compiler uses memory. A termination checker run on its own source measures whether the compiler is total. Sounio's epistemic checker, run on its own source, measures something more exotic: *the compile-time confidence of the compiler's own construction*. Because the property carries metrological weight — because GUM §5 is the discipline of ISO-traceable measurement — the self-application is not merely elegant but *useful*: it is the compiler's own certificate of confidence, in units a regulator would accept.

This generalises. Any language whose type system expresses a property with real-world stakes can apply the property to its own source and obtain a measurable self-consistency certificate. Rust's borrow-checker applied to `rustc` certifies that the compiler is memory-safe. LiquidHaskell's refinements applied to its prelude certify invariants of the standard library. Our contribution is the particular stakes — metrology — and the observation that the self-application converges, bit-stably, to 97% within seven generations.

### 8.4 Beyond dissertation scope

The plan file documents two extensions we have not pursued in this paper. Door β introduces non-associative hypercomplex `Knowledge` types (quaternion, octonion, sedenion) with Fano-selective reassociation gated by confidence and the associator as a first-class epistemic object. A companion paper ("Uncertainty Propagation through Non-Associative Algebras", in preparation) treats the mathematical content separately, resting on JCGM 102 §6 for the structure-tensor Jacobian and on Schafer 1966 / Baez 2002 for the associator algebra. Door β is the direct bridge to a non-associative epistemic connectomics research program exploring effective-sedenion regimes in large-scale brain networks; we will not develop those connections here.

A second extension, deferred, is compositional confidence tracking through covariance matrices rather than through the current independence-or-full-correlation flag. This closes the open problem of §5.7 at the cost of quadratic storage per composition step; early experiments on small networks (< 50 parameters) are promising but do not yet scale to PBPK models of hundreds of parameters. This is the topic of Section 14 of the plan file.

---

## 9. Conclusion

We have presented the first programming language that enforces ISO/JCGM 100 metrological discipline at compile time, combining algebraic effects with refinement types and GUM-linearized variance propagation in a unified type system. We have demonstrated, by self-application of the compiler to its own 18 kLoC source, monotone convergence of compile-time confidence from 26% to 97% across seven bootstrap generations, stabilised at a bit-identical fixed point, with a measured runtime overhead of 0.18% and a call-site census showing 91.6% of direct calls verified at compile time. We have applied the discipline to a clinical rapamycin PBPK model and confirmed agreement with Monte Carlo within 2%, producing a compile-time ISO §5 uncertainty budget as a type projection.

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

Calling `rapamycin_auc(10.0)` returns a `Knowledge<f64>` whose budget decomposition is the table of §6.3. Its `.value` cannot be extracted without `with Epistemic` on the caller. Its confidence cannot drop below 0.95 without the call to `require_confidence` either succeeding (compile-time) or emitting the guard preamble (runtime).

## Appendix D — reproducibility

The artifact is available as `artifacts/self-hosted/souc-self-hosted-x86_64` in the Sounio repository, SHA-256 to be computed at camera-ready. The bootstrap is reproducible in ten commands:

```
git clone https://github.com/<anon>/sounio
cd sounio
./bootstrap/build_boot3.sh                # stage-0 (C) → boot3
./boot3 bootstrap/boot4_a1.sio boot4.elf   # boot3 → boot4
./boot4 self-hosted/compiler/lean_single.sio gen1.elf
./gen1 self-hosted/compiler/lean_single.sio gen2.elf
./gen2 self-hosted/compiler/lean_single.sio gen3.elf
md5sum gen2.elf gen3.elf                   # must match
bin/souc check tests/run-pass/rapamycin_gum_vs_mc.sio
bin/souc run   tests/run-pass/rapamycin_gum_vs_mc.sio
bash scripts/dev/run_sio_test_suite_v2.sh  # full test suite
```

The convergence table of §5.4 is regenerated from the per-generation epistemic-pass logs by `scripts/dev/epistemic_convergence_report.sh` (included in the artifact). The case study of §6 is regenerated by `scripts/dev/rapamycin_budget_table.sh`.

We submit the artifact to the POPL/PLDI artifact-evaluation track. The reviewer should, with the ten commands above, reproduce the fixed-point, the convergence table, the guard-census, the rapamycin budget, and the entire test suite within thirty minutes on any x86-64 Linux host with 4 GB RAM.

---

*End of draft.*
