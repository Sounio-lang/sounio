# Epistemic Systematic Review & Meta-Analysis Suite (Ultraplan)

**Status:** Draft — for refinement
**Date:** 2026-04-23
**Target:** `stdlib/meta/` — new module
**Sequencing:** Phase 3 (kernel) → Phase 1 (dissertation SR) → Phase 2 (full SOTA++++)
**Motivating use case:** Rapamycin PBPK prior synthesis for 2026-09 dissertation; standalone stdlib paper claim.

## Motivation

Meta-analysis IS epistemic computing. Every existing tool (`metafor`, `PyMARE`, `RevMan`, `GRADEpro`, `netmeta`) bolts uncertainty onto a numerics layer that was not built for it: effect-size SEs, τ² for heterogeneity, GRADE certainty ratings, and PRISMA audit trails are all conventions riding on plain floats.

Sounio already provides, as language features:

- `Knowledge<T>` with GUM propagation (budget, confidence, provenance)
- Refinement types (compile-time invariants on data)
- Units (dimensional analysis — OR ≠ RR ≠ HR as types)
- Effects (`with Observe`, `with IO`, extensible) — stage-tracked pipelines
- Compile-time confidence gates (dissertation novel contribution #2)

This means SR/MA is not an application of Sounio — it is the natural expression of Sounio. The stdlib suite should out-express every existing tool on properties that matter to evidence synthesis:

1. **PRISMA 2020 compliance is a proof, not a form.** Stage transitions are effects; the flow diagram is derivable from the effect trace.
2. **Pooling is type-gated.** You cannot pool studies of incommensurable effect kind, mismatched units, or incomplete RoB assessment — the program will not compile.
3. **GRADE certainty is compile-time.** The five downgrade + three upgrade domains reduce to a refinement on the output `Knowledge<T>` confidence; "moderate certainty" is a type, not a spreadsheet cell.
4. **Uncertainty is load-bearing, not ornamental.** τ², SE, CI are products of GUM propagation on typed operations, not separately-maintained parallel scalars.
5. **Living SR is a re-compile, not a re-run.** Effects-tracked search + extraction stages detect when a new record invalidates the synthesis.
6. **NMA inconsistency as an algebraic defect.** Treatment networks carry a cohomological (H¹) inconsistency object on the additive scale; in multivariate or sequential NMA, composition may be non-commutative or non-associative, giving a richer detector than scalar z-tests. *Novelty and algebra choice gated on the validation sim in [nma_nonassociative_algebra_note.md](nma_nonassociative_algebra_note.md).*

## Non-goals

- Not a replacement for Covidence/Rayyan UI; stdlib emits artifacts, humans adjudicate.
- Not a replication of R's `metafor` API surface for its own sake — where `metafor` makes choices Sounio can improve on, we improve.
- Not a library for general Bayesian modeling — we use `stdlib/bayes/` where it exists.

## Phase 3 — Epistemic kernel (v0.1, ~400 LOC)

**Exit criterion:** A 5-study toy meta-analysis pools end-to-end with RoB gate, τ² estimate, prediction interval, and forest plot. Unit/kind mismatch fails to type-check.

### Module layout

```
stdlib/meta/
  lib.sio          -- re-exports
  study.sio        -- Study + refinement
  effect.sio       -- EffectSize<Kind>, distinct types per kind
  convert.sio      -- delta-method kind conversions
  pool.sio         -- IV fixed, DerSimonian-Laird random
  rob.sio          -- RoB 2 schema as refinement
  forest.sio       -- minimal SVG forest (optional — can defer to Phase 1)
```

### Core types (sketch)

```sio
// effect.sio
struct EffectSize<K: EffectKind> {
    estimate: Knowledge<f64>,        // log-scale for ratio kinds
    kind: K,                         // phantom — compile-time kind tag
    n_total: i64 | n_total > 0,      // refinement
}

// Concrete kinds (distinct types, not enum variants):
struct OR {}   // odds ratio (log-scale estimate)
struct RR {}   // risk ratio (log-scale)
struct HR {}   // hazard ratio (log-scale)
struct MD {}   // mean difference (natural scale, carries unit)
struct SMD {}  // standardized mean difference (Hedges' g by default)

// study.sio
struct Study<K: EffectKind> {
    id: StudyId,
    effect: EffectSize<K>,
    rob: RobAssessment,              // refinement: complete
    design: StudyDesign,
    year: i32 | year > 1900,
}

// refinement: poolable only if rob is complete AND kind is commensurable
type PoolableSet<K> = [Study<K>; _] where all(s.rob.complete()) && all(s.effect.kind == K)

// pool.sio
fn pool_fixed_iv<K>(studies: PoolableSet<K>) -> PooledEstimate<K> with Synthesize
fn pool_random_dl<K>(studies: PoolableSet<K>) -> PooledEstimate<K> with Synthesize

struct PooledEstimate<K> {
    estimate: Knowledge<f64>,        // GUM-budgeted
    tau_sq: Knowledge<f64>,
    i_sq: Knowledge<f64>,
    q: f64, q_df: i64, q_p: f64,
    prediction_interval: (f64, f64),
    n_studies: i64,
}
```

### Type-level properties to verify in Phase 3 tests

- Pooling `[Study<OR>, Study<RR>]` fails at type-check (kind mismatch).
- Pooling `[Study<MD in mg/L>, Study<MD in nmol/L>]` fails at unit-check.
- Pooling studies with incomplete RoB fails the refinement predicate.
- `EffectSize<OR>.to_rr(baseline_risk: f64) -> EffectSize<RR>` propagates variance via delta method; `Knowledge<f64>` budget grows appropriately.

## Phase 1 — Dissertation SR (~600 LOC on top of kernel)

**Exit criterion:** Rapamycin PBPK dissertation priors are produced by a reproducible Sounio systematic review; prior feed into `stdlib/darwin_pbpk/` uses the kernel's `PooledEstimate` directly — no manual transcription.

### Additional modules

```
stdlib/meta/
  pk_effects.sio    -- EffectSize<LogAucRatio|LogCmaxRatio|LogClRatio|LogVdRatio>
  species.sio       -- cross-species downweight schema
  prior_feed.sio    -- MA posterior -> PBPK prior distribution adapter
examples/conversational_ossm/
  rapamycin_sr.sio  -- the actual dissertation SR (new file)
```

### Dissertation contribution (novel)

*"In traditional PBPK, prior distributions are hand-picked from selected references. In this dissertation, priors are produced by a compile-time-audited systematic review whose provenance, risk-of-bias gate, heterogeneity estimate, and PRISMA flow are all properties of the program that produces them. A prior that loses its provenance will not type-check."*

Maps directly to dissertation novel contributions #1 (GUM-through-ODE) and #2 (compile-time confidence gates) — the SR is the upstream side of the same epistemic chain.

## Phase 2 — Full SOTA++++ (~5000 LOC)

Eight waves, each independently shippable.

### Wave 2.1 — Screening / PRISMA (~600 LOC)
- Effects `Screen`, `Extract`, `Adjudicate`; stage transitions effect-tracked.
- Dedup: DOI exact + title Jaro-Winkler + author-year.
- Dual-screener κ, conflict queue, adjudicator override.
- PRISMA 2020 flow numbers derived from effect trace (not hand-counted).

### Wave 2.2 — Extended RoB (~500 LOC)
- ROBINS-I (non-randomized), QUADAS-2 (diagnostic), SYRCLE (animal), ROBINS-E (exposure).
- Domain → overall judgment as refinement derivation.

### Wave 2.3 — Extended pooling + heterogeneity (~800 LOC)
- Fixed: Mantel-Haenszel, Peto (in addition to IV).
- Random: REML, PM, Sidik-Jonkman, Hartung-Knapp-Sidik-Jonkman adjustment.
- Bayesian NNHM with half-normal τ prior (via `stdlib/bayes/`).
- Q-profile τ² CI, H², prediction interval, subgroup + meta-regression.

### Wave 2.4 — Bias + sensitivity (~600 LOC)
- Funnel (contour-enhanced), Egger, Begg, Harbord, Peters.
- Trim-and-fill, PET-PEESE, Copas selection, 3PSM.
- Leave-one-out, cumulative, Cook's D, DFBETAS, Viechtbauer-Cheung outliers.

### Wave 2.5 — GRADE (~400 LOC)
- 5 downgrade + 3 upgrade domains as refinement decisions.
- Certainty → `Knowledge<T>` confidence level (reuses dissertation gate machinery).
- Summary of Findings table auto-emitted.

### Wave 2.6 — NMA + algebraic consistency detector (~1200 LOC) ★ paper claim, spun out
- Frequentist graph-theoretical (netmeta-style).
- Bayesian hierarchical (Lu-Ades).
- **Algebraic inconsistency detector:** primary framing is cohomological (H¹ of the treatment graph); multivariate / sequential extensions may invoke non-commutative or non-associative composition. Algebra choice gated on validation sim — see [nma_nonassociative_algebra_note.md](nma_nonassociative_algebra_note.md).
- SUCRA, P-scores, design-by-treatment inconsistency.
- *Wave 2.6 is spun out of the stdlib paper per Revision 1 (A2). Tracked separately; Sounio implementation follows the sim.*

### Wave 2.7 — Specialty MA (~800 LOC)
- Diagnostic: bivariate, HSROC.
- Dose-response: Greenland-Longnecker, restricted cubic splines (one/two-stage).
- IPD: one-stage + two-stage with treatment-covariate interactions.

### Wave 2.8 — Living SR + reporting (~700 LOC)
- Effects-tracked update detection; versioned syntheses.
- Native SVG emission: forest, funnel, network, PRISMA flow.
- PRISMA 2020 checklist generator with traceability to effect trace.

## Paper claims

1. **Stdlib paper** (Phase 2 overall): "Type-Gated Evidence Synthesis: A Language-Level Approach to Systematic Review and Meta-Analysis." Venue candidates: *Research Synthesis Methods*, *Journal of Clinical Epidemiology*, or *Journal of Statistical Software*.
2. **Algebraic NMA consistency paper** (Wave 2.6 standalone, working title): *"Cohomological (and possibly non-associative) Inconsistency in Network Meta-Analysis."* Venue candidates: *Statistics in Medicine*, *Biometrics*. Decoupled from the octonion/connectomics program per [nma_nonassociative_algebra_note.md](nma_nonassociative_algebra_note.md); algebra chosen by validation sim, not by loyalty to 𝕆.
3. **Dissertation contribution** (Phase 1): incorporated into dissertation novel contribution #2.

## Open questions for refinement

1. **`Knowledge<T>` vs separate τ² field.** Should τ² live inside `Knowledge<f64>` as an extra budget component, or stay as a sibling field on `PooledEstimate`? Implications for GUM chaining into downstream PBPK priors.
2. **Effect-kind extensibility.** Kernel uses phantom-typed struct tags (`struct OR {}`). Does the current Sounio type system support this cleanly, or should kinds be trait-based? (Depends on `lean_single.sio` trait support — memory says traits are NOT real yet.)
3. **SR data input format.** RIS + BibTeX + Cochrane RM5 import — does this belong in the kernel, Phase 1, or Wave 2.1?
4. **NMA paper timing.** Does the NMA associator paper need to wait for Wave 2.6 Sounio implementation, or can it land earlier as a theoretical note with a reference prototype?
5. **Advisor/co-authorship.** User's pharmacologist advisor is excited about dissertation — does the stdlib paper and/or NMA paper get offered to them, or stays solo per authorship-ethics memory?
6. **Scope of "SOTA++++".** Should we pre-commit to parity with `metafor` feature-by-feature, or stake a claim at "everything that matters for evidence synthesis, with structural guarantees" and accept that niche `metafor` features (e.g. multilevel MA for crossed random effects) land later or never?
7. **Butterfly thread.** Where does this sit relative to the octonion/connectomics/ORC program? NMA associator is the obvious link; is there a deeper one through the Garden?

## Proposed immediate next steps (after plan approval)

1. Resolve open questions 1–3 (type-system-dependent — blocks Phase 3).
2. Create `stdlib/meta/` with `lib.sio` skeleton + `CONVENTIONS.md` matching `stdlib/epistemic/` style.
3. Implement `effect.sio` + `study.sio` (kernel types, no pooling yet).
4. Write the failing-to-compile tests first (kind mismatch, unit mismatch, incomplete RoB) — these pin down the refinement contracts.
5. Implement `pool.sio` IV fixed + DerSimonian-Laird random; 5-study toy passes.
6. Commit `[epistemic] meta kernel v0.1 (phase 3)`.

## Revision 4 — audit correction + expanded probes (2026-04-24)

The R3 audit below was wrong about refinements being entirely disabled — my test syntax used the wrong constructor. Correcting and expanding:

### Probes that FIRE at compile-time

| Case | Example | Result |
|---|---|---|
| Inline refinement on fn param, **literal** arg | `fn f(x: { n: i32 \| n > 0 }); f(-5)` | `error: refinement type violation — value -5` ✓ |
| Unit mismatch on **binary arithmetic** | `mg_val + kg_val` | `error: unit mismatch` ✓ |
| Kind mismatch via **distinct structs** | `OrEffect` ≠ `RrEffect` as fn arg | `error[E001]: Type mismatch in call argument` ✓ |

### Probes that DO NOT FIRE (but should for the stdlib paper to work)

| Case | Example | Result |
|---|---|---|
| Named-type alias at fn param | `type Positive = {…}; fn f(x: Positive); f(-5)` | silent pass |
| Refined **variable** arg (non-literal) | `let v = -5; takes_positive(v)` | silent pass |
| Refined **typed variable** arg | `let v: i32 = -5; takes_positive(v)` | silent pass |
| Unit mismatch on **fn arg** (literal or var) | `fn take_mg(x: mg); take_mg(0.5_kg)` | silent pass |
| **Struct field** refinement | `struct Study { year: { y: i32 \| y > 1900 } }; Study { year: 1800 }` | silent pass |

### What this really means

The ONLY compile-time refinement gate that fires today is:
- **Literal argument** (not variable)
- Passed to a function with **inline** refinement on the parameter (not named alias, not struct field)

In real stdlib code, arguments are variables. `pool_or_iv(studies)` takes an array of studies whose fields are variables. The refinement gate, as currently implemented, fires on essentially zero production-code paths.

Compile-time gates that DO survive in real code today:
- **Kind mismatch via distinct struct types** (the R3-a pivot stands — this is real and load-bearing).
- **Binary-op unit mismatch** (useful for internal stdlib arithmetic but not for pooling API boundaries).

### Fork B scope — honestly revised

Fork B now requires four compiler extensions, not two:

1. **Variable-flow refinement propagation** — the refinement predicate must flow through `let v = -5; f(v)`. This is value-range analysis / abstract interpretation over the existing refinement predicate machinery. **~400-800 LOC, non-trivial.** This is the largest piece and the critical one.
2. **Named-type-alias propagation** — when a fn parameter's declared type is a named refinement alias, inline the predicate. Probably small — ~100-200 LOC lookup fix.
3. **Struct-field refinement enforcement** — fire the refinement check at struct-constructor field assignment sites. ~200-400 LOC, needs new check in the constructor-expression code path.
4. **Unit-mismatch on fn-call args** — copy the existing pattern from binary-op site (line 13660) to the function-call arg check. ~30-80 LOC.

**Total compiler work: ~750-1500 LOC across four enhancements.** This is a genuine compiler research increment, not a patch set. Realistic timeline: weeks to a few months of focused compiler work, NOT days.

### Updated fork calibration

- **Fork A (ship now, narrower claim):** kind-mismatch-via-distinct-structs as the sole compile-time gate; everything else runtime. Dissertation unblocked. Paper claim honest but weaker.
- **Fork B (compiler first, full claim):** the four-PR compiler work above. Dissertation PBPK kernel work probably pauses during compiler sprints. Paper claim strong. **Timeline: 2-3 months of compiler work before stdlib Phase 3 can start with the full gate story.**
- **Fork C (parallel tracks):** run compiler extensions and stdlib Phase 3 on distinct-struct gates concurrently. Stdlib retrofits to stronger gates as compiler PRs land. Highest throughput, some duplicated effort where retrofit cost is high.

User has locked Fork B ("high stakes, high reward") with the earlier audit data. The revised scope is 2-3x larger than the "50-300 LOC" estimate that was in front of the user when they chose. **This revision ought to be re-surfaced before compiler code starts.**

## Revision 3 — `lean_single.sio` type-system audit (2026-04-24) [SUPERSEDED IN PART BY R4]

Ran the five R2-d audit questions as minimal `.sio` probes against `./bin/souc` (souc 1.0.0-beta.5). Verdicts below. **Some results overturn the Phase 3 design.**

### Audit probe results

| # | Probe | Verdict |
|---|-------|---------|
| A1 | Refinement type `type Positive = { x: f64 \| x > 0.0 }` + constructor | **E200 at construction — refinement types disabled.** `.sio.disabled` files across stdlib confirm this. No compile-time enforcement. |
| A2 | Generic struct `Box<T>` | **Works.** Type parameter accepted, value recovered at runtime. Generics are usable for storage only. |
| A3 | Unit mismatch `let m: kg = 0.5; take_mg(m)` via `unit mg; unit kg;` | **NOT caught, exit 0.** Units parsed but do not discriminate at the type level. *Runtime-asserted only / not enforced.* |
| A4 | Phantom kind `EffectSize<OR>` vs `EffectSize<RR>` passed to `pool_or(s: EffectSize<OR>)` | **NOT discriminated, exit 0, runs.** Type parameter erased; `EffectSize<RR>` silently accepted where `EffectSize<OR>` was declared. |
| A5 | Distinct struct types `OrEffect` vs `RrEffect` passed to `pool_or(s: OrEffect)` | **DISCRIMINATED: E001 at compile-time.** `"Type mismatch in call argument — declared type does not match at line 11"`. Real compile-time gate. |

### What this means for Phase 3

**The phantom-kind design is dead.** Type parameters are erased; `EffectSize<OR>` is not a distinct type from `EffectSize<RR>`. Kind enforcement must come from **distinct struct types** (A5 works) rather than phantom-parameterized structs (A4 fails).

**Revised Phase 3 kind architecture:**

```sio
// effect.sio — distinct struct per kind, NOT EffectSize<K>
struct OrEffect  { estimate: f64, se: f64, n: i64 }   // log-OR
struct RrEffect  { estimate: f64, se: f64, n: i64 }   // log-RR
struct HrEffect  { estimate: f64, se: f64, n: i64 }   // log-HR
struct MdEffect  { estimate: f64, se: f64, n: i64 }   // mean diff (unit-tagged via companion field or wrapper)
struct SmdEffect { estimate: f64, se: f64, n: i64 }   // Hedges' g

// pool.sio — per-kind pooling functions
fn pool_or_iv(studies: &[OrStudy])  -> OrPooled with Synthesize { ... }
fn pool_rr_iv(studies: &[RrStudy])  -> RrPooled with Synthesize { ... }
fn pool_hr_iv(studies: &[HrStudy])  -> HrPooled with Synthesize { ... }
// ... five kinds × (IV fixed + DL random) = 10 pooling functions
```

LOC inflation: ~30-50% over the phantom design. Grok-code's estimate upward-correction is now confirmed by the audit, though for a different reason than grok-code gave.

### Status of each refinement the original draft asserted

| Original claim | Actual enforcement | Notes |
|----|----|----|
| "Pooling `[Study<OR>, Study<RR>]` fails at type-check" | **Compile-time** | Works when kinds are distinct structs (not phantom params). Pool functions accept only `[OrStudy; N]`; an `RrStudy` in the array triggers E001. |
| "Pooling mismatched units fails at type-check" | **Runtime-asserted** (today) | Units are nominal but not enforced. Workaround: wrap MD in kind-specific structs (`MdEffectMgPerL`, `MdEffectNmolPerL`) as distinct types — feasible but ugly. Compiler work needed for real unit-discrimination. |
| "Pooling with incomplete RoB fails" | **Runtime-asserted** | `{ x : T \| predicate }` refinement disabled. Must use `fn validate_poolable(studies: &[OrStudy]) -> Result<...>` at runtime. |
| "GRADE certainty as refinement on output `Knowledge<f64>`" | **Runtime-asserted** | Same reason — refinement predicate unavailable. |
| "Quantified refinements `all(s.rob.complete())` over arrays" | **Runtime-asserted** | A1 blocks all refinement syntax. |

### R3 amendments adopted

- **R3-a** — Phase 3 effect-kind design changes from `EffectSize<K>` phantom to distinct per-kind structs (`OrEffect`, `RrEffect`, `HrEffect`, `MdEffect`, `SmdEffect`). Kind mismatch is the **single compile-time gate that survives the audit** — it now carries the entire R2-b minimum-novelty-floor claim.
- **R3-b** — Unit enforcement drops to runtime for now. Workaround via distinct wrapper structs for MD unit variants, documented as a known limitation. File a compiler issue: "enable unit mismatch at type-check." This is genuinely important for the paper claim and is compiler work, not stdlib work.
- **R3-c** — Refinement-type claims throughout the proposal downgraded to runtime validators. RoB-complete gate becomes `Result<PoolableSet, RobIncomplete>` from a validator, not a type constructor. File a compiler issue: "re-enable refinement types for struct fields and newtypes."
- **R3-d** — Phase 3 exit criterion rewritten: *"Pool 5-study OR toy end-to-end; passing an `RrStudy` in an `[OrStudy; N]` array fails at type-check (compile-time); runtime RoB-complete validator rejects incomplete assessments; unit-variant-wrapped MD pooling rejects mismatched wrapper types at compile-time as a unit-discrimination surrogate."*
- **R3-e** — Phase 3 LOC revised: **400 → 550-650 LOC** to account for per-kind pooling functions and runtime validators replacing refinement types. Phase 1 likely **600 → 800 LOC** by the same logic. Total to dissertation: **~1400-1500 LOC** (was ~1000).
- **R3-f** — Two new compiler issues are now on the critical path for the stdlib paper's novelty floor:
  1. Enable unit mismatch at type-check (convert `unit X;` decls from nominal-only to type-discriminating).
  2. Re-enable refinement types for struct fields and newtypes (the `.sio.disabled` files show this was working at some point).

Both issues are **language/compiler work**, not stdlib work. They must be resolved before the stdlib paper can claim "unit-mismatched pooling fails to compile" and "incomplete-RoB pooling fails to compile." Without them, the paper's defensible compile-time gate is *kind mismatch only* (distinct structs), which is narrower than the original pitch but still real.

### Resolved R2-d audit questions

- **Q1** (`all(...)` over arrays) → **runtime-asserted.** Blocked on refinement-type re-enablement (compiler issue).
- **Q2** (`where` on heterogeneous array properties) → **runtime-asserted.** Same blocker.
- **Q3** (unit mismatch at compile-time) → **runtime-asserted.** Blocked on unit-discrimination compiler issue. *Workaround:* per-unit-variant wrapper structs.
- **Q4** (chained refinement derivations) → **runtime-asserted.** Same blocker as Q1.
- **Q5** (refinement × `Knowledge<T>` GUM interaction) → **runtime-asserted** for the refinement portion; `Knowledge<T>` GUM propagation itself is unaffected and works at runtime.

**Summary verdict:** the current compiler enforces distinct-struct-kind mismatch at compile-time; everything else in the original plan reduces to runtime validators. The R2-b novelty-floor target (unit mismatch at type-check) requires compiler work to land. The stdlib paper narrative is viable, but it needs two compiler PRs landed before it can claim its original strong gates.

## Revision 2 — post second-pass review + algebra audit (2026-04-23)

Second-pass review (grok-code; minimax/qwen/groq blocked by plan/credit/key issues) on the v2 plan plus a targeted algebra audit. Three changes:

### R2-a. Octonion branding dropped from NMA paper

The "why octonions" defense sketched in R1 (treatment comparisons form a loop / quasigroup-with-identity under reversal) does not survive the algebra audit in [nma_nonassociative_algebra_note.md](nma_nonassociative_algebra_note.md). Native NMA consistency is an H¹ cocycle-defect object on the treatment graph — abelian, associative. Non-associativity enters only via *Route A* (multivariate NMA with non-commutative composition, landing on quaternions) or *Route B* (sequential/crossover NMA with path memory, landing on a free non-associative magma). Neither route lands on 𝕆 specifically. Route C ("just embed in 𝕆") is an imposed structure, not one arising from the data.

**Action:** strip octonion-specific language from the stdlib plan. Working title for the Wave 2.6 spin-out becomes *"Cohomological (and possibly non-associative) Inconsistency in Network Meta-Analysis."* The 𝕆 research thread (connectomics, G₂, triality, 7-sphere) is preserved as its own program, decoupled from NMA.

### R2-b. Honest-framing guard-rail

Grok-code flagged that R1's honest framing may over-correct — "effect-traced, type-gated-where-possible" can read as "glorified wrapper" to a *RSM*/*JSS* reviewer. Mitigation: the stdlib paper must demonstrate **at least one compile-time guarantee that no existing tool provides** (target: unit-mismatched pooling fails to type-check, as distinct from a runtime assertion). That single irreducible compile-time claim is the paper's minimum novelty floor. Without it, the paper is technical-report-grade.

### R2-c. LOC estimate corrected upward

Grok-code estimates the ~3500 LOC to stdlib paper is 50-100% low — hidden costs: metafor-interop validation (~500 LOC across 3-5 reference datasets), phantom-struct per-kind instantiation (+100-200 LOC in Phase 3), Sounio effects+units boilerplate not accounted for. **Revised total to stdlib paper: 4500-6000 LOC.** Dissertation subset (Phase 3 + Phase 1) likely ~1200-1500 LOC, not 1000. Schedule realism still holds for the dissertation subset; Phase 2 slippage is expected.

### R2-d. `lean_single.sio` pre-Phase-3 audit checklist

Before a single line of Phase 3 code is written, these five type-system questions must be resolved by prototyping in a scratch `.sio` file:

1. Quantified refinements over arrays (`all(...)`) — compile-time-enforced or linear-only?
2. `where` clauses on heterogeneous array properties (uniformity of a kind tag across elements).
3. Unit mismatches at compile-time via dimensional types (the minimum-novelty-floor check from R2-b).
4. Chained refinement derivations (RoB domain-level → overall-level as a derivation, not a duplication).
5. Refinement × `Knowledge<T>` GUM propagation — can variance budgets update through delta-method conversions without runtime checks?

Each question gets a one-line verdict in this doc before Phase 3 starts: *compile-time / effect-traced / runtime-asserted*.

## Revision 1 — post multi-model review (2026-04-23)

Ran `scripts/mcp/llm-offload.sh` across grok + deepseek (gemini credit-blocked) on the original draft above. Reviews converged strongly on five points. This section records the amendments; the original draft is preserved so the delta is visible and the reasoning auditable.

### What both reviewers agreed on

1. **Compile-time guarantees are oversold.** `lean_single.sio` does not currently support the refinement predicates the draft relies on (e.g. `all(s.rob.complete())` over heterogeneous arrays, `PoolableSet<K>` with quantified `where` clauses). Many "will not compile" claims reduce to runtime assertions. Deepseek: *"if it's a runtime panic, this is not novel — it's a wrapper around existing methods with assertions."* Grok: *"forcing runtime checks or weaker linear refinements."*
2. **NMA associator needs validation before paper commitment.** Both reviewers found no prior art for octonion/associator methods in NMA but demanded a simulation showing the associator catches inconsistency that node-splitting or design-by-treatment interaction miss on a synthetic network with known ground truth. Otherwise the method collapses to a reparameterization.
3. **Scope is 3-5× too large for a solo medical-student dissertation timeline.** Grok independently recommended shrinking Phase 2 to ~1500 LOC; deepseek cuts Waves 2.4, 2.7, 2.8.
4. **Missing methods a *Research Synthesis Methods* / *JCE* reviewer would flag.** p-curve, p-uniform*, Hedges-Vevea / McShane-Böckenholt-Hansen selection models, robust variance estimation (RVE), multilevel / multivariate MA, network geometry metrics, and — most importantly — **simulation-based Type I/II error control against `metafor`** as a venue-entry requirement.
5. **Phantom structs are the right choice for effect kinds**, not traits. Zero-cost static dispatch, extensible by adding new kind structs, feasible today. Tradeoff: generic pooling functions may need per-kind instantiation until Sounio grows trait-like generics.

### Amendments adopted

- **A1 — Honest framing.** Replace every occurrence of "will not compile" claims with the enforcement mechanism that is actually available. Before Phase 3 begins, audit `lean_single.sio` for each refinement the plan asserts; reclassify into *(a) compile-time-enforced*, *(b) effect-trace-audited*, *(c) runtime-asserted*. Revise the Motivation section accordingly. The paper claim becomes **"effect-traced, type-gated-where-possible evidence synthesis"** — weaker than the draft, defensible.

- **A2 — NMA detector spin-out, decoupled from octonions.** Remove Wave 2.6 from the stdlib paper. Track as a separate work stream:
  1. **Algebra audit first** — the "why octonions" defense does not survive scrutiny (see [nma_nonassociative_algebra_note.md](nma_nonassociative_algebra_note.md)). NMA consistency is natively a group-cohomology object (H¹ defect); non-associativity only enters via Routes A/B (multivariate or sequential NMA) and even then lands on quaternions or free magmas, not 𝕆. Octonion branding dropped from the NMA paper; the connectomics 𝕆 program keeps its own thread independently.
  2. Validation prototype in Python/NumPy (~200 LOC, one weekend) — synthetic NMA with injected inconsistency, running both additive cocycle-defect and non-commutative variants; which detector flags inconsistency that node-splitting misses?
  3. If any variant wins → standalone paper (*Statistics in Medicine* or *Biometrics*), framing determined by which detector won. Sounio implementation follows.
  4. If none wins → drop the spin-out. No sunk-cost loyalty.

- **A3 — Phase 2 cut to 4 waves (~2500 LOC).** Retained: 2.1 PRISMA, 2.2 RoB, 2.3 pooling+heterogeneity, 2.5 GRADE. Dropped from dissertation-timeline scope: 2.4, 2.6, 2.7, 2.8. Moved to "Phase 2 post-dissertation roadmap" — not cut entirely, but not gated on 2026-09.

- **A4 — New Wave 2.3.5 — Validation.** Before any paper submission, simulations must match `metafor` fixed + DL random output on 3-5 reference datasets (e.g. BCG tuberculosis, Pignon head-and-neck, Colditz childhood vaccination) to within numerical tolerance, plus Type I/II error control sims. ~200 LOC. Not optional.

- **A5 — Method gaps added to post-dissertation roadmap.** p-curve, p-uniform*, Hedges-Vevea + McShane-Böckenholt-Hansen selection models, RVE, multilevel MA (crossed random effects), multivariate MA (correlated outcomes), network geometry metrics (density, diversity, thin/thick loops), influence diagnostics (net heat plot, leverage).

- **A6 — Phantom-struct commitment.** Phase 3 uses phantom-typed struct kinds (`struct OR {}`, etc.). Generic pooling functions may need per-kind instantiation until trait-like generics land. Accepted as a tradeoff, not a blocker.

### Revised sequencing and size

| Phase | Scope | LOC | Deadline anchor |
|-------|-------|-----|-----------------|
| Phase 3 kernel | effect, study, rob, pool (IV + DL), forest | ~400 | pre-Phase-1 |
| Phase 1 dissertation SR | PK effect kinds, species, prior_feed, rapamycin_sr | ~600 | 2026-09 dissertation |
| Phase 2 (cut) | PRISMA + RoB + pooling+hetero + GRADE + validation | ~2500 | stdlib paper submission |
| NMA associator (spin-out) | Python validation first, then Sounio | ~200 + TBD | standalone paper |
| Post-dissertation roadmap | 2.4 bias, 2.7 specialty, 2.8 living, + p-curve/RVE/multilevel | TBD | post-2026-09 |

Total to dissertation: **~1000 LOC** (Phase 3 + Phase 1). Total to stdlib paper: **~3500 LOC**. NMA paper: gated on validation sim.

### Open questions now resolved

- **Q2 (phantom vs traits)** → phantom structs, per A6.
- **Q4 (NMA paper timing)** → spin-out track, validation-gated, per A2.

### Open questions still pending user refinement

- **Q1** (τ² in `Knowledge<T>` vs sibling field)
- **Q3** (RIS/BibTeX/RM5 import placement)
- **Q5** (advisor / co-authorship ethics)
- **Q6** (pre-commit to metafor parity vs structural-guarantee claim)
- **Q7** (butterfly thread — where does SR/MA sit in the Garden)

## Risk register

- **Refinement-type expressiveness.** If current `lean_single.sio` cannot express the refinement predicates cleanly, some compile-time gates become runtime gates. Mitigation: catalog required refinements before Phase 3 commit, escalate to compiler work if needed.
- **Dissertation schedule competition.** Phase 3 is ~400 LOC over ~2-3 sessions; Phase 1 is ~600 LOC over ~3-4 sessions. Phase 2 is paper-scope (multi-month). Risk: Phase 2 eats dissertation runway. Mitigation: Phase 2 waves gated behind dissertation milestones.
- **NMA associator novelty.** Need a lit-review pass to confirm no one has published this. Mitigation: `/offload-review` across Grok + DeepSeek + Kimi on "non-associative algebra for network meta-analysis consistency" before committing to paper claim.
