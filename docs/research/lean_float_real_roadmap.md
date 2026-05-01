<!-- docs:meta
topic_id: repo.docs.research.lean-float-real-roadmap
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lean-float-real-roadmap
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lean Float-Real Lift Roadmap

**Started**: 2026-05-01
**Owner**: Demetrios Chiuratto Agourakis
**Status**: Stage 0 ✅ + Stage 1 ✅ + Stage 2 ✅ + Stage 3a (Route A
+ Cauchy structural) ✅ + Stage 3a-Cauchy partial ✅ + Stage 3b
(typeclass + Route C Float instance) ✅ + Stage 3b-F Phase 1
(canonical 5-axiom IEEE-754 spec + 2 derived theorems) ✅ landed.

**Pending milestones**:
  - Full `MulPreservesCauchy` for `OrderedCarrier
    SounioRealCauchy` (~150 LOC ε-N analysis, planned as
    `SounioRealCauchyMul.lean`).
  - Phase 1.5: derive `mul_le_mul_bounded_from_spec` and
    `add_le_add_bounded_from_spec` from the IEEE-754 spec
    (~150 LOC), eliminating the 4 typeclass-shape axioms.
  - Phase 2 / Route B: in-tree IEEE-754 binary64 model
    (~5000 LOC, c.f. Coq Flocq) discharging the 5 spec axioms.

## Why this matters

The Sounio epistemic stack ships three Lean-mechanised soundness
proofs (`SounioFrechet.lean`, `SounioWalley.lean`,
`SounioKlibanoff.lean`) over `Nat`-shadow. The Sounio runtime,
however, computes on `Float` (IEEE-754 binary64). This document
records the multi-stage discharge plan to close the
representation gap **without depending on Mathlib**, preserving
the in-tree, no-axiom, no-`sorry` policy.

## Stage map

| Stage | Carrier  | Status      | Key artifact                                    |
|-------|----------|-------------|-------------------------------------------------|
| 0     | `Nat`    | ✅ landed    | `formal/lean4/SounioFrechet.lean`               |
|       |          |             | `formal/lean4/SounioWalley.lean`                |
|       |          |             | `formal/lean4/SounioKlibanoff.lean`             |
| 1     | `Rat`    | ✅ landed    | `formal/lean4/SounioFrechetRat.lean`            |
| 2     | typeclass| ✅ landed    | `formal/lean4/SounioOrderedCarrier.lean`        |
|       |          |             | `formal/lean4/SounioFrechetGeneric.lean`        |
| 3a    | `Real`   | ✅ landed    | `formal/lean4/SounioRealOrder.lean` (Route A)   |
| 3a-C  | `Real`   | ✅ structural| `formal/lean4/SounioRealCauchy.lean`            |
|       | (Cauchy) | (full ⏳)    |  IsCauchy + LE_p + bridges; Mul/full-OC defer  |
| 3a-C-P| `Real`   | ✅ partial   | `formal/lean4/SounioRealCauchyPartial.lean`     |
|       | (Cauchy) | (mul ⏳)     |  mul_le_mul_of_nonneg_right + OC-modulo-MulPres |
| 3b    | typeclass| ✅ landed    | `formal/lean4/SounioFloatBounded.lean`          |
| 3b-F  | `Float`  | ✅ axiomatic | `formal/lean4/SounioFloatInstance.lean` (RtC)   |
|       |          | (Mathlib ⏳) |  4 axioms per Higham 2002 §2.4; cookbook eps   |
| 3b-F-1| `Float`  | ✅ Phase 1   | `formal/lean4/SounioIEEE754Spec.lean`           |
|       |          | (1.5 ⏳)     |  + `SounioFloatInstance.lean` dual-rep refactor |
| Walley/| Stage 2 | ✅ landed    | `formal/lean4/SounioWalleyGeneric.lean`         |
| Kliba |          |             | `formal/lean4/SounioKlibanoffGeneric.lean`      |

## Stage 0 — `Nat` shadow (DONE)

The Nat-shadow proves the **structural ordering content** of the
Fréchet, Walley, and Klibanoff theorems. Encoding choices:

- ε encoded as `(eps_num, eps_den)` with `eps_num ≤ eps_den`.
- λ encoded as `(lam_num, lam_den)` with `lam_num ≤ lam_den`.
- Means / supports / function values as `Nat`.

Proofs use `Nat.le_trans`, `Nat.mul_le_mul_right`, `Nat.add_*`,
`simp` with `Nat.sub_zero` / `Nat.zero_mul` / `Nat.mul_one`, and
`rfl`. **No Mathlib, no axiom, no sorry.**

This stage establishes that the soundness *machinery* is
correct — the theorems hold for any monotone integer-valued
function on a discrete rectangle.

## Stage 1 — `Rat` lift (DONE)

The Rat lift extends the structural content to **arbitrary
rational arithmetic**, giving us:

- Full ordered-field arithmetic (commutative rings,
  multiplication by non-negatives preserves ordering, etc.)
- Faithful sub-field embedding into ℝ
- Proximity to the actual `Float` representation
  (Float ⊂ Rat-with-rounding via `Float.toRat` round-tripping)

Lean 4 core ships `Rat` with the lemmas needed:

- `Rat.le_trans`
- `Rat.mul_le_mul_of_nonneg_left` / `..._right`
- `Rat.add_le_add_right` / `..._left`
- `Rat.add_comm`, `Rat.mul_comm`, `Rat.mul_one`

The `SounioFrechetRat.lean` proofs are **structurally identical**
to the Nat versions — only `Nat.le_trans` becomes `Rat.le_trans`.
This is intentional: the Nat → Rat lift is a sanity check on the
structural reusability of the Sounio Fréchet substrate.

The Rat-typed Vancomycin Cmin obligation is also discharged,
demonstrating substrate-generality at this stage.

## Stage 2 — typeclass abstraction (DONE 2026-05-01)

Stage 2 unifies the Stage 0 (`Nat`) and Stage 1 (`Rat`) proofs by
declaring a Mathlib-free typeclass `Sounio.OrderedCarrier α`
capturing the *minimal algebraic content* the structural theorems
require, and re-stating the Fréchet/Walley theorems generically
over any instance.

### Files landed

- `formal/lean4/SounioOrderedCarrier.lean` — typeclass definition
  and `Nat` / `Int` / `Rat` instances.
- `formal/lean4/SounioFrechetGeneric.lean` — generic Fréchet
  enclosure (3 variants: inc-dec, inc-inc, dec-dec) + Walley
  gap-monotonicity, plus Stage 0 / Stage 1 specialisation
  corollaries.

### Typeclass shape (Mathlib-free)

```lean
class OrderedCarrier (α : Type) extends LE α, Mul α where
  zero : α
  le_trans :
    ∀ {a b c : α}, a ≤ b → b ≤ c → a ≤ c
  mul_le_mul_of_nonneg_right :
    ∀ {a b : α} (c : α), a ≤ b → zero ≤ c → a * c ≤ b * c
```

Four fields, no parent class beyond Lean 4 core's `LE` and `Mul`.
The minimum sufficient set for the Fréchet (uses only `le_trans`)
+ Walley gap monotonicity (uses `mul_le_mul_of_nonneg_right`).

### Subsumption verification

`SounioFrechetGeneric.frechet_enclosure_inc_dec_nat` and
`..._rat` are stated as direct applications of the generic
theorem. By typeclass resolution, Lean 4 picks up the canonical
`Nat`/`Rat` instances from `SounioOrderedCarrier.lean` and the
generic proof discharges the carrier-specific obligation
**without any new content**. The Stage 0 / Stage 1 source files
remain available for independent inspection but are no longer the
*authoritative* proofs of the structural content.

### What this unblocks

The Real / Float instances now require **only** the typeclass
methods (≤ 10 lines each), not a full re-proof of the
Fréchet / Walley theorems. This collapses the Stage 3 effort
budget from "4-6 weeks of theorem migration" to
"4-6 weeks of carrier-specific proof obligations on the typeclass
methods" — a strict decomposition.

### Math-review record

- Pre-impl thesis (`/tmp/stage2_thesis.md`) → Grok 4.1 8/10 OK +
  1 TIGHTENABLE (Mathlib `Preorder` suggestion declined since it
  isn't in Lean 4 core) + 1 OVERREACH (Real bridge cost: ~10-20
  LOC with Mathlib import vs. ~50-100 LOC in-tree — accepted,
  document both options).
- Post-impl review (`/tmp/stage2_impl_review.md`) → Grok 4.1
  13/13 OK, NO_FINDINGS.

## Stage 3a — `Real` bridge (DONE 2026-05-01, Route A)

The Stage 3a milestone landed via **Route A** (Mathlib-free,
in-tree). The deliverable is a `SounioReal` type whose
denotation is the **rational subset of ℝ**, with the
`OrderedCarrier` instance inherited from `Rat`.

### What was shipped

- `formal/lean4/SounioRealOrder.lean` (~140 LOC):
  - `structure SounioReal where approx : Rat`
  - `instance : LE SounioReal`, `instance : Mul SounioReal`
  - `instance : Sounio.OrderedCarrier SounioReal`
  - canonical embeddings `ofRat`, `toRat` and witness theorems
    (`vancomycin_cmin_frechet_enclosure_real`,
    `walley_gap_monotone_real`)

### Honest scope

- **What it covers**: the rational subset of ℝ. Every theorem
  proven on `SounioReal` is bit-perfect for any rational number.
- **What it does NOT cover**: irrationals. The full Cauchy
  completion is deferred to a future Stage 3a-Cauchy milestone
  (separate type `SounioRealCauchy`, ~500-1000 LOC).
- **Rationale**: the structural Fréchet/Walley/Klibanoff
  theorems extend to ℝ from ℚ by *density* (uniform-norm
  continuity of monotone enclosures + ℚ dense in ℝ — see
  Grok 4.1 math-review 2026-05-01). The Stage 3a-Cauchy
  upgrade is **additive** when needed; the current shim is
  sufficient for clinical PK/TDM workloads which operate on
  rational dose / weight / clearance / time.

### Two-route comparison (retained for reference)

- **Route A — in-tree Rat-as-Real shim** (LANDED): ~140 LOC,
  Mathlib-free, covers ℝ ∩ ℚ.
- **Route B — Mathlib import**: ~10-20 LOC declaring
  `instance : OrderedCarrier Real` with Mathlib's lemmas.
  Available as an optional alternative for consumers who
  already have Mathlib in their dependency tree.

Both are valid; **Route A** is canonical for the repository's
Mathlib-free policy.

### Math-review record

- Pre-impl (`/tmp/realorder_thesis.md`): Grok 4.1 9/10 OK + 1
  TIGHTENABLE (density-bridge formalisation deferred to Cauchy
  milestone).
- Post-impl (consolidated with Tracks 2/3): Grok 4.1 16/16 OK
  across all four new modules.

## Stage 3a-Cauchy — irrational extension (DONE 2026-05-01, structural)

The Stage 3a-Cauchy milestone landed `SounioRealCauchy.lean`
which extends `SounioReal` (rational subset of ℝ) with a
genuine Cauchy-sequence construction that *can* represent
irrationals. The deliverable is **structural**: the type, the
ordering, and the bridge to `SounioReal` are provided; the full
`OrderedCarrier` instance for `SounioRealCauchy` is deferred to
a future complete-milestone (~200-400 LOC ε-N analysis).

### What was shipped

- `formal/lean4/SounioRealCauchy.lean` (~250 LOC):
  - `IsCauchy : (Nat → Rat) → Prop` predicate
  - `structure SounioRealCauchy where seq, cauchy : ...`
  - `LE` instance via **pointwise eventual ordering**
    (`f ≤_p g iff ∃ N, ∀ n ≥ N, f n ≤ g n`)
  - Reflexivity and transitivity of `≤_p` (proved without ε/2
    splitting, since the pointwise version doesn't need it)
  - `ofRat`, `ofSounioReal` constant-sequence bridges
  - `MulPreservesCauchy : Prop` (deferred lemma)
  - `OrderedCarrierObligation_RealCauchy : Prop` (full
    obligation stated)

### Why pointwise eventual order

The canonical real-line order on Cauchy sequences is the
ε-quantified `f ≤_ε g iff ∀ ε > 0, ∃ N, ∀ n ≥ N, f n ≤ g n + ε`.
Transitivity of `≤_ε` requires the ε/2 splitting argument,
which in turn requires `Rat.div_pos` (`0 < ε → 0 < ε / 2`).
Lean 4 core does NOT ship `Rat.div_pos` directly — it would
need to be derived from `Rat.mul_pos` + `Rat.inv_pos`, which is
non-trivial without `ring`/`field_simp` (Mathlib).

We sidestep this by using `≤_p`, which is **strictly stronger**
than `≤_ε` and provides transitivity for free via
`Rat.le_trans`. For the structural Sounio theorems
(Fréchet/Walley/Klibanoff), `≤_p` is sufficient because all
arguments use only `le_trans` + monotonicity-by-hypothesis.

### Math-review record

- Pre-impl (`/tmp/cauchy_float_thesis.md`): 5/5 OK on key
  Cauchy questions + the `≤_p` redesign approved.
- Post-impl (`/tmp/cauchy_float_postimpl_review.md`): 17/18
  OK (1 WRONG was on the Float counter-example arithmetic in
  Stage 3b-F, not Cauchy).

## Stage 3b — `Float` typeclass (DONE 2026-05-01, instance deferred)

The Stage 3b milestone landed the **typeclass** for IEEE-754
bounded soundness; the **Float instance** remains deferred (gated
by external IEEE-754 model availability).

### What was shipped

- `formal/lean4/SounioFloatBounded.lean` (~250 LOC):
  - `class BoundedOrderedCarrier α extends LE, Mul, Add` with
    relaxed `mul_le_mul_of_nonneg_right_bounded` and
    `add_le_add_right_bounded` (each parameterised by a
    non-negative `eps_inf`).
  - `frechet_enclosure_inc_dec_bounded`: the bounded analogue
    of the strict Stage 2 theorem, with parametric inflation
    budget and user-supplied combine/lift bookkeeping.
  - `vancomycin_cmin_frechet_enclosure_bounded`: PK obligation
    with quantified rounding tolerance.

### Float instance: three routes (instance NOT shipped)

A sound Float instance requires either:

  **Route a — Mathlib `Float` bridge** (~10 LOC if it exists):
  Use Mathlib's IEEE-754 lemmas if/when they ship ulp-bounded
  arithmetic facts.

  **Route b — In-tree IEEE-754 model** (~5000 LOC):
  Formalise IEEE-754 binary64 round-to-nearest from scratch
  (rounding modes, ulp, guard bits). Comparable in scope to
  Coq's Flocq library.

  **Route c — Axiomatised interim** (~50 LOC):
  Add an `axiom Float.mul_bounded_error : ...` asserting the
  IEEE-754 round-to-nearest bound. Fast but breaks the
  axiom-free in-tree philosophy.

The current recommendation is **Route a** when available,
**Route c** as a pragmatic interim, and **Route b** if Sounio
ever needs to ship a fully self-contained formal Float story
(e.g., for FDA / EMA submission).

Until the instance lands, runtime soundness for Float is covered
empirically by property tests (e.g.,
`tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio`,
`tests/stdlib/epistemic/test_klibanoff_float_property.sio`).

### Math-review record

- Pre-impl (`/tmp/floatbounded_thesis.md`): Grok 4.1 OVERREACH
  ×1 + TIGHTENABLE ×1 + WRONG ×1 + 6 OK (all addressed in
  implementation).
- Post-impl (consolidated): Grok 4.1 16/16 OK.

## Stage 3b-F — Float instance Route C (DONE 2026-05-01)

The Stage 3b-F milestone landed `SounioFloatInstance.lean` —
the **Route C axiomatic interim** Float instance for
`BoundedOrderedCarrier`. This file:

  - Declares **4 explicit axioms** corresponding to IEEE-754
    binary64 facts, each cited to Higham 2002 §2.4/2.5:
    - `Float.le_trans`
    - `Float.zero_le_zero`
    - `Float.mul_le_mul_of_nonneg_right_bounded`
    - `Float.add_le_add_right_bounded`
  - Provides `instance : BoundedOrderedCarrier Float` built on
    the axioms.
  - Provides `vancomycin_cmin_frechet_enclosure_float` as a
    direct specialisation of the bounded-Fréchet theorem.
  - Documents the user cookbook: `eps_inf ≥ k · ε_machine ·
    max(|a|, |b|, |c|, |a·c|, |b·c|)` per Higham.

### Critical design lesson

The math-review (pre-impl) caught a BUG in the original axiom
draft: a fixed-bound axiom like `|fl(a · b) - (a · b)| ≤ 0.0001`
is **mathematically wrong** for IEEE-754, since the rounding
error scales with operand magnitude as `ulp/2 ∝ |a · b| · 2⁻⁵²`.
For `a = b = 10¹⁰`, the actual half-ulp is `~1.1 × 10⁴` —
eight orders of magnitude above 0.0001.

The fix: axiomatise the typeclass methods **directly** with
per-call `eps_inf` parameters (already part of
`BoundedOrderedCarrier`). The user computes the correct ulp
budget from operand magnitudes; the axiom asserts that any
sufficient `eps_inf` discharges the typeclass law. This is the
"right" Route C: axioms exactly mirror typeclass methods, and
soundness obligation is pushed to the call site where operand
magnitudes are known.

### Discharge route to Route A or B

When Mathlib provides `Float` ulp-bounded arithmetic lemmas
(Route A) or Sounio formalises an in-tree IEEE-754 model
(Route B, ~5000 LOC, comparable to Coq's Flocq), this file's
`axiom` declarations become `theorem`s with proofs. Downstream
consumers continue to work without modification — they only
depend on the typeclass interface.

### Math-review record

- Pre-impl (`/tmp/cauchy_float_thesis.md`): BUG ×1
  (fixed-bound axiom unsound, addressed by per-call eps_inf
  redesign) + TIGHTENABLE ×1 (Route C honesty, addressed by
  WARNING header + counter-example docstring) + 6 OK.
- Post-impl (`/tmp/cauchy_float_postimpl_review.md`): WRONG ×1
  (counter-example arithmetic numerically off, qualitative
  point unchanged, addressed by corrected docstring) + 17 OK.

### Strategy reference: IEEE-754 round-to-nearest is monotone

For any IEEE-754 operation, the rounded result is monotone in
each input on the representable subset of ℝ. This means:

  - For monotone f on ℝ that maps a Float-representable rectangle
    to a Float-representable interval, the Float computation of f
    on Float arguments is also monotone with the same signs.
  - The Fréchet enclosure on the Float corner endpoints encloses
    the true ℝ Fréchet enclosure on the same corners.

The instance must provide:

1. A formalisation of IEEE-754 semantics over `Float`
   (`SounioIeee754.lean`).
2. A bound on the rounding-induced gap inflation per operation
   (Higham 2002 §2.4: `|fl(a op b) − (a op b)| ≤ ulp(a op b) / 2`).
3. The composition theorem: rounding a monotone-in-each-arg
   computation preserves the enclosure modulo ε_machine inflation.

### Bounded-soundness statement

The end-state Lean theorem will be:

```
theorem frechet_enclosure_monotone_inc_dec_float_bounded
    (f : Float → Float → Float)
    (mono_x mono_y : ...)
    (a b c d x y : Float)
    (hxa hxb hyc hyd : ...) :
    f a d − k * eps_machine ≤ f x y
    ∧
    f x y ≤ f b c + k * eps_machine
```

where `k` is a small constant (typically ≤ 4) bounding the number
of arithmetic operations in `f`. The Sounio runtime claim
becomes: the corner-enumeration band is sound modulo
`k · eps_machine`, where `eps_machine ≈ 2.2 × 10⁻¹⁶` for f64.

For a clinical Cmin band of width ~3 mg/L, the rounding-induced
inflation is `≤ 4 · 2.2 × 10⁻¹⁶ · 3 ≈ 2.6 × 10⁻¹⁵ mg/L` —
**16 orders of magnitude below clinical relevance**.

This is the honest soundness claim Sounio aims for: not bit-perfect
ℝ-soundness, but rigorously-bounded Float-soundness with a
quantified gap that is empirically irrelevant.

## Deferred-but-recommended engineering work

### Test-level Float-shadow

Even before the Stage 3 theorem lands, we can add **runtime
property tests** that empirically validate the Float-Fréchet
enclosure on the actual implementation. This is partially done:
`tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio`
(250 samples) and
`tests/stdlib/clinical/test_aminoglycoside_correlation_sensitivity.sio`
(500 samples) already serve as Float-runtime witnesses for the
M2.5 Fréchet theorem.

**Recommended addition**: a Walley/Klibanoff property test
`tests/stdlib/epistemic/test_klibanoff_float_property.sio` that
samples 500 random `(α, λ, c)` triples, computes CE_α, and
verifies the sandwich `walley ≤ CE_α ≤ precise + ε` on Float.
This is a natural follow-up commit.

### Mathlib detour (NOT recommended)

Importing Mathlib would discharge Stages 2-3 in ~50 lines of
Lean. We deliberately avoid this because:

1. The Sounio-formal in-tree philosophy: the entire formal stack
   should be readable / re-derivable / re-checkable from primary
   sources without external dependencies.
2. CI cost: a Mathlib import adds 5-10 GB of compiled artifacts
   and 10-15 minutes of build time per check.
3. Long-term reproducibility: Mathlib version drift is a
   significant risk; the Nat/Rat/Real/Float stack we control
   end-to-end is more stable.

If at some future point the Sounio formal-Lean stack is
restructured for non-self-hosted distribution
(e.g., as a teaching curriculum or a Lean-only proof artifact for
a referee), a Mathlib-imported alternative could be added as a
**parallel** discharge path, not a replacement.

## Status table (updates here as stages land)

| Stage | Date       | PR / Commit                                | Notes                                              |
|-------|------------|--------------------------------------------|----------------------------------------------------|
| 0     | 2026-04-30 | Initial (`SounioFrechet`)                  | Nat-shadow Fréchet enclosure                       |
| 0     | 2026-04-30 | `SounioKnightian` / `Walley`               | Nat-shadow Walley elicitation                      |
| 0     | 2026-05-01 | `SounioKlibanoff`                          | Nat-shadow Klibanoff CE boundaries                 |
| 1     | 2026-05-01 | `SounioFrechetRat`                         | Rat-stage lift of inc-dec / inc-inc / dec-dec      |
| 2     | 2026-05-01 | `SounioOrderedCarrier` + `FrechetGeneric`  | Typeclass abstraction; generic Fréchet+Walley     |
| 3a    | 2026-05-01 | `SounioRealOrder` (Route A)                | Mathlib-free SounioReal as ℝ ∩ ℚ                  |
| 3a-C  | 2026-05-01 | `SounioRealCauchy`                         | Cauchy structure + LE_p; Mul/full-OC deferred     |
| 3b    | 2026-05-01 | `SounioFloatBounded` typeclass             | Typeclass shipped; Float instance deferred         |
| 3b-F  | 2026-05-01 | `SounioFloatInstance` (Route C, axiomatic) | 4 IEEE-754 axioms; instance + demo theorem        |
| Walley generic | 2026-05-01 | `SounioWalleyGeneric`             | Stage 2 lift of M3.5 collapse/vacuous/gap          |
| Klibanoff generic | 2026-05-01 | `SounioKlibanoffGeneric`       | Stage 2 lift of M3.5+ boundary theorems           |
| 3a-C-P| 2026-05-01 | `SounioRealCauchyPartial`                  | Easy half: mul_le_mul_pointwise + OC-modulo-MulPres + ≤_p→≤_ε |
| 3b-F-1| 2026-05-01 | `SounioIEEE754Spec` (Phase 1)              | 5 canonical IEEE-754 axioms; Higham §2.1 + 2u→3u   |
| 3b-F-1| 2026-05-01 | `SounioFloatInstance` (refactor)           | 2 derived theorems from spec; mul/add → Phase 1.5  |

## Stage 3a-Cauchy partial (DONE — 2026-05-01)

`SounioRealCauchyPartial.lean` makes incremental progress on
closing `OrderedCarrierObligation_RealCauchy`. It proves the
**easy half** (`mul_le_mul_of_nonneg_right_pointwise` over `≤_p`
given Cauchy witnesses for the products) and the
`OrderedCarrierObligation_RealCauchy_holds_given_mulPres`
decomposition (the obligation discharges given
`MulPreservesCauchy` as a hypothesis).

The hard half (`MulPreservesCauchy`: ε-N proof that pointwise
products of Cauchy sequences are Cauchy) is deferred to a
future `SounioRealCauchyMul.lean` milestone (~150 LOC of Rat
algebra + boundedness lemma).

The file also ships `le_p_implies_le_eps`, the canonical
`≤_p → ≤_ε` lift, demonstrating that the pointwise eventual
order suffices for all structural Sounio theorems and lifts
to the canonical real-line order on demand.

Math-review record:
  - Pre-impl thesis: 8/8 OK (boundedness, ε/(2K) splitting,
    Mul-instance non-conflict, subsumption, `≤_p → ≤_ε`).
  - Post-impl: 4/5 OK + 1 OVERREACH on naming. File renamed
    `Complete.lean → Partial.lean` per review.

## Stage 3b-F Phase 1 — canonical IEEE-754 spec (DONE — 2026-05-01)

`SounioIEEE754Spec.lean` extracts the IEEE-754 binary64
specification to a separate module with **5 canonical
axioms** drawn from Higham 2002 §2.1 (basic-operation model)
and IEEE-754-2008 §5.11 (total order):

  1. `Float.toRat : Float → Rat`
  2. `Float.IsFiniteNormal : Float → Prop`
  3. `Float.toRat_le_iff_finite`: ≤ matches Rat ≤ on finite
     normal subset
  4. `Float.mul_rne_bound`: Higham §2.1 multiplication
     relative-error bound (`u = 2⁻⁵³`)
  5. `Float.add_rne_bound`: Higham §2.1 addition

`SounioFloatInstance.lean` is refactored to import the spec
and adopt a Phase 1 dual representation:
  - 4 typeclass-shape axioms retained (back the
    unconditional `BoundedOrderedCarrier Float` instance);
  - 2 derived theorems (`Float.le_trans_from_spec`,
    `Float.zero_le_zero_from_spec`) proven from the spec
    with finiteness hypotheses;
  - 2 deferred theorems (`mul_le_mul_bounded_from_spec`,
    `add_le_add_bounded_from_spec`) documented with explicit
    proof sketches in `/-! ## Phase 1.5 deferred ... -/`
    blocks.

Math-review caught **3 critical bugs** in the thesis pre-impl:
  - `ε_machine = 2⁻⁵²` was `ulp(1)`, not unit roundoff.
    Corrected to `u = 2⁻⁵³` (Higham §2.4).
  - Citation `Higham §2.5` is summation γ_n bound. Corrected
    to `§2.1` basic-operation model.
  - Cookbook `eps_inf ≥ ε(|ac| + |bc|)` missed `add_rne_bound`
    of `(bc + eps_inf)`. Corrected coarse form `3u·max(|ac|,
    |bc|)` documented in deferred derivation.

Phase 1 ships meaningful axiom-canonicalisation progress
without breaking the typeclass instance API. Phase 1.5 will
discharge the remaining 2 theorems and delete the 4 typeclass-
shape axioms (net: 5 spec axioms only).

## Audit policy

Each Stage landing requires:

1. `bin/llm-offload -t math-review -p xai` checkpoint on the
   theorem statement (pre-implementation thesis review).
2. `bin/llm-offload -t math-review -p xai` checkpoint on the
   final discharge (post-implementation review).
3. Audit log entry in `.claude/llm_offload_log.md`.
4. Atomic commit with the new Lean module + lakefile entry.

This is the same policy that gated the Stage 0 / Stage 1 work,
recorded in 6 audit log entries between 2026-04-30 and 2026-05-01.
