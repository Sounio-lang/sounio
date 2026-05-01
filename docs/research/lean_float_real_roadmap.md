# Lean Float-Real Lift Roadmap

**Started**: 2026-05-01
**Owner**: Demetrios Chiuratto Agourakis
**Status**: Stage 0 ✅ + Stage 1 ✅ landed; Stages 2–3 deferred.

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
| 2     | `Real`   | ⏳ 2-3 weeks | TBD (`SounioFrechetReal.lean`)                  |
| 3     | `Float`  | ⏳ 4-6 weeks | TBD (`SounioFrechetFloat.lean`)                 |

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

## Stage 2 — `Real` bridge (DEFERRED 2-3 weeks)

The Stage 2 milestone is to lift the theorems from `Rat` to a
**Mathlib-free `Real` carrier**. Two approaches under consideration:

### Approach 2a: in-tree Real-from-Cauchy

Implement Real numbers as Cauchy sequences of Rat, with the
ordering and arithmetic operations defined and proven in core
Lean 4. This is a non-trivial development (~1000 lines) but
keeps the Mathlib-free policy.

**Pros**: full self-containment, no external dependencies.
**Cons**: 1000+ lines of foundational work for a single discharge.

### Approach 2b: density argument over Rat

Prove that for any Real-valued monotone f and a Rat-rectangle
arbitrarily close to a Real-rectangle, the Fréchet enclosure on
the Rat-rectangle is arbitrarily close to the Real-rectangle's.
Then by density of Rat in Real, the enclosure transfers.

**Pros**: avoids re-doing Real foundations; ~200 lines.
**Cons**: requires a `Cauchy.complete` lemma, also Mathlib-adjacent.

### Recommended path

Start with **Approach 2b** with a minimal `Real`-as-Cauchy-Rat
shim that exposes only the ordering structure
(no analysis, no calculus). This is enough for the structural
Fréchet/Walley/Klibanoff content. The `Real`-shim should live in
`formal/lean4/SounioRealOrder.lean` and import nothing from
Mathlib.

## Stage 3 — `Float` bridge (DEFERRED 4-6 weeks)

The Stage 3 milestone closes the Real → Float gap. Strategy:

### IEEE-754 round-to-nearest is monotone

For any IEEE-754 operation, the rounded result is monotone in
each input on the representable subset of ℝ. This means:

  - For monotone f on ℝ that maps a Float-representable rectangle
    to a Float-representable interval, the Float computation of f
    on Float arguments is also monotone with the same signs.
  - The Fréchet enclosure on the Float corner endpoints encloses
    the true ℝ Fréchet enclosure on the same corners.

This requires:

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

| Stage | Date       | PR / Commit                       | Notes                                          |
|-------|------------|------------------------------------|------------------------------------------------|
| 0     | 2026-04-30 | Initial (`SounioFrechet`)          | Nat-shadow Fréchet enclosure                   |
| 0     | 2026-04-30 | `SounioKnightian` / `Walley`       | Nat-shadow Walley elicitation                  |
| 0     | 2026-05-01 | `SounioKlibanoff`                  | Nat-shadow Klibanoff CE boundaries             |
| 1     | 2026-05-01 | `SounioFrechetRat`                 | Rat-stage lift of inc-dec / inc-inc / dec-dec  |
| 2     | TBD        | (`SounioRealOrder` + bridge)        | 2-3 weeks; Approach 2b recommended             |
| 3     | TBD        | (`SounioIeee754` + bridge)         | 4-6 weeks; bounded-soundness via ulp accounting|

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
