-- formal/lean4/SounioRealCauchy.lean
import SounioOrderedCarrier
import SounioRealOrder

/-!
# Sounio.RealCauchy — irrational extension of `SounioReal` (Stage 3a-Cauchy)

This file extends the Stage 3a Route A milestone (rational-only
`SounioReal` from `SounioRealOrder.lean`) with a Cauchy-sequence
construction that *does* cover irrationals.

## Honest scope statement

A *complete* in-tree Cauchy-quotient construction (with full
`OrderedCarrier` typeclass instance, including the
`mul_le_mul_of_nonneg_right` proof) is a 500-1000 LOC ε-N
analysis project. This file is a **structural milestone** that:

  - **Ships**:
      1. `IsCauchy` predicate over `Nat → Rat`
      2. `structure SounioRealCauchy` carrying a sequence + a
         Cauchy proof
      3. `LE` instance via **pointwise eventual ordering**:
            `f ≤ g iff ∃ N, ∀ n ≥ N, f n ≤ g n`
         (see §3.1 for why we use this instead of the
         ε-quantified version)
      4. Reflexivity and transitivity of the eventual LE
         (proved without ε/2 splitting)
      5. **Constant-sequence bridge**: `ofRat : Rat →
         SounioRealCauchy` and `ofSounioReal :
         SounioReal → SounioRealCauchy`
      6. `Mul` instance via pointwise product (definition only)
      7. `OrderedCarrierObligation_RealCauchy : Prop` that
         *states* the residual proof obligation for the full
         `OrderedCarrier` typeclass instance.

  - **Defers** (to a future `SounioRealCauchyComplete.lean`
    milestone, ~200-400 LOC):
      - The proof that pointwise product preserves `IsCauchy`
        (requires a boundedness lemma + ε/2-K argument).
      - The `mul_le_mul_of_nonneg_right` law over the eventual
        ordering with pointwise product (requires bounded ε
        analysis).
      - The full `OrderedCarrier SounioRealCauchy` instance.
      - The lift from pointwise eventual order to the canonical
        ε-quantified ordering on the Cauchy quotient.

The deferred work is an *honest* future commitment, not a
typeclass placeholder with `sorry`. The Sounio epistemic stack
remains bit-perfect for the rational subset (via `SounioReal`)
and gains a documented bridge to irrationals (via
`SounioRealCauchy`) without compromising the no-`sorry`,
no-`axiom`, no-Mathlib policy.

## §3.1 — Why pointwise eventual order

The standard real-line order on Cauchy sequences is:

    f ≤_ε g iff ∀ ε > 0, ∃ N, ∀ n ≥ N, f n ≤ g n + ε

Transitivity of `≤_ε` requires the ε/2 splitting argument, which
in turn requires `Rat.div_pos` (proving `0 < ε / 2` from
`0 < ε`). Lean 4 core does NOT ship `Rat.div_pos` directly —
it would need to be derived from `Rat.mul_pos` + `Rat.inv_pos`,
which is non-trivial without `ring`/`field_simp` (Mathlib).

We sidestep this by using the **pointwise eventual order**:

    f ≤_p g iff ∃ N, ∀ n ≥ N, f n ≤ g n

This is **strictly stronger** than `≤_ε`: `f ≤_p g → f ≤_ε g`.

For the structural `OrderedCarrier` content (Fréchet enclosure,
Walley gap monotonicity, Klibanoff boundary), `≤_p` is sufficient
because:
  - All structural theorems are stated as `Prop`s closed under
    forward implication along `→`.
  - The downstream consumers (vancomycin Cmin etc.) work with
    rational-valued or `≤_p`-valued bounds.
  - The lift `≤_p → ≤_ε` is straightforward (give ε >= 0 and N
    from the witness).

The full `≤_ε` order is the canonical real-line order; lifting
this file's `≤_p` proofs to `≤_ε` is part of the deferred
`SounioRealCauchyComplete.lean` milestone. We document the
relationship as a `Prop`-level obligation (§7).

## Math-review record

Pre-impl thesis (`/tmp/cauchy_float_thesis.md`):
  - Track 1 (this file): 5/5 OK on key questions (eventual LE
    definition approved; ε/2 transitivity approved as standard;
    defer Mul-Cauchy preservation approved; constant-sequence
    bridge OK; Option 7c obligation Prop without instance OK).
  - Track 2 (companion `SounioFloatInstance.lean`): 1 BUG
    caught + 1 TIGHTENABLE; addressed in that file.

Post-impl: see commit message.

## Status

The structure + LE instance + reflexivity + transitivity +
constant-sequence bridge + obligation proposition build cleanly
under `lake build SounioRealCauchy`. No `axiom`, no `sorry`, no
Mathlib import.
-/

namespace Sounio.RealCauchy

-- ================================================================
-- §1. Cauchy sequences over Rat.
-- ================================================================

/-- A sequence `f : Nat → Rat` is **Cauchy** if it eventually
    becomes ε-close to itself for any ε > 0.

    Formally: ∀ ε > 0, ∃ N, ∀ m n ≥ N,
              -ε ≤ f m - f n ≤ ε

    We split into two halves to avoid `Rat.abs` (which exists in
    core Lean 4 but uses helpers we don't need here). -/
def IsCauchy (f : Nat → Rat) : Prop :=
  ∀ ε : Rat, 0 < ε →
    ∃ N : Nat, ∀ m n : Nat, N ≤ m → N ≤ n →
      f m - f n ≤ ε ∧ f n - f m ≤ ε

/-- The constant sequence is trivially Cauchy: `f m - f n = 0`
    for all m, n, and `0 ≤ ε` for any positive ε. -/
theorem const_isCauchy (q : Rat) : IsCauchy (fun _ => q) := by
  intro ε hε
  refine ⟨0, ?_⟩
  intro m n _ _
  have hsub : (fun (_ : Nat) => q) m - (fun (_ : Nat) => q) n = 0 := by
    show q - q = 0
    exact Rat.sub_self
  rw [hsub]
  exact ⟨Rat.le_of_lt hε, Rat.le_of_lt hε⟩

-- ================================================================
-- §2. The SounioRealCauchy structure.
-- ================================================================

/-- A `SounioRealCauchy` is a Cauchy sequence of rationals,
    representing a real number. Distinct sequences may
    represent the same real (the equivalence
    `(f ~ g) iff (f − g) → 0` is **not** quotiented out at this
    stage; we work with sequence representatives directly).

    For the pointwise eventual order, the underlying sequence
    matters; for the canonical real-line order (≤_ε), only the
    equivalence class matters. See §3.1 in the file header for
    the relationship. -/
structure SounioRealCauchy where
  /-- The underlying rational sequence. -/
  seq : Nat → Rat
  /-- Witness that `seq` is Cauchy. -/
  cauchy : IsCauchy seq

namespace SounioRealCauchy

/-- The constant Cauchy sequence representing the rational `q`. -/
def ofRat (q : Rat) : SounioRealCauchy :=
  ⟨fun _ => q, const_isCauchy q⟩

/-- Bridge from the rational-only Stage 3a `SounioReal` to the
    Cauchy-stage `SounioRealCauchy`. The denotation is the
    canonical embedding ℚ ↪ ℝ. -/
def ofSounioReal (r : Sounio.SounioReal) : SounioRealCauchy :=
  ofRat r.approx

/-- The canonical zero. -/
def zero : SounioRealCauchy := ofRat 0

/-- The canonical one. -/
def one : SounioRealCauchy := ofRat 1

-- ================================================================
-- §3. Pointwise eventual ordering (LE).
-- ================================================================

/-- **Pointwise eventual ordering** for Cauchy sequences:

        f ≤_p g iff ∃ N, ∀ n ≥ N, f n ≤ g n

    Strictly stronger than the canonical ε-quantified order
    `≤_ε` (see §3.1 in the file header). Used here to avoid
    ε/2 splitting that requires `Rat.div_pos` (not in core
    Lean 4 directly). -/
def le (a b : SounioRealCauchy) : Prop :=
  ∃ N : Nat, ∀ n : Nat, N ≤ n → a.seq n ≤ b.seq n

instance : LE SounioRealCauchy where
  le := le

-- ================================================================
-- §4. Reflexivity and transitivity.
-- ================================================================

/-- Pointwise eventual LE is reflexive: take `N = 0` and use
    `Rat.le_refl` for every n. -/
theorem le_refl (a : SounioRealCauchy) : a ≤ a :=
  ⟨0, fun _ _ => Rat.le_refl⟩

/-- Pointwise eventual LE is transitive: combine the two
    witnesses `N₁`, `N₂` via `max` and chain `Rat.le_trans`.

    No ε/2 splitting needed because the order is pointwise. -/
theorem le_trans {a b c : SounioRealCauchy}
    (hab : a ≤ b) (hbc : b ≤ c) : a ≤ c := by
  obtain ⟨N1, h1⟩ := hab
  obtain ⟨N2, h2⟩ := hbc
  refine ⟨max N1 N2, ?_⟩
  intro n hn
  have hn1 : N1 ≤ n := Nat.le_trans (Nat.le_max_left N1 N2) hn
  have hn2 : N2 ≤ n := Nat.le_trans (Nat.le_max_right N1 N2) hn
  exact Rat.le_trans (h1 n hn1) (h2 n hn2)

-- ================================================================
-- §5. Multiplication (definition only).
-- ================================================================

/-- **Pointwise product** of two Cauchy sequences. The proof
    that this is itself Cauchy is **deferred** (it requires a
    boundedness lemma + ε/2-K argument, ~50 LOC).

    For now, we package this as `def mul_seq` returning the raw
    sequence; constructing the `SounioRealCauchy` from it
    requires the deferred preservation lemma. -/
def mul_seq (a b : SounioRealCauchy) : Nat → Rat :=
  fun n => a.seq n * b.seq n

/-- The deferred Cauchy-mul preservation lemma. Stated here as
    a `Prop` so downstream code can refer to it; the proof is
    a separate milestone (`SounioRealCauchyComplete.lean`). -/
def MulPreservesCauchy : Prop :=
  ∀ (a b : SounioRealCauchy), IsCauchy (mul_seq a b)

-- ================================================================
-- §6. Bridge soundness.
-- ================================================================

/-- The `ofRat` embedding preserves `≤`. This is the soundness
    bridge from rational-only `SounioReal` to the Cauchy stage:
    every theorem proven on rationals lifts to the rational-
    embedded subset of `SounioRealCauchy`.

    Proof: take `N = 0`; the constant sequences satisfy
    `(fun _ => a) n = a ≤ b = (fun _ => b) n` directly from
    `h : a ≤ b`. -/
theorem ofRat_le_ofRat (a b : Rat) (h : a ≤ b) :
    ofRat a ≤ ofRat b :=
  ⟨0, fun _ _ => h⟩

/-- The `ofSounioReal` embedding preserves `≤`. Composes with
    `SounioReal`'s LE instance (which delegates to Rat). -/
theorem ofSounioReal_le_ofSounioReal
    (a b : Sounio.SounioReal) (h : a ≤ b) :
    ofSounioReal a ≤ ofSounioReal b :=
  ofRat_le_ofRat a.approx b.approx h

-- ================================================================
-- §7. The deferred OrderedCarrier obligation.
-- ================================================================

/-- The full `Sounio.OrderedCarrier SounioRealCauchy` instance
    requires:
      1. `MulPreservesCauchy` (from §5)
      2. `mul_le_mul_of_nonneg_right` over the eventual
         ordering with pointwise product
      3. The lift from pointwise eventual order `≤_p` to the
         canonical ε-quantified order `≤_ε` (preserves all
         structural theorems but requires `Rat.div_pos` or
         equivalent).

    This proposition states the obligation; its proof is the
    `SounioRealCauchyComplete.lean` milestone. -/
def OrderedCarrierObligation_RealCauchy : Prop :=
  -- Mul preserves Cauchy
  MulPreservesCauchy
    -- mul_le_mul_of_nonneg_right (using the deferred Mul-Cauchy
    -- preservation as hypotheses for the typing)
    ∧ ∀ {a b : SounioRealCauchy} (c : SounioRealCauchy),
        a ≤ b → zero ≤ c →
        ∀ (h_mul_a_c : IsCauchy (mul_seq a c))
          (h_mul_b_c : IsCauchy (mul_seq b c)),
            (⟨mul_seq a c, h_mul_a_c⟩ : SounioRealCauchy)
              ≤ ⟨mul_seq b c, h_mul_b_c⟩

end SounioRealCauchy
end Sounio.RealCauchy
