import SounioZeroDivisorBridge

/-!
# Sedenion Projective Measurement — combinatorial corollaries

This file re-states the algebraic content of
`SounioZeroDivisorBridge.lean` in the language of
projective measurement / decoherence channels on the
16-dimensional state space `ℝ¹⁶ ≅ 𝕊` (sedenions).

## What this file *does* claim

For the canonical sedenion zero-divisor
  `A := e₃ + e₁₀  (primA)`
left multiplication
  `L_A : 𝕊 → 𝕊,  L_A(h) := A * h`
admits, in the runtime-aligned Schafer Cayley-Dickson sign table, a
list of **four** linearly independent primitive imaginary sedenion
right-annihilators (proved exhaustively by `native_decide`):

  `annihilatorsOfA = [ e₄+e₁₃,  e₅−e₁₂,  e₆−e₁₅,  e₇+e₁₄ ]`.

Each is annihilated bit-for-bit, all four are pairwise distinct, all
four lie in xor-fiber 9, and the count is exact — no further primitive
right-annihilators exist (this is `annihilators_complete` in the bridge).

The uniform-4-regularity of the annihilation graph
(`every_primitive_has_4_annihilators`) lifts directly: every one of
the 84 valid primitives has exactly 4 primitive right-annihilators,
yielding 336 ordered ZD pairs and 168 projective classes.

In measurement-channel language this is:
  every primitive zero-divisor `Z` in 𝕊 defines a non-unitary
  operator `L_Z` whose primitive-annihilator subspace has
  combinatorial dimension 4, and there are exactly **168** such
  inequivalent erasure structures.

## What this file does **NOT** claim

The combinatorial count of 4 primitive right-annihilators does **not**
by itself imply `dim_ℝ ker(L_A) = 4` as a real-linear-algebra rank
statement; that requires Hilbert-space infrastructure (Mathlib
`LinearMap.ker`, `FiniteDimensional.finrank`) not loaded here. The
empirical 4-dimensional kernel observed at the runtime in
`examples/sedenion_projective_measurement.sio` (rank 12, kernel 4,
Pythagorean mass split 16 = 4 + 12) is consistent with — and indeed
*upper-bounded by* — the combinatorial count proved below.

We also do not claim any violation of Heisenberg uncertainty.
`σ_x σ_p ≥ ℏ/2` is a theorem about unitary evolution on a complex
Hilbert space; `L_A` is real-linear and non-unitary by construction.

No `sorry`. No Mathlib. All counts by `native_decide`.

References:
  - `SounioZeroDivisorBridge.lean` (parent file, all heavy lifting)
  - `SounioCayleyDickson.lean` (cdSigma sign table)
  - `stdlib/algebra/sedenion.sio` (runtime sed_mul, sign-aligned)
  - `examples/sedenion_projective_measurement.sio` (numerical witness)
  - Moreno (1998), Cawagas (2004), Schafer (1954)
-/

namespace Sounio.SedenionMeasurement

open Sounio.ZeroDivisorBridge

-- ================================================================
-- §1. Renaming: the canonical ZD as a measurement generator
-- ================================================================

/-- The canonical sedenion zero-divisor `A = e₃ + e₁₀`, re-exported under
    the name `measGen` to emphasise its role as a projective-measurement
    generator on `𝕊`. -/
def measGen : PrimSed := primA

/-- The list of primitive right-annihilators of `A`, re-exported as the
    `forgettable basis` of the measurement. -/
def forgettableBasis : List PrimSed := annihilatorsOfA

-- ================================================================
-- §2. Corollaries: the four governing facts of the measurement
-- ================================================================

/-- **Existence of erasure**: `A * v = 0` for each of the four
    primitive vectors in `forgettableBasis`. -/
theorem erasure_holds :
    forgettableBasis.all (fun v => isZeroPair measGen v) :=
  annihilators_are_zero_pairs

/-- **Exact-count corollary**: `A` has exactly 4 primitive right-
    annihilators in the entire 84-element catalogue of valid primitives. -/
theorem erasure_count_four :
    (orderedZDPairs.filter (fun p => p.1 == measGen)).length = 4 :=
  annihilators_complete

/-- **Distinctness**: the four annihilators are pairwise distinct,
    so the forgettable basis has cardinality 4. -/
theorem forgettable_basis_card_four :
    forgettableBasis.length = 4 ∧ forgettableBasis.Nodup := by
  refine ⟨?_, annihilators_distinct⟩
  native_decide

/-- **Intra-fiber locality**: the entire erasure structure of `A`
    lives in its own xor-fiber (label 9 = 3 ⊕ 10). -/
theorem erasure_intra_fiber :
    forgettableBasis.all (fun v => xorLabel v == xorLabel measGen) := by
  native_decide

-- ================================================================
-- §3. Global structure: 168 inequivalent erasure channels in 𝕊
-- ================================================================

/-- **Uniformity**: every primitive in 𝕊 defines an erasure of exactly
    the same combinatorial dimension as `A`. -/
theorem erasure_uniform :
    validPrims.all (fun u =>
      (orderedZDPairs.filter (fun p => p.1 == u)).length = 4) :=
  every_primitive_has_4_annihilators

/-- **The 168 channels**: there are exactly 168 projective equivalence
    classes of primitive sedenion zero-divisors, hence 168 structurally
    inequivalent rank-deficient erasure channels on `𝕊`. -/
theorem inequivalent_channel_count :
    unorderedZDPairs.length = 168 :=
  zd_projective_count_168

/-- **Channel-bundle invariant**: the total measure-of-erasure across
    all primitives equals `84 × 4 = 336`. -/
theorem total_erasure_measure :
    orderedZDPairs.length = validPrims.length * 4 :=
  total_forgettable_dimension.symm

end Sounio.SedenionMeasurement
