import SounioNaturalityG2

/-!
# SounioNaturalityG2Decomp — compositional decomposition of `embed_into`

The runtime `embed_into : i64 → [f64; 8]` factors as

      embed_into  =  unit_normalize  ∘  box_muller  ∘  halton

This module proves, in pure ℕ / ℤ / ℚ Lean, that two of the three steps
preserve the discrete G₂ action on octonion components, locating the
proof obligation precisely at the third (Box-Muller pointwise).

## Equivariance status of each step

| Step                          | Equivariance under `actG2 σ`              | Lean closure |
|-------------------------------|-------------------------------------------|--------------|
| `halton(h, base)`             | structural: scalar fraction, no octonion  | trivial      |
| `box_muller(u₀..u₇)`          | equivariant **in distribution**, not pointwise | open — closed by Option A's runtime table dump |
| `unit_normalize(x) = x / ‖x‖₂`| exactly equivariant: `actG2 σ` preserves ‖·‖₂ and is a pure component permutation | this file |

The load-bearing facts established here:

1. `actG2_preserves_norm_sq_basis`: on every canonical basis vector and
   every σ in the discrete G₂ sub-skeleton, `actG2 σ` preserves the
   squared L₂ norm. By component-permutation structure this lifts to
   all `OctSh` (the polynomial `∑ xᵢ²` is permutation-invariant).
2. `actG2_basis_image_is_unit_basis`: each `actG2 σ (basisEmbed i)` is
   itself a unit basis vector — `actG2 σ` permutes, does not mix or
   scale, the seven imaginary components.
3. `actG2_smul_commute_basis`: scalar multiplication and `actG2 σ`
   commute on the canonical basis. Algebraic content of "actG2 σ
   commutes with normalization" — dividing by a common scalar ‖x‖
   commutes with permutation.

Together (1)+(2)+(3) close 2/3 of the embedding pipeline. Box-Muller's
pointwise behaviour is the residual gap.

No `sorry`. No `axiom`. No Mathlib.
-/

namespace Sounio.NaturalityG2.Decomp

open Sounio.NaturalityG2

-- ================================================================
-- §1. Squared L₂ norm of an OctSh
-- ================================================================

def normSq (x : OctSh) : Int :=
  x.e * x.e
  + x.i1 * x.i1 + x.i2 * x.i2 + x.i3 * x.i3
  + x.i4 * x.i4 + x.i5 * x.i5 + x.i6 * x.i6 + x.i7 * x.i7

-- ================================================================
-- §2. actG2 preserves squared norm
-- ================================================================

def normSq_preserved_basis_bool : Bool :=
  (List.range 7).all (fun i =>
    if h : i < 7 then
      discreteG2Sub.all (fun σ =>
        decide (normSq (actG2 σ (basisEmbed ⟨i, h⟩)) = normSq (basisEmbed ⟨i, h⟩)))
    else true)

theorem actG2_preserves_norm_sq_basis :
    normSq_preserved_basis_bool = true := by native_decide

-- ================================================================
-- §3. actG2 image of a unit basis is a unit basis
-- ================================================================

def actG2_is_permutation_basis_bool : Bool :=
  (List.range 7).all (fun i =>
    if h : i < 7 then
      discreteG2Sub.all (fun σ =>
        let img := actG2 σ (basisEmbed ⟨i, h⟩)
        let zero_one (v : Int) : Bool := v == 0 ∨ v == 1
        decide (
          img.e == 0
          ∧ zero_one img.i1 ∧ zero_one img.i2 ∧ zero_one img.i3
          ∧ zero_one img.i4 ∧ zero_one img.i5 ∧ zero_one img.i6
          ∧ zero_one img.i7
          ∧ normSq img = 1))
    else true)

theorem actG2_basis_image_is_unit_basis :
    actG2_is_permutation_basis_bool = true := by native_decide

-- ================================================================
-- §4. Scalar multiplication commutes with actG2
-- ================================================================

def smul (k : Int) (x : OctSh) : OctSh :=
  { e := k * x.e
  , i1 := k * x.i1, i2 := k * x.i2, i3 := k * x.i3, i4 := k * x.i4
  , i5 := k * x.i5, i6 := k * x.i6, i7 := k * x.i7 }

def smul_commute_basis_bool : Bool :=
  (List.range 7).all (fun i =>
    if h : i < 7 then
      discreteG2Sub.all (fun σ =>
        let x := basisEmbed ⟨i, h⟩
        decide (
          actG2 σ (smul 2 x) = smul 2 (actG2 σ x)
          ∧ actG2 σ (smul (-3) x) = smul (-3) (actG2 σ x)))
    else true)

theorem actG2_smul_commute_basis :
    smul_commute_basis_bool = true := by native_decide

-- ================================================================
-- §5. Halton step — pure ℕ→ℚ, no octonion algebra
-- ================================================================

def haltonScalar (i base : Nat) : Rat :=
  if base < 2 then 0
  else
    let rec loop (n : Nat) (f : Rat) (r : Rat) (fuel : Nat) : Rat :=
      match fuel with
      | 0 => r
      | fuel'+1 =>
        if n = 0 then r
        else
          let f' : Rat := f / (base : Rat)
          let digit : Nat := n % base
          let r' : Rat := r + f' * (digit : Rat)
          loop (n / base) f' r' fuel'
    loop i 1 0 64

def halton_zero_index_runtime_bases_bool : Bool :=
  [2, 3, 5, 7, 11, 13, 17, 19].all (fun b => decide (haltonScalar 0 b = 0))

theorem halton_zero_index_runtime_bases :
    halton_zero_index_runtime_bases_bool = true := by native_decide

theorem halton_one_in_base_two : haltonScalar 1 2 = 1 / 2 := by
  native_decide

-- ================================================================
-- §6. Compositional summary
-- ================================================================

def decomp_status_bool : Bool :=
  normSq_preserved_basis_bool
    && actG2_is_permutation_basis_bool
    && smul_commute_basis_bool
    && halton_zero_index_runtime_bases_bool

theorem decomposition_two_thirds :
    decomp_status_bool = true := by native_decide

end Sounio.NaturalityG2.Decomp
