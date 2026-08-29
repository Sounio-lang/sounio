import SounioCayleyDickson

-- Sounio Composition Algebra: Norm Multiplicativity and Artin Dormancy
-- No sorry. No Mathlib. All by native_decide.

namespace Sounio.CompositionAlgebra

open Sounio.CayleyDickson

-- §1. Helpers

def basisProductNormSq (i j : Nat) : Int :=
  (octSigma i j) * (octSigma i j)

-- §2. Composition Algebra (Hurwitz)

-- All basis pairs satisfy σ(i,j)² = 1 → ||eᵢ·eⱼ||² = 1 = ||eᵢ||²·||eⱼ||²
theorem basis_norm_multiplicative :
    (List.range 8 |>.flatMap fun i =>
      List.range 8 |>.filter fun j =>
        basisProductNormSq i j ≠ 1).length = 0 := by native_decide

theorem sigma_squared_is_one :
    (List.range 8 |>.flatMap fun i =>
      List.range 8 |>.filter fun j =>
        octSigma i j * octSigma i j ≠ 1).length = 0 := by native_decide

-- §3. Alternative Identities (Artin Dormancy)
-- Checks: sign of (eᵢ·eᵢ)·eⱼ = sign of eᵢ·(eᵢ·eⱼ) for all i,j

theorem left_alternative_sign :
    (List.range 8 |>.flatMap fun i =>
      List.range 8 |>.filter fun j =>
        let left_sign := octSigma i i * octSigma (i ^^^ i) j
        let right_sign := octSigma i j * octSigma i (i ^^^ j)
        left_sign ≠ right_sign).length = 0 := by native_decide

theorem right_alternative_sign :
    (List.range 8 |>.flatMap fun i =>
      List.range 8 |>.filter fun j =>
        let left_sign := octSigma j i * octSigma (j ^^^ i) i
        let right_sign := octSigma i i * octSigma j (i ^^^ i)
        left_sign ≠ right_sign).length = 0 := by native_decide

theorem flexible_sign :
    (List.range 8 |>.flatMap fun i =>
      List.range 8 |>.filter fun j =>
        let left_sign := octSigma i j * octSigma (i ^^^ j) i
        let right_sign := octSigma j i * octSigma i (j ^^^ i)
        left_sign ≠ right_sign).length = 0 := by native_decide

-- §4. Non-Associativity Counterexample
-- (e₁·e₂)·e₅ ≠ e₁·(e₂·e₅)

theorem nonassociative_witness :
    octSigma 1 2 * octSigma (1 ^^^ 2) 5 ≠
    octSigma 2 5 * octSigma 1 (2 ^^^ 5) := by native_decide

-- §5. Jacobian Orthogonality
-- R_{eⱼ}^T · R_{eⱼ} = I (right-multiplication matrix is orthogonal)

theorem right_mul_matrix_orthogonal :
    (List.range 8 |>.flatMap fun j =>
      List.range 8 |>.flatMap fun i =>
        List.range 8 |>.filter fun i' =>
          let dot := (List.range 8).foldl (fun acc k =>
            let rki := if (i ^^^ j) = k then octSigma i j else 0
            let rki' := if (i' ^^^ j) = k then octSigma i' j else 0
            acc + rki * rki') 0
          let expected := if i = i' then 1 else 0
          dot ≠ expected).length = 0 := by native_decide

-- §6. Synthesis: norm preservation coexists with non-associativity

theorem composition_algebra_with_nonassociativity :
    ((List.range 8 |>.flatMap fun i =>
       List.range 8 |>.filter fun j =>
         octSigma i j * octSigma i j ≠ 1).length = 0) ∧
    (nonFanoCount = 168) := by
  exact ⟨sigma_squared_is_one, non_fano_count_168⟩

end Sounio.CompositionAlgebra
