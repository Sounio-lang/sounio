import SounioMultiquadParam

open SounioSqrt
open SounioSqrt.RealCauchyField
open SounioSqrt.RealCauchyField.Multiquad

/-- Recursive evaluation (split form); provably equal to `evalS`. -/
noncomputable def evalSRec : List Nat → (Nat → Rat) → Real
  | [], c => qR (c 0)
  | p :: S', c =>
      addR (evalSRec S' (lowCoeff c)) (mulR (sqrtR p) (evalSRec S' (highCoeff c)))

theorem evalSRec_cons (p : Nat) (S' : List Nat) (c : Nat → Rat) :
    evalSRec (p :: S') c =
      addR (evalSRec S' (lowCoeff c)) (mulR (sqrtR p) (evalSRec S' (highCoeff c))) := rfl

theorem evalSRec_nil (c : Nat → Rat) : evalSRec [] c = qR (c 0) := rfl

theorem evalS_eq_evalSRec_nil (c : Nat → Rat) : evalS [] c = evalSRec [] c := by
  rw [evalSRec_nil, evalS_nil]

theorem evalS_eq_evalSRec_cons (p : Nat) (S' : List Nat) (c : Nat → Rat)
    (ih : ∀ c', evalS S' c' = evalSRec S' c') :
    evalS (p :: S') c = evalSRec (p :: S') c := by
  unfold evalS evalSRec
  rw [show (p :: S').length = S'.length + 1 from rfl,
      show 2 ^ (S'.length + 1) = 2 * 2 ^ S'.length from Nat.pow_succ _]
  sorry

#print axioms evalSRec_cons
