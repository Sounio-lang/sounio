import SounioMultiquadParam

open SounioSqrt
open SounioSqrt.RealCauchyField
open SounioSqrt.RealCauchyField.Multiquad

theorem evalS_cons_split_nil (p : Nat) (c : Nat → Rat) :
    evalS (p :: []) c =
      addR (evalS [] (lowCoeff c)) (mulR (sqrtR p) (evalS [] (highCoeff c))) := by
  have hL : evalS [] (lowCoeff c) = qR (c 0) := by
    unfold evalS lowCoeff
    rw [show (2 ^ (0 : Nat)) = 1 from rfl, show List.range 1 = [0] from rfl]
    simp [radS, evalS_nil, mulR_one, addR_zero]
  have hR : evalS [] (highCoeff c) = qR (c 1) := by
    unfold evalS highCoeff
    rw [show (2 ^ (0 : Nat)) = 1 from rfl, show List.range 1 = [0] from rfl]
    simp [radS, evalS_nil, mulR_one, addR_zero]
  unfold evalS
  rw [show (2 ^ ([p] : List Nat).length) = 2 from by simp, show List.range 2 = [0, 1] from rfl, hL, hR]
  simp [radS, List.foldr, mulR_one, addR_zero]

#print axioms evalS_cons_split_nil
