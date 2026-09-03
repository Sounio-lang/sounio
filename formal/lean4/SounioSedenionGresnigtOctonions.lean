/-
  SounioSedenionGresnigtOctonions — cross-check of Gresnigt's three generation-octonions (arXiv:2306.13098)
  and an explicit G₂ (color-side) monomial automorphism permuting them, by Lean native_decide.
  φ = (e₁e₂e₃)(e₅e₆e₇)(e₉e₁₀e₁₁)(e₁₃e₁₄e₁₅) [all signs +1] is a genuine order-3 automorphism of 𝕊
  cycling 𝕆₁→𝕆₂→𝕆₃, fixing the shared quaternion {4,8,12} and e₈; and φ ∈ G₂ (its restriction to the
  base octonion {1..7} is an octonion automorphism). This is NOT Gresnigt's family S₃ (Brown factor,
  non-monomial, disjoint from G₂); it is a color-sector map. Not a physics bridge. Mathlib-free, no sorry.
-/
namespace SounioSedenionGresnigtOctonions

def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        let aHi := a ≥ half; let bHi := b ≥ half; let aLo := a % half; let bLo := b % half
        if !aHi && !bHi then cdSigma aLo bLo (n+1)
        else if !aHi && bHi then cdSigma bLo aLo (n+1)
        else if aHi && !bHi then (if bLo == 0 then cdSigma aLo 0 (n+1) else - cdSigma aLo bLo (n+1))
        else (if bLo == 0 then - cdSigma 0 aLo (n+1) else cdSigma bLo aLo (n+1))
def sg (a b : Nat) : Int := cdSigma a b 4

def piL : List Nat := [0,2,3,1,4,6,7,5,8,10,11,9,12,14,15,13]
def pi (i : Nat) : Nat := piL.getD i 0

/-- φ is a genuine algebra automorphism of 𝕊: all 256 products (sign-free monomial condition). -/
theorem phi_automorphism :
    (List.range 16).all (fun i => (List.range 16).all (fun j =>
      sg (pi i) (pi j) == sg i j && pi (i ^^^ j) == (pi i) ^^^ (pi j))) = true := by native_decide
/-- φ has order 3 and moves something. -/
theorem phi_order_3 :
    (List.range 16).all (fun i => pi (pi (pi i)) == i) && (List.range 16).any (fun i => pi i != i) = true := by
  native_decide
/-- φ fixes the shared quaternion {4,8,12} and e₈. -/
theorem phi_fixes_quaternion_e8 : (pi 4, pi 8, pi 12) = (4, 8, 12) := by native_decide
/-- φ ∈ G₂: it preserves the base octonion {1..7} and restricts there to an octonion automorphism. -/
theorem phi_in_G2 :
    (List.range 7).all (fun i => 1 ≤ pi (i+1) && pi (i+1) ≤ 7)
    && (List.range 8).all (fun i => (List.range 8).all (fun j => cdSigma (pi i) (pi j) 3 == cdSigma i j 3)) = true := by
  native_decide
/-- φ cyclically permutes Gresnigt's three octonions 𝕆₁→𝕆₂→𝕆₃. -/
theorem phi_cycles_octonions :
    (([1,4,5,8,9,12,13].map pi).all (fun x => x ∈ [2,4,6,8,10,12,14])
     && ([2,4,6,8,10,12,14].map pi).all (fun x => x ∈ [3,4,7,8,11,12,15])) = true := by native_decide

end SounioSedenionGresnigtOctonions
