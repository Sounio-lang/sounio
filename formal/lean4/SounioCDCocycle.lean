/-
  SounioCDCocycle — a step toward proving the forward zero-divisor obstruction for ALL Cayley-Dickson
  levels (not just the dims verified by native_decide in SounioCDTowerSeam). The obstruction reduces to
  the OPERATOR identity L_i²=−I, i.e. σ(i,j)·σ(i,i⊕j)=−1 ∀j. On paper that closes by a simultaneous
  induction over the CD sign's four branches, using {L (=L²), R (=R²), antisym (basis units anticommute),
  diag (e_i²=−1)}. Here the CD sign is reformulated on explicit bit-lists `sgn` (MSB first; XOR is
  structural `xorL`, no Nat arithmetic), verified to agree with the Nat `cdSigma` at dim 16 and 32.
  PROVED for ALL n (structural induction): `diag` — e_i² = −1 for every imaginary basis unit. The
  remaining members (antisym, and hence the full L_i²=−I) close on paper but their Mathlib-free
  formalization is left open (the equation-compiler lemmas for the branching `sgn` resist `simp`);
  they remain certified at n=4,5,6 in SounioCDTowerSeam. No sorry. No Mathlib.
-/
namespace SounioCDCocycle
-- bit-list Cayley-Dickson sign, MSB first (head = top bit). Structural XOR, no Nat arithmetic.
def isZ : List Bool → Bool
  | [] => true
  | b :: bs => (!b) && isZ bs
def sgn : List Bool → List Bool → Int
  | [], [] => 1
  | [a], [b] => if a && b then -1 else 1
  | a :: as, b :: bs =>
      if isZ (a :: as) || isZ (b :: bs) then 1
      else
        match a, b with
        | false, false => sgn as bs
        | false, true  => sgn bs as
        | true,  false => - sgn as bs
        | true,  true  => if isZ bs then -1 else sgn bs as
  | _, _ => 0
termination_by a b => a.length + b.length
decreasing_by all_goals (simp_wf; omega)

def xorL : List Bool → List Bool → List Bool
  | a :: as, b :: bs => (xor a b) :: xorL as bs
  | _, _ => []
-- zeros of a given length
def zeros : Nat → List Bool
  | 0 => []
  | n+1 => false :: zeros n

-- sanity: matches the Nat cdSigma at small dims? convert Nat->bits (MSB first, fixed width) and compare.
def bitsOf : Nat → Nat → List Bool     -- width, value
  | 0, _ => []
  | w+1, v => (decide (v ≥ 2^w)) :: bitsOf w (v % 2^w)
def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        if !(a ≥ half) && !(b ≥ half) then cdSigma (a%half) (b%half) (n+1)
        else if !(a ≥ half) && (b ≥ half) then cdSigma (b%half) (a%half) (n+1)
        else if (a ≥ half) && !(b ≥ half) then (if b%half == 0 then cdSigma (a%half) 0 (n+1) else - cdSigma (a%half) (b%half) (n+1))
        else (if b%half == 0 then - cdSigma 0 (a%half) (n+1) else cdSigma (b%half) (a%half) (n+1))
-- agreement at width 4 (dim 16): sgn on bit-lists == cdSigma on Nats
theorem agree4 : (List.range 16).all (fun i => (List.range 16).all (fun j =>
  sgn (bitsOf 4 i) (bitsOf 4 j) == cdSigma i j 4)) = true := by native_decide
theorem agree5 : (List.range 32).all (fun i => (List.range 32).all (fun j =>
  sgn (bitsOf 5 i) (bitsOf 5 j) == cdSigma i j 5)) = true := by native_decide
-- xorL matches Nat xor at width 4
theorem xor_agree4 : (List.range 16).all (fun i => (List.range 16).all (fun j =>
  xorL (bitsOf 4 i) (bitsOf 4 j) == bitsOf 4 (i ^^^ j))) = true := by native_decide

-- diag: e_i^2 = -1 for every imaginary basis unit, ALL n (structural induction).
theorem diag : ∀ i : List Bool, isZ i = false → sgn i i = -1
  | [], h => by simp [isZ] at h
  | [a], h => by cases a <;> simp_all [isZ, sgn]
  | a :: b :: as, h => by
      have hz : isZ (a :: b :: as) = false := h
      cases a with
      | false =>
        have hb : isZ (b :: as) = false := by simpa [isZ] using hz
        simp only [sgn, hz, Bool.or_self, if_false]
        exact diag (b :: as) hb
      | true =>
        by_cases hbz : isZ (b :: as) = true
        · simp [sgn, hz, hbz]
        · have hb : isZ (b :: as) = false := by simpa using hbz
          simp only [sgn, hz, Bool.or_self, hb]
          exact diag (b :: as) hb


end SounioCDCocycle
