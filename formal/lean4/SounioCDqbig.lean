/-
  SounioCDqbig — exact witness (native Int) for the unbounded-ℚ Cayley-Dickson product brick
  (tests/run-pass/sedenion_cd_qbig.sio). Lean's Int is arbitrary precision, so this independently
  confirms, digit-exact: (case 1) a zero divisor scaled by 10^40 annihilates in all 16 components at
  magnitude 10^80; (case 2) the general 16-component product equals the exact values (also produced by
  the Python oracle and, mod 1e9+7, by the souc minimal-BigInt leg). Mathlib-free, no sorry.
-/
namespace SounioCDqbig

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

def cd16 (A B : List Int) : Nat → Int := fun k =>
  (List.range 16).foldl (fun acc i =>
    (List.range 16).foldl (fun acc2 j =>
      if (i ^^^ j) == k then acc2 + cdSigma i j 4 * A.getD i 0 * B.getD j 0 else acc2) acc) 0

-- Case 1: (e3+e10)(e6-e15), each coefficient scaled by 10^40.
def A1 : List Int := (List.range 16).map (fun i => if i == 3 || i == 10 then (10:Int)^40 else 0)
def B1 : List Int := (List.range 16).map (fun i => if i == 6 then (10:Int)^40 else if i == 15 then -(10:Int)^40 else 0)
/-- Exact annihilation at magnitude 10^80: all 16 components are exactly 0. -/
theorem case1_annihilates : (List.range 16).all (fun k => cd16 A1 B1 k == 0) = true := by native_decide

-- Case 2: general 16-comp product, unbounded rational numerators.
def A2 : List Int := (List.range 16).map (fun i =>
  if i == 1 then (10:Int)^30 else if i == 2 then 3*(10:Int)^28 else if i == 4 then -5*(10:Int)^35
  else if i == 8 then (10:Int)^33 else if i == 15 then 7*(10:Int)^20 else 0)
def B2 : List Int := (List.range 16).map (fun i =>
  if i == 1 then 2*(10:Int)^31 else if i == 5 then -(10:Int)^29 else if i == 7 then 4*(10:Int)^27
  else if i == 10 then 9*(10:Int)^34 else 0)
def expected2 : List Int := [-20000000000000000000000000000000000000000000000000000000000000, 50000000000000000000000000000000000000000000000000000000000000000, 90000000000000000000000000000000000000000000000000000000000000000000, -2000600000000000000000000000000000000000000000000000000000000000, 100000000000000000000000000000000000000000000000000000000000, 9999999999817000000000000000000000000000000000000000000000000000000, 4000000000000000000000000000000000000000000000000000000000, -3000000000000000000000000000000000000000000000000000000000, -2699999999999997200000000000000000000000000000000000000000000000, -20000000000000000000000000000000000000000000000000000000000000000, -70000000000000000000000000000000000000000000000000, -90000000000000000000000000000000000000000000000000000000000000000, 0, 100000000000000000000000000000000000000000000000000000000000000, -44999999999999999986000000000000000000000000000000000000000000000000000, -4000000000000000000000000000000000000000000000000000000000000]
/-- The general product equals the exact values (digit-exact witness, native Int). -/
theorem case2_exact : (List.range 16).map (cd16 A2 B2) = expected2 := by native_decide

end SounioCDqbig
