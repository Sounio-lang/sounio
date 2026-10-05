/-
  SounioCDCoreLaw — the CORE-LAW character-sum TWIN RECURSION, formally machine-checked.

  Context (Python-proven, this file is the Lean certificate): the zero-divisor annihilation degree of a
  mixed-half primitive e_lo + s·e_hi equals the middle-slot associator degree #{a : Ψ(lo,a,hi_lo)=1},
  hence D = (2^m − S)/2 with the CHARACTER SUM
        S(m, lo, hi_lo) = Σ_{a<2^m} (−1)^{Ψ(lo,a,hi_lo)},   (−1)^Ψ = ∏ of the four cdSigma factors.
  The core law (fiber degree stratification of the frozen-168 orbit action on the CD zero divisors)
  reduces to two DOUBLING RECURSIONS of S — both derived from the ∀n seam-flip law + the vanishing of the
  associator on degenerate triples:
        (middle-a)     S m lo hi_lo          = 2·S (m−1) lo hi_lo − 8·[hi_lo ≠ 0]        (lo,hi_lo lower)
        (hi_lo-seam)   S m lo (h_lo + 2^{m−1}) = 8 − 2·S (m−1) lo h_lo                     (lo,h_lo lower)
  Together they close S to the octonion base, giving Dmax = 4(2^{m−2}−1) and the per-fiber maximizer
  count 2(2^{m−b}−1) — the core law.  The "non-quadratic wall" (Ψ(lo,·,hi_lo) is quadratic in a only up
  to m=4) is irrelevant: both seam corrections live on the associator's zero locus, so S recurses anyway.

  This file gives the Mathlib-free machine-checked CERTIFICATE of both recursions at dims 16/32/64
  (m = 4,5,6), by native_decide — the lane's regression-anchor convention (cf. lsq_16/32/64,
  coincidence_16/32 in SounioCDTowerSeam).  The ∀n structural proof (formalizing the seam-flip law for Ψ
  and the List.range sum split Mathlib-free) is the tower-wide target, tracked separately; the recursions
  themselves are DERIVED ∀n in Python (scripts/research/cd_tower_core_law_recursion_proof.py).
  Mathlib-free, no sorry.
-/
namespace SounioCDCoreLaw

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

/-- (−1)^{Ψ(lo,a,hi_lo)} as a product of the four cdSigma factors of the associator 3-form. -/
def psiSign (bits lo a hilo : Nat) : Int :=
  cdSigma lo a bits * cdSigma (lo ^^^ a) hilo bits * cdSigma a hilo bits * cdSigma lo (a ^^^ hilo) bits

/-- Character sum  S(bits,lo,hi_lo) = Σ_{a<2^bits} (−1)^{Ψ(lo,a,hi_lo)}. -/
def charSum (bits lo hilo : Nat) : Int :=
  ((List.range (2 ^ bits)).map (fun a => psiSign bits lo a hilo)).foldl (· + ·) 0

/-- middle-a recursion certificate at a given level: for all lo,hi_lo in the LOWER half of level `bits`,
    S bits lo hi_lo = 2·S (bits−1) lo hi_lo − 8·[hi_lo ≠ 0]. -/
def midRecOK (bits : Nat) : Bool :=
  let H := 2 ^ (bits - 1)
  (List.range H).all (fun lo => lo == 0 ||
    (List.range H).all (fun hilo => lo == hilo ||
      charSum bits lo hilo == 2 * charSum (bits-1) lo hilo - (if hilo == 0 then 0 else 8)))

/-- hi_lo-seam recursion certificate: S bits lo (h_lo + 2^{bits−1}) = 8 − 2·S (bits−1) lo h_lo. -/
def hiRecOK (bits : Nat) : Bool :=
  let H := 2 ^ (bits - 1)
  (List.range H).all (fun lo => lo == 0 ||
    (List.range H).all (fun hlo =>
      let hi := hlo + H
      (lo == hlo || lo == hi) ||
      charSum bits lo hi == 8 - 2 * charSum (bits-1) lo hlo))

/-- Dmax certificate: the maximum of the middle-slot associator degree over the fiber equals 4(2^{m−2}−1),
    equivalently the minimum of S equals 8 − 2^m. -/
def dmaxOK (bits : Nat) : Bool :=
  let N := 2 ^ bits
  let smin := 8 - (2 ^ bits : Int)
  -- some pair attains S = 8 − 2^m, and none goes below it
  ((List.range N).any (fun lo => lo == 0 || (List.range N).any (fun hilo =>
      lo != hilo && charSum bits lo hilo == smin)))
  && ((List.range N).all (fun lo => lo == 0 || (List.range N).all (fun hilo =>
      lo == hilo || charSum bits lo hilo ≥ smin)))

-- ===== formal per-dimension certificates (native_decide) =====

/-- Twin recursion at dim 16 (m=4): both the middle-a and the hi_lo-seam recursion hold. -/
theorem twin_16 : midRecOK 4 = true ∧ hiRecOK 4 = true := by native_decide
/-- Twin recursion at dim 32 (m=5). -/
theorem twin_32 : midRecOK 5 = true ∧ hiRecOK 5 = true := by native_decide
/-- Twin recursion at dim 64 (m=6). -/
theorem twin_64 : midRecOK 6 = true ∧ hiRecOK 6 = true := by native_decide

/-- Dmax = 4(2^{m−2}−1) i.e. S_min = 8 − 2^m, at dims 32 and 64. -/
theorem dmax_32 : dmaxOK 5 = true := by native_decide
theorem dmax_64 : dmaxOK 6 = true := by native_decide

end SounioCDCoreLaw
