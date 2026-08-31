/-
  SounioConj68EulerLeg — route-alpha legs of the attack on Conjecture 6.8
  (Guterman–Zhilina, arXiv:2608.26903).  Split from SounioConj68RankBound.lean
  (the pod's thread cap bounds native_decide count per translation unit).

  ⚠ RODADA 10 CORRECTION: the "P3" relation below is VACUOUS — for pure u,w,
  wu = conj(uw) forces u∘w = 2Re(uw)·e₀ (always real), so ⟨T, u∘w⟩ = 0 holds
  trivially and cuts nothing.  Only P1, P2 are genuine pointwise relations;
  the bundle in the Euler argument has rank 9 (> dim 8) and the obstruction
  argument as stated below COLLAPSES.  The theorems in this file remain TRUE
  as verified statements (p3_bilinear proves exactly the identity that makes
  P3 vacuous; the characteristic-class arithmetic is correct for the stated
  ring element), but the topological interpretation awaits a genuine third
  pointwise relation with the right twist, or a deficit-1 argument.
  See attack log Rodada 10.

  THE EULER-CLASS OBSTRUCTION (Rodada 9 of the attack log, AS ORIGINALLY
  STATED — see the correction above):
  three POINTWISE relations of the commutator section, universal for pure u,w:
    (P1) ⟨[u,w], u⟩ = 0    (P2) ⟨[u,w], w⟩ = 0    — φ fully alternating
         (antisymmetric in (u,w) + cyclic via ⟨a,bc⟩ = ⟨ac̄,b⟩);
    (P3) ⟨[u,w], uw + wu⟩ = 0 — reduces to the bilinear identity
         wu = conj(uw) for pure u,w (then ⟨[u,w],uw+wu⟩ = n(uw) − n(wu) = 0
         by inner-product algebra alone).
  With the 4 proven linear relations, the section s(u,w) = T(u,w) lives in a
  rank-8 bundle E ≅ ℝ¹¹ ⊖ γ₁ ⊖ γ₂ ⊖ γ₁γ₂ over ℝP⁴ × ℝP⁴, twisted by γ₁γ₂
  (degeneracy loci to be refined separately).  The mod-2 obstruction
    e = Σᵢ wᵢ(E)(α+β)^{8−i},   w(E) = [(1+α)(1+β)(1+α+β)]⁻¹
  in ℤ/2[α,β]/(α⁵,β⁵) has α⁴β⁴-coefficient 1 — verified below by the kernel —
  so a nowhere-zero section is impossible: every configuration admits a
  witness, which is the last lemma of Conjecture 6.8 modulo the degeneracy-
  loci refinement.  Mathlib-free, no sorry.
-/
namespace SounioConj68EulerLeg

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

def sedSigma (a b : Nat) : Int := cdSigma a b 4

abbrev Vec := List Int

def coord (x : Vec) (k : Nat) : Int := x.getD k 0

def mulC (x y : Vec) (k : Nat) : Int :=
  (List.range 16).foldl (fun acc i => acc + sedSigma i (i ^^^ k) * coord x i * coord y (i ^^^ k)) 0

def mul (x y : Vec) : Vec := (List.range 16).map (mulC x y)

def vsub (x y : Vec) : Vec := (List.range 16).map (fun k => coord x k - coord y k)

def comm (x y : Vec) : Vec := vsub (mul x y) (mul y x)

def dot (x y : Vec) : Int := (List.range 16).foldl (fun acc k => acc + coord x k * coord y k) 0

def e (i : Nat) : Vec := (List.range 16).map (fun k => if k == i then (1 : Int) else 0)

/-- P1/P2 polarized: ⟨[eᵢ,eⱼ],e_k⟩ + ⟨[e_k,eⱼ],eᵢ⟩ = 0 for pure basis indices
    — the polarization of ⟨[u,w],u⟩ = 0 (P2 follows by antisymmetry of [·,·]). -/
def p12Polarized : Bool :=
  (List.range 15).all (fun i0 => (List.range 15).all (fun j0 => (List.range 15).all (fun k0 =>
    let i := i0 + 1; let j := j0 + 1; let k := k0 + 1
    dot (comm (e i) (e j)) (e k) + dot (comm (e k) (e j)) (e i) == 0)))

theorem p12_polarized : p12Polarized = true := by native_decide

/-- P3's bilinear core: wu = conj(uw) on all pure basis pairs (bilinearity
    gives the general pure case). -/
def conjMul (v : Vec) : Vec :=
  (List.range 16).map (fun k => if k == 0 then coord v 0 else - coord v k)

def p3Bilinear : Bool :=
  (List.range 15).all (fun i0 => (List.range 15).all (fun j0 =>
    let i := i0 + 1; let j := j0 + 1
    mul (e j) (e i) == conjMul (mul (e i) (e j))))

theorem p3_bilinear : p3Bilinear = true := by native_decide

/-- Characteristic-class arithmetic in ℤ/2[α,β]/(α⁵,β⁵): the α⁴β⁴ coefficient
    of Σᵢ wᵢ(E)(α+β)^{8−i} with w(E) = [(1+α)(1+β)(1+α+β)]⁻¹ equals 1, and
    the computed inverse is genuine. -/
abbrev Poly := List (List Bool)

def pget (p : Poly) (i j : Nat) : Bool := ((p.getD i []).getD j false)

def pmk (f : Nat → Nat → Bool) : Poly :=
  (List.range 5).map (fun i => (List.range 5).map (fun j => f i j))

def pmulZ2 (p q : Poly) : Poly :=
  pmk (fun i j =>
    ((List.range (i+1)).foldl (fun acc a =>
      ((List.range (j+1)).foldl (fun acc2 b =>
        if pget p a b && pget q (i-a) (j-b) then !acc2 else acc2) acc)) false))

def paddZ2 (p q : Poly) : Poly := pmk (fun i j => pget p i j != pget q i j)

def pone : Poly := pmk (fun i j => i == 0 && j == 0)

def dpoly : Poly :=
  pmulZ2 (pmulZ2 (pmk (fun i j => (i == 0 && j == 0) || (i == 1 && j == 0)))
                 (pmk (fun i j => (i == 0 && j == 0) || (i == 0 && j == 1))))
         (pmk (fun i j => (i == 0 && j == 0) || (i == 1 && j == 0) || (i == 0 && j == 1)))

def dinv : Poly :=
  let n := paddZ2 dpoly pone
  let step := fun (acc : Poly × Poly) (_ : Nat) =>
    (paddZ2 acc.1 acc.2, pmulZ2 acc.2 n)
  ((List.range 10).foldl step (pone, n)).1

def cpol : Poly := pmk (fun i j => (i == 1 && j == 0) || (i == 0 && j == 1))

def cpow (t : Nat) : Poly := (List.range t).foldl (fun acc _ => pmulZ2 acc cpol) pone

def wpart (i : Nat) : Poly := pmk (fun a b => a + b == i && pget dinv a b)

def eulerPoly : Poly :=
  (List.range 9).foldl (fun acc i => paddZ2 acc (pmulZ2 (wpart i) (cpow (8 - i)))) (pmk (fun _ _ => false))

def eulerChecks : Bool :=
  (pmulZ2 dpoly dinv == pone) && pget eulerPoly 4 4

theorem euler_obstruction_nonzero : eulerChecks = true := by native_decide

/-! ## RODADA 11 — the corrected obstruction (associator relations)

Genuine pointwise relations (each proved; non-vacuity measured 2000/2000):
  T ⊥ u (twist γ₁), T ⊥ w (γ₂), T ⊥ [u,w,w] (γ₁), T ⊥ [w,u,u] (γ₂).
Associator relations proof sketch: ⟨[u,w],[u,w,w]⟩ = ⟨[u,w],(uw)w⟩ (by P1 and
w² = −n(w)); the uw-term dies by ⟨a,bc⟩ = ⟨ac̄,b⟩ self-negation; the wu-term,
with v = uw and wu = conj v, reduces to ⟨v², w⟩ = 2Re(v)⟨uw,w⟩ = 0 since
⟨uw,w⟩ = n(w)⟨u,1⟩ = 0.  Symmetrically for [w,u,u].
Bundle: E₇ ≅ ℝ¹¹ ⊖ 2γ₁ ⊖ 2γ₂, rank 7 < dim 8; the PRIMARY obstruction to a
nowhere-zero section of E₇ ⊗ γ₁γ₂ is w₇ = Σᵢ wᵢ(E₇)(α+β)^{7−i} with
w(E₇) = [(1+α)²(1+β)²]⁻¹ = (1+α²+α⁴)(1+β²+β⁴): verified below to be
α⁴β³ + α³β⁴ ≠ 0.  A nonzero primary obstruction forces zeros even for
rank < dim — every configuration admits a witness, modulo the degeneracy-loci
refinement (same caveat class as before). -/

def d2poly : Poly :=
  let fa := pmk (fun i j => (i == 0 && j == 0) || (i == 1 && j == 0))
  let fb := pmk (fun i j => (i == 0 && j == 0) || (i == 0 && j == 1))
  pmulZ2 (pmulZ2 fa fa) (pmulZ2 fb fb)

def d2inv : Poly :=
  let n := paddZ2 d2poly pone
  let step := fun (acc : Poly × Poly) (_ : Nat) =>
    (paddZ2 acc.1 acc.2, pmulZ2 acc.2 n)
  ((List.range 10).foldl step (pone, n)).1

def w7part (i : Nat) : Poly := pmk (fun a b => a + b == i && pget d2inv a b)

def euler7Poly : Poly :=
  (List.range 8).foldl (fun acc i => paddZ2 acc (pmulZ2 (w7part i) (cpow (7 - i)))) (pmk (fun _ _ => false))

def euler7Checks : Bool :=
  (pmulZ2 d2poly d2inv == pone) && pget euler7Poly 4 3 && pget euler7Poly 3 4

theorem euler7_primary_obstruction_nonzero : euler7Checks = true := by native_decide

end SounioConj68EulerLeg
